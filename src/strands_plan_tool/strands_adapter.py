"""Strands Agents adapter: exposes plan submission as one ordinary tool.

The model emits a single ``submit_workflow_plan`` tool call. From the event loop's side
that is an entirely ordinary tool call -- dispatch, one ``ToolResult``, recurse back to
the model -- so the agentic loop is neither modified nor bypassed. The plan's steps run
inside that one call.

Autonomy is preserved because planning is *elected*, never imposed. Tool choice stays
automatic, so the model may submit a plan, or ignore this tool entirely and call tools
one at a time as usual. Nothing here authors a plan on the model's behalf.

Interrupts: verified against strands-agents 1.55.1 on 2026-09-15. A direct tool call
(``agent.tool.<name>(...)``, the path a plan uses) refuses interrupts at two points in
``strands/tools/_caller.py``:

* If the agent is already interrupted, any direct call raises
  ``RuntimeError("cannot directly call tool during interrupt")``.
* If a tool raises an interrupt during a direct call, the caller raises
  ``RuntimeError("cannot raise interrupt in direct tool call")`` -- and first calls
  ``_InterruptState.deactivate()``, which CLEARS the agent's interrupts and context.

That second behaviour is why plannability is settled before execution rather than caught
during it: by the time the error surfaces, the interrupt state is already gone. An
approval-gated tool is therefore refused at validation, with a message telling the model
to call it directly instead. The gate keeps working; it simply is not batchable.

Docs consulted: https://strandsagents.com/docs/api/python/strands.plugins.plugin/
and https://strandsagents.com/docs/user-guide/concepts/agents/interventions/human-in-the-loop/
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError
from strands import tool
from strands.hooks import AfterToolsEvent
from strands.plugins import Plugin

from .errors import (
    AgentInterruptedError,
    InvalidPlanShapeError,
    PlanError,
    ToolExecutionError,
)
from .executor import execute_plan
from .models import MAX_STEPS, JsonValue, PlanResult, StepStatus, WorkflowPlan

if TYPE_CHECKING:  # pragma: no cover
    from strands import Agent

__all__ = [
    "ALL_TOOLS",
    "AgentToolInvoker",
    "WorkflowPlanPlugin",
    "coerce_plan",
    "resolve_plannable",
    "unwrap_tool_result",
]

ALL_TOOLS: Final = ("*",)
"""Default plannable pattern: every registered tool except this plugin's own tool."""

_PLAN_TOOL_NAME: Final = "submit_workflow_plan"

_INTERRUPT_REFUSAL: Final = (
    "cannot directly call tool during interrupt",
    "cannot raise interrupt in direct tool call",
)


def resolve_plannable(patterns: Iterable[str], registered: Iterable[str]) -> frozenset[str]:
    """Expand allow/deny patterns against the registered tool names.

    Pattern syntax matches the SDK's own ``HumanInTheLoop.allowed_tools`` vocabulary so
    the two configurations read alike: a bare name allows it, ``"*"`` allows everything,
    and ``"!name"`` removes one. Negations are applied after inclusions, so ordering
    inside the sequence does not matter.

    Args:
        patterns: Allow/deny patterns.
        registered: Every tool name registered on the agent.

    Returns:
        The tool names a plan may call. This plugin's own tool is always excluded, so a
        plan can never nest a plan.
    """
    names = frozenset(registered)
    allowed: set[str] = set()
    denied: set[str] = set()

    for pattern in patterns:
        if pattern == "*":
            allowed |= names
        elif pattern.startswith("!"):
            denied.add(pattern[1:])
        else:
            allowed.add(pattern)

    return frozenset(allowed - denied - {_PLAN_TOOL_NAME}) & names


def _plan_fully_succeeded(result: PlanResult) -> bool:
    """Return whether every step in a plan completed successfully.

    Args:
        result: A completed plan result.

    Returns:
        True when no step failed or was skipped.
    """
    return all(outcome.status is StepStatus.OK for outcome in result.ledger)


def coerce_plan(plan: WorkflowPlan | Mapping[str, JsonValue]) -> WorkflowPlan:
    """Validate a raw plan payload into the typed model.

    The SDK hands a nested Pydantic tool parameter through as the decoded JSON object, not
    as a model instance, so this coercion is the difference between the tool working
    against a live model and raising ``AttributeError`` on ``plan.steps``.

    Args:
        plan: A model instance, or the raw JSON object the model emitted.

    Returns:
        The validated plan.

    Raises:
        InvalidPlanShapeError: The payload does not match the schema. Carries Pydantic's
            own message, which names the offending field, so the model can repair it on the
            next turn rather than guessing.
    """
    if isinstance(plan, WorkflowPlan):
        return plan
    try:
        return WorkflowPlan.model_validate(dict(plan))
    except ValidationError as exc:
        raise InvalidPlanShapeError(str(exc)) from exc


def unwrap_tool_result(tool: str, result: Mapping[str, JsonValue]) -> JsonValue:
    """Turn a Strands ``ToolResult`` into the value a binding expects to see.

    This is the difference between a plan that works and one that cannot. A ``@tool``
    returning ``{"beast": "leshen"}`` does not arrive as that dict -- Strands wraps it as
    ``{"status": "success", "content": [{"text": '{"beast": "leshen"}'}]}``, a JSON *string*
    inside a content block. Handing that through unchanged makes the obvious binding,
    ``steps.board.beast``, unresolvable, and the model has no way to know why.

    So a lone text block is JSON-decoded when it can be, leaving bindings to read the shape
    the tool's own signature advertises.

    Args:
        tool: Tool name, for the error message.
        result: The ``ToolResult`` from a direct tool call.

    Returns:
        The decoded value: a parsed object for a single JSON text block, the raw string for
        a single non-JSON text block, a list of strings for several text blocks, and the
        content blocks verbatim for anything else.

    Raises:
        ToolExecutionError: The tool reported ``status="error"``. Returning its error text
            as if it were data would let a failed step look successful.
    """
    blocks = result.get("content")
    texts: list[str] = []
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)

    if result.get("status") == "error":
        raise ToolExecutionError(tool, " ".join(texts) or "no detail reported")

    if not isinstance(blocks, list):
        return blocks

    if len(texts) == len(blocks) == 1:
        try:
            decoded: JsonValue = json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
        return decoded

    if texts and len(texts) == len(blocks):
        return list(texts)

    return blocks


@dataclass(frozen=True, slots=True)
class AgentToolInvoker:
    """Runs a plan step as a direct tool call on a Strands agent.

    ``record_direct_tool_call=False`` keeps plan steps out of message history: the model
    asked for a plan, and its result is the plan's aggregate, not a transcript of every
    inner call.
    """

    agent: Agent

    async def __call__(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        """Invoke one registered tool.

        Args:
            name: Registered tool name.
            args: Resolved arguments; copied before dispatch so the caller's mapping is
                never handed to the SDK.

        Returns:
            The tool's value, decoded by :func:`unwrap_tool_result`.

        Raises:
            PermissionError: The SDK refused the call because an interrupt was involved.
                Raised as a distinct type so the step's error names the real cause
                instead of a bare ``RuntimeError``.
            ToolExecutionError: The tool itself reported failure.
        """
        caller = getattr(self.agent.tool, name)
        try:
            result = caller(**dict(args), record_direct_tool_call=False)
        except RuntimeError as exc:
            if any(marker in str(exc) for marker in _INTERRUPT_REFUSAL):
                raise PermissionError(
                    f"tool {name!r} requires human approval and cannot run inside a plan; "
                    "call it directly instead"
                ) from exc
            raise

        return unwrap_tool_result(name, result)


class WorkflowPlanPlugin(Plugin):
    """Adds one tool that accepts a JSONata workflow plan and executes it locally.

    Args:
        plannable: Patterns naming which tools a plan may call, in
            ``HumanInTheLoop.allowed_tools`` syntax. Defaults to every registered tool.
            When the agent has interventions attached you MUST pass this explicitly --
            see :meth:`init_agent`.
        max_steps: Ceiling on steps per plan.
        honor_final: When a plan sets ``final``, end the turn with its returned values
            instead of sampling the model once more to restate them. Saves the closing
            round trip; the cost is that the last message is JSON rather than prose. Pass
            ``False`` to keep the prose.

    Example:
        ```python
        agent = Agent(
            tools=[read_notice_board, consult_bestiary],
            plugins=[WorkflowPlanPlugin()],
        )
        ```

    Note:
        The instance carries per-agent state, so give each agent its own.
    """

    name = "jsonata-workflow-plan"

    def __init__(
        self,
        *,
        plannable: Sequence[str] | None = None,
        max_steps: int = MAX_STEPS,
        honor_final: bool = True,
    ) -> None:
        """Configure the plugin. See the class docstring for argument meanings."""
        super().__init__()
        self._plannable: tuple[str, ...] = tuple(plannable) if plannable is not None else ALL_TOOLS
        self._plannable_was_explicit = plannable is not None
        self._max_steps = max_steps
        self._honor_final = honor_final
        self._agent: Agent | None = None
        self._pending_final: str | None = None

    def init_agent(self, agent: Agent) -> None:
        """Bind the agent and refuse an unsafe default.

        An agent with interventions attached may gate any tool behind human approval,
        and this plugin cannot tell which from the outside. Rather than default to
        "everything is plannable" and risk batching an approval-gated tool, it fails
        here, at construction, where the developer can see it.

        Args:
            agent: The agent this plugin is attached to.

        Raises:
            ValueError: Interventions are attached but ``plannable`` was left at its
                default, or this instance is already bound to a different agent.
        """
        if self._agent is not None and self._agent is not agent:
            raise ValueError(
                f"{self.name}: this plugin instance is already attached to another agent. "
                "It carries per-agent state, so give each agent its own instance."
            )

        registry = getattr(agent, "_intervention_registry", None)
        handlers = getattr(registry, "handlers", None) if registry is not None else None
        if handlers and not self._plannable_was_explicit:
            names = ", ".join(sorted(h.name for h in handlers))
            raise ValueError(
                f"{self.name}: agent has interventions attached ({names}) which may gate "
                "tools behind human approval, and approval cannot be serviced inside a "
                "plan. Pass plannable=[...] naming the tools that are safe to batch, "
                'for example plannable=["*", "!delete_files"].'
            )
        self._agent = agent
        # Registered here rather than via the @hook decorator: the decorator's overloads
        # describe a single-argument callback, so decorating a *method* (which also takes
        # self) does not type-check, even though the SDK documents that form. add_hook with
        # a bound method is the documented alternative and is fully typed.
        agent.add_hook(self.end_turn_after_a_final_plan)

    def end_turn_after_a_final_plan(self, event: AfterToolsEvent) -> None:
        """End the turn when a plan declared itself final, saving the closing round trip.

        A plan marked ``final`` asserts that its returned values answer the request. Taking
        the model at its word lets the loop stop here instead of sampling once more purely
        to have it restate the result -- the difference between two model round trips and
        one.

        The trade is real: the closing message becomes the plan's returned values as JSON
        rather than prose the model wrote. Construct with ``honor_final=False`` to keep the
        prose and pay the extra round trip.

        The pending payload is consumed here, so a later tool cycle in the same turn cannot
        inherit an earlier plan's decision.

        Args:
            event: The batch-completion event whose ``end_turn`` halts the loop.
        """
        payload = self._pending_final
        self._pending_final = None
        if payload is not None:
            event.end_turn = payload

    @tool(name=_PLAN_TOOL_NAME)
    async def submit_workflow_plan(self, plan: WorkflowPlan) -> PlanResult:
        """Run several dependent tool calls in one batch instead of one at a time.

        Use this when you already know which tools to call and how each call's arguments
        come from an earlier call's result, so no judgement is needed in between. Each
        step's `bind` maps an argument name to a JSONata expression evaluated against
        `{"steps": {<step id>: <that step's result>}}`, for example
        `{"beast": "steps.board.beast"}` when the `board` step's tool documents that it
        returns an object with a `beast` field. Read each tool's documented return shape
        and bind to the exact field you need; `steps.board` alone is the whole result
        object, not a field of it. Name in `returns` only the steps whose results you
        actually need back. Set `final` when those results answer the request.

        Do not use this when a step's outcome needs your judgement before you can decide
        what to call next; call those tools one at a time instead. If a tool's return
        shape is not documented, call it directly once to see it rather than guessing in
        a plan.

        Args:
            plan: The steps to run, their dependencies, and which results to return.

        Returns:
            The requested results, plus a per-step ledger reporting what happened.
        """
        return await self.run_plan(plan)

    async def run_plan(self, plan: WorkflowPlan | Mapping[str, JsonValue]) -> PlanResult:
        """Execute a plan against the bound agent.

        Separate from the ``@tool`` wrapper above so it is directly callable and typed:
        the decorator turns the method into a tool descriptor, whose call signature is
        the model's, not Python's. The wrapper carries the model-facing docstring; this
        carries the behaviour.

        Accepts a raw mapping as well as a model instance, because the SDK hands a nested
        Pydantic parameter through as the decoded JSON dict rather than coercing it. Doing
        the coercion here means a malformed plan comes back as a legible validation message
        the model can repair, instead of an ``AttributeError`` it cannot.

        Args:
            plan: The plan to validate and execute, typed or as the raw JSON object.

        Returns:
            The aggregated plan result.

        Raises:
            AgentInterruptedError: The agent has an outstanding interrupt, which makes
                every direct tool call fail, so no step could run.
            InvalidPlanShapeError: ``plan`` does not match the schema.
            strands_plan_tool.errors.PlanError: The plan is malformed or names a tool that is
                not plannable. Nothing was executed.
        """
        agent = self._agent
        if agent is None:  # pragma: no cover - registry always calls init_agent first
            raise RuntimeError(f"{self.name} was not attached to an agent")

        if agent._interrupt_state.activated:
            raise AgentInterruptedError

        validated = coerce_plan(plan)
        allowed = resolve_plannable(self._plannable, agent.tool_registry.registry.keys())
        result = await execute_plan(
            validated,
            AgentToolInvoker(agent),
            max_steps=self._max_steps,
            allowed_tools=allowed,
        )

        # Only arm the early exit when the plan both declared itself final AND actually
        # delivered every step. Ending the turn on a partial result would hide the failure
        # from the model, which is the one thing the ledger exists to prevent.
        if self._honor_final and result.final and _plan_fully_succeeded(result):
            self._pending_final = json.dumps(result.returned, default=str)

        return result


def describe_plan_failure(exc: PlanError) -> str:
    """Render a plan rejection as guidance the model can act on next turn.

    Args:
        exc: The validation error raised before execution.

    Returns:
        A single line naming the problem and the corrective action.
    """
    return f"Plan rejected, nothing was executed: {exc}"
