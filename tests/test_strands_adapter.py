"""The Strands adapter: plannability rules, interrupt refusal, and loop shape.

These tests build a real ``Agent`` with a stub Model. A direct tool call never samples the
model, so nothing here needs credentials or a network.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from strands import Agent, tool
from strands.hooks import AfterToolsEvent
from strands.interrupt import Interrupt, InterruptException
from strands.interventions import InterventionHandler
from strands.models.model import Model

from strands_plan_tool import (
    AgentInterruptedError,
    InvalidPlanShapeError,
    JsonValue,
    PlanStep,
    StepStatus,
    ToolExecutionError,
    ToolNotPlannableError,
    WorkflowPlan,
    execute_plan,
)
from strands_plan_tool.strands_adapter import (
    AgentToolInvoker,
    WorkflowPlanPlugin,
    coerce_plan,
    resolve_plannable,
    unwrap_tool_result,
)

_NEVER_YIELDS: tuple[Any, ...] = ()


class StubModel(Model):
    """Never invoked: a direct tool call does not sample the model."""

    def update_config(self, **model_config: Any) -> None:
        return None

    def get_config(self) -> Any:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        for item in _NEVER_YIELDS:
            yield item
        raise AssertionError("model must not be invoked")

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        # Loops an opaque empty tuple so this is a real async generator without the
        # post-raise `yield` that reads as dead code.
        for item in _NEVER_YIELDS:
            yield item
        raise AssertionError("model must not be invoked")


@tool
def read_board() -> dict[str, str]:
    """Read the notice board."""
    return {"beast": "leshen"}


@tool
def consult(beast: str) -> dict[str, str]:
    """Look up a beast.

    Args:
        beast: Beast name.

    Returns:
        An object shaped {"weakness": str}.
    """
    return {"weakness": f"sign-for-{beast}"}


@tool
def needs_approval(amount: int) -> str:
    """A tool gated behind a human approval interrupt."""
    raise InterruptException(Interrupt(id="v1:test:approve", name="approve", reason=amount))


def build_agent(**kwargs: Any) -> Agent:
    return Agent(model=StubModel(), tools=[read_board, consult, needs_approval], **kwargs)


# --- pattern expansion -------------------------------------------------------------


def test_star_allows_every_registered_tool_but_never_the_plan_tool() -> None:
    allowed = resolve_plannable(["*"], ["read_board", "consult", "submit_workflow_plan"])
    assert allowed == frozenset({"read_board", "consult"})


def test_negation_removes_one_tool() -> None:
    allowed = resolve_plannable(["*", "!consult"], ["read_board", "consult"])
    assert allowed == frozenset({"read_board"})


def test_negation_order_does_not_matter() -> None:
    assert resolve_plannable(["!consult", "*"], ["read_board", "consult"]) == resolve_plannable(
        ["*", "!consult"], ["read_board", "consult"]
    )


def test_bare_names_are_an_allowlist() -> None:
    allowed = resolve_plannable(["read_board"], ["read_board", "consult"])
    assert allowed == frozenset({"read_board"})


def test_unregistered_names_are_dropped() -> None:
    assert resolve_plannable(["ghost"], ["read_board"]) == frozenset()


# --- interrupt resolution ----------------------------------------------------------


async def test_an_approval_gated_tool_is_refused_before_anything_runs() -> None:
    calls: list[str] = []

    async def invoker(name: str, args: Any) -> Any:
        calls.append(name)
        return name

    plan = WorkflowPlan(
        steps=(
            PlanStep(id="board", tool="read_board"),
            PlanStep(id="pay", tool="needs_approval", args={"amount": 10}),
        ),
        returns=("pay",),
    )
    with pytest.raises(ToolNotPlannableError) as caught:
        await execute_plan(plan, invoker, allowed_tools={"read_board", "consult"})

    assert caught.value.tool == "needs_approval"
    assert "Call it directly instead" in str(caught.value)
    assert calls == [], "validation must reject the plan before executing any step"


async def test_the_invoker_names_the_real_cause_when_a_tool_interrupts() -> None:
    # Defence in depth: if an interruptible tool slips past the allowlist, the raw
    # RuntimeError is translated into something the model can act on.
    agent = build_agent()
    invoker = AgentToolInvoker(agent)
    with pytest.raises(PermissionError, match="requires human approval"):
        await invoker("needs_approval", {"amount": 10})


async def test_a_plan_is_refused_while_the_agent_is_interrupted() -> None:
    plugin = WorkflowPlanPlugin()
    agent = build_agent(plugins=[plugin])
    agent._interrupt_state.activate()

    plan = WorkflowPlan(steps=(PlanStep(id="board", tool="read_board"),), returns=("board",))
    with pytest.raises(AgentInterruptedError):
        await plugin.run_plan(plan)


# --- attachment safety -------------------------------------------------------------


class Gate(InterventionHandler):
    """Minimal intervention, present only so the plugin can detect one."""

    name = "test-gate"


def test_attaching_to_an_agent_with_interventions_needs_an_explicit_allowlist() -> None:
    with pytest.raises(ValueError, match="interventions attached"):
        build_agent(plugins=[WorkflowPlanPlugin()], interventions=[Gate()])


def test_an_explicit_allowlist_permits_attachment_alongside_interventions() -> None:
    agent = build_agent(
        plugins=[WorkflowPlanPlugin(plannable=["read_board", "consult"])],
        interventions=[Gate()],
    )
    assert "submit_workflow_plan" in agent.tool_registry.registry


def test_plain_agent_attaches_with_the_default() -> None:
    agent = build_agent(plugins=[WorkflowPlanPlugin()])
    assert "submit_workflow_plan" in agent.tool_registry.registry


# --- tool result unwrapping ---------------------------------------------------------


def tool_result(status: str, *texts: str) -> dict[str, JsonValue]:
    """Build a ToolResult the way Strands wraps one."""
    return {"status": status, "content": [{"text": t} for t in texts]}


def test_a_json_object_text_block_is_decoded() -> None:
    assert unwrap_tool_result("board", tool_result("success", '{"beast": "leshen"}')) == {
        "beast": "leshen"
    }


def test_a_plain_string_text_block_stays_a_string() -> None:
    assert unwrap_tool_result("verdict", tool_result("success", "ready")) == "ready"


def test_several_text_blocks_become_a_list() -> None:
    assert unwrap_tool_result("many", tool_result("success", "a", "b")) == ["a", "b"]


def test_an_error_status_raises_instead_of_looking_like_data() -> None:
    with pytest.raises(ToolExecutionError) as caught:
        unwrap_tool_result("scout_lair", tool_result("error", "Error: no such lair"))
    assert caught.value.tool == "scout_lair"
    assert "no such lair" in caught.value.detail


async def test_the_obvious_binding_resolves_end_to_end() -> None:
    # The regression this whole unwrap exists for: before it, `steps.board.beast` could
    # not resolve, because the result was a JSON string inside a content block.
    plugin = WorkflowPlanPlugin()
    build_agent(plugins=[plugin])

    plan = WorkflowPlan(
        steps=(
            PlanStep(id="board", tool="read_board"),
            PlanStep(id="beast", tool="consult", bind={"beast": "steps.board.beast"}),
        ),
        returns=("beast",),
    )
    result = await plugin.run_plan(plan)

    ledger = {o.id: o for o in result.ledger}
    assert ledger["board"].status is StepStatus.OK
    assert ledger["beast"].status is StepStatus.OK, ledger["beast"].error
    assert result.returned == {"beast": {"weakness": "sign-for-leshen"}}


# --- the shape a live model actually submits -----------------------------------------


async def test_a_raw_dict_plan_is_coerced_the_way_a_live_model_submits_it() -> None:
    # Regression: the SDK hands a nested Pydantic tool parameter through as the decoded
    # JSON dict, not a model instance. Every earlier test called run_plan with a typed
    # WorkflowPlan and so never exercised this, which is why `plan.steps` raised
    # AttributeError against a live model while the suite stayed green.
    plugin = WorkflowPlanPlugin()
    build_agent(plugins=[plugin])

    raw: dict[str, JsonValue] = {
        "steps": [
            {"id": "board", "tool": "read_board"},
            {"id": "beast", "tool": "consult", "bind": {"beast": "steps.board.beast"}},
        ],
        "returns": ["beast"],
        "final": True,
    }
    result = await plugin.run_plan(raw)

    assert result.returned == {"beast": {"weakness": "sign-for-leshen"}}
    assert result.final is True
    assert all(o.status is StepStatus.OK for o in result.ledger)


def test_coerce_plan_passes_a_typed_plan_through() -> None:
    plan = WorkflowPlan(steps=(PlanStep(id="a", tool="read_board"),), returns=("a",))
    assert coerce_plan(plan) is plan


def test_a_malformed_plan_reports_the_offending_field() -> None:
    with pytest.raises(InvalidPlanShapeError) as caught:
        coerce_plan({"steps": [{"id": "a"}], "returns": ["a"]})
    assert "tool" in caught.value.detail


def test_an_empty_plan_is_refused() -> None:
    with pytest.raises(InvalidPlanShapeError):
        coerce_plan({"steps": [], "returns": ["a"]})


# --- final plan ends the turn ---------------------------------------------------------


def _make_after_tools_event(agent: Agent) -> AfterToolsEvent:
    """Build the batch-completion event the loop hands to hooks."""
    return AfterToolsEvent(
        agent=agent,
        message={"role": "user", "content": []},
        invocation_state={},
    )


async def test_a_final_plan_arms_the_early_exit() -> None:
    plugin = WorkflowPlanPlugin()
    agent = build_agent(plugins=[plugin])

    raw: dict[str, JsonValue] = {
        "steps": [{"id": "board", "tool": "read_board"}],
        "returns": ["board"],
        "final": True,
    }
    await plugin.run_plan(raw)

    event = _make_after_tools_event(agent)
    plugin.end_turn_after_a_final_plan(event)
    assert event.end_turn, "a successful final plan must end the turn"
    assert "leshen" in str(event.end_turn)


async def test_a_non_final_plan_leaves_the_turn_running() -> None:
    plugin = WorkflowPlanPlugin()
    agent = build_agent(plugins=[plugin])

    raw: dict[str, JsonValue] = {
        "steps": [{"id": "board", "tool": "read_board"}],
        "returns": ["board"],
        "final": False,
    }
    await plugin.run_plan(raw)

    event = _make_after_tools_event(agent)
    plugin.end_turn_after_a_final_plan(event)
    assert event.end_turn is False


async def test_honor_final_false_keeps_the_closing_round_trip() -> None:
    plugin = WorkflowPlanPlugin(honor_final=False)
    agent = build_agent(plugins=[plugin])

    raw: dict[str, JsonValue] = {
        "steps": [{"id": "board", "tool": "read_board"}],
        "returns": ["board"],
        "final": True,
    }
    await plugin.run_plan(raw)

    event = _make_after_tools_event(agent)
    plugin.end_turn_after_a_final_plan(event)
    assert event.end_turn is False


async def test_a_failed_final_plan_does_not_end_the_turn() -> None:
    # Ending early on a partial result would hide the failure from the model, which is the
    # one thing the ledger exists to prevent.
    plugin = WorkflowPlanPlugin()
    agent = build_agent(plugins=[plugin])

    raw: dict[str, JsonValue] = {
        "steps": [
            {"id": "board", "tool": "read_board"},
            {"id": "bad", "tool": "consult", "bind": {"beast": "steps.board.nope.deeper"}},
        ],
        "returns": ["bad"],
        "final": True,
    }
    result = await plugin.run_plan(raw)
    assert any(o.status is not StepStatus.OK for o in result.ledger)

    event = _make_after_tools_event(agent)
    plugin.end_turn_after_a_final_plan(event)
    assert event.end_turn is False


async def test_the_pending_exit_is_consumed_once() -> None:
    plugin = WorkflowPlanPlugin()
    agent = build_agent(plugins=[plugin])

    raw: dict[str, JsonValue] = {
        "steps": [{"id": "board", "tool": "read_board"}],
        "returns": ["board"],
        "final": True,
    }
    await plugin.run_plan(raw)

    first = _make_after_tools_event(agent)
    plugin.end_turn_after_a_final_plan(first)
    assert first.end_turn

    second = _make_after_tools_event(agent)
    plugin.end_turn_after_a_final_plan(second)
    assert second.end_turn is False, "a later tool cycle must not inherit the decision"


def test_the_hook_is_registered_on_the_agent() -> None:
    plugin = WorkflowPlanPlugin()
    agent = build_agent(plugins=[plugin])
    callbacks = agent.hooks.get_callbacks_for(_make_after_tools_event(agent))
    assert plugin.end_turn_after_a_final_plan in list(callbacks)


def test_one_instance_refuses_a_second_agent() -> None:
    plugin = WorkflowPlanPlugin()
    build_agent(plugins=[plugin])
    with pytest.raises(ValueError, match="already attached"):
        build_agent(plugins=[plugin])


# --- loop shape --------------------------------------------------------------------


def test_the_plan_tool_is_one_ordinary_tool_among_the_others() -> None:
    agent = build_agent(plugins=[WorkflowPlanPlugin()])
    names = set(agent.tool_registry.registry)
    assert {"read_board", "consult", "needs_approval", "submit_workflow_plan"} <= names


def test_the_plan_tool_schema_hides_the_agent_and_self_params() -> None:
    agent = build_agent(plugins=[WorkflowPlanPlugin()])
    spec = agent.tool_registry.registry["submit_workflow_plan"].tool_spec
    properties = spec["inputSchema"]["json"]["properties"]
    assert "plan" in properties
    assert "self" not in properties
    assert "agent" not in properties


async def test_a_real_dependency_chain_runs_through_the_agent() -> None:
    plugin = WorkflowPlanPlugin()
    build_agent(plugins=[plugin])

    plan = WorkflowPlan(
        steps=(
            PlanStep(id="board", tool="read_board"),
            PlanStep(id="beast", tool="consult", bind={"beast": "steps.board[0].text.beast"}),
        ),
        returns=("beast",),
        final=True,
    )
    result = await plugin.run_plan(plan)

    ledger = {o.id: o for o in result.ledger}
    assert ledger["board"].status is StepStatus.OK
    assert result.levels == 2
    assert result.inference_passes_saved == 1
    assert result.final is True
