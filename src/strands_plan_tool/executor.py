"""Local plan execution, scheduled for latency.

Tool invocation sits behind :class:`ToolInvoker`, so this engine has no agent-framework
import and the Strands adapter is a thin wrapper. That seam is also why the latency claim
holds: steps execute in this process, with no hop back to the model provider between them.

Why this scheduler
------------------
The plan is a **DAG**, not a tree: a step may have several dependencies and several
dependents, so in-degree above one is normal and tree structures (BST, heap) are the wrong
shape. A heap would only buy priority ordering, which does not shorten the critical path.

Ordering therefore comes from Kahn's algorithm over in-degree counts, which is
BFS-flavoured. A DFS topological sort is the one choice to avoid: it yields a single linear
sequence, and executing a sequence serialises steps that never depended on each other, so
wall clock becomes the *sum* of every step instead of the longest path.

:func:`~strands_plan_tool.graph.plan_levels` groups steps into levels, and levels are the right
way to *describe* a plan -- the level count is its depth, and that is the number of model
round trips the plan replaces. But levels are the wrong way to *run* one: a level barrier
makes every step in level N+1 wait for the slowest step in level N, even when it never
depended on it. A 2000ms step alongside a 5ms step delays the 5ms step's dependent by
1995ms for nothing.

So execution is barrier-free and dependency-triggered: every step awaits exactly its own
dependencies and starts the instant its last one lands. The event loop is the scheduler,
each step is one task, and each dependency edge is one await -- O(1) amortised per step and
per edge, O(V + E) overall, the same complexity as the level walk but with wall clock bound
by the critical path rather than by the sum of per-level maxima.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, assert_never, runtime_checkable

from .binder import Binder, JsonataBinder
from .errors import BindEvaluationError
from .graph import dependencies_of, plan_levels
from .models import (
    MAX_STEPS,
    JsonValue,
    OnError,
    PlanResult,
    PlanStep,
    StepOutcome,
    StepStatus,
    WorkflowPlan,
)

__all__ = ["ToolInvoker", "execute_plan", "levels_of", "summarize"]


@runtime_checkable
class ToolInvoker(Protocol):
    """Runs one named tool with resolved arguments."""

    async def __call__(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        """Invoke ``name``.

        Args:
            name: Registered tool name.
            args: Fully resolved arguments; the invoker must not mutate them.

        Returns:
            The tool's result as a JSON-compatible value.
        """
        ...


async def execute_plan(
    plan: WorkflowPlan,
    invoker: ToolInvoker,
    *,
    binder: Binder | None = None,
    max_steps: int = MAX_STEPS,
    allowed_tools: Iterable[str] | None = None,
) -> PlanResult:
    """Validate a plan, run it with barrier-free dispatch, and aggregate one result.

    Each step starts the moment its own dependencies complete rather than waiting for a
    level to finish, so wall clock tracks the critical path. Per-step hot-path cost is
    O(1): one dict write for the result, one dict hit for the compiled expression, and no
    rescan of the graph or of prior results.

    Args:
        plan: The plan to execute.
        invoker: Callable that runs a single tool.
        binder: Expression evaluator. Defaults to :class:`~strands_plan_tool.binder.JsonataBinder`
            with ``$eval`` disabled.
        max_steps: Ceiling on step count.
        allowed_tools: Tool names this plan may call; ``None`` disables the check.

    Returns:
        Only the steps named in ``plan.returns``, plus a ledger covering every step in
        level order.

    Raises:
        strands_plan_tool.errors.PlanError: The plan is malformed or names a tool it may not
            call. Nothing was executed.
    """
    levels = plan_levels(plan, max_steps=max_steps, allowed_tools=allowed_tools)
    depth_of = {step.id: depth for depth, level in enumerate(levels) for step in level}
    deps_of = {step.id: dependencies_of(step) for step in plan.steps}
    resolver = binder if binder is not None else JsonataBinder()

    # Single owner: `results` is owned by this call, and `env` is a deliberate read-only
    # alias of it built once rather than per step, keeping binding O(1) instead of copying
    # every prior result for every step.
    #
    # A plain dict is safe under this concurrency because asyncio is cooperative and
    # `_resolve_args` reads `env` synchronously -- there is no await between reading a
    # result and using it, so no other step can interleave a write mid-read.
    results: dict[str, JsonValue] = {}
    env: dict[str, JsonValue] = {"steps": results}

    outcomes: dict[str, StepOutcome] = {}
    tasks: dict[str, asyncio.Task[None]] = {}
    halted = asyncio.Event()

    def skipped(step: PlanStep) -> StepOutcome:
        return StepOutcome(
            id=step.id,
            tool=step.tool,
            status=StepStatus.SKIPPED,
            level=depth_of[step.id],
            duration_ms=0.0,
        )

    async def run(step: PlanStep) -> None:
        """Await only this step's dependencies, then run it. Never raises."""
        dependencies = deps_of[step.id]
        for dependency in dependencies:
            await tasks[dependency]

        unusable = any(outcomes[d].status is not StepStatus.OK for d in dependencies)
        if halted.is_set() or unusable:
            outcomes[step.id] = skipped(step)
            return

        outcome, value = await _run_step(step, depth_of[step.id], invoker, resolver, env)
        outcomes[step.id] = outcome

        if outcome.status is StepStatus.OK:
            results[step.id] = value
            return

        match plan.on_error:
            case OnError.FAIL_FAST:
                halted.set()
            case OnError.CONTINUE:
                pass
            case _ as unreachable:
                assert_never(unreachable)

    # Spawned in level order purely so every `tasks[dependency]` exists before it is
    # awaited; the tasks themselves synchronise on dependencies, not on levels.
    for level in levels:
        for step in level:
            tasks[step.id] = asyncio.create_task(run(step))

    await asyncio.gather(*tasks.values())

    ledger = tuple(outcomes[step.id] for level in levels for step in level)
    return PlanResult(
        returned={sid: results[sid] for sid in plan.returns if sid in results},
        ledger=ledger,
        levels=len(levels),
        inference_passes_saved=max(len(levels) - 1, 0),
        final=plan.final,
    )


async def _run_step(
    step: PlanStep,
    depth: int,
    invoker: ToolInvoker,
    binder: Binder,
    env: Mapping[str, JsonValue],
) -> tuple[StepOutcome, JsonValue]:
    """Resolve one step's arguments and invoke its tool.

    A binding failure is a step failure, never a substituted ``None``: handing a tool an
    argument the plan did not intend is worse than reporting that the plan was wrong.

    Args:
        step: The step to run.
        depth: Level index, recorded on the outcome for reporting.
        invoker: Tool runner.
        binder: Expression evaluator.
        env: Read-only binding environment shaped ``{"steps": {...}}``.

    Returns:
        The outcome, and the tool's value when it succeeded.
    """
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000.0

    def failure(reason: str) -> tuple[StepOutcome, JsonValue]:
        return (
            StepOutcome(
                id=step.id,
                tool=step.tool,
                status=StepStatus.FAILED,
                level=depth,
                duration_ms=elapsed(),
                error=reason,
            ),
            None,
        )

    try:
        args = _resolve_args(step, binder, env)
    except BindEvaluationError as exc:
        return failure(str(exc))

    try:
        value = await invoker(step.tool, args)
    except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
        return failure(f"{type(exc).__name__}: {exc}")

    return (
        StepOutcome(
            id=step.id,
            tool=step.tool,
            status=StepStatus.OK,
            level=depth,
            duration_ms=elapsed(),
        ),
        value,
    )


def _resolve_args(
    step: PlanStep,
    binder: Binder,
    env: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Merge literal args with evaluated bindings, bindings winning.

    Args:
        step: The step whose arguments to build.
        binder: Expression evaluator.
        env: Read-only binding environment shaped ``{"steps": {...}}``.

    Returns:
        A fresh dict; the step's own ``args`` is never mutated.

    Raises:
        BindEvaluationError: An expression failed to compile or evaluate.
    """
    resolved = dict(step.args)

    for parameter, expression in step.bind.items():
        try:
            resolved[parameter] = binder.bind(expression, env)
        except Exception as exc:
            raise BindEvaluationError(step.id, parameter, expression, str(exc)) from exc

    return resolved


def summarize(result: PlanResult) -> str:
    """Render a one-line-per-step ledger for logs and demos.

    Args:
        result: A completed plan result.

    Returns:
        Human-readable multi-line summary.
    """
    lines: list[str] = []
    for outcome in result.ledger:
        match outcome.status:
            case StepStatus.OK:
                mark = "ok     "
            case StepStatus.FAILED:
                mark = "FAILED "
            case StepStatus.SKIPPED:
                mark = "skipped"
            case _ as unreachable:
                assert_never(unreachable)
        detail = f"  {outcome.error}" if outcome.error else ""
        lines.append(
            f"  L{outcome.level} {mark} {outcome.id:<14} {outcome.tool:<20}"
            f" {outcome.duration_ms:7.1f}ms{detail}"
        )
    return "\n".join(lines)


def levels_of(plan: WorkflowPlan) -> Sequence[Sequence[PlanStep]]:
    """Expose validated levels without executing, for inspection and tests.

    Args:
        plan: The plan to lay out.

    Returns:
        Levels in execution order.

    Raises:
        strands_plan_tool.errors.PlanError: The plan is malformed.
    """
    return plan_levels(plan)
