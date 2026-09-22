"""Scheduling behaviour: barrier-free dispatch, and why levels are not enough.

The load-bearing test is ``test_a_slow_step_does_not_delay_an_independent_chain``: under
level-synchronous execution it fails, because level 1 would wait for the slow step in
level 0 that it never depended on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from strands_plan_tool import (
    JsonValue,
    OnError,
    PlanStep,
    StepStatus,
    WorkflowPlan,
    execute_plan,
    plan_levels,
)

SLOW_S = 0.30
FAST_S = 0.02


class TimedInvoker:
    """Tool runner with per-tool durations and a completion timeline."""

    def __init__(self, durations: Mapping[str, float]) -> None:
        self.durations = dict(durations)
        self.finished_at: dict[str, float] = {}
        self._origin = 0.0

    async def __call__(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        loop = asyncio.get_running_loop()
        if not self._origin:
            self._origin = loop.time()
        await asyncio.sleep(self.durations.get(name, 0.0))
        self.finished_at[name] = loop.time() - self._origin
        return {"tool": name, "args": dict(args)}


async def test_a_slow_step_does_not_delay_an_independent_chain() -> None:
    # slow and quick both sit at level 0; dependent depends ONLY on quick.
    # A level barrier would make dependent wait for slow. Barrier-free must not.
    invoker = TimedInvoker({"slow": SLOW_S, "quick": FAST_S, "dependent": FAST_S})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="slow", tool="slow"),
            PlanStep(id="quick", tool="quick"),
            PlanStep(id="dependent", tool="dependent", bind={"x": "steps.quick.tool"}),
        ),
        returns=("dependent",),
    )

    levels = plan_levels(plan)
    assert [[s.id for s in lvl] for lvl in levels] == [["slow", "quick"], ["dependent"]]

    result = await execute_plan(plan, invoker)

    assert result.returned != {}
    dependent_done = invoker.finished_at["dependent"]
    assert dependent_done < SLOW_S, (
        f"dependent finished at {dependent_done:.3f}s, so it waited on the unrelated "
        f"{SLOW_S}s step -- execution is level-synchronous, not dependency-triggered"
    )


async def test_wall_clock_tracks_the_critical_path_not_the_sum() -> None:
    invoker = TimedInvoker({"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="a", tool="a"),
            PlanStep(id="b", tool="b"),
            PlanStep(id="c", tool="c"),
            PlanStep(id="d", tool="d"),
        ),
        returns=("a", "b", "c", "d"),
    )
    started = asyncio.get_running_loop().time()
    result = await execute_plan(plan, invoker)
    span = asyncio.get_running_loop().time() - started

    assert result.levels == 1
    assert span < 0.25, f"four independent 100ms tools took {span:.3f}s, so not concurrent"


async def test_only_the_dependent_subtree_is_skipped_under_continue() -> None:
    # `orphan` shares a level with the failing step's dependent but depends on nothing,
    # so a correct scheduler still runs it.
    async def invoker(name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        if name == "boom":
            raise RuntimeError("tool exploded")
        return name

    plan = WorkflowPlan(
        steps=(
            PlanStep(id="bad", tool="boom"),
            PlanStep(id="downstream", tool="ok", bind={"x": "steps.bad"}),
            PlanStep(id="orphan", tool="ok"),
        ),
        returns=("orphan",),
        on_error=OnError.CONTINUE,
    )
    result = await execute_plan(plan, invoker)

    by_id = {o.id: o for o in result.ledger}
    assert by_id["bad"].status is StepStatus.FAILED
    assert by_id["downstream"].status is StepStatus.SKIPPED
    assert by_id["orphan"].status is StepStatus.OK
    assert result.returned == {"orphan": "ok"}


async def test_ledger_is_reported_in_level_order() -> None:
    invoker = TimedInvoker({"slow": SLOW_S, "quick": FAST_S})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="slow", tool="slow"),
            PlanStep(id="quick", tool="quick"),
            PlanStep(id="last", tool="quick", bind={"x": "steps.slow.tool"}),
        ),
        returns=("last",),
    )
    result = await execute_plan(plan, invoker)
    # Deterministic report order despite out-of-order completion.
    assert [o.id for o in result.ledger] == ["slow", "quick", "last"]
    assert [o.level for o in result.ledger] == [0, 0, 1]


async def test_deep_chain_layers_without_quadratic_blowup() -> None:
    # 4000 steps: O(V^2) layering would do ~16M subset checks and crawl.
    # Kahn over in-degree counts is O(V + E) and stays fast.
    depth = 4000
    steps = [PlanStep(id="s0", tool="noop")]
    steps.extend(
        PlanStep(id=f"s{i}", tool="noop", bind={"x": f"steps.s{i - 1}.tool"})
        for i in range(1, depth)
    )
    plan = WorkflowPlan(steps=tuple(steps), returns=(f"s{depth - 1}",))

    started = asyncio.get_running_loop().time()
    levels = plan_levels(plan, max_steps=depth)
    span = asyncio.get_running_loop().time() - started

    assert len(levels) == depth
    assert all(len(level) == 1 for level in levels)
    assert span < 1.0, f"layering {depth} steps took {span:.3f}s, which is superlinear"
