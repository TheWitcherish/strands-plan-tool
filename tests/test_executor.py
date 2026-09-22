"""Execution: dependency data flow, aggregation, concurrency, and failure policy."""

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
)


class RecordingInvoker:
    """Fake tool runner that records call order and returns canned values."""

    def __init__(self, table: Mapping[str, JsonValue], *, delay: float = 0.0) -> None:
        self.table = dict(table)
        self.delay = delay
        self.calls: list[tuple[str, dict[str, JsonValue]]] = []

    async def __call__(self, name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        self.calls.append((name, dict(args)))
        if self.delay:
            await asyncio.sleep(self.delay)
        if name == "boom":
            raise RuntimeError("tool exploded")
        return self.table.get(name)


async def test_binding_carries_a_result_into_the_next_step() -> None:
    invoker = RecordingInvoker({"lookup": {"id": 42}, "fetch": {"ok": True}})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="lookup", tool="lookup"),
            PlanStep(id="fetch", tool="fetch", bind={"target": "steps.lookup.id"}),
        ),
        returns=("fetch",),
    )
    result = await execute_plan(plan, invoker)

    assert invoker.calls[1] == ("fetch", {"target": 42})
    assert result.returned == {"fetch": {"ok": True}}
    assert result.levels == 2
    assert result.inference_passes_saved == 1


async def test_only_returns_travel_back() -> None:
    invoker = RecordingInvoker({"big": {"rows": [1, 2, 3]}, "small": "summary"})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="big", tool="big"),
            PlanStep(id="small", tool="small", bind={"n": "$count(steps.big.rows)"}),
        ),
        returns=("small",),
    )
    result = await execute_plan(plan, invoker)

    assert result.returned == {"small": "summary"}
    assert "big" not in result.returned
    assert invoker.calls[1] == ("small", {"n": 3})
    assert len(result.ledger) == 2


async def test_bindings_win_over_literal_args() -> None:
    invoker = RecordingInvoker({"a": {"v": 9}, "b": None})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="a", tool="a"),
            PlanStep(id="b", tool="b", args={"x": 1}, bind={"x": "steps.a.v"}),
        ),
        returns=("b",),
    )
    await execute_plan(plan, invoker)
    assert invoker.calls[1][1] == {"x": 9}


async def test_same_level_steps_run_concurrently() -> None:
    invoker = RecordingInvoker({"a": 1, "b": 2, "c": 3}, delay=0.05)
    plan = WorkflowPlan(
        steps=(PlanStep(id="a", tool="a"), PlanStep(id="b", tool="b"), PlanStep(id="c", tool="c")),
        returns=("a", "b", "c"),
    )
    started = asyncio.get_running_loop().time()
    result = await execute_plan(plan, invoker)
    span = asyncio.get_running_loop().time() - started

    assert result.levels == 1
    assert span < 0.12, f"three 50ms tools took {span:.3f}s, so they were not concurrent"


async def test_fail_fast_skips_the_rest_and_reports_every_step() -> None:
    invoker = RecordingInvoker({"a": 1})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="a", tool="a"),
            PlanStep(id="b", tool="boom", bind={"x": "steps.a"}),
            PlanStep(id="c", tool="c", bind={"y": "steps.b"}),
        ),
        returns=("c",),
        on_error=OnError.FAIL_FAST,
    )
    result = await execute_plan(plan, invoker)

    by_id = {o.id: o for o in result.ledger}
    assert by_id["a"].status is StepStatus.OK
    assert by_id["b"].status is StepStatus.FAILED
    assert by_id["c"].status is StepStatus.SKIPPED
    assert result.returned == {}
    assert by_id["b"].error is not None and "tool exploded" in by_id["b"].error


async def test_continue_policy_keeps_going_past_a_failure() -> None:
    invoker = RecordingInvoker({"a": 1, "c": 3})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="a", tool="a"),
            PlanStep(id="b", tool="boom"),
            PlanStep(id="c", tool="c", bind={"x": "steps.a"}),
        ),
        returns=("c",),
        on_error=OnError.CONTINUE,
    )
    result = await execute_plan(plan, invoker)

    by_id = {o.id: o for o in result.ledger}
    assert by_id["b"].status is StepStatus.FAILED
    assert by_id["c"].status is StepStatus.OK
    assert result.returned == {"c": 3}


async def test_a_broken_binding_fails_the_step_instead_of_passing_none() -> None:
    invoker = RecordingInvoker({"a": {"v": 1}})
    plan = WorkflowPlan(
        steps=(
            PlanStep(id="a", tool="a"),
            PlanStep(id="b", tool="b", bind={"x": '$eval("1+1")'}),
        ),
        returns=("b",),
    )
    result = await execute_plan(plan, invoker)

    by_id = {o.id: o for o in result.ledger}
    assert by_id["b"].status is StepStatus.FAILED
    assert by_id["b"].error is not None and "not permitted" in by_id["b"].error
    assert [name for name, _ in invoker.calls] == ["a"]


async def test_final_flag_is_carried_through() -> None:
    invoker = RecordingInvoker({"a": 1})
    plan = WorkflowPlan(steps=(PlanStep(id="a", tool="a"),), returns=("a",), final=True)
    result = await execute_plan(plan, invoker)
    assert result.final is True
    assert result.inference_passes_saved == 0
