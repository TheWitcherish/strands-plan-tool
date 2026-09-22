"""Validation and layering: every rejection path plus implicit dependency inference."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strands_plan_tool import (
    MAX_STEPS,
    DanglingReferenceError,
    DuplicateStepError,
    JsonValue,
    PlanCycleError,
    PlanStep,
    StepLimitError,
    ToolNotPlannableError,
    WorkflowPlan,
    dependencies_of,
    plan_levels,
)


def step(
    sid: str,
    tool: str = "t",
    *,
    args: dict[str, JsonValue] | None = None,
    bind: dict[str, str] | None = None,
    after: tuple[str, ...] = (),
) -> PlanStep:
    """Build a step without repeating the empty-collection defaults in every test."""
    return PlanStep(id=sid, tool=tool, args=args or {}, bind=bind or {}, after=after)


def test_binding_reference_creates_an_implicit_edge() -> None:
    s = step("b", bind={"x": "steps.a.value"})
    assert dependencies_of(s) == frozenset({"a"})


def test_explicit_after_and_binding_edges_union() -> None:
    s = step("c", after=("a",), bind={"x": "steps.b.value + 1"})
    assert dependencies_of(s) == frozenset({"a", "b"})


def test_self_reference_is_dropped_not_treated_as_a_cycle() -> None:
    s = step("a", bind={"x": "steps.a.value"})
    assert dependencies_of(s) == frozenset()


def test_chain_lays_out_one_step_per_level() -> None:
    plan = WorkflowPlan(
        steps=(
            step("a"),
            step("b", bind={"x": "steps.a.v"}),
            step("c", bind={"y": "steps.b.v"}),
        ),
        returns=("c",),
    )
    levels = plan_levels(plan)
    assert [[s.id for s in lvl] for lvl in levels] == [["a"], ["b"], ["c"]]


def test_independent_steps_share_one_level_in_declaration_order() -> None:
    plan = WorkflowPlan(steps=(step("b"), step("a"), step("c")), returns=("a",))
    levels = plan_levels(plan)
    assert len(levels) == 1
    assert [s.id for s in levels[0]] == ["b", "a", "c"]


def test_lookup_then_fanout_is_two_levels() -> None:
    plan = WorkflowPlan(
        steps=(
            step("ids", tool="list_ids"),
            step("one", tool="fetch", bind={"i": "steps.ids.items[0]"}),
            step("two", tool="fetch", bind={"i": "steps.ids.items[1]"}),
        ),
        returns=("one", "two"),
    )
    levels = plan_levels(plan)
    assert [[s.id for s in lvl] for lvl in levels] == [["ids"], ["one", "two"]]


def test_cycle_is_rejected_and_names_the_blocked_steps() -> None:
    plan = WorkflowPlan(
        steps=(
            step("a", bind={"x": "steps.b.v"}),
            step("b", bind={"x": "steps.a.v"}),
        ),
        returns=("a",),
    )
    with pytest.raises(PlanCycleError) as caught:
        plan_levels(plan)
    assert set(caught.value.remaining) == {"a", "b"}


def test_dangling_dependency_is_rejected() -> None:
    plan = WorkflowPlan(steps=(step("a", bind={"x": "steps.nope.v"}),), returns=("a",))
    with pytest.raises(DanglingReferenceError) as caught:
        plan_levels(plan)
    assert caught.value.missing == "nope"


def test_dangling_returns_entry_is_rejected() -> None:
    plan = WorkflowPlan(steps=(step("a"),), returns=("ghost",))
    with pytest.raises(DanglingReferenceError) as caught:
        plan_levels(plan)
    assert caught.value.field == "returns"


def test_duplicate_ids_are_rejected() -> None:
    plan = WorkflowPlan(steps=(step("a"), step("a", tool="other")), returns=("a",))
    with pytest.raises(DuplicateStepError):
        plan_levels(plan)


def test_step_limit_is_enforced() -> None:
    plan = WorkflowPlan(steps=tuple(step(f"s{i}") for i in range(5)), returns=("s0",))
    with pytest.raises(StepLimitError) as caught:
        plan_levels(plan, max_steps=4)
    assert caught.value.limit == 4


def test_default_step_limit_still_applies_after_the_schema_bound_was_relaxed() -> None:
    # The cap moved out of the schema so max_steps is configurable; the default policy
    # ceiling must still reject an oversized plan before anything runs.
    plan = WorkflowPlan(steps=tuple(step(f"s{i}") for i in range(MAX_STEPS + 1)), returns=("s0",))
    with pytest.raises(StepLimitError) as caught:
        plan_levels(plan)
    assert caught.value.limit == MAX_STEPS


def test_tool_allowlist_rejects_an_unlisted_tool() -> None:
    plan = WorkflowPlan(steps=(step("a", tool="danger"),), returns=("a",))
    with pytest.raises(ToolNotPlannableError) as caught:
        plan_levels(plan, allowed_tools={"safe"})
    assert caught.value.tool == "danger"
    assert caught.value.allowed == ("safe",)


def test_tool_allowlist_is_skipped_when_none() -> None:
    plan = WorkflowPlan(steps=(step("a", tool="anything"),), returns=("a",))
    assert len(plan_levels(plan, allowed_tools=None)) == 1


def test_models_are_frozen() -> None:
    s = step("a")
    with pytest.raises(ValidationError):
        s.id = "b"


def test_unknown_field_is_refused_at_the_boundary() -> None:
    # Validated from a dict, because that is how a plan actually arrives: JSON the
    # model wrote. extra="forbid" is what stops a hallucinated field passing silently.
    with pytest.raises(ValidationError):
        PlanStep.model_validate({"id": "a", "tool": "t", "surprise": 1})


def test_malformed_step_id_is_refused() -> None:
    with pytest.raises(ValidationError):
        PlanStep(id="Not-Valid", tool="t")
