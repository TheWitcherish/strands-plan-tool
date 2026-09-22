"""Validation and topological layering.

A plan is checked completely before any tool runs: duplicate ids, dangling references,
non-plannable tools, and cycles are all rejected up front, so a malformed plan costs one
cheap repair round trip instead of half-executing and leaving side effects behind.

Dependencies come from two places. ``PlanStep.after`` is explicit. Anything a binding
reads as ``steps.<id>`` is inferred, so the model does not have to declare the same
edge twice. Inference covers the dotted form only; a binding that reaches a result by
some other syntax must name the edge in ``after``.

Layering is Kahn's algorithm over an indegree count, not a depth-first traversal. DFS
yields a single linear order, which would serialise steps that have no dependency on
each other; Kahn yields *levels*, and every step in a level can run concurrently. Since
the point of a plan is latency, levels are the whole product: the level count is the
chain depth, and it is exactly the number of model round trips the plan replaces.

Cost: each step is enqueued exactly once and each edge is decremented exactly once, so
the work is O(1) amortised per step and per edge, O(V + E) overall, plus O(V log V) total
for keeping each level in declaration order so layering is deterministic. O(V + E) is
optimal -- a plan cannot be validated without reading it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from .errors import (
    DanglingReferenceError,
    DuplicateStepError,
    PlanCycleError,
    StepLimitError,
    ToolNotPlannableError,
)
from .models import MAX_STEPS, PlanStep, WorkflowPlan

__all__ = ["dependencies_of", "plan_levels"]

_STEP_REF = re.compile(r"\bsteps\s*\.\s*([a-z][a-z0-9_]{0,31})\b")


def dependencies_of(step: PlanStep) -> frozenset[str]:
    """Collect every step id this step must wait for.

    Args:
        step: The step to inspect.

    Returns:
        Explicit ``after`` ids unioned with ids inferred from its binding expressions.
        Self-references are dropped so a typo becomes a dangling reference rather than
        a one-node cycle.
    """
    found = set(step.after)
    for expression in step.bind.values():
        found.update(_STEP_REF.findall(expression))
    found.discard(step.id)
    return frozenset(found)


def plan_levels(
    plan: WorkflowPlan,
    *,
    max_steps: int = MAX_STEPS,
    allowed_tools: Iterable[str] | None = None,
) -> tuple[tuple[PlanStep, ...], ...]:
    """Validate a plan and group its steps into dependency levels.

    Args:
        plan: The plan to validate.
        max_steps: Ceiling on step count.
        allowed_tools: Tool names this plan may call. ``None`` disables the check;
            pass a collection to enforce an allowlist. Enforced here rather than at
            execution time because some tools cannot be safely invoked from inside a
            plan at all, and discovering that mid-flight is already too late.

    Returns:
        Levels in execution order, innermost first.

    Raises:
        StepLimitError: The plan declares more steps than ``max_steps``.
        DuplicateStepError: Two steps share an id.
        ToolNotPlannableError: A step names a tool outside ``allowed_tools``.
        DanglingReferenceError: A dependency or a ``returns`` entry names no step.
        PlanCycleError: The graph has no topological order.
    """
    if len(plan.steps) > max_steps:
        raise StepLimitError(len(plan.steps), max_steps)

    by_id: dict[str, PlanStep] = {}
    for step in plan.steps:
        if step.id in by_id:
            raise DuplicateStepError(step.id)
        by_id[step.id] = step

    if allowed_tools is not None:
        permitted = frozenset(allowed_tools)
        for step in plan.steps:
            if step.tool not in permitted:
                raise ToolNotPlannableError(step.id, step.tool, tuple(sorted(permitted)))

    deps_of: dict[str, frozenset[str]] = {}
    for step in plan.steps:
        deps = dependencies_of(step)
        for dep in sorted(deps):
            if dep not in by_id:
                raise DanglingReferenceError(step.id, dep, field="dependencies")
        deps_of[step.id] = deps

    for returned in plan.returns:
        if returned not in by_id:
            raise DanglingReferenceError(returned, returned, field="returns")

    return _layer(by_id, deps_of, plan.steps)


def _layer(
    by_id: Mapping[str, PlanStep],
    deps_of: Mapping[str, frozenset[str]],
    declared_order: Sequence[PlanStep],
) -> tuple[tuple[PlanStep, ...], ...]:
    """Kahn layering over indegree counts, declaration order preserved per level.

    Each edge is relaxed exactly once and each step enters the frontier exactly once,
    so per-step and per-edge work is O(1) amortised.

    Args:
        by_id: Steps indexed by id.
        deps_of: Dependency ids per step.
        declared_order: Steps as the model wrote them, so a level is deterministic
            rather than set-ordered.

    Returns:
        Levels in execution order.

    Raises:
        PlanCycleError: Some steps never reached indegree zero.
    """
    position = {step.id: index for index, step in enumerate(declared_order)}
    indegree = {sid: len(deps) for sid, deps in deps_of.items()}

    successors: dict[str, list[str]] = {sid: [] for sid in by_id}
    for sid, deps in deps_of.items():
        for dep in deps:
            successors[dep].append(sid)

    frontier = sorted(
        (sid for sid, count in indegree.items() if count == 0), key=position.__getitem__
    )
    levels: list[tuple[PlanStep, ...]] = []
    settled = 0

    while frontier:
        levels.append(tuple(by_id[sid] for sid in frontier))
        settled += len(frontier)

        following: list[str] = []
        for sid in frontier:
            for successor in successors[sid]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    following.append(successor)

        following.sort(key=position.__getitem__)
        frontier = following

    if settled != len(by_id):
        raise PlanCycleError(tuple(sid for sid, count in indegree.items() if count > 0))

    return tuple(levels)
