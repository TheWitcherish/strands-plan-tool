"""Smoke test run against a built distribution, not the source tree.

The release workflow installs the wheel and then the sdist in isolation and runs this,
so it catches the packaging failures a normal test run cannot: a module left out of the
wheel, a missing ``py.typed``, an import that only resolves because the source tree
happens to be on the path.

It must therefore import ONLY this package and its declared runtime dependencies --
never ``strands`` (an optional extra) and never anything from ``dev``.

Guarded under ``__main__`` so pytest can collect the file without executing it: the name
matches pytest's ``*_test.py`` pattern, and module-level work would run at import time.
"""

import asyncio
import pathlib
from collections.abc import Mapping

from strands_plan_tool import (
    JsonValue,
    PlanStep,
    StepStatus,
    WorkflowPlan,
    execute_plan,
)


async def check() -> None:
    """Execute a two-step dependent plan and assert the binding carried a value."""
    calls: list[tuple[str, dict[str, JsonValue]]] = []

    async def invoker(name: str, args: Mapping[str, JsonValue]) -> JsonValue:
        calls.append((name, dict(args)))
        return {"id": 7} if name == "lookup" else {"ok": True}

    plan = WorkflowPlan(
        steps=(
            PlanStep(id="lookup", tool="lookup"),
            PlanStep(id="fetch", tool="fetch", bind={"target": "steps.lookup.id"}),
        ),
        returns=("fetch",),
    )
    result = await execute_plan(plan, invoker)

    assert calls[1] == ("fetch", {"target": 7}), f"binding did not resolve: {calls}"
    assert result.returned == {"fetch": {"ok": True}}, result.returned
    assert all(o.status is StepStatus.OK for o in result.ledger), result.ledger
    assert result.levels == 2, result.levels
    assert result.inference_passes_saved == 1, result.inference_passes_saved


def main() -> None:
    """Run every packaging assertion and report."""
    import strands_plan_tool as pkg

    marker = pathlib.Path(pkg.__file__).parent / "py.typed"
    assert marker.exists(), "py.typed missing from the distribution"
    assert len(pkg.__all__) > 20, f"__all__ looks truncated: {len(pkg.__all__)}"

    asyncio.run(check())
    print(f"smoke test passed: {pkg.__name__}, {len(pkg.__all__)} exports, py.typed present")


if __name__ == "__main__":
    main()
