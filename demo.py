"""A depth-3 contract chain, executed from one plan.

Run it:
    uv run demo.py

Each tool sleeps 120ms to stand in for a real network call, and the "inference pass"
line stands in for a model round trip. Watch the level count: three dependent steps,
one plan, one trip back to the model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from strands_plan_tool import JsonValue, PlanStep, WorkflowPlan, execute_plan, summarize

TOOL_LATENCY_S = 0.12

BOARD: dict[str, JsonValue] = {
    "drowner": {"weakness": "Aard", "oil": "Necrophage oil", "bounty": 120},
    "griffin": {"weakness": "Igni", "oil": "Hybrid oil", "bounty": 340},
    "leshen": {"weakness": "Yrden", "oil": "Cursed oil", "bounty": 800},
}


async def contract_board(name: str, args: Mapping[str, JsonValue]) -> JsonValue:
    """Stand-in tool registry: a notice board, a bestiary, and a kit planner."""
    await asyncio.sleep(TOOL_LATENCY_S)
    match name:
        case "read_notice_board":
            return {"contracts": [{"beast": "leshen", "village": "Blackbough"}]}
        case "consult_bestiary":
            beast = args["beast"]
            if not isinstance(beast, str):
                raise TypeError(f"beast must be a name, got {type(beast).__name__}")
            entry = BOARD.get(beast)
            if entry is None:
                raise KeyError(f"no bestiary entry for {beast!r}")
            return entry
        case "pack_satchel":
            return {"sign": args["sign"], "oil": args["oil"], "verdict": "ready"}
        case _:
            raise KeyError(f"unknown tool {name!r}")


PLAN = WorkflowPlan(
    steps=(
        PlanStep(id="board", tool="read_notice_board"),
        PlanStep(
            id="beast",
            tool="consult_bestiary",
            bind={"beast": "steps.board.contracts[0].beast"},
        ),
        PlanStep(
            id="satchel",
            tool="pack_satchel",
            bind={"sign": "steps.beast.weakness", "oil": "steps.beast.oil"},
        ),
    ),
    returns=("satchel",),
    final=True,
)


async def main() -> None:
    """Execute the plan and print the latency accounting."""
    print("The contract:  one notice board, one unknown beast, one satchel to pack.\n")
    print(f"Plan: {len(PLAN.steps)} steps, each tool costs {TOOL_LATENCY_S * 1000:.0f}ms\n")

    started = asyncio.get_running_loop().time()
    result = await execute_plan(PLAN, contract_board)
    wall_ms = (asyncio.get_running_loop().time() - started) * 1000.0

    print(summarize(result))
    print(f"\n  chain depth (levels):        {result.levels}")
    print("  model round trips used:      1")
    print(f"  inference passes avoided:    {result.inference_passes_saved}")
    print(f"  wall clock:                  {wall_ms:.1f}ms")
    print(f"\n  returned to the model:       {result.returned}")

    hidden = len(result.ledger) - len(result.returned)
    print(f"  results the model never saw: {hidden}")
    print(
        "\nThe board and the bestiary entry did the work and never reached the model."
        "\nIt asked for a satchel and got a satchel."
    )


if __name__ == "__main__":
    asyncio.run(main())
