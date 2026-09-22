"""Resolve one RPG combat round: twelve tool calls, one model round trip.

Offline (no credentials, deterministic):
    uv run python -m examples.game_master

Live, letting the model author its own plan:
    AWS_PROFILE=... AWS_REGION=... uv run python -m examples.game_master --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections.abc import Mapping
from typing import Final

from examples.rpg import (
    GAME_MASTER_PLAN,
    get_character,
    narrate_round,
    read_action_log,
    resolve_action,
    roll_check,
    tally_round,
)
from strands_plan_tool import JsonValue, StepStatus, execute_plan, levels_of, summarize

# Verified ACTIVE in account REDACTED / eu-central-1 on 2026-09-22 via
# `aws bedrock list-inference-profiles`. Model ids are perishable -- re-verify, or --model.
DEFAULT_MODEL: Final = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

TOOL_NAMES: Final = (
    "read_action_log",
    "get_character",
    "roll_check",
    "resolve_action",
    "tally_round",
    "narrate_round",
)


def _damages(value: JsonValue) -> list[int]:
    """Coerce a bound JSONata array into the list of ints the tool declares.

    Args:
        value: Whatever the binding produced.

    Returns:
        The damage values.

    Raises:
        TypeError: The binding did not produce a list of ints.
    """
    if not isinstance(value, list) or not all(isinstance(v, int) for v in value):
        raise TypeError(f"damages must be a list of ints, got {value!r}")
    return [v for v in value if isinstance(v, int)]


async def local_invoker(name: str, args: Mapping[str, JsonValue]) -> JsonValue:
    """Dispatch one domain function, with no agent framework involved.

    Dispatched by ``match`` rather than a callable registry so each argument is checked at
    the boundary. A registry of heterogeneous callables would need ``Any`` and would pass a
    wrongly-typed binding straight through to the tool.

    Args:
        name: Tool name from the plan.
        args: Resolved arguments.

    Returns:
        The tool's result.

    Raises:
        KeyError: The plan named a tool that is not registered.
    """
    match name:
        case "read_action_log":
            return await read_action_log()
        case "get_character":
            return await get_character(str(args["name"]))
        case "roll_check":
            return await roll_check(
                actor=str(args["actor"]),
                skill=str(args["skill"]),
                modifier=int(str(args["modifier"])),
            )
        case "resolve_action":
            return await resolve_action(
                actor=str(args["actor"]),
                success=bool(args["success"]),
                intent=str(args["intent"]),
            )
        case "tally_round":
            return await tally_round(damages=_damages(args["damages"]), target=str(args["target"]))
        case "narrate_round":
            return await narrate_round(
                scene=str(args["scene"]),
                dealt=int(str(args["dealt"])),
                defeated=bool(args["defeated"]),
            )
        case _:
            raise KeyError(f"unknown tool {name!r}")


async def run_offline() -> int:
    """Execute the hand-written plan locally and print the round.

    Returns:
        Process exit status.
    """
    plan = GAME_MASTER_PLAN
    levels = levels_of(plan)
    widest = max(len(level) for level in levels)

    print("Three players declared actions. The Game Master resolves the round.\n")
    print(f"plan: {len(plan.steps)} steps, depth {len(levels)}, widest level {widest}")
    for depth, level in enumerate(levels):
        print(f"  L{depth}  {', '.join(s.id for s in level)}")
    print()

    started = time.perf_counter()
    result = await execute_plan(plan, local_invoker)
    wall = (time.perf_counter() - started) * 1000.0

    print(summarize(result))
    print(f"\n  levels (model round trips replaced): {result.levels}")
    print(f"  inference passes avoided:            {result.inference_passes_saved}")
    print(f"  wall clock:                          {wall:.1f}ms")

    hidden = len(result.ledger) - len(result.returned)
    print(f"\n  returned to the model:       {sorted(result.returned)}")
    print(f"  results the model never saw: {hidden}")

    narration = result.returned.get("narration")
    if isinstance(narration, dict):
        print(f"\n  {narration.get('narration')}")

    ok = all(o.status is StepStatus.OK for o in result.ledger)
    print(f"\n{'all steps succeeded' if ok else 'SOME STEPS FAILED -- see ledger above'}")
    return 0 if ok else 1


async def run_live(model_id: str) -> int:
    """Give the tools to a real agent and let the model write its own plan.

    Args:
        model_id: Bedrock model or inference-profile id.

    Returns:
        Process exit status.
    """
    from strands import Agent, tool
    from strands.models import BedrockModel

    from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

    agent = Agent(
        model=BedrockModel(model_id=model_id),
        system_prompt=(
            "You are a tabletop Game Master resolving one combat round.\n\n"
            "You have a submit_workflow_plan tool. When you already know which tools to "
            "call and each call's arguments come mechanically from an earlier call's "
            "result, submit one plan instead of calling the tools one at a time. Read each "
            "tool's documented return shape and bind to the exact field you need. Name in "
            "`returns` only the steps whose results you actually need back."
        ),
        # Built inline so the element type is inferred from what Agent accepts. The domain
        # functions already document their return shapes, which is what lets the model bind
        # to the right field instead of guessing.
        tools=[
            tool(name="read_action_log")(read_action_log),
            tool(name="get_character")(get_character),
            tool(name="roll_check")(roll_check),
            tool(name="resolve_action")(resolve_action),
            tool(name="tally_round")(tally_round),
            tool(name="narrate_round")(narrate_round),
        ],
        plugins=[WorkflowPlanPlugin()],
    )

    prompt = (
        "Resolve this combat round against the bridge troll. Read the action log, look up "
        "each acting character, roll their declared skill against the scene, resolve each "
        "action, tally all the damage onto the troll at once, and narrate the result."
    )

    started = time.perf_counter()
    result = await agent.invoke_async(prompt)
    wall = time.perf_counter() - started

    used = [
        str(block["toolUse"]["name"])
        for message in agent.messages
        for block in message.get("content", ())
        if block.get("toolUse") is not None
    ]

    print(f"\n  tools the model called: {used}")
    print(f"  did it plan?            {'submit_workflow_plan' in used}")
    print(f"  model round trips:      {result.metrics.cycle_count}")
    print("  a plain loop would need: ~7 (six dependency levels plus the answer)")
    print(f"  wall clock:             {wall:.2f}s")
    print(f"  tokens:                 {result.metrics.accumulated_usage.get('totalTokens')}")
    print(f"\n  answer: {result.message.get('content')}")
    return 0


async def main() -> int:
    """Parse arguments and run the chosen mode.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="use a real Bedrock model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.live:
        return await run_offline()

    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        print("Set AWS_PROFILE (or AWS_ACCESS_KEY_ID) and AWS_REGION for --live.")
        return 2
    return await run_live(args.model)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
