"""Live end-to-end demo: one real Bedrock model, one plan, one round trip.

Run it:
    AWS_PROFILE=... AWS_REGION=eu-central-1 uv run demo_bedrock.py

The point is visible in the last two numbers. A five-link dependency chain normally costs
the model six round trips, because it has to be re-sampled after every tool result before
it knows what to call next. Here it writes one plan and gets asked twice.

Nothing is forced. `submit_workflow_plan` is one tool among six with tool choice left
automatic, so the model can ignore it entirely -- and the demo says so when it does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Final

from strands import Agent
from strands.models import BedrockModel

from bench.scenario import build_scenario
from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

# Verified ACTIVE in account REDACTED / eu-central-1 on 2026-09-17 via
# `aws bedrock list-inference-profiles`. Model ids are perishable -- re-verify, or pass --model.
DEFAULT_MODEL: Final = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
TOOL_LATENCY_S: Final = 0.25
PLAN_TOOL: Final = "submit_workflow_plan"

SYSTEM: Final = """You are a terse witcher's quartermaster.

You have a submit_workflow_plan tool. When you already know which tools to call and each
call's arguments come mechanically from an earlier call's result, submit one plan instead
of calling the tools one at a time. Call tools individually when a result needs your
judgement before you can decide what comes next."""


def _submitted_plan(agent: Agent) -> dict[str, object] | None:
    """Return the plan the model wrote, if it wrote one.

    Args:
        agent: The agent after an invocation.

    Returns:
        The plan as the model submitted it, or ``None`` if it never planned.
    """
    for message in agent.messages:
        for block in message.get("content", ()):
            use = block.get("toolUse")
            if use is not None and use.get("name") == PLAN_TOOL:
                raw = use.get("input")
                plan = raw.get("plan") if isinstance(raw, dict) else None
                if isinstance(plan, dict):
                    return plan
    return None


async def main() -> int:
    """Run the demo against a live Bedrock model.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        print("Set AWS_PROFILE (or AWS_ACCESS_KEY_ID) and AWS_REGION first.")
        return 2

    scenario = build_scenario(5, tool_latency_s=TOOL_LATENCY_S)

    print("The contract: a notice board, an unknown beast, and a satchel to pack.")
    print(f"Five tools, each {TOOL_LATENCY_S * 1000:.0f}ms, each needing the one before it.\n")
    print(f"model: {args.model}\n")

    agent = Agent(
        model=BedrockModel(model_id=args.model),
        system_prompt=SYSTEM,
        tools=list(scenario.tools),
        plugins=[WorkflowPlanPlugin()],
    )

    started = time.perf_counter()
    result = await agent.invoke_async(scenario.prompt)
    wall = time.perf_counter() - started

    plan = _submitted_plan(agent)
    if plan is None:
        print("The model chose NOT to plan and called the tools one at a time.")
        print("That is a legitimate outcome: planning is elected, never imposed.")
    else:
        print("The plan the model wrote, unedited:\n")
        print(json.dumps(plan, indent=2))
        steps = plan.get("steps")
        print(f"\nSteps: {len(steps) if isinstance(steps, list) else '?'}")

    usage = result.metrics.accumulated_usage
    trips = result.metrics.cycle_count
    print(f"\n  model round trips:  {trips}")
    print(f"  round trips a plain loop would need: {scenario.depth + 1}")
    print(f"  wall clock:         {wall:.2f}s")
    print(f"  tokens:             {usage.get('totalTokens')}")
    print(f"\n  answer: {result.message.get('content')}")
    asked = "once" if trips == 1 else f"{trips} times"
    print(
        f"\nFive tools ran. The model was asked {asked}."
        "\nIt did not see the bestiary entry, the terrain, or the oil -- only the verdict."
    )
    if trips == 1:
        print(
            "\nThe answer is JSON because the plan declared itself final and we took it at "
            "its word.\nPass honor_final=False for prose and one more round trip."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
