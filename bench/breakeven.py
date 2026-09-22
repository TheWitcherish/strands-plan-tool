"""Break-even benchmark: at what chain depth does planning start paying?

Two arms, same model, same tools, same prompt:

* **loop**  -- an ordinary agent. It must be re-sampled after every tool result, so a
  depth-D chain costs roughly D+1 model round trips.
* **plan**  -- the same agent plus ``WorkflowPlanPlugin``. If the model elects to plan, the
  whole chain runs inside one tool call, so the cost is about 2 round trips regardless of D.

Planning is never forced. ``used_plan`` records whether the model actually chose it, and a
run where it declined is reported as such rather than quietly averaged in -- the model
declining IS a result.

Run it:
    AWS_PROFILE=... AWS_REGION=... uv run python -m bench.breakeven

Every tool sleeps for a fixed interval, so the wall-clock gap between arms is attributable
to round trips rather than to tool speed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from strands import Agent
from strands.models import BedrockModel
from strands.plugins import Plugin

from bench.scenario import MAX_DEPTH, build_scenario
from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

# Verified ACTIVE in eu-central-1 on 2026-09-17 via
# `aws bedrock list-inference-profiles`. Model ids are perishable -- re-verify before
# trusting this default, and override with --model.
DEFAULT_MODEL: Final = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_TOOL_LATENCY_S: Final = 0.25
PLAN_TOOL: Final = "submit_workflow_plan"

_PLAN_NUDGE: Final = (
    "You have a submit_workflow_plan tool. When you already know which tools to call and "
    "each call's arguments come mechanically from an earlier call's result, submit one plan "
    "instead of calling the tools one at a time. Call tools individually when a result "
    "needs your judgement before you can decide what comes next."
)


@dataclass(frozen=True, slots=True)
class Run:
    """One measured agent invocation."""

    arm: str
    depth: int
    wall_s: float
    round_trips: int
    input_tokens: int
    output_tokens: int
    used_plan: bool
    correct: bool


@dataclass(frozen=True, slots=True)
class Cell:
    """Aggregated trials for one arm at one depth."""

    arm: str
    depth: int
    median_wall_s: float
    median_round_trips: float
    median_total_tokens: float
    plan_rate: float
    correct_rate: float


def _tool_names_used(agent: Agent) -> tuple[str, ...]:
    """Collect every tool name the model invoked, read from message history.

    Args:
        agent: The agent after an invocation.

    Returns:
        Tool names in call order.
    """
    names: list[str] = []
    for message in agent.messages:
        for block in message.get("content", ()):
            use = block.get("toolUse") if isinstance(block, dict) else None
            if use is not None:
                names.append(str(use.get("name")))
    return tuple(names)


def _normalize(text: str) -> str:
    """Fold case and word separators so an answer is not marked wrong for prose styling.

    A model writing "Cursed Oil" has answered `cursed_oil` correctly; only the formatting
    differs. Comparing raw strings made both arms look 0% correct at depth 4, which was a
    bug in the check rather than in the agents.

    Args:
        text: Raw text.

    Returns:
        Lowercased text with underscores, hyphens, and spaces removed.
    """
    return text.lower().replace("_", "").replace("-", "").replace(" ", "")


async def measure(arm: str, depth: int, *, model_id: str, tool_latency_s: float) -> Run:
    """Run one arm once and measure it.

    Args:
        arm: ``"loop"`` or ``"plan"``.
        depth: Chain depth.
        model_id: Bedrock model or inference-profile id.
        tool_latency_s: Simulated per-tool duration.

    Returns:
        The measurement.
    """
    scenario = build_scenario(depth, tool_latency_s=tool_latency_s)
    system = "You are a terse witcher's quartermaster. Use the tools available to you."
    plugins: list[Plugin] = []
    if arm == "plan":
        system = f"{system}\n\n{_PLAN_NUDGE}"
        plugins.append(WorkflowPlanPlugin())

    agent = Agent(
        model=BedrockModel(model_id=model_id),
        system_prompt=system,
        tools=list(scenario.tools),
        plugins=plugins,
    )

    started = time.perf_counter()
    result = await agent.invoke_async(scenario.prompt)
    wall = time.perf_counter() - started

    usage = result.metrics.accumulated_usage
    answer = str(result.message.get("content", ""))
    return Run(
        arm=arm,
        depth=depth,
        wall_s=wall,
        round_trips=result.metrics.cycle_count,
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        used_plan=PLAN_TOOL in _tool_names_used(agent),
        correct=_normalize(scenario.expected_substring) in _normalize(answer),
    )


def aggregate(runs: Sequence[Run]) -> Cell:
    """Reduce trials to medians, keeping plan election and correctness as rates.

    Args:
        runs: Trials for one arm at one depth; must be non-empty.

    Returns:
        The aggregate.
    """
    first = runs[0]
    return Cell(
        arm=first.arm,
        depth=first.depth,
        median_wall_s=statistics.median(r.wall_s for r in runs),
        median_round_trips=statistics.median(r.round_trips for r in runs),
        median_total_tokens=statistics.median(r.input_tokens + r.output_tokens for r in runs),
        plan_rate=sum(r.used_plan for r in runs) / len(runs),
        correct_rate=sum(r.correct for r in runs) / len(runs),
    )


def _cell_columns(cell: Cell) -> str:
    """Format one arm's three numeric columns.

    Args:
        cell: The aggregate to render.

    Returns:
        Wall clock, round trips, and total tokens, fixed width.
    """
    return (
        f"{cell.median_wall_s:5.2f}s {cell.median_round_trips:5.1f} {cell.median_total_tokens:8.0f}"
    )


def render(cells: Sequence[Cell]) -> str:
    """Format the sweep as a table plus the break-even reading.

    Args:
        cells: Aggregates for both arms across depths.

    Returns:
        The report.
    """
    by_key = {(c.arm, c.depth): c for c in cells}
    depths = sorted({c.depth for c in cells})

    lines = [
        "depth |            loop            |            plan            | verdict",
        "      |  wall   trips   tokens     |  wall   trips   tokens  pl%|",
        "------+----------------------------+----------------------------+---------------",
    ]
    breakeven: int | None = None
    for depth in depths:
        loop = by_key.get(("loop", depth))
        plan = by_key.get(("plan", depth))
        if loop is None or plan is None:
            continue
        faster = plan.median_wall_s < loop.median_wall_s
        if faster and breakeven is None and plan.plan_rate > 0.5:
            breakeven = depth
        delta = (loop.median_wall_s - plan.median_wall_s) / loop.median_wall_s * 100.0
        verdict = f"plan {delta:+.0f}%" if plan.plan_rate > 0.5 else "model declined"
        loop_cells = _cell_columns(loop)
        plan_cells = _cell_columns(plan)
        lines.append(
            f"  {depth}   | {loop_cells}    | {plan_cells} {plan.plan_rate * 100:3.0f}| {verdict}"
        )

    lines.append("")
    if breakeven is None:
        lines.append("Break-even: not reached in this sweep.")
    else:
        lines.append(f"Break-even: depth {breakeven} -- planning is faster from here on.")
    lines.append("pl% = share of trials where the model chose to plan (never forced).")

    incorrect = [c for c in cells if c.correct_rate < 1.0]
    if incorrect:
        lines.append("")
        lines.append("Correctness below 100% (answers not verified for these cells):")
        for cell in incorrect:
            lines.append(f"  {cell.arm} depth {cell.depth}: {cell.correct_rate * 100:.0f}%")
    return "\n".join(lines)


async def main() -> int:
    """Parse arguments, sweep both arms, print the report.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--tool-latency", type=float, default=DEFAULT_TOOL_LATENCY_S)
    args = parser.parse_args()

    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        print("Set AWS_PROFILE (or AWS_ACCESS_KEY_ID) and AWS_REGION first.")
        return 2

    print(f"model={args.model}  trials={args.trials}  tool_latency={args.tool_latency}s\n")

    cells: list[Cell] = []
    for depth in range(1, args.max_depth + 1):
        for arm in ("loop", "plan"):
            runs: list[Run] = []
            for trial in range(args.trials):
                print(f"  depth {depth} {arm:<4} trial {trial + 1}/{args.trials} ...", flush=True)
                runs.append(
                    await measure(arm, depth, model_id=args.model, tool_latency_s=args.tool_latency)
                )
            cells.append(aggregate(runs))

    print()
    print(render(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
