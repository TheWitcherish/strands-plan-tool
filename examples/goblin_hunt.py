"""The mage hunts the goblin: forest or castle, decided inside one plan with `when`.

Offline (no credentials, the plan is hand-written):
    uv run python -m examples.goblin_hunt
    GOBLIN_LAIR=forest uv run python -m examples.goblin_hunt

Live, letting the model write the plan:
    export AWS_PROFILE=YOUR_PROFILE AWS_REGION=YOUR_AWS_REGION
    uv run python -m examples.goblin_hunt --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Mapping
from typing import Final

from strands_plan_tool import JsonValue, WorkflowPlan, execute_plan, summarize

DEFAULT_MODEL_ID: Final = "eu.anthropic.claude-sonnet-4-6"
MODEL_ID: Final[str] = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)

LAIR: Final[str] = os.environ.get("GOBLIN_LAIR", "castle")


def look_up_monster(name: str) -> dict[str, JsonValue]:
    """Look up a monster's stat block and where it lives.

    Args:
        name: Monster name, e.g. "goblin".

    Returns:
        An object shaped {"name": str, "hp": int, "armour": int, "lair": "forest" | "castle"}.
    """
    return {"name": name, "hp": 7, "armour": 15, "lair": LAIR}


def search_forest() -> dict[str, JsonValue]:
    """Search the forest for monsters.

    Returns:
        An object shaped {"found": str}.
    """
    return {"found": "a very confused owl"}


def storm_castle() -> dict[str, JsonValue]:
    """Break into the castle gates.

    Returns:
        An object shaped {"gate": str}.
    """
    return {"gate": "open"}


def roll_attack(armour: int) -> dict[str, JsonValue]:
    """Roll an attack against an armour value.

    Args:
        armour: The target's armour, from look_up_monster's "armour" field.

    Returns:
        An object shaped {"roll": int, "hits": bool, "damage": int}.
    """
    roll = 18
    return {"roll": roll, "hits": roll >= armour, "damage": 6 if roll >= armour else 0}


# The plan a model writes for "hunt the goblin". The two roads carry opposite `when`
# conditions; the attack follows the castle, so the forest road prunes it too.
PLAN: Final = WorkflowPlan.model_validate(
    {
        "steps": [
            {"id": "monster", "tool": "look_up_monster", "args": {"name": "goblin"}},
            {"id": "forest", "tool": "search_forest", "when": 'steps.monster.lair = "forest"'},
            {"id": "castle", "tool": "storm_castle", "when": 'steps.monster.lair = "castle"'},
            {
                "id": "attack",
                "tool": "roll_attack",
                "bind": {"armour": "steps.monster.armour"},
                "after": ["castle"],
            },
        ],
        "returns": ["forest", "castle", "attack"],
        "final": True,
    }
)


async def local_invoker(name: str, args: Mapping[str, JsonValue]) -> JsonValue:
    """Run one tool by name, checking each argument at the boundary."""
    match name:
        case "look_up_monster":
            return look_up_monster(str(args["name"]))
        case "search_forest":
            return search_forest()
        case "storm_castle":
            return storm_castle()
        case "roll_attack":
            return roll_attack(int(str(args["armour"])))
        case _:
            raise KeyError(f"unknown tool {name!r}")


async def run_offline() -> int:
    """Execute the hand-written plan and print which road the mage took."""
    result = await execute_plan(PLAN, local_invoker)
    print(summarize(result))
    print(f"\nreturned: {result.returned}")
    return 0


def run_live() -> int:
    """Let the model write the plan, branches included."""
    from strands import Agent, tool
    from strands.models import BedrockModel

    from strands_plan_tool.strands_adapter import WorkflowPlanPlugin

    agent = Agent(
        model=BedrockModel(model_id=MODEL_ID),
        system_prompt=(
            "You are a Game Master for a tabletop role-playing game.\n\n"
            "You have a submit_workflow_plan tool. When you already know which tools to call "
            "and each call's arguments come from an earlier call's result, submit one plan "
            "instead of calling the tools one at a time. Each step's `bind` maps an argument "
            'to a JSONata expression over {"steps": {<id>: <result>}}, for example '
            '{"armour": "steps.monster.armour"}. Read each tool\'s documented return shape. '
            "When the next tool depends on a result you have not seen yet, do not look first: "
            "put every road in the plan, each with a `when` condition on that result."
        ),
        tools=[tool(f) for f in (look_up_monster, search_forest, storm_castle, roll_attack)],
        plugins=[WorkflowPlanPlugin()],
        callback_handler=None,
    )
    response = agent(
        "The mage hunts the goblin. Look it up, go to its lair (search the forest, or storm "
        "the castle), and attack it only if it lives in the castle."
    )
    for message in agent.messages:
        for block in message["content"]:
            if "toolUse" in block:
                print(f"model called {block['toolUse']['name']}: {block['toolUse']['input']}")
    print(response.message["content"][0]["text"])
    print(f"model round trips: {response.metrics.cycle_count}  (a plain loop needs 4)")
    return 0


def main() -> int:
    """Parse arguments and run the chosen mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="let a Bedrock model write the plan")
    args = parser.parse_args()
    return run_live() if args.live else asyncio.run(run_offline())


if __name__ == "__main__":
    raise SystemExit(main())
