"""Fake tabletop session: a Game Master resolving one combat round.

The interesting part is the *shape*. A round is not a straight line -- three players act
independently, each action needs that character's sheet before it can be rolled, and the
round cannot be tallied until all three resolve. So the plan is:

    L0  read_action_log                          1 step
    L1  get_character      x3   (fan-out)        3 steps
    L2  roll_check         x3                    3 steps
    L3  resolve_action     x3                    3 steps
    L4  tally_round             (join, in-degree 3)
    L5  narrate_round

Twelve steps, depth 6, three wide. The join is where JSONata earns its place over a
simpler path syntax: `tally_round` needs an *array* of the three damage values, which a
binding builds inline with `[steps.a.damage, steps.b.damage, steps.c.damage]`.

All dice are deterministic so the example is reproducible: a stream can run it twice and
get the same round.
"""

from __future__ import annotations

import zlib
from typing import Final

from strands_plan_tool import JsonValue, PlanStep, WorkflowPlan

ACTION_LOG: Final[dict[str, JsonValue]] = {
    "round": 3,
    "scene": "the collapsed bridge at Kaer Trolde",
    "actions": [
        {"actor": "Vesna", "skill": "stealth", "intent": "cut the rope bridge free"},
        {"actor": "Alder", "skill": "arcana", "intent": "hurl a firebolt"},
        {"actor": "Grimm", "skill": "athletics", "intent": "hold the line"},
    ],
}

CHARACTERS: Final[dict[str, dict[str, JsonValue]]] = {
    "Vesna": {"name": "Vesna", "level": 4, "hp": 27, "skills": {"stealth": 7, "arcana": 1}},
    "Alder": {"name": "Alder", "level": 5, "hp": 22, "skills": {"arcana": 9, "athletics": 0}},
    "Grimm": {"name": "Grimm", "level": 4, "hp": 41, "skills": {"athletics": 6, "stealth": -1}},
}

MONSTER: Final[dict[str, JsonValue]] = {"name": "bridge troll", "hp": 38, "armour": 14}

_DIFFICULTY: Final = 14


def _d20(actor: str, skill: str) -> int:
    """Deterministic d20 for a given actor and skill.

    A real table rolls dice; an example that changes every run cannot be demonstrated.
    CRC32 rather than ``hash`` because Python salts ``hash`` per process.

    Args:
        actor: Character name.
        skill: Skill being tested.

    Returns:
        A value in 1..20, stable across runs and machines.
    """
    return zlib.crc32(f"{actor}:{skill}".encode()) % 20 + 1


async def read_action_log() -> dict[str, JsonValue]:
    """What the players declared this round.

    Returns:
        {"round": int, "scene": str, "actions": [{"actor","skill","intent"}, ...]}
    """
    return dict(ACTION_LOG)


async def get_character(name: str) -> dict[str, JsonValue]:
    """Fetch one character sheet.

    Args:
        name: Character name from the action log.

    Returns:
        {"name": str, "level": int, "hp": int, "skills": {skill: modifier}}

    Raises:
        KeyError: No such character.
    """
    sheet = CHARACTERS.get(name)
    if sheet is None:
        raise KeyError(f"no character sheet for {name!r}")
    return dict(sheet)


async def roll_check(actor: str, skill: str, modifier: int) -> dict[str, JsonValue]:
    """Roll a skill check against the scene difficulty.

    Args:
        actor: Character name.
        skill: Skill being tested.
        modifier: That character's modifier for the skill.

    Returns:
        {"actor": str, "roll": int, "total": int, "success": bool}
    """
    roll = _d20(actor, skill)
    total = roll + modifier
    return {"actor": actor, "roll": roll, "total": total, "success": total >= _DIFFICULTY}


async def resolve_action(actor: str, success: bool, intent: str) -> dict[str, JsonValue]:
    """Turn a check into a concrete effect and damage number.

    Args:
        actor: Character name.
        success: Whether the check beat the difficulty.
        intent: What the player declared.

    Returns:
        {"actor": str, "effect": str, "damage": int}
    """
    damage = 12 if success else 0
    effect = f"{intent} — succeeds" if success else f"{intent} — fails"
    return {"actor": actor, "effect": effect, "damage": damage}


async def tally_round(damages: list[int], target: str) -> dict[str, JsonValue]:
    """Apply every player's damage to the target at once.

    This is the join: it cannot run until all three resolutions exist.

    Args:
        damages: One damage value per player action.
        target: Monster name.

    Returns:
        {"target": str, "hp_before": int, "hp_after": int, "defeated": bool, "dealt": int}
    """
    before = MONSTER["hp"]
    if not isinstance(before, int):  # pragma: no cover - fixture is an int
        raise TypeError("monster hp must be an int")
    dealt = sum(damages)
    after = max(before - dealt, 0)
    return {
        "target": target,
        "hp_before": before,
        "hp_after": after,
        "dealt": dealt,
        "defeated": after == 0,
    }


async def narrate_round(scene: str, dealt: int, defeated: bool) -> dict[str, JsonValue]:
    """Write the Game Master's closing line for the round.

    Args:
        scene: Scene description from the action log.
        dealt: Total damage dealt this round.
        defeated: Whether the target fell.

    Returns:
        {"narration": str}
    """
    tail = "The troll topples into the gorge." if defeated else "The troll still stands."
    return {"narration": f"Round resolved at {scene}: {dealt} damage dealt. {tail}"}


# The plan a Game Master agent would submit. Written out here so the example is
# reproducible offline; in the live run the model authors its own.
GAME_MASTER_PLAN: Final = WorkflowPlan(
    steps=(
        PlanStep(id="log", tool="read_action_log"),
        # Fan-out: three sheets, one per declared action, all at the same level.
        PlanStep(id="c_vesna", tool="get_character", bind={"name": "steps.log.actions[0].actor"}),
        PlanStep(id="c_alder", tool="get_character", bind={"name": "steps.log.actions[1].actor"}),
        PlanStep(id="c_grimm", tool="get_character", bind={"name": "steps.log.actions[2].actor"}),
        # Each roll reaches into its character's skill map with a computed key.
        PlanStep(
            id="r_vesna",
            tool="roll_check",
            bind={
                "actor": "steps.c_vesna.name",
                "skill": "steps.log.actions[0].skill",
                "modifier": "steps.c_vesna.skills.stealth",
            },
        ),
        PlanStep(
            id="r_alder",
            tool="roll_check",
            bind={
                "actor": "steps.c_alder.name",
                "skill": "steps.log.actions[1].skill",
                "modifier": "steps.c_alder.skills.arcana",
            },
        ),
        PlanStep(
            id="r_grimm",
            tool="roll_check",
            bind={
                "actor": "steps.c_grimm.name",
                "skill": "steps.log.actions[2].skill",
                "modifier": "steps.c_grimm.skills.athletics",
            },
        ),
        PlanStep(
            id="a_vesna",
            tool="resolve_action",
            bind={
                "actor": "steps.r_vesna.actor",
                "success": "steps.r_vesna.success",
                "intent": "steps.log.actions[0].intent",
            },
        ),
        PlanStep(
            id="a_alder",
            tool="resolve_action",
            bind={
                "actor": "steps.r_alder.actor",
                "success": "steps.r_alder.success",
                "intent": "steps.log.actions[1].intent",
            },
        ),
        PlanStep(
            id="a_grimm",
            tool="resolve_action",
            bind={
                "actor": "steps.r_grimm.actor",
                "success": "steps.r_grimm.success",
                "intent": "steps.log.actions[2].intent",
            },
        ),
        # The join. A JSONata array literal gathers three separate step results into the
        # single list argument the tool wants -- no extra tool, no extra round trip.
        PlanStep(
            id="tally",
            tool="tally_round",
            args={"target": "bridge troll"},
            bind={"damages": "[steps.a_vesna.damage, steps.a_alder.damage, steps.a_grimm.damage]"},
        ),
        PlanStep(
            id="narration",
            tool="narrate_round",
            bind={
                "scene": "steps.log.scene",
                "dealt": "steps.tally.dealt",
                "defeated": "steps.tally.defeated",
            },
        ),
    ),
    returns=("tally", "narration"),
    final=True,
)
