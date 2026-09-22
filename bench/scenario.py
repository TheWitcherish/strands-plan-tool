"""A dependency chain deep enough to make the loop cost visible.

Five links, each needing the previous link's output before its own arguments exist. A
depth-D run uses the first D links, so one scenario drives the whole break-even sweep.

Each tool sleeps for a fixed interval to stand in for a real API call. That is the honest
way to measure: the difference between the two arms is then attributable to model round
trips rather than to tool speed, which is what the plan changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from strands.types.tools import AgentTool

MAX_DEPTH: Final = 5

_BESTIARY: Final[dict[str, dict[str, str]]] = {
    "leshen": {"weakness": "Yrden", "lair": "blackbough"},
    "griffin": {"weakness": "Igni", "lair": "yaruga_cliffs"},
}

_TERRAIN: Final[dict[str, str]] = {
    "blackbough": "swamp",
    "yaruga_cliffs": "scree",
}

_OILS: Final[dict[str, str]] = {
    "swamp": "cursed_oil",
    "scree": "hybrid_oil",
}


@dataclass(frozen=True, slots=True)
class Scenario:
    """A depth-D slice of the contract chain, plus the prompt that drives it."""

    depth: int
    tools: tuple[AgentTool, ...]
    prompt: str
    expected_substring: str


def build_scenario(depth: int, *, tool_latency_s: float) -> Scenario:
    """Assemble the first ``depth`` links of the chain as Strands tools.

    Latency is closed over rather than read from shared state, so two scenarios in one
    process cannot interfere with each other's timing.

    Args:
        depth: How many links to include, 1 to :data:`MAX_DEPTH`.
        tool_latency_s: Simulated per-tool duration.

    Returns:
        The tools, a prompt requiring all of them, and a substring the final answer
        should contain.

    Raises:
        ValueError: ``depth`` is outside 1..:data:`MAX_DEPTH`.
    """
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be 1..{MAX_DEPTH}, got {depth}")

    from strands import tool

    async def _wait() -> None:
        await asyncio.sleep(tool_latency_s)

    @tool
    async def read_notice_board() -> dict[str, str]:
        """Read the village notice board.

        Returns:
            An object shaped {"beast": str} -- the beast named in the contract.
        """
        await _wait()
        return {"beast": "leshen"}

    @tool
    async def consult_bestiary(beast: str) -> dict[str, str]:
        """Look up a beast's weakness sign and the name of its lair.

        Args:
            beast: Beast name, as returned by read_notice_board.

        Returns:
            An object shaped {"weakness": str, "lair": str}.
        """
        await _wait()
        entry = _BESTIARY.get(beast)
        if entry is None:
            raise KeyError(f"no bestiary entry for {beast!r}")
        return dict(entry)

    @tool
    async def scout_lair(lair: str) -> dict[str, str]:
        """Scout a lair and report its terrain.

        Args:
            lair: Lair name, as returned by consult_bestiary.

        Returns:
            An object shaped {"terrain": str}.
        """
        await _wait()
        terrain = _TERRAIN.get(lair)
        if terrain is None:
            raise KeyError(f"unmapped lair {lair!r}")
        return {"terrain": terrain}

    @tool
    async def pick_oil(terrain: str) -> dict[str, str]:
        """Choose the blade oil suited to a terrain.

        Args:
            terrain: Terrain name, as returned by scout_lair.

        Returns:
            An object shaped {"oil": str}.
        """
        await _wait()
        oil = _OILS.get(terrain)
        if oil is None:
            raise KeyError(f"no oil for terrain {terrain!r}")
        return {"oil": oil}

    @tool
    async def pack_satchel(sign: str, oil: str) -> dict[str, str]:
        """Pack the satchel and confirm the witcher is ready.

        Args:
            sign: Weakness sign, from consult_bestiary's "weakness" field.
            oil: Blade oil, from pick_oil's "oil" field.

        Returns:
            An object shaped {"sign": str, "oil": str, "verdict": str}.
        """
        await _wait()
        return {"sign": sign, "oil": oil, "verdict": "ready"}

    chain: Sequence[AgentTool] = (
        read_notice_board,
        consult_bestiary,
        scout_lair,
        pick_oil,
        pack_satchel,
    )

    prompts: Final[dict[int, tuple[str, str]]] = {
        1: ("Which beast is named on the notice board?", "leshen"),
        2: ("Which sign is the beast on the notice board weak to?", "Yrden"),
        3: ("What terrain is the lair of the beast on the notice board?", "swamp"),
        4: ("Which blade oil suits the lair of the beast on the notice board?", "cursed_oil"),
        5: (
            "Take the contract on the notice board: identify the beast, find its weakness "
            "and lair, scout the terrain, choose the right oil, and pack the satchel. "
            "Report the verdict.",
            "ready",
        ),
    }
    prompt, expected = prompts[depth]

    # pack_satchel needs the sign from link 2 and the oil from link 4, so a depth-5 run
    # is a genuine diamond rather than a straight line.
    return Scenario(
        depth=depth,
        tools=tuple(chain[:depth]),
        prompt=prompt,
        expected_substring=expected,
    )
