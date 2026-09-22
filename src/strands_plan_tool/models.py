"""Pydantic v2 frozen boundary models: the contract the model writes to.

These are the only shapes that cross the tool boundary in either direction, so they
are the whole public contract. A TypeScript port mirrors this file as Zod schemas
with identical field names; nothing here relies on Python-only dynamism.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_STEPS",
    "SCHEMA_MAX_STEPS",
    "OnError",
    "PlanResult",
    "PlanStep",
    "StepOutcome",
    "StepStatus",
    "WorkflowPlan",
]

MAX_STEPS: Final = 32
"""Default policy ceiling on steps per plan. A plan is a short dependency chain, not a
program. Enforced by :func:`~strands_plan_tool.graph.plan_levels` before anything executes, so
it is configurable per deployment rather than baked into the schema."""

SCHEMA_MAX_STEPS: Final = 10_000
"""Hard schema bound, far above any sane policy value, purely to keep a malformed or
hostile payload from allocating without limit. The real limit is ``MAX_STEPS``."""

STEP_ID_PATTERN: Final = r"^[a-z][a-z0-9_]{0,31}$"

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
"""Any value that survives a JSON round trip, which is every value a tool exchanges."""


class OnError(StrEnum):
    """What the executor does when a step fails."""

    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


class StepStatus(StrEnum):
    """Terminal state of a single step."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep(BaseModel):
    """One tool invocation and how its arguments are produced.

    ``args`` carries literal arguments. ``bind`` carries JSONata expressions evaluated
    against ``{"steps": {<id>: <result>}}`` once every dependency has completed, and is
    merged over ``args`` so a binding always wins over a literal of the same name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=STEP_ID_PATTERN)
    tool: str = Field(min_length=1)
    args: dict[str, JsonValue] = Field(default_factory=dict)
    bind: dict[str, str] = Field(default_factory=dict)
    after: tuple[str, ...] = ()


class WorkflowPlan(BaseModel):
    """A dependency-ordered batch of tool calls the model commits to in one turn.

    ``returns`` names the step ids whose results travel back to the model; every other
    result stays inside the executor. ``final`` declares that the returned values are
    the answer, letting the caller end the turn without another model round trip.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=SCHEMA_MAX_STEPS)
    returns: tuple[str, ...] = Field(min_length=1)
    final: bool = False
    on_error: OnError = OnError.FAIL_FAST


class StepOutcome(BaseModel):
    """What actually happened to one step, always reported even when it failed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    tool: str
    status: StepStatus
    level: int
    duration_ms: float
    error: str | None = None


class PlanResult(BaseModel):
    """The single value handed back to the model for the whole plan.

    ``returned`` holds only the steps named in ``WorkflowPlan.returns``. ``ledger``
    reports every step's fate so a partially failed plan is legible rather than silent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    returned: dict[str, JsonValue]
    ledger: tuple[StepOutcome, ...]
    levels: int
    inference_passes_saved: int
    final: bool
