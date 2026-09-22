"""Declarative JSONata workflow plans for agent tool chains.

The model writes one :class:`WorkflowPlan` describing which tools to call, in what
order, and how each step's arguments derive from earlier results. The plan executes
locally in one batch and the model sees only the aggregated result, so a dependency
chain of depth D costs one model round trip instead of D.
"""

from .binder import Binder, JsonataBinder
from .errors import (
    AgentInterruptedError,
    BindEvaluationError,
    DanglingReferenceError,
    DuplicateStepError,
    InvalidPlanShapeError,
    PlanCycleError,
    PlanError,
    StepLimitError,
    ToolExecutionError,
    ToolNotPlannableError,
)
from .executor import ToolInvoker, execute_plan, levels_of, summarize
from .graph import dependencies_of, plan_levels
from .models import (
    MAX_STEPS,
    JsonValue,
    OnError,
    PlanResult,
    PlanStep,
    StepOutcome,
    StepStatus,
    WorkflowPlan,
)

__all__ = [
    "MAX_STEPS",
    "AgentInterruptedError",
    "BindEvaluationError",
    "Binder",
    "DanglingReferenceError",
    "DuplicateStepError",
    "InvalidPlanShapeError",
    "JsonValue",
    "JsonataBinder",
    "OnError",
    "PlanCycleError",
    "PlanError",
    "PlanResult",
    "PlanStep",
    "StepLimitError",
    "StepOutcome",
    "StepStatus",
    "ToolExecutionError",
    "ToolInvoker",
    "ToolNotPlannableError",
    "WorkflowPlan",
    "dependencies_of",
    "execute_plan",
    "levels_of",
    "plan_levels",
    "summarize",
]
