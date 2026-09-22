"""Typed errors for plan validation and execution.

Every failure mode a plan can hit has its own class, so a caller can match on the
kind rather than parsing a message. Nothing in this package raises a bare
``Exception`` or swallows one.
"""

from __future__ import annotations

__all__ = [
    "AgentInterruptedError",
    "BindEvaluationError",
    "DanglingReferenceError",
    "DuplicateStepError",
    "InvalidPlanShapeError",
    "PlanCycleError",
    "PlanError",
    "StepLimitError",
    "ToolExecutionError",
    "ToolNotPlannableError",
]


class PlanError(Exception):
    """Base class for every plan validation or binding failure."""


class DuplicateStepError(PlanError):
    """Two steps declared the same id."""

    def __init__(self, step_id: str) -> None:
        """Record the duplicated id.

        Args:
            step_id: The id that appeared more than once.
        """
        super().__init__(f"duplicate step id: {step_id!r}")
        self.step_id = step_id


class DanglingReferenceError(PlanError):
    """A step referenced an id that no step declares."""

    def __init__(self, step_id: str, missing: str, *, field: str) -> None:
        """Record where the dangling reference was found.

        Args:
            step_id: The step holding the bad reference.
            missing: The referenced id that does not exist.
            field: Which field carried it, for example ``after`` or ``returns``.
        """
        super().__init__(f"step {step_id!r} {field} references unknown step {missing!r}")
        self.step_id = step_id
        self.missing = missing
        self.field = field


class PlanCycleError(PlanError):
    """The dependency graph contains a cycle, so no execution order exists."""

    def __init__(self, remaining: tuple[str, ...]) -> None:
        """Record the steps that could never become ready.

        Args:
            remaining: Ids still blocked when the topological sort stalled.
        """
        super().__init__(f"dependency cycle among steps: {', '.join(sorted(remaining))}")
        self.remaining = remaining


class StepLimitError(PlanError):
    """The plan declared more steps than the configured ceiling allows."""

    def __init__(self, count: int, limit: int) -> None:
        """Record the overrun.

        Args:
            count: How many steps the plan declared.
            limit: The configured maximum.
        """
        super().__init__(f"plan declares {count} steps, limit is {limit}")
        self.count = count
        self.limit = limit


class BindEvaluationError(PlanError):
    """A JSONata binding expression failed to compile or evaluate.

    Raised instead of substituting ``None``: a binding that silently yields nothing
    would hand a downstream tool an argument the plan never intended.
    """

    def __init__(self, step_id: str, parameter: str, expression: str, cause: str) -> None:
        """Record which binding failed and why.

        Args:
            step_id: The step whose binding failed.
            parameter: The argument name the expression was bound to.
            expression: The offending JSONata source.
            cause: Human-readable reason from the evaluator.
        """
        super().__init__(
            f"step {step_id!r} binding {parameter!r} failed: {cause} (in {expression!r})"
        )
        self.step_id = step_id
        self.parameter = parameter
        self.expression = expression
        self.cause = cause


class ToolNotPlannableError(PlanError):
    """A step named a tool that is not on the plannable allowlist.

    Most often this means the tool can raise a human-in-the-loop interrupt, which the
    SDK refuses to service from inside a plan. The tool is still fully available to the
    model through the ordinary loop; it just cannot be batched.
    """

    def __init__(self, step_id: str, tool: str, allowed: tuple[str, ...]) -> None:
        """Record the refused tool and what was permitted.

        Args:
            step_id: The step naming the tool.
            tool: The refused tool name.
            allowed: The permitted tool names, sorted.
        """
        super().__init__(
            f"step {step_id!r} calls {tool!r}, which cannot be run inside a plan. "
            f"Call it directly instead. Plannable tools: {', '.join(allowed) or '(none)'}"
        )
        self.step_id = step_id
        self.tool = tool
        self.allowed = allowed


class ToolExecutionError(PlanError):
    """A planned tool ran and reported failure.

    Distinguished from a transport or binding error so a step's ledger entry names the
    tool's own complaint rather than a generic exception.
    """

    def __init__(self, tool: str, detail: str) -> None:
        """Record the failing tool and its message.

        Args:
            tool: The tool that failed.
            detail: The tool's error text.
        """
        super().__init__(f"tool {tool!r} failed: {detail}")
        self.tool = tool
        self.detail = detail


class InvalidPlanShapeError(PlanError):
    """The submitted plan payload does not match the schema.

    Distinct from the semantic errors above: the plan was not even well-formed, so no
    graph could be built from it. Carries the validator's own message so the model can see
    which field it got wrong.
    """

    def __init__(self, detail: str) -> None:
        """Record the validation detail.

        Args:
            detail: The validator's message.
        """
        super().__init__(f"plan does not match the required shape: {detail}")
        self.detail = detail


class AgentInterruptedError(PlanError):
    """The agent was already in an interrupt state when the plan was submitted.

    The SDK refuses every direct tool call while an interrupt is outstanding, so a plan
    submitted in that state would fail on its first step.
    """

    def __init__(self) -> None:
        """Describe the refusal."""
        super().__init__("agent has an outstanding interrupt; resolve it before submitting a plan")
