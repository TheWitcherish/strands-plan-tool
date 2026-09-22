"""Binding expressions to upstream results.

``Binder`` is a protocol so the expression language is swappable: JSONata here, a
restricted JSON-pointer dialect for teaching material, both behind one seam.

``JsonataBinder`` shuts JSONata's ``$eval`` on every expression it compiles. That is
the library's only dynamic-expression surface, and closing it is what makes a plan
fully inspectable before it runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import jsonata

from .models import JsonValue

__all__ = ["Binder", "JsonataBinder"]

_EVAL_DENIED = "$eval is not permitted inside a workflow plan binding"


@runtime_checkable
class Binder(Protocol):
    """Evaluates one binding expression against the results collected so far."""

    def bind(self, expression: str, env: Mapping[str, JsonValue]) -> JsonValue:
        """Evaluate ``expression`` against ``env``.

        Args:
            expression: Source in whatever dialect this binder implements.
            env: Read-only view of upstream results, shaped ``{"steps": {...}}``.

        Returns:
            The bound value.

        Raises:
            Exception: Any evaluation failure; the executor wraps it in
                :class:`~strands_plan_tool.errors.BindEvaluationError`.
        """
        ...


def _deny_eval(*_args: JsonValue) -> JsonValue:
    """Stand in for ``$eval`` so a plan cannot construct expressions at runtime."""
    raise PermissionError(_EVAL_DENIED)


class JsonataBinder:
    """JSONata binder with a per-expression compile cache and ``$eval`` disabled."""

    __slots__ = ("_allow_eval", "_cache")

    def __init__(self, *, allow_eval: bool = False) -> None:
        """Create a binder.

        Args:
            allow_eval: Leave JSONata's ``$eval`` reachable. Defaults to ``False``;
                enabling it means a plan's bindings can no longer be fully analysed
                before execution.
        """
        self._cache: dict[str, jsonata.Jsonata] = {}
        self._allow_eval = allow_eval

    def bind(self, expression: str, env: Mapping[str, JsonValue]) -> JsonValue:
        """Evaluate a JSONata expression against the collected results.

        Args:
            expression: JSONata source, for example ``steps.lookup.id``.
            env: Read-only view shaped ``{"steps": {<id>: <result>}}``.

        Returns:
            The evaluated value.

        Raises:
            PermissionError: The expression called ``$eval`` while it was disabled.
        """
        compiled = self._cache.get(expression)
        if compiled is None:
            compiled = jsonata.Jsonata(expression)
            if not self._allow_eval:
                # Instance frame is a child of the library's static frame, so this
                # shadows $eval for this expression only and never mutates the global.
                compiled.register_lambda("eval", _deny_eval)
            self._cache[expression] = compiled

        # Passed through without copying: JSONata is a query evaluator and only reads
        # its input, so copying here would make every binding O(prior results) and the
        # whole plan quadratic. Enforced by test_binder_does_not_mutate_the_environment.
        result: JsonValue = compiled.evaluate(env)
        return result
