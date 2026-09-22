"""Binder behaviour, including the $eval lockdown that keeps a plan inspectable."""

from __future__ import annotations

import pytest

from strands_plan_tool import JsonataBinder, JsonValue

ENV: dict[str, JsonValue] = {"steps": {"lookup": {"id": 7, "items": [10, 20, 30]}}}


def test_plain_path_binding() -> None:
    assert JsonataBinder().bind("steps.lookup.id", ENV) == 7


def test_arithmetic_and_indexing() -> None:
    assert JsonataBinder().bind("steps.lookup.items[1] * 2", ENV) == 40


def test_aggregation_keeps_intermediates_out_of_the_result() -> None:
    assert JsonataBinder().bind("$sum(steps.lookup.items)", ENV) == 60


def test_eval_is_denied_by_default() -> None:
    with pytest.raises(PermissionError):
        JsonataBinder().bind('$eval("1+1")', ENV)


def test_eval_can_be_opted_back_in() -> None:
    assert JsonataBinder(allow_eval=True).bind('$eval("1+1")', ENV) == 2


def test_denying_eval_does_not_affect_ordinary_expressions() -> None:
    locked = JsonataBinder()
    opened = JsonataBinder(allow_eval=True)
    assert locked.bind("steps.lookup.id", ENV) == opened.bind("steps.lookup.id", ENV)


def test_compile_cache_returns_a_stable_result() -> None:
    binder = JsonataBinder()
    first = binder.bind("steps.lookup.id", ENV)
    second = binder.bind("steps.lookup.id", ENV)
    assert first == second == 7


def test_binder_does_not_mutate_the_environment() -> None:
    env: dict[str, JsonValue] = {"steps": {"lookup": {"id": 7}}}
    snapshot: dict[str, JsonValue] = {"steps": {"lookup": {"id": 7}}}
    JsonataBinder().bind("steps.lookup.id", env)
    assert env == snapshot
