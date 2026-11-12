# Copyright (c) 2026 AI Biology Lab contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

"""Original-model regression fixtures and bounded numerical-recovery behavior."""
import itertools

import pytest
import swiglpk

from optlang import glpk_interface as gi
from optlang import interface


@pytest.mark.parametrize("presolve", ["auto", False, True])
@pytest.mark.parametrize(
    "a,lb,ub",
    [
        (11, 1, 10),
        (-11, -10, -1),
        (1000000, 250000, 999995),
        (-1000000, -999995, -250000),
    ],
)
@pytest.mark.parametrize("direction", ["min", "max"])
def test_exact_binary_contradiction(a, lb, ub, presolve, direction):
    assert not any(lb <= a * i <= ub for i in (0, 1))
    p = gi.Model()
    x = gi.Variable("x", type="binary")
    g = gi.Variable("g", lb=0, ub=0)
    p.add([x, g])
    p.add(gi.Constraint(g + a * x, lb=lb, ub=ub))
    p.objective = gi.Objective(2 * x + 7, direction=direction)
    p.configuration.presolve = presolve
    assert p.optimize() == interface.INFEASIBLE
    assert p.status == interface.INFEASIBLE
    assert p.configuration.presolve == presolve


@pytest.mark.parametrize("presolve", ["auto", False, True])
@pytest.mark.parametrize("a", [11, -11, 1000000, -1000000])
@pytest.mark.parametrize("direction", ["min", "max"])
def test_feasible_mip_matches_exact_objective(a, presolve, direction):
    lo, hi = sorted((-a, a))
    candidates = [
        (3 * x - 2 * y + 5, x, y)
        for x, y in itertools.product(range(-2, 3), range(-2, 3))
        if lo <= a * x <= hi and -2 <= x + 2 * y <= 2
    ]
    expected = (min if direction == "min" else max)(row[0] for row in candidates)
    p = gi.Model()
    x = gi.Variable("x", lb=-2, ub=2, type="integer")
    y = gi.Variable("y", lb=-2, ub=2, type="integer")
    g = gi.Variable("g", lb=1, ub=1)
    p.add([x, y, g])
    p.add([gi.Constraint(a * x, lb=lo, ub=hi), gi.Constraint(x + 2 * y, lb=-2, ub=2)])
    p.objective = gi.Objective(3 * x - 2 * y + 5 * g, direction=direction)
    p.configuration.presolve = presolve
    assert p.optimize() == interface.OPTIMAL
    assert p.objective.value == pytest.approx(expected, abs=1e-8)
    assert lo <= a * x.primal <= hi and -2 <= x.primal + 2 * y.primal <= 2
    assert p.configuration.presolve == presolve


def contradictory_model():
    p = gi.Model()
    x = gi.Variable("x", type="binary")
    p.add(x)
    p.add(gi.Constraint(11 * x, lb=1, ub=10))
    p.objective = gi.Objective(0)
    p.configuration.presolve = True
    return p


def test_invalid_second_native_optimum_is_undefined(monkeypatch):
    p = contradictory_model()
    original = gi.glp_intopt
    calls = []

    def ignore_disabled_presolve(problem, parameters):
        calls.append(parameters.presolve)
        setting = parameters.presolve
        try:
            parameters.presolve = swiglpk.GLP_ON
            return original(problem, parameters)
        finally:
            parameters.presolve = setting

    monkeypatch.setattr(gi, "glp_intopt", ignore_disabled_presolve)
    assert p.optimize() == interface.UNDEFINED
    assert calls == [swiglpk.GLP_ON, swiglpk.GLP_OFF]
    assert p.configuration.presolve is True


def test_validation_retry_restores_configuration_after_native_exception(monkeypatch):
    p = contradictory_model()
    original = gi.glp_intopt
    calls = []

    def exception_on_disabled_presolve(problem, parameters):
        calls.append(parameters.presolve)
        if parameters.presolve == swiglpk.GLP_OFF:
            raise RuntimeError("injected native retry failure")
        return original(problem, parameters)

    monkeypatch.setattr(gi, "glp_intopt", exception_on_disabled_presolve)
    with pytest.raises(RuntimeError, match="injected native retry failure"):
        p.optimize()
    assert calls == [swiglpk.GLP_ON, swiglpk.GLP_OFF]
    assert p.configuration.presolve is True


def test_existing_presolve_retry_and_validation_restore_explicit_false(monkeypatch):
    p = contradictory_model()
    p.configuration.presolve = False
    original = gi.glp_intopt
    calls = []

    def record(problem, parameters):
        calls.append(parameters.presolve)
        return original(problem, parameters)

    monkeypatch.setattr(gi, "glp_intopt", record)
    assert p.optimize() == interface.INFEASIBLE
    assert calls == [swiglpk.GLP_OFF, swiglpk.GLP_ON, swiglpk.GLP_OFF]
    assert p.configuration.presolve is False


@pytest.mark.parametrize("presolve", ["auto", False, True])
def test_feasible_large_row_roundoff_uses_relative_tolerance(presolve):
    p = gi.Model()
    x = gi.Variable("x", type="binary")
    y = gi.Variable("y", lb=0, ub=10)
    z = gi.Variable("z", lb=0, ub=10)
    p.add([x, y, z])
    p.add(
        [
            gi.Constraint(x, lb=1),
            gi.Constraint(y, lb=3, ub=3),
            gi.Constraint(z, lb=7, ub=7),
            gi.Constraint(1e12 * x + 0.1 * y - 0.1 * z, lb=1e12 - 0.4, ub=1e12 - 0.4),
        ]
    )
    p.objective = gi.Objective(x + y)
    p.configuration.presolve = presolve
    assert p.optimize() == interface.OPTIMAL
    assert (x.primal, y.primal, z.primal) == (1, 3, 7)
    assert p.objective.value == 4
    ae, re = swiglpk.doubleArray(1), swiglpk.doubleArray(1)
    ai, ri = swiglpk.intArray(1), swiglpk.intArray(1)
    swiglpk.glp_check_kkt(
        p.problem, swiglpk.GLP_MIP, swiglpk.GLP_KKT_PE, ae, ai, re, ri
    )
    assert ae[0] > p.configuration.tolerances.feasibility
    assert re[0] < p.configuration.tolerances.feasibility
    assert p.configuration.presolve == presolve


@pytest.mark.parametrize("condition", [swiglpk.GLP_KKT_PE, swiglpk.GLP_KKT_PB])
@pytest.mark.parametrize(
    "absolute,relative,tolerance,expected",
    [
        (1e-4, 1e-9, 1e-7, True),
        (1e-9, 1e-4, 1e-7, True),
        (1e-4, 1e-5, 1e-7, False),
        (1e-4, 1e-5, 1e-3, True),
    ],
)
def test_primal_check_uses_absolute_or_relative_configured_tolerance(
    monkeypatch, condition, absolute, relative, tolerance, expected
):
    p = contradictory_model()
    p.configuration.tolerances.feasibility = tolerance
    seen = []

    def check(problem, solution, kind, ae, ai, re, ri):
        assert solution == swiglpk.GLP_MIP
        assert kind in (swiglpk.GLP_KKT_PE, swiglpk.GLP_KKT_PB)
        seen.append(kind)
        ae[0] = absolute if kind == condition else 0
        re[0] = relative if kind == condition else 0
        ai[0] = ri[0] = 0

    monkeypatch.setattr(gi, "glp_check_kkt", check)
    assert p._mip_solution_is_feasible() is expected
    assert swiglpk.GLP_KKT_PE in seen

    for absolute, relative in [
        (float("inf"), 0),
        (float("nan"), 0),
        (-1, 0),
        (0, float("inf")),
        (0, float("nan")),
        (0, -1),
    ]:
        assert p._mip_solution_is_feasible() is False
