"""Certified convex outer reductions for diagonal quadratic transitions."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from certification.equations import Expr, Op

from .convex import solve_convex_constraints
from .linear import solve_linear_constraints
from ..model.optimization import (
    Polynomial,
    QuadraticConstraint,
    _polynomial,
    conjunctive_comparisons,
    quadratic_constraints,
)
from ..model.proof_rules import (
    LinearInequality,
    ProofDeferred,
    comparison_inequalities,
)
from ..model.reduction import ReducedCase, expression_hash

try:  # pragma: no cover - installation is checked by integration runs
    import numpy as np
    from scipy.optimize import linprog, minimize
except Exception:  # pragma: no cover
    np = None
    linprog = None
    minimize = None


def _oriented_polynomial(comparison) -> tuple[Polynomial, bool]:
    strict = comparison.op in {"<", ">"}
    if comparison.op in {"<=", "<"}:
        expression = Op("-", (comparison.left, comparison.right))
    elif comparison.op in {">=", ">"}:
        expression = Op("-", (comparison.right, comparison.left))
    else:
        raise ProofDeferred(
            "NOT_CONVEX",
            f"unsupported comparison {comparison.op}",
        )
    return _polynomial(expression), strict


def _linear_skeleton(comparisons) -> list[LinearInequality]:
    constraints: list[LinearInequality] = []
    for comparison in comparisons:
        polynomial, _strict = _oriented_polynomial(comparison)
        if any(len(monomial) > 1 for monomial in polynomial):
            continue
        constraints.extend(comparison_inequalities(comparison, set()))
    return constraints


def _convex_skeleton(comparisons) -> list[QuadraticConstraint]:
    constraints: list[QuadraticConstraint] = []
    for comparison in comparisons:
        try:
            constraints.extend(quadratic_constraints(
                Op(comparison.op, (comparison.left, comparison.right)),
                set(),
            ))
        except ProofDeferred:
            continue
    return constraints


def _linear_candidate_bound(
    constraints: list[LinearInequality],
    variable: str,
    *,
    upper: bool,
    timeout_ms: int,
) -> tuple[Fraction, dict[str, Any]]:
    if linprog is None or np is None:
        raise ProofDeferred("BLOCKED_INPUT", "SciPy linear optimization is unavailable")
    variables = sorted({
        name for constraint in constraints for name, _value in constraint.coefficients
    } | {variable})
    rows = [
        [float(constraint.coeff_dict().get(name, 0)) for name in variables]
        for constraint in constraints
    ]
    bounds = [float(constraint.bound) for constraint in constraints]
    objective = [0.0 for _name in variables]
    objective[variables.index(variable)] = -1.0 if upper else 1.0
    result = linprog(
        np.asarray(objective),
        A_ub=np.asarray(rows, dtype=float) if rows else None,
        b_ub=np.asarray(bounds, dtype=float) if rows else None,
        bounds=[(None, None)] * len(variables),
        method="highs",
        options={"time_limit": timeout_ms / 1000.0},
    )
    if int(result.status) == 1:
        raise ProofDeferred("TIMEOUT", str(result.message))
    if not result.success:
        raise ProofDeferred(
            "MISSING_BOUND",
            f"no finite {'upper' if upper else 'lower'} bound for {variable}",
        )
    observed = -float(result.fun) if upper else float(result.fun)
    base = Fraction(str(observed)).limit_denominator(1_000_000)
    expansions = (
        Fraction(0),
        Fraction(1, 1_000_000_000),
        Fraction(1, 1_000_000),
        Fraction(1, 1000),
        Fraction(1),
    )
    for expansion in expansions:
        candidate = base + expansion if upper else base - expansion
        violation = (
            LinearInequality.make({variable: Fraction(-1)}, -candidate, True)
            if upper
            else LinearInequality.make({variable: Fraction(1)}, candidate, True)
        )
        proof = solve_linear_constraints(
            [*constraints, violation],
            timeout_ms=timeout_ms,
        )
        if proof.get("outcome") == "CERTIFIED":
            return candidate, proof
    raise ProofDeferred(
        "CERTIFICATE_RECONSTRUCTION_FAILED",
        f"the numerical {variable} bound did not produce an exact certificate",
    )


def _convex_candidate_bound(
    constraints: list[QuadraticConstraint],
    variable: str,
    *,
    upper: bool,
    timeout_ms: int,
) -> tuple[Fraction, dict[str, Any]]:
    if minimize is None or np is None:
        raise ProofDeferred("BLOCKED_INPUT", "SciPy convex optimization is unavailable")
    variables = sorted({
        name
        for constraint in constraints
        for name, _value in (*constraint.square, *constraint.linear)
    } | {variable})
    variable_index = {name: index for index, name in enumerate(variables)}

    def value(constraint: QuadraticConstraint, point) -> float:
        return (
            sum(
                float(coefficient) * point[variable_index[name]] ** 2
                for name, coefficient in constraint.square
            )
            + sum(
                float(coefficient) * point[variable_index[name]]
                for name, coefficient in constraint.linear
            )
            + float(constraint.constant)
        )

    objective_sign = -1.0 if upper else 1.0
    result = minimize(
        lambda point: objective_sign * point[variable_index[variable]],
        np.zeros(len(variables)),
        method="SLSQP",
        constraints=[
            {
                "type": "ineq",
                "fun": lambda point, item=item: -value(item, point),
            }
            for item in constraints
        ],
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )
    if not result.success:
        raise ProofDeferred(
            "MISSING_BOUND",
            f"no certified convex {'upper' if upper else 'lower'} bound for {variable}",
        )
    observed = float(result.x[variable_index[variable]])
    base = Fraction(str(observed)).limit_denominator(1_000_000)
    expansions = (
        Fraction(0),
        Fraction(1, 1_000_000_000),
        Fraction(1, 1_000_000),
        Fraction(1, 1000),
        Fraction(1),
    )
    for expansion in expansions:
        candidate = base + expansion if upper else base - expansion
        violation = (
            QuadraticConstraint.make(
                {},
                {variable: Fraction(-1)},
                candidate,
                True,
            )
            if upper
            else QuadraticConstraint.make(
                {},
                {variable: Fraction(1)},
                -candidate,
                True,
            )
        )
        proof = solve_convex_constraints(
            [*constraints, violation],
            timeout_ms=timeout_ms,
        )
        if proof.get("outcome") == "CERTIFIED":
            return candidate, proof
    raise ProofDeferred(
        "CERTIFICATE_RECONSTRUCTION_FAILED",
        f"the numerical convex {variable} bound did not produce an exact certificate",
    )


def _candidate_bound(
    linear_constraints: list[LinearInequality],
    convex_constraints: list[QuadraticConstraint],
    variable: str,
    *,
    upper: bool,
    timeout_ms: int,
) -> tuple[Fraction, dict[str, Any], str]:
    candidates: list[tuple[Fraction, dict[str, Any], str]] = []
    failures: list[ProofDeferred] = []
    try:
        value, proof = _linear_candidate_bound(
            linear_constraints,
            variable,
            upper=upper,
            timeout_ms=timeout_ms,
        )
        candidates.append((value, proof, "linear"))
    except ProofDeferred as exc:
        failures.append(exc)
    try:
        value, proof = _convex_candidate_bound(
            convex_constraints,
            variable,
            upper=upper,
            timeout_ms=timeout_ms,
        )
        candidates.append((value, proof, "convex"))
    except ProofDeferred as exc:
        failures.append(exc)
    if not candidates:
        failure = failures[-1]
        raise ProofDeferred(failure.reason_code, failure.detail)
    return (
        min(candidates, key=lambda item: item[0])
        if upper
        else max(candidates, key=lambda item: item[0])
    )


def _relaxed_constraints(
    polynomials: list[tuple[Polynomial, bool]],
    bounds: dict[str, tuple[Fraction, Fraction]],
) -> tuple[list[QuadraticConstraint], list[dict[str, Any]]]:
    constraints: list[QuadraticConstraint] = []
    reductions: list[dict[str, Any]] = []
    auxiliary = {name: f"reach_square__{name}" for name in bounds}
    for polynomial, strict in polynomials:
        linear: dict[str, Fraction] = {}
        constant = Fraction(0)
        for monomial, coefficient in polynomial.items():
            if not monomial:
                constant += coefficient
            elif len(monomial) == 1:
                linear[monomial[0]] = linear.get(monomial[0], Fraction(0)) + coefficient
            elif len(monomial) == 2 and monomial[0] == monomial[1]:
                name = monomial[0]
                if name not in auxiliary:
                    raise ProofDeferred("MISSING_BOUND", f"missing square bound for {name}")
                target = auxiliary[name]
                linear[target] = linear.get(target, Fraction(0)) + coefficient
            else:
                raise ProofDeferred("NOT_CONVEX", "cross terms are not supported")
        constraints.append(QuadraticConstraint.make({}, linear, constant, strict))

    for name, (lower, upper) in sorted(bounds.items()):
        square_name = auxiliary[name]
        constraints.append(QuadraticConstraint.make(
            {name: Fraction(1)},
            {square_name: Fraction(-1)},
            Fraction(0),
            False,
        ))
        constraints.append(QuadraticConstraint.make(
            {},
            {
                square_name: Fraction(1),
                name: -(lower + upper),
            },
            lower * upper,
            False,
        ))
        reductions.append({
            "variable": name,
            "auxiliary_square": square_name,
            "lower": f"{lower.numerator}/{lower.denominator}",
            "upper": f"{upper.numerator}/{upper.denominator}",
            "lower_relation": "square >= tangent envelope",
            "upper_relation": "square <= secant over the certified interval",
            "identity": "(value - lower) * (upper - value) >= 0",
        })
    return constraints, reductions


def run_convex_envelope_checker(
    reduced_case: ReducedCase,
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    """Prove infeasibility after a checked outer reduction of square terms."""

    try:
        comparisons = conjunctive_comparisons(reduced_case.expression, set())
        polynomials = [_oriented_polynomial(item) for item in comparisons]
        square_variables = sorted({
            monomial[0]
            for polynomial, _strict in polynomials
            for monomial in polynomial
            if len(monomial) == 2 and monomial[0] == monomial[1]
        })
        if not square_variables:
            raise ProofDeferred("NOT_CONVEX", "no diagonal square requires an envelope")
        if any(
            len(monomial) > 2
            or (len(monomial) == 2 and monomial[0] != monomial[1])
            for polynomial, _strict in polynomials
            for monomial in polynomial
        ):
            raise ProofDeferred("NOT_CONVEX", "cross terms or degree above two remain")
        linear = _linear_skeleton(comparisons)
        convex = _convex_skeleton(comparisons)
        bounds: dict[str, tuple[Fraction, Fraction]] = {}
        bound_proofs: list[dict[str, Any]] = []
        for variable in square_variables:
            lower, lower_proof, lower_method = _candidate_bound(
                linear,
                convex,
                variable,
                upper=False,
                timeout_ms=timeout_ms,
            )
            upper, upper_proof, upper_method = _candidate_bound(
                linear,
                convex,
                variable,
                upper=True,
                timeout_ms=timeout_ms,
            )
            if lower > upper:
                raise ProofDeferred("MALFORMED_OUTPUT", f"invalid interval for {variable}")
            bounds[variable] = (lower, upper)
            bound_proofs.append({
                "variable": variable,
                "lower": f"{lower.numerator}/{lower.denominator}",
                "upper": f"{upper.numerator}/{upper.denominator}",
                "lower_proof": lower_proof,
                "upper_proof": upper_proof,
                "lower_method": lower_method,
                "upper_method": upper_method,
            })
        relaxed, reductions = _relaxed_constraints(polynomials, bounds)
        linear_outer = [
            LinearInequality.make(
                dict(constraint.linear),
                -constraint.constant,
                constraint.strict,
            )
            for constraint in relaxed
            if not constraint.square
        ]
        result = solve_linear_constraints(linear_outer, timeout_ms=timeout_ms)
        if result.get("outcome") != "CERTIFIED":
            result = solve_convex_constraints(relaxed, timeout_ms=timeout_ms)
        if result.get("outcome") == "CERTIFIED":
            result.setdefault("proof", {}).update({
                "outer_reduction": {
                    "rule": "certified_square_tangent_secant_envelope_v1",
                    "source_expression_sha256": expression_hash(
                        reduced_case.expression
                    ),
                    "bounds": bound_proofs,
                    "relations": reductions,
                }
            })
        result["applicability_checks"] = {
            "accepted": True,
            "reduced_case_required": True,
            "case_id": reduced_case.case_id,
            "source_expression_sha256": expression_hash(reduced_case.expression),
            "square_variables": square_variables,
            "bounds_certified_exactly": True,
            "outer_reduction": True,
            "optimization_timeout_ms": int(timeout_ms),
        }
        return result
    except ProofDeferred as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": exc.reason_code,
            "detail": exc.detail,
            "applicability_checks": {"accepted": False},
        }
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
            "applicability_checks": {"accepted": False},
        }
