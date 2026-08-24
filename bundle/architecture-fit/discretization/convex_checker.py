"""Convex optimization candidates and exact dual-certificate checking."""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

from certification.equations import Expr

from .full_model_reduction import ReducedCase
from .optimization_common import (
    QuadraticConstraint,
    fraction_text,
    parse_fraction,
    quadratic_constraints,
    serialize_quadratic_constraint,
    serialize_quadratic_constraints,
)
from .proof_rules import ProofDeferred

try:  # pragma: no cover - installation is checked by the integration run
    import numpy as np
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    np = None
    minimize = None


def _combined_quadratic(
    constraints: list[QuadraticConstraint],
    multipliers: list[Fraction],
) -> QuadraticConstraint:
    square: dict[str, Fraction] = {}
    linear: dict[str, Fraction] = {}
    constant = Fraction(0)
    strict = False
    for constraint, multiplier in zip(constraints, multipliers):
        for name, value in constraint.square:
            square[name] = square.get(name, Fraction(0)) + multiplier * value
        for name, value in constraint.linear:
            linear[name] = linear.get(name, Fraction(0)) + multiplier * value
        constant += multiplier * constraint.constant
        strict = strict or (constraint.strict and multiplier > 0)
    return QuadraticConstraint.make(square, linear, constant, strict)


def _global_lower_bound(expression: QuadraticConstraint) -> Fraction | None:
    square = dict(expression.square)
    linear = dict(expression.linear)
    result = expression.constant
    for variable in sorted(set(square) | set(linear)):
        quadratic = square.get(variable, Fraction(0))
        coefficient = linear.get(variable, Fraction(0))
        if quadratic < 0:
            return None
        if quadratic == 0:
            if coefficient != 0:
                return None
            continue
        result -= coefficient * coefficient / (4 * quadratic)
    return result


def make_convex_certificate(
    constraints: list[QuadraticConstraint],
    multipliers: list[Fraction],
) -> dict[str, Any]:
    combined = _combined_quadratic(constraints, multipliers)
    lower_bound = _global_lower_bound(combined)
    return {
        "kind": "convex_dual_bound_v1",
        "constraints": serialize_quadratic_constraints(constraints),
        "multipliers": [fraction_text(value) for value in multipliers],
        "combined_quadratic": serialize_quadratic_constraint(combined),
        "global_lower_bound": (
            fraction_text(lower_bound) if lower_bound is not None else None
        ),
        "contradiction": (
            "weighted constraints require a value < 0 but their global lower bound is 0"
            if lower_bound == 0 and combined.strict
            else "weighted constraints are <= 0 but their global lower bound is > 0"
        ),
    }


def verify_convex_certificate(
    certificate: dict[str, Any],
    constraints: list[QuadraticConstraint],
) -> list[str]:
    errors: list[str] = []
    if certificate.get("kind") != "convex_dual_bound_v1":
        errors.append("convex certificate kind is invalid")
    if certificate.get("constraints") != serialize_quadratic_constraints(constraints):
        errors.append("convex certificate constraints do not match the checked problem")
    if any(value < 0 for item in constraints for _name, value in item.square):
        errors.append("convex certificate contains a nonconvex source constraint")
    try:
        multipliers = [parse_fraction(value) for value in certificate["multipliers"]]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("convex certificate multipliers are malformed")
        return errors
    if len(multipliers) != len(constraints):
        errors.append("convex certificate multiplier count is incorrect")
        return errors
    if any(value < 0 for value in multipliers):
        errors.append("convex certificate contains a negative multiplier")
    if not any(value > 0 for value in multipliers):
        errors.append("convex certificate multipliers are all zero")
    combined = _combined_quadratic(constraints, multipliers)
    lower_bound = _global_lower_bound(combined)
    if lower_bound is None or lower_bound < 0 or (
        lower_bound == 0 and not combined.strict
    ):
        errors.append("convex weighted sum does not exclude the constrained region")
    if certificate.get("combined_quadratic") != serialize_quadratic_constraint(combined):
        errors.append("convex certificate combined quadratic is incorrect")
    recorded_bound = fraction_text(lower_bound) if lower_bound is not None else None
    if certificate.get("global_lower_bound") != recorded_bound:
        errors.append("convex certificate global lower bound is incorrect")
    return errors


def _exact_candidate(
    constraints: list[QuadraticConstraint],
    values: list[float],
) -> list[Fraction] | None:
    total = sum(max(0.0, value) for value in values)
    if total <= 0:
        return None
    multipliers = [
        Fraction(str(max(0.0, value) / total)).limit_denominator(1_000_000)
        for value in values
    ]
    certificate = make_convex_certificate(constraints, multipliers)
    if verify_convex_certificate(certificate, constraints):
        return None
    return multipliers


def _float_lower_bound(
    constraints: list[QuadraticConstraint],
    multipliers: Any,
) -> float:
    square: dict[str, float] = {}
    linear: dict[str, float] = {}
    constant = 0.0
    for constraint, multiplier in zip(constraints, multipliers):
        for name, value in constraint.square:
            square[name] = square.get(name, 0.0) + float(multiplier) * float(value)
        for name, value in constraint.linear:
            linear[name] = linear.get(name, 0.0) + float(multiplier) * float(value)
        constant += float(multiplier) * float(constraint.constant)
    result = constant
    for variable in set(square) | set(linear):
        quadratic = square.get(variable, 0.0)
        coefficient = linear.get(variable, 0.0)
        if quadratic <= 1e-12:
            if abs(coefficient) > 1e-10:
                return float("-inf")
            continue
        result -= coefficient * coefficient / (4.0 * quadratic)
    return result


class _SolverTimedOut(RuntimeError):
    pass


def solve_convex_constraints(
    constraints: list[QuadraticConstraint],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    if not constraints:
        return {
            "outcome": "VIOLATION",
            "reason_code": "COUNTEREXAMPLE_REPLAYED",
            "detail": "the counterexample conjunction has no constraints",
            "proof": {"counterexample": {}},
        }

    exact_candidates = []
    for index in range(len(constraints)):
        candidate = [Fraction(0) for _ in constraints]
        candidate[index] = Fraction(1)
        exact_candidates.append(candidate)
    exact_candidates.append([Fraction(1, len(constraints)) for _ in constraints])
    for multipliers in exact_candidates:
        certificate = make_convex_certificate(constraints, multipliers)
        if not verify_convex_certificate(certificate, constraints):
            return {
                "outcome": "CERTIFIED",
                "reason_code": "",
                "detail": "exact convex dual bound verified",
                "proof": {
                    "solver": "exact_dual_candidate",
                    "certificate": certificate,
                    "certificate_verification": "PASSED",
                },
            }

    if minimize is None or np is None:
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "SciPy convex optimization is unavailable",
        }
    deadline = time.monotonic() + timeout_ms / 1000.0

    def objective(values):
        if time.monotonic() > deadline:
            raise _SolverTimedOut
        lower_bound = _float_lower_bound(constraints, values)
        if lower_bound == float("-inf"):
            return 1e100
        return -lower_bound

    try:
        candidate = minimize(
            objective,
            np.full(len(constraints), 1.0 / len(constraints)),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(constraints),
            constraints=[{
                "type": "eq",
                "fun": lambda values: float(np.sum(values) - 1.0),
            }],
            options={"maxiter": 1000, "ftol": 1e-12},
        )
    except _SolverTimedOut:
        return {
            "outcome": "DEFERRED",
            "reason_code": "TIMEOUT",
            "detail": f"convex optimization exceeded {timeout_ms} ms",
        }
    except Exception as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
        }
    if candidate.success:
        multipliers = _exact_candidate(constraints, list(candidate.x))
        if multipliers is not None:
            certificate = make_convex_certificate(constraints, multipliers)
            if not verify_convex_certificate(certificate, constraints):
                return {
                    "outcome": "CERTIFIED",
                    "reason_code": "",
                    "detail": "exact convex dual bound verified",
                    "proof": {
                        "solver": "scipy.optimize.minimize.slsqp",
                        "solver_status": int(candidate.status),
                        "certificate": certificate,
                        "certificate_verification": "PASSED",
                    },
                }
        return {
            "outcome": "DEFERRED",
            "reason_code": "CERTIFICATE_RECONSTRUCTION_FAILED",
            "detail": "the numerical convex result did not produce an exact certificate",
        }
    return {
        "outcome": "DEFERRED",
        "reason_code": (
            "TIMEOUT" if time.monotonic() > deadline else "NUMERIC_BOUND_INCONCLUSIVE"
        ),
        "detail": str(candidate.message),
    }


def run_convex_checker(
    reduced_case: ReducedCase,
    boolean_variables: set[str],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    if not isinstance(reduced_case, ReducedCase):
        return {
            "outcome": "DEFERRED",
            "reason_code": "UNREDUCED_INPUT",
            "detail": "convex checker accepts only a recorded reduced case",
            "applicability_checks": {
                "accepted": False,
                "reduced_case_required": True,
                "optimization_timeout_ms": int(timeout_ms),
            },
        }
    counterexample = reduced_case.expression
    try:
        constraints = quadratic_constraints(counterexample, boolean_variables)
    except ProofDeferred as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": exc.reason_code,
            "detail": exc.detail,
            "applicability_checks": {
                "accepted": False,
                "optimization_timeout_ms": int(timeout_ms),
            },
        }
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
            "applicability_checks": {
                "accepted": False,
                "optimization_timeout_ms": int(timeout_ms),
            },
        }
    try:
        result = solve_convex_constraints(constraints, timeout_ms=timeout_ms)
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        result = {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
        }
    result["applicability_checks"] = {
        "accepted": True,
        "reduced_case_required": True,
        "case_id": reduced_case.case_id,
        "parent_expression_sha256": reduced_case.parent_hash,
        "constraint_count": len(constraints),
        "strict_boundaries": any(item.strict for item in constraints),
        "strict_boundaries_checked_exactly": True,
        "logical_case_enumeration": False,
        "convexity_checked_exactly": True,
        "optimization_timeout_ms": int(timeout_ms),
    }
    return result
