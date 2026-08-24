"""Linear optimization candidates and exact proof-certificate checking."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from certification.equations import Expr

from .full_model_reduction import ReducedCase

from .optimization_common import (
    fraction_text,
    linear_constraints,
    parse_fraction,
    serialize_linear_constraints,
)
from .proof_rules import LinearInequality, ProofDeferred

try:  # pragma: no cover - installation is checked by the integration run
    import numpy as np
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    np = None
    linprog = None


def _combined(
    constraints: list[LinearInequality],
    multipliers: list[Fraction],
) -> tuple[dict[str, Fraction], Fraction]:
    coefficients: dict[str, Fraction] = {}
    bound = Fraction(0)
    for constraint, multiplier in zip(constraints, multipliers):
        for name, value in constraint.coefficients:
            coefficients[name] = coefficients.get(name, Fraction(0)) + multiplier * value
        bound += multiplier * constraint.bound
    return (
        {name: value for name, value in sorted(coefficients.items()) if value},
        bound,
    )


def make_linear_certificate(
    constraints: list[LinearInequality],
    multipliers: list[Fraction],
) -> dict[str, Any]:
    coefficients, bound = _combined(constraints, multipliers)
    combined_strict = any(
        constraint.strict and multiplier > 0
        for constraint, multiplier in zip(constraints, multipliers)
    )
    return {
        "kind": "linear_infeasibility_weights_v1",
        "constraints": serialize_linear_constraints(constraints),
        "multipliers": [fraction_text(value) for value in multipliers],
        "combined_coefficients": {
            name: fraction_text(value) for name, value in coefficients.items()
        },
        "combined_bound": fraction_text(bound),
        "combined_strict": combined_strict,
        "contradiction": (
            "0 < 0"
            if bound == 0 and combined_strict
            else "0 <= combined_bound with combined_bound < 0"
        ),
    }


def verify_linear_certificate(
    certificate: dict[str, Any],
    constraints: list[LinearInequality],
) -> list[str]:
    errors: list[str] = []
    if certificate.get("kind") != "linear_infeasibility_weights_v1":
        errors.append("linear certificate kind is invalid")
    if certificate.get("constraints") != serialize_linear_constraints(constraints):
        errors.append("linear certificate constraints do not match the checked problem")
    try:
        multipliers = [parse_fraction(value) for value in certificate["multipliers"]]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("linear certificate multipliers are malformed")
        return errors
    if len(multipliers) != len(constraints):
        errors.append("linear certificate multiplier count is incorrect")
        return errors
    if any(value < 0 for value in multipliers):
        errors.append("linear certificate contains a negative multiplier")
    coefficients, bound = _combined(constraints, multipliers)
    combined_strict = any(
        constraint.strict and multiplier > 0
        for constraint, multiplier in zip(constraints, multipliers)
    )
    if coefficients:
        errors.append("linear weighted sum does not eliminate every variable")
    if bound > 0 or (bound == 0 and not combined_strict):
        errors.append("linear weighted sum does not produce a contradiction")
    recorded_coefficients = {
        name: fraction_text(value) for name, value in coefficients.items()
    }
    if certificate.get("combined_coefficients") != recorded_coefficients:
        errors.append("linear certificate combined coefficients are incorrect")
    if certificate.get("combined_bound") != fraction_text(bound):
        errors.append("linear certificate combined bound is incorrect")
    if certificate.get("combined_strict") is not combined_strict:
        errors.append("linear certificate combined strictness is incorrect")
    return errors


def _exact_candidate(
    constraints: list[LinearInequality],
    candidate: list[float],
    *,
    normalization: str,
) -> list[Fraction] | None:
    direct = [Fraction(str(value)).limit_denominator(1_000_000) for value in candidate]
    if not verify_linear_certificate(make_linear_certificate(constraints, direct), constraints):
        return direct

    support = [index for index, value in enumerate(candidate) if value > 1e-9]
    if not support:
        return None
    variables = sorted({name for item in constraints for name, _ in item.coefficients})
    rows: list[list[Fraction]] = []
    right: list[Fraction] = []
    for variable in variables:
        rows.append([
            constraints[index].coeff_dict().get(variable, Fraction(0))
            for index in support
        ])
        right.append(Fraction(0))
    rows.append([constraints[index].bound for index in support])
    right.append(Fraction(-1) if normalization == "negative_bound" else Fraction(0))
    if normalization == "strict_weight":
        rows.append([
            Fraction(1) if constraints[index].strict else Fraction(0)
            for index in support
        ])
        right.append(Fraction(1))

    matrix = [row + [value] for row, value in zip(rows, right)]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(len(support)):
        selected = next(
            (row for row in range(pivot_row, len(matrix)) if matrix[row][column]),
            None,
        )
        if selected is None:
            continue
        matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
        pivot = matrix[pivot_row][column]
        matrix[pivot_row] = [value / pivot for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row or matrix[row][column] == 0:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                left - factor * right_value
                for left, right_value in zip(matrix[row], matrix[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    for row in matrix:
        if all(value == 0 for value in row[:-1]) and row[-1] != 0:
            return None

    free_columns = [column for column in range(len(support)) if column not in pivot_columns]
    scaled = list(candidate)
    if normalization == "negative_bound":
        normalizer = -sum(
            float(constraint.bound) * value
            for constraint, value in zip(constraints, candidate)
        )
    else:
        normalizer = sum(
            value
            for constraint, value in zip(constraints, candidate)
            if constraint.strict
        )
    if normalizer > 1e-12:
        scaled = [value / normalizer for value in candidate]
    solution = [Fraction(0) for _ in support]
    for column in free_columns:
        solution[column] = Fraction(str(scaled[support[column]])).limit_denominator(1_000_000)
    for row_index in range(len(pivot_columns) - 1, -1, -1):
        column = pivot_columns[row_index]
        row = matrix[row_index]
        solution[column] = row[-1] - sum(
            row[free] * solution[free] for free in free_columns
        )
    multipliers = [Fraction(0) for _ in constraints]
    for index, value in zip(support, solution):
        multipliers[index] = value
    if verify_linear_certificate(make_linear_certificate(constraints, multipliers), constraints):
        return None
    return multipliers


def _exact_witness(
    constraints: list[LinearInequality],
    variables: list[str],
    candidate: list[float],
) -> dict[str, str] | None:
    values = {
        name: Fraction(str(value)).limit_denominator(1_000_000)
        for name, value in zip(variables, candidate)
    }
    for constraint in constraints:
        left = sum(
            coefficient * values[name]
            for name, coefficient in constraint.coefficients
        )
        if left > constraint.bound:
            return None
        if constraint.strict and left == constraint.bound:
            return None
    return {name: fraction_text(value) for name, value in sorted(values.items())}


def solve_linear_constraints(
    constraints: list[LinearInequality],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    if linprog is None or np is None:
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "SciPy linear optimization is unavailable",
        }
    if not constraints:
        return {
            "outcome": "VIOLATION",
            "reason_code": "COUNTEREXAMPLE_REPLAYED",
            "detail": "the counterexample conjunction has no constraints",
            "proof": {"counterexample": {}},
        }
    variables = sorted({name for item in constraints for name, _ in item.coefficients})
    coefficient_rows = [
        [float(item.coeff_dict().get(name, 0)) for name in variables]
        for item in constraints
    ]
    bounds = [float(item.bound) for item in constraints]
    dual_equalities = [
        [row[column] for row in coefficient_rows]
        for column in range(len(variables))
    ]
    try:
        negative_candidate = linprog(
            np.zeros(len(constraints)),
            A_ub=np.asarray([bounds], dtype=float),
            b_ub=np.asarray([-1.0], dtype=float),
            A_eq=(np.asarray(dual_equalities, dtype=float) if dual_equalities else None),
            b_eq=(np.zeros(len(dual_equalities)) if dual_equalities else None),
            bounds=[(0.0, None)] * len(constraints),
            method="highs",
            options={"time_limit": timeout_ms / 1000.0},
        )
    except Exception as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
        }
    if negative_candidate.success:
        multipliers = _exact_candidate(
            constraints,
            list(negative_candidate.x),
            normalization="negative_bound",
        )
        if multipliers is not None:
            certificate = make_linear_certificate(constraints, multipliers)
            errors = verify_linear_certificate(certificate, constraints)
            if not errors:
                return {
                    "outcome": "CERTIFIED",
                    "reason_code": "",
                    "detail": "exact weighted contradiction verified",
                    "proof": {
                        "solver": "scipy.optimize.linprog.highs",
                        "solver_status": int(negative_candidate.status),
                        "certificate_branch": "negative_combined_bound",
                        "certificate": certificate,
                        "certificate_verification": "PASSED",
                    },
                }
    if int(negative_candidate.status) == 1:
        return {
            "outcome": "DEFERRED",
            "reason_code": "TIMEOUT",
            "detail": str(negative_candidate.message),
        }

    strict_indices = [index for index, item in enumerate(constraints) if item.strict]
    if strict_indices:
        strict_weights = [
            -1.0 if index in strict_indices else 0.0
            for index in range(len(constraints))
        ]
        strict_equalities = list(dual_equalities) + [bounds]
        strict_right = [0.0] * len(dual_equalities) + [0.0]
        try:
            strict_candidate = linprog(
                np.zeros(len(constraints)),
                A_ub=np.asarray([strict_weights], dtype=float),
                b_ub=np.asarray([-1.0], dtype=float),
                A_eq=np.asarray(strict_equalities, dtype=float),
                b_eq=np.asarray(strict_right, dtype=float),
                bounds=[(0.0, None)] * len(constraints),
                method="highs",
                options={"time_limit": timeout_ms / 1000.0},
            )
        except Exception as exc:
            return {
                "outcome": "DEFERRED",
                "reason_code": "MALFORMED_OUTPUT",
                "detail": str(exc),
            }
        if strict_candidate.success:
            multipliers = _exact_candidate(
                constraints,
                list(strict_candidate.x),
                normalization="strict_weight",
            )
            if multipliers is not None:
                certificate = make_linear_certificate(constraints, multipliers)
                errors = verify_linear_certificate(certificate, constraints)
                if not errors:
                    return {
                        "outcome": "CERTIFIED",
                        "reason_code": "",
                        "detail": "exact weighted contradiction verified",
                        "proof": {
                            "solver": "scipy.optimize.linprog.highs",
                            "solver_status": int(strict_candidate.status),
                            "certificate_branch": "zero_bound_with_strict_constraint",
                            "certificate": certificate,
                            "certificate_verification": "PASSED",
                        },
                    }
        if int(strict_candidate.status) == 1:
            return {
                "outcome": "DEFERRED",
                "reason_code": "TIMEOUT",
                "detail": str(strict_candidate.message),
            }

    if not variables:
        feasible = all(
            Fraction(0) < item.bound
            if item.strict
            else Fraction(0) <= item.bound
            for item in constraints
        )
        if feasible:
            return {
                "outcome": "VIOLATION",
                "reason_code": "COUNTEREXAMPLE_REPLAYED",
                "detail": "the constant counterexample was replayed exactly",
                "proof": {
                    "counterexample": {},
                    "constraints": serialize_linear_constraints(constraints),
                    "counterexample_verification": "PASSED",
                },
            }
        return {
            "outcome": "DEFERRED",
            "reason_code": "CERTIFICATE_RECONSTRUCTION_FAILED",
            "detail": "the constant contradiction did not produce an exact certificate",
        }

    try:
        primal = linprog(
            np.zeros(len(variables)),
            A_ub=np.asarray(coefficient_rows, dtype=float),
            b_ub=np.asarray(bounds, dtype=float),
            bounds=[(None, None)] * len(variables),
            method="highs",
            options={"time_limit": timeout_ms / 1000.0},
        )
    except Exception as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
        }
    if primal.success:
        witness = _exact_witness(constraints, variables, list(primal.x))
        if witness is not None:
            return {
                "outcome": "VIOLATION",
                "reason_code": "COUNTEREXAMPLE_REPLAYED",
                "detail": "the linear counterexample was replayed with exact numbers",
                "proof": {
                    "solver": "scipy.optimize.linprog.highs",
                    "counterexample": witness,
                    "constraints": serialize_linear_constraints(constraints),
                    "counterexample_verification": "PASSED",
                },
            }
        return {
            "outcome": "DEFERRED",
            "reason_code": "COUNTEREXAMPLE_REPLAY_FAILED",
            "detail": "the numerical counterexample did not satisfy the exact constraints",
        }
    return {
        "outcome": "DEFERRED",
        "reason_code": (
            "TIMEOUT" if int(primal.status) == 1 else "CERTIFICATE_RECONSTRUCTION_FAILED"
        ),
        "detail": str(primal.message),
    }


def run_linear_checker(
    reduced_case: ReducedCase,
    boolean_variables: set[str],
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    if not isinstance(reduced_case, ReducedCase):
        return {
            "outcome": "DEFERRED",
            "reason_code": "UNREDUCED_INPUT",
            "detail": "linear checker accepts only a recorded reduced case",
            "applicability_checks": {
                "accepted": False,
                "reduced_case_required": True,
                "optimization_timeout_ms": int(timeout_ms),
            },
        }
    counterexample = reduced_case.expression
    try:
        constraints = linear_constraints(counterexample, boolean_variables)
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
        result = solve_linear_constraints(constraints, timeout_ms=timeout_ms)
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
        "optimization_timeout_ms": int(timeout_ms),
    }
    return result
