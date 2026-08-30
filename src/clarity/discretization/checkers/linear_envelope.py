"""Certified linear outer reductions for bounded polynomial products."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from clarity.certification.equations import Const, Expr, Op, RawRef, Var

from .linear import solve_linear_constraints
from ..model.optimization import conjunctive_comparisons, fraction_text
from ..model.proof_rules import (
    LinearInequality,
    ProofDeferred,
    fraction_value,
)
from ..model.reduction import ReducedCase, expression_hash

try:  # pragma: no cover - installation is checked by integration runs
    import numpy as np
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    np = None
    linprog = None


Polynomial = dict[tuple[str, ...], Fraction]
MAX_EXPANDED_TERMS = 10000


def _add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, Fraction(0)) + coefficient
        if result[monomial] == 0:
            del result[monomial]
    if len(result) > MAX_EXPANDED_TERMS:
        raise ProofDeferred(
            "REDUCTION_SIZE_LIMIT",
            f"polynomial expansion exceeds {MAX_EXPANDED_TERMS} terms",
        )
    return result


def _scale(value: Fraction, expression: Polynomial) -> Polynomial:
    return {
        monomial: value * coefficient
        for monomial, coefficient in expression.items()
        if value * coefficient
    }


def _polynomial(expression: Expr) -> Polynomial:
    if isinstance(expression, Const):
        return {(): fraction_value(expression.value)}
    if isinstance(expression, Var):
        return {(expression.name,): Fraction(1)}
    if isinstance(expression, RawRef):
        return {(expression.path,): Fraction(1)}
    if not isinstance(expression, Op):
        raise ProofDeferred(
            "NOT_LINEARIZABLE",
            f"unsupported expression {expression.pretty()}",
        )
    if expression.op == "+":
        result: Polynomial = {}
        for argument in expression.args:
            result = _add(result, _polynomial(argument))
        return result
    if expression.op == "-":
        if len(expression.args) == 1:
            return _scale(Fraction(-1), _polynomial(expression.args[0]))
        result = _polynomial(expression.args[0])
        for argument in expression.args[1:]:
            result = _add(result, _scale(Fraction(-1), _polynomial(argument)))
        return result
    if expression.op == "*" and len(expression.args) == 2:
        left = _polynomial(expression.args[0])
        right = _polynomial(expression.args[1])
        result: Polynomial = {}
        for left_term, left_value in left.items():
            for right_term, right_value in right.items():
                result = _add(result, {
                    tuple(sorted(left_term + right_term)): left_value * right_value
                })
        return result
    if expression.op == "/" and len(expression.args) == 2:
        numerator = _polynomial(expression.args[0])
        denominator = _polynomial(expression.args[1])
        if set(denominator) != {()} or denominator[()] == 0:
            raise ProofDeferred(
                "NOT_LINEARIZABLE",
                "division is not by a nonzero constant",
            )
        return _scale(Fraction(1) / denominator[()], numerator)
    raise ProofDeferred(
        "NOT_LINEARIZABLE",
        f"unsupported operation {expression.op}",
    )


def _oriented_polynomial(comparison) -> tuple[Polynomial, bool]:
    strict = comparison.op in {"<", ">"}
    if comparison.op in {"<=", "<"}:
        expression = Op("-", (comparison.left, comparison.right))
    elif comparison.op in {">=", ">"}:
        expression = Op("-", (comparison.right, comparison.left))
    else:
        raise ProofDeferred(
            "NOT_LINEARIZABLE",
            f"unsupported comparison {comparison.op}",
        )
    return _polynomial(expression), strict


def _linear_constraint(
    polynomial: Polynomial,
    strict: bool,
) -> LinearInequality:
    coefficients: dict[str, Fraction] = {}
    constant = polynomial.get((), Fraction(0))
    for monomial, coefficient in polynomial.items():
        if not monomial:
            continue
        if len(monomial) != 1:
            raise ProofDeferred("NOT_LINEAR", "a nonlinear monomial remains")
        coefficients[monomial[0]] = (
            coefficients.get(monomial[0], Fraction(0)) + coefficient
        )
    return LinearInequality.make(coefficients, -constant, strict)


def _candidate_bound(
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
    right = [float(constraint.bound) for constraint in constraints]
    objective = [0.0 for _name in variables]
    objective[variables.index(variable)] = -1.0 if upper else 1.0
    result = linprog(
        np.asarray(objective),
        A_ub=np.asarray(rows, dtype=float) if rows else None,
        b_ub=np.asarray(right, dtype=float) if rows else None,
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


def _optional_bounds(
    constraints: list[LinearInequality],
    variable: str,
    *,
    timeout_ms: int,
) -> tuple[tuple[Fraction | None, Fraction | None], dict[str, Any]]:
    lower: Fraction | None = None
    upper: Fraction | None = None
    lower_proof: dict[str, Any] | None = None
    upper_proof: dict[str, Any] | None = None
    try:
        lower, lower_proof = _candidate_bound(
            constraints, variable, upper=False, timeout_ms=timeout_ms
        )
    except ProofDeferred as exc:
        if exc.reason_code not in {"MISSING_BOUND", "CERTIFICATE_RECONSTRUCTION_FAILED"}:
            raise
    try:
        upper, upper_proof = _candidate_bound(
            constraints, variable, upper=True, timeout_ms=timeout_ms
        )
    except ProofDeferred as exc:
        if exc.reason_code not in {"MISSING_BOUND", "CERTIFICATE_RECONSTRUCTION_FAILED"}:
            raise
    if lower is not None and upper is not None and lower > upper:
        raise ProofDeferred("MALFORMED_OUTPUT", f"invalid interval for {variable}")
    record: dict[str, Any] = {
        "variable": variable,
        "lower": fraction_text(lower) if lower is not None else None,
        "upper": fraction_text(upper) if upper is not None else None,
    }
    if lower_proof is not None:
        record["lower_proof"] = lower_proof
    if upper_proof is not None:
        record["upper_proof"] = upper_proof
    return (lower, upper), record


def _product_constraints(
    left: str,
    right: str,
    product: str,
    left_bounds: tuple[Fraction | None, Fraction | None],
    right_bounds: tuple[Fraction | None, Fraction | None],
) -> list[LinearInequality]:
    left_lower, left_upper = left_bounds
    right_lower, right_upper = right_bounds
    if left == right:
        constraints = [
            LinearInequality.make({product: Fraction(-1)}, Fraction(0))
        ]
        tangent_points = sorted({
            value for value in (left_lower, left_upper) if value is not None
        })
        for point in tangent_points:
            constraints.append(LinearInequality.make(
                {left: 2 * point, product: Fraction(-1)},
                point * point,
            ))
        if left_lower is not None and left_upper is not None:
            constraints.append(LinearInequality.make(
                {product: Fraction(1), left: -(left_lower + left_upper)},
                -left_lower * left_upper,
            ))
        return constraints
    if None not in {left_lower, left_upper, right_lower, right_upper}:
        assert left_lower is not None and left_upper is not None
        assert right_lower is not None and right_upper is not None
        return [
            LinearInequality.make(
                {right: left_lower, left: right_lower, product: Fraction(-1)},
                left_lower * right_lower,
            ),
            LinearInequality.make(
                {right: left_upper, left: right_upper, product: Fraction(-1)},
                left_upper * right_upper,
            ),
            LinearInequality.make(
                {product: Fraction(1), right: -left_upper, left: -right_lower},
                -left_upper * right_lower,
            ),
            LinearInequality.make(
                {product: Fraction(1), right: -left_lower, left: -right_upper},
                -left_lower * right_upper,
            ),
        ]
    if left_lower is not None and left_lower >= 0 and (
        right_lower is not None and right_upper is not None
    ):
        return [
            LinearInequality.make(
                {left: right_lower, product: Fraction(-1)}, Fraction(0)
            ),
            LinearInequality.make(
                {product: Fraction(1), left: -right_upper}, Fraction(0)
            ),
        ]
    if left_upper is not None and left_upper <= 0 and (
        right_lower is not None and right_upper is not None
    ):
        return [
            LinearInequality.make(
                {left: right_upper, product: Fraction(-1)}, Fraction(0)
            ),
            LinearInequality.make(
                {product: Fraction(1), left: -right_lower}, Fraction(0)
            ),
        ]
    if right_lower is not None and right_lower >= 0 and (
        left_lower is not None and left_upper is not None
    ):
        return [
            LinearInequality.make(
                {right: left_lower, product: Fraction(-1)}, Fraction(0)
            ),
            LinearInequality.make(
                {product: Fraction(1), right: -left_upper}, Fraction(0)
            ),
        ]
    if right_upper is not None and right_upper <= 0 and (
        left_lower is not None and left_upper is not None
    ):
        return [
            LinearInequality.make(
                {right: left_upper, product: Fraction(-1)}, Fraction(0)
            ),
            LinearInequality.make(
                {product: Fraction(1), right: -left_lower}, Fraction(0)
            ),
        ]
    return []


def run_linear_envelope_checker(
    reduced_case: ReducedCase,
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    """Prove infeasibility after checked linear bounds on every product."""

    try:
        comparisons = conjunctive_comparisons(reduced_case.expression, set())
        polynomials = [_oriented_polynomial(item) for item in comparisons]
        nonlinear_monomials = sorted({
            monomial
            for polynomial, _strict in polynomials
            for monomial in polynomial
            if len(monomial) >= 2
        }, key=lambda item: (len(item), item))
        if not nonlinear_monomials:
            raise ProofDeferred("NOT_LINEARIZABLE", "no product requires reduction")

        constraints = [
            _linear_constraint(polynomial, strict)
            for polynomial, strict in polynomials
            if all(len(monomial) <= 1 for monomial in polynomial)
        ]
        skeleton_result = solve_linear_constraints(
            constraints,
            timeout_ms=timeout_ms,
        )
        if skeleton_result.get("outcome") == "CERTIFIED":
            skeleton_result.setdefault("proof", {})["outer_reduction"] = {
                "rule": "linear_skeleton_outer_reduction_v1",
                "source_expression_sha256": expression_hash(reduced_case.expression),
                "dropped_nonlinear_comparison_count": sum(
                    any(len(monomial) >= 2 for monomial in polynomial)
                    for polynomial, _strict in polynomials
                ),
            }
            return skeleton_result

        bound_cache: dict[str, tuple[Fraction | None, Fraction | None]] = {}
        bound_records: list[dict[str, Any]] = []
        base_variables = sorted({
            variable
            for monomial in nonlinear_monomials
            for variable in monomial
        })
        for variable in base_variables:
            interval, record = _optional_bounds(
                constraints,
                variable,
                timeout_ms=timeout_ms,
            )
            bound_cache[variable] = interval
            bound_records.append(record)

        product_names: dict[tuple[str, ...], str] = {}
        product_records: list[dict[str, Any]] = []

        def bound_text(value: Fraction | None) -> str | None:
            return fraction_text(value) if value is not None else None

        def ensure_product(monomial: tuple[str, ...]) -> str:
            if len(monomial) == 1:
                return monomial[0]
            if monomial in product_names:
                return product_names[monomial]
            repeated = next(
                (name for name in monomial if monomial.count(name) >= 2),
                None,
            )
            if repeated is not None:
                left_term = (repeated, repeated)
                remaining = list(monomial)
                remaining.remove(repeated)
                remaining.remove(repeated)
                right_term = tuple(remaining)
                if not right_term:
                    left = repeated
                    right = repeated
                else:
                    left = ensure_product(left_term)
                    right = ensure_product(right_term)
            else:
                split = len(monomial) // 2
                left = ensure_product(monomial[:split])
                right = ensure_product(monomial[split:])
            product_name = "reach_product__" + "__".join(monomial)
            suffix = 2
            while product_name in bound_cache:
                product_name = (
                    "reach_product__" + "__".join(monomial) + f"__{suffix}"
                )
                suffix += 1
            product_names[monomial] = product_name
            added = _product_constraints(
                left,
                right,
                product_name,
                bound_cache[left],
                bound_cache[right],
            )
            constraints.extend(added)
            interval, record = _optional_bounds(
                constraints,
                product_name,
                timeout_ms=timeout_ms,
            )
            bound_cache[product_name] = interval
            bound_records.append(record)
            product_records.append({
                "monomial": list(monomial),
                "left": left,
                "right": right,
                "product": product_name,
                "left_lower": bound_text(bound_cache[left][0]),
                "left_upper": bound_text(bound_cache[left][1]),
                "right_lower": bound_text(bound_cache[right][0]),
                "right_upper": bound_text(bound_cache[right][1]),
                "envelope_constraint_count": len(added),
            })
            return product_name

        for monomial in nonlinear_monomials:
            ensure_product(monomial)

        for polynomial, strict in polynomials:
            transformed: Polynomial = {}
            for monomial, coefficient in polynomial.items():
                target = monomial
                if len(monomial) >= 2:
                    target = (product_names[monomial],)
                transformed[target] = transformed.get(target, Fraction(0)) + coefficient
            constraints.append(_linear_constraint(transformed, strict))

        result = solve_linear_constraints(constraints, timeout_ms=timeout_ms)
        if result.get("outcome") == "CERTIFIED":
            result.setdefault("proof", {})["outer_reduction"] = {
                "rule": "certified_bounded_product_linear_envelope_v1",
                "source_expression_sha256": expression_hash(reduced_case.expression),
                "bounds": bound_records,
                "products": product_records,
                "transformed_comparison_count": len(polynomials),
            }
        else:
            result = {
                "outcome": "DEFERRED",
                "reason_code": (
                    result.get("reason_code")
                    if result.get("outcome") == "DEFERRED"
                    else "REACHABILITY_BOUND_INCONCLUSIVE"
                ),
                "detail": (
                    result.get("detail")
                    if result.get("outcome") == "DEFERRED"
                    else "the checked linear outer reduction remains feasible"
                ),
            }
        result["applicability_checks"] = {
            "accepted": True,
            "reduced_case_required": True,
            "case_id": reduced_case.case_id,
            "source_expression_sha256": expression_hash(reduced_case.expression),
            "bounded_product_count": len(product_records),
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
