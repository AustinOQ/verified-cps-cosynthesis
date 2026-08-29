"""Exact normalized forms shared by the linear and convex certificate paths."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from certification.equations import Const, Expr, Op

from .proof_rules import (
    Comparison,
    LinearInequality,
    ProofDeferred,
    comparison_inequalities,
    expression_symbols,
)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_fraction(value: Any) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("exact number must be a string")
    return Fraction(value)


def _comparison(op: str, left: Expr, right: Expr, truth: bool) -> list[Comparison]:
    if truth:
        if op == "==":
            return [Comparison("<=", left, right), Comparison(">=", left, right)]
        return [Comparison(op, left, right)]
    inverse = {">": "<=", "<": ">=", ">=": "<", "<=": ">"}
    if op in inverse:
        return [Comparison(inverse[op], left, right)]
    if op == "==":
        raise ProofDeferred(
            "NON_CONJUNCTIVE_LOGIC",
            "the negation of numeric equality requires separate cases",
        )
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"unsupported comparison {op}")


def conjunctive_comparisons(
    expr: Expr,
    boolean_variables: set[str],
) -> list[Comparison]:
    """Accept one conjunction without constructing or enumerating logical cases."""

    used_booleans = expression_symbols(expr) & boolean_variables
    if used_booleans:
        raise ProofDeferred(
            "BOOLEAN_CASES_REQUIRED",
            "Boolean variables require the later symbolic checker: "
            + ", ".join(sorted(used_booleans)),
        )

    def visit(node: Expr, truth: bool = True) -> list[Comparison]:
        if isinstance(node, Const) and isinstance(node.value, bool):
            if node.value is truth:
                return []
            return [Comparison("<=", Const(1), Const(0))]
        if not isinstance(node, Op):
            raise ProofDeferred(
                "UNSUPPORTED_EXPRESSION",
                f"non-Boolean expression {node.pretty()}",
            )
        if node.op == "not":
            return visit(node.args[0], not truth)
        if node.op == "and" and truth:
            out: list[Comparison] = []
            for arg in node.args:
                out.extend(visit(arg, True))
            return out
        if node.op == "or" and not truth:
            out = []
            for arg in node.args:
                out.extend(visit(arg, False))
            return out
        if node.op in {"==", ">", "<", ">=", "<="}:
            return _comparison(node.op, node.args[0], node.args[1], truth)
        raise ProofDeferred(
            "NON_CONJUNCTIVE_LOGIC",
            f"operation {node.op} requires logical case enumeration",
        )

    return visit(expr)


def linear_constraints(
    expr: Expr,
    boolean_variables: set[str],
) -> list[LinearInequality]:
    constraints: list[LinearInequality] = []
    for comparison in conjunctive_comparisons(expr, boolean_variables):
        constraints.extend(comparison_inequalities(comparison, boolean_variables))
    return constraints


def serialize_linear_constraint(item: LinearInequality) -> dict[str, Any]:
    return {
        "coefficients": {
            name: fraction_text(value) for name, value in item.coefficients
        },
        "bound": fraction_text(item.bound),
        "strict": item.strict,
    }


def serialize_linear_constraints(
    constraints: list[LinearInequality],
) -> list[dict[str, Any]]:
    return [serialize_linear_constraint(item) for item in constraints]


@dataclass(frozen=True)
class QuadraticConstraint:
    """A diagonal quadratic expression constrained to be at most zero."""

    square: tuple[tuple[str, Fraction], ...]
    linear: tuple[tuple[str, Fraction], ...]
    constant: Fraction
    strict: bool = False

    @staticmethod
    def make(
        square: dict[str, Fraction],
        linear: dict[str, Fraction],
        constant: Fraction,
        strict: bool = False,
    ) -> "QuadraticConstraint":
        return QuadraticConstraint(
            tuple(sorted((name, value) for name, value in square.items() if value)),
            tuple(sorted((name, value) for name, value in linear.items() if value)),
            constant,
            strict,
        )


Polynomial = dict[tuple[str, ...], Fraction]


def _poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for monomial, value in right.items():
        result[monomial] = result.get(monomial, Fraction(0)) + value
        if result[monomial] == 0:
            del result[monomial]
    return result


def _poly_scale(value: Fraction, expression: Polynomial) -> Polynomial:
    return {
        monomial: value * coefficient
        for monomial, coefficient in expression.items()
        if value * coefficient
    }


def _polynomial(expr: Expr) -> Polynomial:
    from certification.equations import RawRef, Var
    from .proof_rules import fraction_value

    if isinstance(expr, Const):
        return {(): fraction_value(expr.value)}
    if isinstance(expr, Var):
        return {(expr.name,): Fraction(1)}
    if isinstance(expr, RawRef):
        return {(expr.path,): Fraction(1)}
    if not isinstance(expr, Op):
        raise ProofDeferred("NOT_CONVEX", f"unsupported expression {expr.pretty()}")
    if expr.op == "+":
        result: Polynomial = {}
        for arg in expr.args:
            result = _poly_add(result, _polynomial(arg))
        return result
    if expr.op == "-":
        if len(expr.args) == 1:
            return _poly_scale(Fraction(-1), _polynomial(expr.args[0]))
        result = _polynomial(expr.args[0])
        for arg in expr.args[1:]:
            result = _poly_add(result, _poly_scale(Fraction(-1), _polynomial(arg)))
        return result
    if expr.op == "*" and len(expr.args) == 2:
        left = _polynomial(expr.args[0])
        right = _polynomial(expr.args[1])
        result: Polynomial = {}
        for left_term, left_value in left.items():
            for right_term, right_value in right.items():
                monomial = tuple(sorted(left_term + right_term))
                if len(monomial) > 2:
                    raise ProofDeferred("NOT_CONVEX", "polynomial degree exceeds two")
                result = _poly_add(
                    result,
                    {monomial: left_value * right_value},
                )
        return result
    if expr.op == "/" and len(expr.args) == 2:
        numerator = _polynomial(expr.args[0])
        denominator = _polynomial(expr.args[1])
        if set(denominator) != {()} or denominator[()] == 0:
            raise ProofDeferred("NOT_CONVEX", "division is not by a nonzero constant")
        return _poly_scale(Fraction(1, 1) / denominator[()], numerator)
    raise ProofDeferred("NOT_CONVEX", f"unsupported operation {expr.op}")


def _quadratic_constraint(expression: Expr, *, strict: bool) -> QuadraticConstraint:
    square: dict[str, Fraction] = {}
    linear: dict[str, Fraction] = {}
    constant = Fraction(0)
    for monomial, coefficient in _polynomial(expression).items():
        if not monomial:
            constant += coefficient
        elif len(monomial) == 1:
            linear[monomial[0]] = linear.get(monomial[0], Fraction(0)) + coefficient
        elif len(monomial) == 2 and monomial[0] == monomial[1]:
            square[monomial[0]] = square.get(monomial[0], Fraction(0)) + coefficient
        else:
            raise ProofDeferred("NOT_CONVEX", "cross terms are not supported")
    if any(value < 0 for value in square.values()):
        raise ProofDeferred("NOT_CONVEX", "a quadratic constraint is not convex")
    return QuadraticConstraint.make(square, linear, constant, strict)


def quadratic_constraints(
    expr: Expr,
    boolean_variables: set[str],
) -> list[QuadraticConstraint]:
    constraints: list[QuadraticConstraint] = []
    for comparison in conjunctive_comparisons(expr, boolean_variables):
        strict = comparison.op in {"<", ">"}
        if comparison.op in {"<=", "<"}:
            expression = Op("-", (comparison.left, comparison.right))
        elif comparison.op in {">=", ">"}:
            expression = Op("-", (comparison.right, comparison.left))
        else:
            raise ProofDeferred("NOT_CONVEX", f"unsupported comparison {comparison.op}")
        constraints.append(_quadratic_constraint(expression, strict=strict))
    return constraints


def serialize_quadratic_constraint(item: QuadraticConstraint) -> dict[str, Any]:
    return {
        "square": {name: fraction_text(value) for name, value in item.square},
        "linear": {name: fraction_text(value) for name, value in item.linear},
        "constant": fraction_text(item.constant),
        "relation": "< 0" if item.strict else "<= 0",
    }


def serialize_quadratic_constraints(
    constraints: list[QuadraticConstraint],
) -> list[dict[str, Any]]:
    return [serialize_quadratic_constraint(item) for item in constraints]
