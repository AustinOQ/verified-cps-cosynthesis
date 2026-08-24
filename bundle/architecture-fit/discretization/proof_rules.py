"""Small exact proof rules used by the discretization certificate checker.

The numerical backends are not trusted.  This module works with exact fractions
and proves small Boolean combinations of linear real constraints by exhaustive
Boolean case splitting followed by exact variable elimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any, Iterable

from certification.equations import Const, EquationModel, Expr, Ite, Op, RawRef, Var


class ProofDeferred(ValueError):
    """The exact proof rule does not support the supplied expression."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


def fraction_value(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ProofDeferred("UNSUPPORTED_EXPRESSION", "Boolean used as a number")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"unsupported number {value!r}")


def expr_to_dict(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, Const):
        return {"type": "const", "value": expr.value}
    if isinstance(expr, Var):
        return {"type": "var", "name": expr.name}
    if isinstance(expr, RawRef):
        return {"type": "raw_ref", "path": expr.path}
    if isinstance(expr, Op):
        return {
            "type": "op",
            "op": expr.op,
            "args": [expr_to_dict(arg) for arg in expr.args],
        }
    if isinstance(expr, Ite):
        return {
            "type": "ite",
            "cond": expr_to_dict(expr.cond),
            "then": expr_to_dict(expr.then_expr),
            "else": expr_to_dict(expr.else_expr),
        }
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", type(expr).__name__)


def expression_symbols(expr: Expr) -> set[str]:
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, RawRef):
        return {expr.path}
    if isinstance(expr, Op):
        out: set[str] = set()
        for arg in expr.args:
            out |= expression_symbols(arg)
        return out
    if isinstance(expr, Ite):
        return (
            expression_symbols(expr.cond)
            | expression_symbols(expr.then_expr)
            | expression_symbols(expr.else_expr)
        )
    return set()


def expand_definitions(
    model: EquationModel,
    expr: Expr,
    *,
    seen: set[str] | None = None,
) -> Expr:
    seen = set(seen or set())
    if isinstance(expr, Var) and expr.name in model.definitions:
        if expr.name in seen:
            raise ProofDeferred(
                "UNSUPPORTED_EXPRESSION",
                f"cyclic definition {expr.name}",
            )
        seen.add(expr.name)
        return expand_definitions(model, model.definitions[expr.name].expr, seen=seen)
    if isinstance(expr, Op):
        return Op(
            expr.op,
            tuple(expand_definitions(model, arg, seen=set(seen)) for arg in expr.args),
        )
    if isinstance(expr, Ite):
        return Ite(
            expand_definitions(model, expr.cond, seen=set(seen)),
            expand_definitions(model, expr.then_expr, seen=set(seen)),
            expand_definitions(model, expr.else_expr, seen=set(seen)),
        )
    return expr


def post_state_expression(model: EquationModel, expr: Expr) -> Expr:
    """Evaluate a property after one sampled transition.

    References inside a transition equation remain references to the current
    sampled state.  Same-cycle definitions are expanded in both places.
    """

    expr = expand_definitions(model, expr)

    def visit(node: Expr) -> Expr:
        if isinstance(node, Var) and node.name in model.state:
            equation = model.transitions.get(node.name)
            if equation is None:
                return node
            return expand_definitions(model, equation.expr)
        if isinstance(node, Op):
            return Op(node.op, tuple(visit(arg) for arg in node.args))
        if isinstance(node, Ite):
            return Ite(visit(node.cond), visit(node.then_expr), visit(node.else_expr))
        return node

    return visit(expr)


def within_interval_expression(
    model: EquationModel,
    expr: Expr,
    continuously_changing: set[str],
) -> Expr:
    """Evaluate held values after the update and leave changing values arbitrary."""

    expr = expand_definitions(model, expr)

    def visit(node: Expr) -> Expr:
        if isinstance(node, Var) and node.name in model.state:
            if node.name in continuously_changing:
                return node
            equation = model.transitions.get(node.name)
            if equation is None:
                return node
            return expand_definitions(model, equation.expr)
        if isinstance(node, Op):
            return Op(node.op, tuple(visit(arg) for arg in node.args))
        if isinstance(node, Ite):
            return Ite(visit(node.cond), visit(node.then_expr), visit(node.else_expr))
        return node

    return visit(expr)


def substitute(expr: Expr, values: dict[str, Expr]) -> Expr:
    if isinstance(expr, Var) and expr.name in values:
        return values[expr.name]
    if isinstance(expr, RawRef) and expr.path in values:
        return values[expr.path]
    if isinstance(expr, Op):
        return Op(expr.op, tuple(substitute(arg, values) for arg in expr.args))
    if isinstance(expr, Ite):
        return Ite(
            substitute(expr.cond, values),
            substitute(expr.then_expr, values),
            substitute(expr.else_expr, values),
        )
    return expr


def simplify_known_conditions(expr: Expr, true_conditions: Iterable[Expr]) -> Expr:
    truths = tuple(true_conditions)

    def known(node: Expr) -> bool:
        return any(node == item for item in truths)

    if isinstance(expr, Ite):
        cond = simplify_known_conditions(expr.cond, truths)
        if known(cond):
            return simplify_known_conditions(expr.then_expr, truths)
        if isinstance(cond, Op) and cond.op == "not" and known(cond.args[0]):
            return simplify_known_conditions(expr.else_expr, truths)
        return Ite(
            cond,
            simplify_known_conditions(expr.then_expr, truths),
            simplify_known_conditions(expr.else_expr, truths),
        )
    if isinstance(expr, Op):
        return Op(
            expr.op,
            tuple(simplify_known_conditions(arg, truths) for arg in expr.args),
        )
    return expr


@dataclass(frozen=True)
class LinearForm:
    coefficients: tuple[tuple[str, Fraction], ...]
    constant: Fraction = Fraction(0)

    @staticmethod
    def make(coefficients: dict[str, Fraction], constant: Fraction = Fraction(0)) -> "LinearForm":
        return LinearForm(
            tuple(sorted((name, value) for name, value in coefficients.items() if value)),
            constant,
        )

    def coeff_dict(self) -> dict[str, Fraction]:
        return dict(self.coefficients)

    def scale(self, value: Fraction) -> "LinearForm":
        return LinearForm.make(
            {name: coeff * value for name, coeff in self.coefficients},
            self.constant * value,
        )

    def add(self, other: "LinearForm") -> "LinearForm":
        coefficients = self.coeff_dict()
        for name, value in other.coefficients:
            coefficients[name] = coefficients.get(name, Fraction(0)) + value
        return LinearForm.make(coefficients, self.constant + other.constant)


def linear_form(expr: Expr, boolean_variables: set[str]) -> LinearForm:
    if isinstance(expr, Const):
        return LinearForm.make({}, fraction_value(expr.value))
    if isinstance(expr, Var):
        if expr.name in boolean_variables:
            raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"Boolean {expr.name} used numerically")
        return LinearForm.make({expr.name: Fraction(1)})
    if isinstance(expr, RawRef):
        return LinearForm.make({expr.path: Fraction(1)})
    if isinstance(expr, Ite):
        raise ProofDeferred("UNSUPPORTED_EXPRESSION", "unresolved numeric conditional")
    if not isinstance(expr, Op):
        raise ProofDeferred("UNSUPPORTED_EXPRESSION", type(expr).__name__)

    if expr.op == "+":
        out = LinearForm.make({})
        for arg in expr.args:
            out = out.add(linear_form(arg, boolean_variables))
        return out
    if expr.op == "-":
        if len(expr.args) == 1:
            return linear_form(expr.args[0], boolean_variables).scale(Fraction(-1))
        out = linear_form(expr.args[0], boolean_variables)
        for arg in expr.args[1:]:
            out = out.add(linear_form(arg, boolean_variables).scale(Fraction(-1)))
        return out
    if expr.op == "*" and len(expr.args) == 2:
        left = linear_form(expr.args[0], boolean_variables)
        right = linear_form(expr.args[1], boolean_variables)
        if not left.coefficients:
            return right.scale(left.constant)
        if not right.coefficients:
            return left.scale(right.constant)
        raise ProofDeferred("NOT_LINEAR", f"variable multiplication in {expr.pretty()}")
    if expr.op == "/" and len(expr.args) == 2:
        numerator = linear_form(expr.args[0], boolean_variables)
        denominator = linear_form(expr.args[1], boolean_variables)
        if denominator.coefficients:
            raise ProofDeferred("NOT_LINEAR", f"symbolic division in {expr.pretty()}")
        if denominator.constant == 0:
            raise ProofDeferred("UNSUPPORTED_EXPRESSION", "division by zero")
        return numerator.scale(Fraction(1, 1) / denominator.constant)
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"nonlinear operation {expr.op}")


@dataclass(frozen=True)
class LinearInequality:
    """A linear inequality represented as coefficients times variables <= bound."""

    coefficients: tuple[tuple[str, Fraction], ...]
    bound: Fraction
    strict: bool = False

    @staticmethod
    def make(
        coefficients: dict[str, Fraction],
        bound: Fraction,
        strict: bool = False,
    ) -> "LinearInequality":
        return LinearInequality(
            tuple(sorted((name, value) for name, value in coefficients.items() if value)),
            bound,
            strict,
        )

    def coeff_dict(self) -> dict[str, Fraction]:
        return dict(self.coefficients)


@dataclass(frozen=True)
class Comparison:
    op: str
    left: Expr
    right: Expr


def _is_boolean(expr: Expr, boolean_variables: set[str]) -> bool:
    if isinstance(expr, Const):
        return isinstance(expr.value, bool)
    if isinstance(expr, Var):
        return expr.name in boolean_variables
    if isinstance(expr, Ite):
        return _is_boolean(expr.then_expr, boolean_variables) and _is_boolean(
            expr.else_expr, boolean_variables
        )
    if isinstance(expr, Op):
        return expr.op in {"not", "and", "or", "implies", "==", ">", "<", ">=", "<="}
    return False


def _cross(left: list[list[Comparison]], right: list[list[Comparison]]) -> list[list[Comparison]]:
    return [a + b for a in left for b in right]


def _comparison_cases(op: str, left: Expr, right: Expr, truth: bool) -> list[list[Comparison]]:
    if truth:
        if op == "==":
            return [[Comparison("<=", left, right), Comparison(">=", left, right)]]
        return [[Comparison(op, left, right)]]
    inverse = {">": "<=", "<": ">=", ">=": "<", "<=": ">"}
    if op in inverse:
        return [[Comparison(inverse[op], left, right)]]
    if op == "==":
        return [[Comparison("<", left, right)], [Comparison(">", left, right)]]
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"unsupported comparison {op}")


def boolean_dnf(
    expr: Expr,
    boolean_variables: set[str],
    *,
    truth: bool = True,
) -> list[list[Comparison]]:
    if isinstance(expr, Const) and isinstance(expr.value, bool):
        return [[]] if expr.value is truth else []
    if isinstance(expr, Var) and expr.name in boolean_variables:
        raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"unassigned Boolean {expr.name}")
    if isinstance(expr, Ite):
        expanded = Op(
            "or",
            (
                Op("and", (expr.cond, expr.then_expr)),
                Op("and", (Op("not", (expr.cond,)), expr.else_expr)),
            ),
        )
        return boolean_dnf(expanded, boolean_variables, truth=truth)
    if not isinstance(expr, Op):
        raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"non-Boolean expression {expr.pretty()}")

    if expr.op == "not":
        return boolean_dnf(expr.args[0], boolean_variables, truth=not truth)
    if expr.op == "implies":
        expanded = Op("or", (Op("not", (expr.args[0],)), expr.args[1]))
        return boolean_dnf(expanded, boolean_variables, truth=truth)
    if expr.op in {"and", "or"}:
        conjunction = (expr.op == "and") == truth
        if conjunction:
            out: list[list[Comparison]] = [[]]
            for arg in expr.args:
                out = _cross(out, boolean_dnf(arg, boolean_variables, truth=truth))
            return out
        out: list[list[Comparison]] = []
        for arg in expr.args:
            out.extend(boolean_dnf(arg, boolean_variables, truth=truth))
        return out
    if expr.op == "==" and all(_is_boolean(arg, boolean_variables) for arg in expr.args):
        left, right = expr.args
        if truth:
            return _cross(
                boolean_dnf(left, boolean_variables, truth=True),
                boolean_dnf(right, boolean_variables, truth=True),
            ) + _cross(
                boolean_dnf(left, boolean_variables, truth=False),
                boolean_dnf(right, boolean_variables, truth=False),
            )
        return _cross(
            boolean_dnf(left, boolean_variables, truth=True),
            boolean_dnf(right, boolean_variables, truth=False),
        ) + _cross(
            boolean_dnf(left, boolean_variables, truth=False),
            boolean_dnf(right, boolean_variables, truth=True),
        )
    if expr.op in {"==", ">", "<", ">=", "<="}:
        return _comparison_cases(expr.op, expr.args[0], expr.args[1], truth)
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"unsupported Boolean operation {expr.op}")


def comparison_inequalities(
    comparison: Comparison,
    boolean_variables: set[str],
) -> list[LinearInequality]:
    left = linear_form(comparison.left, boolean_variables)
    right = linear_form(comparison.right, boolean_variables)
    difference = left.add(right.scale(Fraction(-1)))
    coefficients = difference.coeff_dict()
    bound = -difference.constant

    if comparison.op == "<=":
        return [LinearInequality.make(coefficients, bound, False)]
    if comparison.op == "<":
        return [LinearInequality.make(coefficients, bound, True)]
    if comparison.op == ">=":
        return [LinearInequality.make({k: -v for k, v in coefficients.items()}, -bound, False)]
    if comparison.op == ">":
        return [LinearInequality.make({k: -v for k, v in coefficients.items()}, -bound, True)]
    raise ProofDeferred("UNSUPPORTED_EXPRESSION", comparison.op)


def _constant_contradiction(inequality: LinearInequality) -> bool:
    if inequality.coefficients:
        return False
    return inequality.bound < 0 or (inequality.bound == 0 and inequality.strict)


def exact_linear_infeasible(
    inequalities: list[LinearInequality],
    *,
    elimination_limit: int = 50000,
) -> tuple[bool, dict[str, Any]]:
    work = list(dict.fromkeys(inequalities))
    peak = len(work)
    if any(_constant_contradiction(item) for item in work):
        return True, {"variables_eliminated": 0, "peak_inequalities": peak}

    variables = sorted({name for item in work for name, _value in item.coefficients})
    eliminated = 0
    for variable in variables:
        positive: list[tuple[LinearInequality, Fraction]] = []
        negative: list[tuple[LinearInequality, Fraction]] = []
        zero: list[LinearInequality] = []
        for item in work:
            coeff = item.coeff_dict().get(variable, Fraction(0))
            if coeff > 0:
                positive.append((item, coeff))
            elif coeff < 0:
                negative.append((item, coeff))
            else:
                zero.append(item)

        combined: list[LinearInequality] = list(zero)
        if positive and negative:
            for upper, upper_coeff in positive:
                upper_terms = upper.coeff_dict()
                upper_terms.pop(variable, None)
                for lower, lower_coeff in negative:
                    lower_terms = lower.coeff_dict()
                    lower_terms.pop(variable, None)
                    coefficients: dict[str, Fraction] = {}
                    for name, value in upper_terms.items():
                        coefficients[name] = coefficients.get(name, Fraction(0)) + (-lower_coeff) * value
                    for name, value in lower_terms.items():
                        coefficients[name] = coefficients.get(name, Fraction(0)) + upper_coeff * value
                    combined.append(LinearInequality.make(
                        coefficients,
                        (-lower_coeff) * upper.bound + upper_coeff * lower.bound,
                        upper.strict or lower.strict,
                    ))
                    if len(combined) > elimination_limit:
                        raise ProofDeferred(
                            "NUMERIC_BOUND_INCONCLUSIVE",
                            f"exact elimination exceeded {elimination_limit} inequalities",
                        )
        work = list(dict.fromkeys(combined))
        peak = max(peak, len(work))
        eliminated += 1
        if any(_constant_contradiction(item) for item in work):
            return True, {
                "variables_eliminated": eliminated,
                "peak_inequalities": peak,
            }

    return False, {
        "variables_eliminated": eliminated,
        "peak_inequalities": peak,
    }


def prove_implication_exact(
    premises: list[Expr],
    conclusion: Expr,
    boolean_variables: set[str],
) -> dict[str, Any]:
    premise = Const(True) if not premises else Op("and", tuple(premises))
    counterexample = Op("and", (premise, Op("not", (conclusion,))))
    used_booleans = sorted(expression_symbols(counterexample) & boolean_variables)

    assignments_checked = 0
    linear_cases_checked = 0
    peak_inequalities = 0
    for values in product((False, True), repeat=len(used_booleans)):
        assignments_checked += 1
        mapping = {
            name: Const(value)
            for name, value in zip(used_booleans, values)
        }
        assigned = substitute(counterexample, mapping)
        cases = boolean_dnf(assigned, boolean_variables)
        for comparisons in cases:
            linear_cases_checked += 1
            inequalities: list[LinearInequality] = []
            for comparison in comparisons:
                inequalities.extend(comparison_inequalities(comparison, boolean_variables))
            infeasible, detail = exact_linear_infeasible(inequalities)
            peak_inequalities = max(peak_inequalities, int(detail["peak_inequalities"]))
            if not infeasible:
                return {
                    "proved": False,
                    "reason_code": "COUNTEREXAMPLE_CASE_REMAINS",
                    "boolean_assignments_checked": assignments_checked,
                    "linear_cases_checked": linear_cases_checked,
                    "unresolved_boolean_assignment": {
                        name: value for name, value in zip(used_booleans, values)
                    },
                    "unresolved_inequality_count": len(inequalities),
                    "peak_inequalities": peak_inequalities,
                }
    return {
        "proved": True,
        "method": "exact_boolean_cases_and_linear_elimination_v1",
        "boolean_variables": used_booleans,
        "boolean_assignments_checked": assignments_checked,
        "linear_cases_checked": linear_cases_checked,
        "peak_inequalities": peak_inequalities,
    }


def expression_is_linear(expr: Expr, boolean_variables: set[str]) -> tuple[bool, str]:
    try:
        if _is_boolean(expr, boolean_variables):
            used_booleans = sorted(expression_symbols(expr) & boolean_variables)
            for values in product((False, True), repeat=len(used_booleans)):
                assigned = substitute(expr, {
                    name: Const(value) for name, value in zip(used_booleans, values)
                })
                for comparisons in boolean_dnf(assigned, boolean_variables):
                    for comparison in comparisons:
                        comparison_inequalities(comparison, boolean_variables)
        else:
            linear_form(expr, boolean_variables)
        return True, ""
    except ProofDeferred as exc:
        return False, f"{exc.reason_code}: {exc.detail}"
