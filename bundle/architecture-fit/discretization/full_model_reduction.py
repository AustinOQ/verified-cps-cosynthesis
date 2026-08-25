"""Reduce a complete SysML controller interval to checked arithmetic cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any, Iterable

from certification.equations import Const, Equation, EquationModel, Expr, Ite, Op, RawRef, Var

from .proof_rules import (
    Comparison,
    ProofDeferred,
    boolean_dnf,
    expr_to_dict,
    expression_symbols,
    expand_definitions,
    substitute,
)


INTERVAL_TIME = "proof_interval_time"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def expression_hash(expr: Expr) -> str:
    return hashlib.sha256(_canonical_bytes(expr_to_dict(expr))).hexdigest()


@dataclass(frozen=True)
class ReducedCase:
    """One completely recorded arithmetic case produced by the reducer."""

    case_id: str
    expression: Expr
    parent_hash: str
    boolean_assignment: tuple[tuple[str, bool], ...]
    time_reduction: str
    obligation: str
    reachability_expression: Expr | None = None


@dataclass(frozen=True)
class ReachabilityContext:
    """Exact inputs used to prove that a reduced case is unreachable."""

    domain: Expr
    initial_constraints: tuple[Expr, ...]
    initial_variables: tuple[str, ...]
    post_values: tuple[tuple[str, Expr], ...]
    action_variables: tuple[str, ...]
    boolean_variables: tuple[str, ...]
    integer_variables: tuple[str, ...]

    def post_dict(self) -> dict[str, Expr]:
        return dict(self.post_values)


def _and(expressions: Iterable[Expr]) -> Expr:
    items = tuple(expressions)
    if not items:
        return Const(True)
    if len(items) == 1:
        return items[0]
    return Op("and", items)


def _comparison_expression(comparison: Comparison) -> Expr:
    return Op(comparison.op, (comparison.left, comparison.right))


def _time_degree(expr: Expr) -> int | None:
    """Return the exact polynomial degree in the interval time alone."""

    if isinstance(expr, (Const, RawRef)):
        return 0
    if isinstance(expr, Var):
        return 1 if expr.name == INTERVAL_TIME else 0
    if isinstance(expr, Ite):
        condition = _time_degree(expr.cond)
        then_degree = _time_degree(expr.then_expr)
        else_degree = _time_degree(expr.else_expr)
        if condition != 0 or then_degree is None or else_degree is None:
            return None
        return max(then_degree, else_degree)
    if not isinstance(expr, Op):
        return None
    degrees = [_time_degree(arg) for arg in expr.args]
    if any(value is None for value in degrees):
        return None
    known = [int(value) for value in degrees if value is not None]
    if expr.op in {"+", "-"}:
        return max(known, default=0)
    if expr.op == "*" and len(known) == 2:
        return known[0] + known[1]
    if expr.op == "/" and len(known) == 2 and known[1] == 0:
        return known[0]
    if expr.op in {">", "<", ">=", "<=", "==", "and", "or", "not", "implies"}:
        return max(known, default=0)
    return None


def _is_interval_bound(comparison: Comparison) -> bool:
    expression = _comparison_expression(comparison)
    symbols = expression_symbols(expression)
    if symbols != {INTERVAL_TIME}:
        return False
    constants = [
        item.value
        for item in (comparison.left, comparison.right)
        if isinstance(item, Const)
    ]
    return bool(constants)


def _first_ite(expr: Expr) -> Ite | None:
    if isinstance(expr, Ite):
        return expr
    if isinstance(expr, Op):
        for arg in expr.args:
            found = _first_ite(arg)
            if found is not None:
                return found
    return None


def _assume_condition(expr: Expr, condition: Expr, truth: bool) -> Expr:
    if expr == condition:
        return Const(truth)
    if (
        isinstance(expr, Op)
        and expr.op == "not"
        and len(expr.args) == 1
        and expr.args[0] == condition
    ):
        return Const(not truth)
    if isinstance(expr, Op):
        return simplify(Op(
            expr.op,
            tuple(
                _assume_condition(argument, condition, truth)
                for argument in expr.args
            ),
        ))
    if isinstance(expr, Ite):
        return simplify(Ite(
            _assume_condition(expr.cond, condition, truth),
            _assume_condition(expr.then_expr, condition, truth),
            _assume_condition(expr.else_expr, condition, truth),
        ))
    return expr


def _split_conditionals(expr: Expr, *, limit: int = 512) -> list[tuple[str, Expr]]:
    pending: list[tuple[str, Expr]] = [("root", simplify(expr))]
    complete: list[tuple[str, Expr]] = []
    while pending:
        branch_id, current = pending.pop(0)
        conditional = _first_ite(current)
        if conditional is None:
            complete.append((branch_id, current))
            continue
        if len(pending) + len(complete) + 2 > limit:
            raise ProofDeferred(
                "INCOMPLETE_CASE_COVERAGE",
                f"conditional case split exceeds {limit} branches",
            )
        then_expression = _assume_condition(current, conditional.cond, True)
        else_expression = _assume_condition(current, conditional.cond, False)
        pending.append((
            branch_id + ".then",
            simplify(_and([conditional.cond, then_expression])),
        ))
        pending.append((
            branch_id + ".else",
            simplify(_and([Op("not", (conditional.cond,)), else_expression])),
        ))
    return complete


def simplify(expr: Expr) -> Expr:
    """Apply only exact local Boolean and conditional simplifications."""

    if isinstance(expr, (Const, Var, RawRef)):
        return expr
    if isinstance(expr, Ite):
        condition = simplify(expr.cond)
        then_expr = simplify(expr.then_expr)
        else_expr = simplify(expr.else_expr)
        if isinstance(condition, Const) and isinstance(condition.value, bool):
            return then_expr if condition.value else else_expr
        if then_expr == else_expr:
            return then_expr
        return Ite(condition, then_expr, else_expr)
    if not isinstance(expr, Op):
        return expr
    args = tuple(simplify(arg) for arg in expr.args)
    if args and all(isinstance(arg, Const) for arg in args):
        values = [arg.value for arg in args]
        if expr.op == "==" and len(values) == 2:
            return Const(values[0] == values[1])
        if expr.op in {">", "<", ">=", "<="} and len(values) == 2:
            try:
                left = Fraction(str(values[0]))
                right = Fraction(str(values[1]))
                result = {
                    ">": left > right,
                    "<": left < right,
                    ">=": left >= right,
                    "<=": left <= right,
                }[expr.op]
                return Const(result)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        if expr.op in {"+", "-", "*", "/"}:
            try:
                numbers = [Fraction(str(value)) for value in values]
                if expr.op == "+":
                    value = sum(numbers, Fraction(0))
                elif expr.op == "-":
                    value = -numbers[0] if len(numbers) == 1 else numbers[0] - sum(numbers[1:], Fraction(0))
                elif expr.op == "*" and len(numbers) == 2:
                    value = numbers[0] * numbers[1]
                elif expr.op == "/" and len(numbers) == 2 and numbers[1] != 0:
                    value = numbers[0] / numbers[1]
                else:
                    value = None
                if value is not None:
                    return Const(
                        value.numerator
                        if value.denominator == 1
                        else f"{value.numerator}/{value.denominator}"
                    )
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    if expr.op == "not" and len(args) == 1:
        if isinstance(args[0], Const) and isinstance(args[0].value, bool):
            return Const(not args[0].value)
        if isinstance(args[0], Op) and args[0].op == "not":
            return args[0].args[0]
    if expr.op in {"and", "or"}:
        flattened: list[Expr] = []
        for arg in args:
            if isinstance(arg, Op) and arg.op == expr.op:
                flattened.extend(arg.args)
            else:
                flattened.append(arg)
        identity = expr.op == "and"
        absorbing = not identity
        if any(
            isinstance(arg, Const)
            and isinstance(arg.value, bool)
            and arg.value == absorbing
            for arg in flattened
        ):
            return Const(absorbing)
        kept = [
            arg
            for arg in flattened
            if not (
                isinstance(arg, Const)
                and isinstance(arg.value, bool)
                and arg.value == identity
            )
        ]
        if not kept:
            return Const(identity)
        if len(kept) == 1:
            return kept[0]
        return Op(expr.op, tuple(kept))
    if expr.op == "implies" and len(args) == 2:
        left, right = args
        if isinstance(left, Const) and isinstance(left.value, bool):
            return right if left.value else Const(True)
        if isinstance(right, Const) and isinstance(right.value, bool):
            return Const(True) if right.value else simplify(Op("not", (left,)))
    if expr.op == "==" and len(args) == 2:
        for boolean_value, other in ((args[0], args[1]), (args[1], args[0])):
            if not (
                isinstance(boolean_value, Const)
                and isinstance(boolean_value.value, bool)
            ):
                continue
            if isinstance(other, Op) and other.op in {
                "not", "and", "or", "implies", "==", ">", "<", ">=", "<="
            }:
                return other if boolean_value.value else simplify(Op("not", (other,)))
    if expr.op == "==" and len(args) == 2 and args[0] == args[1]:
        return Const(True)
    return Op(expr.op, args)


def _equation(model: EquationModel, name: str) -> Equation | None:
    return model.definitions.get(name) or model.transitions.get(name)


def _path_to_continuous(
    model: EquationModel,
    start: str,
    continuous: set[str],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Find one deterministic equation path from a sampled value to physics."""

    queue: list[tuple[str, list[dict[str, Any]]]] = [(start, [])]
    visited: set[str] = set()
    found: list[tuple[str, list[dict[str, Any]]]] = []
    while queue:
        name, path = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        if name in continuous and name != start:
            if _is_physical_target(name):
                found.append((name, path))
            continue
        equation = _equation(model, name)
        if equation is None:
            continue
        row = {
            "target": equation.target,
            "kind": equation.kind,
            "source": equation.source,
            "expression": expr_to_dict(equation.expr),
        }
        for reference in sorted(expression_symbols(equation.expr)):
            if reference == name:
                continue
            queue.append((reference, path + [row]))
    targets = sorted({target for target, _path in found})
    if len(targets) != 1:
        return None
    target = targets[0]
    candidates = [path for candidate, path in found if candidate == target]
    candidates.sort(key=lambda value: (len(value), _canonical_bytes(value)))
    return target, candidates[0]


def _is_physical_target(name: str) -> bool:
    return "time" not in name.lower()


def physical_aliases(
    model: EquationModel,
    expressions: Iterable[Expr],
    continuous: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    aliases: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    symbols: set[str] = set()
    for expression in expressions:
        symbols |= expression_symbols(expression)
    for symbol in sorted(symbols):
        if symbol in continuous:
            continue
        path = _path_to_continuous(model, symbol, continuous)
        if path is None or not _is_physical_target(path[0]):
            continue
        target, equations = path
        aliases[symbol] = target
        records.append({
            "sampled_value": symbol,
            "physical_value": target,
            "equation_path": equations,
            "rule": "ordered_sensor_path_to_current_physical_value_v1",
        })
    return aliases, records


def _reaches_target(
    model: EquationModel,
    expr: Expr,
    target: str,
    *,
    seen: set[str] | None = None,
) -> bool:
    seen = set(seen or set())
    if isinstance(expr, Var):
        if expr.name == target:
            return True
        if expr.name in seen:
            return False
        equation = _equation(model, expr.name)
        return bool(
            equation
            and _reaches_target(
                model,
                equation.expr,
                target,
                seen=seen | {expr.name},
            )
        )
    if isinstance(expr, Op):
        return any(_reaches_target(model, arg, target, seen=set(seen)) for arg in expr.args)
    if isinstance(expr, Ite):
        return any(
            _reaches_target(model, arg, target, seen=set(seen))
            for arg in (expr.cond, expr.then_expr, expr.else_expr)
        )
    return False


def _required_mapping_guards(
    model: EquationModel,
    sampled: str,
    physical: str,
) -> list[Expr]:
    required: list[Expr] = []
    visited: set[str] = set()
    current = sampled
    while current != physical and current not in visited:
        visited.add(current)
        equation = _equation(model, current)
        if equation is None:
            break
        expression = equation.expr
        if isinstance(expression, Ite):
            then_reaches = _reaches_target(
                model, expression.then_expr, physical, seen=set(visited)
            )
            else_reaches = _reaches_target(
                model, expression.else_expr, physical, seen=set(visited)
            )
            if then_reaches and not else_reaches:
                required.append(expand_definitions(model, expression.cond))
                expression = expression.then_expr
            elif else_reaches and not then_reaches:
                required.append(
                    Op("not", (expand_definitions(model, expression.cond),))
                )
                expression = expression.else_expr
            else:
                break
        references = [
            name
            for name in sorted(expression_symbols(expression))
            if _reaches_target(model, Var(name), physical, seen=set(visited))
        ]
        if len(references) != 1:
            break
        current = references[0]
    return required


def _expand_post_update(
    model: EquationModel,
    expr: Expr,
    continuous: set[str],
    *,
    seen: set[str] | None = None,
) -> Expr:
    """Use the controller and actuator value effective for the next interval."""

    seen = set(seen or set())
    if isinstance(expr, Var):
        if expr.name in model.definitions:
            if expr.name in seen:
                raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"cyclic definition {expr.name}")
            return _expand_post_update(
                model,
                model.definitions[expr.name].expr,
                continuous,
                seen=seen | {expr.name},
            )
        if expr.name in model.state and expr.name not in continuous:
            equation = model.transitions.get(expr.name)
            if equation is not None and expr.name not in seen:
                return _expand_post_update(
                    model,
                    equation.expr,
                    continuous,
                    seen=seen | {expr.name},
                )
        return expr
    if isinstance(expr, Op):
        return Op(
            expr.op,
            tuple(
                _expand_post_update(model, arg, continuous, seen=set(seen))
                for arg in expr.args
            ),
        )
    if isinstance(expr, Ite):
        return Ite(
            _expand_post_update(model, expr.cond, continuous, seen=set(seen)),
            _expand_post_update(model, expr.then_expr, continuous, seen=set(seen)),
            _expand_post_update(model, expr.else_expr, continuous, seen=set(seen)),
        )
    return expr


def _trajectory(
    model: EquationModel,
    target: str,
    continuous: set[str],
    constant_values: dict[str, Expr],
) -> tuple[Expr, list[str]]:
    equation = model.transitions.get(target)
    if equation is None:
        raise ProofDeferred("MISSING_WITHIN_STEP_MEANING", f"no transition for {target}")
    symbols = expression_symbols(equation.expr)
    dt_symbols = sorted(name for name in symbols if name == "dt" or name.endswith("_dt"))
    if not dt_symbols:
        raise ProofDeferred(
            "MISSING_WITHIN_STEP_MEANING",
            f"continuous assignment for {target} does not use dt",
        )
    expression = substitute(
        equation.expr,
        {name: Var(INTERVAL_TIME) for name in dt_symbols},
    )
    expression = _expand_post_update(model, expression, continuous)
    expression = substitute(
        expression,
        {name: value for name, value in constant_values.items() if name not in dt_symbols},
    )
    return simplify(expression), dt_symbols


def _case_split(
    expression: Expr,
    boolean_variables: set[str],
    property_id: str,
    obligation: str,
) -> tuple[list[ReducedCase], dict[str, Any]]:
    used_booleans = sorted(expression_symbols(expression) & boolean_variables)
    parent_hash = expression_hash(expression)
    cases: list[ReducedCase] = []
    case_rows: list[dict[str, Any]] = []
    index = 0
    conditional_branch_count = 0
    for values in product((False, True), repeat=len(used_booleans)):
        assignment = tuple(zip(used_booleans, values))
        assigned = simplify(substitute(expression, {
            name: Const(value) for name, value in assignment
        }))
        conditional_branches = _split_conditionals(assigned)
        conditional_branch_count += len(conditional_branches)
        for conditional_branch, branch_expression in conditional_branches:
            alternatives = boolean_dnf(branch_expression, boolean_variables)
            for comparisons in alternatives:
                interval_bounds = [
                    item for item in comparisons if _is_interval_bound(item)
                ]
                body = [
                    item for item in comparisons if not _is_interval_bound(item)
                ]
                time_gate_relaxations = [
                    item
                    for item in body
                    if INTERVAL_TIME
                    in expression_symbols(_comparison_expression(item))
                    and all(
                        name == INTERVAL_TIME or "time" in name.lower()
                        for name in expression_symbols(_comparison_expression(item))
                    )
                ]
                body = [item for item in body if item not in time_gate_relaxations]
                changing = [
                    item
                    for item in body
                    if INTERVAL_TIME
                    in expression_symbols(_comparison_expression(item))
                ]
                endpoint_values: list[tuple[str, Expr | None]]
                if not changing:
                    endpoint_values = [("time_independent", None)]
                elif len(changing) == 1 and (
                    _time_degree(changing[0].left) is not None
                    and _time_degree(changing[0].left) <= 1
                    and _time_degree(changing[0].right) is not None
                    and _time_degree(changing[0].right) <= 1
                ):
                    endpoint_values = [
                        ("affine_time_endpoint_zero", Const(0)),
                        ("affine_time_endpoint_dt", Const("__DT__")),
                    ]
                else:
                    endpoint_values = [("unreduced_interval_time", None)]

                for time_reduction, endpoint in endpoint_values:
                    if time_gate_relaxations:
                        time_reduction = (
                            "time_gate_outer_relaxation+" + time_reduction
                        )
                    selected = (
                        body
                        if time_reduction != "unreduced_interval_time"
                        else comparisons
                    )
                    if time_reduction.endswith("unreduced_interval_time"):
                        selected = [
                            item
                            for item in comparisons
                            if item not in time_gate_relaxations
                        ]
                    expressions = [
                        _comparison_expression(item) for item in selected
                    ]
                    reachability_expressions = [
                        _comparison_expression(item) for item in comparisons
                    ]
                    if endpoint is not None:
                        value = endpoint
                        if isinstance(endpoint, Const) and endpoint.value == "__DT__":
                            upper = next(
                                (
                                    item.right
                                    for item in interval_bounds
                                    if isinstance(item.left, Var)
                                    and item.left.name == INTERVAL_TIME
                                    and item.op == "<="
                                ),
                                None,
                            )
                            if upper is None:
                                raise ProofDeferred(
                                    "INCOMPLETE_CASE_COVERAGE",
                                    "the interval upper bound is missing",
                                )
                            value = upper
                        expressions = [
                            substitute(item, {INTERVAL_TIME: value})
                            for item in expressions
                        ]
                        reachability_expressions = [
                            substitute(item, {INTERVAL_TIME: value})
                            for item in reachability_expressions
                        ]
                    case_expression = simplify(_and(expressions))
                    reachability_expression = simplify(_and(
                        reachability_expressions
                    ))
                    case_id = f"{property_id}.{obligation}.case.{index:04d}"
                    case = ReducedCase(
                        case_id,
                        case_expression,
                        parent_hash,
                        assignment,
                        time_reduction,
                        obligation,
                        reachability_expression,
                    )
                    cases.append(case)
                    case_rows.append({
                        "case_id": case_id,
                        "obligation": obligation,
                        "boolean_assignment": dict(assignment),
                        "conditional_branch": conditional_branch,
                        "time_reduction": time_reduction,
                        "expression": expr_to_dict(case_expression),
                        "expression_sha256": expression_hash(case_expression),
                        "reachability_expression": expr_to_dict(
                            reachability_expression
                        ),
                        "reachability_expression_sha256": expression_hash(
                            reachability_expression
                        ),
                    })
                    index += 1
    return cases, {
        "rule": "exhaustive_boolean_assignment_then_exact_dnf_v1",
        "obligation": obligation,
        "parent_expression_sha256": parent_hash,
        "boolean_variables": used_booleans,
        "assignment_count": 2 ** len(used_booleans),
        "conditional_branch_count": conditional_branch_count,
        "case_count": len(cases),
        "complete": True,
        "time_reduction_rule": (
            "A single comparison affine in interval time holds somewhere on the "
            "closed interval exactly when it holds at at least one endpoint. A "
            "comparison involving only modeled time is removed only as an outer "
            "relaxation, so infeasibility still proves the original case."
        ),
        "cases": case_rows,
    }


def _equation_inventory(
    model: EquationModel,
    included_targets: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for equation in sorted(model.all_equations(), key=lambda item: (item.kind, item.target)):
        included = equation.target in included_targets
        rows.append({
            "target": equation.target,
            "kind": equation.kind,
            "source": equation.source,
            "expression": expr_to_dict(equation.expr),
            "included": included,
            "reason": (
                "dependency closure of the interval counterexample"
                if included
                else "outside the dependency closure of the interval counterexample"
            ),
        })
    return rows


def _dependency_closure(model: EquationModel, seeds: set[str]) -> set[str]:
    closure = set(seeds)
    queue = list(sorted(seeds))
    while queue:
        name = queue.pop(0)
        equation = _equation(model, name)
        if equation is None:
            continue
        for reference in sorted(expression_symbols(equation.expr)):
            if reference not in closure:
                closure.add(reference)
                queue.append(reference)
    return closure


def build_reduction(
    extractor,
    model: EquationModel,
    equation: Equation,
    shield_expression: Expr,
    guards: list[Expr],
    scenario_constraints: list[Expr],
    scenario_initial_constraints: list[Expr],
    constant_values: dict[str, Expr],
    continuous: set[str],
    boolean_variables: set[str],
    integer_variables: set[str],
    dt_record: dict[str, Any],
) -> tuple[list[ReducedCase], dict[str, Any], ReachabilityContext]:
    """Build the full physical interval counterexample and its reduction trace."""

    original = expand_definitions(model, equation.expr)
    terminal_equation = model.terminals.get("env.completion.done")
    terminal = (
        expand_definitions(model, terminal_equation.expr)
        if terminal_equation is not None
        else Const(False)
    )
    all_changing = {
        target
        for target, transition in model.transitions.items()
        if any(
            name == "dt" or name.endswith("_dt")
            for name in expression_symbols(transition.expr)
        )
    }
    aliases, alias_records = physical_aliases(
        model,
        [original, shield_expression, terminal],
        all_changing,
    )
    required_changing = set(aliases.values()) | (
        expression_symbols(original) & all_changing
    )
    missing_annotations = sorted(required_changing - continuous)
    if missing_annotations:
        raise ProofDeferred(
            "MISSING_WITHIN_STEP_MEANING",
            "missing #ContinuousRate on " + ", ".join(missing_annotations),
        )
    guard_evidence: list[dict[str, Any]] = []
    for mapping in alias_records:
        required_guards = _required_mapping_guards(
            model,
            mapping["sampled_value"],
            mapping["physical_value"],
        )
        for required_guard in required_guards:
            if required_guard not in guards:
                raise ProofDeferred(
                    "BLOCKED_INPUT",
                    "sensor mapping guard is not a controller call guard for "
                    + mapping["sampled_value"],
                )
        mapping["guard_evidence"] = [
            expr_to_dict(item) for item in required_guards
        ]
        guard_evidence.extend({
            "sampled_value": mapping["sampled_value"],
            "guard": expr_to_dict(item),
            "matched_controller_call_guard": True,
        } for item in required_guards)
    needed_physical = set(aliases.values()) | (
        expression_symbols(original) & continuous
    )
    trajectories: dict[str, Expr] = {}
    trajectory_rows: list[dict[str, Any]] = []
    for target in sorted(needed_physical):
        trajectory, dt_symbols = _trajectory(
            model, target, continuous, constant_values
        )
        trajectories[target] = trajectory
        trajectory_rows.append({
            "physical_value": target,
            "source_assignment": expr_to_dict(model.transitions[target].expr),
            "dt_symbols": dt_symbols,
            "trajectory": expr_to_dict(trajectory),
            "rule": "continuous_rate_assignment_with_held_effective_action_v1",
        })

    physical_start = {sampled: Var(target) for sampled, target in aliases.items()}
    shield_at_start = simplify(substitute(shield_expression, physical_start))
    terminal_at_start = _expand_post_update(
        model,
        substitute(terminal, physical_start),
        continuous,
    )
    terminal_at_start = simplify(substitute(
        terminal_at_start,
        constant_values,
    ))

    interval_values: dict[str, Expr] = {
        sampled: trajectories[target]
        for sampled, target in aliases.items()
        if target in trajectories
    }
    interval_values.update(trajectories)
    interval_property = _expand_post_update(
        model,
        substitute(original, physical_start),
        continuous,
    )
    interval_property = simplify(substitute(interval_property, interval_values))
    interval_property = simplify(substitute(interval_property, constant_values))

    start_values: dict[str, Expr] = {
        sampled: Var(target) for sampled, target in aliases.items()
    }
    start_values.update({target: Var(target) for target in continuous})
    start_property = _expand_post_update(
        model,
        substitute(original, physical_start),
        continuous,
    )
    start_property = simplify(substitute(start_property, start_values))
    start_property = simplify(substitute(start_property, constant_values))

    dt_exact = Const(dt_record["canonical"])
    controller_premises = [
        simplify(substitute(shield_at_start, constant_values)),
        *(simplify(substitute(item, constant_values)) for item in guards),
        *(simplify(substitute(item, constant_values)) for item in scenario_constraints),
    ]
    sampled_point_counterexample = simplify(_and([
        *controller_premises,
        Op("not", (start_property,)),
    ]))
    interval_premises = [
        *controller_premises,
        start_property,
        Op(">=", (Var(INTERVAL_TIME), Const(0))),
        Op("<=", (Var(INTERVAL_TIME), dt_exact)),
    ]
    counterexample = simplify(_and([
        *interval_premises,
        Op("not", (interval_property,)),
    ]))
    sampled_cases, sampled_coverage = _case_split(
        sampled_point_counterexample,
        boolean_variables,
        equation.target,
        "sampled_point",
    )
    interval_cases, interval_coverage = _case_split(
        counterexample,
        boolean_variables,
        equation.target,
        "physical_interval",
    )
    cases = sampled_cases + interval_cases
    coverage = {
        "rule": "sampled_point_and_physical_interval_obligations_v2",
        "complete": sampled_coverage["complete"] and interval_coverage["complete"],
        "case_count": len(cases),
        "cases": sampled_coverage["cases"] + interval_coverage["cases"],
        "obligations": {
            "sampled_point": sampled_coverage,
            "physical_interval": interval_coverage,
        },
    }

    dependency_seeds = expression_symbols(counterexample)
    for record in alias_records:
        dependency_seeds.add(record["sampled_value"])
        dependency_seeds.add(record["physical_value"])
    included_targets = _dependency_closure(model, dependency_seeds)

    domain = simplify(_and([
        *controller_premises,
        Op("not", (terminal_at_start,)),
    ]))
    specified_initial_constraints = tuple(
        Op("==", (Var(target), Const(value)))
        for target, value in sorted(model.initial_values.items())
    )
    initial_constraints = (
        specified_initial_constraints
        + tuple(scenario_constraints)
        + tuple(scenario_initial_constraints)
    )
    reachability_targets = set(model.transitions)
    reachability_targets.update(expression_symbols(domain) & model.state)
    for case in cases:
        reachability_targets.update(expression_symbols(case.expression) & model.state)
        if case.reachability_expression is not None:
            reachability_targets.update(
                expression_symbols(case.reachability_expression) & model.state
            )
    post_values: dict[str, Expr] = {}
    pending = list(sorted(reachability_targets))
    while pending:
        target = pending.pop(0)
        if target in post_values:
            continue
        if target in continuous:
            trajectory = trajectories.get(target)
            if trajectory is None:
                trajectory, _dt_symbols = _trajectory(
                    model, target, continuous, constant_values
                )
            post = simplify(substitute(
                trajectory,
                {INTERVAL_TIME: dt_exact},
            ))
        else:
            transition = model.transitions.get(target)
            if transition is None:
                post = Var(target)
            else:
                post = simplify(substitute(
                    _expand_post_update(
                        model,
                        transition.expr,
                        continuous,
                    ),
                    constant_values,
                ))
        post_values[target] = post
        for reference in sorted(expression_symbols(post) & model.state):
            if reference not in post_values and reference not in pending:
                pending.append(reference)
    cycle_order = [
        "queued actuator state machine changes",
        "same cycle constraint propagation",
        *[
            f"owned step action {index}: {fqn}"
            for index, (fqn, _body) in enumerate(extractor.parser.step_action_bodies)
        ],
    ]
    record = {
        "kind": "full_sysml_interval_reduction_v3",
        "property_id": equation.target.removeprefix("status."),
        "annotation": equation.source,
        "original_property": expr_to_dict(original),
        "physical_start_property": expr_to_dict(start_property),
        "physical_interval_property": expr_to_dict(interval_property),
        "sampled_point_counterexample": expr_to_dict(sampled_point_counterexample),
        "sampled_point_counterexample_sha256": expression_hash(
            sampled_point_counterexample
        ),
        "interval_counterexample": expr_to_dict(counterexample),
        "interval_counterexample_sha256": expression_hash(counterexample),
        "interval": {
            "time_variable": INTERVAL_TIME,
            "lower": "0/1",
            "upper": dt_record["canonical"],
            "fixed_dt": dt_record,
        },
        "cycle_order": cycle_order,
        "action_use": (
            "The controller output selected at the current reading is held through "
            "the following physical interval and is substituted through actuator equations."
        ),
        "sensor_to_physical_mappings": alias_records,
        "sensor_mapping_guard_evidence": guard_evidence,
        "trajectories": trajectory_rows,
        "sampled_point_premise": {
            "controller_contract": expr_to_dict(shield_at_start),
            "physical_property_at_interval_start": expr_to_dict(start_property),
            "controller_call_guards": [expr_to_dict(item) for item in guards],
            "scenario_constraints": [expr_to_dict(item) for item in scenario_constraints],
            "proof_rule": (
                "The controller contract and controller call guards must imply the "
                "physical property at every controller update."
            ),
        },
        "endpoint_checks": [
            {
                "physical_value": target,
                "trajectory_at_dt": expr_to_dict(simplify(substitute(
                    trajectory, {INTERVAL_TIME: dt_exact}
                ))),
                "extracted_next_value": expr_to_dict(simplify(substitute(
                    _expand_post_update(model, model.transitions[target].expr, continuous),
                    constant_values,
                ))),
                "matches": simplify(substitute(
                    trajectory, {INTERVAL_TIME: dt_exact}
                )) == simplify(substitute(
                    _expand_post_update(model, model.transitions[target].expr, continuous),
                    constant_values,
                )),
            }
            for target, trajectory in sorted(trajectories.items())
        ],
        "equation_inventory": _equation_inventory(model, included_targets),
        "case_coverage": coverage,
        "reachability_mapping": {
            "rule": "exact_initial_values_and_complete_sampled_transition_v1",
            "domain": expr_to_dict(domain),
            "completion_condition": expr_to_dict(terminal_at_start),
            "nonterminal_intervals_only": True,
            "initial_constraints": [
                expr_to_dict(item) for item in initial_constraints
            ],
            "specified_initial_constraints": [
                expr_to_dict(item) for item in specified_initial_constraints
            ],
            "scenario_initial_constraints": [
                expr_to_dict(item) for item in scenario_initial_constraints
            ],
            "initial_variables": sorted(model.initial_values),
            "post_values": {
                target: expr_to_dict(value)
                for target, value in sorted(post_values.items())
            },
            "action_variables": sorted(model.actions),
            "boolean_variables": sorted(boolean_variables),
            "integer_variables": sorted(integer_variables),
            "complete_for_case_symbols": True,
        },
        "human_description": (
            "The SysML sensor equations map the controller reading to the listed "
            "physical state. The controller contract selects the held action. The "
            "annotated physical equations then define every listed trajectory for "
            "all times from zero through dt. The cases are the exhaustive unsafe "
            "alternatives of that complete interval statement."
        ),
    }
    if not all(item["matches"] for item in record["endpoint_checks"]):
        raise ProofDeferred(
            "MISSING_WITHIN_STEP_MEANING",
            "a physical trajectory does not match its extracted sampled endpoint",
        )
    reachability = ReachabilityContext(
        domain=domain,
        initial_constraints=initial_constraints,
        initial_variables=tuple(sorted(model.initial_values)),
        post_values=tuple(sorted(post_values.items())),
        action_variables=tuple(sorted(model.actions)),
        boolean_variables=tuple(sorted(boolean_variables)),
        integer_variables=tuple(sorted(integer_variables)),
    )
    return cases, record, reachability
