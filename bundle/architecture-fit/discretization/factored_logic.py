"""Lazy proof search over factored Boolean and arithmetic obligations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

from certification.equations import Const, Expr, Ite, Op, RawRef, Var

from .full_model_reduction import (
    INTERVAL_TIME,
    ReducedCase,
    _time_degree,
    expression_hash,
    simplify,
)
from .proof_rules import expr_to_dict, expression_symbols, substitute


Attempt = dict[str, Any]
Checker = Callable[[ReducedCase, int], Attempt]
Predicate = Callable[[Expr], bool]


def _and(expressions: list[Expr]) -> Expr:
    if not expressions:
        return Const(True)
    if len(expressions) == 1:
        return expressions[0]
    return _canonical(Op("and", tuple(expressions)))


def _or(expressions: list[Expr]) -> Expr:
    if not expressions:
        return Const(False)
    if len(expressions) == 1:
        return expressions[0]
    return _canonical(Op("or", tuple(expressions)))


def _canonical(expression: Expr) -> Expr:
    if isinstance(expression, (Const, Var, RawRef)):
        return expression
    if isinstance(expression, Ite):
        return simplify(Ite(
            _canonical(expression.cond),
            _canonical(expression.then_expr),
            _canonical(expression.else_expr),
        ))
    if not isinstance(expression, Op):
        return expression
    arguments = tuple(_canonical(item) for item in expression.args)
    normalized = simplify(Op(expression.op, arguments))
    if not isinstance(normalized, Op) or normalized.op not in {"and", "or"}:
        return normalized
    flattened: list[Expr] = []
    for item in normalized.args:
        if isinstance(item, Op) and item.op == normalized.op:
            flattened.extend(item.args)
        else:
            flattened.append(item)
    unique = {expression_hash(item): item for item in flattened}
    ordered = tuple(unique[key] for key in sorted(unique))
    if len(ordered) == 1:
        return ordered[0]
    return simplify(Op(normalized.op, ordered))


def _is_boolean(expression: Expr, boolean_variables: set[str]) -> bool:
    if isinstance(expression, Const):
        return isinstance(expression.value, bool)
    if isinstance(expression, Var):
        return expression.name in boolean_variables
    if isinstance(expression, Ite):
        return _is_boolean(
            expression.then_expr,
            boolean_variables,
        ) and _is_boolean(expression.else_expr, boolean_variables)
    if isinstance(expression, Op):
        return expression.op in {
            "not", "and", "or", "implies", "==", ">", "<", ">=", "<=",
        }
    return False


def _normal_boolean(
    expression: Expr,
    boolean_variables: set[str],
    *,
    truth: bool = True,
) -> Expr:
    """Push logical negation to atoms without distributing conjunctions."""

    expression = _canonical(expression)
    if isinstance(expression, Const) and isinstance(expression.value, bool):
        return Const(expression.value is truth)
    if isinstance(expression, Var) and expression.name in boolean_variables:
        return expression if truth else Op("not", (expression,))
    if isinstance(expression, Ite) and _is_boolean(
        expression,
        boolean_variables,
    ):
        return _or([
            _and([
                _normal_boolean(
                    expression.cond,
                    boolean_variables,
                    truth=True,
                ),
                _normal_boolean(
                    expression.then_expr,
                    boolean_variables,
                    truth=truth,
                ),
            ]),
            _and([
                _normal_boolean(
                    expression.cond,
                    boolean_variables,
                    truth=False,
                ),
                _normal_boolean(
                    expression.else_expr,
                    boolean_variables,
                    truth=truth,
                ),
            ]),
        ])
    if not isinstance(expression, Op):
        return expression if truth else Op("not", (expression,))
    if expression.op == "not":
        return _normal_boolean(
            expression.args[0],
            boolean_variables,
            truth=not truth,
        )
    if expression.op == "implies":
        left, right = expression.args
        expanded = _or([
            _normal_boolean(left, boolean_variables, truth=False),
            _normal_boolean(right, boolean_variables, truth=True),
        ])
        return _normal_boolean(expanded, boolean_variables, truth=truth)
    if expression.op in {"and", "or"}:
        operation = expression.op if truth else (
            "or" if expression.op == "and" else "and"
        )
        items = [
            _normal_boolean(
                item,
                boolean_variables,
                truth=truth,
            )
            for item in expression.args
        ]
        return _and(items) if operation == "and" else _or(items)
    if expression.op == "==" and all(
        _is_boolean(item, boolean_variables) for item in expression.args
    ):
        left, right = expression.args
        if truth:
            return _or([
                _and([
                    _normal_boolean(left, boolean_variables, truth=True),
                    _normal_boolean(right, boolean_variables, truth=True),
                ]),
                _and([
                    _normal_boolean(left, boolean_variables, truth=False),
                    _normal_boolean(right, boolean_variables, truth=False),
                ]),
            ])
        return _or([
            _and([
                _normal_boolean(left, boolean_variables, truth=True),
                _normal_boolean(right, boolean_variables, truth=False),
            ]),
            _and([
                _normal_boolean(left, boolean_variables, truth=False),
                _normal_boolean(right, boolean_variables, truth=True),
            ]),
        ])
    if expression.op in {"==", ">", "<", ">=", "<="}:
        left, right = expression.args
        if truth:
            if expression.op == "==":
                return _and([Op("<=", (left, right)), Op(">=", (left, right))])
            return expression
        inverse = {">": "<=", "<": ">=", ">=": "<", "<=": ">"}
        if expression.op == "==":
            return _or([Op("<", (left, right)), Op(">", (left, right))])
        return Op(inverse[expression.op], (left, right))
    return expression if truth else Op("not", (expression,))


def _conjuncts(expression: Expr) -> list[Expr]:
    if isinstance(expression, Const) and expression.value is True:
        return []
    if isinstance(expression, Op) and expression.op == "and":
        result: list[Expr] = []
        for item in expression.args:
            result.extend(_conjuncts(item))
        return result
    return [expression]


def _common_conjuncts(expression: Expr) -> dict[str, Expr]:
    if isinstance(expression, Const):
        return {}
    if isinstance(expression, Op) and expression.op == "and":
        result: dict[str, Expr] = {}
        for item in expression.args:
            result.update(_common_conjuncts(item))
        return result
    if isinstance(expression, Op) and expression.op == "or":
        children = [_common_conjuncts(item) for item in expression.args]
        if not children:
            return {}
        shared = set(children[0])
        for child in children[1:]:
            shared.intersection_update(child)
        return {key: children[0][key] for key in sorted(shared)}
    if isinstance(expression, Ite):
        return {}
    return {expression_hash(expression): expression}


def _boolean_occurrences(
    expression: Expr,
    boolean_variables: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()

    def visit(item: Expr) -> None:
        if isinstance(item, Var) and item.name in boolean_variables:
            counts[item.name] += 1
        elif isinstance(item, Op):
            for argument in item.args:
                visit(argument)
        elif isinstance(item, Ite):
            visit(item.cond)
            visit(item.then_expr)
            visit(item.else_expr)

    visit(expression)
    return counts


def _first_ite(expression: Expr) -> Ite | None:
    if isinstance(expression, Ite):
        return expression
    if isinstance(expression, Op):
        for item in expression.args:
            found = _first_ite(item)
            if found is not None:
                return found
    return None


def _first_or(expression: Expr) -> Op | None:
    if isinstance(expression, Op) and expression.op == "or":
        return expression
    if isinstance(expression, Op):
        for item in expression.args:
            found = _first_or(item)
            if found is not None:
                return found
    if isinstance(expression, Ite):
        for item in (
            expression.cond,
            expression.then_expr,
            expression.else_expr,
        ):
            found = _first_or(item)
            if found is not None:
                return found
    return None


def _replace(expression: Expr, target: Expr, replacement: Expr) -> Expr:
    if expression == target:
        return replacement
    if isinstance(expression, Op):
        return _canonical(Op(
            expression.op,
            tuple(_replace(item, target, replacement) for item in expression.args),
        ))
    if isinstance(expression, Ite):
        return _canonical(Ite(
            _replace(expression.cond, target, replacement),
            _replace(expression.then_expr, target, replacement),
            _replace(expression.else_expr, target, replacement),
        ))
    return expression


def _affine_interval_endpoint_split(
    expression: Expr,
) -> tuple[list[Expr], dict[str, Any]] | None:
    """Reduce one affine interval comparison to its two endpoints."""

    conjuncts = _conjuncts(expression)
    lower_bound: Expr | None = None
    upper_bound: Expr | None = None
    interval_bounds: list[Expr] = []
    for item in conjuncts:
        if not isinstance(item, Op) or len(item.args) != 2:
            continue
        left, right = item.args
        if (
            isinstance(left, Var)
            and left.name == INTERVAL_TIME
            and isinstance(right, Const)
            and right.value == 0
            and item.op == ">="
        ):
            lower_bound = item
            interval_bounds.append(item)
        elif (
            isinstance(left, Var)
            and left.name == INTERVAL_TIME
            and item.op == "<="
            and INTERVAL_TIME not in expression_symbols(right)
        ):
            upper_bound = right
            interval_bounds.append(item)
    if lower_bound is None or upper_bound is None:
        return None

    body = [item for item in conjuncts if item not in interval_bounds]
    time_gates = [
        item
        for item in body
        if INTERVAL_TIME in expression_symbols(item)
        and all(
            name == INTERVAL_TIME or "time" in name.lower()
            for name in expression_symbols(item)
        )
    ]
    reduced_body = [item for item in body if item not in time_gates]
    changing = [
        item
        for item in reduced_body
        if INTERVAL_TIME in expression_symbols(item)
    ]
    if len(changing) != 1:
        return None
    changing_item = changing[0]
    if (
        not isinstance(changing_item, Op)
        or changing_item.op not in {">", "<", ">=", "<="}
        or len(changing_item.args) != 2
        or _time_degree(changing_item.args[0]) is None
        or int(_time_degree(changing_item.args[0]) or 0) > 1
        or _time_degree(changing_item.args[1]) is None
        or int(_time_degree(changing_item.args[1]) or 0) > 1
    ):
        return None

    relaxed_expression = _and(reduced_body)
    endpoints = [Const(0), upper_bound]
    children = [
        _canonical(substitute(
            relaxed_expression,
            {INTERVAL_TIME: endpoint},
        ))
        for endpoint in endpoints
    ]
    return children, {
        "relaxed_expression": relaxed_expression,
        "removed_interval_bounds": interval_bounds,
        "removed_time_gates": time_gates,
        "changing_comparison": changing_item,
        "time_variable": INTERVAL_TIME,
        "endpoints": endpoints,
    }


@dataclass
class _Metrics:
    visited_node_count: int = 0
    actual_split_count: int = 0
    proof_leaf_count: int = 0
    cache_hits: int = 0
    checker_query_count: int = 0

    def record(self) -> dict[str, int]:
        return {
            "visited_node_count": self.visited_node_count,
            "actual_split_count": self.actual_split_count,
            "proof_leaf_count": self.proof_leaf_count,
            "cache_hits": self.cache_hits,
            "checker_query_count": self.checker_query_count,
        }


def run_lazy_factored_checker(
    reduced_case: ReducedCase,
    boolean_variables: set[str],
    checker_name: str,
    checker: Checker,
    accepted: Predicate,
    *,
    timeout_ms: int,
    context_sha256: str,
    total_timeout_ms: int | None = None,
    node_limit: int = 4096,
) -> Attempt:
    """Prove a factored formula impossible while splitting only on demand."""

    deadline = (
        None
        if total_timeout_ms is None
        else monotonic() + total_timeout_ms / 1000.0
    )
    metrics = _Metrics()
    cache: dict[str, tuple[str, dict[str, Any]]] = {}
    expression_pool: dict[str, dict[str, Any]] = {}

    def intern(expression: Expr) -> str:
        key = expression_hash(expression)
        expression_pool.setdefault(key, expr_to_dict(expression))
        return key

    def attempt(expression: Expr, suffix: str) -> Attempt:
        metrics.checker_query_count += 1
        leaf = ReducedCase(
            case_id=f"{reduced_case.case_id}.{checker_name}.{suffix}",
            expression=expression,
            parent_hash=expression_hash(reduced_case.expression),
            boolean_assignment=(),
            time_reduction="lazy_factored_formula",
            obligation=reduced_case.obligation,
            reachability_expression=expression,
            factored=True,
        )
        return checker(leaf, timeout_ms)

    def solve(expression: Expr) -> tuple[str, dict[str, Any]]:
        if deadline is not None and monotonic() >= deadline:
            key = intern(expression)
            return "DEFERRED", {
                "rule": "unresolved_factored_formula_v1",
                "reason_code": "TIMEOUT",
                "expression_sha256": key,
            }
        if metrics.visited_node_count >= node_limit:
            key = intern(expression)
            return "DEFERRED", {
                "rule": "unresolved_factored_formula_v1",
                "reason_code": "NODE_LIMIT",
                "expression_sha256": key,
            }
        expression = _canonical(_normal_boolean(
            expression,
            boolean_variables,
        ))
        key = intern(expression)
        cache_key = f"{context_sha256}:{checker_name}:{key}"
        if cache_key in cache:
            metrics.cache_hits += 1
            outcome, _node = cache[cache_key]
            return outcome, {
                "rule": "reused_factored_proof_v1",
                "expression_sha256": key,
                "context_sha256": context_sha256,
                "checker": checker_name,
            }
        metrics.visited_node_count += 1
        if isinstance(expression, Const) and expression.value is False:
            metrics.proof_leaf_count += 1
            result = ("CERTIFIED", {
                "rule": "constant_false_factored_leaf_v1",
                "expression_sha256": key,
            })
            cache[cache_key] = result
            return result

        common = [
            item
            for item in _common_conjuncts(expression).values()
            if accepted(item)
        ]
        if common:
            common_expression = _and(common)
            common_attempt = attempt(common_expression, "common")
            if common_attempt.get("outcome") == "CERTIFIED":
                metrics.proof_leaf_count += 1
                result = ("CERTIFIED", {
                    "rule": "common_conjunct_factored_proof_v1",
                    "expression_sha256": key,
                    "common_expression_sha256": intern(common_expression),
                    "checker_attempt": common_attempt,
                })
                cache[cache_key] = result
                return result

        occurrences = _boolean_occurrences(expression, boolean_variables)
        if occurrences:
            pivot = min(
                occurrences,
                key=lambda name: (-occurrences[name], name),
            )
            children = [
                _canonical(substitute(expression, {pivot: Const(value)}))
                for value in (False, True)
            ]
            split_kind = "boolean_variable"
            split_record: dict[str, Any] = {"variable": pivot}
        else:
            conditional = _first_ite(expression)
            alternative = _first_or(expression)
            if conditional is not None:
                children = [
                    _and([
                        _normal_boolean(
                            conditional.cond,
                            boolean_variables,
                            truth=value,
                        ),
                        _replace(
                            expression,
                            conditional,
                            conditional.then_expr
                            if value
                            else conditional.else_expr,
                        ),
                    ])
                    for value in (False, True)
                ]
                split_kind = "conditional"
                split_record = {
                    "condition_sha256": intern(conditional.cond),
                }
            elif alternative is not None:
                alternatives = list(alternative.args)
                left = alternatives[0]
                right = _or(alternatives[1:])
                children = [
                    _replace(expression, alternative, left),
                    _replace(expression, alternative, right),
                ]
                split_kind = "logical_alternative"
                split_record = {
                    "alternative_expression_sha256": intern(alternative),
                }
            else:
                endpoint_split = _affine_interval_endpoint_split(expression)
                if endpoint_split is not None:
                    children, endpoint_record = endpoint_split
                    split_kind = "affine_interval_endpoints"
                    split_record = {
                        "relaxed_expression_sha256": intern(
                            endpoint_record["relaxed_expression"]
                        ),
                        "removed_interval_bound_sha256": [
                            intern(item)
                            for item in endpoint_record["removed_interval_bounds"]
                        ],
                        "removed_time_gate_sha256": [
                            intern(item)
                            for item in endpoint_record["removed_time_gates"]
                        ],
                        "changing_comparison_sha256": intern(
                            endpoint_record["changing_comparison"]
                        ),
                        "time_variable": endpoint_record["time_variable"],
                        "endpoint_sha256": [
                            intern(item) for item in endpoint_record["endpoints"]
                        ],
                    }
                else:
                    conjuncts = _conjuncts(expression)
                    if conjuncts and all(accepted(item) for item in conjuncts):
                        leaf_attempt = attempt(expression, "leaf")
                        outcome = str(leaf_attempt.get("outcome", "DEFERRED"))
                        metrics.proof_leaf_count += 1
                        result = (outcome, {
                            "rule": "factored_checker_leaf_v1",
                            "expression_sha256": key,
                            "checker_attempt": leaf_attempt,
                        })
                        cache[cache_key] = result
                        return result
                    result = ("DEFERRED", {
                        "rule": "unresolved_factored_formula_v1",
                        "reason_code": "INAPPLICABLE_REDUCED_FORM",
                        "expression_sha256": key,
                    })
                    cache[cache_key] = result
                    return result

        metrics.actual_split_count += 1
        child_records: list[dict[str, Any]] = []
        outcome = "CERTIFIED"
        for index, child in enumerate(children):
            child_outcome, child_record = solve(child)
            child_records.append({
                "branch": index,
                "expression_sha256": intern(child),
                "outcome": child_outcome,
                "proof": child_record,
            })
            if (
                child_outcome == "VIOLATION"
                and split_kind == "affine_interval_endpoints"
            ):
                outcome = "DEFERRED"
                continue
            if child_outcome == "VIOLATION":
                outcome = "VIOLATION"
                break
            if child_outcome != "CERTIFIED":
                outcome = "DEFERRED"
        result = (outcome, {
            "rule": "exact_factored_split_v1",
            "split_kind": split_kind,
            "expression_sha256": key,
            **split_record,
            "children": child_records,
        })
        cache[cache_key] = result
        return result

    normalized_root = reduced_case.expression
    intern(normalized_root)
    logical_root = _canonical(_normal_boolean(
        normalized_root,
        boolean_variables,
    ))
    intern(logical_root)
    outcome, proof_tree = solve(logical_root)
    if expression_hash(logical_root) != expression_hash(normalized_root):
        proof_tree = {
            "rule": "exact_boolean_normalization_v1",
            "expression_sha256": expression_hash(normalized_root),
            "normalized_expression_sha256": expression_hash(logical_root),
            "outcome": outcome,
            "proof": proof_tree,
        }
    if outcome == "DEFERRED":
        proof_record = {
            "rule": "lazy_factored_formula_attempt_v2",
            "checker": checker_name,
            "context_sha256": context_sha256,
            "source_expression_sha256": expression_hash(normalized_root),
            "coverage_complete": False,
            "metrics": metrics.record(),
        }
    else:
        proof_record = {
            "rule": "lazy_factored_formula_coverage_v2",
            "checker": checker_name,
            "context_sha256": context_sha256,
            "boolean_variables": sorted(boolean_variables),
            "source_expression_sha256": expression_hash(normalized_root),
            "expression_pool": {
                key: expression_pool[key] for key in sorted(expression_pool)
            },
            "expression_pool_count": len(expression_pool),
            "coverage_complete": outcome == "CERTIFIED",
            "metrics": metrics.record(),
            "tree": proof_tree,
        }
    return {
        "outcome": outcome,
        "reason_code": "" if outcome == "CERTIFIED" else (
            "COUNTEREXAMPLE_REPLAYED"
            if outcome == "VIOLATION"
            else "UNRESOLVED_FACTORED_FORMULA"
        ),
        "detail": (
            f"the {checker_name} checker certified the complete factored formula"
            if outcome == "CERTIFIED"
            else f"the {checker_name} checker left part of the factored formula unresolved"
        ),
        "applicability_checks": {
            "accepted": True,
            "case_id": reduced_case.case_id,
            "factored_formula": True,
            **metrics.record(),
        },
        "proof": proof_record,
    }
