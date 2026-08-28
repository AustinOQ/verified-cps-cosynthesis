"""Certified linear and convex exclusion of unreachable unsafe cases."""

from __future__ import annotations

import hashlib
import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import count, product
from time import monotonic
from typing import Any, Callable

from certification.equations import Const, EquationModel, Expr, Ite, Op, RawRef, Var
from certification.solver import Encoder, infer_sorts

from .convex_checker import run_convex_checker
from .convex_envelope_checker import run_convex_envelope_checker
from .exact_replay import replay_serialized_boolean_expression, serialize_exact_value
from .full_model_reduction import (
    ReducedCase,
    ReachabilityContext,
    _comparison_expression,
    _split_conditionals,
    expression_hash,
    simplify,
)
from .factored_logic import run_lazy_factored_checker
from .linear_envelope_checker import run_linear_envelope_checker
from .linear_checker import run_linear_checker
from .optimization_common import conjunctive_comparisons, quadratic_constraints
from .proof_rules import (
    ProofDeferred,
    boolean_dnf,
    expr_to_dict,
    expression_is_linear,
    expression_symbols,
    prove_implication_exact,
    substitute,
)

try:  # pragma: no cover - integration environment determines availability
    import z3  # type: ignore
except Exception:  # pragma: no cover
    z3 = None
else:  # Proof generation must be enabled before this process creates a solver.
    z3.set_param(proof=True)


MAX_ARITHMETIC_BRANCHES = 4096


@dataclass
class _SharedRelationalRegion:
    context_sha256: str
    invariant: tuple[Expr, ...]
    record: dict[str, Any]
    query_runner: _IncrementalSmtQueryRunner


@dataclass
class SharedReachabilityCache:
    """Reuse one checked reachable region and identical safety queries."""

    regions: dict[str, _SharedRelationalRegion] = field(default_factory=dict)
    region_failures: dict[str, dict[str, Any]] = field(default_factory=dict)
    safety_queries: dict[str, dict[str, Any]] = field(default_factory=dict)
    safety_expressions: dict[str, Expr] = field(default_factory=dict, repr=False)

    def export(self) -> dict[str, Any]:
        return {
            "regions": {
                key: value.record for key, value in sorted(self.regions.items())
            },
            "region_failures": {
                key: value for key, value in sorted(self.region_failures.items())
            },
            "safety_queries": {
                key: value for key, value in sorted(self.safety_queries.items())
            },
        }


def _remaining_timeout_ms(deadline: float) -> int:
    remaining = int((deadline - monotonic()) * 1000)
    if remaining <= 0:
        raise ProofDeferred("TIMEOUT", "reachability checker timeout")
    return remaining


@contextmanager
def _hard_timeout(timeout_ms: int):
    def handle_timeout(_signum: int, _frame: Any) -> None:
        raise ProofDeferred("TIMEOUT", "reachability checker timeout")

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        signal.signal(signal.SIGALRM, handle_timeout)
    except (AttributeError, ValueError):
        yield
        return
    started = monotonic()
    signal.setitimer(signal.ITIMER_REAL, timeout_ms / 1000.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_timer[0] - elapsed),
                previous_timer[1],
            )


def _and(expressions: list[Expr] | tuple[Expr, ...]) -> Expr:
    items = tuple(expressions)
    if not items:
        return Const(True)
    if len(items) == 1:
        return items[0]
    return simplify(Op("and", items))


def _mode_condition(
    reduced_case: ReducedCase,
    action_names: dict[str, str],
) -> Expr:
    tests: list[Expr] = []
    for name, value in reduced_case.boolean_assignment:
        tests.append(Op("==", (Var(action_names.get(name, name)), Const(value))))
    return _and(tests)


def _step_action_names(
    context: ReachabilityContext,
    step: int,
) -> dict[str, str]:
    if step == 0:
        return {name: name for name in context.action_variables}
    return {
        name: f"reach_action_{step}__{name}"
        for name in context.action_variables
    }


def _lift(
    expression: Expr,
    state_values: dict[str, Expr],
    action_names: dict[str, str],
) -> Expr:
    action_values = {
        name: Var(renamed) for name, renamed in action_names.items()
    }
    return simplify(substitute(
        substitute(expression, action_values),
        state_values,
    ))


def _state_sequence(
    context: ReachabilityContext,
    depth: int,
    *,
    deadline: float | None = None,
) -> tuple[list[dict[str, Expr]], list[dict[str, str]], list[Expr]]:
    """Build linearly sized state frames and explicit transition equalities."""

    generic_post = context.post_dict()
    state_names = sorted(generic_post)
    states: list[dict[str, Expr]] = []
    for step in range(depth + 1):
        states.append({
            name: Var(name if step == 0 else f"reach_state_{step}__{name}")
            for name in state_names
        })
    actions = [_step_action_names(context, step) for step in range(depth + 1)]
    transitions: list[Expr] = []
    for step in range(depth):
        if deadline is not None:
            _remaining_timeout_ms(deadline)
        equations: list[Expr] = []
        for target, expression in generic_post.items():
            if deadline is not None:
                _remaining_timeout_ms(deadline)
            equations.append(Op("==", (
                states[step + 1][target],
                _lift(expression, states[step], actions[step]),
            )))
        transitions.append(_and(equations))
    return states, actions, transitions


def _arithmetic_cases(
    expression: Expr,
    boolean_variables: set[str],
    prefix: str,
    context: ReachabilityContext,
    valid_action_modes: tuple[tuple[tuple[str, bool], ...], ...],
    *,
    deadline: float,
) -> list[ReducedCase]:
    _remaining_timeout_ms(deadline)
    used_booleans = expression_symbols(expression) & boolean_variables
    boolean_actions = sorted(
        set(context.action_variables) & set(context.boolean_variables)
    )
    step_action_groups: list[tuple[int, dict[str, str]]] = []
    used_steps: set[int] = set()
    if set(boolean_actions) & used_booleans:
        used_steps.add(0)
    for name in used_booleans:
        if not name.startswith("reach_action_") or "__" not in name:
            continue
        step_text, source = name.split("__", 1)
        step_text = step_text.removeprefix("reach_action_")
        if step_text.isdigit() and source in boolean_actions:
            used_steps.add(int(step_text))
    for step in sorted(used_steps):
        names = _step_action_names(context, step)
        if any(names[name] in used_booleans for name in boolean_actions):
            step_action_groups.append((step, names))
    grouped_names = {
        names[name]
        for _step, names in step_action_groups
        for name in boolean_actions
    }
    remaining_booleans = sorted(used_booleans - grouped_names)
    grouped_assignments: list[list[dict[str, bool]]] = []
    for _step, names in step_action_groups:
        grouped_assignments.append([
            {names[name]: value for name, value in mode}
            for mode in valid_action_modes
        ])
    if not grouped_assignments:
        grouped_assignments = [[{}]]
    cases: list[ReducedCase] = []
    parent_hash = expression_hash(expression)
    index = 0
    for selected_modes in product(*grouped_assignments):
        _remaining_timeout_ms(deadline)
        grouped = {
            name: value
            for mapping in selected_modes
            for name, value in mapping.items()
        }
        for values in product((False, True), repeat=len(remaining_booleans)):
            _remaining_timeout_ms(deadline)
            assignment = tuple(sorted({
                **grouped,
                **dict(zip(remaining_booleans, values)),
            }.items()))
            assigned = simplify(substitute(expression, {
                name: Const(value) for name, value in assignment
            }))
            for _branch_name, branch in _split_conditionals(
                assigned,
                limit=MAX_ARITHMETIC_BRANCHES,
            ):
                for comparisons in boolean_dnf(branch, boolean_variables):
                    arithmetic = _and([
                        _comparison_expression(item) for item in comparisons
                    ])
                    cases.append(ReducedCase(
                        f"{prefix}.{index:05d}",
                        arithmetic,
                        parent_hash,
                        assignment,
                        "reachability",
                        "reachability",
                    ))
                    index += 1
                    if len(cases) > MAX_ARITHMETIC_BRANCHES:
                        raise ProofDeferred(
                            "INCOMPLETE_CASE_COVERAGE",
                            f"reachability split exceeds {MAX_ARITHMETIC_BRANCHES} branches",
                        )
    return cases


def _valid_action_modes(
    context: ReachabilityContext,
) -> tuple[tuple[tuple[str, bool], ...], ...]:
    boolean_actions = sorted(
        set(context.action_variables) & set(context.boolean_variables)
    )
    if not boolean_actions:
        return ((),)
    modes: list[tuple[tuple[str, bool], ...]] = []
    remaining_booleans = set(context.boolean_variables) - set(boolean_actions)
    for values in product((False, True), repeat=len(boolean_actions)):
        mode = tuple(zip(boolean_actions, values))
        assigned = simplify(substitute(context.domain, {
            name: Const(value) for name, value in mode
        }))
        try:
            infeasible = prove_implication_exact(
                [assigned],
                Const(False),
                remaining_booleans,
            ).get("proved") is True
        except ProofDeferred:
            infeasible = False
        if not infeasible:
            modes.append(mode)
    if not modes:
        raise ProofDeferred(
            "BLOCKED_INPUT",
            "the controller contract has no feasible Boolean action mode",
        )
    return tuple(modes)


def _query_set(
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    depth: int,
    *,
    deadline: float | None = None,
) -> tuple[list[tuple[str, Expr]], Expr, set[str]]:
    if deadline is not None:
        _remaining_timeout_ms(deadline)
    states, actions, transitions = _state_sequence(
        context,
        depth,
        deadline=deadline,
    )
    unsafe_expression = (
        reduced_case.reachability_expression or reduced_case.expression
    )
    domains = [
        _lift(context.domain, states[step], actions[step])
        for step in range(depth + 1)
    ]
    if deadline is not None:
        _remaining_timeout_ms(deadline)
    case_at = [
        _and([
            _mode_condition(reduced_case, actions[step]),
            _lift(unsafe_expression, states[step], actions[step]),
        ])
        for step in range(depth + 1)
    ]
    safe_at = [simplify(Op("not", (item,))) for item in case_at]

    bases: list[tuple[str, Expr]] = []
    for step in range(depth):
        if deadline is not None:
            _remaining_timeout_ms(deadline)
        bases.append((
            f"base_step_{step}",
            _and([
                *context.initial_constraints,
                *domains[:step],
                *transitions[:step],
                case_at[step],
            ]),
        ))
    induction = _and([
        *domains[:depth],
        *transitions[:depth],
        *safe_at[:depth],
        case_at[depth],
    ])

    renamed_booleans = set(context.boolean_variables)
    source_boolean_actions = set(context.boolean_variables) & set(
        context.action_variables
    )
    for step in range(1, depth + 1):
        renamed_booleans.update(
            actions[step][name] for name in source_boolean_actions
        )
    return bases, induction, renamed_booleans


def _raw_reference_names(expression: Expr) -> set[str]:
    if isinstance(expression, RawRef):
        return {expression.path}
    if isinstance(expression, Op):
        result: set[str] = set()
        for argument in expression.args:
            result.update(_raw_reference_names(argument))
        return result
    if isinstance(expression, Ite):
        return (
            _raw_reference_names(expression.cond)
            | _raw_reference_names(expression.then_expr)
            | _raw_reference_names(expression.else_expr)
        )
    return set()


def _z3_exact_value(value: Any) -> bool | Fraction:
    if z3.is_true(value):
        return True
    if z3.is_false(value):
        return False
    if z3.is_int_value(value):
        return Fraction(value.as_long())
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    raise ValueError(f"Z3 value is not an exact rational or Boolean: {value}")


def _exact_query_values(
    model: EquationModel,
    encoder: Encoder,
    solver_model: Any,
    expression: Expr,
) -> dict[str, bool | str]:
    raw_references = _raw_reference_names(expression)
    values: dict[str, bool | str] = {}
    for name in sorted(expression_symbols(expression)):
        encoded = (
            encoder.const_var(name)
            if name in raw_references
            else encoder.encode_expr(Var(name), 1, "current")
        )
        exact = _z3_exact_value(
            solver_model.eval(encoded, model_completion=True)
        )
        values[name] = serialize_exact_value(exact)
    return values


def _reachability_sorts(
    model: EquationModel,
    context: ReachabilityContext,
    expression: Expr,
) -> tuple[dict[str, str], list[str]]:
    sorts, conflicts = infer_sorts(model)
    for name in context.integer_variables:
        sorts[name] = "Int"
    for name in expression_symbols(expression):
        if "__" not in name:
            continue
        source = name.split("__", 1)[1]
        if (
            name.startswith("rel_state_")
            or name.startswith("reach_state_")
        ) and source in (
            set(context.post_dict()) | set(context.initial_variables)
        ):
            sorts[name] = (
                "Bool" if source in context.boolean_variables
                else "Int" if source in context.integer_variables
                else sorts.get(source, "Real")
            )
        elif (
            name.startswith("reach_action_")
            or name.startswith("rel_action_")
        ) and source in context.action_variables:
            sorts[name] = (
                "Bool" if source in context.boolean_variables
                else "Int" if source in context.integer_variables
                else sorts.get(source, "Real")
            )
    return sorts, conflicts


class _IncrementalSmtQueryRunner:
    def __init__(
        self,
        model: EquationModel,
        context: ReachabilityContext,
        type_scope: Expr,
    ) -> None:
        self.model = model
        self.context = context
        self.sorts, self.conflicts = _reachability_sorts(
            model,
            context,
            type_scope,
        )
        self.encoder = Encoder(model, self.sorts)
        self.solver = z3.Solver()

    def run(
        self,
        expression: Expr,
        *,
        timeout_ms: int,
        record_sat_values: bool,
        replay_sat_query: bool,
    ) -> dict[str, Any]:
        serialized = expr_to_dict(expression)
        query_hash = expression_hash(expression)
        if self.conflicts:
            return {
                "solver_status": "unsupported",
                "reason_code": "UNSUPPORTED_EXPRESSION",
                "detail": "; ".join(self.conflicts),
                "query_expression": serialized,
                "query_expression_sha256": query_hash,
            }
        self.solver.push()
        try:
            self.solver.set(timeout=int(timeout_ms))
            self.solver.add(self.encoder.encode_expr(expression, 1, "current"))
            query_smt2 = self.solver.to_smt2()
            result = self.solver.check()
            record: dict[str, Any] = {
                "solver": "z3",
                "solver_reuse": "persistent_push_pop_v1",
                "solver_status": str(result),
                "query_expression": serialized,
                "query_expression_sha256": query_hash,
                "query_smt2": query_smt2,
                "query_smt2_sha256": hashlib.sha256(
                    query_smt2.encode("utf-8")
                ).hexdigest(),
            }
            if result == z3.unknown:
                reason = self.solver.reason_unknown()
                record["reason_code"] = (
                    "TIMEOUT" if "timeout" in reason.lower() else "PROOF_REJECTED"
                )
                record["detail"] = reason
                return record
            if result == z3.unsat:
                proof_text = self.solver.proof().sexpr()
                record["z3_proof"] = proof_text
                record["z3_proof_sha256"] = hashlib.sha256(
                    proof_text.encode("utf-8")
                ).hexdigest()
                return record
            if not record_sat_values:
                return record
            exact_values = _exact_query_values(
                self.model,
                self.encoder,
                self.solver.model(),
                expression,
            )
            record["exact_values"] = exact_values
            if replay_sat_query:
                record["exact_replay"] = replay_serialized_boolean_expression(
                    serialized,
                    exact_values,
                )
            return record
        finally:
            self.solver.pop()


def _run_smt_query(
    model: EquationModel,
    context: ReachabilityContext,
    expression: Expr,
    *,
    timeout_ms: int,
    record_sat_values: bool = True,
    replay_sat_query: bool = True,
    runner: _IncrementalSmtQueryRunner | None = None,
) -> dict[str, Any]:
    if runner is not None:
        return runner.run(
            expression,
            timeout_ms=timeout_ms,
            record_sat_values=record_sat_values,
            replay_sat_query=replay_sat_query,
        )
    serialized = expr_to_dict(expression)
    query_hash = expression_hash(expression)
    sorts, conflicts = _reachability_sorts(model, context, expression)
    if conflicts:
        return {
            "solver_status": "unsupported",
            "reason_code": "UNSUPPORTED_EXPRESSION",
            "detail": "; ".join(conflicts),
            "query_expression": serialized,
            "query_expression_sha256": query_hash,
        }
    encoder = Encoder(model, sorts)
    solver = z3.Solver()
    solver.set(timeout=int(timeout_ms))
    solver.add(encoder.encode_expr(expression, 1, "current"))
    query_smt2 = solver.to_smt2()
    query_smt2_hash = hashlib.sha256(query_smt2.encode("utf-8")).hexdigest()
    result = solver.check()
    record: dict[str, Any] = {
        "solver": "z3",
        "solver_status": str(result),
        "query_expression": serialized,
        "query_expression_sha256": query_hash,
        "query_smt2": query_smt2,
        "query_smt2_sha256": query_smt2_hash,
    }
    if result == z3.unknown:
        reason = solver.reason_unknown()
        record["reason_code"] = (
            "TIMEOUT" if "timeout" in reason.lower() else "PROOF_REJECTED"
        )
        record["detail"] = reason
        return record
    if result == z3.unsat:
        proof_text = solver.proof().sexpr()
        record["z3_proof"] = proof_text
        record["z3_proof_sha256"] = hashlib.sha256(
            proof_text.encode("utf-8")
        ).hexdigest()
        return record
    if not record_sat_values:
        return record
    exact_values = _exact_query_values(
        model,
        encoder,
        solver.model(),
        expression,
    )
    record["exact_values"] = exact_values
    if replay_sat_query:
        record["exact_replay"] = replay_serialized_boolean_expression(
            serialized,
            exact_values,
        )
    return record


def _run_smt_reachability_checker(
    model: EquationModel,
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    """Check exact finite prefixes and induction with SMT."""

    if reduced_case.obligation != "physical_interval":
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "SMT reachability is only required for physical interval cases",
            "applicability_checks": {"accepted": False},
        }
    if z3 is None:
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "z3-solver is unavailable",
            "applicability_checks": {"accepted": False},
        }
    if not set(context.post_dict()).issubset(set(context.initial_variables)):
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "the SMT trace state is not completely initialized",
            "applicability_checks": {"accepted": False},
        }

    source_hash = expression_hash(reduced_case.expression)
    reachability_hash = expression_hash(
        reduced_case.reachability_expression or reduced_case.expression
    )
    depth_attempts: list[dict[str, Any]] = []
    deadline = monotonic() + (timeout_ms / 1000.0)
    try:
        for depth in count(1):
            _remaining_timeout_ms(deadline)
            bases, induction, _boolean_variables = _query_set(
                reduced_case,
                context,
                depth,
                deadline=deadline,
            )
            base_records: list[dict[str, Any]] = []
            incomplete = False
            for name, expression in bases:
                remaining_ms = _remaining_timeout_ms(deadline)
                query = _run_smt_query(
                    model,
                    context,
                    expression,
                    timeout_ms=remaining_ms,
                )
                base_records.append({"name": name, **query})
                if query.get("solver_status") == "sat":
                    if query.get("exact_replay") is not True:
                        return {
                            "outcome": "DEFERRED",
                            "reason_code": "COUNTEREXAMPLE_REPLAY_FAILED",
                            "detail": "the SMT reachability trace did not replay exactly",
                            "applicability_checks": {"accepted": True},
                            "proof": {
                                "rule": "smt_finite_prefix_counterexample_v1",
                                "case_expression_sha256": source_hash,
                                "reachability_expression_sha256": reachability_hash,
                                "prefix_depth": depth,
                                "trace_query": {"name": name, **query},
                            },
                        }
                    return {
                        "outcome": "VIOLATION",
                        "reason_code": "COUNTEREXAMPLE_REPLAYED",
                        "detail": (
                            "an exact SMT trace reaches the unsafe case from the "
                            "SysML initial state"
                        ),
                        "applicability_checks": {
                            "accepted": True,
                            "case_id": reduced_case.case_id,
                            "complete_initialization": True,
                            "complete_transition_encoding": True,
                        },
                        "proof": {
                            "rule": "smt_finite_prefix_counterexample_v1",
                            "case_id": reduced_case.case_id,
                            "case_expression_sha256": source_hash,
                            "reachability_expression_sha256": reachability_hash,
                            "prefix_depth": depth,
                            "trace_query": {"name": name, **query},
                        },
                    }
                if query.get("solver_status") != "unsat":
                    incomplete = True
                    break
            induction_record: dict[str, Any] | None = None
            if not incomplete:
                remaining_ms = _remaining_timeout_ms(deadline)
                induction_record = _run_smt_query(
                    model,
                    context,
                    induction,
                    timeout_ms=remaining_ms,
                )
            proved = bool(
                not incomplete
                and induction_record is not None
                and induction_record.get("solver_status") == "unsat"
            )
            depth_attempts.append({
                "depth": depth,
                "base_queries": base_records,
                "induction_query": induction_record,
                "proved": proved,
            })
            if proved:
                return {
                    "outcome": "CERTIFIED",
                    "reason_code": "",
                    "detail": (
                        f"exact SMT queries prove the unsafe case unreachable by "
                        f"{depth}-step induction from the SysML initial state"
                    ),
                    "applicability_checks": {
                        "accepted": True,
                        "case_id": reduced_case.case_id,
                        "induction_depth": depth,
                        "complete_initialization": True,
                        "complete_transition_encoding": True,
                    },
                    "proof": {
                        "rule": "smt_finite_prefix_and_inductive_exclusion_v1",
                        "case_id": reduced_case.case_id,
                        "case_expression_sha256": source_hash,
                        "reachability_expression_sha256": reachability_hash,
                        "depth_attempts": depth_attempts,
                    },
                }
    except ProofDeferred as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": exc.reason_code,
            "detail": exc.detail,
            "applicability_checks": {
                "accepted": True,
                "case_id": reduced_case.case_id,
                "complete_initialization": True,
                "complete_transition_encoding": True,
            },
            "proof": {
                "rule": "smt_finite_prefix_and_inductive_exclusion_v1",
                "case_id": reduced_case.case_id,
                "case_expression_sha256": source_hash,
                "reachability_expression_sha256": reachability_hash,
                "depth_attempts": depth_attempts,
            },
        }
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": str(exc),
            "applicability_checks": {"accepted": False},
            "proof": {
                "rule": "smt_finite_prefix_and_inductive_exclusion_v1",
                "case_id": reduced_case.case_id,
                "case_expression_sha256": source_hash,
                "reachability_expression_sha256": reachability_hash,
                "depth_attempts": depth_attempts,
            },
        }

    last_query = next(
        (
            attempt.get("induction_query")
            for attempt in reversed(depth_attempts)
            if isinstance(attempt.get("induction_query"), dict)
        ),
        None,
    ) or {}
    return {
        "outcome": "DEFERRED",
        "reason_code": str(
            last_query.get("reason_code") or "REACHABILITY_BOUND_INCONCLUSIVE"
        ),
        "detail": str(
            last_query.get("detail")
            or "the exact SMT induction query remains feasible"
        ),
        "applicability_checks": {
            "accepted": True,
            "case_id": reduced_case.case_id,
            "complete_initialization": True,
            "complete_transition_encoding": True,
        },
        "proof": {
            "rule": "smt_finite_prefix_and_inductive_exclusion_v1",
            "case_id": reduced_case.case_id,
            "case_expression_sha256": source_hash,
            "reachability_expression_sha256": reachability_hash,
            "depth_attempts": depth_attempts,
        },
    }


def run_smt_reachability_checker(
    model: EquationModel,
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    *,
    timeout_ms: int,
) -> dict[str, Any]:
    """Run SMT reachability within one complete wall clock timeout."""

    try:
        with _hard_timeout(timeout_ms):
            return _run_smt_reachability_checker(
                model,
                reduced_case,
                context,
                timeout_ms=timeout_ms,
            )
    except ProofDeferred as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": exc.reason_code,
            "detail": exc.detail,
            "applicability_checks": {
                "accepted": True,
                "case_id": reduced_case.case_id,
            },
        }


def _attempt_query(
    expression: Expr,
    boolean_variables: set[str],
    prefix: str,
    checker: Callable[..., dict[str, Any]],
    timeout_ms: int,
    context: ReachabilityContext,
    *,
    deadline: float,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
    _remaining_timeout_ms(deadline)
    source_case = ReducedCase(
        case_id=prefix + ".root",
        expression=expression,
        parent_hash=expression_hash(expression),
        boolean_assignment=(),
        time_reduction="lazy_factored_reachability",
        obligation="reachability",
        reachability_expression=expression,
        factored=True,
    )
    factored_timeout_ms = min(
        timeout_ms,
        _remaining_timeout_ms(deadline),
    )
    attempt = run_lazy_factored_checker(
        source_case,
        boolean_variables,
        "reachability_arithmetic",
        lambda leaf, remaining: checker(
            leaf,
            set(),
            timeout_ms=min(timeout_ms, remaining),
        ),
        lambda item: (
            isinstance(item, Op)
            and item.op in {"==", ">", "<", ">=", "<="}
        ),
        timeout_ms=factored_timeout_ms,
        context_sha256=shared_reachability_context_sha256(context),
        total_timeout_ms=factored_timeout_ms,
    )
    record = {
        "case_id": source_case.case_id,
        "expression": expr_to_dict(expression),
        "expression_sha256": expression_hash(expression),
        "attempt": attempt,
    }
    if attempt.get("outcome") == "CERTIFIED":
        return True, [record], None
    return False, [record], attempt


def _run_reachability_checker(
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    *,
    method: str,
    timeout_ms: int,
) -> dict[str, Any]:
    """Prove an unsafe case unreachable with checked finite-prefix induction."""

    if reduced_case.obligation != "physical_interval":
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "reachability is only required for physical interval cases",
            "applicability_checks": {"accepted": False},
        }
    if method == "linear":
        def checker(
            arithmetic_case: ReducedCase,
            boolean_variables: set[str],
            *,
            timeout_ms: int,
        ) -> dict[str, Any]:
            if expression_is_linear(
                arithmetic_case.expression,
                boolean_variables,
            )[0]:
                linear = run_linear_checker(
                    arithmetic_case,
                    boolean_variables,
                    timeout_ms=timeout_ms,
                )
                if linear.get("outcome") in {"CERTIFIED", "VIOLATION"}:
                    linear.setdefault("applicability_checks", {})[
                        "reachability_submethod"
                    ] = "linear"
                    return linear
            envelope = run_linear_envelope_checker(
                arithmetic_case,
                timeout_ms=timeout_ms,
            )
            envelope.setdefault("applicability_checks", {})[
                "reachability_submethod"
            ] = "bounded_product_linear_envelope"
            return envelope
    elif method == "convex":
        def checker(
            arithmetic_case: ReducedCase,
            boolean_variables: set[str],
            *,
            timeout_ms: int,
        ) -> dict[str, Any]:
            if expression_is_linear(
                arithmetic_case.expression,
                boolean_variables,
            )[0]:
                linear = run_linear_checker(
                    arithmetic_case,
                    boolean_variables,
                    timeout_ms=timeout_ms,
                )
                if linear.get("outcome") == "CERTIFIED":
                    linear.setdefault("applicability_checks", {})[
                        "reachability_submethod"
                    ] = "linear"
                    return linear
            try:
                quadratic_constraints(
                    arithmetic_case.expression,
                    boolean_variables,
                )
            except ProofDeferred:
                convex = None
            else:
                convex = run_convex_checker(
                    arithmetic_case,
                    boolean_variables,
                    timeout_ms=timeout_ms,
                )
                convex.setdefault("applicability_checks", {})[
                    "reachability_submethod"
                ] = "convex"
                if convex.get("outcome") == "CERTIFIED":
                    return convex
            envelope = run_convex_envelope_checker(
                arithmetic_case,
                timeout_ms=timeout_ms,
            )
            envelope.setdefault("applicability_checks", {})[
                "reachability_submethod"
            ] = "convex_square_envelope"
            return envelope
    else:
        return {
            "outcome": "DEFERRED",
            "reason_code": "MALFORMED_OUTPUT",
            "detail": f"unknown reachability method {method}",
            "applicability_checks": {"accepted": False},
        }

    depth_records: list[dict[str, Any]] = []
    deadline = monotonic() + (timeout_ms / 1000.0)
    for depth in count(1):
        try:
            _remaining_timeout_ms(deadline)
            bases, induction, boolean_variables = _query_set(
                reduced_case,
                context,
                depth,
                deadline=deadline,
            )
            base_records: list[dict[str, Any]] = []
            failed: dict[str, Any] | None = None
            for name, expression in bases:
                proved, records, failed = _attempt_query(
                    expression,
                    boolean_variables,
                    f"{reduced_case.case_id}.reach.depth{depth}.{name}",
                    checker,
                    timeout_ms,
                    context,
                    deadline=deadline,
                )
                base_records.extend(records)
                if not proved:
                    if (
                        failed is not None
                        and failed.get("outcome") == "VIOLATION"
                        and set(context.post_dict()).issubset(
                            set(context.initial_variables)
                        )
                    ):
                        return {
                            "outcome": "VIOLATION",
                            "reason_code": "COUNTEREXAMPLE_REPLAYED",
                            "detail": (
                                f"an exact unsafe trace from the SysML initial "
                                f"state was replayed at prefix step {len(base_records) - 1}"
                            ),
                            "applicability_checks": {
                                "accepted": True,
                                "method": method,
                                "case_id": reduced_case.case_id,
                                "complete_initialization": True,
                                "complete_transition_mode_coverage": True,
                            },
                            "proof": {
                                "rule": "exact_finite_prefix_counterexample_v1",
                                "method": method,
                                "case_id": reduced_case.case_id,
                                "case_expression_sha256": expression_hash(
                                    reduced_case.expression
                                ),
                                "reachability_expression_sha256": expression_hash(
                                    reduced_case.reachability_expression
                                    or reduced_case.expression
                                ),
                                "prefix_depth": depth,
                                "base_query": name,
                                "base_obligations": base_records,
                                "counterexample": (
                                    failed.get("proof") or {}
                                ).get("counterexample", {}),
                            },
                        }
                    break
            induction_records: list[dict[str, Any]] = []
            if failed is None:
                proved, induction_records, failed = _attempt_query(
                    induction,
                    boolean_variables,
                    f"{reduced_case.case_id}.reach.depth{depth}.induction",
                    checker,
                    timeout_ms,
                    context,
                    deadline=deadline,
                )
            else:
                proved = False
            depth_record = {
                "depth": depth,
                "base_query_count": len(bases),
                "base_obligations": base_records,
                "induction_obligations": induction_records,
                "proved": bool(proved and failed is None),
            }
            depth_records.append(depth_record)
            if depth_record["proved"]:
                return {
                    "outcome": "CERTIFIED",
                    "reason_code": "",
                    "detail": (
                        f"the unsafe case is excluded by checked {depth}-step "
                        f"{method} induction from the SysML initial state"
                    ),
                    "applicability_checks": {
                        "accepted": True,
                        "method": method,
                        "induction_depth": depth,
                        "case_id": reduced_case.case_id,
                        "complete_initial_prefix": True,
                        "complete_transition_mode_coverage": True,
                        "symbolic_action_modes": True,
                        "eager_action_mode_product": False,
                    },
                    "proof": {
                        "rule": "finite_prefix_and_inductive_case_exclusion_v1",
                        "method": method,
                        "case_id": reduced_case.case_id,
                        "case_expression_sha256": expression_hash(
                            reduced_case.expression
                        ),
                        "reachability_expression_sha256": expression_hash(
                            reduced_case.reachability_expression
                            or reduced_case.expression
                        ),
                        "depth_attempts": depth_records,
                    },
                }
        except ProofDeferred as exc:
            depth_records.append({
                "depth": depth,
                "proved": False,
                "reason_code": exc.reason_code,
                "detail": exc.detail,
            })
            if exc.reason_code in {
                "NOT_LINEAR",
                "NOT_CONVEX",
                "UNSUPPORTED_EXPRESSION",
                "INCOMPLETE_CASE_COVERAGE",
                "TIMEOUT",
            }:
                break
        except Exception as exc:  # pragma: no cover - fail-closed boundary
            depth_records.append({
                "depth": depth,
                "proved": False,
                "reason_code": "MALFORMED_OUTPUT",
                "detail": str(exc),
            })
            break

    last_failure = next(
        (
            record.get("attempt")
            for depth_record in reversed(depth_records)
            for record in reversed(
                depth_record.get("induction_obligations", [])
                + depth_record.get("base_obligations", [])
            )
            if record.get("attempt", {}).get("outcome") != "CERTIFIED"
        ),
        None,
    )
    return {
        "outcome": "DEFERRED",
        "reason_code": str(
            (last_failure or {}).get("reason_code")
            or depth_records[-1].get("reason_code")
            or "REACHABILITY_BOUND_INCONCLUSIVE"
        ),
        "detail": str(
            (last_failure or {}).get("detail")
            or depth_records[-1].get("detail")
            or "the checked reachable-state overapproximation intersects the unsafe case"
        ),
        "applicability_checks": {
            "accepted": True,
            "method": method,
            "case_id": reduced_case.case_id,
            "complete_initial_prefix": True,
            "complete_transition_mode_coverage": True,
            "symbolic_action_modes": True,
            "eager_action_mode_product": False,
        },
        "proof": {
            "rule": "finite_prefix_and_inductive_case_exclusion_v1",
            "method": method,
            "case_id": reduced_case.case_id,
            "case_expression_sha256": expression_hash(reduced_case.expression),
            "reachability_expression_sha256": expression_hash(
                reduced_case.reachability_expression or reduced_case.expression
            ),
            "depth_attempts": depth_records,
        },
    }


def run_reachability_checker(
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    *,
    method: str,
    timeout_ms: int,
) -> dict[str, Any]:
    """Run optimized reachability within one complete wall clock timeout."""

    try:
        with _hard_timeout(timeout_ms):
            return _run_reachability_checker(
                reduced_case,
                context,
                method=method,
                timeout_ms=timeout_ms,
            )
    except ProofDeferred as exc:
        return {
            "outcome": "DEFERRED",
            "reason_code": exc.reason_code,
            "detail": exc.detail,
            "applicability_checks": {
                "accepted": True,
                "method": method,
                "case_id": reduced_case.case_id,
            },
        }


def _relational_names(
    context: ReachabilityContext,
) -> tuple[set[str], set[str]]:
    states = set(context.post_dict()) | set(context.initial_variables)
    return states, set(context.action_variables)


def _relational_lift(
    expression: Expr,
    context: ReachabilityContext,
    frame: int,
) -> Expr:
    states, actions = _relational_names(context)
    mapping = {
        name: Var(f"rel_state_{frame}__{name}")
        for name in states
    }
    mapping.update({
        name: Var(f"rel_action_{frame}__{name}")
        for name in actions
    })
    return simplify(substitute(expression, mapping))


def _relational_transition(
    context: ReachabilityContext,
) -> Expr:
    states, _actions = _relational_names(context)
    post = context.post_dict()
    equations = []
    for name in sorted(states):
        next_value = Var(f"rel_state_1__{name}")
        source = post.get(name, Var(name))
        equations.append(Op("==", (
            next_value,
            _relational_lift(source, context, 0),
        )))
    return _and(equations)


def _comparison_atoms(expression: Expr) -> list[Expr]:
    atoms: list[Expr] = []
    if isinstance(expression, Op):
        if expression.op in {"<", "<=", ">", ">=", "=="}:
            atoms.append(expression)
        for argument in expression.args:
            atoms.extend(_comparison_atoms(argument))
    elif isinstance(expression, Ite):
        atoms.extend(_comparison_atoms(expression.cond))
        atoms.extend(_comparison_atoms(expression.then_expr))
        atoms.extend(_comparison_atoms(expression.else_expr))
    return atoms


def _numeric_value(expression: Expr) -> Fraction | None:
    if not isinstance(expression, Const) or isinstance(expression.value, bool):
        return None
    try:
        return Fraction(str(expression.value))
    except (ValueError, ZeroDivisionError):
        return None


def _numeric_expression(value: Fraction) -> Const:
    if value.denominator == 1:
        return Const(value.numerator)
    return Const(f"{value.numerator}/{value.denominator}")


def _numeric_constants(expression: Expr) -> set[Fraction]:
    if isinstance(expression, Const):
        value = _numeric_value(expression)
        return set() if value is None else {abs(value)}
    if isinstance(expression, Op):
        values: set[Fraction] = set()
        for argument in expression.args:
            values.update(_numeric_constants(argument))
        return values
    if isinstance(expression, Ite):
        return (
            _numeric_constants(expression.cond)
            | _numeric_constants(expression.then_expr)
            | _numeric_constants(expression.else_expr)
        )
    return set()


def _initial_numeric_values(
    context: ReachabilityContext,
) -> dict[str, Fraction]:
    values: dict[str, Fraction] = {}
    for expression in context.initial_constraints:
        if not isinstance(expression, Op) or expression.op != "==":
            continue
        left, right = expression.args
        right_value = _numeric_value(right)
        left_value = _numeric_value(left)
        if isinstance(left, Var) and right_value is not None:
            values[left.name] = right_value
        elif isinstance(right, Var) and left_value is not None:
            values[right.name] = left_value
    return values


def _clock_steps(context: ReachabilityContext) -> dict[str, Fraction]:
    clocks: dict[str, Fraction] = {}
    for name, expression in context.post_values:
        if not isinstance(expression, Op) or expression.op != "+":
            continue
        arguments = list(expression.args)
        variable_count = sum(
            isinstance(argument, Var) and argument.name == name
            for argument in arguments
        )
        constants = [
            value
            for argument in arguments
            if (value := _numeric_value(argument)) is not None
        ]
        if variable_count == 1 and len(constants) == 1 and constants[0] > 0:
            clocks[name] = constants[0]
    return clocks


def _raw_lower_bounds(
    context: ReachabilityContext,
) -> dict[str, Fraction]:
    bounds: dict[str, Fraction] = {}
    for expression in context.initial_constraints:
        if not isinstance(expression, Op) or expression.op not in {">=", "<="}:
            continue
        left, right = expression.args
        if expression.op == ">=" and isinstance(left, RawRef):
            value = _numeric_value(right)
            if value is not None:
                bounds[left.path] = max(bounds.get(left.path, value), value)
        elif expression.op == "<=" and isinstance(right, RawRef):
            value = _numeric_value(left)
            if value is not None:
                bounds[right.path] = max(bounds.get(right.path, value), value)
    return bounds


def _rate_candidates(expression: Expr) -> list[Fraction]:
    constants = sorted(value for value in _numeric_constants(expression) if value)
    rates: set[Fraction] = {Fraction(1)}
    for numerator in constants:
        for denominator in constants:
            rate = numerator / denominator
            if Fraction(1) <= rate <= Fraction(10):
                rates.add(rate)
                rates.add(Fraction(rate.numerator // rate.denominator))
    return sorted(rate for rate in rates if rate > 0)


def _dynamical_candidates(
    context: ReachabilityContext,
) -> list[Expr]:
    post = context.post_dict()
    states, _actions = _relational_names(context)
    initial = _initial_numeric_values(context)
    clocks = _clock_steps(context)
    raw_lower = _raw_lower_bounds(context)
    candidates: list[Expr] = []
    for dependent, dependent_post in sorted(post.items()):
        if dependent not in initial:
            continue
        raw_references = _raw_reference_names(dependent_post)
        bounded_references = sorted(raw_references & set(raw_lower))
        if not bounded_references:
            continue
        source_states = sorted(
            (expression_symbols(dependent_post) & states) - {dependent}
        )
        for source in source_states:
            if source not in initial or source not in post:
                continue
            rates = _rate_candidates(post[source])
            if not rates:
                continue
            for clock, step in sorted(clocks.items()):
                if clock not in initial:
                    continue
                clock_delta = Op("-", (
                    Var(clock),
                    _numeric_expression(initial[clock]),
                ))
                source_offset = _numeric_expression(initial[source])
                dependent_offset = _numeric_expression(initial[dependent])
                for name in (clock, source, dependent):
                    value = _numeric_expression(initial[name])
                    candidates.extend([
                        Op(">=", (Var(name), value)),
                        Op("<=", (Var(name), value)),
                    ])
                for upper in range(1, 21):
                    candidates.append(Op("<=", (
                        Var(clock),
                        _numeric_expression(initial[clock] + upper),
                    )))
                for rate in rates:
                    change = Op("*", (
                        _numeric_expression(rate),
                        clock_delta,
                    ))
                    line = Op("+", (source_offset, change))
                    candidates.extend([
                        Op("<=", (Var(source), line)),
                        Op(">=", (Var(source), line)),
                    ])
                    for reference in bounded_references:
                        lower = raw_lower[reference]
                        linear = lower + rate * step / 2
                        curve = Op("+", (
                            dependent_offset,
                            Op("*", (
                                _numeric_expression(linear),
                                clock_delta,
                            )),
                            Op("*", (
                                _numeric_expression(-rate / 2),
                                Op("*", (clock_delta, clock_delta)),
                            )),
                        ))
                        candidates.append(Op(">=", (
                            Var(dependent),
                            curve,
                        )))
    return candidates


def _candidate_invariants(
    context: ReachabilityContext,
) -> list[Expr]:
    states, actions = _relational_names(context)
    known_initial: dict[str, Expr] = {}
    for expression in context.initial_constraints:
        if not isinstance(expression, Op) or expression.op != "==":
            continue
        left, right = expression.args
        if isinstance(left, Var) and isinstance(right, Const):
            known_initial[left.name] = right
        elif isinstance(right, Var) and isinstance(left, Const):
            known_initial[right.name] = left

    candidates: dict[str, Expr] = {}
    for expression in context.initial_constraints:
        projected = simplify(substitute(expression, known_initial))
        symbols = expression_symbols(projected)
        if symbols & (states | actions):
            continue
        if isinstance(projected, Const) and projected.value is True:
            continue
        candidates[expression_hash(projected)] = projected

    boolean_variables = set(context.boolean_variables)
    for atom in _comparison_atoms(context.domain):
        if len(atom.args) != 2:
            continue
        symbols = expression_symbols(atom)
        if not (symbols & states) or symbols & actions:
            continue
        if symbols & boolean_variables:
            continue
        operation = {"<": "<=", ">": ">="}.get(atom.op, atom.op)
        candidate = simplify(Op(operation, atom.args))
        candidates[expression_hash(candidate)] = candidate
    for candidate in _dynamical_candidates(context):
        candidate = simplify(candidate)
        candidates[expression_hash(candidate)] = candidate
    return [candidates[key] for key in sorted(candidates)]


def _relational_query(
    expressions: list[Expr] | tuple[Expr, ...],
) -> Expr:
    return _and(list(expressions))


def _conjunct_hashes(expression: Expr) -> set[str]:
    if isinstance(expression, Op) and expression.op == "and":
        hashes: set[str] = set()
        for argument in expression.args:
            hashes.update(_conjunct_hashes(argument))
        return hashes
    return {expression_hash(expression)}


def _shared_context_record(context: ReachabilityContext) -> dict[str, Any]:
    return {
        "domain": expr_to_dict(context.domain),
        "initial_constraints": [
            expr_to_dict(item)
            for item in sorted(context.initial_constraints, key=expression_hash)
        ],
        "initial_variables": sorted(context.initial_variables),
        "post_values": {
            name: expr_to_dict(expression)
            for name, expression in sorted(context.post_values)
        },
        "action_variables": sorted(context.action_variables),
        "boolean_variables": sorted(context.boolean_variables),
        "integer_variables": sorted(context.integer_variables),
    }


def shared_reachability_context_sha256(
    context: ReachabilityContext,
) -> str:
    payload = json.dumps(
        _shared_context_record(context),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _batched_candidate_filter(
    candidates: list[Expr],
    antecedent: Expr,
    context: ReachabilityContext,
    frame: int,
    run_query: Callable[[Expr], dict[str, Any]],
) -> tuple[list[Expr], list[Expr]]:
    proved: list[Expr] = []
    rejected: list[Expr] = []

    def check(group: list[Expr]) -> None:
        remaining = list(group)
        while remaining:
            lifted = [
                _relational_lift(item, context, frame) for item in remaining
            ]
            consequent = _relational_query(lifted)
            query = run_query(_relational_query([
                antecedent,
                Op("not", (consequent,)),
            ]))
            if query.get("solver_status") == "unsat":
                proved.extend(remaining)
                return
            failed: list[Expr] = []
            if query.get("solver_status") == "sat" and isinstance(
                query.get("exact_values"), dict
            ):
                for candidate, lifted_candidate in zip(remaining, lifted):
                    try:
                        holds = replay_serialized_boolean_expression(
                            expr_to_dict(lifted_candidate),
                            query["exact_values"],
                        )
                    except (TypeError, ValueError, ZeroDivisionError):
                        holds = None
                    if holds is False:
                        failed.append(candidate)
            if failed:
                failed_set = set(failed)
                rejected.extend(failed)
                remaining = [
                    item for item in remaining if item not in failed_set
                ]
                continue
            if len(remaining) == 1:
                rejected.extend(remaining)
                return
            middle = len(remaining) // 2
            check(remaining[:middle])
            check(remaining[middle:])
            return

    check(candidates)
    return proved, rejected


def _batch_proof_record(
    candidates: list[Expr],
    query: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidates": [expr_to_dict(item) for item in candidates],
        "candidate_sha256": [expression_hash(item) for item in candidates],
        "query": query,
    }


def _prepare_shared_relational_region(
    model: EquationModel,
    context: ReachabilityContext,
    *,
    timeout_ms: int,
) -> _SharedRelationalRegion:
    deadline = monotonic() + (timeout_ms / 1000.0)
    context_record = _shared_context_record(context)
    context_sha256 = shared_reachability_context_sha256(context)
    initial = _relational_query([
        _relational_lift(item, context, 0)
        for item in context.initial_constraints
    ])
    domain = _relational_lift(context.domain, context, 0)
    transition = _relational_transition(context)
    candidates = _candidate_invariants(context)
    query_runner = _IncrementalSmtQueryRunner(
        model,
        context,
        _relational_query([
            initial,
            domain,
            transition,
            *[
                _relational_lift(item, context, frame)
                for frame in (0, 1)
                for item in candidates
            ],
        ]),
    )
    query_cache: dict[str, dict[str, Any]] = {}
    query_cache_hits = 0

    def run_query(expression: Expr) -> dict[str, Any]:
        nonlocal query_cache_hits
        key = expression_hash(expression)
        if key in query_cache:
            query_cache_hits += 1
            return query_cache[key]
        result = _run_smt_query(
            model,
            context,
            expression,
            timeout_ms=_remaining_timeout_ms(deadline),
            replay_sat_query=False,
            runner=query_runner,
        )
        query_cache[key] = result
        return result

    initiated, _rejected_initial = _batched_candidate_filter(
        candidates,
        initial,
        context,
        0,
        run_query,
    )

    active = list(initiated)
    changed = True
    while changed and active:
        current_invariant = _relational_query([
            _relational_lift(item, context, 0) for item in active
        ])
        preservation_antecedent = _relational_query([
            current_invariant,
            domain,
            transition,
        ])
        preserved, rejected = _batched_candidate_filter(
            active,
            preservation_antecedent,
            context,
            1,
            run_query,
        )
        changed = bool(rejected)
        active = preserved

    if not active:
        raise ProofDeferred(
            "INVARIANT_NOT_FOUND",
            "no generated state constraint was both initial and preserved",
        )

    preserved_candidate_count = len(active)
    implication_removals: list[dict[str, Any]] = []
    reduced = list(active)
    for candidate in list(active):
        remaining = [item for item in reduced if item != candidate]
        if not remaining:
            continue
        implication_query = _relational_query([
            *[_relational_lift(item, context, 0) for item in remaining],
            Op("not", (_relational_lift(candidate, context, 0),)),
        ])
        query = run_query(implication_query)
        if query.get("solver_status") != "unsat":
            continue
        reduced = remaining
        implication_removals.append({
            "removed_candidate": expr_to_dict(candidate),
            "removed_candidate_sha256": expression_hash(candidate),
            "remaining_invariant_sha256": [
                expression_hash(item) for item in remaining
            ],
            "query": query,
        })
    active = reduced

    current_invariant = _relational_query([
        _relational_lift(item, context, 0) for item in active
    ])
    initiation_expression = _relational_query([
        initial,
        Op("not", (_relational_query([
            _relational_lift(item, context, 0) for item in active
        ]),)),
    ])
    initiation_query = run_query(initiation_expression)
    if initiation_query.get("solver_status") != "unsat":
        raise ProofDeferred(
            str(initiation_query.get("reason_code") or "INVARIANT_NOT_INITIAL"),
            str(initiation_query.get("detail") or "the retained state constraints are not initial"),
        )

    preservation_expression = _relational_query([
        current_invariant,
        domain,
        transition,
        Op("not", (_relational_query([
            _relational_lift(item, context, 1) for item in active
        ]),)),
    ])
    preservation_query = run_query(preservation_expression)
    if preservation_query.get("solver_status") != "unsat":
        raise ProofDeferred(
            str(preservation_query.get("reason_code") or "INVARIANT_NOT_PRESERVED"),
            str(preservation_query.get("detail") or "the retained state constraints are not preserved"),
        )

    record = {
        "rule": "shared_relational_reachable_region_v2",
        "context": context_record,
        "context_sha256": context_sha256,
        "initial_expression": expr_to_dict(initial),
        "initial_expression_sha256": expression_hash(initial),
        "domain_expression": expr_to_dict(domain),
        "domain_expression_sha256": expression_hash(domain),
        "transition_expression": expr_to_dict(transition),
        "transition_expression_sha256": expression_hash(transition),
        "candidate_count": len(candidates),
        "initiated_candidate_count": len(initiated),
        "preserved_candidate_count": preserved_candidate_count,
        "invariant": [expr_to_dict(item) for item in active],
        "invariant_sha256": [expression_hash(item) for item in active],
        "implication_removals": implication_removals,
        "initiation_batches": [_batch_proof_record(active, initiation_query)],
        "preservation_batches": [_batch_proof_record(active, preservation_query)],
        "unique_query_count": len(query_cache),
        "query_cache_hits": query_cache_hits,
    }
    return _SharedRelationalRegion(
        context_sha256=context_sha256,
        invariant=tuple(active),
        record=record,
        query_runner=query_runner,
    )


def _run_relational_invariant_checker(
    model: EquationModel,
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    *,
    timeout_ms: int,
    cache: SharedReachabilityCache,
) -> dict[str, Any]:
    if reduced_case.obligation != "physical_interval":
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "the relational invariant applies only to physical interval cases",
            "applicability_checks": {"accepted": False},
        }
    if z3 is None:
        return {
            "outcome": "DEFERRED",
            "reason_code": "BLOCKED_INPUT",
            "detail": "z3-solver is unavailable",
            "applicability_checks": {"accepted": False},
        }

    deadline = monotonic() + (timeout_ms / 1000.0)
    source_hash = expression_hash(reduced_case.expression)
    reachability_hash = expression_hash(
        reduced_case.reachability_expression or reduced_case.expression
    )
    context_sha256 = shared_reachability_context_sha256(context)
    if context_sha256 in cache.region_failures:
        return cache.region_failures[context_sha256]
    shared = cache.regions.get(context_sha256)
    if shared is None:
        shared = _prepare_shared_relational_region(
            model,
            context,
            timeout_ms=_remaining_timeout_ms(deadline),
        )
        cache.regions[context_sha256] = shared
    domain = _relational_lift(context.domain, context, 0)
    current_invariant = _relational_query([
        _relational_lift(item, context, 0) for item in shared.invariant
    ])
    unsafe = reduced_case.reachability_expression or reduced_case.expression
    mode = _mode_condition(reduced_case, {
        name: f"rel_action_0__{name}"
        for name in context.action_variables
    })
    safety_expression = _relational_query([
        current_invariant,
        domain,
        mode,
        _relational_lift(unsafe, context, 0),
    ])
    safety_expression_sha256 = expression_hash(safety_expression)
    safety_key = hashlib.sha256(
        f"{context_sha256}:{safety_expression_sha256}".encode("utf-8")
    ).hexdigest()
    safety_record = cache.safety_queries.get(safety_key)
    reused_safety_query = safety_record is not None
    merge_rule = "identical_expression" if reused_safety_query else "new_query"
    if safety_record is None:
        current_conjuncts = _conjunct_hashes(safety_expression)
        for existing_key, existing_expression in cache.safety_expressions.items():
            existing_record = cache.safety_queries[existing_key]
            if existing_record.get("context_sha256") != context_sha256:
                continue
            if existing_record.get("query", {}).get("solver_status") != "unsat":
                continue
            if _conjunct_hashes(existing_expression) <= current_conjuncts:
                safety_key = existing_key
                safety_record = existing_record
                reused_safety_query = True
                merge_rule = "conjunct_containment"
                break
    if safety_record is None:
        safety_query = _run_smt_query(
            model,
            context,
            safety_expression,
            timeout_ms=_remaining_timeout_ms(deadline),
            runner=shared.query_runner,
        )
        safety_record = {
            "rule": "merged_relational_safety_query_v1",
            "context_sha256": context_sha256,
            "safety_expression": expr_to_dict(safety_expression),
            "safety_expression_sha256": safety_expression_sha256,
            "case_ids": [],
            "covered_case_expressions": [],
            "query": safety_query,
        }
        cache.safety_queries[safety_key] = safety_record
        cache.safety_expressions[safety_key] = safety_expression
    safety_record["case_ids"].append(reduced_case.case_id)
    safety_record["covered_case_expressions"].append({
        "case_id": reduced_case.case_id,
        "expression": expr_to_dict(safety_expression),
        "expression_sha256": safety_expression_sha256,
        "merge_rule": merge_rule,
    })
    safety_query = safety_record["query"]
    proof = {
        "rule": "shared_relational_inductive_invariant_v2",
        "case_id": reduced_case.case_id,
        "case_expression_sha256": source_hash,
        "reachability_expression_sha256": reachability_hash,
        "shared_reachable_region_sha256": context_sha256,
        "merged_safety_query_sha256": safety_key,
        "case_safety_expression_sha256": safety_expression_sha256,
        "merge_rule": merge_rule,
        "reused_safety_query": reused_safety_query,
    }
    if safety_query.get("solver_status") == "unsat":
        return {
            "outcome": "CERTIFIED",
            "reason_code": "",
            "detail": "a checked relational invariant excludes the unsafe physical interval",
            "applicability_checks": {
                "accepted": True,
                "case_id": reduced_case.case_id,
                "complete_initialization": True,
                "complete_transition_encoding": True,
                "candidate_count": shared.record["candidate_count"],
                "invariant_clause_count": len(shared.invariant),
                "shared_reachable_region": True,
                "merged_identical_case": reused_safety_query,
            },
            "proof": proof,
        }
    return {
        "outcome": "DEFERRED",
        "reason_code": str(safety_query.get("reason_code") or "INVARIANT_TOO_WEAK"),
        "detail": str(safety_query.get("detail") or "the preserved state constraint intersects the unsafe interval"),
        "applicability_checks": {
            "accepted": True,
            "case_id": reduced_case.case_id,
        },
        "proof": proof,
    }


def _run_relational_invariant_group(
    model: EquationModel,
    reduced_cases: list[ReducedCase],
    context: ReachabilityContext,
    *,
    timeout_ms: int,
    cache: SharedReachabilityCache,
) -> dict[str, dict[str, Any]]:
    if not reduced_cases:
        return {}
    if any(item.obligation != "physical_interval" for item in reduced_cases):
        return {
            item.case_id: {
                "outcome": "DEFERRED",
                "reason_code": "BLOCKED_INPUT",
                "detail": "grouped relational checks require physical interval cases",
                "applicability_checks": {"accepted": False},
            }
            for item in reduced_cases
        }
    if z3 is None:
        return {
            item.case_id: {
                "outcome": "DEFERRED",
                "reason_code": "BLOCKED_INPUT",
                "detail": "z3-solver is unavailable",
                "applicability_checks": {"accepted": False},
            }
            for item in reduced_cases
        }

    deadline = monotonic() + (timeout_ms / 1000.0)
    context_sha256 = shared_reachability_context_sha256(context)
    if context_sha256 in cache.region_failures:
        failure = cache.region_failures[context_sha256]
        return {item.case_id: dict(failure) for item in reduced_cases}
    shared = cache.regions.get(context_sha256)
    if shared is None:
        shared = _prepare_shared_relational_region(
            model,
            context,
            timeout_ms=_remaining_timeout_ms(deadline),
        )
        cache.regions[context_sha256] = shared

    domain = _relational_lift(context.domain, context, 0)
    current_invariant = _relational_query([
        _relational_lift(item, context, 0) for item in shared.invariant
    ])
    case_safety: list[tuple[ReducedCase, Expr]] = []
    for reduced_case in reduced_cases:
        unsafe = reduced_case.reachability_expression or reduced_case.expression
        mode = _mode_condition(reduced_case, {
            name: f"rel_action_0__{name}"
            for name in context.action_variables
        })
        case_safety.append((
            reduced_case,
            _relational_query([
                current_invariant,
                domain,
                mode,
                _relational_lift(unsafe, context, 0),
            ]),
        ))
    group_expression = simplify(Op(
        "or",
        tuple(expression for _case, expression in case_safety),
    ))
    group_expression_sha256 = expression_hash(group_expression)
    safety_key = hashlib.sha256(
        f"{context_sha256}:{group_expression_sha256}".encode("utf-8")
    ).hexdigest()
    safety_record = cache.safety_queries.get(safety_key)
    if safety_record is None:
        query = _run_smt_query(
            model,
            context,
            group_expression,
            timeout_ms=_remaining_timeout_ms(deadline),
            runner=shared.query_runner,
        )
        safety_record = {
            "rule": "grouped_relational_safety_query_v2",
            "context_sha256": context_sha256,
            "safety_expression": expr_to_dict(group_expression),
            "safety_expression_sha256": group_expression_sha256,
            "case_ids": [item.case_id for item, _expression in case_safety],
            "covered_case_expressions": [
                {
                    "case_id": item.case_id,
                    "expression": expr_to_dict(expression),
                    "expression_sha256": expression_hash(expression),
                    "merge_rule": "group_disjunction",
                }
                for item, expression in case_safety
            ],
            "query": query,
        }
        cache.safety_queries[safety_key] = safety_record
        cache.safety_expressions[safety_key] = group_expression
    query = safety_record["query"]
    certified = query.get("solver_status") == "unsat"
    attempts: dict[str, dict[str, Any]] = {}
    for reduced_case, safety_expression in case_safety:
        proof = {
            "rule": "shared_relational_inductive_invariant_v2",
            "case_id": reduced_case.case_id,
            "case_expression_sha256": expression_hash(reduced_case.expression),
            "reachability_expression_sha256": expression_hash(
                reduced_case.reachability_expression or reduced_case.expression
            ),
            "shared_reachable_region_sha256": context_sha256,
            "merged_safety_query_sha256": safety_key,
            "case_safety_expression_sha256": expression_hash(safety_expression),
            "merge_rule": "group_disjunction",
            "reused_safety_query": len(reduced_cases) > 1,
        }
        attempts[reduced_case.case_id] = {
            "outcome": "CERTIFIED" if certified else "DEFERRED",
            "reason_code": "" if certified else str(
                query.get("reason_code") or "GROUP_SAFETY_INCONCLUSIVE"
            ),
            "detail": (
                f"one checked relational query excludes {len(reduced_cases)} unsafe cases"
                if certified
                else "the grouped unsafe cases require automatic subdivision"
            ),
            "applicability_checks": {
                "accepted": True,
                "case_id": reduced_case.case_id,
                "group_case_count": len(reduced_cases),
                "complete_initialization": True,
                "complete_transition_encoding": True,
                "candidate_count": shared.record["candidate_count"],
                "invariant_clause_count": len(shared.invariant),
                "shared_reachable_region": True,
            },
            "proof": proof,
        }
    return attempts


def run_relational_invariant_group_checker(
    model: EquationModel,
    reduced_cases: list[ReducedCase],
    context: ReachabilityContext,
    *,
    timeout_ms: int,
    cache: SharedReachabilityCache,
) -> dict[str, dict[str, Any]]:
    """Certify groups and divide only groups whose combined query is inconclusive."""

    results: dict[str, dict[str, Any]] = {}

    def check(group: list[ReducedCase]) -> None:
        try:
            with _hard_timeout(timeout_ms):
                attempts = _run_relational_invariant_group(
                    model,
                    group,
                    context,
                    timeout_ms=timeout_ms,
                    cache=cache,
                )
        except ProofDeferred as exc:
            attempts = {
                item.case_id: {
                    "outcome": "DEFERRED",
                    "reason_code": exc.reason_code,
                    "detail": exc.detail,
                    "applicability_checks": {
                        "accepted": True,
                        "case_id": item.case_id,
                    },
                }
                for item in group
            }
            context_sha256 = shared_reachability_context_sha256(context)
            cache.region_failures.setdefault(
                context_sha256,
                next(iter(attempts.values())),
            )
        if attempts and all(
            item.get("outcome") == "CERTIFIED" for item in attempts.values()
        ):
            results.update(attempts)
            return
        if len(group) == 1:
            results.update(attempts)
            return
        middle = len(group) // 2
        check(group[:middle])
        check(group[middle:])

    check(list(reduced_cases))
    return results


def run_relational_invariant_checker(
    model: EquationModel,
    reduced_case: ReducedCase,
    context: ReachabilityContext,
    *,
    timeout_ms: int,
    cache: SharedReachabilityCache | None = None,
) -> dict[str, Any]:
    """Prove safety from a compact current-to-next-state invariant."""

    shared_cache = cache if cache is not None else SharedReachabilityCache()
    context_sha256 = shared_reachability_context_sha256(context)
    try:
        with _hard_timeout(timeout_ms):
            return _run_relational_invariant_checker(
                model,
                reduced_case,
                context,
                timeout_ms=timeout_ms,
                cache=shared_cache,
            )
    except ProofDeferred as exc:
        failure = {
            "outcome": "DEFERRED",
            "reason_code": exc.reason_code,
            "detail": exc.detail,
            "applicability_checks": {
                "accepted": True,
                "case_id": reduced_case.case_id,
            },
        }
        shared_cache.region_failures.setdefault(context_sha256, failure)
        return failure
