"""Build deterministic discretization-safety proof obligations and results."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from certification.equations import Const, EquationModel, Expr, Ite, Op, RawRef, Var
from certification.relevance import equation_refs
from certification.solver import Encoder, infer_sorts
from certification.strict_extract import CertificationExtractor
from sysml_parser import IfStmt, PerformStmt, SubactionCallStmt

from .certificates.replay import replay_serialized_boolean_expression, serialize_exact_value
from .checkers.convex import run_convex_checker
from .checkers.convex_envelope import run_convex_envelope_checker
from .checkers.factored import run_lazy_factored_checker
from .checkers.linear import run_linear_checker
from .checkers.linear_envelope import run_linear_envelope_checker
from .checkers.reachability import (
    SharedReachabilityCache,
    run_reachability_checker,
    run_relational_invariant_group_checker,
    run_smt_reachability_checker,
    shared_reachability_context_sha256,
)
from .model.optimization import quadratic_constraints
from .model.proof_rules import (
    ProofDeferred,
    expand_definitions,
    expression_is_linear,
    expression_symbols,
    prove_implication_exact,
    substitute,
)
from .model.reduction import (
    ReducedCase,
    build_reduction,
    expression_hash,
    expr_to_dict,
)

try:  # pragma: no cover - integration environment determines availability
    import z3  # type: ignore
except Exception:  # pragma: no cover
    z3 = None


CHECKER_ORDER = [
    "linear",
    "convex",
    "reachability_linear",
    "reachability_convex",
    "exact_symbolic",
    "smt_fallback",
    "relational_invariant",
    "smt_reachability",
]


def canonical_dt(text: str | float) -> dict[str, Any]:
    raw = str(text).strip()
    value = Fraction(raw)
    if value <= 0:
        raise ValueError("dt must be positive")
    return {
        "input": raw,
        "numerator": value.numerator,
        "denominator": value.denominator,
        "canonical": f"{value.numerator}/{value.denominator}",
        "decimal": float(value),
    }


def _stage(
    checker: str,
    outcome: str,
    *,
    reason_code: str = "",
    detail: str = "",
    applicability_checks: dict[str, Any] | None = None,
    proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checker": checker,
        "checker_version": 1,
        "outcome": outcome,
        "reason_code": reason_code,
        "detail": detail,
        "applicability_checks": applicability_checks or {},
        "proof": proof or {},
    }


def _attempt_stage(checker: str, attempt: dict[str, Any]) -> dict[str, Any]:
    return _stage(
        checker,
        str(attempt.get("outcome", "DEFERRED")),
        reason_code=str(attempt.get("reason_code", "")),
        detail=str(attempt.get("detail", "")),
        applicability_checks=attempt.get("applicability_checks") or {},
        proof=attempt.get("proof") or {},
    )


def _validated_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    if attempt.get("outcome") in {"CERTIFIED", "VIOLATION", "DEFERRED"}:
        return attempt
    return {
        "outcome": "DEFERRED",
        "reason_code": "MALFORMED_OUTPUT",
        "detail": "checker returned an invalid outcome",
        "applicability_checks": attempt.get("applicability_checks") or {},
    }


def _case_conjuncts(expression: Expr) -> list[Expr]:
    if isinstance(expression, Op) and expression.op == "and":
        result: list[Expr] = []
        for argument in expression.args:
            result.extend(_case_conjuncts(argument))
        return result
    return [expression]


def _case_conjunction(expressions: list[Expr]) -> Expr:
    if not expressions:
        return Const(True)
    if len(expressions) == 1:
        return expressions[0]
    return Op("and", tuple(expressions))


def _expression_is_convex(expression: Expr) -> bool:
    try:
        quadratic_constraints(expression, set())
        return True
    except ProofDeferred:
        return False


def _deduplicate_proof_certificates(
    properties: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    certificates: dict[str, dict[str, Any]] = {}

    def digest(value: dict[str, Any]) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    key == "certificate"
                    and isinstance(item, dict)
                    and item.get("kind") in {
                        "linear_infeasibility_weights_v1",
                        "convex_dual_bound_v1",
                    }
                ):
                    key_hash = digest(item)
                    counts[key_hash] = counts.get(key_hash, 0) + 1
                    certificates[key_hash] = item
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(properties)
    repeated = {
        key: certificates[key]
        for key in sorted(certificates)
        if counts[key] > 1
    }

    def replace(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "certificate" and isinstance(item, dict):
                    key_hash = digest(item)
                    if key_hash in repeated:
                        value[key] = {
                            "shared_certificate_sha256": key_hash,
                        }
                        continue
                replace(item)
        elif isinstance(value, list):
            for item in value:
                replace(item)

    replace(properties)
    return {
        "rule": "content_addressed_proof_certificate_pool_v1",
        "certificates": repeated,
        "reference_count": sum(
            count for key, count in counts.items() if key in repeated
        ),
    }


def _grouped_case_attempts(
    reduced_cases: list[ReducedCase],
    checker_name: str,
    checker: Callable[[ReducedCase], dict[str, Any]],
    conjunct_filter: Callable[[Expr], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    conjunct_maps = {
        item.case_id: {
            expression_hash(expression): expression
            for expression in _case_conjuncts(item.expression)
            if conjunct_filter is None or conjunct_filter(expression)
        }
        for item in reduced_cases
    }
    results: dict[str, dict[str, Any]] = {}
    attempt_cache: dict[str, dict[str, Any]] = {}

    def attempt_for(expression: Expr, obligation: str) -> dict[str, Any]:
        expression_sha256 = expression_hash(expression)
        attempt = attempt_cache.get(expression_sha256)
        if attempt is None:
            group_case = ReducedCase(
                f"group.{checker_name}.{expression_sha256[:12]}",
                expression,
                expression_sha256,
                (),
                "common_constraint_group",
                obligation,
                expression,
            )
            attempt = _validated_attempt(checker(group_case))
            attempt_cache[expression_sha256] = attempt
        return attempt

    def minimize_certified_subset(
        expression: Expr,
        obligation: str,
        attempt: dict[str, Any],
    ) -> tuple[Expr, dict[str, Any], int]:
        selected = sorted(
            _case_conjuncts(expression),
            key=expression_hash,
        )
        checks = 0
        if checker_name == "linear":
            certificate = (attempt.get("proof") or {}).get("certificate") or {}
            multipliers = certificate.get("multipliers")
            if isinstance(multipliers, list) and len(multipliers) == len(selected):
                try:
                    supported = [
                        item
                        for item, multiplier in zip(selected, multipliers)
                        if Fraction(str(multiplier)) != 0
                    ]
                except (TypeError, ValueError, ZeroDivisionError):
                    supported = []
                if 0 < len(supported) < len(selected):
                    supported_expression = _case_conjunction(supported)
                    supported_attempt = attempt_for(
                        supported_expression,
                        obligation,
                    )
                    checks += 1
                    if supported_attempt.get("outcome") == "CERTIFIED":
                        return supported_expression, supported_attempt, checks
        for candidate in tuple(selected):
            remaining = [item for item in selected if item is not candidate]
            if not remaining:
                continue
            candidate_expression = _case_conjunction(remaining)
            candidate_attempt = attempt_for(candidate_expression, obligation)
            checks += 1
            if candidate_attempt.get("outcome") == "CERTIFIED":
                selected = remaining
                attempt = candidate_attempt
        return _case_conjunction(selected), attempt, checks

    def partition(group: list[ReducedCase]) -> tuple[list[ReducedCase], list[ReducedCase]]:
        sets = [set(conjunct_maps[item.case_id]) for item in group]
        union = set().union(*sets)
        shared = set.intersection(*sets)
        choices = []
        for key in sorted(union - shared):
            present = sum(key in item for item in sets)
            if 0 < present < len(group):
                choices.append((abs(2 * present - len(group)), key))
        if choices:
            _distance, pivot = min(choices)
            left = [item for item in group if pivot in conjunct_maps[item.case_id]]
            right = [item for item in group if pivot not in conjunct_maps[item.case_id]]
            return left, right
        middle = len(group) // 2
        return group[:middle], group[middle:]

    def check(group: list[ReducedCase]) -> None:
        group = [item for item in group if item.case_id not in results]
        if not group:
            return
        if (
            len(group) == 1
            and conjunct_filter is not None
            and not conjunct_maps[group[0].case_id]
        ):
            results[group[0].case_id] = {
                "outcome": "DEFERRED",
                "reason_code": "INAPPLICABLE_REDUCED_FORM",
                "detail": (
                    f"the reduced equation form has no constraints accepted by "
                    f"the {checker_name} checker"
                ),
                "applicability_checks": {
                    "accepted": False,
                    "case_id": group[0].case_id,
                    "reduced_form_checked": True,
                },
            }
            return
        common = set(conjunct_maps[group[0].case_id])
        for item in group[1:]:
            common.intersection_update(conjunct_maps[item.case_id])
        common_expression = _case_conjunction([
            conjunct_maps[group[0].case_id][key] for key in sorted(common)
        ])
        if len(group) == 1 and conjunct_filter is None:
            common_expression = group[0].expression
        attempt = attempt_for(common_expression, group[0].obligation)
        if attempt.get("outcome") == "CERTIFIED":
            selected_expression, attempt, minimization_checks = (
                minimize_certified_subset(
                    common_expression,
                    group[0].obligation,
                    attempt,
                )
            )
            selected_hash = expression_hash(selected_expression)
            selected_conjuncts = {
                expression_hash(item)
                for item in _case_conjuncts(selected_expression)
            }
            covered_cases = [
                item
                for item in reduced_cases
                if item.case_id not in results
                and selected_conjuncts <= set(conjunct_maps[item.case_id])
            ]
            for item in covered_cases:
                covered = deepcopy(attempt)
                covered.setdefault("applicability_checks", {}).update({
                    "case_id": item.case_id,
                    "shared_case_group": len(covered_cases) > 1,
                    "shared_case_count": len(covered_cases),
                })
                if (
                    selected_hash != expression_hash(item.expression)
                    or len(covered_cases) > 1
                ):
                    covered.setdefault("proof", {})["group_reduction"] = {
                        "rule": "certified_conjunctive_subset_v2",
                        "source_expression_sha256": expression_hash(item.expression),
                        "shared_expression": expr_to_dict(selected_expression),
                        "shared_expression_sha256": selected_hash,
                        "shared_conjunct_sha256": sorted(selected_conjuncts),
                        "covered_case_count": len(covered_cases),
                        "starting_group_case_count": len(group),
                        "subset_minimization_checks": minimization_checks,
                    }
                results[item.case_id] = covered
            return
        if len(group) == 1:
            results[group[0].case_id] = deepcopy(attempt)
            return
        left, right = partition(group)
        check(left)
        check(right)

    check(list(reduced_cases))
    return results


def _controller_context(extractor: CertificationExtractor) -> list[str]:
    return extractor.legacy.ctrl_fqn.split("::")


def _shield_expression(
    extractor: CertificationExtractor,
    model: EquationModel,
    mdp_certificate: dict[str, Any],
) -> tuple[Expr, dict[str, Any]]:
    shield = mdp_certificate["mdp_obligations"]["shield"]
    predicate = shield.get("predicate_ast")
    interface = shield.get("interface") or {}
    subject = str(shield.get("subject_var") or "p")
    if not isinstance(predicate, dict):
        raise ProofDeferred("BLOCKED_INPUT", "shield predicate AST is missing")

    input_sources = {
        row["param"]: row["source_target"]
        for row in shield.get("input_coverage", [])
        if row.get("covered_by_q_and_action") is True
    }
    output_sources = {
        row["param"]: row["action_vars"][0]
        for row in shield.get("output_action_mapping", [])
        if row.get("unique_action_var") is True and len(row.get("action_vars", [])) == 1
    }
    input_params = set(interface.get("input_params", []))
    output_params = set(interface.get("output_params", []))
    ctrl_ctx = _controller_context(extractor)

    def convert(node: dict[str, Any]) -> Expr:
        node_type = node.get("type")
        if node_type == "literal":
            return Const(node.get("value"))
        if node_type == "ref":
            path = list(node.get("path", []))
            if len(path) == 2 and path[0] == subject:
                param = path[1]
                if param in input_params:
                    source = input_sources.get(param)
                    observation_key = str(source).removeprefix("obs.")
                    if observation_key in model.observations:
                        return model.observations[observation_key].expr
                    if source in model.terminals:
                        return model.terminals[source].expr
                    raise ProofDeferred(
                        "BLOCKED_INPUT",
                        f"shield input {param} has no checked equation",
                    )
                if param in output_params:
                    action = output_sources.get(param)
                    if action is None:
                        raise ProofDeferred(
                            "BLOCKED_INPUT",
                            f"shield output {param} has no unique action variable",
                        )
                    return Var(action)
            return extractor._resolve_ref(  # pylint: disable=protected-access
                path,
                ctrl_ctx,
                allow_legacy_fallback=False,
            )
        if node_type == "unary":
            return Op(str(node.get("op")), (convert(node["operand"]),))
        if node_type == "binary":
            return Op(
                str(node.get("op")),
                (convert(node["left"]), convert(node["right"])),
            )
        if node_type == "ternary":
            return Ite(
                convert(node["condition"]),
                convert(node["true"]),
                convert(node["false"]),
            )
        raise ProofDeferred("UNSUPPORTED_EXPRESSION", f"shield node {node_type}")

    expression = expand_definitions(model, convert(predicate))
    return expression, {
        "input_sources": input_sources,
        "output_sources": output_sources,
        "predicate": expr_to_dict(expression),
    }


def _policy_call_guards(
    extractor: CertificationExtractor,
    model: EquationModel,
) -> list[Expr]:
    controller = extractor._controller_part_def()  # pylint: disable=protected-access
    neural = extractor._neural_action_def()  # pylint: disable=protected-access
    if controller is None or neural is None:
        raise ProofDeferred("BLOCKED_INPUT", "controller or neural action is missing")
    actions = {action.name: action for action in controller.actions}
    guards: list[Expr] = []

    def walk(statements, conditions: list[Expr], seen_actions: set[str]) -> None:
        for statement in statements:
            if isinstance(statement, SubactionCallStmt) and statement.type_name == neural.name:
                guards.extend(conditions)
            elif isinstance(statement, IfStmt):
                condition = extractor._expr(  # pylint: disable=protected-access
                    statement.condition,
                    _controller_context(extractor),
                    allow_legacy_fallback=False,
                )
                walk(statement.body, conditions + [condition], seen_actions)
                if statement.else_body:
                    walk(
                        statement.else_body,
                        conditions + [Op("not", (condition,))],
                        seen_actions,
                    )
            elif isinstance(statement, PerformStmt):
                if statement.action_name in seen_actions:
                    raise ProofDeferred(
                        "UNSUPPORTED_EXPRESSION",
                        f"recursive performed action {statement.action_name}",
                    )
                action = actions.get(statement.action_name)
                if action is not None:
                    walk(action.body, conditions, seen_actions | {statement.action_name})

    for action in controller.actions:
        if action.name == "step":
            walk(action.body, [], {"step"})
    unique: list[Expr] = []
    for guard in guards:
        guard = expand_definitions(model, guard)
        if guard not in unique:
            unique.append(guard)
    return unique


def _scenario_constraints(
    extractor: CertificationExtractor,
    model: EquationModel,
) -> tuple[list[Expr], list[Expr]]:
    parameter_constraints: list[Expr] = []
    initial_state_constraints: list[Expr] = []
    for constraint in extractor.parser.parsed_constraints:
        if "ScenarioConstraint" not in getattr(constraint, "metadata", []):
            continue
        ctx = extractor._ctx(constraint.context)  # pylint: disable=protected-access
        for conjunct in extractor._conjuncts(constraint.expression):  # pylint: disable=protected-access
            expression = expand_definitions(
                model,
                extractor._expr(  # pylint: disable=protected-access
                    conjunct,
                    ctx,
                    allow_legacy_fallback=False,
                ),
            )
            if expression_symbols(expression) & model.state:
                initial_state_constraints.append(expression)
            else:
                parameter_constraints.append(expression)
    return parameter_constraints, initial_state_constraints


def _boolean_variables(
    extractor: CertificationExtractor,
    model: EquationModel,
    mdp_certificate: dict[str, Any],
) -> set[str]:
    variables = set()
    for fqn, instance in extractor.parser.part_instances.items():
        part = extractor.parser.part_defs.get(instance.part_type)
        if part is None:
            continue
        for attribute, type_name in part.attributes.items():
            if type_name.lower() in {"bool", "boolean"}:
                variables.add(extractor.legacy._canon(fqn.split("::") + [attribute]))
    interface = mdp_certificate["mdp_obligations"]["shield"].get("interface") or {}
    output_types = interface.get("output_param_types") or {}
    output_mapping = mdp_certificate["mdp_obligations"]["shield"].get(
        "output_action_mapping", []
    )
    for row in output_mapping:
        if str(output_types.get(row.get("param"), "")).lower() in {"bool", "boolean"}:
            variables.update(row.get("action_vars", []))
    return variables & (model.state | model.actions | set(model.definitions))


def _integer_variables(
    extractor: CertificationExtractor,
    model: EquationModel,
) -> set[str]:
    variables: set[str] = set()
    for fqn, instance in extractor.parser.part_instances.items():
        part = extractor.parser.part_defs.get(instance.part_type)
        if part is None:
            continue
        for attribute, type_name in part.attributes.items():
            if type_name.lower() == "integer":
                variables.add(
                    extractor.legacy._canon(fqn.split("::") + [attribute])
                )
    return variables & (
        model.state
        | model.actions
        | model.constants
        | set(model.definitions)
        | set(model.observations)
    )


def _continuous_targets(
    extractor: CertificationExtractor,
    model: EquationModel,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    annotated: set[str] = set()
    dt_updated: set[str] = set()
    records: list[dict[str, Any]] = []
    for action in extractor.parser.step_actions:
        target = extractor.legacy._canon(action.target_key.split("::"))
        equation = model.transitions.get(target)
        if equation is None:
            continue
        symbols = expression_symbols(equation.expr)
        uses_dt = any(name == "dt" or name.endswith("_dt") for name in symbols)
        if uses_dt:
            dt_updated.add(target)
        if "ContinuousRate" in action.metadata:
            annotated.add(target)
            records.append({
                "target": target,
                "metadata": list(action.metadata),
                "equation": equation.pretty(),
                "uses_dt": uses_dt,
            })
    return annotated, dt_updated, records


def _specified_constant_values(
    extractor: CertificationExtractor,
    model: EquationModel,
    dt_record: dict[str, Any],
) -> dict[str, Expr]:
    values: dict[str, Expr] = {}
    for parameter in extractor.parser.parameters:
        target = extractor.legacy._canon(parameter.qualified_name.split("::"))
        if target not in model.constants or "ScenarioInput" in parameter.metadata:
            continue
        values[target] = Const(parameter.value)
    for target in model.constants:
        if target == "dt" or target.endswith("_dt"):
            values[target] = Const(dt_record["canonical"])
    return values


def _timing_record(
    mdp_certificate: dict[str, Any],
    dt_record: dict[str, Any],
) -> dict[str, Any]:
    sampled = []
    for fact in mdp_certificate.get("equation_proof", {}).get("facts", []):
        if fact.get("rule") != "sampled_memory_bound":
            continue
        detail = fact.get("detail") or {}
        sampled.append({
            "target": detail.get("target", fact.get("var")),
            "source": detail.get("source"),
            "max_delay_steps": int(detail.get("max_delay", 1)),
            "schedule_state": detail.get("schedule_state"),
        })
    return {
        "fixed_dt": dt_record,
        "sampled_memory_cases": sampled,
        "information_delay_treatment": (
            "The recorded observation and action history reconstructs the modeled "
            "state used at the controller update. It is retained as information "
            "history and is not converted into elapsed physical time."
        ),
        "interval_length_use": (
            "Every physical trajectory is checked for interval time from zero through "
            "the single fixed dt. Sensor and schedule guards are retained in the "
            "sampled point premise."
        ),
        "observation_defect_envelope": {
            "physical_time_upper": dt_record["canonical"],
            "fixed_dt": True,
            "information_history": sampled,
        },
    }


def _top_level_conjuncts(expression: Expr) -> list[Expr]:
    if isinstance(expression, Op) and expression.op == "and":
        result: list[Expr] = []
        for argument in expression.args:
            result.extend(_top_level_conjuncts(argument))
        return result
    return [expression]


def _conjunction(expressions: list[Expr]) -> Expr:
    if not expressions:
        return Const(True)
    if len(expressions) == 1:
        return expressions[0]
    return Op("and", tuple(expressions))


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
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    raise ValueError(f"Z3 value is not an exact rational or Boolean: {value}")


def _exact_model_values(
    model: EquationModel,
    encoder: Encoder,
    solver_model: Any,
    expression: Expr,
) -> dict[str, bool | str]:
    raw_references = _raw_reference_names(expression)
    values: dict[str, bool | str] = {}
    for name in sorted(expression_symbols(expression)):
        if name in raw_references:
            encoded = encoder.const_var(name)
        else:
            encoded = encoder.encode_expr(Var(name), 1, "current")
        exact = _z3_exact_value(
            solver_model.eval(encoded, model_completion=True)
        )
        values[name] = serialize_exact_value(exact)
    return values


def _core_recertification_attempts(
    reduced_case: ReducedCase,
    selected_constraints: list[Expr],
    *,
    timeout_ms: int,
) -> tuple[ReducedCase, list[dict[str, Any]], dict[str, Any] | None]:
    core_case = ReducedCase(
        case_id=reduced_case.case_id + ".smt_core",
        expression=_conjunction(selected_constraints),
        parent_hash=expression_hash(reduced_case.expression),
        boolean_assignment=reduced_case.boolean_assignment,
        time_reduction=reduced_case.time_reduction,
        obligation=reduced_case.obligation,
    )
    checks = [
        (
            "linear",
            lambda: run_linear_checker(core_case, set(), timeout_ms=timeout_ms),
        ),
        (
            "linear_envelope",
            lambda: run_linear_envelope_checker(core_case, timeout_ms=timeout_ms),
        ),
        (
            "convex",
            lambda: run_convex_checker(core_case, set(), timeout_ms=timeout_ms),
        ),
        (
            "convex_envelope",
            lambda: run_convex_envelope_checker(core_case, timeout_ms=timeout_ms),
        ),
    ]
    attempts: list[dict[str, Any]] = []
    for checker, run in checks:
        attempt = _validated_attempt(run())
        record = {
            "checker": checker,
            "outcome": attempt.get("outcome", "DEFERRED"),
            "reason_code": attempt.get("reason_code", ""),
            "detail": attempt.get("detail", ""),
            "applicability_checks": attempt.get("applicability_checks") or {},
            "proof": attempt.get("proof") or {},
        }
        attempts.append(record)
        if record["outcome"] == "CERTIFIED":
            return core_case, attempts, record
    return core_case, attempts, None


def _smt_fallback(
    model: EquationModel,
    reduced_case: ReducedCase,
    *,
    timeout_ms: int,
    recertification_timeout_ms: int,
) -> dict[str, Any]:
    if z3 is None:
        return _stage(
            "smt_fallback",
            "DEFERRED",
            reason_code="BLOCKED_INPUT",
            detail="z3-solver is unavailable",
        )
    try:
        sorts, conflicts = infer_sorts(model)
        if conflicts:
            return _stage(
                "smt_fallback",
                "DEFERRED",
                reason_code="UNSUPPORTED_EXPRESSION",
                detail="; ".join(conflicts),
            )
        encoder = Encoder(model, sorts)
        solver = z3.Solver()
        solver.set(timeout=int(timeout_ms))
        solver.set(unsat_core=True)
        source_constraints = _top_level_conjuncts(reduced_case.expression)
        trackers = [
            z3.Bool(f"discretization_core_{index}")
            for index in range(len(source_constraints))
        ]
        for index, constraint in enumerate(source_constraints):
            solver.add(z3.Implies(
                trackers[index],
                encoder.encode_expr(constraint, 1, "current"),
            ))
        result = solver.check(*trackers)
        if result == z3.unknown:
            reason = solver.reason_unknown()
            code = "TIMEOUT" if "timeout" in reason.lower() else "PROOF_REJECTED"
            return _stage(
                "smt_fallback",
                "DEFERRED",
                reason_code=code,
                detail=reason,
                proof={"solver_status": "unknown"},
            )
        if result == z3.unsat:
            selected_indices = sorted({
                int(str(item).removeprefix("discretization_core_"))
                for item in solver.unsat_core()
            })
            minimization_checks = 0
            minimization_complete = True
            for index in tuple(selected_indices):
                candidate = [
                    item for item in selected_indices if item != index
                ]
                candidate_result = solver.check(*[
                    trackers[item] for item in candidate
                ])
                minimization_checks += 1
                if candidate_result == z3.unsat:
                    selected_indices = candidate
                elif candidate_result == z3.unknown:
                    minimization_complete = False
            selected_constraints = [
                source_constraints[index] for index in selected_indices
            ]
            core_case, attempts, certified = _core_recertification_attempts(
                reduced_case,
                selected_constraints,
                timeout_ms=recertification_timeout_ms,
            )
            proof = {
                "rule": "solver_selected_subset_recertification_v1",
                "solver_status": "unsat",
                "source_expression_sha256": expression_hash(
                    reduced_case.expression
                ),
                "source_constraint_count": len(source_constraints),
                "selected_indices": selected_indices,
                "subset_minimization_checks": minimization_checks,
                "subset_minimization_complete": minimization_complete,
                "selected_constraints": [
                    expr_to_dict(item) for item in selected_constraints
                ],
                "selected_expression": expr_to_dict(core_case.expression),
                "selected_expression_sha256": expression_hash(
                    core_case.expression
                ),
                "recertification_attempts": attempts,
            }
            if certified is not None:
                proof["certifying_checker"] = certified["checker"]
                proof["certificate_attempt"] = certified
                return _stage(
                    "smt_fallback",
                    "CERTIFIED",
                    detail=(
                        "a solver-selected subset of the source constraints "
                        "has an independently checked linear or convex certificate"
                    ),
                    proof=proof,
                )
            return _stage(
                "smt_fallback",
                "DEFERRED",
                reason_code="PROOF_REJECTED",
                detail=(
                    "the solver-selected source constraint subset was not "
                    "certified by the linear or convex checkers"
                ),
                proof=proof,
            )
        exact_values = _exact_model_values(
            model,
            encoder,
            solver.model(),
            reduced_case.expression,
        )
        replayed = replay_serialized_boolean_expression(
            expr_to_dict(reduced_case.expression),
            exact_values,
        )
        if replayed:
            return _stage(
                "smt_fallback",
                "DEFERRED",
                reason_code="REACHABILITY_BOUND_INCONCLUSIVE",
                detail=(
                    "the exact values replay the local unsafe constraints, but "
                    "do not establish reachability from the declared initial state"
                ),
                proof={
                    "rule": "exact_local_feasibility_replay_v1",
                    "solver_status": "sat",
                    "source_expression_sha256": expression_hash(
                        reduced_case.expression
                    ),
                    "exact_values": exact_values,
                    "exact_replay": True,
                },
            )
        return _stage(
            "smt_fallback",
            "DEFERRED",
            reason_code="COUNTEREXAMPLE_REPLAY_FAILED",
            detail="solver candidate was not independently replayed",
            proof={"solver_status": "sat"},
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _stage(
            "smt_fallback",
            "DEFERRED",
            reason_code="MALFORMED_OUTPUT",
            detail=str(exc),
        )


def analyze_model(
    model_path: str | Path,
    mdp_certificate: dict[str, Any],
    *,
    dt_text: str,
    optimization_timeout_ms: int = 250,
    smt_timeout_ms: int = 30000,
) -> dict[str, Any]:
    path = str(Path(model_path).resolve())
    dt_record = canonical_dt(dt_text)
    extractor = CertificationExtractor(path)
    model = extractor.extract()
    boolean_variables = _boolean_variables(extractor, model, mdp_certificate)
    integer_variables = _integer_variables(extractor, model)
    timing = _timing_record(mdp_certificate, dt_record)
    annotated, _dt_updated, continuous_records = _continuous_targets(extractor, model)
    constant_values = _specified_constant_values(extractor, model, dt_record)

    blockers = [
        diagnostic.pretty()
        for diagnostic in model.diagnostics
        if diagnostic.severity in {"warning", "error"}
    ]
    if blockers:
        return {
            "schema_version": 3,
            "result": "NOT_CERTIFIED",
            "claim": "full_sysml_discretization_safety_preservation_v3",
            "timing": timing,
            "continuous_rate_assignments": continuous_records,
            "properties": [],
            "blocking_diagnostics": blockers,
        }

    try:
        shield_expression, _shield_record = _shield_expression(
            extractor, model, mdp_certificate
        )
        guards = _policy_call_guards(extractor, model)
        scenario_constraints, scenario_initial_constraints = _scenario_constraints(
            extractor, model
        )
        shield_expression = substitute(shield_expression, constant_values)
        guards = [substitute(item, constant_values) for item in guards]
        scenario_constraints = [
            substitute(item, constant_values) for item in scenario_constraints
        ]
        scenario_initial_constraints = [
            substitute(item, constant_values)
            for item in scenario_initial_constraints
        ]
    except ProofDeferred as exc:
        return {
            "schema_version": 3,
            "result": "NOT_CERTIFIED",
            "claim": "full_sysml_discretization_safety_preservation_v3",
            "timing": timing,
            "continuous_rate_assignments": continuous_records,
            "properties": [],
            "blocking_diagnostics": [f"{exc.reason_code}: {exc.detail}"],
        }

    properties: list[dict[str, Any]] = []
    shared_reachability_cache = SharedReachabilityCache()
    for target, equation in sorted(model.requirements.items()):
        if equation.source not in {"Prohibition", "Obligation"}:
            continue
        property_id = target.removeprefix("status.")
        dependencies = equation_refs(model, equation)
        try:
            reduced_cases, reduction, reachability_context = build_reduction(
                extractor,
                model,
                equation,
                shield_expression,
                guards,
                scenario_constraints,
                scenario_initial_constraints,
                constant_values,
                annotated,
                boolean_variables,
                integer_variables,
                dt_record,
            )
        except ProofDeferred as exc:
            properties.append({
                "property_id": property_id,
                "annotation": equation.source,
                "source": equation.pretty(),
                "dependencies": sorted(dependencies),
                "reduction": {
                    "kind": "full_sysml_interval_reduction_v3",
                    "outcome": "DEFERRED",
                    "reason_code": exc.reason_code,
                    "detail": exc.detail,
                },
                "cases": [],
                "progression": [
                    _stage(
                        checker,
                        "DEFERRED",
                        reason_code=exc.reason_code,
                        detail=exc.detail,
                    )
                    for checker in CHECKER_ORDER
                ],
                "result": "NOT_CERTIFIED",
            })
            continue

        if any(item.factored for item in reduced_cases):
            context_sha256 = shared_reachability_context_sha256(
                reachability_context
            )
            linear_group_attempts = {
                item.case_id: run_lazy_factored_checker(
                    item,
                    boolean_variables,
                    "linear",
                    lambda leaf, remaining: run_linear_checker(
                        leaf,
                        set(),
                        timeout_ms=min(optimization_timeout_ms, remaining),
                    ),
                    lambda expression: expression_is_linear(
                        expression,
                        set(),
                    )[0],
                    timeout_ms=optimization_timeout_ms,
                    context_sha256=context_sha256,
                )
                for item in reduced_cases
            }
            convex_group_attempts = {
                item.case_id: run_lazy_factored_checker(
                    item,
                    boolean_variables,
                    "convex",
                    lambda leaf, remaining: run_convex_checker(
                        leaf,
                        set(),
                        timeout_ms=min(optimization_timeout_ms, remaining),
                    ),
                    _expression_is_convex,
                    timeout_ms=optimization_timeout_ms,
                    context_sha256=context_sha256,
                )
                for item in reduced_cases
                if linear_group_attempts[item.case_id].get("outcome")
                != "CERTIFIED"
            }
        else:
            linear_group_attempts = _grouped_case_attempts(
                reduced_cases,
                "linear",
                lambda item: run_linear_checker(
                    item,
                    set(),
                    timeout_ms=optimization_timeout_ms,
                ),
                conjunct_filter=lambda expression: expression_is_linear(
                    expression,
                    set(),
                )[0],
            )
            convex_group_attempts = _grouped_case_attempts(
                [
                    item for item in reduced_cases
                    if linear_group_attempts[item.case_id].get("outcome")
                    != "CERTIFIED"
                ],
                "convex",
                lambda item: run_convex_checker(
                    item,
                    set(),
                    timeout_ms=optimization_timeout_ms,
                ),
                conjunct_filter=_expression_is_convex,
            )
        case_records: list[dict[str, Any]] = []
        pending_relational: list[tuple[ReducedCase, dict[str, Any]]] = []
        for reduced_case in reduced_cases:
            case_progression: list[dict[str, Any]] = []
            linear_attempt = deepcopy(linear_group_attempts[reduced_case.case_id])
            if linear_attempt.get("outcome") == "VIOLATION":
                linear_attempt = {
                    **linear_attempt,
                    "outcome": "DEFERRED",
                    "reason_code": "COUNTEREXAMPLE_REPLAY_FAILED",
                    "detail": (
                        "the arithmetic candidate satisfies the local proof obligation, "
                        "but reachability from the declared initial state was not replayed"
                    ),
                }
            case_progression.append(_attempt_stage("linear", linear_attempt))
            outcome = str(linear_attempt.get("outcome", "DEFERRED"))

            if outcome == "DEFERRED":
                convex_attempt = deepcopy(
                    convex_group_attempts[reduced_case.case_id]
                )
                if convex_attempt.get("outcome") == "VIOLATION":
                    convex_attempt = {
                        **convex_attempt,
                        "outcome": "DEFERRED",
                        "reason_code": "COUNTEREXAMPLE_REPLAY_FAILED",
                        "detail": (
                            "the arithmetic candidate satisfies the local proof obligation, "
                            "but reachability from the declared initial state was not replayed"
                        ),
                    }
                case_progression.append(_attempt_stage("convex", convex_attempt))
                outcome = str(convex_attempt.get("outcome", "DEFERRED"))

            if outcome == "DEFERRED":
                reachability_linear = _validated_attempt(
                    run_reachability_checker(
                        reduced_case,
                        reachability_context,
                        method="linear",
                        timeout_ms=optimization_timeout_ms,
                    )
                )
                case_progression.append(_attempt_stage(
                    "reachability_linear",
                    reachability_linear,
                ))
                outcome = str(
                    reachability_linear.get("outcome", "DEFERRED")
                )

            if outcome == "DEFERRED":
                reachability_convex = _validated_attempt(
                    run_reachability_checker(
                        reduced_case,
                        reachability_context,
                        method="convex",
                        timeout_ms=optimization_timeout_ms,
                    )
                )
                case_progression.append(_attempt_stage(
                    "reachability_convex",
                    reachability_convex,
                ))
                outcome = str(
                    reachability_convex.get("outcome", "DEFERRED")
                )

            if outcome == "DEFERRED":
                if reduced_case.factored:
                    exact_proof = {
                        "proved": False,
                        "reason_code": "FACTORED_FORMULA_PRESERVED",
                        "detail": (
                            "the factored formula is passed intact to the next "
                            "checker without exhaustive symbolic case expansion"
                        ),
                    }
                else:
                    try:
                        exact_proof = prove_implication_exact(
                            [reduced_case.expression],
                            Const(False),
                            set(),
                        )
                    except ProofDeferred as exc:
                        exact_proof = {
                            "proved": False,
                            "reason_code": exc.reason_code,
                            "detail": exc.detail,
                        }
                    except Exception as exc:  # pragma: no cover
                        exact_proof = {
                            "proved": False,
                            "reason_code": "MALFORMED_OUTPUT",
                            "detail": str(exc),
                        }
                if exact_proof.get("proved"):
                    case_progression.append(_stage(
                        "exact_symbolic",
                        "CERTIFIED",
                        applicability_checks={
                            "reduced_case_required": True,
                            "case_id": reduced_case.case_id,
                        },
                        proof={
                            "rule": "reduced_case_exact_infeasibility_v2",
                            "within_interval_proof": exact_proof,
                        },
                    ))
                    outcome = "CERTIFIED"
                else:
                    case_progression.append(_stage(
                        "exact_symbolic",
                        "DEFERRED",
                        reason_code=str(
                            exact_proof.get("reason_code")
                            or "UNSUPPORTED_EXPRESSION"
                        ),
                        detail=str(
                            exact_proof.get("detail")
                            or "the reduced case remains feasible"
                        ),
                        applicability_checks={
                            "reduced_case_required": True,
                            "case_id": reduced_case.case_id,
                        },
                        proof=exact_proof,
                    ))
                    outcome = "DEFERRED"

            if outcome == "DEFERRED":
                smt_stage = _smt_fallback(
                    model,
                    reduced_case,
                    timeout_ms=smt_timeout_ms,
                    recertification_timeout_ms=optimization_timeout_ms,
                )
                smt_stage.setdefault("applicability_checks", {}).update({
                    "reduced_case_required": True,
                    "case_id": reduced_case.case_id,
                })
                case_progression.append(smt_stage)
                outcome = str(smt_stage.get("outcome", "DEFERRED"))

            case_record = {
                "case_id": reduced_case.case_id,
                "expression": expr_to_dict(reduced_case.expression),
                "reachability_expression": expr_to_dict(
                    reduced_case.reachability_expression
                    or reduced_case.expression
                ),
                "parent_expression_sha256": reduced_case.parent_hash,
                "boolean_assignment": dict(reduced_case.boolean_assignment),
                "time_reduction": reduced_case.time_reduction,
                "obligation": reduced_case.obligation,
                "progression": case_progression,
                "result": (
                    outcome
                    if outcome in {"CERTIFIED", "VIOLATION"}
                    else "NOT_CERTIFIED"
                ),
            }
            case_records.append(case_record)
            if outcome == "DEFERRED":
                pending_relational.append((reduced_case, case_record))

        if pending_relational:
            relational_attempts = run_relational_invariant_group_checker(
                model,
                [item for item, _record in pending_relational],
                reachability_context,
                timeout_ms=smt_timeout_ms,
                cache=shared_reachability_cache,
            )
            for reduced_case, case_record in pending_relational:
                relational_invariant = _validated_attempt(
                    relational_attempts.get(reduced_case.case_id, {
                        "outcome": "DEFERRED",
                        "reason_code": "MALFORMED_OUTPUT",
                        "detail": "grouped relational checker omitted the case",
                    })
                )
                case_record["progression"].append(_attempt_stage(
                    "relational_invariant",
                    relational_invariant,
                ))
                outcome = str(
                    relational_invariant.get("outcome", "DEFERRED")
                )
                if outcome == "DEFERRED":
                    smt_reachability = _validated_attempt(
                        run_smt_reachability_checker(
                            model,
                            reduced_case,
                            reachability_context,
                            timeout_ms=smt_timeout_ms,
                        )
                    )
                    case_record["progression"].append(_attempt_stage(
                        "smt_reachability",
                        smt_reachability,
                    ))
                    outcome = str(
                        smt_reachability.get("outcome", "DEFERRED")
                    )
                case_record["result"] = (
                    outcome
                    if outcome in {"CERTIFIED", "VIOLATION"}
                    else "NOT_CERTIFIED"
                )

        if any(item["result"] == "VIOLATION" for item in case_records):
            property_result = "VIOLATION"
        elif all(item["result"] == "CERTIFIED" for item in case_records):
            property_result = "CERTIFIED"
        else:
            property_result = "NOT_CERTIFIED"
        property_progression: list[dict[str, Any]] = []
        for checker in CHECKER_ORDER:
            attempts = [
                stage
                for case in case_records
                for stage in case["progression"]
                if stage["checker"] == checker
            ]
            if not attempts:
                continue
            property_progression.append(_stage(
                checker,
                (
                    "VIOLATION"
                    if any(stage["outcome"] == "VIOLATION" for stage in attempts)
                    else (
                        "CERTIFIED"
                        if all(stage["outcome"] == "CERTIFIED" for stage in attempts)
                        else "DEFERRED"
                    )
                ),
                reason_code=(
                    "COUNTEREXAMPLE_REPLAYED"
                    if any(stage["outcome"] == "VIOLATION" for stage in attempts)
                    else (
                        ""
                        if all(stage["outcome"] == "CERTIFIED" for stage in attempts)
                        else "UNRESOLVED_CASES"
                    )
                ),
                detail=f"{sum(stage['outcome'] == 'CERTIFIED' for stage in attempts)}/{len(attempts)} attempted cases certified",
                applicability_checks={"case_attempt_count": len(attempts)},
            ))

        properties.append({
            "property_id": property_id,
            "annotation": equation.source,
            "source": equation.pretty(),
            "dependencies": sorted(dependencies),
            "reduction": reduction,
            "cases": case_records,
            "progression": property_progression,
            "result": property_result,
        })

    if any(item["result"] == "VIOLATION" for item in properties):
        result = "VIOLATION"
    elif properties and all(item["result"] == "CERTIFIED" for item in properties):
        result = "CERTIFIED"
    else:
        result = "NOT_CERTIFIED"
    shared_proof_certificates = _deduplicate_proof_certificates(properties)
    return {
        "schema_version": 3,
        "result": result,
        "claim": "full_sysml_discretization_safety_preservation_v3",
        "claim_scope": (
            "For the complete SysML physical process from each shielded controller "
            "update through every time in the following fixed dt interval, for every "
            "parsed #Prohibition and #Obligation."
        ),
        "timing": timing,
        "continuous_rate_semantics": (
            "#ContinuousRate marks x := x + rate * dt as a rate held over the "
            "following physical interval, with the selected controller action held "
            "through the actuator and physical equations."
        ),
        "continuous_rate_assignments": continuous_records,
        "specified_constant_values": {
            name: expr_to_dict(value) for name, value in sorted(constant_values.items())
        },
        "checker_order": CHECKER_ORDER,
        "markov_process_evidence": {
            "buffer": mdp_certificate.get("buffer", {}),
            "reconstructed_state": mdp_certificate.get("sets", {}).get("q", []),
            "executed_actions": mdp_certificate.get("sets", {}).get("actions", []),
            "time_variables": mdp_certificate.get("sets", {}).get("time_vars", []),
        },
        "optimization_timeout_ms": int(optimization_timeout_ms),
        "shared_proof_certificates": shared_proof_certificates,
        "shared_reachability": shared_reachability_cache.export(),
        "properties": properties,
        "blocking_diagnostics": [],
    }
