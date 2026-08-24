"""Build deterministic discretization-safety proof obligations and results."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from certification.equations import Const, EquationModel, Expr, Ite, Op, RawRef, Var
from certification.relevance import equation_refs
from certification.solver import Encoder, infer_sorts
from certification.strict_extract import CertificationExtractor
from sysml_parser import IfStmt, PerformStmt, SubactionCallStmt

from .convex_checker import run_convex_checker
from .linear_checker import run_linear_checker
from .proof_rules import (
    ProofDeferred,
    expand_definitions,
    expr_to_dict,
    expression_symbols,
    post_state_expression,
    prove_implication_exact,
    simplify_known_conditions,
    substitute,
    within_interval_expression,
)

try:  # pragma: no cover - integration environment determines availability
    import z3  # type: ignore
except Exception:  # pragma: no cover
    z3 = None


CHECKER_ORDER = [
    "linear",
    "convex",
    "exact_symbolic",
    "smt_fallback",
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


def _and(expressions: list[Expr]) -> Expr:
    if not expressions:
        return Const(True)
    if len(expressions) == 1:
        return expressions[0]
    return Op("and", tuple(expressions))


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


def _scenario_parameter_constraints(
    extractor: CertificationExtractor,
    model: EquationModel,
) -> list[Expr]:
    constraints: list[Expr] = []
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
                continue
            constraints.append(expression)
    return constraints


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
            values[target] = Const(dt_record["decimal"])
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
            "The recorded steps describe how much observation and action history "
            "reconstructs state. They are not treated as elapsed physical time."
        ),
        "interval_length_use": (
            "The exact proof quantifies changing property values without restricting "
            "the interval length, so it does not convert the information delay to time."
        ),
    }


def _smt_fallback(
    model: EquationModel,
    premises: list[Expr],
    conclusion: Expr,
    *,
    timeout_ms: int,
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
        for premise in premises:
            solver.add(encoder.encode_expr(premise, 1, "current"))
        solver.add(z3.Not(encoder.encode_expr(conclusion, 1, "current")))
        result = solver.check()
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
            return _stage(
                "smt_fallback",
                "DEFERRED",
                reason_code="PROOF_REJECTED",
                detail="solver returned unsat without a proof supported by the independent checker",
                proof={"solver_status": "unsat"},
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
    smt_timeout_ms: int = 2000,
) -> dict[str, Any]:
    path = str(Path(model_path).resolve())
    dt_record = canonical_dt(dt_text)
    extractor = CertificationExtractor(path)
    model = extractor.extract()
    boolean_variables = _boolean_variables(extractor, model, mdp_certificate)
    timing = _timing_record(mdp_certificate, dt_record)
    annotated, dt_updated, continuous_records = _continuous_targets(extractor, model)
    constant_values = _specified_constant_values(extractor, model, dt_record)

    blockers = [
        diagnostic.pretty()
        for diagnostic in model.diagnostics
        if diagnostic.severity in {"warning", "error"}
    ]
    if blockers:
        return {
            "schema_version": 1,
            "result": "NOT_CERTIFIED",
            "claim": "discretization_safety_preservation_v1",
            "timing": timing,
            "continuous_rate_assignments": continuous_records,
            "properties": [],
            "blocking_diagnostics": blockers,
        }

    try:
        shield_expression, shield_record = _shield_expression(
            extractor, model, mdp_certificate
        )
        guards = _policy_call_guards(extractor, model)
        scenario_constraints = _scenario_parameter_constraints(extractor, model)
        shield_expression = substitute(shield_expression, constant_values)
        guards = [substitute(item, constant_values) for item in guards]
        scenario_constraints = [
            substitute(item, constant_values) for item in scenario_constraints
        ]
        shield_record["predicate"] = expr_to_dict(shield_expression)
    except ProofDeferred as exc:
        return {
            "schema_version": 1,
            "result": "NOT_CERTIFIED",
            "claim": "discretization_safety_preservation_v1",
            "timing": timing,
            "continuous_rate_assignments": continuous_records,
            "properties": [],
            "blocking_diagnostics": [f"{exc.reason_code}: {exc.detail}"],
        }

    premises = [shield_expression, *guards, *scenario_constraints]
    properties: list[dict[str, Any]] = []
    for target, equation in sorted(model.requirements.items()):
        if equation.source not in {"Prohibition", "Obligation"}:
            continue
        property_id = target.removeprefix("status.")
        current_expression = substitute(
            expand_definitions(model, equation.expr),
            constant_values,
        )
        post_expression = post_state_expression(model, current_expression)
        post_expression = substitute(post_expression, constant_values)
        post_expression = simplify_known_conditions(post_expression, guards)
        interval_expression = within_interval_expression(
            model,
            current_expression,
            annotated,
        )
        interval_expression = substitute(interval_expression, constant_values)
        interval_expression = simplify_known_conditions(interval_expression, guards)
        dependencies = equation_refs(model, equation)
        continuous_dependencies = sorted(
            expression_symbols(current_expression) & dt_updated
        )
        unannotated_continuous = sorted(set(continuous_dependencies) - annotated)
        progression: list[dict[str, Any]] = []
        counterexample = _and([*premises, Op("not", (interval_expression,))])

        if unannotated_continuous:
            missing_detail = (
                "missing #ContinuousRate on " + ", ".join(unannotated_continuous)
            )
            linear_attempt = {
                "outcome": "DEFERRED",
                "reason_code": "MISSING_WITHIN_STEP_MEANING",
                "detail": missing_detail,
                "applicability_checks": {
                    "accepted": False,
                    "all_continuous_dependencies_annotated": False,
                    "optimization_timeout_ms": int(optimization_timeout_ms),
                },
            }
        else:
            linear_attempt = run_linear_checker(
                counterexample,
                boolean_variables,
                timeout_ms=optimization_timeout_ms,
            )
            linear_attempt.setdefault("applicability_checks", {})[
                "all_continuous_dependencies_annotated"
            ] = True
        linear_attempt = _validated_attempt(linear_attempt)
        progression.append(_attempt_stage("linear", linear_attempt))

        outcome = str(linear_attempt.get("outcome", "DEFERRED"))
        if outcome == "DEFERRED":
            if unannotated_continuous:
                convex_attempt = {
                    "outcome": "DEFERRED",
                    "reason_code": "MISSING_WITHIN_STEP_MEANING",
                    "detail": missing_detail,
                    "applicability_checks": {
                        "accepted": False,
                        "all_continuous_dependencies_annotated": False,
                        "optimization_timeout_ms": int(optimization_timeout_ms),
                    },
                }
            else:
                convex_attempt = run_convex_checker(
                    counterexample,
                    boolean_variables,
                    timeout_ms=optimization_timeout_ms,
                )
                convex_attempt.setdefault("applicability_checks", {})[
                    "all_continuous_dependencies_annotated"
                ] = True
            convex_attempt = _validated_attempt(convex_attempt)
            progression.append(_attempt_stage("convex", convex_attempt))
            outcome = str(convex_attempt.get("outcome", "DEFERRED"))

        if outcome == "DEFERRED":
            try:
                exact_proof = prove_implication_exact(
                    premises,
                    interval_expression,
                    boolean_variables,
                )
            except ProofDeferred as exc:
                exact_proof = {
                    "proved": False,
                    "reason_code": exc.reason_code,
                    "detail": exc.detail,
                }
            except Exception as exc:  # pragma: no cover - fail-closed boundary
                exact_proof = {
                    "proved": False,
                    "reason_code": "MALFORMED_OUTPUT",
                    "detail": str(exc),
                }
            exact_applicability = {
                "sampled_point_implication_proved": bool(exact_proof.get("proved")),
                "continuous_property_dependencies": continuous_dependencies,
                "property_variables_held_between_controller_updates": not continuous_dependencies,
                "all_continuous_dependencies_annotated": not unannotated_continuous,
                "time_variables_quantified_without_restriction": sorted(
                    dependencies & set(mdp_certificate.get("sets", {}).get("time_vars", []))
                ),
            }
            if exact_proof.get("proved") and not unannotated_continuous:
                progression.append(_stage(
                    "exact_symbolic",
                    "CERTIFIED",
                    applicability_checks=exact_applicability,
                    proof={
                        "rule": "shielded_update_implies_property_for_arbitrary_within_interval_continuous_values_v1",
                        "within_interval_proof": exact_proof,
                    },
                ))
                outcome = "CERTIFIED"
            else:
                reason = (
                    "MISSING_WITHIN_STEP_MEANING"
                    if unannotated_continuous
                    else str(exact_proof.get("reason_code") or "UNSUPPORTED_EXPRESSION")
                )
                progression.append(_stage(
                    "exact_symbolic",
                    "DEFERRED",
                    reason_code=reason,
                    detail=str(
                        exact_proof.get("detail")
                        or "exact rule did not certify the property"
                    ),
                    applicability_checks=exact_applicability,
                    proof=exact_proof,
                ))
                outcome = "DEFERRED"

        if outcome == "DEFERRED":
            progression.append(_smt_fallback(
                model,
                premises,
                post_expression,
                timeout_ms=smt_timeout_ms,
            ))
            outcome = "DEFERRED"

        property_result = outcome if outcome in {"CERTIFIED", "VIOLATION"} else "NOT_CERTIFIED"

        properties.append({
            "property_id": property_id,
            "annotation": equation.source,
            "source": equation.pretty(),
            "dependencies": sorted(dependencies),
            "continuous_dependencies": continuous_dependencies,
            "sampled_point_premises": {
                "shield": shield_record,
                "controller_call_guards": [expr_to_dict(item) for item in guards],
                "scenario_parameter_constraints": [
                    expr_to_dict(item) for item in scenario_constraints
                ],
            },
            "post_action_property": expr_to_dict(post_expression),
            "within_interval_property": expr_to_dict(interval_expression),
            "progression": progression,
            "result": property_result,
        })

    result = (
        "CERTIFIED"
        if properties and all(item["result"] == "CERTIFIED" for item in properties)
        else "NOT_CERTIFIED"
    )
    return {
        "schema_version": 1,
        "result": result,
        "claim": "discretization_safety_preservation_v1",
        "claim_scope": (
            "From each shielded controller update after environment reset to the next "
            "controller update, for every parsed #Prohibition and #Obligation."
        ),
        "timing": timing,
        "continuous_rate_semantics": (
            "#ContinuousRate marks x := x + rate * dt as a rate held over the "
            "corresponding sampled interval under the extracted simultaneous transition semantics."
        ),
        "continuous_rate_assignments": continuous_records,
        "specified_constant_values": {
            name: expr_to_dict(value) for name, value in sorted(constant_values.items())
        },
        "checker_order": CHECKER_ORDER,
        "optimization_timeout_ms": int(optimization_timeout_ms),
        "properties": properties,
        "blocking_diagnostics": [],
    }
