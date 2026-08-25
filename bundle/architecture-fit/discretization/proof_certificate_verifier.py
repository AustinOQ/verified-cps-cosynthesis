"""Exact verifier for recorded linear and convex proof certificates."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

from .exact_replay import replay_serialized_boolean_expression
from .optimization_common import fraction_text, parse_fraction

try:  # pragma: no cover - integration environment determines availability
    import z3  # type: ignore
except Exception:  # pragma: no cover
    z3 = None


def _fraction_mapping(value: Any) -> dict[str, Fraction]:
    if not isinstance(value, dict):
        raise ValueError("coefficient mapping is malformed")
    return {str(name): parse_fraction(number) for name, number in value.items()}


def verify_recorded_linear_certificate(certificate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if certificate.get("kind") != "linear_infeasibility_weights_v1":
        errors.append("linear certificate kind is invalid")
    try:
        constraints = certificate["constraints"]
        if not isinstance(constraints, list):
            raise ValueError
        coefficients = [_fraction_mapping(item["coefficients"]) for item in constraints]
        bounds = [parse_fraction(item["bound"]) for item in constraints]
        strict = [item.get("strict") is True for item in constraints]
        if any(item.get("strict") not in {True, False} for item in constraints):
            errors.append("linear certificate strictness is malformed")
        multipliers = [parse_fraction(value) for value in certificate["multipliers"]]
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("linear certificate contents are malformed")
        return errors
    if len(multipliers) != len(constraints):
        errors.append("linear certificate multiplier count is incorrect")
        return errors
    if any(value < 0 for value in multipliers):
        errors.append("linear certificate contains a negative multiplier")
    combined: dict[str, Fraction] = {}
    combined_bound = Fraction(0)
    for row, bound, multiplier in zip(coefficients, bounds, multipliers):
        for name, value in row.items():
            combined[name] = combined.get(name, Fraction(0)) + multiplier * value
        combined_bound += multiplier * bound
    combined = {name: value for name, value in sorted(combined.items()) if value}
    combined_strict = any(
        is_strict and multiplier > 0
        for is_strict, multiplier in zip(strict, multipliers)
    )
    if combined:
        errors.append("linear weighted sum does not eliminate every variable")
    if combined_bound > 0 or (combined_bound == 0 and not combined_strict):
        errors.append("linear weighted sum does not produce a contradiction")
    if certificate.get("combined_coefficients") != {
        name: fraction_text(value) for name, value in combined.items()
    }:
        errors.append("linear certificate combined coefficients are incorrect")
    if certificate.get("combined_bound") != fraction_text(combined_bound):
        errors.append("linear certificate combined bound is incorrect")
    if certificate.get("combined_strict") is not combined_strict:
        errors.append("linear certificate combined strictness is incorrect")
    return errors


def verify_recorded_linear_counterexample(proof: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        witness = _fraction_mapping(proof["counterexample"])
        constraints = proof["constraints"]
        if not isinstance(constraints, list):
            raise ValueError
        for item in constraints:
            coefficients = _fraction_mapping(item["coefficients"])
            bound = parse_fraction(item["bound"])
            if item.get("strict") not in {True, False}:
                raise ValueError
            if any(name not in witness for name in coefficients):
                errors.append("linear counterexample omits a required variable")
                continue
            left = sum(
                coefficient * witness[name]
                for name, coefficient in coefficients.items()
            )
            if left > bound or (item["strict"] and left == bound):
                errors.append("linear counterexample violates a recorded constraint")
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("linear counterexample contents are malformed")
    return errors


def verify_recorded_outer_reduction(
    proof: dict[str, Any],
    source_expression_sha256: str,
) -> list[str]:
    outer = proof.get("outer_reduction")
    if outer is None:
        return []
    errors: list[str] = []
    if outer.get("rule") not in {
        "linear_skeleton_outer_reduction_v1",
        "certified_bounded_product_linear_envelope_v1",
        "certified_square_tangent_secant_envelope_v1",
    }:
        errors.append("outer reduction rule is invalid")
    if outer.get("source_expression_sha256") != source_expression_sha256:
        errors.append("outer reduction source expression hash is invalid")
    for bound in outer.get("bounds", []):
        variable = bound.get("variable")
        if not isinstance(variable, str):
            errors.append("outer reduction bound variable is malformed")
            continue
        for direction in ("lower", "upper"):
            value = bound.get(direction)
            recorded_proof = bound.get(direction + "_proof")
            if value is None:
                if recorded_proof is not None:
                    errors.append("outer reduction records a proof for a missing bound")
                continue
            try:
                exact_value = parse_fraction(value)
                certificate = recorded_proof["proof"]["certificate"]
                constraints = certificate["constraints"]
                final = constraints[-1]
                expected_coefficients = {
                    variable: fraction_text(
                        Fraction(1) if direction == "lower" else Fraction(-1)
                    )
                }
                expected_bound = fraction_text(
                    exact_value if direction == "lower" else -exact_value
                )
            except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
                errors.append("outer reduction bound proof is malformed")
                continue
            if recorded_proof.get("outcome") != "CERTIFIED":
                errors.append("outer reduction bound is not certified")
            errors.extend(
                "outer reduction bound certificate: " + error
                for error in verify_recorded_linear_certificate(certificate)
            )
            if (
                final.get("coefficients") != expected_coefficients
                or final.get("bound") != expected_bound
                or final.get("strict") is not True
            ):
                errors.append("outer reduction bound proof checks the wrong inequality")
    return errors


def verify_recorded_convex_certificate(certificate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if certificate.get("kind") != "convex_dual_bound_v1":
        errors.append("convex certificate kind is invalid")
    try:
        constraints = certificate["constraints"]
        if not isinstance(constraints, list):
            raise ValueError
        squares = [_fraction_mapping(item["square"]) for item in constraints]
        linears = [_fraction_mapping(item["linear"]) for item in constraints]
        constants = [parse_fraction(item["constant"]) for item in constraints]
        strict = [item.get("relation") == "< 0" for item in constraints]
        if any(item.get("relation") not in {"<= 0", "< 0"} for item in constraints):
            errors.append("convex certificate relation is malformed")
        multipliers = [parse_fraction(value) for value in certificate["multipliers"]]
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("convex certificate contents are malformed")
        return errors
    if len(multipliers) != len(constraints):
        errors.append("convex certificate multiplier count is incorrect")
        return errors
    if any(value < 0 for value in multipliers):
        errors.append("convex certificate contains a negative multiplier")
    if not any(value > 0 for value in multipliers):
        errors.append("convex certificate multipliers are all zero")
    if any(value < 0 for square in squares for value in square.values()):
        errors.append("convex certificate contains a nonconvex source constraint")

    combined_square: dict[str, Fraction] = {}
    combined_linear: dict[str, Fraction] = {}
    combined_constant = Fraction(0)
    combined_strict = False
    for square, linear, constant, multiplier in zip(
        squares, linears, constants, multipliers
    ):
        for name, value in square.items():
            combined_square[name] = combined_square.get(name, Fraction(0)) + multiplier * value
        for name, value in linear.items():
            combined_linear[name] = combined_linear.get(name, Fraction(0)) + multiplier * value
        combined_constant += multiplier * constant
    combined_strict = any(
        is_strict and multiplier > 0
        for is_strict, multiplier in zip(strict, multipliers)
    )
    combined_square = {
        name: value for name, value in sorted(combined_square.items()) if value
    }
    combined_linear = {
        name: value for name, value in sorted(combined_linear.items()) if value
    }
    lower_bound: Fraction | None = combined_constant
    for variable in sorted(set(combined_square) | set(combined_linear)):
        quadratic = combined_square.get(variable, Fraction(0))
        linear = combined_linear.get(variable, Fraction(0))
        if quadratic < 0 or (quadratic == 0 and linear != 0):
            lower_bound = None
            break
        if quadratic > 0:
            lower_bound -= linear * linear / (4 * quadratic)
    if lower_bound is None or lower_bound < 0 or (
        lower_bound == 0 and not combined_strict
    ):
        errors.append("convex weighted sum does not exclude the constrained region")
    expected_combined = {
        "square": {
            name: fraction_text(value) for name, value in combined_square.items()
        },
        "linear": {
            name: fraction_text(value) for name, value in combined_linear.items()
        },
        "constant": fraction_text(combined_constant),
        "relation": "< 0" if combined_strict else "<= 0",
    }
    if certificate.get("combined_quadratic") != expected_combined:
        errors.append("convex certificate combined quadratic is incorrect")
    expected_bound = fraction_text(lower_bound) if lower_bound is not None else None
    if certificate.get("global_lower_bound") != expected_bound:
        errors.append("convex certificate global lower bound is incorrect")
    return errors


def _serialized_expression_hash(expression: Any) -> str:
    return hashlib.sha256(json.dumps(
        expression,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _serialized_conjuncts(expression: Any) -> list[Any]:
    if (
        isinstance(expression, dict)
        and expression.get("type") == "op"
        and expression.get("op") == "and"
        and isinstance(expression.get("args"), list)
    ):
        result: list[Any] = []
        for argument in expression["args"]:
            result.extend(_serialized_conjuncts(argument))
        return result
    return [expression]


def _serialized_conjunction(expressions: list[Any]) -> Any:
    if not expressions:
        return {"type": "const", "value": True}
    if len(expressions) == 1:
        return expressions[0]
    return {"type": "op", "op": "and", "args": expressions}


def _verify_common_case_group(
    proof: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    reduction = proof.get("group_reduction")
    if reduction is None:
        return []
    if not isinstance(reduction, dict):
        return ["common case group reduction is malformed"]
    errors: list[str] = []
    if reduction.get("rule") != "common_conjunctive_case_group_v1":
        errors.append("common case group reduction rule is invalid")
    source = case.get("expression")
    shared = reduction.get("shared_expression")
    if not isinstance(source, dict) or reduction.get(
        "source_expression_sha256"
    ) != _serialized_expression_hash(source):
        errors.append("common case group source expression hash is invalid")
        return errors
    if not isinstance(shared, dict) or reduction.get(
        "shared_expression_sha256"
    ) != _serialized_expression_hash(shared):
        errors.append("common case group expression hash is invalid")
        return errors
    source_hashes = {
        _serialized_expression_hash(item) for item in _serialized_conjuncts(source)
    }
    shared_hashes = sorted(
        _serialized_expression_hash(item) for item in _serialized_conjuncts(shared)
    )
    if reduction.get("shared_conjunct_sha256") != shared_hashes:
        errors.append("common case group conjunct hashes are invalid")
    if not set(shared_hashes) <= source_hashes:
        errors.append("common case group is not a conjunction subset")
    if not isinstance(reduction.get("covered_case_count"), int) or reduction.get(
        "covered_case_count"
    ) <= 1:
        errors.append("common case group size is invalid")
    return errors


def _normalized_bound_values(record: Any) -> tuple[dict[str, Fraction], Fraction, bool]:
    if not isinstance(record, dict) or not isinstance(record.get("coefficients"), dict):
        raise ValueError("normalized bound is malformed")
    coefficients = {
        str(name): parse_fraction(value)
        for name, value in record["coefficients"].items()
    }
    bound = parse_fraction(record.get("bound"))
    strict = record.get("strict")
    if not isinstance(strict, bool) or not coefficients:
        raise ValueError("normalized bound is malformed")
    return coefficients, bound, strict


def _serialized_linear_form(
    expression: Any,
) -> tuple[dict[str, Fraction], Fraction]:
    if not isinstance(expression, dict):
        raise ValueError("linear expression is malformed")
    kind = expression.get("type")
    if kind == "const":
        value = expression.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError("linear constant is malformed")
        return {}, Fraction(str(value))
    if kind in {"var", "raw_ref"}:
        field = "name" if kind == "var" else "path"
        name = expression.get(field)
        if not isinstance(name, str):
            raise ValueError("linear variable is malformed")
        return {name: Fraction(1)}, Fraction(0)
    if kind != "op" or not isinstance(expression.get("args"), list):
        raise ValueError("linear operation is malformed")

    operation = expression.get("op")
    arguments = expression["args"]

    def add(
        left: tuple[dict[str, Fraction], Fraction],
        right: tuple[dict[str, Fraction], Fraction],
        scale: Fraction = Fraction(1),
    ) -> tuple[dict[str, Fraction], Fraction]:
        coefficients = dict(left[0])
        for name, value in right[0].items():
            coefficients[name] = coefficients.get(name, Fraction(0)) + scale * value
        return (
            {name: value for name, value in coefficients.items() if value},
            left[1] + scale * right[1],
        )

    if operation == "+":
        result = ({}, Fraction(0))
        for argument in arguments:
            result = add(result, _serialized_linear_form(argument))
        return result
    if operation == "-" and arguments:
        if len(arguments) == 1:
            coefficients, constant = _serialized_linear_form(arguments[0])
            return (
                {name: -value for name, value in coefficients.items()},
                -constant,
            )
        result = _serialized_linear_form(arguments[0])
        for argument in arguments[1:]:
            result = add(result, _serialized_linear_form(argument), Fraction(-1))
        return result
    if operation == "*" and len(arguments) == 2:
        left = _serialized_linear_form(arguments[0])
        right = _serialized_linear_form(arguments[1])
        if left[0] and right[0]:
            raise ValueError("variable multiplication is not linear")
        variable, constant = (left, right[1]) if left[0] else (right, left[1])
        return (
            {name: value * constant for name, value in variable[0].items()},
            variable[1] * constant,
        )
    if operation == "/" and len(arguments) == 2:
        numerator = _serialized_linear_form(arguments[0])
        denominator = _serialized_linear_form(arguments[1])
        if denominator[0] or denominator[1] == 0:
            raise ValueError("linear divisor is malformed")
        scale = Fraction(1) / denominator[1]
        return (
            {name: value * scale for name, value in numerator[0].items()},
            numerator[1] * scale,
        )
    raise ValueError("operation is not linear")


def _serialized_normalized_linear_bound(
    expression: Any,
) -> tuple[dict[str, Fraction], Fraction, bool]:
    if not isinstance(expression, dict) or expression.get("type") != "op":
        raise ValueError("linear comparison is malformed")
    operation = expression.get("op")
    arguments = expression.get("args")
    if operation not in {"<", "<=", ">", ">="} or not isinstance(
        arguments, list
    ) or len(arguments) != 2:
        raise ValueError("linear comparison is malformed")
    left = _serialized_linear_form(arguments[0])
    right = _serialized_linear_form(arguments[1])
    coefficients = dict(left[0])
    for name, value in right[0].items():
        coefficients[name] = coefficients.get(name, Fraction(0)) - value
    constant = left[1] - right[1]
    if operation in {">", ">="}:
        coefficients = {name: -value for name, value in coefficients.items()}
        constant = -constant
    coefficients = {
        name: value for name, value in sorted(coefficients.items()) if value
    }
    if not coefficients:
        raise ValueError("linear comparison has no variable")
    scale = abs(next(iter(coefficients.values())))
    return (
        {name: value / scale for name, value in coefficients.items()},
        -constant / scale,
        operation in {"<", ">"},
    )


def _verify_constraint_removals(records: Any) -> list[str]:
    if not isinstance(records, list):
        return ["constraint removal records are malformed"]
    errors: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            errors.append("constraint removal record is malformed")
            continue
        removed = record.get("removed_expression")
        if not isinstance(removed, dict) or record.get(
            "removed_expression_sha256"
        ) != _serialized_expression_hash(removed):
            errors.append("removed constraint expression hash is invalid")
            continue
        rule = record.get("rule")
        if rule == "exact_duplicate_constraint_v1":
            if record.get("retained_expression_sha256") != record.get(
                "removed_expression_sha256"
            ):
                errors.append("duplicate constraint removal is not identical")
            continue
        if rule != "normalized_linear_bound_dominance_v1":
            errors.append("constraint removal rule is invalid")
            continue
        dominating = record.get("dominating_expression")
        if not isinstance(dominating, dict) or record.get(
            "dominating_expression_sha256"
        ) != _serialized_expression_hash(dominating):
            errors.append("dominating constraint expression hash is invalid")
        try:
            removed_coefficients, removed_bound, removed_strict = (
                _normalized_bound_values(record.get("removed_normalized_bound"))
            )
            dominating_coefficients, dominating_bound, dominating_strict = (
                _normalized_bound_values(record.get("dominating_normalized_bound"))
            )
        except (TypeError, ValueError, ZeroDivisionError):
            errors.append("normalized constraint dominance record is malformed")
            continue
        try:
            if (
                removed_coefficients,
                removed_bound,
                removed_strict,
            ) != _serialized_normalized_linear_bound(removed):
                errors.append("removed normalized bound does not match its expression")
            if (
                dominating_coefficients,
                dominating_bound,
                dominating_strict,
            ) != _serialized_normalized_linear_bound(dominating):
                errors.append(
                    "dominating normalized bound does not match its expression"
                )
        except (TypeError, ValueError, ZeroDivisionError):
            errors.append("constraint dominance expression is not linear")
        if removed_coefficients != dominating_coefficients:
            errors.append("constraint dominance uses different normalized left sides")
        if not (
            dominating_bound < removed_bound
            or (
                dominating_bound == removed_bound
                and (dominating_strict or not removed_strict)
            )
        ):
            errors.append("dominating constraint is not stronger")
    return errors


def _verify_smt_stage(
    stage: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    proof = stage.get("proof") or {}
    rule = proof.get("rule")
    source_expression = case.get("expression")
    if not isinstance(source_expression, dict):
        return ["source case expression is malformed"]
    source_hash = _serialized_expression_hash(source_expression)
    if proof.get("source_expression_sha256") != source_hash:
        errors.append("source expression hash is invalid")

    if rule == "exact_local_feasibility_replay_v1":
        if stage.get("outcome") != "DEFERRED":
            errors.append("local feasibility replay must remain deferred")
        if proof.get("solver_status") != "sat":
            errors.append("local feasibility replay solver status is invalid")
        try:
            replayed = replay_serialized_boolean_expression(
                source_expression,
                proof.get("exact_values"),
            )
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"local feasibility replay is malformed: {exc}")
        else:
            if replayed is not True or proof.get("exact_replay") is not True:
                errors.append("local feasibility values do not satisfy the source case")
        return errors

    if rule != "solver_selected_subset_recertification_v1":
        if stage.get("outcome") == "CERTIFIED":
            errors.append("certified solver fallback proof rule is invalid")
        return errors
    if proof.get("solver_status") != "unsat":
        errors.append("selected subset solver status is invalid")

    source_constraints = _serialized_conjuncts(source_expression)
    if proof.get("source_constraint_count") != len(source_constraints):
        errors.append("source constraint count is invalid")
    indices = proof.get("selected_indices")
    if (
        not isinstance(indices, list)
        or any(not isinstance(index, int) for index in indices)
        or indices != sorted(set(indices))
        or any(index < 0 or index >= len(source_constraints) for index in indices)
    ):
        errors.append("selected constraint indices are malformed")
        return errors
    selected = [source_constraints[index] for index in indices]
    if proof.get("selected_constraints") != selected:
        errors.append("selected constraints are not the recorded source subset")
    selected_expression = _serialized_conjunction(selected)
    if proof.get("selected_expression") != selected_expression:
        errors.append("selected constraint expression is invalid")
    selected_hash = _serialized_expression_hash(selected_expression)
    if proof.get("selected_expression_sha256") != selected_hash:
        errors.append("selected constraint expression hash is invalid")

    if stage.get("outcome") != "CERTIFIED":
        return errors
    attempt = proof.get("certificate_attempt")
    if not isinstance(attempt, dict) or attempt.get("outcome") != "CERTIFIED":
        errors.append("selected constraint certificate attempt is missing")
        return errors
    if proof.get("certifying_checker") != attempt.get("checker"):
        errors.append("selected constraint certifying checker is inconsistent")
    certificate = (attempt.get("proof") or {}).get("certificate")
    if not isinstance(certificate, dict):
        errors.append("selected constraint certificate is missing")
        return errors
    kind = certificate.get("kind")
    if kind == "linear_infeasibility_weights_v1":
        errors.extend(
            "selected constraint linear certificate: " + error
            for error in verify_recorded_linear_certificate(certificate)
        )
    elif kind == "convex_dual_bound_v1":
        errors.extend(
            "selected constraint convex certificate: " + error
            for error in verify_recorded_convex_certificate(certificate)
        )
    else:
        errors.append("selected constraint certificate kind is invalid")
    errors.extend(
        "selected constraint outer reduction: " + error
        for error in verify_recorded_outer_reduction(
            attempt.get("proof") or {},
            selected_hash,
        )
    )
    return errors


def _verify_smt_reachability_query(query: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(query, dict):
        return ["SMT reachability query is malformed"]
    expression = query.get("query_expression")
    if not isinstance(expression, dict):
        errors.append("SMT reachability query expression is malformed")
    elif query.get("query_expression_sha256") != _serialized_expression_hash(
        expression
    ):
        errors.append("SMT reachability query expression hash is invalid")
    smt2 = query.get("query_smt2")
    if not isinstance(smt2, str) or not smt2:
        errors.append("SMT reachability query text is missing")
    elif query.get("query_smt2_sha256") != hashlib.sha256(
        smt2.encode("utf-8")
    ).hexdigest():
        errors.append("SMT reachability query text hash is invalid")
    status = query.get("solver_status")
    if status == "unsat":
        proof_text = query.get("z3_proof")
        if not isinstance(proof_text, str) or not proof_text:
            errors.append("SMT reachability no solution proof is missing")
        elif query.get("z3_proof_sha256") != hashlib.sha256(
            proof_text.encode("utf-8")
        ).hexdigest():
            errors.append("SMT reachability no solution proof hash is invalid")
    elif status == "sat":
        try:
            replayed = replay_serialized_boolean_expression(
                expression,
                query.get("exact_values"),
            )
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"SMT reachability trace is malformed: {exc}")
        else:
            if replayed is not True or query.get("exact_replay") is not True:
                errors.append("SMT reachability trace does not satisfy its query")
    elif status not in {"unknown", "unsupported"}:
        errors.append("SMT reachability solver status is invalid")
    return errors


def _recheck_smt_no_solution(query: dict[str, Any]) -> list[str]:
    if z3 is None:
        return ["z3-solver is unavailable for SMT reachability proof replay"]
    smt2 = query.get("query_smt2")
    if not isinstance(smt2, str) or not smt2:
        return ["SMT reachability query text is missing"]
    try:
        assertions = z3.parse_smt2_string(smt2)
        solver = z3.Solver()
        solver.add(assertions)
        result = solver.check()
    except Exception as exc:  # pragma: no cover - fail-closed boundary
        return [f"SMT reachability proof replay failed: {exc}"]
    if result != z3.unsat:
        return [f"SMT reachability no solution result did not replay: {result}"]
    return []


def _verify_smt_reachability_stage(
    stage: dict[str, Any],
    case: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    proof = stage.get("proof") or {}
    if stage.get("outcome") == "DEFERRED" and not proof:
        return []
    rule = proof.get("rule")
    source_expression = case.get("expression")
    reachability_expression = case.get("reachability_expression")
    if not isinstance(source_expression, dict):
        return ["SMT reachability source expression is malformed"]
    if not isinstance(reachability_expression, dict):
        return ["SMT reachability expression is malformed"]
    if proof.get("case_expression_sha256") != _serialized_expression_hash(
        source_expression
    ):
        errors.append("SMT reachability source expression hash is invalid")
    if proof.get("reachability_expression_sha256") != _serialized_expression_hash(
        reachability_expression
    ):
        errors.append("SMT reachability expression hash is invalid")

    if rule == "smt_finite_prefix_counterexample_v1":
        if stage.get("outcome") != "VIOLATION":
            errors.append("SMT reachability trace must report a violation")
        trace_query = proof.get("trace_query")
        errors.extend(_verify_smt_reachability_query(trace_query))
        if isinstance(trace_query, dict) and trace_query.get("solver_status") != "sat":
            errors.append("SMT reachability trace is not a solution")
        return errors

    if rule != "smt_finite_prefix_and_inductive_exclusion_v1":
        if stage.get("outcome") in {"CERTIFIED", "VIOLATION"}:
            errors.append("SMT reachability proof rule is invalid")
        return errors
    attempts = proof.get("depth_attempts")
    if not isinstance(attempts, list):
        return errors + ["SMT reachability depth attempts are malformed"]
    for attempt in attempts:
        if not isinstance(attempt, dict):
            errors.append("SMT reachability depth attempt is malformed")
            continue
        for query in attempt.get("base_queries", []):
            errors.extend(_verify_smt_reachability_query(query))
        induction = attempt.get("induction_query")
        if induction is not None:
            errors.extend(_verify_smt_reachability_query(induction))
    if stage.get("outcome") == "CERTIFIED":
        certified = [
            attempt for attempt in attempts if attempt.get("proved") is True
        ]
        if not certified:
            errors.append("SMT reachability proof has no certified depth")
        for attempt in certified:
            bases = attempt.get("base_queries")
            induction = attempt.get("induction_query")
            if not isinstance(bases, list) or not bases:
                errors.append("SMT reachability certified depth has no base query")
            elif any(query.get("solver_status") != "unsat" for query in bases):
                errors.append("SMT reachability base query is not proved impossible")
            else:
                for query in bases:
                    errors.extend(_recheck_smt_no_solution(query))
            if not isinstance(induction, dict) or induction.get(
                "solver_status"
            ) != "unsat":
                errors.append("SMT reachability induction query is not proved impossible")
            else:
                errors.extend(_recheck_smt_no_solution(induction))
    return errors


def _verify_relational_invariant_stage(
    stage: dict[str, Any],
    case: dict[str, Any],
    shared_regions: dict[str, Any] | None = None,
    shared_safety_queries: dict[str, Any] | None = None,
) -> list[str]:
    proof = stage.get("proof") or {}
    if stage.get("outcome") != "CERTIFIED":
        return []
    errors: list[str] = []
    if proof.get("rule") == "shared_relational_inductive_invariant_v2":
        source_expression = case.get("expression")
        reachability_expression = case.get("reachability_expression")
        if not isinstance(source_expression, dict) or proof.get(
            "case_expression_sha256"
        ) != _serialized_expression_hash(source_expression):
            errors.append("shared relational source expression hash is invalid")
        if not isinstance(reachability_expression, dict) or proof.get(
            "reachability_expression_sha256"
        ) != _serialized_expression_hash(reachability_expression):
            errors.append("shared relational reachability expression hash is invalid")
        region_key = proof.get("shared_reachable_region_sha256")
        safety_key = proof.get("merged_safety_query_sha256")
        region = (shared_regions or {}).get(region_key)
        safety = (shared_safety_queries or {}).get(safety_key)
        if not isinstance(region, dict):
            errors.append("shared reachable region reference is missing")
        if not isinstance(safety, dict):
            errors.append("merged relational safety query reference is missing")
        else:
            if safety.get("context_sha256") != region_key:
                errors.append("merged relational safety query context is inconsistent")
            if case.get("case_id") not in safety.get("case_ids", []):
                errors.append("merged relational safety query omits its case")
            coverage = [
                item
                for item in safety.get("covered_case_expressions", [])
                if isinstance(item, dict)
                and item.get("case_id") == case.get("case_id")
                and item.get("expression_sha256")
                == proof.get("case_safety_expression_sha256")
                and item.get("merge_rule") == proof.get("merge_rule")
            ]
            if not coverage:
                errors.append("merged relational safety query has no case coverage")
            query = safety.get("query")
            if not isinstance(query, dict) or query.get("solver_status") != "unsat":
                errors.append("merged relational safety obligation is not proved impossible")
        return errors
    if proof.get("rule") != "relational_inductive_invariant_v1":
        return ["relational invariant proof rule is invalid"]
    source_expression = case.get("expression")
    reachability_expression = case.get("reachability_expression")
    if not isinstance(source_expression, dict):
        errors.append("relational invariant source expression is malformed")
    elif proof.get("case_expression_sha256") != _serialized_expression_hash(
        source_expression
    ):
        errors.append("relational invariant source expression hash is invalid")
    if not isinstance(reachability_expression, dict):
        errors.append("relational invariant reachability expression is malformed")
    elif proof.get(
        "reachability_expression_sha256"
    ) != _serialized_expression_hash(reachability_expression):
        errors.append("relational invariant reachability expression hash is invalid")

    transition = proof.get("transition_expression")
    if not isinstance(transition, dict):
        errors.append("relational transition expression is malformed")
    elif proof.get("transition_expression_sha256") != _serialized_expression_hash(
        transition
    ):
        errors.append("relational transition expression hash is invalid")

    invariants = proof.get("invariant")
    invariant_hashes = proof.get("invariant_sha256")
    if not isinstance(invariants, list) or not invariants:
        errors.append("relational invariant is missing")
        invariants = []
    if not isinstance(invariant_hashes, list) or len(invariant_hashes) != len(
        invariants
    ):
        errors.append("relational invariant hashes are malformed")
        invariant_hashes = []
    for index, expression in enumerate(invariants):
        if not isinstance(expression, dict):
            errors.append("relational invariant clause is malformed")
        elif index < len(invariant_hashes) and invariant_hashes[
            index
        ] != _serialized_expression_hash(expression):
            errors.append("relational invariant clause hash is invalid")

    initiation = proof.get("initiation_queries")
    preservation = proof.get("preservation_queries")
    if not isinstance(initiation, list):
        errors.append("relational invariant initiation queries are malformed")
        initiation = []
    if not isinstance(preservation, list):
        errors.append("relational invariant preservation queries are malformed")
        preservation = []
    initiated_hashes = {
        item.get("candidate_sha256")
        for item in initiation
        if isinstance(item, dict)
    }
    preserved_hashes = {
        item.get("candidate_sha256")
        for item in preservation
        if isinstance(item, dict)
    }
    for invariant_hash in invariant_hashes:
        if invariant_hash not in initiated_hashes:
            errors.append("relational invariant clause has no initiation proof")
        if invariant_hash not in preserved_hashes:
            errors.append("relational invariant clause has no preservation proof")
    for record in [*initiation, *preservation]:
        if not isinstance(record, dict):
            errors.append("relational invariant query record is malformed")
            continue
        candidate = record.get("candidate")
        if not isinstance(candidate, dict):
            errors.append("relational invariant query candidate is malformed")
        elif record.get("candidate_sha256") != _serialized_expression_hash(
            candidate
        ):
            errors.append("relational invariant query candidate hash is invalid")
        query = record.get("query")
        errors.extend(_verify_smt_reachability_query(query))
        if not isinstance(query, dict) or query.get("solver_status") != "unsat":
            errors.append("relational invariant obligation is not proved impossible")
        else:
            errors.extend(_recheck_smt_no_solution(query))

    safety_query = proof.get("safety_query")
    errors.extend(_verify_smt_reachability_query(safety_query))
    if not isinstance(safety_query, dict) or safety_query.get(
        "solver_status"
    ) != "unsat":
        errors.append("relational invariant safety obligation is not proved impossible")
    else:
        errors.extend(_recheck_smt_no_solution(safety_query))
    return errors


def _serialized_conjunct_hashes(expression: dict[str, Any]) -> set[str]:
    if expression.get("type") == "op" and expression.get("op") == "and":
        hashes: set[str] = set()
        for argument in expression.get("args", []):
            if isinstance(argument, dict):
                hashes.update(_serialized_conjunct_hashes(argument))
        return hashes
    return {_serialized_expression_hash(expression)}


def _serialized_disjunct_hashes(expression: dict[str, Any]) -> set[str]:
    if expression.get("type") == "op" and expression.get("op") == "or":
        hashes: set[str] = set()
        for argument in expression.get("args", []):
            if isinstance(argument, dict):
                hashes.update(_serialized_disjunct_hashes(argument))
        return hashes
    return {_serialized_expression_hash(expression)}


def _verify_shared_candidate_batches(
    records: Any,
    invariant_hashes: list[Any],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(records, list) or not records:
        return [f"shared reachable region {label} batches are malformed"]
    covered: set[Any] = set()
    for record in records:
        if not isinstance(record, dict):
            errors.append(f"shared reachable region {label} batch is malformed")
            continue
        candidates = record.get("candidates")
        candidate_hashes = record.get("candidate_sha256")
        if not isinstance(candidates, list) or not isinstance(candidate_hashes, list):
            errors.append(f"shared reachable region {label} candidates are malformed")
            continue
        if len(candidates) != len(candidate_hashes):
            errors.append(f"shared reachable region {label} candidate hashes are malformed")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or index >= len(
                candidate_hashes
            ) or candidate_hashes[index] != _serialized_expression_hash(candidate):
                errors.append(f"shared reachable region {label} candidate hash is invalid")
            elif index < len(candidate_hashes):
                covered.add(candidate_hashes[index])
        query = record.get("query")
        errors.extend(_verify_smt_reachability_query(query))
        if not isinstance(query, dict) or query.get("solver_status") != "unsat":
            errors.append(f"shared reachable region {label} batch is not proved impossible")
        else:
            errors.extend(_recheck_smt_no_solution(query))
    for invariant_hash in invariant_hashes:
        if invariant_hash not in covered:
            errors.append(f"shared reachable region clause has no {label} proof")
    return errors


def _verify_shared_reachability(
    analysis: dict[str, Any],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    shared = analysis.get("shared_reachability") or {}
    regions = shared.get("regions") or {}
    safety_queries = shared.get("safety_queries") or {}
    if not isinstance(regions, dict):
        return ["shared reachable regions are malformed"], {}, {}
    if not isinstance(safety_queries, dict):
        return ["shared relational safety queries are malformed"], regions, {}

    for key, region in regions.items():
        if not isinstance(region, dict):
            errors.append("shared reachable region is malformed")
            continue
        context = region.get("context")
        if not isinstance(context, dict):
            errors.append("shared reachable region context is malformed")
        else:
            context_hash = hashlib.sha256(json.dumps(
                context,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if key != context_hash or region.get("context_sha256") != context_hash:
                errors.append("shared reachable region context hash is invalid")
        rule = region.get("rule")
        if rule not in {
            "shared_relational_reachable_region_v1",
            "shared_relational_reachable_region_v2",
        }:
            errors.append("shared reachable region proof rule is invalid")
        for name in ("initial_expression", "domain_expression", "transition_expression"):
            expression = region.get(name)
            if not isinstance(expression, dict) or region.get(
                name + "_sha256"
            ) != _serialized_expression_hash(expression):
                errors.append(f"shared reachable region {name} hash is invalid")

        invariants = region.get("invariant")
        invariant_hashes = region.get("invariant_sha256")
        if not isinstance(invariants, list) or not invariants:
            errors.append("shared reachable region invariant is missing")
            invariants = []
        if not isinstance(invariant_hashes, list) or len(invariant_hashes) != len(
            invariants
        ):
            errors.append("shared reachable region invariant hashes are malformed")
            invariant_hashes = []
        for index, expression in enumerate(invariants):
            if not isinstance(expression, dict) or index >= len(
                invariant_hashes
            ) or invariant_hashes[index] != _serialized_expression_hash(expression):
                errors.append("shared reachable region invariant clause hash is invalid")

        if rule == "shared_relational_reachable_region_v2":
            errors.extend(_verify_shared_candidate_batches(
                region.get("initiation_batches"),
                invariant_hashes,
                "initiation",
            ))
            errors.extend(_verify_shared_candidate_batches(
                region.get("preservation_batches"),
                invariant_hashes,
                "preservation",
            ))
        else:
            initiation = region.get("initiation_queries")
            preservation = region.get("preservation_queries")
            if not isinstance(initiation, list):
                errors.append("shared reachable region initiation queries are malformed")
                initiation = []
            if not isinstance(preservation, list):
                errors.append("shared reachable region preservation queries are malformed")
                preservation = []
            initiated_hashes = {
                item.get("candidate_sha256")
                for item in initiation
                if isinstance(item, dict)
            }
            preserved_hashes = {
                item.get("candidate_sha256")
                for item in preservation
                if isinstance(item, dict)
            }
            for invariant_hash in invariant_hashes:
                if invariant_hash not in initiated_hashes:
                    errors.append("shared reachable region clause has no initiation proof")
                if invariant_hash not in preserved_hashes:
                    errors.append("shared reachable region clause has no preservation proof")
            for record in [*initiation, *preservation]:
                if not isinstance(record, dict):
                    errors.append("shared reachable region query record is malformed")
                    continue
                candidate = record.get("candidate")
                if not isinstance(candidate, dict) or record.get(
                    "candidate_sha256"
                ) != _serialized_expression_hash(candidate):
                    errors.append("shared reachable region candidate hash is invalid")
                query = record.get("query")
                errors.extend(_verify_smt_reachability_query(query))
                if not isinstance(query, dict) or query.get("solver_status") != "unsat":
                    errors.append("shared reachable region obligation is not proved impossible")
                else:
                    errors.extend(_recheck_smt_no_solution(query))

        for removal in region.get("implication_removals", []):
            if not isinstance(removal, dict):
                errors.append("shared reachable region implication removal is malformed")
                continue
            candidate = removal.get("removed_candidate")
            if not isinstance(candidate, dict) or removal.get(
                "removed_candidate_sha256"
            ) != _serialized_expression_hash(candidate):
                errors.append("removed reachable region constraint hash is invalid")
            query = removal.get("query")
            errors.extend(_verify_smt_reachability_query(query))
            if not isinstance(query, dict) or query.get("solver_status") != "unsat":
                errors.append("reachable region constraint removal is not proved")
            else:
                errors.extend(_recheck_smt_no_solution(query))

    for key, safety in safety_queries.items():
        if not isinstance(safety, dict):
            errors.append("merged relational safety query is malformed")
            continue
        expression = safety.get("safety_expression")
        if not isinstance(expression, dict):
            errors.append("merged relational safety expression is malformed")
            continue
        expression_hash = _serialized_expression_hash(expression)
        if safety.get("safety_expression_sha256") != expression_hash:
            errors.append("merged relational safety expression hash is invalid")
        expected_key = hashlib.sha256(
            f"{safety.get('context_sha256')}:{expression_hash}".encode("utf-8")
        ).hexdigest()
        if key != expected_key:
            errors.append("merged relational safety query hash is invalid")
        if safety.get("context_sha256") not in regions:
            errors.append("merged relational safety query has no reachable region")
        covered = safety.get("covered_case_expressions")
        if not isinstance(covered, list) or not covered:
            errors.append("merged relational safety query has no covered cases")
            covered = []
        source_conjuncts = _serialized_conjunct_hashes(expression)
        source_disjuncts = _serialized_disjunct_hashes(expression)
        for coverage in covered:
            if not isinstance(coverage, dict):
                errors.append("merged relational safety case coverage is malformed")
                continue
            covered_expression = coverage.get("expression")
            if not isinstance(covered_expression, dict) or coverage.get(
                "expression_sha256"
            ) != _serialized_expression_hash(covered_expression):
                errors.append("merged relational safety case expression hash is invalid")
                continue
            rule = coverage.get("merge_rule")
            covered_conjuncts = _serialized_conjunct_hashes(covered_expression)
            if rule == "identical_expression":
                if covered_conjuncts != source_conjuncts:
                    errors.append("identical relational safety merge is not identical")
            elif rule == "conjunct_containment":
                if not source_conjuncts <= covered_conjuncts:
                    errors.append("relational safety containment merge is invalid")
            elif rule == "new_query":
                if coverage.get("expression_sha256") != expression_hash:
                    errors.append("new relational safety query coverage is inconsistent")
            elif rule == "group_disjunction":
                if coverage.get("expression_sha256") not in source_disjuncts:
                    errors.append("grouped relational safety coverage is invalid")
            else:
                errors.append("relational safety merge rule is invalid")
        query = safety.get("query")
        errors.extend(_verify_smt_reachability_query(query))
        if isinstance(query, dict) and query.get("query_expression") != expression:
            errors.append("merged relational safety query expression does not match")
        if isinstance(query, dict) and query.get("solver_status") == "unsat":
            errors.extend(_recheck_smt_no_solution(query))
    return errors, regions, safety_queries


def verify_recorded_optimization_certificates(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(analysis, dict):
        return ["recorded analysis is malformed"]
    shared_errors, shared_regions, shared_safety_queries = (
        _verify_shared_reachability(analysis)
    )
    errors.extend(shared_errors)
    for property_record in analysis.get("properties", []):
        property_id = property_record.get("property_id", "unknown")
        reduction = property_record.get("reduction") or {}
        if reduction.get("outcome") != "DEFERRED":
            if reduction.get("kind") != "full_sysml_interval_reduction_v3":
                errors.append(f"property {property_id} reduction kind is invalid")
            counterexample = reduction.get("interval_counterexample")
            if not isinstance(counterexample, dict):
                errors.append(f"property {property_id} interval counterexample is missing")
            else:
                observed_hash = hashlib.sha256(json.dumps(
                    counterexample,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                if observed_hash != reduction.get("interval_counterexample_sha256"):
                    errors.append(f"property {property_id} interval counterexample hash is invalid")
            sampled_counterexample = reduction.get("sampled_point_counterexample")
            if not isinstance(sampled_counterexample, dict):
                errors.append(f"property {property_id} sampled point counterexample is missing")
            else:
                sampled_hash = hashlib.sha256(json.dumps(
                    sampled_counterexample,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                if sampled_hash != reduction.get("sampled_point_counterexample_sha256"):
                    errors.append(f"property {property_id} sampled point counterexample hash is invalid")
            pre_case_reduction = reduction.get("pre_case_constraint_reduction") or {}
            if pre_case_reduction.get(
                "rule"
            ) != "recursive_canonical_constraint_reduction_v1":
                errors.append(f"property {property_id} pre case reduction rule is invalid")
            for removal_error in (
                _verify_constraint_removals(
                    pre_case_reduction.get("sampled_point_removed_constraints")
                )
                + _verify_constraint_removals(
                    pre_case_reduction.get("physical_interval_removed_constraints")
                )
            ):
                errors.append(
                    f"property {property_id} pre case reduction: {removal_error}"
                )
            endpoint_checks = reduction.get("endpoint_checks")
            if not isinstance(endpoint_checks, list) or any(
                item.get("matches") is not True for item in endpoint_checks
            ):
                errors.append(f"property {property_id} endpoint checks are incomplete")
            inventory = reduction.get("equation_inventory")
            inventory_by_target = {
                item.get("target"): item
                for item in inventory or []
                if isinstance(item, dict)
            }
            for trajectory in reduction.get("trajectories", []):
                target = trajectory.get("physical_value")
                if inventory_by_target.get(target, {}).get("included") is not True:
                    errors.append(
                        f"property {property_id} physical equation {target} is omitted"
                    )
            for mapping in reduction.get("sensor_to_physical_mappings", []):
                required = [mapping.get("physical_value")] + [
                    item.get("target") for item in mapping.get("equation_path", [])
                ]
                for target in required:
                    if inventory_by_target.get(target, {}).get("included") is not True:
                        errors.append(
                            f"property {property_id} sensor mapping equation {target} is omitted"
                        )
            if any(
                item.get("matched_controller_call_guard") is not True
                for item in reduction.get("sensor_mapping_guard_evidence", [])
            ):
                errors.append(
                    f"property {property_id} sensor mapping guard evidence is incomplete"
                )
            coverage = reduction.get("case_coverage") or {}
            if coverage.get("complete") is not True:
                errors.append(f"property {property_id} case coverage is incomplete")
            coverage_cases = coverage.get("cases") or []
            recorded_cases = property_record.get("cases") or []
            if [item.get("case_id") for item in coverage_cases] != [
                item.get("case_id") for item in recorded_cases
            ]:
                errors.append(f"property {property_id} case identifiers do not match coverage")
            coverage_by_id = {item.get("case_id"): item for item in coverage_cases}
            for case in recorded_cases:
                source = coverage_by_id.get(case.get("case_id"), {})
                if source.get("expression") != case.get("expression"):
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} expression does not match coverage"
                    )
                if source.get("reachability_expression") is not None:
                    reachability_hash = hashlib.sha256(json.dumps(
                        source["reachability_expression"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")).hexdigest()
                    if reachability_hash != source.get(
                        "reachability_expression_sha256"
                    ):
                        errors.append(
                            f"property {property_id} case {case.get('case_id')} reachability expression hash is invalid"
                        )
                if source.get("time_reduction") != case.get("time_reduction"):
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} time reduction does not match coverage"
                    )
                if source.get("obligation") != case.get("obligation"):
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} obligation does not match coverage"
                    )
                constraint_reduction = source.get("constraint_reduction") or {}
                if constraint_reduction.get(
                    "rule"
                ) != "canonical_conjunction_reduction_v1":
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} constraint reduction rule is invalid"
                    )
                for removal_error in (
                    _verify_constraint_removals(
                        constraint_reduction.get("removed_constraints")
                    )
                    + _verify_constraint_removals(
                        constraint_reduction.get("reachability_removed_constraints")
                    )
                ):
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} constraint reduction: {removal_error}"
                    )
                if isinstance(source.get("expression"), dict):
                    case_hash = hashlib.sha256(json.dumps(
                        source["expression"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")).hexdigest()
                    if case_hash != source.get("expression_sha256"):
                        errors.append(
                            f"property {property_id} case {case.get('case_id')} expression hash is invalid"
                        )

        stages = [
            (case, stage)
            for case in property_record.get("cases", [])
            for stage in case.get("progression", [])
        ]
        for case, stage in stages:
            checker = stage.get("checker")
            proof = stage.get("proof") or {}
            if checker == "smt_fallback":
                errors.extend(
                    f"property {property_id} smt fallback certificate: {error}"
                    for error in _verify_smt_stage(stage, case)
                )
                continue
            if checker == "smt_reachability":
                errors.extend(
                    f"property {property_id} smt reachability certificate: {error}"
                    for error in _verify_smt_reachability_stage(stage, case)
                )
                continue
            if checker == "relational_invariant":
                errors.extend(
                    f"property {property_id} relational invariant certificate: {error}"
                    for error in _verify_relational_invariant_stage(
                        stage,
                        case,
                        shared_regions,
                        shared_safety_queries,
                    )
                )
                continue
            if stage.get("outcome") == "VIOLATION":
                if checker != "reachability_linear":
                    errors.append(
                        f"property {property_id} unsupported violation checker {checker}"
                    )
                    continue
                if proof.get("rule") != "exact_finite_prefix_counterexample_v1":
                    errors.append(
                        f"property {property_id} reachability violation proof rule is invalid"
                    )
                    continue
                obligations = proof.get("base_obligations", [])
                replayed = [
                    obligation.get("attempt", {})
                    for obligation in obligations
                    if obligation.get("attempt", {}).get("outcome") == "VIOLATION"
                ]
                if not replayed:
                    errors.append(
                        f"property {property_id} reachability violation has no replayed obligation"
                    )
                for attempt in replayed:
                    for error in verify_recorded_linear_counterexample(
                        attempt.get("proof") or {}
                    ):
                        errors.append(
                            f"property {property_id} reachability counterexample: {error}"
                        )
                continue
            if stage.get("outcome") != "CERTIFIED":
                continue
            certificate = proof.get("certificate")
            if checker == "linear":
                if not isinstance(certificate, dict):
                    stage_errors = ["linear proof certificate is missing"]
                else:
                    stage_errors = verify_recorded_linear_certificate(certificate)
            elif checker == "convex":
                if not isinstance(certificate, dict):
                    stage_errors = ["convex proof certificate is missing"]
                else:
                    stage_errors = verify_recorded_convex_certificate(certificate)
            elif checker in {"reachability_linear", "reachability_convex"}:
                method = (
                    "linear"
                    if checker == "reachability_linear"
                    else "convex"
                )
                if proof.get("rule") != "finite_prefix_and_inductive_case_exclusion_v1":
                    stage_errors = ["reachability proof rule is invalid"]
                elif proof.get("method") != method:
                    stage_errors = ["reachability proof method is invalid"]
                else:
                    stage_errors = []
                    certified_depths = [
                        item
                        for item in proof.get("depth_attempts", [])
                        if item.get("proved") is True
                    ]
                    if not certified_depths:
                        stage_errors.append("reachability proof has no certified depth")
                    for depth in certified_depths:
                        obligations = (
                            depth.get("base_obligations", [])
                            + depth.get("induction_obligations", [])
                        )
                        if not obligations:
                            stage_errors.append(
                                "reachability proof has no arithmetic obligations"
                            )
                        for obligation in obligations:
                            attempt = obligation.get("attempt") or {}
                            if obligation.get("outer_case_group") is True:
                                if (
                                    not isinstance(
                                        obligation.get("covered_case_first_id"),
                                        str,
                                    )
                                    or not isinstance(
                                        obligation.get("covered_case_last_id"),
                                        str,
                                    )
                                    or not isinstance(
                                        obligation.get("covered_case_count"),
                                        int,
                                    )
                                    or obligation.get("covered_case_count") <= 1
                                ):
                                    stage_errors.append(
                                        "reachability outer case group coverage is malformed"
                                    )
                            certificate = (attempt.get("proof") or {}).get(
                                "certificate"
                            )
                            if attempt.get("outcome") != "CERTIFIED":
                                stage_errors.append(
                                    "reachability arithmetic obligation is not certified"
                                )
                            elif (
                                (attempt.get("proof") or {}).get("rule")
                                == "exhaustive_case_split_empty_v1"
                            ):
                                if attempt.get("applicability_checks", {}).get(
                                    "arithmetic_case_count"
                                ) != 0:
                                    stage_errors.append(
                                        "empty reachability split has a nonzero case count"
                                    )
                            elif not isinstance(certificate, dict):
                                stage_errors.append(
                                    "reachability arithmetic certificate is missing"
                                )
                            elif certificate.get("kind") == "linear_infeasibility_weights_v1":
                                stage_errors.extend(
                                    "reachability linear certificate: " + error
                                    for error in verify_recorded_linear_certificate(
                                        certificate
                                    )
                                )
                                stage_errors.extend(
                                    "reachability outer reduction: " + error
                                    for error in verify_recorded_outer_reduction(
                                        attempt.get("proof") or {},
                                        str(obligation.get("expression_sha256", "")),
                                    )
                                )
                            elif certificate.get("kind") == "convex_dual_bound_v1":
                                stage_errors.extend(
                                    "reachability convex certificate: " + error
                                    for error in verify_recorded_convex_certificate(
                                        certificate
                                    )
                                )
                                stage_errors.extend(
                                    "reachability outer reduction: " + error
                                    for error in verify_recorded_outer_reduction(
                                        attempt.get("proof") or {},
                                        str(obligation.get("expression_sha256", "")),
                                    )
                                )
                            else:
                                stage_errors.append(
                                    "reachability certificate kind is invalid"
                                )
            else:
                continue
            if checker in {"linear", "convex"}:
                stage_errors.extend(_verify_common_case_group(proof, case))
            errors.extend(
                f"property {property_id} {checker} certificate: {error}"
                for error in stage_errors
            )
    return errors
