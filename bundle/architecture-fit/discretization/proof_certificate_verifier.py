"""Exact verifier for recorded linear and convex proof certificates."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

from .optimization_common import fraction_text, parse_fraction


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


def verify_recorded_optimization_certificates(analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(analysis, dict):
        return ["recorded analysis is malformed"]
    for property_record in analysis.get("properties", []):
        property_id = property_record.get("property_id", "unknown")
        reduction = property_record.get("reduction") or {}
        if reduction.get("outcome") != "DEFERRED":
            if reduction.get("kind") != "full_sysml_interval_reduction_v2":
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
                if source.get("time_reduction") != case.get("time_reduction"):
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} time reduction does not match coverage"
                    )
                if source.get("obligation") != case.get("obligation"):
                    errors.append(
                        f"property {property_id} case {case.get('case_id')} obligation does not match coverage"
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
            stage
            for case in property_record.get("cases", [])
            for stage in case.get("progression", [])
        ]
        for stage in stages:
            if stage.get("outcome") != "CERTIFIED":
                continue
            checker = stage.get("checker")
            proof = stage.get("proof") or {}
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
            else:
                continue
            errors.extend(
                f"property {property_id} {checker} certificate: {error}"
                for error in stage_errors
            )
    return errors
