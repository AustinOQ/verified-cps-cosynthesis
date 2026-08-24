"""Exact verifier for recorded linear and convex proof certificates."""

from __future__ import annotations

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
        for stage in property_record.get("progression", []):
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
