#!/usr/bin/env python3
"""Validation battery for the discretization-safety proof rules and checker."""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from certification.certificate import load_certificate as load_mdp_certificate
from certification.equations import Const, Op, Var

from .analysis import CHECKER_ORDER, analyze_model
from .certificate import certificate_hash, check_certificate, load_certificate
from .convex_checker import run_convex_checker, solve_convex_constraints
from .full_model_reduction import ReducedCase, expression_hash
from .linear_checker import run_linear_checker, solve_linear_constraints
from .optimization_common import QuadraticConstraint
from .proof_rules import expression_is_linear, prove_implication_exact
from .proof_rules import LinearInequality
from .proof_certificate_verifier import (
    verify_recorded_convex_certificate,
    verify_recorded_linear_certificate,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def reduced_case(case_id: str, expression) -> ReducedCase:
    return ReducedCase(
        case_id,
        expression,
        expression_hash(expression),
        (),
        "test",
        "test",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", nargs="+")
    args = parser.parse_args()

    x = Var("x")
    sound = prove_implication_exact(
        [Op("<=", (x, Const(0)))],
        Op("<=", (x, Const(1))),
        set(),
    )
    require(sound.get("proved") is True, "valid exact implication was rejected")
    unsound = prove_implication_exact(
        [Op("<=", (x, Const(1)))],
        Op("<=", (x, Const(0))),
        set(),
    )
    require(unsound.get("proved") is False, "invalid exact implication was accepted")
    linear, _detail = expression_is_linear(Op("*", (x, x)), set())
    require(linear is False, "variable multiplication was classified as linear")

    linear_problem = [
        LinearInequality.make({"x": Fraction(1)}, Fraction(0)),
        LinearInequality.make({"x": Fraction(-1)}, Fraction(-1)),
    ]
    linear_result = solve_linear_constraints(linear_problem, timeout_ms=250)
    require(linear_result.get("outcome") == "CERTIFIED", "linear proof was rejected")
    linear_certificate = linear_result["proof"]["certificate"]
    require(
        not verify_recorded_linear_certificate(linear_certificate),
        "linear proof certificate did not verify",
    )
    bad_linear_certificate = copy.deepcopy(linear_certificate)
    bad_linear_certificate["multipliers"][0] = "0/1"
    require(
        bool(verify_recorded_linear_certificate(bad_linear_certificate)),
        "invalid linear proof certificate was accepted",
    )
    strict_problem = [
        LinearInequality.make({"x": Fraction(1)}, Fraction(0), False),
        LinearInequality.make({"x": Fraction(-1)}, Fraction(0), True),
    ]
    strict_result = solve_linear_constraints(strict_problem, timeout_ms=250)
    require(
        strict_result.get("outcome") == "CERTIFIED",
        "exact strict linear proof was rejected",
    )

    convex_problem = [
        QuadraticConstraint.make({"x": Fraction(1)}, {}, Fraction(1))
    ]
    convex_result = solve_convex_constraints(convex_problem, timeout_ms=250)
    require(convex_result.get("outcome") == "CERTIFIED", "convex proof was rejected")
    convex_certificate = convex_result["proof"]["certificate"]
    require(
        not verify_recorded_convex_certificate(convex_certificate),
        "convex proof certificate did not verify",
    )
    bad_convex_certificate = copy.deepcopy(convex_certificate)
    bad_convex_certificate["global_lower_bound"] = "0/1"
    require(
        bool(verify_recorded_convex_certificate(bad_convex_certificate)),
        "invalid convex proof certificate was accepted",
    )
    strict_convex_result = solve_convex_constraints(
        [QuadraticConstraint.make({"x": Fraction(1)}, {}, Fraction(0), True)],
        timeout_ms=250,
    )
    require(
        strict_convex_result.get("outcome") == "CERTIFIED",
        "exact strict convex proof was rejected",
    )
    nonlinear_counterexample = Op(
        "<=",
        (Op("+", (Op("*", (x, x)), Const(1))), Const(0)),
    )
    require(
        run_linear_checker(
            reduced_case("nonlinear", nonlinear_counterexample),
            set(),
            timeout_ms=250,
        ).get("outcome")
        == "DEFERRED",
        "nonlinear problem was accepted by the linear checker",
    )
    require(
        run_convex_checker(
            reduced_case("convex", nonlinear_counterexample),
            set(),
            timeout_ms=250,
        ).get("outcome")
        == "CERTIFIED",
        "supported nonlinear problem was not certified by the convex checker",
    )
    require(
        run_linear_checker(nonlinear_counterexample, set(), timeout_ms=250).get(
            "reason_code"
        )
        == "UNREDUCED_INPUT",
        "linear checker accepted an unreduced expression",
    )
    require(
        run_convex_checker(nonlinear_counterexample, set(), timeout_ms=250).get(
            "reason_code"
        )
        == "UNREDUCED_INPUT",
        "convex checker accepted an unreduced expression",
    )
    with patch(
        "discretization.linear_checker.solve_linear_constraints",
        side_effect=RuntimeError("forced linear failure"),
    ):
        failed_linear = run_linear_checker(
            reduced_case("failed-linear", Op("<=", (x, Const(0)))),
            set(),
            timeout_ms=250,
        )
    require(
        failed_linear.get("outcome") == "DEFERRED"
        and failed_linear.get("reason_code") == "MALFORMED_OUTPUT",
        "linear backend failure did not defer",
    )
    with patch(
        "discretization.convex_checker.solve_convex_constraints",
        side_effect=RuntimeError("forced convex failure"),
    ):
        failed_convex = run_convex_checker(
            reduced_case("failed-convex", nonlinear_counterexample),
            set(),
            timeout_ms=250,
        )
    require(
        failed_convex.get("outcome") == "DEFERRED"
        and failed_convex.get("reason_code") == "MALFORMED_OUTPUT",
        "convex backend failure did not defer",
    )

    for path in args.certificate:
        certificate = load_certificate(path)
        errors = check_certificate(certificate)
        require(not errors, f"valid certificate failed: {path}: {errors}")
        mutated = copy.deepcopy(certificate)
        properties = mutated.get("analysis", {}).get("properties", [])
        require(bool(properties), f"certificate has no checked properties: {path}")
        reduction = next(
            (
                item.get("reduction", {})
                for item in properties
                if item.get("reduction", {}).get("trajectories")
            ),
            None,
        )
        require(reduction is not None, f"certificate has no physical trajectory: {path}")
        physical_target = reduction["trajectories"][0]["physical_value"]
        inventory = reduction["equation_inventory"]
        target_row = next(item for item in inventory if item["target"] == physical_target)
        target_row["included"] = False
        mutated["self_sha256"] = certificate_hash(mutated)
        mutation_errors = check_certificate(mutated)
        require(
            any(
                "physical equation" in error or "independent replay" in error
                for error in mutation_errors
            ),
            f"checker accepted an omitted physical equation: {path}",
        )
        all_inventory_targets = {
            item.get("target")
            for property_record in properties
            for item in property_record.get("reduction", {}).get("equation_inventory", [])
        }
        for required_target in ("vehicle_speedMps", "vehicle_gapMeters"):
            if required_target not in all_inventory_targets:
                continue
            missing_physics = copy.deepcopy(certificate)
            changed = False
            for property_record in missing_physics["analysis"]["properties"]:
                for item in property_record.get("reduction", {}).get("equation_inventory", []):
                    if item.get("target") == required_target and item.get("included") is True:
                        item["included"] = False
                        changed = True
            require(changed, f"{required_target} was never included: {path}")
            missing_physics["self_sha256"] = certificate_hash(missing_physics)
            require(
                any(
                    required_target in error
                    for error in check_certificate(missing_physics)
                ),
                f"checker accepted omitted {required_target}: {path}",
            )
        mapped_property = next(
            (
                item for item in properties
                if item.get("reduction", {}).get("sensor_to_physical_mappings")
            ),
            None,
        )
        if mapped_property is not None:
            bad_mapping = copy.deepcopy(certificate)
            selected = next(
                item for item in bad_mapping["analysis"]["properties"]
                if item.get("property_id") == mapped_property.get("property_id")
            )
            selected["reduction"]["sensor_to_physical_mappings"][0][
                "physical_value"
            ] += "_corrupted"
            bad_mapping["self_sha256"] = certificate_hash(bad_mapping)
            require(
                any(
                    "independent replay" in error or "sensor mapping equation" in error
                    for error in check_certificate(bad_mapping)
                ),
                f"checker accepted a corrupted sensor mapping: {path}",
            )
        print(f"{path}: VALIDATION PASSED")

    marker = "#ContinuousRate assign currentTime := currentTime + dt;"
    base = None
    model_path = None
    source = ""
    for path in args.certificate:
        candidate = load_certificate(path)
        candidate_path = Path(candidate["model"]["path"])
        candidate_source = candidate_path.read_text(encoding="utf-8")
        if marker in candidate_source:
            base = candidate
            model_path = candidate_path
            source = candidate_source
            break
    require(base is not None and model_path is not None, "validation models lack the expected time annotation")
    with tempfile.TemporaryDirectory(prefix="discretization-cascade-") as directory:
        modified_path = Path(directory) / model_path.name
        modified_path.write_text(
            source.replace(marker, "assign currentTime := currentTime + dt;", 1),
            encoding="utf-8",
        )
        mdp = load_mdp_certificate(base["markov_process_certificate"]["path"])
        analysis = analyze_model(
            modified_path,
            mdp,
            dt_text=base["settings"]["dt"]["input"],
            smt_timeout_ms=base["settings"]["smt_timeout_ms"],
        )
        require(
            analysis.get("result") == "NOT_CERTIFIED",
            "missing within-step meaning did not block certification",
        )
        cascaded = [
            item for item in analysis.get("properties", [])
            if item.get("progression", [{}])[0].get("reason_code")
            == "MISSING_WITHIN_STEP_MEANING"
        ]
        require(bool(cascaded), "missing annotation did not produce the required reason")
        for item in cascaded:
            progression = item["progression"]
            require(
                [stage["checker"] for stage in progression] == CHECKER_ORDER,
                "deferred property did not traverse the complete checker order",
            )
            require(
                all(stage["outcome"] == "DEFERRED" for stage in progression),
                "an unsupported checker produced a successful outcome",
            )
    print("exact proof rule validation: PASSED")
    print("nonlinear rejection validation: PASSED")
    print("linear proof certificate validation: PASSED")
    print("convex proof certificate validation: PASSED")
    print("linear to convex progression validation: PASSED")
    print("backend failure deferral validation: PASSED")
    print("full model equation omission rejection: PASSED")
    print("unreduced checker input rejection: PASSED")
    print("sensor to physical mapping mutation rejection: PASSED")
    print("loud deferral and checker progression: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
