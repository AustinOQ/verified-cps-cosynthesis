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

from clarity.certification.certificate import load_certificate as load_mdp_certificate
from clarity.certification.equations import Const, EquationModel, Op, Var

from .analysis import CHECKER_ORDER, analyze_model
from .certificate import certificate_hash, check_certificate, load_certificate
from .certificates.replay import replay_serialized_boolean_expression
from .certificates.verifier import (
    verify_recorded_convex_certificate,
    _verify_lazy_factored_stage,
    verify_recorded_linear_certificate,
    verify_recorded_outer_reduction,
)
from .checkers.convex import run_convex_checker, solve_convex_constraints
from .checkers.convex_envelope import run_convex_envelope_checker
from .checkers.factored import run_lazy_factored_checker
from .checkers.linear import run_linear_checker, solve_linear_constraints
from .checkers.linear_envelope import run_linear_envelope_checker
from .checkers.reachability import run_reachability_checker, run_smt_reachability_checker
from .model.optimization import QuadraticConstraint
from .model.proof_rules import (
    LinearInequality,
    expr_to_dict,
    expression_is_linear,
    prove_implication_exact,
)
from .model.reduction import (
    INTERVAL_TIME,
    ReachabilityContext,
    ReducedCase,
    expression_hash,
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


def nested_objects(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from nested_objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_objects(item)


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
    exact_replay_expression = {
        "type": "op",
        "op": "and",
        "args": [
            {
                "type": "op",
                "op": "==",
                "args": [
                    {
                        "type": "op",
                        "op": "*",
                        "args": [
                            {"type": "var", "name": "x"},
                            {"type": "const", "value": 2},
                        ],
                    },
                    {"type": "const", "value": 1},
                ],
            },
            {
                "type": "op",
                "op": ">",
                "args": [
                    {"type": "var", "name": "x"},
                    {"type": "const", "value": 0},
                ],
            },
        ],
    }
    require(
        replay_serialized_boolean_expression(
            exact_replay_expression,
            {"x": "1/2"},
        ) is True,
        "exact recorded values did not replay",
    )
    require(
        replay_serialized_boolean_expression(
            exact_replay_expression,
            {"x": "-1/2"},
        ) is False,
        "invalid recorded values replayed",
    )

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
    bounded_product = Op(
        "and",
        (
            Op(">=", (x, Const(-1))),
            Op("<=", (x, Const(1))),
            Op(">=", (Op("*", (x, x)), Const(2))),
        ),
    )
    bounded_product_result = run_linear_envelope_checker(
        reduced_case("bounded-product", bounded_product),
        timeout_ms=250,
    )
    require(
        bounded_product_result.get("outcome") == "CERTIFIED",
        "bounded product was not certified by the linear outer reduction",
    )
    require(
        not verify_recorded_outer_reduction(
            bounded_product_result["proof"],
            expression_hash(bounded_product),
        ),
        "bounded product reduction certificate did not verify",
    )
    require(
        run_convex_envelope_checker(
            reduced_case("bounded-square", bounded_product),
            timeout_ms=250,
        ).get("outcome") == "CERTIFIED",
        "bounded square was not certified by the convex outer reduction",
    )
    feasible_product = Op(
        "and",
        (
            Op(">=", (x, Const(-1))),
            Op("<=", (x, Const(1))),
            Op(">=", (Op("*", (x, x)), Const(0))),
        ),
    )
    require(
        run_linear_envelope_checker(
            reduced_case("feasible-product", feasible_product),
            timeout_ms=250,
        ).get("outcome") == "DEFERRED",
        "feasible linear outer reduction reported a proof",
    )

    interval_time = Var(INTERVAL_TIME)
    factored_expression = Op("and", (
        Op(">=", (interval_time, Const(0))),
        Op("<=", (interval_time, Const(1))),
        Op(">=", (x, Const(0))),
        Op("<", (Op("*", (x, interval_time)), Const(0))),
    ))
    factored_case = ReducedCase(
        "factored-endpoint",
        factored_expression,
        expression_hash(factored_expression),
        (),
        "test",
        "physical_interval",
        factored_expression,
        True,
    )
    factored_attempt = run_lazy_factored_checker(
        factored_case,
        set(),
        "linear",
        lambda leaf, remaining: run_linear_checker(
            leaf,
            set(),
            timeout_ms=remaining,
        ),
        lambda expression: expression_is_linear(expression, set())[0],
        timeout_ms=250,
        context_sha256="validation",
    )
    require(
        factored_attempt.get("outcome") == "CERTIFIED",
        "factored endpoint proof was rejected",
    )
    factored_stage = {"checker": "linear", **factored_attempt}
    factored_record = {"expression": expr_to_dict(factored_expression)}
    require(
        not _verify_lazy_factored_stage(factored_stage, factored_record),
        "factored endpoint certificate did not verify",
    )
    bad_factored_split = copy.deepcopy(factored_stage)
    split_node = next(
        item
        for item in nested_objects(bad_factored_split)
        if item.get("rule") == "exact_factored_split_v1"
    )
    split_node["children"][0]["expression_sha256"] = split_node[
        "expression_sha256"
    ]
    require(
        bool(_verify_lazy_factored_stage(bad_factored_split, factored_record)),
        "invalid factored split was accepted",
    )
    bad_endpoint = copy.deepcopy(factored_stage)
    endpoint_node = next(
        item
        for item in nested_objects(bad_endpoint)
        if item.get("split_kind") == "affine_interval_endpoints"
    )
    endpoint_node["changing_comparison_sha256"] = endpoint_node[
        "expression_sha256"
    ]
    require(
        bool(_verify_lazy_factored_stage(bad_endpoint, factored_record)),
        "invalid factored endpoint reduction was accepted",
    )

    reachability_context = ReachabilityContext(
        domain=Const(True),
        initial_constraints=(Op("==", (x, Const(0))),),
        initial_variables=("x",),
        post_values=(("x", Op("+", (x, Const(1)))),),
        action_variables=(),
        boolean_variables=(),
        integer_variables=(),
    )
    safe_reachability_case = ReducedCase(
        "safe-reachability",
        Op("<", (x, Const(0))),
        expression_hash(Op("<", (x, Const(0)))),
        (),
        "time_independent",
        "physical_interval",
        Op("<", (x, Const(0))),
    )
    safe_reachability_result = run_reachability_checker(
        safe_reachability_case,
        reachability_context,
        method="linear",
        timeout_ms=250,
    )
    require(
        safe_reachability_result.get("outcome") == "CERTIFIED",
        "linear reachability proof was rejected",
    )
    certified_depth = next(
        item
        for item in safe_reachability_result["proof"]["depth_attempts"]
        if item.get("proved") is True
    )
    for obligation in (
        certified_depth["base_obligations"]
        + certified_depth["induction_obligations"]
    ):
        require(
            not _verify_lazy_factored_stage(
                obligation["attempt"],
                {"expression": obligation["expression"]},
            ),
            "factored reachability certificate did not verify",
        )
    unsafe_reachability_case = ReducedCase(
        "unsafe-reachability",
        Op(">=", (x, Const(0))),
        expression_hash(Op(">=", (x, Const(0)))),
        (),
        "time_independent",
        "physical_interval",
        Op(">=", (x, Const(0))),
    )
    require(
        run_reachability_checker(
            unsafe_reachability_case,
            reachability_context,
            method="linear",
            timeout_ms=250,
        ).get("outcome") == "VIOLATION",
        "exact initial reachability violation was not replayed",
    )
    smt_model = EquationModel(
        "synthetic",
        {"x"},
        set(),
        initial_values={"x": 0},
    )
    require(
        run_smt_reachability_checker(
            smt_model,
            safe_reachability_case,
            reachability_context,
            timeout_ms=250,
        ).get("outcome") == "CERTIFIED",
        "SMT reachability proof was rejected",
    )
    smt_violation = run_smt_reachability_checker(
        smt_model,
        unsafe_reachability_case,
        reachability_context,
        timeout_ms=250,
    )
    require(
        smt_violation.get("outcome") == "VIOLATION"
        and smt_violation.get("proof", {}).get("trace_query", {}).get(
            "exact_replay"
        ) is True,
        "SMT reachability trace was not replayed",
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
        "clarity.discretization.checkers.linear.solve_linear_constraints",
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
        "clarity.discretization.checkers.convex.solve_convex_constraints",
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
        del mutated
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
            del missing_physics
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
            del bad_mapping
        exact_stage = next(
            (
                stage
                for property_record in properties
                for case in property_record.get("cases", [])
                for stage in case.get("progression", [])
                if (stage.get("proof") or {}).get("rule")
                == "exact_local_feasibility_replay_v1"
            ),
            None,
        )
        if exact_stage is not None:
            bad_values = copy.deepcopy(certificate)
            selected_stage = next(
                stage
                for property_record in bad_values["analysis"]["properties"]
                for case in property_record.get("cases", [])
                for stage in case.get("progression", [])
                if (stage.get("proof") or {}).get("rule")
                == "exact_local_feasibility_replay_v1"
            )
            first_name = sorted(selected_stage["proof"]["exact_values"])[0]
            selected_stage["proof"]["exact_values"][first_name] = None
            bad_values["self_sha256"] = certificate_hash(bad_values)
            require(
                any(
                    "local feasibility" in error
                    for error in check_certificate(bad_values, check_files=False)
                ),
                f"checker accepted corrupted exact values: {path}",
            )
            del bad_values
        subset_stage = next(
            (
                stage
                for property_record in properties
                for case in property_record.get("cases", [])
                for stage in case.get("progression", [])
                if stage.get("outcome") == "CERTIFIED"
                and (stage.get("proof") or {}).get("rule")
                == "solver_selected_subset_recertification_v1"
            ),
            None,
        )
        if subset_stage is not None:
            bad_subset = copy.deepcopy(certificate)
            selected_stage = next(
                stage
                for property_record in bad_subset["analysis"]["properties"]
                for case in property_record.get("cases", [])
                for stage in case.get("progression", [])
                if stage.get("outcome") == "CERTIFIED"
                and (stage.get("proof") or {}).get("rule")
                == "solver_selected_subset_recertification_v1"
            )
            selected_stage["proof"]["selected_indices"][0] = -1
            bad_subset["self_sha256"] = certificate_hash(bad_subset)
            require(
                any(
                    "selected constraint" in error
                    for error in check_certificate(bad_subset, check_files=False)
                ),
                f"checker accepted corrupted selected constraints: {path}",
            )
            del bad_subset
        smt_reachability_stage = next(
            (
                stage
                for property_record in properties
                for case in property_record.get("cases", [])
                for stage in case.get("progression", [])
                if stage.get("checker") == "smt_reachability"
                and stage.get("outcome") == "CERTIFIED"
            ),
            None,
        )
        if smt_reachability_stage is not None:
            bad_smt_proof = copy.deepcopy(certificate)
            selected_stage = next(
                stage
                for property_record in bad_smt_proof["analysis"]["properties"]
                for case in property_record.get("cases", [])
                for stage in case.get("progression", [])
                if stage.get("checker") == "smt_reachability"
                and stage.get("outcome") == "CERTIFIED"
            )
            certified_depth = next(
                item
                for item in selected_stage["proof"]["depth_attempts"]
                if item.get("proved") is True
            )
            certified_depth["induction_query"]["z3_proof_sha256"] = "corrupted"
            bad_smt_proof["self_sha256"] = certificate_hash(bad_smt_proof)
            require(
                any(
                    "SMT reachability no solution proof hash" in error
                    for error in check_certificate(
                        bad_smt_proof,
                        check_files=False,
                    )
                ),
                f"checker accepted a corrupted SMT reachability proof: {path}",
            )
            del bad_smt_proof
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
            if (item.get("progression") or [{}])[0].get("reason_code")
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
    print("factored proof certificate validation: PASSED")
    print("linear to convex progression validation: PASSED")
    print("backend failure deferral validation: PASSED")
    print("full model equation omission rejection: PASSED")
    print("unreduced checker input rejection: PASSED")
    print("sensor to physical mapping mutation rejection: PASSED")
    print("exact solver value replay validation: PASSED")
    print("selected constraint certificate validation: PASSED")
    print("SMT reachability proof and trace validation: PASSED")
    print("loud deferral and checker progression: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
