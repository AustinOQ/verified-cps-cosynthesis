#!/usr/bin/env python3
"""Large validation battery for strict-Q certification artifacts.

This battery is deliberately broader than validate_reference_models.py:

1. Known-positive complex SysML models must keep their expected buffers and
   discharged reward/done obligations.
2. Known-negative model variants are created in temporary files and must not
   certify.
3. Mutated certificate artifacts must be rejected by the independent checker.

The script writes validation outputs to the selected output directory.
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from clarity.models import models_root
from clarity.sysml.inputs import discover_sysml
from .certificate import (
    PROFILE_OBLIGATIONS_DISCHARGED,
    build_certificate_for_path,
    check_certificate,
    load_certificate,
    write_certificate,
)
from .equation_reconstruct import equation_reconstruction_trace, fact_key as equation_fact_key
from .equations import Const, Equation, EquationModel, Ite, Op, Var
from .relevance import compute_transition_closed_relevance, equation_refs
from .solver import MAX_SOLVER_POLYNOMIAL_DEGREE, one_step_transition_closure
from .strict_extract import extract_equation_model
from clarity.certification.reconstruct import get_strict_model, reconstruct
from clarity.sysml.runtime_settings import DEFAULT_DT


_DISCOVERED = {
    item.key: item.path
    for item in discover_sysml(
        [], models_root=models_root()
    )
}
MODELS = {
    "mixing": _DISCOVERED["tank-filling-system"],
    "thermostat": _DISCOVERED["thermostat"],
    "cruise-discrete": _DISCOVERED["cruise-control"],
}
TEST_DT = DEFAULT_DT


EXPECTED = {
    "mixing": {
        "buffer": (2, 1),
        "state": 11,
        "definitions": 27,
        "transitions": 11,
        "q": 10,
        "terminal_state_refs": {
            "feederTank1_currentLevelMl",
            "feederTank2_currentLevelMl",
        },
        "solver_status": "discharged",
    },
    "thermostat": {
        "buffer": (1, 2),
        "state": 10,
        "definitions": 6,
        "transitions": 10,
        "q": 8,
        "terminal_state_refs": {
            "controller_acOn",
            "controller_heaterOn",
            "thermometer_temperatureReading_temperatureCelcius",
        },
        "solver_status": "discharged",
    },
    "cruise-discrete": {
        "buffer": (1, 2),
        "state": 13,
        "definitions": 9,
        "transitions": 13,
        "q": 11,
        "terminal_state_refs": {
            "controller_brakeOn",
            "controller_throttleOn",
            "speedSensor_speedReading_speedMps",
        },
        "solver_status": "discharged",
    },
}


def _first_closure(model: dict[str, Any], max_obs: int = 2, max_act: int = 4):
    target = set(model.get("R", set())) or set(model["STATE"])
    for b_obs in range(max_obs + 1):
        for b_act in range(max_act + 1):
            missing, _ = reconstruct(model, b_obs, b_act, horizon=8, target=target)
            if not missing:
                return b_obs, b_act
    return None


class Battery:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.cases: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def record(self, name: str, passed: bool, detail: str = "", data: dict[str, Any] | None = None) -> None:
        self.cases.append({
            "name": name,
            "passed": passed,
            "detail": detail,
            "data": data or {},
        })
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            self.failures.append(f"{name}: {detail}")

    def require(self, name: str, condition: bool, detail: str) -> None:
        self.record(name, condition, detail)

    def report(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cases": self.cases,
            "failures": self.failures,
            "passed": not self.failures,
        }
        (self.out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _assert_positive_model(battery: Battery, name: str, cert_dir: Path) -> dict[str, Any]:
    expected = EXPECTED[name]
    model = extract_equation_model(MODELS[name])
    relevance = compute_transition_closed_relevance(model)
    terminals = list(model.terminals.values())
    terminal = terminals[0] if len(terminals) == 1 else None
    terminal_refs = set()
    if terminal is not None:
        terminal_refs = {
            ref for ref in equation_refs(model, terminal)
            if ref in model.state
        }

    battery.require(
        f"{name}/shape",
        len(model.state) == expected["state"]
        and len(model.definitions) == expected["definitions"]
        and len(model.transitions) == expected["transitions"]
        and len(relevance.q) == expected["q"],
        (
            f"state={len(model.state)} def={len(model.definitions)} "
            f"trans={len(model.transitions)} q={len(relevance.q)}"
        ),
    )
    battery.require(
        f"{name}/terminal_refs",
        terminal_refs == expected["terminal_state_refs"],
        f"terminal_state_refs={sorted(terminal_refs)}",
    )
    blocked_diags = [
        diag.pretty()
        for diag in model.diagnostics
        if diag.severity in {"warning", "error"}
    ]
    unresolved_raw = {
        eq.target: sorted(eq.raw_refs() - model.constants)
        for eq in model.all_equations()
        if eq.raw_refs() - model.constants
    }
    missing_ignored_reasons = sorted(
        var for var in relevance.ignored_state
        if "backward cone" not in relevance.ignored_state_reasons.get(var, "")
    )
    battery.require(
        f"{name}/equation_audit",
        not blocked_diags and not unresolved_raw and not missing_ignored_reasons,
        (
            f"blocked_diags={blocked_diags[:3]} "
            f"unresolved_raw={unresolved_raw} "
            f"missing_ignored_reasons={missing_ignored_reasons}"
        ),
    )

    strict_model = get_strict_model(MODELS[name], dt=TEST_DT)
    observed_buffer = _first_closure(strict_model)
    battery.require(
        f"{name}/first_closure",
        observed_buffer == expected["buffer"],
        f"expected={expected['buffer']} observed={observed_buffer}",
    )

    exp_obs, exp_act = expected["buffer"]
    target = set(strict_model.get("R", set())) or set(strict_model["STATE"])
    prior_failures = []
    for b_obs in range(exp_obs + 1):
        for b_act in range(5):
            if (b_obs, b_act) >= (exp_obs, exp_act):
                continue
            missing, _ = reconstruct(strict_model, b_obs, b_act, horizon=8, target=target)
            if not missing:
                prior_failures.append((b_obs, b_act))
    battery.require(
        f"{name}/smaller_buffers_fail",
        not prior_failures,
        f"unexpected earlier passing buffers={prior_failures}",
    )

    cert = build_certificate_for_path(MODELS[name], dt=TEST_DT)
    cert_path = cert_dir / f"{name}.certificate.json"
    write_certificate(cert, cert_path)
    disk_cert = load_certificate(cert_path)
    errors = check_certificate(disk_cert)
    buffer = None if cert["buffer"] is None else (
        cert["buffer"]["b_obs"], cert["buffer"]["b_act"]
    )
    obligations = cert["mdp_obligations"]
    gate = cert["theorem_gate"]
    equation_proof = cert["equation_proof"]
    ignored_state = set(cert["sets"]["ignored_state"])
    ignored_reasons = cert["sets"]["ignored_state_reasons"]
    one_step_solver = cert["solver_advisory"]["one_step_transition_closure"]
    certificate_unresolved_raw = [
        (section, eq["target"], eq.get("unresolved_raw_refs", []))
        for section, equations in cert["equations"].items()
        for eq in equations
        if eq.get("unresolved_raw_refs")
    ]
    battery.require(
        f"{name}/certificate",
        not errors
        and cert["result"] == "PASS"
        and cert["claim"]["profile_mdp_theorem"] == "discharged"
        and cert["claim"]["mdp_theorem"] == "discharged"
        and cert["claim"]["solver_backed_mdp_theorem"] == "discharged"
        and buffer == expected["buffer"]
        and obligations["terminal_state"]["status"] == "covered_by_q_and_action"
        and obligations["truncation"]["status"] == "discharged_as_finite_horizon_augmented_state"
        and obligations["reward"]["status"] == "discharged_over_augmented_state"
        and obligations["done"]["status"] == "discharged_over_augmented_state"
        and obligations["shield"]["status"] == "discharged"
        and obligations["shield"]["semantic_status"] == "discharged_discrete_exact_ast"
        and obligations["overall_mdp_theorem_status"] == PROFILE_OBLIGATIONS_DISCHARGED
        and gate["profile_mdp_theorem"] == "discharged"
        and gate["solver_backed_mdp_theorem"] == "discharged"
        and gate["profile_obligations_discharged"] is True
        and gate["full_mdp_theorem_claim_allowed"] is True
        and not gate["full_mdp_theorem_blockers"]
        and gate["solver_backed_uniqueness"]["status"] == "discharged"
        and gate["solver_backed_uniqueness"]["result_status"] == "discharged"
        and cert["search"]["legacy_and_equation_search_agree"] is True
        and equation_proof["proof_engine"] == "equation_ir_syntactic_v1"
        and equation_proof["passes"] is True
        and not equation_proof["missing_target_facts"]
        and one_step_solver["status"] == expected["solver_status"]
        and (
            one_step_solver["claim"] == "one_step_transition_closure"
            if one_step_solver["status"] == "discharged"
            else one_step_solver["claim"] == "not_claimed_by_this_artifact"
        )
        and not any(
            fact["rule"] == "equation_guarded_copy_inversion"
            and fact["detail"].get("guard_status") != "proven_true"
            for fact in equation_proof["facts"]
        )
        and not certificate_unresolved_raw
        and all(
            "backward cone" in ignored_reasons.get(var, "")
            for var in ignored_state
        ),
        (
            f"buffer={buffer} checker_errors={errors} "
            f"solver={one_step_solver['status']} "
            f"unresolved_raw={certificate_unresolved_raw[:3]}"
        ),
    )
    return cert


def _assert_solver_units(battery: Battery) -> None:
    def power_expr(name: str, degree: int):
        expr = Var(name)
        for _ in range(degree - 1):
            expr = Op("*", (expr, Var(name)))
        return expr

    closed = EquationModel(
        model_path="<solver-closed>",
        state={"x"},
        actions={"u"},
        transitions={
            "x": Equation(
                target="x",
                expr=Op("+", (Var("x"), Var("u"))),
                kind="transition",
                source="synthetic closed linear transition",
            )
        },
        observations={
            "xObs": Equation(
                target="obs.xObs",
                expr=Var("x"),
                kind="observation",
                source="synthetic observation",
            )
        },
    )
    closed_result = one_step_transition_closure(closed, {"x"})
    battery.require(
        "solver_unit/closed_linear_unsat",
        closed_result["status"] == "discharged",
        f"result={closed_result}",
    )

    hidden = EquationModel(
        model_path="<solver-hidden-dependency>",
        state={"x", "hidden"},
        actions=set(),
        transitions={
            "x": Equation(
                target="x",
                expr=Var("hidden"),
                kind="transition",
                source="synthetic hidden dependency",
            )
        },
        observations={
            "xObs": Equation(
                target="obs.xObs",
                expr=Var("x"),
                kind="observation",
                source="synthetic observation",
            )
        },
    )
    hidden_result = one_step_transition_closure(hidden, {"x"})
    battery.require(
        "solver_unit/hidden_dependency_sat",
        hidden_result["status"] == "counterexample"
        and hidden_result.get("disagreement", {}).get("term") == "next_q.x",
        f"result={hidden_result}",
    )

    boolean_closed = EquationModel(
        model_path="<solver-boolean-ite>",
        state={"safe"},
        actions={"command"},
        transitions={
            "safe": Equation(
                target="safe",
                expr=Ite(Var("command"), Const(True), Const(False)),
                kind="transition",
                source="synthetic boolean transition",
            )
        },
        requirements={
            "status.safe": Equation(
                target="status.safe",
                expr=Var("safe"),
                kind="requirement",
                source="synthetic requirement",
            )
        },
    )
    boolean_result = one_step_transition_closure(boolean_closed, {"safe"})
    battery.require(
        "solver_unit/boolean_ite_unsat",
        boolean_result["status"] == "discharged",
        f"result={boolean_result}",
    )

    quadratic = EquationModel(
        model_path="<solver-quadratic>",
        state={"x"},
        actions=set(),
        transitions={
            "x": Equation(
                target="x",
                expr=Op("*", (Var("x"), Var("x"))),
                kind="transition",
                source="synthetic quadratic transition",
            )
        },
    )
    quadratic_result = one_step_transition_closure(quadratic, {"x"})
    battery.require(
        "solver_unit/quadratic_unsat",
        quadratic_result["status"] == "discharged"
        and quadratic_result.get("logic") == "QF_UFNRA"
        and quadratic_result.get("max_polynomial_degree") == 2,
        f"result={quadratic_result}",
    )

    hidden_quadratic = EquationModel(
        model_path="<solver-hidden-quadratic>",
        state={"x", "hidden"},
        actions=set(),
        transitions={
            "x": Equation(
                target="x",
                expr=Op("*", (Var("x"), Var("hidden"))),
                kind="transition",
                source="synthetic hidden quadratic transition",
            )
        },
    )
    hidden_quadratic_result = one_step_transition_closure(hidden_quadratic, {"x"})
    battery.require(
        "solver_unit/hidden_quadratic_sat",
        hidden_quadratic_result["status"] == "counterexample"
        and hidden_quadratic_result.get("logic") == "QF_UFNRA"
        and hidden_quadratic_result.get("max_polynomial_degree") == 2
        and hidden_quadratic_result.get("disagreement", {}).get("term") == "next_q.x",
        f"result={hidden_quadratic_result}",
    )

    for degree in range(3, MAX_SOLVER_POLYNOMIAL_DEGREE + 1):
        polynomial = EquationModel(
            model_path=f"<solver-degree-{degree}>",
            state={"x"},
            actions=set(),
            transitions={
                "x": Equation(
                    target="x",
                    expr=power_expr("x", degree),
                    kind="transition",
                    source=f"synthetic degree {degree} transition",
                )
            },
        )
        polynomial_result = one_step_transition_closure(polynomial, {"x"})
        battery.require(
            f"solver_unit/degree_{degree}_unsat",
            polynomial_result["status"] == "discharged"
            and polynomial_result.get("logic") == "QF_UFNRA"
            and polynomial_result.get("max_polynomial_degree") == degree,
            f"result={polynomial_result}",
        )

        hidden_polynomial = EquationModel(
            model_path=f"<solver-hidden-degree-{degree}>",
            state={"x", "hidden"},
            actions=set(),
            transitions={
                "x": Equation(
                    target="x",
                    expr=Op("*", (power_expr("x", degree - 1), Var("hidden"))),
                    kind="transition",
                    source=f"synthetic hidden degree {degree} transition",
                )
            },
        )
        hidden_polynomial_result = one_step_transition_closure(hidden_polynomial, {"x"})
        battery.require(
            f"solver_unit/hidden_degree_{degree}_sat",
            hidden_polynomial_result["status"] == "counterexample"
            and hidden_polynomial_result.get("logic") == "QF_UFNRA"
            and hidden_polynomial_result.get("max_polynomial_degree") == degree
            and hidden_polynomial_result.get("disagreement", {}).get("term") == "next_q.x",
            f"result={hidden_polynomial_result}",
        )

    unsupported_degree = MAX_SOLVER_POLYNOMIAL_DEGREE + 1
    too_high_degree = EquationModel(
        model_path=f"<solver-degree-{unsupported_degree}>",
        state={"x"},
        actions=set(),
        transitions={
            "x": Equation(
                target="x",
                expr=power_expr("x", unsupported_degree),
                kind="transition",
                source=f"synthetic degree {unsupported_degree} transition",
            )
        },
    )
    too_high_result = one_step_transition_closure(too_high_degree, {"x"})
    battery.require(
        f"solver_unit/degree_{unsupported_degree}_unknown",
        too_high_result["status"] == "unknown"
        and too_high_result.get("reason") == "unsupported_fragment"
        and any(
            f"polynomial degree {unsupported_degree}" in item
            for item in too_high_result.get("unsupported", [])
        ),
        f"result={too_high_result}",
    )

    symbolic_division = EquationModel(
        model_path="<solver-symbolic-division>",
        state={"x", "hidden"},
        actions=set(),
        transitions={
            "x": Equation(
                target="x",
                expr=Op("/", (Var("x"), Var("hidden"))),
                kind="transition",
                source="synthetic symbolic division transition",
            )
        },
    )
    symbolic_division_result = one_step_transition_closure(symbolic_division, {"x"})
    battery.require(
        "solver_unit/symbolic_division_unknown",
        symbolic_division_result["status"] == "unknown"
        and symbolic_division_result.get("reason") == "unsupported_fragment"
        and any("division by symbolic expression" in item for item in symbolic_division_result.get("unsupported", [])),
        f"result={symbolic_division_result}",
    )


def _assert_equation_reconstructibility_units(battery: Battery) -> None:
    direct = EquationModel(
        model_path="<synthetic-direct-copy>",
        state={"x", "y"},
        actions=set(),
        observations={
            "yObs": Equation(
                target="obs.yObs",
                expr=Var("y"),
                kind="observation",
                source="synthetic direct observation",
            )
        },
        transitions={
            "y": Equation(
                target="y",
                expr=Var("x"),
                kind="transition",
                source="synthetic direct copy",
            )
        },
    )
    direct_trace = equation_reconstruction_trace(
        direct,
        b_obs=0,
        b_act=0,
        dt=TEST_DT,
        horizon=2,
        target={"x"},
        enable_sampled_memory=False,
    )
    battery.require(
        "equation_unit/direct_copy_inversion",
        equation_fact_key("x", -1) in {fact["key"] for fact in direct_trace["facts"]},
        f"facts={[fact['key'] for fact in direct_trace['facts']]}",
    )

    guarded = EquationModel(
        model_path="<synthetic-guarded-copy>",
        state={"guard", "x", "y"},
        actions=set(),
        observations={
            "yObs": Equation(
                target="obs.yObs",
                expr=Var("y"),
                kind="observation",
                source="synthetic direct observation",
            )
        },
        transitions={
            "y": Equation(
                target="y",
                expr=Ite(Var("guard"), Var("x"), Var("y")),
                kind="transition",
                source="synthetic guarded copy",
            )
        },
    )
    guarded_trace = equation_reconstruction_trace(
        guarded,
        b_obs=0,
        b_act=0,
        dt=TEST_DT,
        horizon=2,
        target={"x"},
        enable_sampled_memory=False,
    )
    guarded_facts = {fact["key"] for fact in guarded_trace["facts"]}
    battery.require(
        "equation_unit/guarded_copy_not_inverted",
        equation_fact_key("x", -1) not in guarded_facts
        and bool(guarded_trace["blocked_unsafe_inversions"]),
        (
            f"x@-1_present={equation_fact_key('x', -1) in guarded_facts} "
            f"blocked={guarded_trace['blocked_unsafe_inversions'][:1]}"
        ),
    )

    proven_guarded = EquationModel(
        model_path="<synthetic-proven-guarded-copy>",
        state={"x", "y"},
        actions=set(),
        observations={
            "yObs": Equation(
                target="obs.yObs",
                expr=Var("y"),
                kind="observation",
                source="synthetic direct observation",
            )
        },
        transitions={
            "y": Equation(
                target="y",
                expr=Ite(Const(True), Var("x"), Var("y")),
                kind="transition",
                source="synthetic proven guarded copy",
            )
        },
    )
    proven_trace = equation_reconstruction_trace(
        proven_guarded,
        b_obs=0,
        b_act=0,
        dt=TEST_DT,
        horizon=2,
        target={"x"},
        enable_sampled_memory=False,
    )
    battery.require(
        "equation_unit/proven_guarded_copy_inverted",
        equation_fact_key("x", -1) in {fact["key"] for fact in proven_trace["facts"]},
        f"blocked={proven_trace['blocked_unsafe_inversions']}",
    )


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(f"{label}: expected one replacement target, found {count}")
    return text.replace(old, new, 1)


def _write_temp_model(src: str, new_text: str, tmp: Path, name: str) -> str:
    model_dir = tmp / name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "model.sysml"
    path.write_text(new_text, encoding="utf-8")
    return str(path)


def _assert_negative_models(battery: Battery) -> None:
    with tempfile.TemporaryDirectory(prefix="cert-battery-models-") as td:
        tmp = Path(td)

        thermostat_text = Path(MODELS["thermostat"]).read_text(encoding="utf-8")
        thermostat_no_done = _replace_once(
            thermostat_text,
            (
                "                in done := setPointCelcius <= reading.temperatureCelcius + toleranceCelcius and\n"
                "                setPointCelcius >= reading.temperatureCelcius - toleranceCelcius and not heaterOn and not acOn;\n"
            ),
            "",
            "thermostat remove done binding",
        )
        thermostat_no_done_path = _write_temp_model(
            MODELS["thermostat"], thermostat_no_done, tmp, "thermostat-no-done"
        )
        cert = build_certificate_for_path(thermostat_no_done_path, dt=TEST_DT)
        errors = check_certificate(cert, check_hash=False)
        blocking_codes = {d["code"] for d in cert["diagnostics"]["blocking"]}
        battery.require(
            "negative_model/missing_terminal_binding",
            cert["result"] != "PASS"
            and errors
            and "missing_terminal_binding" in blocking_codes,
            f"result={cert['result']} blocking={sorted(blocking_codes)} errors={errors[:3]}",
        )

        mixing_text = Path(MODELS["mixing"]).read_text(encoding="utf-8")
        hidden_done = _replace_once(
            mixing_text,
            (
                "                in done := (tank1OriginalLevelMl - volume1Res.response >= tank1TransferMl) and\n"
                "                            (tank2OriginalLevelMl - volume2Res.response >= tank2TransferMl);\n"
            ),
            "                in done := fillingTank.currentLevelMl >= 0;\n",
            "mixing hidden terminal dependency",
        )
        hidden_done_path = _write_temp_model(
            MODELS["mixing"], hidden_done, tmp, "mixing-hidden-terminal"
        )
        cert = build_certificate_for_path(hidden_done_path, dt=TEST_DT)
        errors = check_certificate(cert, check_hash=False)
        battery.require(
            "negative_model/hidden_terminal_dependency",
            cert["result"] != "PASS" and errors,
            f"result={cert['result']} buffer={cert['buffer']} errors={errors[:3]}",
        )

        thermostat_no_neural_req = _replace_once(
            thermostat_text,
            "#NeuralRequirement requirement def 'Neural Controller Soundness'",
            "requirement def 'Neural Controller Soundness'",
            "thermostat remove NeuralRequirement metadata",
        )
        thermostat_no_neural_req_path = _write_temp_model(
            MODELS["thermostat"],
            thermostat_no_neural_req,
            tmp,
            "thermostat-no-neural-requirement",
        )
        cert = build_certificate_for_path(thermostat_no_neural_req_path, dt=TEST_DT)
        errors = check_certificate(cert, check_hash=False)
        shield = cert["mdp_obligations"]["shield"]
        battery.require(
            "negative_model/missing_neural_requirement",
            errors
            and shield["status"] == "not_discharged"
            and shield["semantic_status"] == "missing_neural_requirement_ast",
            f"shield={shield.get('semantic_status')} errors={errors[:3]}",
        )

        thermostat_unknown_shield_ref = _replace_once(
            thermostat_text,
            "(not (p.heaterState and p.acState))",
            "((not (p.heaterState and p.acState)) and p.unmappedHiddenInput)",
            "thermostat unknown shield ref",
        )
        thermostat_unknown_shield_ref_path = _write_temp_model(
            MODELS["thermostat"],
            thermostat_unknown_shield_ref,
            tmp,
            "thermostat-unknown-shield-ref",
        )
        cert = build_certificate_for_path(
            thermostat_unknown_shield_ref_path, dt=TEST_DT
        )
        errors = check_certificate(cert, check_hash=False)
        shield = cert["mdp_obligations"]["shield"]
        battery.require(
            "negative_model/unknown_shield_ref",
            errors
            and shield["status"] == "not_discharged"
            and shield["semantic_status"] == "unknown_requirement_refs",
            f"unknown={shield.get('unknown_requirement_refs')} errors={errors[:3]}",
        )

        thermostat_unparsed_requirement = _replace_once(
            thermostat_text,
            (
                "                currentTime > 0 implies \n"
                "                    (s.lastObservedTemperature < s.controller.setPointCelcius - s.controller.toleranceCelcius\n"
                "                        implies s.controller.heaterOn)\n"
            ),
            "                @unparsedRequirementExpression\n",
            "thermostat unparsed requirement expression",
        )
        thermostat_unparsed_requirement_path = _write_temp_model(
            MODELS["thermostat"],
            thermostat_unparsed_requirement,
            tmp,
            "thermostat-unparsed-requirement",
        )
        cert = build_certificate_for_path(
            thermostat_unparsed_requirement_path, dt=TEST_DT
        )
        errors = check_certificate(cert, check_hash=False)
        blocking_codes = {d["code"] for d in cert["diagnostics"]["blocking"]}
        battery.require(
            "negative_model/unparsed_requirement",
            cert["result"] != "PASS"
            and errors
            and "skipped_requirement_parse" in blocking_codes,
            f"blocking={sorted(blocking_codes)} errors={errors[:3]}",
        )

    strict_model = get_strict_model(
        MODELS["mixing"], dt=TEST_DT, enable_sampled_memory=False
    )
    battery.require(
        "negative_closure/mixing_without_sampled_memory",
        _first_closure(strict_model) is None,
        "mixing must not certify if sampled-memory rules are disabled",
    )


def _target_fact_index(cert: dict[str, Any]) -> int:
    target_keys = set(cert["proof"]["target_facts"])
    for idx, fact in enumerate(cert["proof"]["facts"]):
        if fact["key"] in target_keys:
            return idx
    raise AssertionError("could not find a target fact to mutate")


def _forward_fact_index(cert: dict[str, Any]) -> int:
    for idx, fact in enumerate(cert["proof"]["facts"]):
        if fact["rule"] == "forward_transition" and fact.get("premises"):
            return idx
    raise AssertionError("could not find a forward fact to mutate")


def _equation_forward_fact_index(cert: dict[str, Any]) -> int:
    for idx, fact in enumerate(cert["equation_proof"]["facts"]):
        if fact["rule"] == "equation_forward_transition" and fact.get("premises"):
            return idx
    raise AssertionError("could not find an equation-forward fact to mutate")


def _equation_target_fact_index(cert: dict[str, Any]) -> int:
    target_keys = set(cert["equation_proof"]["target_facts"])
    for idx, fact in enumerate(cert["equation_proof"]["facts"]):
        if fact["key"] in target_keys:
            return idx
    raise AssertionError("could not find an equation target fact to mutate")


def _assert_mutation_rejected(
    battery: Battery,
    base: dict[str, Any],
    name: str,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    cert = copy.deepcopy(base)
    mutator(cert)
    errors = check_certificate(cert)
    battery.require(
        f"negative_certificate/{name}",
        bool(errors),
        f"errors={errors[:3]}",
    )


def _assert_certificate_mutations(battery: Battery, base: dict[str, Any]) -> None:
    mutations: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("result_fail", lambda c: c.__setitem__("result", "FAIL")),
        (
            "claim_mdp_not_discharged",
            lambda c: c["claim"].__setitem__("mdp_theorem", "not_claimed_by_this_artifact"),
        ),
        (
            "claim_profile_not_discharged",
            lambda c: c["claim"].__setitem__("profile_mdp_theorem", "not_discharged"),
        ),
        (
            "claim_solver_backed_mdp_not_discharged",
            lambda c: c["claim"].__setitem__(
                "solver_backed_mdp_theorem", "not_claimed_by_this_artifact"
            ),
        ),
        (
            "missing_theorem_gate",
            lambda c: c.pop("theorem_gate", None),
        ),
        (
            "theorem_gate_profile_not_discharged",
            lambda c: c["theorem_gate"].__setitem__("profile_mdp_theorem", "not_discharged"),
        ),
        (
            "theorem_gate_obligations_false",
            lambda c: c["theorem_gate"].__setitem__("profile_obligations_discharged", False),
        ),
        (
            "theorem_gate_blocks_full_claim",
            lambda c: c["theorem_gate"].__setitem__("full_mdp_theorem_claim_allowed", False),
        ),
        (
            "theorem_gate_solver_uniqueness_not_discharged",
            lambda c: c["theorem_gate"]["solver_backed_uniqueness"].__setitem__(
                "status", "not_discharged"
            ),
        ),
        (
            "theorem_gate_solver_logic_mismatch",
            lambda c: c["theorem_gate"]["solver_backed_uniqueness"].__setitem__(
                "logic", "QF_BV"
            ),
        ),
        (
            "strict_mode_disabled",
            lambda c: c["proof_mode"].__setitem__("strict_no_legacy_fallback", False),
        ),
        (
            "legacy_transition_kind",
            lambda c: c["equations"]["transitions"][0].__setitem__(
                "kind", "legacy_transition_dependency"
            ),
        ),
        (
            "legacy_dep_expr",
            lambda c: c["equations"]["transitions"][0].__setitem__(
                "expr", {"type": "op", "op": "legacy_dep", "args": []}
            ),
        ),
        (
            "equation_unresolved_raw_ref",
            lambda c: c["equations"]["observations"][0].__setitem__(
                "unresolved_raw_refs", ["unclassifiedSymbol"]
            ),
        ),
        (
            "equation_raw_ref_audit_mismatch",
            lambda c: c["equations"]["observations"][0].__setitem__(
                "raw_refs", ["not_in_expr"]
            ),
        ),
        (
            "missing_ignored_state_reason",
            lambda c: c["sets"]["ignored_state_reasons"].pop(
                c["sets"]["ignored_state"][0], None
            ),
        ),
        (
            "missing_terminal_section",
            lambda c: c["mdp_obligations"].pop("terminal_state", None),
        ),
        (
            "terminal_not_covered",
            lambda c: c["mdp_obligations"]["terminal_state"].__setitem__(
                "status", "not_covered_by_q_and_action"
            ),
        ),
        (
            "terminal_equations_empty",
            lambda c: c["mdp_obligations"]["terminal_state"].__setitem__(
                "equations", []
            ),
        ),
        (
            "truncation_not_discharged",
            lambda c: c["mdp_obligations"]["truncation"].__setitem__(
                "status", "modeled_as_finite_horizon_augmented_state"
            ),
        ),
        (
            "reward_not_discharged",
            lambda c: c["mdp_obligations"]["reward"].__setitem__(
                "status", "not_discharged"
            ),
        ),
        (
            "reward_terminal_uncovered",
            lambda c: c["mdp_obligations"]["reward"]["covered_terms"].__setitem__(
                "all_terminal_state_terms_covered", False
            ),
        ),
        (
            "reward_external_terms_present",
            lambda c: c["mdp_obligations"]["reward"].__setitem__(
                "external_terms_not_yet_in_q", ["unmodeled terminal source"]
            ),
        ),
        (
            "done_not_discharged",
            lambda c: c["mdp_obligations"]["done"].__setitem__(
                "status", "not_discharged"
            ),
        ),
        (
            "done_covered_false",
            lambda c: c["mdp_obligations"]["done"].__setitem__("covered", False),
        ),
        (
            "shield_not_discharged",
            lambda c: c["mdp_obligations"]["shield"].__setitem__(
                "status", "not_discharged"
            ),
        ),
        (
            "shield_bad_semantic_status",
            lambda c: c["mdp_obligations"]["shield"].__setitem__(
                "semantic_status", "interface_recorded_not_semantically_proven"
            ),
        ),
        (
            "shield_missing_input",
            lambda c: c["mdp_obligations"]["shield"].__setitem__(
                "missing_input_params", ["done"]
            ),
        ),
        (
            "shield_missing_output",
            lambda c: c["mdp_obligations"]["shield"].__setitem__(
                "missing_output_params", ["shouldOpenValve1"]
            ),
        ),
        (
            "shield_unknown_ref",
            lambda c: c["mdp_obligations"]["shield"].__setitem__(
                "unknown_requirement_refs", ["hidden"]
            ),
        ),
        (
            "shield_input_uncovered",
            lambda c: c["mdp_obligations"]["shield"]["input_coverage"][0].__setitem__(
                "covered_by_q_and_action", False
            ),
        ),
        (
            "shield_output_not_unique",
            lambda c: c["mdp_obligations"]["shield"]["output_action_mapping"][0].__setitem__(
                "unique_action_var", False
            ),
        ),
        ("missing_proof", lambda c: c.__setitem__("proof", None)),
        ("missing_equation_proof", lambda c: c.__setitem__("equation_proof", None)),
        (
            "equation_proof_not_passes",
            lambda c: c["equation_proof"].__setitem__("passes", False),
        ),
        (
            "equation_target_fact_removed",
            lambda c: c["equation_proof"]["facts"].pop(_equation_target_fact_index(c)),
        ),
        (
            "equation_bad_forward_premise",
            lambda c: c["equation_proof"]["facts"][
                _equation_forward_fact_index(c)
            ].__setitem__("premises", ["not_a_real_equation_fact@0"]),
        ),
        (
            "equation_unsafe_guarded_inversion",
            lambda c: c["equation_proof"]["facts"].append(
                {
                    "key": "fakeSource@-1",
                    "var": "fakeSource",
                    "tau": -1,
                    "rule": "equation_guarded_copy_inversion",
                    "premises": [],
                    "detail": {
                        "equation_target": "fakeTarget",
                        "guard_status": "not_proven_true",
                        "inversion": "guarded_transition_copy",
                    },
                }
            ),
        ),
        (
            "missing_solver_advisory",
            lambda c: c.pop("solver_advisory", None),
        ),
        (
            "solver_advisory_bad_status",
            lambda c: c["solver_advisory"]["one_step_transition_closure"].__setitem__(
                "status", "magically_proven"
            ),
        ),
        (
            "solver_advisory_unknown_claims",
            lambda c: (
                c["solver_advisory"]["one_step_transition_closure"].__setitem__(
                    "status", "unknown"
                ),
                c["solver_advisory"]["one_step_transition_closure"].__setitem__(
                    "claim", "one_step_transition_closure"
                ),
            ),
        ),
        (
            "solver_advisory_unknown_not_claimed",
            lambda c: (
                c["solver_advisory"]["one_step_transition_closure"].__setitem__(
                    "status", "unknown"
                ),
                c["solver_advisory"]["one_step_transition_closure"].__setitem__(
                    "claim", "not_claimed_by_this_artifact"
                ),
            ),
        ),
        (
            "target_fact_removed",
            lambda c: c["proof"]["facts"].pop(_target_fact_index(c)),
        ),
        (
            "bad_forward_premise",
            lambda c: c["proof"]["facts"][_forward_fact_index(c)].__setitem__(
                "premises", ["not_a_real_fact@0"]
            ),
        ),
        (
            "model_hash_changed",
            lambda c: c["model"].__setitem__("sha256", "0" * 64),
        ),
    ]
    for name, mutator in mutations:
        _assert_mutation_rejected(battery, base, name, mutator)


def main() -> int:
    out_dir = Path("results") / f"certification_battery_{time.strftime('%Y%m%d-%H%M%S')}"
    cert_dir = out_dir / "positive_certificates"
    cert_dir.mkdir(parents=True, exist_ok=True)
    battery = Battery(out_dir)

    _assert_solver_units(battery)
    _assert_equation_reconstructibility_units(battery)

    positive_certs = {}
    for name in ("mixing", "thermostat", "cruise-discrete"):
        positive_certs[name] = _assert_positive_model(battery, name, cert_dir)

    _assert_negative_models(battery)
    _assert_certificate_mutations(battery, positive_certs["mixing"])

    battery.report()
    print(f"\nWROTE {out_dir / 'report.json'}")
    if battery.failures:
        print("\nCERTIFICATION BATTERY FAILED")
        for failure in battery.failures:
            print(f"  - {failure}")
        return 1

    print("\nCERTIFICATION BATTERY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
