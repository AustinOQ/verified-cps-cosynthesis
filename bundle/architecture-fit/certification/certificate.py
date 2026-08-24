#!/usr/bin/env python3
"""Proof-artifact emission and checking for strict Q reconstructibility.

The certificate combines the replayed strict-Q reconstruction proof with a
solver-backed one-step uniqueness proof over the extracted equation semantics.
Passing artifacts may claim the solver-backed strict-Q MDP theorem only when
both gates discharge and the independent checker verifies the recorded evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from reconstruct_closure import get_strict_model

from .equations import Const, Diagnostic, Equation, Expr, Ite, Op, RawRef, Var
from .equation_reconstruct import (
    equation_reconstruction_trace,
    equation_search,
    fact_key as equation_fact_key,
)
from .relevance import compute_transition_closed_relevance, equation_refs
from .solver import MAX_SOLVER_POLYNOMIAL_DEGREE, one_step_transition_closure
from .strict_extract import extract_equation_model
from sysml_parser import BinaryExpr, LiteralExpr, RefExpr, SysMLParser, TernaryExpr, UnaryExpr
from runtime_settings import DEFAULT_DT, validate_dt


SCHEMA_VERSION = 2
PROOF_PROFILE = "strict_q_syntactic_reconstructibility_v1"
PROFILE_MDP_THEOREM = "strict_q_profile_mdp_v1"
SOLVER_BACKED_MDP_THEOREM = "strict_q_solver_backed_mdp_v1"
PROFILE_OBLIGATIONS_DISCHARGED = "profile_obligations_discharged_solver_gate_required"
BLOCKING_DIAGNOSTIC_SEVERITIES = {"warning", "error"}
BLOCKING_DIAGNOSTIC_CODES = {"legacy_dependency_fallback"}
SAFETY_REQUIREMENT_KINDS = {"Prohibition", "Obligation", "Requirement", ""}


def model_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fact_key(var: str, tau: int) -> str:
    return f"{var}@{tau}"


def _expr_to_dict(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, Const):
        return {"type": "const", "value": expr.value, "pretty": expr.pretty()}
    if isinstance(expr, Var):
        return {"type": "var", "name": expr.name, "pretty": expr.pretty()}
    if isinstance(expr, RawRef):
        return {"type": "raw_ref", "path": expr.path, "pretty": expr.pretty()}
    if isinstance(expr, Op):
        return {
            "type": "op",
            "op": expr.op,
            "args": [_expr_to_dict(arg) for arg in expr.args],
            "pretty": expr.pretty(),
        }
    if isinstance(expr, Ite):
        return {
            "type": "ite",
            "cond": _expr_to_dict(expr.cond),
            "then": _expr_to_dict(expr.then_expr),
            "else": _expr_to_dict(expr.else_expr),
            "pretty": expr.pretty(),
        }
    return {"type": "unknown", "pretty": expr.pretty()}


def _expr_raw_refs(expr_dict: dict[str, Any]) -> set[str]:
    if expr_dict.get("type") == "raw_ref":
        return {expr_dict.get("path", "")}
    refs: set[str] = set()
    for item in expr_dict.get("args", []):
        refs |= _expr_raw_refs(item)
    for key in ("cond", "then", "else"):
        item = expr_dict.get(key)
        if isinstance(item, dict):
            refs |= _expr_raw_refs(item)
    return refs


def _equation_to_dict(eq: Equation, constants: set[str] | None = None) -> dict[str, Any]:
    constants = constants or set()
    raw_refs = eq.raw_refs()
    return {
        "target": eq.target,
        "kind": eq.kind,
        "source": eq.source,
        "refs": sorted(eq.refs()),
        "raw_refs": sorted(raw_refs),
        "constant_refs": sorted(raw_refs & constants),
        "unresolved_raw_refs": sorted(raw_refs - constants),
        "expr": _expr_to_dict(eq.expr),
        "pretty": eq.pretty(),
    }


def _diagnostic_to_dict(diag: Diagnostic) -> dict[str, str]:
    return {
        "severity": diag.severity,
        "code": diag.code,
        "message": diag.message,
        "subject": diag.subject,
        "pretty": diag.pretty(),
    }


def _blocking_diagnostics(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    return [
        d for d in diagnostics
        if d.severity in BLOCKING_DIAGNOSTIC_SEVERITIES
        or d.code in BLOCKING_DIAGNOSTIC_CODES
    ]


def _expr_contains_legacy_dep(expr_dict: dict[str, Any]) -> bool:
    if expr_dict.get("type") == "op" and expr_dict.get("op") == "legacy_dep":
        return True
    for key in ("args",):
        for item in expr_dict.get(key, []):
            if _expr_contains_legacy_dep(item):
                return True
    for key in ("cond", "then", "else"):
        item = expr_dict.get(key)
        if isinstance(item, dict) and _expr_contains_legacy_dep(item):
            return True
    return False


def _theorem_gate(
    *,
    proof: dict[str, Any] | None,
    equation_proof: dict[str, Any] | None,
    blocking: list[Diagnostic],
    profile_obligations_discharged: bool,
    solver_result: dict[str, Any] | None,
) -> dict[str, Any]:
    profile_discharged = bool(
        proof
        and proof.get("passes")
        and equation_proof
        and equation_proof.get("passes")
        and not blocking
        and profile_obligations_discharged
    )
    solver_discharged = bool(
        solver_result
        and solver_result.get("status") == "discharged"
        and solver_result.get("claim") == "one_step_transition_closure"
    )
    blockers = []
    if not profile_discharged:
        blockers.append("profile_mdp_theorem_not_discharged")
    if not solver_discharged:
        blockers.append("solver_one_step_transition_closure_not_discharged")
    full_claim_allowed = profile_discharged and solver_discharged
    solver_gate = {
        "status": "discharged" if solver_discharged else "not_discharged",
        "source": "solver_advisory.one_step_transition_closure",
        "result_status": None if solver_result is None else solver_result.get("status"),
        "claim": None if solver_result is None else solver_result.get("claim"),
        "solver": None if solver_result is None else solver_result.get("solver"),
        "logic": None if solver_result is None else solver_result.get("logic"),
        "max_polynomial_degree": (
            None if solver_result is None else solver_result.get("max_polynomial_degree")
        ),
        "timeout_ms": None if solver_result is None else solver_result.get("timeout_ms"),
        "q_size": None if solver_result is None else solver_result.get("q_size"),
        "actions_size": (
            None if solver_result is None else solver_result.get("actions_size")
        ),
        "visible_terms_checked": (
            [] if solver_result is None else solver_result.get("visible_terms_checked", [])
        ),
        "theorem_statement": (
            "No two one-step executions of the extracted equation model can "
            "agree on current q and executed action while disagreeing on next "
            "q, next observations, next terminal equations, or next requirement "
            "status equations."
        ),
    }
    return {
        "profile": PROFILE_MDP_THEOREM,
        "profile_mdp_theorem": "discharged" if profile_discharged else "not_discharged",
        "solver_backed_profile": SOLVER_BACKED_MDP_THEOREM,
        "solver_backed_mdp_theorem": (
            "discharged" if full_claim_allowed else "not_discharged"
        ),
        "profile_theorem_statement": (
            "Within the extracted strict-Q syntactic semantics, the selected "
            "finite buffer reconstructs q and determines observations, "
            "requirement statuses, reward, done, and exact shield execution "
            "as recorded in mdp_obligations."
        ),
        "solver_backed_theorem_statement": (
            "Within the extracted equation semantics and stated solver "
            "fragment, the selected buffer is a sufficient Markov state for "
            "the transition-closed relevant state, visible outputs, reward, "
            "done, and shielded executed action."
        ),
        "profile_scope": [
            "extracted equation/dependency semantics",
            "finite-horizon augmented env.step_count",
            "deterministic exact shield semantics",
            "selected bounded buffer and replayed proof trace",
            "solver-backed one-step self-composition over q and executed action",
        ],
        "profile_requires": [
            "strict proof trace reconstructs every q variable at tau=0",
            "equation-IR proof trace reconstructs every q variable at tau=0",
            "no blocking extraction diagnostics",
            "no legacy dependency fallback transitions",
            "observation obligations discharged",
            "requirement-status obligations discharged",
            "reward and done obligations discharged over augmented state",
            "shield semantics discharged from covered inputs and mapped outputs",
            "solver proves one-step transition closure over extracted equations",
        ],
        "profile_obligations_discharged": profile_obligations_discharged,
        "solver_backed_uniqueness": solver_gate,
        "full_mdp_theorem_claim_allowed": full_claim_allowed,
        "full_mdp_theorem_blockers": blockers,
        "claim_policy": (
            "claim.profile_mdp_theorem, claim.solver_backed_mdp_theorem, and "
            "claim.mdp_theorem may be discharged only when profile obligations "
            "and solver-backed one-step uniqueness are both discharged."
        ),
    }


def _coverage_for_equation(
    eq_model,
    eq: Equation,
    q: set[str],
    actions: set[str],
) -> dict[str, Any]:
    refs = equation_refs(eq_model, eq)
    state_refs = {ref for ref in refs if ref in eq_model.state}
    action_refs = {ref for ref in refs if ref in actions}
    unresolved_refs = (
        refs - state_refs - action_refs - set(eq_model.definitions) - set(eq_model.constants)
    )
    raw_refs = eq.raw_refs()
    unresolved_raw_refs = raw_refs - set(eq_model.constants)
    return {
        "target": eq.target,
        "kind": eq.kind,
        "source": eq.source,
        "refs": sorted(refs),
        "state_refs": sorted(state_refs),
        "action_refs": sorted(action_refs),
        "constant_refs": sorted(raw_refs & set(eq_model.constants)),
        "unresolved_refs": sorted(unresolved_refs),
        "unresolved_raw_refs": sorted(unresolved_raw_refs),
        "covered_by_q_and_action": (
            state_refs <= q
            and action_refs <= actions
            and not unresolved_refs
            and not unresolved_raw_refs
        ),
        "pretty": eq.pretty(),
    }


def _parser_expr_to_dict(expr) -> dict[str, Any]:
    if isinstance(expr, LiteralExpr):
        return {"type": "literal", "value": expr.value}
    if isinstance(expr, RefExpr):
        return {"type": "ref", "path": list(expr.path), "pretty": ".".join(expr.path)}
    if isinstance(expr, UnaryExpr):
        return {
            "type": "unary",
            "op": expr.op,
            "operand": _parser_expr_to_dict(expr.operand),
        }
    if isinstance(expr, BinaryExpr):
        return {
            "type": "binary",
            "op": expr.op,
            "left": _parser_expr_to_dict(expr.left),
            "right": _parser_expr_to_dict(expr.right),
        }
    if isinstance(expr, TernaryExpr):
        return {
            "type": "ternary",
            "condition": _parser_expr_to_dict(expr.condition),
            "true": _parser_expr_to_dict(expr.true_expr),
            "false": _parser_expr_to_dict(expr.false_expr),
        }
    return {"type": type(expr).__name__}


def _neural_interface_summary(model_path: str, actions: set[str]) -> dict[str, Any]:
    parser = SysMLParser(model_path)
    parser.parse()

    ctrl_fqn = parser.controller_part
    if not ctrl_fqn:
        return {"status": "missing_controller", "input_params": [], "output_params": []}
    ctrl_inst = parser.part_instances.get(ctrl_fqn)
    if ctrl_inst is None:
        return {"status": "missing_controller_instance", "input_params": [], "output_params": []}
    ctrl_def = parser.part_defs.get(ctrl_inst.part_type)
    if ctrl_def is None:
        return {"status": "missing_controller_definition", "input_params": [], "output_params": []}

    neural_def = None
    for action_def in ctrl_def.action_defs:
        if "Neural" in action_def.metadata:
            neural_def = action_def
            break
    input_params = [] if neural_def is None else [p.name for p in neural_def.in_params]
    output_params = [] if neural_def is None else [p.name for p in neural_def.out_params]
    input_param_types = {} if neural_def is None else {
        p.name: getattr(p, "type_name", "") for p in neural_def.in_params
    }
    completion_params = [] if neural_def is None else [
        p.name for p in neural_def.in_params if "Completion" in p.metadata
    ]
    output_param_types = {} if neural_def is None else {
        p.name: getattr(p, "type_name", "") for p in neural_def.out_params
    }

    neural_requirement = None
    for req_name, subject_var, subject_type, req_expr, req_meta in ctrl_def.requirements:
        if "NeuralRequirement" in req_meta:
            neural_requirement = {
                "name": req_name,
                "subject_var": subject_var,
                "subject_type": subject_type,
                "metadata": req_meta,
                "raw_expression": req_expr,
            }
            break

    output_matches = {
        out: sorted(action for action in actions if action.endswith("_" + out) or action == out)
        for out in output_params
    }
    status = "recorded_not_semantically_proven"
    if neural_def is None:
        status = "missing_neural_action"
    elif neural_requirement is None:
        status = "missing_neural_requirement"
    elif any(not matches for matches in output_matches.values()):
        status = "output_action_mapping_incomplete"

    return {
        "status": status,
        "controller_part": ctrl_fqn,
        "input_params": input_params,
        "input_param_types": input_param_types,
        "completion_params": completion_params,
        "output_params": output_params,
        "output_param_types": output_param_types,
        "output_action_matches": output_matches,
        "neural_requirement": neural_requirement,
    }


def _shield_semantics_summary(
    eq_model,
    relevance,
    model_path: str,
    neural: dict[str, Any],
    observations: list[dict[str, Any]],
    terminals: list[dict[str, Any]],
) -> dict[str, Any]:
    observation_by_param = {
        item["target"].split(".", 1)[-1]: item for item in observations
    }
    terminal_by_param = {
        item["target"].split(".")[-1]: item
        for item in terminals
        if item["target"].startswith("env.completion.")
    }
    completion_params = set(neural.get("completion_params", []))

    if neural.get("neural_requirement") is None:
        return {
            "status": "not_discharged",
            "semantic_status": "missing_neural_requirement_ast",
            "runtime_class": "SpecShield",
            "unknown_requirement_refs": [],
            "interface": neural,
        }

    try:
        from shield import SpecShield, _collect_refs, _flatten_and  # type: ignore
    except Exception as exc:  # pragma: no cover - environment/configuration failure
        return {
            "status": "not_discharged",
            "semantic_status": "spec_shield_import_failed",
            "error": str(exc),
        }

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec = SpecShield(model_path)
    except Exception as exc:
        message = str(exc)
        unresolved_marker = "#NeuralRequirement contains unresolved references:"
        if unresolved_marker in message:
            unknown_refs = [
                item.strip()
                for item in message.split(unresolved_marker, 1)[1].split(",")
                if item.strip()
            ]
            return {
                "status": "not_discharged",
                "semantic_status": "unknown_requirement_refs",
                "runtime_class": "SpecShield",
                "unknown_requirement_refs": sorted(unknown_refs),
                "interface": neural,
                "error": message,
            }
        return {
            "status": "not_discharged",
            "semantic_status": "spec_shield_construction_failed",
            "error": message,
        }

    output_types = neural.get("output_param_types", {})
    continuous = (
        len(spec.out_params) == 1
        and output_types.get(spec.out_params[0]) == "Real"
    )
    if continuous:
        try:
            from continuous_shield import ContinuousShield  # type: ignore
            with contextlib.redirect_stdout(io.StringIO()):
                continuous_shield = ContinuousShield(model_path)
            continuous_details = {
                "runtime_class": "ContinuousShield",
                "projection": (
                    "safe output = clamp(proposed output, interval read from the "
                    "current SysML requirement); an empty interval is an error"
                ),
                "static_range": [
                    continuous_shield.act_low,
                    continuous_shield.act_high,
                ],
                "epsilon": continuous_shield.eps,
            }
        except Exception as exc:
            return {
                "status": "not_discharged",
                "semantic_status": "continuous_shield_construction_failed",
                "error": str(exc),
            }
    else:
        continuous_details = {}

    input_coverages = []
    missing_inputs = []
    for param in spec.in_params:
        if param in completion_params:
            source = terminal_by_param.get(param)
            source_kind = "terminal_state"
        else:
            source = observation_by_param.get(param)
            source_kind = "observation"
        covered = bool(source) and bool(source.get("covered_by_q_and_action"))
        if not covered:
            missing_inputs.append(param)
        input_coverages.append({
            "param": param,
            "source_kind": source_kind,
            "source_target": None if source is None else source["target"],
            "covered_by_q_and_action": covered,
        })

    output_action_matches = neural.get("output_action_matches", {})
    output_mappings = []
    missing_outputs = []
    for param in spec.out_params:
        matches = sorted(output_action_matches.get(param, []))
        if len(matches) != 1:
            missing_outputs.append(param)
        output_mappings.append({
            "param": param,
            "type": output_types.get(param, ""),
            "action_vars": matches,
            "unique_action_var": len(matches) == 1,
        })

    req_refs = set(_collect_refs(spec.req_ast)) if spec.req_ast is not None else set()
    known_refs = (
        {spec.subject_var}
        | set(spec.in_params)
        | set(spec.out_params)
        | set(spec.unchanging.keys())
    )
    unknown_refs = sorted(req_refs - known_refs)
    clauses = [] if spec.req_ast is None else [
        _parser_expr_to_dict(clause) for clause in _flatten_and(spec.req_ast)
    ]

    if spec.req_ast is None:
        semantic_status = "missing_neural_requirement_ast"
    elif unknown_refs:
        semantic_status = "unknown_requirement_refs"
    elif missing_inputs:
        semantic_status = "shield_inputs_not_covered"
    elif missing_outputs:
        semantic_status = "shield_outputs_not_mapped_to_actions"
    else:
        semantic_status = (
            "discharged_continuous_interval_projection"
            if continuous
            else "discharged_discrete_exact_ast"
        )

    discharged = semantic_status.startswith("discharged_")
    return {
        "status": "discharged" if discharged else "not_discharged",
        "semantic_status": semantic_status,
        "runtime_class": "ContinuousShield" if continuous else "SpecShield",
        "action_space": (
            "continuous_single_real_output"
            if continuous
            else "discrete_boolean_output_bitvector"
        ),
        "determinism_claim": (
            "executed action is a deterministic function of covered shield inputs "
            "and the policy proposed action"
        ),
        "executed_action_history": (
            "buffered environments push the shielded/executed action, not the raw proposal"
        ),
        "subject_var": spec.subject_var,
        "input_coverage": input_coverages,
        "missing_input_params": missing_inputs,
        "output_action_mapping": output_mappings,
        "missing_output_params": missing_outputs,
        "unknown_requirement_refs": unknown_refs,
        "constants": {k: spec.unchanging[k] for k in sorted(spec.unchanging)},
        "dead_actions": sorted(spec.dead_actions),
        "action_map": {
            str(action_id): spec.action_map[action_id]
            for action_id in sorted(spec.action_map)
        } if not continuous else {},
        "continuous_details": continuous_details,
        "predicate_ast": None if spec.req_ast is None else _parser_expr_to_dict(spec.req_ast),
        "clauses": clauses,
        "interface": neural,
    }


def _rl_obligations(eq_model, relevance, model_path: str) -> dict[str, Any]:
    q = set(relevance.q)
    actions = set(eq_model.actions)
    observations = [
        _coverage_for_equation(eq_model, eq, q, actions)
        for eq in sorted(eq_model.observations.values(), key=lambda e: e.target)
    ]
    terminals = [
        _coverage_for_equation(eq_model, eq, q, actions)
        for eq in sorted(eq_model.terminals.values(), key=lambda e: e.target)
    ]
    requirements = [
        _coverage_for_equation(eq_model, eq, q, actions)
        for eq in sorted(eq_model.requirements.values(), key=lambda e: e.target)
    ]
    safety_requirements = [
        item for item in requirements
        if item["source"].split(",", 1)[0] in SAFETY_REQUIREMENT_KINDS
    ]
    all_observations_covered = all(item["covered_by_q_and_action"] for item in observations)
    all_requirements_covered = all(item["covered_by_q_and_action"] for item in requirements)
    all_safety_covered = all(item["covered_by_q_and_action"] for item in safety_requirements)
    terminal_covered = bool(terminals) and all(
        item["covered_by_q_and_action"] for item in terminals
    )
    truncation_discharged = True
    reward_done_discharged = all_safety_covered and terminal_covered and truncation_discharged
    neural = _neural_interface_summary(model_path, actions)
    shield = _shield_semantics_summary(
        eq_model, relevance, model_path, neural, observations, terminals
    )
    profile_obligations_discharged = (
        all_observations_covered
        and all_requirements_covered
        and reward_done_discharged
        and shield.get("status") == "discharged"
    )

    return {
        "overall_mdp_theorem_status": (
            PROFILE_OBLIGATIONS_DISCHARGED
            if profile_obligations_discharged
            else "not_fully_discharged"
        ),
        "observation": {
            "status": (
                "covered_by_q" if all_observations_covered
                else "not_covered_by_q"
            ),
            "equations": observations,
        },
        "requirement_status": {
            "status": (
                "covered_by_q_and_action"
                if all_requirements_covered
                else "not_covered_by_q_and_action"
            ),
            "equations": requirements,
        },
        "terminal_state": {
            "status": (
                "covered_by_q_and_action"
                if terminals and all(item["covered_by_q_and_action"] for item in terminals)
                else "not_covered_by_q_and_action"
                if terminals
                else "absent_from_simulator_state"
            ),
            "runtime_source": "SysMLEnv reads the current #Completion input",
            "equations": terminals,
        },
        "truncation": {
            "status": "discharged_as_finite_horizon_augmented_state",
            "augmented_state_variable": "env.step_count",
            "reset": "env.step_count := 0 on reset after warmup calls",
            "transition": "env.step_count' := env.step_count + 1 after every env.step",
            "terminal_condition": "env.step_count >= env.max_steps",
            "reward_override": "if truncated and not already done: done=True, reward=0.0",
        },
        "reward": {
            "status": (
                "discharged_over_augmented_state"
                if reward_done_discharged
                else "not_discharged"
            ),
            "environment_formula": (
                "phase 2: if any safety requirement status is false, reward=-1 and done=True; "
                "else if the current #Completion input is true, reward=1 and done=True; "
                "else reward=-0.01 and done=False; max-step truncation overrides reward to 0."
            ),
            "covered_terms": {
                "safety_requirement_statuses": safety_requirements,
                "all_safety_statuses_covered": all_safety_covered,
                "terminal_state": terminals,
                "all_terminal_state_terms_covered": terminal_covered,
                "finite_horizon_truncation_covered": truncation_discharged,
            },
            "modeled_external_terms": [
                "current #Completion input via terminal_state equation",
                "environment step_count >= max_steps via augmented finite-horizon state",
            ] if reward_done_discharged else [],
            "external_terms_not_yet_in_q": [] if reward_done_discharged else [
                "reward/done not discharged because terminal_state or safety coverage failed",
            ],
        },
        "done": {
            "status": (
                "discharged_over_augmented_state"
                if reward_done_discharged
                else "not_discharged"
            ),
            "depends_on": [
                "safety requirement violation",
                "current #Completion input",
                "environment step_count >= max_steps",
            ],
            "covered": reward_done_discharged,
        },
        "shield": shield,
    }


def _add_fact(
    facts: dict[str, dict[str, Any]],
    var: str,
    tau: int,
    rule: str,
    premises: list[str] | None = None,
    detail: dict[str, Any] | None = None,
) -> bool:
    key = fact_key(var, tau)
    if key in facts:
        return False
    facts[key] = {
        "key": key,
        "var": var,
        "tau": tau,
        "rule": rule,
        "premises": premises or [],
        "detail": detail or {},
    }
    return True


def reconstruction_trace(
    model: dict[str, Any],
    b_obs: int,
    b_act: int,
    *,
    horizon: int = 14,
    target: set[str] | None = None,
) -> dict[str, Any]:
    """Replay the current syntactic proof path and record every derived fact."""

    state = set(model["STATE"])
    actions = set(model["ACTIONS"])
    nsupp = {k: set(v) for k, v in model["nsupp"].items()}
    copies = set(model["copies"])
    obs = set(model["OBS"])
    sampled_memories = list(model.get("sampled_memories", []))
    if target is None:
        target = state

    time_vars = set(model.get("known_schedule_state", []))
    taus = range(-horizon, 1)
    facts: dict[str, dict[str, Any]] = {}

    for tau in taus:
        for var in sorted(time_vars):
            _add_fact(facts, var, tau, "deterministic_time_known")

    for var in sorted(obs):
        for tau in range(-b_obs, 1):
            _add_fact(facts, var, tau, "observation_buffer")

    for var in sorted(actions):
        for tau in range(-b_act, 0):
            _add_fact(facts, var, tau, "executed_action_history")

    changed = True
    while changed:
        changed = False
        for var in sorted(state):
            supp = set(nsupp.get(var, set()))
            for tau in taus:
                next_tau = tau + 1
                if next_tau in taus and fact_key(var, next_tau) not in facts:
                    premises = [fact_key(u, tau) for u in sorted(supp)]
                    if all(p in facts for p in premises):
                        changed |= _add_fact(
                            facts,
                            var,
                            next_tau,
                            "forward_transition",
                            premises,
                            {"transition_var": var, "from_tau": tau, "support": sorted(supp)},
                        )

                if var in copies and supp:
                    source = sorted(supp)[0]
                    premise = fact_key(var, next_tau)
                    if premise in facts and fact_key(source, tau) not in facts:
                        changed |= _add_fact(
                            facts,
                            source,
                            tau,
                            "copy_inversion",
                            [premise],
                            {
                                "transition_var": var,
                                "source": source,
                                "known_transition_tau": next_tau,
                            },
                        )

        for rule in sampled_memories:
            target_var = rule["target"]
            source = rule["source"]
            delay = int(rule["max_delay"])
            for tau in taus:
                if fact_key(target_var, tau) in facts:
                    continue
                start = tau - delay
                if start < min(taus):
                    continue
                premises = [fact_key(source, src_tau) for src_tau in range(start, tau)]
                if all(p in facts for p in premises):
                    changed |= _add_fact(
                        facts,
                        target_var,
                        tau,
                        "sampled_memory_bound",
                        premises,
                        {
                            "target": target_var,
                            "source": source,
                            "max_delay": delay,
                            "tau": tau,
                        },
                    )

    missing = sorted(var for var in target if fact_key(var, 0) not in facts)
    return {
        "horizon": horizon,
        "b_obs": b_obs,
        "b_act": b_act,
        "facts": list(facts.values()),
        "target_facts": [fact_key(var, 0) for var in sorted(target)],
        "missing_target_facts": [fact_key(var, 0) for var in missing],
        "passes": not missing,
    }


def _strict_search(
    model: dict[str, Any],
    *,
    max_obs: int,
    max_act: int,
    horizon: int,
) -> tuple[tuple[int, int] | None, list[dict[str, Any]]]:
    target = set(model.get("R", set())) or set(model["STATE"])
    attempts: list[dict[str, Any]] = []
    selected: tuple[int, int] | None = None
    for b_obs in range(max_obs + 1):
        for b_act in range(max_act + 1):
            trace = reconstruction_trace(
                model, b_obs, b_act, horizon=horizon, target=target
            )
            missing = [item.rsplit("@", 1)[0] for item in trace["missing_target_facts"]]
            attempts.append(
                {
                    "b_obs": b_obs,
                    "b_act": b_act,
                    "passes": trace["passes"],
                    "missing": sorted(missing),
                }
            )
            if trace["passes"] and selected is None:
                selected = (b_obs, b_act)
    return selected, attempts


def build_certificate_for_path(
    model_path: str,
    *,
    b_obs: int | None = None,
    b_act: int | None = None,
    max_obs: int = 2,
    max_act: int = 4,
    horizon: int = 14,
    dt: float,
    enable_sampled_memory: bool = True,
    include_solver_artifacts: bool = False,
) -> dict[str, Any]:
    dt = validate_dt(dt)
    eq_model = extract_equation_model(model_path)
    relevance = compute_transition_closed_relevance(eq_model)
    strict_model = get_strict_model(
        model_path, dt=dt, enable_sampled_memory=enable_sampled_memory
    )
    target = set(strict_model.get("R", set())) or set(strict_model["STATE"])
    blocking = _blocking_diagnostics(eq_model.diagnostics)

    selected, attempts = _strict_search(
        strict_model, max_obs=max_obs, max_act=max_act, horizon=horizon
    )
    equation_selected, equation_attempts = equation_search(
        eq_model,
        max_obs=max_obs,
        max_act=max_act,
        horizon=horizon,
        target=set(relevance.q),
        dt=dt,
        enable_sampled_memory=enable_sampled_memory,
    )
    if b_obs is None or b_act is None:
        if selected is not None:
            b_obs, b_act = selected
    else:
        selected = (b_obs, b_act)

    proof = None
    equation_proof = None
    result = "UNKNOWN"
    if b_obs is not None and b_act is not None:
        proof = reconstruction_trace(
            strict_model, b_obs, b_act, horizon=horizon, target=target
        )
        equation_proof = equation_reconstruction_trace(
            eq_model,
            b_obs,
            b_act,
            horizon=horizon,
            target=set(relevance.q),
            dt=dt,
            enable_sampled_memory=enable_sampled_memory,
        )
        result = "PASS" if proof["passes"] and not blocking else "FAIL"

    ignored = sorted(set(eq_model.state) - set(relevance.q))
    transition_kinds = {
        eq.target: eq.kind for eq in eq_model.transitions.values()
    }
    legacy_fallbacks = sorted(
        target for target, kind in transition_kinds.items()
        if kind == "legacy_transition_dependency"
    )
    equation_sections = {
        "definitions": [_equation_to_dict(eq, eq_model.constants) for eq in sorted(
            eq_model.definitions.values(), key=lambda e: e.target
        )],
        "observations": [_equation_to_dict(eq, eq_model.constants) for eq in sorted(
            eq_model.observations.values(), key=lambda e: e.target
        )],
        "terminals": [_equation_to_dict(eq, eq_model.constants) for eq in sorted(
            eq_model.terminals.values(), key=lambda e: e.target
        )],
        "transitions": [_equation_to_dict(eq, eq_model.constants) for eq in sorted(
            eq_model.transitions.values(), key=lambda e: e.target
        )],
        "requirements": [_equation_to_dict(eq, eq_model.constants) for eq in sorted(
            eq_model.requirements.values(), key=lambda e: e.target
        )],
    }
    mdp_obligations = _rl_obligations(eq_model, relevance, model_path)
    profile_obligations_discharged = (
        mdp_obligations.get("overall_mdp_theorem_status")
        == PROFILE_OBLIGATIONS_DISCHARGED
    )
    solver_advisory = {
        "one_step_transition_closure": one_step_transition_closure(
            eq_model,
            set(relevance.q),
            timeout_ms=1000,
            include_artifacts=include_solver_artifacts,
        )
    }
    theorem_gate = _theorem_gate(
        proof=proof,
        equation_proof=equation_proof,
        blocking=blocking,
        profile_obligations_discharged=profile_obligations_discharged,
        solver_result=solver_advisory["one_step_transition_closure"],
    )
    if proof is not None:
        result = (
            "PASS"
            if theorem_gate["full_mdp_theorem_claim_allowed"] is True
            else "FAIL"
        )
    mdp_theorem_claim = (
        "discharged"
        if theorem_gate["full_mdp_theorem_claim_allowed"] is True
        else "not_claimed_by_this_artifact"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "proof_profile": PROOF_PROFILE,
        "result": result,
        "claim": {
            "level": (
                SOLVER_BACKED_MDP_THEOREM
                if mdp_theorem_claim == "discharged"
                else PROFILE_MDP_THEOREM
            ),
            "profile_mdp_theorem": theorem_gate["profile_mdp_theorem"],
            "mdp_theorem": mdp_theorem_claim,
            "solver_backed_mdp_theorem": theorem_gate["solver_backed_mdp_theorem"],
            "description": (
                "Proof that the selected finite buffer reconstructs the "
                "transition-closed relevant state q under the extracted "
                "equation/dependency semantics, and that Z3 discharges the "
                "one-step self-composed uniqueness obligation over q, executed "
                "actions, observations, terminal equations, and requirement "
                "status equations. Reward and done obligations are discharged "
                "over the finite-horizon augmented state when mdp_obligations "
                "says so. Shield obligations are discharged as deterministic "
                "exact-shield semantics when mdp_obligations says so."
            ),
        },
        "model": {
            "path": os.path.abspath(model_path),
            "sha256": model_hash(model_path),
        },
        "settings": {
            "dt": dt,
            "max_obs": max_obs,
            "max_act": max_act,
            "horizon": horizon,
            "enable_sampled_memory": enable_sampled_memory,
            "search_order": "lexicographic b_obs then b_act within stated bounds",
        },
        "proof_mode": {
            "strict_no_legacy_fallback": True,
            "blocking_diagnostic_severities": sorted(BLOCKING_DIAGNOSTIC_SEVERITIES),
            "blocking_diagnostic_codes": sorted(BLOCKING_DIAGNOSTIC_CODES),
            "pass_requires": [
                "selected buffer reconstructs every q variable at tau=0",
                "equation-IR proof trace reconstructs every q variable at tau=0",
                "no warning/error diagnostics",
                "no legacy dependency fallback transitions",
                "reward and done obligations discharged over the augmented state",
                "shield semantics discharged as deterministic function of covered inputs",
                "solver-backed one-step transition closure is discharged",
                "independent checker can replay every proof fact",
            ],
        },
        "buffer": (
            None if b_obs is None or b_act is None
            else {"b_obs": b_obs, "b_act": b_act}
        ),
        "search": {
            "selected_first_passing": (
                None if selected is None
                else {"b_obs": selected[0], "b_act": selected[1]}
            ),
            "attempts": attempts,
            "equation_ir_selected_first_passing": (
                None if equation_selected is None
                else {"b_obs": equation_selected[0], "b_act": equation_selected[1]}
            ),
            "equation_ir_attempts": equation_attempts,
            "legacy_and_equation_search_agree": selected == equation_selected,
            "minimality_claim": "first passing pair in stated bounded lexicographic scan",
        },
        "sets": {
            "state": sorted(eq_model.state),
            "actions": sorted(eq_model.actions),
            "constants": sorted(eq_model.constants),
            "initial_values": {
                key: eq_model.initial_values[key]
                for key in sorted(eq_model.initial_values)
            },
            "observed_state_vars": sorted(strict_model["OBS"]),
            "time_vars": sorted(strict_model.get("known_schedule_state", [])),
            "q": sorted(relevance.q),
            "q_seeds": sorted(relevance.seeds),
            "action_deps": sorted(relevance.action_deps),
            "ignored_state": ignored,
            "ignored_state_reasons": {
                key: relevance.ignored_state_reasons[key]
                for key in sorted(relevance.ignored_state_reasons)
            },
            "seed_equations": {
                key: relevance.seed_equations[key]
                for key in sorted(relevance.seed_equations)
            },
        },
        "dependency_model": {
            "nsupp": {k: sorted(v) for k, v in sorted(strict_model["nsupp"].items())},
            "copies": sorted(strict_model["copies"]),
            "sampled_memories": strict_model.get("sampled_memories", []),
        },
        "equations": equation_sections,
        "diagnostics": {
            "all": [_diagnostic_to_dict(diag) for diag in eq_model.diagnostics],
            "blocking": [_diagnostic_to_dict(diag) for diag in blocking],
            "legacy_dependency_fallback_transitions": legacy_fallbacks,
        },
        "assumptions": strict_model.get("assumptions", []),
        "mdp_obligations": mdp_obligations,
        "theorem_gate": theorem_gate,
        "solver_advisory": solver_advisory,
        "proof": proof,
        "equation_proof": equation_proof,
    }


def write_certificate(certificate: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")


def load_certificate(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_certificate(certificate: dict[str, Any], *, check_hash: bool = True) -> list[str]:
    errors: list[str] = []
    if certificate.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={certificate.get('schema_version')}")
    if certificate.get("proof_profile") != PROOF_PROFILE:
        errors.append(f"unsupported proof_profile={certificate.get('proof_profile')}")
    if certificate.get("result") != "PASS":
        errors.append(f"certificate result is not PASS: {certificate.get('result')}")
    claim = certificate.get("claim", {})
    if claim.get("level") != SOLVER_BACKED_MDP_THEOREM:
        errors.append(f"unsupported claim level={claim.get('level')}")
    if claim.get("profile_mdp_theorem") != "discharged":
        errors.append("profile MDP theorem is not discharged")
    if claim.get("mdp_theorem") != "discharged":
        errors.append("solver-backed MDP theorem is not discharged")
    if claim.get("solver_backed_mdp_theorem") != "discharged":
        errors.append("certificate does not claim discharged solver-backed MDP theorem")

    model_info = certificate.get("model", {})
    path = model_info.get("path")
    if check_hash and path and os.path.exists(path):
        observed = model_hash(path)
        if observed != model_info.get("sha256"):
            errors.append("model sha256 does not match certificate")

    diagnostics = certificate.get("diagnostics", {})
    if diagnostics.get("blocking"):
        errors.append("certificate contains blocking diagnostics")
    if diagnostics.get("legacy_dependency_fallback_transitions"):
        errors.append("certificate uses legacy dependency fallback transitions")
    proof_mode = certificate.get("proof_mode", {})
    if not proof_mode.get("strict_no_legacy_fallback"):
        errors.append("certificate proof mode is not strict no-fallback")
    for section, equations in certificate.get("equations", {}).items():
        for eq in equations:
            if eq.get("kind") == "legacy_transition_dependency":
                errors.append(f"{section} contains legacy transition dependency: {eq.get('target')}")
            if _expr_contains_legacy_dep(eq.get("expr", {})):
                errors.append(f"{section} contains legacy_dep expression: {eq.get('target')}")
            if eq.get("unresolved_raw_refs"):
                errors.append(f"{section} contains unresolved raw refs: {eq.get('target')}")
            expr_raw_refs = _expr_raw_refs(eq.get("expr", {}))
            declared_raw_refs = set(eq.get("raw_refs", []))
            if expr_raw_refs != declared_raw_refs:
                errors.append(f"{section} raw-ref audit mismatch: {eq.get('target')}")
            if set(eq.get("unresolved_raw_refs", [])) & set(eq.get("constant_refs", [])):
                errors.append(f"{section} raw-ref classification conflict: {eq.get('target')}")

    obligations = certificate.get("mdp_obligations", {})
    if obligations.get("overall_mdp_theorem_status") != PROFILE_OBLIGATIONS_DISCHARGED:
        errors.append("profile obligations are not discharged or theorem boundary is unclear")
    theorem_gate = certificate.get("theorem_gate", {})
    if theorem_gate.get("profile") != PROFILE_MDP_THEOREM:
        errors.append(f"unsupported theorem gate profile={theorem_gate.get('profile')}")
    if theorem_gate.get("solver_backed_profile") != SOLVER_BACKED_MDP_THEOREM:
        errors.append(
            f"unsupported solver-backed theorem profile={theorem_gate.get('solver_backed_profile')}"
        )
    if theorem_gate.get("profile_mdp_theorem") != "discharged":
        errors.append("theorem gate profile theorem is not discharged")
    if theorem_gate.get("solver_backed_mdp_theorem") != "discharged":
        errors.append("theorem gate solver-backed MDP theorem is not discharged")
    if theorem_gate.get("profile_obligations_discharged") is not True:
        errors.append("theorem gate profile obligations are not discharged")
    solver_gate = theorem_gate.get("solver_backed_uniqueness", {})
    if solver_gate.get("status") != "discharged":
        errors.append("solver-backed uniqueness status is not discharged")
    if solver_gate.get("source") != "solver_advisory.one_step_transition_closure":
        errors.append("solver-backed uniqueness source is not the recorded solver advisory")
    if solver_gate.get("result_status") != "discharged":
        errors.append("solver-backed uniqueness result status is not discharged")
    if solver_gate.get("claim") != "one_step_transition_closure":
        errors.append("solver-backed uniqueness claim label is wrong")
    if solver_gate.get("solver") != "z3":
        errors.append("solver-backed uniqueness must be discharged by z3")
    if solver_gate.get("logic") not in {"QF_UFLRA", "QF_UFNRA"}:
        errors.append(f"unsupported solver logic={solver_gate.get('logic')}")
    degree = solver_gate.get("max_polynomial_degree")
    if (
        not isinstance(degree, int)
        or degree < 0
        or degree > MAX_SOLVER_POLYNOMIAL_DEGREE
    ):
        errors.append(f"unsupported solver polynomial degree={degree}")
    if not solver_gate.get("visible_terms_checked"):
        errors.append("solver-backed uniqueness records no visible terms checked")
    if theorem_gate.get("full_mdp_theorem_claim_allowed") is not True:
        errors.append("theorem gate does not allow the solver-backed MDP claim")
    if theorem_gate.get("full_mdp_theorem_blockers"):
        errors.append("theorem gate records blockers despite discharged solver-backed claim")

    sets_for_noninterference = certificate.get("sets", {})
    ignored_state = set(sets_for_noninterference.get("ignored_state", []))
    ignored_reasons = sets_for_noninterference.get("ignored_state_reasons", {})
    if ignored_state and not isinstance(ignored_reasons, dict):
        errors.append("ignored-state reasons are not recorded as a mapping")
        ignored_reasons = {}
    for var in sorted(ignored_state):
        reason = ignored_reasons.get(var)
        if not isinstance(reason, str) or "backward cone" not in reason:
            errors.append(f"ignored state lacks noninterference reason: {var}")
    for var in sorted(set(ignored_reasons) - ignored_state):
        errors.append(f"ignored-state reason recorded for non-ignored state: {var}")

    solver_advisory = certificate.get("solver_advisory", {})
    one_step = solver_advisory.get("one_step_transition_closure")
    if one_step is None:
        errors.append("missing solver advisory one-step transition-closure section")
    elif one_step.get("status") not in {
        "discharged",
        "counterexample",
        "unknown",
        "unavailable",
    }:
        errors.append(f"unsupported solver advisory status={one_step.get('status')}")
    elif one_step.get("status") == "discharged":
        if one_step.get("claim") != "one_step_transition_closure":
            errors.append("solver advisory discharged status has wrong claim label")
    elif one_step.get("claim") != "not_claimed_by_this_artifact":
        errors.append("non-discharged solver advisory must not claim theorem support")
    if one_step is not None:
        if one_step.get("status") != "discharged":
            errors.append("solver advisory one-step transition closure is not discharged")
        for key in (
            "claim",
            "solver",
            "logic",
            "max_polynomial_degree",
            "timeout_ms",
            "q_size",
            "actions_size",
            "visible_terms_checked",
        ):
            if solver_gate.get(key) != one_step.get(key):
                errors.append(f"solver theorem gate/advisory mismatch: {key}")
    for key in (
        "observation",
        "requirement_status",
        "terminal_state",
        "truncation",
        "reward",
        "done",
        "shield",
    ):
        if key not in obligations:
            errors.append(f"missing MDP obligation section: {key}")
    observation = obligations.get("observation", {})
    if observation and observation.get("status") != "covered_by_q":
        errors.append("observation obligations are not covered by q")
    requirement_status = obligations.get("requirement_status", {})
    if requirement_status and requirement_status.get("status") != "covered_by_q_and_action":
        errors.append("requirement-status obligations are not covered by q and action")
    reward = obligations.get("reward", {})
    if reward:
        if reward.get("status") != "discharged_over_augmented_state":
            errors.append("reward obligation is not discharged over augmented state")
        covered_terms = reward.get("covered_terms", {})
        if covered_terms.get("all_safety_statuses_covered") is not True:
            errors.append("reward safety-status terms are not fully covered")
        if covered_terms.get("all_terminal_state_terms_covered") is not True:
            errors.append("reward terminal-state terms are not fully covered")
        if covered_terms.get("finite_horizon_truncation_covered") is not True:
            errors.append("reward truncation term is not covered")
        if reward.get("external_terms_not_yet_in_q"):
            errors.append("reward obligation still has external terms outside q")
    terminal_state = obligations.get("terminal_state", {})
    if terminal_state and terminal_state.get("status") not in {
        "covered_by_q_and_action",
        "not_covered_by_q_and_action",
        "absent_from_simulator_state",
    }:
        errors.append("terminal-state obligation has unknown status")
    if terminal_state:
        if terminal_state.get("status") != "covered_by_q_and_action":
            errors.append("terminal-state obligation is not covered by q and action")
        if not terminal_state.get("equations"):
            errors.append("terminal-state obligation does not record terminal equations")
    truncation = obligations.get("truncation", {})
    if truncation and truncation.get("status") != "discharged_as_finite_horizon_augmented_state":
        errors.append("truncation obligation is not discharged as finite-horizon augmented state")
    done = obligations.get("done", {})
    if done:
        if done.get("status") != "discharged_over_augmented_state":
            errors.append("done obligation is not discharged over augmented state")
        if done.get("covered") is not True:
            errors.append("done obligation covered flag is not true")
    shield = obligations.get("shield", {})
    if shield:
        if shield.get("status") != "discharged":
            errors.append("shield obligation is not discharged")
        if shield.get("semantic_status") not in {
            "discharged_discrete_exact_ast",
            "discharged_continuous_interval_projection",
        }:
            errors.append("shield semantic status is not discharged")
        if shield.get("runtime_class") not in {"SpecShield", "ContinuousShield"}:
            errors.append("shield runtime class is not recognized")
        if shield.get("missing_input_params"):
            errors.append("shield has missing input params")
        if shield.get("missing_output_params"):
            errors.append("shield has missing output params")
        if shield.get("unknown_requirement_refs"):
            errors.append("shield has unknown requirement refs")
        if not shield.get("predicate_ast"):
            errors.append("shield predicate AST is missing")
        if not shield.get("input_coverage"):
            errors.append("shield input coverage is missing")
        for item in shield.get("input_coverage", []):
            if item.get("covered_by_q_and_action") is not True:
                errors.append(f"shield input is not covered: {item.get('param')}")
        if not shield.get("output_action_mapping"):
            errors.append("shield output action mapping is missing")
        for item in shield.get("output_action_mapping", []):
            if item.get("unique_action_var") is not True:
                errors.append(f"shield output is not uniquely mapped: {item.get('param')}")
        if not shield.get("executed_action_history"):
            errors.append("shield does not record executed-action history semantics")

    proof = certificate.get("proof")
    if not proof:
        errors.append("missing proof section")
        return errors
    if not proof.get("passes"):
        errors.append("proof section does not pass")

    settings = certificate.get("settings", {})
    try:
        validate_dt(settings.get("dt"))
    except (TypeError, ValueError):
        errors.append(f"invalid certificate dt={settings.get('dt')}")
    buffer = certificate.get("buffer") or {}
    sets = certificate.get("sets", {})
    deps = certificate.get("dependency_model", {})

    b_obs = int(buffer.get("b_obs", -1))
    b_act = int(buffer.get("b_act", -1))
    horizon = int(proof.get("horizon", settings.get("horizon", -1)))
    if proof.get("b_obs") != b_obs or proof.get("b_act") != b_act:
        errors.append("proof buffer does not match certificate buffer")

    state = set(sets.get("state", []))
    actions = set(sets.get("actions", []))
    obs = set(sets.get("observed_state_vars", []))
    time_vars = set(sets.get("time_vars", []))
    initial_values = sets.get("initial_values", {})
    if not isinstance(initial_values, dict):
        errors.append("sets.initial_values is not an object")
        initial_values = {}
    for var in sorted(time_vars):
        if var not in initial_values:
            errors.append(f"known schedule state lacks a SysML initial value: {var}")
    nsupp = {k: set(v) for k, v in deps.get("nsupp", {}).items()}
    copies = set(deps.get("copies", []))
    sampled_memories = {
        rule["target"]: rule for rule in deps.get("sampled_memories", [])
    }
    min_tau = -horizon
    max_tau = 0

    known: set[str] = set()
    for fact in proof.get("facts", []):
        key = fact.get("key")
        var = fact.get("var")
        tau = fact.get("tau")
        rule = fact.get("rule")
        premises = fact.get("premises", [])
        detail = fact.get("detail", {})

        if key != fact_key(var, tau):
            errors.append(f"malformed fact key: {fact}")
            continue
        if key in known:
            errors.append(f"duplicate fact: {key}")
            continue
        if not isinstance(tau, int) or tau < min_tau or tau > max_tau:
            errors.append(f"fact tau out of proof horizon: {key}")
            continue

        if rule == "deterministic_time_known":
            if var not in time_vars:
                errors.append(f"time fact for non-time variable: {key}")
                continue
        elif rule == "observation_buffer":
            if var not in obs or not (-b_obs <= tau <= 0):
                errors.append(f"invalid observation-buffer fact: {key}")
                continue
        elif rule == "executed_action_history":
            if var not in actions or not (-b_act <= tau < 0):
                errors.append(f"invalid action-history fact: {key}")
                continue
        elif rule == "forward_transition":
            transition_var = detail.get("transition_var")
            from_tau = detail.get("from_tau")
            support = set(detail.get("support", []))
            expected_premises = {fact_key(u, from_tau) for u in support}
            if var != transition_var or tau != from_tau + 1:
                errors.append(f"invalid forward-transition target: {key}")
                continue
            if support != nsupp.get(var, set()):
                errors.append(f"forward support mismatch for {key}")
                continue
            if set(premises) != expected_premises or not expected_premises <= known:
                errors.append(f"forward premises not established for {key}")
                continue
        elif rule == "copy_inversion":
            transition_var = detail.get("transition_var")
            known_transition_tau = detail.get("known_transition_tau")
            if transition_var not in copies:
                errors.append(f"copy inversion from non-copy transition: {key}")
                continue
            support = nsupp.get(transition_var, set())
            if support != {var}:
                errors.append(f"copy inversion support mismatch for {key}")
                continue
            expected = {fact_key(transition_var, known_transition_tau)}
            if known_transition_tau != tau + 1 or set(premises) != expected or not expected <= known:
                errors.append(f"copy inversion premise not established for {key}")
                continue
        elif rule == "sampled_memory_bound":
            target_var = detail.get("target")
            source = detail.get("source")
            delay = int(detail.get("max_delay", -1))
            if var != target_var or target_var not in sampled_memories:
                errors.append(f"invalid sampled-memory fact: {key}")
                continue
            rule_def = sampled_memories[target_var]
            if rule_def.get("source") != source or int(rule_def.get("max_delay")) != delay:
                errors.append(f"sampled-memory rule mismatch for {key}")
                continue
            start = tau - delay
            expected = {fact_key(source, src_tau) for src_tau in range(start, tau)}
            if set(premises) != expected or not expected <= known:
                errors.append(f"sampled-memory premises not established for {key}")
                continue
        else:
            errors.append(f"unknown proof rule {rule!r} for {key}")
            continue

        known.add(key)

    for target_key in proof.get("target_facts", []):
        if target_key not in known:
            errors.append(f"target fact not established: {target_key}")
    for missing in proof.get("missing_target_facts", []):
        errors.append(f"proof reports missing target fact: {missing}")

    equation_proof = certificate.get("equation_proof")
    if not equation_proof:
        errors.append("missing equation proof section")
        return errors
    if equation_proof.get("proof_engine") != "equation_ir_syntactic_v1":
        errors.append("equation proof engine is unsupported")
    if not equation_proof.get("passes"):
        errors.append("equation proof section does not pass")
    if equation_proof.get("b_obs") != b_obs or equation_proof.get("b_act") != b_act:
        errors.append("equation proof buffer does not match certificate buffer")

    eq_horizon = int(equation_proof.get("horizon", settings.get("horizon", -1)))
    eq_min_tau = -eq_horizon
    eq_max_tau = 0
    transition_equations = {
        eq.get("target"): eq for eq in certificate.get("equations", {}).get("transitions", [])
    }
    observation_equations = {
        eq.get("target"): eq for eq in certificate.get("equations", {}).get("observations", [])
    }
    eq_known: set[str] = set()
    for fact in equation_proof.get("facts", []):
        key = fact.get("key")
        var = fact.get("var")
        tau = fact.get("tau")
        rule = fact.get("rule")
        premises = fact.get("premises", [])
        detail = fact.get("detail", {})

        if key != equation_fact_key(var, tau):
            errors.append(f"malformed equation-proof fact key: {fact}")
            continue
        if key in eq_known:
            errors.append(f"duplicate equation-proof fact: {key}")
            continue
        if not isinstance(tau, int) or tau < eq_min_tau or tau > eq_max_tau:
            errors.append(f"equation-proof fact tau out of horizon: {key}")
            continue

        if rule == "deterministic_time_known":
            if var not in time_vars:
                errors.append(f"equation proof time fact for non-time variable: {key}")
                continue
        elif rule == "equation_observation_direct_copy":
            if var not in state or not (-b_obs <= tau <= 0):
                errors.append(f"invalid equation observation fact: {key}")
                continue
            eq_target = detail.get("equation_target")
            eq = observation_equations.get(eq_target)
            if eq is None or eq.get("pretty") != detail.get("equation_pretty"):
                errors.append(f"equation observation fact does not cite a known equation: {key}")
                continue
            if detail.get("source") != var:
                errors.append(f"equation observation source mismatch: {key}")
                continue
        elif rule == "executed_action_history":
            if var not in actions or not (-b_act <= tau < 0):
                errors.append(f"invalid equation action-history fact: {key}")
                continue
        elif rule == "equation_forward_transition":
            eq_target = detail.get("equation_target")
            from_tau = detail.get("from_tau")
            support = set(detail.get("support", []))
            expected_premises = {equation_fact_key(ref, from_tau) for ref in support}
            eq = transition_equations.get(eq_target)
            if eq_target != var or tau != from_tau + 1:
                errors.append(f"invalid equation-forward target: {key}")
                continue
            if eq is None or eq.get("pretty") != detail.get("equation_pretty"):
                errors.append(f"equation-forward fact does not cite a known transition: {key}")
                continue
            if set(premises) != expected_premises or not expected_premises <= eq_known:
                errors.append(f"equation-forward premises not established for {key}")
                continue
        elif rule == "equation_direct_copy_inversion":
            eq_target = detail.get("equation_target")
            known_transition_tau = detail.get("known_transition_tau")
            eq = transition_equations.get(eq_target)
            expected = {equation_fact_key(eq_target, known_transition_tau)}
            if detail.get("source") != var:
                errors.append(f"equation copy-inversion source mismatch: {key}")
                continue
            if detail.get("inversion") != "direct_transition_copy":
                errors.append(f"equation copy-inversion kind is not direct: {key}")
                continue
            if eq is None or eq.get("pretty") != detail.get("equation_pretty"):
                errors.append(f"equation copy-inversion does not cite a known transition: {key}")
                continue
            if known_transition_tau != tau + 1 or set(premises) != expected or not expected <= eq_known:
                errors.append(f"equation copy-inversion premise not established for {key}")
                continue
        elif rule == "equation_guarded_copy_inversion":
            if detail.get("guard_status") != "proven_true":
                errors.append(f"unsafe guarded-copy inversion in equation proof: {key}")
                continue
            if detail.get("inversion") != "guarded_transition_copy":
                errors.append(f"malformed guarded-copy inversion in equation proof: {key}")
                continue
            expected = {equation_fact_key(detail.get("equation_target"), tau + 1)}
            if set(premises) != expected or not expected <= eq_known:
                errors.append(f"guarded-copy inversion premise not established for {key}")
                continue
        elif rule == "sampled_memory_bound":
            target_var = detail.get("target")
            source = detail.get("source")
            delay = int(detail.get("max_delay", -1))
            if var != target_var or delay < 1:
                errors.append(f"invalid equation sampled-memory fact: {key}")
                continue
            start = tau - delay
            expected = {equation_fact_key(source, src_tau) for src_tau in range(start, tau)}
            if set(premises) != expected or not expected <= eq_known:
                errors.append(f"equation sampled-memory premises not established for {key}")
                continue
        else:
            errors.append(f"unknown equation-proof rule {rule!r} for {key}")
            continue

        eq_known.add(key)

    for target_key in equation_proof.get("target_facts", []):
        if target_key not in eq_known:
            errors.append(f"equation target fact not established: {target_key}")
    for missing in equation_proof.get("missing_target_facts", []):
        errors.append(f"equation proof reports missing target fact: {missing}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+", help="SysML file path")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out", default=None, help="single output file; only valid for one model")
    ap.add_argument("--max-obs", type=int, default=2)
    ap.add_argument("--max-act", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    ap.add_argument("--no-sampled-memory", action="store_true")
    ap.add_argument("--check", action="store_true", help="run the independent checker")
    args = ap.parse_args()

    if args.out and len(args.model) != 1:
        raise SystemExit("--out requires exactly one model")

    out_dir = Path(args.out_dir or "certificates")
    status = 0
    for item in args.model:
        path = item
        cert = build_certificate_for_path(
            path,
            max_obs=args.max_obs,
            max_act=args.max_act,
            horizon=args.horizon,
            dt=args.dt,
            enable_sampled_memory=not args.no_sampled_memory,
        )
        out = Path(args.out) if args.out else out_dir / f"{Path(path).stem}-{item}.certificate.json"
        write_certificate(cert, out)
        print(f"WROTE {out} result={cert['result']} buffer={cert.get('buffer')}")
        if args.check:
            errors = check_certificate(cert)
            if errors:
                status = 1
                print("CHECK FAILED")
                for err in errors:
                    print(f"  - {err}")
            else:
                print("CHECK PASSED")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
