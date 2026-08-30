#!/usr/bin/env python3
"""Certified reduced-MDP architecture contract.

The strict-Q certificate proves that a finite observation/action buffer is a
sound Markov state for a SysML model. This module turns that proof result into
the small runtime contract consumed by training and evaluation code:

  - exact certified buffer lengths;
  - exact policy input layout;
  - exact program/AST shield semantics;
  - model and certificate hashes that make stale artifacts rejectable.

The contract intentionally does not certify learned weights. It certifies the
architecture interface that learned controllers must use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


from clarity.runtime.env import SysMLEnv
from clarity.runtime.oracle import extract_interface
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt

from .certificate import (
    SOLVER_BACKED_MDP_THEOREM,
    check_certificate,
    load_certificate,
    model_hash,
)
from .feedforward_architecture import (
    check_continuous_feedforward_architecture,
    check_feedforward_architecture,
    derive_continuous_feedforward_architecture,
    derive_feedforward_architecture,
)


SCHEMA_VERSION = 1
SPEC_KIND = "certified_reduced_mdp_architecture_v1"
DISCRETE_SHIELD_TYPE = "program_ast_spec_shield"
CONTINUOUS_SHIELD_TYPE = "program_ast_continuous_interval_shield"
SHIELD_TYPE = DISCRETE_SHIELD_TYPE
RUNTIME_SHIELD_CLASS = "SpecShield"


def canonical_json_bytes(obj: Any) -> bytes:
    """Return stable JSON bytes for hashing proof-adjacent artifacts."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spec_hash(spec: dict[str, Any]) -> str:
    """Hash a spec excluding its own self-hash field."""
    clone = json.loads(json.dumps(spec))
    clone.pop("self_sha256", None)
    return sha256_bytes(canonical_json_bytes(clone))


def _json_action_map(action_map: dict[int, dict[str, bool]]) -> dict[str, dict[str, bool]]:
    return {
        str(int(action_id)): {
            str(name): bool(value)
            for name, value in sorted(actuators.items())
        }
        for action_id, actuators in sorted(action_map.items())
    }


def _bool_output_space(out_params: list[tuple[str, str]]) -> bool:
    bool_types = {"bool", "boolean"}
    return all((type_name or "").lower() in bool_types for _name, type_name in out_params)


def _inspect_env(model_path: str, *, dt: float, max_steps: int) -> dict[str, Any]:
    probe = SysMLEnv(model_path, dt=dt, max_steps=max_steps, phase=1, rng_seed=0)
    try:
        out_params = [(str(name), str(type_name)) for name, type_name in probe._out_params]
        is_discrete = _bool_output_space(out_params)
        base = {
            "action_kind": "discrete" if is_discrete else "continuous",
            "observation_keys": list(probe._obs_keys),
            "base_observation_dim": int(probe.obs_dim),
            "out_params": out_params,
            "obs_scale": float(probe._obs_scale),
        }
        if is_discrete:
            base.update({
                "action_count": int(probe.n_actions),
                "action_map": _json_action_map(probe._action_map),
            })
            return base
    finally:
        probe.close()

    from continuous_env import SysMLContinuousEnv

    env = SysMLContinuousEnv(model_path, dt=dt, max_steps=max_steps, phase=1, rng_seed=0)
    try:
        base.update({
            "act_dim": int(env.act_dim),
            "out_names": list(env._out_names),
        })
        return base
    finally:
        env.close()


def _layout_from_buffer(
    *,
    observation_keys: list[str],
    action_kind: str,
    action_width: int,
    b_obs: int,
    b_act: int,
    action_scale: float = 1.0,
) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    index = 0
    for obs_index, name in enumerate(observation_keys):
        layout.append({
            "index": index,
            "source": "current_observation",
            "lag": 0,
            "name": name,
            "base_observation_index": obs_index,
            "encoding": "scalar_normalized_by_sysml_env",
        })
        index += 1
    if action_kind == "continuous":
        for lag in range(1, b_act + 1):
            for action_index in range(action_width):
                layout.append({
                    "index": index,
                    "source": "past_executed_action",
                    "lag": lag,
                    "action_index": action_index,
                    "encoding": "scalar_divided_by_action_scale",
                    "action_scale": float(action_scale),
                    "name": f"executed_action[-{lag}][{action_index}]",
                })
                index += 1
        for lag in range(1, b_obs + 1):
            for obs_index, name in enumerate(observation_keys):
                layout.append({
                    "index": index,
                    "source": "past_observation",
                    "lag": lag,
                    "name": name,
                    "base_observation_index": obs_index,
                    "encoding": "scalar_normalized_by_sysml_env",
                })
                index += 1
    else:
        for lag in range(1, b_obs + 1):
            for obs_index, name in enumerate(observation_keys):
                layout.append({
                    "index": index,
                    "source": "past_observation",
                    "lag": lag,
                    "name": name,
                    "base_observation_index": obs_index,
                    "encoding": "scalar_normalized_by_sysml_env",
                })
                index += 1
        for lag in range(1, b_act + 1):
            for action_id in range(action_width):
                layout.append({
                    "index": index,
                    "source": "past_executed_action",
                    "lag": lag,
                    "action_id": action_id,
                    "encoding": "one_hot",
                    "name": f"executed_action[-{lag}]={action_id}",
                })
                index += 1
    return layout


def build_reduced_mdp_spec(
    model_path: str | Path,
    *,
    certificate: dict[str, Any] | None = None,
    certificate_path: str | Path | None = None,
    dt: float,
    max_steps: int = 5000,
) -> dict[str, Any]:
    """Build a training/eval architecture contract from a checked certificate."""
    dt = validate_dt(dt)
    model_abs = os.path.abspath(model_path)
    if certificate is None:
        if certificate_path is None:
            raise ValueError("certificate or certificate_path is required")
        certificate = load_certificate(certificate_path)

    cert_errors = check_certificate(certificate)
    if cert_errors:
        raise ValueError(
            "certificate cannot support reduced-MDP spec:\n"
            + "\n".join(f"  - {err}" for err in cert_errors)
        )

    cert_model = certificate.get("model", {})
    cert_model_path = os.path.abspath(cert_model.get("path", model_abs))
    if cert_model_path != model_abs:
        raise ValueError(
            "certificate model path does not match requested model: "
            f"{cert_model_path} != {model_abs}"
        )
    observed_model_hash = model_hash(model_abs)
    if observed_model_hash != cert_model.get("sha256"):
        raise ValueError("model sha256 does not match certificate")

    settings = certificate.get("settings", {})
    certificate_dt = validate_dt(settings.get("dt"))
    if certificate_dt != dt:
        raise ValueError(
            f"certificate dt does not match requested dt: {certificate_dt} != {dt}"
        )
    buffer = certificate.get("buffer") or {}
    b_obs = int(buffer["b_obs"])
    b_act = int(buffer["b_act"])

    env_info = _inspect_env(model_abs, dt=dt, max_steps=max_steps)
    action_kind = env_info["action_kind"]
    if action_kind == "discrete":
        iface = extract_interface(model_abs)
        shield = iface["spec_shield"]
        action_width = int(env_info["action_count"])
        if list(shield.action_map.keys()) != [int(k) for k in sorted(env_info["action_map"], key=int)]:
            raise ValueError("shield and environment action spaces disagree")
        shield_info = {
            "type": DISCRETE_SHIELD_TYPE,
            "runtime_class": "SpecShield",
            "source": "SysML #NeuralRequirement AST via clarity.runtime.shield.SpecShield",
            "obs_names": list(iface["obs_names"]),
            "in_params": list(shield.in_params),
            "out_params": list(shield.out_params),
            "dead_actions": sorted(int(a) for a in shield.dead_actions),
            "action_map": _json_action_map(shield.action_map),
            "history_semantics": "history records executed actions after exact shield override",
        }
        action_space = {
            "type": "discrete_boolean_output_bitvector",
            "n_actions": action_width,
            "action_names": list(iface["action_names"]),
            "out_params": [
                {"name": name, "type": type_name}
                for name, type_name in env_info["out_params"]
            ],
            "action_map": env_info["action_map"],
        }
        layout_version = "current_obs_then_past_obs_then_past_executed_actions_v1"
        action_scale = 1.0
    else:
        from continuous_shield import ContinuousShield

        shield = ContinuousShield(model_abs)
        action_width = int(env_info["act_dim"])
        if action_width != 1:
            raise ValueError(
                "continuous reduced-MDP specs currently support exactly one real output; "
                f"observed act_dim={action_width}"
            )
        action_scale = max(abs(float(shield.act_low)), abs(float(shield.act_high)))
        if not math.isfinite(action_scale) or action_scale <= 0:
            raise ValueError(
                "continuous #NeuralRequirement must provide a finite nonzero action range"
            )
        shield_info = {
            "type": CONTINUOUS_SHIELD_TYPE,
            "runtime_class": "ContinuousShield",
            "source": (
                "SysML #NeuralRequirement AST via "
                "rl/continuous_shield.py::ContinuousShield"
            ),
            "obs_names": list(env_info["observation_keys"]),
            "in_params": list(shield.in_params),
            "out_params": list(shield.out_params),
            "dead_actions": [],
            "action_map": {},
            "static_interval": {
                "low": float(shield.act_low),
                "high": float(shield.act_high),
            },
            "history_semantics": "history records executed actions after exact shield projection",
        }
        action_space = {
            "type": "continuous_single_real_output",
            "act_dim": action_width,
            "action_names": list(env_info["out_names"]),
            "out_params": [
                {"name": name, "type": type_name}
                for name, type_name in env_info["out_params"]
            ],
            "action_history_encoding": "scalar_divided_by_action_scale",
            "action_scale": float(action_scale),
        }
        layout_version = "current_obs_then_past_executed_actions_then_past_obs_v1"

    layout = _layout_from_buffer(
        observation_keys=env_info["observation_keys"],
        action_kind=action_kind,
        action_width=action_width,
        b_obs=b_obs,
        b_act=b_act,
        action_scale=action_scale,
    )
    input_dim = len(layout)
    feedforward_architecture = None
    if action_kind == "discrete":
        feedforward_architecture = derive_feedforward_architecture(
            shield,
            input_dim=input_dim,
            action_count=action_width,
        )
    else:
        feedforward_architecture = derive_continuous_feedforward_architecture(
            shield,
            input_dim=input_dim,
            action_dim=action_width,
        )

    one_step = (
        certificate.get("solver_advisory", {})
        .get("one_step_transition_closure", {})
    )
    cert_artifact = {
        "path": None if certificate_path is None else os.path.abspath(certificate_path),
        "file_sha256": None if certificate_path is None else file_sha256(certificate_path),
        "canonical_sha256": sha256_bytes(canonical_json_bytes(certificate)),
        "claim": certificate.get("claim", {}),
        "buffer": {"b_obs": b_obs, "b_act": b_act},
    }
    spec = {
        "schema_version": SCHEMA_VERSION,
        "kind": SPEC_KIND,
        "model": {
            "path": model_abs,
            "sha256": observed_model_hash,
            "dt": dt,
            "max_steps_for_shape_probe": int(max_steps),
        },
        "certificate": cert_artifact,
        "certified_buffer": {
            "b_obs": b_obs,
            "b_act": b_act,
            "semantics": (
                "current observation plus certified finite history of past "
                "observations and past shielded/executed actions"
            ),
        },
        "policy_input": {
            "layout_version": layout_version,
            "input_dim": input_dim,
            "base_observation_dim": env_info["base_observation_dim"],
            "observation_keys": env_info["observation_keys"],
            "layout": layout,
        },
        "action_space": action_space,
        "shield": shield_info,
        "feedforward_architecture": feedforward_architecture,
        "proof_summary": {
            "claim_level": certificate.get("claim", {}).get("level"),
            "solver_backed_mdp_theorem": certificate.get("claim", {}).get(
                "solver_backed_mdp_theorem"
            ),
            "one_step_transition_closure": {
                "status": one_step.get("status"),
                "solver": one_step.get("solver"),
                "logic": one_step.get("logic"),
                "max_polynomial_degree": one_step.get("max_polynomial_degree"),
                "q_size": one_step.get("q_size"),
                "actions_size": one_step.get("actions_size"),
                "visible_terms_checked": one_step.get("visible_terms_checked"),
            },
            "mdp_obligation_statuses": {
                key: value.get("status")
                for key, value in certificate.get("mdp_obligations", {}).items()
                if isinstance(value, dict)
            },
        },
        "training_contract": {
            "policy_class": "memoryless",
            "recurrent_state_allowed": False,
            "trainer_must_use_policy_input_layout_exactly": True,
            "trainer_must_use_exact_program_shield": action_kind == "discrete",
            "trainer_must_use_exact_runtime_shield": True,
            "trainer_must_record_executed_actions_not_raw_proposals": True,
            "safe_checkpoint_selection": (
                "select only zero-safety-violation checkpoints, then highest "
                "accuracy, then lowest override rate"
            ),
        },
    }
    spec["self_sha256"] = spec_hash(spec)
    return spec


def write_reduced_mdp_spec(spec: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(spec))
    spec["self_sha256"] = spec_hash(spec)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reduced_mdp_spec(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_reduced_mdp_spec(
    spec: dict[str, Any],
    *,
    check_files: bool = True,
    check_certificate_artifact: bool = True,
) -> list[str]:
    errors: list[str] = []
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={spec.get('schema_version')}")
    if spec.get("kind") != SPEC_KIND:
        errors.append(f"unsupported kind={spec.get('kind')}")
    if spec.get("self_sha256") != spec_hash(spec):
        errors.append("self_sha256 does not match spec contents")

    model_info = spec.get("model", {})
    model_path = model_info.get("path")
    try:
        model_dt = validate_dt(model_info.get("dt"))
    except (TypeError, ValueError):
        model_dt = None
        errors.append(f"invalid reduced-MDP spec dt={model_info.get('dt')}")
    if check_files and model_path:
        if not os.path.exists(model_path):
            errors.append(f"model path does not exist: {model_path}")
        elif model_hash(model_path) != model_info.get("sha256"):
            errors.append("model sha256 does not match reduced-MDP spec")

    cert = spec.get("certificate", {})
    claim = cert.get("claim", {})
    if claim.get("level") != SOLVER_BACKED_MDP_THEOREM:
        errors.append(f"unsupported certificate claim level={claim.get('level')}")
    if claim.get("solver_backed_mdp_theorem") != "discharged":
        errors.append("certificate claim does not discharge solver-backed MDP theorem")

    buffer = spec.get("certified_buffer", {})
    cert_buffer = cert.get("buffer", {})
    for key in ("b_obs", "b_act"):
        if not isinstance(buffer.get(key), int) or buffer.get(key) < 0:
            errors.append(f"invalid certified_buffer.{key}={buffer.get(key)}")
        if cert_buffer and buffer.get(key) != cert_buffer.get(key):
            errors.append(f"certificate/spec buffer mismatch for {key}")

    policy_input = spec.get("policy_input", {})
    layout = policy_input.get("layout", [])
    if not isinstance(layout, list):
        errors.append("policy_input.layout is not a list")
        layout = []
    input_dim = policy_input.get("input_dim")
    if input_dim != len(layout):
        errors.append("policy_input.input_dim does not match layout length")
    base_dim = policy_input.get("base_observation_dim")
    obs_keys = policy_input.get("observation_keys", [])
    if base_dim != len(obs_keys):
        errors.append("base_observation_dim does not match observation_keys")

    action_space = spec.get("action_space", {})
    action_type = action_space.get("type")
    if action_type == "discrete_boolean_output_bitvector":
        action_width = action_space.get("n_actions")
        if not isinstance(action_width, int) or action_width <= 0:
            errors.append(f"invalid action count={action_width}")
            action_width = 0
    elif action_type == "continuous_single_real_output":
        action_width = action_space.get("act_dim")
        if action_width != 1:
            errors.append(f"invalid continuous act_dim={action_width}")
            action_width = 0
        if action_space.get("action_history_encoding") != "scalar_divided_by_action_scale":
            errors.append("continuous action history encoding is not recorded")
        scale = action_space.get("action_scale")
        if not isinstance(scale, (int, float)) or scale <= 0:
            errors.append(f"invalid continuous action_scale={scale}")
    else:
        errors.append(f"unsupported action_space.type={action_type}")
        action_width = 0

    if isinstance(input_dim, int) and isinstance(base_dim, int):
        expected_dim = base_dim * (1 + int(buffer.get("b_obs", 0))) + (
            int(buffer.get("b_act", 0)) * action_width
        )
        if input_dim != expected_dim:
            errors.append(
                "policy_input.input_dim does not match certified buffer dimensions: "
                f"expected {expected_dim}, observed {input_dim}"
            )

    for expected_index, item in enumerate(layout):
        if not isinstance(item, dict):
            errors.append(f"layout item {expected_index} is not an object")
            continue
        if item.get("index") != expected_index:
            errors.append(f"layout index mismatch at {expected_index}")
        if item.get("source") not in {
            "current_observation",
            "past_observation",
            "past_executed_action",
        }:
            errors.append(f"unknown layout source at {expected_index}: {item.get('source')}")

    shield = spec.get("shield", {})
    if action_type == "discrete_boolean_output_bitvector":
        if shield.get("type") != DISCRETE_SHIELD_TYPE:
            errors.append(f"unsupported discrete shield type={shield.get('type')}")
        if shield.get("runtime_class") != "SpecShield":
            errors.append(f"unsupported discrete shield runtime_class={shield.get('runtime_class')}")
        if shield.get("history_semantics") != "history records executed actions after exact shield override":
            errors.append("discrete shield history semantics are not exact executed-action semantics")
        if shield.get("action_map") != action_space.get("action_map"):
            errors.append("shield action map does not match action space action map")
        architecture = spec.get("feedforward_architecture")
        if not isinstance(architecture, dict):
            errors.append("discrete spec lacks a derived feedforward architecture")
        elif isinstance(input_dim, int) and isinstance(action_width, int):
            errors.extend(check_feedforward_architecture(
                architecture,
                input_dim=input_dim,
                action_count=action_width,
            ))
            if check_files and model_path and os.path.exists(model_path):
                try:
                    observed_interface = extract_interface(model_path)
                    observed_architecture = derive_feedforward_architecture(
                        observed_interface["spec_shield"],
                        input_dim=input_dim,
                        action_count=action_width,
                    )
                    if architecture != observed_architecture:
                        errors.append(
                            "feedforward architecture does not match the SysML requirement"
                        )
                except Exception as exc:
                    errors.append(
                        "could not recompute feedforward architecture from SysML: "
                        f"{exc}"
                    )
    elif action_type == "continuous_single_real_output":
        if shield.get("type") != CONTINUOUS_SHIELD_TYPE:
            errors.append(f"unsupported continuous shield type={shield.get('type')}")
        if shield.get("runtime_class") != "ContinuousShield":
            errors.append(
                f"unsupported continuous shield runtime_class={shield.get('runtime_class')}"
            )
        if shield.get("history_semantics") != "history records executed actions after exact shield projection":
            errors.append("continuous shield history semantics are not exact projection semantics")
        interval = shield.get("static_interval", {})
        if not isinstance(interval.get("low"), (int, float)):
            errors.append("continuous shield static interval lacks numeric low")
        if not isinstance(interval.get("high"), (int, float)):
            errors.append("continuous shield static interval lacks numeric high")
        architecture = spec.get("feedforward_architecture")
        if not isinstance(architecture, dict):
            errors.append("continuous spec lacks a derived feedforward architecture")
        elif isinstance(input_dim, int) and isinstance(action_width, int):
            errors.extend(check_continuous_feedforward_architecture(
                architecture,
                input_dim=input_dim,
                action_dim=action_width,
            ))
            if check_files and model_path and os.path.exists(model_path):
                try:
                    from continuous_shield import ContinuousShield

                    observed_shield = ContinuousShield(model_path)
                    observed_architecture = derive_continuous_feedforward_architecture(
                        observed_shield,
                        input_dim=input_dim,
                        action_dim=action_width,
                    )
                    if architecture != observed_architecture:
                        errors.append(
                            "continuous feedforward architecture does not match "
                            "the SysML requirement"
                        )
                except Exception as exc:
                    errors.append(
                        "could not recompute continuous feedforward architecture "
                        f"from SysML: {exc}"
                    )

    contract = spec.get("training_contract", {})
    if contract.get("policy_class") != "memoryless":
        errors.append("training contract is not memoryless")
    if contract.get("recurrent_state_allowed") is not False:
        errors.append("training contract allows recurrent state")
    if action_type == "discrete_boolean_output_bitvector" and (
        contract.get("trainer_must_use_exact_program_shield") is not True
    ):
        errors.append("training contract does not require exact program shield")
    if contract.get("trainer_must_use_exact_runtime_shield") is not True:
        errors.append("training contract does not require exact runtime shield")
    if contract.get("trainer_must_record_executed_actions_not_raw_proposals") is not True:
        errors.append("training contract does not require executed-action history")

    proof = spec.get("proof_summary", {})
    if proof.get("claim_level") != SOLVER_BACKED_MDP_THEOREM:
        errors.append(f"unsupported proof summary claim={proof.get('claim_level')}")
    if proof.get("solver_backed_mdp_theorem") != "discharged":
        errors.append("proof summary does not discharge solver-backed theorem")
    one_step = proof.get("one_step_transition_closure", {})
    if one_step.get("status") != "discharged":
        errors.append("proof summary one-step transition closure is not discharged")
    if one_step.get("solver") != "z3":
        errors.append("proof summary solver is not z3")

    if check_files and check_certificate_artifact:
        cert_path = cert.get("path")
        if cert_path:
            if not os.path.exists(cert_path):
                errors.append(f"certificate path does not exist: {cert_path}")
            else:
                if file_sha256(cert_path) != cert.get("file_sha256"):
                    errors.append("certificate file sha256 does not match spec")
                loaded = load_certificate(cert_path)
                if sha256_bytes(canonical_json_bytes(loaded)) != cert.get("canonical_sha256"):
                    errors.append("certificate canonical sha256 does not match spec")
                for err in check_certificate(loaded):
                    errors.append(f"embedded certificate check failed: {err}")
                loaded_buffer = loaded.get("buffer") or {}
                try:
                    loaded_dt = validate_dt(loaded.get("settings", {}).get("dt"))
                except (TypeError, ValueError):
                    loaded_dt = None
                    errors.append("loaded certificate has an invalid dt")
                if model_dt is not None and loaded_dt is not None and loaded_dt != model_dt:
                    errors.append("loaded certificate dt does not match spec")
                if loaded_buffer.get("b_obs") != buffer.get("b_obs"):
                    errors.append("loaded certificate b_obs does not match spec")
                if loaded_buffer.get("b_act") != buffer.get("b_act"):
                    errors.append("loaded certificate b_act does not match spec")
        else:
            errors.append("spec has no certificate artifact path")

    return errors


def reduced_buffer_from_spec(spec: dict[str, Any]) -> tuple[int, int]:
    """Return ``(b_obs, b_act)`` after a structural spec check."""
    errors = check_reduced_mdp_spec(spec)
    if errors:
        raise ValueError(
            "reduced-MDP spec check failed:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
    buffer = spec["certified_buffer"]
    return int(buffer["b_obs"]), int(buffer["b_act"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--certificate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    spec = build_reduced_mdp_spec(
        args.model,
        certificate_path=args.certificate,
        dt=args.dt,
        max_steps=args.max_steps,
    )
    write_reduced_mdp_spec(spec, args.out)
    print(
        f"WROTE {args.out} "
        f"buffer={spec['certified_buffer']} "
        f"input_dim={spec['policy_input']['input_dim']} "
        f"self_sha256={spec['self_sha256']}"
    )
    if args.check:
        loaded = load_reduced_mdp_spec(args.out)
        errors = check_reduced_mdp_spec(loaded)
        if errors:
            print("FAIL")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("PASS reduced-MDP spec check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
