#!/usr/bin/env python3
"""Validation battery for the handmade reduced-MDP stack."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
REPO = ARCH.parent
for path in (REPO, ARCH, REPO / "rl", REPO / "sysml-models"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from certification.certificate import (
    build_certificate_for_path,
    check_certificate,
    write_certificate,
)
from certification.feedforward_architecture import derive_feedforward_architecture
from certification.reduced_mdp_spec import (
    build_reduced_mdp_spec,
    check_reduced_mdp_spec,
    load_reduced_mdp_spec,
    spec_hash,
    write_reduced_mdp_spec,
)
from handmade.io import load_policy, save_policy
from handmade.losses import ppo_update_grads
from mlp_buffer import BufferedContinuousEnv
from oracle import extract_interface
from runtime_settings import DEFAULT_DT

from reduced_handmade.buffered_env import BufferedDiscreteEnv
from reduced_handmade.collection import (
    CollectionSettings,
    DiscreteRuntime,
    build_episode_collector,
    generate_oracle_data_with_backend,
    make_episode_jobs,
)
from reduced_handmade.composite import ProgramShieldComposite
from reduced_handmade.policy import MLPActorCritic
from reduced_handmade.train_one_seed import train_one_seed
from sysml_inputs import discover_sysml


_DISCOVERED = {
    item.key: item.path
    for item in discover_sysml([], models_root=REPO / "sysml-models")
}
MODELS = {
    "cruise": _DISCOVERED["cruise-control"],
    "cruise-continuous": _DISCOVERED["cruise-control-continuous"],
    "mixing": _DISCOVERED["tank-filling-system"],
    "thermostat": _DISCOVERED["thermostat"],
}
TEST_DT = DEFAULT_DT
NONDEFAULT_TEST_DT = DEFAULT_DT / 2.0


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _loss_from_policy(policy: MLPActorCritic, obs: np.ndarray,
                      dlogits: np.ndarray, dvalues: np.ndarray) -> float:
    logits, values, _ = policy.forward_sequence(
        obs, policy.initial_hidden(obs.shape[0]))
    return float((logits * dlogits).sum() + (values * dvalues).sum())


def check_policy_gradients(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    policy = MLPActorCritic(obs_dim=5, n_actions=3, hidden_dim=4, seed=seed)
    obs = rng.normal(size=(2, 3, 5)).astype(np.float32)
    dlogits = rng.normal(size=(2, 3, 3)).astype(np.float32)
    dvalues = rng.normal(size=(2, 3)).astype(np.float32)
    _, _, cache = policy.forward_sequence(obs, policy.initial_hidden(2))
    grads = policy.backward_sequence(dlogits, dvalues, cache)

    eps = 1e-3
    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    for name, param in policy.parameters().items():
        flat = param.reshape(-1)
        if flat.size <= 8:
            indices = np.arange(flat.size)
        else:
            indices = rng.choice(flat.size, size=8, replace=False)
        grad_flat = grads[name].reshape(-1)
        for idx in indices:
            old = float(flat[idx])
            flat[idx] = old + eps
            plus = _loss_from_policy(policy, obs, dlogits, dvalues)
            flat[idx] = old - eps
            minus = _loss_from_policy(policy, obs, dlogits, dvalues)
            flat[idx] = old
            numeric = (plus - minus) / (2.0 * eps)
            analytic = float(grad_flat[idx])
            abs_err = abs(numeric - analytic)
            rel_err = abs_err / max(1.0, abs(numeric), abs(analytic))
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            checked += 1
    _assert(max_abs < 2e-2 and max_rel < 2e-2,
            f"policy gradient check failed: max_abs={max_abs}, max_rel={max_rel}")
    return {"checked_entries": checked, "max_abs_error": max_abs, "max_rel_error": max_rel}


def check_ppo_direct_gradients(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=0.3, size=(2, 3, 4)).astype(np.float64)
    values = rng.normal(scale=0.2, size=(2, 3)).astype(np.float64)
    actions = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64)
    old_logp = np.zeros((2, 3), dtype=np.float64)
    advantages = rng.normal(scale=0.5, size=(2, 3)).astype(np.float64)
    returns = rng.normal(scale=0.5, size=(2, 3)).astype(np.float64)
    mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.float64)

    loss, _, dlogits, dvalues = ppo_update_grads(
        logits, values, actions, old_logp, advantages, returns, mask,
        clip_eps=0.2, value_coeff=0.5, entropy_coeff=-0.25, bc_coeff=0.0)

    eps = 1e-5
    max_abs = 0.0
    max_rel = 0.0
    checked = 0

    def f(cur_logits, cur_values):
        return ppo_update_grads(
            cur_logits, cur_values, actions, old_logp, advantages, returns,
            mask, clip_eps=0.2, value_coeff=0.5, entropy_coeff=-0.25,
            bc_coeff=0.0)[0]

    for arr, grad, label in ((logits, dlogits, "logits"), (values, dvalues, "values")):
        flat = arr.reshape(-1)
        grad_flat = grad.reshape(-1)
        picks = rng.choice(flat.size, size=min(12, flat.size), replace=False)
        for idx in picks:
            old = float(flat[idx])
            flat[idx] = old + eps
            plus = f(logits, values)
            flat[idx] = old - eps
            minus = f(logits, values)
            flat[idx] = old
            numeric = (plus - minus) / (2.0 * eps)
            analytic = float(grad_flat[idx])
            abs_err = abs(numeric - analytic)
            rel_err = abs_err / max(1.0, abs(numeric), abs(analytic))
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            checked += 1
    _assert(math.isfinite(loss), "PPO loss is not finite")
    _assert(max_abs < 1e-4 and max_rel < 1e-4,
            f"PPO gradient check failed: max_abs={max_abs}, max_rel={max_rel}")
    return {"checked_entries": checked, "max_abs_error": max_abs, "max_rel_error": max_rel}


def check_checkpoint_roundtrip(seed: int = 2) -> dict:
    p1 = MLPActorCritic(obs_dim=7, n_actions=4, hidden_dim=5, seed=seed)
    saved = {k: v.copy() for k, v in p1.parameters().items()}
    p2 = MLPActorCritic(obs_dim=7, n_actions=4, hidden_dim=5, seed=seed + 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.npz"
        save_policy(p1, path)
        load_policy(p2, path)
    for key, val in saved.items():
        _assert(np.array_equal(val, p2.parameters()[key]),
                f"checkpoint roundtrip mismatch for {key}")
    return {"parameters_checked": len(saved)}


def check_composite_uses_program_shield(seed: int = 3) -> dict:
    class DummyPolicy:
        def initial_hidden(self, batch_size):
            return np.zeros((batch_size, 0), dtype=np.float32)

        def step(self, obs, hidden):
            del hidden
            logits = np.array([[-10.0, 10.0, -10.0]], dtype=np.float32)
            value = np.array([0.0], dtype=np.float32)
            return logits, value, self.initial_hidden(obs.shape[0])

    calls = []

    def exact_program_shield(proposed, obs_dict):
        calls.append((proposed, dict(obs_dict)))
        return 0 if proposed == 1 else proposed

    comp = ProgramShieldComposite(DummyPolicy(), exact_program_shield, ["x"])
    action, logp, value, hidden, overridden, probs, timing = comp.act(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 0), dtype=np.float32),
        {"x": 7.0},
        greedy=True,
        rng=np.random.default_rng(seed),
    )
    _assert(comp.shield_type == "program_ast_spec_shield", "wrong shield_type marker")
    _assert(calls == [(1, {"x": 7.0})], "exact shield callable was not invoked as expected")
    _assert(action == 0 and overridden, "shield override did not affect final action")
    _assert(np.isfinite(logp) and np.isfinite(value), "composite returned non-finite outputs")
    return {"calls": len(calls), "final_action": action, "overridden": overridden}


def check_buffered_env(model_key: str = "mixing") -> dict:
    model_path = str(MODELS[model_key])
    env = BufferedDiscreteEnv(
        model_path, dt=TEST_DT, max_steps=20, phase=1,
        rng_seed=11, n_obs=2, n_act=1)
    try:
        base = env._base_obs_dim
        n_actions = env.n_actions
        obs0 = env.reset()
        _assert(obs0.shape == (base + 2 * base + n_actions,),
                f"unexpected reset obs shape {obs0.shape}")
        _assert(np.allclose(obs0[base:base + 2 * base], 0.0),
                "past observations are not zero-filled at reset")
        _assert(np.allclose(obs0[-n_actions:], 0.0),
                "past actions are not zero-filled at reset")
        obs1, _, _, _ = env.step(0)
        _assert(obs1.shape == obs0.shape, "step changed augmented obs shape")
        expected = np.zeros(n_actions, dtype=np.float32)
        expected[0] = 1.0
        _assert(np.allclose(obs1[-n_actions:], expected),
                "last-action onehot not placed at augmented observation tail")
    finally:
        env.close()
    return {"model": model_key, "base_obs_dim": base, "n_actions": n_actions,
            "augmented_obs_dim": int(obs0.shape[0])}


def check_derived_feedforward_architectures() -> dict:
    cases = {
        "thermostat": {"n_obs": 1, "n_act": 2, "hidden": 2, "params": 47},
        "cruise": {"n_obs": 1, "n_act": 2, "hidden": 3, "params": 77},
        "mixing": {"n_obs": 2, "n_act": 1, "hidden": 4, "params": 245},
    }
    details = {}
    for model_key, expected in cases.items():
        model_path = str(MODELS[model_key])
        env = BufferedDiscreteEnv(
            model_path,
            dt=TEST_DT,
            max_steps=20,
            phase=1,
            rng_seed=41,
            n_obs=expected["n_obs"],
            n_act=expected["n_act"],
        )
        try:
            input_dim = env.obs_dim
            action_count = env.n_actions
        finally:
            env.close()
        iface = extract_interface(model_path)
        architecture = derive_feedforward_architecture(
            iface["spec_shield"],
            input_dim=input_dim,
            action_count=action_count,
        )
        _assert(architecture["hidden_dim"] == expected["hidden"],
                f"unexpected derived hidden size for {model_key}")
        _assert(architecture["parameter_count"] == expected["params"],
                f"unexpected derived parameter count for {model_key}")
        policy = MLPActorCritic(
            input_dim, action_count, architecture["hidden_dim"], seed=43)
        actual_params = sum(value.size for value in policy.parameters().values())
        _assert(actual_params == architecture["parameter_count"],
                f"runtime parameter count differs for {model_key}")
        details[model_key] = architecture
    return details


def check_parallel_collection_equivalence(model_key: str = "cruise") -> dict:
    model_path = str(MODELS[model_key].resolve())
    n_obs = 1
    n_act = 1
    probe = BufferedDiscreteEnv(
        model_path, dt=TEST_DT, max_steps=20, phase=2,
        rng_seed=19, n_obs=n_obs, n_act=n_act)
    try:
        obs_dim = probe.obs_dim
        n_actions = probe.n_actions
    finally:
        probe.close()

    iface = extract_interface(model_path)
    policy = MLPActorCritic(obs_dim, n_actions, hidden_dim=2, seed=23)
    composite = ProgramShieldComposite(
        policy, iface["spec_shield"], iface["obs_names"])
    runtime = DiscreteRuntime(
        model_path=model_path,
        dt=TEST_DT,
        max_steps=20,
        phase=2,
        n_obs=n_obs,
        n_act=n_act,
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=2,
    )
    serial_settings = CollectionSettings(backend="serial", workers=1)
    process_settings = CollectionSettings(backend="process", workers=2)
    jobs = make_episode_jobs(4, seed_base=29)

    with build_episode_collector(
        serial_settings, runtime, composite
    ) as serial_collector, build_episode_collector(
        process_settings, runtime, composite
    ) as process_collector:
        serial_episodes = serial_collector.collect(policy, jobs, greedy=False)
        process_episodes = process_collector.collect(policy, jobs, greedy=False)
        _assert(len(serial_episodes) == len(process_episodes),
                "serial and process collectors returned different episode counts")
        array_fields = (
            "obs", "actions", "rewards", "values", "log_probs",
            "dones", "overrides",
        )
        for index, (serial_ep, process_ep) in enumerate(zip(
            serial_episodes, process_episodes
        )):
            for field in array_fields:
                _assert(
                    np.array_equal(
                        np.asarray(getattr(serial_ep, field)),
                        np.asarray(getattr(process_ep, field)),
                    ),
                    f"episode {index} differs in {field}",
                )
            _assert(serial_ep.outcome == process_ep.outcome,
                    f"episode {index} differs in outcome")
            _assert(serial_ep.violations == process_ep.violations,
                    f"episode {index} differs in violations")
            _assert(serial_ep.safety_viol == process_ep.safety_viol,
                    f"episode {index} differs in safety result")

        eval_jobs = make_episode_jobs(4, seed_base=31)
        serial_summary = serial_collector.evaluate(policy, eval_jobs)
        process_summary = process_collector.evaluate(policy, eval_jobs)
        deterministic_summary_fields = (
            "n_episodes", "n_steps", "success_rate", "violation_rate",
            "truncated_rate", "pooled_override_rate", "mean_episode_steps",
            "mean_reward", "safety_violation_rate",
        )
        for field in deterministic_summary_fields:
            _assert(
                np.isclose(
                    getattr(serial_summary, field),
                    getattr(process_summary, field),
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"serial and process evaluation differ in {field}",
            )

    oracle_runtime = replace(runtime, phase=1)
    serial_oracle = generate_oracle_data_with_backend(
        oracle_runtime,
        serial_settings,
        iface,
        32,
        seed_base=37,
        max_resets=16,
    )
    process_oracle = generate_oracle_data_with_backend(
        oracle_runtime,
        process_settings,
        iface,
        32,
        seed_base=37,
        max_resets=16,
    )
    _assert(np.array_equal(serial_oracle[0], process_oracle[0]),
            "serial and process oracle observations differ")
    _assert(np.array_equal(serial_oracle[1], process_oracle[1]),
            "serial and process oracle actions differ")
    _assert(serial_oracle[2:] == process_oracle[2:],
            "serial and process oracle metadata differ")

    return {
        "model": model_key,
        "episodes": len(jobs),
        "oracle_samples": int(len(serial_oracle[1])),
        "workers": process_settings.workers,
    }


def check_certificate_gate(model_key: str = "mixing") -> dict:
    cert = build_certificate_for_path(
        str(MODELS[model_key]), max_obs=2, max_act=6, horizon=14, dt=TEST_DT)
    errors = check_certificate(cert)
    _assert(not errors, f"fresh certificate failed checker: {errors}")

    bad = copy.deepcopy(cert)
    bad["claim"]["level"] = "profile_only_overclaim"
    _assert(check_certificate(bad), "mutated claim level was not rejected")

    bad = copy.deepcopy(cert)
    bad["claim"]["solver_backed_mdp_theorem"] = "not_discharged"
    _assert(check_certificate(bad), "mutated solver theorem was not rejected")
    return {
        "model": model_key,
        "buffer": cert.get("buffer"),
        "claim_level": cert.get("claim", {}).get("level"),
    }


def check_reduced_mdp_spec_contract(out_dir: Path, model_key: str = "mixing") -> dict:
    spec_dir = out_dir / "reduced_mdp_spec_contract"
    spec_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(MODELS[model_key])
    cert = build_certificate_for_path(
        model_path, max_obs=2, max_act=6, horizon=14, dt=TEST_DT)
    cert_path = spec_dir / "certificate.json"
    write_certificate(cert, cert_path)
    spec = build_reduced_mdp_spec(
        model_path,
        certificate=cert,
        certificate_path=cert_path,
        dt=TEST_DT,
        max_steps=20,
    )
    spec_path = spec_dir / "reduced_mdp_spec.json"
    write_reduced_mdp_spec(spec, spec_path)
    loaded = load_reduced_mdp_spec(spec_path)
    errors = check_reduced_mdp_spec(loaded)
    _assert(not errors, f"fresh reduced-MDP spec failed checker: {errors}")

    env = BufferedDiscreteEnv(
        model_path, dt=TEST_DT, max_steps=20, phase=1, rng_seed=13,
        n_obs=loaded["certified_buffer"]["b_obs"],
        n_act=loaded["certified_buffer"]["b_act"],
    )
    try:
        _assert(env.obs_dim == loaded["policy_input"]["input_dim"],
                "spec input_dim does not match buffered env")
        _assert(env.n_actions == loaded["action_space"]["n_actions"],
                "spec action count does not match buffered env")
    finally:
        env.close()

    bad = copy.deepcopy(loaded)
    bad["certified_buffer"]["b_act"] += 1
    bad["self_sha256"] = spec_hash(bad)
    _assert(check_reduced_mdp_spec(bad),
            "mutated buffer length was not rejected")

    bad = copy.deepcopy(loaded)
    bad["policy_input"]["input_dim"] += 1
    bad["self_sha256"] = spec_hash(bad)
    _assert(check_reduced_mdp_spec(bad),
            "mutated policy input dimension was not rejected")

    bad = copy.deepcopy(loaded)
    bad["shield"]["type"] = "dnn_coarchitecture_shield"
    bad["self_sha256"] = spec_hash(bad)
    _assert(check_reduced_mdp_spec(bad),
            "mutated shield type was not rejected")

    bad = copy.deepcopy(loaded)
    bad["feedforward_architecture"]["hidden_dim"] += 1
    bad["self_sha256"] = spec_hash(bad)
    _assert(check_reduced_mdp_spec(bad),
            "mutated feedforward hidden size was not rejected")

    return {
        "model": model_key,
        "spec": str(spec_path),
        "buffer": loaded["certified_buffer"],
        "input_dim": loaded["policy_input"]["input_dim"],
        "n_actions": loaded["action_space"]["n_actions"],
        "feedforward_architecture": loaded["feedforward_architecture"],
        "self_sha256": loaded["self_sha256"],
    }


def check_continuous_reduced_mdp_spec_contract(out_dir: Path) -> dict:
    spec_dir = out_dir / "continuous_reduced_mdp_spec_contract"
    spec_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(MODELS["cruise-continuous"])
    cert = build_certificate_for_path(
        model_path, max_obs=2, max_act=6, horizon=14, dt=TEST_DT)
    cert_path = spec_dir / "certificate.json"
    write_certificate(cert, cert_path)
    spec = build_reduced_mdp_spec(
        model_path,
        certificate=cert,
        certificate_path=cert_path,
        dt=TEST_DT,
        max_steps=20,
    )
    spec_path = spec_dir / "reduced_mdp_spec.json"
    write_reduced_mdp_spec(spec, spec_path)
    loaded = load_reduced_mdp_spec(spec_path)
    errors = check_reduced_mdp_spec(loaded)
    _assert(not errors, f"fresh continuous reduced-MDP spec failed checker: {errors}")
    _assert(loaded["action_space"]["type"] == "continuous_single_real_output",
            "continuous spec did not record continuous action space")

    env = BufferedContinuousEnv(
        model_path, dt=TEST_DT, max_steps=20, phase=1, rng_seed=17,
        n_obs=loaded["certified_buffer"]["b_obs"],
        n_act=loaded["certified_buffer"]["b_act"],
    )
    try:
        _assert(env.obs_dim == loaded["policy_input"]["input_dim"],
                "continuous spec input_dim does not match buffered env")
        _assert(env.act_dim == loaded["action_space"]["act_dim"],
                "continuous spec action dimension does not match buffered env")
    finally:
        env.close()

    bad = copy.deepcopy(loaded)
    bad["shield"]["runtime_class"] = "SpecShield"
    bad["self_sha256"] = spec_hash(bad)
    _assert(check_reduced_mdp_spec(bad),
            "mutated continuous shield runtime class was not rejected")

    return {
        "model": "cruise-continuous",
        "spec": str(spec_path),
        "buffer": loaded["certified_buffer"],
        "input_dim": loaded["policy_input"]["input_dim"],
        "act_dim": loaded["action_space"]["act_dim"],
        "shield": loaded["shield"]["runtime_class"],
    }


def check_dt_propagation_and_mismatches(out_dir: Path) -> dict:
    case_dir = out_dir / "dt_propagation"
    case_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(MODELS["thermostat"])
    cert = build_certificate_for_path(
        model_path,
        max_obs=2,
        max_act=6,
        horizon=14,
        dt=NONDEFAULT_TEST_DT,
    )
    _assert(
        cert["settings"]["dt"] == NONDEFAULT_TEST_DT,
        "certificate did not record the supplied nondefault dt",
    )
    cert_path = case_dir / "certificate.json"
    write_certificate(cert, cert_path)
    spec = build_reduced_mdp_spec(
        model_path,
        certificate=cert,
        certificate_path=cert_path,
        dt=NONDEFAULT_TEST_DT,
        max_steps=20,
    )
    spec_path = case_dir / "reduced_mdp_spec.json"
    write_reduced_mdp_spec(spec, spec_path)
    _assert(
        spec["model"]["dt"] == NONDEFAULT_TEST_DT,
        "reduced-MDP spec did not record the certificate dt",
    )
    _assert(
        not check_reduced_mdp_spec(load_reduced_mdp_spec(spec_path)),
        "nondefault-dt reduced-MDP spec failed its checker",
    )

    env = BufferedDiscreteEnv(
        model_path,
        dt=NONDEFAULT_TEST_DT,
        max_steps=20,
        phase=1,
        rng_seed=53,
        n_obs=spec["certified_buffer"]["b_obs"],
        n_act=spec["certified_buffer"]["b_act"],
    )
    try:
        _assert(
            env._twin._dt == NONDEFAULT_TEST_DT,
            "simulator did not receive the supplied nondefault dt",
        )
    finally:
        env.close()

    try:
        build_reduced_mdp_spec(
            model_path,
            certificate=cert,
            certificate_path=cert_path,
            dt=TEST_DT,
            max_steps=20,
        )
    except ValueError as exc:
        _assert("certificate dt does not match" in str(exc),
                "certificate/spec dt mismatch raised the wrong error")
    else:
        raise AssertionError("certificate/spec dt mismatch was accepted")

    bad_spec = copy.deepcopy(spec)
    bad_spec["model"]["dt"] = TEST_DT
    bad_spec["self_sha256"] = spec_hash(bad_spec)
    mismatch_errors = check_reduced_mdp_spec(bad_spec)
    _assert(
        "loaded certificate dt does not match spec" in mismatch_errors,
        "reduced-MDP checker accepted a certificate/spec dt mismatch",
    )

    try:
        train_one_seed(
            model_path,
            seed=0,
            out_dir=case_dir / "trainer_mismatch",
            dt=TEST_DT,
            reduced_mdp_spec_path=spec_path,
        )
    except RuntimeError as exc:
        _assert("training dt does not match" in str(exc),
                "trainer dt mismatch raised the wrong error")
    else:
        raise AssertionError("trainer accepted a reduced-MDP spec with another dt")

    bad_cert = copy.deepcopy(cert)
    bad_cert["settings"]["dt"] = 0.0
    _assert(check_certificate(bad_cert), "certificate checker accepted an invalid dt")
    return {
        "dt": NONDEFAULT_TEST_DT,
        "certificate": str(cert_path),
        "spec": str(spec_path),
    }


def check_smoke_training(out_dir: Path) -> dict:
    run_dir = out_dir / "smoke_mixing_h4"
    result = train_one_seed(
        str(MODELS["mixing"]),
        seed=0,
        out_dir=run_dir,
        dt=TEST_DT,
        ensure_class_coverage=4,
        config={
            "oracle_samples": 64,
            "oracle_epochs": 2,
            "ppo_episodes": 8,
            "episodes_per_update": 4,
            "eval_interval": 4,
            "eval_episodes": 4,
            "test_episodes": 4,
            "n_ppo_epochs": 1,
            "minibatch_size": 4,
        },
    )
    _assert(result["shield_type"] == "program_ast_spec_shield",
            "smoke run did not use exact program shield")
    _assert(result["certificate_claim"]["level"] == "strict_q_solver_backed_mdp_v1",
            "smoke run did not use solver-backed certificate")
    _assert(result["architecture_source"] == "local_certificate",
            "local smoke run did not record local certificate architecture source")
    _assert(result["reduced_mdp_spec_path"],
            "local smoke run did not write a reduced-MDP spec")
    _assert(result["test"]["safety_violation_rate"] == 0.0,
            "smoke run had safety violations")

    spec_run_dir = out_dir / "smoke_mixing_h4_from_spec"
    spec_result = train_one_seed(
        str(MODELS["mixing"]),
        seed=1,
        out_dir=spec_run_dir,
        dt=TEST_DT,
        ensure_class_coverage=4,
        reduced_mdp_spec_path=result["reduced_mdp_spec_path"],
        collection_backend="process",
        collection_workers=2,
        config={
            "oracle_samples": 64,
            "oracle_epochs": 2,
            "ppo_episodes": 8,
            "episodes_per_update": 4,
            "eval_interval": 4,
            "eval_episodes": 4,
            "test_episodes": 4,
            "n_ppo_epochs": 1,
            "minibatch_size": 4,
        },
    )
    _assert(spec_result["architecture_source"] == "provided_reduced_mdp_spec",
            "spec smoke run did not consume the provided architecture spec")
    _assert(spec_result["certified_buffer"] == result["certified_buffer"],
            "spec smoke run changed the certified buffer")
    _assert(spec_result["obs_dim"] == result["obs_dim"],
            "spec smoke run changed the policy input dimension")
    _assert(spec_result["shield_type"] == "program_ast_spec_shield",
            "spec smoke run did not use exact program shield")
    _assert(spec_result["collection"]["backend"] == "process",
            "spec smoke run did not use process episode collection")
    _assert(spec_result["hidden_dim_source"] == "sysml_comparison_output_width_v1",
            "spec smoke run did not use the derived feedforward architecture")
    _assert(spec_result["test"]["safety_violation_rate"] == 0.0,
            "spec smoke run had safety violations")
    return {
        "summary": str(run_dir / "summary.json"),
        "spec_summary": str(spec_run_dir / "summary.json"),
        "spec": result["reduced_mdp_spec_path"],
        "params": result["parameter_count"],
        "test_accuracy": result["test"]["success_rate"],
        "test_override": result["test"]["pooled_override_rate"],
        "peak_rss_mb": result["peak_rss_mb"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (
        ARCH / "results" / f"handmade_reduced_validation_{time.strftime('%Y%m%d-%H%M%S')}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = [
        ("policy_gradients", lambda: check_policy_gradients()),
        ("ppo_direct_gradients", lambda: check_ppo_direct_gradients()),
        ("checkpoint_roundtrip", lambda: check_checkpoint_roundtrip()),
        ("program_shield_composite", lambda: check_composite_uses_program_shield()),
        ("buffered_env", lambda: check_buffered_env()),
        ("derived_feedforward_architectures",
         lambda: check_derived_feedforward_architectures()),
        ("parallel_collection_equivalence",
         lambda: check_parallel_collection_equivalence()),
        ("certificate_gate", lambda: check_certificate_gate()),
        ("reduced_mdp_spec_contract", lambda: check_reduced_mdp_spec_contract(out_dir)),
        ("continuous_reduced_mdp_spec_contract",
         lambda: check_continuous_reduced_mdp_spec_contract(out_dir)),
        ("dt_propagation_and_mismatches",
         lambda: check_dt_propagation_and_mismatches(out_dir)),
    ]
    if not args.skip_smoke:
        checks.append(("end_to_end_smoke_training", lambda: check_smoke_training(out_dir)))

    results = []
    status = 0
    for name, fn in checks:
        print(f"RUN {name}")
        start = time.time()
        try:
            detail = fn()
            elapsed = time.time() - start
            results.append({"name": name, "status": "PASS", "seconds": elapsed,
                            "detail": detail})
            print(f"PASS {name} ({elapsed:.3f}s)")
        except Exception as exc:  # noqa: BLE001 - validation should report all context.
            elapsed = time.time() - start
            status = 1
            results.append({"name": name, "status": "FAIL", "seconds": elapsed,
                            "error": repr(exc)})
            print(f"FAIL {name}: {exc}")

    report = {
        "status": "PASS" if status == 0 else "FAIL",
        "out_dir": str(out_dir),
        "checks": results,
    }
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"WROTE {report_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
