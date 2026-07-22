#!/usr/bin/env python3
"""Stage-wise RSS profile for handmade reduced-MDP training."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
REPO = ARCH.parent
for path in (REPO, ARCH, REPO / "rl", REPO / "sysml-models"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from certification.certificate import build_certificate_for_path, check_certificate
from certification.reduced_mdp_spec import build_reduced_mdp_spec
from handmade.optim import Adam
from handmade.ppo_update import ppo_update
from handmade.train_oracle import train_oracle
from oracle import extract_interface

from reduced_handmade.buffered_env import BufferedDiscreteEnv
from reduced_handmade.composite import ProgramShieldComposite
from reduced_handmade.episode import collect_episode
from reduced_handmade.oracle_data import generate_oracle_data
from reduced_handmade.policy import MLPActorCritic
from reduced_handmade.train_one_seed import certify_minimal_buffer
from sysml_inputs import inspect_sysml


def _ru_maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _proc_status_mb() -> dict[str, float | None]:
    out = {"vmrss_mb": None, "vmhwm_mb": None}
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            out["vmrss_mb"] = float(line.split()[1]) / 1024.0
        elif line.startswith("VmHWM:"):
            out["vmhwm_mb"] = float(line.split()[1]) / 1024.0
    return out


def _mark(stages: list[dict], stage: str, start: float,
          detail: dict | None = None) -> None:
    gc.collect()
    proc = _proc_status_mb()
    row = {
        "stage": stage,
        "seconds": time.time() - start,
        "ru_maxrss_mb": _ru_maxrss_mb(),
        **proc,
    }
    if detail:
        row["detail"] = detail
    stages.append(row)
    print(
        f"{stage:28s} current={row['vmrss_mb']} "
        f"hwm={row['vmhwm_mb']} ru_max={row['ru_maxrss_mb']:.3f}"
    )


def profile_one(model_path: str, out_dir: Path, *,
                seed: int = 0, dt: float = 0.1, max_steps: int = 5000,
                oracle_samples: int = 256, ensure_class_coverage: int = 16,
                oracle_epochs: int = 1, ppo_episodes: int = 8) -> dict:
    model = inspect_sysml(model_path)
    if model.action_kind != "discrete":
        raise ValueError(f"memory profiler requires Boolean #Neural outputs: {model.path}")
    model_path = str(model.path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    stages: list[dict] = []
    start = time.time()
    _mark(stages, "process_start", start)

    b_obs, b_act = certify_minimal_buffer(
        model_path, dt=dt, max_obs=2, max_act=6, horizon=14)
    cert = build_certificate_for_path(
        model_path, b_obs=b_obs, b_act=b_act, max_obs=2, max_act=6,
        horizon=14, dt=dt)
    errors = check_certificate(cert)
    if errors:
        raise RuntimeError(errors)
    reduced_spec = build_reduced_mdp_spec(
        model_path,
        certificate=cert,
        dt=dt,
        max_steps=max_steps,
    )
    hidden_dim = int(reduced_spec["feedforward_architecture"]["hidden_dim"])
    _mark(stages, "after_certificate", start,
          {"b_obs": b_obs, "b_act": b_act, "claim": cert.get("claim", {}).get("level")})

    probe = BufferedDiscreteEnv(
        model_path, dt=dt, max_steps=max_steps, phase=1, rng_seed=seed,
        n_obs=b_obs, n_act=b_act)
    obs_dim = probe.obs_dim
    n_actions = probe.n_actions
    probe.close()
    _mark(stages, "after_probe_env", start,
          {"obs_dim": obs_dim, "n_actions": n_actions})

    iface = extract_interface(model_path, dt=dt)
    _mark(stages, "after_extract_interface", start,
          {"obs_names": iface["obs_names"], "shield_type": "program_ast_spec_shield"})

    policy = MLPActorCritic(obs_dim, n_actions, hidden_dim, seed=seed)
    composite = ProgramShieldComposite(
        policy, iface["spec_shield"], iface["obs_names"])
    n_params = int(sum(v.size for v in policy.parameters().values()))
    _mark(stages, "after_policy_init", start,
          {"hidden_dim": hidden_dim, "parameter_count": n_params,
           "shield_type": composite.shield_type})

    oracle_env = BufferedDiscreteEnv(
        model_path, dt=dt, max_steps=max_steps, phase=1, rng_seed=seed,
        n_obs=b_obs, n_act=b_act)
    try:
        obs_data, act_data, class_counts, resets = generate_oracle_data(
            iface, oracle_env, oracle_samples,
            min_class_count=ensure_class_coverage, max_steps=max_steps)
    finally:
        oracle_env.close()
    _mark(stages, "after_oracle_data", start,
          {"samples": int(len(act_data)), "class_counts": class_counts,
           "resets": resets})

    history = train_oracle(
        policy, obs_data, act_data, batch_size=512, n_epochs=oracle_epochs,
        lr=1e-3, seed=seed, verbose=False)
    _mark(stages, "after_oracle_ce", start,
          {"epochs": oracle_epochs, "final_acc": history[-1]["acc"]})

    if ppo_episodes > 0:
        train_env = BufferedDiscreteEnv(
            model_path, dt=dt, max_steps=max_steps, phase=2,
            rng_seed=seed + 2, n_obs=b_obs, n_act=b_act)
        episodes = []
        try:
            for _ in range(ppo_episodes):
                episodes.append(collect_episode(
                    train_env, composite, rng=rng, greedy=False))
        finally:
            train_env.close()
        _mark(stages, "after_collect_ppo_buffer", start,
              {"episodes": ppo_episodes,
               "steps": int(sum(len(ep.actions) for ep in episodes))})
        opt = Adam(policy.parameters(), lr=1e-4)
        metrics = ppo_update(
            policy, opt, episodes, obs_dim=obs_dim,
            gamma=0.99, lam=0.95, clip_eps=0.2, value_coeff=0.5,
            entropy_coeff=-1.0, n_epochs=1,
            minibatch_size=max(1, min(4, len(episodes))),
            max_grad_norm=0.5, rng=rng)
        _mark(stages, "after_one_ppo_update", start, metrics)

    result = {
        "model": model.key,
        "model_name": model.name,
        "model_path": model_path,
        "model_sha256": model.sha256,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "device": "cpu",
        "shield_type": "program_ast_spec_shield",
        "certified_buffer": {"b_obs": b_obs, "b_act": b_act},
        "parameter_count": n_params,
        "stages": stages,
    }
    path = out_dir / f"{model.key}_h{hidden_dim}_memory_profile.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"WROTE {path}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+", help="SysML file path")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle-samples", type=int, default=256)
    ap.add_argument("--ensure-class-coverage", type=int, default=16)
    ap.add_argument("--oracle-epochs", type=int, default=1)
    ap.add_argument("--ppo-episodes", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (
        ARCH / "results" / f"handmade_reduced_memory_profile_{time.strftime('%Y%m%d-%H%M%S')}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_path in args.model:
        results.append(profile_one(
            model_path, out_dir, seed=args.seed,
            oracle_samples=args.oracle_samples,
            ensure_class_coverage=args.ensure_class_coverage,
            oracle_epochs=args.oracle_epochs,
            ppo_episodes=args.ppo_episodes,
        ))
    aggregate = {"out_dir": str(out_dir), "profiles": results}
    (out_dir / "memory_profiles.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"WROTE {out_dir / 'memory_profiles.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
