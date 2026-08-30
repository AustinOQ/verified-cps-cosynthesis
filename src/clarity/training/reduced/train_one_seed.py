#!/usr/bin/env python3
"""Single-seed handmade reduced-MDP trainer.

This is the NumPy implementation of the reduced feedforward training stage for
discrete models:

  1. certify strict-Q buffer reconstructibility and write/check a certificate;
  2. build a memoryless handmade MLP over the certified buffer;
  3. train with handmade oracle CE + handmade PPO/backprop on CPU;
  4. select only safe checkpoints, then max accuracy, then min override.

The runtime shield is always the exact program/AST ``SpecShield``. The DNN
co-architecture shield is not used here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


from clarity.certification.certificate import (
    build_certificate_for_path,
    check_certificate,
    load_certificate,
    write_certificate,
)
from clarity.certification.reduced_mdp_spec import (
    build_reduced_mdp_spec,
    check_reduced_mdp_spec,
    load_reduced_mdp_spec,
    spec_hash,
    write_reduced_mdp_spec,
)
from clarity.certification.reconstruct import get_strict_model, reconstruct
from clarity.runtime.oracle import extract_interface
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt
from clarity.training.recurrent.io import save_policy
from clarity.training.recurrent.optim import Adam
from clarity.training.recurrent.ppo_update import ppo_update
from clarity.training.recurrent.train_oracle import train_oracle

from clarity.training.reduced.buffered_env import BufferedDiscreteEnv
from clarity.training.reduced.collection import (
    CollectionSettings,
    DiscreteRuntime,
    build_episode_collector,
    generate_oracle_data_with_backend,
    make_episode_jobs,
)
from clarity.training.reduced.composite import ProgramShieldComposite
from clarity.training.reduced.oracle_data import (
    balance_classes,
)
from clarity.training.reduced.policy import MLPActorCritic


DEFAULT_CONFIG = {
    "oracle_samples": 2000,
    "oracle_epochs": 100,
    "oracle_batch_size": 512,
    "oracle_lr": 1e-3,
    "ppo_episodes": 2000,
    "episodes_per_update": 100,
    "eval_interval": 100,
    "ppo_lr": 1e-4,
    "hidden_dim": 64,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "value_coeff": 0.5,
    "entropy_start": -1.0,
    "entropy_end": -5.0,
    "max_grad_norm": 0.5,
    "n_ppo_epochs": 4,
    "minibatch_size": 25,
    "test_episodes": 200,
    "eval_episodes": 100,
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def certify_minimal_buffer(model_path: str, *, dt: float, max_obs: int,
                           max_act: int, horizon: int) -> tuple[int, int]:
    model = get_strict_model(model_path, dt=dt)
    blocking = [
        diag for diag in model.get("diagnostics", [])
        if getattr(diag, "severity", "") in {"warning", "error"}
    ]
    if blocking:
        pretty = "\n".join(f"  - {diag.pretty()}" for diag in blocking)
        raise RuntimeError(f"strict extractor emitted blocking diagnostics:\n{pretty}")
    target = set(model.get("R", set())) or set(model["STATE"])
    for b_obs in range(max_obs + 1):
        for b_act in range(max_act + 1):
            missing, _ = reconstruct(
                model, b_obs, b_act, horizon=horizon, target=target)
            if not missing:
                return b_obs, b_act
    raise RuntimeError(
        "no strict-Q reconstructible buffer within "
        f"max_obs={max_obs}, max_act={max_act}, horizon={horizon}"
    )


def train_one_seed(model_path: str, seed: int, out_dir: str | Path,
                   *, dt: float, max_steps: int = 5000,
                   max_obs: int = 2, max_act: int = 6, horizon: int = 14,
                   hidden_dim: int | None = None,
                   ensure_class_coverage: int = 200,
                   balance_oracle_classes: bool = False,
                   reduced_mdp_spec_path: str | Path | None = None,
                   reduced_mdp_spec_out: str | Path | None = None,
                   collection_backend: str = "serial",
                   collection_workers: int = 1,
                   collection_start_method: str = "spawn",
                   config: dict | None = None) -> dict:
    dt = validate_dt(dt)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    t0 = time.time()
    architecture_source = "local_certificate"
    reduced_spec = None
    reduced_spec_path_for_result = None
    if reduced_mdp_spec_path is not None:
        reduced_spec_path = Path(reduced_mdp_spec_path)
        reduced_spec = load_reduced_mdp_spec(reduced_spec_path)
        spec_errors = check_reduced_mdp_spec(reduced_spec)
        if spec_errors:
            raise RuntimeError(
                "reduced-MDP spec checker failed:\n"
                + "\n".join(f"  - {err}" for err in spec_errors)
            )
        spec_model = os.path.abspath(reduced_spec["model"]["path"])
        if spec_model != os.path.abspath(model_path):
            raise RuntimeError(
                "reduced-MDP spec is for a different model: "
                f"{spec_model} != {os.path.abspath(model_path)}"
            )
        spec_dt = validate_dt(reduced_spec["model"].get("dt"))
        if spec_dt != dt:
            raise RuntimeError(
                f"training dt does not match reduced-MDP spec: {dt} != {spec_dt}"
            )
        b_obs = int(reduced_spec["certified_buffer"]["b_obs"])
        b_act = int(reduced_spec["certified_buffer"]["b_act"])
        cert_path = Path(reduced_spec["certificate"]["path"])
        certificate = load_certificate(cert_path)
        architecture_source = "provided_reduced_mdp_spec"
        reduced_spec_path_for_result = reduced_spec_path
    else:
        b_obs, b_act = certify_minimal_buffer(
            model_path, dt=dt, max_obs=max_obs, max_act=max_act, horizon=horizon)
        certificate = build_certificate_for_path(
            model_path,
            b_obs=b_obs,
            b_act=b_act,
            max_obs=max_obs,
            max_act=max_act,
            horizon=horizon,
            dt=dt,
            include_solver_artifacts=True,
        )
        cert_path = out_path / "certificate.json"
        write_certificate(certificate, cert_path)
        cert_errors = check_certificate(certificate)
        if cert_errors:
            raise RuntimeError(
                "solver-backed certificate checker failed:\n"
                + "\n".join(f"  - {err}" for err in cert_errors)
            )
        reduced_spec = build_reduced_mdp_spec(
            model_path,
            certificate=certificate,
            certificate_path=cert_path,
            dt=dt,
            max_steps=max_steps,
        )
        reduced_spec_path_for_result = Path(
            reduced_mdp_spec_out or (out_path / "reduced_mdp_spec.json")
        )
        write_reduced_mdp_spec(reduced_spec, reduced_spec_path_for_result)

    feedforward_architecture = reduced_spec.get("feedforward_architecture")
    if not isinstance(feedforward_architecture, dict):
        raise RuntimeError(
            "reduced-MDP spec does not contain a derived feedforward architecture"
        )
    derived_hidden_dim = int(feedforward_architecture["hidden_dim"])
    if hidden_dim is None:
        hidden_dim = derived_hidden_dim
        hidden_dim_source = feedforward_architecture["method"]
    else:
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        hidden_dim_source = "explicit_override"
    cfg["hidden_dim"] = hidden_dim

    # Keep this equal to the certified action buffer for all current models.
    # If a future model certifies b_act=0, using zero is the more literal MDP
    # representation and avoids giving the learner extra non-certified inputs.
    policy_n_act = b_act
    policy_n_obs = b_obs

    probe = BufferedDiscreteEnv(
        model_path,
        dt=dt,
        max_steps=max_steps,
        phase=1,
        rng_seed=seed,
        n_obs=policy_n_obs,
        n_act=policy_n_act,
    )
    obs_dim = probe.obs_dim
    n_actions = probe.n_actions
    buffer_spec = probe.buffer_spec
    probe.close()
    if reduced_spec is not None:
        spec_input = reduced_spec["policy_input"]
        spec_actions = reduced_spec["action_space"]
        if obs_dim != int(spec_input["input_dim"]):
            raise RuntimeError(
                "runtime buffered env input dimension does not match "
                f"reduced-MDP spec: env={obs_dim}, spec={spec_input['input_dim']}"
            )
        if buffer_spec["base_obs_dim"] != int(spec_input["base_observation_dim"]):
            raise RuntimeError(
                "runtime base observation dimension does not match reduced-MDP spec"
            )
        if n_actions != int(spec_actions["n_actions"]):
            raise RuntimeError(
                "runtime action count does not match reduced-MDP spec: "
                f"env={n_actions}, spec={spec_actions['n_actions']}"
            )

    iface = extract_interface(model_path)
    policy = MLPActorCritic(obs_dim, n_actions, hidden_dim, seed=seed)
    composite = ProgramShieldComposite(
        policy, iface["spec_shield"], iface["obs_names"])
    n_params = int(sum(arr.size for arr in policy.parameters().values()))
    if hidden_dim == derived_hidden_dim and (
        n_params != int(feedforward_architecture["parameter_count"])
    ):
        raise RuntimeError(
            "derived feedforward parameter count does not match runtime policy"
        )

    print("=" * 72)
    print("HANDMADE REDUCED-MDP TRAINING")
    print("=" * 72)
    print(f"model: {model_path}")
    print(f"seed: {seed}")
    print(f"architecture_source: {architecture_source}")
    print(f"hidden_dim_source: {hidden_dim_source}")
    print(f"certificate: {cert_path}")
    print(f"reduced_mdp_spec: {reduced_spec_path_for_result}")
    print(f"certified buffer: b_obs={b_obs}, b_act={b_act}")
    print(f"shield_type: {composite.shield_type}")
    print(f"policy: handmade numpy MLP, hidden_dim={hidden_dim}, params={n_params}")
    print(f"obs_dim={obs_dim}, n_actions={n_actions}, buffer={buffer_spec}")
    collection_settings = CollectionSettings(
        backend=collection_backend,
        workers=collection_workers,
        start_method=collection_start_method,
    ).resolved()
    runtime = DiscreteRuntime(
        model_path=os.path.abspath(model_path),
        dt=dt,
        max_steps=max_steps,
        phase=2,
        n_obs=policy_n_obs,
        n_act=policy_n_act,
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=hidden_dim,
    )
    print(
        "collection: "
        f"backend={collection_settings.backend}, "
        f"workers={collection_settings.workers}, "
        f"start_method={collection_settings.start_method}"
    )

    obs_data, act_data, class_counts, resets = generate_oracle_data_with_backend(
        replace(runtime, phase=1),
        collection_settings,
        iface,
        cfg["oracle_samples"],
        min_class_count=ensure_class_coverage,
        seed_base=seed,
    )
    print(
        "oracle data: "
        f"{len(act_data)} samples, class_counts={class_counts}, resets={resets}"
    )
    if balance_oracle_classes:
        obs_data, act_data = balance_classes(obs_data, act_data, seed=seed)
        unique, counts = np.unique(act_data, return_counts=True)
        print(
            "oracle data balanced: "
            f"{len(act_data)} samples, "
            f"class_counts={dict(zip(unique.tolist(), counts.tolist()))}"
        )

    oracle_history = train_oracle(
        policy,
        obs_data,
        act_data,
        batch_size=cfg["oracle_batch_size"],
        n_epochs=cfg["oracle_epochs"],
        lr=cfg["oracle_lr"],
        seed=seed,
        verbose=False,
    )
    print(
        "oracle final: "
        f"loss={oracle_history[-1]['loss']:.4f}, "
        f"acc={oracle_history[-1]['acc']:.1%}, "
        f"elapsed={oracle_history[-1]['elapsed_seconds']:.1f}s"
    )

    ppo_opt = Adam(policy.parameters(), lr=cfg["ppo_lr"])
    rng_update = np.random.default_rng(seed + 2)
    buffer = []
    checkpoints = []
    train_eval_history = []
    t_train = time.time()
    with build_episode_collector(
        collection_settings, runtime, composite
    ) as collector:
        ep_i = 0
        while ep_i < cfg["ppo_episodes"]:
            next_update = (
                (ep_i // cfg["episodes_per_update"] + 1)
                * cfg["episodes_per_update"]
            )
            next_eval = (
                (ep_i // cfg["eval_interval"] + 1)
                * cfg["eval_interval"]
            )
            chunk_end = min(cfg["ppo_episodes"], next_update, next_eval)
            jobs = make_episode_jobs(
                chunk_end - ep_i,
                seed_base=seed + 2,
                start_index=ep_i,
            )
            buffer.extend(collector.collect(policy, jobs, greedy=False))
            ep_i = chunk_end

            if ep_i % cfg["episodes_per_update"] == 0:
                progress = (ep_i - cfg["episodes_per_update"]) / max(
                    cfg["ppo_episodes"] - cfg["episodes_per_update"], 1)
                lr_frac = 1.0 - progress
                ppo_opt.lr = cfg["ppo_lr"] * max(lr_frac, 0.0)
                ent = (
                    cfg["entropy_start"]
                    + progress * (cfg["entropy_end"] - cfg["entropy_start"])
                )
                metrics = ppo_update(
                    policy,
                    ppo_opt,
                    buffer,
                    obs_dim=obs_dim,
                    gamma=cfg["gamma"],
                    lam=cfg["lam"],
                    clip_eps=cfg["clip_eps"],
                    value_coeff=cfg["value_coeff"],
                    entropy_coeff=ent,
                    n_epochs=cfg["n_ppo_epochs"],
                    minibatch_size=cfg["minibatch_size"],
                    max_grad_norm=cfg["max_grad_norm"],
                    rng=rng_update,
                )
                buffer = []
                print(
                    f"ppo ep {ep_i:5d}/{cfg['ppo_episodes']} "
                    f"lr={ppo_opt.lr:.2e} ent={ent:+.2f} "
                    f"p_loss={metrics['policy_loss']:+.4f} "
                    f"v_loss={metrics['value_loss']:.4f} "
                    f"elapsed={time.time() - t_train:.0f}s"
                )

            if ep_i % cfg["eval_interval"] == 0:
                eval_seed = seed * 100000 + ep_i
                summary = collector.evaluate(
                    policy,
                    make_episode_jobs(
                        cfg["eval_episodes"], seed_base=eval_seed),
                )
                params = {k: v.copy() for k, v in policy.parameters().items()}
                checkpoint = {
                    "episode": ep_i,
                    "safety_violation_rate": summary.safety_violation_rate,
                    "success_rate": summary.success_rate,
                    "override_rate": summary.pooled_override_rate,
                    "mean_episode_steps": summary.mean_episode_steps,
                    "params": params,
                }
                checkpoints.append(checkpoint)
                train_eval_history.append({
                    k: v for k, v in checkpoint.items() if k != "params"
                })
                print(
                    f"  eval ep {ep_i}: "
                    f"safety={summary.safety_violation_rate:.3f} "
                    f"acc={summary.success_rate:.3f} "
                    f"override={summary.pooled_override_rate:.4f} "
                    f"steps={summary.mean_episode_steps:.1f}"
                )

        safe = [c for c in checkpoints if c["safety_violation_rate"] == 0.0]
        if not safe:
            raise RuntimeError("no safe checkpoint; refusing to select a model")
        best = sorted(
            safe,
            key=lambda c: (-c["success_rate"], c["override_rate"], c["episode"]),
        )[0]
        for key, val in best["params"].items():
            policy.parameters()[key][...] = val
        save_policy(policy, out_path / "best.npz")

        eval_summary = collector.evaluate(
            policy,
            make_episode_jobs(
                cfg["eval_episodes"], seed_base=10_000 + seed),
        )
        test_summary = collector.evaluate(
            policy,
            make_episode_jobs(
                cfg["test_episodes"], seed_base=20_000 + seed),
        )
    train_seconds = time.time() - t0
    result = {
        "model_path": model_path,
        "dt": dt,
        "seed": seed,
        "training_stack": "handmade_numpy_reduced_mdp",
        "device": "cpu",
        "collection": collection_settings.to_dict(),
        "shield_type": composite.shield_type,
        "policy_type": "memoryless_mlp",
        "architecture_source": architecture_source,
        "hidden_dim_source": hidden_dim_source,
        "feedforward_architecture": feedforward_architecture,
        "hidden_dim": hidden_dim,
        "parameter_count": n_params,
        "certificate_path": str(cert_path),
        "certificate_claim": certificate.get("claim", {}),
        "reduced_mdp_spec_path": (
            None if reduced_spec_path_for_result is None
            else str(reduced_spec_path_for_result)
        ),
        "reduced_mdp_spec_sha256": (
            None if reduced_spec is None else spec_hash(reduced_spec)
        ),
        "certified_buffer": {"b_obs": b_obs, "b_act": b_act},
        "policy_buffer": {"n_obs": policy_n_obs, "n_act": policy_n_act},
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "train_seconds": train_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "oracle": {
            "samples": int(len(act_data)),
            "class_counts": class_counts,
            "resets": resets,
            "final_loss": oracle_history[-1]["loss"],
            "final_acc": oracle_history[-1]["acc"],
        },
        "selected_checkpoint": {
            k: v for k, v in best.items() if k != "params"
        },
        "eval": asdict(eval_summary),
        "test": asdict(test_summary),
        "training_eval_history": train_eval_history,
        "config": cfg,
        "ensure_class_coverage": ensure_class_coverage,
        "balance_oracle_classes": balance_oracle_classes,
    }
    (out_path / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\nSELECTED")
    print(
        f"  ep={best['episode']} safety={best['safety_violation_rate']:.3f} "
        f"acc={best['success_rate']:.3f} override={best['override_rate']:.4f}"
    )
    print("HELD-OUT TEST")
    print(
        f"  safety={test_summary.safety_violation_rate:.3f} "
        f"acc={test_summary.success_rate:.3f} "
        f"override={test_summary.pooled_override_rate:.4f} "
        f"mean_steps={test_summary.mean_episode_steps:.1f}"
    )
    print(f"WROTE {out_path / 'summary.json'}")
    return result


def _config_overrides(args) -> dict:
    keys = [
        "oracle_samples",
        "oracle_epochs",
        "oracle_batch_size",
        "ppo_episodes",
        "episodes_per_update",
        "eval_interval",
        "eval_episodes",
        "test_episodes",
        "n_ppo_epochs",
        "minibatch_size",
    ]
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="override the SysML-derived hidden size",
    )
    ap.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--max-obs", type=int, default=2)
    ap.add_argument("--max-act", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument(
        "--reduced-mdp-spec",
        default=None,
        help="Use a reduced-MDP architecture spec generated earlier in this run.",
    )
    ap.add_argument(
        "--reduced-mdp-spec-out",
        default=None,
        help="Where to write the spec when this run certifies locally.",
    )
    ap.add_argument("--ensure-class-coverage", type=int, default=200)
    ap.add_argument("--balance-oracle-classes", action="store_true")
    ap.add_argument("--oracle-samples", type=int)
    ap.add_argument("--oracle-epochs", type=int)
    ap.add_argument("--oracle-batch-size", type=int)
    ap.add_argument("--ppo-episodes", type=int)
    ap.add_argument("--episodes-per-update", type=int)
    ap.add_argument("--eval-interval", type=int)
    ap.add_argument("--eval-episodes", type=int)
    ap.add_argument("--test-episodes", type=int)
    ap.add_argument("--n-ppo-epochs", type=int)
    ap.add_argument("--minibatch-size", type=int)
    ap.add_argument(
        "--collection-backend",
        choices=("serial", "process", "auto"),
        default="serial",
        help="episode collection backend",
    )
    ap.add_argument(
        "--collection-workers",
        type=int,
        default=1,
        help="independent simulator processes used for episode collection",
    )
    ap.add_argument(
        "--collection-start-method",
        default="spawn",
        help="multiprocessing start method; spawn is safe with device runtimes",
    )
    args = ap.parse_args()
    train_one_seed(
        args.model,
        args.seed,
        args.out_dir,
        dt=args.dt,
        max_steps=args.max_steps,
        max_obs=args.max_obs,
        max_act=args.max_act,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        ensure_class_coverage=args.ensure_class_coverage,
        balance_oracle_classes=args.balance_oracle_classes,
        reduced_mdp_spec_path=args.reduced_mdp_spec,
        reduced_mdp_spec_out=args.reduced_mdp_spec_out,
        collection_backend=args.collection_backend,
        collection_workers=args.collection_workers,
        collection_start_method=args.collection_start_method,
        config=_config_overrides(args),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
