"""Top-level single-seed CPU training runner.

Pipeline (matches rl/train.py:run_training_mode):
  1. Build env + iface
  2. Phase 1: generate oracle data, optionally balance, train CE
  3. Phase 2: PPO with episode collection, GAE, clipped surrogate
  4. Periodic eval to track best safe checkpoint by success-then-override
  5. Final eval/test and checkpoint safety announcement

Saves per-seed metrics under <seed_dir>/.
"""

from __future__ import annotations

import json
import os
import random
import resource
import time
from dataclasses import asdict

import numpy as np

from clarity.runtime.env import SysMLEnv
from clarity.runtime.oracle import extract_interface
from clarity.sysml.runtime_settings import validate_dt

from .composite import Composite
from .episode import collect_episode, evaluate
from .io import save_policy
from .optim import Adam
from .oracle_data import generate_oracle_data, balance_classes
from .policy import RecurrentActorCritic
from .ppo_update import ppo_update
from .train_oracle import train_oracle


# Config matching rl/train.py MODE_CONFIG["full"]
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
    "bptt_chunk_size": 200,
    "bc_aux_coeff": 0.0,
}


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _is_better_candidate(result, best_success: float,
                         best_override: float) -> bool:
    """Higher success wins; exact success ties use lower override rate."""
    eps = 1e-12
    if result.success_rate > best_success + eps:
        return True
    if abs(result.success_rate - best_success) <= eps:
        return result.pooled_override_rate < best_override - eps
    return False


def _copy_params(policy):
    return {k: v.copy() for k, v in policy.parameters().items()}


def _restore_params(policy, params: dict):
    for k, v in params.items():
        policy.parameters()[k][...] = v


def train_one_seed(model_path: str, seed: int, seed_dir: str,
                   dt: float, max_steps: int = 5000,
                   ensure_class_coverage: int = 0,
                   balance_oracle_classes: bool = False,
                   eval_episodes: int = 100,
                   test_episodes: int = 200,
                   config: dict | None = None):
    dt = validate_dt(dt)
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    os.makedirs(seed_dir, exist_ok=True)

    _seed_everything(seed)

    # ---- probe env / iface ----
    probe = SysMLEnv(model_path, dt=dt, max_steps=max_steps, phase=1,
                     rng_seed=seed)
    obs_dim, n_actions = probe.obs_dim, probe.n_actions
    probe.close()
    iface = extract_interface(model_path)

    # ---- build policy + composite ----
    policy = RecurrentActorCritic(obs_dim, n_actions, cfg["hidden_dim"],
                                  seed=seed)
    composite = Composite(policy, iface["spec_shield"], iface["obs_names"])

    print(f"\n[seed {seed}] policy params: "
          f"{sum(v.size for v in policy.parameters().values()):,}")
    print(f"  obs_dim={obs_dim}  n_actions={n_actions}  "
          f"hidden={cfg['hidden_dim']}")

    # =========================================================================
    # Phase 1: Oracle CE pretraining
    # =========================================================================
    t0 = time.time()
    oracle_env = SysMLEnv(model_path, dt=dt, max_steps=max_steps, phase=1,
                          rng_seed=seed)
    print(f"\n  Phase 1: Oracle data collection "
          f"(target {cfg['oracle_samples']}"
          f"{', coverage>=' + str(ensure_class_coverage) if ensure_class_coverage > 0 else ''})")
    obs_data, act_data = generate_oracle_data(
        iface, oracle_env, cfg["oracle_samples"],
        min_class_count=ensure_class_coverage)
    oracle_env.close()
    unique, counts = np.unique(act_data, return_counts=True)
    print(f"  Oracle data (raw): {len(act_data)} samples, "
          f"class dist: {dict(zip(unique.tolist(), counts.tolist()))}")
    if balance_oracle_classes:
        obs_data, act_data = balance_classes(obs_data, act_data, seed=seed)
        unique, counts = np.unique(act_data, return_counts=True)
        print(f"  Oracle data (balanced): {len(act_data)} samples, "
              f"class dist: {dict(zip(unique.tolist(), counts.tolist()))}")

    print(f"  Training oracle CE for {cfg['oracle_epochs']} epochs ...")
    oracle_history = train_oracle(
        policy, obs_data, act_data,
        batch_size=cfg["oracle_batch_size"],
        n_epochs=cfg["oracle_epochs"],
        lr=cfg["oracle_lr"],
        seed=seed,
        verbose=False)
    last = oracle_history[-1]
    print(f"  Oracle final: loss={last['loss']:.4f} acc={last['acc']:.1%} "
          f"({last['elapsed_seconds']:.1f}s)")

    # Post-oracle eval
    rng_eval = np.random.default_rng(seed + 1)
    post_oracle = evaluate(
        env_factory=lambda: SysMLEnv(model_path, dt=dt, max_steps=max_steps,
                                     phase=2, rng_seed=seed + 1),
        composite=composite, n_episodes=20, rng=rng_eval, greedy=True)
    print(f"  Post-oracle eval (20 ep): success={post_oracle.success_rate:.1%} "
          f"override={post_oracle.pooled_override_rate:.4f}")

    # =========================================================================
    # Phase 2: PPO
    # =========================================================================
    print(f"\n  Phase 2: PPO ({cfg['ppo_episodes']} episodes)")
    total_updates = cfg["ppo_episodes"] // cfg["episodes_per_update"]
    ppo_opt = Adam(policy.parameters(), lr=cfg["ppo_lr"])
    rng_train = np.random.default_rng(seed + 2)

    # Linear entropy schedule
    entropy_start = cfg["entropy_start"]
    entropy_end = cfg["entropy_end"]

    train_env = SysMLEnv(model_path, dt=dt, max_steps=max_steps,
                         phase=2, rng_seed=seed + 2)
    best_safe_success = -1.0
    best_safe_override = float("inf")
    best_safe_episode = None
    best_safe_params = None
    best_safe_safety = None
    best_any_success = -1.0
    best_any_override = float("inf")
    best_any_episode = None
    best_any_params = None
    best_any_safety = None
    update_idx = 0
    buffer: list = []
    eval_history = []
    t_phase2 = time.time()
    try:
        for ep_i in range(1, cfg["ppo_episodes"] + 1):
            ep = collect_episode(train_env, composite, rng=rng_train,
                                  greedy=False)
            buffer.append(ep)
            if ep_i % cfg["episodes_per_update"] == 0:
                frac = min(update_idx / max(total_updates, 1), 1.0)
                ent_coeff = entropy_start + frac * (entropy_end - entropy_start)
                metrics = ppo_update(
                    policy, ppo_opt, buffer, obs_dim=obs_dim,
                    gamma=cfg["gamma"], lam=cfg["lam"],
                    clip_eps=cfg["clip_eps"],
                    value_coeff=cfg["value_coeff"],
                    entropy_coeff=ent_coeff,
                    bc_coeff=cfg["bc_aux_coeff"],
                    n_epochs=cfg["n_ppo_epochs"],
                    minibatch_size=cfg["minibatch_size"],
                    bptt_chunk_size=cfg["bptt_chunk_size"],
                    max_grad_norm=cfg["max_grad_norm"],
                    rng=rng_train)
                buffer = []
                update_idx += 1
                recent_succ = sum(1 for e in [ep] if e.outcome == "SUCCESS")  # placeholder
                # (we just print the most recent update's metrics; per-episode logging is in eval below)
                elapsed = time.time() - t_phase2
                print(f"  [PPO] ep {ep_i:5d}/{cfg['ppo_episodes']} | "
                      f"ent_c={ent_coeff:+.2f} | "
                      f"p_loss={metrics['policy_loss']:+.4f} | "
                      f"v_loss={metrics['value_loss']:.4f} | "
                      f"entropy={metrics['entropy']:.3f} | "
                      f"{elapsed:.0f}s")

            if ep_i % cfg["eval_interval"] == 0:
                eval_env_seed = seed * 100_000 + ep_i
                results = evaluate(
                    env_factory=lambda s=eval_env_seed: SysMLEnv(
                        model_path, dt=dt, max_steps=max_steps,
                        phase=2, rng_seed=s),
                    composite=composite, n_episodes=100,
                    rng=np.random.default_rng(eval_env_seed),
                    greedy=True)
                print(f"     eval ep {ep_i}: success={results.success_rate:.2f} "
                      f"override={results.pooled_override_rate:.4f} "
                      f"safety={results.safety_violation_rate:.4f}")
                is_safe = (results.safety_violation_rate == 0.0)
                eval_history.append({
                    "episode": ep_i,
                    "success_rate": results.success_rate,
                    "override_rate": results.pooled_override_rate,
                    "safety_violation_rate": results.safety_violation_rate,
                    "is_safe": is_safe,
                })
                if _is_better_candidate(
                        results, best_any_success, best_any_override):
                    best_any_success = results.success_rate
                    best_any_override = results.pooled_override_rate
                    best_any_episode = ep_i
                    best_any_safety = results.safety_violation_rate
                    best_any_params = _copy_params(policy)
                    print(f"     >> new best candidate "
                          f"(succ={best_any_success:.2f}, "
                          f"override={best_any_override:.4f}, "
                          f"safety={best_any_safety:.4f})")
                if is_safe and _is_better_candidate(
                        results, best_safe_success, best_safe_override):
                    best_safe_success = results.success_rate
                    best_safe_override = results.pooled_override_rate
                    best_safe_episode = ep_i
                    best_safe_safety = results.safety_violation_rate
                    best_safe_params = _copy_params(policy)
                    print(f"     >> new best SAFE checkpoint "
                          f"(succ={best_safe_success:.2f}, "
                          f"override={best_safe_override:.4f})")
    finally:
        train_env.close()
    train_seconds = time.time() - t0

    # Restore the best safe checkpoint. If none was ever observed during
    # periodic eval, fall back to the best candidate and mark it unsafe.
    selected_checkpoint_source = "safe_eval"
    selected_episode = best_safe_episode
    selected_success = best_safe_success
    selected_override = best_safe_override
    selected_safety = best_safe_safety
    selected_params = best_safe_params
    if selected_params is None:
        selected_checkpoint_source = "unsafe_fallback"
        selected_episode = best_any_episode
        selected_success = best_any_success
        selected_override = best_any_override
        selected_safety = best_any_safety
        selected_params = best_any_params

    if selected_params is not None:
        _restore_params(policy, selected_params)
        if selected_checkpoint_source == "safe_eval":
            print(f"\n  Selected SAFE checkpoint from eval ep "
                  f"{selected_episode}: success={selected_success:.4f} "
                  f"override={selected_override:.5f}")
        else:
            print(f"\n  WARNING: no safe periodic-eval checkpoint was found; "
                  f"using UNSAFE fallback from eval ep {selected_episode}: "
                  f"success={selected_success:.4f} "
                  f"override={selected_override:.5f} "
                  f"safety={selected_safety:.5f}")
    else:
        selected_checkpoint_source = "final_weights_no_eval"
        selected_episode = None
        selected_success = None
        selected_override = None
        selected_safety = None
        print("\n  WARNING: no periodic-eval checkpoint was recorded; "
              "saving final training weights.")

    # Persist best checkpoint to disk for inference-only RSS measurement.
    save_policy(policy, os.path.join(seed_dir, "best.npz"))

    # =========================================================================
    # Phase 3: Final eval (eval + test rollouts)
    # =========================================================================
    print(f"\n  Final eval ({eval_episodes} ep) ...")
    eval_summary = evaluate(
        env_factory=lambda: SysMLEnv(model_path, dt=dt, max_steps=max_steps,
                                     phase=2, rng_seed=10_000 + seed),
        composite=composite, n_episodes=eval_episodes,
        rng=np.random.default_rng(10_000 + seed), greedy=True)
    print(f"  Final test ({test_episodes} ep) ...")
    test_summary = evaluate(
        env_factory=lambda: SysMLEnv(model_path, dt=dt, max_steps=max_steps,
                                     phase=2, rng_seed=20_000 + seed),
        composite=composite, n_episodes=test_episodes,
        rng=np.random.default_rng(20_000 + seed), greedy=True)
    final_checkpoint_safe = (
        eval_summary.safety_violation_rate == 0.0
        and test_summary.safety_violation_rate == 0.0
    )
    print(f"  Final eval summary: success={eval_summary.success_rate:.4f} "
          f"override={eval_summary.pooled_override_rate:.5f} "
          f"safety={eval_summary.safety_violation_rate:.5f}")
    print(f"  Final test summary: success={test_summary.success_rate:.4f} "
          f"override={test_summary.pooled_override_rate:.5f} "
          f"safety={test_summary.safety_violation_rate:.5f}")
    if final_checkpoint_safe:
        print("  Final chosen checkpoint is SAFE on final eval/test.")
    else:
        print("  WARNING: final chosen checkpoint is UNSAFE on final "
              "eval/test.")

    summary = {
        "seed": seed,
        "model_path": model_path,
        "dt": dt,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "train_seconds": train_seconds,
        "peak_rss_mb": _peak_rss_mb(),
        "eval": asdict(eval_summary),
        "test": asdict(test_summary),
        "best_during_training": {
            "selection_rule": (
                "safe checkpoints only; highest success, tie-break lower "
                "override; fallback to best candidate if no safe checkpoint "
                "exists"
            ),
            "selected_checkpoint_source": selected_checkpoint_source,
            "selected_episode": selected_episode,
            "success_rate": selected_success,
            "override_rate": selected_override,
            "safety_violation_rate": selected_safety,
            "final_checkpoint_safe": final_checkpoint_safe,
            "best_safe": {
                "episode": best_safe_episode,
                "success_rate": best_safe_success
                if best_safe_episode is not None else None,
                "override_rate": best_safe_override
                if best_safe_episode is not None else None,
                "safety_violation_rate": best_safe_safety,
            },
            "best_any": {
                "episode": best_any_episode,
                "success_rate": best_any_success
                if best_any_episode is not None else None,
                "override_rate": best_any_override
                if best_any_episode is not None else None,
                "safety_violation_rate": best_any_safety,
            },
        },
        "training_eval_history": eval_history,
        "config_used": cfg,
        "ensure_class_coverage": ensure_class_coverage,
        "balance_oracle_classes": balance_oracle_classes,
    }
    with open(os.path.join(seed_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    return summary
