"""Inference-only worker.

Loads a saved policy checkpoint and runs eval + test rollouts. Records the
peak RSS for the process in `inference_peak_rss_mb` — this is the inference-
only footprint (no training-time activations, no PPO buffers, no Adam state).

Intended to be invoked as a fresh subprocess per seed, so the RSS measurement
reflects what a deployment-style process would actually allocate.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict

import numpy as np

from clarity.runtime.env import SysMLEnv
from clarity.runtime.oracle import extract_interface
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt

from clarity.training.recurrent.composite import Composite
from clarity.training.recurrent.episode import evaluate
from clarity.training.recurrent.io import load_policy
from clarity.training.recurrent.policy import RecurrentActorCritic


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--test-episodes", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    # Probe env for dims
    probe = SysMLEnv(args.model_path, dt=args.dt, max_steps=args.max_steps,
                     phase=1, rng_seed=args.seed)
    obs_dim, n_actions = probe.obs_dim, probe.n_actions
    probe.close()

    iface = extract_interface(args.model_path)
    policy = RecurrentActorCritic(obs_dim, n_actions, args.hidden_dim,
                                  seed=args.seed)
    load_policy(policy, args.ckpt)
    composite = Composite(policy, iface["spec_shield"], iface["obs_names"])

    t0 = time.time()
    eval_sum = evaluate(
        env_factory=lambda: SysMLEnv(args.model_path, dt=args.dt,
                                     max_steps=args.max_steps,
                                     phase=2, rng_seed=10_000 + args.seed),
        composite=composite, n_episodes=args.eval_episodes,
        rng=np.random.default_rng(10_000 + args.seed), greedy=True)
    eval_seconds = time.time() - t0
    t0 = time.time()
    test_sum = evaluate(
        env_factory=lambda: SysMLEnv(args.model_path, dt=args.dt,
                                     max_steps=args.max_steps,
                                     phase=2, rng_seed=20_000 + args.seed),
        composite=composite, n_episodes=args.test_episodes,
        rng=np.random.default_rng(20_000 + args.seed), greedy=True)
    test_seconds = time.time() - t0

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    final_checkpoint_safe = (
        eval_sum.safety_violation_rate == 0.0
        and test_sum.safety_violation_rate == 0.0
    )

    out = {
        "seed": args.seed,
        "dt": args.dt,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "eval": asdict(eval_sum),
        "test": asdict(test_sum),
        "eval_seconds": eval_seconds,
        "test_seconds": test_seconds,
        "inference_peak_rss_mb": peak_rss,
        "final_checkpoint_safe": final_checkpoint_safe,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    print(f"[eval_only seed={args.seed}] "
          f"test_succ={test_sum.success_rate:.4f} "
          f"test_ov={test_sum.pooled_override_rate:.5f} "
          f"test_safety={test_sum.safety_violation_rate:.5f} "
          f"inf_peak_rss_mb={peak_rss:.1f} "
          f"eval_s={eval_seconds:.1f} test_s={test_seconds:.1f}",
          flush=True)
    if not final_checkpoint_safe:
        print(f"[eval_only seed={args.seed}] WARNING: loaded checkpoint is "
              "UNSAFE on eval/test", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
