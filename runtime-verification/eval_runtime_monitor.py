#!/usr/bin/env python3
"""Standalone Torch-checkpoint runtime monitor evaluator.

The active artifact pipeline evaluates NumPy CPU checkpoints through
``run_pipeline.sh`` and writes results under ``metrics/runtime_results/``. This
script is retained for comparison experiments with Torch checkpoints under
``rl/checkpoints/<system>/best.pt``.

It runs greedy episodes with a trained Torch policy + shield, and at each step
evaluates the ``#NeuralRequirement`` via the AST evaluator from ``verify.py``.

Usage:
    python runtime-verification/eval_runtime_monitor.py --episodes 100
    python runtime-verification/eval_runtime_monitor.py --episodes 100 --margin 2.0
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "sysml-models"))
sys.path.insert(0, str(ROOT_DIR / "rl"))
sys.path.insert(0, str(SCRIPT_DIR))

from verify import load_model, evaluate  # noqa: E402
from env import SysMLEnv  # noqa: E402
from composite_model import build_composite_model  # noqa: E402
from model import RecurrentActorCritic  # noqa: E402

SYSTEMS = [
    ("thermostat", "sysml-models/thermostat/model.sysml"),
    ("cruise", "sysml-models/cruise-controller-model/model.sysml"),
    ("mixing", "sysml-models/mixing-sysml-model/model.sysml"),
]


def eval_system(name, sysml_rel, episodes, margin):
    sysml_path = str(ROOT_DIR / sysml_rel)
    ckpt = ROOT_DIR / "rl" / "checkpoints" / name / "best.pt"
    if not ckpt.exists():
        print(f"  [{name}] No checkpoint at {ckpt}, skipping")
        return None

    # Load environment and model
    env = SysMLEnv(sysml_path, phase=2)
    policy = RecurrentActorCritic(env.obs_dim, env.n_actions)
    composite = build_composite_model(sysml_path, policy)

    # Checkpoint may be saved as bare policy weights or composite weights
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if any(k.startswith("policy.") for k in state):
        composite.load_state_dict(state, strict=False)
    else:
        composite.policy.load_state_dict(state, strict=False)
    composite.eval()

    # Load monitor AST
    monitor = load_model(sysml_path)
    if monitor is None:
        print(f"  [{name}] Could not load monitor AST")
        return None

    total_steps = 0
    actual_violations = 0
    detected_violations = 0
    false_positives = 0
    false_negatives = 0
    check_times_us = []
    override_count = 0

    for ep in range(episodes):
        obs = env.reset()
        hidden = torch.zeros(1, 1, 64)
        done = False
        # Get initial raw state via a no-op peek at the twin's last output
        last_raw = env._twin._last_state if hasattr(env._twin, '_last_state') else {}

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            # Build obs_dict from raw engine state (unnormalized)
            obs_dict = {}
            eng_state = env._twin._engine.state
            for k in env._obs_keys:
                # Try short key from last twin output first
                if k in last_raw:
                    obs_dict[k] = last_raw[k]
                else:
                    # Fall back to qualified engine state
                    for sk, sv in eng_state.items():
                        if sk.endswith("::" + k) or sk == k:
                            obs_dict[k] = sv
                            break
                # Last resort: use denormalized obs
                if k not in obs_dict:
                    idx = env._obs_keys.index(k)
                    obs_dict[k] = float(obs[idx]) * env._obs_scale

            action, _, _, hidden, overridden = composite.act(
                obs_t, hidden, obs_dict, greedy=True)
            if overridden:
                override_count += 1

            # Build values dict for monitor
            actuators = env._action_map[action]
            values = {**monitor.unchanging, **obs_dict, **actuators}

            # Time the monitor check
            t0 = time.perf_counter_ns()
            try:
                satisfied = evaluate(monitor.requirement_ast, values,
                                     monitor.subject_var)
            except Exception:
                satisfied = True  # can't evaluate = don't flag
            t1 = time.perf_counter_ns()
            check_times_us.append((t1 - t0) / 1000.0)

            obs, reward, done, info = env.step(action)
            total_steps += 1
            if "state" in info:
                last_raw = info["state"]

            # Ground truth: did the environment flag a violation?
            is_actual_violation = (reward == -1.0 and done)
            monitor_flagged = not satisfied

            if is_actual_violation:
                actual_violations += 1
            if monitor_flagged:
                detected_violations += 1
            if monitor_flagged and not is_actual_violation:
                false_positives += 1
            if is_actual_violation and not monitor_flagged:
                false_negatives += 1

    env.close()

    times = np.array(check_times_us)
    return {
        "system": name,
        "episodes": episodes,
        "total_steps": total_steps,
        "actual_violations": actual_violations,
        "detected_violations": detected_violations,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "override_rate": f"{override_count / max(total_steps, 1) * 100:.1f}%",
        "avg_check_us": f"{times.mean():.2f}",
        "median_check_us": f"{np.median(times):.2f}",
        "p99_check_us": f"{np.percentile(times, 99):.2f}",
        "max_check_us": f"{times.max():.2f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = ROOT_DIR / "metrics" / "runtime_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_margin{args.margin}" if args.margin > 0 else ""
    out_file = out_dir / f"runtime_monitor_summary{suffix}.csv"

    results = []
    for name, sysml_rel in SYSTEMS:
        print(f"Evaluating {name} ({args.episodes} episodes)...")
        row = eval_system(name, sysml_rel, args.episodes, args.margin)
        if row:
            results.append(row)
            print(f"  steps={row['total_steps']} violations={row['actual_violations']} "
                  f"detected={row['detected_violations']} FP={row['false_positives']} "
                  f"FN={row['false_negatives']} avg={row['avg_check_us']}us "
                  f"p99={row['p99_check_us']}us override={row['override_rate']}")

    if results:
        with open(out_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    main()
