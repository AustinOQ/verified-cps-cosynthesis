"""Parallel-train + sequential-inference orchestrator for handmade n30.

Two phases:

  Phase 1 (parallel, --jobs workers):
    Each worker trains a single seed, saves best.npz, writes a partial
    summary with training metrics + training peak RSS.

  Phase 2 (sequential, one process at a time):
    For each seed, spawn the packaged evaluation program as a fresh subprocess. The
    subprocess loads best.npz, runs eval + test rollouts, and records
    *contention-free* per-step latencies (policy/shield/total in µs)
    plus inference peak RSS. Sequential to keep timing clean.

Outputs per seed:
  seed_<n>/training_summary.json
  seed_<n>/best.npz
  seed_<n>/inference_summary.json   (with latency stats)
  seed_<n>/summary.json             (combined)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from typing import List

from clarity.models import model_path
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt
from clarity.training.recurrent.train_one_seed import train_one_seed


def _train_worker(args_tuple):
    """One seed: train + save checkpoint. NO inference here."""
    (seed, model_name, run_dir, dt, max_steps, ensure_class_coverage,
     balance_oracle_classes, eval_episodes, test_episodes,
     oracle_samples, oracle_epochs, ppo_episodes,
     minibatch_size, bptt_chunk_size, bc_aux_coeff) = args_tuple

    sysml_path = model_path(model_name)
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    cfg = {
        "oracle_samples": oracle_samples,
        "oracle_epochs": oracle_epochs,
        "ppo_episodes": ppo_episodes,
        "minibatch_size": minibatch_size,
        "bptt_chunk_size": bptt_chunk_size,
        "bc_aux_coeff": bc_aux_coeff,
    }

    try:
        train_summary = train_one_seed(
            model_path=str(sysml_path), seed=seed, seed_dir=seed_dir,
            dt=dt, max_steps=max_steps,
            ensure_class_coverage=ensure_class_coverage,
            balance_oracle_classes=balance_oracle_classes,
            eval_episodes=eval_episodes, test_episodes=test_episodes,
            config=cfg)
    except Exception as e:
        import traceback
        return ("error", {"seed": seed, "error": str(e),
                          "traceback": traceback.format_exc()})

    with open(os.path.join(seed_dir, "training_summary.json"), "w") as f:
        json.dump(train_summary, f, indent=2, sort_keys=True, default=str)
    return ("ok", train_summary)


def _run_inference(seed: int, model_path: str, dt: float, max_steps: int,
                   eval_episodes: int, test_episodes: int,
                   seed_dir: str) -> dict:
    """Spawn eval_only as a fresh subprocess; return inference dict."""
    ckpt = os.path.join(seed_dir, "best.npz")
    out_json = os.path.join(seed_dir, "inference_summary.json")
    cmd = [
        sys.executable, "-m", "clarity.training.recurrent.eval_only",
        "--model-path", model_path,
        "--ckpt", ckpt,
        "--seed", str(seed),
        "--dt", str(dt),
        "--max-steps", str(max_steps),
        "--eval-episodes", str(eval_episodes),
        "--test-episodes", str(test_episodes),
        "--out", out_json,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)
    return {"error": proc.stderr[-3000:], "stdout": proc.stdout[-1500:]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["cruise", "mixing", "thermostat"])
    p.add_argument("--seeds", default="0..29")
    p.add_argument("--jobs", type=int, default=30,
                   help="Parallel workers for TRAIN phase. "
                        "Inference is always sequential.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    p.add_argument("--ensure-class-coverage", type=int, default=200)
    p.add_argument("--balance-oracle-classes", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--test-episodes", type=int, default=200)
    p.add_argument("--oracle-samples", type=int, default=2000)
    p.add_argument("--oracle-epochs", type=int, default=100)
    p.add_argument("--ppo-episodes", type=int, default=2000)
    p.add_argument("--minibatch-size", type=int, default=25)
    p.add_argument("--bptt-chunk-size", type=int, default=200)
    p.add_argument("--bc-aux-coeff", type=float, default=0.0)
    args = p.parse_args(argv)

    if ".." in args.seeds:
        lo, hi = (int(x) for x in args.seeds.split(".."))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump({
            "model": args.model, "seeds": seeds,
            "train_jobs": args.jobs,
            "dt": args.dt,
            "max_steps": args.max_steps,
            "ensure_class_coverage": args.ensure_class_coverage,
            "balance_oracle_classes": args.balance_oracle_classes,
            "eval_episodes": args.eval_episodes,
            "test_episodes": args.test_episodes,
            "oracle_samples": args.oracle_samples,
            "oracle_epochs": args.oracle_epochs,
            "ppo_episodes": args.ppo_episodes,
            "minibatch_size": args.minibatch_size,
            "bptt_chunk_size": args.bptt_chunk_size,
            "bc_aux_coeff": args.bc_aux_coeff,
            "started_at": _dt.datetime.now().isoformat(),
        }, f, indent=2, sort_keys=True)

    work = [(s, args.model, args.run_dir, args.dt, args.max_steps,
              args.ensure_class_coverage, args.balance_oracle_classes,
              args.eval_episodes, args.test_episodes,
              args.oracle_samples, args.oracle_epochs, args.ppo_episodes,
              args.minibatch_size, args.bptt_chunk_size, args.bc_aux_coeff)
             for s in seeds]

    sysml_path = str(model_path(args.model))

    train_results: List[dict] = []
    errors: List[dict] = []

    # ---------------- Phase 1: parallel training ----------------
    t_train_start = time.time()
    print(f"\n[{args.model}] PHASE 1 — training {len(work)} seeds with "
          f"{args.jobs} parallel workers", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.jobs) as pool:
        for status, payload in pool.imap_unordered(_train_worker, work):
            if status == "error":
                errors.append(payload)
                print(f"  TRAIN seed {payload.get('seed')}: ERROR  "
                      f"{payload.get('error')}", flush=True)
                continue
            train_results.append(payload)
            elapsed = time.time() - t_train_start
            print(f"  TRAIN seed {payload['seed']:>2} done  "
                  f"train={payload['train_seconds']:.1f}s  "
                  f"train_rss={payload['peak_rss_mb']:.1f}MB  "
                  f"[{elapsed:.0f}s]", flush=True)
    train_seconds = time.time() - t_train_start
    print(f"[{args.model}] PHASE 1 done in {train_seconds:.0f}s; "
          f"errors={len(errors)}", flush=True)

    # ---------------- Phase 2: sequential inference ----------------
    train_by_seed = {tr["seed"]: tr for tr in train_results}
    inf_results: List[dict] = []
    t_inf_start = time.time()
    print(f"\n[{args.model}] PHASE 2 — sequential inference "
          f"(clean latency) for {len(train_results)} seeds", flush=True)
    for seed in sorted(train_by_seed):
        seed_dir = os.path.join(args.run_dir, f"seed_{seed}")
        inf = _run_inference(seed, sysml_path, args.dt, args.max_steps,
                              args.eval_episodes, args.test_episodes,
                              seed_dir)
        inf_results.append((seed, inf))
        if "error" in inf:
            print(f"  INF   seed {seed:>2}: ERROR  {inf['error'][:300]}",
                  flush=True)
            continue
        test = inf.get("test", {})
        print(f"  INF   seed {seed:>2}  "
              f"inf_rss={inf.get('inference_peak_rss_mb', 0):.1f}MB  "
              f"test_succ={test.get('success_rate', 0):.3f}  "
              f"test_ov={test.get('pooled_override_rate', 0):.4f}  "
              f"shield_us_mean={test.get('shield_us_mean', 0):.2f}  "
              f"shield_us_p99={test.get('shield_us_p99', 0):.2f}",
              flush=True)
    inf_seconds = time.time() - t_inf_start
    print(f"[{args.model}] PHASE 2 done in {inf_seconds:.0f}s", flush=True)

    # ---------------- Combine + write per-seed + run-level summaries ----------------
    per_seed: List[dict] = []
    for seed, inf in inf_results:
        tr = train_by_seed[seed]
        combined = {
            "seed": seed,
            "model": args.model,
            "training": {
                "train_seconds": tr["train_seconds"],
                "training_peak_rss_mb": tr["peak_rss_mb"],
                "eval": tr["eval"],
                "test": tr["test"],
                "best_during_training": tr["best_during_training"],
                "training_eval_history": tr["training_eval_history"],
            },
            "inference_only": inf,
            "config": tr["config_used"],
            "ensure_class_coverage": args.ensure_class_coverage,
            "balance_oracle_classes": args.balance_oracle_classes,
        }
        per_seed.append(combined)
        seed_dir = os.path.join(args.run_dir, f"seed_{seed}")
        with open(os.path.join(seed_dir, "summary.json"), "w") as f:
            json.dump(combined, f, indent=2, sort_keys=True, default=str)

    per_seed.sort(key=lambda x: x["seed"])
    summary = {
        "model": args.model,
        "dt": args.dt,
        "n_seeds": len(per_seed),
        "n_errors": len(errors),
        "train_wall_seconds": train_seconds,
        "inference_wall_seconds": inf_seconds,
        "per_seed": per_seed,
        "errors": errors,
    }
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)

    if per_seed:
        rows = []
        for p_ in per_seed:
            t_ = p_["training"]
            i_ = p_.get("inference_only", {})
            it = i_.get("test", {})
            rows.append({
                "seed": p_["seed"],
                "model": p_["model"],
                "train_seconds": t_["train_seconds"],
                "training_peak_rss_mb": t_["training_peak_rss_mb"],
                "inference_peak_rss_mb": i_.get("inference_peak_rss_mb", ""),
                "test_success_rate": it.get("success_rate", ""),
                "test_pooled_override_rate": it.get("pooled_override_rate", ""),
                "test_safety_violation_rate": it.get("safety_violation_rate", ""),
                "test_mean_episode_steps": it.get("mean_episode_steps", ""),
                "test_n_steps": it.get("n_steps", ""),
                "shield_us_mean": it.get("shield_us_mean", ""),
                "shield_us_p95":  it.get("shield_us_p95", ""),
                "shield_us_p99":  it.get("shield_us_p99", ""),
                "policy_us_mean": it.get("policy_us_mean", ""),
                "policy_us_p95":  it.get("policy_us_p95", ""),
                "policy_us_p99":  it.get("policy_us_p99", ""),
                "total_us_mean":  it.get("total_us_mean", ""),
                "total_us_p95":   it.get("total_us_p95", ""),
                "total_us_p99":   it.get("total_us_p99", ""),
                "inference_eval_seconds": i_.get("eval_seconds", ""),
                "inference_test_seconds": i_.get("test_seconds", ""),
            })
        with open(os.path.join(args.run_dir, "per_seed.csv"), "w",
                   newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    total = time.time() - t_train_start
    print(f"\n[{args.model}] all done: train={train_seconds:.0f}s "
          f"inference={inf_seconds:.0f}s total={total:.0f}s; "
          f"errors={len(errors)}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
