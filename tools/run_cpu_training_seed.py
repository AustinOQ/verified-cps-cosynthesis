#!/usr/bin/env python3
"""Run one fixed-seed CPU training job.

This is a thin artifact-local entry point around the NumPy CPU training
implementation. It keeps the pipeline output and CLI language independent of
the implementation package name while still reusing the updated trainer.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one CPU training seed for a SysML controller.")
    parser.add_argument("--program-root", required=True,
                        help="Repository containing the CPU training program.")
    parser.add_argument("--model-path", required=True,
                        help="SysML model file to train against.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--ensure-class-coverage", type=int, default=200)
    parser.add_argument("--balance-oracle-classes", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--oracle-samples", type=int, default=2000)
    parser.add_argument("--oracle-epochs", type=int, default=100)
    parser.add_argument("--ppo-episodes", type=int, default=2000)
    parser.add_argument("--minibatch-size", type=int, default=25)
    parser.add_argument("--bptt-chunk-size", type=int, default=200)
    parser.add_argument("--bc-aux-coeff", type=float, default=0.0)
    args = parser.parse_args(argv)

    program_root = os.path.abspath(args.program_root)
    if not os.path.isdir(program_root):
        raise SystemExit(
            f"CPU training program root does not exist: {program_root}")

    sys.path.insert(0, program_root)
    sys.path.insert(0, os.path.join(program_root, "rl"))

    from handmade.train_one_seed import train_one_seed

    os.makedirs(args.out, exist_ok=True)
    summary = train_one_seed(
        model_path=os.path.abspath(args.model_path),
        seed=args.seed,
        seed_dir=args.out,
        dt=args.dt,
        max_steps=args.max_steps,
        ensure_class_coverage=args.ensure_class_coverage,
        balance_oracle_classes=args.balance_oracle_classes,
        eval_episodes=args.eval_episodes,
        test_episodes=args.test_episodes,
        config={
            "oracle_samples": args.oracle_samples,
            "oracle_epochs": args.oracle_epochs,
            "ppo_episodes": args.ppo_episodes,
            "minibatch_size": args.minibatch_size,
            "bptt_chunk_size": args.bptt_chunk_size,
            "bc_aux_coeff": args.bc_aux_coeff,
        },
    )

    test = summary["test"]
    best = summary.get("best_during_training", {})
    print("\n========= CPU training summary =========")
    print(f"  model            : {args.model_name}")
    print(f"  seed             : {summary['seed']}")
    print(f"  selected_source  : "
          f"{best.get('selected_checkpoint_source', 'unknown')}")
    print(f"  selected_episode : {best.get('selected_episode', '')}")
    print(f"  selected_safe    : {best.get('final_checkpoint_safe', '')}")
    print(f"  train_seconds    : {summary['train_seconds']:.1f}")
    print(f"  peak_rss_mb      : {summary['peak_rss_mb']:.1f}")
    print(f"  test.success     : {test['success_rate']:.4f}")
    print(f"  test.override    : {test['pooled_override_rate']:.4f}")
    print(f"  test.safety_viol : {test['safety_violation_rate']:.4f}")
    if best.get("final_checkpoint_safe") is False:
        print("  WARNING          : final chosen checkpoint is UNSAFE")
    print(f"  output           : {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
