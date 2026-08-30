"""CLI runner for one recurrent controller training seed.

Usage:
  python -m clarity.training.recurrent.run --model thermostat --seed 3 \
    --ensure-class-coverage 200 \
    --out outputs/recurrent_training/thermostat/seed_3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from clarity.models import model_path
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt
from clarity.training.recurrent.train_one_seed import train_one_seed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="thermostat",
                   choices=["cruise", "mixing", "thermostat"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Output directory under outputs/recurrent_training by default")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    p.add_argument("--ensure-class-coverage", type=int, default=0)
    p.add_argument("--balance-oracle-classes", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--test-episodes", type=int, default=200)
    # PPO/oracle config overrides (defaults match rl/train.py MODE_CONFIG["full"])
    p.add_argument("--oracle-samples", type=int, default=2000)
    p.add_argument("--oracle-epochs", type=int, default=100)
    p.add_argument("--ppo-episodes", type=int, default=2000)
    p.add_argument("--minibatch-size", type=int, default=25,
                   help="0 = full batch; default 25.")
    p.add_argument("--bptt-chunk-size", type=int, default=200,
                   help="0 = full BPTT; default 200 (chunked + truncated).")
    p.add_argument("--bc-aux-coeff", type=float, default=0.0)
    args = p.parse_args(argv)

    sysml_path = model_path(args.model)

    out_dir = args.out or str(
        Path.cwd() / "outputs" / "recurrent_training" / args.model / f"seed_{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    config_overrides = {
        "oracle_samples": args.oracle_samples,
        "oracle_epochs": args.oracle_epochs,
        "ppo_episodes": args.ppo_episodes,
        "minibatch_size": args.minibatch_size,
        "bptt_chunk_size": args.bptt_chunk_size,
        "bc_aux_coeff": args.bc_aux_coeff,
    }

    summary = train_one_seed(
        model_path=str(sysml_path),
        seed=args.seed,
        seed_dir=out_dir,
        dt=args.dt,
        max_steps=args.max_steps,
        ensure_class_coverage=args.ensure_class_coverage,
        balance_oracle_classes=args.balance_oracle_classes,
        eval_episodes=args.eval_episodes,
        test_episodes=args.test_episodes,
        config=config_overrides,
    )

    print("\n========= summary =========")
    print(f"  seed             : {summary['seed']}")
    print(f"  model            : {args.model}")
    print(f"  train_seconds    : {summary['train_seconds']:.1f}")
    print(f"  peak_rss_mb      : {summary['peak_rss_mb']:.1f}")
    print(f"  eval.success     : {summary['eval']['success_rate']:.4f}")
    print(f"  eval.override    : {summary['eval']['pooled_override_rate']:.4f}")
    print(f"  test.success     : {summary['test']['success_rate']:.4f}")
    print(f"  test.override    : {summary['test']['pooled_override_rate']:.4f}")
    print(f"  test.safety_viol : {summary['test']['safety_violation_rate']:.4f}")
    print(f"  output           : {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
