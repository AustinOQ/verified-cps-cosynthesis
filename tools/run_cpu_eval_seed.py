#!/usr/bin/env python3
"""Evaluate one CPU-trained checkpoint.

This artifact-local entry point keeps the top-level pipeline independent of
the implementation package name used by the neighboring CPU training program.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one CPU-trained SysML controller checkpoint.")
    parser.add_argument("--program-root", required=True,
                        help="Repository containing the CPU training program.")
    parser.add_argument("--model-path", required=True,
                        help="SysML model file to evaluate against.")
    parser.add_argument("--ckpt", required=True,
                        help="CPU checkpoint to evaluate.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu-mode", choices=("single", "aggressive"),
                        default="single")
    parser.add_argument("--cpu-affinity-core", default="0")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    program_root = os.path.abspath(args.program_root)
    if not os.path.isdir(program_root):
        raise SystemExit(
            f"CPU training program root does not exist: {program_root}")

    sys.path.insert(0, program_root)
    sys.path.insert(0, os.path.join(program_root, "rl"))

    from handmade.eval_only import main as eval_main

    rc = eval_main([
        "--model-path", os.path.abspath(args.model_path),
        "--ckpt", os.path.abspath(args.ckpt),
        "--seed", str(args.seed),
        "--dt", str(args.dt),
        "--max-steps", str(args.max_steps),
        "--eval-episodes", str(args.eval_episodes),
        "--test-episodes", str(args.test_episodes),
        "--out", os.path.abspath(args.out),
    ])
    if rc == 0:
        with open(args.out) as f:
            data = json.load(f)
        data["execution"] = {
            "cpu_mode": args.cpu_mode,
            "cpu_affinity_core": args.cpu_affinity_core
            if args.cpu_mode == "single" else "",
        }
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
