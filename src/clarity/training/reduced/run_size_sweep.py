#!/usr/bin/env python3
"""Run one SysML-derived handmade reduced-MDP architecture per model.

Each run is a separate process so ``ru_maxrss`` in its summary is meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


PY = Path(sys.executable)

from clarity.sysml.inputs import inspect_sysml

FIELDNAMES = [
    "run_id",
    "model",
    "hidden_dim",
    "seed",
    "status",
    "returncode",
    "seconds",
    "params",
    "peak_rss_mb",
    "b_obs",
    "b_act",
    "test_safety",
    "test_accuracy",
    "test_override",
    "test_mean_steps",
    "summary_path",
    "log_path",
]


@dataclass(frozen=True)
class Job:
    model: str
    model_path: Path
    seed: int
    out_dir: Path
    overrides: list[str]

    @property
    def run_id(self) -> str:
        return f"{self.model}_seed{self.seed}"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


def _run_job(job: Job) -> dict[str, str]:
    run_dir = job.out_dir / "runs" / job.model / f"seed_{job.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    cmd = [
        str(PY),
        "-m",
        "clarity.training.reduced.train_one_seed",
        str(job.model_path),
        "--out-dir",
        str(run_dir),
        "--seed",
        str(job.seed),
        *job.overrides,
    ]
    start = time.time()
    proc = subprocess.run(
        cmd,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    seconds = time.time() - start
    log_path.write_text(proc.stdout, encoding="utf-8")

    row = {key: "" for key in FIELDNAMES}
    row.update(
        run_id=job.run_id,
        model=job.model,
        seed=str(job.seed),
        returncode=str(proc.returncode),
        seconds=f"{seconds:.3f}",
        log_path=str(log_path),
        summary_path=str(run_dir / "summary.json"),
    )
    summary_path = run_dir / "summary.json"
    if proc.returncode == 0 and summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        row.update(
            status="ok",
            hidden_dim=str(data.get("hidden_dim", "")),
            params=str(data.get("parameter_count", "")),
            peak_rss_mb=f"{float(data.get('peak_rss_mb', 0.0)):.3f}",
            b_obs=str(data.get("certified_buffer", {}).get("b_obs", "")),
            b_act=str(data.get("certified_buffer", {}).get("b_act", "")),
            test_safety=str(data.get("test", {}).get("safety_violation_rate", "")),
            test_accuracy=str(data.get("test", {}).get("success_rate", "")),
            test_override=str(data.get("test", {}).get("pooled_override_rate", "")),
            test_mean_steps=str(data.get("test", {}).get("mean_episode_steps", "")),
        )
    else:
        row["status"] = "failed"
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+", help="SysML file path")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="use tiny training settings for validation")
    args = ap.parse_args()

    models = [inspect_sysml(path) for path in args.model]
    non_discrete = [model.path for model in models if model.action_kind != "discrete"]
    if non_discrete:
        raise SystemExit(f"Boolean #Neural outputs required: {non_discrete}")
    keys = [model.key for model in models]
    if len(set(keys)) != len(keys):
        raise SystemExit("SysML package names must be unique")

    out_dir = Path(args.out_dir or (
        Path.cwd() / "outputs" / f"reduced_size_sweep_{time.strftime('%Y%m%d-%H%M%S')}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides: list[str] = []
    if args.smoke:
        overrides = [
            "--oracle-samples", "64",
            "--oracle-epochs", "2",
            "--ensure-class-coverage", "4",
            "--ppo-episodes", "8",
            "--episodes-per-update", "4",
            "--eval-interval", "4",
            "--eval-episodes", "4",
            "--test-episodes", "4",
            "--n-ppo-epochs", "1",
            "--minibatch-size", "4",
        ]

    manifest = {
        "models": [model.to_dict() for model in models],
        "seed": args.seed,
        "jobs": args.jobs,
        "smoke": args.smoke,
        "out_dir": str(out_dir),
        "shield_type": "program_ast_spec_shield",
        "device": "cpu",
        "notes": "one process per run so peak_rss_mb is per learning model",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    jobs = [
        Job(model.key, model.path, args.seed, out_dir, overrides)
        for model in models
    ]
    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        pending = {pool.submit(_run_job, job): job for job in jobs}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                job = pending.pop(fut)
                row = fut.result()
                rows.append(row)
                _write_csv(out_dir / "per_run.csv", rows)
                print(
                    f"{row['status'].upper():6s} {job.run_id:24s} "
                    f"rss={row['peak_rss_mb']} acc={row['test_accuracy']} "
                    f"override={row['test_override']} seconds={row['seconds']}"
                )
    _write_csv(out_dir / "per_run.csv", rows)
    print(f"WROTE {out_dir / 'per_run.csv'}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
