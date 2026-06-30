#!/usr/bin/env python3
"""Generate a compact controller fitting sequence artifact.

Each run rebuilds generated artifacts from the bundled SysML files, then trains
reduced feedforward controllers from the run-local certified architecture specs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
ARTIFACT = THIS.parents[1]
DEFAULT_ARCH = ARTIFACT / "bundle" / "architecture-fit"
ARCH = DEFAULT_ARCH.resolve()
REPO = ARCH.parent

MODEL_ORDER = ["thermostat", "cruise-discrete", "cruise-continuous", "mixing"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key in fieldnames) + " |")
    return "\n".join(lines)


def display_command(cmd: list[str]) -> str:
    labels = [
        (str(ARTIFACT), "$ARTIFACT"),
        (str(ARCH), "$ARCH"),
        (str(REPO), "$REPO"),
    ]
    parts = []
    for part in cmd:
        shown = part
        for prefix, label in labels:
            if shown.startswith(prefix):
                shown = label + shown[len(prefix):]
                break
        parts.append(shown)
    return " ".join(parts)


def run_command(cmd: list[str], log_path: Path, out_dir: Path) -> tuple[dict[str, Any], str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join([
        str(REPO),
        str(ARCH),
        str(REPO / "sysml-models"),
        str(REPO / "rl"),
    ])
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        cmd,
        cwd=ARCH,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(compact_output(proc.stdout), encoding="utf-8")
    return {
        "command": display_command(cmd),
        "returncode": proc.returncode,
        "log": str(log_path.relative_to(out_dir)),
        "captured_stdout_bytes": len(proc.stdout.encode("utf-8")),
    }, proc.stdout


def compact_output(text: str, max_lines: int = 80) -> str:
    lines = [
        line for line in text.splitlines()
        if "ANTLR runtime and generated code versions disagree" not in line
    ]
    if len(lines) <= max_lines:
        return "\n".join(lines) + ("\n" if lines else "")
    head = lines[: max_lines // 2]
    tail = lines[-max_lines // 2 :]
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join(head + [f"... omitted {omitted} lines ..."] + tail) + "\n"


def model_from_path(line: str) -> str | None:
    if "thermostat/model.sysml" in line:
        return "thermostat"
    if "cruise-controller-model/model.sysml" in line:
        return "cruise-discrete"
    if "cruise-continuous-model/model.sysml" in line:
        return "cruise-continuous"
    if "mixing-sysml-model/model.sysml" in line:
        return "mixing"
    return None


def parse_closure_text(text: str, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: str | None = None
    pattern = re.compile(r"last (?P<b_act>\d+) actions \+ current \+ (?P<b_obs>\d+) past obs")
    for line in text.splitlines():
        key = model_from_path(line)
        if key:
            current = key
            continue
        match = pattern.search(line)
        if match and current:
            rows.append({
                "model": current,
                "stage": stage,
                "b_obs": int(match.group("b_obs")),
                "b_act": int(match.group("b_act")),
                "claim": (
                    "memoryless current-decision controller"
                    if stage == "memoryless"
                    else "provable Markov/MDP buffer"
                ),
            })
            current = None
    return sorted(rows, key=lambda r: MODEL_ORDER.index(r["model"]))


def write_closure_log(log_path: Path, run: dict[str, Any], rows: list[dict[str, Any]],
                      raw_text: str) -> None:
    lines = [
        f"command: {run['command']}",
        f"returncode: {run['returncode']}",
        "",
        "buffer results:",
    ]
    for row in rows:
        lines.append(
            f"- {row['model']}: b_obs={row['b_obs']}, b_act={row['b_act']}; "
            f"{row['claim']}"
        )
    if run["returncode"] != 0:
        lines.extend(["", "failure tail:", compact_output(raw_text, max_lines=40)])
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_command_summary_log(log_path: Path, run: dict[str, Any],
                              summary_lines: list[str], raw_text: str) -> None:
    lines = [
        f"command: {run['command']}",
        f"returncode: {run['returncode']}",
        "",
        *summary_lines,
    ]
    if run["returncode"] != 0:
        lines.extend(["", "failure tail:", compact_output(raw_text, max_lines=40)])
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def stage_affine(out_dir: Path, py: str) -> dict[str, Any]:
    stage_dir = out_dir / "01_affine_rule"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    jobs = str(min(8, os.cpu_count() or 1))
    generated_specs = [
        (
            "thermostat",
            [py, "analytic_fit/thermostat_rule_fit.py", "--episodes", "20",
             "--jobs", jobs, "--out-dir", str(stage_dir / "thermostat_rule")],
            log_dir / "01_thermostat_rule_fit.txt",
            stage_dir / "thermostat_rule" / "report.json",
        ),
        (
            "cruise",
            [py, "analytic_fit/cruise_rule_fit.py", "--episodes", "20",
             "--jobs", jobs, "--out-dir", str(stage_dir / "cruise_rule")],
            log_dir / "01_cruise_rule_fit.txt",
            stage_dir / "cruise_rule" / "report.json",
        ),
    ]
    rule_runs: list[dict[str, Any]] = []
    for label, cmd, log_path, report_path in generated_specs:
        run, raw_text = run_command(cmd, log_path, out_dir)
        rule_runs.append({"label": label, "run": run})
        if run["returncode"] != 0 or not report_path.exists():
            raise RuntimeError(f"{label} rule evaluation failed; see {log_path}")
        report = read_json(report_path)
        episodes = report.get("episodes", report.get("episodes_per_model", ""))
        generated_rows = []
        for row in report.get("comparison_rows", []):
            if row.get("method") != "analytic rule sidecar":
                continue
            generated = {
                "model": row["model"],
                "method": "NeuralRequirement rule",
                "learned_params": row.get("learned_params"),
                "recurrent_state": "no",
                "episodes": episodes,
                "success_rate": row.get("success_rate"),
                "safety_violation_rate": row.get("safety_violation_rate"),
                "override_rate": row.get("override_rate"),
                "pointwise_agreement": row.get("pointwise_agreement"),
                "freshness": "fresh artifact run",
                "source": str(report_path.relative_to(out_dir)),
            }
            rows.append(generated)
            generated_rows.append(generated)
        write_command_summary_log(
            log_path,
            run,
            [
                f"{label} NeuralRequirement rule validation:",
                *[
                    f"- {row['model']}: episodes={row['episodes']}, "
                    f"success={float(row['success_rate']):.3f}, "
                    f"safety={float(row['safety_violation_rate']):.3f}, "
                    f"override={float(row['override_rate']):.3f}, "
                    f"pointwise={float(row['pointwise_agreement']):.3f}"
                    for row in generated_rows
                ],
                f"- metrics json: {report_path.relative_to(out_dir)}",
            ],
            raw_text,
        )

    mix_json = stage_dir / "mixing_rule_eval.json"
    mix_log = log_dir / "01_mixing_rule_eval.txt"
    mix_cmd = [
        py,
        str(ARTIFACT / "src" / "mixing_rule_eval.py"),
        "--episodes",
        "20",
        "--out-json",
        str(mix_json),
    ]
    mix_run, mix_text = run_command(mix_cmd, mix_log, out_dir)
    if mix_run["returncode"] != 0 or not mix_json.exists():
        raise RuntimeError(f"mixing rule evaluation failed; see {mix_log}")
    mix_report = read_json(mix_json)
    mix_metrics = mix_report["metrics"]
    write_command_summary_log(
        mix_log,
        mix_run,
        [
            "mixing NeuralRequirement rule validation:",
            (
                f"- episodes={mix_metrics['episodes']}, "
                f"success={mix_metrics['success_rate']:.3f}, "
                f"safety={mix_metrics['safety_violation_rate']:.3f}, "
                f"override={mix_metrics['override_rate']:.3f}, "
                f"pointwise={mix_metrics['pointwise_agreement']:.3f}"
            ),
            f"- metrics json: {mix_json.relative_to(out_dir)}",
        ],
        mix_text,
    )
    rows.append({
        "model": "mixing",
        "method": "NeuralRequirement rule",
        "learned_params": 0,
        "recurrent_state": "no",
        "episodes": mix_report["episodes"],
        "success_rate": mix_metrics["success_rate"],
        "safety_violation_rate": mix_metrics["safety_violation_rate"],
        "override_rate": mix_metrics["override_rate"],
        "pointwise_agreement": mix_metrics["pointwise_agreement"],
        "freshness": "fresh artifact run",
        "source": str(mix_json.relative_to(out_dir)),
    })

    rows = sorted(rows, key=lambda r: MODEL_ORDER.index(r["model"]))
    fields = [
        "model",
        "method",
        "learned_params",
        "recurrent_state",
        "episodes",
        "success_rate",
        "safety_violation_rate",
        "override_rate",
        "pointwise_agreement",
        "freshness",
        "source",
    ]
    write_csv(stage_dir / "summary.csv", rows, fields)
    write_json(stage_dir / "summary.json", {
        "rule_runs": rule_runs,
        "mixing_run": mix_run,
        "rows": rows,
    })
    return {"rule_runs": rule_runs, "mixing_run": mix_run, "rows": rows, "fields": fields}


def stage_weak(out_dir: Path, py: str) -> dict[str, Any]:
    stage_dir = out_dir / "02_memoryless"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "02_memoryless_check.txt"
    cmd = [
        py,
        "reconstruct_closure.py",
        "--legacy",
        "--max-obs",
        "2",
        "--max-act",
        "4",
        "thermostat",
        "cruise-continuous",
        "cruise-discrete",
        "mixing",
    ]
    run, raw_text = run_command(cmd, log_path, out_dir)
    rows = parse_closure_text(raw_text, "memoryless")
    for row in rows:
        row["command_log"] = str(log_path.relative_to(out_dir))
    write_closure_log(log_path, run, rows, raw_text)
    fields = ["model", "stage", "b_obs", "b_act", "claim", "command_log"]
    write_csv(stage_dir / "summary.csv", rows, fields)
    write_json(stage_dir / "summary.json", {"run": run, "rows": rows})
    return {"run": run, "rows": rows, "fields": fields}


def stage_strict(out_dir: Path, py: str) -> dict[str, Any]:
    stage_dir = out_dir / "03_markov_mdp"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    closure_log = log_dir / "03_markov_mdp_buffer_check.txt"
    generation_log = log_dir / "03_markov_mdp_z3_generation.txt"
    cmd = [
        py,
        "reconstruct_closure.py",
        "--max-obs",
        "2",
        "--max-act",
        "4",
        "thermostat",
        "cruise-continuous",
        "cruise-discrete",
        "mixing",
    ]
    closure_run, closure_text = run_command(cmd, closure_log, out_dir)
    closure_rows = parse_closure_text(closure_text, "markov_mdp")
    write_closure_log(closure_log, closure_run, closure_rows, closure_text)

    generation_json = stage_dir / "markov_mdp_generation.json"
    generation_cmd = [
        py,
        str(ARTIFACT / "src" / "generate_markov_mdp.py"),
        "--out-json",
        str(generation_json),
        "--artifact-dir",
        str(stage_dir),
        "--max-obs",
        "2",
        "--max-act",
        "4",
        "thermostat",
        "cruise-continuous",
        "cruise-discrete",
        "mixing",
    ]
    generation_run, generation_text = run_command(
        generation_cmd, generation_log, out_dir
    )
    generation_summary = (
        read_json(generation_json) if generation_json.exists() else {"rows": []}
    )
    rows = generation_summary.get("rows", [])
    rows = sorted(rows, key=lambda r: MODEL_ORDER.index(r["model"]))
    write_command_summary_log(
        generation_log,
        generation_run,
        [
            "fresh Markov/MDP proof generation:",
            *[
                f"- {row['model']}: checker={row['checker']}, "
                f"solver_status={row['solver_status']}, "
                f"buffer=b_obs={row['b_obs']}, b_act={row['b_act']}"
                for row in rows
            ],
            "- certificates, reduced specs, SMT-LIB queries, and proof/counterexample files are saved fresh",
            f"- compact summary json: {generation_json.relative_to(out_dir)}",
        ],
        generation_text,
    )
    fields = [
        "model",
        "b_obs",
        "b_act",
        "claim",
        "checker",
        "solver_status",
        "solver",
        "logic",
        "max_polynomial_degree",
        "source",
        "certificate_saved",
        "certificate_path",
        "reduced_mdp_spec_path",
        "smt2_path",
        "proof_or_disproof_path",
    ]
    write_csv(stage_dir / "summary.csv", rows, fields)
    write_json(stage_dir / "summary.json", {
        "closure_run": closure_run,
        "generation_run": generation_run,
        "closure_rows": closure_rows,
        "generation_rows": rows,
        "certificate_policy": generation_summary.get("certificate_policy", ""),
    })
    return {
        "closure_run": closure_run,
        "generation_run": generation_run,
        "closure_rows": closure_rows,
        "rows": rows,
        "fields": fields,
    }


def _model_path(name: str) -> Path:
    paths = {
        "thermostat": REPO / "sysml-models" / "thermostat" / "model.sysml",
        "cruise-discrete": REPO / "sysml-models" / "cruise-controller-model" / "model.sysml",
        "cruise-continuous": REPO / "sysml-models" / "cruise-continuous-model" / "model.sysml",
        "mixing": REPO / "sysml-models" / "mixing-sysml-model" / "model.sysml",
    }
    return paths[name]


def _spec_path(out_dir: Path, name: str) -> Path:
    return out_dir / "03_markov_mdp" / "reduced_mdp_specs" / f"{name}.reduced_mdp_spec.json"


def _read_training_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    test = data.get("test", {})
    selected = data.get("selected_checkpoint", {})
    buffer = data.get("certified_buffer", {})
    return {
        "hidden_dim": data.get("hidden_dim", ""),
        "params": data.get("parameter_count", ""),
        "b_obs": buffer.get("b_obs", ""),
        "b_act": buffer.get("b_act", ""),
        "device": data.get("device", ""),
        "shield": data.get("shield_type", ""),
        "policy": data.get("policy_type", ""),
        "train_seconds": data.get("train_seconds", ""),
        "peak_rss_mb": data.get("peak_rss_mb", ""),
        "selected_success": selected.get("success_rate", selected.get("acc", "")),
        "selected_override": selected.get("override_rate", selected.get("override", "")),
        "test_safety": test.get("safety_violation_rate", ""),
        "test_success": test.get("success_rate", ""),
        "test_override": test.get("pooled_override_rate", ""),
        "summary_json": "",
        "weights": "",
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_training_job(job: dict[str, Any], py: str, out_dir: Path) -> dict[str, Any]:
    name = job["model"]
    hidden_dim = int(job["hidden_dim"])
    kind = job["kind"]
    run_dir = job["run_dir"]
    log_path = job["log_path"]
    if kind == "discrete":
        cmd = [
            py,
            "-m",
            "reduced_handmade.train_one_seed",
            str(_model_path(name)),
            "--out-dir",
            str(run_dir),
            "--seed",
            str(job["seed"]),
            "--hidden-dim",
            str(hidden_dim),
            "--max-obs",
            "2",
            "--max-act",
            "4",
            "--reduced-mdp-spec",
            str(job["spec_path"]),
            *job.get("overrides", []),
        ]
        weight_path = run_dir / "best.npz"
    else:
        summary_json = run_dir / "summary.json"
        cmd = [
            py,
            "train_mlp_buffer.py",
            str(_model_path(name)),
            "--save-dir",
            str(run_dir),
            "--summary-json",
            str(summary_json),
            "--reduced-mdp-spec",
            str(job["spec_path"]),
            "--seed",
            str(job["seed"]),
            "--hidden-dim",
            str(hidden_dim),
            "--episodes",
            str(job["continuous_episodes"]),
            "--episodes-per-update",
            str(job["continuous_episodes_per_update"]),
            "--eval-interval",
            str(job["continuous_eval_interval"]),
            "--eval-episodes",
            str(job["continuous_eval_episodes"]),
            "--max-steps",
            str(job["continuous_max_steps"]),
            "--anneal-lr",
        ]
        weight_path = run_dir / "best.pt"

    started = time.time()
    run, raw_text = run_command(cmd, log_path, out_dir)
    seconds = time.time() - started
    summary_json = run_dir / "summary.json"
    if run["returncode"] != 0 or not summary_json.exists():
        return {
            "model": name,
            "kind": kind,
            "status": "failed",
            "hidden_dim": hidden_dim,
            "seconds": seconds,
            "command_log": str(log_path.relative_to(out_dir)),
            "summary_json": "",
            "weights": "",
            "returncode": run["returncode"],
        }
    row = _read_training_summary(summary_json)
    row.update({
        "model": name,
        "kind": kind,
        "status": "trained",
        "seconds": seconds,
        "summary_json": str(summary_json.relative_to(out_dir)),
        "weights": str(weight_path.relative_to(out_dir)),
        "command_log": str(log_path.relative_to(out_dir)),
        "returncode": run["returncode"],
    })
    return row


def _select_smallest_good(
    rows: list[dict[str, Any]],
    *,
    override_tolerance: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for model in sorted({r["model"] for r in rows}, key=MODEL_ORDER.index):
        candidates = [
            r for r in rows
            if r["model"] == model
            and r.get("status") == "trained"
            and _float_or_none(r.get("test_safety")) == 0.0
        ]
        if not candidates:
            selected.append({
                "model": model,
                "selection_status": "no_safe_trained_candidate",
            })
            continue
        best_acc = max(_float_or_none(r.get("test_success")) or 0.0 for r in candidates)
        best_acc_rows = [
            r for r in candidates
            if (_float_or_none(r.get("test_success")) or 0.0) == best_acc
        ]
        best_override = min(
            _float_or_none(r.get("test_override")) or float("inf")
            for r in best_acc_rows
        )
        eligible = [
            r for r in best_acc_rows
            if (_float_or_none(r.get("test_override")) or float("inf"))
            <= best_override + override_tolerance
        ]
        choice = sorted(
            eligible,
            key=lambda r: (
                int(r.get("params") or 10**18),
                int(r.get("hidden_dim") or 10**18),
                _float_or_none(r.get("test_override")) or float("inf"),
            ),
        )[0]
        selected.append({
            "model": model,
            "selection_status": "selected",
            "hidden_dim": choice.get("hidden_dim", ""),
            "params": choice.get("params", ""),
            "test_safety": choice.get("test_safety", ""),
            "test_success": choice.get("test_success", ""),
            "test_override": choice.get("test_override", ""),
            "best_success": best_acc,
            "best_override_at_best_success": best_override,
            "override_tolerance": override_tolerance,
            "summary_json": choice.get("summary_json", ""),
            "weights": choice.get("weights", ""),
        })
    return selected


def stage_training(out_dir: Path, py: str, *, smoke: bool = False,
                   jobs: int | None = None,
                   override_tolerance: float = 0.01) -> dict[str, Any]:
    stage_dir = out_dir / "04_reduced_training"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    hidden_dims = [2, 4, 8, 16, 32, 64]
    discrete_models = ["thermostat", "cruise-discrete", "mixing"]
    continuous_models = ["cruise-continuous"]
    job_count = max(1, jobs if jobs is not None else min(6, os.cpu_count() or 1))
    discrete_overrides: list[str] = []
    continuous_settings = {
        "episodes": 2000,
        "episodes_per_update": 20,
        "eval_interval": 100,
        "eval_episodes": 100,
        "max_steps": 1200,
    }
    if smoke:
        hidden_dims = [2, 4]
        discrete_overrides = [
            "--oracle-samples", "96",
            "--oracle-epochs", "4",
            "--oracle-batch-size", "64",
            "--ensure-class-coverage", "2",
            "--ppo-episodes", "16",
            "--episodes-per-update", "4",
            "--eval-interval", "4",
            "--eval-episodes", "4",
            "--test-episodes", "4",
            "--n-ppo-epochs", "1",
            "--minibatch-size", "4",
            "--max-steps", "300",
        ]
        continuous_settings = {
            "episodes": 8,
            "episodes_per_update": 2,
            "eval_interval": 2,
            "eval_episodes": 3,
            "max_steps": 300,
        }
        job_count = min(job_count, 4)

    runs: list[dict[str, Any]] = []

    training_jobs: list[dict[str, Any]] = []
    for name in discrete_models:
        spec_path = _spec_path(out_dir, name)
        if not spec_path.exists():
            rows.append({
                "model": name,
                "kind": "discrete",
                "status": "skipped_no_run_local_certified_spec",
                "summary_json": "",
                "weights": "",
                "command_log": "",
            })
            continue
        for hidden_dim in hidden_dims:
            run_dir = stage_dir / "runs" / name / f"h{hidden_dim}" / "seed_0"
            training_jobs.append({
                "model": name,
                "kind": "discrete",
                "hidden_dim": hidden_dim,
                "seed": 0,
                "run_dir": run_dir,
                "log_path": log_dir / f"04_train_{name}_h{hidden_dim}.txt",
                "spec_path": spec_path,
                "overrides": discrete_overrides,
            })

    for name in continuous_models:
        spec_path = _spec_path(out_dir, name)
        if not spec_path.exists():
            rows.append({
                "model": name,
                "kind": "continuous",
                "status": "skipped_no_run_local_certified_spec",
                "summary_json": "",
                "weights": "",
                "command_log": "",
            })
            continue
        for hidden_dim in hidden_dims:
            run_dir = stage_dir / "runs" / name / f"h{hidden_dim}" / "seed_0"
            training_jobs.append({
                "model": name,
                "kind": "continuous",
                "hidden_dim": hidden_dim,
                "seed": 0,
                "run_dir": run_dir,
                "log_path": log_dir / f"04_train_{name}_h{hidden_dim}.txt",
                "spec_path": spec_path,
                "continuous_episodes": continuous_settings["episodes"],
                "continuous_episodes_per_update": continuous_settings["episodes_per_update"],
                "continuous_eval_interval": continuous_settings["eval_interval"],
                "continuous_eval_episodes": continuous_settings["eval_episodes"],
                "continuous_max_steps": continuous_settings["max_steps"],
            })

    manifest = {
        "mode": "smoke" if smoke else "full",
        "hidden_dims": hidden_dims,
        "seed": 0,
        "jobs": job_count,
        "discrete_training_defaults": (
            "train_one_seed defaults: 2000 PPO episodes, 100 episodes per update, "
            "100-episode eval/checkpoint interval, 100 eval episodes per checkpoint, "
            "200 final test episodes, 2000 oracle samples, 100 oracle epochs"
            if not smoke else "smoke overrides"
        ),
        "continuous_training_settings": continuous_settings,
        "selection_rule": (
            "safe candidates only; maximize test success; among max-success candidates "
            f"choose the smallest parameter count within {override_tolerance:g} absolute "
            "override of the best override at that success"
        ),
    }
    write_json(stage_dir / "training_manifest.json", manifest)

    with ThreadPoolExecutor(max_workers=job_count) as pool:
        pending = {
            pool.submit(_run_training_job, job, py, out_dir): job
            for job in training_jobs
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                job = pending.pop(fut)
                row = fut.result()
                rows.append(row)
                runs.append({
                    "model": job["model"],
                    "hidden_dim": job["hidden_dim"],
                    "status": row.get("status"),
                    "command_log": row.get("command_log", ""),
                    "returncode": row.get("returncode", ""),
                })
                write_csv(stage_dir / "all_runs.csv", rows, [
                    "model", "kind", "status", "hidden_dim", "params",
                    "b_obs", "b_act", "device", "shield", "policy",
                    "train_seconds", "peak_rss_mb", "test_safety",
                    "test_success", "test_override", "summary_json",
                    "weights", "command_log",
                ])

    rows = sorted(rows, key=lambda r: MODEL_ORDER.index(r["model"]))
    selected_rows = _select_smallest_good(rows, override_tolerance=override_tolerance)
    fields = [
        "model",
        "kind",
        "status",
        "hidden_dim",
        "params",
        "b_obs",
        "b_act",
        "device",
        "shield",
        "policy",
        "test_safety",
        "test_success",
        "test_override",
        "summary_json",
        "weights",
        "command_log",
    ]
    write_csv(stage_dir / "summary.csv", rows, fields)
    selected_fields = [
        "model", "selection_status", "hidden_dim", "params", "test_safety",
        "test_success", "test_override", "best_success",
        "best_override_at_best_success", "override_tolerance",
        "summary_json", "weights",
    ]
    write_csv(stage_dir / "selected_smallest_good.csv", selected_rows, selected_fields)
    write_json(stage_dir / "summary.json", {
        "manifest": manifest,
        "runs": runs,
        "rows": rows,
        "selected_smallest_good": selected_rows,
    })
    return {
        "runs": runs,
        "rows": rows,
        "fields": fields,
        "selected_rows": selected_rows,
        "selected_fields": selected_fields,
        "manifest": manifest,
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Controller Fitting Sequence Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("This report demonstrates the simplified controller fitting sequence:")
    lines.append("")
    lines.append("```text")
    lines.append("NeuralRequirement affine/rule fit")
    lines.append("  -> memoryless controller check")
    lines.append("    -> provable Markov/MDP controller check")
    lines.append("      -> reduced feedforward training")
    lines.append("```")
    lines.append("")
    lines.append("The run overwrites generated outputs and rebuilds certificates, specs,")
    lines.append("proof files, and trained weights from the bundled SysML models.")
    lines.append("")

    affine = summary["stages"]["affine_rule"]
    lines.append("## 1. Affine/Rule Fit")
    lines.append("")
    lines.append("Claim: the NeuralRequirement already induces a transparent rule/affine-predicate controller.")
    lines.append("")
    lines.append(markdown_table(affine["rows"], affine["fields"]))
    lines.append("")
    lines.append("Every row in this stage is generated fresh by this artifact.")
    lines.append("")
    lines.append("Fit status: this is the direct analytical path. The controller has zero")
    lines.append("learned parameters when the requirement itself gives the rule.")
    lines.append("")

    weak = summary["stages"]["memoryless"]
    lines.append("## 2. Memoryless Controller Check")
    lines.append("")
    lines.append("Claim: a finite buffer is enough for the current controller decision.")
    lines.append("This supports a memoryless controller, but it does not prove the full")
    lines.append("modeled process is Markov.")
    lines.append("")
    lines.append("Fit status: this stage does not fit weights. It provides a non-recurrent")
    lines.append("input form that can be used by a later feedforward fit.")
    lines.append("")
    lines.append(markdown_table(weak["rows"], weak["fields"]))
    lines.append("")
    lines.append(f"Command log: `{weak['run']['log']}`")
    lines.append("")

    strict = summary["stages"]["markov_mdp"]
    lines.append("## 3. Provable Markov/MDP Controller Check")
    lines.append("")
    lines.append("Claim: a finite buffer is enough to prove the next modeled step is")
    lines.append("determined. This stage generates the proof obligations from the bundled")
    lines.append("SysML files and calls Z3 during the run. It saves certificates,")
    lines.append("reduced-MDP specs, SMT-LIB queries, and proof/counterexample transcripts.")
    lines.append("")
    lines.append("Fit status: this stage does not fit weights. It provides a certified")
    lines.append("augmented state for later feedforward, tabular, or analytical fitting.")
    lines.append("")
    lines.append(markdown_table(strict["rows"], strict["fields"]))
    lines.append("")
    lines.append(f"Reconstructibility log: `{strict['closure_run']['log']}`")
    lines.append(f"Z3 generation log: `{strict['generation_run']['log']}`")
    lines.append("")

    training = summary["stages"]["reduced_training"]
    lines.append("## 4. Reduced Feedforward Training")
    lines.append("")
    lines.append("Claim: the run-local certified specs can be consumed by non-recurrent")
    lines.append("feedforward trainers. This stage sweeps hidden dimensions")
    lines.append("2, 4, 8, 16, 32, and 64, then selects the smallest good model.")
    lines.append("Discrete models use the handmade NumPy PPO stack.")
    lines.append("The continuous cruise model uses the bundled continuous MLP PPO stack.")
    lines.append("All training runs on CPU and keeps the exact program shield or exact")
    lines.append("continuous projection as the safety authority.")
    lines.append("")
    lines.append("Selection rule: safe candidates only. Maximize held-out test success.")
    lines.append("Among the max-success candidates, choose the smallest parameter count")
    lines.append("within the configured override tolerance of the best override at that")
    lines.append("success level.")
    lines.append("")
    lines.append("### Selected Smallest Good")
    lines.append("")
    lines.append(markdown_table(training["selected_rows"], training["selected_fields"]))
    lines.append("")
    lines.append("### All Training Runs")
    lines.append("")
    lines.append(markdown_table(training["rows"], training["fields"]))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Stage 1 is the cheapest: no learned parameters when the NeuralRequirement")
    lines.append("  already defines the controller.")
    lines.append("- Stage 2 removes controller recurrence for the current decision.")
    lines.append("- Stage 3 is the stronger provable Markov/MDP claim.")
    lines.append("- Stage 4 trains small feedforward policies from the Stage 3 specs.")
    lines.append("")
    lines.append("The exact shield / exact continuous projection remains the safety authority")
    lines.append("across deployed/evaluated stages.")
    lines.append("")
    (out_dir / "fitting_sequence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ARTIFACT / "outputs" / "latest"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--smoke-training", action="store_true",
                        help="run a short Stage 4 wiring test instead of the full training sweep")
    parser.add_argument("--training-jobs", type=int, default=None,
                        help="parallel Stage 4 training jobs; default=min(6,cpu_count)")
    parser.add_argument("--override-tolerance", type=float, default=0.01,
                        help="absolute override slack for selecting a smaller max-accuracy model")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture_fit_root": str(ARCH),
        "python_bin": args.python_bin,
        "stages": {},
    }
    summary["stages"]["affine_rule"] = stage_affine(out_dir, args.python_bin)
    summary["stages"]["memoryless"] = stage_weak(out_dir, args.python_bin)
    summary["stages"]["markov_mdp"] = stage_strict(out_dir, args.python_bin)
    summary["stages"]["reduced_training"] = stage_training(
        out_dir,
        args.python_bin,
        smoke=args.smoke_training,
        jobs=args.training_jobs,
        override_tolerance=args.override_tolerance,
    )

    write_json(out_dir / "fitting_sequence_summary.json", summary)
    write_report(out_dir, summary)
    print(f"WROTE {out_dir / 'fitting_sequence_report.md'}")
    print(f"WROTE {out_dir / 'fitting_sequence_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
