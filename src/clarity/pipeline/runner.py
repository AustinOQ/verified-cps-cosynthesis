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
ARTIFACT = THIS.parents[3]
DEFAULT_ARCH = ARTIFACT / "bundle" / "architecture-fit"
ARCH = DEFAULT_ARCH.resolve()
REPO = ARCH.parent
for import_path in (ARCH, REPO / "sysml-models", REPO / "rl"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sysml_inputs import SysMLInput, discover_sysml
from runtime_settings import DEFAULT_DT, validate_dt
from shield import SpecShield, _collect_refs


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
    python_paths = [
        str(REPO),
        str(ARCH),
        str(REPO / "sysml-models"),
        str(REPO / "rl"),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
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


def _model_order(models: list[SysMLInput]) -> dict[str, int]:
    return {model.key: index for index, model in enumerate(models)}


def parse_closure_text(
    text: str,
    stage: str,
    models: list[SysMLInput],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: str | None = None
    paths = {str(model.path): model for model in models}
    pattern = re.compile(r"last (?P<b_act>\d+) actions \+ current \+ (?P<b_obs>\d+) past obs")
    for line in text.splitlines():
        model = paths.get(line.strip())
        if model is not None:
            current = model.key
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
    order = _model_order(models)
    return sorted(rows, key=lambda row: order[row["model"]])


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


def stage_affine(
    out_dir: Path,
    py: str,
    models: list[SysMLInput],
    dt: float,
) -> dict[str, Any]:
    stage_dir = out_dir / "01_affine_rule"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rule_runs: list[dict[str, Any]] = []
    for model in models:
        report_path = stage_dir / "models" / f"{model.key}.json"
        log_path = log_dir / f"01_rule_{model.key}.txt"
        cmd = [
            py,
            str(ARTIFACT / "src" / "clarity" / "pipeline" / "affine_rule.py"),
            str(model.path),
            "--episodes",
            "20",
            "--dt",
            str(dt),
            "--out-json",
            str(report_path),
        ]
        run, raw_text = run_command(cmd, log_path, out_dir)
        rule_runs.append({"model": model.key, "run": run})
        if run["returncode"] != 0 or not report_path.exists():
            raise RuntimeError(f"{model.key} rule extraction failed; see {log_path}")
        report = read_json(report_path)
        report_dt = validate_dt(report.get("dt"))
        if report_dt != dt:
            raise RuntimeError(
                f"{model.key} affine-stage dt does not match run dt: "
                f"{report_dt} != {dt}"
            )
        row = {
            "model": model.key,
            "model_name": model.name,
            "action_kind": model.action_kind,
            "dt": report_dt,
            "status": report.get("status", ""),
            "method": report.get("method", ""),
            "learned_params": report.get("learned_params", ""),
            "recurrent_state": report.get("recurrent_state", ""),
            "episodes": report.get("episodes", ""),
            "success_rate": report.get("success_rate", ""),
            "safety_violation_rate": report.get("safety_violation_rate", ""),
            "override_rate": report.get("override_rate", ""),
            "pointwise_agreement": report.get("pointwise_agreement", ""),
            "source": str(report_path.relative_to(out_dir)),
        }
        rows.append(row)
        metric_text = (
            f"episodes={row['episodes']}, success={float(row['success_rate']):.3f}, "
            f"safety={float(row['safety_violation_rate']):.3f}, "
            f"override={float(row['override_rate']):.3f}"
            if row["status"] == "evaluated"
            else f"status={row['status']}"
        )
        write_command_summary_log(
            log_path,
            run,
            [
                f"{model.key} #NeuralRequirement extraction:",
                f"- {metric_text}",
                f"- metrics json: {report_path.relative_to(out_dir)}",
            ],
            raw_text,
        )
    order = _model_order(models)
    rows = sorted(rows, key=lambda row: order[row["model"]])
    fields = [
        "model",
        "model_name",
        "action_kind",
        "dt",
        "status",
        "method",
        "learned_params",
        "recurrent_state",
        "episodes",
        "success_rate",
        "safety_violation_rate",
        "override_rate",
        "pointwise_agreement",
        "source",
    ]
    write_csv(stage_dir / "summary.csv", rows, fields)
    write_json(stage_dir / "summary.json", {
        "settings": {"dt": dt},
        "rule_runs": rule_runs,
        "rows": rows,
    })
    return {"rule_runs": rule_runs, "rows": rows, "fields": fields}


def stage_weak(
    out_dir: Path,
    models: list[SysMLInput],
) -> dict[str, Any]:
    stage_dir = out_dir / "02_memoryless"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "02_memoryless_check.txt"
    rows = []
    log_lines = []
    for model in models:
        requirement = SpecShield(str(model.path))
        references = _collect_refs(requirement.req_ast)
        allowed = (
            set(requirement.in_params)
            | set(requirement.out_params)
            | set(requirement.unchanging)
            | {requirement.subject_var}
        )
        unresolved = sorted(references - allowed)
        if unresolved:
            raise RuntimeError(
                f"{model.key} requirement references values outside its current "
                f"neural inputs and outputs: {unresolved}"
            )
        row = {
            "model": model.key,
            "model_name": model.name,
            "stage": "memoryless",
            "b_obs": 0,
            "b_act": 0,
            "claim": "current neural inputs determine the current controller constraint",
            "source": str(model.path),
            "model_sha256": model.sha256,
            "command_log": str(log_path.relative_to(out_dir)),
        }
        rows.append(row)
        log_lines.append(
            f"- {model.key}: current inputs={list(requirement.in_params)}, "
            f"outputs={list(requirement.out_params)}"
        )
    log_path.write_text(
        "SysML current-decision checks:\n" + "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    run = {
        "command": "read #Neural action and #NeuralRequirement from each supplied SysML file",
        "returncode": 0,
        "log": str(log_path.relative_to(out_dir)),
    }
    fields = [
        "model", "model_name", "stage", "b_obs", "b_act", "claim",
        "source", "model_sha256", "command_log",
    ]
    write_csv(stage_dir / "summary.csv", rows, fields)
    write_json(stage_dir / "summary.json", {"run": run, "rows": rows})
    return {"run": run, "rows": rows, "fields": fields}


def stage_strict(
    out_dir: Path,
    py: str,
    models: list[SysMLInput],
    dt: float,
) -> dict[str, Any]:
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
        "--dt",
        str(dt),
        *[str(model.path) for model in models],
    ]
    closure_run, closure_text = run_command(cmd, closure_log, out_dir)
    closure_rows = parse_closure_text(closure_text, "markov_mdp", models)
    for row in closure_rows:
        row["dt"] = dt
    if closure_run["returncode"] != 0 or len(closure_rows) != len(models):
        raise RuntimeError(
            "Markov/MDP buffer extraction did not produce one result for every "
            f"SysML file; see {closure_log}"
        )
    write_closure_log(closure_log, closure_run, closure_rows, closure_text)

    generation_json = stage_dir / "markov_mdp_generation.json"
    generation_cmd = [
        py,
        str(ARTIFACT / "src" / "clarity" / "pipeline" / "markov_mdp.py"),
        "--out-json",
        str(generation_json),
        "--artifact-dir",
        str(stage_dir),
        "--max-obs",
        "2",
        "--max-act",
        "4",
        "--dt",
        str(dt),
        *[str(model.path) for model in models],
    ]
    generation_run, generation_text = run_command(
        generation_cmd, generation_log, out_dir
    )
    if generation_run["returncode"] != 0 or not generation_json.exists():
        raise RuntimeError(
            f"Markov/MDP proof generation failed; see {generation_log}"
        )
    generation_summary = (
        read_json(generation_json) if generation_json.exists() else {"rows": []}
    )
    generation_dt = validate_dt(generation_summary.get("settings", {}).get("dt"))
    if generation_dt != dt:
        raise RuntimeError(
            f"Markov/MDP generation dt does not match run dt: {generation_dt} != {dt}"
        )
    rows = generation_summary.get("rows", [])
    if len(rows) != len(models):
        raise RuntimeError(
            "Markov/MDP proof generation did not produce one result for every SysML file"
        )
    order = _model_order(models)
    for row in rows:
        row_dt = validate_dt(row.get("dt"))
        if row_dt != dt:
            raise RuntimeError(
                f"{row.get('model')} Markov/MDP row dt does not match run dt: "
                f"{row_dt} != {dt}"
            )
    rows = sorted(rows, key=lambda row: order[row["model"]])
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
        "dt",
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
        "settings": {"dt": dt},
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


def _spec_path(out_dir: Path, name: str) -> Path:
    return out_dir / "03_markov_mdp" / "reduced_mdp_specs" / f"{name}.reduced_mdp_spec.json"


def stage_discretization(
    out_dir: Path,
    py: str,
    models: list[SysMLInput],
    dt: float,
    *,
    dt_text: str,
    optimization_timeout_ms: int,
    smt_timeout_ms: int,
) -> dict[str, Any]:
    stage_dir = out_dir / "04_discretization_safety"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "04_discretization_safety.txt"
    summary_path = stage_dir / "generation.json"
    cmd = [
        py,
        str(ARTIFACT / "src" / "clarity" / "pipeline" / "discretization_safety.py"),
        "--out-json",
        str(summary_path),
        "--artifact-dir",
        str(stage_dir),
        "--mdp-dir",
        str(out_dir / "03_markov_mdp"),
        "--dt",
        dt_text,
        "--optimization-timeout-ms",
        str(optimization_timeout_ms),
        "--smt-timeout-ms",
        str(smt_timeout_ms),
        *[str(model.path) for model in models],
    ]
    run, raw_text = run_command(cmd, log_path, out_dir)
    generated = read_json(summary_path) if summary_path.exists() else {"rows": []}
    rows = generated.get("rows", [])
    write_command_summary_log(
        log_path,
        run,
        [
            "discretization-safety certificate generation:",
            *[
                f"- {row['model']}: result={row['result']}, "
                f"checker={row['checker']}, properties="
                f"{row['properties_certified']}/{row['properties_checked']}, "
                f"seconds={float(row['elapsed_seconds']):.6f}"
                for row in rows
            ],
            "- every certificate is replayed by the independent checker",
        ],
        raw_text,
    )
    if run["returncode"] != 0 or not summary_path.exists():
        raise RuntimeError(
            f"discretization-safety certification failed; see {log_path}"
        )
    generated_dt = validate_dt(generated.get("settings", {}).get("dt"))
    if generated_dt != dt:
        raise RuntimeError(
            "discretization-safety dt does not match run dt: "
            f"{generated_dt} != {dt}"
        )
    if len(rows) != len(models):
        raise RuntimeError(
            "discretization-safety generation did not produce one result for every model"
        )
    if any(
        row.get("result") != "CERTIFIED" or row.get("checker") != "passed"
        for row in rows
    ):
        raise RuntimeError(
            f"one or more discretization-safety certificates failed; see {log_path}"
        )
    order = _model_order(models)
    rows = sorted(rows, key=lambda row: order[row["model"]])
    fields = [
        "model",
        "dt",
        "result",
        "checker",
        "properties_checked",
        "properties_certified",
        "elapsed_seconds",
        "certificate_path",
    ]
    write_csv(stage_dir / "summary.csv", rows, fields)
    write_json(stage_dir / "summary.json", {
        "settings": generated.get("settings", {}),
        "run": run,
        "rows": rows,
        "result": generated.get("result"),
    })
    return {"run": run, "rows": rows, "fields": fields}


def _feedforward_architecture_from_spec(
    spec_path: Path,
    *,
    dt: float,
) -> dict[str, Any]:
    spec = read_json(spec_path)
    spec_dt = validate_dt(spec.get("model", {}).get("dt"))
    if spec_dt != dt:
        raise RuntimeError(
            f"reduced-MDP spec dt does not match run dt: {spec_dt} != {dt}: "
            f"{spec_path}"
        )
    architecture = spec.get("feedforward_architecture")
    if not isinstance(architecture, dict):
        raise RuntimeError(
            f"reduced-MDP spec lacks a feedforward architecture: {spec_path}"
        )
    hidden_dim = architecture.get("hidden_dim")
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise RuntimeError(
            f"reduced-MDP spec has an invalid hidden size: {spec_path}"
        )
    return architecture


def _read_training_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    test = data.get("test", {})
    selected = data.get("selected_checkpoint", {})
    buffer = data.get("certified_buffer", {})
    return {
        "dt": data.get("dt", data.get("config", {}).get("dt", "")),
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
            str(job["model_path"]),
            "--out-dir",
            str(run_dir),
            "--seed",
            str(job["seed"]),
            "--max-obs",
            "2",
            "--max-act",
            "4",
            "--dt",
            str(job["dt"]),
            "--reduced-mdp-spec",
            str(job["spec_path"]),
            "--collection-backend",
            str(job["collection_backend"]),
            "--collection-workers",
            str(job["collection_workers"]),
            "--collection-start-method",
            str(job["collection_start_method"]),
            *job.get("overrides", []),
        ]
        weight_path = run_dir / "best.npz"
    else:
        summary_json = run_dir / "summary.json"
        cmd = [
            py,
            "train_mlp_buffer.py",
            str(job["model_path"]),
            "--save-dir",
            str(run_dir),
            "--summary-json",
            str(summary_json),
            "--reduced-mdp-spec",
            str(job["spec_path"]),
            "--seed",
            str(job["seed"]),
            "--dt",
            str(job["dt"]),
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
    write_command_summary_log(
        log_path,
        run,
        [
            f"{name} reduced training:",
            f"- dt={job['dt']}",
            f"- elapsed_seconds={seconds:.6f}",
        ],
        raw_text,
    )
    summary_json = run_dir / "summary.json"
    if run["returncode"] != 0 or not summary_json.exists():
        return {
            "model": name,
            "kind": kind,
            "status": "failed",
            "dt": job["dt"],
            "hidden_dim": hidden_dim,
            "seconds": seconds,
            "command_log": str(log_path.relative_to(out_dir)),
            "summary_json": "",
            "weights": "",
            "returncode": run["returncode"],
        }
    row = _read_training_summary(summary_json)
    trained_dt = validate_dt(row.get("dt"))
    if trained_dt != job["dt"]:
        raise RuntimeError(
            f"{name} training result dt does not match run dt: "
            f"{trained_dt} != {job['dt']}"
        )
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


def _select_models(
    rows: list[dict[str, Any]],
    *,
    override_tolerance: float,
    models: list[SysMLInput],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    order = _model_order(models)
    for model in sorted({r["model"] for r in rows}, key=order.__getitem__):
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
            "selection_method": "sysml_derived_single_architecture",
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


def stage_training(out_dir: Path, py: str, models: list[SysMLInput], *, dt: float,
                   smoke: bool = False,
                   jobs: int | None = None,
                   override_tolerance: float = 0.01,
                   collection_backend: str = "auto",
                   collection_workers_per_job: int | None = None,
                   collection_start_method: str = "spawn") -> dict[str, Any]:
    stage_dir = out_dir / "05_reduced_training"
    log_dir = out_dir / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    discrete_models = [model for model in models if model.action_kind == "discrete"]
    continuous_models = [model for model in models if model.action_kind == "continuous"]
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

    cpu_count = max(1, os.cpu_count() or 1)
    if collection_workers_per_job is None:
        collection_workers_per_job = max(
            1, min(8, cpu_count // max(job_count, 1))
        )
    else:
        collection_workers_per_job = max(1, collection_workers_per_job)
    if collection_backend == "serial":
        collection_workers_per_job = 1

    runs: list[dict[str, Any]] = []

    training_jobs: list[dict[str, Any]] = []
    architectures: dict[str, dict[str, Any]] = {}
    for model in discrete_models:
        name = model.key
        spec_path = _spec_path(out_dir, name)
        if not spec_path.exists():
            raise RuntimeError(
                f"current run did not generate a reduced-MDP spec for {name}"
            )
        architecture = _feedforward_architecture_from_spec(spec_path, dt=dt)
        architectures[name] = architecture
        hidden_dim = int(architecture["hidden_dim"])
        run_dir = stage_dir / "runs" / name / f"h{hidden_dim}" / "seed_0"
        training_jobs.append({
            "model": name,
            "model_path": model.path,
            "kind": "discrete",
            "dt": dt,
            "hidden_dim": hidden_dim,
            "seed": 0,
            "run_dir": run_dir,
            "log_path": log_dir / f"05_train_{name}_h{hidden_dim}.txt",
            "spec_path": spec_path,
            "overrides": discrete_overrides,
            "collection_backend": collection_backend,
            "collection_workers": collection_workers_per_job,
            "collection_start_method": collection_start_method,
        })

    for model in continuous_models:
        name = model.key
        spec_path = _spec_path(out_dir, name)
        if not spec_path.exists():
            raise RuntimeError(
                f"current run did not generate a reduced-MDP spec for {name}"
            )
        architecture = _feedforward_architecture_from_spec(spec_path, dt=dt)
        architectures[name] = architecture
        hidden_dim = int(architecture["hidden_dim"])
        run_dir = stage_dir / "runs" / name / f"h{hidden_dim}" / "seed_0"
        training_jobs.append({
            "model": name,
            "model_path": model.path,
            "kind": "continuous",
            "dt": dt,
            "hidden_dim": hidden_dim,
            "seed": 0,
            "run_dir": run_dir,
            "log_path": log_dir / f"05_train_{name}_h{hidden_dim}.txt",
            "spec_path": spec_path,
            "continuous_episodes": continuous_settings["episodes"],
            "continuous_episodes_per_update": continuous_settings["episodes_per_update"],
            "continuous_eval_interval": continuous_settings["eval_interval"],
            "continuous_eval_episodes": continuous_settings["eval_episodes"],
            "continuous_max_steps": continuous_settings["max_steps"],
        })

    manifest = {
        "dt": dt,
        "mode": "smoke" if smoke else "full",
        "architecture_selection": "SysML-derived; one architecture per model",
        "architectures": architectures,
        "seed": 0,
        "jobs": job_count,
        "collection_backend": collection_backend,
        "collection_workers_per_discrete_job": collection_workers_per_job,
        "collection_start_method": collection_start_method,
        "discrete_training_defaults": (
            "train_one_seed defaults: 2000 PPO episodes, 100 episodes per update, "
            "100-episode eval/checkpoint interval, 100 eval episodes per checkpoint, "
            "200 final test episodes, 2000 oracle samples, 100 oracle epochs"
            if not smoke else "smoke overrides"
        ),
        "continuous_training_settings": continuous_settings,
        "selection_rule": (
            "each model uses the sole SysML-derived architecture; a trained model is "
            "selected only from a zero-violation checkpoint, using test success and "
            "then override rate"
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
                    "model", "kind", "status", "dt", "hidden_dim", "params",
                    "b_obs", "b_act", "device", "shield", "policy",
                    "train_seconds", "peak_rss_mb", "test_safety",
                    "test_success", "test_override", "summary_json",
                    "weights", "command_log",
                ])

    order = _model_order(models)
    rows = sorted(rows, key=lambda row: order[row["model"]])
    failed = [row for row in rows if row.get("status") != "trained"]
    if failed:
        details = ", ".join(
            f"{row['model']} ({row.get('command_log', 'no log')})"
            for row in failed
        )
        raise RuntimeError(f"reduced training failed: {details}")
    selected_rows = _select_models(
        rows, override_tolerance=override_tolerance, models=models
    )
    fields = [
        "model",
        "kind",
        "status",
        "dt",
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
        "model", "selection_status", "selection_method", "hidden_dim", "params",
        "test_safety", "test_success", "test_override", "best_success",
        "best_override_at_best_success", "override_tolerance",
        "summary_json", "weights",
    ]
    write_csv(stage_dir / "selected_models.csv", selected_rows, selected_fields)
    write_json(stage_dir / "summary.json", {
        "manifest": manifest,
        "runs": runs,
        "rows": rows,
        "selected_models": selected_rows,
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
    lines.append("      -> discretization safety certification")
    if "reduced_training" in summary["stages"]:
        lines.append("        -> reduced feedforward training")
    lines.append("```")
    lines.append("")
    lines.append("The run overwrites generated outputs and rebuilds each selected stage")
    lines.append("from the SysML files supplied to this run.")
    lines.append("")

    affine = summary["stages"]["affine_rule"]
    lines.append("## 1. Affine/Rule Fit")
    lines.append("")
    lines.append("This stage reads each #NeuralRequirement directly from its SysML file.")
    lines.append("A Boolean requirement is evaluated only when it determines one action.")
    lines.append("")
    lines.append(markdown_table(affine["rows"], affine["fields"]))
    lines.append("")
    lines.append("Every row in this stage is generated fresh by this artifact.")
    lines.append("")
    lines.append("No learned parameters are used when the requirement itself gives the action.")
    lines.append("")

    weak = summary["stages"]["memoryless"]
    lines.append("## 2. Memoryless Controller Check")
    lines.append("")
    lines.append("Claim: the current controller inputs are enough for the current decision.")
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
    lines.append("determined. This stage generates the proof obligations from the supplied")
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

    discretization = summary["stages"]["discretization_safety"]
    lines.append("## 4. Discretization Safety Certification")
    lines.append("")
    lines.append("Claim: the checked controller contract implies every extracted safety")
    lines.append("property throughout each interval between controller updates.")
    lines.append("The stage consumes the Stage 3 certificate and the run-wide dt value.")
    lines.append("")
    lines.append(markdown_table(discretization["rows"], discretization["fields"]))
    lines.append("")
    lines.append(f"Certificate log: `{discretization['run']['log']}`")
    lines.append("")

    training = summary["stages"].get("reduced_training")
    if training is not None:
        lines.append("## 5. Reduced Feedforward Training")
        lines.append("")
        lines.append("Claim: the run-local certified specs can be consumed by non-recurrent")
        lines.append("feedforward trainers. For each model, the SysML requirement determines")
        lines.append("one hidden size from its comparison boundaries and outputs.")
        lines.append("Discrete models use the handmade NumPy PPO stack.")
        lines.append("Real-valued actions use the bundled continuous MLP PPO stack.")
        lines.append("All training runs on CPU and checks actions against the requirement")
        lines.append("read from the current SysML file.")
        lines.append("")
        lines.append("Selection rule: train and evaluate the sole SysML-derived size.")
        lines.append("")
        lines.append("### Selected Models")
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
    lines.append("- Stage 4 certifies safety throughout each discretized interval.")
    if training is not None:
        lines.append("- Stage 5 trains small feedforward policies from the Stage 3 specs.")
        lines.append("")
        lines.append("Training and evaluation check each action against the requirement")
        lines.append("read from the current SysML file.")
    lines.append("")
    (out_dir / "fitting_sequence_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="SysML file paths")
    parser.add_argument(
        "--models-root",
        default=str(REPO / "sysml-models"),
        help="directory searched recursively when no file paths are supplied",
    )
    parser.add_argument("--out-dir", default=str(ARTIFACT / "outputs" / "latest"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--dt",
        default=str(DEFAULT_DT),
        help=f"simulation time step used by every applicable stage; default={DEFAULT_DT}",
    )
    parser.add_argument("--smoke-training", action="store_true",
                        help="run a short Stage 5 wiring test instead of full training")
    parser.add_argument("--training-jobs", type=int, default=None,
                        help="parallel Stage 5 training jobs; default=min(6,cpu_count)")
    parser.add_argument(
        "--preprocessing-only",
        action="store_true",
        help="run Stages 1 through 4 and stop before training",
    )
    parser.add_argument(
        "--discretization-optimization-timeout-ms",
        type=int,
        default=250,
        help="solver timeout for the linear and convex checkers in Stage 4",
    )
    parser.add_argument(
        "--discretization-smt-timeout-ms",
        type=int,
        default=30000,
        help="solver timeout for the final fallback in Stage 4",
    )
    parser.add_argument(
        "--collection-backend",
        choices=("auto", "serial", "process"),
        default="auto",
        help="discrete episode collection backend",
    )
    parser.add_argument(
        "--collection-workers-per-job",
        type=int,
        default=None,
        help="simulator workers per discrete training job; default uses the CPU budget",
    )
    parser.add_argument(
        "--collection-start-method",
        default="spawn",
        help="multiprocessing start method for discrete episode workers",
    )
    parser.add_argument("--override-tolerance", type=float, default=0.01,
                        help="absolute override slack for selecting a smaller max-accuracy model")
    args = parser.parse_args()
    dt_text = str(args.dt)
    args.dt = validate_dt(args.dt)
    if args.discretization_smt_timeout_ms <= 0:
        parser.error("--discretization-smt-timeout-ms must be positive")
    if args.discretization_optimization_timeout_ms <= 0:
        parser.error("--discretization-optimization-timeout-ms must be positive")
    models = discover_sysml(args.models, models_root=args.models_root)

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture_fit_root": str(ARCH),
        "python_bin": args.python_bin,
        "settings": {"dt": args.dt, "dt_input": dt_text},
        "sysml_inputs": [model.to_dict() for model in models],
        "stages": {},
    }
    write_json(out_dir / "sysml_inputs.json", summary["sysml_inputs"])
    summary["stages"]["affine_rule"] = stage_affine(
        out_dir, args.python_bin, models, args.dt
    )
    summary["stages"]["memoryless"] = stage_weak(
        out_dir, models
    )
    summary["stages"]["markov_mdp"] = stage_strict(
        out_dir, args.python_bin, models, args.dt
    )
    summary["stages"]["discretization_safety"] = stage_discretization(
        out_dir,
        args.python_bin,
        models,
        args.dt,
        dt_text=dt_text,
        optimization_timeout_ms=args.discretization_optimization_timeout_ms,
        smt_timeout_ms=args.discretization_smt_timeout_ms,
    )
    if not args.preprocessing_only:
        summary["stages"]["reduced_training"] = stage_training(
            out_dir,
            args.python_bin,
            models,
            dt=args.dt,
            smoke=args.smoke_training,
            jobs=args.training_jobs,
            override_tolerance=args.override_tolerance,
            collection_backend=args.collection_backend,
            collection_workers_per_job=args.collection_workers_per_job,
            collection_start_method=args.collection_start_method,
        )

    write_json(out_dir / "fitting_sequence_summary.json", summary)
    write_report(out_dir, summary)
    print(f"WROTE {out_dir / 'fitting_sequence_report.md'}")
    print(f"WROTE {out_dir / 'fitting_sequence_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
