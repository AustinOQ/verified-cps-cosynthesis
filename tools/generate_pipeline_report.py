#!/usr/bin/env python3
"""Generate a human-readable pipeline report from generated metrics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(row: dict, key: str):
    value = str(row.get(key, "")).strip().lower()
    if value in ("true", "1", "yes"):
        return 1.0
    if value in ("false", "0", "no"):
        return 0.0
    return None


def _ci95(values: list[float]) -> tuple[float | None, float, float, int]:
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None, 0.0, 0.0, 0
    if n == 1:
        return vals[0], 0.0, 0.0, 1
    m = mean(vals)
    sd = stdev(vals)
    half = 1.96 * sd / math.sqrt(n)
    return m, sd, half, n


def _fmt(values: list[float], digits: int = 4) -> str:
    m, _sd, half, n = _ci95(values)
    if m is None:
        return "-"
    return f"{m:.{digits}f} +/- {half:.{digits}f} (n={n})"


def _fmt_raw(value: str, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value or "-"


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        system = row.get("system")
        if system and system != "TOTAL":
            grouped[system].append(row)
    return dict(grouped)


def _num_list(rows: list[dict], key: str) -> list[float]:
    vals = []
    for row in rows:
        value = _float(row, key)
        if value is not None:
            vals.append(value)
    return vals


def _bool_list(rows: list[dict], key: str) -> list[float]:
    vals = []
    for row in rows:
        value = _bool(row, key)
        if value is not None:
            vals.append(value)
    return vals


def _runtime_steps(rows: list[dict]) -> list[float]:
    vals = []
    for row in rows:
        total = _float(row, "total_steps")
        episodes = _float(row, "episodes")
        if total is not None and episodes:
            vals.append(total / episodes)
    return vals


def _seeds(rows: list[dict]) -> str:
    seeds = sorted({str(row.get("seed", "")) for row in rows
                    if row.get("seed", "") != ""}, key=lambda s: int(s))
    return ", ".join(seeds) if seeds else "-"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--cpu-mode", default="")
    parser.add_argument("--cpu-affinity-core", default="")
    args = parser.parse_args(argv)

    metrics_dir = Path(args.metrics_dir)
    out_path = Path(args.out) if args.out else metrics_dir / "pipeline_report.md"

    verification = _read_csv(metrics_dir / "evaluation_summary.csv")
    training = _read_csv(metrics_dir / "cpu_training_summary.csv")
    runtime = _read_csv(metrics_dir / "runtime_results" /
                        "runtime_monitor_summary.csv")
    train_by_system = _group(training)
    runtime_by_system = _group(runtime)

    model_order = [m for m in args.models.split(",") if m]
    if not model_order:
        model_order = []
        for row in verification:
            system = row.get("system")
            if system and system != "TOTAL":
                model_order.append(system)
        for source in (train_by_system, runtime_by_system):
            for system in source:
                if system not in model_order:
                    model_order.append(system)

    all_seed_rows = training or runtime
    seed_text = args.seeds or _seeds(all_seed_rows)
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    lines = [
        "# Pipeline Report",
        "",
        f"Generated: {now}",
        f"Models: {', '.join(model_order) if model_order else '-'}",
        f"Seeds: {seed_text}",
        f"CPU execution mode: {args.cpu_mode or '-'}"
        + (f" (core {args.cpu_affinity_core})"
           if args.cpu_mode == "single" and args.cpu_affinity_core else ""),
        "",
        "Confidence intervals are 95% CI half-widths computed across the "
        "seed rows generated for this pipeline call. For n=1, the CI "
        "half-width is reported as 0.",
        "",
        "## SMV Verification",
        "",
        "| system | IC3 proven | IC3 failed | IC3 runtime ms | IC3 peak MB | "
        "BMC proven | BMC failed | BMC runtime ms | BMC peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    ver_by_system = {row.get("system"): row for row in verification}
    for system in model_order:
        row = ver_by_system.get(system, {})
        lines.append(
            f"| {system} | {row.get('ic3_proven', '-')} | "
            f"{row.get('ic3_failed', '-')} | "
            f"{_fmt_raw(row.get('ic3_runtime_ms', ''), 0)} | "
            f"{_fmt_raw(row.get('ic3_peak_mb', ''), 1)} | "
            f"{row.get('bmc_proven', '-')} | "
            f"{row.get('bmc_failed', '-')} | "
            f"{_fmt_raw(row.get('bmc_runtime_ms', ''), 0)} | "
            f"{_fmt_raw(row.get('bmc_peak_mb', ''), 1)} |")

    total = ver_by_system.get("TOTAL")
    if total:
        lines.append(
            f"| TOTAL | {total.get('ic3_proven', '-')} | "
            f"{total.get('ic3_failed', '-')} | - | - | "
            f"{total.get('bmc_proven', '-')} | "
            f"{total.get('bmc_failed', '-')} | - | - |")

    lines.extend([
        "",
        "## Trained Model Results",
        "",
        "| system | seeds | selected safe rate | final test success | "
        "final test safety violation rate | final test override rate | "
        "avg steps to completion | train time s | train peak RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for system in model_order:
        rows = train_by_system.get(system, [])
        lines.append(
            f"| {system} | {_seeds(rows)} | "
            f"{_fmt(_bool_list(rows, 'final_checkpoint_safe'), 4)} | "
            f"{_fmt(_num_list(rows, 'test_success_rate'), 4)} | "
            f"{_fmt(_num_list(rows, 'test_safety_violation_rate'), 4)} | "
            f"{_fmt(_num_list(rows, 'test_override_rate'), 5)} | "
            f"{_fmt(_num_list(rows, 'test_mean_episode_steps'), 2)} | "
            f"{_fmt(_num_list(rows, 'train_seconds'), 1)} | "
            f"{_fmt(_num_list(rows, 'training_peak_rss_mb'), 1)} |")

    lines.extend([
        "",
        "## Runtime Verification",
        "",
        "| system | seeds | runtime success | runtime safety violation rate | "
        "runtime override rate | avg steps to completion | inference peak "
        "RSS MB | policy us mean | shield us mean | total us mean | "
        "test wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for system in model_order:
        rows = runtime_by_system.get(system, [])
        lines.append(
            f"| {system} | {_seeds(rows)} | "
            f"{_fmt(_num_list(rows, 'success_rate'), 4)} | "
            f"{_fmt(_num_list(rows, 'safety_violation_rate'), 4)} | "
            f"{_fmt(_num_list(rows, 'override_rate'), 5)} | "
            f"{_fmt(_runtime_steps(rows), 2)} | "
            f"{_fmt(_num_list(rows, 'inference_peak_rss_mb'), 1)} | "
            f"{_fmt(_num_list(rows, 'policy_us_mean'), 2)} | "
            f"{_fmt(_num_list(rows, 'shield_us_mean'), 2)} | "
            f"{_fmt(_num_list(rows, 'total_us_mean'), 2)} | "
            f"{_fmt(_num_list(rows, 'test_seconds'), 2)} |")

    lines.extend([
        "",
        "## Runtime Latency Details",
        "",
        "| system | seeds | policy us p95 | policy us p99 | shield us p95 | "
        "shield us p99 | total us p95 | total us p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for system in model_order:
        rows = runtime_by_system.get(system, [])
        lines.append(
            f"| {system} | {_seeds(rows)} | "
            f"{_fmt(_num_list(rows, 'policy_us_p95'), 2)} | "
            f"{_fmt(_num_list(rows, 'policy_us_p99'), 2)} | "
            f"{_fmt(_num_list(rows, 'shield_us_p95'), 2)} | "
            f"{_fmt(_num_list(rows, 'shield_us_p99'), 2)} | "
            f"{_fmt(_num_list(rows, 'total_us_p95'), 2)} | "
            f"{_fmt(_num_list(rows, 'total_us_p99'), 2)} |")

    lines.extend([
        "",
        "## Selected Checkpoints",
        "",
        "| system | seed | source | selected episode | selected success | "
        "selected override | selected safety violation | final safe |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ])
    for system in model_order:
        for row in train_by_system.get(system, []):
            lines.append(
                f"| {system} | {row.get('seed', '-')} | "
                f"{row.get('selected_checkpoint_source', '-')} | "
                f"{row.get('selected_episode', '-')} | "
                f"{_fmt_raw(row.get('selected_success_rate', ''), 4)} | "
                f"{_fmt_raw(row.get('selected_override_rate', ''), 5)} | "
                f"{_fmt_raw(row.get('selected_safety_violation_rate', ''), 5)} | "
                f"{row.get('final_checkpoint_safe', '-')} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Pipeline report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
