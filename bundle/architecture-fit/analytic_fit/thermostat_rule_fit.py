#!/usr/bin/env python3
"""Analytic sidecar for the thermostat model.

This does not train a neural net. It validates the direct rule induced by the
thermostat NeuralRequirement:

  heater iff setPoint >= temperatureCelcius + toleranceCelcius
  ac     iff setPoint <= temperatureCelcius - toleranceCelcius
  else both off

The rule is checked pointwise against the existing program-based SpecShield
oracle and then run closed-loop through the existing SysML simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cruise_rule_fit import (
    ARCH,
    REPO,
    _blank_metrics,
    _finish_rates,
    _find_action,
    _obs_from_inputs,
    _peak_rss_mb,
    _safety_violation,
)

from env import SysMLEnv
from oracle import extract_interface, spec_oracle


MODEL = REPO / "sysml-models" / "thermostat" / "model.sysml"


@dataclass(frozen=True)
class ThermostatRule:
    tolerance_c: float
    off_action: int
    heat_action: int
    cool_action: int

    @classmethod
    def from_spec(cls, spec: Any) -> "ThermostatRule":
        return cls(
            tolerance_c=float(spec.unchanging["toleranceCelcius"]),
            off_action=_find_action(spec.action_map, heaterState=False, acState=False),
            heat_action=_find_action(spec.action_map, heaterState=True, acState=False),
            cool_action=_find_action(spec.action_map, heaterState=False, acState=True),
        )

    def primitive_predicates(self, obs: dict[str, float]) -> dict[str, bool]:
        set_point = float(obs["setPoint"])
        temp = float(obs["temperatureCelcius"])
        return {
            "below_or_at_heat_threshold": set_point - temp - self.tolerance_c >= 0.0,
            "above_or_at_cool_threshold": temp - set_point - self.tolerance_c >= 0.0,
        }

    def action(self, obs: dict[str, float]) -> int:
        pred = self.primitive_predicates(obs)
        if pred["below_or_at_heat_threshold"]:
            return self.heat_action
        if pred["above_or_at_cool_threshold"]:
            return self.cool_action
        return self.off_action


def _merge_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _blank_metrics(rows[0]["model"])
    for row in rows:
        for key, value in row.items():
            if key == "model":
                continue
            if key == "peak_rss_mb":
                merged[key] = max(float(merged[key]), float(value))
            else:
                merged[key] += value
    _finish_rates(merged)
    return merged


def _eval_worker(model_path: str, episodes: int, seed: int,
                 dt: float, max_steps: int) -> dict[str, Any]:
    start = time.time()
    iface = extract_interface(model_path, dt=dt)
    spec = iface["spec_shield"]
    obs_names = list(iface["obs_names"])
    rule = ThermostatRule.from_spec(spec)
    env = SysMLEnv(model_path, dt=dt, max_steps=max_steps, phase=2, rng_seed=seed)
    metrics = _blank_metrics("thermostat")
    try:
        for _episode in range(episodes):
            env.reset()
            done = False
            last_reward = 0.0
            last_info: dict[str, Any] = {}
            while not done:
                obs = _obs_from_inputs(env._twin._model_inputs, obs_names)
                proposed = rule.action(obs)  # type: ignore[arg-type]
                expected = spec_oracle(spec, obs)
                if proposed != expected:
                    metrics["pointwise_failures"] += 1
                metrics["pointwise_checks"] += 1

                final_action = spec(proposed, obs)
                if final_action != proposed:
                    metrics["overrides"] += 1
                _next_obs, reward, done, info = env.step(final_action)
                metrics["steps"] += 1
                last_reward = float(reward)
                last_info = info

            metrics["episodes"] += 1
            if last_reward > 0:
                metrics["successes"] += 1
            elif last_reward < 0 or _safety_violation(last_info):
                metrics["safety_violations"] += 1
            else:
                metrics["truncations"] += 1
    finally:
        env.close()
    metrics["seconds"] = time.time() - start
    metrics["peak_rss_mb"] = _peak_rss_mb()
    return metrics


def _split_work(total: int, jobs: int, seed: int) -> list[tuple[int, int]]:
    jobs = max(1, min(jobs, total))
    base = total // jobs
    rem = total % jobs
    chunks = []
    for i in range(jobs):
        n = base + (1 if i < rem else 0)
        if n:
            chunks.append((n, seed + 100_000 * i))
    return chunks


def evaluate(episodes: int, jobs: int, seed: int, dt: float,
             max_steps: int) -> dict[str, Any]:
    chunks = _split_work(episodes, jobs, seed)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(_eval_worker, str(MODEL), n, chunk_seed, dt, max_steps)
            for n, chunk_seed in chunks
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return _merge_metrics(rows)


def _boundary_observations(tolerance_c: float) -> list[dict[str, float | bool]]:
    eps = 1e-6
    observations: list[dict[str, float | bool]] = []
    for set_point in [13.0, 18.3, 25.0, 33.0]:
        for offset in [
            -3.0,
            -tolerance_c - eps,
            -tolerance_c,
            -tolerance_c + eps,
            0.0,
            tolerance_c - eps,
            tolerance_c,
            tolerance_c + eps,
            3.0,
        ]:
            observations.append({
                "setPoint": set_point,
                "temperatureCelcius": set_point + offset,
                "done": False,
            })
    return observations


def boundary_stress() -> dict[str, Any]:
    iface = extract_interface(str(MODEL))
    spec = iface["spec_shield"]
    rule = ThermostatRule.from_spec(spec)
    cases = _boundary_observations(rule.tolerance_c)
    failures = []
    counts: dict[str, int] = {}
    for obs in cases:
        proposed = rule.action(obs)  # type: ignore[arg-type]
        expected = spec_oracle(spec, obs)
        counts[str(proposed)] = counts.get(str(proposed), 0) + 1
        if proposed != expected:
            failures.append({
                "obs": obs,
                "proposed": proposed,
                "expected": expected,
            })
    return {
        "thermostat": {
            "cases": len(cases),
            "failures": len(failures),
            "agreement": 1.0 - len(failures) / max(len(cases), 1),
            "predicted_action_counts": counts,
            "failure_examples": failures[:5],
        }
    }


def architecture_notes() -> dict[str, Any]:
    return {
        "learned_parameters": 0,
        "uses_recurrent_state": False,
        "uses_certified_buffer_at_runtime": False,
        "primitive_affine_predicates": [
            "setPoint - temperatureCelcius - toleranceCelcius >= 0",
            "temperatureCelcius - setPoint - toleranceCelcius >= 0",
        ],
        "rule_layer": [
            "heater iff predicate_0",
            "AC iff predicate_1",
            "otherwise both off",
        ],
        "single_raw_affine_classifier_is_exact": True,
        "single_raw_affine_classifier_note": (
            "The two on-actions are opposite halfspaces around a deadband. "
            "A multiclass affine score can represent this simpler thermostat "
            "partition exactly, but the direct rule is clearer and has no "
            "learned parameters."
        ),
    }


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = report["validation"]["thermostat"]
    rows.append({
        "model": "thermostat",
        "method": "analytic rule sidecar",
        "learned_params": 0,
        "training_seconds": 0.0,
        "validation_seconds": round(float(metrics["seconds"]), 3),
        "success_rate": round(float(metrics["success_rate"]), 6),
        "safety_violation_rate": round(float(metrics["safety_violation_rate"]), 6),
        "override_rate": round(float(metrics["override_rate"]), 6),
        "mean_steps": round(float(metrics["mean_episode_steps"]), 3),
        "pointwise_agreement": round(float(metrics["pointwise_agreement"]), 6),
        "peak_rss_mb": round(float(metrics["peak_rss_mb"]), 3),
    })
    return rows


def write_markdown(report: dict[str, Any], path: Path) -> None:
    headers = [
        "model", "method", "learned_params", "training_seconds",
        "success_rate", "safety_violation_rate", "override_rate",
        "mean_steps", "pointwise_agreement", "peak_rss_mb",
    ]
    lines = [
        "# Analytic Thermostat Fit Report",
        "",
        "This sidecar validates a zero-learned-parameter thermostat rule extracted from the NeuralRequirement.",
        "",
        "## Architecture",
        "",
        f"- learned parameters: {report['architecture']['learned_parameters']}",
        f"- recurrent state: {report['architecture']['uses_recurrent_state']}",
        f"- primitive affine predicates: {len(report['architecture']['primitive_affine_predicates'])}",
        f"- exact raw affine classifier: {report['architecture']['single_raw_affine_classifier_is_exact']}",
        "",
        "## Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in report["comparison_rows"]:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.extend([
        "",
        "## Boundary Stress",
        "",
    ])
    for name, stress in report["boundary_stress"].items():
        lines.append(
            f"- {name}: {stress['cases']} cases, {stress['failures']} failures, "
            f"agreement={stress['agreement']:.6f}")
    lines.extend([
        "",
        "## Validation Notes",
        "",
        "- Pointwise agreement means the sidecar action equaled `spec_oracle(SpecShield, obs)`.",
        "- Sequential success/safety/override metrics were collected through the existing SysML simulator.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (
        ARCH / "outputs" / f"analytic_thermostat_fit_{time.strftime('%Y%m%d-%H%M%S')}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": args.episodes,
        "jobs": args.jobs,
        "dt": args.dt,
        "max_steps": args.max_steps,
        "model": str(MODEL),
        "architecture": architecture_notes(),
        "validation": {
            "thermostat": evaluate(
                episodes=args.episodes,
                jobs=args.jobs,
                seed=args.seed,
                dt=args.dt,
                max_steps=args.max_steps,
            )
        },
        "boundary_stress": boundary_stress(),
        "total_seconds": time.time() - start,
    }
    report["comparison_rows"] = comparison_rows(report)

    json_path = out_dir / "report.json"
    md_path = out_dir / "REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)

    print(f"WROTE {json_path}")
    print(f"WROTE {md_path}")
    print(json.dumps(report["comparison_rows"], indent=2, sort_keys=True))

    validation = report["validation"]["thermostat"]
    boundary = report["boundary_stress"]["thermostat"]
    ok = (
        validation["safety_violation_rate"] == 0.0
        and validation["pointwise_agreement"] == 1.0
        and boundary["failures"] == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
