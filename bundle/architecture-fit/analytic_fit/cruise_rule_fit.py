#!/usr/bin/env python3
"""Analytic sidecar for the two cruise-controller models.

This program does not train a neural net. It reads the existing SysML models
and validates a direct rule induced by their NeuralRequirement clauses:

  accel iff targetSpeed > currentSpeedMps + toleranceMps
           and gapMeters >= safeFollowingDistanceMeters
  brake iff targetSpeed < currentSpeedMps - toleranceMps
          or gapMeters < safeFollowingDistanceMeters

For the discrete model the rule emits the exact boolean action. For the
continuous model it emits +100, -100, or 0 and checks that value against the
existing interval shield.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
REPO = ARCH.parent
for path in (REPO, ARCH, REPO / "rl", REPO / "sysml-models"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from continuous_env import SysMLContinuousEnv
from continuous_shield import ContinuousShield
from env import SysMLEnv
from oracle import extract_interface, spec_oracle


MODELS = {
    "cruise-discrete": REPO / "sysml-models" / "cruise-controller-model" / "model.sysml",
    "cruise-continuous": REPO / "sysml-models" / "cruise-continuous-model" / "model.sysml",
}


@dataclass(frozen=True)
class DiscreteCruiseRule:
    tolerance_mps: float
    safe_gap_m: float
    coast_action: int
    throttle_action: int
    brake_action: int

    @classmethod
    def from_spec(cls, spec: Any) -> "DiscreteCruiseRule":
        return cls(
            tolerance_mps=float(spec.unchanging["toleranceMps"]),
            safe_gap_m=float(spec.unchanging["safeFollowingDistanceMeters"]),
            coast_action=_find_action(spec.action_map, applyThrottle=False, applyBrake=False),
            throttle_action=_find_action(spec.action_map, applyThrottle=True, applyBrake=False),
            brake_action=_find_action(spec.action_map, applyThrottle=False, applyBrake=True),
        )

    def primitive_predicates(self, obs: dict[str, float]) -> dict[str, bool]:
        current = float(obs["currentSpeedMps"])
        target = float(obs["targetSpeed"])
        gap = float(obs["gapMeters"])
        return {
            "speed_below_target_band": target - current - self.tolerance_mps > 0.0,
            "speed_above_target_band": current - target - self.tolerance_mps > 0.0,
            "gap_too_close": self.safe_gap_m - gap > 0.0,
        }

    def action(self, obs: dict[str, float]) -> int:
        pred = self.primitive_predicates(obs)
        if pred["speed_below_target_band"] and not pred["gap_too_close"]:
            return self.throttle_action
        if pred["speed_above_target_band"] or pred["gap_too_close"]:
            return self.brake_action
        return self.coast_action


@dataclass(frozen=True)
class ContinuousCruiseRule:
    tolerance_mps: float
    safe_gap_m: float
    brake_force: float = -100.0
    coast_force: float = 0.0
    throttle_force: float = 100.0

    @classmethod
    def from_shield(cls, shield: ContinuousShield) -> "ContinuousCruiseRule":
        return cls(
            tolerance_mps=float(shield.unchanging["toleranceMps"]),
            safe_gap_m=float(shield.unchanging["safeFollowingDistanceMeters"]),
        )

    def primitive_predicates(self, obs: dict[str, float]) -> dict[str, bool]:
        current = float(obs["currentSpeedMps"])
        target = float(obs["targetSpeed"])
        gap = float(obs["gapMeters"])
        return {
            "speed_below_target_band": target - current - self.tolerance_mps > 0.0,
            "speed_above_target_band": current - target - self.tolerance_mps > 0.0,
            "gap_too_close": self.safe_gap_m - gap > 0.0,
        }

    def force(self, obs: dict[str, float]) -> float:
        pred = self.primitive_predicates(obs)
        if pred["speed_below_target_band"] and not pred["gap_too_close"]:
            return self.throttle_force
        if pred["speed_above_target_band"] or pred["gap_too_close"]:
            return self.brake_force
        return self.coast_force


def _find_action(action_map: dict[int, dict[str, bool]], **want: bool) -> int:
    for action_id, actuators in action_map.items():
        if all(actuators.get(name) == value for name, value in want.items()):
            return int(action_id)
    raise KeyError(f"no action matching {want}")


def _obs_from_inputs(raw: dict[str, Any], names: list[str]) -> dict[str, float | bool]:
    obs: dict[str, float | bool] = {}
    for name in names:
        value = raw[name]
        if isinstance(value, bool):
            obs[name] = value
        else:
            obs[name] = float(value)
    return obs


def _safety_violation(info: dict[str, Any]) -> bool:
    statuses = info.get("statuses", {})
    for entry in statuses.values():
        kind = entry.get("kind")
        if kind in {"Prohibition", "Obligation", None} and not entry.get("status", True):
            return True
    return False


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _blank_metrics(model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "episodes": 0,
        "successes": 0,
        "safety_violations": 0,
        "truncations": 0,
        "steps": 0,
        "overrides": 0,
        "pointwise_checks": 0,
        "pointwise_failures": 0,
        "seconds": 0.0,
        "peak_rss_mb": 0.0,
    }


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


def _finish_rates(m: dict[str, Any]) -> None:
    episodes = max(int(m["episodes"]), 1)
    steps = max(int(m["steps"]), 1)
    checks = max(int(m["pointwise_checks"]), 1)
    m["success_rate"] = m["successes"] / episodes
    m["safety_violation_rate"] = m["safety_violations"] / episodes
    m["truncation_rate"] = m["truncations"] / episodes
    m["override_rate"] = m["overrides"] / steps
    m["mean_episode_steps"] = m["steps"] / episodes
    m["pointwise_agreement"] = 1.0 - (m["pointwise_failures"] / checks)


def _eval_discrete_worker(model_path: str, episodes: int, seed: int,
                          dt: float, max_steps: int) -> dict[str, Any]:
    start = time.time()
    iface = extract_interface(model_path, dt=dt)
    spec = iface["spec_shield"]
    obs_names = list(iface["obs_names"])
    rule = DiscreteCruiseRule.from_spec(spec)
    env = SysMLEnv(model_path, dt=dt, max_steps=max_steps, phase=2, rng_seed=seed)
    metrics = _blank_metrics("cruise-discrete")
    try:
        for _episode in range(episodes):
            env.reset()
            done = False
            last_reward = 0.0
            last_info: dict[str, Any] = {}
            while not done:
                obs = _obs_from_inputs(env._twin._model_inputs, obs_names)
                proposed = rule.action(obs)  # type: ignore[arg-type]
                oracle_action = spec_oracle(spec, obs)
                if proposed != oracle_action:
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


def _eval_continuous_worker(model_path: str, episodes: int, seed: int,
                            dt: float, max_steps: int) -> dict[str, Any]:
    start = time.time()
    shield = ContinuousShield(model_path)
    rule = ContinuousCruiseRule.from_shield(shield)
    env = SysMLContinuousEnv(model_path, dt=dt, max_steps=max_steps, phase=2, rng_seed=seed)
    obs_names = [name for name in env._obs_keys if name.lower() != "done"]
    metrics = _blank_metrics("cruise-continuous")
    eps = 1e-7
    try:
        for _episode in range(episodes):
            env.reset()
            done = False
            last_reward = 0.0
            last_info: dict[str, Any] = {}
            while not done:
                obs = _obs_from_inputs(env._twin._model_inputs, obs_names)
                proposed = rule.force(obs)  # type: ignore[arg-type]
                lo, hi = shield.safe_interval(obs)
                if not (lo - eps <= proposed <= hi + eps):
                    metrics["pointwise_failures"] += 1
                metrics["pointwise_checks"] += 1

                final_force, overridden = shield(proposed, obs)
                if overridden:
                    metrics["overrides"] += 1
                _next_obs, reward, done, info = env.step(final_force)
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
    out = []
    for i in range(jobs):
        n = base + (1 if i < rem else 0)
        if n:
            out.append((n, seed + 100_000 * i))
    return out


def evaluate_model(model_name: str, model_path: Path, episodes: int, jobs: int,
                   seed: int, dt: float, max_steps: int) -> dict[str, Any]:
    chunks = _split_work(episodes, jobs, seed)
    worker = _eval_discrete_worker if model_name == "cruise-discrete" else _eval_continuous_worker
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(worker, str(model_path), n, chunk_seed, dt, max_steps)
            for n, chunk_seed in chunks
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return _merge_metrics(rows)


def architecture_notes() -> dict[str, Any]:
    return {
        "learned_parameters": 0,
        "uses_recurrent_state": False,
        "uses_certified_buffer_at_runtime": False,
        "primitive_affine_predicates": [
            "targetSpeed - currentSpeedMps - toleranceMps > 0",
            "currentSpeedMps - targetSpeed - toleranceMps > 0",
            "safeFollowingDistanceMeters - gapMeters > 0",
        ],
        "rule_layer": [
            "throttle/positive force iff predicate_0 and not predicate_2",
            "brake/negative force iff predicate_1 or predicate_2",
            "otherwise coast/zero force",
        ],
        "single_raw_affine_classifier_is_exact": False,
        "single_raw_affine_classifier_note": (
            "The exact throttle region is a conjunction of two halfspaces, "
            "and the exact brake region is a disjunction. A one-layer affine "
            "threshold over the raw inputs is therefore the wrong exact class; "
            "a tiny threshold/rule layer is the natural exact representation."
        ),
    }


def _boundary_observations(tolerance_mps: float, safe_gap_m: float) -> list[dict[str, float | bool]]:
    observations: list[dict[str, float | bool]] = []
    eps = 1e-6
    targets = [5.0, 15.0, 25.0]
    speed_offsets = [
        -5.0,
        -tolerance_mps - eps,
        -tolerance_mps,
        -tolerance_mps + eps,
        0.0,
        tolerance_mps - eps,
        tolerance_mps,
        tolerance_mps + eps,
        5.0,
    ]
    gaps = [0.0, safe_gap_m - eps, safe_gap_m, safe_gap_m + eps, 100.0]
    for target in targets:
        for offset in speed_offsets:
            for gap in gaps:
                observations.append({
                    "currentSpeedMps": target + offset,
                    "targetSpeed": target,
                    "gapMeters": gap,
                    "done": False,
                })
    return observations


def boundary_stress() -> dict[str, Any]:
    """Probe threshold boundaries independently of simulator trajectories."""
    out: dict[str, Any] = {}

    discrete_iface = extract_interface(str(MODELS["cruise-discrete"]))
    discrete_spec = discrete_iface["spec_shield"]
    discrete_rule = DiscreteCruiseRule.from_spec(discrete_spec)
    discrete_cases = _boundary_observations(
        discrete_rule.tolerance_mps, discrete_rule.safe_gap_m)
    discrete_failures = []
    discrete_actions: dict[str, int] = {}
    for obs in discrete_cases:
        proposed = discrete_rule.action(obs)  # type: ignore[arg-type]
        expected = spec_oracle(discrete_spec, obs)
        discrete_actions[str(proposed)] = discrete_actions.get(str(proposed), 0) + 1
        if proposed != expected:
            discrete_failures.append({
                "obs": obs,
                "proposed": proposed,
                "expected": expected,
            })
    out["cruise-discrete"] = {
        "cases": len(discrete_cases),
        "failures": len(discrete_failures),
        "agreement": 1.0 - len(discrete_failures) / max(len(discrete_cases), 1),
        "predicted_action_counts": discrete_actions,
        "failure_examples": discrete_failures[:5],
    }

    continuous_shield = ContinuousShield(str(MODELS["cruise-continuous"]))
    continuous_rule = ContinuousCruiseRule.from_shield(continuous_shield)
    continuous_cases = _boundary_observations(
        continuous_rule.tolerance_mps, continuous_rule.safe_gap_m)
    continuous_failures = []
    continuous_forces: dict[str, int] = {}
    eps = 1e-7
    for obs in continuous_cases:
        proposed = continuous_rule.force(obs)  # type: ignore[arg-type]
        lo, hi = continuous_shield.safe_interval(obs)
        continuous_forces[str(proposed)] = continuous_forces.get(str(proposed), 0) + 1
        if not (lo - eps <= proposed <= hi + eps):
            continuous_failures.append({
                "obs": obs,
                "proposed": proposed,
                "safe_interval": [lo, hi],
            })
    out["cruise-continuous"] = {
        "cases": len(continuous_cases),
        "failures": len(continuous_failures),
        "agreement": 1.0 - len(continuous_failures) / max(len(continuous_cases), 1),
        "predicted_force_counts": continuous_forces,
        "failure_examples": continuous_failures[:5],
    }
    return out


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in report["validation"].items():
        rows.append({
            "model": name,
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
    rows = report["comparison_rows"]
    headers = [
        "model", "method", "learned_params", "training_seconds",
        "success_rate", "safety_violation_rate", "override_rate",
        "pointwise_agreement", "peak_rss_mb",
    ]
    lines = [
        "# Analytic Cruise Fit Report",
        "",
        "This sidecar validates a zero-learned-parameter rule extracted from the two cruise NeuralRequirement shapes.",
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
    for row in rows:
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
        "- Discrete pointwise agreement means the sidecar action equaled `spec_oracle(SpecShield, obs)`.",
        "- Continuous pointwise agreement means the sidecar force was inside `ContinuousShield.safe_interval(obs)`.",
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
        ARCH / "outputs" / f"analytic_cruise_fit_{time.strftime('%Y%m%d-%H%M%S')}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    validation = {
        name: evaluate_model(
            name,
            path,
            episodes=args.episodes,
            jobs=args.jobs,
            seed=args.seed + i * 10_000,
            dt=args.dt,
            max_steps=args.max_steps,
        )
        for i, (name, path) in enumerate(MODELS.items())
    }
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes_per_model": args.episodes,
        "jobs": args.jobs,
        "dt": args.dt,
        "max_steps": args.max_steps,
        "models": {name: str(path) for name, path in MODELS.items()},
        "architecture": architecture_notes(),
        "validation": validation,
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
    ok = all(
        row["safety_violation_rate"] == 0.0 and row["pointwise_agreement"] == 1.0
        for row in validation.values()
    ) and all(row["failures"] == 0 for row in report["boundary_stress"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
