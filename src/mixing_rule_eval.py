#!/usr/bin/env python3
"""Fresh artifact-local evaluator for the mixing NeuralRequirement rule.

This sidecar validates that the direct affine-predicate rule induced by the
mixing NeuralRequirement agrees with the program shield/oracle and succeeds
closed-loop in the bundled SysML simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
ARTIFACT = THIS.parents[1]
DEFAULT_ARCH = ARTIFACT / "bundle" / "architecture-fit"
ARCH = DEFAULT_ARCH.resolve()
REPO = ARCH.parent
for path in (REPO, ARCH, REPO / "rl", REPO / "sysml-models"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from env import SysMLEnv
from oracle import extract_interface, spec_oracle


MODEL = REPO / "sysml-models" / "mixing-sysml-model" / "model.sysml"


@dataclass(frozen=True)
class MixingRule:
    both_off: int
    line1_on: int
    line2_on: int
    both_on: int

    @classmethod
    def from_spec(cls, spec: Any) -> "MixingRule":
        def find_action(line1: bool, line2: bool) -> int:
            want = {
                "shouldOpenValve1": line1,
                "shouldTurnOnPump1": line1,
                "shouldOpenValve2": line2,
                "shouldTurnOnPump2": line2,
            }
            for action_id, actuators in spec.action_map.items():
                if all(actuators.get(name) == value for name, value in want.items()):
                    return int(action_id)
            raise KeyError(f"no action matching {want}")

        return cls(
            both_off=find_action(False, False),
            line1_on=find_action(True, False),
            line2_on=find_action(False, True),
            both_on=find_action(True, True),
        )

    def primitive_predicates(self, obs: dict[str, float | bool]) -> dict[str, bool]:
        line1_remaining = (
            float(obs["tank1OriginalMl"])
            - float(obs["tank1VolumeMl"])
            < float(obs["tank1TargetTransferMl"])
        )
        line2_remaining = (
            float(obs["tank2OriginalMl"])
            - float(obs["tank2VolumeMl"])
            < float(obs["tank2TargetTransferMl"])
        )
        return {
            "line1_remaining": line1_remaining,
            "line2_remaining": line2_remaining,
        }

    def action(self, obs: dict[str, float | bool]) -> int:
        pred = self.primitive_predicates(obs)
        if pred["line1_remaining"] and pred["line2_remaining"]:
            return self.both_on
        if pred["line1_remaining"]:
            return self.line1_on
        if pred["line2_remaining"]:
            return self.line2_on
        return self.both_off


def obs_from_inputs(raw: dict[str, Any], names: list[str]) -> dict[str, float | bool]:
    obs: dict[str, float | bool] = {}
    for name in names:
        value = raw[name]
        obs[name] = value if isinstance(value, bool) else float(value)
    return obs


def safety_violation(info: dict[str, Any]) -> bool:
    for entry in info.get("statuses", {}).values():
        if entry.get("kind") in {"Prohibition", "Obligation", None} and not entry.get("status", True):
            return True
    return False


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def finish_rates(metrics: dict[str, Any]) -> None:
    episodes = max(int(metrics["episodes"]), 1)
    steps = max(int(metrics["steps"]), 1)
    checks = max(int(metrics["pointwise_checks"]), 1)
    metrics["success_rate"] = metrics["successes"] / episodes
    metrics["safety_violation_rate"] = metrics["safety_violations"] / episodes
    metrics["truncation_rate"] = metrics["truncations"] / episodes
    metrics["override_rate"] = metrics["overrides"] / steps
    metrics["mean_episode_steps"] = metrics["steps"] / episodes
    metrics["pointwise_agreement"] = 1.0 - metrics["pointwise_failures"] / checks


def evaluate(episodes: int, seed: int, dt: float, max_steps: int) -> dict[str, Any]:
    start = time.time()
    iface = extract_interface(str(MODEL), dt=dt)
    spec = iface["spec_shield"]
    obs_names = list(iface["obs_names"])
    rule = MixingRule.from_spec(spec)
    env = SysMLEnv(str(MODEL), dt=dt, max_steps=max_steps, phase=2, rng_seed=seed)
    metrics: dict[str, Any] = {
        "model": "mixing",
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
    try:
        for _episode in range(episodes):
            env.reset()
            done = False
            last_reward = 0.0
            last_info: dict[str, Any] = {}
            while not done:
                obs = obs_from_inputs(env._twin._model_inputs, obs_names)
                proposed = rule.action(obs)
                expected = int(spec_oracle(spec, obs))
                if proposed != expected:
                    metrics["pointwise_failures"] += 1
                metrics["pointwise_checks"] += 1

                final_action = int(spec(proposed, obs))
                if final_action != proposed:
                    metrics["overrides"] += 1
                _next_obs, reward, done, info = env.step(final_action)
                metrics["steps"] += 1
                last_reward = float(reward)
                last_info = info

            metrics["episodes"] += 1
            if last_reward > 0:
                metrics["successes"] += 1
            elif last_reward < 0 or safety_violation(last_info):
                metrics["safety_violations"] += 1
            else:
                metrics["truncations"] += 1
    finally:
        env.close()

    metrics["seconds"] = time.time() - start
    metrics["peak_rss_mb"] = peak_rss_mb()
    finish_rates(metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    metrics = evaluate(args.episodes, args.seed, args.dt, args.max_steps)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(MODEL),
        "method": "NeuralRequirement rule",
        "learned_params": 0,
        "uses_recurrent_state": False,
        "uses_certified_buffer_at_runtime": False,
        "primitive_affine_predicates": [
            "tank1OriginalMl - tank1VolumeMl < tank1TargetTransferMl",
            "tank2OriginalMl - tank2VolumeMl < tank2TargetTransferMl",
        ],
        "episodes": args.episodes,
        "seed": args.seed,
        "dt": args.dt,
        "max_steps": args.max_steps,
        "metrics": metrics,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "mixing_rule_eval: "
        f"episodes={metrics['episodes']} "
        f"success={metrics['success_rate']:.3f} "
        f"safety={metrics['safety_violation_rate']:.3f} "
        f"override={metrics['override_rate']:.3f} "
        f"pointwise={metrics['pointwise_agreement']:.3f} "
        f"seconds={metrics['seconds']:.2f}"
    )
    print(f"WROTE {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
