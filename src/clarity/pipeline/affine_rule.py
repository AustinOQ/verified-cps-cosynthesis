#!/usr/bin/env python3
"""Evaluate the action read from one SysML neural requirement."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
ARTIFACT = THIS.parents[3]
ARCH = (ARTIFACT / "bundle" / "architecture-fit").resolve()
REPO = ARCH.parent
for path in (ARTIFACT / "src", ARCH, REPO / "rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from env import SysMLEnv  # noqa: E402
from oracle import extract_interface, spec_oracle  # noqa: E402
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt  # noqa: E402
from clarity.sysml.inputs import inspect_sysml  # noqa: E402


def _observation(raw: dict[str, Any], names: list[str]) -> dict[str, float | bool]:
    return {
        name: value if isinstance(value := raw[name], bool) else float(value)
        for name in names
    }


def evaluate(model_path: Path, episodes: int, seed: int, dt: float,
             max_steps: int) -> dict[str, Any]:
    dt = validate_dt(dt)
    model = inspect_sysml(model_path)
    base = {
        "model": model.key,
        "model_name": model.name,
        "model_path": str(model.path),
        "model_sha256": model.sha256,
        "action_kind": model.action_kind,
        "dt": dt,
        "method": "action extracted from #NeuralRequirement",
        "learned_params": 0,
        "recurrent_state": False,
    }
    if model.action_kind != "discrete":
        return {
            **base,
            "status": "requirement_defines_allowed_real_values_not_one_action",
            "episodes": 0,
        }

    started = time.time()
    interface = extract_interface(str(model.path))
    shield = interface["spec_shield"]
    observation_names = list(interface["obs_names"])
    env = SysMLEnv(
        str(model.path), dt=dt, max_steps=max_steps, phase=2, rng_seed=seed
    )
    counts = {
        "episodes": 0,
        "successes": 0,
        "safety_violations": 0,
        "truncations": 0,
        "steps": 0,
        "overrides": 0,
        "pointwise_checks": 0,
        "pointwise_failures": 0,
    }
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            done = False
            last_reward = 0.0
            requirement_violated = False
            while not done:
                obs = _observation(env._twin._model_inputs, observation_names)
                action = int(spec_oracle(shield, obs))
                executed = int(shield(action, obs))
                counts["pointwise_checks"] += 1
                if executed != action:
                    counts["pointwise_failures"] += 1
                    counts["overrides"] += 1
                if int(shield(executed, obs)) != executed:
                    requirement_violated = True
                _next_obs, reward, done, info = env.step(executed)
                counts["steps"] += 1
                last_reward = float(reward)

            counts["episodes"] += 1
            if requirement_violated:
                counts["safety_violations"] += 1
            if last_reward > 0:
                counts["successes"] += 1
            elif last_reward == 0:
                counts["truncations"] += 1
    finally:
        env.close()

    episode_count = max(counts["episodes"], 1)
    step_count = max(counts["steps"], 1)
    check_count = max(counts["pointwise_checks"], 1)
    return {
        **base,
        **counts,
        "status": "evaluated",
        "success_rate": counts["successes"] / episode_count,
        "safety_violation_rate": counts["safety_violations"] / episode_count,
        "truncation_rate": counts["truncations"] / episode_count,
        "override_rate": counts["overrides"] / step_count,
        "pointwise_agreement": 1.0 - counts["pointwise_failures"] / check_count,
        "mean_episode_steps": counts["steps"] / episode_count,
        "seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    report = evaluate(
        Path(args.model).resolve(), args.episodes, args.seed, args.dt, args.max_steps
    )
    out = Path(args.out_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
