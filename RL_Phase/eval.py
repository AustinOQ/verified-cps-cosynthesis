"""
eval.py — Evaluate a trained RL controller and break down episode outcomes.

Runs the trained policy deterministically for N episodes and reports:
  - Success rate (goal reached)
  - Violation rate (prohibition violated, by type)
  - Timeout rate (max steps exceeded)
  - Episode length statistics for successful episodes
  - Detailed state traces for failed/timeout episodes (for debugging)

Usage:
    python -m RL_Phase.eval path/to/model.sysml [-m rl_output/model] [-e 100]
    python -m RL_Phase.eval model.sysml -m rl_output/model_best --randomize -e 200
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .extractor import extract_neural_interface
from .env import SysMLEnv


def evaluate(model_path: str, weights_path: str = "./rl_output/model",
             n_episodes: int = 100, dt: float = 0.1, max_steps: int = 200,
             randomize: bool = False, done_threshold: float = 0.0):
    """Evaluate a trained PPO controller on the SysML environment.

    Args:
        model_path:     Path to the .sysml model file.
        weights_path:   Path to saved SB3 model weights (without .zip).
        n_episodes:     Number of evaluation episodes.
        dt:             Simulation timestep (should match training).
        max_steps:      Max steps per episode (should match training).
        randomize:      Randomize start states (should match training).
        done_threshold: Done threshold (should match training).
    """
    interface = extract_neural_interface(model_path)
    env = SysMLEnv(model_path, interface, dt=dt, max_steps=max_steps,
                   randomize=randomize, done_threshold=done_threshold)
    model = PPO.load(weights_path)

    done_eps = []
    violation_eps = []
    timeout_eps = []

    for ep in range(n_episodes):
        obs, _ = env.reset()

        # Capture initial state for debugging.
        init_state = dict(env._twin._model_inputs)
        states = [init_state]
        actions_taken = []

        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            states.append(dict(env._twin._model_inputs))
            actions_taken.append(action.tolist())

            if terminated or truncated:
                record = {
                    "ep": ep,
                    "steps": step,
                    "reward": reward,
                    "violation": info.get("violation"),
                    "init_state": states[0],
                    "final_state": states[-1],
                    "last_action": actions_taken[-1],
                }
                if reward > 0:
                    done_eps.append(record)
                elif info.get("violation"):
                    violation_eps.append(record)
                else:
                    timeout_eps.append(record)
                break

    env.close()

    # Print summary.
    total = len(done_eps) + len(violation_eps) + len(timeout_eps)
    print(f"\n{'='*60}")
    print(f"Evaluation: {total} episodes")
    print(f"{'='*60}")
    print(f"  Done (goal reached):   {len(done_eps):4d}  ({100*len(done_eps)/total:.1f}%)")
    print(f"  Violation (prohib.):   {len(violation_eps):4d}  ({100*len(violation_eps)/total:.1f}%)")
    print(f"  Timeout (max steps):   {len(timeout_eps):4d}  ({100*len(timeout_eps)/total:.1f}%)")
    print()

    if violation_eps:
        by_type = {}
        for r in violation_eps:
            v = r["violation"]
            by_type.setdefault(v, []).append(r)
        print("--- Violation breakdown ---")
        for vtype, records in by_type.items():
            print(f"  {vtype}: {len(records)} episodes")
        print()

        print("--- Failed episode details (up to 5) ---")
        for r in violation_eps[:5]:
            print(f"  Episode {r['ep']}: step={r['steps']}  violation={r['violation']}")
            print(f"    init:  { {k: round(v, 2) if isinstance(v, float) else v for k, v in r['init_state'].items()} }")
            print(f"    final: { {k: round(v, 2) if isinstance(v, float) else v for k, v in r['final_state'].items()} }")
            print(f"    last_action: {r['last_action']}")
            print()

    if timeout_eps:
        print("--- Timeout episode details (up to 5) ---")
        for r in timeout_eps[:5]:
            print(f"  Episode {r['ep']}: step={r['steps']}")
            print(f"    init:  { {k: round(v, 2) if isinstance(v, float) else v for k, v in r['init_state'].items()} }")
            print(f"    final: { {k: round(v, 2) if isinstance(v, float) else v for k, v in r['final_state'].items()} }")
            print(f"    last_action: {r['last_action']}")
            print()

    if done_eps:
        lengths = [r["steps"] for r in done_eps]
        print(f"--- Done episodes: avg_steps={np.mean(lengths):.1f}  min={min(lengths)}  max={max(lengths)} ---")


def main():
    p = argparse.ArgumentParser(description="Evaluate trained RL controller.")
    p.add_argument("model_path", help="Path to .sysml model file.")
    p.add_argument("--weights", "-m", default="./rl_output/model",
                   help="Path to saved SB3 model weights.")
    p.add_argument("--episodes", "-e", type=int, default=100,
                   help="Number of evaluation episodes.")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Simulation timestep (should match training).")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Max steps per episode (should match training).")
    p.add_argument("--randomize", action="store_true",
                   help="Randomize start states.")
    p.add_argument("--done-threshold", type=float, default=0.0,
                   help="Done threshold (should match training).")
    args = p.parse_args()

    evaluate(args.model_path, args.weights, args.episodes, args.dt, args.max_steps,
             args.randomize, args.done_threshold)


if __name__ == "__main__":
    main()
