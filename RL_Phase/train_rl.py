"""
train_rl.py — Train an RL controller from a SysML v2 model.

This is the main entry point for the RL training pipeline. Given a path
to a SysML v2 model file, it:

  1. Extracts the #Neural action def interface (observations, actions,
     done condition) using extractor.py
  2. Constructs a Gymnasium environment (env.py) wrapping the SysML
     digital twin simulator
  3. Trains a PPO controller using stable-baselines3
  4. Saves the trained model and best checkpoint

The pipeline is plug-and-play: point it at any SysML model with a #Neural
action def and it will auto-configure the observation space, action space,
and reward signal from the model's requirements.

Training Modes
--------------
- **Sparse** (default): Reward is +1 for goal, -1 for prohibition violation,
  0 otherwise. Requires high entropy for exploration. Works but slow.
- **Shaped** (--shaping): Reward is +10 for goal, -1 for violation, plus
  per-step normalized distance improvement derived from the done expression.
  Converges much faster and more reliably. Recommended.

PPO Hyperparameters
-------------------
- n_steps=4096: Large rollout buffer for sparse reward signals.
- batch_size=128: Standard minibatch size.
- target_kl=0.015: Early stopping on KL divergence to prevent policy collapse.
- ent_coef: 0.15 (sparse) or 0.01 (shaped) — high entropy needed for
  exploration with sparse rewards, low entropy fine with dense shaping.

Usage:
    python -m RL_Phase path/to/model.sysml [-o output_dir] [-n timesteps]
    python -m RL_Phase model.sysml --shaping --randomize --max-steps 1000
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .extractor import extract_neural_interface
from .env import SysMLEnv


class LogCallback(BaseCallback):
    """SB3 callback that saves the best model via periodic micro-tests.

    Every `eval_freq` timesteps, runs `eval_episodes` deterministic episodes
    on a separate evaluation environment (default initial state, no
    randomization). The model with the highest goal-completion rate (and
    fewest violations) is saved as model_best.
    """
    def __init__(self, save_path: str, eval_env: SysMLEnv,
                 eval_freq: int = 50_000, eval_episodes: int = 100):
        super().__init__()
        self.save_path = save_path
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.episode_rewards = []
        self.episode_violations = []
        self._best_goal_rate = -1.0
        self._last_eval_step = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if info.get("violation"):
                self.episode_violations.append(info["violation"])

        # Periodic micro-test evaluation.
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        return True

    def _run_eval(self):
        """Run deterministic micro-test and save if best."""
        goals, violations = 0, 0
        for ep in range(self.eval_episodes):
            obs, _ = self.eval_env.reset(seed=ep)
            done = truncated = False
            info = {}
            while not done and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = self.eval_env.step(action)
                if info.get("violation"):
                    violations += 1
                    break
            if done and not info.get("violation"):
                goals += 1

        goal_rate = goals / self.eval_episodes
        print(f"  [eval @ {self.num_timesteps:,} steps: "
              f"goals={goals}/{self.eval_episodes}  violations={violations}]")
        if goal_rate > self._best_goal_rate:
            self._best_goal_rate = goal_rate
            self.model.save(self.save_path)
            print(f"  [best model saved: goal_rate={goal_rate:.2%}]")


def train(model_path: str, output_dir: str = "./rl_output",
          total_timesteps: int = 200_000, dt: float = 0.1,
          max_steps: int = 200, randomize: bool = False,
          shaping: bool = False, seed: int = None) -> dict:
    """Train a PPO controller from a SysML model.

    Args:
        model_path:       Path to the .sysml model file.
        output_dir:       Directory to save trained models and interface.
        total_timesteps:  Total environment steps for training.
        dt:               Simulation timestep in seconds.
        max_steps:        Max steps per episode before truncation.
        randomize:        Randomize start states each episode.
        shaping:          Use distance-based reward shaping.

    Returns:
        Dict with training summary (episode count, output directory).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Extract interface from SysML model.
    interface = extract_neural_interface(model_path)
    print(f"Model: {model_path}")
    print(f"Obs ({len(interface.obs_names)}): {interface.obs_names}")
    print(f"Act ({len(interface.action_names)}): {interface.action_names}")
    if interface.done_expr:
        print(f"Done: {interface.done_expr}")
    if randomize:
        print("Start state randomization: ON")
    if shaping:
        print("Reward shaping: ON (distance-based)")

    # Save extracted interface for downstream use.
    # Exclude non-serializable dataclass objects (scenario_inputs, scenario_constraints).
    serializable = {k: v for k, v in vars(interface).items()
                    if k not in ('scenario_inputs', 'scenario_constraints')}
    with open(out / "interface.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # 2. Build Gymnasium environment.
    env = SysMLEnv(model_path, interface, dt=dt, max_steps=max_steps,
                   randomize=randomize, shaping=shaping)

    # 3. Train with SB3 PPO.
    # With shaping, less exploration needed (dense signal available).
    ent = 0.01 if shaping else 0.15
    model = PPO(
        "MlpPolicy", env,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=ent,
        target_kl=0.015,
        verbose=1,
        seed=seed,
    )

    # Build a separate eval environment (same randomization as training).
    eval_env = SysMLEnv(model_path, interface, dt=dt, max_steps=max_steps,
                        randomize=randomize, shaping=shaping)

    best_path = str(out / "model_best")
    cb = LogCallback(save_path=best_path, eval_env=eval_env,
                     eval_freq=50_000, eval_episodes=100)
    model.learn(total_timesteps=total_timesteps, callback=cb)

    print(f"\nSaved to {out}/")
    env.close()
    eval_env.close()

    return {"episodes": len(cb.episode_rewards), "output_dir": str(out)}


def main():
    p = argparse.ArgumentParser(description="Train RL controller from SysML model.")
    p.add_argument("model_path", help="Path to .sysml model file.")
    p.add_argument("--output-dir", "-o", default="./rl_output",
                   help="Directory to save trained models.")
    p.add_argument("--timesteps", "-n", type=int, default=200_000,
                   help="Total environment steps for training.")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Simulation timestep in seconds.")
    p.add_argument("--max-steps", type=int, default=200,
                   help="Max steps per episode.")
    p.add_argument("--randomize", action="store_true",
                   help="Randomize start states each episode.")
    p.add_argument("--shaping", action="store_true",
                   help="Enable distance-based reward shaping.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    args = p.parse_args()

    train(args.model_path, args.output_dir, args.timesteps, args.dt, args.max_steps,
          args.randomize, args.shaping, args.seed)


if __name__ == "__main__":
    main()
