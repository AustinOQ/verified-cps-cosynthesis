#!/usr/bin/env python3
"""
phased_training.py — Single-step phased training from SysML v2 models.

Phase 1: Goal only — reward = d_before - d_after (distance improvement).
Phase 2: Goal + Safety — same reward, but violations get a large penalty.

Each step: observe state -> sample action from model -> step sim ->
compute distance change -> reward-weighted BCE update.

This is closer to supervised learning than RL: no full episodes needed,
every single step produces a gradient signal. Random warmup after resets
gives diverse states across the trajectory space.

Usage:
    python phased_training.py sysml-models/thermostat/thermostat.sysml --randomize
    python phased_training.py sysml-models/mixing-sysml-model/model.sysml --randomize
    python phased_training.py sysml-models/cruise-controller-model/model.sysml --randomize
"""

import argparse
import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "sysml-models"))

from RL_Phase.extractor import extract_neural_interface
from RL_Phase.env import SysMLEnv


class PolicyNet(nn.Module):
    def __init__(self, n_obs, n_act, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _obs_dict(env):
    return env._obs_dict(dict(env._twin._model_inputs))


def _reset_to_random(env, max_warmup=50):
    """Reset env then take 0-max_warmup random steps for state diversity."""
    obs, _ = env.reset()
    warmup = random.randint(0, max_warmup)
    for _ in range(warmup):
        obs, _, d, t, info = env.step(env.action_space.sample())
        if d or t or info.get("violation"):
            obs, _ = env.reset()
            break
    return obs


def train_phased(
    model_path,
    output_dir="./phased_output",
    phase1_steps=100_000,
    phase2_steps=100_000,
    dt=0.1,
    max_steps=200,
    batch_size=64,
    lr=1e-3,
    randomize=False,
    eval_episodes=50,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    interface = extract_neural_interface(model_path)
    print(f"Model: {model_path}")
    print(f"Obs: {interface.obs_names}")
    print(f"Act: {interface.action_names}")
    print(f"Done: {interface.done_expr}")

    env = SysMLEnv(
        model_path, interface, dt=dt, max_steps=max_steps,
        randomize=randomize, shaping=True,
    )

    n_obs = len(env._obs_indices)
    n_act = len(interface.action_names)

    policy = PolicyNet(n_obs, n_act)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # ---- Training --------------------------------------------------------
    for phase in [1, 2]:
        steps = phase1_steps if phase == 1 else phase2_steps
        label = "Goal only" if phase == 1 else "Goal + Safety"
        print(f"\n{'=' * 55}")
        print(f"  Phase {phase}: {label}  ({steps:,} steps, batch={batch_size})")
        print(f"{'=' * 55}")

        buf_obs, buf_act, buf_rew = [], [], []
        stats = {"improve": 0, "worsen": 0, "viol": 0, "goal": 0}
        last_loss = 0.0

        obs = _reset_to_random(env)
        d_prev = env._goal_distance(_obs_dict(env))
        ep_steps = 0

        for step in range(steps):
            # --- sample action from current policy ---
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                p = policy(obs_t).squeeze(0)
            action = (torch.rand_like(p) < p).int().numpy()

            # --- step ---
            next_obs, _, done, truncated, info = env.step(action)
            d_curr = env._goal_distance(_obs_dict(env))
            ep_steps += 1

            # --- reward ---
            reward = d_prev - d_curr  # positive = got closer

            if done and d_curr == 0:
                reward = max(abs(d_prev), 1.0)
                stats["goal"] += 1

            if phase == 2 and info.get("violation"):
                reward = -max(abs(d_prev), 1.0)
                stats["viol"] += 1

            if reward > 0:
                stats["improve"] += 1
            else:
                stats["worsen"] += 1

            buf_obs.append(obs)
            buf_act.append(action)
            buf_rew.append(reward)

            # --- gradient step on full batch ---
            if len(buf_obs) >= batch_size:
                o = torch.FloatTensor(np.array(buf_obs))
                a = torch.FloatTensor(np.array(buf_act))
                r = torch.FloatTensor(buf_rew)

                if r.std() > 1e-8:
                    r = (r - r.mean()) / (r.std() + 1e-8)

                probs = policy(o)
                log_p = a * torch.log(probs + 1e-8) + \
                        (1 - a) * torch.log(1 - probs + 1e-8)
                loss = -(r.unsqueeze(1) * log_p).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

                buf_obs, buf_act, buf_rew = [], [], []

            # --- advance or reset ---
            if done or truncated or info.get("violation") or ep_steps >= max_steps:
                obs = _reset_to_random(env)
                d_prev = env._goal_distance(_obs_dict(env))
                ep_steps = 0
            else:
                obs = next_obs
                d_prev = d_curr

            # --- log ---
            if (step + 1) % 10_000 == 0:
                print(
                    f"  [{step+1:>7,}/{steps:,}]  loss={last_loss:.4f}  "
                    f"improve={stats['improve']}  worsen={stats['worsen']}  "
                    f"goal={stats['goal']}  viol={stats['viol']}"
                )
                stats = {"improve": 0, "worsen": 0, "viol": 0, "goal": 0}

    # ---- Save ------------------------------------------------------------
    torch.save(policy.state_dict(), str(out / "policy.pt"))
    import json
    json.dump(
        {"n_obs": n_obs, "n_act": n_act, "model_path": model_path},
        open(out / "info.json", "w"),
    )

    # ---- Evaluation ------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  Evaluation ({eval_episodes} episodes, deterministic)")
    print(f"{'=' * 55}")

    goals, violations, timeouts = 0, 0, 0
    viol_detail = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=ep)
        done = truncated = False
        while not done and not truncated:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                p = policy(obs_t).squeeze(0)
            action = (p > 0.5).int().numpy()
            obs, _, done, truncated, info = env.step(action)
            if info.get("violation"):
                violations += 1
                viol_detail.append(info["violation"])
                break
        if done and not info.get("violation"):
            goals += 1
        elif not done and not info.get("violation"):
            timeouts += 1

    env.close()

    print(f"\n  goals={goals}/{eval_episodes}  violations={violations}  "
          f"timeouts={timeouts}")
    if viol_detail:
        print(f"  violations: {dict(Counter(viol_detail))}")
    print(f"\n  Saved to {out}/")


def main():
    p = argparse.ArgumentParser(
        description="Single-step phased training from SysML model."
    )
    p.add_argument("model_path", help="Path to .sysml model file.")
    p.add_argument("-o", "--output-dir", default="./phased_output")
    p.add_argument("--phase1-steps", type=int, default=100_000)
    p.add_argument("--phase2-steps", type=int, default=100_000)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=50)
    args = p.parse_args()

    train_phased(
        args.model_path, args.output_dir,
        args.phase1_steps, args.phase2_steps,
        args.dt, args.max_steps, args.batch_size, args.lr,
        args.randomize, args.eval_episodes,
    )


if __name__ == "__main__":
    main()
