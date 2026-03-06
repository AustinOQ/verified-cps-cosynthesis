#!/usr/bin/env python3
"""
Demo: chemical mixing model running through the train.py protocol.

Wires up:
    1. SimulatorTwin  — digital twin from the SysML model
    2. requirement_statuses — live requirement evaluation from the engine
    3. A hand-coded bang-bang policy (stand-in for a neural network)
    4. train() from train.py

Run from the repo root:
    python demo_mixing.py
"""

import sys
sys.path.insert(0, "sysml-models")

from simulator_adapter import SimulatorTwin
from train import make_reward_fn, train


# --- Twin -------------------------------------------------------------------
twin = SimulatorTwin("sysml-models/mixing-sysml-model/model.sysml", dt=0.1)


# --- Requirement statuses (bound after first reset) -------------------------
# We need the engine to exist before we can bind requirement_statuses.
# SimulatorTwin creates a fresh engine on each reset, so we wrap the call.
def requirement_statuses():
    return twin._engine.requirement_statuses()


# --- Policy ------------------------------------------------------------------
class BangBangPolicy:
    """Hand-coded policy that opens valves and runs pumps until done.

    This is a stand-in for a neural policy. It demonstrates the
    policy protocol expected by train.py:
        .act(state) -> action
        .store(state, action, reward, next_state, done)
        .update()
    """

    def __init__(self):
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_count = 0

    def act(self, state: dict) -> dict:
        # Simple strategy: if not done, open everything; if done, close everything.
        if state.get("done"):
            return {
                "shouldOpenValve1": False,
                "shouldOpenValve2": False,
                "shouldTurnOnPump1": False,
                "shouldTurnOnPump2": False,
            }
        return {
            "shouldOpenValve1": True,
            "shouldOpenValve2": True,
            "shouldTurnOnPump1": True,
            "shouldTurnOnPump2": True,
        }

    def store(self, state, action, reward, next_state, done):
        self.episode_reward += reward
        self.episode_steps += 1
        if done:
            self.episode_count += 1
            tag = "OK" if reward > 0 else "VIOLATION" if reward < 0 else "TRUNCATED"
            print(
                f"  Episode {self.episode_count:3d}: "
                f"steps={self.episode_steps:4d}  "
                f"reward={self.episode_reward:+.1f}  "
                f"outcome={tag}"
            )
            # Show final requirement statuses on first episode
            if self.episode_count == 1:
                statuses = requirement_statuses()
                for name, entry in statuses.items():
                    marker = "PASS" if entry["status"] else "FAIL"
                    print(f"    [{marker}] {entry['kind']}: {name}")
            self.episode_reward = 0.0
            self.episode_steps = 0

    def update(self):
        pass


# --- Run ---------------------------------------------------------------------
if __name__ == "__main__":
    policy = BangBangPolicy()

    print("=" * 60)
    print("Chemical Mixing Model — train.py protocol demo")
    print("=" * 60)

    # Quick initial check
    state = twin()
    print(f"\nInitial state keys: {list(state.keys())}")
    print(f"Initial state: { {k: (f'{v:.1f}' if isinstance(v, float) else v) for k, v in state.items()} }")
    print(f"\nRequirements:")
    for name, entry in twin._engine.requirement_statuses().items():
        print(f"  {entry['kind']:12s}  {name}")
    twin._stop()

    # Run 5 episodes through train()
    print(f"\nRunning 5 episodes (max 200 steps each, dt=0.1s)...\n")
    train(
        twin=twin,
        requirement_statuses=requirement_statuses,
        policy=policy,
        n_episodes=5,
        max_steps=200,
        dt=0.1,
    )

    twin._stop()
    print(f"\nDone.")
