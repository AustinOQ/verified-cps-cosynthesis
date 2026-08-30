"""GAE — generalized advantage estimation. Pure numpy port of rl/ppo.py."""

from __future__ import annotations

import numpy as np


def compute_gae(rewards, values, dones, gamma: float = 0.99,
                lam: float = 0.95):
    """rewards, values, dones: lists/1D arrays of length T (single episode).
    Returns (advantages, returns) as 1D float32 arrays.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    last_value = 0.0   # bootstrap from terminal state = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else last_value
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns
