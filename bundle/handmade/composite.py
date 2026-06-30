"""Composite: numpy RecurrentActorCritic + SpecShield.

The shield (rl/shield.py:SpecShield) evaluates the #NeuralRequirement AST
directly — no torch involved. We use it as a runtime correctness layer:
if the policy proposes a dead/invalid action, the shield returns the spec-
correct action instead, and we mark the step as `overridden`.

`act()` records per-call latencies (policy forward, shield call, total) in
microseconds via time.perf_counter_ns(). Negligible (~50 ns/probe) overhead.
"""

from __future__ import annotations

import time

import numpy as np

from .nn import softmax
from .policy import RecurrentActorCritic


class Composite:
    def __init__(self, policy: RecurrentActorCritic, spec_shield,
                 obs_names):
        self.policy = policy
        self.spec_shield = spec_shield
        self.obs_names = obs_names

    def act(self, obs_norm: np.ndarray, h_prev: np.ndarray,
            raw_obs: dict, greedy: bool, rng=None):
        """obs_norm: (B=1, obs_dim) normalized obs. h_prev: (1, H).
        Returns (final_action, log_prob, value, h_new, overridden,
                 dist_probs, timing_us)
        where timing_us is a dict with policy_us, shield_us, total_us.
        """
        t0 = time.perf_counter_ns()
        logits, value, h_new = self.policy.step(obs_norm, h_prev)
        t1 = time.perf_counter_ns()
        probs = softmax(logits, axis=-1)
        if greedy:
            proposed = int(np.argmax(logits[0]))
        else:
            if rng is None:
                rng = np.random
            # Categorical sample (Gumbel-max)
            u = rng.random(size=logits.shape[-1])
            g = -np.log(-np.log(np.clip(u, 1e-30, None)))
            proposed = int(np.argmax(logits[0] + g))
        obs_dict = {name: float(raw_obs.get(name, 0))
                    for name in self.obs_names}
        t2 = time.perf_counter_ns()
        final = self.spec_shield(proposed, obs_dict)
        t3 = time.perf_counter_ns()
        overridden = (final != proposed)
        log_prob = float(np.log(np.clip(probs[0, final], 1e-30, None)))
        timing_us = {
            "policy_us": (t1 - t0) / 1000.0,
            "shield_us": (t3 - t2) / 1000.0,
            "total_us":  (t3 - t0) / 1000.0,
        }
        return (final, log_prob, float(value[0]), h_new, overridden,
                probs[0], timing_us)
