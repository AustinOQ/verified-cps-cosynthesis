"""Exact program-shield composite for handmade reduced policies."""

from __future__ import annotations

import time

import numpy as np

from handmade.nn import softmax


class ProgramShieldComposite:
    """Policy + exact program/AST ``SpecShield``.

    This deliberately does not use the DNN co-architecture shield. The shield
    object is the callable returned by ``oracle.extract_interface`` from
    ``rl/shield.py`` and evaluates the SysML neural requirement programmatically.
    """

    shield_type = "program_ast_spec_shield"

    def __init__(self, policy, spec_shield, obs_names):
        self.policy = policy
        self.spec_shield = spec_shield
        self.obs_names = list(obs_names)

    def act(self, obs_norm: np.ndarray, h_prev: np.ndarray,
            raw_obs: dict, greedy: bool, rng=None):
        t0 = time.perf_counter_ns()
        logits, value, h_new = self.policy.step(obs_norm, h_prev)
        t1 = time.perf_counter_ns()

        probs = softmax(logits, axis=-1)
        if greedy:
            proposed = int(np.argmax(logits[0]))
        else:
            if rng is None:
                rng = np.random.default_rng()
            u = rng.random(size=logits.shape[-1])
            gumbel = -np.log(-np.log(np.clip(u, 1e-30, None)))
            proposed = int(np.argmax(logits[0] + gumbel))

        obs_dict = {name: float(raw_obs.get(name, 0.0))
                    for name in self.obs_names}
        t2 = time.perf_counter_ns()
        final = int(self.spec_shield(proposed, obs_dict))
        t3 = time.perf_counter_ns()

        log_prob = float(np.log(np.clip(probs[0, final], 1e-30, None)))
        timing_us = {
            "policy_us": (t1 - t0) / 1000.0,
            "shield_us": (t3 - t2) / 1000.0,
            "total_us": (t3 - t0) / 1000.0,
        }
        return (
            final,
            log_prob,
            float(value[0]),
            h_new,
            final != proposed,
            probs[0],
            timing_us,
        )

