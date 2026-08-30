"""Episode collection and evaluation rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Episode:
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    overrides: list = field(default_factory=list)
    # Per-step latencies (µs), recorded via time.perf_counter_ns inside
    # Composite.act. Empty if Composite isn't used (e.g., raw-policy rollout).
    policy_us: list = field(default_factory=list)
    shield_us: list = field(default_factory=list)
    total_us: list = field(default_factory=list)
    outcome: str = ""
    violations: list = field(default_factory=list)


def collect_episode(env, composite, rng=None, greedy: bool = False) -> Episode:
    """Roll out one episode using composite.act. Records data for PPO and
    per-step latencies for inference profiling.
    """
    obs = env.reset()
    h = composite.policy.initial_hidden(1)
    ep = Episode()
    done = False
    info = {}
    reward = 0.0
    while not done:
        obs_norm = obs[None, :].astype(np.float32)
        raw_obs = env._twin._model_inputs
        (final_action, log_p, value, h_new, overridden, _,
         timing_us) = composite.act(obs_norm, h, raw_obs, greedy=greedy,
                                     rng=rng)
        next_obs, reward, done, info = env.step(final_action)
        ep.obs.append(obs)
        ep.actions.append(final_action)
        ep.rewards.append(reward)
        ep.values.append(value)
        ep.log_probs.append(log_p)
        ep.dones.append(bool(done))
        ep.overrides.append(bool(overridden))
        ep.policy_us.append(timing_us["policy_us"])
        ep.shield_us.append(timing_us["shield_us"])
        ep.total_us.append(timing_us["total_us"])
        h = h_new
        obs = next_obs
    ep.outcome = ("SUCCESS" if reward > 0
                  else ("VIOLATION" if reward < 0 else "TRUNCATED"))
    if "statuses" in info:
        ep.violations = [n for n, s in info["statuses"].items()
                          if not s["status"]]
    return ep


@dataclass
class RolloutSummary:
    n_episodes: int
    n_steps: int
    success_rate: float
    violation_rate: float
    truncated_rate: float
    pooled_override_rate: float
    mean_episode_steps: float
    mean_reward: float
    safety_violation_rate: float
    # Latency stats (µs) pooled across every step of every episode.
    policy_us_mean: float
    policy_us_p95: float
    policy_us_p99: float
    shield_us_mean: float
    shield_us_p95: float
    shield_us_p99: float
    total_us_mean: float
    total_us_p95: float
    total_us_p99: float


def evaluate(env_factory, composite, n_episodes: int, rng=None,
             greedy: bool = True) -> RolloutSummary:
    """Run n_episodes rollouts and aggregate metrics + latency stats.

    env_factory() is called ONCE; the env's RNG advances per reset() so
    consecutive episodes get varied scenarios (matches rl/train.py:evaluate).
    """
    SAFETY_NAMES = {"No Dry Running", "No Dead Heading"}
    n_succ = 0; n_viol = 0; n_trunc = 0
    n_safety_eps = 0
    total_steps = 0; total_overrides = 0
    total_reward = 0.0
    all_policy = []
    all_shield = []
    all_total = []
    env = env_factory()
    try:
        for _ in range(n_episodes):
            ep = collect_episode(env, composite, rng=rng, greedy=greedy)
            if ep.outcome == "SUCCESS": n_succ += 1
            elif ep.outcome == "VIOLATION": n_viol += 1
            else: n_trunc += 1
            if any(v in SAFETY_NAMES for v in ep.violations):
                n_safety_eps += 1
            total_steps += len(ep.actions)
            total_overrides += sum(ep.overrides)
            total_reward += sum(ep.rewards)
            all_policy.extend(ep.policy_us)
            all_shield.extend(ep.shield_us)
            all_total.extend(ep.total_us)
    finally:
        env.close()

    if all_total:
        policy_arr = np.asarray(all_policy)
        shield_arr = np.asarray(all_shield)
        total_arr  = np.asarray(all_total)
        p_mean = float(policy_arr.mean());  p_p95 = float(np.percentile(policy_arr, 95));  p_p99 = float(np.percentile(policy_arr, 99))
        s_mean = float(shield_arr.mean());  s_p95 = float(np.percentile(shield_arr, 95));  s_p99 = float(np.percentile(shield_arr, 99))
        t_mean = float(total_arr.mean());   t_p95 = float(np.percentile(total_arr, 95));   t_p99 = float(np.percentile(total_arr, 99))
    else:
        p_mean = p_p95 = p_p99 = 0.0
        s_mean = s_p95 = s_p99 = 0.0
        t_mean = t_p95 = t_p99 = 0.0

    return RolloutSummary(
        n_episodes=n_episodes,
        n_steps=total_steps,
        success_rate=n_succ / n_episodes,
        violation_rate=n_viol / n_episodes,
        truncated_rate=n_trunc / n_episodes,
        pooled_override_rate=total_overrides / max(total_steps, 1),
        mean_episode_steps=total_steps / n_episodes,
        mean_reward=total_reward / n_episodes,
        safety_violation_rate=n_safety_eps / n_episodes,
        policy_us_mean=p_mean, policy_us_p95=p_p95, policy_us_p99=p_p99,
        shield_us_mean=s_mean, shield_us_p95=s_p95, shield_us_p99=s_p99,
        total_us_mean=t_mean,  total_us_p95=t_p95,  total_us_p99=t_p99,
    )
