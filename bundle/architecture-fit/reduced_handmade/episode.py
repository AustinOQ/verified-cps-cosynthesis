"""Episode collection and evaluation for reduced handmade policies."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    policy_us: list = field(default_factory=list)
    shield_us: list = field(default_factory=list)
    total_us: list = field(default_factory=list)
    outcome: str = ""
    violations: list = field(default_factory=list)
    safety_viol: bool = False


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
    policy_us_mean: float
    policy_us_p95: float
    policy_us_p99: float
    shield_us_mean: float
    shield_us_p95: float
    shield_us_p99: float
    total_us_mean: float
    total_us_p95: float
    total_us_p99: float


def _safety_violated(statuses: dict) -> bool:
    return any(
        entry.get("kind") == "Prohibition" and not entry.get("status", True)
        for entry in statuses.values()
    )


def collect_episode(env, composite, rng=None, greedy: bool = False) -> Episode:
    obs = env.reset()
    hidden = composite.policy.initial_hidden(1)
    ep = Episode()
    done = False
    reward = 0.0
    info = {}
    while not done:
        obs_norm = obs[None, :].astype(np.float32, copy=False)
        raw_obs = env._twin._model_inputs
        (final_action, log_prob, value, hidden, overridden, _,
         timing_us) = composite.act(obs_norm, hidden, raw_obs,
                                    greedy=greedy, rng=rng)
        next_obs, reward, done, info = env.step(final_action)
        ep.obs.append(obs)
        ep.actions.append(final_action)
        ep.rewards.append(reward)
        ep.values.append(value)
        ep.log_probs.append(log_prob)
        ep.dones.append(bool(done))
        ep.overrides.append(bool(overridden))
        ep.policy_us.append(timing_us["policy_us"])
        ep.shield_us.append(timing_us["shield_us"])
        ep.total_us.append(timing_us["total_us"])
        obs = next_obs

    ep.outcome = (
        "SUCCESS" if reward > 0 else ("VIOLATION" if reward < 0 else "TRUNCATED")
    )
    statuses = info.get("statuses", {})
    ep.violations = [
        name for name, entry in statuses.items()
        if not entry.get("status", True)
    ]
    ep.safety_viol = _safety_violated(statuses)
    return ep


def _latency_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def evaluate(env_factory, composite, n_episodes: int, rng=None,
             greedy: bool = True) -> RolloutSummary:
    n_succ = 0
    n_viol = 0
    n_trunc = 0
    n_safety = 0
    total_steps = 0
    total_overrides = 0
    total_reward = 0.0
    all_policy = []
    all_shield = []
    all_total = []
    env = env_factory()
    try:
        for _ in range(n_episodes):
            ep = collect_episode(env, composite, rng=rng, greedy=greedy)
            if ep.outcome == "SUCCESS":
                n_succ += 1
            elif ep.outcome == "VIOLATION":
                n_viol += 1
            else:
                n_trunc += 1
            n_safety += int(ep.safety_viol)
            total_steps += len(ep.actions)
            total_overrides += sum(ep.overrides)
            total_reward += sum(ep.rewards)
            all_policy.extend(ep.policy_us)
            all_shield.extend(ep.shield_us)
            all_total.extend(ep.total_us)
    finally:
        env.close()

    p_mean, p_p95, p_p99 = _latency_stats(all_policy)
    s_mean, s_p95, s_p99 = _latency_stats(all_shield)
    t_mean, t_p95, t_p99 = _latency_stats(all_total)
    return RolloutSummary(
        n_episodes=n_episodes,
        n_steps=total_steps,
        success_rate=n_succ / max(n_episodes, 1),
        violation_rate=n_viol / max(n_episodes, 1),
        truncated_rate=n_trunc / max(n_episodes, 1),
        pooled_override_rate=total_overrides / max(total_steps, 1),
        mean_episode_steps=total_steps / max(n_episodes, 1),
        mean_reward=total_reward / max(n_episodes, 1),
        safety_violation_rate=n_safety / max(n_episodes, 1),
        policy_us_mean=p_mean,
        policy_us_p95=p_p95,
        policy_us_p99=p_p99,
        shield_us_mean=s_mean,
        shield_us_p95=s_p95,
        shield_us_p99=s_p99,
        total_us_mean=t_mean,
        total_us_p95=t_p95,
        total_us_p99=t_p99,
    )

