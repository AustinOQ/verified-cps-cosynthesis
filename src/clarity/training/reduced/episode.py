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

    def compact(self) -> "Episode":
        """Pack step data into arrays before cross-process transfer."""
        return Episode(
            obs=np.asarray(self.obs, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            rewards=np.asarray(self.rewards, dtype=np.float64),
            values=np.asarray(self.values, dtype=np.float64),
            log_probs=np.asarray(self.log_probs, dtype=np.float64),
            dones=np.asarray(self.dones, dtype=np.bool_),
            overrides=np.asarray(self.overrides, dtype=np.bool_),
            policy_us=np.asarray(self.policy_us, dtype=np.float64),
            shield_us=np.asarray(self.shield_us, dtype=np.float64),
            total_us=np.asarray(self.total_us, dtype=np.float64),
            outcome=self.outcome,
            violations=list(self.violations),
            safety_viol=bool(self.safety_viol),
        )


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


@dataclass
class RolloutMeasurements:
    """Mergeable episode measurements used by serial and process collectors."""

    n_episodes: int
    n_succ: int
    n_viol: int
    n_trunc: int
    n_safety: int
    total_steps: int
    total_overrides: int
    total_reward: float
    policy_us: np.ndarray
    shield_us: np.ndarray
    total_us: np.ndarray


def collect_episode(env, composite, rng=None, greedy: bool = False,
                    reset_seed: int | None = None) -> Episode:
    obs = env.reset(seed=reset_seed)
    hidden = composite.policy.initial_hidden(1)
    ep = Episode()
    done = False
    reward = 0.0
    info = {}
    requirement_violated = False
    while not done:
        obs_norm = obs[None, :].astype(np.float32, copy=False)
        raw_obs = env._twin._model_inputs
        (final_action, log_prob, value, hidden, overridden, _,
         timing_us) = composite.act(obs_norm, hidden, raw_obs,
                                    greedy=greedy, rng=rng)
        if not composite.requirement_holds(final_action, raw_obs):
            requirement_violated = True
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
    ep.safety_viol = requirement_violated
    return ep


def _latency_stats(values) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(arr.mean()),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
    )


def measure_episodes(episodes) -> RolloutMeasurements:
    n_succ = 0
    n_viol = 0
    n_trunc = 0
    n_safety = 0
    total_steps = 0
    total_overrides = 0
    total_reward = 0.0
    policy_parts = []
    shield_parts = []
    total_parts = []
    n_episodes = 0
    for ep in episodes:
        n_episodes += 1
        if ep.outcome == "SUCCESS":
            n_succ += 1
        elif ep.outcome == "VIOLATION":
            n_viol += 1
        else:
            n_trunc += 1
        n_safety += int(ep.safety_viol)
        total_steps += len(ep.actions)
        total_overrides += int(np.asarray(ep.overrides, dtype=np.int64).sum())
        total_reward += float(np.asarray(ep.rewards, dtype=np.float64).sum())
        policy_parts.append(np.asarray(ep.policy_us, dtype=np.float64))
        shield_parts.append(np.asarray(ep.shield_us, dtype=np.float64))
        total_parts.append(np.asarray(ep.total_us, dtype=np.float64))

    def joined(parts) -> np.ndarray:
        nonempty = [part for part in parts if part.size]
        if not nonempty:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(nonempty)

    return RolloutMeasurements(
        n_episodes=n_episodes,
        n_succ=n_succ,
        n_viol=n_viol,
        n_trunc=n_trunc,
        n_safety=n_safety,
        total_steps=total_steps,
        total_overrides=total_overrides,
        total_reward=total_reward,
        policy_us=joined(policy_parts),
        shield_us=joined(shield_parts),
        total_us=joined(total_parts),
    )


def merge_measurements(parts) -> RolloutMeasurements:
    parts = list(parts)
    if not parts:
        return measure_episodes([])

    def joined(name: str) -> np.ndarray:
        arrays = [np.asarray(getattr(part, name), dtype=np.float64)
                  for part in parts if getattr(part, name).size]
        if not arrays:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(arrays)

    return RolloutMeasurements(
        n_episodes=sum(part.n_episodes for part in parts),
        n_succ=sum(part.n_succ for part in parts),
        n_viol=sum(part.n_viol for part in parts),
        n_trunc=sum(part.n_trunc for part in parts),
        n_safety=sum(part.n_safety for part in parts),
        total_steps=sum(part.total_steps for part in parts),
        total_overrides=sum(part.total_overrides for part in parts),
        total_reward=sum(part.total_reward for part in parts),
        policy_us=joined("policy_us"),
        shield_us=joined("shield_us"),
        total_us=joined("total_us"),
    )


def summarize_measurements(data: RolloutMeasurements) -> RolloutSummary:
    p_mean, p_p95, p_p99 = _latency_stats(data.policy_us)
    s_mean, s_p95, s_p99 = _latency_stats(data.shield_us)
    t_mean, t_p95, t_p99 = _latency_stats(data.total_us)
    return RolloutSummary(
        n_episodes=data.n_episodes,
        n_steps=data.total_steps,
        success_rate=data.n_succ / max(data.n_episodes, 1),
        violation_rate=data.n_viol / max(data.n_episodes, 1),
        truncated_rate=data.n_trunc / max(data.n_episodes, 1),
        pooled_override_rate=data.total_overrides / max(data.total_steps, 1),
        mean_episode_steps=data.total_steps / max(data.n_episodes, 1),
        mean_reward=data.total_reward / max(data.n_episodes, 1),
        safety_violation_rate=data.n_safety / max(data.n_episodes, 1),
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


def evaluate(env_factory, composite, n_episodes: int, rng=None,
             greedy: bool = True) -> RolloutSummary:
    env = env_factory()
    try:
        measurements = measure_episodes(
            collect_episode(env, composite, rng=rng, greedy=greedy)
            for _ in range(n_episodes)
        )
    finally:
        env.close()
    return summarize_measurements(measurements)
