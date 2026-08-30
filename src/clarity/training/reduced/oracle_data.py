"""Oracle data for buffered reduced-MDP policies."""

from __future__ import annotations

from collections import Counter

import numpy as np


def collect_oracle_episode(iface, env, *, max_steps: int = 5000,
                           reset_seed: int | None = None,
                           sample_limit: int | None = None,
                           include_terminal: bool = False
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Collect one full rule-labeled episode from a seeded environment."""
    from clarity.runtime.oracle import spec_oracle

    spec_shield = iface["spec_shield"]
    obs_names = iface["obs_names"]

    def oracle_action() -> int:
        raw_obs = env._twin._model_inputs
        missing = [name for name in obs_names if name not in raw_obs]
        if missing:
            raise KeyError(f"SysML simulation omitted neural inputs: {missing}")
        obs_dict = {name: float(raw_obs[name]) for name in obs_names}
        return int(spec_oracle(spec_shield, obs_dict))

    obs_all = []
    act_all = []
    obs = env.reset(seed=reset_seed)
    done = False
    steps = 0
    while not done and steps < max_steps:
        if sample_limit is not None and len(obs_all) >= sample_limit:
            break
        action = oracle_action()
        obs_all.append(np.asarray(obs, dtype=np.float32).copy())
        act_all.append(action)
        obs, _, done, _ = env.step(action)
        steps += 1

    if include_terminal and (
        sample_limit is None or len(obs_all) < sample_limit
    ):
        obs_all.append(np.asarray(obs, dtype=np.float32).copy())
        act_all.append(oracle_action())

    return (
        np.asarray(obs_all, dtype=np.float32),
        np.asarray(act_all, dtype=np.int64),
    )


def generate_oracle_data(iface, env, n_samples: int,
                         min_class_count: int = 0,
                         max_steps: int = 5000,
                         max_resets: int = 20000):
    """Collect buffered observations labeled by the exact spec oracle.

    This rolls full oracle episodes and probes the terminal observation, matching
    the coverage-oriented behavior used by the reduced PyTorch runner. Coverage
    is over observed oracle classes; structurally dead actions are not forced.
    """
    obs_all: list[np.ndarray] = []
    act_all: list[int] = []
    by_class: Counter[int] = Counter()

    def need_more() -> bool:
        if len(obs_all) < n_samples:
            return True
        if min_class_count <= 0:
            return False
        return any(count < min_class_count for count in by_class.values())

    resets = 0
    while need_more() and resets < max_resets:
        remaining = None if min_class_count > 0 else n_samples - len(obs_all)
        obs_ep, act_ep = collect_oracle_episode(
            iface,
            env,
            max_steps=max_steps,
            sample_limit=remaining,
            include_terminal=min_class_count > 0,
        )
        resets += 1
        for obs, action in zip(obs_ep, act_ep):
            obs_all.append(obs)
            action = int(action)
            act_all.append(action)
            by_class[action] += 1

    if not obs_all:
        raise RuntimeError("oracle collection produced no samples")
    if need_more():
        raise RuntimeError(
            "oracle coverage target was not reached: "
            f"n={len(obs_all)}, class_counts={dict(by_class)}, "
            f"min_class_count={min_class_count}, max_resets={max_resets}"
        )
    if min_class_count <= 0:
        obs_all = obs_all[:n_samples]
        act_all = act_all[:n_samples]

    obs_arr = np.asarray(obs_all, dtype=np.float32)
    act_arr = np.asarray(act_all, dtype=np.int64)
    return obs_arr, act_arr, dict(sorted(by_class.items())), resets


def balance_classes(obs: np.ndarray, acts: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(acts, return_counts=True)
    target = int(counts.max())
    obs_chunks = [obs]
    act_chunks = [acts]
    for cls, count in zip(unique.tolist(), counts.tolist()):
        if count >= target:
            continue
        idx = np.where(acts == cls)[0]
        sampled = rng.choice(idx, size=target - int(count), replace=True)
        obs_chunks.append(obs[sampled])
        act_chunks.append(acts[sampled])
    out_obs = np.concatenate(obs_chunks, axis=0)
    out_acts = np.concatenate(act_chunks, axis=0)
    perm = rng.permutation(len(out_acts))
    return out_obs[perm], out_acts[perm]
