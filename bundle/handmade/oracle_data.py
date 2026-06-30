"""Oracle data generation — port of rl/train.py:generate_oracle_data plus
the coverage-extension fix. Also class balancing via oversampling.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def generate_oracle_data(iface, env, n_samples: int, max_steps: int = 1000,
                         min_class_count: int = 0,
                         hard_cap: int | None = None):
    """Collect (obs, action) pairs by rolling out the spec oracle.

    If min_class_count > 0, after the baseline n_samples is reached we keep
    sampling until every observed action class has at least min_class_count
    examples (or hard_cap is hit). Mirrors rl/train.py exactly.
    """
    from oracle import spec_oracle   # injected via sys.path by caller

    spec_shield = iface["spec_shield"]
    obs_names = iface["obs_names"]
    if hard_cap is None:
        hard_cap = max(n_samples * 25, 50_000)

    obs_all, act_all = [], []

    def _need_more():
        if len(obs_all) < n_samples:
            return True
        if min_class_count <= 0:
            return False
        c = Counter(act_all)
        return any(v < min_class_count for v in c.values())

    while _need_more() and len(obs_all) < hard_cap:
        norm_obs = env.reset()
        for step in range(max_steps):
            raw_obs = env._twin._model_inputs
            obs_dict = {name: float(raw_obs.get(name, 0)) for name in obs_names}
            discrete = spec_oracle(spec_shield, obs_dict)
            obs_all.append(norm_obs.copy())
            act_all.append(discrete)
            if min_class_count <= 0 and len(obs_all) >= n_samples:
                break
            if len(obs_all) >= hard_cap:
                break
            norm_obs, reward, done, info = env.step(discrete)
            if done:
                break

    if min_class_count <= 0:
        obs_arr = np.array(obs_all[:n_samples], dtype=np.float32)
        act_arr = np.array(act_all[:n_samples], dtype=np.int64)
    else:
        obs_arr = np.array(obs_all, dtype=np.float32)
        act_arr = np.array(act_all, dtype=np.int64)
    return obs_arr, act_arr


def balance_classes(obs, acts, seed: int = 0):
    """Oversample minority action classes up to the majority class count.
    Returns shuffled (obs, acts) arrays with all classes equal-sized.
    """
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(acts, return_counts=True)
    target = counts.max()
    obs_chunks = [obs]
    act_chunks = [acts]
    for cls, cnt in zip(unique.tolist(), counts.tolist()):
        if cnt >= target:
            continue
        idx = np.where(acts == cls)[0]
        n_needed = int(target) - int(cnt)
        sampled = rng.choice(idx, size=n_needed, replace=True)
        obs_chunks.append(obs[sampled])
        act_chunks.append(acts[sampled])
    obs_bal = np.concatenate(obs_chunks, axis=0)
    act_bal = np.concatenate(act_chunks, axis=0)
    perm = rng.permutation(len(obs_bal))
    return obs_bal[perm], act_bal[perm]
