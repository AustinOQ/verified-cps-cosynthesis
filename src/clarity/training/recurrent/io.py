"""Numpy-only checkpoint save/load for RecurrentActorCritic.

Stores all parameters in a single .npz with keys matching policy.parameters().
"""

from __future__ import annotations

import numpy as np


def save_policy(policy, path: str) -> None:
    np.savez(path, **policy.parameters())


def load_policy(policy, path: str) -> None:
    data = np.load(path)
    params = policy.parameters()
    for k, v in params.items():
        if k not in data:
            raise KeyError(f"checkpoint missing param '{k}'")
        v[...] = data[k]
