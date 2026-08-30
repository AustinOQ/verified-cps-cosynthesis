"""Numpy-side certified-buffer environment wrappers.

The wrapped SysML simulator is unchanged. The policy observation is augmented
with the certified finite history:

    current obs ++ past observations ++ past executed actions

The SysML requirement check still reads the simulator's current raw
``_model_inputs``; the buffer only changes what the memoryless learner sees.
"""

from __future__ import annotations

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCH = os.path.dirname(_HERE)
_REPO = os.path.dirname(_ARCH)
_RL = os.path.join(_REPO, "rl")
_SRC = os.path.join(os.path.dirname(_REPO), "src")
for _path in (_RL, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from env import SysMLEnv


class BufferedDiscreteEnv(SysMLEnv):
    """Discrete SysML env with finite history in the policy observation."""

    def __init__(self, model_path: str, dt: float, max_steps: int = 5000,
                 phase: int = 2, rng_seed: int | None = None,
                 n_act: int = 1, n_obs: int = 0):
        super().__init__(model_path, dt=dt, max_steps=max_steps,
                         phase=phase, rng_seed=rng_seed)
        self._base_obs_dim = self.obs_dim
        self._n_act = int(n_act)
        self._n_obs = int(n_obs)
        self._n_actions = self.n_actions
        self.obs_dim = (
            self._base_obs_dim
            + self._n_obs * self._base_obs_dim
            + self._n_act * self._n_actions
        )
        self._obs_hist: list[np.ndarray] = []
        self._act_hist: list[np.ndarray] = []

    @property
    def buffer_spec(self) -> dict[str, int]:
        return {
            "n_obs": self._n_obs,
            "n_act": self._n_act,
            "base_obs_dim": self._base_obs_dim,
            "augmented_obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
        }

    def _onehot(self, action: int) -> np.ndarray:
        out = np.zeros(self._n_actions, dtype=np.float32)
        if 0 <= int(action) < self._n_actions:
            out[int(action)] = 1.0
        return out

    def _augment(self, current_obs: np.ndarray) -> np.ndarray:
        parts = [np.asarray(current_obs, dtype=np.float32)]
        parts.extend(self._obs_hist)
        parts.extend(self._act_hist)
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _push_obs(self, obs: np.ndarray) -> None:
        self._obs_hist = (
            [np.asarray(obs, dtype=np.float32).copy()] + self._obs_hist
        )[:self._n_obs]

    def _push_act(self, action: int) -> None:
        self._act_hist = ([self._onehot(action)] + self._act_hist)[:self._n_act]

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = super().reset(seed=seed)
        self._obs_hist = [
            np.zeros(self._base_obs_dim, dtype=np.float32)
            for _ in range(self._n_obs)
        ]
        self._act_hist = [
            np.zeros(self._n_actions, dtype=np.float32)
            for _ in range(self._n_act)
        ]
        aug = self._augment(obs)
        self._push_obs(obs)
        return aug

    def step(self, action: int):
        self._push_act(action)
        obs, reward, done, info = super().step(action)
        aug = self._augment(obs)
        self._push_obs(obs)
        return aug, reward, done, info
