"""Handmade memoryless actor-critic for certified reduced MDPs.

Architecture matches the PyTorch reduced discrete policy shape:

    augmented_obs -> Linear -> tanh -> Linear -> tanh
                  -> policy logits / value

There is deliberately no recurrent state. The ``hidden`` arguments are dummy
compatibility hooks so the existing handmade PPO/oracle code can call this
policy through the same interface as the handmade GRU policy.
"""

from __future__ import annotations

import numpy as np

from clarity.training.recurrent.nn import Linear


def _tanh_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.tanh(x)
    return out, out


def _tanh_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return dout * (1.0 - cache * cache)


class MLPActorCritic:
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.fc1 = Linear(obs_dim, hidden_dim, seed=int(rng.integers(2**31)))
        self.fc2 = Linear(hidden_dim, hidden_dim, seed=int(rng.integers(2**31)))
        self.actor = Linear(hidden_dim, n_actions, seed=int(rng.integers(2**31)))
        self.value = Linear(hidden_dim, 1, seed=int(rng.integers(2**31)))
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

    def parameters(self) -> dict[str, np.ndarray]:
        out = {}
        for prefix, mod in [
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("actor", self.actor),
            ("value", self.value),
        ]:
            for key, val in mod.parameters().items():
                out[f"{prefix}.{key}"] = val
        return out

    def _flatten_grads(self, grads_per_module: dict[str, dict[str, np.ndarray]]
                       ) -> dict[str, np.ndarray]:
        flat = {}
        for prefix, grads in grads_per_module.items():
            for key, val in grads.items():
                flat[f"{prefix}.{key}"] = val.astype(np.float32, copy=False)
        return flat

    def initial_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, 0), dtype=np.float32)

    def forward_sequence(self, obs_seq: np.ndarray, h0: np.ndarray | None = None,
                         bptt_chunk_size: int = 0):
        del h0, bptt_chunk_size
        z1, fc1_cache = self.fc1.forward(obs_seq)
        h1, tanh1_cache = _tanh_forward(z1)
        z2, fc2_cache = self.fc2.forward(h1)
        h2, tanh2_cache = _tanh_forward(z2)
        logits, actor_cache = self.actor.forward(h2)
        values, value_cache = self.value.forward(h2)
        cache = {
            "fc1_cache": fc1_cache,
            "tanh1_cache": tanh1_cache,
            "fc2_cache": fc2_cache,
            "tanh2_cache": tanh2_cache,
            "actor_cache": actor_cache,
            "value_cache": value_cache,
        }
        return logits, values[..., 0], cache

    def backward_sequence(self, dlogits: np.ndarray, dvalues: np.ndarray,
                          cache: dict):
        dvalues_unsq = dvalues[..., None]
        dh_from_value, value_grads = self.value.backward(
            dvalues_unsq, cache["value_cache"])
        dh_from_actor, actor_grads = self.actor.backward(
            dlogits, cache["actor_cache"])
        dh2 = dh_from_value + dh_from_actor
        dz2 = _tanh_backward(dh2, cache["tanh2_cache"])
        dh1, fc2_grads = self.fc2.backward(dz2, cache["fc2_cache"])
        dz1 = _tanh_backward(dh1, cache["tanh1_cache"])
        _, fc1_grads = self.fc1.backward(dz1, cache["fc1_cache"])
        return self._flatten_grads({
            "fc1": fc1_grads,
            "fc2": fc2_grads,
            "actor": actor_grads,
            "value": value_grads,
        })

    def step(self, obs: np.ndarray, h_prev: np.ndarray | None = None):
        del h_prev
        z1 = obs @ self.fc1.W.T + self.fc1.b
        h1 = np.tanh(z1)
        z2 = h1 @ self.fc2.W.T + self.fc2.b
        h2 = np.tanh(z2)
        logits = h2 @ self.actor.W.T + self.actor.b
        value = h2 @ self.value.W.T + self.value.b
        return logits, value[..., 0], self.initial_hidden(obs.shape[0])
