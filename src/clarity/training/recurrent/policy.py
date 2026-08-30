"""RecurrentActorCritic in numpy — mirror of rl/model.py for compatibility,
but built from the handmade nn.* layers with hand-written backwards.

Architecture:
   obs -> encoder (Linear)  -> relu
       -> GRU (single-layer) -> hidden seq
       -> actor head (Linear -> logits)
       -> value head (Linear -> 1)
"""

from __future__ import annotations

import numpy as np

from .nn import Linear, GRU, relu_forward, relu_backward


class RecurrentActorCritic:
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        # Use distinct seeds so weight matrices don't share patterns
        self.encoder = Linear(obs_dim, hidden_dim, seed=int(rng.integers(2**31)))
        self.gru = GRU(hidden_dim, hidden_dim, seed=int(rng.integers(2**31)))
        self.actor = Linear(hidden_dim, n_actions, seed=int(rng.integers(2**31)))
        self.value = Linear(hidden_dim, 1, seed=int(rng.integers(2**31)))
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.obs_dim = obs_dim

    # ---- parameter management ----
    def parameters(self) -> dict:
        out = {}
        for prefix, mod in [("encoder", self.encoder), ("gru.cell", self.gru.cell),
                            ("actor", self.actor), ("value", self.value)]:
            for k, v in mod.parameters().items():
                out[f"{prefix}.{k}"] = v
        return out

    def assign_grads(self, grads_per_module: dict) -> dict:
        """Re-flatten {module_prefix: {name: dparam}} into a single dict
        whose keys match self.parameters().
        """
        flat = {}
        for prefix, sub in grads_per_module.items():
            for k, v in sub.items():
                flat[f"{prefix}.{k}"] = v
        return flat

    def initial_hidden(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_dim), dtype=np.float32)

    # ---- forward (sequence) ----
    def forward_sequence(self, obs_seq: np.ndarray, h0: np.ndarray,
                         bptt_chunk_size: int = 0):
        """obs_seq: (B, T, obs_dim), h0: (B, H). Returns logits, values, cache.
        logits: (B, T, n_actions); values: (B, T).

        If bptt_chunk_size > 0, the GRU operates in gradient-checkpointed +
        truncated-BPTT mode: only h at chunk boundaries is kept after forward;
        backward re-runs the GRU forward chunk-by-chunk and the hidden-state
        gradient is detached across boundaries.
        """
        enc_pre, enc_cache = self.encoder.forward(obs_seq)
        enc, relu_cache = relu_forward(enc_pre)
        gru_out, h_final, gru_info = self.gru.forward_sequence(
            enc, h0, bptt_chunk_size=bptt_chunk_size)
        logits, actor_cache = self.actor.forward(gru_out)
        values, value_cache = self.value.forward(gru_out)
        values = values[..., 0]                      # (B, T)
        cache = {
            "enc_cache": enc_cache,
            "relu_cache": relu_cache,
            "gru_info": gru_info,
            "actor_cache": actor_cache,
            "value_cache": value_cache,
        }
        return logits, values, cache

    # ---- backward (sequence) ----
    def backward_sequence(self, dlogits: np.ndarray, dvalues: np.ndarray,
                          cache: dict):
        """dlogits: (B, T, n_actions); dvalues: (B, T). Returns dict of
        per-module gradients with keys matching parameters().
        """
        # Value head backward
        dvalues_unsq = dvalues[..., None]    # (B, T, 1)
        dgru_from_value, value_grads = self.value.backward(
            dvalues_unsq, cache["value_cache"])
        # Actor head backward
        dgru_from_actor, actor_grads = self.actor.backward(
            dlogits, cache["actor_cache"])
        # Sum incoming gradients into the GRU output
        dgru_out = dgru_from_value + dgru_from_actor   # (B, T, H)
        # GRU backward through time (full or chunked-checkpointed)
        denc, _, gru_grads = self.gru.backward_sequence(
            dgru_out, cache["gru_info"])
        # ReLU backward
        denc_pre = relu_backward(denc, cache["relu_cache"])
        # Encoder backward
        _, encoder_grads = self.encoder.backward(denc_pre, cache["enc_cache"])

        grads_per_module = {
            "encoder": encoder_grads,
            "gru.cell": gru_grads,
            "actor": actor_grads,
            "value": value_grads,
        }
        return self.assign_grads(grads_per_module)

    # ---- single-step forward (used in env rollouts) ----
    def step(self, obs: np.ndarray, h_prev: np.ndarray):
        """obs: (B, obs_dim) -- here typically B=1 during rollout."""
        enc_pre = obs @ self.encoder.W.T + self.encoder.b
        enc = np.maximum(enc_pre, 0.0)
        h_new, _ = self.gru.cell.forward(enc, h_prev)
        logits = h_new @ self.actor.W.T + self.actor.b
        value = h_new @ self.value.W.T + self.value.b
        return logits, value[..., 0], h_new
