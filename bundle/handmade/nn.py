"""Numpy-only neural network building blocks with hand-written backwards.

Layers/ops here implement forward + backward in matching pairs. Backwards
compute parameter gradients AND propagate dL/dinput so callers can chain.

All tensors are np.float32 unless noted. Backward functions accept a `cache`
returned by forward and return either:
  (dx, {param_name: dparam, ...})    for parameterized layers
  dx                                  for stateless ops

Convention: weight matrices are stored in row-major (out_dim, in_dim) shape
to match torch.nn.Linear.weight semantics so the eventual port is mechanical.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Activations and parameter-free ops
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu_forward(x):
    out = np.maximum(x, 0.0)
    cache = (x > 0).astype(x.dtype)
    return out, cache


def relu_backward(dout, cache):
    return dout * cache


def softmax(logits, axis=-1):
    """Numerically stable softmax."""
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(logits, axis=-1):
    z = logits - logits.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


# ============================================================================
# Linear  (Y = X @ W.T + b)
# ============================================================================

class Linear:
    _param_names = ("W", "b")

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = np.sqrt(1.0 / in_dim)
        self.W = rng.uniform(-k, k, (out_dim, in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.in_dim, self.out_dim = in_dim, out_dim

    def parameters(self):
        return {"W": self.W, "b": self.b}

    def forward(self, x):
        # x: (..., in_dim); out: (..., out_dim)
        out = x @ self.W.T + self.b
        return out, x  # cache = x

    def backward(self, dout, cache):
        x = cache
        x_flat = x.reshape(-1, x.shape[-1])
        dout_flat = dout.reshape(-1, dout.shape[-1])
        dW = dout_flat.T @ x_flat              # (out_dim, in_dim)
        db = dout_flat.sum(axis=0)             # (out_dim,)
        dx = dout @ self.W                     # (..., in_dim)
        return dx, {"W": dW, "b": db}


# ============================================================================
# GRU cell  (matches torch.nn.GRUCell formulas exactly)
#
#   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
#   z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
#   n = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
#   h_new = (1 - z) * n + z * h
#
# Weight layout matches torch:
#   W_ih (3H, input)  = [W_ir; W_iz; W_in]   stacked along axis 0
#   W_hh (3H, H)      = [W_hr; W_hz; W_hn]
#   b_ih (3H,)        = [b_ir, b_iz, b_in]
#   b_hh (3H,)        = [b_hr, b_hz, b_hn]
# ============================================================================

class GRUCell:
    _param_names = ("W_ih", "W_hh", "b_ih", "b_hh")

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        k = np.sqrt(1.0 / hidden_dim)
        self.W_ih = rng.uniform(-k, k, (3 * hidden_dim, input_dim)).astype(np.float32)
        self.W_hh = rng.uniform(-k, k, (3 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b_ih = np.zeros(3 * hidden_dim, dtype=np.float32)
        self.b_hh = np.zeros(3 * hidden_dim, dtype=np.float32)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

    def parameters(self):
        return {"W_ih": self.W_ih, "W_hh": self.W_hh,
                "b_ih": self.b_ih, "b_hh": self.b_hh}

    def forward(self, x, h_prev):
        # x: (B, input_dim), h_prev: (B, H)
        gi = x @ self.W_ih.T + self.b_ih    # (B, 3H)
        gh = h_prev @ self.W_hh.T + self.b_hh
        i_r, i_z, i_n = np.split(gi, 3, axis=-1)
        h_r, h_z, h_n = np.split(gh, 3, axis=-1)
        r = sigmoid(i_r + h_r)
        z = sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)
        h_new = (1.0 - z) * n + z * h_prev
        cache = (x, h_prev, r, z, n, h_n)
        return h_new, cache

    def backward(self, dh_new, cache):
        x, h_prev, r, z, n, h_n = cache
        # h_new = (1-z) n + z h_prev
        dz = dh_new * (h_prev - n)
        dn = dh_new * (1.0 - z)
        dh_prev = dh_new * z                     # one contribution to dh_prev

        # n = tanh(i_n + r * h_n) ; dpre_n = dn * (1 - n^2)
        dpre_n = dn * (1.0 - n * n)
        di_n = dpre_n
        dr = dpre_n * h_n
        dh_n = dpre_n * r

        # z = sigmoid(.) ; dpre_z = dz * z * (1-z)
        dpre_z = dz * z * (1.0 - z)
        # r = sigmoid(.) ; dpre_r = dr * r * (1-r)
        dpre_r = dr * r * (1.0 - r)

        # i_* are split of gi; gi = x @ W_ih.T + b_ih
        dgi = np.concatenate([dpre_r, dpre_z, di_n], axis=-1)   # (B, 3H)
        dgh = np.concatenate([dpre_r, dpre_z, dh_n], axis=-1)

        dW_ih = dgi.T @ x                       # (3H, input)
        db_ih = dgi.sum(axis=0)
        dx = dgi @ self.W_ih                    # (B, input)

        dW_hh = dgh.T @ h_prev                  # (3H, H)
        db_hh = dgh.sum(axis=0)
        dh_prev = dh_prev + (dgh @ self.W_hh)   # accumulate second contribution

        grads = {"W_ih": dW_ih, "W_hh": dW_hh, "b_ih": db_ih, "b_hh": db_hh}
        return dx, dh_prev, grads


class GRU:
    """Sequence wrapper around GRUCell. Does NOT mask internally — the loss
    layer is responsible for masking padded positions out of dout.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0):
        self.cell = GRUCell(input_dim, hidden_dim, seed=seed)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

    def parameters(self):
        return self.cell.parameters()

    def forward_sequence(self, x_seq, h0, bptt_chunk_size: int = 0):
        """If bptt_chunk_size > 0 and < T, only store h at chunk boundaries
        (gradient-checkpointing mode). The backward pass will re-run forward
        chunk-by-chunk to regenerate the per-step caches. Across chunks,
        the hidden-state gradient is detached (truncated BPTT).
        """
        B, T, _ = x_seq.shape
        outputs = np.zeros((B, T, self.hidden_dim), dtype=h0.dtype)
        h = h0
        if bptt_chunk_size <= 0 or bptt_chunk_size >= T:
            caches = []
            for t in range(T):
                h, cache = self.cell.forward(x_seq[:, t], h)
                outputs[:, t] = h
                caches.append(cache)
            return outputs, h, {"mode": "full", "caches": caches}
        # Chunked mode: only save hidden state at chunk boundaries.
        chunk_h0s = [h.copy()]
        for t in range(T):
            h, _ = self.cell.forward(x_seq[:, t], h)
            outputs[:, t] = h
            if (t + 1) % bptt_chunk_size == 0 and (t + 1) < T:
                chunk_h0s.append(h.copy())
        return outputs, h, {
            "mode": "chunked",
            "chunk_h0s": chunk_h0s,
            "x_seq": x_seq,
            "chunk_size": bptt_chunk_size,
        }

    def backward_sequence(self, dout, info):
        if info["mode"] == "full":
            return self._backward_full(dout, info["caches"])
        return self._backward_chunked(dout, info)

    def _backward_full(self, dout, caches):
        B, T, _ = dout.shape
        dx_seq = np.zeros((B, T, self.cell.input_dim), dtype=dout.dtype)
        dh_prev = np.zeros((B, self.cell.hidden_dim), dtype=dout.dtype)
        tot = {k: 0.0 for k in self.cell._param_names}
        for t in reversed(range(T)):
            dh = dout[:, t] + dh_prev
            dx, dh_prev, g = self.cell.backward(dh, caches[t])
            dx_seq[:, t] = dx
            for k in tot:
                tot[k] = tot[k] + g[k]
        return dx_seq, dh_prev, tot

    def _backward_chunked(self, dout, info):
        chunk_h0s = info["chunk_h0s"]
        x_seq = info["x_seq"]
        chunk_size = info["chunk_size"]
        B, T, _ = dout.shape
        dx_seq = np.zeros((B, T, self.cell.input_dim), dtype=dout.dtype)
        tot = {k: 0.0 for k in self.cell._param_names}
        n_chunks = (T + chunk_size - 1) // chunk_size
        for ci in reversed(range(n_chunks)):
            t0 = ci * chunk_size
            t1 = min(t0 + chunk_size, T)
            # Re-run forward for this chunk to get per-step caches.
            h = chunk_h0s[ci]
            chunk_caches = [None] * (t1 - t0)
            for local in range(t1 - t0):
                h, c = self.cell.forward(x_seq[:, t0 + local], h)
                chunk_caches[local] = c
            # Backward through this chunk with detached boundary (truncated BPTT).
            dh_prev = np.zeros((B, self.cell.hidden_dim), dtype=dout.dtype)
            for local in reversed(range(t1 - t0)):
                dh = dout[:, t0 + local] + dh_prev
                dx, dh_prev, g = self.cell.backward(dh, chunk_caches[local])
                dx_seq[:, t0 + local] = dx
                for k in tot:
                    tot[k] = tot[k] + g[k]
            # chunk_caches goes out of scope here -> freed
        # No gradient propagates past first chunk's boundary.
        dh0 = np.zeros((B, self.cell.hidden_dim), dtype=dout.dtype)
        return dx_seq, dh0, tot
