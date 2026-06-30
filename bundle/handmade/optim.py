"""Adam optimizer with optional global gradient clipping."""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, params: dict, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        """params: {name: ndarray}. Buffers are stored by name."""
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: dict, max_grad_norm: float = 0.0):
        """In-place update. grads must have the same keys as self.params."""
        if max_grad_norm > 0.0:
            total_norm_sq = sum(float((g * g).sum()) for g in grads.values())
            total_norm = np.sqrt(total_norm_sq)
            if total_norm > max_grad_norm:
                scale = max_grad_norm / (total_norm + 1e-8)
                grads = {k: g * scale for k, g in grads.items()}
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k in self.params:
            g = grads[k].astype(np.float32)
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            self.params[k] -= (self.lr * m_hat /
                                (np.sqrt(v_hat) + self.eps)).astype(np.float32)
