"""Oracle CE pretraining loop — supervised cloning of spec_oracle's labels
on the oracle data. Mirrors rl/train.py:train_oracle but with our numpy
policy and Adam.

Treats the dataset as flat (B, obs_dim) samples — uses GRU with T=1 (so
the hidden state propagates within a sample, not across the dataset).
This matches the torch version exactly: it just calls forward_policy with
h0 = initial_hidden(batch) and bs single-step batches.
"""

from __future__ import annotations

import time

import numpy as np

from .losses import cross_entropy
from .optim import Adam


def train_oracle(policy, obs_data: np.ndarray, action_data: np.ndarray,
                 batch_size: int = 1024, n_epochs: int = 30,
                 lr: float = 1e-3, seed: int = 0, verbose: bool = True):
    """In-place updates policy's parameters via CE on (obs, action) pairs.

    obs_data:    (N, obs_dim) float32
    action_data: (N,)         int64
    """
    rng = np.random.default_rng(seed)
    n_samples = obs_data.shape[0]
    opt = Adam(policy.parameters(), lr=lr)

    history = []
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        perm = rng.permutation(n_samples)
        total_loss = 0.0
        correct = 0
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            b_obs = obs_data[idx]                # (B, obs_dim)
            b_act = action_data[idx]             # (B,)
            B = b_obs.shape[0]
            # Treat each sample as a length-1 sequence so the GRU passes
            # through. h0 starts at zero — same as torch oracle phase.
            obs_seq = b_obs[:, None, :]          # (B, 1, obs_dim)
            h0 = policy.initial_hidden(B)
            logits, values, cache = policy.forward_sequence(obs_seq, h0)
            flat_logits = logits[:, 0, :]         # (B, n_actions)
            loss, dflat = cross_entropy(flat_logits, b_act)
            dlogits = dflat[:, None, :]           # (B, 1, n_actions)
            dvalues = np.zeros_like(values)
            grads = policy.backward_sequence(dlogits, dvalues, cache)
            opt.step(grads, max_grad_norm=1.0)
            total_loss += loss
            correct += int((flat_logits.argmax(axis=-1) == b_act).sum())
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        acc = correct / n_samples
        history.append({"epoch": epoch, "loss": mean_loss, "acc": acc,
                        "elapsed_seconds": time.time() - t0})
        if verbose:
            print(f"  [Oracle] epoch {epoch:2d}/{n_epochs} | "
                  f"loss={mean_loss:.4f} | acc={acc:.1%}")
    return history
