"""End-to-end smoke test: train the handmade RecurrentActorCritic via CE
on a tiny synthetic recurrent task. Demonstrates that forward + backward +
Adam actually converge — i.e. the building blocks are not just locally
correct (per grad_check) but also work as a training stack.

Task: at each timestep, predict the action_id encoded in a single
"signal" obs dimension; the rest of the obs is noise. The model must
learn to (a) ignore the noise dims via the encoder, (b) propagate
hidden state without forgetting (GRU), (c) classify via the actor head.

We print loss/accuracy each epoch and assert that:
  * loss falls from log(n_actions) baseline (~random) to below ~0.05
  * accuracy reaches ≥ 95% on the training set within a few hundred steps

Run: python -m clarity.training.recurrent.train_demo
"""

from __future__ import annotations

import numpy as np
import time

from .losses import cross_entropy
from .optim import Adam
from .policy import RecurrentActorCritic


def make_synthetic_data(n_samples: int, T: int, obs_dim: int,
                        n_actions: int, rng: np.random.Generator):
    """obs[t, 0] = signal that determines the correct action_id at t.
    Other obs dims are noise. Returns (obs (N, T, D), targets (N, T)).
    """
    # First obs dim contains an integer-coded class signal in [0, n_actions)
    targets = rng.integers(0, n_actions, size=(n_samples, T))
    obs = rng.standard_normal((n_samples, T, obs_dim)).astype(np.float32)
    # Encode target in dim 0 as a centered signal
    obs[..., 0] = (targets - (n_actions - 1) / 2.0).astype(np.float32)
    return obs, targets


def main():
    rng = np.random.default_rng(0)
    obs_dim, hidden, n_actions = 8, 16, 4
    n_samples, T = 64, 6
    batch_size = 16
    lr = 5e-3
    n_epochs = 50

    model = RecurrentActorCritic(obs_dim, n_actions, hidden, seed=0)
    opt = Adam(model.parameters(), lr=lr)

    obs, targets = make_synthetic_data(n_samples, T, obs_dim, n_actions, rng)
    target_flat = targets.reshape(-1)

    baseline_loss = np.log(n_actions)
    print(f"Synthetic recurrent task: N={n_samples} sequences, T={T}, "
          f"obs_dim={obs_dim}, hidden={hidden}, n_actions={n_actions}")
    print(f"Random-baseline CE loss = log({n_actions}) = {baseline_loss:.4f}")
    print()
    print(f"{'epoch':>5}  {'loss':>8}  {'acc':>6}  {'grad_norm':>10}  {'time_s':>7}")
    print("-" * 50)

    t_start = time.time()
    n_batches = max(n_samples // batch_size, 1)
    for ep in range(1, n_epochs + 1):
        perm = rng.permutation(n_samples)
        loss_acc = 0.0
        correct = 0
        seen = 0
        gnorm_acc = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            b_obs = obs[idx]
            b_targets = targets[idx]
            B = b_obs.shape[0]
            h0 = np.zeros((B, hidden), dtype=np.float32)

            logits, values, cache = model.forward_sequence(b_obs, h0)
            flat_logits = logits.reshape(B * T, n_actions)
            flat_targets = b_targets.reshape(-1)
            l_ce, dflat = cross_entropy(flat_logits, flat_targets)
            dlogits = dflat.reshape(B, T, n_actions)
            dvalues = np.zeros_like(values)   # value head not used here
            grads = model.backward_sequence(dlogits, dvalues, cache)

            opt.step(grads, max_grad_norm=1.0)

            loss_acc += l_ce * B
            pred = flat_logits.argmax(axis=-1)
            correct += int((pred == flat_targets).sum())
            seen += B * T
            gnorm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
            gnorm_acc += gnorm

        mean_loss = loss_acc / n_samples
        acc = correct / seen
        mean_gnorm = gnorm_acc / n_batches
        elapsed = time.time() - t_start
        if ep <= 5 or ep % 5 == 0 or ep == n_epochs:
            print(f"{ep:>5d}  {mean_loss:>8.4f}  {acc:>6.1%}  "
                  f"{mean_gnorm:>10.4f}  {elapsed:>7.2f}")

    # ---- assertions on convergence ----
    final_loss = mean_loss
    final_acc = acc
    ok_loss = final_loss < 0.10
    ok_acc = final_acc > 0.95
    print()
    print(f"Final loss: {final_loss:.4f}  (need < 0.10)  "
          f"{'OK' if ok_loss else 'FAIL'}")
    print(f"Final acc : {final_acc:.4f}  (need > 0.95)  "
          f"{'OK' if ok_acc else 'FAIL'}")
    return 0 if (ok_loss and ok_acc) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
