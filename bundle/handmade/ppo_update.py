"""PPO update — builds padded batch from collected episodes, computes GAE,
runs multi-epoch minibatched updates via ppo_update_grads, and steps Adam.
"""

from __future__ import annotations

import numpy as np

from .gae import compute_gae
from .losses import ppo_update_grads


def _build_padded_batch(episodes: list, obs_dim: int, gamma: float,
                        lam: float):
    """Pad all episodes to max length; return tensors + mask. Also computes
    GAE advantages/returns per episode and normalizes advantages across the
    valid positions.
    """
    B = len(episodes)
    T = max(len(e.obs) for e in episodes)
    obs = np.zeros((B, T, obs_dim), dtype=np.float32)
    actions = np.zeros((B, T), dtype=np.int64)
    old_logp = np.zeros((B, T), dtype=np.float32)
    advantages = np.zeros((B, T), dtype=np.float32)
    returns = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, ep in enumerate(episodes):
        Ti = len(ep.obs)
        obs[i, :Ti] = np.array(ep.obs, dtype=np.float32)
        actions[i, :Ti] = np.array(ep.actions, dtype=np.int64)
        old_logp[i, :Ti] = np.array(ep.log_probs, dtype=np.float32)
        adv, ret = compute_gae(ep.rewards, ep.values, ep.dones,
                                gamma=gamma, lam=lam)
        advantages[i, :Ti] = adv
        returns[i, :Ti] = ret
        mask[i, :Ti] = 1.0
    # Normalize advantages across valid positions
    flat_mask = mask.astype(bool)
    av = advantages[flat_mask]
    if av.size > 1:
        advantages[flat_mask] = (av - av.mean()) / (av.std() + 1e-8)
    return obs, actions, old_logp, advantages, returns, mask


def ppo_update(policy, optimizer, episodes: list, obs_dim: int,
               gamma: float = 0.99, lam: float = 0.95,
               clip_eps: float = 0.2, value_coeff: float = 0.5,
               entropy_coeff: float = -1.0, bc_coeff: float = 0.0,
               n_epochs: int = 4, minibatch_size: int = 0,
               bptt_chunk_size: int = 0,
               max_grad_norm: float = 0.5, rng=None) -> dict:
    """Run n_epochs of PPO updates over the collected episodes."""
    obs, actions, old_logp, advantages, returns, mask = _build_padded_batch(
        episodes, obs_dim, gamma, lam)
    B = obs.shape[0]

    if rng is None:
        rng = np.random.default_rng(0)

    metrics_acc = {"policy_loss": 0.0, "value_loss": 0.0,
                   "entropy": 0.0, "approx_kl": 0.0}
    n_steps = 0
    for _ in range(n_epochs):
        if 0 < minibatch_size < B:
            perm = rng.permutation(B)
            slices = [perm[i:i + minibatch_size]
                      for i in range(0, B, minibatch_size)]
        else:
            slices = [np.arange(B)]
        for idx in slices:
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_logp = old_logp[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]
            mb_mask = mask[idx]
            h0 = policy.initial_hidden(len(idx))

            logits, values, cache = policy.forward_sequence(
                mb_obs, h0, bptt_chunk_size=bptt_chunk_size)
            loss, m, dlogits, dvalues = ppo_update_grads(
                logits, values, mb_actions, mb_old_logp, mb_adv, mb_ret,
                mb_mask, clip_eps=clip_eps, value_coeff=value_coeff,
                entropy_coeff=entropy_coeff, bc_coeff=bc_coeff)
            grads = policy.backward_sequence(dlogits, dvalues, cache)
            optimizer.step(grads, max_grad_norm=max_grad_norm)

            for k in metrics_acc:
                metrics_acc[k] += m[k]
            n_steps += 1

    return {k: v / max(n_steps, 1) for k, v in metrics_acc.items()}
