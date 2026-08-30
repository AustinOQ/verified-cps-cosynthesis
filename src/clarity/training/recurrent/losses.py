"""Loss functions with hand-written gradients. Each returns
(scalar_loss, dlogits_or_dvalues) so callers can chain into backward passes.

Mask convention: mask is (B, T) with 1 for real positions and 0 for padding.
All masked-mean losses divide by mask.sum() (not B*T) for stable scaling.
"""

from __future__ import annotations

import numpy as np

from .nn import softmax


# ============================================================================
# Cross-entropy on logits (oracle pretraining)
# ============================================================================

def cross_entropy(logits, targets):
    """logits: (N, C), targets: (N,) int. Returns (loss_scalar, dlogits).

    Loss = mean over N of -log p[target]. Gradient: (probs - one_hot) / N.
    """
    N = logits.shape[0]
    probs = softmax(logits, axis=-1)
    # -log p_target
    log_p_target = np.log(np.clip(probs[np.arange(N), targets], 1e-30, None))
    loss = -log_p_target.mean()
    dlogits = probs.copy()
    dlogits[np.arange(N), targets] -= 1.0
    dlogits /= N
    return float(loss), dlogits


# ============================================================================
# Value MSE (masked mean)
# ============================================================================

def value_mse(values, returns, mask):
    """values, returns, mask: (B, T). Returns (loss, dvalues)."""
    diff = (values - returns) * mask
    msum = max(mask.sum(), 1.0)
    loss = (diff * diff).sum() / msum
    dvalues = 2.0 * (values - returns) * mask / msum
    return float(loss), dvalues


# ============================================================================
# Categorical log_prob / entropy on a sequence of logits with a mask
# ============================================================================

def categorical_log_prob_and_entropy(logits, actions, mask):
    """logits: (B, T, C), actions: (B, T) int, mask: (B, T).

    Returns dict with:
      log_prob (B, T)  : log p_action at each position
      entropy  (B, T)  : entropy of the categorical at each position
      probs    (B, T, C): softmax for reuse in backward
    """
    probs = softmax(logits, axis=-1)
    log_probs_all = np.log(np.clip(probs, 1e-30, None))
    B, T, C = logits.shape
    idx_b = np.arange(B)[:, None].repeat(T, axis=1)
    idx_t = np.arange(T)[None, :].repeat(B, axis=0)
    log_prob = log_probs_all[idx_b, idx_t, actions]
    # H = -sum_i p_i log p_i (per position)
    entropy = -(probs * log_probs_all).sum(axis=-1)
    return {"log_prob": log_prob, "entropy": entropy,
            "probs": probs, "log_probs_all": log_probs_all}


# ============================================================================
# PPO clipped surrogate + entropy + BC, with gradient w.r.t. logits and values.
#
# Matches rl/ppo.py's update():
#   ratio = exp(new_logp - old_logp)
#   surr1 = ratio * adv
#   surr2 = clip(ratio, 1-eps, 1+eps) * adv
#   policy_loss = -min(surr1, surr2)
#   value_loss  = (values - returns)^2
#   entropy_bonus = -entropy
#   bc_loss = -new_logp
#   loss = mean_masked(policy_loss + value_coeff*value_loss
#                      + entropy_coeff*entropy_bonus + bc_coeff*bc_loss)
# ============================================================================

def ppo_update_grads(logits, values, actions, old_logp, advantages, returns,
                     mask, clip_eps: float, value_coeff: float,
                     entropy_coeff: float, bc_coeff: float):
    """Returns (loss, metrics_dict, dlogits, dvalues).

    All shapes are (B, T, *). The gradients dlogits (B, T, C) and dvalues
    (B, T) are ready to feed into the policy's backward_sequence step.
    """
    B, T, C = logits.shape
    cat = categorical_log_prob_and_entropy(logits, actions, mask)
    new_logp = cat["log_prob"]                  # (B, T)
    entropy = cat["entropy"]                    # (B, T)
    probs = cat["probs"]                        # (B, T, C)
    log_probs_all = cat["log_probs_all"]        # (B, T, C)

    ratio = np.exp(new_logp - old_logp)         # (B, T)
    surr1 = ratio * advantages
    clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratio * advantages
    use_surr1 = (surr1 <= surr2).astype(np.float32)  # mask where min is surr1
    policy_loss_pt = -np.minimum(surr1, surr2)      # (B, T)

    value_loss_pt = (values - returns) ** 2          # (B, T)
    entropy_bonus_pt = -entropy
    bc_loss_pt = -new_logp

    total_pt = (policy_loss_pt
                + value_coeff * value_loss_pt
                + entropy_coeff * entropy_bonus_pt
                + bc_coeff * bc_loss_pt)
    msum = max(mask.sum(), 1.0)
    loss = float((total_pt * mask).sum() / msum)

    # ----- backward -----
    # Each per-position contribution to dL is mask[b,t] / msum.
    # d(policy_loss_pt)/d(new_logp) =
    #   if min == surr1 (not clipped): d(-ratio*adv)/d(new_logp) = -ratio*adv
    #   else (clipped, surr2 dominates): 0  (the clip detaches the gradient)
    dpolicy_dnew_logp = -ratio * advantages * use_surr1
    # d(bc_loss)/d(new_logp) = -1
    dbc_dnew_logp = -np.ones_like(new_logp)
    # d(entropy_bonus)/d(probs) handled via dentropy_dlogits below
    # Combine d(loss)/d(new_logp)
    dnew_logp = (dpolicy_dnew_logp + bc_coeff * dbc_dnew_logp) * mask / msum

    # value gradient: 2*(v-ret) * mask / msum, scaled by value_coeff
    dvalues = value_coeff * 2.0 * (values - returns) * mask / msum

    # Logits gradient from new_logp:
    #   d log p_a / d logit_j = delta(a,j) - p_j
    # so d(loss)/d(logits[b,t,j]) += dnew_logp[b,t] * (one_hot - probs)
    dlogits = -probs * dnew_logp[..., None]
    idx_b = np.arange(B)[:, None].repeat(T, axis=1)
    idx_t = np.arange(T)[None, :].repeat(B, axis=0)
    dlogits[idx_b, idx_t, actions] += dnew_logp

    # Entropy gradient:
    #   dH / d logit_j = -p_j * (log p_j + H)
    # entropy_bonus = -H, so d entropy_bonus / d logit_j = +p_j * (log p_j + H)
    # Coefficient: entropy_coeff. Per-position scaling: mask / msum.
    dent_dlogits = probs * (log_probs_all + entropy[..., None])
    dlogits += (entropy_coeff * dent_dlogits * (mask / msum)[..., None])

    metrics = {
        "policy_loss": float((policy_loss_pt * mask).sum() / msum),
        "value_loss":  float((value_loss_pt  * mask).sum() / msum),
        "entropy":     float((entropy        * mask).sum() / msum),
        "approx_kl":   float(((old_logp - new_logp) * mask).sum() / msum),
    }
    return loss, metrics, dlogits, dvalues
