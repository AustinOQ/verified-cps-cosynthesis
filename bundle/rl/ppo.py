"""
Recurrent PPO with episode-level sequence training.

Handles variable-length episodes by padding and masking.
Trains the GRU over full episode sequences so it learns temporal patterns.
"""

import torch
import torch.nn as nn
import numpy as np

from model import RecurrentActorCritic


def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                gamma: float = 0.99, lam: float = 0.95) -> tuple[list[float], list[float]]:
    """Compute Generalized Advantage Estimation for one episode.

    Returns:
        advantages: per-step advantage estimates
        returns: per-step discounted returns (for value target)
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T

    last_gae = 0.0
    last_value = 0.0  # Bootstrap from terminal state = 0

    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else last_value
        next_non_terminal = 0.0 if dones[t] else 1.0

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]

    return advantages, returns


class EpisodeBuffer:
    """Stores transitions from multiple episodes for batch training."""

    def __init__(self):
        self.episodes = []

    def add_episode(self, obs: list, actions: list, rewards: list,
                    values: list, log_probs: list, dones: list):
        self.episodes.append({
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "dones": dones,
        })

    def clear(self):
        self.episodes = []

    def build_batch(self, gamma: float, lam: float, device: torch.device):
        """Pad episodes to same length and compute GAE.

        Returns dict of tensors:
            obs:        (batch, max_len, obs_dim)
            actions:    (batch, max_len)
            old_logp:   (batch, max_len)
            advantages: (batch, max_len)
            returns:    (batch, max_len)
            mask:       (batch, max_len)  — 1 for real steps, 0 for padding
        """
        batch_size = len(self.episodes)
        max_len = max(len(ep["obs"]) for ep in self.episodes)
        obs_dim = len(self.episodes[0]["obs"][0])

        obs = np.zeros((batch_size, max_len, obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, max_len), dtype=np.int64)
        old_logp = np.zeros((batch_size, max_len), dtype=np.float32)
        advantages = np.zeros((batch_size, max_len), dtype=np.float32)
        returns = np.zeros((batch_size, max_len), dtype=np.float32)
        mask = np.zeros((batch_size, max_len), dtype=np.float32)

        for i, ep in enumerate(self.episodes):
            T = len(ep["obs"])
            adv, ret = compute_gae(ep["rewards"], ep["values"], ep["dones"],
                                   gamma, lam)

            obs[i, :T] = np.array(ep["obs"])
            actions[i, :T] = np.array(ep["actions"])
            old_logp[i, :T] = np.array(ep["log_probs"])
            advantages[i, :T] = np.array(adv)
            returns[i, :T] = np.array(ret)
            mask[i, :T] = 1.0

        # Normalize advantages across the entire batch
        valid = mask.astype(bool)
        adv_valid = advantages[valid]
        if len(adv_valid) > 1:
            advantages[valid] = (adv_valid - adv_valid.mean()) / (adv_valid.std() + 1e-8)

        return {
            "obs": torch.tensor(obs, device=device),
            "actions": torch.tensor(actions, device=device),
            "old_logp": torch.tensor(old_logp, device=device),
            "advantages": torch.tensor(advantages, device=device),
            "returns": torch.tensor(returns, device=device),
            "mask": torch.tensor(mask, device=device),
        }


class RecurrentPPO:
    """PPO trainer for recurrent actor-critic."""

    def __init__(self, model: RecurrentActorCritic,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 entropy_coeff: float = 0.01,
                 entropy_coeff_end: float = None,
                 total_updates: int = 1,
                 value_coeff: float = 0.5,
                 n_epochs: int = 4,
                 max_grad_norm: float = 0.5,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coeff_start = entropy_coeff
        self.entropy_coeff_end = entropy_coeff_end if entropy_coeff_end is not None else entropy_coeff
        self.total_updates = total_updates
        self.update_count = 0
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

    def update(self, buffer: EpisodeBuffer,
               use_gradient_checkpoint: bool = False,
               minibatch_size: int = 0,
               bptt_chunk_size: int = 0,
               bc_aux_coeff: float = 0.0) -> dict:
        """Run PPO update on collected episodes.

        Optional memory-savers (default off — disabling restores the
        original behavior exactly):
          use_gradient_checkpoint:  if True, the model's forward_sequence
            re-runs the encoder+GRU body during backward instead of caching
            activations. Trades ~30% compute for ~50-70% activation memory.
          minibatch_size:  if > 0 and < batch_size, split the collected
            episodes into chunks of this size and do one forward+backward+
            optimizer step per chunk inside each epoch. Cuts per-step
            activation memory linearly with the minibatch size.
          bptt_chunk_size:  if > 0, use truncated BPTT inside
            forward_sequence — detach GRU hidden state every `chunk_size`
            timesteps so the backward graph stays within-chunk. Cuts per-
            step activation memory by roughly (max_seq_len / chunk_size).
            Loses gradient flow for credit assignment beyond `chunk_size`
            timesteps. No-op when seq_len <= chunk_size.

        Returns dict of loss metrics for logging.
        """
        # Linear entropy decay
        frac = min(self.update_count / max(self.total_updates, 1), 1.0)
        self.entropy_coeff = (self.entropy_coeff_start
                              + frac * (self.entropy_coeff_end - self.entropy_coeff_start))
        self.update_count += 1

        batch = buffer.build_batch(self.gamma, self.lam, self.device)

        obs = batch["obs"]              # (B, T, obs_dim)
        actions = batch["actions"]      # (B, T)
        old_logp = batch["old_logp"]    # (B, T)
        advantages = batch["advantages"]  # (B, T)
        returns = batch["returns"]      # (B, T)
        mask = batch["mask"]            # (B, T)

        B = obs.shape[0]

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_minibatch_steps = 0  # how many optimizer steps we actually took

        for _ in range(self.n_epochs):
            # Decide minibatch index slices for this epoch.
            if minibatch_size and 0 < minibatch_size < B:
                perm = torch.randperm(B, device=self.device)
                mb_slices = [perm[i:i + minibatch_size]
                             for i in range(0, B, minibatch_size)]
            else:
                mb_slices = [torch.arange(B, device=self.device)]

            for mb_idx in mb_slices:
                mb_obs = obs.index_select(0, mb_idx)
                mb_actions = actions.index_select(0, mb_idx)
                mb_old_logp = old_logp.index_select(0, mb_idx)
                mb_adv = advantages.index_select(0, mb_idx)
                mb_ret = returns.index_select(0, mb_idx)
                mb_mask = mask.index_select(0, mb_idx)

                h0_mb = self.model.initial_hidden(mb_obs.shape[0]).to(
                    self.device)

                logits, values = self.model.forward_sequence(
                    mb_obs, h0_mb, mb_mask,
                    use_gradient_checkpoint=use_gradient_checkpoint,
                    bptt_chunk_size=bptt_chunk_size)
                values = values.squeeze(-1)  # (b, T)

                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio,
                                    1 - self.clip_eps,
                                    1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2)

                value_loss = (values - mb_ret).pow(2)
                entropy_bonus = -entropy

                # Behavior-cloning auxiliary loss: pull the policy distribution
                # toward the action that was actually executed (which is the
                # shield-corrected action). Independent of advantages, so it
                # acts as a constant force in parameter space toward the
                # shield-aligned attractor. λ=0 disables.
                bc_loss = -new_logp

                loss = (policy_loss
                        + self.value_coeff * value_loss
                        + self.entropy_coeff * entropy_bonus
                        + bc_aux_coeff * bc_loss)
                mask_sum = mb_mask.sum().clamp_min(1.0)
                loss = (loss * mb_mask).sum() / mask_sum

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    metrics["policy_loss"] += (policy_loss * mb_mask).sum().item() / mask_sum.item()
                    metrics["value_loss"] += (value_loss * mb_mask).sum().item() / mask_sum.item()
                    metrics["entropy"] += (entropy * mb_mask).sum().item() / mask_sum.item()
                n_minibatch_steps += 1

        for k in metrics:
            metrics[k] /= max(n_minibatch_steps, 1)

        return metrics
