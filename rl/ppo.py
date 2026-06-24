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

    def update(self, buffer: EpisodeBuffer) -> dict:
        """Run PPO update on collected episodes.

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
        h0 = self.model.initial_hidden(B).to(self.device)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        for _ in range(self.n_epochs):
            logits, values = self.model.forward_sequence(obs, h0, mask)
            # logits: (B, T, n_actions), values: (B, T, 1)
            values = values.squeeze(-1)  # (B, T)

            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(actions)       # (B, T)
            entropy = dist.entropy()                 # (B, T)

            # PPO clipped surrogate
            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2)

            # Value loss (clipped)
            value_loss = (values - returns).pow(2)

            # Entropy bonus
            entropy_bonus = -entropy

            # Combine losses, masked to valid timesteps
            loss = (policy_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_bonus)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"] += (policy_loss * mask).sum().item() / mask.sum().item()
                metrics["value_loss"] += (value_loss * mask).sum().item() / mask.sum().item()
                metrics["entropy"] += (entropy * mask).sum().item() / mask.sum().item()

        for k in metrics:
            metrics[k] /= self.n_epochs

        return metrics
