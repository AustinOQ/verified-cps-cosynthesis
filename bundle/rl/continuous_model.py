"""
Gaussian recurrent actor-critic for CONTINUOUS control, plus the continuous
shielded composite policy.

Counterpart to model.py (Categorical / discrete) and composite_model.py.
Plan A: a plain diagonal Gaussian over the real action(s) — no tanh squash.
Bounding is delegated to the ContinuousShield, which clamps the executed
action onto the safe interval; a plain-Gaussian log_prob is well-defined at
the clamped value, so it composes cleanly with the existing "train on the
executed (shield-corrected) action" scheme.

  head: encoder -> GRU -> mean_head (act_dim) + value_head (1)
  std : a single global learnable log_std parameter (state-independent)
  dist: Normal(mean, std)
  eval action  = mean (the mode / highest-density point)
  train action = sample ~ Normal(mean, std)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianRecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 1,
                 hidden_dim: int = 64, gru_layers: int = 1,
                 init_log_std: float = -0.7):
        # The Gaussian operates in a NORMALIZED action space (~unit scale);
        # the composite multiplies by action_scale to reach physical units.
        # init_log_std = -0.7 -> std ~ 0.5, giving O(1) log-prob gradients on
        # the mean (a force-unit std of ~20 divides those gradients by ~400).
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        # State-independent log-std (init std = exp(init_log_std)).
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch_size, self.hidden_dim)

    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp(min=1e-3, max=1e3)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Single timestep. obs: (B, obs_dim). Returns (Normal, value, hidden)."""
        feats = self.encoder(obs).unsqueeze(1)            # (B, 1, H)
        gru_out, hidden = self.gru(feats, hidden)         # (B, 1, H)
        gru_out = gru_out.squeeze(1)                       # (B, H)
        mean = self.mean_head(gru_out)                    # (B, act_dim)
        value = self.value_head(gru_out)                  # (B, 1)
        std = self.std().expand_as(mean)
        return Normal(mean, std), value, hidden

    def forward_sequence(self, obs_seq: torch.Tensor, hidden: torch.Tensor,
                         mask: torch.Tensor = None):
        """Padded-sequence forward for PPO. Returns (mean, values).

        mean:   (B, T, act_dim)
        values: (B, T, 1)
        The PPO update reconstructs Normal(mean, std) with the global std.
        """
        feats = self.encoder(obs_seq)                     # (B, T, H)
        gru_out, _ = self.gru(feats, hidden)              # (B, T, H)
        mean = self.mean_head(gru_out)
        values = self.value_head(gru_out)
        return mean, values


class CompositeContinuousPolicy(nn.Module):
    """Trainable Gaussian policy + frozen continuous interval shield.

    The policy acts in a normalized space; `action_scale` maps it to physical
    force units for the shield/env. The shield clamps in force units; the
    executed action is converted back to normalized units (`z_exec`) so PPO's
    log-prob/storage stay in the same well-conditioned space as the policy.
    """

    def __init__(self, policy: GaussianRecurrentActorCritic, shield,
                 action_scale: float = 100.0):
        super().__init__()
        self.policy = policy
        self.shield = shield  # ContinuousShield (not an nn.Module)
        self.action_scale = float(action_scale)

    def initial_hidden(self, batch_size: int = 1):
        return self.policy.initial_hidden(batch_size)

    def forward_sequence(self, obs_seq, hidden, mask=None):
        return self.policy.forward_sequence(obs_seq, hidden, mask)

    def act(self, obs_t, hidden, obs_dict, greedy: bool = False):
        """Policy proposes a normalized action; shield clamps in force units.

        Returns (safe_force, z_exec, dist, value, hidden, overridden, clamp_dist):
          safe_force : physical action to pass to env.step
          z_exec     : executed action in normalized units, for log_prob/storage
          clamp_dist : |proposed - safe| in NORMALIZED units (shield-intervention
                       magnitude, used for the intervention-penalty reward term)
        For act_dim == 1 both forces are scalar floats.
        """
        with torch.no_grad():
            dist, value, hidden = self.policy(obs_t, hidden)

        z = dist.mean if greedy else dist.sample()    # (1, act_dim), normalized
        z_val = float(z.squeeze().item())             # act_dim == 1
        proposed_force = z_val * self.action_scale

        safe_force, overridden = self.shield(proposed_force, obs_dict)
        z_exec = safe_force / self.action_scale
        clamp_dist = abs(proposed_force - safe_force) / self.action_scale
        return safe_force, z_exec, dist, value, hidden, overridden, clamp_dist


def build_continuous_composite(model_path: str,
                               policy: GaussianRecurrentActorCritic,
                               action_scale: float = 100.0):
    """Construct the continuous shield and wrap the policy."""
    from continuous_shield import ContinuousShield
    shield = ContinuousShield(model_path)
    return CompositeContinuousPolicy(policy, shield, action_scale=action_scale)
