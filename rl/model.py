"""
GRU Actor-Critic for POMDP mixing control.

Architecture:
    Obs (7 floats) → MLP encoder → GRU → policy head (4 actions) + value head
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class RecurrentActorCritic(nn.Module):
    """GRU-based actor-critic for the mixing POMDP.

    The GRU maintains hidden state across timesteps within an episode,
    allowing the policy to infer latent state (pump ramp, sensor delay)
    from observation history.
    """

    def __init__(self, obs_dim: int = 7, n_actions: int = 4,
                 hidden_dim: int = 64, gru_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Temporal backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        # Heads
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Zero-initialize GRU hidden state."""
        return torch.zeros(self.gru_layers, batch_size, self.hidden_dim)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """Forward pass for a single timestep.

        Args:
            obs: (batch, obs_dim)
            hidden: (gru_layers, batch, hidden_dim)

        Returns:
            dist: Categorical distribution over actions
            value: state value estimate (batch, 1)
            hidden: updated GRU hidden state
        """
        features = self.encoder(obs)                    # (batch, hidden_dim)
        features = features.unsqueeze(1)                # (batch, 1, hidden_dim)
        gru_out, hidden = self.gru(features, hidden)    # gru_out: (batch, 1, hidden_dim)
        gru_out = gru_out.squeeze(1)                    # (batch, hidden_dim)

        logits = self.policy_head(gru_out)              # (batch, n_actions)
        value = self.value_head(gru_out)                # (batch, 1)

        return Categorical(logits=logits), value, hidden

    def forward_sequence(self, obs_seq: torch.Tensor, hidden: torch.Tensor,
                         mask: torch.Tensor = None):
        """Forward pass over a padded sequence (for training).

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            hidden: (gru_layers, batch, hidden_dim)
            mask: (batch, seq_len) — 1 for valid steps, 0 for padding

        Returns:
            logits: (batch, seq_len, n_actions)
            values: (batch, seq_len, 1)
        """
        batch, seq_len, _ = obs_seq.shape
        features = self.encoder(obs_seq)                # (batch, seq_len, hidden_dim)

        if mask is not None:
            # Pack padded sequences for efficient GRU processing
            lengths = mask.sum(dim=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False)
            gru_out, _ = self.gru(packed, hidden)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=seq_len)
        else:
            gru_out, _ = self.gru(features, hidden)

        logits = self.policy_head(gru_out)              # (batch, seq_len, n_actions)
        values = self.value_head(gru_out)               # (batch, seq_len, 1)

        return logits, values
