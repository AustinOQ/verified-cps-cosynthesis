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

    def _seq_body(self, obs_seq: torch.Tensor, hidden: torch.Tensor,
                  mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Encoder + GRU body used by `forward_sequence`. Factored out so it
        can be wrapped by `torch.utils.checkpoint.checkpoint` when
        `use_gradient_checkpoint=True`."""
        features = self.encoder(obs_seq)
        if mask is not None:
            lengths = mask.sum(dim=1).long().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths, batch_first=True, enforce_sorted=False)
            gru_out, _ = self.gru(packed, hidden)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                gru_out, batch_first=True, total_length=seq_len)
        else:
            gru_out, _ = self.gru(features, hidden)
        return gru_out

    def _seq_body_chunked(self, obs_seq: torch.Tensor, hidden: torch.Tensor,
                          seq_len: int, chunk_size: int) -> torch.Tensor:
        """Truncated-BPTT forward with per-chunk activation checkpointing.

        Each chunk's encoder+GRU forward is wrapped in
        torch.utils.checkpoint(use_reentrant=False), so the activations of
        that chunk are NOT retained after forward — they are recomputed
        during backward.  Combined with `h.detach()` at chunk boundaries,
        this gives the truncated-BPTT memory profile (only one chunk's
        worth of activations alive at any one time during the
        forward+backward dance).

        The encoder is also applied INSIDE each checkpointed chunk so its
        activations are bounded by chunk_size too — not by seq_len.

        Mask isn't used here; masking is still applied to the per-timestep
        loss downstream so padded steps' contributions are zeroed.
        """
        from torch.utils.checkpoint import checkpoint

        # Per-chunk forward: encoder + GRU. Activations live only while
        # this function is on the stack (or during recompute on backward).
        def _chunk_fwd(obs_chunk, h_in):
            feats = self.encoder(obs_chunk)
            out, h_new = self.gru(feats, h_in)
            return out, h_new

        outputs = []
        h = hidden
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            obs_chunk = obs_seq[:, start:end, :]
            chunk_out, h = checkpoint(_chunk_fwd, obs_chunk, h,
                                       use_reentrant=False)
            outputs.append(chunk_out)
            # Sever the gradient graph between chunks so backward doesn't
            # try to flow gradients from chunk N+1 into chunk N's compute.
            h = h.detach()
        return torch.cat(outputs, dim=1)

    def forward_sequence(self, obs_seq: torch.Tensor, hidden: torch.Tensor,
                         mask: torch.Tensor = None,
                         use_gradient_checkpoint: bool = False,
                         bptt_chunk_size: int = 0):
        """Forward pass over a padded sequence (for training).

        Args:
            obs_seq: (batch, seq_len, obs_dim)
            hidden: (gru_layers, batch, hidden_dim)
            mask: (batch, seq_len) — 1 for valid steps, 0 for padding
            use_gradient_checkpoint: if True, wrap the encoder+GRU body in
                torch.utils.checkpoint so activations are not stored across
                the forward pass and are recomputed during backward.
                Trades ~30% compute for ~50-70% less activation memory.
                Default False (no behavior change).
            bptt_chunk_size: if > 0 and seq_len > chunk_size, use truncated
                BPTT: split the sequence into chunks of this size and detach
                the GRU hidden state at each chunk boundary so the backward
                pass only sees one chunk's activations at a time.
                Loses long-horizon credit assignment beyond `chunk_size`.
                Default 0 = full BPTT. Mutually exclusive with
                `use_gradient_checkpoint`; if both are set, truncated BPTT
                wins (it already keeps activation memory bounded by chunk
                size, so checkpointing adds little).

        Returns:
            logits: (batch, seq_len, n_actions)
            values: (batch, seq_len, 1)
        """
        batch, seq_len, _ = obs_seq.shape

        if bptt_chunk_size > 0 and seq_len > bptt_chunk_size:
            gru_out = self._seq_body_chunked(obs_seq, hidden, seq_len,
                                              bptt_chunk_size)
        elif use_gradient_checkpoint:
            from torch.utils.checkpoint import checkpoint
            gru_out = checkpoint(self._seq_body, obs_seq, hidden, mask,
                                 seq_len, use_reentrant=False)
        else:
            gru_out = self._seq_body(obs_seq, hidden, mask, seq_len)

        logits = self.policy_head(gru_out)              # (batch, seq_len, n_actions)
        values = self.value_head(gru_out)               # (batch, seq_len, 1)

        return logits, values
