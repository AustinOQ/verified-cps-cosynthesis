"""
Non-recurrent policy plus an observation/action buffer for real-valued actions.

Motivation: the reconstructibility closure (reconstruct_closure.py) proves that
(current obs + last 2 actions + 1 past obs) is a sufficient statistic for the
full current state, so the GRU's recurrence is unnecessary. Here we realize that
as:
  - BufferedContinuousEnv : returns an AUGMENTED observation = current obs ++
    last n_act actions ++ last n_obs past observations. The buffer lives in the
    env, so the policy that consumes it is plain/memoryless. The shield still
    reads the RAW current obs via env._twin._model_inputs (unchanged), so
    safety plumbing is identical.
  - GaussianMLPActorCritic : same interface as GaussianRecurrentActorCritic
    (act/forward/forward_sequence/initial_hidden/std) but with the GRU removed;
    `hidden` is a dummy passed through so the existing training loop is reused
    verbatim.

Nothing about the dynamics is baked into the policy: it just receives a flat
vector. The dynamics were used only to *prove* the buffer suffices.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical

_RL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl")
if _RL not in sys.path:
    sys.path.insert(0, _RL)
from continuous_env import SysMLContinuousEnv
from env import SysMLEnv


class GaussianMLPActorCritic(nn.Module):
    """Memoryless diagonal-Gaussian actor-critic (GRU removed)."""

    def __init__(self, obs_dim: int, act_dim: int = 1,
                 hidden_dim: int = 64, init_log_std: float = -0.7):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(1)  # dummy; this policy has no recurrent state

    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std).clamp(min=1e-3, max=1e3)

    def forward(self, obs: torch.Tensor, hidden):
        feats = self.encoder(obs)
        mean = self.mean_head(feats)
        value = self.value_head(feats)
        std = self.std().expand_as(mean)
        return Normal(mean, std), value, hidden

    def forward_sequence(self, obs_seq: torch.Tensor, hidden, mask=None):
        feats = self.encoder(obs_seq)        # (B,T,O) -> (B,T,H)
        mean = self.mean_head(feats)
        values = self.value_head(feats)
        return mean, values


class BufferedContinuousEnv(SysMLContinuousEnv):
    """Augments the observation with the last n_act actions and n_obs past obs.

    aug[t] = [ obs[t],  a[t-1..t-n_act] (normalized),  obs[t-1..t-n_obs] ]
    The shield path is unaffected (it reads env._twin._model_inputs).
    """

    def __init__(self, model_path, dt: float, max_steps: int = 1200,
                 phase: int = 1, rng_seed: int = None,
                 n_act: int = 2, n_obs: int = 1, action_scale: float = 100.0):
        super().__init__(model_path, dt=dt, max_steps=max_steps,
                         phase=phase, rng_seed=rng_seed)
        self._base_obs_dim = self.obs_dim
        self._n_act = n_act
        self._n_obs = n_obs
        self._action_scale = float(action_scale)
        self.obs_dim = self._base_obs_dim + n_act + n_obs * self._base_obs_dim
        self._act_hist = None
        self._obs_hist = None

    def _augment(self, cur_obs):
        parts = [np.asarray(cur_obs, np.float32),
                 np.asarray(self._act_hist, np.float32)]
        parts.extend(self._obs_hist)
        return np.concatenate(parts).astype(np.float32)

    def _push_act(self, a):
        self._act_hist = ([a] + self._act_hist)[:self._n_act]

    def _push_obs(self, o):
        self._obs_hist = ([np.asarray(o, np.float32).copy()] + self._obs_hist)[:self._n_obs]

    def reset(self):
        obs0 = super().reset()
        self._act_hist = [0.0] * self._n_act
        self._obs_hist = [np.zeros(self._base_obs_dim, np.float32)
                          for _ in range(self._n_obs)]
        aug = self._augment(obs0)          # pre-step history (zeros) at t=0
        self._push_obs(obs0)
        return aug

    def step(self, action):
        a_norm = float(action) / self._action_scale
        self._push_act(a_norm)             # include the just-taken action
        obs1, r, done, info = super().step(action)
        aug = self._augment(obs1)          # uses prev obs (current not yet pushed)
        self._push_obs(obs1)
        return aug, r, done, info


# =====================================================================
# DISCRETE counterparts (Categorical policy + buffered discrete env)
# =====================================================================

class CategoricalMLPActorCritic(nn.Module):
    """Memoryless Categorical actor-critic (GRU removed). Conforms to the
    RecurrentActorCritic interface so the shield composite (ExactShieldComposite)
    wraps it unchanged; `hidden` is a dummy."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int = 1):
        return torch.zeros(1)

    def forward(self, obs, hidden):
        feats = self.encoder(obs)
        return Categorical(logits=self.policy_head(feats)), self.value_head(feats), hidden

    def forward_sequence(self, obs_seq, hidden, mask=None, seq_len=None,
                         use_gradient_checkpoint=False):
        feats = self.encoder(obs_seq)          # (B,T,O)->(B,T,H)
        return self.policy_head(feats), self.value_head(feats)


class BufferedDiscreteEnv(SysMLEnv):
    """Augments the obs with the last n_act actions (one-hot) and n_obs past
    observations. Shield path unchanged (reads env._twin._model_inputs)."""

    def __init__(self, model_path, dt: float, max_steps: int = 1200,
                 phase: int = 2, rng_seed: int = None, n_act: int = 3, n_obs: int = 1):
        super().__init__(model_path, dt=dt, max_steps=max_steps,
                         phase=phase, rng_seed=rng_seed)
        self._base_obs_dim = self.obs_dim
        self._n_act = n_act
        self._n_obs = n_obs
        self._na = self.n_actions
        self.obs_dim = self._base_obs_dim + n_obs * self._base_obs_dim + n_act * self._na
        self._act_hist = None
        self._obs_hist = None

    def _onehot(self, a):
        v = np.zeros(self._na, np.float32)
        if 0 <= int(a) < self._na:
            v[int(a)] = 1.0
        return v

    def _augment(self, cur_obs):
        return np.concatenate(
            [np.asarray(cur_obs, np.float32)] + list(self._obs_hist) + list(self._act_hist)
        ).astype(np.float32)

    def _push_act(self, oh):
        # slice to exactly n_act so n_act=0 stays empty ([x]+[])[:0] == [] and a
        # full buffer drops the oldest; [x]+hist[:-1] would GROW an empty buffer.
        self._act_hist = ([oh] + self._act_hist)[:self._n_act]

    def _push_obs(self, o):
        self._obs_hist = ([np.asarray(o, np.float32).copy()] + self._obs_hist)[:self._n_obs]

    def reset(self):
        obs0 = super().reset()
        self._act_hist = [np.zeros(self._na, np.float32) for _ in range(self._n_act)]
        self._obs_hist = [np.zeros(self._base_obs_dim, np.float32) for _ in range(self._n_obs)]
        aug = self._augment(obs0)
        self._push_obs(obs0)
        return aug

    def step(self, action):
        self._push_act(self._onehot(action))
        obs1, r, done, info = super().step(action)
        aug = self._augment(obs1)
        self._push_obs(obs1)
        return aug, r, done, info
