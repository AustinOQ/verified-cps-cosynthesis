"""
Composite shielded policy with two modes:
  - full:        frozen neural shield from entire #NeuralRequirement
  - none:        no shield, policy acts directly

Both modes expose the same .act() interface.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from shield import (FullSpecShield,
                    _evaluate, _flatten_and, _collect_refs)
from sysml_parser import BinaryExpr, RefExpr, LiteralExpr, UnaryExpr, TernaryExpr


# ---------------------------------------------------------------------------
# FullShieldNet: wraps FullSpecShield as nn.Module
# Uses prohibition neural net for dead actions + AST eval for state-dep
# ---------------------------------------------------------------------------

class FullShieldNet(nn.Module):
    """Neural prohibition net + full AST requirement evaluation.

    Dead action blocking is pure tensor math (prohibition clauses).
    State-dependent requirement evaluation uses the FullSpecShield.
    """

    def __init__(self, spec_shield):
        super().__init__()
        self.in_params = spec_shield.in_params
        self.out_params = spec_shield.out_params
        n_out = len(self.out_params)
        n_actions = 2 ** n_out
        self.n_actions = n_actions
        self.n_out = n_out
        self.n_obs = len(self.in_params)
        self._spec_shield = spec_shield

        dead_mask = torch.ones(n_actions)
        for a in spec_shield.dead_actions:
            dead_mask[a] = 0.0
        self.register_buffer('dead_mask', dead_mask)

        all_action_bits = torch.zeros(n_actions, n_out)
        for action_id in range(n_actions):
            for bit in range(n_out):
                all_action_bits[action_id, bit] = float(bool(action_id & (1 << bit)))
        self.register_buffer('all_action_bits', all_action_bits)

        for param in self.parameters():
            param.requires_grad = False

        print(f"  [FullShieldNet] dead={sorted(spec_shield.dead_actions)}, "
              f"n_valid={n_actions - len(spec_shield.dead_actions)}")

    @torch.no_grad()
    def forward(self, proposed_action, policy_probs=None, obs_dict=None):
        if obs_dict is None:
            obs_dict = {}
        result = self._spec_shield(proposed_action, obs_dict)
        overridden = (result != proposed_action)
        return result, overridden


# ---------------------------------------------------------------------------
# NoShieldNet: passthrough, never overrides
# ---------------------------------------------------------------------------

class NoShieldNet(nn.Module):
    """No shield — passes every action through."""

    def __init__(self, in_params, out_params):
        super().__init__()
        self.in_params = in_params
        self.out_params = out_params
        n_out = len(out_params)
        self.n_actions = 2 ** n_out
        self.n_out = n_out
        self.n_obs = len(in_params)

        dead_mask = torch.ones(self.n_actions)
        self.register_buffer('dead_mask', dead_mask)

        all_action_bits = torch.zeros(self.n_actions, n_out)
        for action_id in range(self.n_actions):
            for bit in range(n_out):
                all_action_bits[action_id, bit] = float(bool(action_id & (1 << bit)))
        self.register_buffer('all_action_bits', all_action_bits)

        print(f"  [NoShield] All {self.n_actions} actions valid")

    @torch.no_grad()
    def forward(self, proposed_action, policy_probs=None, obs_dict=None):
        return proposed_action, False


# ---------------------------------------------------------------------------
# CompositeShieldedPolicy: single model, all modes
# ---------------------------------------------------------------------------

class CompositeShieldedPolicy(nn.Module):
    """Single model: trainable policy + frozen shield (any mode)."""

    def __init__(self, policy, shield_net, mode="full"):
        super().__init__()
        self.policy = policy
        self.shield = shield_net
        self.mode = mode

    def initial_hidden(self, batch_size=1):
        return self.policy.initial_hidden(batch_size)

    def forward_policy(self, obs_t, hidden):
        return self.policy(obs_t, hidden)

    def forward_sequence(self, obs_seq, hidden, mask=None):
        return self.policy.forward_sequence(obs_seq, hidden, mask)

    def act(self, obs_t, hidden, obs_dict, greedy=False):
        with torch.no_grad():
            dist, value, hidden = self.policy(obs_t, hidden)

        proposed = dist.probs.argmax(dim=-1).item() if greedy else dist.sample().item()
        policy_probs = dist.probs.detach().squeeze(0)

        if self.mode == "full":
            final_action, overridden = self.shield(proposed, policy_probs,
                                                   obs_dict=obs_dict)
        else:
            final_action, overridden = proposed, False

        return final_action, dist, value, hidden, overridden


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_full_shield_model(model_path, policy):
    """Build composite with full requirement shield."""
    print("  Building FULL specification shield...")
    spec_shield = FullSpecShield(model_path)
    shield_net = FullShieldNet(spec_shield)
    return CompositeShieldedPolicy(policy, shield_net, mode="full")


def build_no_shield_model(model_path, policy):
    """Build composite with no shield (passthrough)."""
    print("  Building NO SHIELD (passthrough)...")
    from shield import _extract_base
    d = _extract_base(model_path)
    shield_net = NoShieldNet(d["in_params"], d["out_params"])
    return CompositeShieldedPolicy(policy, shield_net, mode="none")
