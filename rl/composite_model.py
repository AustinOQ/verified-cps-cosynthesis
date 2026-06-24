"""Torch composite policy utilities.

The active artifact pipeline trains and evaluates NumPy CPU policies through
``run_pipeline.sh``. This module remains for the Torch reference path and uses
the same architecture-level shield interface: the policy proposes an action,
then the Python ``SpecShield`` function evaluates the SysML
``#NeuralRequirement`` AST directly and returns either the proposed action or a
spec-compliant override.

``ShieldNet`` below is retained as a reference implementation of an encoded
shield, but ``build_composite_model`` uses ``SpecShield`` directly.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from shield import SpecShield, _evaluate, _flatten_and, _collect_refs
from sysml_parser import BinaryExpr, RefExpr, LiteralExpr, UnaryExpr, TernaryExpr

SCALE = 1000.0


# ---------------------------------------------------------------------------
# Linearize arithmetic expressions to (weight_vector, bias)
# ---------------------------------------------------------------------------

def _linearize(expr, var_to_idx, n_vars, constants, subject_var):
    """Convert arithmetic expression to (weights, bias) on input vector."""
    if isinstance(expr, LiteralExpr):
        return torch.zeros(n_vars), float(expr.value)

    if isinstance(expr, RefExpr):
        path = list(expr.path)
        if path and path[0] == subject_var:
            path = path[1:]
        name = path[-1]
        if name in var_to_idx:
            w = torch.zeros(n_vars)
            w[var_to_idx[name]] = 1.0
            return w, 0.0
        if name in constants:
            return torch.zeros(n_vars), float(constants[name])
        return None

    if isinstance(expr, BinaryExpr):
        if expr.op == '+':
            l = _linearize(expr.left, var_to_idx, n_vars, constants, subject_var)
            r = _linearize(expr.right, var_to_idx, n_vars, constants, subject_var)
            if l is None or r is None:
                return None
            return l[0] + r[0], l[1] + r[1]
        if expr.op == '-':
            l = _linearize(expr.left, var_to_idx, n_vars, constants, subject_var)
            r = _linearize(expr.right, var_to_idx, n_vars, constants, subject_var)
            if l is None or r is None:
                return None
            return l[0] - r[0], l[1] - r[1]
        if expr.op == '*':
            l = _linearize(expr.left, var_to_idx, n_vars, constants, subject_var)
            r = _linearize(expr.right, var_to_idx, n_vars, constants, subject_var)
            if l is None or r is None:
                return None
            if l[0].abs().sum() == 0:
                return r[0] * l[1], r[1] * l[1]
            if r[0].abs().sum() == 0:
                return l[0] * r[1], l[1] * r[1]
            return None

    if isinstance(expr, UnaryExpr) and expr.op == '-':
        inner = _linearize(expr.operand, var_to_idx, n_vars, constants, subject_var)
        if inner is None:
            return None
        return -inner[0], -inner[1]

    return None


# ---------------------------------------------------------------------------
# General propositional compilation: AST → atoms + boolean formula → CNF
# ---------------------------------------------------------------------------

class AtomCollector:
    """Walk the requirement AST, identify atoms, build a boolean formula.

    Atoms are:
      - Arithmetic comparisons (>, >=, <, <=) → need comp_layer neurons
      - Output bit references used in boolean context → available as inputs

    The boolean formula is represented as nested tuples:
      ('comp', idx)   — comparison atom
      ('bit', idx)    — output bit atom
      ('and', [children])
      ('or', [children])
      ('not', child)
    """

    def __init__(self, out_set, in_set, const_set, subject_var):
        self.out_set = out_set
        self.in_set = in_set
        self.const_set = const_set
        self.subject_var = subject_var
        self.comp_atoms = []
        self.bit_atoms = []
        self._comp_cache = {}
        self._bit_cache = {}

    def _get_ref_name(self, expr):
        if isinstance(expr, RefExpr):
            path = list(expr.path)
            if path and path[0] == self.subject_var:
                path = path[1:]
            return path[-1]
        return None

    def _add_comp_atom(self, expr):
        eid = id(expr)
        if eid in self._comp_cache:
            return self._comp_cache[eid]
        idx = len(self.comp_atoms)
        self.comp_atoms.append(expr)
        self._comp_cache[eid] = idx
        return idx

    def _add_bit_atom(self, name):
        if name in self._bit_cache:
            return self._bit_cache[name]
        idx = len(self.bit_atoms)
        self.bit_atoms.append(name)
        self._bit_cache[name] = idx
        return idx

    def to_formula(self, expr):
        """Convert AST expression to boolean formula over atoms."""

        # Output bit reference → bit atom
        if isinstance(expr, RefExpr):
            name = self._get_ref_name(expr)
            if name and name in self.out_set:
                idx = self._add_bit_atom(name)
                return ('bit', idx)
            return None

        # Arithmetic comparison → comp atom
        if (isinstance(expr, BinaryExpr) and
                expr.op in ('>', '>=', '<', '<=')):
            idx = self._add_comp_atom(expr)
            return ('comp', idx)

        # not
        if isinstance(expr, UnaryExpr) and expr.op == 'not':
            child = self.to_formula(expr.operand)
            if child is None:
                return None
            return ('not', child)

        # and
        if isinstance(expr, BinaryExpr) and expr.op == 'and':
            left = self.to_formula(expr.left)
            right = self.to_formula(expr.right)
            if left is None or right is None:
                return None
            return ('and', [left, right])

        # or
        if isinstance(expr, BinaryExpr) and expr.op == 'or':
            left = self.to_formula(expr.left)
            right = self.to_formula(expr.right)
            if left is None or right is None:
                return None
            return ('or', [left, right])

        # implies: A → B = ¬A ∨ B
        if isinstance(expr, BinaryExpr) and expr.op == 'implies':
            left = self.to_formula(expr.left)
            right = self.to_formula(expr.right)
            if left is None or right is None:
                return None
            return ('or', [('not', left), right])

        # biconditional ==: A == B = (¬A ∨ B) ∧ (¬B ∨ A)
        if isinstance(expr, BinaryExpr) and expr.op == '==':
            left = self.to_formula(expr.left)
            right = self.to_formula(expr.right)
            if left is None or right is None:
                return None
            return ('and', [
                ('or', [('not', left), right]),
                ('or', [('not', right), left]),
            ])

        return None


def _to_nnf(formula):
    """Convert formula to Negation Normal Form (push NOT to atoms)."""
    if formula is None:
        return None
    tag = formula[0]

    if tag in ('comp', 'bit'):
        return formula

    if tag == 'not':
        child = formula[1]
        ctag = child[0]

        if ctag == 'not':
            return _to_nnf(child[1])

        if ctag == 'and':
            return ('or', [_to_nnf(('not', c)) for c in child[1]])

        if ctag == 'or':
            return ('and', [_to_nnf(('not', c)) for c in child[1]])

        if ctag in ('comp', 'bit'):
            return formula

        return ('not', _to_nnf(child))

    if tag == 'and':
        return ('and', [_to_nnf(c) for c in formula[1]])

    if tag == 'or':
        return ('or', [_to_nnf(c) for c in formula[1]])

    return formula


def _to_cnf(formula):
    """Convert NNF formula to CNF. Returns list of clauses.

    Each clause is a list of literals: ('pos', tag, idx) or ('neg', tag, idx).
    """
    if formula is None:
        return []
    tag = formula[0]

    if tag in ('comp', 'bit'):
        return [[('pos', tag, formula[1])]]

    if tag == 'not':
        child = formula[1]
        if child[0] in ('comp', 'bit'):
            return [[('neg', child[0], child[1])]]

    if tag == 'and':
        clauses = []
        for child in formula[1]:
            clauses.extend(_to_cnf(child))
        return clauses

    if tag == 'or':
        children_cnf = [_to_cnf(c) for c in formula[1]]
        result = children_cnf[0]
        for i in range(1, len(children_cnf)):
            new_result = []
            for c1 in result:
                for c2 in children_cnf[i]:
                    new_result.append(c1 + c2)
            result = new_result
        return result

    return []


# ---------------------------------------------------------------------------
# ShieldNet
# ---------------------------------------------------------------------------

class ShieldNet(nn.Module):
    """Neural shield compiled from the full #NeuralRequirement.

    General compilation:
      1. Collect atoms (comparisons + output bits) from AST
      2. Convert boolean formula to CNF over atoms
      3. Each comparison atom → one comp_layer neuron
      4. Each CNF clause → one clause_layer neuron
      5. conj_layer ANDs all clauses
    """

    def __init__(self, spec_shield: SpecShield):
        super().__init__()

        shield = spec_shield
        in_params = shield.in_params
        out_params = shield.out_params
        n_obs = len(in_params)
        n_out = len(out_params)
        n_actions = 2 ** n_out
        n_input = n_obs + n_out

        self.n_obs = n_obs
        self.n_out = n_out
        self.n_actions = n_actions
        self.in_params = in_params
        self.out_params = out_params

        var_to_idx = {}
        for i, name in enumerate(in_params):
            var_to_idx[name] = i
        for i, name in enumerate(out_params):
            var_to_idx[name] = n_obs + i

        constants = shield.unchanging
        subject_var = shield.subject_var

        # Dead actions mask
        dead_mask = torch.ones(n_actions)
        for a in shield.dead_actions:
            dead_mask[a] = 0.0
        self.register_buffer('dead_mask', dead_mask)

        # All action bit patterns
        all_action_bits = torch.zeros(n_actions, n_out)
        for action_id in range(n_actions):
            for bit in range(n_out):
                all_action_bits[action_id, bit] = float(
                    bool(action_id & (1 << bit)))
        self.register_buffer('all_action_bits', all_action_bits)

        # --- Step 1: Collect atoms and build boolean formula ---
        collector = AtomCollector(
            out_set=set(out_params),
            in_set=set(in_params),
            const_set=set(constants.keys()),
            subject_var=subject_var)

        if shield.req_ast is not None:
            formula = collector.to_formula(shield.req_ast)
            if formula is None:
                raise RuntimeError(
                    "ShieldNet: could not convert requirement AST to "
                    "boolean formula. Unsupported expression in AST.")
        else:
            formula = None

        # --- Step 2: Convert to CNF ---
        if formula is not None:
            nnf = _to_nnf(formula)
            cnf_clauses = _to_cnf(nnf)
        else:
            cnf_clauses = []

        if not cnf_clauses and shield.req_ast is not None:
            raise RuntimeError(
                "ShieldNet: CNF conversion produced zero clauses from "
                "a non-empty requirement. Something went wrong.")

        # --- Step 3: Compile comparison atoms into comp_layer ---
        n_comp = len(collector.comp_atoms)
        n_bits = len(collector.bit_atoms)

        bit_atom_to_input_idx = {}
        for i, name in enumerate(collector.bit_atoms):
            bit_atom_to_input_idx[i] = var_to_idx[name]

        if n_comp > 0:
            self.comp_layer = nn.Linear(n_input, n_comp, bias=True)
            with torch.no_grad():
                for j, comp_expr in enumerate(collector.comp_atoms):
                    result = self._compile_comparison(
                        comp_expr, var_to_idx, n_input,
                        constants, subject_var)
                    if result is None:
                        raise RuntimeError(
                            f"ShieldNet: comparison atom {j} could not "
                            f"be linearized: {comp_expr}")
                    w, b = result
                    self.comp_layer.weight[j] = w
                    self.comp_layer.bias[j] = b
        else:
            self.comp_layer = None

        # --- Step 4: Wire CNF clauses into clause_layer ---
        clause_input_dim = n_comp + n_input
        n_clauses = len(cnf_clauses)

        if n_clauses > 0:
            self.clause_layer = nn.Linear(clause_input_dim, n_clauses,
                                          bias=True)
            with torch.no_grad():
                self.clause_layer.weight.zero_()
                self.clause_layer.bias.zero_()

                for i, clause in enumerate(cnf_clauses):
                    n_neg = 0
                    for polarity, atom_type, atom_idx in clause:
                        if atom_type == 'comp':
                            layer_idx = atom_idx
                        elif atom_type == 'bit':
                            layer_idx = n_comp + bit_atom_to_input_idx[atom_idx]
                        else:
                            continue

                        if polarity == 'pos':
                            self.clause_layer.weight[i, layer_idx] += 1.0
                        else:
                            self.clause_layer.weight[i, layer_idx] += -1.0
                            n_neg += 1

                    self.clause_layer.bias[i] = n_neg - 0.5
        else:
            self.clause_layer = None

        # --- Step 5: Conjunction layer ---
        if n_clauses > 0:
            self.conj_layer = nn.Linear(n_clauses, 1, bias=True)
            with torch.no_grad():
                self.conj_layer.weight.fill_(1.0)
                self.conj_layer.bias.fill_(-(n_clauses - 0.5))
        else:
            self.conj_layer = None

        # Freeze
        for param in self.parameters():
            param.requires_grad = False

        # Verify
        mismatches = self._verify(spec_shield, n_samples=1000)

        print(f"  [ShieldNet] Compiled: {n_comp} comparison atoms, "
              f"{n_bits} bit atoms, {n_clauses} CNF clauses")
        print(f"  [ShieldNet] dead_actions={sorted(shield.dead_actions)}, "
              f"n_valid={n_actions - len(shield.dead_actions)}")
        print(f"  [ShieldNet] Verification (1000 samples): "
              f"{mismatches} mismatches")

    # -------------------------------------------------------------------
    # Compile comparison to (weights, bias)
    # -------------------------------------------------------------------

    def _compile_comparison(self, expr, var_to_idx, n_input,
                            constants, subject_var):
        if isinstance(expr, BinaryExpr) and expr.op in ('>=', '>', '<=', '<'):
            left = _linearize(expr.left, var_to_idx, n_input,
                              constants, subject_var)
            right = _linearize(expr.right, var_to_idx, n_input,
                               constants, subject_var)
            if left is None or right is None:
                return None
            if expr.op in ('>=', '>'):
                w = left[0] - right[0]
                b = left[1] - right[1]
                if expr.op == '>':
                    b -= 1e-10
                return w, b
            else:
                w = right[0] - left[0]
                b = right[1] - left[1]
                if expr.op == '<':
                    b -= 1e-10
                return w, b

        if isinstance(expr, RefExpr):
            path = list(expr.path)
            if path and path[0] == subject_var:
                path = path[1:]
            name = path[-1]
            if name in var_to_idx:
                w = torch.zeros(n_input)
                w[var_to_idx[name]] = 1.0
                return w, -0.5

        return None

    # -------------------------------------------------------------------
    # Forward pass: pure tensor math
    # -------------------------------------------------------------------

    def _forward_layers(self, x):
        """x: (..., n_obs + n_out) → validity: (...)"""
        if self.comp_layer is not None:
            comp = torch.sigmoid(SCALE * self.comp_layer(x))
            combined = torch.cat([comp, x], dim=-1)
        else:
            combined = x

        if self.clause_layer is not None:
            clauses = torch.sigmoid(SCALE * self.clause_layer(combined))
            valid = torch.sigmoid(SCALE * self.conj_layer(clauses))
            return valid.squeeze(-1)
        else:
            return torch.ones(x.shape[:-1], device=x.device)

    @torch.no_grad()
    def evaluate_all_actions(self, obs_values):
        """Evaluate requirement for all actions. Pure tensor math."""
        obs_expanded = obs_values.unsqueeze(0).expand(self.n_actions, -1)
        x = torch.cat([obs_expanded, self.all_action_bits], dim=-1)
        return self._forward_layers(x)

    @torch.no_grad()
    def forward(self, obs_values, proposed_action, policy_probs=None):
        """Shield decision. Pure tensor math."""
        action_bits = self.all_action_bits[proposed_action].unsqueeze(0)
        x = torch.cat([obs_values.unsqueeze(0), action_bits], dim=-1)
        proposed_valid = self._forward_layers(x).item() > 0.5

        if proposed_valid and self.dead_mask[proposed_action] > 0:
            return proposed_action, False

        all_validity = self.evaluate_all_actions(obs_values)

        if policy_probs is not None:
            scores = all_validity * self.dead_mask + policy_probs
        else:
            intervention_bias = torch.zeros(self.n_actions,
                                            device=obs_values.device)
            for aid in range(self.n_actions):
                intervention_bias[aid] = -0.01 * bin(aid).count('1')
            scores = all_validity * self.dead_mask + intervention_bias

        scores[(all_validity < 0.5) | (self.dead_mask < 0.5)] = float('-inf')

        if scores.max() == float('-inf'):
            return 0, True

        return scores.argmax().item(), True

    # -------------------------------------------------------------------
    # Verification
    # -------------------------------------------------------------------

    def _verify(self, spec_shield, n_samples=1000):
        """Verify neural shield matches SpecShield on random inputs."""
        rng = np.random.default_rng(12345)
        mismatches = 0

        for _ in range(n_samples):
            obs_dict = {}
            obs_vals = []
            for name in self.in_params:
                if name == 'done':
                    v = False
                elif 'OriginalMl' in name:
                    v = 1000.0
                elif 'VolumeMl' in name:
                    v = float(rng.uniform(0, 1000))
                elif 'TargetTransfer' in name:
                    v = float(rng.uniform(50, 950))
                elif 'emperatur' in name:
                    v = float(rng.uniform(10, 40))
                elif 'etPoint' in name or 'setPoint' in name:
                    v = float(rng.uniform(15, 35))
                elif 'peed' in name or 'Speed' in name:
                    v = float(rng.uniform(0, 50))
                elif 'gap' in name or 'Gap' in name or 'istance' in name:
                    v = float(rng.uniform(0, 100))
                else:
                    v = float(rng.uniform(0, 1000))
                obs_dict[name] = v
                obs_vals.append(float(v))

            action_id = rng.integers(0, self.n_actions)

            shield_result = spec_shield(action_id, obs_dict)
            shield_passed = (shield_result == action_id)

            obs_t = torch.tensor(obs_vals, dtype=torch.float32)
            action_bits = self.all_action_bits[action_id]
            x = torch.cat([obs_t, action_bits]).unsqueeze(0)
            neural_score = self._forward_layers(x).item()
            neural_passed = (neural_score > 0.5)

            if shield_passed != neural_passed:
                mismatches += 1

        return mismatches


# ---------------------------------------------------------------------------
# CompositeShieldedPolicy
# ---------------------------------------------------------------------------

class CompositeShieldedPolicy(nn.Module):
    """Single nn.Module: trainable policy + external SpecShield function."""

    def __init__(self, policy, spec_shield):
        super().__init__()
        self.policy = policy
        self.shield = spec_shield

    def initial_hidden(self, batch_size=1):
        return self.policy.initial_hidden(batch_size)

    def forward_policy(self, obs_t, hidden):
        return self.policy(obs_t, hidden)

    def forward_sequence(self, obs_seq, hidden, mask=None):
        return self.policy.forward_sequence(obs_seq, hidden, mask)

    def act(self, obs_t, hidden, obs_dict, greedy=False):
        """Policy proposes, external SpecShield checks/overrides."""
        with torch.no_grad():
            dist, value, hidden = self.policy(obs_t, hidden)

        proposed = (dist.probs.argmax(dim=-1).item() if greedy
                    else dist.sample().item())

        shield_obs = {
            name: float(obs_dict.get(name, 0))
            for name in self.shield.in_params
        }
        final_action = self.shield(proposed, shield_obs)
        overridden = (final_action != proposed)

        return final_action, dist, value, hidden, overridden


def build_composite_model(model_path, policy):
    """Build composite: policy + external SpecShield callable."""
    spec_shield = SpecShield(model_path)
    return CompositeShieldedPolicy(policy, spec_shield)
