"""
Specification-derived oracle for SysML-extracted neural controllers.

Derives correct action labels directly from the #NeuralRequirement AST —
no simulation, no propagationDelay, no lookahead. The requirement is a
static function from observations to actions; the oracle evaluates it.

Also extracts obs_names, action_names, is_done, and goal_distance from
the SysML model for use by the environment and evaluation code.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from sysml_parser import (SysMLParser, InputBindingStmt, SubactionCallStmt,
                           RefExpr, BinaryExpr, LiteralExpr, UnaryExpr,
                           TernaryExpr)
from shield import SpecShield


# ---------------------------------------------------------------------------
# Expression helpers (for is_done / goal_distance from done AST)
# ---------------------------------------------------------------------------

def _eval_expr(expr, ns):
    if isinstance(expr, LiteralExpr):
        return expr.value
    if isinstance(expr, RefExpr):
        return ns.get(expr.path[0], 0.0)
    if isinstance(expr, BinaryExpr):
        l, r = _eval_expr(expr.left, ns), _eval_expr(expr.right, ns)
        if l is None: l = 0.0
        if r is None: r = 0.0
        ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,
               '*': lambda a,b: a*b, '/': lambda a,b: a/b if b else 0,
               '>=': lambda a,b: a>=b, '>': lambda a,b: a>b,
               '<=': lambda a,b: a<=b, '<': lambda a,b: a<b,
               '==': lambda a,b: a==b,
               'and': lambda a,b: a and b, 'or': lambda a,b: a or b,
               'implies': lambda a,b: (not a) or b}
        return ops[expr.op](l, r)
    if isinstance(expr, UnaryExpr):
        v = _eval_expr(expr.operand, ns)
        if expr.op == 'not': return not v
        if expr.op == '-': return -(v or 0)
    if isinstance(expr, TernaryExpr):
        c = _eval_expr(expr.condition, ns)
        return _eval_expr(expr.true_expr, ns) if c else _eval_expr(expr.false_expr, ns)
    return 0.0


def _flatten_and(expr):
    if isinstance(expr, BinaryExpr) and expr.op == 'and':
        return _flatten_and(expr.left) + _flatten_and(expr.right)
    return [expr]


# ---------------------------------------------------------------------------
# SysML interface extraction
# ---------------------------------------------------------------------------

def extract_interface(model_path: str, dt: float = 0.1):
    """Extract obs_names, action_names, is_done, goal_distance from SysML.

    Returns dict with everything the oracle and environment need.
    No simulation engine, no propagationDelay — the oracle is derived
    directly from the #NeuralRequirement via SpecShield.
    """
    parser = SysMLParser(model_path)
    parser.parse()

    # Find #Neural action def
    ctrl_inst = parser.part_instances[parser.controller_part]
    ctrl_def = parser.part_defs[ctrl_inst.part_type]

    neural_def = None
    for ad in ctrl_def.action_defs:
        if 'Neural' in ad.metadata:
            neural_def = ad
            break
    if not neural_def:
        raise ValueError("No #Neural action def found")

    # Find the SubactionCallStmt that invokes it
    def find_call(stmts, type_name):
        for s in stmts:
            if isinstance(s, SubactionCallStmt) and s.type_name == type_name:
                return s
            if hasattr(s, 'body') and isinstance(s.body, list):
                r = find_call(s.body, type_name)
                if r:
                    return r
        return None

    call_stmt = None
    for action in ctrl_def.actions:
        call_stmt = find_call(action.body, neural_def.name)
        if call_stmt:
            break
    if not call_stmt:
        raise ValueError(f"No call to {neural_def.name} found")

    # Extract obs/action names
    obs_names = [p.name for p in neural_def.in_params if p.name.lower() != 'done']
    action_names = [p.name for p in neural_def.out_params]

    # Build ref -> obs_name map, extract done expr
    ref_to_obs = {}
    done_expr_ast = None
    for b in call_stmt.bindings:
        if not isinstance(b, InputBindingStmt):
            continue
        if b.name.lower() == 'done':
            done_expr_ast = b.expr
        elif isinstance(b.expr, RefExpr):
            ref_to_obs['.'.join(b.expr.path)] = b.name

    # Rewrite done expr to use obs names
    def rewrite(expr):
        if isinstance(expr, RefExpr):
            key = '.'.join(expr.path)
            if key in ref_to_obs:
                return RefExpr([ref_to_obs[key]])
            return expr
        if isinstance(expr, BinaryExpr):
            return BinaryExpr(expr.op, rewrite(expr.left), rewrite(expr.right))
        if isinstance(expr, UnaryExpr):
            return UnaryExpr(expr.op, rewrite(expr.operand))
        return expr

    done_ast = rewrite(done_expr_ast) if done_expr_ast else None

    def is_done(obs_dict):
        if not done_ast:
            return False
        return bool(_eval_expr(done_ast, obs_dict))

    goal_terms = _flatten_and(done_ast) if done_ast else []

    def goal_distance(obs_dict):
        total = 0.0
        for term in goal_terms:
            if isinstance(term, BinaryExpr) and term.op in ('>=', '>', '<=', '<', '=='):
                l = _eval_expr(term.left, obs_dict)
                r = _eval_expr(term.right, obs_dict)
                if term.op in ('>=', '>'):
                    gap = max(0.0, r - l)
                elif term.op in ('<=', '<'):
                    gap = max(0.0, l - r)
                else:
                    gap = abs(l - r)
                total += gap ** 2
        return math.sqrt(total)

    # Build SpecShield for oracle labeling
    spec_shield = SpecShield(model_path)

    return {
        "obs_names": obs_names,
        "action_names": action_names,
        "is_done": is_done,
        "goal_distance": goal_distance,
        "spec_shield": spec_shield,
    }


# ---------------------------------------------------------------------------
# spec_oracle: label actions from the requirement AST
# ---------------------------------------------------------------------------

def spec_oracle(spec_shield: SpecShield, obs_dict: dict):
    """Derive the correct action directly from the #NeuralRequirement.

    Evaluates the requirement AST for every valid action combo against
    the current observations. Returns the correct action as an integer.

    This is equivalent to what the shield does at runtime, but used at
    training time to generate (observation, action) labels for the oracle
    cloning phase.
    """
    return spec_shield._requirement_action(obs_dict)
