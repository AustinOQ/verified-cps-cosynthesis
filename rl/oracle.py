"""
Brute-force oracle for SysML-extracted neural controllers.

Fully general — works for ANY SysML model:
  1. For each possible action combo:
     a. Re-initialize a fresh engine (clean BindRefs, SMs, mailboxes)
     b. Copy physical state values from the live engine
     c. Apply that action for propagationDelay steps
     d. Read engine.state directly to measure goal distance
  2. Pick the action that reduces goal distance most without violations.

The SysML model MUST define `attribute propagationDelay : Real = N;`
on its system part def. This tells the oracle how many steps to
simulate so the action's effect is observable.
"""

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from sysml_parser import (SysMLParser, InputBindingStmt, SubactionCallStmt,
                           RefExpr, BinaryExpr, LiteralExpr, UnaryExpr,
                           TernaryExpr)
from simulator import SimulationEngine, BindRef


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
# SyncEngine: re-initializes cleanly for each trial
# ---------------------------------------------------------------------------

class SyncEngine:
    """Synchronous simulation engine for oracle evaluation.

    Key idea: for each action trial, we fully re-initialize the engine
    (restoring clean BindRefs, constraint solver, SM states, mailboxes)
    and then overlay the current physical state values. This prevents
    stale sensor data or old actions from interfering with the reading.
    """

    def __init__(self, model_path: str, dt: float = 0.1):
        self._model_path = model_path
        self._parser = SysMLParser(model_path)
        self._parser.parse()
        self._engine = SimulationEngine(self._parser)
        self._engine.initialize()
        self._dt = dt
        self._captured_inputs = {}

    def fresh_trial(self, source_engine, action_dict, n_steps):
        """Re-initialize, copy physical state, apply action, return result.

        Returns (captured_inputs_dict, violation_name_or_None).
        """
        eng = self._engine

        # 1. Full re-initialization — clean BindRefs, SMs, mailboxes, solver
        eng.initialize()

        # 2. Copy non-BindRef state values from source (physical state)
        for key, val in source_engine.state.items():
            if not isinstance(val, BindRef):
                # Only overwrite if the key exists in our engine
                # (both engines parse the same model, so keys match)
                if key in eng.state and not isinstance(eng.state[key], BindRef):
                    eng.state[key] = val

        eng.time = source_engine.time

        # 3. Re-solve constraints with the copied state
        eng.solver.solve(eng.current_sm_state)

        # 4. Set model to return the trial action and capture inputs
        self._captured_inputs = {}
        eng.model = lambda inputs: (
            self._captured_inputs.update(inputs) or action_dict
        )

        # 5. Step forward propagationDelay times
        for _ in range(n_steps):
            eng.step(self._dt)

        # 6. Check requirement violations
        for name, entry in eng.requirement_statuses().items():
            if entry["kind"] in ("Prohibition", "Obligation") and not entry["status"]:
                return dict(self._captured_inputs), name

        return dict(self._captured_inputs), None


# ---------------------------------------------------------------------------
# SysML interface extraction
# ---------------------------------------------------------------------------

def extract_interface(model_path: str, dt: float = 0.1):
    """Extract obs_names, action_names, is_done, goal_distance from SysML.

    Returns dict with everything the oracle needs.
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

    # Create synchronous engine for oracle evaluation
    sync_eng = SyncEngine(model_path, dt=dt)

    # Read propagationDelay from the SysML model (required attribute on system part)
    prop_delay = None
    for key, val in sync_eng._engine.state.items():
        if key.endswith("::propagationDelay") or key == "propagationDelay":
            prop_delay = int(val)
            break
    if prop_delay is None:
        raise ValueError(
            "SysML model must define 'attribute propagationDelay : Real = N;' "
            "on the system part def. This tells the oracle how many steps to "
            "simulate forward when evaluating each action."
        )

    return {
        "obs_names": obs_names,
        "action_names": action_names,
        "is_done": is_done,
        "goal_distance": goal_distance,
        "sync_engine": sync_eng,
        "propagation_delay": prop_delay,
    }


# ---------------------------------------------------------------------------
# brute_force_oracle: try all actions, pick the best
# ---------------------------------------------------------------------------

def brute_force_oracle(sync_engine: SyncEngine, source_engine,
                       obs_names: list, action_names: list,
                       goal_distance, is_done, n_steps: int = 3):
    """Try all 2^N action combos. For each one:
      1. Re-initialize engine with clean state
      2. Copy physical state from source engine
      3. Step with that action for n_steps
      4. Read captured_inputs to measure goal distance
      5. Score it

    Returns boolean array of best action.
    """
    n_act = len(action_names)

    # Get d_before: re-init with no-op to read current obs
    no_op = {n: False for n in action_names}
    cur_obs_raw, _ = sync_engine.fresh_trial(source_engine, no_op, n_steps=1)
    cur_obs = {n: float(cur_obs_raw.get(n, 0)) for n in obs_names}
    d_before = goal_distance(cur_obs)

    best_action = None
    best_score = float('-inf')

    for bits in itertools.product([False, True], repeat=n_act):
        act_dict = {name: bits[i] for i, name in enumerate(action_names)}

        # Fresh engine for each trial
        new_obs_raw, viol = sync_engine.fresh_trial(
            source_engine, act_dict, n_steps=n_steps)
        if viol:
            continue

        new_obs = {n: float(new_obs_raw.get(n, 0)) for n in obs_names}
        d_after = goal_distance(new_obs)
        done = is_done(new_obs)
        score = 1e6 if done else (d_before - d_after)

        if score > best_score:
            best_score = score
            best_action = np.array([float(b) for b in bits], dtype=np.float32)

    if best_action is None:
        return np.zeros(n_act, dtype=np.float32)
    return best_action
