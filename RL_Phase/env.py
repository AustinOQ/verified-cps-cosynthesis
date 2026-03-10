"""
env.py — Gymnasium environment wrapping a SysML SimulatorTwin.

This module provides `SysMLEnv`, a Gymnasium-compatible environment that
wraps any SysML v2 model's digital twin (SimulatorTwin) as an RL environment.
The environment is constructed entirely from extracted SysML metadata — no
model-specific code is needed.

Scenario Randomization
----------------------
Start-state randomization is driven entirely by SysML metadata:
  - `#ScenarioInput` on attributes identifies which parameters to randomize.
  - `#ScenarioConstraint` on constraints defines valid ranges for sampling.
Each `#ScenarioInput` parameter's `qualified_name` maps directly to an
engine state key — no naming heuristics are needed.

Reward Protocol
---------------
- **Sparse mode** (default): +1 goal, -1 prohibition violation, 0 otherwise
- **Shaped mode** (--shaping): +10 goal, -1 prohibition (terminate),
  per-step normalized distance improvement (~1.0 total over a full episode)
- Only `#Prohibition` requirements trigger penalties. `#Obligation`
  requirements are NOT penalized — the agent learns to satisfy them
  through the goal reward.

Termination
-----------
Episodes terminate on: goal reached, prohibition violation, or max_steps.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add sysml-models directory to path for simulator imports.
_SYSML_MODELS_DIR = str(
    Path(__file__).resolve().parent.parent / "sysml-models"
)
if _SYSML_MODELS_DIR not in sys.path:
    sys.path.insert(0, _SYSML_MODELS_DIR)

from simulator_adapter import SimulatorTwin
from .extractor import NeuralInterface


def _to_float(val) -> float:
    """Convert a simulator value to float. Handles None, bool, numeric."""
    if val is None:
        return 0.0
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    return float(val)


# ------------------------------------------------------------------
# Goal-distance computation from done expression (GENERAL)
#
# These functions parse the done_expr (a Python expression string
# extracted from the SysML model) into its AST and compute a
# continuous distance metric. This enables reward shaping without
# any model-specific knowledge — the done expression IS the
# specification of what "goal reached" means.
#
# Supported operators:
#   >=, >, <=, <  -> one-sided squared distance (0 when satisfied)
#   ==            -> squared distance
#   and           -> sum of sub-distances
#   or            -> min of sub-distances
#   not           -> boolean negation distance (value^2, 0 when falsy)
#   +, -, *, /    -> arithmetic in comparands
# ------------------------------------------------------------------

def _goal_distance_from_expr(done_expr: str, obs_dict: dict) -> float:
    """Compute continuous distance to the goal encoded in done_expr.

    Returns 0.0 when the done condition is satisfied.

    Example:
        done_expr = "(x >= 10) and (y >= 20)"
        obs_dict = {"x": 5.0, "y": 15.0}
        -> (10-5)^2 + (20-15)^2 = 25 + 25 = 50.0
    """
    tree = ast.parse(done_expr, mode='eval')
    return _eval_dist_node(tree.body, obs_dict)


def _eval_dist_node(node, ns):
    """Evaluate distance for an AST node."""
    if isinstance(node, ast.BoolOp):
        parts = [_eval_dist_node(v, ns) for v in node.values]
        # AND = all must be satisfied = sum distances
        # OR = any one suffices = take minimum distance
        return sum(parts) if isinstance(node.op, ast.And) else min(parts)
    if isinstance(node, ast.Compare):
        left = _eval_arith_node(node.left, ns)
        total = 0.0
        cur = left
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_arith_node(comp, ns)
            if isinstance(op, (ast.GtE, ast.Gt)):
                # a >= b: distance = max(0, b - a)^2
                total += max(0.0, right - cur) ** 2
            elif isinstance(op, (ast.LtE, ast.Lt)):
                # a <= b: distance = max(0, a - b)^2
                total += max(0.0, cur - right) ** 2
            elif isinstance(op, ast.Eq):
                total += (cur - right) ** 2
            cur = right
        return total
    # FIX 3: Handle boolean negation.
    # `not x` should have distance 0 when x is falsy, positive when truthy.
    # We evaluate the operand as an arithmetic value and use val^2 as
    # the distance — 0 when the operand is 0 (False), 1 when it's 1 (True).
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        val = _eval_arith_node(node.operand, ns)
        return val ** 2
    return 0.0


def _eval_arith_node(node, ns):
    """Evaluate an arithmetic AST node using observation values."""
    if isinstance(node, ast.Name):
        return float(ns.get(node.id, 0.0))
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        l = _eval_arith_node(node.left, ns)
        r = _eval_arith_node(node.right, ns)
        if isinstance(node.op, ast.Sub): return l - r
        if isinstance(node.op, ast.Add): return l + r
        if isinstance(node.op, ast.Mult): return l * r
        if isinstance(node.op, ast.Div): return l / r if r else 0.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arith_node(node.operand, ns)
    return 0.0


# ------------------------------------------------------------------
# Done expression variable discovery
# ------------------------------------------------------------------

def _find_done_expr_vars(done_expr: str) -> set[str]:
    """Return all variable names referenced in the done expression."""
    tree = ast.parse(done_expr, mode='eval')
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


# ------------------------------------------------------------------
# Scenario constraint evaluation
# ------------------------------------------------------------------

def _normalize_constraint_expr(raw_text: str) -> str:
    """Convert a ScenarioConstraint raw_text to a Python expression.

    Strips dotted prefixes (e.g. "controller.tank1TransferMl" -> "tank1TransferMl")
    and collapses whitespace so the expression can be eval'd with Parameter.name
    as variable names.
    """
    # FIX 1: Only match identifiers (starting with letter/underscore), not
    # decimal numbers like 12.78. The old regex r'(\w+)\.(\w+)' treated
    # '12.78' as two \w+ groups separated by a dot, replacing it with '78'.
    expr = re.sub(r'([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)', r'\2', raw_text)
    return ' '.join(expr.split())


def _parse_constraint_bounds(constraint_expr: str, param_names: list[str]) -> dict[str, tuple[float, float]]:
    """Extract [lower, upper] bounds for each parameter from a constraint expression.

    Parses patterns like 'paramName >= 12.78' and 'paramName <= 37.78' from
    the normalized constraint string.

    Returns:
        Dict mapping param name -> (lower_bound, upper_bound).
        Uses -inf/+inf when a bound is not found in the constraint.
    """
    bounds = {}
    for name in param_names:
        lower = -float('inf')
        upper = float('inf')
        # Match: name >= value  or  value <= name
        for m in re.finditer(rf'{re.escape(name)}\s*>=\s*([-\d.eE]+)', constraint_expr):
            lower = max(lower, float(m.group(1)))
        for m in re.finditer(rf'([-\d.eE]+)\s*<=\s*{re.escape(name)}', constraint_expr):
            lower = max(lower, float(m.group(1)))
        # Match: name <= value  or  value >= name
        for m in re.finditer(rf'{re.escape(name)}\s*<=\s*([-\d.eE]+)', constraint_expr):
            upper = min(upper, float(m.group(1)))
        for m in re.finditer(rf'([-\d.eE]+)\s*>=\s*{re.escape(name)}', constraint_expr):
            upper = min(upper, float(m.group(1)))
        bounds[name] = (lower, upper)
    return bounds


class SysMLEnv(gym.Env):
    """Gymnasium environment wrapping a SysML digital twin.

    This environment is constructed from a NeuralInterface (extracted from
    the SysML model) and uses SimulatorTwin for dynamics. No model-specific
    code is needed for the core train/eval loop.

    The `done` signal from the SysML model's Neural action def is used
    directly for episode termination. It is excluded from the policy's
    observation vector (the agent doesn't see it as an input).

    Args:
        model_path:     Path to the .sysml model file.
        interface:      NeuralInterface from extractor.py.
        dt:             Simulation timestep in seconds.
        max_steps:      Maximum steps per episode before truncation.
        randomize:      If True, randomize scenario inputs each episode
                        using #ScenarioInput / #ScenarioConstraint metadata.
        shaping:        If True, use distance-based reward shaping instead
                        of sparse rewards.
    """

    def __init__(self, model_path: str, interface: NeuralInterface,
                 dt: float = 0.1, max_steps: int = 200,
                 randomize: bool = False, shaping: bool = False):
        super().__init__()
        self.interface = interface
        self.max_steps = max_steps
        self._twin = SimulatorTwin(model_path, dt=dt)
        self._engine = None
        self._step_count = 0
        self._done_expr = interface.done_expr
        self._randomize = randomize
        self._shaping = shaping

        # Distance tracking for reward shaping.
        self._d_initial = 0.0
        self._last_distance = 0.0

        # Exclude `done` from policy observations (used only for termination).
        self._obs_indices = []
        for i, (name, typ) in enumerate(zip(interface.obs_names, interface.obs_types)):
            if name.lower() == 'done' and typ == 'Boolean':
                continue
            self._obs_indices.append(i)

        n_obs = len(self._obs_indices)
        n_act = len(interface.action_names)

        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.MultiBinary(n_act)
        self._obs_scale = None

        # Pre-compile scenario constraint expression for rejection sampling.
        self._scenario_constraint_expr = None
        self._scenario_bounds = {}
        if interface.scenario_constraints:
            raw = ' and '.join(c.raw_text for c in interface.scenario_constraints)
            self._scenario_constraint_expr = _normalize_constraint_expr(raw)
            # FIX 4: Parse explicit bounds from constraint for sampling.
            param_names = [p.name for p in (interface.scenario_inputs or [])]
            self._scenario_bounds = _parse_constraint_bounds(
                self._scenario_constraint_expr, param_names)

        # FIX 2: Discover variables in done_expr that are NOT observations.
        # Build a static lookup for parameters (e.g. toleranceCelcius) and
        # identify engine-state variables (e.g. heaterOn, acOn) that must
        # be read from the engine each step.
        self._done_extra_static = {}   # name -> float (constant params)
        self._done_extra_dynamic = []  # names to read from engine state
        if self._done_expr:
            from sysml_parser import SysMLParser
            obs_set = {n for n in interface.obs_names if n.lower() != 'done'}
            all_vars = _find_done_expr_vars(self._done_expr)
            missing = all_vars - obs_set

            if missing:
                # Try to resolve from parser parameters (static values).
                parser = self._twin._parser
                param_by_name = {p.name: p.value for p in parser.parameters}
                for var in missing:
                    if var in param_by_name:
                        self._done_extra_static[var] = param_by_name[var]
                    else:
                        # Must be dynamic state (e.g. controller attributes
                        # like heaterOn, acOn). Will be resolved from engine
                        # state at each step.
                        self._done_extra_dynamic.append(var)

    # ------------------------------------------------------------------
    # Goal distance (GENERAL)
    # ------------------------------------------------------------------

    def _obs_dict(self, state: dict) -> dict:
        """Build {var_name: value} dict for distance computation.

        Includes all variables referenced in done_expr:
          - Observations from the Neural action's in-params (from state dict)
          - Static parameters like toleranceCelcius (from parser, set at init)
          - Dynamic engine state like heaterOn, acOn (resolved each call)
        """
        d = {}
        # Observations from state dict.
        for name in self.interface.obs_names:
            if name.lower() != 'done':
                d[name] = _to_float(state.get(name))
        # FIX 2: Add static parameter values (e.g. toleranceCelcius).
        d.update(self._done_extra_static)
        # FIX 2: Add dynamic engine state values (e.g. heaterOn, acOn).
        if self._engine and self._done_extra_dynamic:
            for var in self._done_extra_dynamic:
                # Search engine state for a key ending with ::varName.
                for key, val in self._engine.state.items():
                    if key.endswith('::' + var):
                        d[var] = _to_float(val)
                        break
                else:
                    # Also check state dict directly (twin may surface it).
                    d.setdefault(var, _to_float(state.get(var, 0.0)))
        return d

    def _goal_distance(self, state: dict) -> float:
        """Continuous distance to done condition. 0 when goal is met."""
        if not self._done_expr:
            return 0.0
        return _goal_distance_from_expr(self._done_expr, self._obs_dict(state))

    # ------------------------------------------------------------------
    # Scenario randomization (metadata-driven, no heuristics)
    # ------------------------------------------------------------------

    def _randomize_start(self) -> dict:
        """Randomize #ScenarioInput parameters and patch the engine.

        Samples random values for each #ScenarioInput parameter using
        bounds parsed from #ScenarioConstraint, validates them via
        rejection sampling, and writes them directly to the engine state
        using each parameter's qualified_name.

        Returns the updated state dict (matching twin._model_inputs format).
        """
        rng = self.np_random
        scenario_inputs = self.interface.scenario_inputs or []
        if not scenario_inputs:
            return dict(self._twin._model_inputs)

        for _ in range(1000):
            vals = {}
            for param in scenario_inputs:
                # FIX 4: Sample from constraint bounds when available,
                # falling back to [0, default_value] if no bounds found.
                lo, hi = self._scenario_bounds.get(
                    param.name, (0.0, param.value))
                # Replace infinite bounds with reasonable defaults.
                if lo == -float('inf'):
                    lo = min(0.0, param.value)
                if hi == float('inf'):
                    hi = max(0.0, param.value)
                # Ensure lo <= hi.
                if lo > hi:
                    lo, hi = hi, lo
                vals[param.name] = rng.uniform(lo, hi)

            # Validate against #ScenarioConstraint if present.
            if self._scenario_constraint_expr:
                ns = dict(vals)
                ns["__builtins__"] = {}
                try:
                    if not eval(self._scenario_constraint_expr, ns):
                        continue
                except Exception:
                    continue

            # Valid sample — patch engine state.
            for param in scenario_inputs:
                self._engine.state[param.qualified_name] = vals[param.name]

            # Return updated model_inputs state.
            new_state = dict(self._twin._model_inputs)
            for param in scenario_inputs:
                # Find the obs_name that binds to this parameter.
                if self.interface.obs_bindings:
                    for obs_name, ref in self.interface.obs_bindings.items():
                        if ref == param.name:
                            new_state[obs_name] = vals[param.name]
                            break
            return new_state

        # Fallback: return unmodified state if no valid sample found.
        return dict(self._twin._model_inputs)

    # ------------------------------------------------------------------
    # Core Gymnasium methods (GENERAL)
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self._twin()           # Reset simulator twin.
        self._engine = self._twin._engine
        self._step_count = 0

        if self._randomize:
            state = self._randomize_start()

        # Compute initial goal distance for reward shaping.
        if self._shaping:
            self._d_initial = self._goal_distance(state)
            self._last_distance = self._d_initial

        raw = self._raw_obs(state)
        if self._obs_scale is None:
            self._obs_scale = np.maximum(np.abs(raw), 1.0)

        return raw / self._obs_scale, {}

    def step(self, action):
        # Convert MultiBinary array to named action dict.
        action_dict = {}
        for i, name in enumerate(self.interface.action_names):
            action_dict[name] = bool(action[i])

        state = self._twin(action_dict)
        self._step_count += 1
        obs = self._raw_obs(state) / self._obs_scale

        # 1. Check done using the simulator's done signal.
        done = bool(state.get('done', False))
        if done:
            reward = 10.0 if self._shaping else 1.0
            return obs, reward, True, False, {"violation": None}

        # 2. Then check prohibition violations.
        violation = self._check_violation()
        if violation:
            return obs, -1.0, True, False, {"violation": violation}

        # 3. Normal step — add distance shaping reward if enabled.
        truncated = self._step_count >= self.max_steps
        reward = 0.0
        if self._shaping and self._d_initial > 0:
            d_now = self._goal_distance(state)
            # Normalized improvement: fraction of initial distance closed.
            reward = (self._last_distance - d_now) / self._d_initial
            self._last_distance = d_now

        return obs, reward, False, truncated, {"violation": None}

    def _raw_obs(self, state: dict) -> np.ndarray:
        """Extract observation vector from state dict."""
        obs = np.zeros(len(self._obs_indices), dtype=np.float32)
        for j, i in enumerate(self._obs_indices):
            name = self.interface.obs_names[i]
            obs[j] = _to_float(state.get(name))
        return obs

    def _check_violation(self) -> Optional[str]:
        """Check if any #Prohibition requirement is currently violated.

        Returns the requirement name if violated, None otherwise.
        Only Prohibition requirements trigger penalties — Obligation
        requirements are left for the agent to learn naturally.
        """
        for req_name, entry in self._engine.requirement_statuses().items():
            if entry["kind"] == "Prohibition" and not entry["status"]:
                return req_name
        return None

    def close(self):
        if self._twin is not None:
            self._twin._stop()
