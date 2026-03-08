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
# Scenario constraint evaluation
# ------------------------------------------------------------------

def _normalize_constraint_expr(raw_text: str) -> str:
    """Convert a ScenarioConstraint raw_text to a Python expression.

    Strips dotted prefixes (e.g. "controller.tank1TransferMl" -> "tank1TransferMl")
    and collapses whitespace so the expression can be eval'd with Parameter.name
    as variable names.
    """
    expr = re.sub(r'(\w+)\.(\w+)', r'\2', raw_text)
    return ' '.join(expr.split())


class SysMLEnv(gym.Env):
    """Gymnasium environment wrapping a SysML digital twin.

    This environment is constructed from a NeuralInterface (extracted from
    the SysML model) and uses SimulatorTwin for dynamics. No model-specific
    code is needed for the core train/eval loop.

    The simulator's `done` binding value is unreliable (the expression
    evaluator doesn't properly resolve sensor response refs), so we
    exclude `done` from observations and recompute it from the extracted
    done_expr using actual observation values.

    Args:
        model_path:     Path to the .sysml model file.
        interface:      NeuralInterface from extractor.py.
        dt:             Simulation timestep in seconds.
        max_steps:      Maximum steps per episode before truncation.
        randomize:      If True, randomize scenario inputs each episode
                        using #ScenarioInput / #ScenarioConstraint metadata.
        done_threshold: Reduce target values by this amount when checking
                        done, making goal conditions easier to satisfy.
                        Useful when exact convergence is slow.
        shaping:        If True, use distance-based reward shaping instead
                        of sparse rewards.
    """

    def __init__(self, model_path: str, interface: NeuralInterface,
                 dt: float = 0.1, max_steps: int = 200,
                 randomize: bool = False, done_threshold: float = 0.0,
                 shaping: bool = False):
        super().__init__()
        self.interface = interface
        self.max_steps = max_steps
        self._twin = SimulatorTwin(model_path, dt=dt)
        self._engine = None
        self._step_count = 0
        self._done_expr = interface.done_expr
        self._randomize = randomize
        self._done_threshold = done_threshold
        self._shaping = shaping

        # Distance tracking for reward shaping.
        self._d_initial = 0.0
        self._last_distance = 0.0

        # Exclude `done` from obs (simulator value is unreliable).
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
        if interface.scenario_constraints:
            raw = ' and '.join(c.raw_text for c in interface.scenario_constraints)
            self._scenario_constraint_expr = _normalize_constraint_expr(raw)

        # Build set of #ScenarioInput parameter names for done_threshold.
        self._scenario_input_names = set()
        if interface.scenario_inputs:
            self._scenario_input_names = {p.name for p in interface.scenario_inputs}

    # ------------------------------------------------------------------
    # Goal distance (GENERAL)
    # ------------------------------------------------------------------

    def _obs_dict(self, state: dict) -> dict:
        """Build {obs_name: value} dict from state for distance computation."""
        d = {}
        for name in self.interface.obs_names:
            if name.lower() != 'done':
                d[name] = _to_float(state.get(name))
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

        Samples random values for each #ScenarioInput parameter, validates
        them against the #ScenarioConstraint expression via rejection
        sampling, and writes them directly to the engine state using each
        parameter's qualified_name.

        Returns the updated state dict (matching twin._model_inputs format).
        """
        rng = self.np_random
        scenario_inputs = self.interface.scenario_inputs or []
        if not scenario_inputs:
            return dict(self._twin._model_inputs)

        for _ in range(1000):
            vals = {}
            for param in scenario_inputs:
                # Sample uniformly in [0, default_value].
                vals[param.name] = rng.uniform(0.0, param.value)

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

        # 1. Check done (goal reached) first.
        done = self._check_done(state)
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

    def _check_done(self, state: dict) -> bool:
        """Evaluate the done expression against current observation values.

        Uses the Python expression extracted from the SysML model's done
        binding. Optionally applies done_threshold to make convergence
        easier (reduces target values before checking).
        """
        if not self._done_expr:
            return False
        try:
            ns = {}
            for name in self.interface.obs_names:
                if name.lower() == 'done':
                    continue
                ns[name] = _to_float(state.get(name))
            # Apply threshold: reduce scenario input values so done triggers
            # earlier. Only applies to #ScenarioInput-bound observations.
            if self._done_threshold > 0 and self._scenario_input_names:
                bindings = self.interface.obs_bindings or {}
                for obs_name in list(ns):
                    ref = bindings.get(obs_name, '')
                    if ref in self._scenario_input_names:
                        ns[obs_name] = max(0.0, ns[obs_name] - self._done_threshold)
            ns["__builtins__"] = {}
            return bool(eval(self._done_expr, ns))
        except Exception:
            return False

    def close(self):
        if self._twin is not None:
            self._twin._stop()
