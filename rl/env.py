"""
Gym-style environment wrapping the SysML SimulatorTwin.

Fully general — all structure derived from SysML extraction:
    - Observation space: Neural action in-params
    - Action space: 2^N for N boolean Neural out-params
    - Scenario randomization: bounds from #ScenarioConstraint
    - Normalization: global scale from initial obs values
    - Reward: requirement_statuses from the engine
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from sysml_parser import SysMLParser, BinaryExpr, RefExpr, LiteralExpr
from simulator_adapter import SimulatorTwin

_SAFETY_KINDS = {"Prohibition", "Obligation", None}


class SysMLEnv:
    """RL environment derived from any SysML model with a #Neural action.

    All dimensions, action mappings, scenario ranges, and normalization
    are read from the parsed SysML model — nothing is hardcoded.
    """

    def __init__(self, model_path: str, dt: float = 0.1,
                 max_steps: int = 1200, phase: int = 1,
                 rng_seed: int = None):
        self._model_path = model_path
        self._twin = SimulatorTwin(model_path, dt=dt)
        self._max_steps = max_steps
        self.phase = phase
        self._step_count = 0
        self._rng = np.random.default_rng(rng_seed)

        parser = self._twin._parser

        # Find #Neural action def and extract in/out params
        self._out_params = []
        self._obs_keys = []
        for pdef in parser.part_defs.values():
            for ad in pdef.action_defs:
                if "Neural" in ad.metadata:
                    self._obs_keys = [p.name for p in ad.in_params]
                    self._out_params = [(p.name, p.type_name) for p in ad.out_params]
                    break
            if self._out_params:
                break

        # Build action map: 2^N for N boolean outputs
        n_out = len(self._out_params)
        self._action_map = {}
        for action_id in range(2 ** n_out):
            actuators = {}
            for bit, (name, _) in enumerate(self._out_params):
                actuators[name] = bool(action_id & (1 << bit))
            self._action_map[action_id] = actuators

        self.obs_dim = len(self._obs_keys)
        self.n_actions = len(self._action_map)

        # Extract scenario inputs with bounds from ScenarioConstraint
        self._scenario_inputs = _extract_scenario_bounds(parser)

        # Compute global normalization scale from initial obs
        self._obs_scale = self._compute_obs_scale()

    def _compute_obs_scale(self):
        """Run one init step and use max absolute obs value as global scale."""
        self._twin()  # reset
        state = self._twin(self._action_map[0])  # step with no-op
        scale = 1.0
        for key in self._obs_keys:
            val = state.get(key, 0.0)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                scale = max(scale, abs(val))
        return scale

    def _randomize_scenario(self):
        """Sample each ScenarioInput uniformly within its constraint bounds."""
        eng = self._twin._engine
        for qname, info in self._scenario_inputs.items():
            lo, hi = info["lower"], info["upper"]
            if lo == hi:
                eng.state[qname] = lo
            elif isinstance(lo, int) and isinstance(hi, int):
                eng.state[qname] = float(self._rng.integers(lo, hi + 1))
            else:
                eng.state[qname] = self._rng.uniform(lo, hi)

    def _state_to_obs(self, state: dict) -> np.ndarray:
        """Convert twin state dict to normalized observation vector."""
        obs = []
        for key in self._obs_keys:
            val = state.get(key, 0.0)
            if isinstance(val, bool):
                obs.append(float(val))
            else:
                obs.append(float(val) / self._obs_scale)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, state: dict) -> tuple[float, bool]:
        """Compute reward and done flag.

        Phase 1: ignore safety violations (oracle pretraining).
        Phase 2: -1 terminal penalty for any safety violation.
        """
        statuses = self._twin._engine.requirement_statuses()

        if self.phase == 2:
            for entry in statuses.values():
                if entry["kind"] in _SAFETY_KINDS and not entry["status"]:
                    return -1.0, True

        if state.get("done"):
            return 1.0 - 1.0 / self._max_steps, True

        return -1.0 / self._max_steps, False

    def reset(self) -> np.ndarray:
        """Reset environment with randomized scenario inputs."""
        self._twin()
        self._randomize_scenario()
        self._step_count = 0
        state = self._twin(self._action_map[0])
        self._step_count = 0
        return self._state_to_obs(state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take one step. Returns (obs, reward, done, info)."""
        actuators = self._action_map[action]
        state = self._twin(actuators)
        self._step_count += 1

        reward, done = self._compute_reward(state)

        truncated = self._step_count >= self._max_steps
        if truncated and not done:
            done = True
            reward = 0.0

        info = {"step": self._step_count, "state": state}
        if done:
            info["statuses"] = self._twin._engine.requirement_statuses()

        return self._state_to_obs(state), reward, done, info

    def close(self):
        """Clean up simulator thread."""
        self._twin._stop()


# ---------------------------------------------------------------------------
# ScenarioConstraint bound extraction
# ---------------------------------------------------------------------------

def _extract_scenario_bounds(parser: SysMLParser) -> dict:
    """Extract per-ScenarioInput bounds from #ScenarioConstraint expressions.

    Returns dict mapping qualified_name -> {default, lower, upper}.
    """
    inputs = {}
    for p in parser.parameters:
        if "ScenarioInput" in p.metadata:
            inputs[p.qualified_name] = {
                "default": p.value,
                "lower": None,
                "upper": None,
            }

    # Walk ScenarioConstraint expressions for constant bounds
    for constraint in parser.parsed_constraints:
        if 'ScenarioConstraint' in getattr(constraint, 'metadata', []):
            _walk_bounds(constraint.expression, inputs, parser.system_part)

    # Fill missing bounds with sensible defaults
    for qname, info in inputs.items():
        if info["lower"] is None:
            info["lower"] = 0.0
        if info["upper"] is None:
            info["upper"] = info["default"]

    return inputs


def _walk_bounds(expr, inputs: dict, system: str):
    """Recursively extract constant bounds from a constraint AST."""
    if not isinstance(expr, BinaryExpr):
        return
    if expr.op == 'and':
        _walk_bounds(expr.left, inputs, system)
        _walk_bounds(expr.right, inputs, system)
        return

    # var >= const  →  lower bound
    # var <= const  →  upper bound
    # var == const  →  fixed (lower = upper = const)
    if expr.op in ('>=', '<=', '=='):
        ref, lit = None, None
        if isinstance(expr.left, RefExpr) and isinstance(expr.right, LiteralExpr):
            ref, lit, op = expr.left, expr.right, expr.op
        elif isinstance(expr.left, LiteralExpr) and isinstance(expr.right, RefExpr):
            ref, lit = expr.right, expr.left
            op = {'>=': '<=', '<=': '>=', '==': '=='}[expr.op]
        else:
            return

        qname = system + "::" + "::".join(ref.path)
        if qname not in inputs:
            return

        if op == '==' or op == '>=':
            inputs[qname]["lower"] = max(
                inputs[qname]["lower"] or float('-inf'), lit.value)
        if op == '==' or op == '<=':
            cur = inputs[qname]["upper"]
            inputs[qname]["upper"] = lit.value if cur is None else min(cur, lit.value)
