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

from sysml_parser import SysMLParser, BinaryExpr, RefExpr, LiteralExpr, UnaryExpr
from simulator_adapter import SimulatorTwin

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
        neural_defs = []
        for pdef in parser.part_defs.values():
            for ad in pdef.action_defs:
                if "Neural" in ad.metadata:
                    neural_defs.append(ad)
        if len(neural_defs) != 1:
            raise ValueError(
                f"expected exactly one #Neural action, found {len(neural_defs)}"
            )
        neural = neural_defs[0]
        completion = [
            p.name for p in neural.in_params if "Completion" in p.metadata
        ]
        if len(completion) != 1:
            raise ValueError("#Neural action must mark exactly one input #Completion")
        self._completion_key = completion[0]
        self._obs_keys = [
            p.name for p in neural.in_params if "Completion" not in p.metadata
        ]
        self._out_params = [(p.name, p.type_name) for p in neural.out_params]

        # Build the output representation from the types declared in SysML.
        n_out = len(self._out_params)
        self._action_map = {}
        output_types = {(type_name or "").lower() for _name, type_name in self._out_params}
        if output_types <= {"bool", "boolean"}:
            self._initial_action = {name: False for name, _type in self._out_params}
            for action_id in range(2 ** n_out):
                actuators = {}
                for bit, (name, _) in enumerate(self._out_params):
                    actuators[name] = bool(action_id & (1 << bit))
                self._action_map[action_id] = actuators
        elif output_types <= {"real", "float", "double", "integer", "int"}:
            self._initial_action = {name: 0.0 for name, _type in self._out_params}
        else:
            raise ValueError(f"unsupported or mixed #Neural output types: {output_types}")

        self.obs_dim = len(self._obs_keys)
        self.n_actions = len(self._action_map)

        # Extract scenario inputs with bounds from ScenarioConstraint
        self._scenario_inputs = _extract_scenario_bounds(parser)

        # Extract relations among scenario inputs and initialized model state.
        self._cross_constraints = _extract_cross_constraints(parser)
        self._scenario_state_bindings = _extract_scenario_state_bindings(parser)

        # Compute global normalization scale from initial obs
        self._obs_scale = self._compute_obs_scale()

    @staticmethod
    def _state_value(state: dict, key: str):
        if key not in state:
            raise KeyError(f"SysML simulation did not produce neural input {key}")
        return state[key]

    def _compute_obs_scale(self):
        """Run one init step and use max absolute obs value as global scale."""
        self._twin()  # reset
        state = self._twin(self._initial_action)
        scale = 1.0
        for key in self._obs_keys:
            val = self._state_value(state, key)
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
        # Enforce relations among sampled inputs exactly as written in SysML.
        for greater_qname, lesser_qname in self._cross_constraints:
            g = self._state_value(eng.state, greater_qname)
            l = self._state_value(eng.state, lesser_qname)
            if g < l:
                eng.state[lesser_qname] = g
        for state_qname, input_qname in self._scenario_state_bindings:
            eng.state[state_qname] = self._state_value(eng.state, input_qname)

    def _state_to_obs(self, state: dict) -> np.ndarray:
        """Convert twin state dict to normalized observation vector."""
        obs = []
        for key in self._obs_keys:
            val = self._state_value(state, key)
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
                if not entry["status"]:
                    return -1.0, True

        if bool(self._state_value(state, self._completion_key)):
            return 1.0, True

        return -0.01, False

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._twin()
        self._randomize_scenario()
        self._step_count = 0
        self._twin(self._initial_action)
        state = self._twin(self._initial_action)
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

    Returns dict mapping qualified_name -> {lower, upper}.
    """
    inputs = {}
    for p in parser.parameters:
        if "ScenarioInput" in p.metadata:
            inputs[p.qualified_name] = {
                "lower": None,
                "upper": None,
            }

    # Walk ScenarioConstraint expressions for constant bounds
    for constraint in parser.parsed_constraints:
        if 'ScenarioConstraint' in getattr(constraint, 'metadata', []):
            _walk_bounds(constraint.expression, inputs, parser.system_part)

    # Every randomized input must have complete bounds in its SysML file.
    for qname, info in inputs.items():
        if info["lower"] is None or info["upper"] is None:
            raise ValueError(
                f"#ScenarioInput {qname} needs lower and upper bounds in "
                "a #ScenarioConstraint"
            )

    return inputs


def _extract_cross_constraints(parser: SysMLParser) -> list:
    """Extract var >= var constraints from #ScenarioConstraint.

    Returns list of (greater_qname, lesser_qname) pairs meaning
    greater_qname >= lesser_qname must hold after sampling.
    """
    constraints = []
    system = parser.system_part

    # Collect ScenarioInput qnames for matching
    input_qnames = set()
    for p in parser.parameters:
        if "ScenarioInput" in p.metadata:
            input_qnames.add(p.qualified_name)

    for constraint in parser.parsed_constraints:
        if 'ScenarioConstraint' in getattr(constraint, 'metadata', []):
            _walk_cross_constraints(
                constraint.expression, input_qnames, system, constraints)

    return constraints


def _extract_scenario_state_bindings(parser: SysMLParser) -> list[tuple[str, str]]:
    """Read state == ScenarioInput relations used during scenario setup."""
    input_qnames = {
        parameter.qualified_name
        for parameter in parser.parameters
        if "ScenarioInput" in parameter.metadata
    }
    bindings: list[tuple[str, str]] = []

    def walk(expr) -> None:
        if not isinstance(expr, BinaryExpr):
            return
        if expr.op == "and":
            walk(expr.left)
            walk(expr.right)
            return
        if expr.op != "==" or not (
            isinstance(expr.left, RefExpr) and isinstance(expr.right, RefExpr)
        ):
            return
        left = parser.system_part + "::" + "::".join(expr.left.path)
        right = parser.system_part + "::" + "::".join(expr.right.path)
        if left in input_qnames and right not in input_qnames:
            bindings.append((right, left))
        elif right in input_qnames and left not in input_qnames:
            bindings.append((left, right))

    for constraint in parser.parsed_constraints:
        if "ScenarioConstraint" in getattr(constraint, "metadata", []):
            walk(constraint.expression)
    return bindings


def _walk_cross_constraints(expr, input_qnames, system, out):
    """Extract var >= var constraints recursively."""
    if not isinstance(expr, BinaryExpr):
        return
    if expr.op == 'and':
        _walk_cross_constraints(expr.left, input_qnames, system, out)
        _walk_cross_constraints(expr.right, input_qnames, system, out)
        return

    if expr.op in ('>=', '<='):
        if isinstance(expr.left, RefExpr) and isinstance(expr.right, RefExpr):
            left_qname = system + "::" + "::".join(expr.left.path)
            right_qname = system + "::" + "::".join(expr.right.path)
            if left_qname in input_qnames and right_qname in input_qnames:
                if expr.op == '>=':
                    out.append((left_qname, right_qname))
                else:
                    out.append((right_qname, left_qname))


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
        ref, value = None, None
        right_value = _numeric_literal(expr.right)
        left_value = _numeric_literal(expr.left)
        if isinstance(expr.left, RefExpr) and right_value is not None:
            ref, value, op = expr.left, right_value, expr.op
        elif left_value is not None and isinstance(expr.right, RefExpr):
            ref, value = expr.right, left_value
            op = {'>=': '<=', '<=': '>=', '==': '=='}[expr.op]
        else:
            return

        qname = system + "::" + "::".join(ref.path)
        if qname not in inputs:
            return

        if op == '==' or op == '>=':
            current = inputs[qname]["lower"]
            inputs[qname]["lower"] = max(
                float('-inf') if current is None else current, value)
        if op == '==' or op == '<=':
            cur = inputs[qname]["upper"]
            inputs[qname]["upper"] = value if cur is None else min(cur, value)


def _numeric_literal(expr):
    """Return a numeric constant, including a signed SysML literal."""
    if isinstance(expr, LiteralExpr) and isinstance(expr.value, (int, float)):
        return expr.value
    if isinstance(expr, UnaryExpr) and expr.op == '-':
        value = _numeric_literal(expr.operand)
        return None if value is None else -value
    return None
