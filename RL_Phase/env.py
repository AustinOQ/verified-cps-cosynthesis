"""
env.py — Gymnasium environment wrapping a SysML SimulatorTwin.

This module provides `SysMLEnv`, a Gymnasium-compatible environment that
wraps any SysML v2 model's digital twin (SimulatorTwin) as an RL environment.
The environment is constructed entirely from extracted SysML metadata — no
model-specific code is needed for basic training.

Generality Tiers
-----------------
The code has three tiers of generality:

1. **Fully general** (works for any SysML model with a #Neural action def):
   - Observation/action space construction from NeuralInterface
   - Step/reset using SimulatorTwin
   - Reward from requirement_statuses() (#Prohibition -> penalty)
   - Done checking from extracted done_expr

2. **General for numeric done conditions** (reward shaping):
   - Parses done_expr AST to compute continuous goal distance
   - Works for any done expression using >=, <=, ==, and/or operators
   - Example: "(x >= 10) and (y >= 20)" -> distance = max(0,10-x)^2 + max(0,20-y)^2

3. **Model-specific heuristics** (start-state randomization):
   - _discover_mapping() and _randomize_start() use naming conventions
     (e.g., "target", "original", "capacity") to identify observation roles
   - These work for fluid-transfer models but may need adaptation for
     other domains. Override these methods for new model families.

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
from simulator import BindRef
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
        randomize:      If True, randomize start states each episode.
                        (Uses model-specific heuristics — see _randomize_start.)
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

        # State mapping for randomization (built on first reset).
        self._state_map = None   # obs_name -> [engine state keys]
        self._capacities = None  # obs_name -> capacity value

    # ------------------------------------------------------------------
    # State mapping discovery (MODEL-SPECIFIC HEURISTICS)
    #
    # This method discovers which internal simulator state keys correspond
    # to each observation, enabling start-state randomization. It uses
    # naming conventions from the fluid-transfer model family:
    #   - ".response" in binding ref -> sensor reading -> find physical
    #     state by matching digits and looking for "currentLevelMl"
    #   - "capacityMl" suffix -> tank capacity for normalization
    #   - "volumeSensor" -> sensor device readings
    #
    # For other model families, override this method or extend the
    # heuristics. The core train/eval loop works WITHOUT this method
    # (only needed when randomize=True).
    # ------------------------------------------------------------------

    def _discover_mapping(self):
        """Build obs_name -> engine state keys mapping using binding refs.

        Uses heuristics based on SysML naming conventions to find the
        simulator state keys that correspond to each observation. This
        is needed for start-state randomization to patch the right keys.
        """
        state = self._engine.state
        bindings = self.interface.obs_bindings or {}

        self._state_map = {}
        self._capacities = {}

        # Find controller context prefix (e.g., "system::controller").
        ctrl_prefix = None
        for key in state:
            if '::controller::' in key:
                parts = key.split('::')
                idx = parts.index('controller')
                ctrl_prefix = '::'.join(parts[:idx + 1])
                break

        for obs_name in self.interface.obs_names:
            if obs_name.lower() == 'done':
                continue
            ref = bindings.get(obs_name)
            if not ref:
                continue

            keys = []

            if '.response' in ref:
                # Sensor reading — find the physical state key by digit matching.
                digit = ''.join(c for c in ref if c.isdigit())
                for sk in state:
                    if (sk.endswith('::currentLevelMl')
                            and digit in sk
                            and 'feeder' in sk.lower()):
                        keys.append(sk)
                        prefix = sk.rsplit('::', 1)[0]
                        cap_key = f"{prefix}::capacityMl"
                        cap_val = state.get(cap_key)
                        if isinstance(cap_val, (int, float)):
                            self._capacities[obs_name] = float(cap_val)

                # Also patch the controller's cached sensor response.
                if ctrl_prefix:
                    cache_key = f"{ctrl_prefix}::{ref.replace('.', '::')}"
                    if cache_key in state:
                        keys.append(cache_key)

                # Also patch volumeSensor device readings.
                for sk in state:
                    if ('volumeSensor' in sk and digit in sk
                            and ('reading' in sk or 'rsp' in sk)
                            and not isinstance(state[sk], BindRef)
                            and isinstance(state.get(sk), (int, float))):
                        keys.append(sk)
            else:
                # Direct controller attribute.
                if ctrl_prefix:
                    key = f"{ctrl_prefix}::{ref}"
                    if key in state:
                        keys.append(key)

            self._state_map[obs_name] = keys

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
    # Start state randomization (MODEL-SPECIFIC HEURISTICS)
    #
    # Generates random valid start states for domain randomization.
    # Uses naming conventions to classify observations:
    #   - obs with discovered capacity -> "volume" (randomize in [1, capacity])
    #   - obs with "target"/"transfer" in name -> constrained to [1, volume]
    #   - obs with "original" in name -> set to initial volume
    #
    # For other model families, override this method.
    # ------------------------------------------------------------------

    def _randomize_start(self) -> dict:
        """Generate a random valid start state and patch the engine.

        Returns the new state dict (matching twin._model_inputs format).
        """
        rng = self.np_random
        obs_names = [self.interface.obs_names[i] for i in self._obs_indices]

        # Classify obs by role using naming conventions.
        volume_obs = []   # (obs_name, capacity)
        target_obs = []   # obs_name
        original_obs = [] # obs_name
        for name in obs_names:
            lower = name.lower()
            if name in self._capacities:
                volume_obs.append((name, self._capacities[name]))
            elif 'target' in lower or 'transfer' in lower:
                target_obs.append(name)
            elif 'original' in lower:
                original_obs.append(name)

        # Rejection sampling for a valid state.
        for _ in range(1000):
            new_vals = {}

            # Random volumes in [1, capacity].
            for vname, cap in volume_obs:
                new_vals[vname] = rng.uniform(1.0, cap)

            # Random targets — pair with correct volume by shared digits.
            valid = True
            for tname in target_obs:
                t_digits = {c for c in tname if c.isdigit()}
                paired_vol = None
                for vname, _ in volume_obs:
                    v_digits = {c for c in vname if c.isdigit()}
                    if t_digits & v_digits:
                        paired_vol = vname
                        break
                if paired_vol is None:
                    cap = max(c for _, c in volume_obs) if volume_obs else 1000.0
                    new_vals[tname] = rng.uniform(1.0, cap * 0.5)
                else:
                    max_target = new_vals[paired_vol]
                    if max_target < 1.0:
                        valid = False
                        break
                    new_vals[tname] = rng.uniform(1.0, max_target)

            if not valid:
                continue

            # Originals = volume at start (pair by shared digits).
            for oname in original_obs:
                o_digits = {c for c in oname if c.isdigit()}
                for vname, _ in volume_obs:
                    v_digits = {c for c in vname if c.isdigit()}
                    if o_digits & v_digits:
                        new_vals[oname] = new_vals[vname]
                        break

            break  # Valid state found

        # Patch engine state.
        for obs_name, val in new_vals.items():
            for sk in self._state_map.get(obs_name, []):
                self._engine.state[sk] = val

        # Update derived booleans (isEmpty, isFull).
        for sk in list(self._engine.state):
            if sk.endswith('::isEmpty'):
                prefix = sk.rsplit('::', 1)[0]
                lvl = self._engine.state.get(f"{prefix}::currentLevelMl")
                if isinstance(lvl, (int, float)):
                    self._engine.state[sk] = lvl <= 0
            elif sk.endswith('::isFull'):
                prefix = sk.rsplit('::', 1)[0]
                lvl = self._engine.state.get(f"{prefix}::currentLevelMl")
                cap = self._engine.state.get(f"{prefix}::capacityMl")
                if isinstance(lvl, (int, float)) and isinstance(cap, (int, float)):
                    self._engine.state[sk] = lvl >= cap

        # Build state dict matching model_inputs format.
        new_state = dict(self._twin._model_inputs)
        for obs_name, val in new_vals.items():
            new_state[obs_name] = val
        return new_state

    # ------------------------------------------------------------------
    # Core Gymnasium methods (GENERAL)
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self._twin()           # Reset simulator twin.
        self._engine = self._twin._engine
        self._step_count = 0

        if self._state_map is None:
            self._discover_mapping()

        if self._randomize:
            state = self._randomize_start()

        # Compute initial goal distance for reward shaping.
        if self._shaping:
            self._d_initial = self._goal_distance(state)
            self._last_distance = self._d_initial

        raw = self._raw_obs(state)
        if self._obs_scale is None:
            if self._randomize and self._capacities:
                # Use capacities for stable normalization across episodes.
                scale = np.ones(len(self._obs_indices), dtype=np.float32)
                for j, i in enumerate(self._obs_indices):
                    name = self.interface.obs_names[i]
                    if name in self._capacities:
                        scale[j] = self._capacities[name]
                    else:
                        scale[j] = max(self._capacities.values())
                self._obs_scale = np.maximum(scale, 1.0)
            else:
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
            # Apply threshold: reduce target values so done triggers earlier.
            if self._done_threshold > 0:
                for name in self.interface.obs_names:
                    lower = name.lower()
                    if 'target' in lower or 'transfer' in lower:
                        ns[name] = max(0.0, ns[name] - self._done_threshold)
            ns["__builtins__"] = {}
            return bool(eval(self._done_expr, ns))
        except Exception:
            return False

    def close(self):
        if self._twin is not None:
            self._twin._stop()
