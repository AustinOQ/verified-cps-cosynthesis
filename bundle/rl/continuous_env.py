"""
Continuous-action environment: same SysML simulation twin as SysMLEnv, but the
action is a real vector (the #Neural action's real out-params) instead of a
discrete 2^N index.

Reuses everything from SysMLEnv except the action plumbing:
  - observation space, normalization, scenario randomization, reward, done
    are inherited unchanged.
  - the action is passed to the twin as {out_param_name: float} rather than
    looked up in a boolean action map.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from env import SysMLEnv


class SysMLContinuousEnv(SysMLEnv):
    """RL environment with a continuous (real-valued) action space."""

    def __init__(self, model_path, dt: float, max_steps: int = 1200,
                 phase: int = 1, rng_seed: int = None):
        # super() builds the parser/twin, scenario bounds, obs scale, and a
        # (now-unused) discrete action map. We override the action handling.
        super().__init__(model_path, dt=dt, max_steps=max_steps,
                         phase=phase, rng_seed=rng_seed)
        self._out_names = [name for name, _type in self._out_params]
        self.act_dim = len(self._out_names)
        self._noop = dict(self._initial_action)

    def _act_to_dict(self, action) -> dict:
        if self.act_dim == 1:
            try:
                val = float(action)
            except TypeError:
                val = float(action[0])
            return {self._out_names[0]: val}
        return {name: float(action[i]) for i, name in enumerate(self._out_names)}

    def reset(self):
        self._twin()
        self._randomize_scenario()
        self._step_count = 0
        self._twin(self._noop)          # sensor sends; controller hasn't read yet
        state = self._twin(self._noop)  # controller reads sensor, done correct
        self._step_count = 0
        return self._state_to_obs(state)

    def step(self, action):
        actuators = self._act_to_dict(action)
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
