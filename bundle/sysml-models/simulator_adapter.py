"""
Adapter that wraps the SysML simulator as a twin callable for train.py.

Twin protocol (from train.py):
    state = twin()          # reset: returns initial state dict
    state = twin(action)    # step: advances simulation, returns new state dict

Implementation uses coroutine-style control flow via threading:
    - The simulator runs in a background thread, calling engine.step(dt) in a loop.
    - When the simulator hits a Neural sub-action, it calls engine.model(inputs).
    - engine.model blocks the sim thread and unblocks the main thread (twin caller).
    - twin(action) unblocks engine.model (which returns action), sim continues.
"""

import threading
from typing import Any, Optional

from sysml_parser import SysMLParser
from simulator import SimulationEngine


class SimulatorTwin:
    """Wraps SimulationEngine as a twin callable for the RL training loop.

    The model function on the engine is the coroutine bridge: when the
    simulator calls model(inputs), control yields back to the twin caller.
    twin(action) resumes the simulator by making model() return action.
    """

    def __init__(self, model_path: str, dt: float = 0.1):
        self._model_path = model_path
        self._dt = dt
        self._parser = SysMLParser(model_path)
        self._parser.parse()

        self._engine: Optional[SimulationEngine] = None
        self._sim_thread: Optional[threading.Thread] = None

        # Synchronization between twin() calls and model() calls
        self._model_called = threading.Event()
        self._action_ready = threading.Event()
        self._model_inputs: dict = {}
        self._action: dict = {}
        self._sim_error: Optional[Exception] = None
        self._sim_done = threading.Event()

    def _model_fn(self, inputs: dict) -> dict:
        """Injected as engine.model. Blocks sim thread, yields to twin caller."""
        self._model_inputs = dict(inputs)
        self._model_called.set()
        self._action_ready.wait()
        self._action_ready.clear()
        if self._sim_done.is_set():
            raise _SimulationStopped
        return dict(self._action)

    def _run_simulation(self):
        """Simulation loop running in background thread."""
        try:
            while not self._sim_done.is_set():
                self._engine.step(self._dt)
        except _SimulationStopped:
            pass
        except Exception as e:
            self._sim_error = e
            self._model_called.set()  # unblock caller on error

    def _stop(self):
        """Stop any running simulation thread."""
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_done.set()
            self._action_ready.set()  # unblock model_fn if waiting
            self._sim_thread.join(timeout=2.0)

    def __call__(self, action: Optional[dict] = None) -> dict:
        if action is None:
            return self._reset()
        return self._step(action)

    def _reset(self) -> dict:
        self._stop()

        self._engine = SimulationEngine(self._parser)
        self._engine.model = self._model_fn
        self._engine.initialize()

        self._model_called.clear()
        self._action_ready.clear()
        self._sim_done.clear()
        self._sim_error = None

        self._sim_thread = threading.Thread(
            target=self._run_simulation, daemon=True)
        self._sim_thread.start()

        self._model_called.wait()
        self._model_called.clear()
        if self._sim_error:
            raise self._sim_error

        return self._build_state()

    def _step(self, action: dict) -> dict:
        self._action = action
        self._action_ready.set()

        self._model_called.wait()
        self._model_called.clear()
        if self._sim_error:
            raise self._sim_error

        return self._build_state()

    def _build_state(self) -> dict:
        """Build the state dict returned to the training loop.

        Contains the Neural action's in-params (the policy observation)
        plus simulator state needed for reward/done checks.
        """
        state = dict(self._model_inputs)
        return state


class _SimulationStopped(Exception):
    """Raised inside model_fn to cleanly exit the sim thread on reset."""
    pass
