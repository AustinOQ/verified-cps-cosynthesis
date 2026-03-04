"""
train.py — RL training loop with requirement-derived sparse rewards.

Architecture Context
--------------------
This module sits between two stages of the pipeline:

    UPSTREAM:   SysML v2 parser + simulator produce a digital twin whose
                SimulationEngine exposes requirement_statuses(), returning
                the evaluated status of every requirement in the model.
    THIS FILE:  Trains a neural controller using the twin, rewarding only
                via requirement satisfaction checks (sparse rewards).
    DOWNSTREAM: The trained controller is verified in two phases:
                  1. β-CROWN checks every non-temporal property on the network
                     directly (per-input-point verification).
                  2. nuXmv model-checks the full temporal LTL specification
                     over a finite transition system whose valid transitions
                     are constrained by the non-temporal properties.

Requirement kinds and their role in training:

    Prohibition / Obligation (safety properties):
        Per-step checks. If any requirement of these kinds has status=False,
        the requirement is violated → -1 reward, episode terminates. The
        agent learns to avoid these states entirely.

        During verification (downstream), the same requirements define which
        transitions are POSSIBLE in the nuXmv abstraction. β-CROWN proves
        the trained network satisfies each property for all valid inputs.

    None (unclassified):
        Treated as safety properties (same as Prohibition/Obligation).
        If status=False → violation.

Goal completion is signalled separately via the "done" key in the state
dict. The termination condition is computed by the simulator and passed
as an input to the neural policy (e.g., `in done : Boolean`), so the
controller can observe when the task is complete. The twin surfaces this
in the state dict returned to the training loop.

Why Sparse Rewards
------------------
The reward signal is intentionally sparse: -1 / 0 / +1, nothing else.

    -1  A safety requirement (Prohibition, Obligation, or unclassified) was
        violated this timestep. Episode terminates immediately — no recovery,
        no partial credit.

     0  No violations, goal not yet reached. The agent gets no guidance about
        whether it's "close" to the goal or "far" from a violation.

    +1  The twin signals task completion (done=True in the state dict) and
        no safety requirement was violated on this final step.

    Why not shaped rewards?
        Shaped rewards (e.g., distance-to-goal, proximity-to-violation-boundary)
        were tested in early experiments and led to reward hacking: agents
        optimized the shaping signal without actually satisfying the temporal
        specification. For example, a mixer controller learned to hover near
        the target volume without committing to a transfer, because the
        distance reward was maximized by staying close without finishing.

        Sparse rewards eliminate this failure mode. The agent gets no credit
        for "almost" satisfying the spec — only for actually satisfying it.
        The cost is slower convergence, which we accept because training time
        remains modest (~10-15 minutes for both case studies).

    Why not -1 magnitude shaping?
        All violations are equally fatal: -1 regardless of "how badly" the
        property was violated. This matches the semantics of the requirement,
        where □(level ≥ 2) is either satisfied or not — there is no notion
        of "level = 1.9 is less bad than level = 0.1." A violation is a
        violation. This also simplifies the reward function to a pure boolean
        check per requirement, which is exactly what β-CROWN will verify
        downstream.

Digital Twin Protocol
---------------------
The twin is a single callable with two modes:

    state = twin()          # reset: returns initial state dict
    state = twin(action)    # step:  advances simulation by dt, returns new state

Why a single function instead of a class with reset()/step()?

    - Minimal interface. One callable is the smallest possible contract.
      Any simulation can be wrapped in this interface: a Gym env, a Simulink
      model, an NVIDIA Isaac sim, a real hardware-in-the-loop rig. The
      wrapper is trivially small because there's only one method to implement.

    - Plug-and-play composition. The training loop doesn't know or care what
      the twin does internally. It could be a 247-line Python physics model
      generated from SysML v2 (chemical mixer), or a high-fidelity NVIDIA
      Omniverse simulation (drone). The training code is identical.

Requirement Statuses Protocol
-----------------------------
The requirement_statuses callable returns the current status of every
requirement in the model:

    statuses = requirement_statuses()
    -> {"No Dry Running": {"kind": "Prohibition", "status": True},
        "Stop Simulation": {"kind": "Termination", "status": False}, ...}

Each entry maps a requirement name to:
    kind:   "Prohibition", "Obligation", "Termination", or None
    status: True if the requirement currently holds, False if violated

This callable is provided by SimulationEngine.requirement_statuses and
evaluates the parsed SysML requirement expressions against the live
simulator state. The requirement expressions, their metadata annotations
(#Prohibition, #Obligation, #Termination), and the simulation variables
they reference all originate from the same SysML v2 model — one source
of truth, end to end.

Timestep Convention
-------------------
Each call to twin(action) advances the simulation clock by dt seconds
(default 0.05s, i.e., 20 Hz). The step counter in the training loop
tracks elapsed ticks:

    wall_time = step * dt

This matches the physical timestep of the simulation and determines the
temporal resolution at which safety properties are checked. A property
violated between ticks is not caught — the simulation fidelity must be
sufficient that dt-spaced checks capture all physically realizable
violations. For the case studies in the paper (chemical mixing at thermal
time constants of ~seconds, drone attitude at control rates of 20-50 Hz),
dt=0.05s is conservative.

Policy Protocol
---------------
The policy object must implement three methods:

    policy.act(state: dict) -> action
        Select an action given the current state. During training this
        includes exploration (e.g., PPO's stochastic policy). The action
        format is opaque to this module — it's passed directly to the twin.

    policy.store(state, action, reward, next_state, done)
        Buffer a single transition for learning. Called once per timestep.

    policy.update()
        Perform a policy update using buffered transitions. Called once
        per episode. For PPO, this is where the clipped surrogate
        objective is optimized over the episode's trajectory.

    This interface is deliberately minimal. Any actor-critic implementation
    (PPO, A2C, SAC) can conform to it. The training loop does not assume
    anything about the policy's internals — network architecture, optimizer,
    advantage estimation, etc. are all encapsulated in the policy object.
"""

# Requirement kinds that are treated as safety properties.
# Violation (status=False) of any of these → -1 reward, episode terminates.
_SAFETY_KINDS = {"Prohibition", "Obligation", None}


def make_reward_fn(requirement_statuses):
    """
    Build a reward function from a requirement_statuses callable.

    Returns a closure:  state_dict -> (float, bool)

    The returned function calls requirement_statuses() to check safety
    properties, and inspects state["done"] for goal completion:

        -1.0, True   A safety requirement (Prohibition, Obligation, or
                      unclassified) has status=False. Episode should
                      terminate immediately.
        +1.0, True   state["done"] is True and no safety requirement
                      is violated. Goal reached.
         0.0, False  No violations, goal not yet reached. Continue.

    Parameters
    ----------
    requirement_statuses : callable
        Returns dict[str, {"kind": str|None, "status": bool}].
        Typically SimulationEngine.requirement_statuses.
    """
    def reward(state: dict) -> tuple[float, bool]:
        statuses = requirement_statuses()
        for entry in statuses.values():
            if entry["kind"] in _SAFETY_KINDS and not entry["status"]:
                return -1.0, True
        if state.get("done"):
            return 1.0, True
        return 0.0, False
    return reward


def train(twin, requirement_statuses, policy,
          n_episodes=1000, max_steps=200, dt=0.05):
    """
    Train a controller via sparse requirement-derived rewards on a digital twin.

    Parameters
    ----------
    twin : callable
        The digital twin / simulation environment.
        twin()       -> dict : reset, return initial state.
        twin(action) -> dict : step by dt seconds, return new state.
        State dict must include "done" (bool) when the goal is reached.

    requirement_statuses : callable
        Returns the current status of every requirement in the model.
        -> dict[str, {"kind": str|None, "status": bool}]
        Typically bound to SimulationEngine.requirement_statuses.

    policy : object
        RL policy with methods:
            .act(state: dict) -> action
            .store(state, action, reward, next_state, done)
            .update()

    n_episodes : int
        Number of training episodes. Default 1000.

    max_steps : int
        Maximum timesteps per episode before truncation. Default 200
        (= 10 seconds at dt=0.05).

    dt : float
        Simulation timestep in seconds. Each twin(action) call advances
        the clock by this amount. Default 0.05 (20 Hz).
    """
    reward_fn = make_reward_fn(requirement_statuses)

    for ep in range(n_episodes):
        state = twin()
        step = 0

        while step < max_steps:
            action = policy.act(state)
            next_state = twin(action)
            step += 1  # clock += dt

            reward, done = reward_fn(next_state)

            policy.store(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        policy.update()
