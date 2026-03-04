"""
train.py — RL training loop with LTL-derived sparse rewards for digital twins.

Architecture Context
--------------------
This module sits between two stages of the pipeline:

    UPSTREAM:   SysML v2 extractor produces LTL formulas + a digital twin.
    THIS FILE:  Trains a neural controller using the twin, rewarding only
                via LTL violation detection (sparse rewards).
    DOWNSTREAM: The trained controller is verified in two phases:
                  1. β-CROWN checks every non-temporal property on the network
                     directly (per-input-point verification).
                  2. nuXmv model-checks the full temporal LTL specification
                     over a finite transition system whose valid transitions
                     are constrained by the non-temporal properties.

The non-temporal properties (always / never) serve as the bridge between
training and verification:

    During training (here):
        They are per-step violation checks. Any violation → -1 reward,
        episode terminates. The agent learns to avoid these states entirely.

    During verification (downstream):
        The same properties define which transitions are POSSIBLE in the
        nuXmv abstraction. β-CROWN proves that the trained network satisfies
        each non-temporal property for all valid inputs. nuXmv then only
        needs to explore transition sequences within that verified envelope.
        This is what keeps temporal model checking tractable — without the
        non-temporal constraints pruning the state space, nuXmv would face
        an exponential blowup in reachable states.

    The critical invariant is that the SAME expressions used for reward here
    are used for transition constraints in nuXmv. There is one source of
    truth: the LTL extracted from the SysML v2 specification.

Why Sparse Rewards
------------------
The reward signal is intentionally sparse: -1 / 0 / +1, nothing else.

    -1  A non-temporal safety property was violated this timestep.
         Episode terminates immediately — no recovery, no partial credit.

     0  No violation, goal not yet reached. The agent gets no guidance about
         whether it's "close" to the goal or "far" from a violation.

    +1  The digital twin signals task completion (done=True) and no property
         was violated on this final step.

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
        property was violated. This matches the semantics of the LTL spec,
        where □(level ≥ 2) is either satisfied or not — there is no notion
        of "level = 1.9 is less bad than level = 0.1." A violation is a
        violation. This also simplifies the reward function to a pure boolean
        check per property, which is exactly what β-CROWN will verify downstream.

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

State dict contract:

    The twin returns a plain Python dict. Keys are strings that MUST match
    the variable names used in the LTL specification. This is the critical
    design decision that eliminates the mapping layer:

        SysML v2 spec says:  "pump_on", "valve_open", "level"
        LTL extractor uses:   pump_on, valve_open, level
        Twin returns:         {"pump_on": 1, "valve_open": 1, "level": 5.2}
        Violation expression: "(pump_on) and not (valve_open)"
        eval() namespace:     the twin's state dict directly

    No var_map, no index translation, no observation vector flattening.
    The variable names flow from the SysML v2 specification through the
    LTL extractor, into the twin's state dict, and directly into eval().
    One vocabulary, end to end.

    The special key "done" (bool) signals that the task's goal condition
    has been achieved. This corresponds to the liveness properties in the
    LTL spec (e.g., ◇(|transferred - target| ≤ 2L)). The twin is
    responsible for checking goal conditions because they require domain
    logic (e.g., "close enough to target volume") that the generic reward
    function cannot infer from the non-temporal properties alone.

    The liveness properties themselves are verified by nuXmv over the
    transition system, not by this training loop. The +1 reward for "done"
    simply provides the learning signal that drives the agent toward
    goal-reaching behavior. Verification that the goal is ALWAYS eventually
    reached (under the verified transition constraints) is nuXmv's job.

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

from ltl_to_constraints import validate_violations


# Restricted eval namespace: no builtins except safe math functions.
# The state dict is passed as locals, so its keys become the only accessible
# variables. This prevents the violation expressions from doing anything
# beyond arithmetic and comparisons on state values.
_EVAL_NS = {"__builtins__": {}, "abs": abs, "max": max, "min": min}


def make_reward_fn(violations: list[str]):
    """
    Build a reward function that checks non-temporal LTL violations.

    Returns a closure:  state_dict -> float

    The returned function evaluates each violation expression against the
    state dict. Because the twin's state dict keys match the LTL variable
    names, the dict IS the eval namespace — no translation needed.

    Return values:
        -1.0  First violation expression that evaluates to True. Short-circuits:
              remaining expressions are not checked. The episode should
              terminate immediately (the state is unsafe).
        +1.0  No violations AND state["done"] is True. The task goal was
              reached safely. This is the only positive reward the agent
              ever receives.
         0.0  No violations, goal not yet reached. Continue.

    Parameters
    ----------
    violations : list[str]
        Violation expressions from ltl_to_violations(). Each must eval to
        True when the corresponding safety property is violated.
    """
    def reward(state: dict) -> float:
        for v in violations:
            if eval(v, _EVAL_NS, state):
                return -1.0
        if state.get("done"):
            return 1.0
        return 0.0
    return reward


def train(twin, violations, policy,
          n_episodes=1000, max_steps=200, dt=0.05):
    """
    Train a controller via sparse LTL rewards on a plug-and-play digital twin.

    Parameters
    ----------
    twin : callable
        The digital twin / simulation environment.
        twin()       -> dict : reset, return initial state.
        twin(action) -> dict : step by dt seconds, return new state.
        State dict keys must match the variable names in the LTL spec.
        Set "done": True in the state dict when the goal is reached.

    violations : list[str]
        Violation expressions from ltl_to_violations(). These are the
        non-temporal (always/never) properties converted to per-step checks.
        The same expressions will later constrain the nuXmv transition
        system for temporal verification.

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
    reward_fn = make_reward_fn(violations)

    # Validate violation expressions against the twin's state dict keys
    # before training. Catches variable name mismatches (e.g., LTL says
    # "pump_on" but twin returns "pump_active") immediately rather than
    # mid-training. This is the one-vocabulary invariant being enforced.
    init = twin()
    validate_violations(violations, [k for k in init if k != "done"])

    for ep in range(n_episodes):
        state = twin()
        step = 0

        while step < max_steps:
            action = policy.act(state)
            next_state = twin(action)
            step += 1  # clock += dt

            reward = reward_fn(next_state)
            done = next_state.get("done", False) or reward < 0

            policy.store(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        policy.update()
