# RL_Phase — Automated RL Controller Synthesis from SysML v2

Plug-and-play reinforcement learning pipeline that takes any SysML v2 model
with a `#Neural` action def and produces a trained neural controller using
PPO (stable-baselines3). Everything — observations, actions, rewards, done
conditions — is extracted automatically from the SysML specification.

This module is part of the **Verified CPS Co-Synthesis** pipeline. It sits
between the SysML model/simulator (upstream) and formal verification
(downstream).

## Quick Start

```bash
# Install dependencies
pip install stable-baselines3 gymnasium numpy

# Train a controller (from repo root):
python -m RL_Phase sysml-models/mixing-sysml-model/model.sysml --shaping

# Evaluate the trained model:
python -m RL_Phase.eval sysml-models/mixing-sysml-model/model.sysml \
    -m rl_output/model_best --randomize --max-steps 1000
```

## What SysML Models Must Provide

For this pipeline to work, the SysML v2 model needs three things:

### 1. A `#Neural` Action Def

Exactly one action def annotated with `#Neural`. Its `in` parameters become
**observations** and `out` parameters become **actions**:

```sysml
#Neural
action def NeuralPolicy {
    in sensorReading1 : Real;     // observation
    in sensorReading2 : Real;     // observation
    in targetValue : Real;        // observation
    in done : Boolean;            // goal-completion signal (special)
    out controlSignal1 : Boolean; // action
    out controlSignal2 : Boolean; // action
}
```

- All `in` parameters (except `done`) become the observation vector.
- All `out` parameters become the action vector (currently Boolean only).
- A `done` input (case-insensitive, Boolean type) is excluded from observations
  and instead used for goal detection.

### 2. A Done Binding Expression

The `#Neural` action must be invoked via a `SubactionCallStmt` with
`InputBindingStmt`s that bind each observation to a SysML expression.
The `done` binding defines the **goal condition** — when it evaluates to
True, the episode succeeds:

```sysml
perform step : NeuralPolicy {
    in sensorReading1 = volumeSensor1.response;
    in sensorReading2 = volumeSensor2.response;
    in targetValue = controller.targetAmount;
    in done = (sensorReading1 >= targetValue) and (sensorReading2 >= targetValue);
    out controlSignal1;
    out controlSignal2;
}
```

The done expression is converted to Python and used for:
- **Boolean goal checking**: episode terminates with positive reward when True
- **Continuous distance computation**: for reward shaping, the expression
  AST is parsed to compute a differentiable distance-to-goal metric

### 3. Requirements with Metadata Annotations

Safety properties become the reward signal. Requirements annotated with
`#Prohibition` trigger penalties when violated:

```sysml
#Prohibition
requirement def 'No Dry Running' {
    doc /* Pump must not run when tank is empty */
    ...
}

#Obligation
requirement def 'Must Complete Transfer' {
    doc /* All transfers must eventually complete */
    ...
}
```

Reward protocol:
- **`#Prohibition` violated** -> -1 reward, episode terminates immediately
- **Goal reached** (`done=True`) -> +1 reward (or +10 with shaping)
- **`#Obligation` violated** -> no penalty (agent learns to satisfy these
  through the goal reward, not through punishment)
- **Normal step** -> 0 (sparse) or distance improvement (shaped)

### 4. Physical Dynamics

The model must have dynamics (state machines, flows, `action step` blocks)
that the `SimulationEngine` can evaluate. The `SimulatorTwin` wraps the
engine as a callable: `twin()` resets, `twin(action_dict)` steps.

## Architecture

```
SysML Model (.sysml)
    |
    v
+-------------+     +--------------+
|  Extractor   |---->|  Neural      |
| (extractor)  |     |  Interface   |
+-------------+     +------+-------+
                           |
    +----------------------+--------------------+
    |                      |                    |
    v                      v                    v
+---------+        +-------------+      +-----------+
|  SysML   |        |   SB3 PPO   |      | Reward    |
|   Env    |<------>|   MlpPolicy |      | (sparse/  |
|  (env)   |        |             |      |  shaped)  |
+---------+        +-------------+      +-----------+
    |                      |
    |  SimulatorTwin       |
    |  (threading)         |
    v                      v
+---------+        +-------------+
| model   |        | model_best  |
| .zip    |        | .zip        |
+---------+        +-------------+
```

## Modules

| File | Role | Generality |
|------|------|------------|
| `extractor.py` | Parse SysML, extract #Neural interface | Fully general |
| `env.py` | Gymnasium env wrapping SimulatorTwin | Core: general. Randomization: model-specific |
| `train_rl.py` | PPO training loop with best-model checkpointing | Fully general |
| `eval.py` | Deterministic evaluation with outcome breakdown | Fully general |

## Training Modes

### Sparse Rewards (default)

```bash
python -m RL_Phase model.sysml
```

Reward: +1 goal, -1 violation, 0 otherwise. Requires high entropy (0.15)
for exploration. Works but converges slowly and can suffer policy collapse.

### Distance-Based Reward Shaping (recommended)

```bash
python -m RL_Phase model.sysml --shaping
```

The done expression is parsed into an AST to compute continuous distance:

- `(a >= b)` contributes `max(0, b-a)^2` (0 when satisfied)
- `(a and b)` sums sub-distances
- `(a or b)` takes minimum sub-distance

Per-step reward = `(last_distance - current_distance) / initial_distance`,
so total shaping reward over a successful episode is ~1.0. Goal bonus is +10
to keep the terminal reward dominant.

This works for any done expression using `>=`, `<=`, `==`, `and`, `or` —
no model-specific code needed.

### Domain Randomization

```bash
python -m RL_Phase model.sysml --shaping --randomize --max-steps 1000
```

Randomizes start states each episode for robust generalization. The
randomization heuristics in `_randomize_start()` use naming conventions
("target", "original", "capacity") to classify observations — these may
need adaptation for new model families.

## CLI Reference

### Training

```bash
python -m RL_Phase <model_path> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `model_path` | (required) | Path to `.sysml` model file |
| `--output-dir`, `-o` | `./rl_output` | Output directory for models |
| `--timesteps`, `-n` | 200000 | Total training timesteps |
| `--dt` | 0.1 | Simulation timestep (seconds) |
| `--max-steps` | 200 | Max steps per episode |
| `--randomize` | off | Randomize start states |
| `--done-threshold` | 0.0 | Tolerance for done condition |
| `--shaping` | off | Distance-based reward shaping |

### Evaluation

```bash
python -m RL_Phase.eval <model_path> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `model_path` | (required) | Path to `.sysml` model file |
| `--weights`, `-m` | `./rl_output/model` | Path to saved model |
| `--episodes`, `-e` | 100 | Number of eval episodes |
| `--dt` | 0.1 | Simulation timestep |
| `--max-steps` | 200 | Max steps per episode |
| `--randomize` | off | Randomize start states |
| `--done-threshold` | 0.0 | Done threshold |

## Output Files

| File | Description |
|------|-------------|
| `model.zip` | Final trained PPO model (SB3 format) |
| `model_best.zip` | Best model checkpoint (highest rolling mean reward) |
| `interface.json` | Extracted Neural interface (obs/action names, types, done_expr) |

## Programmatic Usage

```python
from RL_Phase import extract_neural_interface, train

# Extract interface (useful for inspection):
interface = extract_neural_interface("path/to/model.sysml")
print(f"Observations: {interface.obs_names}")
print(f"Actions: {interface.action_names}")
print(f"Done condition: {interface.done_expr}")

# Train:
result = train("path/to/model.sysml", output_dir="./results",
               total_timesteps=300_000, shaping=True, randomize=True,
               max_steps=1000)
```

## Generality Notes

### What is fully general (works for any SysML model)

- Interface extraction from #Neural action def
- Gymnasium env construction (obs/action spaces from interface)
- Reward from `requirement_statuses()` (#Prohibition -> -1, terminate)
- Done checking from extracted done_expr
- PPO training with SB3
- Distance-based reward shaping from done_expr AST

### What uses model-specific heuristics

These methods in `env.py` use naming conventions from fluid-transfer models
and may need adaptation for other domains:

- **`_discover_mapping()`**: Maps observation names to simulator state keys
  by looking for patterns like `currentLevelMl`, `capacityMl`, `feeder`,
  `volumeSensor`. Only needed for start-state randomization.

- **`_randomize_start()`**: Classifies observations as volumes, targets,
  or originals by name ("target", "transfer", "original") and generates
  valid random states. Uses digit matching to pair related observations.

- **`_check_done()` threshold**: Applies `done_threshold` only to
  observations with "target" or "transfer" in their name.

For new model families, override these methods in a subclass or extend
the heuristics. The core pipeline works without randomization.

## Design Decisions

- **SB3 PPO with MlpPolicy**: Standard, well-tested RL algorithm. MLP is
  sufficient for the observation spaces in typical CPS models.
- **Distance shaping from done_expr**: Provides dense reward signal without
  any model-specific reward engineering. The done expression IS the spec.
- **target_kl=0.015**: Prevents catastrophic policy updates that cause
  training collapse with sparse/shaped rewards.
- **Best-model checkpointing**: Saves model at peak performance, not at
  end of training (which may be degraded due to entropy decay).
- **Only Prohibition penalties**: Obligations are learned through goal
  reward, not punishment. Penalizing obligations leads to overly
  conservative policies.
- **MultiBinary action space**: All Boolean actions with independent
  Bernoulli distributions. Scales to any number of actuators.

## Troubleshooting

### "Expected exactly one #Neural action def, found 0"
The SysML model must contain exactly one action def annotated with `#Neural`.

### Training gets stuck at negative reward
The agent is violating prohibitions every episode. Try:
- `--shaping` to give gradient signal toward the goal
- Increase `--timesteps` for more exploration time

### 100% timeout, 0% goal
The agent isn't reaching the goal within max_steps. Try:
- `--max-steps 1000` or higher
- `--shaping` for dense reward signal
- `--done-threshold 5.0` to relax the goal condition slightly

### Policy collapse (reward drops suddenly during training)
The `target_kl=0.015` should prevent this. If it persists, try reducing
`learning_rate` to `1e-4`.
