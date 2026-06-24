# Controller Environment and Torch Reference Path

The active artifact pipeline trains controllers with the NumPy CPU training
program invoked by `run_pipeline.sh`. That path writes per-seed checkpoints and
summaries under `metrics/cpu_training/`.

This directory contains shared controller utilities plus a Torch reference
training path retained for comparison experiments:

```
rl/
├── env.py              # Gym-style wrapper over the SysML SimulatorTwin
├── oracle.py           # Brute-force oracle for behavioral cloning
├── shield.py           # Specification-derived safety shield from SysML AST
├── model.py            # GRU actor-critic network used by the Torch path
├── ppo.py              # Recurrent PPO implementation used by the Torch path
├── composite_model.py  # Torch policy + external specification shield wrapper
└── train.py            # Torch reference training entry point
```

## Shared Pieces

`env.py`, `oracle.py`, and `shield.py` are used as reusable control, simulation,
and specification utilities. `shield.py` evaluates the `#NeuralRequirement`
constraint AST at runtime and chooses a specification-compliant override when
the proposed action is unsafe.

## Torch Reference Training

`train.py` runs oracle behavioral cloning followed by PPO using the full
`#NeuralRequirement` shield. It saves Torch checkpoints under
`rl/checkpoints/<system>/` when run directly. The top-level pipeline does not
use these checkpoints for the current reported CPU training or runtime results.
