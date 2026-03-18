# Training Guide

## Quick Start

From the repo root:

```bash
python rl/train.py sysml-models/thermostat/model.sysml
```

## Available Models

```bash
python rl/train.py sysml-models/thermostat/model.sysml
python rl/train.py sysml-models/cruise-controller-model/model.sysml
python rl/train.py sysml-models/mixing-sysml-model/model.sysml
```

If no argument is given, defaults to the mixing model.

## What It Does

1. **Phase 1** — Oracle behavioral cloning (supervised pretraining)
2. **Phase 2** — PPO fine-tuning with full specification shield
3. **Evaluation** — 100-episode greedy eval, passes if ≥95% success and 0 safety violations

Checkpoints saved to `rl/checkpoints/full/` (`oracle.pt`, `best.pt`, `final.pt`).

## Changing Training Parameters

All constants are at the top of `train.py` inside `MODE_CONFIG` and `run_training_mode`:

```python
# In MODE_CONFIG (top of file):
"oracle_samples": 5_000,    # number of oracle demonstration pairs
"oracle_epochs": 30,        # supervised training epochs
"ppo_episodes": 5_000,      # total PPO training episodes

# In run_training_mode (line ~410):
HIDDEN_DIM = 64             # GRU + MLP hidden size
ORACLE_BATCH_SIZE = 1024    # batch size for oracle training
ORACLE_LR = 1e-3            # oracle learning rate
EPISODES_PER_UPDATE = 80    # episodes collected per PPO update
EVAL_INTERVAL = 500         # greedy eval every N episodes
RL_LR = 3e-4                # PPO learning rate
ENTROPY_START = 0.01        # initial entropy coefficient
ENTROPY_END = 0.001         # final entropy coefficient (linear decay)
MAX_STEPS = 1200            # max timesteps per episode
DT = 0.1                    # simulation timestep (seconds)
SEED = 42                   # random seed
```

## Pass Criteria

```python
PASS_SUCCESS_RATE = 0.95    # ≥95% of 100 eval episodes reach goal
PASS_SAFETY_VIOLATIONS = 0  # zero requirement violations across all eval episodes
```

## Output

Training logs print to stdout. Key lines to watch:

```
[Oracle] epoch  1/30 | loss=0.3012 | acc=85.2%     # Phase 1 progress
[full/P2] ep   480/5000 | avg_r=+0.821 | success=90% # Phase 2 progress
Detailed eval: 97/100 success, 0 violations → PASS  # Final result
```
