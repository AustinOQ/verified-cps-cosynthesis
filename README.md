# Verified CPS Co-Synthesis Pipeline

Automated pipeline for synthesizing formally verified neural controllers from SysML v2 models.
Given a SysML v2 system model with requirements, the pipeline:

1. **Extracts** an SMV formal model and Python simulation twin
2. **Verifies** safety properties via nuXmv (IC3 / BMC)
3. **Trains** a recurrent neural controller (behavioral cloning + PPO)
4. **Evaluates** the controller with a runtime safety monitor

## Quick Start

```bash
# Full pipeline (all 3 models, seed=42)
bash run_pipeline.sh

# Force re-run even if checkpoints exist
bash run_pipeline.sh --force-full

# Skip training (verification + metrics only)
bash run_pipeline.sh --skip-training
```

### Prerequisites

- Python 3.10+ with `torch`, `numpy`, `matplotlib`
- nuXmv (on PATH)
- ANTLR4 runtime (`antlr4-python3-runtime==4.13.2`)

## Directory Structure

```
clean_pipeline/
├── sysml-models/           # SysML v2 source models
│   ├── thermostat/model.sysml
│   ├── cruise-controller-model/model.sysml
│   ├── mixing-sysml-model/model.sysml
│   └── mc-extract.py       # SysML → SMV/Python extractor
├── SMV/                    # Generated nuXmv models + verification results
│   ├── thermostat/model.smv
│   ├── cruise-control-model/out.smv
│   ├── mixing-model/model.smv
│   └── verification_results.txt
├── rl/                     # RL training code + checkpoints
│   ├── train.py            # Oracle + PPO training loop
│   ├── model.py            # RecurrentActorCritic network
│   ├── env.py              # SysML simulation environment
│   ├── oracle.py           # Brute-force oracle for behavioral cloning
│   └── checkpoints/        # Trained models (best.pt, final.pt per system)
├── runtime-verification/   # Runtime safety monitor
│   ├── verify.py           # Requirement AST parser + evaluator
│   └── eval_runtime_monitor.py  # Monitor evaluation script
├── metrics/                # All logged metrics (CSV)
│   ├── *_training_metrics.csv   # Per-model training logs
│   ├── evaluation_summary.csv   # Verification results table
│   ├── automation_metrics.csv   # LOC automation ratios
│   └── runtime_results/         # Runtime monitor evaluation
├── results/                # Paper figures and analysis
│   ├── code/generate_figures.py
│   └── graphs_tables/      # Generated PNG figures
└── run_pipeline.sh         # Main entry point
```

## Metrics and Logging

### Training Metrics (`metrics/<model>_training_metrics.csv`)

Logged per-model during training. Columns:

| Column | Description |
|--------|-------------|
| `timestamp` | ISO 8601 wall-clock time |
| `phase` | `oracle`, `ppo`, `eval`, `final_eval`, `best_eval` |
| `epoch_or_episode` | Oracle epoch or PPO episode number |
| `loss`, `accuracy` | Oracle phase only |
| `avg_reward`, `avg_steps` | PPO phase: batch averages |
| `success_rate`, `violation_rate` | Fraction of episodes succeeding / violating |
| `entropy`, `policy_loss`, `value_loss` | PPO diagnostics |
| `reward_mean_last100`, `reward_std_last100` | Rolling window stats |
| `elapsed_seconds`, `memory_mb` | Resource tracking |

**Phases:**
- `oracle` — behavioral cloning from brute-force oracle (~30 epochs)
- `ppo` — reinforcement learning updates (logged every 80 episodes, ~250 rows)
- `eval` — periodic checkpoint evaluation (every 500 episodes, 20-episode eval)
- `final_eval` — last model evaluated over 20 episodes
- `best_eval` — best checkpoint re-evaluated over 20 episodes

### Verification Summary (`metrics/evaluation_summary.csv`)

One row per system + TOTAL row. Tracks requirements verified/failed, verification method, strengthening invariant counts, and heuristics applied.

### Automation Metrics (`metrics/automation_metrics.csv`)

LOC comparison: SysML input vs auto-extracted SMV vs shared reward/env code. Shows extraction ratio (~37-41% SMV/SysML).

### Runtime Monitor (`metrics/runtime_results/runtime_monitor_summary.csv`)

Per-system runtime monitor evaluation: steps, actual violations, detected violations, false positives/negatives, detection rate, check latency (avg/median/p99/max in microseconds), memory usage.

## Generating Figures

```bash
cd results/code
python generate_figures.py              # default 200 DPI
python generate_figures.py --dpi 300    # publication quality
python generate_figures.py --style ggplot  # alternative style
```

Produces 7 figures in `results/graphs_tables/`:

| Figure | Description |
|--------|-------------|
| `fig1_training_curves.png` | PPO reward + success rate over episodes |
| `fig2_oracle_training.png` | Behavioral cloning loss + accuracy |
| `fig3_verification_table.png` | Formal verification results table |
| `fig4_automation_metrics.png` | SysML vs SMV vs reward LOC |
| `fig5_runtime_monitor.png` | Runtime monitor detection + latency |
| `fig6_memory_usage.png` | Training RSS memory over time |
| `fig7_final_rewards.png` | Final vs best model reward comparison |

## Runtime Monitor

The runtime monitor evaluates the `#NeuralRequirement` from each SysML model at every simulation step. It supports a configurable tolerance margin:

```bash
cd runtime-verification
python eval_runtime_monitor.py                  # strict (margin=0)
python eval_runtime_monitor.py --margin 2.0     # tolerance band
python eval_runtime_monitor.py --episodes 50    # fewer episodes
```

The `--margin` flag adds a dead zone around comparison boundaries. When a numeric condition is within `margin` of its threshold, the monitor does not flag it. This trades false positives for potential false negatives — tunable per deployment.

## Results Summary (seed=42)

| System | Success Rate | Reward (mean +/- std) | Requirements | Verified |
|--------|-------------|----------------------|-------------|----------|
| Thermostat | 100% | 0.895 +/- 0.091 | 5 | 5/5 (IC3) |
| Cruise Control | 100% | 0.968 +/- 0.016 | 7 | 7/7 (IC3) |
| Chemical Mixing | 55% | -0.341 +/- 0.999 | 2 | 0/2 (BMC) |

Mixing model verification: 6/8 strengthening invariants proven (IC3), 2 pump ordering invariants fail due to scan-phase timing. 2 requirements produce BMC counterexamples at step 16 (mid-scan-cycle overshoot).
