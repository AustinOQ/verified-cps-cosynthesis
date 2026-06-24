# clarity: Verified CPS Co-Synthesis Pipeline

This artifact takes SysML v2 controller models through the full experiment
pipeline:

1. extract SMV models from SysML,
2. verify safety requirements with nuXmv IC3 and BMC,
3. train CPU NumPy recurrent controllers,
4. evaluate the selected checkpoints with the external specification shield,
5. write CSV metrics and a human-readable report.

The default run processes all three models (`thermostat`, `cruise`, `mixing`)
with one fixed seed, `42`.

## Running The Pipeline

```bash
bash run_pipeline.sh
```

Useful flags:

| flag | effect |
|---|---|
| `--force-full` | Regenerate requested outputs, including CPU training and runtime evaluation. |
| `--verify-only` | Skip training; extract SMVs, run nuXmv, and reuse available CPU checkpoints for runtime/report data. |
| `--model NAME` | Run one model: `thermostat`, `cruise`, or `mixing`. |
| `--num-seeds N` | Run `N` deterministic seeds starting at `42`. |
| `--seeds N` | Alias for `--num-seeds N`. |
| `--cpu-mode single` | Default. Cap numerical libraries to one thread and pin timed subprocesses to one CPU core when `taskset` is available. |
| `--cpu-mode aggressive` | Use normal multicore scheduling for faster local runs. |
| `--single-core` | Alias for `--cpu-mode single`. |
| `--aggressive-multicore` | Alias for `--cpu-mode aggressive`. |
| `--cpu-affinity-core N` | Core used by single-core mode. Defaults to `0`. |

Common commands:

```bash
# One seed per model, seed 42
bash run_pipeline.sh

# Rebuild everything for the default seed
bash run_pipeline.sh --force-full

# Reproduce the 30-seed statistical run
bash run_pipeline.sh --force-full --num-seeds 30

# Faster workstation run, not for reported timing statistics
bash run_pipeline.sh --force-full --num-seeds 30 --cpu-mode aggressive

# Re-run only extraction/formal verification for the mixing model
bash run_pipeline.sh --verify-only --model mixing
```

Environment overrides:

| variable | purpose |
|---|---|
| `PYTHON_BIN` | Python interpreter. Defaults to `~/git_stuff/AI_venv/bin/python` when present. |
| `CPU_TRAINING_REPO` | Path to the CPU training program repo. Defaults to `../shield-pipeline-new-sysml`. |
| `NUXMV_BIN` | Path to the nuXmv binary. Defaults to the included Linux binary or `nuXmv` on `PATH`. |
| `NUXMV_MEM_LIMIT_KB` | Memory cap for nuXmv. Defaults to 10 GB. |
| `CPU_EXECUTION_MODE` | `single` or `aggressive`. Defaults to `single`. |
| `CPU_AFFINITY_CORE` | Core used in single-core mode. Defaults to `0`. |

## Outputs

Key generated files:

| file | contents |
|---|---|
| `SMV/*/*.smv` | Generated nuXmv models. |
| `SMV/*/ic3_output.txt`, `SMV/*/bmc_output.txt` | Raw nuXmv output. |
| `SMV/verification_results.txt` | Combined IC3/BMC verification log. |
| `metrics/evaluation_summary.csv` | Per-model IC3/BMC proof counts, runtime, and peak RSS. |
| `metrics/cpu_training_summary.csv` | Per-model/per-seed training, selected checkpoint, safety, override, step, time, and RSS metrics. |
| `metrics/runtime_results/runtime_monitor_summary.csv` | Runtime success, safety violation, override, latency, and inference RSS metrics. |
| `metrics/pipeline_report.md` | Human-readable summary with means and 95% confidence intervals across seeds. |

The pipeline defaults to single-core execution for reported timing statistics.
In this mode, common numerical-library thread pools are capped at one thread and
timed subprocesses are pinned with `taskset` when available. Use
`--cpu-mode aggressive` only for faster local runs where timing comparability is
not the goal.

nuXmv settings are fixed in the pipeline: IC3 is unbounded, and BMC uses
`-bmc -bmc_length 200 -bmc_inc_invar_alg een-sorensson`. The simulator and
training/evaluation path use `dt = 0.1`.

## Hardware For Statistical Experiments

The paper's statistical experiments were run on a dedicated CPU workstation:

| component | value |
|---|---|
| CPU | AMD Ryzen 9 9950X 16-Core Processor |
| CPU threads | 32 logical CPUs, 16 physical cores |
| Memory | 123 GiB RAM, 16 GiB swap |
| OS | Ubuntu/Kubuntu Linux, kernel `6.17.0-35-generic`, x86_64 |
| Training mode | CPU-only NumPy training; no GPU is required for the reported CPU runs. |

Peak RSS for training, verification, and runtime evaluation is recorded in the
generated metrics and summarized in `metrics/pipeline_report.md`.

## Requirements

- Python 3.10+ with `numpy` and `antlr4-python3-runtime==4.13.2`
- nuXmv 2.1.0 or compatible; a Linux x86_64 binary is included
- The CPU training program repo next to this artifact, or `CPU_TRAINING_REPO`
  set explicitly

Install Python dependencies with:

```bash
pip install -r requirements.txt
```
