# N30 Experiment Outputs

These files preserve the 30-seed aggressive multicore run:

```bash
bash run_pipeline.sh --force-full --num-seeds 30 --cpu-mode aggressive --jobs 32
```

The normal pipeline output filenames were copied to matching
`*_n30_experiment.*` filenames so future pipeline runs can overwrite the
default outputs without replacing this experiment snapshot.

Primary files:

- `metrics/pipeline_report_n30_experiment.md`
- `metrics/cpu_training_summary_n30_experiment.csv`
- `metrics/runtime_results/runtime_monitor_summary_n30_experiment.csv`
- `metrics/evaluation_summary_n30_experiment.csv`
- `SMV/verification_results_n30_experiment.txt`
- `pipeline_30seed_aggressive_full_n30_experiment.log`
