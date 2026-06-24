# Runtime Verification Utilities

`verify.py` loads SysML models, parses `#NeuralRequirement` expressions into an
AST, and evaluates those requirements against concrete input/output values.
This is the runtime counterpart to the static nuXmv checks: nuXmv verifies the
SMV model over all modeled states, while this evaluator checks the concrete
controller decision at a simulation step.

The active artifact pipeline invokes runtime evaluation through
`run_pipeline.sh`, using the NumPy CPU checkpoints in `metrics/cpu_training/`.
Aggregated runtime results are written to
`metrics/runtime_results/runtime_monitor_summary.csv` and summarized in
`metrics/pipeline_report.md`.

## Components

`verify.py`:
Requirement AST parser and evaluator. It can also serve as a Flask endpoint for
manual testing.

`eval_runtime_monitor.py`:
Standalone Torch-checkpoint evaluator retained for comparison experiments. It
expects checkpoints under `rl/checkpoints/<system>/best.pt`; those are separate
from the CPU checkpoints used by the current top-level pipeline.
