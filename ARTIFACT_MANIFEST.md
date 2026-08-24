# Fitting Artifact Manifest

This directory contains the code and models required by the `architecture-fit`
controller simplification artifact. A run reads from `bundle/` and writes
generated results to `outputs/latest/`.

## Source Files

| file | role |
|---|---|
| `run_fitting_sequence.sh` | portable entry point using `python3` or `PYTHON_BIN` |
| `make_overleaf_archive.sh` | creates a clean source archive for Overleaf |
| `requirements.txt` | exact Python dependency versions used to validate the artifact |
| `src/run_fitting_sequence.py` | orchestrates the artifact report and SysML-derived fitted training |
| `src/evaluate_sysml_rule.py` | reads and evaluates a Boolean action from the required SysML-encoded `#NeuralRequirement` |
| `src/generate_markov_mdp.py` | generates Markov/MDP certificates, specs, SMT-LIB queries, and Z3 proof/counterexample files from supplied SysML |
| `src/generate_discretization_safety.py` | generates and checks discretization safety certificates from the current SysML and Stage 3 result |
| `README.md` | setup, usage, input rules, and archive command |
| `ABOUT.md` | concise cruise-control example |

## Bundled Inputs

| path | contents |
|---|---|
| `bundle/architecture-fit/` | SysML discovery, buffer checker, certification package, and training code |
| `bundle/sysml-models/` | thermostat, discrete cruise control, continuous cruise control, and chemical mixing plant models plus parser and simulator support |
| `bundle/rl/` | shield, oracle, and environment code used by the checks |

## Dependencies

Runtime tools:

| dependency | required for | notes |
|---|---|---|
| Bash | `run_fitting_sequence.sh` | Used only to choose the Python executable and start the runner. |
| Python 3.12 or newer | all stages | The artifact defaults to `python3`. Another compatible interpreter can be supplied with `PYTHON_BIN`. |
| `numpy` | checks and discrete fitted training | Required by the bundled environment, oracle code, and NumPy trainer. |
| `z3-solver` | Markov/MDP proof check | Required by the default run. The artifact builds the Z3 query from the bundled SysML files at run time. |
| `torch` | continuous fitted training | Used by the bundled continuous MLP PPO trainer. It is forced to CPU by the artifact. |

Optional Python packages:

| dependency | needed when | notes |
|---|---|---|
| `pysysml2` | optional parser assist | The bundled parser also reads the SysML forms used by this artifact directly. |

Bundled code required by the default run:

| path | role |
|---|---|
| `bundle/architecture-fit/sysml_inputs.py` | Discovers and describes models from their current SysML contents. |
| `bundle/architecture-fit/reconstruct_closure.py` | Finds finite buffers for the memoryless and Markov/MDP checks. |
| `bundle/architecture-fit/sysml_deps.py` | Extracts dependency information from SysML for the memoryless check. |
| `bundle/architecture-fit/certification/` | Builds and checks the Markov/MDP proof result in memory. |
| `bundle/architecture-fit/discretization/` | Runs the linear, convex, exact symbolic, and SMT progression and independently checks the linear and convex proof certificates. |
| `bundle/architecture-fit/reduced_handmade/` | Trains fitted discrete feedforward policies with NumPy PPO. |
| `bundle/architecture-fit/train_mlp_buffer.py` | Trains a fitted real-valued-action MLP policy on CPU. |
| `bundle/architecture-fit/mlp_buffer.py` | Defines buffered continuous/discrete MLP environment wrappers and policy classes. |
| `bundle/handmade/` | NumPy neural network, optimizer, oracle, PPO, and checkpoint helpers. |
| `bundle/sysml-models/sysml_parser.py` | Parses the bundled SysML models. |
| `bundle/sysml-models/simulator.py` | Runs the bundled simulation engine. |
| `bundle/sysml-models/simulator_adapter.py` | Connects the simulator to the environment wrapper. |
| `bundle/sysml-models/runtime_settings.py` | Defines and validates the shared simulation time step. |
| `bundle/rl/env.py` | Environment wrapper whose dimensions, completion input, and scenario values are read from SysML. |
| `bundle/rl/continuous_env.py` | Real-valued-action environment wrapper. |
| `bundle/rl/oracle.py` | Extracts the controller interface and requirement oracle. |
| `bundle/rl/shield.py` | Boolean-action shield rebuilt from the current SysML requirement. |
| `bundle/rl/continuous_shield.py` | Real-valued shield extracted from the current SysML requirement. |

Bundled SysML models:

| model | path |
|---|---|
| Thermostat | `bundle/sysml-models/thermostat/model.sysml` |
| Discrete cruise controller | `bundle/sysml-models/cruise-controller-model/model.sysml` |
| Continuous cruise controller | `bundle/sysml-models/cruise-continuous-model/model.sysml` |
| Chemical mixing plant | `bundle/sysml-models/mixing-sysml-model/model.sysml` |

Generated data:

Each run overwrites `outputs/latest/` and rebuilds generated artifacts from the
SysML files supplied to that run. The Markov/MDP stage saves certificates,
reduced-MDP specs, SMT-LIB queries, and proof or counterexample transcripts.

## Generated Output

Running `bash run_fitting_sequence.sh` from this directory regenerates
`outputs/latest/`.

| output | role |
|---|---|
| `outputs/latest/fitting_sequence_report.md` | concise human-readable report |
| `outputs/latest/fitting_sequence_summary.json` | machine-readable report data |
| `outputs/latest/01_affine_rule/` | requirement extraction and evaluation summaries by model |
| `outputs/latest/02_memoryless/` | memoryless controller summary |
| `outputs/latest/03_markov_mdp/` | provable Markov/MDP certificates, specs, queries, proof files, and summaries |
| `outputs/latest/04_discretization_safety/` | discretization safety certificates and model summaries |
| `outputs/latest/05_reduced_training/` | SysML-derived fitted feedforward runs, selected-model table, summaries, logs, and weights |
| `outputs/latest/logs/` | concise stage command summaries |
