# Fitting Artifact Manifest

This directory is a presentable, branch-friendly artifact for the
`architecture-fit` controller simplification work. It carries local copies of
the code and models needed by the demo. A run reads from `bundle/` and writes
generated results to `outputs/latest/`.

Read `README.md` first for the plain-language overview. In short, the
memoryless stage checks whether a finite buffer is enough for the current
controller decision. The Markov/MDP stage checks whether a finite buffer is
enough for a proof that the next modeled step is determined.

There are three simplified checks plus one training stage. The affine/rule
stage is analytical. The memoryless and Markov/MDP stages produce buffer
evidence. The final stage sweeps small feedforward controllers from the
Markov/MDP specs generated in the same run and selects the smallest good model.

## Source Files

| file | role |
|---|---|
| `run_fitting_sequence.sh` | portable entry point using `python3` or `PYTHON_BIN` |
| `src/run_fitting_sequence.py` | orchestrates the artifact report and full reduced training sweep |
| `src/mixing_rule_eval.py` | fresh artifact-local validation of the mixing NeuralRequirement rule |
| `src/generate_markov_mdp.py` | generates Markov/MDP certificates, specs, SMT-LIB queries, and Z3 proof/counterexample files from bundled SysML |
| `README.md` | quick usage and limitations |
| `ABOUT.md` | concise checker overview and example output rows |

## Bundled Inputs

| path | contents |
|---|---|
| `bundle/architecture-fit/` | buffer checker, certification package, and analytic rule sidecars |
| `bundle/sysml-models/` | thermostat, discrete cruise, continuous cruise, and mixing SysML models plus parser/simulator support |
| `bundle/rl/` | exact shield, continuous shield, oracle, and environment code used by the fresh checks |

## Dependencies

Runtime tools:

| dependency | required for | notes |
|---|---|---|
| Bash | `run_fitting_sequence.sh` | Used only to choose the Python executable and start the runner. |
| Python 3 | all stages | The artifact defaults to `python3`. Any compatible Python can be supplied with `PYTHON_BIN`. |
| `numpy` | fresh checks and handmade discrete training | Required by the bundled environment, oracle code, and handmade NumPy trainer. |
| `z3-solver` | fresh Markov/MDP proof check | Required by the default run. The artifact builds the Z3 query from the bundled SysML files at run time. |
| `torch` | continuous reduced training | Used by the bundled continuous MLP PPO trainer. It is forced to CPU by the artifact. |

Optional Python packages:

| dependency | needed when | notes |
|---|---|---|
| `pysysml2` | optional parser assist | The bundled parser has a regex fallback for these models. |

Bundled code required by the default run:

| path | role |
|---|---|
| `bundle/architecture-fit/analytic_fit/` | Runs fresh thermostat and cruise affine/rule checks from SysML. |
| `bundle/architecture-fit/reconstruct_closure.py` | Finds finite buffers for the memoryless and Markov/MDP checks. |
| `bundle/architecture-fit/sysml_deps.py` | Extracts dependency information from SysML for the memoryless check. |
| `bundle/architecture-fit/certification/` | Builds and checks the Markov/MDP proof result in memory. |
| `bundle/architecture-fit/reduced_handmade/` | Trains reduced discrete feedforward policies with handmade NumPy PPO. |
| `bundle/architecture-fit/train_mlp_buffer.py` | Trains the reduced continuous cruise MLP policy on CPU. |
| `bundle/architecture-fit/mlp_buffer.py` | Defines buffered continuous/discrete MLP environment wrappers and policy classes. |
| `bundle/handmade/` | Handmade NumPy neural network, optimizer, oracle, PPO, and checkpoint helpers. |
| `bundle/sysml-models/sysml_parser.py` | Parses the bundled SysML models. |
| `bundle/sysml-models/simulator.py` | Runs the bundled simulation engine. |
| `bundle/sysml-models/simulator_adapter.py` | Connects the simulator to the environment wrapper. |
| `bundle/rl/env.py` | Environment wrapper used by the fresh mixing rule evaluation. |
| `bundle/rl/continuous_env.py` | Continuous-action environment wrapper used by the continuous cruise support code. |
| `bundle/rl/oracle.py` | Extracts the controller interface and requirement oracle. |
| `bundle/rl/shield.py` | Exact program shield used by the fresh mixing rule evaluation. |
| `bundle/rl/continuous_shield.py` | Exact continuous projection shield used by the continuous cruise support code. |

Bundled SysML models:

| model | path |
|---|---|
| Thermostat | `bundle/sysml-models/thermostat/model.sysml` |
| Discrete cruise controller | `bundle/sysml-models/cruise-controller-model/model.sysml` |
| Continuous cruise controller | `bundle/sysml-models/cruise-continuous-model/model.sysml` |
| Mixing machine | `bundle/sysml-models/mixing-sysml-model/model.sysml` |

Generated data:

Each run overwrites `outputs/latest/` and rebuilds generated artifacts from the
bundled SysML files. The Markov/MDP stage saves fresh certificates,
reduced-MDP specs, SMT-LIB queries, and proof or counterexample transcripts.

## Generated Output

Running `bash run_fitting_sequence.sh` from this directory regenerates
`outputs/latest/`.

| output | role |
|---|---|
| `outputs/latest/fitting_sequence_report.md` | concise human-readable report |
| `outputs/latest/fitting_sequence_summary.json` | machine-readable report data |
| `outputs/latest/01_affine_rule/` | affine/rule stage summaries and mixing metrics |
| `outputs/latest/02_memoryless/` | memoryless controller summary |
| `outputs/latest/03_markov_mdp/` | provable Markov/MDP certificates, specs, queries, proof files, and summaries |
| `outputs/latest/04_reduced_training/` | full reduced feedforward sweep, selected-smallest-good table, summaries, logs, and weights |
| `outputs/latest/logs/` | concise command summaries for fresh rerun stages |
