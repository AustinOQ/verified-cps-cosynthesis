# Fitting Artifact Manifest

This directory is a presentable, branch-friendly artifact for the
`architecture-fit` controller simplification work. It carries local copies of
the code and models needed by the demo. It does not modify central programs,
certification code, SysML models, or training experiments.

Read `README.md` first for the plain-language overview. In short, the
memoryless stage checks whether a finite buffer is enough for the current
controller decision. The Markov/MDP stage checks whether a finite buffer is
enough for a proof that the next modeled step is determined.

There are three simplified checks. The affine/rule stage is the only analytical
fit implemented directly by this artifact. The memoryless and Markov/MDP stages
produce buffer evidence that supports later feedforward or analytical fitting.

## Source Files

| file | role |
|---|---|
| `run_fitting_sequence.sh` | entry point using the project venv by default |
| `src/run_fitting_sequence.py` | orchestrates the three-stage artifact report |
| `src/mixing_rule_eval.py` | fresh artifact-local validation of the mixing NeuralRequirement rule |
| `src/generate_markov_mdp.py` | generates the Markov/MDP Z3 check from bundled SysML without saving certificates |
| `README.md` | quick usage and claim boundary |
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
| Python 3 | all stages | Tested with Python `3.13.7`. The artifact defaults to `/home/csned/git_stuff/AI_venv/bin/python`. Any compatible Python can be supplied with `PYTHON_BIN=/path/to/python`. |
| `numpy` | fresh mixing rule evaluation | Tested with `numpy 1.26.4`. Required by the bundled environment and oracle code in `bundle/rl/`. |
| `z3-solver` | fresh Markov/MDP proof check | Required by the default run. The artifact builds the Z3 query from the bundled SysML files at run time. |

Optional Python packages:

| dependency | needed when | notes |
|---|---|---|
| `pysysml2` | optional parser assist | The bundled parser has a regex fallback for these models. The default artifact does not require `pysysml2`. |

Not required for the default artifact run:

| dependency | why not required |
|---|---|
| `torch` | The artifact does not train neural networks. |
| `pandas` | The runner uses Python standard library CSV and JSON readers. |
| `scikit-learn` | No fitting routines from scikit-learn are used. |

Bundled code required by the default run:

| path | role |
|---|---|
| `bundle/architecture-fit/analytic_fit/` | Runs fresh thermostat and cruise affine/rule checks from SysML. |
| `bundle/architecture-fit/reconstruct_closure.py` | Finds finite buffers for the memoryless and Markov/MDP checks. |
| `bundle/architecture-fit/sysml_deps.py` | Extracts dependency information from SysML for the memoryless check. |
| `bundle/architecture-fit/certification/` | Builds and checks the Markov/MDP proof result in memory. |
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

Data policy:

The artifact does not bundle or save full Markov/MDP proof certificates. The
Markov/MDP stage builds the symbolic proof obligation from the bundled SysML
files. It calls Z3 during the run. It records only compact summaries of the
result.

The artifact also does not bundle old training or fit result files. The default
output is regenerated from the bundled code and SysML files.

## Generated Output

Running `bash run_fitting_sequence.sh` from this directory regenerates
`outputs/latest/`.

| output | role |
|---|---|
| `outputs/latest/fitting_sequence_report.md` | concise human-readable report |
| `outputs/latest/fitting_sequence_summary.json` | machine-readable report data |
| `outputs/latest/01_affine_rule/` | affine/rule stage summaries and mixing metrics |
| `outputs/latest/02_memoryless/` | memoryless controller summary |
| `outputs/latest/03_markov_mdp/` | provable Markov/MDP summary |
| `outputs/latest/logs/` | concise command summaries for fresh rerun stages |
