# Controller Fitting Sequence Artifact

This is a compact, self-contained artifact for the `architecture-fit` work. It
is intended to be branch/upload friendly: one runner, one bundled code/model
tree, one output directory, and one human-readable report that demonstrates the
sequence of simplified controller fitting/training methods developed in this
subproject.

The artifact uses local copies under `bundle/`. It does **not** modify the
central certification or training programs.

## Run

From this directory:

```bash
bash run_fitting_sequence.sh
```

This regenerates `outputs/latest/`.

## What It Demonstrates

The artifact asks whether a SysML model can be controlled by a simple
non-recurrent controller instead of a larger recurrent learner.

The sequence is:

```text
1. Affine/rule fit from NeuralRequirements
2. Memoryless controller check
3. Provable Markov/MDP controller check
```

The checks are ordered from cheapest to strongest. The early stages try to avoid
training. The later checks add stronger evidence that a finite buffer is enough.

## Plain Language

- A SysML `NeuralRequirement` is the part of the model that says which
  controller outputs are safe or required for the current observation.
- The affine/rule stage asks if that requirement already gives a direct
  controller. For the cruise model this is just a few speed and distance
  inequalities.
- The memoryless stage asks if a finite history buffer is enough for the
  current controller decision. This is useful when the goal is to remove a GRU
  or other recurrent learner.
- The Markov/MDP stage asks a stronger question. It checks whether the buffer is
  enough to make the next step of the modeled process determined. This stage
  generates proof obligations from the bundled SysML files. It calls Z3 during
  the run. It does not save full proof certificates but can modified to do so.
- The exact shield is the program-based safety logic from the SysML
  requirement. It is not a learned shield.

## What Gets Fitted

| stage | what the checker does | fitting status in this artifact |
|---|---|---|
| Affine/rule fit | Checks whether the requirement directly gives a controller. | This is analytical. It has `0` learned parameters. Thermostat, cruise, and mixing are rerun fresh from the bundled SysML files. |
| Memoryless controller check | Checks whether a finite buffer is enough for the current decision. | This does not fit weights. It supports later feedforward fitting by removing recurrent state from the controller input. |
| Provable Markov/MDP check | Checks whether a finite buffer is enough to prove the modeled next step is determined. | This does not fit weights. It supports later feedforward, tabular, or analytical fitting on a certified augmented state. |

Important claim boundary:

- The memoryless stage supports a simple controller for the current decision.
  It does not prove the full process is Markov.
- The Markov/MDP stage is the stronger provable check.
- The exact program shield / exact continuous projection remains the safety
  authority throughout deployed/evaluated stages.

## Environment

The runner defaults to:

```text
/home/csned/git_stuff/AI_venv/bin/python
```

Override with:

```bash
PYTHON_BIN=/path/to/python bash run_fitting_sequence.sh
```

The demo still needs a Python environment with the normal project packages,
including `numpy` and `z3-solver`.

Advanced override:

```bash
ARCHITECTURE_FIT_ROOT=/path/to/architecture-fit
```

This override is usually not needed. The default uses `bundle/architecture-fit/`.
