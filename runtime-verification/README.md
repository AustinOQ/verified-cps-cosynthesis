# Runtime Verification Server

Checks whether a neural controller's decisions satisfy the formal `#NeuralRequirement` from a SysML model. Given concrete policy inputs and outputs, the server evaluates the requirement constraint and returns whether it holds.

This is the runtime counterpart to the static verification done by nuXmv: where nuXmv proves the requirement holds for *all* possible inputs, this server checks it for a *specific* input/output pair. Use it during training or inference to verify individual policy decisions on the fly.

## Running the server

```
python runtime-verification/verify.py --port 8080 --model-dir sysml-models/
```

On startup the server scans `--model-dir` for subdirectories containing a `model.sysml` file and loads each model that has a `#Neural` action def and a `#NeuralRequirement`. It prints what it loaded:

```
Loaded model: cruise-controller-model  (in=['currentSpeedMps', 'targetSpeed', 'gapMeters', 'done'], out=['applyThrottle', 'applyBrake'], unchanging={'toleranceMps': 1.0, 'safeFollowingDistanceMeters': 10.0})
```

## Manual testing

Send a POST to `/models/<model-name>` with a JSON body containing `in` and `out` keys:

```bash
# Should return {"satisfied": true} — throttle when below target with safe gap
curl -X POST http://localhost:8080/models/cruise-controller-model \
  -H 'Content-Type: application/json' \
  -d '{
    "in":  {"currentSpeedMps": 15.0, "targetSpeed": 25.0, "gapMeters": 50.0, "done": false},
    "out": {"applyThrottle": true, "applyBrake": false}
  }'

# Should return {"satisfied": false} — both throttle and brake at once
curl -X POST http://localhost:8080/models/cruise-controller-model \
  -H 'Content-Type: application/json' \
  -d '{
    "in":  {"currentSpeedMps": 15.0, "targetSpeed": 25.0, "gapMeters": 50.0, "done": false},
    "out": {"applyThrottle": true, "applyBrake": true}
  }'
```

The `in` keys must match the `in` parameters of the model's `#Neural` action def, and the `out` keys must match its `out` parameters. Missing parameters return a 400 with details on what's missing.

## Design

The server has three components:

**Model loading** (`load_model`). For each model, the SysML parser finds the controller part, locates its `#Neural` action def (which declares the policy's input/output signature), and finds the `#NeuralRequirement` (which constrains the relationship between inputs and outputs). The requirement's raw expression text is parsed into an AST using `ExpressionParser`. Controller attributes that aren't neural parameters (e.g. `toleranceMps`, `safeFollowingDistanceMeters`) are collected as unchanging quantities — these are constants baked into the model that the requirement references.

**AST evaluator** (`evaluate`). A recursive function that walks the parsed expression tree. References prefixed with the requirement's subject variable (e.g. `p.targetSpeed`) are stripped to bare names and looked up in a flat values dictionary. The dictionary is the union of the request's `in`/`out` values and the model's unchanging quantities. Supports arithmetic (`+`, `-`, `*`, `/`), comparison (`==`, `>`, `<`, `>=`, `<=`), logical (`and`, `or`, `not`, `implies`), and ternary operators.

**Flask route** (`POST /models/<model_name>`). Validates the request, merges values, evaluates the AST, and returns `{"satisfied": true|false}`.
