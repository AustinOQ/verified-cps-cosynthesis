# RL Controller Synthesis Pipeline

## Directory Structure

```
rl/
├── train.py            # Training entry point (oracle + PPO, full shield)
├── model.py            # GRU actor-critic network
├── ppo.py              # Recurrent PPO with GAE and episode-level batching
├── env.py              # Gym-style wrapper over SysML SimulatorTwin
├── oracle.py           # Brute-force oracle for behavioral cloning
├── shield.py           # Specification-derived safety shields from SysML AST
├── composite_model.py  # Policy + frozen shield composite
└── checkpoints/        # Saved model weights per system
```

## Architecture

### Network (`model.py`)

`RecurrentActorCritic`: MLP encoder → GRU → dual heads.

```
obs (obs_dim) → Linear(obs_dim, 64) → Tanh → Linear(64, 64) → Tanh
             → GRU(64, 64, 1 layer) → policy_head Linear(64, n_actions)
                                     → value_head  Linear(64, 1)
```

- Encoder: 2-layer MLP with Tanh activations, hidden_dim=64.
- Temporal backbone: single-layer GRU, hidden_dim=64.
- Policy head: logits over `2^N` discrete actions (N = number of boolean outputs).
- Value head: scalar state-value estimate.
- `forward(obs, hidden)`: single-timestep inference, returns `Categorical` distribution, value, updated hidden.
- `forward_sequence(obs_seq, hidden, mask)`: batched sequence inference with `pack_padded_sequence` for variable-length episodes.

### Environment (`env.py`)

`SysMLEnv`: wraps `SimulatorTwin` as a standard `reset()`/`step()` RL environment.

- **Observation space**: extracted from `#Neural` action def `in` parameters. Normalized by global scale (max absolute value from initial observation).
- **Action space**: `2^N` discrete actions for N boolean `out` parameters. Each action ID is a bitmask over the output parameter list.
- **Scenario randomization**: `#ScenarioInput` parameters are sampled uniformly within bounds extracted from `#ScenarioConstraint` expressions at each `reset()`.
- **Reward** (Phase 2):
  - `-1.0` + terminal: any `Prohibition`/`Obligation` requirement has `status=False`.
  - `+1.0 - 1/max_steps` + terminal: `done=True` in state dict (goal reached).
  - `-1/max_steps`: otherwise (living cost).
  - `0.0` + terminal: truncation at `max_steps=1200`.
- **Phase 1** (oracle data collection): safety violations ignored, only goal completion and living cost active.

### Oracle (`oracle.py`)

Brute-force optimal action selection for behavioral cloning.

**`SyncEngine`**: maintains a second `SimulationEngine` instance for counterfactual evaluation. For each candidate action:
1. Fully re-initializes engine (clean BindRefs, state machines, mailboxes, constraint solver).
2. Copies all non-BindRef physical state values from the live engine.
3. Sets `engine.model` to return the candidate action and capture inputs.
4. Steps forward `propagationDelay` times (attribute defined on the SysML system part).
5. Checks `requirement_statuses()` for violations.

**`brute_force_oracle`**: enumerates all `2^N` action combinations. Scores each by `goal_distance_before - goal_distance_after`. Actions causing requirement violations are excluded. Returns the highest-scoring action as a boolean array.

**`extract_interface`**: parses the SysML model to extract:
- `obs_names`: `#Neural` action `in` parameters (excluding `done`).
- `action_names`: `#Neural` action `out` parameters.
- `is_done(obs_dict)`: evaluates the `done` binding expression from the `SubactionCallStmt`.
- `goal_distance(obs_dict)`: L2 distance derived from `done` expression conjuncts (each comparison term contributes a squared gap).
- `propagation_delay`: integer from `attribute propagationDelay` on the system part def.

### Shield (`shield.py`)

Specification-derived action filtering from the `#NeuralRequirement` constraint AST.

**`_extract_base(model_path)`** returns:
- `req_ast`: parsed AST of the `#NeuralRequirement` constraint expression.
- `subject_var`: the requirement's subject variable name (e.g., `"p"`).
- `in_params`, `out_params`: neural action parameter names.
- `unchanging`: controller attributes that are not neural I/O (e.g., `toleranceCelcius`, `safeFollowingDistanceMeters`). Looked up by prefix-matching against the controller's qualified name.
- `prohibition_clauses`: conjuncts of `req_ast` whose references are entirely within `out_params ∪ unchanging` (i.e., evaluable without observations).
- `dead_actions`: action IDs where any prohibition clause evaluates to `False` under `unchanging ∪ actuator_values`.
- `action_map`: `{action_id: {param_name: bool}}` for all `2^N` actions.

**`FullSpecShield`**: runtime shield evaluating the complete `#NeuralRequirement`.

- `__call__(proposed_action, obs_dict) → int`: evaluates `req_ast` with `unchanging ∪ obs_dict ∪ actuators`. If the proposed action satisfies the requirement, returns it unchanged. Otherwise calls `_requirement_action`.
- `_requirement_action(obs_dict)`: iterates all non-dead actions in ascending bit-count order. Returns the first action that satisfies `req_ast`. Falls back to action 0.

**`ProhibitionSpecShield`**: enforces only output-only prohibition clauses. Returns proposed action if not in `dead_actions`, otherwise returns nearest valid action by Hamming distance.

### Composite Model (`composite_model.py`)

`CompositeShieldedPolicy`: wraps a trainable `RecurrentActorCritic` with a frozen shield.

Two modes:
- **`full`**: `FullShieldNet` wraps `FullSpecShield` as `nn.Module`. Evaluates full `#NeuralRequirement` AST at runtime using `obs_dict`.
- **`none`**: `NoShieldNet` passes all actions through unchanged.

**`act(obs_t, hidden, obs_dict, greedy)`**:
1. Forward pass through policy network → `Categorical` distribution + value.
2. Sample (training) or argmax (eval) → `proposed_action`.
3. If `mode == "full"`: pass `proposed_action` + `policy_probs` + `obs_dict` through `FullShieldNet`. Shield either accepts or overrides.
4. Returns `(final_action, dist, value, hidden, overridden)`.

Gradients flow only through the policy network. The shield has `requires_grad=False` on all parameters and operates under `torch.no_grad()`.

**`FullShieldNet`** registered buffers:
- `dead_mask`: `(n_actions,)` tensor, 0 for dead actions, 1 for valid.
- `all_action_bits`: `(n_actions, n_out)` tensor encoding each action's boolean outputs.

### PPO (`ppo.py`)

`RecurrentPPO`: proximal policy optimization for recurrent policies.

**Episode collection**: `collect_episode` runs one episode through `composite.act()`, storing `(obs, action, reward, value, log_prob, done)` per timestep.

**`EpisodeBuffer.build_batch`**: pads variable-length episodes to `max_len`, computes GAE per episode, normalizes advantages across the batch. Returns tensors `(B, T, ...)` with binary `mask`.

**`RecurrentPPO.update`**:
- Entropy coefficient: linear decay from `entropy_coeff_start` to `entropy_coeff_end` over `total_updates`.
- Per epoch (4 epochs per update):
  - Forward `policy.forward_sequence(obs, h0, mask)` → logits, values.
  - PPO clipped surrogate: `ratio = exp(new_logp - old_logp)`, clipped to `[1-ε, 1+ε]`.
  - Loss = `policy_loss + 0.5 * value_loss - entropy_coeff * entropy`, masked to valid timesteps.
  - Gradient clipping: `max_grad_norm=0.5`.
  - Optimizer: Adam, `lr=3e-4`.

**Hyperparameters** (from `train.py`):
| Parameter | Value |
|---|---|
| `HIDDEN_DIM` | 64 |
| `EPISODES_PER_UPDATE` | 80 |
| `PPO_EPOCHS` | 4 |
| `CLIP_EPS` | 0.2 |
| `GAMMA` | 0.99 |
| `GAE_LAMBDA` | 0.95 |
| `LR` | 3e-4 |
| `ENTROPY_START` | 0.01 |
| `ENTROPY_END` | 0.001 |
| `MAX_GRAD_NORM` | 0.5 |
| `VALUE_COEFF` | 0.5 |
| `MAX_STEPS` | 1200 |
| `DT` | 0.1 |
| `ORACLE_SAMPLES` | 5,000 |
| `ORACLE_EPOCHS` | 30 |
| `PPO_EPISODES` | 5,000 |

## Training Regime (`train.py`)

Two-phase sequential training with full specification shield:

### Phase 1: Oracle Behavioral Cloning
1. Collect `5,000` `(observation, oracle_action)` pairs via `brute_force_oracle`.
2. Train policy network supervised with `CrossEntropyLoss`, `lr=1e-3`, `batch_size=1024`, `30` epochs.
3. Save `oracle.pt`.

### Phase 2: PPO Fine-Tuning
1. `5,000` episodes, `80` episodes per PPO update (62 updates total).
2. Shield active during rollout collection: policy proposes, shield overrides if `#NeuralRequirement` violated.
3. Greedy evaluation every `500` episodes (20 episodes). Best checkpoint saved as `best.pt`.
4. Entropy decays linearly from `0.01` → `0.001` over all updates.
5. Save `final.pt`.

### Pass Criteria
Detailed evaluation on 100 episodes with greedy policy:
- ≥95% success rate (goal reached).
- 0 safety violations (no `Prohibition`/`Obligation` with `status=False`).

## Usage

```bash
# From repo root:
python rl/train.py sysml-models/thermostat/model.sysml
python rl/train.py sysml-models/cruise-controller-model/model.sysml
python rl/train.py sysml-models/mixing-sysml-model/model.sysml
```

Checkpoints saved to `rl/checkpoints/full/{oracle,best,final}.pt`.
