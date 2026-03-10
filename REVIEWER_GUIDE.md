# Verified CPS Co-Synthesis — Reviewer Guide

## What This System Does

This pipeline automatically synthesizes and verifies neural controllers for
cyber-physical systems (CPS) from SysML v2 specifications. Given a SysML model
of a physical plant (tanks, thermostats, vehicles, etc.), the pipeline:

1. **Extracts** the neural controller interface from the SysML model — what the
   controller can observe, what actions it can take, what "done" means, and
   what safety rules it must follow.
2. **Simulates** the plant as a digital twin using a discrete-time engine built
   from the SysML dynamics, state machines, constraints, and flows.
3. **Trains** a neural controller via PPO (reinforcement learning) that learns
   to reach the goal while respecting safety prohibitions.
4. **Verifies** (downstream) the trained controller using beta-CROWN
   (per-input-point property verification) and nuXmv (temporal model checking).

The key idea: one SysML model is the single source of truth for the plant
physics, the controller interface, the safety requirements, AND the
verification properties. No manual reward engineering or environment code is
needed — point the pipeline at a `.sysml` file and it does the rest.

---

## Setup

```bash
cd verified-cps-cosynthesis

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Running All Three Examples

From `verified-cps-cosynthesis/`, with the venv active:

```bash
# 1. Chemical Mixing System
python -m RL_Phase.train_rl sysml-models/mixing-sysml-model/model.sysml \
    --shaping --randomize --timesteps 500000 -o rl_output_mixing

# 2. Thermostat
python -m RL_Phase.train_rl sysml-models/thermostat/thermostat.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 1000 -o rl_output_thermostat

# 3. Cruise Control
python -m RL_Phase.train_rl sysml-models/cruise-controller-model/model.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 500 -o rl_output_cruise
```

Each command runs extraction and training end-to-end. Outputs (trained model,
best checkpoint, interface metadata) are saved to the specified `-o` directory.

### Run All Three as a Script

```bash
source .venv/bin/activate

echo "=== Training Mixing System ==="
python -m RL_Phase.train_rl sysml-models/mixing-sysml-model/model.sysml \
    --shaping --randomize --timesteps 500000 -o rl_output_mixing

echo "=== Training Thermostat ==="
python -m RL_Phase.train_rl sysml-models/thermostat/thermostat.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 1000 -o rl_output_thermostat

echo "=== Training Cruise Control ==="
python -m RL_Phase.train_rl sysml-models/cruise-controller-model/model.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 500 -o rl_output_cruise

echo "=== All training complete ==="
```

Expected runtime: ~10 minutes per model on CPU (~30 minutes total).

### CLI Flags

| Flag | Purpose |
|------|---------|
| `--shaping` | Distance-based reward shaping (recommended, much faster convergence) |
| `--randomize` | Randomize start states each episode using `#ScenarioInput` metadata |
| `--timesteps N` | Total environment steps for training (default: 200,000; use 500,000 for reliable convergence) |
| `--max-steps N` | Max steps per episode before truncation (default: 200) |
| `-o DIR` | Output directory for trained models |

---

## The Three Models

### 1. Chemical Mixing System (`mixing-sysml-model/model.sysml`)

**Plant:** Two feeder tanks drain through pumps and valves into a single
filling tank. Pumps and valves are controlled over Modbus.

**Controller observes:** Tank volumes, transfer targets, original levels.

**Controller actions:** Open/close each valve, turn on/off each pump (4 boolean outputs).

**Goal (done):** Both tanks have transferred at least their target amounts.

**Safety rules:**

| Rule | Kind | Plain English |
|------|------|---------------|
| No Dry Running | Prohibition | Don't run a pump when its tank is empty. |
| No Dead Heading | Prohibition | Don't run a pump when its valve is closed. |
| Fluid Transfer Termination Safety | Obligation | Once a tank has transferred enough fluid, stop its pump and close its valve. |
| Fluid Transfer Liveness | Obligation | If a tank still needs to transfer fluid, its pump and valve should be active. |

**Randomization:** Transfer targets and original tank levels are randomized each episode.

### 2. Thermostat (`thermostat/thermostat.sysml`)

**Plant:** A room with a heater and an air conditioner. The room loses heat
to the outside environment via Newton's law of cooling.

**Controller observes:** Current temperature, setpoint.

**Controller actions:** Turn heater on/off, turn AC on/off (2 boolean outputs).

**Goal (done):** Temperature is within tolerance of setpoint.

**Safety rules:**

| Rule | Kind | Plain English |
|------|------|---------------|
| No Simultaneous Heating and Cooling | Prohibition | Don't run the heater and AC at the same time. |
| Heat When Cold | Obligation | If the temperature is below the setpoint minus tolerance, the heater should be on. |
| Cool When Hot | Obligation | If the temperature is above the setpoint plus tolerance, the AC should be on. |

**Randomization:** Setpoint and outside temperature are randomized each episode.

### 3. Cruise Control (`cruise-controller-model/model.sysml`)

**Plant:** A vehicle with mass and aerodynamic drag. An engine applies forward
force; a brake applies opposing force.

**Controller observes:** Current speed, target speed.

**Controller actions:** Apply throttle, apply brake (2 boolean outputs).

**Goal (done):** Speed is within tolerance of target speed.

**Safety rules:**

| Rule | Kind | Plain English |
|------|------|---------------|
| No Simultaneous Throttle and Brake | Prohibition | Don't apply throttle and brake at the same time. |
| No Throttle Above Target | Obligation | Don't throttle when speed is above the target speed. |
| No Brake Below Target | Obligation | Don't brake when speed is below the target speed. |
| Accelerate When Below Target | Obligation | If speed is below target minus tolerance, throttle should be on. |
| Brake When Above Target | Obligation | If speed is above target plus tolerance, brake should be on. |

**Randomization:** Target speed (5-25 m/s) is randomized each episode.

---

## What "Prohibition" vs "Obligation" Means for Training

- **Prohibitions** are enforced during RL training. Violating one gives -1
  reward and immediately ends the episode. The agent learns to never enter
  these states.
- **Obligations** are NOT enforced during training. The agent learns to satisfy
  them naturally through the goal reward. They are verified formally downstream
  by beta-CROWN and nuXmv.

---

## Expected Training Results

| Model | Final `ep_rew_mean` | Final `ep_len_mean` | Interpretation |
|-------|--------------------|--------------------|----------------|
| Mixing | ~10.9 | ~55 steps | Pumps both paths, transfers target volumes |
| Thermostat | ~10.5 | ~540 steps | Heats/cools to setpoint within tolerance |
| Cruise Control | ~11 | ~75 steps | Throttles to target speed, coasts into tolerance band |

After training, all three models should achieve **100% goal completion** and
**0 safety violations** when evaluated with the deterministic policy over 100
randomized episodes.

---

## Output Files

Each training run produces:

| File | Contents |
|------|----------|
| `model.zip` | Final PPO policy (stable-baselines3 format) |
| `model_best.zip` | Best checkpoint by rolling mean reward |
| `interface.json` | Extracted neural interface (obs names, action names, done expression) |
