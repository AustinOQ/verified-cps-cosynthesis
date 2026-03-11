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
    --shaping --randomize --timesteps 500000 --seed 42 -o rl_output_mixing

# 2. Thermostat
python -m RL_Phase.train_rl sysml-models/thermostat/thermostat.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 1000 --seed 42 -o rl_output_thermostat

# 3. Cruise Control
python -m RL_Phase.train_rl sysml-models/cruise-controller-model/model.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 500 --seed 42 -o rl_output_cruise
```

Each command runs extraction and training end-to-end. The best model checkpoint
(`model_best.zip`) is saved based on periodic micro-test evaluations during
training (100 deterministic episodes with randomized scenarios every 50,000 steps).

### Run All Three as a Script

```bash
source .venv/bin/activate

echo "=== Training Mixing System ==="
python -m RL_Phase.train_rl sysml-models/mixing-sysml-model/model.sysml \
    --shaping --randomize --timesteps 500000 --seed 42 -o rl_output_mixing

echo "=== Training Thermostat ==="
python -m RL_Phase.train_rl sysml-models/thermostat/thermostat.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 1000 --seed 42 -o rl_output_thermostat

echo "=== Training Cruise Control ==="
python -m RL_Phase.train_rl sysml-models/cruise-controller-model/model.sysml \
    --shaping --randomize --timesteps 500000 --max-steps 500 --seed 42 -o rl_output_cruise

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
| `--seed N` | Random seed for reproducibility (recommended: 42) |
| `-o DIR` | Output directory for trained models |

---

## The Three Models

### 1. Chemical Mixing System (`mixing-sysml-model/model.sysml`)

**Plant:** Two feeder tanks drain through pumps and valves into a single
filling tank. Pumps and valves are controlled over Modbus.

**Controller observes:** Tank volumes, transfer targets, original levels.

**Controller actions:** Open/close each valve, turn on/off each pump (4 boolean outputs).

**Goal (done):** Both tanks have transferred at least their target amounts.

**Safety rules enforced during training (Prohibitions):**

| Rule | Plain English |
|------|---------------|
| No Dry Running | Don't run a pump when its tank is empty. |
| No Dead Heading | Don't run a pump when its valve is closed. |

**Randomization:** Transfer targets and original tank levels are randomized each episode.

### 2. Thermostat (`thermostat/thermostat.sysml`)

**Plant:** A room with a heater and an air conditioner. The room loses heat
to the outside environment via Newton's law of cooling.

**Controller observes:** Current temperature, setpoint.

**Controller actions:** Turn heater on/off, turn AC on/off (2 boolean outputs).

**Goal (done):** Temperature is within tolerance of setpoint.

**Safety rules enforced during training (Prohibitions):**

| Rule | Plain English |
|------|---------------|
| No Simultaneous Heating and Cooling | Don't run the heater and AC at the same time. |

**Randomization:** Setpoint and outside temperature are randomized each episode.

### 3. Cruise Control (`cruise-controller-model/model.sysml`)

**Plant:** A vehicle with mass and aerodynamic drag. An engine applies forward
force; a brake applies opposing force.

**Controller observes:** Current speed, target speed.

**Controller actions:** Apply throttle, apply brake (2 boolean outputs).

**Goal (done):** Speed is within tolerance of target speed.

**Safety rules enforced during training (Prohibitions):**

| Rule | Plain English |
|------|---------------|
| No Simultaneous Throttle and Brake | Don't apply throttle and brake at the same time. |

**Randomization:** Target speed (5-25 m/s) is randomized each episode.

---

## How Training Works

### Reward Protocol
- **Goal reached:** +10 reward, episode terminates successfully.
- **Prohibition violated:** -1 reward, episode terminates immediately.
- **Normal step:** Per-step distance improvement toward the goal (reward shaping).

### Model Selection
During training, a micro-test evaluation runs every 50,000 steps: 100 episodes
with randomized scenarios, deterministic policy (no exploration). The model
checkpoint with the highest goal-completion rate is saved as `model_best.zip`.
This guards against PPO's occasional late-training policy instability.

### Reproducibility
All training uses `--seed 42`, which seeds Python's random, NumPy, and PyTorch
via stable-baselines3. The deterministic evaluation policy (`predict(obs,
deterministic=True)`) produces identical actions for identical observations.

---

## Expected Training Results

| Model | Final `ep_rew_mean` | Final `ep_len_mean` | Interpretation |
|-------|--------------------|--------------------|----------------|
| Mixing | ~10.9 | ~55 steps | Pumps both paths, transfers target volumes |
| Thermostat | ~10.5 | ~540 steps | Heats/cools to setpoint within tolerance |
| Cruise Control | ~11 | ~75 steps | Throttles to target speed, coasts into tolerance band |

After training, all three models achieve **100% goal completion** and
**0 safety violations** when evaluated with the deterministic policy over 100
randomized episodes (verified with seed 42).

---

## Verifying Trained Models

After training, verify the results by running the deterministic policy on 100
randomized episodes per model:

```python
# Run from verified-cps-cosynthesis/ with venv active
python -c "
from RL_Phase.extractor import extract_neural_interface
from RL_Phase.env import SysMLEnv
from stable_baselines3 import PPO

models = [
    ('Mixing', 'sysml-models/mixing-sysml-model/model.sysml', 'rl_output_mixing/model_best.zip', 200, 0.1),
    ('Thermostat', 'sysml-models/thermostat/thermostat.sysml', 'rl_output_thermostat/model_best.zip', 1000, 0.1),
    ('Cruise', 'sysml-models/cruise-controller-model/model.sysml', 'rl_output_cruise/model_best.zip', 500, 0.1),
]

for name, sysml, ckpt, max_steps, dt in models:
    iface = extract_neural_interface(sysml)
    env = SysMLEnv(sysml, iface, dt=dt, max_steps=max_steps, randomize=True, shaping=True)
    agent = PPO.load(ckpt, env=env)
    goals, violations, timeouts = 0, 0, 0
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        done = truncated = False
        info = {}
        while not done and not truncated:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            if info.get('violation'):
                violations += 1
                break
        if done and not info.get('violation'):
            goals += 1
        elif not done and not info.get('violation'):
            timeouts += 1
    status = 'PASS' if goals == 100 and violations == 0 else 'FAIL'
    print(f'[{status}] {name}: goals={goals}/100  violations={violations}  timeouts={timeouts}')
    env.close()
"
```

Expected output:
```
[PASS] Mixing: goals=100/100  violations=0  timeouts=0
[PASS] Thermostat: goals=100/100  violations=0  timeouts=0
[PASS] Cruise: goals=100/100  violations=0  timeouts=0
```

---

## Output Files

Each training run produces:

| File | Contents |
|------|----------|
| `model_best.zip` | Best PPO policy checkpoint (stable-baselines3 format), selected by micro-test evaluation |
| `interface.json` | Extracted neural interface (obs names, action names, done expression) |
