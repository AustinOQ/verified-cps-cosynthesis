# Strengthening Invariant Heuristics

## Motivation

nuXmv's `msat_check_invar_bmc` with the een-sorensson algorithm uses k-induction to verify INVARSPEC properties on infinite-state (real-valued) models. k-induction works by checking whether a property is *k-inductive*: if it holds for k consecutive states, does it hold for the next state?

Many safety properties are not self-inductive because they depend on implicit relationships between variables that the checker cannot discover on its own. For example, verifying "a pump does not run when its feeder tank is empty" requires knowing that the sensor reading matches the tank level — a fact that is true in the model but not stated as a property.

**Strengthening invariants** are auxiliary INVARSPEC lines that encode these implicit relationships. They are conjoined with the main properties during k-induction, making the combined set inductive at low k values (typically k < 10 instead of k > 1000).

## How They Are Extracted

All four categories are derived mechanically from the SysML model structure — specifically from the scan-cycle phase numbering, the DEFINE chain topology, and the step-action / do-action classification.

### Data Sources

The extraction relies on metadata collected during SMV generation:

- **Phase effects** (`_phase_effects`): For each `SendStmt` in the scan cycle's action sequence, the translator records which do-action target is assigned, at which phase number, under which condition, and with which value expression.

- **Sensor read phases** (`_sensor_read_phases`): A subset of phase effects where a do-action target receives its value from a sensor read (an `AcceptStmt` → `SendStmt` → do-action assignment chain).

- **DEFINE chains** (`_defines`): The dictionary of all DEFINE aliases. These encode flow connections (port-to-port propagation), bind statements, and derived attributes. Following a chain of simple name aliases resolves a sensor's value source back to the underlying physical variable.

- **Step-action keys** (`_sa_keys`): The set of Euler-integrated state variables (tank levels, temperatures, etc.) updated via `next(x) := x + f(x) * dt`.

## Category 1: Sensor–Physical Variable Synchronization

**Pattern:** After a sensor read phase completes, the sensor's response variable equals the physical variable it measures, and remains equal until the next Euler integration step (phase 0).

**Generated form:**
```smv
INVARSPEC (scan_phase >= P+1 & scan_phase <= N) -> (sensor_var = physical_var)
```
where P is the phase at which the sensor read occurs and N is the maximum phase number.

**Extraction algorithm:**

1. For each entry in `_sensor_read_phases`, resolve the value source through the DEFINE chain using `_resolve_define_chain()`. This follows simple name-to-name aliases (e.g., `sensor_tankConnection_reading` → `tank_volumeSensorPort_reading` → `tank_currentLevel`).

2. If the resolved variable is a step-action target (i.e., it appears in `_sa_keys`), emit the invariant. This filter excludes actuator command targets (coil writes, etc.) whose value source does not chain back to a physical quantity.

**Why it helps:** Without this invariant, k-induction considers arbitrary states where the sensor reading differs from the tank level even though the scan cycle has already performed the read. This makes properties that depend on the sensor (e.g., "shouldRunPump is based on how much fluid has been transferred") non-inductive.

## Category 2: Non-Negativity of Euler-Integrated Variables

**Pattern:** Step-action variables (those updated by Euler integration) are non-negative.

**Generated form:**
```smv
INVARSPEC var >= 0
```

**Extraction algorithm:** Emit one invariant for each entry in `_step_actions`.

**Why it helps:** The Euler integration expression `x + (inflow - outflow) * dt` does not include clamping. However, the controller logic stops outflow before the variable reaches zero. Without stating non-negativity, k-induction can start from a state with a negative level, which breaks the inductive step for properties that depend on the level being physically meaningful.

## Category 3: Actuator–Condition Ordering

**Pattern:** After a scan cycle phase that deactivates an actuator (under a negated condition), if that condition remains false, the actuator must be off.

**Generated form:**
```smv
INVARSPEC (scan_phase >= P & scan_phase <= N & negated_cond) -> !actuator_var
```
where P is the last phase at which the actuator is turned off under the negated condition.

**Extraction algorithm:**

1. Group all phase effects by their target variable.

2. Skip targets already handled by Category 1 (sensor reads that resolve to step-action variables).

3. For each remaining target, scan its phase effects for negated conditions. A condition is considered negated if its SMV string starts with `(!` or `!`. The latest such phase becomes P, and the negated condition string becomes the guard.

**Why it helps:** k-induction may consider states mid-scan-cycle where an actuator is on despite the controlling condition being false. This invariant rules out those unreachable states by encoding the scan cycle's sequential execution order.

## Category 4: Actuator Coupling

**Pattern:** When two boolean do-action targets are controlled by the same condition within the scan cycle, and the cycle always activates one before the other (and deactivates in reverse order), then the later one being active implies the earlier one is active.

**Generated form:**
```smv
INVARSPEC later_actuator -> earlier_actuator
```

**Extraction algorithm:**

1. For each phase effect, strip the condition to its base form (removing leading `!` or `(!...)`). Classify each phase as "on" (non-negated condition) or "off" (negated condition) for its target.

2. Build a map: `target → {base_condition → {on: [phases], off: [phases]}}`.

3. For each pair of targets (A, B) sharing a base condition, check whether:
   - All of A's "on" phases precede all of B's "on" phases: `max(A.on) < min(B.on)`
   - All of B's "off" phases precede all of A's "off" phases: `max(B.off) < min(A.off)`

   If both hold, then B being active implies A is active (A "wraps" B in the phase ordering). Emit `B -> A`.

   The reverse check (B wraps A) is also performed.

**Why it helps:** In a typical scan cycle, the controller opens a flow-control device before starting a flow-producing device, and stops the producer before closing the controller. k-induction can consider states where the producer is active but the controller is not, which are unreachable due to the sequential phase ordering. This invariant eliminates those states.

## Applicability

These heuristics apply to any SysML model that uses:
- Scan-cycle sub-stepping (phase-based `SendStmt` sequencing)
- Euler-integrated step-action variables
- Sensor reads via do-action assignments
- Boolean do-action actuator targets controlled by conditions

Models without scan-cycle sub-stepping (e.g., a simple thermostat with direct state-machine control) produce no strengthening invariants, since their properties are typically self-inductive without assistance.
