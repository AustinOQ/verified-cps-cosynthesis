# SysML v2 Exploration

This project explores using SysML v2 as a machine-readable system model from which
multiple downstream artefacts — a simulation and a formal verification model — can be
derived automatically. The idea is that the SysML model is the single source of truth:
both tools read it directly and derive their behaviour from it, with no hardcoded domain
knowledge in the tooling.

## Example models

### `model.sysml` — fluid-transfer system with Modbus

Two feeder tanks fill a single filling tank via pumps and valves controlled over Modbus:

```
feederTank1 ──► pump1 ──► valve1 ──┐
                                    ├──► fillingTank
feederTank2 ──► pump2 ──► valve2 ──┘
```

A `Controller` has no state machine — its behaviour is defined by action bodies. A
step action checks whether enough time has elapsed since the last scan cycle, and if so
performs a `ScanCycle` action that:

1. Reads volume sensors over Modbus (`send` request → `accept` response).
2. Computes `shouldRunPump` by comparing the measured level to a transfer threshold.
3. Sends `WriteCoilReqMsg` commands to pumps and valves over Modbus.

The six Modbus devices (2 pumps, 2 valves, 2 volume sensors) each have a single-state
state machine that accepts a request on `modbusPort.req`, performs the corresponding
action, and sends a response on `modbusPort.resp`.

### `thermostat.sysml` — HVAC thermostat

A more realistic model with real-valued temperatures, multiple concurrent state machines,
and inter-component message passing via typed port connections:

```
environment ──► thermometer ──► controller ──► heater / ac
                                              (feedback via ports)
```

The `Controller` reads temperature from a `TemperatureSensor` and drives separate
`HeatingCoolingSource` state machines for a heater and an AC unit:

```
waiting ──(temp < setPoint − tolerance)──► heating ──(temp ≥ setPoint − tolerance)──► waiting
waiting ──(temp > setPoint + tolerance)──► cooling ──(temp ≤ setPoint + tolerance)──► waiting
```

This model exercises real arithmetic, `bind` statements, `flow` connections, `connect`
for port-to-port command routing, and typed `send`/`accept` triggers.

## Repository layout

| File | Purpose |
|------|---------|
| `model.sysml` | Fluid-transfer system model (Modbus, action bodies) |
| `thermostat.sysml` | HVAC thermostat model (state machines, real arithmetic) |
| `sysml_parser.py` | Shared parser: reads `.sysml` files into Python data structures |
| `simulator.py` | Discrete-time simulation driven by the parsed model |
| `mc-extract.py` | Translates the parsed model to nuXmv SMV for formal verification |
| `nuXmv` | nuXmv 2.1.0 binary (ARM64 / x86\_64 universal) |
| `sim-tests/` | Simulation regression tests (3 test cases) |
| `do-sim-tests.sh` | Runner script for simulation tests |
| `nuxmv-model-extraction-plan.md` | Implementation plan for mc-extract.py phases |
| `nuxmv-extraction-tests/` | Per-phase test models (`test-1a.sysml` … `test-3c.sysml`) |

## Architecture

### `sysml_parser.py`

Both downstream tools import `sysml_parser`. It is responsible for everything up to
the point of having a structured in-memory representation of the model.

**Parsing pipeline** (`SysMLParser.parse()`):

1. **PySysML2** — the [pysysml2](https://github.com/nicholasRenninger/pySysML2) library
   parses the file and extracts part instances and their attribute values (numeric
   parameters). This gives us fully-qualified names like `system::heater`.

2. **Regex passes** — several targeted regex passes extract what PySysML2 does not:
   part/item/port definitions with their attribute and member types, derived-attribute
   expressions, state machine declarations and transitions (triggers, guards, do-actions),
   `bind` statements, `flow` and `connect` declarations, `send`/`accept` patterns,
   action definitions, and `if`/`perform` statements.

3. **`_identify_system_part`** — looks for an explicit `part system : SomeType`
   declaration. When found it remaps all PySysML2-produced FQNs from `SomeType::*`
   to `system::*`, so the rest of the toolchain always works in terms of the instance
   namespace.

4. **`_build_constraints`** — instantiates constraints and derived attributes from
   part definitions onto each concrete part instance.

**Key data structures produced:**

- `parameters` — numeric attribute values (temperatures, capacities, flow rates, …)
- `parsed_constraints` — expression ASTs for constraints, tagged with their instance context
- `derived_attributes` — expression ASTs for derived boolean/real attributes
- `state_machines` — state names and transitions (triggers, guards, do-actions)
- `step_action_bodies` — action statement lists per instance (for SM-less controllers)
- `flows` — directed port-to-port connections (`flow from A.portX to B.portY`)
- `connects` — port-to-port command links (`connect A.portX to B.portY`)
- `parsed_bindings` — `bind lhs = rhs` aliases
- `instance_state_machines` — map from instance FQN to `StateMachine` object
- `item_def_attrs`, `port_def_items`, `part_def_ports` — type metadata for chain resolution
- `item_type_parents` — inheritance chain for `item def X :> Y`

**Expression AST** — expressions are parsed into a small AST:
`LiteralExpr`, `RefExpr`, `BinaryExpr` (`+`, `-`, `*`, `/`, `==`, `>=`, `and`,
`or`, `implies`, …), `TernaryExpr`, `UnaryExpr`, and `DerExpr` (`der(variable)`).

### `simulator.py` — discrete-time simulation

The simulator runs a fixed-timestep Euler integration loop driven entirely by what the
parser extracted.

**`ConstraintSolver`** classifies each constraint on first encounter:

- Contains `der()` → extract as a `DynamicsEquation`
- Contains `implies` → state-machine constraint (applied when the condition holds)
- Otherwise → algebraic assignment constraint

Each simulation step:

1. **Action body execution** — for parts with step action bodies (e.g. the controller),
   execute the action statements: `if`/`perform` control flow, `send`/`accept` for
   Modbus message passing, `assign` for state updates, and `attribute` declarations.
   Sends trigger recipient state machine transitions immediately (mid-step).
2. **State machine transitions** — check triggers and guards; advance the current state.
3. **Constraint solving** — evaluate derived attributes and implies-constraints; iterate
   flow propagation three times to reach a fixed point.
4. **Euler integration** — evaluate `der(x) = expr` RHS; apply `x += expr * dt`.

Run the simulator:

```
python simulator.py                           # default: model.sysml, 20s, dt=0.1
python simulator.py -m thermostat.sysml -d 60
python simulator.py --duration 15 --timestep 0.1
python simulator.py --list-parameters         # show all tunable values
```

### `mc-extract.py` — nuXmv SMV generator

Translates the parsed model to [nuXmv](https://nuxmv.fbk.eu) SMV. The translation
implements a multi-phase pipeline described in `nuxmv-model-extraction-plan.md`.

**SysML → SMV mapping:**

| SysML | SMV |
|-------|-----|
| State machine instance (≥ 2 states) | `VAR inst_state : {s1, s2, …}` + `next()` case |
| Single-state SM | Elided (nuXmv treats as constant) |
| Real-typed attribute (`Real`) | `VAR attr : real` + unclamped Euler `next()` |
| Integer-typed step-action target | Bounded integer `VAR` + clamped `next()` (or `real` when `--dt` is fractional) |
| `accept Trigger via port` | Boolean `IVAR inst_port_Trigger_available` |
| Trigger-var attribute (e.g. `reading.temperatureCelcius`) | `IVAR attr : real` promoted to `DEFINE` in Phase 3b |
| `bind lhs = rhs` | `DEFINE lhs := rhs` (Phase 2) |
| `flow from A.portX to B.portY` | `DEFINE B_portY_attr := A_portX_attr` (Phase 3a) |
| `connect A.outPort to B.inPort` + `send … via outPort` | Trigger IVAR replaced by sender firing condition `DEFINE` (Phase 3c) |
| Controller action body (SM-less) | `scan_fires`, `shouldRunPump`, coupled trigger DEFINEs (Phase 3d) |
| Numeric parameters | `DEFINE` constants |
| Derived attributes | `DEFINE` expressions |
| Unconnected flow ports | `DEFINE port_flowRate := 0` |

**Extraction pipeline phases:**

- **Phase 1** (1a–1f): Multiple SM instances, real vs. integer arithmetic detection,
  guard AST translation, boolean trigger IVARs, do-action `VAR` entries, placeholder specs.
- **Phase 2**: `bind` → `DEFINE` propagation.
- **Implies constraints**: Non-behaviour implies constraints → conditional `DEFINE` (case
  expressions for pump/valve flow rates). Must run before Phase 3a.
- **Phase 3a**: Generic `flow` → `DEFINE` propagation (fixed-point iteration).
- **Phase 3b**: Trigger-var real IVARs promoted to `DEFINE` aliases following the
  flow/bind chain to the ultimate source variable.
- **Phase 3c**: `connect` + `send` coupling — free boolean trigger IVARs replaced by
  `DEFINE` expressions derived from the sender's firing conditions, eliminating
  spurious nondeterminism.
- **Phase 3d**: Action-body-aware coupling for SM-less controllers. Walks the controller's
  step and `ScanCycle` action bodies to emit:
  - `scan_fires` — condition for the scan cycle to execute
  - `shouldRunPump` DEFINEs — computed from accept-var resolution through the connect map
  - Trigger-var attribute coupling (`coilValue`, `coilAddress`, `registerAddress`)
  - Boolean trigger IVARs coupled to `scan_fires`

Run the extractor:

```
python mc-extract.py model.sysml -o out.smv                    # default dt=1
python mc-extract.py model.sysml -o out.smv --dt 0.1           # fractional timestep
python mc-extract.py model.sysml -o out.smv --max-int 10000    # cap integer ranges
python mc-extract.py thermostat.sysml -o out.smv               # thermostat model
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dt DT` | `1` | Time step for Euler integration. Fractional values (e.g. `0.1`) promote all integer step-action targets to `real`. |
| `--max-int N` | `2147483647` | Upper bound for integer variable ranges. Use a smaller value (e.g. `10000`) to make the SMT solver tractable. |
| `-o FILE` | stdout | Write SMV output to a file instead of stdout. |

## Verification with nuXmv

The generated SMV models use nuXmv's native `real` arithmetic (via the MathSAT5 SMT
backend). The interactive mode is required; the batch `./nuXmv out.smv` invocation only
supports integer-arithmetic solvers.

**Start an interactive session:**

```
./nuXmv -int out.smv
```

### Verification methods

nuXmv offers several verification methods. The right choice depends on whether the model
is finite-state (all integer/enum/boolean variables) or infinite-state (contains `real`
variables), and whether you need a full proof or just bug-finding.

#### 1. IC3 / PDR — unbounded proof (finite-state only)

```
go_msat
check_ltlspec_ic3
```

Uses the IC3/PDR (Property Directed Reachability) algorithm. Proves or disproves each
`LTLSPEC` for **all reachable states and all time**. Returns a counterexample trace if
the property fails.

**Limitation:** IC3 only works on finite-state models. If the model contains `real`
variables (which happens when `--dt` is fractional, or when SysML attributes are typed
`Real`), IC3 cannot verify the model — it operates on finite abstractions and will
either reject the model or fail to converge.

#### 2. BMC — bounded model checking (finite or infinite-state)

```
go_msat
msat_check_ltlspec_bmc -k 60       # LTLSPEC properties
msat_check_invar_bmc -k 60         # INVARSPEC properties
```

Unrolls the transition relation `k` steps and checks whether any counterexample of
length ≤ k exists. Works with both finite and infinite-state models via SMT solving.
Can find bugs up to depth k but **cannot prove safety** — the absence of a
counterexample at depth k does not mean the property holds at depth k+1.

`msat_check_ltlspec_bmc_onepb -k N` checks exactly length-N paths only (useful for
liveness).

#### 3. k-Induction — unbounded proof (infinite-state)

```
go_msat
msat_check_invar_bmc -a een-sorensson
```

Uses the een-sorensson algorithm, which combines BMC with k-induction. For each
`INVARSPEC`, it tries to find a k such that:

1. The property holds for all states reachable in 0..k steps (base case).
2. If the property holds for k consecutive states, it holds for the next (inductive step).

If both succeed, the property is proven for **all reachable states**. This is the
primary method for verifying real-valued models.

**Important:** Safety properties must be written as `INVARSPEC φ`, not `LTLSPEC G(φ)`.
The two are semantically equivalent for invariant properties, but `msat_check_invar_bmc`
only processes `INVARSPEC`.

##### Strengthening invariants

k-induction requires properties to be *k-inductive*: the inductive step must go through
using only the properties themselves as hypotheses. Many safety properties are not
self-inductive because they depend on implicit relationships between variables (e.g.,
a sensor reading matches its tank level after the read phase). Without help, the
checker considers unreachable states where these relationships are violated, and the
inductive step fails.

`mc-extract.py` automatically generates **strengthening invariants** that encode these
implicit relationships. They are conjoined with the main properties during k-induction,
making the combined set inductive at low k values (typically k < 10). Four categories
are generated:

1. **Sensor–variable sync** — after a sensor read phase, the sensor response equals
   the physical variable it measures.
2. **Non-negativity** — Euler-integrated variables (tank levels, temperatures) are ≥ 0.
3. **Actuator–condition ordering** — after an actuator's "off" phase, if the controlling
   condition is false, the actuator is off.
4. **Actuator coupling** — when two actuators share a controlling condition and the scan
   cycle activates one before the other (deactivating in reverse order), the later one
   being on implies the earlier one is on.

All four categories are derived mechanically from model structure (phase ordering,
DEFINE chains, step-action classification). See
[invariant-heuristics.md](invariant-heuristics.md) for details.

### LTLSPEC vs INVARSPEC

| Form | Use with | Notes |
|------|----------|-------|
| `LTLSPEC G(φ)` | `check_ltlspec_ic3`, `msat_check_ltlspec_bmc` | Full LTL syntax, but k-induction not available |
| `INVARSPEC φ` | `msat_check_invar_bmc` | Required for k-induction (`-a een-sorensson`) |

For safety properties (always-true invariants), `LTLSPEC G(φ)` and `INVARSPEC φ` are
semantically equivalent. `mc-extract.py` emits `INVARSPEC` by default for models that
use scan-cycle sub-stepping, since k-induction is the only viable unbounded proof method
for real-valued models.

### Choosing `k` relative to `dt`

The physical time horizon of a BMC run is `k × dt`. For the thermostat at `--dt 1`:

- Each step changes temperature by ≈ 0.083 °C (2000 W into 20 m³ of air).
- Cooling from the initial 23.9 °C to the 19.3 °C setpoint + tolerance takes ≈ 55 steps.
- A run intended to witness temperature regulation should use at least `k = 60`.

To reduce the number of steps required, increase `--dt`:

| `--dt` | Steps to cool 4.6 °C | Physical seconds modelled |
|--------|---------------------|--------------------------|
| 1      | ~55                 | 55 s                     |
| 5      | ~11                 | 55 s                     |
| 10     | ~6                  | 60 s                     |

Larger `dt` is less accurate (coarser Euler steps) but dramatically reduces the BMC
depth needed to observe slow physical dynamics.

### Scan-cycle sub-stepping and verification depth

Models with scan-cycle controllers (e.g., the fluid-transfer model) use phase-based
sub-stepping: each scan cycle is broken into N sub-phases (e.g., 14 phases for the
mixing model). This means one physical time step requires N+1 nuXmv transitions.
A BMC bound of k covers only `k / (N+1)` physical time steps.

With strengthening invariants and k-induction (`-a een-sorensson`), the proof typically
converges at very low k (often k < 10), regardless of sub-stepping depth.

For the fluid-transfer model, use `--max-int` to cap integer ranges if using integer
mode (the default 2147483647 causes `go_msat` to hang):

```
python mc-extract.py model.sysml --max-int 10000 -o out.smv
```

## Simulation tests

Three regression tests verify the simulator against known-good traces:

```
bash do-sim-tests.sh
```

| Test | What it covers |
|------|----------------|
| `01-transition-ambiguity` | State machine transition priority / guard evaluation |
| `02-integer-real-promotion` | Mixed integer/real arithmetic in constraints |
| `03-action-send-accept` | Controller action body execution: if/perform, Modbus send/accept |

Each test has a `.sysml` model and a `.py` script that runs the simulator and asserts
expected state values at specific time steps.

## Limitations and design notes

**Parser coverage.** The parser is purpose-built for the patterns used in these models.
It handles `bind`, `flow`, `connect`, `send`/`accept`, real arithmetic, multi-SM
patterns, action bodies (`if`, `perform`, `assign`, `attribute`, `accept`, `send`),
and Modbus message hierarchies, but does not implement the full SysML v2 grammar.
PySysML2 provides the structural backbone; regex passes fill in the rest.

**Continuous vs. discrete.** The simulator uses floating-point Euler integration.
`mc-extract.py` generates `real`-typed SMV variables for `Real`-typed SysML attributes
(thermostat), and bounded integer variables for integer-typed attributes (model.sysml).
When `--dt` is fractional, integer targets are promoted to `real` to avoid nuXmv type
errors. Properties verified by nuXmv hold for the discretised model. For the fluid-transfer
model the discretisation is exact (at integer `dt`); for the thermostat it is an
approximation.

**Ref bindings are instance-scoped.** References inside `part def Controller` that
cross into sibling parts (e.g. `pump1.isRunning`) are resolved by explicit
`ref :>> pump1 = system::pump1` bindings in the controller instance.

**Single-state SMs are elided.** Devices like pumps, valves, and volume sensors that have
only one SM state (`waiting`) are not emitted as SMV variables — nuXmv would collapse
them to constants and reject assignments to them. Their transition guards are simplified
to drop the always-true state condition.
