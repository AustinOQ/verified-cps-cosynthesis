# Discretization Safety Certification Design

## Purpose

This document specifies a preprocessing stage that determines whether the
chosen `dt` preserves the SysML safety properties between controller updates.
The stage consumes the checked Markov decision process certificate, the SysML
equations, the safety properties, and the timing information. It produces a
checked safety certificate, a checked violation, or a loud `NOT_CERTIFIED`
result.

The earlier mathematical margin proof remains a human readable explanation.
It is not the required computational method. The computational stage selects
the least expensive checker whose assumptions it can prove from the extracted
model.

## Nonnegotiable trust rule

A checker may report `CERTIFIED` only when both of the following are true.

1. The checker has proved that every relevant part of the extracted model is
   within the exact class of equations that the checker supports.
2. An independent certificate checker has accepted the resulting proof.

Every checker invocation for one property case returns exactly one of these
outcomes.

| Outcome | Meaning | Pipeline action |
|---|---|---|
| `CERTIFIED` | A complete certificate was independently checked. | Stop with a certified result. |
| `VIOLATION` | A complete counterexample was independently replayed. | Stop with a violation result. |
| `DEFERRED` | The method did not establish its assumptions or did not reach a conclusion. | Fail loudly for this method, record the reason, and pass the case to the next checker. |

An unsupported expression, missing bound, unresolved reference, unhandled
guard, numerical failure, checker crash, malformed output, or timeout can
never be converted into `CERTIFIED`. If any case remains deferred after the
SMT checker, including because that checker times out, the final result is
`NOT_CERTIFIED`.

`DEFERRED` is used instead of `PASS` because the existing Markov decision
process certificate already uses `PASS` to mean that its own proof succeeded.
The two meanings must not share one machine readable value.

## Claim being certified

The Markov decision process certificate establishes that the selected current
observation and action buffer is enough to determine the modeled next step. It
does not by itself establish what can happen between controller updates.

For each safety property, the new stage asks whether there is any permitted
sampled state, executed action, and time before the next usable controller
update for which the physical state reaches the unsafe condition. The maximum
time from the physical state represented by the controller input to the next
effective controller action is the blind interval. The timing analysis derives
this interval from the fixed `dt`, the controller schedule, observation age,
and actuation timing. The existing information delay addressed by the
observation and action buffer remains part of the Markov decision process
certificate and is not silently treated as elapsed physical time.

The claim is conditional on a checked sampled point premise. A sampled point
premise is the condition already proved to hold whenever the controller acts.
It includes the relationship among the reconstructed physical state, the
current observation and action buffer, and the action actually executed after
the shield. The new stage must either derive and check this premise from the
Markov decision process certificate and controller contract or consume a
separate checked sampled point certificate. It cannot assume that a property
holds at controller updates merely because a shield exists.

The physical state at the end of one blind interval must also satisfy the
sampled point premise needed for the next interval. The certificate must
therefore prove either one step preservation of that premise or a stated finite
horizon beginning from the model's initial and scenario conditions. A local
between step calculation without this connection is not a complete
`CERTIFIED` result.

The existence of such an execution is the authoritative counterexample
question. Every checker must answer this same question. A checker may change
how the question is calculated, but it may not weaken the question.

The expression of each property is preserved exactly. An observed quantity is
not replaced by its physical source unless the sensor and timing equations
prove the required relationship. A property written only over an observation
certifies that observation based statement. A property intended to constrain
the physical quantity must identify that quantity or provide a checked mapping
to it.

## Required inputs

The stage accepts the following checked inputs for each model.

1. The SysML model and its file hash.
2. The single validated `dt` used by the complete artifact run.
3. The Markov decision process certificate and its hash.
4. The reduced Markov decision process specification generated from that
   certificate.
5. The transition closed relevant state identified by the existing relevance
   analysis.
6. The current and buffered observations and executed actions used to
   reconstruct that state.
7. The SysML state equations, same cycle definitions, guards, actions,
   requirements, prohibitions, obligations, completion condition, scenario
   bounds, and initial values.
8. The rule describing how an executed action is held or changed before the
   next controller update.
9. The extracted timing defect envelope. A zero envelope is permitted only
   when the timing analysis certifies that value.
10. The sampled point premise and the evidence that establishes it at the
    first controller update and preserves it at later updates.
11. The exact source text for every numeric constant, including `dt`, so the
    proof representation does not begin with a rounded floating point value.

Missing or ambiguous inputs are recorded and passed through every checker that
might support them. If no checker can establish their meaning, the final result
is `NOT_CERTIFIED`.

## Common preprocessing

### 1. Validate the inputs

The stage first invokes the existing independent Markov decision process
certificate checker. It then compares the model hash, certificate hash,
reduced specification hash, and `dt`. It requires the existing certificate's
`theorem_gate.full_mdp_theorem_claim_allowed` field to be true and its
`claim.solver_backed_mdp_theorem` field to be `discharged`. The new stage may
reuse the reconstructed state, buffer, schedule facts, and executed action
meaning. It may not reuse the existing one step transition claim as a claim
about continuous behavior unless the endpoint relationship described below is
also certified. Any mismatch prevents certification.

### 2. Fix the numeric meaning

The proof uses mathematical integers and real numbers. Decimal constants are
parsed directly from their source text into exact fractions. The command line
representation of `dt` must therefore be retained as text and converted to one
canonical exact fraction for certificates. Converting `dt` to a Python float
before certification is not permitted.

The extractor checks every assignment against its declared SysML numeric type.
An assignment to an integer quantity that can produce a noninteger value is
unsupported unless the model declares its rounding rule. A certificate about
the mathematical equations does not also certify floating point execution.
Any claimed floating point implementation correspondence requires separate
outward rounded error bounds.

### 3. Extract cycle ordering and within step meaning

The current equation representation records next step assignments. The new
stage must first record the actual order of state machine transitions,
constraint propagation, physical updates, sensor updates, controller calls,
and actuator changes. Assignments that execute sequentially cannot be treated
as simultaneous without a checked rule saying they are simultaneous.

The stage then classifies how each relevant physical quantity behaves inside
one step. The supported forms are a held value, a constant rate, a rate that
depends on the evolving state, an instantaneous change, or a guarded selection
among these forms.

The extractor may use a within step rule only when that rule is explicit in
the SysML model or in a separately checked translation rule selected by an
explicit model annotation. A next step assignment alone does not justify
inventing a continuous interpolation. For example, an annotated assignment of
the exact form `x := x + rate * dt` may declare that the instantaneous rate of
`x` equals `rate` between updates. The annotation must also state whether
coupled rates use the same physical state or the sequential assignment order.

The extractor creates two relations. The first describes physical evolution
inside the blind interval. The second describes the sampled next state used by
the existing Markov decision process proof. It then creates an endpoint
obligation showing that the physical state at the next sample agrees with the
sampled transition or lies within a separately certified error bound. Failure
to establish this endpoint obligation prevents reuse of the sampled transition
claim.

### 4. Derive the timing envelope

The timing analysis records the time at which a physical value exists, the
time at which it is observed, the time at which the controller decides, and the
time at which the resulting action becomes effective. It computes a bound for
the complete interval during which the controller must act without newer
usable physical information.

A time varying delay is represented as a checked function of the reconstructed
schedule state or as a finite set of schedule cases. A single global envelope
may be obtained by taking the checked maximum over those cases. An
unsynchronized scan condition is evaluated with exact `dt` arithmetic and the
actual cycle ordering. Unused attributes do not contribute a delay merely
because their names suggest one.

The timing analysis also records the initialization sequence before the first
controller action, including default actions and simulator priming steps. The
first sampled point premise must be checked after that sequence, and every
physical interval before it must be covered separately. Completion and maximum
step limits are taken from the checked reduced specification, with the final
physical interval and any terminal instantaneous change included.

### 5. Establish the sampled point premise

The executed action relation comes from the checked shield and includes every
action the shield can actually produce, not the proposed neural output alone.
For a continuous action, the relation includes the complete permitted output
interval and any state dependent restriction. For a Boolean action, it
includes every permitted action mapping.

The stage relates this executed action to the reconstructed physical state and
to the exact safety property at controller update points. If the controller
contract implies the sampled point premise, that implication receives its own
checked proof. If it does not, the pipeline must obtain a separate sampled
point certificate or finish as `NOT_CERTIFIED`.

### 6. Reduce the model for each property

For each safety property, the stage follows equation references backward from
the unsafe condition. It retains every state quantity, action, definition,
guard, and timing term that can affect that condition. It then closes this set
through the relevant next step equations. Nothing may be removed without a
recorded noninterference reason showing that it cannot affect the property.

### 7. Form the unsafe condition

The default proof targets are every parsed `#Prohibition` and `#Obligation` in
the model. The `#NeuralRequirement` defines the controller action relation used
by the sampled point premise. The stage translates each target statement into
the condition under which it is false. This translation covers quantity
constraints and forbidden state and action combinations. The translated
condition retains the source location and annotation kind from the model.

Boolean structure is converted into a complete set of unsafe alternatives.
Strict and nonstrict comparisons retain their original boundary meaning. A
zero value at a boundary may be safe or unsafe depending on that original
comparison, so the checker may not impose a positive margin universally.

### 8. Build the case coverage record

Each possible guard selection and instantaneous change is treated as a
separate case. The cases must cover every permitted execution. If the stage
cannot establish complete coverage, no later calculation over only the known
cases may certify the property.

The case coverage record is the complete list of cases whose union contains
every permitted execution for one property. Each checker works on individual
unresolved cases. It may certify some cases and defer others. Only the
unresolved cases proceed to the next checker, but the property remains
uncertified until every case in the record has a checked certificate. No case
may disappear between stages.

If a checker divides one case into smaller state or time regions, it records
the child case identifiers and a checked proof that their union contains the
parent case. The parent is replaced only after that coverage proof succeeds.

## Machine readable progression contract

Every checker invocation writes one result object before the next invocation
begins. The required fields are `property_id`, `case_id`, `checker`,
`checker_version`, `outcome`, `applicability_checks`, `reason_code`,
`evidence_paths`, `input_hashes`, `started_at`, and `elapsed_seconds`.

The initial reason codes are `UNSUPPORTED_EXPRESSION`, `MISSING_BOUND`,
`NOT_LINEAR`, `MISSING_WITHIN_STEP_MEANING`, `INCOMPLETE_CASE_COVERAGE`, `BLOCKED_INPUT`,
`NUMERIC_BOUND_INCONCLUSIVE`, `TIMEOUT`, `MALFORMED_OUTPUT`,
`PROOF_REJECTED`, and `COUNTEREXAMPLE_REPLAY_FAILED`. A new reason code requires
a schema version change rather than being written as an unstructured success
or failure string.

The orchestrator applies this fixed control rule to every unresolved case.

```text
for each checker in the declared order
    run the checker on every unresolved case
    retain every independently checked certificate
    stop immediately on an independently replayed violation
    pass every deferred case to the next checker

if every case has an independently checked certificate
    return CERTIFIED
otherwise
    return NOT_CERTIFIED
```

The stage exit codes are `0` for `CERTIFIED`, `2` for `VIOLATION`, and `3` for
`NOT_CERTIFIED`. Checker process exit codes are evidence only and never become
proof outcomes without parsing and checking the corresponding result object.

## Checker progression

A computed outer bound can prove safety only when it does not intersect the
unsafe condition. Intersection with an outer bound is inconclusive because the
intersecting point may not be physically reachable. It becomes `VIOLATION`
only after exact counterexample replay succeeds. Otherwise the case is
`DEFERRED`.

### Initial supported expression set

The first implementation supports exact rational constants, Boolean values,
integer and real state quantities, addition, subtraction, multiplication,
division by a proved nonzero constant, comparisons, conjunction, disjunction,
negation, implication, and conditional expressions that can be split into
complete guard cases. Polynomial expressions are retained exactly. Any other
operation is `UNSUPPORTED_EXPRESSION` and is passed through the progression.

This set covers the arithmetic currently present in the four bundled physical
models. Supporting an operation syntactically does not make it applicable to a
particular checker. Variable multiplication is retained for the convex, exact
symbolic, or SMT stages while the linear stage must return `NOT_LINEAR`.

### Checker 1. Linear checker

This checker handles equations and constraints that are linear in the relevant
state quantities and actions. Before solving anything, it recursively
normalizes every algebraic expression into a constant plus a weighted sum of
variables. The normalization either returns that exact representation or
returns `NOT_LINEAR` with the expression and source location.

The following always produce `NOT_LINEAR` unless one side was already proved
constant.

- Multiplication of two variable dependent expressions
- Division by a variable dependent expression
- A variable raised to a power other than one
- A nonlinear function
- A conditional expression whose cases were not separated completely
- A nonlinear equality hidden behind a same cycle definition

The linear checker may proceed only if every retained equation, bound, guard
case, timing expression, and unsafe condition passes this test. It then uses
linear optimization to determine whether the reachable states can intersect
the unsafe condition.

The current setup accepts only a complete counterexample question that is
already one conjunction of linear constraints. It does not enumerate Boolean
or disjunctive cases. Those inputs produce `DEFERRED` and pass to the convex
checker. Strict comparisons retain their exact direction.

A safety certificate records numbers showing that a weighted sum of the
accepted linear constraints contradicts the unsafe condition. The independent
checker recomputes that weighted sum using exact rational numbers. A floating
point optimizer result is only a candidate for this exact certificate. Strict
comparisons are checked using their exact original direction rather than an
arbitrary numerical tolerance.

### Checker 2. Certified convex bound checker

This checker accepts a nonlinear relation only after exact extraction proves
that it is a diagonal convex quadratic constraint. A diagonal quadratic has no
products between different variables. Unsupported products, compositions, or
logical case splits produce `DEFERRED`.

The current certificate form accepts a conjunction of diagonal convex
quadratic constraints. The solver searches for nonnegative dual weights. The
independent checker combines the constraints with exact rational arithmetic,
computes the exact global lower bound of the combined quadratic, and accepts
only when that bound contradicts the unsafe constraints. A numerical solution
that cannot be converted into this exact certificate produces `DEFERRED`.

The claim that individual relations are convex is not enough. A nonlinear
equality, unsupported composition, incomplete input range, guard, or unsafe
condition can make this checker inapplicable. In those cases it returns
`DEFERRED`. If a bound is valid but too broad to prove separation, it also
returns `DEFERRED` after recording the inconclusive bound.

### Checker 3. Exact symbolic checker

This checker handles the logical cases that the linear and convex solvers do
not enumerate. It substitutes definitions, enumerates the required Boolean
assignments, converts each resulting arithmetic case to exact linear
constraints, and applies exact variable elimination. Unsupported expressions
or a remaining counterexample case produce `DEFERRED`.

### Checker 4. SMT checker

The final checker receives the same authoritative counterexample question and
all unresolved cases. It does not receive a weakened approximation merely to
make the query easier.

The SMT checker may accept a case only when the complete case has an exact
encoding in the solver's supported arithmetic. An unresolved continuous rate
equation cannot be replaced by a sampled equation. It first needs a checked
finite encoding or a checked reachable outer bound. Otherwise the case is
`DEFERRED` before the solver runs.

An unsatisfiable result becomes `CERTIFIED` only when the independent checker
can replay a supported proof made of exact algebraic contradictions, checked
case splits, and checked reachable bounds. Merely rerunning the same solver or
recording its `unsat` response is not independent proof checking. A
satisfiable result becomes `VIOLATION` only when the reported execution can be
replayed against the extracted equations and bounds. `unknown`, timeout,
unsupported arithmetic, missing proof output, unsupported proof step, or
failed replay produces `DEFERRED` and therefore the final result
`NOT_CERTIFIED`.

## Stage applicability records

Every attempted checker records its name, version, outcome, applicability
checks, reason for `DEFERRED`, and proof or replayed counterexample when one is
available. The linear and convex records also contain the accepted constraint
count, treatment of strict comparisons, and configured optimization timeout.
A backend timeout, crash, unsupported input, malformed output, or rejected
certificate is recorded as `DEFERRED` and the next checker is attempted.

## Certificate contents

A successful certificate contains:

1. Model, Markov decision process certificate, reduced specification, and
   timing input hashes.
2. The exact fixed `dt`, numeric interpretation, cycle order, and certified
   blind interval.
3. The sampled point premise and its checked source.
4. The reconstructed state and action interface inherited from the checked
   Markov decision process certificate.
5. The original safety property and translated unsafe condition.
6. The complete property specific equation set and within step meaning.
7. The endpoint relationship between physical evolution and the sampled
   transition.
8. The complete case coverage record, including guards, timing cases, and
   instantaneous changes.
9. Every successful checker's applicability proof.
10. Every calculation needed to establish that the reachable states cannot
   intersect the unsafe condition.
11. The independent checker version and result.

The final model result is `CERTIFIED` only if every requested safety property
has its own accepted certificate. The generated report lists every checker in
the order attempted, including all `DEFERRED` results.

## Trust boundary

The trusted code is the smallest code base whose correctness is assumed by the
certificate claim. It consists of the SysML expression parser and strict
translation rules, exact number parser, independent certificate checker, and
its exact arithmetic and outward rounded interval operations. Optimizers, SMT
solvers, neural training code, simulators, and orchestration code are not
trusted to assert safety.

The independent checker reparses the hashed SysML source, rebuilds the
property specific equations and case coverage record, and compares them with
the certificate before replaying the proof. It does not import optimization or
solver backend code. Sharing the source parser remains an explicit part of the
trust boundary rather than being described as independent verification of the
parser itself.

## Artifact integration

The new preprocessing stage belongs after the current Markov decision process
stage and before fitted training. It should read the freshly generated files
under `outputs/latest/03_markov_mdp/` rather than reconstructing or trusting an
unverified buffer independently.

The proposed generated directory is
`outputs/latest/04_discretization_safety/`. Fitted training would then become
Stage 5. The run level `dt` is retained as its original command line text,
validated once, and passed to the new stage in exact canonical form. Every
applicability record and certificate contains that form.

The new implementation belongs under
`bundle/architecture-fit/discretization/` with separate modules for common
extraction, timing analysis, exact checking, linear checking, convex checking,
interval checking, SMT checking, certificate construction, and independent
certificate checking. The checker imports only the trusted parsing and
arithmetic modules identified above.

The required changes to existing files are fixed as follows.

| Existing file | Required change |
|---|---|
| `bundle/sysml-models/runtime_settings.py` | Preserve the exact source representation of `dt` beside the validated simulator value. |
| `bundle/sysml-models/sysml_parser.py` | Retain numeric source text, explicit within step annotations, source locations, and ordered action phases. |
| `bundle/architecture-fit/certification/equations.py` | Represent numeric types, exact constants, physical rate equations, instantaneous changes, and cycle phases. |
| `bundle/architecture-fit/certification/strict_extract.py` | Extract the new information without accepting unresolved fallbacks. |
| `bundle/architecture-fit/certification/relevance.py` | Include timing, guard, physical rate, and endpoint dependencies in each property specific reduction. |
| `src/run_fitting_sequence.py` | Run the new Stage 4, enforce its exit result, and move fitted training to Stage 5. |
| `requirements.txt` | Add and pin SciPy for numerical linear optimization candidates. |
| `README.md` and `ARTIFACT_MANIFEST.md` | Document the new input rule, stage, outputs, and dependencies. |

Each property directory contains `cases.json`, one applicability record per
checker, every candidate proof or counterexample, the checked certificate when
one exists, and `result.json`. A final `NOT_CERTIFIED` result makes the new
stage and the complete artifact runner return a nonzero exit status before
fitted training begins.

## Implementation plan

### Phase 1. Freeze the claim and result formats

Define the counterexample question, blind interval record, three checker
outcomes, case coverage record, applicability record, certificate format,
trusted code boundary, and final `NOT_CERTIFIED` behavior. Add schema
validation before implementing numerical methods.

### Phase 2. Extend extraction without weakening strict behavior

Extend the existing equation representation with explicit numeric types,
exact numeric source text, state and action bounds, ordered cycle phases,
within step rules, guard sources, instantaneous changes, and timing sources.
Add an explicit model annotation selecting the within step meaning for every
physical update used by a property. Any unsupported SysML construct remains an
error with its source location. Do not replace unresolved content with a
dependency only approximation.

### Phase 3. Connect the Markov decision process evidence

Load and independently check the Stage 3 certificate. Reuse its transition
closed relevant state, reconstruction buffer, executed action meaning, model
hash, and reduced specification hash. Add the safety property specific
dependency reduction and verify its coverage. Build and discharge the sampled
point premise and endpoint relationship before permitting a complete safety
claim.

### Phase 4. Implement and check exact and linear certificates

Implement the recursive exact linearity classifier first. Test it independently
from the optimizer. Implement exact symbolic checking, exact affine trajectory
checking, certified linear continuous flow bounds, linear optimization, exact
rational certificate emission, and a separate certificate replay path.
SciPy supplies numerical linear optimization candidates. A candidate becomes a
certificate only after its coefficients are converted to exact fractions and
the independent checker verifies the resulting contradiction.

### Phase 5. Implement certified convex bounds

Add bounded curvature checks and checked upper and lower relation bounds. Add
certified rate integration and range subdivision only as refinements of a
proved outer bound. The stage must preserve the original counterexample
question and emit `DEFERRED` whenever any bound lacks proof.

### Phase 6. Implement interval and polynomial remainder checking

Add outward rounded interval arithmetic, time subdivision, polynomial
remainder bounds, guard crossing checks, and instantaneous change checks. Make
coverage of every generated region a certificate requirement.

### Phase 7. Add the SMT fallback

Compile unresolved counterexample cases into the supported SMT input language.
Record the exact query, solver settings, timeout, output, and proof or
counterexample artifact. Implement only proof steps the independent checker
can replay. Connect certification and violation claims only to successful
independent checking.

### Phase 8. Integrate the progression

Add the new stage to `src/run_fitting_sequence.py`, preserve the centralized
`dt`, and generate per model and per property summaries. The orchestrator must
invoke the next checker after every `DEFERRED` result. It must never infer certification
from process exit success alone. A final `NOT_CERTIFIED` result returns nonzero
and stops the runner before training.

### Phase 9. Build the validation battery

First add small positive models with hand checked results for held values,
constant rates, a scalar linear continuous rate, two coupled linear rates, a
convex quadratic rate, a guarded instantaneous change, and a time varying scan
schedule. Each positive test must verify both certificate generation and
independent replay.

The required tests deliberately present each checker with unsupported or
corrupted inputs. At minimum they cover variable multiplication, powers,
symbolic division, nonlinear equalities, hidden nonlinear definitions,
incomplete guard cases, instantaneous changes, missing bounds, unresolved
references, incorrect hashes, mismatched `dt`, rounded `dt`, invalid numeric
type assignments, sequential assignments incorrectly marked simultaneous,
missing within step meaning, missing sampled point evidence, invalid endpoint
relationships, observed quantities incorrectly replaced by physical
quantities, disappearing case identifiers, invalid convex bounds, unbounded
polynomial remainders, certificate mutations, solver timeout, solver
`unsat` without a replayable proof, and counterexamples that fail replay.

Each test verifies both that the current checker refuses certification and
that the next checker is invoked. A final timeout test verifies that the run
reports `NOT_CERTIFIED`.

## Current model readiness audit

The current models provide source equations, guards, scenario bounds,
controller contracts, executed action descriptions, fixed `dt` plumbing, and
`#ContinuousRate` annotations on the physical timestep assignments. The
preprocessing stage checks those annotations, reconstructs the corresponding
physical trajectories, and verifies that each trajectory at `dt` matches the
extracted next value.

The thermostat temperature update has a rate that is linear in temperature
after its declared constants and held heater outputs are resolved. It is a
candidate for the linear continuous flow path, but only after the continuous
rate meaning, action hold interval, cycle order, and sampled point premise are
checked.

Both cruise models contain the product of speed with itself in the drag term.
The linear checker must return `NOT_LINEAR` for that case every time. The case
then proceeds to the convex or interval checker, which must also account for
the coupled gap rate and the exact force action relation.

The mixing model contains constant rate tank updates selected by Boolean pump
and valve modes. It is a candidate for exact guarded linear checking. Its scan
condition requires exact schedule analysis, and its integer tank declarations
combined with real `dt` require the numeric type check before certification.
The model's observations are current when a scan occurs but expose only part
of the physical state. That information limitation remains represented by the
checked observation and action buffer rather than being converted into elapsed
sensor time.

## Design reevaluation

A single global derivative bound is cheaper but discards relationships among
state quantities and can fail to prove properties that the equations prove
directly. A margin first formulation has the same problem and does not
naturally represent every forbidden state and action combination. Simulation
cannot certify the absence of an execution. Starting with SMT ignores the
linear and convex structure that the artifact can extract. Using only convex
optimization is unsound when nonlinear equalities, guards, or unsupported
compositions remain.

The staged design avoids these failures because the counterexample question is
fixed once, each checker proves its own applicability, and no approximation
may exclude real behavior. Exact and linear cases receive inexpensive proofs.
More general cases retain certified bounds. SMT remains available without
making every model pay its full search cost. The result is deliberately
incomplete: inability to prove the claim is reported as `NOT_CERTIFIED`, never
as safety.

The current models appear likely to exercise the exact and linear stages
frequently because their extracted equations contain copies, bounded rates,
affine updates, and guarded assignments. That is an implementation hypothesis,
not a certification result. The classifier and certificate checker must make
the determination separately for every model and property.
