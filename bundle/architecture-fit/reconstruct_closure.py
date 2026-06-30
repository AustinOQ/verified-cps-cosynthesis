#!/usr/bin/env python3
"""
Reconstructibility closure (dynamics-aware) -- SysML-native.

Reads the transition dependency graph DIRECTLY from the SysML model (sysml_deps,
which uses the same parser the shield is built from). NO SMV involved, so the
controller->actuator wire is present natively (no manual coupling patch).

Default question: given (current obs + last b_obs observations + last b_act
actions), can the dependency closure reconstruct the strict extractor's
transition-closed relevant state q? Use --legacy to reproduce the historical
R union OBS target and sysml_deps dependency graph.

Propagation over time-indexed nodes (var, tau<=0):
  forward   : v@(tau+1) known if all nsupp(v)@tau known
  inversion : a [copy] var (nsupp = single source) is invertible -> source@tau
              known if v@(tau+1) known  (this is the sensor identity: the reading
              at t+1 IS the true state at t)
Cost: linear in (#vars * horizon).
"""
import argparse
import math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certification.equations import Const, Ite, Op, Var
from certification.relevance import compute_transition_closed_relevance, equation_refs
from certification.strict_extract import extract_equation_model
from sysml_deps import get_model


def _copy_source(eq_model, expr, seen=None):
    if not isinstance(expr, Var):
        return None
    if expr.name not in eq_model.definitions:
        return expr.name
    seen = set(seen or set())
    if expr.name in seen:
        return None
    seen.add(expr.name)
    return _copy_source(eq_model, eq_model.definitions[expr.name].expr, seen)


def _const_number(eq_model, expr):
    if isinstance(expr, Const) and isinstance(expr.value, (int, float)):
        return float(expr.value)
    if isinstance(expr, Var):
        definition = eq_model.definitions.get(expr.name)
        if definition is not None:
            return _const_number(eq_model, definition.expr)
    return None


def _scan_delay_steps(eq_model, cond, dt):
    if dt is None or dt <= 0:
        return None
    if not isinstance(cond, Op) or cond.op != ">=" or len(cond.args) != 2:
        return None

    lhs, rhs = cond.args
    if not isinstance(lhs, Op) or lhs.op != "-" or len(lhs.args) != 2:
        return None
    if not all(isinstance(arg, Var) for arg in lhs.args):
        return None
    current, last = lhs.args
    if "time" not in current.name.lower() or "last" not in last.name.lower():
        return None

    period = None
    if isinstance(rhs, Op) and rhs.op == "/" and len(rhs.args) == 2:
        numerator = _const_number(eq_model, rhs.args[0])
        denominator = _const_number(eq_model, rhs.args[1])
        if numerator is not None and denominator not in (None, 0):
            period = numerator / denominator
    else:
        period = _const_number(eq_model, rhs)

    if period is None or period <= 0:
        return None
    # The simulator evaluates these guards in Python floats. When period/dt is
    # mathematically integral, roundoff can still make an equality-step guard
    # miss once; the next step then necessarily exceeds the threshold. Add one
    # step as a conservative implementation-semantics margin.
    return max(1, int(math.ceil((period / dt) - 1e-12)) + 1)


def _sampled_memory_rules(eq_model, dt):
    rules = []
    assumptions = []

    for target, eq in sorted(eq_model.transitions.items()):
        expr = eq.expr
        if not isinstance(expr, Ite):
            continue
        if expr.else_expr != Var(target):
            continue

        source = _copy_source(eq_model, expr.then_expr)
        if source not in eq_model.state:
            continue

        delay = _scan_delay_steps(eq_model, expr.cond, dt)
        if delay is None:
            continue

        rules.append({"target": target, "source": source, "max_delay": delay})
        assumptions.append(
            "sampled_memory_bound: "
            f"{target} equals a {source} sample no older than {delay} step(s), "
            f"derived from scan guard under dt={dt}; certificate state is after "
            "the environment's reset/warmup scan opportunities"
        )

    return rules, assumptions


def get_strict_model(path, dt=0.1, enable_sampled_memory=True):
    eq_model = extract_equation_model(path)
    relevance = compute_transition_closed_relevance(eq_model)
    nsupp = {}
    copies = set()

    for target, eq in eq_model.transitions.items():
        refs = equation_refs(eq_model, eq)
        nsupp[target] = {
            ref for ref in refs if ref in eq_model.state or ref in eq_model.actions
        }
        copy_source = _copy_source(eq_model, eq.expr)
        if copy_source in eq_model.state:
            copies.add(target)

    obs = set()
    for eq in eq_model.observations.values():
        obs |= {ref for ref in equation_refs(eq_model, eq) if ref in eq_model.state}

    sampled_memories, assumptions = ([], [])
    if enable_sampled_memory:
        sampled_memories, assumptions = _sampled_memory_rules(eq_model, dt)

    time_vars = sorted(
        v for v in eq_model.state if "currenttime" in v.lower() or "timeseconds" in v.lower()
    )
    if time_vars:
        assumptions.insert(
            0,
            "deterministic_time_known: verifier treats these time variables as "
            f"certificate-known counters: {time_vars}",
        )

    return dict(
        STATE=eq_model.state,
        ACTIONS=eq_model.actions,
        nsupp=nsupp,
        copies=copies,
        OBS=obs,
        R=relevance.q,
        diagnostics=eq_model.diagnostics,
        assumptions=assumptions,
        sampled_memories=sampled_memories,
        target_label="TRANSITION-CLOSED Q",
        closure_label="STRICT Q CLOSURE",
    )


def reconstruct(model, b_obs, b_act, horizon=8, target=None):
    STATE = model["STATE"]; nsupp = model["nsupp"]; copies = model["copies"]
    OBS = model["OBS"]; ACT = model["ACTIONS"]
    sampled_memories = model.get("sampled_memories", [])
    if target is None:
        target = STATE
    TIME = {v for v in STATE if "currenttime" in v.lower() or "timeseconds" in v.lower()}
    taus = range(-horizon, 1)
    known = set()
    for tau in taus:
        for v in TIME: known.add((v, tau))          # deterministic step counter
    for v in OBS:
        for tau in range(-b_obs, 1): known.add((v, tau))   # current + b_obs past obs
    for v in ACT:
        for tau in range(-b_act, 0): known.add((v, tau))   # last b_act actions
    changed = True
    while changed:
        changed = False
        for v in STATE:
            supp = nsupp.get(v, set())
            for tau in taus:
                if tau + 1 in taus and (v, tau + 1) not in known and \
                        all((u, tau) in known for u in supp):
                    known.add((v, tau + 1)); changed = True
                if v in copies and supp:
                    u = next(iter(supp))
                    if (v, tau + 1) in known and (u, tau) not in known:
                        known.add((u, tau)); changed = True
        for rule in sampled_memories:
            target_var = rule["target"]
            source = rule["source"]
            delay = rule["max_delay"]
            for tau in taus:
                if (target_var, tau) in known:
                    continue
                start = tau - delay
                if start < min(taus):
                    continue
                if all((source, src_tau) in known for src_tau in range(start, tau)):
                    known.add((target_var, tau)); changed = True
    missing = sorted(v for v in target if (v, 0) not in known)
    return missing, sorted(STATE)


def sweep(path, max_obs=2, max_act=4, legacy=False, dt=0.1, enable_sampled_memory=True):
    if legacy:
        model = get_model(path)
        # Historical target = what the policy/shield/reward directly read
        # (requirement-relevant vars R + the observation).
        relevant = set(model.get("R", set())) | set(model["OBS"])
        model["target_label"] = "LEGACY R union OBS"
        model["closure_label"] = "LEGACY CLOSURE"
    else:
        model = get_strict_model(path, dt=dt, enable_sampled_memory=enable_sampled_memory)
        relevant = set(model.get("R", set()))

    if not relevant:
        relevant = set(model["STATE"])

    print(f"\n{path}")
    print(f"  ACTIONS ={sorted(model['ACTIONS'])}")
    print(f"  OBS     ={sorted(model['OBS'])}")
    print(f"  {model['target_label']}={sorted(relevant)}")
    irrelevant = sorted(set(model["STATE"]) - relevant)
    if irrelevant:
        print(f"  (ignored by this target): {irrelevant}")
    blocking = [
        d for d in model.get("diagnostics", [])
        if getattr(d, "severity", "") in {"warning", "error"}
    ]
    if blocking:
        print("  extractor diagnostics relevant to certification:")
        for diag in blocking:
            print(f"    - {diag.pretty()}")
    if model.get("assumptions"):
        print("  certificate assumptions:")
        for assumption in model["assumptions"]:
            print(f"    - {assumption}")
    print(f"  {'b_obs':>5} {'b_act':>5} | missing target state@now")
    best = None
    for b_obs in range(0, max_obs + 1):
        for b_act in range(0, max_act + 1):
            missing, STATE = reconstruct(model, b_obs, b_act, target=relevant)
            tag = ""
            if not missing and best is None:
                best = (b_obs, b_act)
                tag = f"   <== {model['closure_label']} (first in bounded scan)"
            shown = missing if missing else "NONE - relevant state reconstructed"
            print(f"  {b_obs:>5} {b_act:>5} | {shown}{tag}")
    if best:
        print(
            f"  => first {model['closure_label'].lower()} in scan: "
            f"last {best[1]} actions + current + {best[0]} past obs"
        )
    else:
        print("  => no closure within swept depth")
    return best


_SM = os.path.join(os.path.dirname(__file__), "..", "sysml-models")
MODELS = {
    "thermostat": f"{_SM}/thermostat/model.sysml",
    "cruise-continuous": f"{_SM}/cruise-continuous-model/model.sysml",
    "cruise-discrete": f"{_SM}/cruise-controller-model/model.sysml",
    "mixing": f"{_SM}/mixing-sysml-model/model.sysml",
}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="model key or path")
    ap.add_argument("--legacy", action="store_true", help="use the historical sysml_deps target")
    ap.add_argument("--max-obs", type=int, default=2)
    ap.add_argument("--max-act", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--no-sampled-memory", action="store_true",
                    help="disable bounded sampled-memory reconstruction rules")
    ns = ap.parse_args()

    items = ns.model or ["thermostat", "cruise-continuous", "cruise-discrete", "mixing"]
    for item in items:
        sweep(
            MODELS.get(item, item),
            ns.max_obs,
            ns.max_act,
            legacy=ns.legacy,
            dt=ns.dt,
            enable_sampled_memory=not ns.no_sampled_memory,
        )
