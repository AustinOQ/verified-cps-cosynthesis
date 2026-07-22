#!/usr/bin/env python3
"""
Reconstructibility closure (dynamics-aware) -- SysML-native.

Reads the transition dependency graph DIRECTLY from the SysML model (sysml_deps,
which uses the same parser the shield is built from). NO SMV involved, so the
controller-to-actuator wire is read from the SysML connections and actions.

Given current observations, past observations, and past actions, the program
checks whether the extracted equations reconstruct the transition-closed
relevant state.

Propagation over time-indexed nodes (var, tau<=0):
  forward   : v@(tau+1) known if all nsupp(v)@tau known
  inversion : a [copy] var (nsupp = single source) is invertible -> source@tau
              known if v@(tau+1) known  (this is the sensor identity: the reading
              at t+1 IS the true state at t)
Cost: linear in (#vars * horizon).
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certification.equations import Var
from certification.equation_reconstruct import (
    deterministic_state_variables,
    sampled_memory_rules,
)
from certification.relevance import compute_transition_closed_relevance, equation_refs
from certification.strict_extract import extract_equation_model


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
        sampled_memories, assumptions = sampled_memory_rules(eq_model, dt)

    schedule_state = sorted(deterministic_state_variables(eq_model) | {
        rule["schedule_state"] for rule in sampled_memories
        if rule.get("schedule_state") in eq_model.state
    })
    if schedule_state:
        assumptions.insert(
            0,
            "deterministic_schedule_known: verifier treats these state values as "
            f"known schedule counters because their equations depend only on themselves: "
            f"{schedule_state}",
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
        known_schedule_state=schedule_state,
        target_label="TRANSITION-CLOSED Q",
        closure_label="STRICT Q CLOSURE",
    )


def reconstruct(model, b_obs, b_act, horizon=8, target=None):
    STATE = model["STATE"]; nsupp = model["nsupp"]; copies = model["copies"]
    OBS = model["OBS"]; ACT = model["ACTIONS"]
    sampled_memories = model.get("sampled_memories", [])
    if target is None:
        target = STATE
    schedule_state = set(model.get("known_schedule_state", []))
    taus = range(-horizon, 1)
    known = set()
    for tau in taus:
        for v in schedule_state: known.add((v, tau))
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


def sweep(path, max_obs=2, max_act=4, dt=0.1, enable_sampled_memory=True):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="*", help="SysML file path")
    ap.add_argument("--max-obs", type=int, default=2)
    ap.add_argument("--max-act", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--no-sampled-memory", action="store_true",
                    help="disable bounded sampled-memory reconstruction rules")
    ns = ap.parse_args()

    if not ns.model:
        raise SystemExit("provide one or more SysML file paths")
    items = ns.model
    for item in items:
        sweep(
            item,
            ns.max_obs,
            ns.max_act,
            dt=ns.dt,
            enable_sampled_memory=not ns.no_sampled_memory,
        )
