"""Equation-IR reconstructibility proof trace.

This is the sound syntactic fast path over ``EquationModel``. It deliberately
allows only direct-copy inversions unless a guarded copy has a guard proof. The
first implementation has no general guard solver, so non-constant guards are
recorded as blocked unsafe inversions instead of being used.
"""

from __future__ import annotations

import math
from typing import Any

from .equations import Const, Equation, EquationModel, Expr, Ite, Op, Var
from .relevance import equation_refs


def fact_key(var: str, tau: int) -> str:
    return f"{var}@{tau}"


def _add_fact(
    facts: dict[str, dict[str, Any]],
    var: str,
    tau: int,
    rule: str,
    premises: list[str] | None = None,
    detail: dict[str, Any] | None = None,
) -> bool:
    key = fact_key(var, tau)
    if key in facts:
        return False
    facts[key] = {
        "key": key,
        "var": var,
        "tau": tau,
        "rule": rule,
        "premises": premises or [],
        "detail": detail or {},
    }
    return True


def _copy_source(model: EquationModel, expr: Expr, seen: set[str] | None = None) -> str | None:
    """Return the copied source variable for direct copies through definitions."""

    if not isinstance(expr, Var):
        return None
    if expr.name not in model.definitions:
        return expr.name
    seen = set(seen or set())
    if expr.name in seen:
        return None
    seen.add(expr.name)
    return _copy_source(model, model.definitions[expr.name].expr, seen)


def _const_number(model: EquationModel, expr: Expr) -> float | None:
    if isinstance(expr, Const) and isinstance(expr.value, (int, float)):
        return float(expr.value)
    if isinstance(expr, Var):
        definition = model.definitions.get(expr.name)
        if definition is not None:
            return _const_number(model, definition.expr)
    return None


def _scan_delay_steps(model: EquationModel, cond: Expr, dt: float | None) -> int | None:
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
        numerator = _const_number(model, rhs.args[0])
        denominator = _const_number(model, rhs.args[1])
        if numerator is not None and denominator not in (None, 0):
            period = numerator / denominator
    else:
        period = _const_number(model, rhs)

    if period is None or period <= 0:
        return None
    return max(1, int(math.ceil((period / dt) - 1e-12)) + 1)


def sampled_memory_rules(
    model: EquationModel,
    dt: float | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rules: list[dict[str, Any]] = []
    assumptions: list[str] = []

    for target, eq in sorted(model.transitions.items()):
        expr = eq.expr
        if not isinstance(expr, Ite):
            continue
        if expr.else_expr != Var(target):
            continue
        source = _copy_source(model, expr.then_expr)
        if source not in model.state:
            continue
        delay = _scan_delay_steps(model, expr.cond, dt)
        if delay is None:
            continue
        rules.append(
            {
                "target": target,
                "source": source,
                "max_delay": delay,
                "equation_pretty": eq.pretty(),
                "equation_source": eq.source,
            }
        )
        assumptions.append(
            "sampled_memory_bound: "
            f"{target} equals a {source} sample no older than {delay} step(s), "
            f"derived from scan guard under dt={dt}; certificate state is after "
            "the environment's reset/warmup scan opportunities"
        )

    return rules, assumptions


def _known_support(
    model: EquationModel,
    eq: Equation,
    facts: dict[str, dict[str, Any]],
    tau: int,
) -> tuple[bool, list[str], list[str]]:
    refs = equation_refs(model, eq)
    support = sorted(ref for ref in refs if ref in model.state or ref in model.actions)
    unknown = sorted(
        ref for ref in refs
        if ref not in model.state
        and ref not in model.actions
        and ref not in model.definitions
        and ref not in model.constants
    )
    if eq.raw_refs() - model.constants:
        unknown.extend(sorted(eq.raw_refs() - model.constants))
    premises = [fact_key(ref, tau) for ref in support]
    return not unknown and all(p in facts for p in premises), support, unknown


def _guard_known_true(_model: EquationModel, cond: Expr, _facts: dict[str, Any], _tau: int) -> bool:
    """Sound but intentionally tiny guard proof.

    Knowing all variables in a guard is not the same as knowing the guard is
    true, so only literal true is accepted until a solver/domain proof exists.
    """

    return isinstance(cond, Const) and cond.value is True


def _equation_detail(eq: Equation, support: list[str] | None = None) -> dict[str, Any]:
    detail = {
        "equation_target": eq.target,
        "equation_kind": eq.kind,
        "equation_source": eq.source,
        "equation_pretty": eq.pretty(),
    }
    if support is not None:
        detail["support"] = support
    return detail


def equation_reconstruction_trace(
    model: EquationModel,
    b_obs: int,
    b_act: int,
    *,
    horizon: int = 14,
    target: set[str] | None = None,
    dt: float | None = 0.1,
    enable_sampled_memory: bool = True,
) -> dict[str, Any]:
    """Replay equation-IR reconstructibility and record every derived fact."""

    state = set(model.state)
    actions = set(model.actions)
    target = set(target or state)
    time_vars = {
        v for v in state if "currenttime" in v.lower() or "timeseconds" in v.lower()
    }
    taus = range(-horizon, 1)
    facts: dict[str, dict[str, Any]] = {}
    blocked_inversions: dict[tuple[str, int, str], dict[str, Any]] = {}
    observation_sources: dict[str, str] = {}

    for tau in taus:
        for var in sorted(time_vars):
            _add_fact(facts, var, tau, "deterministic_time_known")

    for obs_name, eq in sorted(model.observations.items()):
        source = _copy_source(model, eq.expr)
        if source in state:
            observation_sources[obs_name] = source
            for tau in range(-b_obs, 1):
                _add_fact(
                    facts,
                    source,
                    tau,
                    "equation_observation_direct_copy",
                    detail={
                        **_equation_detail(eq),
                        "observation": obs_name,
                        "source": source,
                    },
                )

    for var in sorted(actions):
        for tau in range(-b_act, 0):
            _add_fact(facts, var, tau, "executed_action_history")

    memory_rules, memory_assumptions = ([], [])
    if enable_sampled_memory:
        memory_rules, memory_assumptions = sampled_memory_rules(model, dt)

    changed = True
    while changed:
        changed = False
        for target_var, eq in sorted(model.transitions.items()):
            for tau in taus:
                next_tau = tau + 1
                if next_tau in taus and fact_key(target_var, next_tau) not in facts:
                    available, support, unknown = _known_support(model, eq, facts, tau)
                    if available:
                        changed |= _add_fact(
                            facts,
                            target_var,
                            next_tau,
                            "equation_forward_transition",
                            [fact_key(ref, tau) for ref in support],
                            {
                                **_equation_detail(eq, support),
                                "from_tau": tau,
                                "unknown_refs": unknown,
                            },
                        )

                direct_source = _copy_source(model, eq.expr)
                if direct_source in state:
                    premise = fact_key(target_var, next_tau)
                    if premise in facts and fact_key(direct_source, tau) not in facts:
                        changed |= _add_fact(
                            facts,
                            direct_source,
                            tau,
                            "equation_direct_copy_inversion",
                            [premise],
                            {
                                **_equation_detail(eq),
                                "source": direct_source,
                                "known_transition_tau": next_tau,
                                "inversion": "direct_transition_copy",
                            },
                        )
                    continue

                expr = eq.expr
                if (
                    isinstance(expr, Ite)
                    and expr.else_expr == Var(target_var)
                    and fact_key(target_var, next_tau) in facts
                ):
                    guarded_source = _copy_source(model, expr.then_expr)
                    if guarded_source not in state:
                        continue
                    if _guard_known_true(model, expr.cond, facts, tau):
                        if fact_key(guarded_source, tau) not in facts:
                            changed |= _add_fact(
                                facts,
                                guarded_source,
                                tau,
                                "equation_guarded_copy_inversion",
                                [fact_key(target_var, next_tau)],
                                {
                                    **_equation_detail(eq),
                                    "source": guarded_source,
                                    "known_transition_tau": next_tau,
                                    "guard_status": "proven_true",
                                    "guard_pretty": expr.cond.pretty(),
                                    "inversion": "guarded_transition_copy",
                                },
                            )
                    else:
                        key = (target_var, tau, guarded_source)
                        blocked_inversions[key] = {
                            **_equation_detail(eq),
                            "target": target_var,
                            "source": guarded_source,
                            "tau": tau,
                            "known_transition_tau": next_tau,
                            "guard_status": "not_proven_true",
                            "guard_pretty": expr.cond.pretty(),
                            "reason": (
                                "guarded copy inversion is unsupported unless the guard "
                                "is reconstructed and proven true"
                            ),
                        }

        for rule in memory_rules:
            target_var = rule["target"]
            source = rule["source"]
            delay = int(rule["max_delay"])
            for tau in taus:
                if fact_key(target_var, tau) in facts:
                    continue
                start = tau - delay
                if start < min(taus):
                    continue
                premises = [fact_key(source, src_tau) for src_tau in range(start, tau)]
                if all(p in facts for p in premises):
                    changed |= _add_fact(
                        facts,
                        target_var,
                        tau,
                        "sampled_memory_bound",
                        premises,
                        {
                            "target": target_var,
                            "source": source,
                            "max_delay": delay,
                            "tau": tau,
                            "equation_pretty": rule["equation_pretty"],
                            "equation_source": rule["equation_source"],
                        },
                    )

    missing = sorted(var for var in target if fact_key(var, 0) not in facts)
    return {
        "proof_engine": "equation_ir_syntactic_v1",
        "horizon": horizon,
        "b_obs": b_obs,
        "b_act": b_act,
        "observation_sources": observation_sources,
        "sampled_memory_assumptions": memory_assumptions,
        "facts": list(facts.values()),
        "target_facts": [fact_key(var, 0) for var in sorted(target)],
        "missing_target_facts": [fact_key(var, 0) for var in missing],
        "blocked_unsafe_inversions": [
            blocked_inversions[key] for key in sorted(blocked_inversions)
        ],
        "passes": not missing,
    }


def equation_search(
    model: EquationModel,
    *,
    max_obs: int,
    max_act: int,
    horizon: int,
    target: set[str],
    dt: float | None = 0.1,
    enable_sampled_memory: bool = True,
) -> tuple[tuple[int, int] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    selected: tuple[int, int] | None = None
    for b_obs in range(max_obs + 1):
        for b_act in range(max_act + 1):
            trace = equation_reconstruction_trace(
                model,
                b_obs,
                b_act,
                horizon=horizon,
                target=target,
                dt=dt,
                enable_sampled_memory=enable_sampled_memory,
            )
            missing = [item.rsplit("@", 1)[0] for item in trace["missing_target_facts"]]
            attempts.append(
                {
                    "b_obs": b_obs,
                    "b_act": b_act,
                    "passes": trace["passes"],
                    "missing": sorted(missing),
                    "blocked_unsafe_inversions": len(trace["blocked_unsafe_inversions"]),
                }
            )
            if trace["passes"] and selected is None:
                selected = (b_obs, b_act)
    return selected, attempts
