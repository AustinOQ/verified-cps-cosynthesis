"""Advisory solver-backed checks for equation models.

The first solver-backed query is intentionally narrow: one-step transition
closure. It asks whether two runs with equal current q and equal executed action
can disagree on next q or next externally visible equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .equations import Const, Equation, EquationModel, Expr, Ite, Op, RawRef, Var

try:  # pragma: no cover - exercised by integration tests when z3 is installed
    import z3  # type: ignore
except Exception:  # pragma: no cover - environment-dependent
    z3 = None


BOOL_OPS = {"and", "or", "implies"}
COMPARE_OPS = {"==", ">", "<", ">=", "<="}
ARITH_OPS = {"+", "-", "*", "/"}
MAX_SOLVER_POLYNOMIAL_DEGREE = 5


@dataclass
class TypeEnv:
    sorts: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def require(self, name: str, sort: str) -> None:
        prev = self.sorts.get(name)
        if prev is None:
            self.sorts[name] = sort
        elif prev != sort:
            self.conflicts.append(f"{name}: {prev} vs {sort}")


def _symbol_name(run: int | None, role: str, name: str) -> str:
    if run is None:
        return f"const::{name}"
    return f"r{run}::{role}::{name}"


def _var_symbol(
    symbols: dict[tuple[int | None, str, str], Any],
    sort: str,
    run: int | None,
    role: str,
    name: str,
):
    key = (run, role, name)
    if key not in symbols:
        z3_name = _symbol_name(run, role, name)
        if sort == "Bool":
            symbols[key] = z3.Bool(z3_name)
        elif sort == "Int":
            symbols[key] = z3.Int(z3_name)
        else:
            symbols[key] = z3.Real(z3_name)
    return symbols[key]


def _merge_sort(a: str, b: str) -> str:
    if a == b:
        return a
    return "Unknown"


def _expr_sort(expr: Expr, env: TypeEnv, expected: str | None = None) -> str:
    if isinstance(expr, Const):
        if isinstance(expr.value, bool):
            return "Bool"
        if isinstance(expr.value, (int, float)):
            return "Real"
        return "Unknown"
    if isinstance(expr, RawRef):
        env.require(expr.path, expected or env.sorts.get(expr.path, "Real"))
        return env.sorts.get(expr.path, expected or "Real")
    if isinstance(expr, Var):
        env.require(expr.name, expected or env.sorts.get(expr.name, "Real"))
        return env.sorts.get(expr.name, expected or "Real")
    if isinstance(expr, Ite):
        _expr_sort(expr.cond, env, "Bool")
        then_sort = _expr_sort(expr.then_expr, env, expected)
        else_sort = _expr_sort(expr.else_expr, env, expected)
        return _merge_sort(then_sort, else_sort)
    if isinstance(expr, Op):
        if expr.op == "not":
            for arg in expr.args:
                _expr_sort(arg, env, "Bool")
            return "Bool"
        if expr.op in BOOL_OPS:
            for arg in expr.args:
                _expr_sort(arg, env, "Bool")
            return "Bool"
        if expr.op in {">", "<", ">=", "<="}:
            for arg in expr.args:
                _expr_sort(arg, env, "Real")
            return "Bool"
        if expr.op == "==":
            if expected == "Bool":
                for arg in expr.args:
                    _expr_sort(arg, env, "Bool")
            else:
                left_sort = _expr_sort(expr.args[0], env, expected)
                right_expected = left_sort if left_sort != "Unknown" else expected
                _expr_sort(expr.args[1], env, right_expected)
            return "Bool"
        if expr.op in ARITH_OPS:
            for arg in expr.args:
                _expr_sort(arg, env, "Real")
            return "Real"
    return "Unknown"


def infer_sorts(model: EquationModel) -> tuple[dict[str, str], list[str]]:
    env = TypeEnv()
    for name in model.constants:
        env.require(name, "Real")
    for eq in model.terminals.values():
        _expr_sort(eq.expr, env, "Bool")
    for eq in model.requirements.values():
        _expr_sort(eq.expr, env, "Bool")
    for name, eq in model.definitions.items():
        _expr_sort(eq.expr, env, env.sorts.get(name))
    for name, eq in model.observations.items():
        _expr_sort(eq.expr, env, env.sorts.get(name))
    for name, eq in model.transitions.items():
        _expr_sort(eq.expr, env, env.sorts.get(name))
    for name in model.state | model.actions | model.constants:
        if name not in env.sorts:
            env.require(name, "Real")
    return env.sorts, env.conflicts


def _dynamic_refs(
    expr: Expr,
    model: EquationModel,
    seen: set[str] | None = None,
) -> set[str]:
    seen = set(seen or set())
    if isinstance(expr, Const):
        return set()
    if isinstance(expr, RawRef):
        return set()
    if isinstance(expr, Var):
        if expr.name in model.state or expr.name in model.actions:
            return {expr.name}
        if expr.name in model.definitions and expr.name not in seen:
            seen.add(expr.name)
            return _dynamic_refs(model.definitions[expr.name].expr, model, seen)
        return set()
    if isinstance(expr, Ite):
        return (
            _dynamic_refs(expr.cond, model, seen)
            | _dynamic_refs(expr.then_expr, model, seen)
            | _dynamic_refs(expr.else_expr, model, seen)
        )
    if isinstance(expr, Op):
        out: set[str] = set()
        for arg in expr.args:
            out |= _dynamic_refs(arg, model, seen)
        return out
    return set()


def _is_const_like(expr: Expr, model: EquationModel) -> bool:
    return not _dynamic_refs(expr, model)


def _polynomial_degree(
    expr: Expr,
    model: EquationModel,
    seen: set[str] | None = None,
) -> tuple[int | None, list[str]]:
    """Return polynomial degree over state/action vars, or None if unsupported."""

    seen = set(seen or set())
    if isinstance(expr, Const):
        return 0, []
    if isinstance(expr, RawRef):
        if expr.path in model.constants:
            return 0, []
        return None, [f"unresolved raw ref {expr.path}"]
    if isinstance(expr, Var):
        if expr.name in model.state or expr.name in model.actions:
            return 1, []
        if expr.name in model.constants:
            return 0, []
        if expr.name in model.definitions:
            if expr.name in seen:
                return None, [f"cyclic definition reference {expr.name}"]
            seen.add(expr.name)
            return _polynomial_degree(model.definitions[expr.name].expr, model, seen)
        return 0, []
    if isinstance(expr, Ite):
        cond_degree, cond_reasons = _polynomial_degree(expr.cond, model, seen)
        then_degree, then_reasons = _polynomial_degree(expr.then_expr, model, seen)
        else_degree, else_reasons = _polynomial_degree(expr.else_expr, model, seen)
        reasons = cond_reasons + then_reasons + else_reasons
        if cond_degree is None or then_degree is None or else_degree is None:
            return None, reasons
        return max(then_degree, else_degree), reasons
    if isinstance(expr, Op):
        if expr.op == "legacy_dep":
            return None, ["legacy dependency expression"]
        if expr.op not in BOOL_OPS | COMPARE_OPS | ARITH_OPS | {"not"}:
            return None, [f"unsupported op {expr.op}"]
        child_degrees: list[int] = []
        reasons: list[str] = []
        for arg in expr.args:
            degree, child_reasons = _polynomial_degree(arg, model, seen)
            reasons.extend(child_reasons)
            if degree is None:
                return None, reasons
            child_degrees.append(degree)
        if expr.op in {"not"} | BOOL_OPS | COMPARE_OPS:
            return max(child_degrees, default=0), reasons
        if expr.op in {"+", "-"}:
            return max(child_degrees, default=0), reasons
        if expr.op == "*":
            if len(child_degrees) != 2:
                return None, reasons + ["unsupported n-ary multiplication"]
            return child_degrees[0] + child_degrees[1], reasons
        if expr.op == "/":
            if len(child_degrees) != 2:
                return None, reasons + ["unsupported n-ary division"]
            if child_degrees[1] != 0:
                return None, reasons + ["division by symbolic expression"]
            return child_degrees[0], reasons
    return None, [f"unsupported expression {type(expr).__name__}"]


def _unsupported_expr(expr: Expr, model: EquationModel, path: str = "") -> list[str]:
    here = path or expr.pretty()
    degree, reasons = _polynomial_degree(expr, model)
    out = [f"{reason} in {here}" for reason in reasons]
    if degree is None:
        if not out:
            out.append(f"unsupported expression in {here}")
        return out
    if degree > MAX_SOLVER_POLYNOMIAL_DEGREE:
        out.append(
            f"polynomial degree {degree} exceeds supported degree "
            f"{MAX_SOLVER_POLYNOMIAL_DEGREE} polynomial fragment in {here}"
        )
    return out


def _max_polynomial_degree(model: EquationModel, q: set[str]) -> int:
    visible = (
        [model.transitions[name] for name in sorted(q) if name in model.transitions]
        + list(model.observations.values())
        + list(model.terminals.values())
        + list(model.requirements.values())
    )
    degrees = []
    for eq in visible:
        degree, _reasons = _polynomial_degree(eq.expr, model)
        if degree is not None:
            degrees.append(degree)
    return max(degrees, default=0)


def unsupported_reasons(model: EquationModel, q: set[str]) -> list[str]:
    reasons: list[str] = []
    visible = (
        [model.transitions[name] for name in sorted(q) if name in model.transitions]
        + list(model.observations.values())
        + list(model.terminals.values())
        + list(model.requirements.values())
    )
    for eq in visible:
        reasons.extend(_unsupported_expr(eq.expr, model, eq.target))
    return sorted(set(reasons))


class Encoder:
    def __init__(self, model: EquationModel, sorts: dict[str, str]):
        self.model = model
        self.sorts = sorts
        self.symbols: dict[tuple[int | None, str, str], Any] = {}
        self.next_cache: dict[tuple[int, str], Any] = {}
        self.definition_cache: dict[tuple[int, str, str], Any] = {}

    def sort_of(self, name: str) -> str:
        return self.sorts.get(name, "Real")

    def current_var(self, run: int, name: str):
        return _var_symbol(self.symbols, self.sort_of(name), run, "cur", name)

    def action_var(self, run: int, name: str):
        return _var_symbol(self.symbols, self.sort_of(name), run, "act", name)

    def const_var(self, name: str):
        return _var_symbol(self.symbols, self.sort_of(name), None, "const", name)

    def next_var(self, run: int, name: str):
        key = (run, name)
        if key not in self.next_cache:
            eq = self.model.transitions.get(name)
            if eq is None:
                self.next_cache[key] = self.current_var(run, name)
            else:
                self.next_cache[key] = self.encode_expr(eq.expr, run, "current")
        return self.next_cache[key]

    def definition(self, run: int, name: str, role: str):
        key = (run, role, name)
        if key not in self.definition_cache:
            eq = self.model.definitions[name]
            self.definition_cache[key] = self.encode_expr(eq.expr, run, role)
        return self.definition_cache[key]

    def encode_expr(self, expr: Expr, run: int, role: str):
        if isinstance(expr, Const):
            if isinstance(expr.value, bool):
                return z3.BoolVal(expr.value)
            if isinstance(expr.value, int):
                return z3.RealVal(expr.value)
            if isinstance(expr.value, float):
                return z3.RealVal(str(expr.value))
            if isinstance(expr.value, str):
                return z3.RealVal(expr.value)
            raise ValueError(f"unsupported constant {expr.value!r}")
        if isinstance(expr, RawRef):
            return self.const_var(expr.path)
        if isinstance(expr, Var):
            if expr.name in self.model.definitions:
                return self.definition(run, expr.name, role)
            if expr.name in self.model.actions:
                return self.action_var(run, expr.name)
            if expr.name in self.model.constants:
                return self.const_var(expr.name)
            if role == "next":
                return self.next_var(run, expr.name)
            return self.current_var(run, expr.name)
        if isinstance(expr, Ite):
            return z3.If(
                self.encode_expr(expr.cond, run, role),
                self.encode_expr(expr.then_expr, run, role),
                self.encode_expr(expr.else_expr, run, role),
            )
        if isinstance(expr, Op):
            args = [self.encode_expr(arg, run, role) for arg in expr.args]
            if expr.op == "not":
                return z3.Not(args[0])
            if expr.op == "and":
                return z3.And(*args)
            if expr.op == "or":
                return z3.Or(*args)
            if expr.op == "implies":
                return z3.Implies(args[0], args[1])
            if expr.op == "==":
                return args[0] == args[1]
            if expr.op == ">":
                return args[0] > args[1]
            if expr.op == "<":
                return args[0] < args[1]
            if expr.op == ">=":
                return args[0] >= args[1]
            if expr.op == "<=":
                return args[0] <= args[1]
            if expr.op == "+":
                out = args[0]
                for arg in args[1:]:
                    out = out + arg
                return out
            if expr.op == "-":
                if len(args) == 1:
                    return -args[0]
                out = args[0]
                for arg in args[1:]:
                    out = out - arg
                return out
            if expr.op == "*":
                return args[0] * args[1]
            if expr.op == "/":
                return args[0] / args[1]
        raise ValueError(f"unsupported expression {expr.pretty()}")


def _neq(a, b):
    return a != b


def _model_value(solver_model, expr) -> str:
    val = solver_model.eval(expr, model_completion=True)
    return str(val)


def one_step_transition_closure(
    model: EquationModel,
    q: set[str],
    *,
    timeout_ms: int = 1000,
    include_artifacts: bool = False,
) -> dict[str, Any]:
    """Check one-step transition closure by self-composition.

    Returns ``discharged`` when no two runs can agree on current q/actions while
    disagreeing on next q/observation/terminal/requirement equations.
    """

    if z3 is None:
        return {
            "status": "unavailable",
            "reason": "z3-solver is not installed",
            "claim": "not_claimed_by_this_artifact",
        }

    sorts, type_conflicts = infer_sorts(model)
    unsupported = unsupported_reasons(model, q)
    if type_conflicts or unsupported:
        return {
            "status": "unknown",
            "reason": "unsupported_fragment",
            "type_conflicts": type_conflicts,
            "unsupported": unsupported,
            "claim": "not_claimed_by_this_artifact",
        }

    max_degree = _max_polynomial_degree(model, q)
    # The extracted transition formulas mix Boolean control structure with real
    # arithmetic. Use UF-enabled logics while the syntactic gate above keeps the
    # arithmetic fragment bounded to the configured polynomial degree.
    logic = "QF_UFNRA" if max_degree > 1 else "QF_UFLRA"
    solver = z3.SolverFor(logic)
    solver.set(timeout=timeout_ms)
    enc = Encoder(model, sorts)

    equalities = []
    for name in sorted(q):
        equalities.append(enc.current_var(1, name) == enc.current_var(2, name))
    for action in sorted(model.actions):
        equalities.append(enc.action_var(1, action) == enc.action_var(2, action))

    disagreements: list[tuple[str, Any, Any]] = []
    for name in sorted(q):
        disagreements.append((f"next_q.{name}", enc.next_var(1, name), enc.next_var(2, name)))
    for name, eq in sorted(model.observations.items()):
        disagreements.append((
            f"next_observation.{name}",
            enc.encode_expr(eq.expr, 1, "next"),
            enc.encode_expr(eq.expr, 2, "next"),
        ))
    for name, eq in sorted(model.terminals.items()):
        disagreements.append((
            f"next_terminal.{name}",
            enc.encode_expr(eq.expr, 1, "next"),
            enc.encode_expr(eq.expr, 2, "next"),
        ))
    for name, eq in sorted(model.requirements.items()):
        disagreements.append((
            f"next_requirement.{name}",
            enc.encode_expr(eq.expr, 1, "next"),
            enc.encode_expr(eq.expr, 2, "next"),
        ))

    if equalities:
        solver.add(z3.And(*equalities))
    solver.add(z3.Or(*[_neq(left, right) for _label, left, right in disagreements]))

    result = solver.check()

    def _artifacts() -> dict[str, Any]:
        if not include_artifacts:
            return {}
        payload: dict[str, Any] = {
            "smt2": solver.to_smt2(),
            "z3_check_sat": str(result),
            "replay": (
                "Install z3 and run `z3 query.smt2`. The expected answer is "
                "`unsat` for a discharged proof obligation, `sat` for a "
                "counterexample, or `unknown` for a timeout/unsupported case."
            ),
        }
        if result == z3.unsat:
            try:
                payload["z3_proof"] = str(solver.proof())
                payload["z3_proof_available"] = True
            except Exception as exc:
                payload["z3_proof"] = ""
                payload["z3_proof_available"] = False
                payload["z3_proof_error"] = str(exc)
        elif result == z3.sat:
            try:
                payload["z3_model"] = solver.model().sexpr()
            except Exception as exc:
                payload["z3_model_error"] = str(exc)
        elif result == z3.unknown:
            payload["reason_unknown"] = solver.reason_unknown()
        return {"artifacts": payload}

    if result == z3.unsat:
        return {
            "status": "discharged",
            "claim": "one_step_transition_closure",
            "solver": "z3",
            "logic": logic,
            "max_polynomial_degree": max_degree,
            "timeout_ms": timeout_ms,
            "q_size": len(q),
            "actions_size": len(model.actions),
            "visible_terms_checked": [label for label, _left, _right in disagreements],
            **_artifacts(),
        }
    if result == z3.unknown:
        return {
            "status": "unknown",
            "reason": solver.reason_unknown(),
            "claim": "not_claimed_by_this_artifact",
            "solver": "z3",
            "logic": logic,
            "max_polynomial_degree": max_degree,
            "timeout_ms": timeout_ms,
            **_artifacts(),
        }

    solver_model = solver.model()
    disagreement_witness = None
    for label, left, right in disagreements:
        left_val = _model_value(solver_model, left)
        right_val = _model_value(solver_model, right)
        if left_val != right_val:
            disagreement_witness = {
                "term": label,
                "run1": left_val,
                "run2": right_val,
            }
            break

    q_witness = {
        name: {
            "run1": _model_value(solver_model, enc.current_var(1, name)),
            "run2": _model_value(solver_model, enc.current_var(2, name)),
        }
        for name in sorted(q)
    }
    action_witness = {
        name: {
            "run1": _model_value(solver_model, enc.action_var(1, name)),
            "run2": _model_value(solver_model, enc.action_var(2, name)),
        }
        for name in sorted(model.actions)
    }
    hidden_state = sorted(set(model.state) - set(q))
    hidden_witness = {
        name: {
            "run1": _model_value(solver_model, enc.current_var(1, name)),
            "run2": _model_value(solver_model, enc.current_var(2, name)),
        }
        for name in hidden_state
    }
    return {
        "status": "counterexample",
        "claim": "not_claimed_by_this_artifact",
        "solver": "z3",
        "logic": logic,
        "max_polynomial_degree": max_degree,
        "timeout_ms": timeout_ms,
        "disagreement": disagreement_witness,
        "same_q": q_witness,
        "same_action": action_witness,
        "hidden_state": hidden_witness,
        **_artifacts(),
    }
