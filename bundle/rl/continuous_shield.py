"""
Interval-projection safety shield for a CONTINUOUS (single real-valued)
neural control output.

Counterpart to the discrete `SpecShield` (shield.py): where that shield
enumerates 2^N boolean actions and picks a valid one, this shield reads the
same #NeuralRequirement envelope and, for a given observation, collapses it
to a single safe interval [lo, hi] for the real output, then projects
(clamps) the policy's proposed value onto it.

Mapping from the discrete shield:
  - dead-action mask          -> static range bound (e.g. force in [-100,100])
  - fast-path validity check  -> lo <= proposed <= hi  (pass-through)
  - override scan + pick      -> clamp proposed to nearest point in [lo,hi]

The clamp is the continuous analog of "the valid action the policy most
prefers": for a Gaussian policy the highest-density value inside [lo,hi] is
the point nearest the mean, i.e. the projection.

Supported envelope clause forms (after operator-precedence fixup), conjoined:
  - biconditional   condition == (out <cmp> threshold)
  - one-directional condition implies (out <cmp> threshold)
  - bare bound      out <cmp> threshold
  - negated bound   not (out <cmp> threshold)
where <cmp> in {>, >=, <, <=}, `condition` references only observations and
constants, and `threshold` is constant-evaluable for the given observation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from shield import SpecShield, _evaluate, _flatten_and, _collect_refs
from sysml_parser import BinaryExpr, RefExpr, LiteralExpr, UnaryExpr


_NEG = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}
_FLIP = {">": "<", "<": ">", ">=": "<=", "<=": ">="}


class ContinuousShield:
    """Interval-projection shield for one real-valued control output.

    Callable:  shield(proposed_force, obs_dict) -> (safe_force, overridden)
    """

    def __init__(self, model_path: str, eps: float = 1e-6):
        # Reuse the requirement parser so the interval is rebuilt from the
        # current SysML file.
        spec = SpecShield(model_path)

        self.req_ast = spec.req_ast
        self.in_params = list(spec.in_params)
        self.out_params = list(spec.out_params)
        self.unchanging = dict(spec.unchanging)
        self.subject_var = spec.subject_var
        self.eps = eps

        if len(self.out_params) != 1:
            raise ValueError(
                "ContinuousShield handles exactly one real output; got "
                f"{self.out_params}")
        if (spec.output_types[0] or "").lower() not in {
            "real", "float", "double", "integer", "int"
        }:
            raise ValueError("ContinuousShield output must be numeric")
        self.out_names = set(self.out_params)

        if self.req_ast is None:
            raise ValueError("ContinuousShield: model has no #NeuralRequirement")

        # Static range from bare output bounds in the current requirement.
        self.act_low, self.act_high = self._static_range()

    # ------------------------------------------------------------------
    # Interval derivation
    # ------------------------------------------------------------------

    def _output_comparison_bound(self, comp, values):
        """For an output comparison `out <cmp> K` (or `K <cmp> out`), return
        (op, K) normalised so the output is on the left, else None."""
        if not (isinstance(comp, BinaryExpr) and comp.op in (">", ">=", "<", "<=")):
            return None
        left_has = bool(_collect_refs(comp.left) & self.out_names)
        right_has = bool(_collect_refs(comp.right) & self.out_names)
        try:
            if left_has and not right_has:
                return comp.op, float(_evaluate(comp.right, values, self.subject_var))
            if right_has and not left_has:
                return _FLIP[comp.op], float(_evaluate(comp.left, values, self.subject_var))
        except Exception as exc:
            raise ValueError(f"could not evaluate continuous bound: {exc}") from exc
        return None

    def _apply(self, op, k, lo, hi):
        if op == ">":
            lo = max(lo, k + self.eps)
        elif op == ">=":
            lo = max(lo, k)
        elif op == "<":
            hi = min(hi, k - self.eps)
        elif op == "<=":
            hi = min(hi, k)
        return lo, hi

    def _refs_output(self, expr):
        return bool(_collect_refs(expr) & self.out_names)

    def safe_interval(self, obs_dict: dict):
        """Collapse the envelope to [lo, hi] for the given observation."""
        values = {**self.unchanging, **obs_dict}
        lo, hi = float("-inf"), float("inf")

        for clause in _flatten_and(self.req_ast):
            lo, hi = self._process_clause(clause, values, lo, hi)
        return lo, hi

    def _process_clause(self, c, values, lo, hi):
        # biconditional:  condition == (out <cmp> K)
        if isinstance(c, BinaryExpr) and c.op == "==":
            if self._refs_output(c.left) and not self._refs_output(c.right):
                comp, cond = c.left, c.right
            elif self._refs_output(c.right) and not self._refs_output(c.left):
                comp, cond = c.right, c.left
            else:
                raise ValueError("unsupported equality in continuous requirement")
            res = self._output_comparison_bound(comp, values)
            if res is None:
                raise ValueError("continuous equality does not contain an output bound")
            op, k = res
            cond_true = bool(_evaluate(cond, values, self.subject_var))
            return self._apply(op if cond_true else _NEG[op], k, lo, hi)

        # one-directional:  condition implies (out <cmp> K)
        if isinstance(c, BinaryExpr) and c.op == "implies":
            cond, comp = c.left, c.right
            if self._refs_output(comp) and not self._refs_output(cond):
                res = self._output_comparison_bound(comp, values)
                if res is not None:
                    if bool(_evaluate(cond, values, self.subject_var)):
                        return self._apply(res[0], res[1], lo, hi)
            return lo, hi

        # negated bound:  not (out <cmp> K)
        if isinstance(c, UnaryExpr) and c.op == "not":
            res = self._output_comparison_bound(c.operand, values)
            if res is not None:
                return self._apply(_NEG[res[0]], res[1], lo, hi)
            return lo, hi

        # bare bound:  out <cmp> K
        if isinstance(c, BinaryExpr) and c.op in (">", ">=", "<", "<="):
            if self._refs_output(c):
                res = self._output_comparison_bound(c, values)
                if res is not None:
                    return self._apply(res[0], res[1], lo, hi)
        return lo, hi

    def _static_range(self):
        """Range implied by bare (condition-free) output bounds only."""
        lo, hi = float("-inf"), float("inf")
        for clause in _flatten_and(self.req_ast):
            if (isinstance(clause, BinaryExpr)
                    and clause.op in (">", ">=", "<", "<=")
                    and self._refs_output(clause)):
                res = self._output_comparison_bound(clause, {**self.unchanging})
                if res is not None:
                    lo, hi = self._apply(res[0], res[1], lo, hi)
        if lo == float("-inf") or hi == float("inf"):
            raise ValueError(
                "continuous #NeuralRequirement must state lower and upper output bounds"
            )
        return lo, hi

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def __call__(self, proposed: float, obs_dict: dict):
        """Project the proposed real output onto the safe interval.

        Returns (safe_force, overridden)."""
        lo, hi = self.safe_interval(obs_dict)
        if lo > hi:
            raise ValueError("#NeuralRequirement gives an empty continuous interval")
        safe = min(max(float(proposed), lo), hi)
        overridden = abs(safe - float(proposed)) > 1e-9
        return float(safe), overridden
