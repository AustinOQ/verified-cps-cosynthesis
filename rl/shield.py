"""
Specification-derived safety shields for SysML neural controllers.

Two shield modes:
  FullSpecShield:        evaluates entire #NeuralRequirement (prohibitions + obligations)
  ProhibitionSpecShield: enforces only output-only clauses (structural prohibitions)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from sysml_parser import (
    SysMLParser, ExpressionParser, BinaryExpr, RefExpr, LiteralExpr,
    UnaryExpr, TernaryExpr, Expr,
)


# ---------------------------------------------------------------------------
# AST evaluator
# ---------------------------------------------------------------------------

def _evaluate(expr, values: dict, subject_var: str = ""):
    if isinstance(expr, LiteralExpr):
        return expr.value
    if isinstance(expr, RefExpr):
        path = list(expr.path)
        if path and path[0] == subject_var:
            path = path[1:]
        key = ".".join(path)
        if key in values:
            return values[key]
        if len(path) == 1 and path[0] in values:
            return values[path[0]]
        raise KeyError(f"Unknown ref: {'.'.join(expr.path)}")
    if isinstance(expr, BinaryExpr):
        if expr.op == "implies":
            left = _evaluate(expr.left, values, subject_var)
            return True if not left else _evaluate(expr.right, values, subject_var)
        left = _evaluate(expr.left, values, subject_var)
        right = _evaluate(expr.right, values, subject_var)
        ops = {
            "+": lambda a, b: a + b, "-": lambda a, b: a - b,
            "*": lambda a, b: a * b, "/": lambda a, b: a / b if b else 0,
            "==": lambda a, b: a == b, ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            "and": lambda a, b: a and b, "or": lambda a, b: a or b,
        }
        return ops[expr.op](left, right)
    if isinstance(expr, UnaryExpr):
        val = _evaluate(expr.operand, values, subject_var)
        if expr.op == "not":
            return not val
        if expr.op == "-":
            return -val
    if isinstance(expr, TernaryExpr):
        cond = _evaluate(expr.condition, values, subject_var)
        return _evaluate(expr.true_expr if cond else expr.false_expr,
                         values, subject_var)
    return 0


def _collect_refs(expr) -> set:
    refs = set()
    if isinstance(expr, RefExpr):
        refs.add(expr.path[-1])
    elif isinstance(expr, BinaryExpr):
        refs.update(_collect_refs(expr.left))
        refs.update(_collect_refs(expr.right))
    elif isinstance(expr, UnaryExpr):
        refs.update(_collect_refs(expr.operand))
    elif isinstance(expr, TernaryExpr):
        refs.update(_collect_refs(expr.condition))
        refs.update(_collect_refs(expr.true_expr))
        refs.update(_collect_refs(expr.false_expr))
    return refs


def _flatten_and(expr) -> list:
    if isinstance(expr, BinaryExpr) and expr.op == 'and':
        return _flatten_and(expr.left) + _flatten_and(expr.right)
    return [expr]


# ---------------------------------------------------------------------------
# Base extraction (shared by both shield types)
# ---------------------------------------------------------------------------

def _extract_base(model_path):
    """Extract common data from SysML model. Returns dict."""
    parser = SysMLParser(model_path)
    parser.parse()

    ctrl_fqn = parser.controller_part
    ctrl_inst = parser.part_instances[ctrl_fqn]
    ctrl_def = parser.part_defs[ctrl_inst.part_type]

    neural_def = None
    for ad in ctrl_def.action_defs:
        if "Neural" in ad.metadata:
            neural_def = ad
            break

    in_params = [p.name for p in neural_def.in_params]
    out_params = [p.name for p in neural_def.out_params]
    in_set = set(in_params)
    out_set = set(out_params)

    req_ast = None
    subject_var = ""
    for req_name, sv, _st, req_expr, req_meta in ctrl_def.requirements:
        if "NeuralRequirement" in req_meta:
            req_ast = ExpressionParser(req_expr).parse()
            subject_var = sv
            break

    ctrl_prefix = ctrl_fqn + "::"
    neural_names = in_set | out_set
    unchanging = {}
    for p in parser.parameters:
        if p.qualified_name.startswith(ctrl_prefix) and p.name not in neural_names:
            unchanging[p.name] = p.value

    n_out = len(out_params)
    action_map = {}
    for action_id in range(2 ** n_out):
        actuators = {}
        for bit, name in enumerate(out_params):
            actuators[name] = bool(action_id & (1 << bit))
        action_map[action_id] = actuators

    # Output-only (prohibition) clauses
    prohibition_clauses = []
    if req_ast:
        clauses = _flatten_and(req_ast)
        for clause in clauses:
            refs = _collect_refs(clause)
            refs = {r for r in refs if r != subject_var}
            if refs and refs.issubset(out_set | set(unchanging.keys())):
                prohibition_clauses.append(clause)

    # Dead actions from prohibition clauses only
    dead_actions = set()
    for action_id, actuators in action_map.items():
        values = {**unchanging, **actuators}
        for clause in prohibition_clauses:
            try:
                if not _evaluate(clause, values, subject_var):
                    dead_actions.add(action_id)
                    break
            except Exception:
                pass

    return {
        "in_params": in_params, "out_params": out_params,
        "in_set": in_set, "out_set": out_set,
        "req_ast": req_ast, "subject_var": subject_var,
        "unchanging": unchanging, "action_map": action_map,
        "prohibition_clauses": prohibition_clauses,
        "dead_actions": dead_actions,
    }


# ---------------------------------------------------------------------------
# FullSpecShield: evaluates entire #NeuralRequirement
# ---------------------------------------------------------------------------

class FullSpecShield:
    """Shield that evaluates the full #NeuralRequirement at runtime."""

    def __init__(self, model_path: str):
        d = _extract_base(model_path)
        self.in_params = d["in_params"]
        self.out_params = d["out_params"]
        self.req_ast = d["req_ast"]
        self.subject_var = d["subject_var"]
        self.unchanging = d["unchanging"]
        self.action_map = d["action_map"]
        self.prohibition_clauses = d["prohibition_clauses"]
        self.dead_actions = d["dead_actions"]
        self.mode = "full"

        print(f"  [FullShield] dead_actions={sorted(self.dead_actions)}, "
              f"n_valid={len(self.action_map) - len(self.dead_actions)}")

    def _requirement_action(self, obs_dict: dict) -> int:
        priority = sorted(self.action_map.keys(), key=lambda a: bin(a).count('1'))
        for action_id in priority:
            if action_id in self.dead_actions:
                continue
            actuators = self.action_map[action_id]
            values = {**self.unchanging, **obs_dict, **actuators}
            try:
                if _evaluate(self.req_ast, values, self.subject_var):
                    return action_id
            except Exception:
                continue
        return 0

    def __call__(self, proposed_action: int, obs_dict: dict) -> int:
        if proposed_action in self.dead_actions:
            return self._requirement_action(obs_dict)
        actuators = self.action_map[proposed_action]
        values = {**self.unchanging, **obs_dict, **actuators}
        try:
            if not _evaluate(self.req_ast, values, self.subject_var):
                return self._requirement_action(obs_dict)
        except Exception:
            return self._requirement_action(obs_dict)
        return proposed_action


# ---------------------------------------------------------------------------
# ProhibitionSpecShield: output-only clauses only
# ---------------------------------------------------------------------------

class ProhibitionSpecShield:
    """Shield that enforces only structural prohibitions."""

    def __init__(self, model_path: str):
        d = _extract_base(model_path)
        self.in_params = d["in_params"]
        self.out_params = d["out_params"]
        self.req_ast = d["req_ast"]
        self.subject_var = d["subject_var"]
        self.unchanging = d["unchanging"]
        self.action_map = d["action_map"]
        self.prohibition_clauses = d["prohibition_clauses"]
        self.dead_actions = d["dead_actions"]
        self.mode = "prohibition"

        print(f"  [ProhibitionShield] {len(self.prohibition_clauses)} clauses, "
              f"dead_actions={sorted(self.dead_actions)}, "
              f"n_valid={len(self.action_map) - len(self.dead_actions)}")

    def __call__(self, proposed_action: int, obs_dict: dict) -> int:
        if proposed_action not in self.dead_actions:
            return proposed_action
        best = 0
        best_dist = float('inf')
        for action_id in self.action_map:
            if action_id in self.dead_actions:
                continue
            dist = bin(proposed_action ^ action_id).count('1')
            if dist < best_dist:
                best_dist = dist
                best = action_id
        return best
