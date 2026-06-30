"""
Specification-derived safety shield for SysML neural controllers.

Extracts from the #NeuralRequirement:
  - Full AST (prohibitions + obligations)
  - Output-only clauses → structurally dead actions
  - All clause structure for neural compilation

Used only at construction time by ShieldNet in composite_model.py.
At runtime, ShieldNet makes all decisions via tensor math.
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


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

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


def _fixup_precedence(req_ast, out_set, const_set, subject_var):
    """Fix operator precedence issues where == binds tighter than and/or.

    Detects two patterns:

    Pattern 1 — AND split:
      The spec says: (comp1 AND comp2) == output
      The parser produces: comp1 AND (comp2 == output)
      After flattening: [comp1, (comp2 == output)]
      Fix: merge into [(comp1 AND comp2) == output]

    Pattern 2 — OR with biconditional child:
      The spec says: (comp1 OR comp2) == output
      The parser produces: comp1 OR (comp2 == output)
      Fix: restructure to (comp1 OR comp2) == output

    Detection: a "bare comparison" is a clause that references only
    observations and constants — no output parameters. If it sits
    next to a biconditional, they should be merged.
    """
    ignore = {subject_var} | const_set

    def _refs_no_outputs(expr):
        refs = _collect_refs(expr) - ignore
        return refs and refs.isdisjoint(out_set)

    def _is_biconditional_with_output(expr):
        if isinstance(expr, BinaryExpr) and expr.op == '==':
            left_refs = _collect_refs(expr.left) - ignore
            right_refs = _collect_refs(expr.right) - ignore
            if right_refs & out_set:
                return True
            if left_refs & out_set:
                return True
        return False

    def _get_biconditional_parts(expr):
        """Return (comparison_side, output_side) of a biconditional."""
        left_refs = _collect_refs(expr.left) - ignore
        right_refs = _collect_refs(expr.right) - ignore
        if right_refs & out_set:
            return expr.left, expr.right
        if left_refs & out_set:
            return expr.right, expr.left
        return None, None

    # Pattern 2: fix OR nodes with bare comp + biconditional
    def _fixup_or(expr):
        if not isinstance(expr, BinaryExpr) or expr.op != 'or':
            return expr
        # Recurse first
        left = _fixup_or(expr.left)
        right = _fixup_or(expr.right)

        # Check: left is bare comparison, right is biconditional
        if _refs_no_outputs(left) and _is_biconditional_with_output(right):
            comp_side, out_side = _get_biconditional_parts(right)
            if comp_side is not None:
                return BinaryExpr('==',
                                  BinaryExpr('or', left, comp_side),
                                  out_side)

        # Check: right is bare comparison, left is biconditional
        if _refs_no_outputs(right) and _is_biconditional_with_output(left):
            comp_side, out_side = _get_biconditional_parts(left)
            if comp_side is not None:
                return BinaryExpr('==',
                                  BinaryExpr('or', comp_side, right),
                                  out_side)

        return BinaryExpr('or', left, right)

    # Apply OR fixup to the whole AST first
    def _walk_fix_or(expr):
        if isinstance(expr, BinaryExpr):
            left = _walk_fix_or(expr.left)
            right = _walk_fix_or(expr.right)
            fixed = BinaryExpr(expr.op, left, right)
            if expr.op == 'or':
                return _fixup_or(fixed)
            return fixed
        if isinstance(expr, UnaryExpr):
            return UnaryExpr(expr.op, _walk_fix_or(expr.operand))
        return expr

    fixed_ast = _walk_fix_or(req_ast)

    # Pattern 1: flatten AND, merge bare comparisons into adjacent biconditionals
    clauses = _flatten_and(fixed_ast)
    merged = []
    i = 0
    while i < len(clauses):
        clause = clauses[i]

        # Look ahead: bare comparison followed by biconditional
        if (_refs_no_outputs(clause) and
                i + 1 < len(clauses) and
                _is_biconditional_with_output(clauses[i + 1])):
            bicon = clauses[i + 1]
            comp_side, out_side = _get_biconditional_parts(bicon)
            if comp_side is not None:
                merged.append(BinaryExpr('==',
                                         BinaryExpr('and', clause, comp_side),
                                         out_side))
                i += 2
                continue

        # Look behind: biconditional followed by bare comparison
        if (_is_biconditional_with_output(clause) and
                i + 1 < len(clauses) and
                _refs_no_outputs(clauses[i + 1])):
            bare = clauses[i + 1]
            comp_side, out_side = _get_biconditional_parts(clause)
            if comp_side is not None:
                merged.append(BinaryExpr('==',
                                         BinaryExpr('and', comp_side, bare),
                                         out_side))
                i += 2
                continue

        merged.append(clause)
        i += 1

    # Rebuild conjunction
    if not merged:
        return fixed_ast
    result = merged[0]
    for j in range(1, len(merged)):
        result = BinaryExpr('and', result, merged[j])
    return result


# ---------------------------------------------------------------------------
# SpecShield — extraction only, used at construction time
# ---------------------------------------------------------------------------

class SpecShield:
    """Extracts shield data from SysML specification.

    After construction, provides:
      - req_ast: full requirement AST
      - in_params, out_params: neural interface
      - unchanging: controller constants
      - action_map: action_id → actuator dict
      - dead_actions: structurally invalid actions
      - prohibition_clauses: output-only clauses

    Also callable for verification: shield(action_id, obs_dict) → action_id
    """

    def __init__(self, model_path: str):
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

        self.in_params = [p.name for p in neural_def.in_params]
        self.out_params = [p.name for p in neural_def.out_params]
        in_set = set(self.in_params)
        out_set = set(self.out_params)

        self.req_ast = None
        self.subject_var = ""
        for req_name, sv, _st, req_expr, req_meta in ctrl_def.requirements:
            if "NeuralRequirement" in req_meta:
                self.req_ast = ExpressionParser(req_expr).parse()
                self.subject_var = sv
                break

        ctrl_prefix = ctrl_fqn + "::"
        neural_names = in_set | out_set
        self.unchanging = {}
        for p in parser.parameters:
            if p.qualified_name.startswith(ctrl_prefix) and p.name not in neural_names:
                self.unchanging[p.name] = p.value

        # Pick up controller constants not tagged #ScenarioInput
        # (e.g., toleranceMl) that appear in the requirement AST
        if self.req_ast:
            req_refs = _collect_refs(self.req_ast)
            req_refs.discard(self.subject_var)
            missing = req_refs - in_set - out_set - set(self.unchanging.keys())
            if missing:
                from simulator import SimulationEngine, BindRef
                eng = SimulationEngine(parser)
                eng.initialize()
                for key, val in eng.state.items():
                    if isinstance(val, BindRef):
                        continue
                    if key.startswith(ctrl_prefix):
                        name = key[len(ctrl_prefix):]
                        if name in missing:
                            self.unchanging[name] = val
                            missing.discard(name)
                if missing:
                    print(f"  [SpecShield] WARNING: unresolved refs "
                          f"in requirement: {missing}")

        n_out = len(self.out_params)
        self.action_map = {}
        for action_id in range(2 ** n_out):
            actuators = {}
            for bit, name in enumerate(self.out_params):
                actuators[name] = bool(action_id & (1 << bit))
            self.action_map[action_id] = actuators

        # Fix operator precedence issues (== binding tighter than and/or)
        if self.req_ast:
            self.req_ast = _fixup_precedence(
                self.req_ast, out_set,
                set(self.unchanging.keys()), self.subject_var)

        # Dead actions from output-only clauses
        self.dead_actions = set()
        self.prohibition_clauses = []
        if self.req_ast:
            clauses = _flatten_and(self.req_ast)
            for clause in clauses:
                refs = _collect_refs(clause)
                refs = {r for r in refs if r != self.subject_var}
                if refs and refs.issubset(out_set | set(self.unchanging.keys())):
                    self.prohibition_clauses.append(clause)

        for action_id, actuators in self.action_map.items():
            values = {**self.unchanging, **actuators}
            for clause in self.prohibition_clauses:
                try:
                    if not _evaluate(clause, values, self.subject_var):
                        self.dead_actions.add(action_id)
                        break
                except Exception:
                    pass

        print(f"  [SpecShield] Loaded from {model_path}")
        print(f"  [SpecShield] dead_actions={sorted(self.dead_actions)}, "
              f"n_valid={2**n_out - len(self.dead_actions)}")

    def _requirement_action(self, obs_dict: dict) -> int:
        if self.req_ast is None:
            return 0
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
        """Evaluate shield decision via AST. Used only for verification."""
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
