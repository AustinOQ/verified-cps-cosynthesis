#!/usr/bin/env python3
"""
SysML v2 → nuXmv SMV Translator

Translates a SysML v2 model (in the format used by this project) into an
SMV model suitable for verification with nuXmv.

Translation strategy:
  - State machine          → enum VAR + next() case expression
  - Step-action variables  → VAR (real or bounded integer) + Euler next()
  - Do-action variables    → VAR (real or bounded integer) + next() case expression
  - Controlled booleans    → boolean VAR + next() derived from state machine
  - Command triggers       → boolean IVAR (nondeterministic inputs)
  - Trigger-var attributes → real IVAR (Phase 1; promoted to DEFINE in Phase 3)
  - Parameters/constants   → DEFINE
  - Derived attributes     → DEFINE
  - Flow constraints       → DEFINE (computed flow rates)
  - Flow connections       → DEFINE (port-to-port propagation)
  - Bind statements        → DEFINE (port attribute aliases)

Integer step-action targets use bounded integer arithmetic with Euler
integration and clamping; real targets use nuXmv native real arithmetic
without clamping.  dt = 1 time unit per step.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from sysml_parser import (
    Expr, LiteralExpr, RefExpr, BinaryExpr, TernaryExpr, UnaryExpr,
    AssignStmt, ItemDeclStmt, SendStmt,
    Constraint, DerivedAttribute, StepAction, Parameter,
    State, Transition, StateMachine, Flow, PartInstance, PartDef,
    SysMLParser,
)


# =============================================================================
# Expression → SMV string translator
# =============================================================================

def expr_to_smv(expr: Expr, context: str, system: str,
                ref_bindings: dict[str, str],
                state_var: str = "controller_state") -> str:
    """Recursively translate an expression AST node to SMV syntax."""

    def smv_name(key: str) -> str:
        parts = key.split("::")
        if parts and parts[0] == system:
            parts = parts[1:]
        return "_".join(parts)

    def resolve_ref(path: list[str]) -> str:
        # behavior.stateName → state variable comparison
        if len(path) == 2 and path[0] == "behavior":
            return f"{state_var} = {path[1]}"
        # dt is the global timestep constant
        if len(path) == 1 and path[0] == "dt":
            return "dt"
        # Follow ref binding if available
        if context and ref_bindings:
            ref_key = f"{context}::{path[0]}"
            if ref_key in ref_bindings:
                target = ref_bindings[ref_key]
                full = target + "::" + "::".join(path[1:]) if len(path) > 1 else target
                return smv_name(full)

        # Context-qualified path
        if context:
            return smv_name(context + "::" + "::".join(path))
        return smv_name("::".join(path))

    def trans(e: Expr) -> str:
        if isinstance(e, LiteralExpr):
            v = e.value
            if isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)

        elif isinstance(e, RefExpr):
            return resolve_ref(e.path)

        elif isinstance(e, BinaryExpr):
            op_map = {
                "+": "+", "-": "-", "*": "*", "/": "/",
                "==": "=", ">=": ">=", "<=": "<=", ">": ">", "<": "<",
                "and": "&", "or": "|", "implies": "->",
            }
            L = trans(e.left)
            R = trans(e.right)
            op = op_map.get(e.op, e.op)
            return f"({L} {op} {R})"

        elif isinstance(e, TernaryExpr):
            C = trans(e.condition)
            T = trans(e.true_expr)
            F = trans(e.false_expr)
            return f"(case {C} : {T}; TRUE : {F}; esac)"

        elif isinstance(e, UnaryExpr):
            V = trans(e.operand)
            if e.op == "not":
                return f"(!{V})"
            elif e.op == "-":
                return f"(-{V})"

        return "TRUE"

    return trans(expr)


# =============================================================================
# SMV Generator
# =============================================================================

def _contains_implies(expr: Expr) -> bool:
    if isinstance(expr, BinaryExpr):
        return expr.op == "implies" or _contains_implies(expr.left) or _contains_implies(expr.right)
    return False


class SMVGenerator:
    """Generates a nuXmv SMV model from a parsed SysML model."""

    def __init__(self, parser: SysMLParser):
        self.parser = parser
        self.system = parser.system_part
        self.ref_bindings = parser.ref_bindings

        # Per-instance state machines: fqn → SM
        self._sm_instances: dict[str, StateMachine] = dict(parser.instance_state_machines)

        # Compute STATE_VAR for backward compat (controlled booleans, implies constraints).
        # Prefer the controller instance; fall back to the first SM instance.
        ctrl_fqn = parser.controller_part  # already a full FQN like "system::ctrl"
        if ctrl_fqn and ctrl_fqn in self._sm_instances:
            self.STATE_VAR = self._state_var(ctrl_fqn)
        elif self._sm_instances:
            self.STATE_VAR = self._state_var(next(iter(self._sm_instances)))
        else:
            self.STATE_VAR = "controller_state"

        # Populated by _build_info()
        self._step_actions: list[StepAction] = []
        self._sa_keys: set[str] = set()            # step-action target keys
        self._da_keys: set[str] = set()            # do-action target keys
        self._defines: dict[str, str] = {}         # smv_name → smv_expr
        self._controlled: dict[str, dict[str, Any]] = {}  # smv_name → {state: value}
        self._bool_ivars: list[str] = []           # SMV names for boolean IVARs
        self._real_ivars: list[str] = []           # SMV names for real IVARs

        self._build_info()

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    def _smv(self, key: str) -> str:
        """Convert a qualified key to a flat SMV identifier."""
        parts = key.split("::")
        if parts and parts[0] == self.system:
            parts = parts[1:]
        return "_".join(parts)

    def _state_var(self, inst_fqn: str) -> str:
        """Return the SMV state variable name for a part instance FQN."""
        return self._smv(inst_fqn) + "_state"

    def _attr_type(self, key: str) -> Optional[str]:
        """Return the SysML type string for a qualified attribute key, or None.

        Resolves multi-segment paths (e.g. inst::portName::itemName::attrName)
        by walking part_def_ports → port_def_items → item_def_attrs.
        Local item declarations in step/do-action bodies are also consulted.
        """
        parts = key.split("::")
        if parts and parts[0] == self.system:
            parts = parts[1:]
        if len(parts) < 2:
            return None
        inst_name = parts[0]
        path = parts[1:]

        inst = self.parser.part_instances.get(f"{self.system}::{inst_name}")
        if inst is None:
            return None

        current_type = inst.part_type
        current_kind = "part"  # "part" | "port" | "item"

        for i, segment in enumerate(path):
            is_last = (i == len(path) - 1)

            if current_kind == "part":
                part_def = self.parser.part_defs.get(current_type)
                if part_def and segment in part_def.attributes:
                    return part_def.attributes[segment] if is_last else None
                port_type = self.parser.part_def_ports.get(current_type, {}).get(segment)
                if port_type:
                    current_type = port_type
                    current_kind = "port"
                    continue
                local_type = self._resolve_local_item(inst_name, segment)
                if local_type:
                    current_type = local_type
                    current_kind = "item"
                    continue
                return None

            elif current_kind == "port":
                item_type = self.parser.port_def_items.get(current_type, {}).get(segment)
                if item_type:
                    current_type = item_type
                    current_kind = "item"
                    continue
                return None

            elif current_kind == "item":
                item_attrs = self.parser.item_def_attrs.get(current_type, {})
                if segment in item_attrs:
                    return item_attrs[segment] if is_last else None
                return None

        return None

    def _resolve_local_item(self, inst_name: str, var_name: str) -> Optional[str]:
        """Return the item type of a local 'item var : Type' declaration found
        in step or do-action bodies for the given part instance name."""
        inst_fqn = f"{self.system}::{inst_name}"
        for fqn, stmts in self.parser.step_action_bodies:
            if fqn == inst_fqn:
                for stmt in stmts:
                    if isinstance(stmt, ItemDeclStmt) and stmt.name == var_name:
                        return stmt.type_name
        sm = self._sm_instances.get(inst_fqn)
        if sm:
            for trans in sm.transitions:
                if trans.do_action:
                    for stmt in trans.do_action:
                        if isinstance(stmt, ItemDeclStmt) and stmt.name == var_name:
                            return stmt.type_name
        return None

    def _is_real(self, key: str) -> bool:
        """Return True if the attribute identified by key has SysML type Real."""
        return self._attr_type(key) == "Real"

    def _trigger_ivar_name(self, inst_fqn: str, trigger_port: Optional[str],
                           trigger: str) -> str:
        """Return the boolean IVAR name for a trigger on a part instance."""
        inst_smv = self._smv(inst_fqn)
        if trigger_port:
            return f"{inst_smv}_{trigger_port}_{trigger}_available"
        return f"{inst_smv}_{trigger}_available"

    def _is_item_type_or_subtype(self, sent_type: str, accepted_type: str) -> bool:
        """Return True if sent_type equals accepted_type or is a subtype of it."""
        t: Optional[str] = sent_type
        while t is not None:
            if t == accepted_type:
                return True
            t = self.parser.item_type_parents.get(t)
        return False

    def _resolve_ref(self, path: list[str], context: str) -> str:
        """Resolve a path to its SMV identifier, following ref bindings."""
        if len(path) == 2 and path[0] == "behavior":
            return f"{self.STATE_VAR} = {path[1]}"
        if len(path) == 1 and path[0] == "dt":
            return "dt"
        if context and self.ref_bindings:
            ref_key = f"{context}::{path[0]}"
            if ref_key in self.ref_bindings:
                target = self.ref_bindings[ref_key]
                full = target + "::" + "::".join(path[1:]) if len(path) > 1 else target
                return self._smv(full)
        if context:
            return self._smv(context + "::" + "::".join(path))
        return self._smv("::".join(path))

    def _to_smv(self, expr: Expr, context: str) -> str:
        return expr_to_smv(expr, context, self.system, self.ref_bindings, self.STATE_VAR)

    # ------------------------------------------------------------------
    # Info collection
    # ------------------------------------------------------------------

    def _build_info(self):
        # Collect step actions and their target keys
        self._step_actions = list(self.parser.step_actions)
        for sa in self._step_actions:
            self._sa_keys.add(sa.target_key)

        # dt is a global constant (dt = 1 for Euler integration)
        self._defines['dt'] = '1'

        # Collect trigger IVARs from all SM instances
        seen_bool: set[str] = set()
        seen_real: set[str] = set()
        for inst_fqn, sm in self._sm_instances.items():
            for trans in sm.transitions:
                if trans.trigger:
                    name = self._trigger_ivar_name(inst_fqn, trans.trigger_port, trans.trigger)
                    if name not in seen_bool:
                        seen_bool.add(name)
                        self._bool_ivars.append(name)
                if trans.trigger_var and trans.guard:
                    self._collect_trigger_var_ivars(
                        trans.trigger_var, trans.guard, inst_fqn, seen_real)

        # Collect do-action target keys
        for inst_fqn, sm in self._sm_instances.items():
            for trans in sm.transitions:
                if trans.do_action:
                    for stmt in trans.do_action:
                        if isinstance(stmt, AssignStmt):
                            target_key = inst_fqn + "::" + "::".join(stmt.target)
                            self._da_keys.add(target_key)

        # Process assignment constraints → DEFINE
        for c in self.parser.parsed_constraints:
            if _contains_implies(c.expression):
                continue
            e = c.expression
            if isinstance(e, BinaryExpr) and e.op == "==" and isinstance(e.left, RefExpr):
                lhs = self._resolve_ref(e.left.path, c.context)
                rhs = self._to_smv(e.right, c.context)
                self._defines[lhs] = rhs

        # Process derived attributes → DEFINE
        for attr in self.parser.derived_attributes:
            smv_n = self._smv(attr.qualified_name)
            self._defines[smv_n] = self._to_smv(attr.expression, attr.context)

        # Bind statements → DEFINE aliases (Phase 2)
        for lhs_key, rhs_key in self.parser.parsed_bindings.items():
            lhs_smv = self._smv(lhs_key)
            rhs_smv = self._smv(rhs_key)
            if lhs_smv not in self._defines:
                self._defines[lhs_smv] = rhs_smv

        # Phase 3a: Generic flow DEFINE propagation.
        # For each flow from A.portX to B.portY, scan all currently known SMV
        # names for the prefix smv(A.portX)_ and emit a DEFINE alias
        # smv(B.portY)_suffix := smv(A.portX)_suffix for each match.
        # Iterate to a fixed point so chained flows are fully propagated.
        changed = True
        while changed:
            changed = False
            known = (
                {self._state_var(fqn) for fqn in self._sm_instances}
                | {self._smv(k) for k in self._sa_keys}
                | {self._smv(k) for k in self._da_keys}
                | set(self._bool_ivars)
                | set(self._real_ivars)
                | set(self._defines.keys())
            )
            for flow in self.parser.flows:
                from_smv = self._smv(
                    f"{self.system}::{flow.from_port.replace('.', '::')}")
                to_smv = self._smv(
                    f"{self.system}::{flow.to_port.replace('.', '::')}")
                from_prefix = from_smv + "_"
                for name in sorted(known):
                    if name.startswith(from_prefix):
                        suffix = name[len(from_prefix):]
                        dest = to_smv + "_" + suffix
                        if dest not in self._defines:
                            self._defines[dest] = name
                            changed = True

        # Phase 3b: Promote trigger-var real IVARs to DEFINE aliases.
        # For each real IVAR produced in Phase 1d, check whether Phases 2/3a
        # have created a DEFINE that corresponds to the same attribute reached
        # via the trigger port.  Matching: given IVAR
        #   {inst}_{trigger_var}_{attr_suffix}
        # look for a DEFINE key starting with {inst}_{trigger_port}_ and
        # ending with _{attr_suffix}.  If exactly one match exists, the IVAR
        # is redundant — replace it with a DEFINE alias pointing to that key.
        promoted: dict[str, str] = {}  # ivar_name → matched define key
        for inst_fqn, sm in self._sm_instances.items():
            inst_smv = self._smv(inst_fqn)
            for trans in sm.transitions:
                if not (trans.trigger_var and trans.trigger_port and trans.guard):
                    continue
                refs: list[str] = []
                self._collect_trigger_var_refs(
                    trans.trigger_var, trans.guard, inst_fqn, refs)
                tv_prefix = f"{inst_smv}_{trans.trigger_var}_"
                port_prefix = f"{inst_smv}_{trans.trigger_port}_"
                for ivar_name in refs:
                    if ivar_name not in self._real_ivars:
                        continue
                    if not ivar_name.startswith(tv_prefix):
                        continue
                    attr_suffix = ivar_name[len(tv_prefix):]
                    candidates = [
                        k for k in self._defines
                        if k.startswith(port_prefix)
                        and k.endswith("_" + attr_suffix)
                    ]
                    if len(candidates) == 1:
                        promoted[ivar_name] = candidates[0]

        for ivar_name, define_name in promoted.items():
            self._real_ivars.remove(ivar_name)
            if ivar_name not in self._defines:
                self._defines[ivar_name] = define_name

        # Phase 3c: Couple sender do-action sends to receiver trigger IVARs.
        # For each boolean IVAR produced in Phase 1d from a receiver's accept trigger,
        # check whether a connected sender's SM transitions contain do-action SendStmts
        # that send the matching item type via the connected port.  If so, replace the
        # free IVAR with a DEFINE equal to the disjunction of the sender's firing
        # conditions, making the causal link explicit.
        #
        # Uses parser.connects [(from_dot, to_dot)] to map sender.outputPort →
        # receiver.inputPort.  Each dot token is relative to the enclosing system def.
        inst_by_name: dict[str, str] = {
            fqn.split("::")[-1]: fqn for fqn in self._sm_instances
        }
        connect_map: dict[str, str] = {
            to_dot: from_dot for from_dot, to_dot in self.parser.connects
        }
        coupled: dict[str, str] = {}
        for recv_fqn, recv_sm in self._sm_instances.items():
            recv_name = recv_fqn.split("::")[-1]
            for trans in recv_sm.transitions:
                if not trans.trigger or not trans.trigger_port:
                    continue
                ivar_name = self._trigger_ivar_name(
                    recv_fqn, trans.trigger_port, trans.trigger)
                if ivar_name not in self._bool_ivars:
                    continue
                send_dot = connect_map.get(f"{recv_name}.{trans.trigger_port}")
                if send_dot is None:
                    continue
                send_parts = send_dot.split(".", 1)
                if len(send_parts) != 2:
                    continue
                send_part_name, send_port_name = send_parts
                send_fqn = inst_by_name.get(send_part_name)
                if send_fqn is None or send_fqn not in self._sm_instances:
                    continue
                send_sm = self._sm_instances[send_fqn]
                send_state_var = self._state_var(send_fqn)
                firing_conditions: list[str] = []
                for strans in send_sm.transitions:
                    if not strans.do_action:
                        continue
                    for stmt in strans.do_action:
                        if not isinstance(stmt, SendStmt) or stmt.port != send_port_name:
                            continue
                        item_type = next(
                            (d.type_name for d in strans.do_action
                             if isinstance(d, ItemDeclStmt) and d.name == stmt.item_name),
                            None,
                        )
                        if item_type is None:
                            continue
                        if not self._is_item_type_or_subtype(item_type, trans.trigger):
                            continue
                        cond_parts = [f"{send_state_var} = {strans.from_state}"]
                        if strans.trigger:
                            cond_parts.append(self._trigger_ivar_name(
                                send_fqn, strans.trigger_port, strans.trigger))
                        if strans.guard:
                            cond_parts.append(
                                f"({self._to_smv(strans.guard, send_fqn)})")
                        firing_conditions.append(" & ".join(cond_parts))
                if firing_conditions:
                    expr = (
                        " | ".join(f"({c})" for c in firing_conditions)
                        if len(firing_conditions) > 1
                        else firing_conditions[0]
                    )
                    coupled[ivar_name] = expr

        for ivar_name, expr in coupled.items():
            self._bool_ivars.remove(ivar_name)
            if ivar_name not in self._defines:
                self._defines[ivar_name] = expr

        # Process implies constraints → controlled boolean assignments per state
        for c in self.parser.parsed_constraints:
            if not _contains_implies(c.expression):
                continue
            self._process_implies(c)

        # Process non-behavior implies constraints → conditional DEFINE (case expression)
        cond_branches: dict[str, list[tuple[str, str]]] = {}
        for c in self.parser.parsed_constraints:
            if not _contains_implies(c.expression):
                continue
            self._collect_cond_branches(c.expression, c.context, cond_branches)
        for lhs_smv, branches in cond_branches.items():
            if lhs_smv not in self._defines:
                case_str = " ".join(f"{cond} : {val};" for cond, val in branches)
                self._defines[lhs_smv] = f"(case {case_str} TRUE : 0; esac)"

    def _collect_trigger_var_ivars(self, trigger_var: str, guard: Expr,
                                   inst_fqn: str, seen: set[str]):
        """Collect real IVAR names for trigger-variable attributes used in a guard."""
        refs: list[str] = []
        self._collect_trigger_var_refs(trigger_var, guard, inst_fqn, refs)
        for r in refs:
            if r not in seen:
                seen.add(r)
                self._real_ivars.append(r)

    def _collect_trigger_var_refs(self, trigger_var: str, expr: Expr,
                                  inst_fqn: str, refs: list[str]):
        """Walk expr and collect SMV names for paths starting with trigger_var."""
        if isinstance(expr, RefExpr) and expr.path and expr.path[0] == trigger_var:
            refs.append(self._resolve_ref(expr.path, inst_fqn))
        elif isinstance(expr, BinaryExpr):
            self._collect_trigger_var_refs(trigger_var, expr.left, inst_fqn, refs)
            self._collect_trigger_var_refs(trigger_var, expr.right, inst_fqn, refs)
        elif isinstance(expr, UnaryExpr):
            self._collect_trigger_var_refs(trigger_var, expr.operand, inst_fqn, refs)
        elif isinstance(expr, TernaryExpr):
            self._collect_trigger_var_refs(trigger_var, expr.condition, inst_fqn, refs)
            self._collect_trigger_var_refs(trigger_var, expr.true_expr, inst_fqn, refs)
            self._collect_trigger_var_refs(trigger_var, expr.false_expr, inst_fqn, refs)

    def _process_implies(self, constraint: Constraint):
        """Extract state → {var: bool} from a 'behavior.X implies (...)' constraint."""
        e = constraint.expression
        if not (isinstance(e, BinaryExpr) and e.op == "implies"):
            return
        if not (isinstance(e.left, RefExpr) and len(e.left.path) == 2
                and e.left.path[0] == "behavior"):
            return
        state_name = e.left.path[1]
        self._extract_assignments(e.right, state_name, constraint.context)

    def _collect_cond_branches(self, expr: Expr, context: str,
                               result: dict[str, list[tuple[str, str]]]):
        """Collect (condition_smv, value_smv) branches from non-behavior implies."""
        if isinstance(expr, BinaryExpr) and expr.op == "and":
            self._collect_cond_branches(expr.left, context, result)
            self._collect_cond_branches(expr.right, context, result)
            return
        if not (isinstance(expr, BinaryExpr) and expr.op == "implies"):
            return
        lhs, rhs = expr.left, expr.right
        # Skip behavior.X implies (...)  — handled by _process_implies
        if isinstance(lhs, RefExpr) and len(lhs.path) == 2 and lhs.path[0] == "behavior":
            return
        # Only handle X == expr on RHS
        if isinstance(rhs, BinaryExpr) and rhs.op == "==" and isinstance(rhs.left, RefExpr):
            target_smv = self._resolve_ref(rhs.left.path, context)
            cond_smv = self._to_smv(lhs, context)
            val_smv = self._to_smv(rhs.right, context)
            result.setdefault(target_smv, []).append((cond_smv, val_smv))

    def _extract_assignments(self, expr: Expr, state_name: str, context: str):
        if isinstance(expr, BinaryExpr):
            if expr.op == "and":
                self._extract_assignments(expr.left, state_name, context)
                self._extract_assignments(expr.right, state_name, context)
            elif expr.op == "==" and isinstance(expr.left, RefExpr):
                smv_n = self._resolve_ref(expr.left.path, context)
                if isinstance(expr.right, LiteralExpr):
                    if smv_n not in self._controlled:
                        self._controlled[smv_n] = {}
                    self._controlled[smv_n][state_name] = expr.right.value

    # ------------------------------------------------------------------
    # Capacity / bounds helpers
    # ------------------------------------------------------------------

    def _capacity_for(self, level_key: str) -> int:
        """Find the capacity value for a level variable (e.g. feederTank1::currentLevel)."""
        cap_key = level_key.rsplit("::", 1)[0] + "::capacity"
        for p in self.parser.parameters:
            if p.qualified_name == cap_key:
                return int(p.value)
        return 2147483647  # fallback: unbounded

    def _init_for(self, key: str) -> str:
        """Find the initial value for a parameter key, formatted for its type."""
        for p in self.parser.parameters:
            if p.qualified_name == key:
                v = p.value
                if self._is_real(key):
                    if isinstance(v, float) and v == int(v):
                        return f"{int(v)}.0"
                    return str(v)
                return str(int(v)) if isinstance(v, float) else str(v)
        return "0.0" if self._is_real(key) else "0"

    # ------------------------------------------------------------------
    # Undefined reference detection
    # ------------------------------------------------------------------

    def _collect_refs_in_expr(self, expr: Expr, context: str, refs: set[str]):
        """Collect all SMV names referenced in an expression."""
        if isinstance(expr, RefExpr):
            name = self._resolve_ref(expr.path, context)
            # Skip state comparisons (they resolve to 'state_var = X')
            if " = " not in name:
                refs.add(name)
        elif isinstance(expr, BinaryExpr):
            self._collect_refs_in_expr(expr.left, context, refs)
            self._collect_refs_in_expr(expr.right, context, refs)
        elif isinstance(expr, TernaryExpr):
            self._collect_refs_in_expr(expr.condition, context, refs)
            self._collect_refs_in_expr(expr.true_expr, context, refs)
            self._collect_refs_in_expr(expr.false_expr, context, refs)
        elif isinstance(expr, UnaryExpr):
            self._collect_refs_in_expr(expr.operand, context, refs)

    def _find_undefined_refs(self) -> set[str]:
        """Find SMV names referenced in expressions but not declared anywhere."""
        all_refs: set[str] = set()
        params_smv = {self._smv(p.qualified_name) for p in self.parser.parameters}
        sa_smv = {self._smv(k) for k in self._sa_keys}
        da_smv = {self._smv(k) for k in self._da_keys}
        ivar_smv = set(self._bool_ivars) | set(self._real_ivars)
        state_vars = {self._state_var(fqn) for fqn in self._sm_instances}
        declared = (set(self._defines.keys()) | params_smv | sa_smv | da_smv
                    | ivar_smv | state_vars | set(self._controlled.keys()))

        for sa in self._step_actions:
            self._collect_refs_in_expr(sa.expression, sa.context, all_refs)

        for attr in self.parser.derived_attributes:
            self._collect_refs_in_expr(attr.expression, attr.context, all_refs)

        return all_refs - declared

    # ------------------------------------------------------------------
    # SMV section generators
    # ------------------------------------------------------------------

    def _gen_var(self) -> list[str]:
        lines = ["VAR"]

        # State machine state variables (one per SM instance)
        for inst_fqn, sm in self._sm_instances.items():
            state_var = self._state_var(inst_fqn)
            states = ", ".join(s.name for s in sm.states)
            lines.append(f"  {state_var} : {{{states}}};")

        # Step-action target variables
        for sa in self._step_actions:
            smv_n = self._smv(sa.target_key)
            if self._is_real(sa.target_key):
                lines.append(f"  {smv_n} : real;")
            else:
                upper = self._capacity_for(sa.target_key)
                lines.append(f"  {smv_n} : 0..{upper};")

        # Do-action controlled variables
        for key in sorted(self._da_keys):
            smv_n = self._smv(key)
            if self._is_real(key):
                lines.append(f"  {smv_n} : real;")
            else:
                upper = self._capacity_for(key)
                lines.append(f"  {smv_n} : 0..{upper};")

        # Controlled booleans (from implies constraints, backward compat)
        for smv_n in sorted(self._controlled.keys()):
            lines.append(f"  {smv_n} : boolean;")

        return lines

    def _gen_ivar(self) -> list[str]:
        if not self._bool_ivars and not self._real_ivars:
            return []
        lines = ["IVAR"]
        for name in self._bool_ivars:
            lines.append(f"  {name} : boolean;")
        for name in self._real_ivars:
            lines.append(f"  {name} : real;")
        return lines

    def _gen_define(self) -> list[str]:
        lines = ["DEFINE"]

        # Constants from parameters (excluding VAR targets)
        for p in self.parser.parameters:
            if p.qualified_name in self._sa_keys or p.qualified_name in self._da_keys:
                continue
            smv_n = self._smv(p.qualified_name)
            if self._is_real(p.qualified_name):
                v = p.value
                val = f"{int(v)}.0" if isinstance(v, float) and v == int(v) else str(v)
            else:
                val = int(p.value) if isinstance(p.value, float) and p.value == int(p.value) else p.value
            lines.append(f"  {smv_n} := {val};")

        # Computed expressions
        for smv_n, smv_expr in self._defines.items():
            lines.append(f"  {smv_n} := {smv_expr};")

        # Zero-define any flow-rate names that appear in dynamics expressions
        # but are not declared anywhere (e.g. an unconnected inlet port)
        undefined = self._find_undefined_refs()
        flow_rate_undefined = {n for n in undefined if n.endswith("_flowRate")}
        for n in sorted(flow_rate_undefined):
            lines.append(f"  {n} := 0;  -- unconnected port, defaults to zero")

        return lines

    def _gen_assign(self) -> list[str]:
        lines = ["ASSIGN"]

        # Determine the primary SM (for controlled-boolean init, backward compat)
        ctrl_fqn = self.parser.controller_part
        if ctrl_fqn and ctrl_fqn in self._sm_instances:
            primary_sm = self._sm_instances[ctrl_fqn]
        elif self._sm_instances:
            primary_sm = next(iter(self._sm_instances.values()))
        else:
            primary_sm = None

        # init() for state machine state variables
        for inst_fqn, sm in self._sm_instances.items():
            state_var = self._state_var(inst_fqn)
            lines.append(f"  init({state_var}) := {sm.initial_state};")

        # init() for step-action targets
        for sa in self._step_actions:
            smv_n = self._smv(sa.target_key)
            val = self._init_for(sa.target_key)
            lines.append(f"  init({smv_n}) := {val};")

        # init() for do-action targets
        for key in sorted(self._da_keys):
            smv_n = self._smv(key)
            val = self._init_for(key)
            lines.append(f"  init({smv_n}) := {val};")

        # init() for controlled booleans (from primary SM initial state)
        if primary_sm:
            init_state = primary_sm.initial_state
            for smv_n in sorted(self._controlled.keys()):
                init_val = self._controlled[smv_n].get(init_state, False)
                lines.append(f"  init({smv_n}) := {'TRUE' if init_val else 'FALSE'};")

        # next() for state machine transitions (one block per SM instance)
        for inst_fqn, sm in self._sm_instances.items():
            state_var = self._state_var(inst_fqn)
            lines.append(f"  next({state_var}) :=")
            lines.append("    case")
            for trans in sm.transitions:
                cond_parts = [f"{state_var} = {trans.from_state}"]
                if trans.trigger:
                    cond_parts.append(
                        self._trigger_ivar_name(inst_fqn, trans.trigger_port, trans.trigger))
                if trans.guard:
                    cond_parts.append(f"({self._to_smv(trans.guard, inst_fqn)})")
                cond = " & ".join(cond_parts)
                lines.append(f"      {cond} : {trans.to_state};")
            lines.append(f"      TRUE : {state_var};")
            lines.append("    esac;")

        # next() for controlled booleans (backward compat, uses primary SM STATE_VAR)
        for smv_n in sorted(self._controlled.keys()):
            vals = self._controlled[smv_n]
            lines.append(f"  next({smv_n}) :=")
            lines.append("    case")
            for state_name, val in vals.items():
                smv_val = "TRUE" if val else "FALSE"
                lines.append(f"      next({self.STATE_VAR}) = {state_name} : {smv_val};")
            lines.append(f"      TRUE : {smv_n};")
            lines.append("    esac;")

        # next() for step-action targets (real: unclamped; integer: clamped case)
        for sa in self._step_actions:
            smv_n = self._smv(sa.target_key)
            new_val = self._to_smv(sa.expression, sa.context)
            if self._is_real(sa.target_key):
                lines.append(f"  next({smv_n}) := ({new_val});")
            else:
                upper = self._capacity_for(sa.target_key)
                lines.append(f"  next({smv_n}) :=")
                lines.append("    case")
                lines.append(f"      ({new_val}) < 0 : 0;")
                lines.append(f"      ({new_val}) > {upper} : {upper};")
                lines.append(f"      TRUE : ({new_val});")
                lines.append("    esac;")

        # next() for do-action targets (case expression over active transitions)
        da_branches: dict[str, list[tuple[str, str]]] = {}
        for inst_fqn, sm in self._sm_instances.items():
            state_var = self._state_var(inst_fqn)
            for trans in sm.transitions:
                if not trans.do_action:
                    continue
                cond_parts = [f"{state_var} = {trans.from_state}"]
                if trans.trigger:
                    cond_parts.append(
                        self._trigger_ivar_name(inst_fqn, trans.trigger_port, trans.trigger))
                if trans.guard:
                    cond_parts.append(f"({self._to_smv(trans.guard, inst_fqn)})")
                cond = " & ".join(cond_parts)
                for stmt in trans.do_action:
                    if isinstance(stmt, AssignStmt):
                        target_key = inst_fqn + "::" + "::".join(stmt.target)
                        if target_key not in self._da_keys:
                            continue
                        target_smv = self._smv(target_key)
                        val_smv = self._to_smv(stmt.expr, inst_fqn)
                        da_branches.setdefault(target_smv, []).append((cond, val_smv))

        for target_smv in sorted(da_branches.keys()):
            branches = da_branches[target_smv]
            lines.append(f"  next({target_smv}) :=")
            lines.append("    case")
            for cond, val in branches:
                lines.append(f"      {cond} : {val};")
            lines.append(f"      TRUE : {target_smv};")
            lines.append("    esac;")

        return lines

    def _gen_specs(self) -> list[str]:
        lines = ["-- Verification properties", ""]

        if not self._sm_instances:
            return lines

        # Determine primary SM for controlled-boolean specs (backward compat)
        ctrl_fqn = self.parser.controller_part
        if ctrl_fqn and ctrl_fqn in self._sm_instances:
            primary_sm = self._sm_instances[ctrl_fqn]
        elif self._sm_instances:
            primary_sm = next(iter(self._sm_instances.values()))
        else:
            primary_sm = None

        # Safety: actuators follow state machine (controlled booleans, backward compat)
        if primary_sm and self._controlled:
            for state_name in [s.name for s in primary_sm.states]:
                parts_cond = []
                for smv_n, vals in sorted(self._controlled.items()):
                    if state_name in vals:
                        parts_cond.append(smv_n if vals[state_name] else f"!{smv_n}")
                if parts_cond:
                    body = " & ".join(parts_cond)
                    lines.append(f"-- When {state_name}: actuators match")
                    lines.append(f"LTLSPEC G({self.STATE_VAR} = {state_name} -> ({body}))")
                    lines.append("")

        # Safety: integer step-action targets stay within physical bounds
        for sa in self._step_actions:
            if not self._is_real(sa.target_key):
                smv_n = self._smv(sa.target_key)
                upper = self._capacity_for(sa.target_key)
                lines.append(f"-- {smv_n} never leaves its physical bounds")
                lines.append(f"LTLSPEC G({smv_n} >= 0 & {smv_n} <= {upper})")
                lines.append("")

        # Phase 1f: placeholder LTL spec per SM instance
        lines.append("-- TODO: verification properties (to be specified in SysML)")
        for inst_fqn in self._sm_instances:
            state_var = self._state_var(inst_fqn)
            lines.append(f"LTLSPEC G TRUE  -- placeholder for {state_var}")
        lines.append("")

        return lines

    # ------------------------------------------------------------------
    # Top-level generation
    # ------------------------------------------------------------------

    def generate(self) -> str:
        parts = [
            "-- Auto-generated nuXmv SMV model",
            f"-- Source: {self.parser.file_path}",
            "",
            "MODULE main",
        ]
        parts.extend(self._gen_var())

        ivar = self._gen_ivar()
        if ivar:
            parts.append("")
            parts.extend(ivar)

        parts.append("")
        parts.extend(self._gen_define())
        parts.append("")
        parts.extend(self._gen_assign())
        parts.append("")
        parts.extend(self._gen_specs())

        return "\n".join(parts) + "\n"


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Translate a SysML v2 model to nuXmv SMV for model checking."
    )
    ap.add_argument("model_file", help="Path to the SysML v2 model file")
    ap.add_argument("-o", "--output", help="Output SMV file (default: stdout)")
    args = ap.parse_args()

    if not Path(args.model_file).exists():
        print(f"Error: model file not found: {args.model_file}", file=sys.stderr)
        return 1

    try:
        parser = SysMLParser(args.model_file)
        parser.parse()
    except Exception as e:
        print(f"Error parsing model: {e}", file=sys.stderr)
        return 1

    gen = SMVGenerator(parser)
    smv = gen.generate()

    if args.output:
        Path(args.output).write_text(smv)
        print(f"Wrote {args.output}")
    else:
        print(smv, end="")

    return 0


if __name__ == "__main__":
    sys.exit(main())
