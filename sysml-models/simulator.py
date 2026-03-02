#!/usr/bin/env python3
"""
SysML v2 Model Simulator

Parses a SysML v2 model file and simulates its behavior.
All simulation logic is derived from the model - no hardcoded domain knowledge.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Optional
import warnings

warnings.filterwarnings("ignore")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

class BindRef:
    """Marks a state entry as an alias for another state key."""
    __slots__ = ('target_key',)

    def __init__(self, target_key: str):
        self.target_key = target_key

    def __repr__(self) -> str:
        return f"→{self.target_key}"


def resolve_value(state: dict, key: str) -> Any:
    """Follow any BindRef chain in state to return the underlying value."""
    seen: set = set()
    while True:
        val = state.get(key)
        if not isinstance(val, BindRef):
            return val
        if key in seen:
            return None  # cycle guard
        seen.add(key)
        key = val.target_key


def canonical_key(state: dict, key: str) -> str:
    """Follow the BindRef chain to the canonical (non-aliased) key."""
    seen: set = set()
    while isinstance(state.get(key), BindRef):
        if key in seen:
            return key  # cycle guard
        seen.add(key)
        key = state[key].target_key
    return key


from sysml_parser import (
    PYSYSML2_AVAILABLE,
    Expr, LiteralExpr, RefExpr, BinaryExpr, TernaryExpr, UnaryExpr,
    ExpressionParser,
    Constraint, DerivedAttribute, StepAction, Parameter,
    AssignStmt, ItemDeclStmt, SendStmt, AcceptStmt, AttributeDeclStmt, IfStmt, PerformStmt,
    Action, State, Transition, StateMachine, Flow, PartInstance, PartDef,
    SysMLParser,
)


# =============================================================================
# Expression Evaluator
# =============================================================================

class ExpressionEvaluator:
    """Evaluates parsed expressions against a state dictionary."""

    def __init__(self, state: dict[str, Any], context: str = "",
                 ref_bindings: dict[str, str] = None, system: str = ""):
        self.state = state
        self.context = context
        self.ref_bindings = ref_bindings or {}
        self.system = system

    def evaluate(self, expr: Expr) -> Any:
        if isinstance(expr, LiteralExpr):
            return expr.value

        elif isinstance(expr, RefExpr):
            return self._resolve_ref(expr.path)

        elif isinstance(expr, BinaryExpr):
            if expr.op == 'implies':
                left = self.evaluate(expr.left)
                if not left:
                    return True
                return self.evaluate(expr.right)

            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)

            # Handle None values
            if left is None:
                left = 0.0 if expr.op in ['+', '-', '*', '/', '>=', '<=', '>', '<'] else False
            if right is None:
                right = 0.0 if expr.op in ['+', '-', '*', '/', '>=', '<=', '>', '<'] else False

            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b if b != 0 else 0,
                '==': lambda a, b: a == b,
                '>=': lambda a, b: a >= b,
                '<=': lambda a, b: a <= b,
                '>': lambda a, b: a > b,
                '<': lambda a, b: a < b,
                'and': lambda a, b: a and b,
                'or': lambda a, b: a or b,
            }
            return ops[expr.op](left, right)

        elif isinstance(expr, TernaryExpr):
            cond = self.evaluate(expr.condition)
            return self.evaluate(expr.true_expr) if cond else self.evaluate(expr.false_expr)

        elif isinstance(expr, UnaryExpr):
            val = self.evaluate(expr.operand)
            if expr.op == 'not':
                return not val
            elif expr.op == '-':
                return -(val or 0)

        return None

    def _resolve_ref(self, path: list[str]) -> Any:
        """Resolve a reference path to a value in the state, following any BindRefs."""
        original_path = path.copy()
        dot_path = '.'.join(path)

        # Check if first element is a ref that needs to be resolved
        if len(path) >= 1 and self.ref_bindings and self.context:
            first = path[0]
            ref_key = f"{self.context}::{first}"
            if ref_key in self.ref_bindings:
                target = self.ref_bindings[ref_key]
                path = target.split('::') + path[1:]
                abs_key = '::'.join(path)
                if abs_key in self.state:
                    return resolve_value(self.state, abs_key)

        # Try with context prefix and dot notation (for behavior.state patterns)
        if self.context:
            ctx_dot_key = f"{self.context}::{dot_path}"
            if ctx_dot_key in self.state:
                return resolve_value(self.state, ctx_dot_key)

        # Try bare dot key
        if dot_path in self.state:
            return resolve_value(self.state, dot_path)

        # Try as absolute path with :: separators
        abs_key = '::'.join(path)
        if abs_key in self.state:
            return resolve_value(self.state, abs_key)

        # Try with context prefix and :: separators
        if self.context:
            full_key = f"{self.context}::" + '::'.join(original_path)
            if full_key in self.state:
                return resolve_value(self.state, full_key)

        # Try with system prefix
        if self.system and len(original_path) >= 1:
            sys_key = f"{self.system}::" + '::'.join(original_path)
            if sys_key in self.state:
                return resolve_value(self.state, sys_key)

        return None


# =============================================================================
# Constraint Solver
# =============================================================================

class ConstraintSolver:
    """Solves constraints by propagating values."""

    def __init__(self, state: dict[str, Any], system: str,
                 ref_bindings: dict[str, str] = None, flows: list = None,
                 instance_state_machines: dict = None):
        self.state = state
        self.system = system
        self.ref_bindings = ref_bindings or {}
        self.flows = flows or []
        self.instance_state_machines = instance_state_machines or {}
        self.constraints: list[Constraint] = []
        self.derived_attrs: list[DerivedAttribute] = []
        self.state_constraints: list[Constraint] = []
        self.assignment_constraints: list[Constraint] = []

    def add_constraint(self, constraint: Constraint):
        """Categorize and add a constraint."""
        if self._contains_implies(constraint.expression):
            self.state_constraints.append(constraint)
        else:
            self.assignment_constraints.append(constraint)
        self.constraints.append(constraint)

    def add_derived_attribute(self, attr: DerivedAttribute):
        self.derived_attrs.append(attr)

    def _contains_implies(self, expr: Expr) -> bool:
        if isinstance(expr, BinaryExpr):
            if expr.op == 'implies':
                return True
            return self._contains_implies(expr.left) or self._contains_implies(expr.right)
        return False

    def solve(self, current_sm_states: dict[str, str]):
        """Solve all constraints given current state machine states."""
        self._set_state_predicates(current_sm_states)

        # Solve derived attributes
        for attr in self.derived_attrs:
            evaluator = ExpressionEvaluator(self.state, attr.context,
                                            self.ref_bindings, self.system)
            value = evaluator.evaluate(attr.expression)
            self.state[attr.qualified_name] = value

        # Solve state-based constraints (implies)
        for constraint in self.state_constraints:
            self._solve_implies_constraint(constraint)

        # Solve assignment constraints with flow propagation
        for _ in range(3):
            self._propagate_flows()
            for constraint in self.assignment_constraints:
                self._solve_assignment_constraint(constraint)

    def _set_state_predicates(self, current_sm_states: dict[str, str]):
        """Set boolean predicates for state machine states, one per instance."""
        # Clear all behavior predicates
        for key in list(self.state.keys()):
            if 'behavior.' in key:
                self.state[key] = False

        for inst_fqn, current_state in current_sm_states.items():
            sm = self.instance_state_machines.get(inst_fqn)
            if sm:
                # Per-instance predicates: {fqn}::behavior.{state}
                for state in sm.states:
                    self.state[f"{inst_fqn}::behavior.{state.name}"] = (state.name == current_state)
            # Global fallback for constraints that reference bare 'behavior.X'
            self.state[f"behavior.{current_state}"] = True

    def _solve_implies_constraint(self, constraint: Constraint):
        """Solve a constraint of form: condition implies assignments."""
        expr = constraint.expression

        # Handle conjunction of implies constraints: (A implies B) and (C implies D)
        if isinstance(expr, BinaryExpr) and expr.op == 'and':
            for sub_expr in (expr.left, expr.right):
                sub = Constraint(constraint.name, sub_expr, constraint.raw_text, constraint.context)
                self._solve_implies_constraint(sub)
            return

        if not isinstance(expr, BinaryExpr) or expr.op != 'implies':
            return

        evaluator = ExpressionEvaluator(self.state, constraint.context,
                                        self.ref_bindings, self.system)
        condition = evaluator.evaluate(expr.left)
        if not condition:
            return

        self._apply_assignments(expr.right, constraint.context)

    def _apply_assignments(self, expr: Expr, context: str):
        """Extract and apply assignments from an expression."""
        if isinstance(expr, BinaryExpr):
            if expr.op == 'and':
                self._apply_assignments(expr.left, context)
                self._apply_assignments(expr.right, context)
            elif expr.op == '==':
                if isinstance(expr.left, RefExpr):
                    evaluator = ExpressionEvaluator(self.state, context,
                                                    self.ref_bindings, self.system)
                    value = evaluator.evaluate(expr.right)
                    key = self._resolve_ref_to_key(expr.left.path, context)
                    if key:
                        self.state[canonical_key(self.state, key)] = value

    def _solve_assignment_constraint(self, constraint: Constraint):
        """Solve a constraint of form: ref == expr."""
        expr = constraint.expression
        if not isinstance(expr, BinaryExpr) or expr.op != '==':
            return

        evaluator = ExpressionEvaluator(self.state, constraint.context,
                                        self.ref_bindings, self.system)

        if isinstance(expr.left, RefExpr):
            value = evaluator.evaluate(expr.right)
            key = self._resolve_ref_to_key(expr.left.path, constraint.context)
            if key:
                self.state[canonical_key(self.state, key)] = value

    def _resolve_ref_to_key(self, path: list[str], context: str) -> Optional[str]:
        """Convert a reference path to a state key."""
        if len(path) >= 1 and self.ref_bindings and context:
            first = path[0]
            ref_key = f"{context}::{first}"
            if ref_key in self.ref_bindings:
                target = self.ref_bindings[ref_key]
                resolved_path = target.split('::') + path[1:]
                return '::'.join(resolved_path)

        if context:
            return f"{context}::" + '::'.join(path)
        return '::'.join(path)

    def _propagate_flows(self):
        """Propagate all port attribute values through flow connections."""
        for flow in self.flows:
            from_prefix = f"{self.system}::{flow.from_port.replace('.', '::')}::"
            to_prefix = f"{self.system}::{flow.to_port.replace('.', '::')}::"
            propagations = [
                (key, to_prefix + key[len(from_prefix):])
                for key in self.state
                if key.startswith(from_prefix)
            ]
            for from_key, to_key in propagations:
                from_val = resolve_value(self.state, from_key)
                if from_val is not None:
                    self.state[canonical_key(self.state, to_key)] = from_val


# =============================================================================
# Simulation Engine
# =============================================================================

class SimulationEngine:
    """Generic simulation engine driven by parsed SysML model."""

    def __init__(self, parser: SysMLParser):
        self.parser = parser
        self.state: dict[str, Any] = {}
        self.current_sm_state: dict[str, str] = {}
        self.solver: Optional[ConstraintSolver] = None
        self.port_mailboxes: dict[str, list] = {}
        self.time = 0.0

    def initialize(self, overrides: dict[str, float] = None) -> None:
        overrides = overrides or {}

        # Initialize state from parameters
        for param in self.parser.parameters:
            key = param.qualified_name
            if param.cli_name in overrides:
                self.state[key] = overrides[param.cli_name]
            elif param.qualified_name in overrides:
                self.state[key] = overrides[param.qualified_name]
            else:
                self.state[key] = param.value

        # Initialize state machines — one entry per part instance
        for fqn, sm in self.parser.instance_state_machines.items():
            self.current_sm_state[fqn] = sm.initial_state

        # Initialize boolean and real attributes from part definitions
        for fqn, inst in self.parser.part_instances.items():
            part_def = self.parser.part_defs.get(inst.part_type)
            if part_def:
                for attr_name, attr_type in part_def.attributes.items():
                    key = f"{fqn}::{attr_name}"
                    if key not in self.state:
                        if attr_type == 'Boolean':
                            self.state[key] = False
                        elif attr_type == 'Real':
                            self.state[key] = 0.0

        # Initialize per-instance behavior state predicates
        for fqn, sm in self.parser.instance_state_machines.items():
            for s in sm.states:
                self.state[f"{fqn}::behavior.{s.name}"] = (s.name == sm.initial_state)

        # Populate BindRef entries for all bind statements
        for lhs_key, rhs_key in self.parser.parsed_bindings.items():
            self.state[lhs_key] = BindRef(rhs_key)

        # Pre-initialize transition do-action target keys so they exist for flow propagation
        for fqn, sm in self.parser.instance_state_machines.items():
            for trans in sm.transitions:
                if trans.do_action:
                    for stmt in trans.do_action:
                        if isinstance(stmt, AssignStmt):
                            key = f"{fqn}::" + "::".join(stmt.target)
                            if key not in self.state:
                                self.state[key] = 0.0

        # Create constraint solver
        self.solver = ConstraintSolver(self.state, self.parser.system_part,
                                       self.parser.ref_bindings, self.parser.flows,
                                       self.parser.instance_state_machines)
        for constraint in self.parser.parsed_constraints:
            self.solver.add_constraint(constraint)
        for attr in self.parser.derived_attributes:
            self.solver.add_derived_attribute(attr)

        # Initial constraint solving
        self.solver.solve(self.current_sm_state)

    def step(self, dt: float) -> None:
        if not self.parser.system_part:
            return

        # Process state machine transitions
        self._process_state_machines()

        # Solve constraints
        self.solver.solve(self.current_sm_state)

        # Execute step action bodies (assigns, sends, if/perform, accept)
        self.state['dt'] = dt
        for fqn, stmts in self.parser.step_action_bodies:
            self._execute_action_stmts(stmts, fqn)

        self.time += dt

    def _process_state_machine(self, inst_fqn: str) -> bool:
        """Try to fire one transition for the given instance. Returns True if fired."""
        sm = self.parser.instance_state_machines.get(inst_fqn)
        if not sm:
            return False
        current = self.current_sm_state.get(inst_fqn)

        for trans in sm.transitions:
            if trans.from_state != current:
                continue

            # Check trigger
            matched_item = None
            if trans.trigger:
                if trans.trigger_port:
                    # Item-based trigger: look in the port's mailbox
                    port_key = f"{inst_fqn}::{trans.trigger_port.replace('.', '::')}"
                    for item in self.port_mailboxes.get(port_key, []):
                        if self._is_subtype(item['type'], trans.trigger):
                            matched_item = item
                            break
                    if matched_item is None:
                        continue
                    # Expose trigger_var attributes in state for guard evaluation
                    if trans.trigger_var:
                        for attr_path, val in matched_item['attrs'].items():
                            state_key = (f"{inst_fqn}::{trans.trigger_var}::"
                                         f"{attr_path.replace('.', '::')}")
                            self.state[state_key] = val
                else:
                    # Legacy state-flag trigger
                    trigger_key = f"{inst_fqn}::{trans.trigger}"
                    if not self.state.get(trigger_key, False):
                        continue
                    self.state[trigger_key] = False

            # Check guard
            if trans.guard is not None:
                evaluator = ExpressionEvaluator(self.state, inst_fqn,
                                                self.parser.ref_bindings,
                                                self.parser.system_part)
                if not evaluator.evaluate(trans.guard):
                    continue

            # Execute do action
            if trans.do_action:
                self._execute_action_stmts(trans.do_action, inst_fqn)

            # Consume matched item from mailbox
            if matched_item is not None:
                port_key = f"{inst_fqn}::{trans.trigger_port.replace('.', '::')}"
                mailbox = self.port_mailboxes.get(port_key, [])
                if matched_item in mailbox:
                    mailbox.remove(matched_item)

            # Perform transition
            self.current_sm_state[inst_fqn] = trans.to_state
            return True
        return False

    def _process_state_machines(self) -> None:
        for inst_fqn in self.parser.instance_state_machines:
            self._process_state_machine(inst_fqn)

    def _is_subtype(self, type_name: str, expected_type: str) -> bool:
        """Return True if type_name is expected_type or descends from it via :>."""
        t: Optional[str] = type_name
        while t is not None:
            if t == expected_type:
                return True
            t = self.parser.item_type_parents.get(t)
        return False

    def _find_connected_port(self, port_key: str) -> Optional[str]:
        """Return the port key connected to port_key via a connect statement.

        Handles sub-ports: if port_key is system::controller::volumeSensor1Port::req
        and a connect maps controller.volumeSensor1Port <-> volumeSensor1.modbusPort,
        returns system::volumeSensor1::modbusPort::req (suffix transferred).
        """
        system = self.parser.system_part
        for from_port, to_port in self.parser.connects:
            from_key = f"{system}::{from_port.replace('.', '::')}"
            to_key = f"{system}::{to_port.replace('.', '::')}"
            if port_key == from_key:
                return to_key
            if port_key == to_key:
                return from_key
            # Check if port_key is a sub-port of either side
            if port_key.startswith(from_key + "::"):
                suffix = port_key[len(from_key):]
                return to_key + suffix
            if port_key.startswith(to_key + "::"):
                suffix = port_key[len(to_key):]
                return from_key + suffix
        return None

    def _send_item(self, type_name: str, attrs: dict,
                   sender_fqn: str, port_name: str) -> Optional[str]:
        """Deliver an item to the port connected to sender_fqn::port_name.

        Single-slot-per-type: a new item replaces any existing item of the same
        type in the destination mailbox, so the consumer always sees the latest value.
        Returns the destination port key (so caller can trigger recipient SM), or None.
        """
        sender_port_key = f"{sender_fqn}::{port_name.replace('.', '::')}"
        dest_port_key = self._find_connected_port(sender_port_key)
        if dest_port_key:
            mailbox = self.port_mailboxes.setdefault(dest_port_key, [])
            new_item = {'type': type_name, 'attrs': attrs}
            for i, existing in enumerate(mailbox):
                if existing['type'] == type_name:
                    mailbox[i] = new_item
                    return dest_port_key
            mailbox.append(new_item)
            return dest_port_key
        return None

    def _find_action(self, context: str, action_name: str):
        """Look up a named action from the part def of the given context."""
        inst = self.parser.part_instances.get(context)
        if not inst:
            return None
        pdef = self.parser.part_defs.get(inst.part_type)
        if not pdef:
            return None
        for action in pdef.actions:
            if action.name == action_name:
                return action
        return None

    def _execute_action_stmts(self, stmts: list, context: str, local_items: dict = None) -> None:
        """Evaluate and apply a list of ActionStmt nodes in the given part context."""
        # Use passed-in locals (for recursive calls) or create fresh
        if local_items is None:
            local_items = {}

        # Collect local item declarations first
        for stmt in stmts:
            if isinstance(stmt, ItemDeclStmt):
                local_items[stmt.name] = {'type': stmt.type_name, 'attrs': {}}

        evaluator = ExpressionEvaluator(self.state, context,
                                        self.parser.ref_bindings,
                                        self.parser.system_part)
        for stmt in stmts:
            if isinstance(stmt, AssignStmt):
                if local_items and stmt.target[0] in local_items:
                    # Local item attribute — track in item dict and expose in state
                    attr_key = '.'.join(stmt.target[1:])
                    value = evaluator.evaluate(stmt.expr)
                    if value is not None:
                        local_items[stmt.target[0]]['attrs'][attr_key] = value
                        state_key = f"{context}::" + "::".join(stmt.target)
                        self.state[state_key] = value
                else:
                    key = f"{context}::" + "::".join(stmt.target)
                    value = evaluator.evaluate(stmt.expr)
                    if value is not None:
                        ckey = canonical_key(self.state, key)
                        cap = self._capacity_for(ckey)
                        if cap is not None:
                            value = min(cap, value)
                        self.state[ckey] = value
            elif isinstance(stmt, SendStmt):
                if stmt.item_name in local_items:
                    item = local_items[stmt.item_name]
                    dest = self._send_item(item['type'], item['attrs'], context, stmt.port)
                    # Trigger recipient SM so it can process the message immediately
                    if dest:
                        # dest is like "system::volumeSensor1::modbusPort::req"
                        # extract the instance FQN (everything before the port path)
                        for inst_fqn in self.parser.instance_state_machines:
                            if dest.startswith(inst_fqn + "::"):
                                self._process_state_machine(inst_fqn)
                                break
            elif isinstance(stmt, IfStmt):
                cond_val = evaluator.evaluate(stmt.condition)
                if cond_val:
                    self._execute_action_stmts(stmt.body, context, local_items)
            elif isinstance(stmt, PerformStmt):
                action = self._find_action(context, stmt.action_name)
                if action:
                    self._execute_action_stmts(action.body, context, local_items)
            elif isinstance(stmt, AcceptStmt):
                port_key = f"{context}::{stmt.port.replace('.', '::')}"
                mailbox = self.port_mailboxes.get(port_key, [])
                for i, item in enumerate(mailbox):
                    if self._is_subtype(item['type'], stmt.type_name):
                        if stmt.var_name:
                            local_items[stmt.var_name] = item
                            # Expose item attrs in state so expressions can resolve them
                            for attr_path, val in item.get('attrs', {}).items():
                                state_key = f"{context}::{stmt.var_name}::{attr_path.replace('.', '::')}"
                                self.state[state_key] = val
                        mailbox.pop(i)
                        break
            elif isinstance(stmt, AttributeDeclStmt):
                if stmt.init_expr:
                    value = evaluator.evaluate(stmt.init_expr)
                    if value is not None:
                        self.state[f"{context}::{stmt.name}"] = value

    def _execute_step_sends(self, dt: float) -> None:
        """Send items produced by step action bodies (e.g. thermometer reading)."""
        self.state['dt'] = dt
        for fqn, stmts in self.parser.step_action_bodies:
            # Only process bodies that actually declare and send items
            local_items: dict[str, dict] = {}
            for stmt in stmts:
                if isinstance(stmt, ItemDeclStmt):
                    local_items[stmt.name] = {'type': stmt.type_name, 'attrs': {}}
            if not local_items:
                continue
            evaluator = ExpressionEvaluator(self.state, fqn,
                                            self.parser.ref_bindings,
                                            self.parser.system_part)
            for stmt in stmts:
                if isinstance(stmt, AssignStmt) and stmt.target[0] in local_items:
                    attr_key = '.'.join(stmt.target[1:])
                    value = evaluator.evaluate(stmt.expr)
                    if value is not None:
                        local_items[stmt.target[0]]['attrs'][attr_key] = value
                elif isinstance(stmt, SendStmt) and stmt.item_name in local_items:
                    item = local_items[stmt.item_name]
                    self._send_item(item['type'], item['attrs'], fqn, stmt.port)

    def _capacity_for(self, target_key: str) -> Optional[float]:
        """Find the capacity bound for a step-action target variable."""
        part_key = target_key.rsplit('::', 1)[0]
        cap_key = f"{part_key}::capacity"
        for param in self.parser.parameters:
            if param.qualified_name == cap_key:
                return param.value
        return None

    def _apply_step_actions(self, dt: float) -> None:
        """Evaluate and apply all owned step actions for one timestep."""
        self.state['dt'] = dt
        for sa in self.parser.step_actions:
            evaluator = ExpressionEvaluator(self.state, sa.context,
                                            self.parser.ref_bindings,
                                            self.parser.system_part)
            new_value = evaluator.evaluate(sa.expression)
            if new_value is not None:
                cap = self._capacity_for(sa.target_key)
                if cap is not None:
                    new_value = min(cap, new_value)
                self.state[canonical_key(self.state, sa.target_key)] = new_value

    def get_status(self) -> dict:
        # Report the controller's state; fall back to any instance
        ctrl = self.parser.controller_part
        if ctrl and ctrl in self.current_sm_state:
            sm_state = self.current_sm_state[ctrl]
        elif self.current_sm_state:
            sm_state = next(iter(self.current_sm_state.values()))
        else:
            sm_state = "unknown"

        # Collect state variables - those updated by step actions
        state_vars = {}
        for sa in self.parser.step_actions:
            key = sa.target_key
            if key in self.state:
                # Create a short display name from the key
                parts = key.split('::')
                if len(parts) >= 2:
                    display_name = f"{parts[-2]}.{parts[-1]}"
                else:
                    display_name = parts[-1]
                state_vars[display_name] = self.state[key]

        # Find flow rate from any port
        flow_rate = 0.0
        for key, val in self.state.items():
            if '::flowRate' in key and val:
                flow_rate = val
                break

        return {
            "time": self.time,
            "state": sm_state,
            "state_vars": state_vars,
            "flow_rate": flow_rate,
        }

    def issue_command(self, command: str) -> None:
        ctrl = self.parser.controller_part
        key = f"{ctrl}::{command}" if ctrl else command
        self.state[key] = True


# =============================================================================
# CLI and Main
# =============================================================================

def load_config(config_path: Path) -> dict:
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for config files.")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_argument_parser(parser: SysMLParser) -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        description="Simulate SysML model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    argparser.add_argument("-m", "--model-file", default="model.sysml",
                          help="Path to SysML model file (default: model.sysml)")
    argparser.add_argument("-c", "--config", type=Path,
                          help="Path to YAML configuration file")
    argparser.add_argument("-t", "--timestep", type=float, default=0.1,
                          help="Simulation timestep in seconds (default: 0.1)")
    argparser.add_argument("-d", "--duration", type=float, default=20.0,
                          help="Simulation duration in seconds (default: 20.0)")
    argparser.add_argument("-o", "--output-interval", type=float, default=1.0,
                          help="Output interval in seconds (default: 1.0)")
    argparser.add_argument("-s", "--initial-state",
                          help="Initial state machine state")
    argparser.add_argument("-q", "--quiet", action="store_true",
                          help="Suppress step-by-step output")
    argparser.add_argument("--list-parameters", action="store_true",
                          help="List all configurable parameters and exit")

    param_group = argparser.add_argument_group("Model Parameters")
    for param in parser.parameters:
        param_group.add_argument(
            f"--{param.cli_name}", type=float, metavar="VALUE",
            help=f"{param.qualified_name} (default: {param.value})"
        )

    return argparser


def run_simulation(engine: SimulationEngine, parser: SysMLParser,
                   timestep: float, duration: float,
                   output_interval: float, quiet: bool) -> list[dict]:
    history = []
    steps = int(duration / timestep)
    output_steps = max(1, int(output_interval / timestep))

    # Issue first trigger from state machine (to start the simulation)
    triggers = parser.get_triggers()
    if triggers:
        engine.issue_command(triggers[0])

    for step in range(steps + 1):
        if step % output_steps == 0:
            status = engine.get_status()
            history.append(status)

            if not quiet:
                vars_str = " | ".join(f"{k}={v:7.1f}" for k, v in status['state_vars'].items())
                print(
                    f"t={status['time']:6.1f}s | "
                    f"state={status['state']:8s} | "
                    f"{vars_str} | "
                    f"flow={status['flow_rate']:5.1f}"
                )

        if step < steps:
            engine.step(timestep)

    return history


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-m", "--model-file", default=None)
    pre_parser.add_argument("-c", "--config", type=Path)
    pre_parser.add_argument("--list-parameters", action="store_true")
    pre_args, remaining = pre_parser.parse_known_args()

    model_path = pre_args.model_file
    if model_path is None:
        for arg in remaining:
            if not arg.startswith('-') and (arg.endswith('.sysml') or Path(arg).exists()):
                model_path = arg
                break
        if model_path is None:
            model_path = "model.sysml"

    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    try:
        parser = SysMLParser(model_path)
        parser.parse()
    except Exception as e:
        print(f"Error parsing model: {e}", file=sys.stderr)
        return 1

    argparser = build_argument_parser(parser)
    args = argparser.parse_args()

    if args.list_parameters:
        print("Configurable parameters:\n")
        for param in parser.parameters:
            print(f"  --{param.cli_name}")
            print(f"      {param.qualified_name}")
            print(f"      Default: {param.value}\n")
        return 0

    config = {}
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = load_config(args.config)

    overrides = {}
    for key, value in config.items():
        if key not in ('timestep', 'duration', 'output_interval', 'initial_state'):
            overrides[key] = value

    for param in parser.parameters:
        cli_value = getattr(args, param.cli_name.replace('-', '_'), None)
        if cli_value is not None:
            overrides[param.cli_name] = cli_value

    engine = SimulationEngine(parser)
    engine.initialize(overrides)

    if args.initial_state:
        valid = {s.name for sm in parser.state_machines for s in sm.states}
        if args.initial_state not in valid:
            print(f"Error: Invalid state '{args.initial_state}'. Valid: {sorted(valid)}", file=sys.stderr)
            return 1
        for fqn, sm in parser.instance_state_machines.items():
            if args.initial_state in [s.name for s in sm.states]:
                engine.current_sm_state[fqn] = args.initial_state
        engine.solver.solve(engine.current_sm_state)

    timestep = config.get('timestep', args.timestep)
    duration = config.get('duration', args.duration)
    output_interval = config.get('output_interval', args.output_interval)

    if not args.quiet:
        print("=" * 70)
        print(f"SysML Model Simulation: {model_path}")
        print("=" * 70)
        print(f"System: {parser.system_part}")
        print(f"Controller: {parser.controller_part}")
        print(f"State machines: {[sm.name for sm in parser.state_machines]}")
        ctrl = parser.controller_part
        init_state = (engine.current_sm_state.get(ctrl)
                      if ctrl else next(iter(engine.current_sm_state.values()), 'N/A'))
        print(f"Initial state: {init_state}")
        print(f"\nStep actions: {len(parser.step_actions)}")
        for sa in parser.step_actions:
            print(f"  {sa.action_name}({sa.target_key}) := ...")
        print(f"\nParameters:")
        for param in parser.parameters:
            value = engine.state.get(param.qualified_name, param.value)
            print(f"  {param.qualified_name}: {value}")
        print(f"\nTimestep: {timestep}s, Duration: {duration}s")
        print("=" * 70 + "\n")

    run_simulation(engine, parser, timestep, duration, output_interval, args.quiet)

    if not args.quiet:
        print("\n" + "=" * 70)
        print("Simulation complete")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
