#!/usr/bin/env python3
"""
SysML v2 Model Simulator

Parses a SysML v2 model file and simulates its behavior.
All simulation logic is derived from the model - no hardcoded domain knowledge.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import warnings

warnings.filterwarnings("ignore")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from pysysml2.modeling.model import Model as PySysML2Model
    PYSYSML2_AVAILABLE = True
except ImportError:
    PYSYSML2_AVAILABLE = False


# =============================================================================
# Expression AST
# =============================================================================

@dataclass
class Expr:
    """Base class for expressions."""
    pass


@dataclass
class LiteralExpr(Expr):
    value: Any


@dataclass
class RefExpr(Expr):
    path: list[str]


@dataclass
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class TernaryExpr(Expr):
    condition: Expr
    true_expr: Expr
    false_expr: Expr


@dataclass
class UnaryExpr(Expr):
    op: str
    operand: Expr


@dataclass
class DerExpr(Expr):
    """Derivative expression: der(variable)"""
    variable: RefExpr


# =============================================================================
# Expression Parser
# =============================================================================

class ExpressionParser:
    """Parses SysML constraint expressions into an AST."""

    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> Expr:
        expr = self._parse_implies()
        self._skip_whitespace()
        return expr

    def _skip_whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n':
            self.pos += 1

    def _peek(self, n: int = 1) -> str:
        return self.text[self.pos:self.pos + n]

    def _consume(self, expected: str) -> bool:
        self._skip_whitespace()
        if self.text[self.pos:self.pos + len(expected)] == expected:
            self.pos += len(expected)
            return True
        return False

    def _parse_implies(self) -> Expr:
        left = self._parse_or()
        self._skip_whitespace()
        if self._consume('implies'):
            right = self._parse_implies()
            return BinaryExpr('implies', left, right)
        return left

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while True:
            self._skip_whitespace()
            if self._consume('or'):
                right = self._parse_and()
                left = BinaryExpr('or', left, right)
            else:
                break
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_equality()
        while True:
            self._skip_whitespace()
            if self._consume('and'):
                right = self._parse_equality()
                left = BinaryExpr('and', left, right)
            else:
                break
        return left

    def _parse_equality(self) -> Expr:
        left = self._parse_comparison()
        self._skip_whitespace()
        if self._consume('=='):
            right = self._parse_comparison()
            return BinaryExpr('==', left, right)
        return left

    def _parse_comparison(self) -> Expr:
        left = self._parse_additive()
        self._skip_whitespace()
        for op in ['>=', '<=', '>', '<']:
            if self._consume(op):
                right = self._parse_additive()
                return BinaryExpr(op, left, right)
        return left

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while True:
            self._skip_whitespace()
            if self._consume('+'):
                right = self._parse_multiplicative()
                left = BinaryExpr('+', left, right)
            elif self._consume('-'):
                right = self._parse_multiplicative()
                left = BinaryExpr('-', left, right)
            else:
                break
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_unary()
        while True:
            self._skip_whitespace()
            if self._consume('*'):
                right = self._parse_unary()
                left = BinaryExpr('*', left, right)
            elif self._consume('/'):
                right = self._parse_unary()
                left = BinaryExpr('/', left, right)
            else:
                break
        return left

    def _parse_unary(self) -> Expr:
        self._skip_whitespace()
        if self._consume('not '):
            return UnaryExpr('not', self._parse_unary())
        if self._consume('-'):
            return UnaryExpr('-', self._parse_unary())
        return self._parse_ternary()

    def _parse_ternary(self) -> Expr:
        self._skip_whitespace()
        if self._peek() == '(':
            start_pos = self.pos
            self.pos += 1
            inner = self._parse_implies()
            self._skip_whitespace()
            if self._consume('?'):
                true_expr = self._parse_implies()
                self._skip_whitespace()
                if self._consume(':'):
                    false_expr = self._parse_implies()
                    self._skip_whitespace()
                    self._consume(')')
                    return TernaryExpr(inner, true_expr, false_expr)
            self.pos = start_pos
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        self._skip_whitespace()

        # Parenthesized expression
        if self._consume('('):
            expr = self._parse_implies()
            self._skip_whitespace()
            self._consume(')')
            return expr

        # der() function for derivatives
        if self._consume('der('):
            var_match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*', self.text[self.pos:])
            if var_match:
                self.pos += len(var_match.group())
                var_path = var_match.group().split('.')
                self._skip_whitespace()
                self._consume(')')
                return DerExpr(RefExpr(var_path))

        # Boolean literals
        if self._consume('true'):
            return LiteralExpr(True)
        if self._consume('false'):
            return LiteralExpr(False)

        # Numeric literal
        match = re.match(r'[-]?\d+\.?\d*', self.text[self.pos:])
        if match:
            self.pos += len(match.group())
            val = match.group()
            return LiteralExpr(float(val) if '.' in val else int(val))

        # Reference (identifier with optional dots)
        match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*', self.text[self.pos:])
        if match:
            self.pos += len(match.group())
            path = match.group().split('.')
            return RefExpr(path)

        raise ValueError(f"Unexpected token at position {self.pos}: {self.text[self.pos:self.pos+20]}")


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

        elif isinstance(expr, DerExpr):
            # der() expressions are handled specially by the dynamics solver
            return None

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
        """Resolve a reference path to a value in the state."""
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
                    return self.state[abs_key]

        # Try with context prefix and dot notation (for behavior.state patterns)
        if self.context:
            ctx_dot_key = f"{self.context}::{dot_path}"
            if ctx_dot_key in self.state:
                return self.state[ctx_dot_key]

        # Try bare dot key
        if dot_path in self.state:
            return self.state[dot_path]

        # Try as absolute path with :: separators
        abs_key = '::'.join(path)
        if abs_key in self.state:
            return self.state[abs_key]

        # Try with context prefix and :: separators
        if self.context:
            full_key = f"{self.context}::" + '::'.join(original_path)
            if full_key in self.state:
                return self.state[full_key]

        # Try with system prefix
        if self.system and len(original_path) >= 1:
            sys_key = f"{self.system}::" + '::'.join(original_path)
            if sys_key in self.state:
                return self.state[sys_key]

        return None


# =============================================================================
# Model Data Structures
# =============================================================================

@dataclass
class Constraint:
    """A parsed constraint with its expression AST."""
    name: str
    expression: Expr
    raw_text: str
    context: str


@dataclass
class DerivedAttribute:
    """An attribute with a derived value expression."""
    name: str
    qualified_name: str
    expression: Expr
    context: str


@dataclass
class DynamicsEquation:
    """A differential equation: der(variable) == expression."""
    variable_path: list[str]  # The variable being differentiated
    variable_key: str  # Fully qualified state key
    expression: Expr  # The RHS expression
    context: str  # Part context for evaluation


@dataclass
class Parameter:
    """A configurable parameter extracted from the model."""
    name: str
    qualified_name: str
    value: float
    part_path: list[str]
    cli_name: str = ""

    def __post_init__(self):
        parts = self.qualified_name.split('::')
        if len(parts) > 1:
            parts = parts[1:]
        name = "-".join(parts)
        name = re.sub(r'([a-z])([A-Z])', r'\1-\2', name).lower()
        self.cli_name = name


@dataclass
class State:
    name: str


@dataclass
class Transition:
    name: str
    from_state: str
    to_state: str
    trigger: Optional[str] = None
    guard: Optional[str] = None


@dataclass
class StateMachine:
    name: str
    states: list[State] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    initial_state: Optional[str] = None


@dataclass
class Flow:
    item_type: str
    from_port: str
    to_port: str


@dataclass
class PartInstance:
    name: str
    part_type: str
    attributes: dict[str, float] = field(default_factory=dict)
    parent: Optional[str] = None


@dataclass
class PartDef:
    """A part definition with its attributes and constraints."""
    name: str
    attributes: dict[str, str] = field(default_factory=dict)
    derived_attributes: dict[str, str] = field(default_factory=dict)
    constraints: list[tuple[str, str]] = field(default_factory=list)
    refs: dict[str, str] = field(default_factory=dict)
    exhibits_state: Optional[str] = None  # State machine name if this part exhibits one


# =============================================================================
# Constraint Solver
# =============================================================================

class ConstraintSolver:
    """Solves constraints by propagating values."""

    def __init__(self, state: dict[str, Any], system: str,
                 ref_bindings: dict[str, str] = None, flows: list = None):
        self.state = state
        self.system = system
        self.ref_bindings = ref_bindings or {}
        self.flows = flows or []
        self.constraints: list[Constraint] = []
        self.derived_attrs: list[DerivedAttribute] = []
        self.dynamics: list[DynamicsEquation] = []
        self.state_constraints: list[Constraint] = []
        self.assignment_constraints: list[Constraint] = []

    def add_constraint(self, constraint: Constraint):
        """Categorize and add a constraint."""
        # Check if it's a dynamics constraint (contains der())
        if self._contains_der(constraint.expression):
            self._extract_dynamics(constraint)
        elif self._contains_implies(constraint.expression):
            self.state_constraints.append(constraint)
        else:
            self.assignment_constraints.append(constraint)
        self.constraints.append(constraint)

    def add_derived_attribute(self, attr: DerivedAttribute):
        self.derived_attrs.append(attr)

    def _contains_der(self, expr: Expr) -> bool:
        if isinstance(expr, DerExpr):
            return True
        if isinstance(expr, BinaryExpr):
            return self._contains_der(expr.left) or self._contains_der(expr.right)
        if isinstance(expr, TernaryExpr):
            return (self._contains_der(expr.condition) or
                    self._contains_der(expr.true_expr) or
                    self._contains_der(expr.false_expr))
        if isinstance(expr, UnaryExpr):
            return self._contains_der(expr.operand)
        return False

    def _contains_implies(self, expr: Expr) -> bool:
        if isinstance(expr, BinaryExpr):
            if expr.op == 'implies':
                return True
            return self._contains_implies(expr.left) or self._contains_implies(expr.right)
        return False

    def _extract_dynamics(self, constraint: Constraint):
        """Extract a dynamics equation from a constraint like der(x) == expr."""
        expr = constraint.expression
        if isinstance(expr, BinaryExpr) and expr.op == '==':
            if isinstance(expr.left, DerExpr):
                var_path = expr.left.variable.path
                var_key = f"{constraint.context}::" + '::'.join(var_path)
                self.dynamics.append(DynamicsEquation(
                    variable_path=var_path,
                    variable_key=var_key,
                    expression=expr.right,
                    context=constraint.context
                ))
            elif isinstance(expr.right, DerExpr):
                var_path = expr.right.variable.path
                var_key = f"{constraint.context}::" + '::'.join(var_path)
                self.dynamics.append(DynamicsEquation(
                    variable_path=var_path,
                    variable_key=var_key,
                    expression=expr.left,
                    context=constraint.context
                ))

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

    def integrate_dynamics(self, dt: float):
        """Integrate all dynamics equations by one timestep."""
        # First evaluate all derivatives
        derivatives = {}
        for dyn in self.dynamics:
            evaluator = ExpressionEvaluator(self.state, dyn.context,
                                            self.ref_bindings, self.system)
            derivative = evaluator.evaluate(dyn.expression)
            if derivative is not None:
                derivatives[dyn.variable_key] = derivative

        # Then update all state variables (Euler integration)
        for var_key, derivative in derivatives.items():
            current = self.state.get(var_key, 0.0)
            if current is not None:
                new_value = current + derivative * dt
                # Clamp to non-negative (physical constraint for levels/amounts)
                new_value = max(0.0, new_value)
                self.state[var_key] = new_value

    def _set_state_predicates(self, current_sm_states: dict[str, str]):
        """Set boolean predicates for state machine states."""
        # Clear all behavior predicates
        for key in list(self.state.keys()):
            if 'behavior.' in key:
                self.state[key] = False

        # Set current state predicates to true
        for sm_name, current_state in current_sm_states.items():
            self.state[f"behavior.{current_state}"] = True
            # Also set with any context prefixes found in state
            for key in list(self.state.keys()):
                if key.endswith(f"::behavior.{current_state}"):
                    self.state[key] = True
                elif f"::behavior." in key:
                    # Set this state's predicate based on whether it matches
                    state_name = key.split('::behavior.')[-1]
                    self.state[key] = (state_name == current_state)

    def _solve_implies_constraint(self, constraint: Constraint):
        """Solve a constraint of form: condition implies assignments."""
        expr = constraint.expression
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
                        self.state[key] = value

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
                self.state[key] = value

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
        """Propagate flow values through connections."""
        for flow in self.flows:
            from_key = f"{self.system}::{flow.from_port.replace('.', '::')}::flowRate"
            to_key = f"{self.system}::{flow.to_port.replace('.', '::')}::flowRate"
            from_val = self.state.get(from_key, 0.0)
            self.state[to_key] = from_val


# =============================================================================
# SysML Parser
# =============================================================================

class SysMLParser:
    """Parses SysML v2 files to extract simulation-relevant information."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = Path(file_path).read_text()

        self.parameters: list[Parameter] = []
        self.state_machines: list[StateMachine] = []
        self.parsed_constraints: list[Constraint] = []
        self.derived_attributes: list[DerivedAttribute] = []
        self.flows: list[Flow] = []
        self.part_defs: dict[str, PartDef] = {}
        self.part_instances: dict[str, PartInstance] = {}
        self.system_part: Optional[str] = None
        self.system_type: Optional[str] = None  # The part def type of system_part
        self.ref_bindings: dict[str, str] = {}
        self.controller_part: Optional[str] = None  # Part that exhibits state machine

    def parse(self) -> None:
        self._parse_with_pysysml2()
        self._parse_part_definitions()
        self._parse_state_machines()
        self._identify_system_part()
        self._parse_part_instances_regex()  # Fallback for instances PySysML2 missed
        self._parse_ref_bindings()
        self._parse_flows()
        self._find_controller()
        self._build_constraints()

    def _parse_with_pysysml2(self) -> None:
        if not PYSYSML2_AVAILABLE:
            raise ImportError("PySysML2 is required.")

        import io, contextlib
        with contextlib.redirect_stderr(io.StringIO()):
            model = PySysML2Model()
            model.from_sysml2_file(self.file_path)

        raw_data = model.sysml2_visitor.model_table_dict

        for idx, elem in raw_data.items():
            name = elem.get('name', '').split('@')[0]
            keywords = elem.get('keywords', [])
            fqn = elem.get('fully_qualified_name', '')
            constants = elem.get('constants', [])

            if 'part' in keywords:
                if 'specializes' in keywords:
                    related = elem.get('related_element_name', '').split('@')[0]
                    parent_idx = elem.get('idx_parent')
                    parent_name = None
                    if parent_idx is not None and parent_idx in raw_data:
                        parent_name = raw_data[parent_idx].get('fully_qualified_name', '')
                    self.part_instances[fqn] = PartInstance(
                        name=name, part_type=related, parent=parent_name
                    )

            elif 'attribute' in keywords and 'redefines' in keywords:
                if constants and constants[0] is not None:
                    try:
                        value = float(constants[0])
                        parent_idx = elem.get('idx_parent')
                        if parent_idx is not None and parent_idx in raw_data:
                            parent_fqn = raw_data[parent_idx].get('fully_qualified_name', '')
                            if parent_fqn in self.part_instances:
                                self.part_instances[parent_fqn].attributes[name] = value
                        parts = fqn.split('::')
                        self.parameters.append(Parameter(
                            name=name, qualified_name=fqn, value=value, part_path=parts[:-1]
                        ))
                    except (ValueError, TypeError):
                        pass

    def _parse_part_definitions(self) -> None:
        """Parse part definitions to extract attributes, derived attributes, and constraints."""
        part_def_pattern = r'part\s+def\s+(\w+)\s*\{'

        for match in re.finditer(part_def_pattern, self.content):
            part_name = match.group(1)
            part_start = match.end()
            part_body = self._extract_block(part_start)

            part_def = PartDef(name=part_name)

            # Parse attributes with derivations
            attr_pattern = r'attribute\s+(\w+)\s*:\s*(\w+)(?:\s*=\s*([^;]+))?;'
            for attr_match in re.finditer(attr_pattern, part_body):
                attr_name = attr_match.group(1)
                attr_type = attr_match.group(2)
                derivation = attr_match.group(3)

                part_def.attributes[attr_name] = attr_type
                if derivation:
                    part_def.derived_attributes[attr_name] = derivation.strip()

            # Parse constraints
            constraint_pattern = r'constraint\s+(\w+)\s*\{\s*([^}]+)\s*\}'
            for const_match in re.finditer(constraint_pattern, part_body):
                const_name = const_match.group(1)
                const_expr = const_match.group(2).strip()
                part_def.constraints.append((const_name, const_expr))

            # Parse refs
            ref_pattern = r'ref\s+(\w+)\s*:\s*(\w+)\s*;'
            for ref_match in re.finditer(ref_pattern, part_body):
                ref_name = ref_match.group(1)
                ref_type = ref_match.group(2)
                part_def.refs[ref_name] = ref_type

            # Check if exhibits state machine
            exhibit_pattern = r'exhibit\s+state\s+\w+\s*:\s*(\w+)\s*;'
            exhibit_match = re.search(exhibit_pattern, part_body)
            if exhibit_match:
                part_def.exhibits_state = exhibit_match.group(1)

            self.part_defs[part_name] = part_def

    def _parse_state_machines(self) -> None:
        state_def_pattern = r'state\s+def\s+(\w+)\s*\{'

        for match in re.finditer(state_def_pattern, self.content):
            sm_name = match.group(1)
            sm_body = self._extract_block(match.end())

            sm = StateMachine(name=sm_name)

            state_pattern = r'state\s+(\w+)\s*;'
            for state_match in re.finditer(state_pattern, sm_body):
                state_name = state_match.group(1)
                sm.states.append(State(name=state_name))
                if sm.initial_state is None:
                    sm.initial_state = state_name

            trans_pattern = r'transition\s+(\w+)\s*first\s+(\w+)\s*(?:accept\s+(\w+))?\s*(?:if\s+(\w+))?\s*then\s+(\w+)\s*;'
            for trans_match in re.finditer(trans_pattern, sm_body):
                sm.transitions.append(Transition(
                    name=trans_match.group(1),
                    from_state=trans_match.group(2),
                    to_state=trans_match.group(5),
                    trigger=trans_match.group(3),
                    guard=trans_match.group(4)
                ))

            self.state_machines.append(sm)

    def _parse_ref_bindings(self) -> None:
        """Parse ref bindings like: ref :>> pump = FillingSystem::pump"""
        ref_binding_pattern = r'ref\s+:>>\s*(\w+)\s*=\s*([\w:]+)\s*;'

        for match in re.finditer(ref_binding_pattern, self.content):
            ref_name = match.group(1)
            target = match.group(2)
            pos = match.start()
            context = self._find_context_at(pos)
            if context:
                self.ref_bindings[f"{context}::{ref_name}"] = target

    def _parse_flows(self) -> None:
        flow_pattern = r'flow\s*:\s*(\w+)\s+from\s+([\w.]+)\s+to\s+([\w.]+)\s*;'
        for match in re.finditer(flow_pattern, self.content):
            self.flows.append(Flow(
                item_type=match.group(1),
                from_port=match.group(2),
                to_port=match.group(3)
            ))

    def _identify_system_part(self) -> None:
        # Look for an explicit top-level part named 'system'
        match = re.search(r'\bpart\s+system\s*:\s*(\w+)\s*[;{]', self.content)
        if match:
            self.system_part = "system"
            self.system_type = match.group(1)
            # Remap part instances and parameters from system_type::* to system::*
            prefix = f"{self.system_type}::"
            new_instances = {}
            for fqn, inst in self.part_instances.items():
                if fqn.startswith(prefix):
                    new_fqn = "system::" + fqn[len(prefix):]
                    if inst.parent == self.system_type:
                        inst.parent = "system"
                    new_instances[new_fqn] = inst
                # Drop any instances not under system_type (e.g. package-level parts)
            self.part_instances = new_instances
            for param in self.parameters:
                if param.qualified_name.startswith(prefix):
                    new_qn = "system::" + param.qualified_name[len(prefix):]
                    param.qualified_name = new_qn
                    param.part_path = ["system"] + param.part_path[1:]
                    parts = new_qn.split('::')[1:]
                    cli = "-".join(parts)
                    param.cli_name = re.sub(r'([a-z])([A-Z])', r'\1-\2', cli).lower()
            return

        # Fall back: infer system part from the common FQN prefix of all instances
        parents = set()
        for fqn in self.part_instances.keys():
            parts = fqn.split('::')
            if len(parts) >= 2:
                parents.add(parts[0])
        if len(parents) == 1:
            self.system_part = parents.pop()
        elif parents:
            for p in parents:
                if 'System' in p:
                    self.system_part = p
                    return
            self.system_part = sorted(parents)[0]

    def _parse_part_instances_regex(self) -> None:
        """Fallback regex parsing for part instances that PySysML2 missed."""
        if not self.system_part:
            return

        # Find part instances inside the system part definition.
        # When system_part is an instance of a named type, look up that type's body.
        lookup_name = self.system_type if self.system_type else self.system_part
        system_def_pattern = rf'part\s+def\s+{re.escape(lookup_name)}\s*\{{'
        match = re.search(system_def_pattern, self.content)
        if not match:
            return

        system_body = self._extract_block(match.end())

        # Find part instances: part name : Type { ... }
        part_inst_pattern = r'part\s+(\w+)\s*:\s*(\w+)\s*\{'
        for inst_match in re.finditer(part_inst_pattern, system_body):
            inst_name = inst_match.group(1)
            inst_type = inst_match.group(2)
            fqn = f"{self.system_part}::{inst_name}"

            # Only add if not already present (PySysML2 might have found it)
            if fqn not in self.part_instances:
                self.part_instances[fqn] = PartInstance(
                    name=inst_name,
                    part_type=inst_type,
                    parent=self.system_part
                )

    def _find_controller(self) -> None:
        """Find the part instance that exhibits a state machine."""
        for fqn, inst in self.part_instances.items():
            part_def = self.part_defs.get(inst.part_type)
            if part_def and part_def.exhibits_state:
                self.controller_part = fqn
                break

    def _build_constraints(self) -> None:
        """Build parsed constraints from part definitions."""
        # For each part instance, instantiate constraints from its definition
        for fqn, instance in self.part_instances.items():
            part_def = self.part_defs.get(instance.part_type)
            if not part_def:
                continue

            # Add derived attributes
            for attr_name, expr_text in part_def.derived_attributes.items():
                try:
                    parser = ExpressionParser(expr_text)
                    expr = parser.parse()
                    self.derived_attributes.append(DerivedAttribute(
                        name=attr_name,
                        qualified_name=f"{fqn}::{attr_name}",
                        expression=expr,
                        context=fqn
                    ))
                except Exception:
                    pass

            # Add constraints
            for const_name, const_expr in part_def.constraints:
                try:
                    parser = ExpressionParser(const_expr)
                    expr = parser.parse()
                    self.parsed_constraints.append(Constraint(
                        name=const_name,
                        expression=expr,
                        raw_text=const_expr,
                        context=fqn
                    ))
                except Exception:
                    pass

        # Add system-level constraints
        if self.system_part:
            for name, pdef in self.part_defs.items():
                if 'System' in name or name == self.system_part or name == self.system_type:
                    for const_name, const_expr in pdef.constraints:
                        try:
                            parser = ExpressionParser(const_expr)
                            expr = parser.parse()
                            self.parsed_constraints.append(Constraint(
                                name=const_name,
                                expression=expr,
                                raw_text=const_expr,
                                context=self.system_part
                            ))
                        except Exception:
                            pass

    def _extract_block(self, start_pos: int) -> str:
        """Extract content between braces starting at given position."""
        brace_count = 1
        pos = start_pos
        while brace_count > 0 and pos < len(self.content):
            if self.content[pos] == '{':
                brace_count += 1
            elif self.content[pos] == '}':
                brace_count -= 1
            pos += 1
        return self.content[start_pos:pos-1]

    def _find_context_at(self, pos: int) -> Optional[str]:
        """Find the part instance context at a given position."""
        part_inst_pattern = r'part\s+(\w+)\s*:\s*(\w+)\s*\{'
        best_match = None
        best_pos = -1

        for match in re.finditer(part_inst_pattern, self.content[:pos]):
            if match.start() > best_pos:
                best_pos = match.start()
                best_match = match.group(1)

        if best_match and self.system_part:
            return f"{self.system_part}::{best_match}"
        return None

    def get_triggers(self) -> list[str]:
        """Get all trigger names from state machine transitions."""
        triggers = []
        for sm in self.state_machines:
            for trans in sm.transitions:
                if trans.trigger and trans.trigger not in triggers:
                    triggers.append(trans.trigger)
        return triggers


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

        # Initialize state machines
        for sm in self.parser.state_machines:
            self.current_sm_state[sm.name] = sm.initial_state

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

        # Initialize behavior state predicates
        if self.parser.controller_part:
            for sm in self.parser.state_machines:
                for s in sm.states:
                    key = f"{self.parser.controller_part}::behavior.{s.name}"
                    self.state[key] = (s.name == sm.initial_state)

        # Create constraint solver
        self.solver = ConstraintSolver(self.state, self.parser.system_part,
                                       self.parser.ref_bindings, self.parser.flows)
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

        # Integrate dynamics
        self.solver.integrate_dynamics(dt)

        self.time += dt

    def _process_state_machines(self) -> None:
        ctrl = self.parser.controller_part

        for sm in self.parser.state_machines:
            current = self.current_sm_state.get(sm.name)

            for trans in sm.transitions:
                if trans.from_state != current:
                    continue

                # Check trigger
                if trans.trigger:
                    trigger_key = f"{ctrl}::{trans.trigger}" if ctrl else trans.trigger
                    if not self.state.get(trigger_key, False):
                        continue
                    self.state[trigger_key] = False

                # Check guard
                if trans.guard:
                    guard_key = f"{ctrl}::{trans.guard}" if ctrl else trans.guard
                    if not self.state.get(guard_key, False):
                        continue

                # Perform transition
                self.current_sm_state[sm.name] = trans.to_state
                break

    def get_status(self) -> dict:
        sm_state = "unknown"
        for sm in self.parser.state_machines:
            sm_state = self.current_sm_state.get(sm.name, "unknown")
            break

        # Collect state variables - those that have dynamics (der()) defined
        state_vars = {}
        for dyn in self.solver.dynamics:
            key = dyn.variable_key
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
        for sm in parser.state_machines:
            if args.initial_state in [s.name for s in sm.states]:
                engine.current_sm_state[sm.name] = args.initial_state
                engine.solver.solve(engine.current_sm_state)
            else:
                valid_states = [s.name for s in sm.states]
                print(f"Error: Invalid state '{args.initial_state}'. Valid: {valid_states}", file=sys.stderr)
                return 1

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
        print(f"Initial state: {list(engine.current_sm_state.values())[0] if engine.current_sm_state else 'N/A'}")
        print(f"\nDynamics equations: {len(engine.solver.dynamics)}")
        for dyn in engine.solver.dynamics:
            print(f"  der({dyn.variable_key}) = ...")
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
