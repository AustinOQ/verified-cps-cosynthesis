#!/usr/bin/env python3
"""
SysML v2 Model Parser

Parses SysML v2 model files to extract structure, constraints, dynamics,
and state machines. Used by both the simulator and the model checker extractor.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import warnings

warnings.filterwarnings("ignore")

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


# =============================================================================
# Action Statement AST
# =============================================================================

@dataclass
class ActionStmt:
    """Base class for action statement AST nodes."""
    pass


@dataclass
class InParamStmt(ActionStmt):
    """in param : Type;"""
    name: str
    type_name: str


@dataclass
class ItemDeclStmt(ActionStmt):
    """item name : Type;"""
    name: str
    type_name: str


@dataclass
class AssignStmt(ActionStmt):
    """assign target := expr;"""
    target: list[str]
    expr: Expr


@dataclass
class SendStmt(ActionStmt):
    """send item via port;"""
    item_name: str
    port: str


@dataclass
class AcceptStmt(ActionStmt):
    """accept [var :] Type via port;"""
    var_name: Optional[str]   # None for unnamed accepts
    type_name: str
    port: str


@dataclass
class AttributeDeclStmt(ActionStmt):
    """attribute name : Type [= expr];"""
    name: str
    type_name: str
    init_expr: Optional[Expr]  # None if no = expr


@dataclass
class IfStmt(ActionStmt):
    """if condition { body } [else { else_body }]"""
    condition: Expr
    body: list  # list[ActionStmt]
    else_body: list = field(default_factory=list)  # list[ActionStmt]


@dataclass
class PerformStmt(ActionStmt):
    """perform action Name;"""
    action_name: str


@dataclass
class InputBindingStmt(ActionStmt):
    """in paramName := expr; (input binding in a sub-action call)"""
    name: str
    expr: Expr


@dataclass
class OutputBindingStmt(ActionStmt):
    """out paramName := expr; (output binding in a sub-action call)"""
    name: str
    expr: Expr


@dataclass
class SubactionCallStmt(ActionStmt):
    """action name : ActionDefType { bindings }"""
    name: str
    type_name: str
    bindings: list[ActionStmt]  # list of InputBindingStmt / OutputBindingStmt


@dataclass
class OutParamStmt(ActionStmt):
    """out param : Type;"""
    name: str
    type_name: str


@dataclass
class ActionDef:
    """An action definition (action def) with in/out parameter declarations."""
    name: str
    in_params: list[InParamStmt]
    out_params: list[OutParamStmt]
    metadata: list[str] = field(default_factory=list)  # e.g. ['Neural']


@dataclass
class Action:
    """A named action with a parsed body."""
    name: str
    body: list[ActionStmt]


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
class Requirement:
    """A parsed requirement with its constraint expression AST."""
    name: str
    expression: Expr
    raw_text: str
    context: str        # instance FQN (e.g. "system")
    subject_var: str    # subject variable name (e.g. "s")
    metadata: list[str] = field(default_factory=list)  # e.g. ["Prohibition"]


@dataclass
class DerivedAttribute:
    """An attribute with a derived value expression."""
    name: str
    qualified_name: str
    expression: Expr
    context: str


@dataclass
class StepAction:
    """An owned action that updates a state variable by assignment."""
    action_name: str        # Name of the action (e.g. "step")
    target_path: list[str]  # LHS path (e.g. ['currentLevel'])
    target_key: str         # Fully qualified state key
    expression: Expr        # RHS expression
    context: str            # Part instance FQN
    condition: Optional[Expr] = None  # Enclosing if-condition, if any


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
    trigger: Optional[str] = None       # item type name, e.g. TemperatureReading
    trigger_var: Optional[str] = None   # bound variable name, e.g. temperatureReading
    trigger_port: Optional[str] = None  # via port name, e.g. thermometerPort
    guard: Optional[Expr] = None        # parsed guard expression AST
    do_action: Optional[list] = None    # list[ActionStmt]


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
    actions: list[Action] = field(default_factory=list)
    action_defs: list[ActionDef] = field(default_factory=list)
    bindings: list[tuple[str, str]] = field(default_factory=list)  # (lhs_path, rhs_path)
    requirements: list[tuple[str, str, str, list[str]]] = field(default_factory=list)  # (name, subject_var, expr_text, metadata)
    exhibits_state: Optional[str] = None  # State machine name if this part exhibits one


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
        self.parsed_requirements: list[Requirement] = []
        self.derived_attributes: list[DerivedAttribute] = []
        self.flows: list[Flow] = []
        self.part_defs: dict[str, PartDef] = {}
        self.part_instances: dict[str, PartInstance] = {}
        self.system_part: Optional[str] = None
        self.system_type: Optional[str] = None  # The part def type of system_part
        self.ref_bindings: dict[str, str] = {}
        self.parsed_bindings: dict[str, str] = {}  # lhs_key -> rhs_key (from bind stmts)
        self.step_actions: list[StepAction] = []
        self.controller_part: Optional[str] = None  # Part that exhibits state machine
        self.instance_state_machines: dict[str, 'StateMachine'] = {}  # fqn -> SM
        self.connects: list[tuple[str, str]] = []  # (from_port, to_port) dot-notation
        self.item_type_parents: dict[str, Optional[str]] = {}  # type_name -> parent or None
        self.step_action_bodies: list[tuple[str, list]] = []  # (inst_fqn, stmts)
        self.item_def_attrs: dict[str, dict[str, str]] = {}   # item_type -> {attr -> type}
        self.port_def_items: dict[str, dict[str, str]] = {}   # port_type -> {item_name -> item_type}
        self.part_def_ports: dict[str, dict[str, str]] = {}   # part_type -> {port_name -> port_type}

    def parse(self) -> None:
        self._parse_with_pysysml2()
        self._parse_part_definitions()
        self._parse_state_machines()
        self._identify_system_part()
        self._parse_part_instances_regex()   # Fallback for instances PySysML2 missed
        self._parse_parameters_regex()       # Fallback for attribute :>> values
        self._parse_ref_bindings()
        self._parse_flows()
        self._parse_item_types()
        self._parse_item_and_port_defs()
        self._parse_connects()
        self._find_controller()
        self._build_constraints()
        self._build_instance_state_machines()

    def _parse_with_pysysml2(self) -> None:
        if not PYSYSML2_AVAILABLE:
            raise ImportError("PySysML2 is required.")

        import io, contextlib
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                model = PySysML2Model()
                model.from_sysml2_file(self.file_path)
        except Exception:
            # PySysML2 can't handle all SysML v2 syntax; fall back to regex parsing.
            return

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

    @staticmethod
    def _walk_assigns(stmts: list, enclosing_condition=None) -> list:
        """Return (AssignStmt, condition_or_None) tuples, recursing into IfStmt bodies."""
        result = []
        for stmt in stmts:
            if isinstance(stmt, AssignStmt):
                result.append((stmt, enclosing_condition))
            elif isinstance(stmt, IfStmt):
                result.extend(SysMLParser._walk_assigns(stmt.body, stmt.condition))
        return result

    @staticmethod
    def _extract_block_from_string(text: str, start_pos: int) -> tuple[str, int]:
        """Extract content between braces in an arbitrary string.

        Args:
            text: The string to search in.
            start_pos: Position immediately after the opening '{'.

        Returns:
            (block_content, end_pos) where end_pos is just past the closing '}'.
        """
        brace_count = 1
        pos = start_pos
        while brace_count > 0 and pos < len(text):
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        return text[start_pos:pos - 1], pos

    @staticmethod
    def _strip_line_comments(text: str) -> str:
        """Remove // line comments while preserving string positions."""
        return re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group()), text)

    def _parse_bindings(self, body: str) -> list:
        """Parse input/output bindings inside a sub-action call block.

        Handles lines like:
            in tank1VolumeMl := volume1Res.response;
            out shouldOpenValve1 := someExpr;
        """
        body = self._strip_line_comments(body)
        bindings: list = []
        for m in re.finditer(
                r'\b(in|out)\s+(\w+)\s*:=\s*([^;]+?)\s*;', body):
            direction = m.group(1)
            name = m.group(2)
            try:
                expr = ExpressionParser(m.group(3).strip()).parse()
            except Exception:
                continue
            if direction == 'in':
                bindings.append(InputBindingStmt(name=name, expr=expr))
            else:
                bindings.append(OutputBindingStmt(name=name, expr=expr))
        return bindings

    def _parse_action_body(self, body: str) -> list:
        """Parse an action body string into a list of ActionStmt nodes, in source order."""
        # Strip line comments so `// Check if we ...` doesn't match `if` patterns
        body = self._strip_line_comments(body)

        all_matches: list[tuple[int, int, str, Any]] = []  # (start, end, kind, data)

        # --- Simple statement patterns ---
        simple_patterns = [
            ('in_param',       r'\bin\s+(\w+)\s*:\s*(\w+)\s*;'),
            ('item_decl',      r'\bitem\s+(\w+)\s*:\s*(\w+)\s*;'),
            ('assign',         r'\bassign\s+([\w.]+)\s*:=\s*([^;]+?)\s*;'),
            ('send',           r'\bsend\s+(\w+)\s+via\s+([\w.]+)\s*;'),
            ('accept_named',   r'\baccept\s+(\w+)\s*:\s*(\w+)\s+via\s+([\w.]+)\s*;'),
            ('accept_unnamed', r'\baccept\s+(\w+)\s+via\s+([\w.]+)\s*;'),
            ('attribute',      r'\battribute\s+(\w+)\s*:\s*(\w+)\s*(?:=\s*([^;]+?))?\s*;'),
            ('perform',        r'\bperform\s+action\s+(\w+)\s*;'),
        ]
        for kind, pat in simple_patterns:
            for m in re.finditer(pat, body, re.DOTALL):
                all_matches.append((m.start(), m.end(), kind, m))

        # --- if/else blocks (need brace matching + recursive parse) ---
        for m in re.finditer(r'\bif\s+(.*?)\s*\{', body, re.DOTALL):
            block_content, block_end = self._extract_block_from_string(body, m.end())
            # Check for else { ... } immediately after the if block
            else_content = None
            total_end = block_end
            rest = body[block_end:]
            else_match = re.match(r'\s*else\s*\{', rest)
            if else_match:
                else_content, else_end = self._extract_block_from_string(
                    body, block_end + else_match.end())
                total_end = else_end
            all_matches.append((m.start(), total_end, 'if_block',
                                (m.group(1), block_content, else_content)))

        # --- sub-action calls: action name : Type { bindings } ---
        for m in re.finditer(r'\baction\s+(\w+)\s*:\s*(\w+)\s*\{', body):
            block_content, block_end = self._extract_block_from_string(body, m.end())
            all_matches.append((m.start(), block_end, 'subaction_call',
                                (m.group(1), m.group(2), block_content)))

        all_matches.sort(key=lambda x: x[0])

        # --- Build statement list, skipping matches inside if-block spans ---
        stmts: list = []
        skip_until = -1
        for start, end, kind, data in all_matches:
            if start < skip_until:
                continue

            if kind == 'in_param':
                stmts.append(InParamStmt(name=data.group(1), type_name=data.group(2)))
            elif kind == 'item_decl':
                stmts.append(ItemDeclStmt(name=data.group(1), type_name=data.group(2)))
            elif kind == 'assign':
                target = data.group(1).split('.')
                try:
                    expr = ExpressionParser(data.group(2).strip()).parse()
                    stmts.append(AssignStmt(target=target, expr=expr))
                except Exception:
                    pass
            elif kind == 'send':
                stmts.append(SendStmt(item_name=data.group(1), port=data.group(2)))
            elif kind == 'accept_named':
                stmts.append(AcceptStmt(
                    var_name=data.group(1), type_name=data.group(2), port=data.group(3)))
            elif kind == 'accept_unnamed':
                # Only emit if not already captured as accept_named at the same position
                if not any(s2 == start and k2 == 'accept_named'
                           for s2, _, k2, _ in all_matches):
                    stmts.append(AcceptStmt(
                        var_name=None, type_name=data.group(1), port=data.group(2)))
            elif kind == 'attribute':
                init_expr = None
                if data.group(3):
                    try:
                        init_expr = ExpressionParser(data.group(3).strip()).parse()
                    except Exception:
                        pass
                stmts.append(AttributeDeclStmt(
                    name=data.group(1), type_name=data.group(2), init_expr=init_expr))
            elif kind == 'perform':
                stmts.append(PerformStmt(action_name=data.group(1)))
            elif kind == 'if_block':
                cond_str, block_content, else_content = data
                skip_until = end
                try:
                    condition = ExpressionParser(cond_str.strip()).parse()
                    inner_stmts = self._parse_action_body(block_content)
                    else_stmts = self._parse_action_body(else_content) if else_content else []
                    stmts.append(IfStmt(condition=condition, body=inner_stmts,
                                        else_body=else_stmts))
                except Exception:
                    pass
            elif kind == 'subaction_call':
                call_name, type_name, block_content = data
                skip_until = end
                bindings = self._parse_bindings(block_content)
                stmts.append(SubactionCallStmt(
                    name=call_name, type_name=type_name, bindings=bindings))
        return stmts

    def _parse_part_definitions(self) -> None:
        """Parse part definitions to extract attributes, derived attributes, and constraints."""
        part_def_pattern = r'part\s+def\s+(\w+)\s*\{'

        for match in re.finditer(part_def_pattern, self.content):
            part_name = match.group(1)
            part_start = match.end()
            part_body = self._extract_block(part_start)

            part_def = PartDef(name=part_name)

            # Build a version of part_body with action/state blocks blanked out,
            # so that attribute/constraint/ref regexes only match part-level decls.
            part_body_toplevel = part_body
            for nested in re.finditer(r'(?:action|state)\s+(?:def\s+)?\w+\s*\{', part_body):
                inner = self._extract_block(part_start + nested.end())
                # Blank the block body (from '{' to '}') with spaces
                start_in_body = nested.end()
                end_in_body = start_in_body + len(inner)
                part_body_toplevel = (part_body_toplevel[:start_in_body]
                                      + ' ' * len(inner)
                                      + part_body_toplevel[end_in_body:])

            # Parse attributes with derivations
            attr_pattern = r'attribute\s+(\w+)\s*:\s*(\w+)(?:\s*=\s*([^;]+))?;'
            for attr_match in re.finditer(attr_pattern, part_body_toplevel):
                attr_name = attr_match.group(1)
                attr_type = attr_match.group(2)
                derivation = attr_match.group(3)

                part_def.attributes[attr_name] = attr_type
                if derivation:
                    part_def.derived_attributes[attr_name] = derivation.strip()

            # Parse constraints
            constraint_pattern = r'constraint\s+(\w+)\s*\{\s*([^}]+)\s*\}'
            for const_match in re.finditer(constraint_pattern, part_body_toplevel):
                const_name = const_match.group(1)
                const_expr = const_match.group(2).strip()
                part_def.constraints.append((const_name, const_expr))

            # Parse requirement defs (with optional #Metadata annotation)
            for req_match in re.finditer(r"(?:#(\w+)\s+)?requirement\s+def\s+(?:'([^']+)'|(\w+))\s*\{", part_body):
                req_metadata = [req_match.group(1)] if req_match.group(1) else []
                req_name = req_match.group(2) or req_match.group(3)
                req_body, _ = self._extract_block_from_string(
                    part_body, req_match.end())
                # Extract subject variable and type
                subj_match = re.search(r'subject\s+(\w+)\s*:\s*(\w+)\s*;', req_body)
                if not subj_match:
                    continue
                subject_var = subj_match.group(1)
                # Extract require constraint body (nested braces)
                rc_match = re.search(r'require\s+constraint\s*\{', req_body)
                if not rc_match:
                    continue
                rc_body, _ = self._extract_block_from_string(
                    req_body, rc_match.end())
                part_def.requirements.append(
                    (req_name, subject_var, rc_body.strip(), req_metadata))

            # Parse refs
            ref_pattern = r'ref\s+(\w+)\s*:\s*(\w+)\s*;'
            for ref_match in re.finditer(ref_pattern, part_body_toplevel):
                ref_name = ref_match.group(1)
                ref_type = ref_match.group(2)
                part_def.refs[ref_name] = ref_type

            # Parse action defs: [#Metadata] action def Name { in/out params }
            action_def_pattern = r'(?:#(\w+)\s+)?action\s+def\s+(\w+)\s*\{'
            for ad_match in re.finditer(action_def_pattern, part_body):
                ad_name = ad_match.group(2)
                ad_body = self._extract_block(part_start + ad_match.end())
                metadata = [ad_match.group(1)] if ad_match.group(1) else []
                in_params = []
                out_params = []
                for pm in re.finditer(r'\bin\s+(\w+)\s*:\s*(\w+)\s*;', ad_body):
                    in_params.append(InParamStmt(name=pm.group(1), type_name=pm.group(2)))
                for pm in re.finditer(r'\bout\s+(\w+)\s*:\s*(\w+)\s*;', ad_body):
                    out_params.append(OutParamStmt(name=pm.group(1), type_name=pm.group(2)))
                part_def.action_defs.append(ActionDef(
                    name=ad_name, in_params=in_params, out_params=out_params,
                    metadata=metadata))

            # Parse owned actions: action name { ... } (skip action def)
            action_pattern = r'action\s+(?!def\s)(\w+)\s*\{'
            for action_match in re.finditer(action_pattern, part_body):
                action_name = action_match.group(1)
                abs_pos = part_start + action_match.end()
                action_body = self._extract_block(abs_pos)
                stmts = self._parse_action_body(action_body)
                part_def.actions.append(Action(name=action_name, body=stmts))

            # Parse bind statements: bind lhs = rhs;
            bind_pattern = r'bind\s+([\w.]+)\s*=\s*([\w.]+)\s*;'
            for bind_match in re.finditer(bind_pattern, part_body):
                part_def.bindings.append((bind_match.group(1), bind_match.group(2)))

            # Parse port declarations: port name : [~]Type;
            ports: dict[str, str] = {}
            for pm in re.finditer(r'\bport\s+(\w+)\s*:\s*~?(\w+)\s*;', part_body):
                ports[pm.group(1)] = pm.group(2)  # strip leading ~ for conjugate ports
            self.part_def_ports[part_name] = ports

            # Check if exhibits state machine
            # Strip single-line comments so that commented-out exhibits are ignored
            part_body_uncommented = re.sub(r'//[^\n]*', '', part_body)
            exhibit_pattern = r'exhibit\s+state\s+\w+\s*:\s*(\w+)\s*;'
            exhibit_match = re.search(exhibit_pattern, part_body_uncommented)
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

            # Flexible multi-line transition pattern.
            # Captures everything between 'first FROM_STATE' and 'then TO_STATE;',
            # then extracts trigger and guard from the middle section.
            trans_pattern = r'transition\s+(\w+)\s+first\s+(\w+)(.*?)then\s+(\w+)\s*;'
            for trans_match in re.finditer(trans_pattern, sm_body, re.DOTALL):
                trans_name = trans_match.group(1)
                from_state = trans_match.group(2)
                middle = trans_match.group(3)
                to_state = trans_match.group(4)

                # Extract trigger: 'accept [varName :] TypeName [via portName]'
                trigger = trigger_var = trigger_port = None
                accept_m = re.search(
                    r'\baccept\s+(?:(\w+)\s*:\s*)?(\w+)(?:\s+via\s+([\w.]+))?', middle
                )
                if accept_m:
                    trigger_var  = accept_m.group(1)   # None when no var binding
                    trigger      = accept_m.group(2)   # item type name
                    trigger_port = accept_m.group(3)   # None when no via clause

                # Extract and parse guard expression
                guard = None
                guard_m = re.search(r'\bif\b\s*(.*?)(?=\s*\bdo\b|\Z)', middle, re.DOTALL)
                if guard_m:
                    guard_text = guard_m.group(1).strip()
                    if guard_text:
                        try:
                            guard = ExpressionParser(guard_text).parse()
                        except Exception:
                            pass

                # Parse do action body
                do_action = None
                do_m = re.search(r'\bdo\s+action\s*\{(.*?)\}', middle, re.DOTALL)
                if do_m:
                    do_action = self._parse_action_body(do_m.group(1))

                sm.transitions.append(Transition(
                    name=trans_name,
                    from_state=from_state,
                    to_state=to_state,
                    trigger=trigger,
                    trigger_var=trigger_var,
                    trigger_port=trigger_port,
                    guard=guard,
                    do_action=do_action,
                ))

            self.state_machines.append(sm)

    def _parse_ref_bindings(self) -> None:
        """Parse ref bindings like: ref :>> pump = system::pump"""
        ref_binding_pattern = r'ref\s+:>>\s*(\w+)\s*=\s*([\w:]+)\s*;'

        for match in re.finditer(ref_binding_pattern, self.content):
            ref_name = match.group(1)
            target = match.group(2)
            pos = match.start()
            context = self._find_context_at(pos)
            if context:
                self.ref_bindings[f"{context}::{ref_name}"] = target

    def _parse_flows(self) -> None:
        flow_pattern = r'flow(?:\s*:\s*(\w+))?\s+from\s+([\w.]+)\s+to\s+([\w.]+)\s*;'
        for match in re.finditer(flow_pattern, self.content):
            self.flows.append(Flow(
                item_type=match.group(1) or '',
                from_port=match.group(2),
                to_port=match.group(3)
            ))

    def _parse_item_types(self) -> None:
        """Parse item def declarations to build a type hierarchy."""
        pattern = r'\bitem\s+def\s+(\w+)(?:\s*:>\s*(\w+))?'
        for m in re.finditer(pattern, self.content):
            self.item_type_parents[m.group(1)] = m.group(2)  # group(2) is None if no parent

    def _parse_item_and_port_defs(self) -> None:
        """Parse item def and port def bodies to extract attribute and item member types."""
        # item def X { attribute y : Type; ... }
        for m in re.finditer(r'\bitem\s+def\s+(\w+)\s*(?::>\s*\w+\s*)?\{', self.content):
            item_name = m.group(1)
            body = self._extract_block(m.end())
            attrs: dict[str, str] = {}
            for am in re.finditer(r'\battribute\s+(\w+)\s*:\s*(\w+)', body):
                attrs[am.group(1)] = am.group(2)
            self.item_def_attrs[item_name] = attrs

        # port def P { in item h : Type; ... }
        for m in re.finditer(r'\bport\s+def\s+(\w+)\s*\{', self.content):
            port_name = m.group(1)
            body = self._extract_block(m.end())
            items: dict[str, str] = {}
            for im in re.finditer(r'\bitem\s+(\w+)\s*:\s*(\w+)', body):
                items[im.group(1)] = im.group(2)
            self.port_def_items[port_name] = items

    def _parse_connects(self) -> None:
        """Parse connect statements."""
        pattern = r'\bconnect\s+([\w.]+)\s+to\s+([\w.]+)\s*;'
        for m in re.finditer(pattern, self.content):
            self.connects.append((m.group(1), m.group(2)))

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

        # Find part instances: part name : Type { ... } or part name : Type;
        part_inst_pattern = r'part\s+(\w+)\s*:\s*(\w+)\s*[{;]'
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

    def _parse_parameters_regex(self) -> None:
        """Regex fallback for attribute :>> name = value patterns.

        PySysML2 normally extracts these, but falls back to this when it fails.
        Searches two locations:
        1. The system instantiation block: part system : Type { part :>> name { ... } }
        2. The system type definition: part def Type { part name : X { attr :>> a = v; } }
        """
        if not self.system_part:
            return

        found_fqns: set[str] = {p.qualified_name for p in self.parameters}

        def _add_param(part_name: str, attr_name: str, attr_value: float) -> None:
            fqn = f"{self.system_part}::{part_name}::{attr_name}"
            if fqn in found_fqns:
                return
            found_fqns.add(fqn)
            parts = fqn.split('::')
            self.parameters.append(Parameter(
                name=attr_name,
                qualified_name=fqn,
                value=attr_value,
                part_path=parts[:-1],
            ))
            inst_fqn = f"{self.system_part}::{part_name}"
            if inst_fqn in self.part_instances:
                self.part_instances[inst_fqn].attributes[attr_name] = attr_value

        attr_pattern = r'attribute\s+:>>\s*(\w+)\s*=\s*(-?[\d.]+)\s*;'

        # 1. System instantiation block: part system : Type { ... }
        sys_inst_pattern = rf'\bpart\s+{re.escape(self.system_part)}\s*:\s*\w+\s*\{{'
        sys_match = re.search(sys_inst_pattern, self.content)
        if sys_match:
            sys_body = self._extract_block(sys_match.end())
            sub_part_pattern = r'part\s+:>>\s*(\w+)\s*\{'
            for sub_match in re.finditer(sub_part_pattern, sys_body):
                sub_body = self._extract_block(sys_match.end() + sub_match.end())
                for attr_match in re.finditer(attr_pattern, sub_body):
                    _add_param(sub_match.group(1), attr_match.group(1),
                               float(attr_match.group(2)))

        # 2. System type definition: part def FillingSystem { part x : T { ... } }
        lookup_name = self.system_type if self.system_type else self.system_part
        sys_def_pattern = rf'part\s+def\s+{re.escape(lookup_name)}\s*\{{'
        def_match = re.search(sys_def_pattern, self.content)
        if def_match:
            def_start = def_match.end()
            def_body = self._extract_block(def_start)
            # Match part instances with blocks: part name : Type { ... }
            part_inst_pattern = r'part\s+(\w+)\s*:\s*\w+\s*\{'
            for inst_match in re.finditer(part_inst_pattern, def_body):
                inst_name = inst_match.group(1)
                inst_body = self._extract_block(def_start + inst_match.end())
                for attr_match in re.finditer(attr_pattern, inst_body):
                    _add_param(inst_name, attr_match.group(1),
                               float(attr_match.group(2)))

    def _find_controller(self) -> None:
        """Find the controller part instance.

        Priority: (1) Controller name + exhibits state, (2) Controller name
        without SM, (3) first exhibiting part.
        """
        # First pass: part whose type contains 'Controller' AND exhibits state
        for fqn, inst in self.part_instances.items():
            part_def = self.part_defs.get(inst.part_type)
            if part_def and part_def.exhibits_state and 'Controller' in inst.part_type:
                self.controller_part = fqn
                return
        # Second pass: part whose type contains 'Controller', no SM required
        for fqn, inst in self.part_instances.items():
            if 'Controller' in inst.part_type:
                self.controller_part = fqn
                return
        # Third pass: take the first exhibiting part
        for fqn, inst in self.part_instances.items():
            part_def = self.part_defs.get(inst.part_type)
            if part_def and part_def.exhibits_state:
                self.controller_part = fqn
                return

    def _build_constraints(self) -> None:
        """Build parsed constraints from part definitions."""
        # For each part instance, instantiate constraints from its definition
        for fqn, instance in self.part_instances.items():
            part_def = self.part_defs.get(instance.part_type)
            if not part_def:
                continue

            # Collect assign targets from all action bodies — these are
            # mutable state, not derived attributes, even if they have = init.
            action_assign_targets: set[str] = set()
            for action in part_def.actions:
                for stmt, _cond in self._walk_assigns(action.body):
                    if len(stmt.target) == 1:
                        action_assign_targets.add(stmt.target[0])

            # Add derived attributes (skip any that are assign targets)
            for attr_name, expr_text in part_def.derived_attributes.items():
                if attr_name in action_assign_targets:
                    continue
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

            # Instantiate owned actions onto this part instance
            for action in part_def.actions:
                if action.name == 'step':
                    self.step_action_bodies.append((fqn, action.body))
                    # Only create StepAction entries from the 'step' action
                    for stmt, cond in self._walk_assigns(action.body):
                        target_key = f"{fqn}::" + "::".join(stmt.target)
                        self.step_actions.append(StepAction(
                            action_name=action.name,
                            target_path=stmt.target,
                            target_key=target_key,
                            expression=stmt.expr,
                            context=fqn,
                            condition=cond,
                        ))

            # Register bind statements as key-to-key attribute bindings
            for lhs, rhs in part_def.bindings:
                lhs_key = f"{fqn}::" + lhs.replace('.', '::')
                rhs_key = f"{fqn}::" + rhs.replace('.', '::')
                self.parsed_bindings[lhs_key] = rhs_key

        # Add system-level constraints and requirements
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
                    for req_name, subject_var, req_expr, req_metadata in pdef.requirements:
                        try:
                            parser = ExpressionParser(req_expr)
                            expr = parser.parse()
                            # subject var binds to the system instance itself
                            self.ref_bindings[
                                f"{self.system_part}::{subject_var}"
                            ] = self.system_part
                            self.parsed_requirements.append(Requirement(
                                name=req_name,
                                expression=expr,
                                raw_text=req_expr,
                                context=self.system_part,
                                subject_var=subject_var,
                                metadata=req_metadata,
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

    def _build_instance_state_machines(self) -> None:
        """Map each part instance that exhibits a state machine to its SM definition."""
        sm_by_name = {sm.name: sm for sm in self.state_machines}
        for fqn, inst in self.part_instances.items():
            part_def = self.part_defs.get(inst.part_type)
            if part_def and part_def.exhibits_state:
                sm = sm_by_name.get(part_def.exhibits_state)
                if sm:
                    self.instance_state_machines[fqn] = sm

    def get_triggers(self) -> list[str]:
        """Get all trigger names from state machine transitions."""
        triggers = []
        for sm in self.state_machines:
            for trans in sm.transitions:
                if trans.trigger and trans.trigger not in triggers:
                    triggers.append(trans.trigger)
        return triggers
