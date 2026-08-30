"""Initial strict-ish semantic extractor for certification.

This module does not replace `sysml_deps.py`; it builds an auditable equation
view beside the legacy dependency graph. Unsupported or approximate pieces are
reported as diagnostics so they cannot silently become certificate assumptions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from clarity.certification.dependencies import SysMLModel  # type: ignore
from clarity.sysml.parser import (  # type: ignore
    AssignStmt,
    BinaryExpr,
    IfStmt,
    InputBindingStmt,
    LiteralExpr,
    PerformStmt,
    RefExpr,
    SubactionCallStmt,
    TernaryExpr,
    UnaryExpr,
    ExpressionParser,
)

from .equations import Const, Equation, EquationModel, Expr, Ite, Op, RawRef, Var


def expr_refs(expr) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []

    def walk(node) -> None:
        if isinstance(node, RefExpr):
            out.append(tuple(node.path))
        elif isinstance(node, BinaryExpr):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, UnaryExpr):
            walk(node.operand)
        elif isinstance(node, TernaryExpr):
            walk(node.condition)
            walk(node.true_expr)
            walk(node.false_expr)

    if expr is not None:
        walk(expr)
    return out


class CertificationExtractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.legacy = SysMLModel(model_path)
        self.parser = self.legacy.p
        self.model = EquationModel(
            model_path=model_path,
            state=set(self.legacy.STATE),
            actions=set(self.legacy.ACTIONS),
            constants=set(getattr(self.legacy, "consts", set())),
        )
        self.flow_value_types: dict[str, str] = {}
        for instance in self.parser.part_instances.values():
            ports = self.parser.part_def_ports.get(instance.part_type, {})
            for port_name, port_type in ports.items():
                for attribute, type_name in self.parser.port_def_attrs.get(
                    port_type, {}
                ).items():
                    self.flow_value_types[
                        self.legacy._canon([instance.name, port_name, attribute])
                    ] = type_name

    def extract(self) -> EquationModel:
        self._augment_state_from_step_body_assignments()
        self._extract_initial_values()
        self._diagnose_parser_surface()
        self._build_same_cycle_definitions()
        self._build_terminal_equations()
        self._build_observation_equations()
        self._build_requirement_equations()
        self._build_transition_equations()
        self._audit_equations()
        return self.model

    def _extract_initial_values(self) -> None:
        owners = [(None, self.parser.system_type)] + [
            (instance.name, instance.part_type)
            for instance in self.parser.part_instances.values()
        ]
        for instance_name, part_type in owners:
            part_def = self.parser.part_defs.get(part_type or "")
            if part_def is None:
                continue
            for attribute, text in part_def.derived_attributes.items():
                target = self.legacy._canon(
                    [attribute] if instance_name is None
                    else [instance_name, attribute]
                )
                if target not in self.model.state:
                    continue
                try:
                    expression = ExpressionParser(text).parse()
                except Exception:
                    continue
                if isinstance(expression, LiteralExpr):
                    self.model.initial_values[target] = expression.value

        for parameter in self.parser.parameters:
            target = self.legacy._canon(parameter.qualified_name.split("::"))
            if target in self.model.state and "ScenarioInput" not in parameter.metadata:
                self.model.initial_values[target] = parameter.value

    def _inst_name(self, fqn: str) -> str:
        return fqn.split("::")[-1]

    def _inst_attr_names(self, inst_name: str) -> set[str]:
        inst = self.legacy.inst_by_name.get(inst_name)
        if not inst:
            return set()
        pdef = self.parser.part_defs.get(inst.part_type)
        return set(pdef.attributes) if pdef else set()

    def _is_instance_attribute_assignment(self, inst_name: str, stmt: AssignStmt) -> bool:
        return len(stmt.target) == 1 and stmt.target[0] in self._inst_attr_names(inst_name)

    def _walk_effective_assigns(self, stmts, inst_name: str, condition: Expr | None = None):
        for stmt in stmts:
            if isinstance(stmt, AssignStmt):
                yield stmt, condition
            elif isinstance(stmt, IfStmt):
                cond = self._expr(stmt.condition, [self.legacy.sys, inst_name])
                body_cond = cond if condition is None else Op("and", (condition, cond))
                yield from self._walk_effective_assigns(stmt.body, inst_name, body_cond)
                if stmt.else_body:
                    else_cond = Op("not", (cond,))
                    if condition is not None:
                        else_cond = Op("and", (condition, else_cond))
                    yield from self._walk_effective_assigns(stmt.else_body, inst_name, else_cond)
            elif isinstance(stmt, PerformStmt) and stmt.action_name in self.legacy.actions_by_name:
                yield from self._walk_effective_assigns(
                    self.legacy.actions_by_name[stmt.action_name], inst_name, condition
                )

    def _augment_state_from_step_body_assignments(self) -> None:
        for fqn, stmts in self.parser.step_action_bodies:
            inst_name = self._inst_name(fqn)
            for stmt, _cond in self._walk_effective_assigns(stmts, inst_name):
                if self._is_instance_attribute_assignment(inst_name, stmt):
                    self.model.state.add(self.legacy._canon([inst_name] + stmt.target))

    def _ctx(self, context: str | None) -> list[str]:
        if context:
            return context.split("::")
        return [self.legacy.sys]

    def _qual_ref_key(
        self, ref: Iterable[str], ctx: list[str], subject_var: str | None = None
    ) -> str:
        parts = list(ref)
        if subject_var and parts and parts[0] == subject_var:
            parts = parts[1:]
        if not parts:
            return subject_var or ""

        return self.legacy._qual(parts, ctx)  # pylint: disable=protected-access

    def _is_const_key(self, key: str) -> bool:
        is_const = key in getattr(self.legacy, "consts", set())
        if is_const:
            self.model.constants.add(key)
        return is_const

    def _is_numeric_flow_value(self, key: str) -> bool:
        return self.flow_value_types.get(key, "").lower() in {
            "integer", "real", "float", "double"
        }

    def _resolve_ref(
        self,
        ref: Iterable[str],
        ctx: list[str],
        subject_var: str | None = None,
        allow_legacy_fallback: bool = True,
    ) -> Expr:
        parts = list(ref)
        key = self._qual_ref_key(parts, ctx, subject_var)
        if not key:
            return RawRef(subject_var or "")

        if self._is_const_key(key):
            return RawRef(key)

        if key in self.model.state or key in self.model.actions or key in self.model.definitions:
            return Var(key)

        # Values declared as numeric port attributes are handled by the flow
        # equations read from this SysML file.
        if self._is_numeric_flow_value(key):
            return Var(key)

        if not allow_legacy_fallback:
            return Var(key)

        try:
            keys = self.legacy._collect_one(parts, ctx)  # pylint: disable=protected-access
        except Exception as exc:  # pragma: no cover - defensive diagnostic path
            self.model.add_diagnostic(
                "error", "unresolved_ref", f"could not resolve reference {'.'.join(parts)}: {exc}"
            )
            return RawRef(".".join(parts))

        if len(keys) == 1:
            return Var(next(iter(keys)))
        if len(keys) > 1:
            return Op("refs", tuple(Var(k) for k in sorted(keys)))
        return RawRef(key)

    def _expr(
        self,
        expr,
        ctx: list[str],
        subject_var: str | None = None,
        allow_legacy_fallback: bool = True,
    ) -> Expr:
        if expr is None:
            return RawRef("<unparsed>")
        if isinstance(expr, LiteralExpr):
            return Const(expr.value)
        if isinstance(expr, RefExpr):
            return self._resolve_ref(expr.path, ctx, subject_var, allow_legacy_fallback)
        if isinstance(expr, UnaryExpr):
            return Op(
                expr.op,
                (self._expr(expr.operand, ctx, subject_var, allow_legacy_fallback),),
            )
        if isinstance(expr, BinaryExpr):
            return Op(
                expr.op,
                (
                    self._expr(expr.left, ctx, subject_var, allow_legacy_fallback),
                    self._expr(expr.right, ctx, subject_var, allow_legacy_fallback),
                ),
            )
        if isinstance(expr, TernaryExpr):
            return Ite(
                self._expr(expr.condition, ctx, subject_var, allow_legacy_fallback),
                self._expr(expr.true_expr, ctx, subject_var, allow_legacy_fallback),
                self._expr(expr.false_expr, ctx, subject_var, allow_legacy_fallback),
            )
        self.model.add_diagnostic(
            "error", "unsupported_expr", f"unsupported expression node {type(expr).__name__}"
        )
        return RawRef(f"<{type(expr).__name__}>")

    def _diagnose_parser_surface(self) -> None:
        parsed_req_names = {req.name for req in self.parser.parsed_requirements}
        for part_def in self.parser.part_defs.values():
            for req_name, _subject_var, _subject_type, _expr_text, metadata in part_def.requirements:
                if "NeuralRequirement" in metadata:
                    continue
                if req_name not in parsed_req_names:
                    self.model.add_diagnostic(
                        "error",
                        "skipped_requirement_parse",
                        "requirement exists in the raw model but has no parsed AST",
                        req_name,
                    )

        parsed_constraint_names = {c.name for c in self.parser.parsed_constraints}
        for part_def in self.parser.part_defs.values():
            for constraint_name, _expr_text, _metadata in part_def.constraints:
                if constraint_name not in parsed_constraint_names:
                    self.model.add_diagnostic(
                        "error",
                        "skipped_constraint_parse",
                        "constraint exists in the raw model but has no parsed AST",
                        constraint_name,
                    )

        for c in self.parser.parsed_constraints:
            if c.expression is None:
                self.model.add_diagnostic(
                    "error", "unparsed_constraint", "constraint expression was not parsed", c.name
                )
        for req in self.parser.parsed_requirements:
            if req.expression is None:
                self.model.add_diagnostic(
                    "error", "unparsed_requirement", "requirement expression was not parsed", req.name
                )

    def _add_definition(self, target: str, expr: Expr, source: str) -> bool:
        existing = self.model.definitions.get(target)
        if existing is not None:
            if existing.expr != expr:
                self.model.add_diagnostic(
                    "error",
                    "duplicate_definition",
                    "same-cycle definition already exists; keeping the first equation",
                    target,
                )
            return False

        self.model.definitions[target] = Equation(
            target=target,
            expr=expr,
            kind="definition",
            source=source,
        )
        return True

    def _conjuncts(self, expr) -> list:
        if isinstance(expr, BinaryExpr) and expr.op == "and":
            return self._conjuncts(expr.left) + self._conjuncts(expr.right)
        return [expr] if expr is not None else []

    def _negates(self, a, b) -> bool:
        return isinstance(a, UnaryExpr) and a.op == "not" and a.operand == b

    def _constraint_equality_definition(self, expr, ctx: list[str]) -> tuple[str, Expr] | None:
        if not isinstance(expr, BinaryExpr) or expr.op != "==":
            return None

        if isinstance(expr.left, RefExpr):
            target = self._qual_ref_key(expr.left.path, ctx)
            rhs = self._expr(expr.right, ctx, allow_legacy_fallback=False)
        elif isinstance(expr.right, RefExpr):
            target = self._qual_ref_key(expr.right.path, ctx)
            rhs = self._expr(expr.left, ctx, allow_legacy_fallback=False)
        else:
            return None

        if self._is_const_key(target):
            return None
        return target, rhs

    def _build_same_cycle_definitions(self) -> None:
        self._build_binding_definitions()
        self._build_derived_attribute_definitions()
        self._build_constraint_definitions()

        referenced_keys = self._collect_referenced_value_keys()
        self._build_flow_definitions(referenced_keys)

    def _build_binding_definitions(self) -> None:
        for lhs, rhs in sorted(self.parser.parsed_bindings.items()):
            target = self.legacy._canon(lhs.split("::"))
            rhs_key = self.legacy._canon(rhs.split("::"))
            if not self._is_const_key(target):
                self._add_definition(target, Var(rhs_key), "bind")

    def _build_derived_attribute_definitions(self) -> None:
        for derived in self.parser.derived_attributes:
            target = self.legacy._canon(derived.qualified_name.split("::"))
            if target in self.model.state:
                continue
            ctx = self._ctx(derived.context)
            expr = self._expr(derived.expression, ctx, allow_legacy_fallback=False)
            self._add_definition(target, expr, "derived attribute")

    def _build_constraint_definitions(self) -> None:
        for constraint in self.parser.parsed_constraints:
            if "ScenarioConstraint" in getattr(constraint, "metadata", []):
                continue
            ctx = self._ctx(constraint.context)
            implied: dict[str, list[tuple[object, Expr]]] = defaultdict(list)

            for conjunct in self._conjuncts(constraint.expression):
                if isinstance(conjunct, BinaryExpr) and conjunct.op == "implies":
                    rule = self._constraint_equality_definition(conjunct.right, ctx)
                    if rule is None:
                        continue
                    target, rhs = rule
                    implied[target].append((conjunct.left, rhs))
                    continue

                rule = self._constraint_equality_definition(conjunct, ctx)
                if rule is None:
                    continue
                target, rhs = rule
                self._add_definition(target, rhs, f"constraint:{constraint.name}")

            for target, rules in sorted(implied.items()):
                if target in self.model.definitions:
                    continue
                if len(rules) == 2:
                    cond_a, rhs_a = rules[0]
                    cond_b, rhs_b = rules[1]
                    if self._negates(cond_b, cond_a):
                        cond = self._expr(cond_a, ctx, allow_legacy_fallback=False)
                        self._add_definition(
                            target, Ite(cond, rhs_a, rhs_b), f"constraint:{constraint.name}"
                        )
                        continue
                    if self._negates(cond_a, cond_b):
                        cond = self._expr(cond_b, ctx, allow_legacy_fallback=False)
                        self._add_definition(
                            target, Ite(cond, rhs_b, rhs_a), f"constraint:{constraint.name}"
                        )
                        continue

                self.model.add_diagnostic(
                    "error",
                    "partial_implied_definition",
                    "implied equality constraint was not a complete two-branch definition",
                    target,
                )

    def _collect_referenced_value_keys(self) -> set[str]:
        keys = set(self.model.state)

        def add_expr(expr, ctx: list[str], subject_var: str | None = None) -> None:
            for ref in expr_refs(expr):
                key = self._qual_ref_key(ref, ctx, subject_var)
                if key and not self._is_const_key(key):
                    keys.add(key)

        for sa in self.parser.step_actions:
            ctx = self._ctx(sa.context)
            add_expr(sa.expression, ctx)
            add_expr(sa.condition, ctx)

        for fqn, stmts in self.parser.step_action_bodies:
            inst_name = self._inst_name(fqn)
            ctx = self._ctx(fqn)
            for stmt, _cond in self._walk_effective_assigns(stmts, inst_name):
                add_expr(stmt.expr, ctx)

        for constraint in self.parser.parsed_constraints:
            add_expr(constraint.expression, self._ctx(constraint.context))

        for req in self.parser.parsed_requirements:
            add_expr(req.expression, self._ctx(req.context), req.subject_var)

        for derived in self.parser.derived_attributes:
            add_expr(derived.expression, self._ctx(derived.context))
            key = self.legacy._canon(derived.qualified_name.split("::"))
            if not self._is_const_key(key):
                keys.add(key)

        for lhs, rhs in self.parser.parsed_bindings.items():
            for key in (
                self.legacy._canon(lhs.split("::")),
                self.legacy._canon(rhs.split("::")),
            ):
                if not self._is_const_key(key):
                    keys.add(key)

        ctrl_ctx = self.legacy.ctrl_fqn.split("::")
        for ref in self.legacy.in_binds.values():
            key = self._qual_ref_key(ref, ctrl_ctx)
            if key and not self._is_const_key(key):
                keys.add(key)

        return keys

    def _keys_below_prefix(self, prefix: str, keys: set[str]) -> set[str]:
        marker = prefix + "_"
        return {key[len(marker):] for key in keys if key.startswith(marker)}

    def _declared_port_suffixes(self, dotted_port: str) -> set[str]:
        parts = dotted_port.split(".")
        if len(parts) < 2:
            return set()
        inst_name, port_name = parts[0], parts[1]
        inst = self.legacy.inst_by_name.get(inst_name)
        if inst is None:
            return set()
        port_type = self.parser.part_def_ports.get(inst.part_type, {}).get(port_name)
        if not port_type:
            return set()

        suffixes: set[str] = set()
        suffixes.update(self.parser.port_def_attrs.get(port_type, {}))
        for item_name, item_type in self.parser.port_def_items.get(port_type, {}).items():
            attrs = self.parser.item_def_attrs.get(item_type, {})
            if attrs:
                for attr in attrs:
                    suffixes.add(self.legacy._canon([item_name, attr]))
            else:
                suffixes.add(item_name)
        return suffixes

    def _flow_suffixes(self, from_port: str, to_port: str, keys: set[str]) -> set[str]:
        from_prefix = self.legacy._canon(from_port.split("."))
        to_prefix = self.legacy._canon(to_port.split("."))
        suffixes = (
            self._keys_below_prefix(from_prefix, keys)
            | self._keys_below_prefix(to_prefix, keys)
            | self._declared_port_suffixes(from_port)
            | self._declared_port_suffixes(to_port)
        )
        return {suffix for suffix in suffixes if suffix}

    def _build_flow_definitions(self, referenced_keys: set[str]) -> None:
        by_target: dict[str, list[str]] = defaultdict(list)
        flow_sources: dict[str, list[str]] = defaultdict(list)

        for flow in self.parser.flows:
            from_prefix = self.legacy._canon(flow.from_port.split("."))
            to_prefix = self.legacy._canon(flow.to_port.split("."))
            suffixes = self._flow_suffixes(flow.from_port, flow.to_port, referenced_keys)
            for suffix in suffixes:
                source = f"{from_prefix}_{suffix}"
                target = f"{to_prefix}_{suffix}"
                by_target[target].append(source)
                flow_sources[target].append(f"{flow.from_port} -> {flow.to_port}")

        for target, sources in sorted(by_target.items()):
            unique_sources = sorted(set(sources))
            if target in self.model.definitions:
                continue
            if len(unique_sources) == 1:
                self._add_definition(target, Var(unique_sources[0]), "flow copy")
                continue
            self.model.add_diagnostic(
                "error",
                "unsupported_flow_merge",
                "multiple flow sources need an equation in the SysML file",
                f"{target} <- {', '.join(unique_sources)}",
            )

    def _diagnose_multiple_flows(self) -> None:
        by_to: dict[str, list[str]] = defaultdict(list)
        for flow in self.parser.flows:
            by_to[self.legacy._canon(flow.to_port.split("."))].append(
                self.legacy._canon(flow.from_port.split("."))
            )
        for to_port, sources in sorted(by_to.items()):
            if len(sources) > 1:
                self.model.add_diagnostic(
                    "warning",
                    "multiple_flow_sources",
                    "multiple flow sources require an explicit merge equation for strict certification",
                    f"{to_port} <- {', '.join(sorted(sources))}",
                )

    def _build_observation_equations(self) -> None:
        ctx = self.legacy.ctrl_fqn.split("::")
        neural_def = self._neural_action_def()
        completion_names = {
            item.name for item in neural_def.in_params
            if "Completion" in item.metadata
        } if neural_def is not None else set()
        for obs_name, ref in sorted(self.legacy.in_binds.items()):
            if obs_name in completion_names:
                continue
            expr = self._resolve_ref(ref, ctx)
            self.model.observations[obs_name] = Equation(
                target=f"obs.{obs_name}",
                expr=expr,
                kind="observation",
                source="Neural input binding",
            )

    def _neural_action_def(self):
        found = []
        for part_def in self.parser.part_defs.values():
            for action_def in part_def.action_defs:
                if "Neural" in action_def.metadata:
                    found.append(action_def)
        return found[0] if len(found) == 1 else None

    def _controller_part_def(self):
        ctrl_inst = self.legacy.inst_by_name.get(self.legacy.ctrl_inst)
        if ctrl_inst is None:
            return None
        return self.parser.part_defs.get(ctrl_inst.part_type)

    def _build_terminal_equations(self) -> None:
        neural_def = self._neural_action_def()
        ctrl_def = self._controller_part_def()
        if neural_def is None or ctrl_def is None:
            self.model.add_diagnostic(
                "warning",
                "missing_terminal_binding",
                "could not locate the unique controller #Neural action",
            )
            return
        completion_names = [
            item.name for item in neural_def.in_params
            if "Completion" in item.metadata
        ]
        if len(completion_names) != 1:
            self.model.add_diagnostic(
                "warning",
                "missing_terminal_binding",
                "#Neural action must mark exactly one input #Completion",
            )
            return
        completion_name = completion_names[0]

        ctx = self.legacy.ctrl_fqn.split("::")
        for action in ctrl_def.actions:
            for stmt in self.legacy._flatten(action.body):  # pylint: disable=protected-access
                if not isinstance(stmt, SubactionCallStmt) or stmt.type_name != neural_def.name:
                    continue
                for binding in stmt.bindings:
                    if not isinstance(binding, InputBindingStmt):
                        continue
                    if binding.name != completion_name:
                        continue
                    target = f"env.completion.{completion_name}"
                    self.model.terminals[target] = Equation(
                        target=target,
                        expr=self._expr(binding.expr, ctx),
                        kind="terminal",
                        source=f"#Completion input binding in {action.name}",
                    )
                    return

        self.model.add_diagnostic(
            "warning",
            "missing_terminal_binding",
            f"#Completion input {completion_name} has no binding",
        )

    def _build_requirement_equations(self) -> None:
        for req in self.parser.parsed_requirements:
            ctx = self._ctx(req.context)
            expr = self._expr(req.expression, ctx, req.subject_var)
            target = f"status.{req.name}"
            self.model.requirements[target] = Equation(
                target=target,
                expr=expr,
                kind="requirement",
                source=",".join(req.metadata) or "Requirement",
            )

    def _build_transition_equations(self) -> None:
        assigned = Counter()
        performed_assigned = Counter()
        for sa in self.parser.step_actions:
            target = self.legacy._canon(sa.target_key.split("::"))
            assigned[target] += 1
            ctx = self._ctx(sa.context)
            rhs = self._expr(sa.expression, ctx)
            if sa.condition is not None:
                cond = self._expr(sa.condition, ctx)
                rhs = Ite(cond, rhs, Var(target))
                self.model.add_diagnostic(
                    "info",
                    "guarded_step_assignment",
                    "guarded step assignment encoded as ite(cond, rhs, old_value)",
                    target,
                )
            self.model.transitions[target] = Equation(
                target=target,
                expr=rhs,
                kind="transition",
                source=f"step:{sa.action_name}",
            )

        for fqn, stmts in self.parser.step_action_bodies:
            inst_name = self._inst_name(fqn)
            ctx = self._ctx(fqn)
            for stmt, cond in self._walk_effective_assigns(stmts, inst_name):
                if not self._is_instance_attribute_assignment(inst_name, stmt):
                    continue
                target = self.legacy._canon([inst_name] + stmt.target)
                performed_assigned[target] += 1
                if target in self.model.transitions:
                    continue
                rhs = self._expr(stmt.expr, ctx)
                if cond is not None:
                    rhs = Ite(cond, rhs, Var(target))
                    self.model.add_diagnostic(
                        "info",
                        "performed_guarded_assignment",
                        "performed-action assignment encoded as ite(cond, rhs, old_value)",
                        target,
                    )
                self.model.transitions[target] = Equation(
                    target=target,
                    expr=rhs,
                    kind="performed_transition",
                    source="flattened performed action inside step",
                )

        self._build_state_machine_equations()

        for target, count in sorted(assigned.items()):
            if count > 1:
                self.model.add_diagnostic(
                    "error",
                    "multiple_step_assignments",
                    "multiple assignments to one target need ordered SSA semantics",
                    target,
                )
        for target, count in sorted(performed_assigned.items()):
            if count > 1:
                self.model.add_diagnostic(
                    "error",
                    "multiple_performed_assignments",
                    "multiple performed-action assignments to one target need ordered SSA semantics",
                    target,
                )

    def _audit_equations(self) -> None:
        for target in sorted(self.model.state - set(self.model.transitions)):
            self.model.add_diagnostic(
                "error",
                "missing_transition_equation",
                "state value has no transition equation in the SysML file",
                target,
            )
        known_vars = self.model.state | self.model.actions | set(self.model.definitions)
        for eq in self.model.all_equations():
            unresolved_vars = sorted(ref for ref in eq.refs() if ref not in known_vars)
            if unresolved_vars:
                self.model.add_diagnostic(
                    "error",
                    "unresolved_equation_ref",
                    "equation references symbols outside state/action/definition sets",
                    f"{eq.target}: {', '.join(unresolved_vars)}",
                )

            unresolved_raw = sorted(
                ref for ref in eq.raw_refs()
                if ref not in self.model.constants
            )
            if unresolved_raw:
                self.model.add_diagnostic(
                    "error",
                    "unresolved_raw_ref",
                    "equation contains raw references that are not classified as constants",
                    f"{eq.target}: {', '.join(unresolved_raw)}",
                )

    def _build_state_machine_equations(self) -> None:
        cmap = {}
        for frm, to in self.parser.connects:
            a = self.legacy._pc(frm)  # pylint: disable=protected-access
            b = self.legacy._pc(to)  # pylint: disable=protected-access
            cmap[a] = b
            cmap[b] = a

        for fqn, sm in self.parser.instance_state_machines.items():
            inst = self._inst_name(fqn)
            ctx = self._ctx(fqn)
            state_changing = any(t.from_state != t.to_state for t in sm.transitions)
            latch = self.legacy._latch_action(sm, inst)  # pylint: disable=protected-access
            trig_vars = {t.trigger_var for t in sm.transitions if t.trigger_var}
            trig_port = next((t.trigger_port for t in sm.transitions if t.trigger_port), None)

            if latch and state_changing:
                self._add_policy_latch_state_machine(inst, sm, latch, ctx)
                continue

            for transition in sm.transitions:
                for stmt in transition.do_action or []:
                    if not isinstance(stmt, AssignStmt):
                        continue
                    target = self.legacy._canon([inst] + stmt.target)
                    refs = expr_refs(stmt.expr)
                    if any(ref and ref[0] in trig_vars for ref in refs):
                        dec = self.legacy._actuator_decision(  # pylint: disable=protected-access
                            inst, trig_port, cmap
                        )
                        if dec:
                            self.model.state.add(target)
                            self.model.transitions[target] = Equation(
                                target=target,
                                expr=Var(dec),
                                kind="state_machine_transition",
                                source=f"{inst}.{transition.name}: coil-write latch",
                            )
                            self.model.add_diagnostic(
                                "info",
                                "coil_latch_equation",
                                "coil-write actuator latch coupled to controller policy output",
                                target,
                            )

    def _add_policy_latch_state_machine(self, inst: str, sm, latch: str, ctx: list[str]) -> None:
        state_var = f"{inst}_state"
        self.model.state.add(state_var)
        self.model.transitions[state_var] = Equation(
            target=state_var,
            expr=Var(latch),
            kind="state_machine_transition",
            source=f"{inst}.{sm.name}: policy latch state",
        )
        self.model.add_diagnostic(
            "info",
            "policy_latch_state_equation",
            "two-state actuator state coupled to controller policy output",
            state_var,
        )

        active_expr = None
        initial_expr = None
        for transition in sm.transitions:
            for stmt in transition.do_action or []:
                if not isinstance(stmt, AssignStmt):
                    continue
                target = self.legacy._canon([inst] + stmt.target)
                self.model.state.add(target)
                expr = self._expr(stmt.expr, ctx)
                if transition.to_state == sm.initial_state:
                    initial_expr = expr
                elif transition.to_state is not None:
                    active_expr = expr

        if active_expr is None or initial_expr is None:
            self.model.add_diagnostic(
                "warning",
                "state_machine_output_partial",
                "could not derive output equations for the initial and alternate states",
                inst,
            )
            return

        for transition in sm.transitions:
            for stmt in transition.do_action or []:
                if not isinstance(stmt, AssignStmt):
                    continue
                target = self.legacy._canon([inst] + stmt.target)
                self.model.transitions[target] = Equation(
                    target=target,
                    expr=Ite(Var(latch), active_expr, initial_expr),
                    kind="state_machine_transition",
                    source=f"{inst}.{sm.name}: policy latch output",
                )
                self.model.add_diagnostic(
                    "info",
                    "policy_latch_output_equation",
                    "two-state output derived from initial and alternate state assignments",
                    target,
                )
                return


def extract_equation_model(model_path: str) -> EquationModel:
    return CertificationExtractor(model_path).extract()
