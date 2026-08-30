"""Small symbolic equation IR for certification.

The first implementation is intentionally lightweight: enough structure to keep
guarded assignments, observations, requirements, and dependency references
explicit while the verifier grows toward the contract in
`CERTIFICATION_SEMANTICS.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


class Expr:
    """Base class for symbolic expressions."""

    def refs(self) -> set[str]:
        return set()

    def raw_refs(self) -> set[str]:
        return set()

    def pretty(self) -> str:
        return str(self)


@dataclass(frozen=True)
class Const(Expr):
    value: Any

    def pretty(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def refs(self) -> set[str]:
        return {self.name}

    def pretty(self) -> str:
        return self.name


@dataclass(frozen=True)
class RawRef(Expr):
    """A reference that could not yet be resolved to a kept model variable."""

    path: str

    def raw_refs(self) -> set[str]:
        return {self.path}

    def pretty(self) -> str:
        return self.path


@dataclass(frozen=True)
class Op(Expr):
    op: str
    args: tuple[Expr, ...]

    def refs(self) -> set[str]:
        out: set[str] = set()
        for arg in self.args:
            out |= arg.refs()
        return out

    def raw_refs(self) -> set[str]:
        out: set[str] = set()
        for arg in self.args:
            out |= arg.raw_refs()
        return out

    def pretty(self) -> str:
        if self.op == "not" and len(self.args) == 1:
            return f"(not {self.args[0].pretty()})"
        if len(self.args) == 1:
            return f"{self.op}({self.args[0].pretty()})"
        if len(self.args) == 2:
            return f"({self.args[0].pretty()} {self.op} {self.args[1].pretty()})"
        return f"{self.op}({', '.join(a.pretty() for a in self.args)})"


@dataclass(frozen=True)
class Ite(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def refs(self) -> set[str]:
        return self.cond.refs() | self.then_expr.refs() | self.else_expr.refs()

    def raw_refs(self) -> set[str]:
        return (
            self.cond.raw_refs()
            | self.then_expr.raw_refs()
            | self.else_expr.raw_refs()
        )

    def pretty(self) -> str:
        return (
            f"ite({self.cond.pretty()}, "
            f"{self.then_expr.pretty()}, {self.else_expr.pretty()})"
        )


@dataclass(frozen=True)
class Equation:
    target: str
    expr: Expr
    kind: str
    source: str = ""

    def refs(self) -> set[str]:
        return self.expr.refs()

    def raw_refs(self) -> set[str]:
        return self.expr.raw_refs()

    def pretty(self) -> str:
        temporal_kinds = {
            "transition",
            "performed_transition",
            "state_machine_transition",
            "legacy_transition_dependency",
        }
        prefix = f"[{self.kind}] " if self.kind else ""
        target = f"{self.target}'" if self.kind in temporal_kinds else self.target
        suffix = f"    # {self.source}" if self.source else ""
        return f"{prefix}{target} = {self.expr.pretty()}{suffix}"


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    subject: str = ""

    def pretty(self) -> str:
        loc = f" ({self.subject})" if self.subject else ""
        return f"{self.severity.upper()} {self.code}{loc}: {self.message}"


@dataclass
class EquationModel:
    model_path: str
    state: set[str]
    actions: set[str]
    constants: set[str] = field(default_factory=set)
    initial_values: dict[str, Any] = field(default_factory=dict)
    definitions: dict[str, Equation] = field(default_factory=dict)
    observations: dict[str, Equation] = field(default_factory=dict)
    terminals: dict[str, Equation] = field(default_factory=dict)
    transitions: dict[str, Equation] = field(default_factory=dict)
    requirements: dict[str, Equation] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def add_diagnostic(self, severity: str, code: str, message: str, subject: str = "") -> None:
        self.diagnostics.append(Diagnostic(severity, code, message, subject))

    def all_equations(self) -> Iterable[Equation]:
        yield from self.definitions.values()
        yield from self.observations.values()
        yield from self.terminals.values()
        yield from self.transitions.values()
        yield from self.requirements.values()
