"""Transition-closed relevance analysis for the equation model."""

from __future__ import annotations

from dataclasses import dataclass

from .equations import EquationModel


@dataclass(frozen=True)
class RelevanceResult:
    seeds: set[str]
    q: set[str]
    action_deps: set[str]
    ignored_state: set[str]
    ignored_state_reasons: dict[str, str]
    seed_equations: dict[str, list[str]]

    def pretty(self) -> str:
        lines = [
            f"seed vars ({len(self.seeds)}): {sorted(self.seeds)}",
            f"transition-closed q ({len(self.q)}): {sorted(self.q)}",
            f"action deps ({len(self.action_deps)}): {sorted(self.action_deps)}",
            f"ignored state ({len(self.ignored_state)}): {sorted(self.ignored_state)}",
        ]
        if self.ignored_state:
            lines.append("ignored-state noninterference reasons:")
            for var in sorted(self.ignored_state):
                lines.append(f"  - {var}: {self.ignored_state_reasons.get(var, '')}")
        return "\n".join(lines)


def expand_definition_refs(
    model: EquationModel, refs: set[str], seen: set[str] | None = None
) -> set[str]:
    """Follow same-cycle definitions until state/action/unknown leaves remain."""

    expanded: set[str] = set()
    stack = list(refs)
    seen_defs = set(seen or set())

    while stack:
        ref = stack.pop()
        definition = model.definitions.get(ref)
        if definition is None:
            expanded.add(ref)
            continue
        if ref in seen_defs:
            expanded.add(ref)
            continue
        seen_defs.add(ref)
        stack.extend(definition.refs())

    return expanded


def equation_refs(model: EquationModel, eq) -> set[str]:
    return expand_definition_refs(model, eq.refs())


def compute_transition_closed_relevance(model: EquationModel) -> RelevanceResult:
    """Compute a first backward-cone relevant state target.

    This is dependency-based over the new equation view. It is not yet a proof of
    reconstructibility; it computes the target `q` that later proof phases should
    reconstruct.
    """

    seed_refs: set[str] = set()
    seed_equations: dict[str, list[str]] = {}
    terminal_equations = list(getattr(model, "terminals", {}).values())
    for eq in (
        list(model.observations.values())
        + terminal_equations
        + list(model.requirements.values())
    ):
        refs = equation_refs(model, eq)
        seed_refs |= refs
        seed_equations[eq.target] = sorted(refs)

    q = {v for v in seed_refs if v in model.state}
    action_deps = {v for v in seed_refs if v in model.actions}

    changed = True
    while changed:
        changed = False
        for target, eq in model.transitions.items():
            if target not in q:
                continue
            refs = equation_refs(model, eq)
            for ref in refs:
                if ref in model.actions:
                    if ref not in action_deps:
                        action_deps.add(ref)
                        changed = True
                elif ref in model.state and ref not in q:
                    q.add(ref)
                    changed = True

    ignored_state = set(model.state) - q
    ignored_state_reasons = {
        var: (
            "outside the backward cone of observation, terminal, requirement, "
            "reward/done, shield-input, and transition-closed q-next equations "
            "under the extracted equation model"
        )
        for var in ignored_state
    }

    return RelevanceResult(
        seeds=q & seed_refs,
        q=q,
        action_deps=action_deps,
        ignored_state=ignored_state,
        ignored_state_reasons=ignored_state_reasons,
        seed_equations=seed_equations,
    )
