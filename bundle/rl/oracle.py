"""
Specification-derived oracle for SysML-extracted neural controllers.

Derives correct action labels directly from the #NeuralRequirement AST —
no simulation, no propagationDelay, no lookahead. The requirement is a
static function from observations to actions; the oracle evaluates it.

Also extracts observation names, action names, and the input marked
`#Completion` from the SysML model.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysml-models"))

from sysml_parser import (
    SysMLParser,
    IfStmt,
    InputBindingStmt,
    PerformStmt,
    SubactionCallStmt,
)
from shield import SpecShield


# ---------------------------------------------------------------------------
# SysML interface extraction
# ---------------------------------------------------------------------------

def extract_interface(model_path: str, dt: float = 0.1):
    """Extract observation, action, and completion names from SysML.

    Returns dict with everything the oracle and environment need.
    No simulation engine, no propagationDelay — the oracle is derived
    directly from the #NeuralRequirement via SpecShield.
    """
    parser = SysMLParser(model_path)
    parser.parse()

    # Find #Neural action def
    ctrl_inst = parser.part_instances[parser.controller_part]
    ctrl_def = parser.part_defs[ctrl_inst.part_type]

    neural_def = None
    for ad in ctrl_def.action_defs:
        if 'Neural' in ad.metadata:
            neural_def = ad
            break
    if not neural_def:
        raise ValueError("No #Neural action def found")

    actions = {action.name: action.body for action in ctrl_def.actions}

    def calls(stmts, active_actions=frozenset()):
        for statement in stmts:
            if isinstance(statement, SubactionCallStmt):
                if statement.type_name == neural_def.name:
                    yield statement
            elif isinstance(statement, IfStmt):
                yield from calls(statement.body, active_actions)
                yield from calls(statement.else_body, active_actions)
            elif (
                isinstance(statement, PerformStmt)
                and statement.action_name in actions
                and statement.action_name not in active_actions
            ):
                yield from calls(
                    actions[statement.action_name],
                    active_actions | {statement.action_name},
                )

    call_by_identity = {
        id(call): call
        for action in ctrl_def.actions
        for call in calls(action.body)
    }
    if len(call_by_identity) != 1:
        raise ValueError(
            f"expected exactly one call to #Neural action {neural_def.name}, "
            f"found {len(call_by_identity)}"
        )
    call_stmt = next(iter(call_by_identity.values()))

    completion_names = [
        p.name for p in neural_def.in_params if "Completion" in p.metadata
    ]
    if len(completion_names) != 1:
        raise ValueError("#Neural action must mark exactly one input #Completion")
    completion_name = completion_names[0]
    obs_names = [
        p.name for p in neural_def.in_params if "Completion" not in p.metadata
    ]
    action_names = [p.name for p in neural_def.out_params]
    completion_bindings = [
        binding for binding in call_stmt.bindings
        if isinstance(binding, InputBindingStmt) and binding.name == completion_name
    ]
    if len(completion_bindings) != 1:
        raise ValueError(
            f"#Completion input {completion_name} must have exactly one binding"
        )

    def is_done(values):
        if completion_name not in values:
            raise KeyError(f"missing #Completion value: {completion_name}")
        return bool(values[completion_name])

    # Build SpecShield for oracle labeling
    spec_shield = SpecShield(model_path)

    return {
        "obs_names": obs_names,
        "action_names": action_names,
        "completion_name": completion_name,
        "is_done": is_done,
        "spec_shield": spec_shield,
    }


# ---------------------------------------------------------------------------
# spec_oracle: label actions from the requirement AST
# ---------------------------------------------------------------------------

def spec_oracle(spec_shield: SpecShield, obs_dict: dict):
    """Derive the correct action directly from the #NeuralRequirement.

    Evaluates the requirement AST for every valid action combo against
    the current observations. Returns the correct action as an integer.

    This is equivalent to what the shield does at runtime, but used at
    training time to generate (observation, action) labels for the oracle
    cloning phase.
    """
    return spec_shield._requirement_action(obs_dict)
