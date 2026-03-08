"""
extractor.py — Extract the RL interface from a SysML v2 model.

This module is the bridge between SysML v2 specifications and RL training.
It parses a SysML model, finds the `#Neural`-annotated action def, and
extracts everything needed to construct a Gymnasium environment:

  - **Observations** (`in` parameters of the #Neural action def)
  - **Actions** (`out` parameters of the #Neural action def)
  - **Done expression** (the binding expression for the `done` input,
    converted to a Python boolean expression using observation names)
  - **Observation bindings** (which SysML ref each observation is bound to,
    enabling the env to map obs names to simulator state keys)

What SysML Models Must Provide
------------------------------
For this extractor to work, the SysML model must contain:

1. Exactly one action def annotated with `#Neural`:

       #Neural action def PolicyName {
           in obs1 : Real;        // becomes an observation
           in obs2 : Real;
           in done : Boolean;     // goal-completion signal
           out act1 : Boolean;    // becomes an action
           out act2 : Boolean;
       }

2. A SubactionCallStmt that invokes that action def with InputBindingStmts
   that bind each `in` parameter to a SysML expression:

       perform policyStep : PolicyName {
           in obs1 = sensorReading.response;
           in done = (obs1 >= threshold) and (obs2 >= threshold);
           out act1;
       }

   The `done` binding expression is converted to Python and used for both
   boolean goal checking and continuous distance-to-goal computation (for
   reward shaping). Other input bindings record the ref path so the env
   can discover the corresponding simulator state keys.

Output
------
A `NeuralInterface` dataclass containing all extracted information.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add sysml-models directory to path so we can import the parser.
_SYSML_MODELS_DIR = str(
    Path(__file__).resolve().parent.parent / "sysml-models"
)
if _SYSML_MODELS_DIR not in sys.path:
    sys.path.insert(0, _SYSML_MODELS_DIR)

from sysml_parser import (
    SysMLParser, SubactionCallStmt, InputBindingStmt,
    RefExpr, BinaryExpr, LiteralExpr, UnaryExpr,
)


@dataclass
class NeuralInterface:
    """Everything the RL env needs from the SysML model.

    Attributes:
        obs_names:    List of observation parameter names (from `in` params).
        obs_types:    Corresponding SysML type names (e.g., "Real", "Boolean").
        action_names: List of action parameter names (from `out` params).
        action_types: Corresponding SysML type names.
        done_expr:    Python expression string for the goal condition, using
                      obs_names as variable names. None if no `done` param.
                      Example: "(tank1VolumeMl) >= (tank1TargetTransferMl)"
        obs_bindings: Maps obs_name -> SysML ref path from the binding stmt.
                      Example: {"tank1VolumeMl": "volume1Res.response"}
                      Used by the env to discover simulator state keys for
                      start-state randomization.
    """
    obs_names: list[str]
    obs_types: list[str]
    action_names: list[str]
    action_types: list[str]
    done_expr: Optional[str] = None
    obs_bindings: Optional[dict[str, str]] = None
    scenario_inputs: list = None       # Parameter objects with #ScenarioInput
    scenario_constraints: list = None  # Constraint objects with #ScenarioConstraint
    neural_requirements: list = None   # (name, raw_text) from #NeuralRequirement


def extract_neural_interface(model_path: str) -> NeuralInterface:
    """Parse a SysML model and return the #Neural action def's interface.

    Args:
        model_path: Path to the .sysml model file.

    Returns:
        NeuralInterface with observations, actions, done expression, and
        observation bindings extracted from the model.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the model has zero or more than one #Neural action def.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SysML model not found: {model_path}")

    parser = SysMLParser(str(path))
    parser.parse()

    # Find the #Neural action def — there must be exactly one.
    neural_defs = []
    for part_def in parser.part_defs.values():
        for action_def in part_def.action_defs:
            if 'Neural' in action_def.metadata:
                neural_defs.append(action_def)

    if len(neural_defs) != 1:
        raise ValueError(
            f"Expected exactly one #Neural action def, found {len(neural_defs)} in {model_path}"
        )

    nd = neural_defs[0]
    obs_names = [p.name for p in nd.in_params]
    obs_types = [p.type_name for p in nd.in_params]
    action_names = [p.name for p in nd.out_params]
    action_types = [p.type_name for p in nd.out_params]

    # Extract the done binding expression and obs bindings from the call site.
    done_expr, obs_bindings = _extract_bindings(parser, nd.name, obs_names)

    # Extract #ScenarioInput parameters and #ScenarioConstraint constraints.
    scenario_inputs = [p for p in parser.parameters
                       if 'ScenarioInput' in p.metadata]
    scenario_constraints = [c for c in parser.parsed_constraints
                            if 'ScenarioConstraint' in c.metadata]

    # Extract #NeuralRequirement constraints on the policy I/O.
    # These are requirement defs inside part defs whose subject is the
    # neural action def. Strip the subject prefix (e.g. "p.") from the
    # raw expression so variable names match obs/action names directly.
    neural_requirements = []
    for part_def in parser.part_defs.values():
        for req_name, subject_var, _subject_type, req_expr, req_metadata in part_def.requirements:
            if 'NeuralRequirement' in req_metadata:
                import re
                clean_expr = re.sub(
                    rf'\b{re.escape(subject_var)}\.', '', req_expr)
                neural_requirements.append((req_name, clean_expr))

    return NeuralInterface(
        obs_names=obs_names, obs_types=obs_types,
        action_names=action_names, action_types=action_types,
        done_expr=done_expr, obs_bindings=obs_bindings,
        scenario_inputs=scenario_inputs,
        scenario_constraints=scenario_constraints,
        neural_requirements=neural_requirements or None,
    )


def _extract_bindings(parser, neural_name: str, obs_names: list[str]) -> tuple[Optional[str], dict[str, str]]:
    """Find the SubactionCallStmt that invokes the Neural action and extract bindings.

    Searches all action bodies in the model for a call to `neural_name`.
    From that call's InputBindingStmts:
      - The `done` param's expression is converted to a Python string.
      - Other params' RefExpr paths are recorded in obs_bindings.
    """
    done_name = None
    for name in obs_names:
        if name.lower() == 'done':
            done_name = name
            break

    for part_def in parser.part_defs.values():
        for action in part_def.actions:
            result = _search_stmts(action.body, neural_name, done_name, obs_names)
            if result is not None:
                return result
    return None, {}


def _search_stmts(stmts, neural_name, done_name, obs_names) -> Optional[tuple[Optional[str], dict[str, str]]]:
    """Recursively search statement lists for the Neural call."""
    for stmt in stmts:
        if isinstance(stmt, SubactionCallStmt) and stmt.type_name == neural_name:
            ref_map = {}
            obs_bindings = {}
            done_expr_ast = None
            for b in stmt.bindings:
                if not isinstance(b, InputBindingStmt):
                    continue
                if done_name and b.name == done_name:
                    done_expr_ast = b.expr
                elif isinstance(b.expr, RefExpr):
                    ref_path = '.'.join(b.expr.path)
                    ref_map[ref_path] = b.name
                    obs_bindings[b.name] = ref_path

            done_expr = None
            if done_expr_ast is not None:
                try:
                    done_expr = _to_python(done_expr_ast, ref_map)
                except Exception:
                    pass
            return done_expr, obs_bindings

        if hasattr(stmt, 'body') and isinstance(getattr(stmt, 'body'), list):
            result = _search_stmts(stmt.body, neural_name, done_name, obs_names)
            if result is not None:
                return result
    return None


def _to_python(expr, ref_map: dict) -> str:
    """Convert a SysML expression AST node to a Python expression string.

    Substitutes RefExprs using ref_map (sensor ref path -> obs name),
    so the resulting expression uses observation variable names directly.

    Example: BinaryExpr(RefExpr(["volume1Res","response"]), ">=", LiteralExpr(100))
             with ref_map {"volume1Res.response": "tank1VolumeMl"}
             -> "(tank1VolumeMl) >= (100)"
    """
    if isinstance(expr, LiteralExpr):
        return repr(expr.value)
    if isinstance(expr, RefExpr):
        key = '.'.join(expr.path)
        return ref_map.get(key, key.replace('.', '_'))
    if isinstance(expr, BinaryExpr):
        left = _to_python(expr.left, ref_map)
        right = _to_python(expr.right, ref_map)
        op = {'and': 'and', 'or': 'or'}.get(expr.op, expr.op)
        return f"({left}) {op} ({right})"
    if isinstance(expr, UnaryExpr):
        operand = _to_python(expr.operand, ref_map)
        return f"{'not ' if expr.op == 'not' else expr.op}({operand})"
    return "False"
