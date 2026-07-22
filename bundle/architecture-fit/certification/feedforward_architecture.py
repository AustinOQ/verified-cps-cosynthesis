"""Derive a feedforward policy width from a SysML neural requirement.

The discrete rules in this artifact have two stages: numeric comparisons over
observations, followed by Boolean output logic.  The existing policy also has
two hidden layers of equal width.  The derived width therefore reserves one
first-layer unit per distinct comparison boundary and one second-layer unit
per Boolean output, using the larger count for both layers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sysml_parser import (
    BinaryExpr,
    LiteralExpr,
    RefExpr,
    TernaryExpr,
    UnaryExpr,
)


METHOD = "sysml_comparison_output_width_v1"
CONTINUOUS_METHOD = "sysml_comparison_real_output_width_v1"
POLICY_CLASS = "two_equal_hidden_layer_tanh_actor_critic"
CONTINUOUS_POLICY_CLASS = "two_equal_hidden_layer_tanh_gaussian_actor_critic"
COMPARISON_OPS = {"<", "<=", ">", ">="}
COMMUTATIVE_OPS = {"+", "*", "and", "or", "=="}


def feedforward_parameter_count(
    input_dim: int,
    action_count: int,
    hidden_dim: int,
) -> int:
    """Exact parameter count for ``MLPActorCritic``."""
    input_dim = int(input_dim)
    action_count = int(action_count)
    hidden_dim = int(hidden_dim)
    if input_dim <= 0 or action_count <= 0 or hidden_dim <= 0:
        raise ValueError("feedforward dimensions must be positive")
    return (
        hidden_dim * hidden_dim
        + (input_dim + action_count + 3) * hidden_dim
        + action_count
        + 1
    )


def continuous_feedforward_parameter_count(
    input_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> int:
    """Exact parameter count for ``GaussianMLPActorCritic``."""
    input_dim = int(input_dim)
    action_dim = int(action_dim)
    hidden_dim = int(hidden_dim)
    if input_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
        raise ValueError("feedforward dimensions must be positive")
    return (
        hidden_dim * hidden_dim
        + (input_dim + action_dim + 3) * hidden_dim
        + 2 * action_dim
        + 1
    )


def _ref_name(expr: RefExpr, subject_var: str) -> tuple[str, ...]:
    path = tuple(str(part) for part in expr.path)
    if path and path[0] == subject_var:
        path = path[1:]
    return path


def _expression_key(expr: Any, subject_var: str):
    if isinstance(expr, LiteralExpr):
        return ("literal", type(expr.value).__name__, repr(expr.value))
    if isinstance(expr, RefExpr):
        return ("ref",) + _ref_name(expr, subject_var)
    if isinstance(expr, UnaryExpr):
        return ("unary", expr.op, _expression_key(expr.operand, subject_var))
    if isinstance(expr, BinaryExpr):
        left = _expression_key(expr.left, subject_var)
        right = _expression_key(expr.right, subject_var)
        if expr.op in COMMUTATIVE_OPS and repr(right) < repr(left):
            left, right = right, left
        return ("binary", expr.op, left, right)
    if isinstance(expr, TernaryExpr):
        return (
            "ternary",
            _expression_key(expr.condition, subject_var),
            _expression_key(expr.true_expr, subject_var),
            _expression_key(expr.false_expr, subject_var),
        )
    return ("unknown", type(expr).__name__)


def _render_expression(expr: Any, subject_var: str) -> str:
    if isinstance(expr, LiteralExpr):
        return repr(expr.value)
    if isinstance(expr, RefExpr):
        return ".".join(_ref_name(expr, subject_var))
    if isinstance(expr, UnaryExpr):
        return f"({expr.op} {_render_expression(expr.operand, subject_var)})"
    if isinstance(expr, BinaryExpr):
        return (
            f"({_render_expression(expr.left, subject_var)} {expr.op} "
            f"{_render_expression(expr.right, subject_var)})"
        )
    if isinstance(expr, TernaryExpr):
        return (
            f"({_render_expression(expr.condition, subject_var)} ? "
            f"{_render_expression(expr.true_expr, subject_var)} : "
            f"{_render_expression(expr.false_expr, subject_var)})"
        )
    return type(expr).__name__


def _reference_names(expr: Any, subject_var: str) -> set[str]:
    if isinstance(expr, RefExpr):
        path = _ref_name(expr, subject_var)
        return {path[-1]} if path else set()
    if isinstance(expr, BinaryExpr):
        return (
            _reference_names(expr.left, subject_var)
            | _reference_names(expr.right, subject_var)
        )
    if isinstance(expr, UnaryExpr):
        return _reference_names(expr.operand, subject_var)
    if isinstance(expr, TernaryExpr):
        return (
            _reference_names(expr.condition, subject_var)
            | _reference_names(expr.true_expr, subject_var)
            | _reference_names(expr.false_expr, subject_var)
        )
    return set()


def _comparison_boundaries(spec_shield) -> tuple[list[dict[str, Any]], int]:
    input_names = set(spec_shield.in_params)
    output_names = set(spec_shield.out_params)
    subject_var = spec_shield.subject_var
    grouped: dict[tuple, dict[str, Any]] = {}
    operators: dict[tuple, set[str]] = defaultdict(set)
    occurrences = 0

    def visit(expr: Any) -> None:
        nonlocal occurrences
        if isinstance(expr, BinaryExpr):
            if expr.op in COMPARISON_OPS:
                references = _reference_names(expr, subject_var)
                if references & input_names and not references & output_names:
                    occurrences += 1
                    left_key = _expression_key(expr.left, subject_var)
                    right_key = _expression_key(expr.right, subject_var)
                    left_text = _render_expression(expr.left, subject_var)
                    right_text = _render_expression(expr.right, subject_var)
                    if repr(right_key) < repr(left_key):
                        left_key, right_key = right_key, left_key
                        left_text, right_text = right_text, left_text
                    key = (left_key, right_key)
                    grouped.setdefault(key, {
                        "left": left_text,
                        "right": right_text,
                    })
                    operators[key].add(expr.op)
            visit(expr.left)
            visit(expr.right)
        elif isinstance(expr, UnaryExpr):
            visit(expr.operand)
        elif isinstance(expr, TernaryExpr):
            visit(expr.condition)
            visit(expr.true_expr)
            visit(expr.false_expr)

    if spec_shield.req_ast is not None:
        visit(spec_shield.req_ast)
    boundaries = []
    for key in sorted(grouped, key=repr):
        boundaries.append({
            **grouped[key],
            "operators": sorted(operators[key]),
        })
    return boundaries, occurrences


def derive_feedforward_architecture(
    spec_shield,
    *,
    input_dim: int,
    action_count: int,
) -> dict[str, Any]:
    """Return the single feedforward architecture induced by the rule AST."""
    boundaries, occurrences = _comparison_boundaries(spec_shield)
    comparison_count = len(boundaries)
    output_count = len(spec_shield.out_params)
    if comparison_count <= 0:
        raise ValueError("neural requirement contains no input comparison boundary")
    if output_count <= 0:
        raise ValueError("neural action has no Boolean outputs")
    if int(action_count) != 2 ** output_count:
        raise ValueError(
            "discrete action count does not match Boolean output encoding: "
            f"actions={action_count}, outputs={output_count}"
        )

    hidden_dim = max(comparison_count, output_count)
    parameter_count = feedforward_parameter_count(
        input_dim, action_count, hidden_dim)
    return {
        "method": METHOD,
        "policy_class": POLICY_CLASS,
        "hidden_layers": [hidden_dim, hidden_dim],
        "hidden_dim": hidden_dim,
        "input_dim": int(input_dim),
        "action_count": int(action_count),
        "distinct_comparison_boundaries": comparison_count,
        "comparison_occurrences": occurrences,
        "boolean_output_count": output_count,
        "comparison_boundaries": boundaries,
        "parameter_count": parameter_count,
        "parameter_bytes_float32": 4 * parameter_count,
    }


def derive_continuous_feedforward_architecture(
    requirement,
    *,
    input_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    """Return one continuous feedforward architecture from the requirement."""
    boundaries, occurrences = _comparison_boundaries(requirement)
    comparison_count = len(boundaries)
    output_count = len(requirement.out_params)
    if comparison_count <= 0:
        raise ValueError("neural requirement contains no input comparison boundary")
    if output_count != int(action_dim):
        raise ValueError(
            "real-valued output count does not match the action dimension: "
            f"outputs={output_count}, action_dim={action_dim}"
        )
    hidden_dim = max(comparison_count, output_count)
    parameter_count = continuous_feedforward_parameter_count(
        input_dim, action_dim, hidden_dim
    )
    return {
        "method": CONTINUOUS_METHOD,
        "policy_class": CONTINUOUS_POLICY_CLASS,
        "hidden_layers": [hidden_dim, hidden_dim],
        "hidden_dim": hidden_dim,
        "input_dim": int(input_dim),
        "action_dim": int(action_dim),
        "distinct_comparison_boundaries": comparison_count,
        "comparison_occurrences": occurrences,
        "real_output_count": output_count,
        "comparison_boundaries": boundaries,
        "parameter_count": parameter_count,
        "parameter_bytes_float32": 4 * parameter_count,
    }


def check_continuous_feedforward_architecture(
    architecture: dict[str, Any],
    *,
    input_dim: int,
    action_dim: int,
) -> list[str]:
    errors: list[str] = []
    if architecture.get("method") != CONTINUOUS_METHOD:
        errors.append(f"unsupported continuous method={architecture.get('method')}")
    if architecture.get("policy_class") != CONTINUOUS_POLICY_CLASS:
        errors.append(
            "unsupported continuous policy_class="
            f"{architecture.get('policy_class')}"
        )
    hidden_dim = architecture.get("hidden_dim")
    comparisons = architecture.get("distinct_comparison_boundaries")
    outputs = architecture.get("real_output_count")
    if not isinstance(comparisons, int) or comparisons <= 0:
        errors.append(f"invalid distinct comparison count={comparisons}")
    if outputs != int(action_dim):
        errors.append(f"real output count does not match action_dim: {outputs}")
    if isinstance(comparisons, int) and isinstance(outputs, int):
        if hidden_dim != max(comparisons, outputs):
            errors.append("continuous hidden_dim does not match structural width")
    if architecture.get("hidden_layers") != [hidden_dim, hidden_dim]:
        errors.append("continuous hidden layers do not match the equal-width policy")
    if architecture.get("input_dim") != int(input_dim):
        errors.append("continuous input_dim does not match policy input")
    if architecture.get("action_dim") != int(action_dim):
        errors.append("continuous action_dim does not match action space")
    if isinstance(hidden_dim, int) and hidden_dim > 0:
        expected = continuous_feedforward_parameter_count(
            input_dim, action_dim, hidden_dim
        )
        if architecture.get("parameter_count") != expected:
            errors.append("continuous parameter_count is incorrect")
        if architecture.get("parameter_bytes_float32") != 4 * expected:
            errors.append("continuous float32 byte count is incorrect")
    return errors


def check_feedforward_architecture(
    architecture: dict[str, Any],
    *,
    input_dim: int,
    action_count: int,
) -> list[str]:
    """Check the arithmetic and shape fields of a derived architecture."""
    errors: list[str] = []
    if architecture.get("method") != METHOD:
        errors.append(f"unsupported feedforward method={architecture.get('method')}")
    if architecture.get("policy_class") != POLICY_CLASS:
        errors.append(
            f"unsupported feedforward policy_class={architecture.get('policy_class')}"
        )
    hidden_dim = architecture.get("hidden_dim")
    comparisons = architecture.get("distinct_comparison_boundaries")
    outputs = architecture.get("boolean_output_count")
    if not isinstance(comparisons, int) or comparisons <= 0:
        errors.append(f"invalid distinct comparison count={comparisons}")
    if not isinstance(outputs, int) or outputs <= 0:
        errors.append(f"invalid Boolean output count={outputs}")
    if isinstance(comparisons, int) and isinstance(outputs, int):
        expected_hidden = max(comparisons, outputs)
        if hidden_dim != expected_hidden:
            errors.append(
                f"hidden_dim does not match structural width: "
                f"expected {expected_hidden}, observed {hidden_dim}"
            )
    if architecture.get("hidden_layers") != [hidden_dim, hidden_dim]:
        errors.append("hidden_layers do not match the equal-width policy")
    if architecture.get("input_dim") != int(input_dim):
        errors.append("feedforward input_dim does not match policy input")
    if architecture.get("action_count") != int(action_count):
        errors.append("feedforward action_count does not match action space")
    if isinstance(hidden_dim, int) and hidden_dim > 0:
        expected_params = feedforward_parameter_count(
            input_dim, action_count, hidden_dim)
        if architecture.get("parameter_count") != expected_params:
            errors.append("feedforward parameter_count is incorrect")
        if architecture.get("parameter_bytes_float32") != 4 * expected_params:
            errors.append("feedforward float32 byte count is incorrect")
    return errors
