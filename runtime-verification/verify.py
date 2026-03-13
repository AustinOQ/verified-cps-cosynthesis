#!/usr/bin/env python3
"""Runtime verification server for SysML neural requirements.

Accepts concrete neural policy inputs/outputs via HTTP POST and checks
whether the #NeuralRequirement constraint from the SysML model holds.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

# Add sysml-models directory to path so we can import the parser.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / "sysml-models"))

from sysml_parser import (  # noqa: E402
    BinaryExpr,
    ExpressionParser,
    Expr,
    LiteralExpr,
    RefExpr,
    SysMLParser,
    TernaryExpr,
    UnaryExpr,
)


# ---------------------------------------------------------------------------
# AST evaluator
# ---------------------------------------------------------------------------

def evaluate(expr: Expr, values: dict[str, Any], subject_var: str) -> Any:
    """Walk a parsed expression AST and evaluate it against concrete values."""
    if isinstance(expr, LiteralExpr):
        return expr.value

    if isinstance(expr, RefExpr):
        path = expr.path
        # Strip subject prefix (e.g. p.targetSpeed -> targetSpeed)
        if path and path[0] == subject_var:
            path = path[1:]
        key = ".".join(path)
        if key in values:
            return values[key]
        raise KeyError(f"Unknown reference: {'.'.join(expr.path)}")

    if isinstance(expr, BinaryExpr):
        if expr.op == "implies":
            left = evaluate(expr.left, values, subject_var)
            return True if not left else evaluate(expr.right, values, subject_var)
        left = evaluate(expr.left, values, subject_var)
        right = evaluate(expr.right, values, subject_var)
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else 0,
            "==": lambda a, b: a == b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
        }
        return ops[expr.op](left, right)

    if isinstance(expr, UnaryExpr):
        val = evaluate(expr.operand, values, subject_var)
        if expr.op == "not":
            return not val
        if expr.op == "-":
            return -val

    if isinstance(expr, TernaryExpr):
        cond = evaluate(expr.condition, values, subject_var)
        return evaluate(expr.true_expr, values, subject_var) if cond else evaluate(expr.false_expr, values, subject_var)

    raise ValueError(f"Unsupported expression type: {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Pre-parsed information about a model's neural requirement."""
    in_params: list[str]
    out_params: list[str]
    subject_var: str
    requirement_ast: Expr
    unchanging: dict[str, Any] = field(default_factory=dict)


def load_model(sysml_path: str) -> ModelInfo | None:
    """Parse a SysML file and extract the neural requirement info."""
    parser = SysMLParser(sysml_path)
    parser.parse()

    ctrl_fqn = parser.controller_part
    if not ctrl_fqn:
        return None
    ctrl_inst = parser.part_instances.get(ctrl_fqn)
    if not ctrl_inst:
        return None
    ctrl_def = parser.part_defs.get(ctrl_inst.part_type)
    if not ctrl_def:
        return None

    # Find #Neural action def.
    neural_def = None
    for ad in ctrl_def.action_defs:
        if "Neural" in ad.metadata:
            neural_def = ad
            break
    if not neural_def:
        return None

    in_names = [p.name for p in neural_def.in_params]
    out_names = [p.name for p in neural_def.out_params]
    neural_param_names = set(in_names + out_names)

    # Find #NeuralRequirement.
    req_ast = None
    subject_var = ""
    for req_name, sv, _st, req_expr, req_meta in ctrl_def.requirements:
        if "NeuralRequirement" not in req_meta:
            continue
        req_ast = ExpressionParser(req_expr).parse()
        subject_var = sv
        break
    if req_ast is None:
        return None

    # Collect unchanging controller quantities (not neural in/out params).
    ctrl_prefix = ctrl_fqn + "::"
    unchanging: dict[str, Any] = {}
    for p in parser.parameters:
        if not p.qualified_name.startswith(ctrl_prefix):
            continue
        bare = p.name
        if bare in neural_param_names:
            continue
        unchanging[bare] = p.value

    return ModelInfo(
        in_params=in_names,
        out_params=out_names,
        subject_var=subject_var,
        requirement_ast=req_ast,
        unchanging=unchanging,
    )


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def create_app(model_dir: str) -> Flask:
    app = Flask(__name__)

    # Load all models from subdirectories.
    models: dict[str, ModelInfo] = {}
    model_root = Path(model_dir)
    for sub in sorted(model_root.iterdir()):
        sysml_file = sub / "model.sysml"
        if not sysml_file.is_file():
            continue
        info = load_model(str(sysml_file))
        if info:
            models[sub.name] = info
            print(f"Loaded model: {sub.name}  "
                  f"(in={info.in_params}, out={info.out_params}, "
                  f"unchanging={info.unchanging})")

    @app.post("/models/<model_name>")
    def check_requirement(model_name: str):
        info = models.get(model_name)
        if info is None:
            return jsonify({"error": f"Unknown model: {model_name}"}), 404

        body = request.get_json(silent=True)
        if not body or "in" not in body or "out" not in body:
            return jsonify({"error": "Body must have 'in' and 'out' keys"}), 400

        in_vals = body["in"]
        out_vals = body["out"]

        missing_in = [p for p in info.in_params if p not in in_vals]
        missing_out = [p for p in info.out_params if p not in out_vals]
        if missing_in or missing_out:
            return jsonify({
                "error": "Missing parameters",
                "missing_in": missing_in,
                "missing_out": missing_out,
            }), 400

        # Merge all values: in params, out params, unchanging quantities.
        values = {**info.unchanging, **in_vals, **out_vals}

        try:
            result = evaluate(info.requirement_ast, values, info.subject_var)
        except Exception as e:
            return jsonify({"error": f"Evaluation failed: {e}"}), 500

        return jsonify({"satisfied": bool(result)})

    return app


def main():
    ap = argparse.ArgumentParser(description="Runtime neural-requirement verification server")
    ap.add_argument("-p", "--port", type=int, default=8080)
    ap.add_argument("-m", "--model-dir", required=True,
                    help="Directory containing model subdirectories")
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        sys.exit(f"Model directory not found: {args.model_dir}")

    app = create_app(args.model_dir)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
