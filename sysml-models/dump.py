#!/usr/bin/env python3
"""Dump parsed SysML model to JSON."""

import argparse
import dataclasses
import json
import sys

from sysml_parser import (
    Expr, LiteralExpr, RefExpr, BinaryExpr, TernaryExpr, UnaryExpr,
    ActionStmt, ActionDef, InputBindingStmt, OutputBindingStmt, SubactionCallStmt,
    SysMLParser,
)


def expr_to_dict(expr: Expr) -> dict:
    """Recursively convert an Expr AST node to a JSON-serializable dict."""
    if isinstance(expr, LiteralExpr):
        return {"type": "Literal", "value": expr.value}
    elif isinstance(expr, RefExpr):
        return {"type": "Ref", "path": expr.path}
    elif isinstance(expr, BinaryExpr):
        return {"type": "Binary", "op": expr.op,
                "left": expr_to_dict(expr.left), "right": expr_to_dict(expr.right)}
    elif isinstance(expr, TernaryExpr):
        return {"type": "Ternary",
                "condition": expr_to_dict(expr.condition),
                "true": expr_to_dict(expr.true_expr),
                "false": expr_to_dict(expr.false_expr)}
    elif isinstance(expr, UnaryExpr):
        return {"type": "Unary", "op": expr.op, "operand": expr_to_dict(expr.operand)}
    return {"type": "Unknown"}


def to_serializable(obj):
    """Recursively convert parsed model objects to JSON-serializable structures."""
    if isinstance(obj, Expr):
        return expr_to_dict(obj)
    elif isinstance(obj, ActionStmt):
        d = {"__type__": type(obj).__name__}
        for f in dataclasses.fields(obj):
            d[f.name] = to_serializable(getattr(obj, f.name))
        return d
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_serializable(getattr(obj, f.name))
                for f in dataclasses.fields(obj)}
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def dump_model(parser: SysMLParser) -> dict:
    return {
        "system_part": parser.system_part,
        "system_type": parser.system_type,
        "controller_part": parser.controller_part,
        "parameters": to_serializable(parser.parameters),
        "part_defs": to_serializable(parser.part_defs),
        "part_instances": to_serializable(parser.part_instances),
        "state_machines": to_serializable(parser.state_machines),
        "flows": to_serializable(parser.flows),
        "step_actions": to_serializable(parser.step_actions),
        "parsed_constraints": to_serializable(parser.parsed_constraints),
        "derived_attributes": to_serializable(parser.derived_attributes),
        "ref_bindings": parser.ref_bindings,
        # Emit SM name only to avoid duplicating full SM definitions
        "instance_state_machines": {
            fqn: sm.name for fqn, sm in parser.instance_state_machines.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump parsed SysML model as JSON")
    ap.add_argument("model_file", nargs="?", default="thermostat.sysml",
                    help="SysML model file to parse (default: thermostat.sysml)")
    ap.add_argument("-i", "--indent", type=int, default=2,
                    help="JSON indentation (default: 2; use 0 for compact)")
    args = ap.parse_args()

    try:
        parser = SysMLParser(args.model_file)
        parser.parse()
    except Exception as e:
        print(f"Error parsing {args.model_file}: {e}", file=sys.stderr)
        return 1

    indent = args.indent or None
    print(json.dumps(dump_model(parser), indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
