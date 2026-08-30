"""Exact replay of recorded values against serialized proof expressions."""

from __future__ import annotations

from fractions import Fraction
from typing import Any


ExactValue = bool | Fraction


def serialize_exact_value(value: ExactValue) -> bool | str:
    if isinstance(value, bool):
        return value
    return f"{value.numerator}/{value.denominator}"


def parse_exact_values(values: Any) -> dict[str, ExactValue]:
    if not isinstance(values, dict):
        raise ValueError("exact values are malformed")
    parsed: dict[str, ExactValue] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        if isinstance(raw_value, bool):
            parsed[name] = raw_value
        elif isinstance(raw_value, str):
            parsed[name] = Fraction(raw_value)
        else:
            raise ValueError(f"exact value for {name} is malformed")
    return parsed


def _number(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean value used as a number")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise ValueError("numeric constant is malformed")


def _require_boolean(value: ExactValue) -> bool:
    if not isinstance(value, bool):
        raise ValueError("numeric value used as a Boolean")
    return value


def _require_number(value: ExactValue) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Boolean value used as a number")
    return value


def evaluate_serialized_expression(
    expression: Any,
    values: dict[str, ExactValue],
) -> ExactValue:
    if not isinstance(expression, dict):
        raise ValueError("expression is malformed")
    kind = expression.get("type")
    if kind == "const":
        value = expression.get("value")
        return value if isinstance(value, bool) else _number(value)
    if kind == "var":
        name = expression.get("name")
        if not isinstance(name, str) or name not in values:
            raise ValueError(f"exact value for {name!r} is missing")
        return values[name]
    if kind == "raw_ref":
        name = expression.get("path")
        if not isinstance(name, str) or name not in values:
            raise ValueError(f"exact value for {name!r} is missing")
        return values[name]
    if kind == "ite":
        condition = _require_boolean(
            evaluate_serialized_expression(expression.get("cond"), values)
        )
        branch = "then" if condition else "else"
        return evaluate_serialized_expression(expression.get(branch), values)
    if kind != "op":
        raise ValueError(f"unsupported expression type {kind!r}")

    operation = expression.get("op")
    raw_arguments = expression.get("args")
    if not isinstance(raw_arguments, list):
        raise ValueError("operation arguments are malformed")
    arguments = [
        evaluate_serialized_expression(argument, values)
        for argument in raw_arguments
    ]
    if operation == "not" and len(arguments) == 1:
        return not _require_boolean(arguments[0])
    if operation == "and":
        return all(_require_boolean(value) for value in arguments)
    if operation == "or":
        return any(_require_boolean(value) for value in arguments)
    if operation == "implies" and len(arguments) == 2:
        return (
            not _require_boolean(arguments[0])
            or _require_boolean(arguments[1])
        )
    if operation == "==" and len(arguments) == 2:
        if isinstance(arguments[0], bool) != isinstance(arguments[1], bool):
            raise ValueError("equality compares incompatible values")
        return arguments[0] == arguments[1]
    if operation in {">", "<", ">=", "<="} and len(arguments) == 2:
        left = _require_number(arguments[0])
        right = _require_number(arguments[1])
        if operation == ">":
            return left > right
        if operation == "<":
            return left < right
        if operation == ">=":
            return left >= right
        return left <= right
    if operation in {"+", "-", "*", "/"}:
        numbers = [_require_number(value) for value in arguments]
        if not numbers:
            raise ValueError("arithmetic operation has no arguments")
        if operation == "+":
            return sum(numbers, Fraction(0))
        if operation == "-":
            if len(numbers) == 1:
                return -numbers[0]
            result = numbers[0]
            for value in numbers[1:]:
                result -= value
            return result
        if operation == "*":
            result = Fraction(1)
            for value in numbers:
                result *= value
            return result
        if len(numbers) != 2 or numbers[1] == 0:
            raise ValueError("division is malformed")
        return numbers[0] / numbers[1]
    raise ValueError(f"unsupported operation {operation!r}")


def replay_serialized_boolean_expression(
    expression: Any,
    raw_values: Any,
) -> bool:
    values = parse_exact_values(raw_values)
    return _require_boolean(evaluate_serialized_expression(expression, values))
