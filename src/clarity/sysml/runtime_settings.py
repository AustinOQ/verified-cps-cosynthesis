"""Shared runtime settings for SysML simulation and fitting."""

from __future__ import annotations

import math


DEFAULT_DT = 0.1


def validate_dt(value: float) -> float:
    """Return a finite positive simulation time step."""
    if isinstance(value, bool):
        raise ValueError("dt must be a finite positive number")
    try:
        dt = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a finite positive number") from exc
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive number")
    return dt
