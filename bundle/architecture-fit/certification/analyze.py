#!/usr/bin/env python3
"""CLI smoke path for the new certification extraction components."""

from __future__ import annotations

import argparse
import os

from .relevance import compute_transition_closed_relevance
from .strict_extract import extract_equation_model


_SM = os.path.join(os.path.dirname(__file__), "..", "..", "sysml-models")
MODELS = {
    "thermostat": os.path.join(_SM, "thermostat", "model.sysml"),
    "cruise-continuous": os.path.join(_SM, "cruise-continuous-model", "model.sysml"),
    "cruise-discrete": os.path.join(_SM, "cruise-controller-model", "model.sysml"),
    "mixing": os.path.join(_SM, "mixing-sysml-model", "model.sysml"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+", help="model key or path")
    ap.add_argument("--show-equations", action="store_true")
    args = ap.parse_args()

    for item in args.model:
        path = MODELS.get(item, item)
        model = extract_equation_model(path)
        relevance = compute_transition_closed_relevance(model)

        print("=" * 78)
        print(f"MODEL: {path}")
        print(f"state={len(model.state)} actions={len(model.actions)} "
              f"def_eq={len(model.definitions)} "
              f"obs_eq={len(model.observations)} transition_eq={len(model.transitions)} "
              f"req_eq={len(model.requirements)} diagnostics={len(model.diagnostics)}")
        print(relevance.pretty())

        if model.diagnostics:
            print("diagnostics:")
            for diag in model.diagnostics:
                print(f"  - {diag.pretty()}")

        if args.show_equations:
            print("same-cycle definitions:")
            for eq in model.definitions.values():
                print(f"  {eq.pretty()}")
            print("observation equations:")
            for eq in model.observations.values():
                print(f"  {eq.pretty()}")
            print("transition equations:")
            for eq in model.transitions.values():
                print(f"  {eq.pretty()}")
            print("requirement equations:")
            for eq in model.requirements.values():
                print(f"  {eq.pretty()}")


if __name__ == "__main__":
    main()
