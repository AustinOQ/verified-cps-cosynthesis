#!/usr/bin/env python3
"""CLI smoke path for the new certification extraction components."""

from __future__ import annotations

import argparse
from .relevance import compute_transition_closed_relevance
from .strict_extract import extract_equation_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+", help="SysML file path")
    ap.add_argument("--show-equations", action="store_true")
    args = ap.parse_args()

    for item in args.model:
        model = extract_equation_model(item)
        relevance = compute_transition_closed_relevance(model)

        print("=" * 78)
        print(f"MODEL: {item}")
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
