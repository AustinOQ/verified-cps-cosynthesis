#!/usr/bin/env python3
"""Generate concise Markov/MDP proof results from bundled SysML models.

This script intentionally does not write certificate artifacts. It builds the
certificate object in memory, invokes the bundled Z3-backed proof obligation,
checks the in-memory result, and writes only a compact summary for the demo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
ARTIFACT = THIS.parents[1]
ARCH = Path(
    os.environ.get(
        "ARCHITECTURE_FIT_ROOT",
        str(ARTIFACT / "bundle" / "architecture-fit"),
    )
).resolve()
REPO = ARCH.parent

for path in (ARCH, REPO / "sysml-models", REPO / "rl"):
    sys.path.insert(0, str(path))

from certification.certificate import build_certificate_for_path, check_certificate  # noqa: E402


MODEL_PATHS = {
    "thermostat": REPO / "sysml-models" / "thermostat" / "model.sysml",
    "cruise-discrete": REPO / "sysml-models" / "cruise-controller-model" / "model.sysml",
    "cruise-continuous": REPO / "sysml-models" / "cruise-continuous-model" / "model.sysml",
    "mixing": REPO / "sysml-models" / "mixing-sysml-model" / "model.sysml",
}


def summarize_certificate(name: str, cert: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    buffer = cert.get("buffer") or {}
    claim = cert.get("claim") or {}
    gate = (cert.get("theorem_gate") or {}).get("solver_backed_uniqueness") or {}
    advisory = (cert.get("solver_advisory") or {}).get("one_step_transition_closure") or {}
    checker = "passed" if not errors else "failed"
    solver_status = gate.get("status") or advisory.get("status") or ""

    row: dict[str, Any] = {
        "model": name,
        "b_obs": buffer.get("b_obs", ""),
        "b_act": buffer.get("b_act", ""),
        "claim": (
            "provable Markov/MDP"
            if checker == "passed" and cert.get("result") == "PASS"
            else "not certified"
        ),
        "checker": checker,
        "solver_status": solver_status,
        "solver": gate.get("solver") or advisory.get("solver") or "",
        "logic": gate.get("logic") or advisory.get("logic") or "",
        "max_polynomial_degree": (
            gate.get("max_polynomial_degree")
            if gate.get("max_polynomial_degree") is not None
            else advisory.get("max_polynomial_degree", "")
        ),
        "source": "generated from bundled SysML at run time",
        "certificate_saved": "no",
        "result": cert.get("result", ""),
        "mdp_theorem": claim.get("mdp_theorem", ""),
        "errors": errors,
    }
    if advisory.get("status") == "counterexample":
        row["counterexample_term"] = (advisory.get("disagreement") or {}).get("term", "")
    elif advisory.get("status") in {"unknown", "unavailable"}:
        row["counterexample_term"] = advisory.get("reason", "")
    else:
        row["counterexample_term"] = ""
    return row


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=list(MODEL_PATHS),
        choices=sorted(MODEL_PATHS),
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-obs", type=int, default=2)
    parser.add_argument("--max-act", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    failures = 0
    for name in args.models:
        model_path = MODEL_PATHS[name]
        cert = build_certificate_for_path(
            str(model_path),
            max_obs=args.max_obs,
            max_act=args.max_act,
            horizon=args.horizon,
            dt=args.dt,
        )
        errors = check_certificate(cert, check_hash=True)
        row = summarize_certificate(name, cert, errors)
        rows.append(row)
        if errors:
            failures += 1
        print(
            f"{name}: result={row['result']} checker={row['checker']} "
            f"solver_status={row['solver_status']} "
            f"buffer=b_obs={row['b_obs']},b_act={row['b_act']} "
            f"certificate_saved={row['certificate_saved']}"
        )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture_fit_root": str(ARCH),
        "models": list(args.models),
        "settings": {
            "max_obs": args.max_obs,
            "max_act": args.max_act,
            "horizon": args.horizon,
            "dt": args.dt,
        },
        "certificate_policy": (
            "Certificates are generated in memory for checking. Full proof "
            "certificate artifacts are not saved by this artifact."
        ),
        "rows": rows,
    }
    out_json = Path(args.out_json).resolve()
    write_json(out_json, summary)
    print(f"WROTE {out_json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
