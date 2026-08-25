#!/usr/bin/env python3
"""Generate and check discretization-safety certificates after the MDP check."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
ARTIFACT = THIS.parents[1]
ARCH = (ARTIFACT / "bundle" / "architecture-fit").resolve()
REPO = ARCH.parent
for path in (ARCH, REPO / "sysml-models", REPO / "rl"):
    sys.path.insert(0, str(path))

from discretization.certificate import (  # noqa: E402
    build_certificate,
    check_certificate,
    write_certificate,
)
from runtime_settings import DEFAULT_DT, validate_dt  # noqa: E402
from sysml_inputs import discover_sysml  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="SysML file paths")
    parser.add_argument(
        "--models-root",
        default=str(REPO / "sysml-models"),
        help="directory searched recursively when no paths are supplied",
    )
    parser.add_argument("--mdp-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--dt", default=str(DEFAULT_DT))
    parser.add_argument("--optimization-timeout-ms", type=int, default=250)
    parser.add_argument("--smt-timeout-ms", type=int, default=30000)
    args = parser.parse_args()
    dt_text = str(args.dt)
    dt = validate_dt(args.dt)
    if args.smt_timeout_ms <= 0:
        parser.error("--smt-timeout-ms must be positive")
    if args.optimization_timeout_ms <= 0:
        parser.error("--optimization-timeout-ms must be positive")

    models = discover_sysml(args.models, models_root=args.models_root)
    mdp_dir = Path(args.mdp_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    certificate_dir = artifact_dir / "certificates"
    if certificate_dir.exists():
        shutil.rmtree(certificate_dir)

    rows: list[dict[str, Any]] = []
    failures = 0
    violations = 0
    for model in models:
        mdp_certificate_path = (
            mdp_dir / "certificates" / f"{model.key}.certificate.json"
        )
        reduced_spec_path = (
            mdp_dir
            / "reduced_mdp_specs"
            / f"{model.key}.reduced_mdp_spec.json"
        )
        certificate_path = certificate_dir / f"{model.key}.certificate.json"
        started = time.perf_counter()
        certificate = build_certificate(
            model.path,
            mdp_certificate_path,
            reduced_spec_path,
            dt_text=dt_text,
            optimization_timeout_ms=args.optimization_timeout_ms,
            smt_timeout_ms=args.smt_timeout_ms,
        )
        write_certificate(certificate, certificate_path)
        errors = check_certificate(certificate)
        elapsed = time.perf_counter() - started
        properties = certificate.get("analysis", {}).get("properties", [])
        row = {
            "model": model.key,
            "model_name": model.name,
            "dt": dt,
            "result": certificate.get("result", "NOT_CERTIFIED"),
            "checker": "passed" if not errors else "failed",
            "properties_checked": len(properties),
            "properties_certified": sum(
                item.get("result") == "CERTIFIED" for item in properties
            ),
            "elapsed_seconds": elapsed,
            "certificate_path": str(certificate_path.relative_to(artifact_dir)),
            "errors": errors,
        }
        rows.append(row)
        if errors or certificate.get("result") != "CERTIFIED":
            failures += 1
        if not errors and certificate.get("result") == "VIOLATION":
            violations += 1
        print(
            f"{model.key}: result={row['result']} checker={row['checker']} "
            f"properties={row['properties_certified']}/{row['properties_checked']} "
            f"seconds={elapsed:.6f} certificate={row['certificate_path']}"
        )
        for error in errors:
            print(f"  - {error}")

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": {
            "dt": dt,
            "dt_input": dt_text,
            "optimization_timeout_ms": args.optimization_timeout_ms,
            "smt_timeout_ms": args.smt_timeout_ms,
        },
        "models": [model.to_dict() for model in models],
        "rows": rows,
        "result": "CERTIFIED" if failures == 0 else "NOT_CERTIFIED",
    }
    write_json(Path(args.out_json).resolve(), summary)
    if violations:
        return 2
    return 0 if failures == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
