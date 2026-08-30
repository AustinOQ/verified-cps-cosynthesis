#!/usr/bin/env python3
"""Generate fresh Markov/MDP proof artifacts from bundled SysML models.

Every run rebuilds the certificate, reduced-MDP architecture spec, SMT-LIB
query, and proof/counterexample transcript from the bundled SysML files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


THIS = Path(__file__).resolve()
ARTIFACT = THIS.parents[3]
ARCH = (ARTIFACT / "bundle" / "architecture-fit").resolve()
REPO = ARCH.parent

for path in (ARTIFACT / "src", ARCH, REPO / "rl"):
    sys.path.insert(0, str(path))

from clarity.certification.certificate import (  # noqa: E402
    build_certificate_for_path,
    check_certificate,
    write_certificate,
)
from clarity.certification.reduced_mdp_spec import (  # noqa: E402
    build_reduced_mdp_spec,
    write_reduced_mdp_spec,
)
from clarity.sysml.inputs import discover_sysml  # noqa: E402
from clarity.sysml.runtime_settings import DEFAULT_DT, validate_dt  # noqa: E402


def summarize_certificate(
    name: str,
    cert: dict[str, Any],
    errors: list[str],
    *,
    certificate_path: Path,
    reduced_spec_path: Path | None,
    smt2_path: Path | None,
    proof_path: Path | None,
    status_path: Path | None,
    out_root: Path,
) -> dict[str, Any]:
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
        "certificate_saved": "yes",
        "certificate_path": str(certificate_path.relative_to(out_root)),
        "reduced_mdp_spec_path": (
            "" if reduced_spec_path is None
            else str(reduced_spec_path.relative_to(out_root))
        ),
        "smt2_path": "" if smt2_path is None else str(smt2_path.relative_to(out_root)),
        "proof_or_disproof_path": (
            "" if proof_path is None else str(proof_path.relative_to(out_root))
        ),
        "solver_status_path": (
            "" if status_path is None else str(status_path.relative_to(out_root))
        ),
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_solver_artifacts(
    name: str,
    cert: dict[str, Any],
    *,
    z3_dir: Path,
) -> tuple[Path | None, Path | None, Path]:
    one_step = (
        cert.get("solver_advisory", {})
        .get("one_step_transition_closure", {})
    )
    artifacts = one_step.get("artifacts", {})
    smt2_path = None
    proof_path = None
    if artifacts.get("smt2"):
        smt2_path = z3_dir / f"{name}.query.smt2"
        write_text(smt2_path, artifacts["smt2"])

    status_path = z3_dir / f"{name}.z3_status.txt"
    status_lines = [
        f"model: {name}",
        "query: one-step self-composition transition closure",
        f"z3_check_sat: {artifacts.get('z3_check_sat', '')}",
        f"certificate_status: {cert.get('result', '')}",
        f"solver_status: {one_step.get('status', '')}",
        f"solver: {one_step.get('solver', '')}",
        f"logic: {one_step.get('logic', '')}",
        "",
        "Interpretation:",
        "- unsat means Z3 found no counterexample to the closure obligation.",
        "- sat means Z3 found a counterexample. See the saved model/witness.",
        "- unknown means Z3 did not decide the query.",
        "",
        "Replay:",
        "- Run `z3 <this model>.query.smt2` from this directory if the z3 CLI is installed.",
        "- The expected first line is the `z3_check_sat` value above.",
    ]
    write_text(status_path, "\n".join(status_lines).rstrip() + "\n")

    proof_path = z3_dir / f"{name}.proof_or_disproof.txt"
    if one_step.get("status") == "discharged":
        if artifacts.get("z3_proof_available"):
            body = artifacts.get("z3_proof", "")
            header = [
                f"model: {name}",
                "z3 result: unsat",
                "meaning: no counterexample exists for the encoded one-step closure obligation",
                "proof_available: yes",
                "",
                "Z3 proof text:",
                "",
            ]
            write_text(proof_path, "\n".join(header) + body + "\n")
        else:
            header = [
                f"model: {name}",
                "z3 result: unsat",
                "meaning: no counterexample exists for the encoded one-step closure obligation",
                "proof_available: no",
                f"proof_error: {artifacts.get('z3_proof_error', '')}",
                "",
                "The SMT-LIB query is saved separately. Running it with Z3 should return `unsat`.",
            ]
            write_text(proof_path, "\n".join(header).rstrip() + "\n")
    elif one_step.get("status") == "counterexample":
        body = artifacts.get("z3_model", "")
        header = [
            f"model: {name}",
            "z3 result: sat",
            "meaning: Z3 found a counterexample to the encoded one-step closure obligation",
            "",
            "Certificate witness:",
            json.dumps({
                "disagreement": one_step.get("disagreement"),
                "same_q": one_step.get("same_q"),
                "same_action": one_step.get("same_action"),
                "hidden_state": one_step.get("hidden_state"),
            }, indent=2, sort_keys=True),
            "",
            "Z3 model:",
            "",
        ]
        write_text(proof_path, "\n".join(header) + body + "\n")
    else:
        header = [
            f"model: {name}",
            f"z3 result: {artifacts.get('z3_check_sat', '')}",
            f"solver_status: {one_step.get('status', '')}",
            f"reason: {one_step.get('reason', artifacts.get('reason_unknown', ''))}",
            "",
            "The SMT-LIB query is saved separately when available.",
        ]
        write_text(proof_path, "\n".join(header).rstrip() + "\n")

    write_json(z3_dir / f"{name}.solver_result.json", one_step)
    return smt2_path, proof_path, status_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="SysML file paths")
    parser.add_argument(
        "--models-root",
        default=str(ARTIFACT / "models"),
        help="directory searched recursively when no file paths are supplied",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--max-obs", type=int, default=2)
    parser.add_argument("--max-act", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--dt", type=validate_dt, default=DEFAULT_DT)
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()
    models = discover_sysml(args.models, models_root=args.models_root)

    rows: list[dict[str, Any]] = []
    failures = 0
    out_json = Path(args.out_json).resolve()
    artifact_dir = (
        Path(args.artifact_dir).resolve()
        if args.artifact_dir is not None
        else out_json.parent
    )
    certificate_dir = artifact_dir / "certificates"
    spec_dir = artifact_dir / "reduced_mdp_specs"
    z3_dir = artifact_dir / "z3"
    for generated_dir in (certificate_dir, spec_dir, z3_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    for model in models:
        name = model.key
        model_path = model.path
        cert = build_certificate_for_path(
            str(model_path),
            max_obs=args.max_obs,
            max_act=args.max_act,
            horizon=args.horizon,
            dt=args.dt,
            include_solver_artifacts=True,
        )
        cert_path = certificate_dir / f"{name}.certificate.json"
        write_certificate(cert, cert_path)
        errors = check_certificate(cert, check_hash=True)
        smt2_path, proof_path, status_path = write_solver_artifacts(
            name, cert, z3_dir=z3_dir
        )

        reduced_spec_path: Path | None = None
        if not errors and cert.get("result") == "PASS":
            reduced_spec = build_reduced_mdp_spec(
                str(model_path),
                certificate=cert,
                certificate_path=cert_path,
                dt=args.dt,
                max_steps=args.max_steps,
            )
            reduced_spec_path = spec_dir / f"{name}.reduced_mdp_spec.json"
            write_reduced_mdp_spec(reduced_spec, reduced_spec_path)

        row = summarize_certificate(
            name,
            cert,
            errors,
            certificate_path=cert_path,
            reduced_spec_path=reduced_spec_path,
            smt2_path=smt2_path,
            proof_path=proof_path,
            status_path=status_path,
            out_root=artifact_dir,
        )
        row["model_name"] = model.name
        row["model_path"] = str(model.path)
        row["model_sha256"] = model.sha256
        row["action_kind"] = model.action_kind
        row["dt"] = args.dt
        rows.append(row)
        if errors:
            failures += 1
        print(
            f"{name}: result={row['result']} checker={row['checker']} "
            f"solver_status={row['solver_status']} "
            f"buffer=b_obs={row['b_obs']},b_act={row['b_act']} "
            f"certificate={row['certificate_path']} "
            f"proof={row['proof_or_disproof_path']}"
        )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture_fit_root": str(ARCH),
        "models": [model.to_dict() for model in models],
        "settings": {
            "max_obs": args.max_obs,
            "max_act": args.max_act,
            "horizon": args.horizon,
            "dt": args.dt,
        },
        "certificate_policy": (
            "Every run rebuilds and overwrites certificates, reduced-MDP specs, "
            "SMT-LIB queries, and proof/counterexample transcripts from the "
            "SysML files supplied to that run. No previous-run artifacts are reused."
        ),
        "artifact_dirs": {
            "certificates": str(certificate_dir),
            "reduced_mdp_specs": str(spec_dir),
            "z3": str(z3_dir),
        },
        "rows": rows,
    }
    write_json(out_json, summary)
    print(f"WROTE {out_json}")
    if failures:
        print(f"NOTE: {failures} model(s) were not certified. This is reported data, not a runner crash.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
