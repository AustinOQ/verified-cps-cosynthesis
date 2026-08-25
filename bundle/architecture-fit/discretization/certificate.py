#!/usr/bin/env python3
"""Build and independently replay discretization-safety certificates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from certification.certificate import (
    check_certificate as check_mdp_certificate,
    load_certificate as load_mdp_certificate,
    model_hash,
)
from certification.reduced_mdp_spec import (
    check_reduced_mdp_spec,
    load_reduced_mdp_spec,
)

from .analysis import analyze_model, canonical_dt
from .proof_certificate_verifier import verify_recorded_optimization_certificates


SCHEMA_VERSION = 3
CERTIFICATE_KIND = "discretization_safety_certificate_v3"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def certificate_hash(certificate: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(certificate))
    clone.pop("self_sha256", None)
    return sha256_bytes(canonical_json_bytes(clone))


TIMED_REACHABILITY_CHECKERS = {
    "reachability_linear",
    "reachability_convex",
    "relational_invariant",
    "smt_reachability",
}


def _contains_timed_reachability(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("checker") in TIMED_REACHABILITY_CHECKERS:
            return True
        return any(_contains_timed_reachability(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_timed_reachability(item) for item in value)
    return False


def _replay_projection(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("checker") in TIMED_REACHABILITY_CHECKERS:
            return {
                "checker": value.get("checker"),
                "checker_version": value.get("checker_version"),
            }
        if (
            isinstance(value.get("progression"), list)
            and _contains_timed_reachability(value["progression"])
        ):
            return {
                key: _replay_projection(item)
                for key, item in value.items()
                if key not in {"progression", "result"}
            }
        has_timed_reachability = _contains_timed_reachability(value)
        return {
            key: _replay_projection(item)
            for key, item in value.items()
            if key not in {
                "exact_values",
                "multipliers",
                "solver",
                "solver_status",
                "selected_indices",
                "selected_constraints",
                "selected_expression",
                "selected_expression_sha256",
                "subset_minimization_checks",
                "subset_minimization_complete",
                "recertification_attempts",
                "certifying_checker",
                "certificate_attempt",
                "query_smt2",
                "query_smt2_sha256",
                "z3_proof",
                "z3_proof_sha256",
            }
            and not (key == "result" and has_timed_reachability)
        }
    if isinstance(value, list):
        return [_replay_projection(item) for item in value]
    return value


def _validate_inputs(
    model_path: Path,
    mdp_path: Path,
    reduced_spec_path: Path,
    dt_text: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    mdp_certificate = load_mdp_certificate(mdp_path)
    errors.extend(
        f"Markov process certificate: {error}"
        for error in check_mdp_certificate(mdp_certificate, check_hash=True)
    )
    if (
        mdp_certificate.get("theorem_gate", {}).get(
            "full_mdp_theorem_claim_allowed"
        )
        is not True
    ):
        errors.append("Markov process certificate does not allow its full claim")
    if (
        mdp_certificate.get("claim", {}).get("solver_backed_mdp_theorem")
        != "discharged"
    ):
        errors.append("Markov process certificate claim is not discharged")

    reduced_spec = load_reduced_mdp_spec(reduced_spec_path)
    errors.extend(
        f"reduced specification: {error}"
        for error in check_reduced_mdp_spec(reduced_spec)
    )

    model_abs = str(model_path.resolve())
    observed_model_hash = model_hash(model_abs)
    for label, source in (
        ("Markov process certificate", mdp_certificate.get("model", {})),
        ("reduced specification", reduced_spec.get("model", {})),
    ):
        if os.path.abspath(str(source.get("path", ""))) != model_abs:
            errors.append(f"{label} model path does not match")
        if source.get("sha256") != observed_model_hash:
            errors.append(f"{label} model hash does not match")

    requested_dt = canonical_dt(dt_text)
    try:
        mdp_dt = canonical_dt(mdp_certificate.get("settings", {}).get("dt"))
        if mdp_dt["canonical"] != requested_dt["canonical"]:
            errors.append("Markov process certificate dt does not match")
    except (TypeError, ValueError, ZeroDivisionError):
        errors.append("Markov process certificate dt is invalid")
    try:
        spec_dt = canonical_dt(reduced_spec.get("model", {}).get("dt"))
        if spec_dt["canonical"] != requested_dt["canonical"]:
            errors.append("reduced specification dt does not match")
    except (TypeError, ValueError, ZeroDivisionError):
        errors.append("reduced specification dt is invalid")
    return mdp_certificate, reduced_spec, errors


def build_certificate(
    model_path: str | Path,
    mdp_certificate_path: str | Path,
    reduced_spec_path: str | Path,
    *,
    dt_text: str,
    optimization_timeout_ms: int = 250,
    smt_timeout_ms: int = 30000,
) -> dict[str, Any]:
    model = Path(model_path).resolve()
    mdp_path = Path(mdp_certificate_path).resolve()
    spec_path = Path(reduced_spec_path).resolve()
    mdp_certificate, reduced_spec, input_errors = _validate_inputs(
        model, mdp_path, spec_path, dt_text
    )
    if input_errors:
        analysis = {
            "schema_version": 3,
            "result": "NOT_CERTIFIED",
            "claim": "full_sysml_discretization_safety_preservation_v3",
            "properties": [],
            "blocking_diagnostics": input_errors,
        }
    else:
        analysis = analyze_model(
            model,
            mdp_certificate,
            dt_text=dt_text,
            optimization_timeout_ms=optimization_timeout_ms,
            smt_timeout_ms=smt_timeout_ms,
        )

    certificate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": CERTIFICATE_KIND,
        "result": analysis.get("result", "NOT_CERTIFIED"),
        "claim": {
            "name": "full_sysml_discretization_safety_preservation_v3",
            "status": (
                "discharged"
                if analysis.get("result") == "CERTIFIED"
                else "not_discharged"
            ),
        },
        "model": {
            "path": str(model),
            "sha256": model_hash(model),
        },
        "settings": {
            "dt": canonical_dt(dt_text),
            "optimization_timeout_ms": int(optimization_timeout_ms),
            "smt_timeout_ms": int(smt_timeout_ms),
        },
        "markov_process_certificate": {
            "path": str(mdp_path),
            "file_sha256": file_sha256(mdp_path),
            "canonical_sha256": sha256_bytes(canonical_json_bytes(mdp_certificate)),
        },
        "reduced_specification": {
            "path": str(spec_path),
            "file_sha256": file_sha256(spec_path),
            "canonical_sha256": sha256_bytes(canonical_json_bytes(reduced_spec)),
            "self_sha256": reduced_spec.get("self_sha256"),
        },
        "analysis": analysis,
    }
    certificate["self_sha256"] = certificate_hash(certificate)
    return certificate


def write_certificate(certificate: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clone = json.loads(json.dumps(certificate))
    clone["self_sha256"] = certificate_hash(clone)
    output.write_text(
        json.dumps(clone, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_certificate(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def check_certificate(
    certificate: dict[str, Any],
    *,
    check_files: bool = True,
) -> list[str]:
    errors: list[str] = []
    if certificate.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version={certificate.get('schema_version')}")
    if certificate.get("kind") != CERTIFICATE_KIND:
        errors.append(f"unsupported kind={certificate.get('kind')}")
    if certificate.get("self_sha256") != certificate_hash(certificate):
        errors.append("self_sha256 does not match certificate contents")
    result = certificate.get("result")
    if result not in {"CERTIFIED", "NOT_CERTIFIED", "VIOLATION"}:
        errors.append(f"certificate result is invalid: {result}")
    expected_status = "discharged" if result == "CERTIFIED" else "not_discharged"
    if certificate.get("claim", {}).get("status") != expected_status:
        errors.append("certificate claim status does not match its result")
    errors.extend(
        verify_recorded_optimization_certificates(certificate.get("analysis", {}))
    )

    model_info = certificate.get("model", {})
    model_path = Path(str(model_info.get("path", "")))
    mdp_info = certificate.get("markov_process_certificate", {})
    mdp_path = Path(str(mdp_info.get("path", "")))
    spec_info = certificate.get("reduced_specification", {})
    spec_path = Path(str(spec_info.get("path", "")))
    if not check_files:
        return errors
    for label, path in (
        ("model", model_path),
        ("Markov process certificate", mdp_path),
        ("reduced specification", spec_path),
    ):
        if not path.is_file():
            errors.append(f"{label} path does not exist: {path}")
    if errors:
        return errors

    if model_hash(model_path) != model_info.get("sha256"):
        errors.append("model sha256 does not match certificate")
    if file_sha256(mdp_path) != mdp_info.get("file_sha256"):
        errors.append("Markov process certificate file sha256 does not match")
    if file_sha256(spec_path) != spec_info.get("file_sha256"):
        errors.append("reduced specification file sha256 does not match")

    mdp_certificate = load_mdp_certificate(mdp_path)
    if sha256_bytes(canonical_json_bytes(mdp_certificate)) != mdp_info.get(
        "canonical_sha256"
    ):
        errors.append("Markov process certificate canonical sha256 does not match")
    errors.extend(
        f"Markov process certificate: {error}"
        for error in check_mdp_certificate(mdp_certificate, check_hash=True)
    )

    reduced_spec = load_reduced_mdp_spec(spec_path)
    if sha256_bytes(canonical_json_bytes(reduced_spec)) != spec_info.get(
        "canonical_sha256"
    ):
        errors.append("reduced specification canonical sha256 does not match")
    if reduced_spec.get("self_sha256") != spec_info.get("self_sha256"):
        errors.append("reduced specification self sha256 does not match")
    errors.extend(
        f"reduced specification: {error}"
        for error in check_reduced_mdp_spec(reduced_spec)
    )

    dt_record = certificate.get("settings", {}).get("dt", {})
    try:
        dt_text = str(dt_record["input"])
        if canonical_dt(dt_text) != dt_record:
            errors.append("certificate dt record is not canonical")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("certificate dt record is invalid")
        return errors
    try:
        optimization_timeout_ms = int(
            certificate.get("settings", {}).get("optimization_timeout_ms")
        )
        if optimization_timeout_ms <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("certificate optimization timeout is invalid")
        return errors
    try:
        timeout_ms = int(certificate.get("settings", {}).get("smt_timeout_ms"))
        if timeout_ms <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("certificate solver timeout is invalid")
        return errors

    rebuilt = analyze_model(
        model_path,
        mdp_certificate,
        dt_text=dt_text,
        optimization_timeout_ms=optimization_timeout_ms,
        smt_timeout_ms=timeout_ms,
    )
    if canonical_json_bytes(_replay_projection(rebuilt)) != canonical_json_bytes(
        _replay_projection(certificate.get("analysis"))
    ):
        errors.append("independent replay does not match recorded analysis")
    return errors
