#!/usr/bin/env python3
"""Independent checker CLI for strict-Q certificate artifacts."""

from __future__ import annotations

import argparse
import sys

from .certificate import check_certificate, load_certificate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("certificate", nargs="+")
    ap.add_argument("--no-hash", action="store_true",
                    help="do not re-check the source model hash")
    args = ap.parse_args()

    status = 0
    for path in args.certificate:
        cert = load_certificate(path)
        errors = check_certificate(cert, check_hash=not args.no_hash)
        if errors:
            status = 1
            print(f"{path}: CHECK FAILED")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"{path}: CHECK PASSED")
    return status


if __name__ == "__main__":
    sys.exit(main())
