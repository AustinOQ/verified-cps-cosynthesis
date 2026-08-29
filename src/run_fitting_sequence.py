#!/usr/bin/env python3
"""Compatibility entry point for the CLARITY pipeline."""

from clarity.pipeline.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
