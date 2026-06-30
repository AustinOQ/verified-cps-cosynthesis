"""Handmade numpy training stack for certified reduced-MDP policies.

This package is intentionally isolated from the existing handmade GRU
experiments and from the PyTorch reduced-MDP runner. It reuses the exact
program/AST shield and SysML simulator, but the policy and gradients are
numpy-only.
"""

