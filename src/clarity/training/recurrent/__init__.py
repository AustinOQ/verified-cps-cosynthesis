"""Handmade numpy-only implementation of the RL training stack.

Layers/losses/optimizer here implement forward + analytic backward by hand,
no autograd library. Goal: train the same recurrent actor-critic + PPO
pipeline as rl/ without paying the torch CPU baseline (~270 MB libtorch).

This is a first cut. See README.md for what's implemented vs not, and
grad_check.py for the validation harness that proves the gradients are
right against finite differences.
"""
