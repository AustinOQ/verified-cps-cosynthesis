# About The Checker Sequence

This document gives a concise example of the pipeline stages. See `README.md`
for setup and usage.

## Cruise-Control Example

- Boolean action extraction. Throttle when target speed exceeds current speed
  by more than the tolerance and the gap is safe. Brake when current speed
  exceeds target speed by more than the tolerance or the following gap is
  unsafe. Otherwise coast.
- Memoryless controller check. The current controller decision is determined
  by the current controller inputs without past observations or previous
  actions.
- Provable Markov/MDP check. The modeled next step is certified with
  the current observation, `1` past observation, and `2` previous actions.
- Discretization safety certification. The checked controller contract is
  proved to satisfy each extracted property between controller updates.
- Fitted feedforward training. A small MLP consumes the certified buffer. The
  shield extracted from the current SysML requirement is applied during
  training and evaluation.
