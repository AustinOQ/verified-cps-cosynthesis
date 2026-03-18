# Evaluation Notes

## Systems Under Evaluation

| System | SysML Model | Description |
|--------|-------------|-------------|
| Thermostat | `sysml-models/thermostat/model.sysml` | Bang-bang temperature controller (heater + AC) |
| Cruise Control | `sysml-models/cruise-controller-model/model.sysml` | Speed + following-distance controller (throttle + brake) |
| Chemical Mixing | `sysml-models/mixing-sysml-model/model.sysml` | Dual-tank fluid transfer with volumetric constraints |

Training data for all three models: `metrics/<model>_training_metrics.csv`
See **Fig 1** (`results/graphs_tables/fig1_training_curves.png`) for reward curves.

---

## Verification Table

> **See:** `metrics/evaluation_summary.csv` and **Fig 3** (`results/graphs_tables/fig3_verification_table.png`)

| System | Req. Total | Verified | Failed | Method | Strengthening (Proven/Failed) | Heuristics |
|--------|-----------|----------|--------|--------|-------------------------------|------------|
| Thermostat | 5 | 5 | 0 | IC3 | 0/0 | Real types, dt workaround |
| Cruise Control | 7 | 7 | 0 | IC3 | 0/0 | Real types, dt workaround |
| Chemical Mixing | 2 | 0 | 2 | BMC | 6/2 | Real types, two-pass IC3, BMC fallback, dt workaround |
| **TOTAL** | **14** | **12** | **2** | --- | **6/2** | 3x real types, 1x two-pass IC3, 1x BMC fallback, 3x dt workaround |

**Heuristic descriptions:**
- **Real types**: Use nuXmv real arithmetic instead of integer (avoids precision loss)
- **Two-pass IC3**: First prove strengthening invariants, then add them as constraints for main requirements
- **BMC fallback**: When IC3 fails, fall back to bounded model checking (k=200)
- **dt workaround**: Encode discrete time step as a real-valued constant to avoid integer overflow

**Mixing failures**: The 2 failing requirements (No Overpumping, Fluid Transfer Termination Safety) produce BMC counterexamples at step 16. This corresponds to a mid-scan-cycle state where the pump has been activated but the valve close hasn't propagated yet. The 2 failing strengthening invariants are pump ordering properties that assume immediate valve-close-before-pump-stop sequencing, which the scan cycle doesn't guarantee within a single phase.

---

## Automation Metrics

> **See:** `metrics/automation_metrics.csv` and **Fig 4** (`results/graphs_tables/fig4_automation_metrics.png`)

| System | SysML LOC | SMV LOC (auto-extracted) | Reward/Env LOC (shared) | SMV/SysML Ratio |
|--------|----------|--------------------------|------------------------|-----------------|
| Thermostat | 269 | 106 | 222 | 39% |
| Cruise Control | 309 | 126 | 222 | 41% |
| Chemical Mixing | 583 | 214 | 222 | 37% |

The SMV formal model is ~37-41% the size of the SysML input — fully auto-extracted by `mc-extract.py`. The reward function and environment code (222 LOC) are shared across all models; they read the SysML model at runtime and generate the simulation + reward automatically.

---

## Scaling Argument

Scaling is approximately linear in SysML model size:

| System | SysML LOC | Extraction (s) | Verification (s) | Training (min) |
|--------|----------|----------------|-------------------|---------------|
| Thermostat | 269 | <2 | <5 | ~55 |
| Cruise Control | 309 | <2 | <5 | ~45 |
| Chemical Mixing | 583 | <3 | ~30 (BMC k=200) | ~588 |

- **Extraction** scales linearly — it's a single-pass AST traversal of the SysML file.
- **Verification** scales with model complexity. IC3 (thermostat, cruise) terminates in seconds. BMC (mixing) with k=200 takes ~30s due to the larger state space (14-phase scan cycle).
- **Training** scales superlinearly with environment complexity. Mixing is ~10x slower per episode (500+ steps vs ~50-100) and converges to lower success rate (95% vs 100%). 

No separate scaling graph is provided — with only 3 data points on the X axis, a table is more informative than a chart.

---

## Runtime Monitoring

> **See:** `metrics/runtime_results/runtime_monitor_summary.csv` and **Fig 5** (`results/graphs_tables/fig5_runtime_monitor.png`)

### Effectiveness

| System | Episodes | Steps | Actual Violations | Detected | FP | FN | Detection Rate |
|--------|---------|-------|-------------------|----------|----|----|---------------|
| Thermostat | 100 | 5,350 | 0 | 843 | 843 | 0 | N/A (no actual violations) |
| Cruise Control | 100 | 4,707 | 0 | 214 | 214 | 0 | N/A (no actual violations) |
| Chemical Mixing | 100 | 55,024 | 22 | 12,560 | 12,538 | 0 | 100% |

**Key safety property: zero false negatives across all systems.** The monitor never misses an actual requirement violation. False positives occur because the `#NeuralRequirement` is a strict biconditional specification (controller output must exactly match the spec for every input), while the RL policy learns a softer approximation. Boundary-condition deviations are expected and configurable via the `--margin` flag (see below).

### Performance

| System | Avg Check (us) | Median (us) | P99 (us) | Max (us) | AST Nodes |
|--------|---------------|-------------|----------|----------|-----------|
| Thermostat | 10.58 | 10.22 | 17.88 | 155.82 | 20 |
| Cruise Control | 14.35 | 13.85 | 24.58 | 45.01 | 28 |
| Chemical Mixing | 10.98 | 10.50 | 19.67 | 192.91 | 23 |

All checks complete in <200us worst-case, <25us P99. Suitable for real-time deployment with dt=100ms.

### Margin Tuning

The `--margin` flag on `eval_runtime_monitor.py` controls the tradeoff between false positives and false negatives:

```bash
python eval_runtime_monitor.py --margin 0.0   # strict: 0 FN, many FP
python eval_runtime_monitor.py --margin 2.0   # moderate: fewer FP, possible FN near boundaries
python eval_runtime_monitor.py --margin 5.0   # relaxed: minimal FP, higher FN risk
```

When margin > 0, comparisons within `margin` of a threshold boundary are treated as passing. Output files are suffixed with the margin value (e.g., `runtime_monitor_summary_margin2.0.csv`).
