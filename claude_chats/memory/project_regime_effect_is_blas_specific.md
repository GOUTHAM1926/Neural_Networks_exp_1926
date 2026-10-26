---
name: project-regime-effect-is-blas-specific
description: "The max_steps/LR regime effect is a property of the BLAS, not the box -- cuBLAS loses 2.28%, our BluBLAS only 0.53%; at the production regime we are 96.07% of cuBLAS, not 94.38%"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-10T07:52:54.000Z
---

MEASURED 2026-09-10 on `quiet-sun-closes-fin-03` (2x H200), 4B config, ZeRO-1 v2 + NCCL
symmetric. 8 runs, ABBA on BOTH arm order and regime order. Window steps 8-17 (val-free).

| condition | mean tok/s | within-condition spread |
|---|---|---|
| cuBLAS ms20   | 36,304.7 | 0.044% |
| cuBLAS ms2000 | 35,476.4 | 0.017% |
| ours ms20     | 34,263.6 | 0.059% |
| ours ms2000   | 34,083.1 | 0.016% |

**cuBLAS regime effect -2.28%. Ours -0.53%. 4.3x less sensitive.**
So [[feedback_never_shorten_max_steps_to_benchmark]] is right that the flag matters, but WRONG
to imply it hits everything equally -- it is BLAS-specific.

**Mechanism (telemetry, both arms power-capped at ~692 W, swPowerCap Active 100%):**
cuBLAS clock 1668/1671 -> 1629/1637 MHz (-34 MHz). Ours 1636/1634 -> 1633/1630 (-3, flat).
cuBLAS's achievable clock is data-dependent; ours is not, because our kernels are ALREADY
power-saturated in both regimes -- consistent with [[project_power_clock_solved]]. Being
power-saturated makes us *immune* to a change that costs cuBLAS 2.3%.

**THE NUMBER THAT MATTERS:**
- ms20 (short benchmark): ours = **94.38%** of cuBLAS
- ms2000 (production): ours = **96.07%** of cuBLAS

Every ms20 figure understates us by **1.69 points**. Re-baseline anything quoted from a
`BLU_MAX_STEPS`-shortened run.

**And at ms2000 both arms run at the SAME ~1633 MHz at the same ~692 W** -- so the remaining
3.93% there is genuine COMPUTE, not clock. That is the real gap to close on 4B, and it is
smaller than the 2026-09-09 numbers implied.

Full write-up: §H186-§H189 in `cublas_rev_eng_binary/analyzations_h200.md`.
Artefacts: `h186_regime_2026_09_10/` (16 files). Relates to [[project_kota_run_gap_resolved]],
[[project_4b_dispatch_gap]], [[project_kota_blas_4b_headtohead]].
