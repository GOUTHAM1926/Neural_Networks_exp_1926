---
name: feedback-never-shorten-max-steps-to-benchmark
description: "BLU_MAX_STEPS=15/20 changes warmup and the LR schedule, which changes switching power by 3.4% and costs 2.46% throughput on a power-capped box -- benchmark at the real 2000-step schedule"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-09T15:05:46.735Z
---

Never shorten a trainer run with `BLU_MAX_STEPS` to make a benchmark cheaper. It
silently changes what you are measuring.

`BLU_MAX_STEPS=N` sets `warmup_steps = N/10` and `stable_steps = N/2`. At N=20 the
warmup is 2 steps, so the run sits at lr=3e-4. At the real default (max_steps=2000)
the warmup is 200 steps, so the first 20 steps sit at lr~2e-5. Different LR ->
different weights -> different data through the multipliers.

Measured on the 4B trainer, 2xH200, same binary, BAAB crossover (2026-09-09):

| condition          | tok/s   | clock       | power     |
|--------------------|---------|-------------|-----------|
| MAX_STEPS=20       | 36,360  | 1670/1658   | 693 W     |
| default 2000       | 35,488  | 1628/1623   | 690 W     |

2.46% apart, within-condition spread 0.005-0.03%.

Locked to 1410 MHz (lock held 100%, 594-619 W, under the cap) the SAME two
conditions are **0.10% apart** (dt 15,263.3 vs 15,247.8 ms) and the power differs
by **+3.4%**. So the compute is identical; the whole gap is data-dependent
switching power -> the 700 W cap -> ~42 MHz of clock.

**Why:** on this box every throughput comparison is a power comparison. Same
instructions with different operand values toggle different numbers of bits, burn
different power, and at a pinned cap that converts straight into clock. Same
mechanism as [[project_power_clock_solved]] (+2.69% there).

**How to apply:** benchmark at the production schedule (don't set BLU_MAX_STEPS),
and cap wall time instead -- `timeout -s INT <seconds> mpirun ...` -- reading the
per-step CSV, which flushes every step. When a throughput difference appears,
re-run both arms at a locked clock before believing it is a compute difference.
Relates to [[feedback_lock_clocks_before_timing]] and
[[feedback_same_step_window_before_comparing]].

This is what explained the whole "why is Kota's cuBLAS slower than ours" thread --
see [[project_kota_run_gap_resolved]].

## CORRECTION 2026-09-10 — the effect is BLAS-SPECIFIC, not universal

Re-measured with 8 ABBA runs on 4B: cuBLAS loses **2.28%**, our BluBLAS only **0.53%** (4.3x
less). So the rule "benchmark at the production schedule" still stands — but do NOT assume the
penalty is the same for both arms and cancels out in a ratio. It does not: ours reads 94.38% of
cuBLAS at ms20 and **96.07%** at ms2000. See [[project_regime_effect_is_blas_specific]].
