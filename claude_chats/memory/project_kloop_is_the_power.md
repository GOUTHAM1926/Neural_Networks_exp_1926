---
name: project_kloop_is_the_power
description: The 4B power excess is the K-LOOP BODY -- parity at short K, +42.6% at long K; alu 4.63x and fma-vs-zero are 86.5% of it
metadata:
  type: project
---

2026-09-11, §H212-§H214, box 86.38.238.116 (1x H200, standalone -- no trainer needed).

**Standalone reproduces the trainer's power gap: +4.15% FLOP-weighted vs §H209's +4.42%
in-trainer, measured on a different box on a different day.** So the excess is a property of
the KERNELS and can be worked on 1 GPU.

**Localised by a K-sweep at fixed M,N (same tile, same 768-tile grid, same kernel in both arms):**

    K=1024  (16 k-iters)  ours 563.9 W  cuBLAS 563.0 W   +0.14%   <- PARITY
    K=4096..16384         ours ~628 W   cuBLAS ~440 W    +42.6 to +44.2%, five values

Prologue/epilogue are fine. **The cost is the k-loop body.** Our power curve is smooth and
monotone; cuBLAS's has a real, reproducible ~60% spike at exactly K=3072 (645.7 W vs ~440
neighbours, same kernel, +-0.3%).

**Per-iteration excess** (census at K=16384 minus K=1024, /240 iters -- removes everything not
scaling with k_iters):

    alu      163,585 vs  35,328   4.63x   57.8% of the excess
    fma       63,744 vs       0   inf     28.7%   <- cuBLAS has NO fma-pipe work in its k-loop
    uniform  211,968 vs 182,784   1.16x   13.2%
    DRAM read                     0.84x   we read LESS

FMA pipe also runs IMAD on Hopper, so that is integer address arithmetic -- matching the
standing note that we rebuild the descriptor base every iteration while cuBLAS keeps it in a
register. See [[project_deadwork_removal_win]], [[project_instruction_excess_worklist]].

**Why:** gives the power hunt an exact target and an exact payoff -- a cut in the loop body
pays 240x more at K=16384 than at K=1024, and 10 of the 15 production shapes run 128x256.

**How to apply:** validate cuts on POWER at a locked clock, never on time
([[project_instruction_cuts_do_not_convert]]). Do NOT reuse §H50's ranking: there uniform led
at 37-44% on 192x192; here alu leads at 57.8% on 128x256. Do not count the "-5.92% win" on
Q/O fwd -- that shape sits on cuBLAS's K=3072 anomaly.

**Tensor is NOT a valid control on this pair:** cuBLAS issues exactly 2x the tensor
instructions per unit of K (narrower wgmma, twice as many). Same math, different instruction
width. See [[feedback_vector_not_total_instructions]].
