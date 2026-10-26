---
name: project_step3_plan
description: THE step-3 plan for BluBLAS on the 1.7B model — 15 shapes, our 8 variants vs cuBLAS's 8; re-read before every step-3 action
metadata:
  type: project
---

**RE-READ THIS BEFORE EVERY STEP-3 ACTION.** The user has had to pull me back on track repeatedly
because I widen the scope on my own.

**The scope, fixed:** the 1.7B BluTrain model calls GEMM at **21 call sites**, 6 of which duplicate
another, giving **15 distinct (M,N,K,layout) shapes**. cuBLAS serves those with **8 distinct
variants**. Layout is PART of the variant identity — same config on a different layout counts as a
different variant (the user's explicit rule: "even if one feature or layout different, consider it
a separate variant"). We already own all 8. Source of truth for the 8: the trainer's own nsys
capture, `profiling_reports/nsys_train_cublas/cuda_gpu_sum.txt`.

**Step 3 = make OUR 8 match then beat cuBLAS's 8 on those 15 shapes.** Nothing else.

The three measurements, each with ONE job:
  * **perf**  — where our 8 stand against cuBLAS's 8 on the 15 shapes
  * **nsys**  — verify both sides issued the variants we expect
  * **ncu + cuBLAS SASS** — backtrack / reverse-engineer to find WHERE our implementation is wrong

**WHAT NOT TO DO — my repeated failures:**
  * Do NOT sweep `--tile`/`--cluster`: it drags in every registered variant (`_rs2`, `_swz128r64`,
    `redA/redB`, `_ep16s2`, `_pd3`…) that nobody asked about. **PIN the 8 exact counterparts.**
  * Do NOT re-open settled questions (coopA/coopB was closed by H12.5; the tile mapping was
    confirmed live by nsys).
  * Do NOT hand-pick a "worst N" subset when the instruction says all 15 shapes.
  * Do NOT surface new side-problems mid-task one at a time — finish the asked thing first.

**Tile mapping, CONFIRMED LIVE (nsys, 2026-08-12):** cuBLAS tile MxN == our tile NxM (operand
swap), cluster label unchanged, cuBLAS layout NN*->our nn, TN*->our nt, NT*->our tn.
Related: [[project_p1_campaign_state]], [[feedback_scope_8_shapes]], [[feedback_verify_claims_and_benchmark]].
