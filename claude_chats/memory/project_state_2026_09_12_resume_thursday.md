---
name: project-state-2026-09-12-resume-thursday
description: BluBLAS/matmul work is ON HOLD from 2026-09-17 - user reassigned to SFT research. This is the frozen resume point; do not restart it until the user says so.
metadata:
  type: project
---

# ON HOLD as of 2026-09-17

**The user was reassigned to SFT (supervised fine-tuning) research in POST-TRAINING.** All
BluBLAS / matmul / GEMM work below is PAUSED. Do not propose restarting it. The user will say
explicitly when to resume. Everything is committed and documented - nothing is at risk.

Remind them of this status periodically, as they asked.

No Verda box running - the 2xH200 `sharp-fish-circles-fin-02` was cut and its IP reassigned.
A new box will be needed whenever this resumes.

## WHERE WE STAND

**4B trainer: 99.80% of cuBLAS at the production schedule** (was 96.07%). See
[[project-4b-trainer-99p8-production]].

| measurement | result |
|---|---|
| standalone @ 1200 MHz locked | **+1.72%** (we win 13/15) |
| standalone @ free clocks (production regime) | **+0.75%** (11/15, both arms at the 691W cap) |
| 4B trainer end-to-end | **-0.20%** |

**The clock regime decides the verdict** - locking 1500 or 1800 both yield only 1440 MHz because
the 700 W cap binds, so the official 989.5 TF is unreachable (real ceiling ~717-725 TF). At
1200 MHz we win; at the clock the GPU actually runs, cuBLAS wins. **Never quote a locked-1200
number as a production ratio.**

**Tensor occupancy: ours 95.54% vs cuBLAS 93.94%** launch-weighted, we lead 14/15.
`corr(tensor%, TFLOPS) = +0.947` - occupancy sets throughput.

**The instruction excess is quantified**: ALU+FMA is **41.4% of our instruction stream vs 10.8%
of cuBLAS's** (6.83x; FMA pipe alone 83x). Removing it would take the instruction ratio from
1.68x to ~1.1x. BUT `corr(instructions, power) = +0.527` only - best predictor, NOT proof of cause.

**Kernel selection is NOT the problem**: both arms pick the identical kernel standalone and
in-trainer on all 15 shapes (§H230, verified by reconstructing all launch counts).

## THE TARGETS, in order

1. **`Q/Owgrad` (tn 3072x3072x8192)** - **-2.53%** at free clocks, 5.9% of FLOPs. Worst shape.
   192x192 tile. §H226's confirmed candidate (`2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_wl`) targets
   exactly this and is **validated but NOT pinned** (+0.538% tput, -0.339% power, 12-round
   crossover with a clean null).
2. **`gate_upwgrad` (tn 3072x16384x8192)** - -0.71% on **15.8% of FLOPs**, the largest share we
   lose on.
3. **`K/Vdgrad`** - worst tensor occupancy (88.34%) and lowest TFLOPS (535.4); K=1024 means only
   16 k-iterations. Biggest per-shape headroom.
4. **128x256 combination sweep, nn and nt layouts** - never run (§H227 did tn only). That tile
   serves 10 of the 15 shapes. See [[feedback-sweep-combinations-not-single-knobs]].
5. Re-run §H226's crossover at a 6 s plateaued window before pinning anything (§H227).
6. Un-register or fix 8 broken registry configs (2 DESC_FROZEN wrong answers, 5 MBAR_SLEEP_TICKS
   hangs, 1 splitk wrong on 128x256 tn).

## WHERE EVERYTHING IS

* **Code + docs + results**: `~/Downloads/blublas` (= `GOUTHAM1926/blublas.git`), HEAD `76f9458`.
  See [[project-local-blublas-checkout-path]].
* **Documentation**: `~/Downloads/blublas/docs/analyzations_h200.md` - 34,534 lines, §H1 -> §H231e.
  **This is the single master. It used to live in the Neural_Networks repo; do not put it back.**
* **Today's run artefacts**: `~/Downloads/Neural_Networks_exp_1926/h229_2026_09_12/` (125 files).
* Both repos: 0 ahead, 0 behind, 0 dirty.

## SETTING UP A NEW BOX (all of this bit us today)

1. Clone `blublas.git`; `blutrain_ours/` is now IN the repo, no rebuild needed.
2. **Install MPI** - `apt install -y libopenmpi-dev openmpi-bin`. Not present by default.
3. **Dataset**: HF `navingv/flux` (public) into
   `/root/.cache/huggingface/hub/datasets--navingv--flux/snapshots/5232b3d133d2eb71905725f288ff6b56dc0df773/`.
   One train shard + one val shard is enough for 30 steps.
4. `bluscript_zero.cpp` needs `#include "Checkpointing.h"` not `"checkpointing/Checkpointing.h"`
   (upstream path mismatch), and `checkpointing = false` for benchmark runs.
5. Build once: `make clean && make all -j88` (box has 88 cores), then **flip `WITH_BLUBLAS` on the
   RUN command** - the stamp mechanism rebuilds the two guarded files. See
   [[feedback-i-was-wrong-on-with-blublas-flag]].
6. Verify the arm from the **nsys kernel names**, never from symbols alone.
