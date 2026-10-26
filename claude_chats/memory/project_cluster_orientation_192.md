---
name: project-cluster-orientation-192
description: At 7 of 8 production call sites we lose exactly where our cluster orientation is flipped vs cuBLAS and tie/win where it matches - both 192x192 rows are flipped and are 73% of the trainer GEMM deficit
metadata:
  type: project
---

**FOUND 2026-08-28 (§H28e) from the nsys traces of the 1.7B trainer, not from a bench pin.**

    call site       cuBLAS tile/cluster      ours          cluster    per-call
    192x192 NN      192x192 1x2 h coopB      192x192 2x1   FLIPPED    +11.07%
    192x192 TN      192x192 2x1 v coopB      192x192 1x2   FLIPPED    +13.67%
    128x256 NN      256x128 1x2 h coopA      128x256 2x1   FLIPPED     +5.82%
    128x256 NT      256x128 1x2 h coopA      128x256 2x1   FLIPPED     +2.76%
    128x128 splitK  128x128 1x2 h splitK     128x128 2x1   FLIPPED    +23.41%
    128x256 NT#2    192x192 2x1 v coopB      128x256 2x1   MATCHES     +0.17%
    256x128 TN      128x256 2x1 v coopA      256x128 2x1   MATCHES     -0.48%
    256x128 big     192x192 2x1 v coopB      256x128 1x2   FLIPPED     -6.22%  <- the exception

**Where the cluster axis matches cuBLAS we tie or win; where it is flipped we lose.** 7 of 8.

**Why it matters:** the two 192x192 rows are **73.3% of the entire trainer GEMM deficit**
(+6,277 ms of +8,569 ms over 5 steps). This is [[project-tile-raster-coupled]] appearing at real
production call sites instead of on a bench pin.

**How to apply:** it is a **pattern, not a mechanism** — no causal claim, and the 640-call
exception is unexplained. Do not write it up as a cause. Test it the cheap way: flip the cluster
axis on the 192x192 dispatch rows to match cuBLAS, rebuild, re-run the §H28b trainer A/B. It is a
dispatch-table edit plus a rebuild.

**Guard:** measure against the dispatch table, not `pins.sh`
([[feedback-measure-against-dispatch-not-pins]]), and validate every shape that routes through a
retargeted row, not only the pinned one — that is the untested explanation for why the §H27bm
retarget appeared to lose.
See [[project-trainer-parity-1p7b]], [[project-open-tasks-2026-08-28]].

**CAVEAT ADDED 2026-08-28 (§H30).** Those per-call gaps (+11.07%, +13.67%) were measured in the
TRAINER at FREE clocks, where the clock deficit inflates every gap. At a LOCKED 1400 MHz the
whole-kernel time gap across 15 pins is only **1.006x**, and V3 (the 192x192 NN pin) is 1.010x.
So most of the apparent 192x192 deficit is clock, not tile geometry. The cluster-orientation
pattern is still worth testing but its ceiling is far smaller than the free-clock numbers
suggested. See [[project-tensor-ceiling]].
