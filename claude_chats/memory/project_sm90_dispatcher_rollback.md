---
name: sm90-dispatcher-rollback
description: The measured-best SM90 dispatcher state (96.4% of cuBLAS) to restore if the same-variant experiments go wrong
metadata:
  node_type: memory
  type: project
---

**ROLLBACK POINT for the SM90 dispatcher.** Before the "use cuBLAS's variant everywhere" experiments,
this is the best measured state. Restore from BluBLAS commits **9cbd369** (tier-1 rows) +
**0f0d8c6** (sync into BluTrain's vendored copy) — restore BOTH copies of
`Bgemm_sm90_dispatch_table.h`, the repo-root one AND
`BluTrain/Tensor-Implementations/BluBridge-BLAS/...` (see [[project_blutrain_integration]]).

    kTableSize_nn=147  nt=148  tn=142    15 rows marked "§H27z.55 1.7B"
    1.7B end to end: 26,182 tok/s = 96.39% of cuBLAS   GEMM 13.305 s = 86.1%, 680 TFLOPS

**Measured cost of pinning cuBLAS's variants at all 21 sites: GEMM 13305 -> 15667 ms, 86.1% ->
73.2%.** Every site gets slower; there is NO site where cuBLAS's config beats our own pick in our
implementation. That number comes from the 3905-run sweep, not a guess.

**Do not need a dispatcher change to debug this.** The sweep already measured our TFLOPS on
cuBLAS's exact config for every shape, so the same-variant head-to-head (e.g. 192x192_1x2_h on
16384x11264x2048: ours 573 TF vs cuBLAS 782) is available standalone. Debug there, keep the
dispatcher fast.

**Cannot match cuBLAS in every feature today:** 4 stages on 256x128/128x256 is blocked by our
full-tile epilogue buffer (13 sites), and split-K exists only on 64x64 not 128x128 (2 sites).

**GATING UNKNOWN — settle before pinning any tile:** cuBLAS's BMxBN may be in swapped
(column-major) coordinates, which would make its 256x128 our 128x256 and flip the tile-match count
from 13/21 to 4/21. Test: run M=16384,N=256 and M=256,N=16384 and see if the tile mirrors.

Full detail: analyzations_h200.md §H27z.57 (depth hypothesis) and §H27z.58 (this rollback point).
Related: [[project_sm90_dispatcher_done]] [[feedback_verify_claims_and_benchmark]]
