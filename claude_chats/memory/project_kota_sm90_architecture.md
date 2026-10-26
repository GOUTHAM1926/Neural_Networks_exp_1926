---
name: project-kota-sm90-architecture
description: Kota's SM90 read at CODE level, not by name - his coopA/B and cluster_2d are our cluster-orientation knobs (we have more), but his regsplit/pingpong/smem-overlap are real and we lack them.
metadata:
  type: project
---

Read from **master @ `ea666ab`** (clone `BLUBLAS_kota/BluBridge-BLAS-kota`, dated copy
`BluBridge-BLAS_29_aug_kota`). **Every label below was checked by reading the code, not the name.**

    his label            what the CODE does                                  do we have it?
    coopA / coopB        cluster along N (A-multicast) / along M             YES, and MORE. Ours:
                         (B-multicast). Literally the cluster axis.          NEEDS_A_MULTICAST = MULTICAST_A && CLUSTER_N>1
                                                                             NEEDS_B_MULTICAST = MULTICAST_B && CLUSTER_M>1
                                                                             195 x 2x1 + 91 x 1x2 configs, all tiles, all 3
                                                                             layouts. His: 2 NN-only files.
    cluster_2d           2x2, DUAL-operand multicast (A and B on             YES - 35 registered _2x2_ configs, and our
                         orthogonal axes)                                    multicast derivation gives both at 2x2.
                                                                             MEASURED losing: 2x2_h 84.26 / 2x2_v 81.63.
    persistent           persistent grid-stride loop                         YES, BY DEFAULT - ours is persistent already.
                                                                             His are separate files because his baseline
                                                                             is not.
    pipelined_overlap    NOT PDL. A ThunderKittens smem trick: ALIAS the     NO - but we solve the SAME problem via
                         load ring onto the tail of the store buffer to      EPI_WARP_LOCAL (STORE_BUF 73,728 -> 16,384
                         fit LOAD_PIPE_DEPTH=4 where naive fits 3.           => depth 4). Both of us have PDL separately.
    pingpong             REAL. Each consumer warpgroup owns ALTERNATING      NO - genuinely missing. Cites CUTLASS
                         tiles, so WG-A's EPILOGUE overlaps WG-B's           sm90_gemm_tma_warpspecialized_pingpong.hpp.
                         MAINLOOP. Cooperative (both of us) forces both
                         to the same epilogue barrier.
    regsplit             REAL setmaxnreg, PRODUCER=24/CONSUMER=240,          NO - ours passes 40/232 but `setmaxnreg`
                         budget-checked, LIVE in 21 files.                   appears ONLY IN COMMENTS, one saying
                                                                             "deliberately NOT used". DEAD KNOB.
    fused swiglu         separate entrypoint; his GEMM dispatcher has        irrelevant to the GEMM comparison.
                         ZERO swiglu references.

**HIS DESIGN = one FILE per structural idea**, each with its own Config struct - hence
`coopB_v2`, `coopB_persistent`, `cluster_2d_sliced`, `cluster_2d_persistent`: a combination needs
a new file. **OURS = one template, every feature a composable knob.** That is the real trade -
his files can be hand-tuned freely (which is how `setmaxnreg` survived on his side and died on
ours, where it had to compose with everything).

**HE IS NOT APPLES-TO-APPLES.** He ports ideas from CUTLASS / ThunderKittens / DeepGEMM and keeps
whatever wins; we reimplemented cuBLAS's chosen variant per shape.

**HIS OWN MEASURED LIMITS, from his comments:** best kernel "trails the baseline by a uniform
0.4-9%" (2026-08-06); coopB LOSES to single-CTA at 8192^3 (93.5%) and 16384^3 (94.7%); the
narrow-M/N huge-K tn family tops out at 50-65% of cuBLAS.

**TILE LADDERS.** ours 17 tiles: 64x64 64x128 64x320 64x448 128x64 128x112 128x128 128x144
128x168 128x176 128x192 128x256 128x320 192x128 192x192 256x128 320x128.
his 11: 128x8 128x16 128x32 128x64 128x128 128x160 128x192 128x256 192x128 192x192 256x128.
**He has narrow-N (8/16/32/160) that we do NOT; we have 64x* and 320x128 that he does not.**

**THE HYPOTHESIS THIS SUGGESTS:** our 102-105% ceiling (§H41) was derived on the LARGE 1.7B
shapes we optimised, where §H50 shows we are already at cycle parity with a 92-99% full tensor
pipe. His 100-200 tok/s more plausibly comes from SMALL/MEDIUM shapes and narrow-N tiles plus the
wave-utilisation heuristic - not from beating us on gate_up-fwd. **Testable, and the first thing
the head-to-head should separate.** See [[project-hybrid-blublas-plan]].

## CORRECTION 2026-09-02 — pingpong and smem-overlap are DEAD CODE in his tree

I previously recorded regsplit / pingpong / smem-overlap as "real gaps" we should take from
him. A transitive include closure over his SM90 module (start from every .cu, follow every
`#include "..."`) shows **5,586 of its 21,161 lines — 26% — are unreachable from any .cu**:

    DEAD: pingpong_template (680), pipelined_overlap_template (573),
          cluster_2d{,_sliced,_persistent} (1458), cluster_nn_persistent (637),
          cluster_nn_coopB_persistent (420), cluster_nn_v3_flowing (475),
          nn_clustered_exp (477), nt_clustered_exp (449), nn_rs_exp (417)

    LIVE: warpspecialized_template (2356) <- the workhorse, 6 dispatcher refs
          swiglu_fused_template (4307)    <- own dispatcher, the single biggest file
          cluster_nn_template (722), cluster_nn_v2 (453),
          cluster_nn_coopB (491), cluster_nn_coopB_v2 (488)
          wgmma_ops (712), epilogue_swizzled (141), bf16.cuh (550)

**So pingpong and the smem-overlap trick contribute NOTHING to his measured 98.4% of cuBLAS.**
Do not budget hybrid work for porting them on the assumption they are proven — they are
unwired experiments, same status as our own parked branches. If we want them we would be
developing them, not adopting them.

The genuinely live things he has and we do not: **swiglu_fused** (a fused SwiGLU+GEMM, 4307
lines, its own dispatcher) and **SM90 addmm**. Those are real.

Architectural contrast for the merge: ours is ONE template with ~40 composable knobs producing
235 registered configs; his is ~6 live separate templates that do not compose with each other.
Related: [[project_hybrid_blublas_plan]], [[project_kota_blas_4b_headtohead]].
