---
name: project-hybrid-blublas-plan
description: THE HYBRID PLAN - merge Gautam's and Kota's SM90 into one library, decided by a 3-arm per-SHAPE measurement; A/B verified as a clean relink; port list is regsplit, small tiles, smem-overlap, pingpong, SM90 addmm.
metadata:
  type: project
---

**Two approaches, now merging. Documented in full as §H55 and §H56.**

*Ours*: every Qwen-report matmul shape, the variant cuBLAS issues recorded per shape, then those
variants reimplemented apples-to-apples across nn/nt/tn -- ~750 configs from ONE composable
template (8,217 lines) + a 967-line hardcoded shape->config table. Only the 1.7B configs
optimised. §H41 ceiling 102-105%; §H50 shows why -- already at cycle parity at locked clock.
**At par, not ahead.**

*Kota's* (`BlubridgeAI/BluBridge-BLAS` master @ `ea666ab`, cloned to
`BLUBLAS_kota/BluBridge-BLAS-kota`, dated copy `BluBridge-BLAS_29_aug_kota`): several SPECIALISED
templates + a 2,051-line rule dispatcher. **Beats cuBLAS by 100-200 tok/s** across the yaml model
configs, and is the version accepted for BluTrain.

**TARGET: one library, one two-level SM90 dispatcher. L1 = hardcoded pins (ours 967 lines + his
grid-keyed overrides). L2 = his wave-utilisation heuristic, which we have no equivalent of.
NOTHING is deleted from either side** - his coopB loses at large squares where ours wins, his
narrow tiles serve shapes ours does not.

**THE A/B IS A CLEAN RELINK - VERIFIED.** 121 exported symbols each; `blublasBgemmStridedBatched`
byte-identical. Only 4 symbols differ and **none are called by BluTrain's own code** (every hit
was inside the VENDORED copy calling its own internals). **Swap mechanism: replace the vendored
tree at `BluTrain/Tensor-Implementations/BluBridge-BLAS/`, rebuild, run. No BluTrain source
change.**

**THE PORT LIST** (see [[project-kota-sm90-architecture]] for what each label actually is):

    regsplit (setmaxnreg 24/240)   knob EXISTS in ours, emit site removed -> revive it. Cheapest.
    small tiles 128x8/16/32/160    new BM/BN/BK instantiations; watch BN % EPI_TILE_N == 0 and
                                   the BN in {64,128,256} paths
    smem-overlap depth-4           knob; then A/B against our EPI_WARP_LOCAL, which solves the
                                   same depth-3 problem a different way
    pingpong                       a DELIBERATE FORK, not a knob - it changes tile OWNERSHIP.
                                   Forcing it into one template doubles the validation matrix
                                   for every future knob, permanently.
    SM90 addmm                     absent on our side entirely (we have SM103 addmm); a whole
                                   call site, not a knob

**THE DECIDING MEASUREMENT.** Three arms - ours / his / cuBLAS - on the training scripts for
every yaml model config, free clocks, same box, same day. **Per-SHAPE, not per-model**: nsys
per-kernel paired by call site (as §H28d), because the decision is per-call-site routing.
Backends swapped by SNAPSHOT with symbol counts verified before each run
([[feedback-backend-ab-snapshot]]). **Order 120M -> 600M -> 1.7B -> 4B.**

**WHY SMALL MODELS FIRST.** §H41's ceiling was derived on the LARGE 1.7B shapes where §H50 shows
cycle parity and a 92-99% full tensor pipe - almost no headroom, and neither regsplit nor
pingpong adds math throughput. His 100-200 tok/s most plausibly comes from the SMALL/MEDIUM
shapes (narrow-N, low tile counts) where wave quantisation dominates and his heuristic and narrow
tiles pay - and where we never looked. Testable, and the cheapest thing to settle first.

The resulting (shape x library x us/call) table **IS** the hybrid's level-1 dispatch. Then re-run
the full training scripts on the hybrid and confirm it reproduces the per-shape wins.

**STATUS 2026-08-29: all local analysis done, nothing measured. Waiting on the GPU, on Gautam's
word.** Remaining local prep: build his library standalone on this toolchain, and write the
3-arm harness.
