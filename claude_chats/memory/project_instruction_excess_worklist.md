---
name: project-instruction-excess-worklist
description: THE tracked list of remaining instruction inefficiencies vs cuBLAS, measured on the DISPATCHED config (1.67x overhead, 9.3M net excess) - work ALL items, and never census a config production does not run
metadata:
  type: project
---

**RULE THE USER SET 2026-08-28: work through EVERY item. Do not pick one, fail, and divert.**

**RULE THIS FILE ALREADY BROKE ONCE: census the config the DISPATCH TABLE names, never the
pins.sh one.** The first version of this list was measured on bare `256x128x64_2x1_h_ep32s2_pdl`
and reported a 2.56x overhead with `R2UR` as the 19% culprit. Production dispatches
`..._dh_dw_uad_ubd_cs`, where R2UR is **already 94% gone**. See
[[feedback-measure-against-dispatch-not-pins]].

## MEASURED STATE (§H36/H37, `2048x2048x16384 tn`, ncu SASS census, ncu clock-locked)

    cuBLAS  nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN   13,954,931 non-tensor
    ours    256x128x64_2x1_h_ep32s2_pdl        (BARE)        35,672,802   2.56x
    ours    ..._pdl_dh_dw_uad_ubd_cs      (DISPATCHED)       23,270,454   1.67x  <- reality
    existing knobs already removed 12,402,348 instructions (34.8%)

    ALREADY BANKED by dh/dw/uad/ubd/cs:
      R2UR      4,194,304 -> 262,144   (cuBLAS 5,120)   -94%
      LOP3.LUT  4,728,832 -> 534,528   (cuBLAS 15,104)  -89%

    net overhead excess remaining: 9,315,523  (~28% of all our instructions)

## REMAINING WORK LIST — status must be updated as each lands

    [~] 1. UNIFORM-PATH ARITHMETIC -- one attempt made, REVERTED (§H40)        ~4.6M
           UIADD3.64 2,752,512 (cuBLAS 67,712), UIADD3.X, ULOP3, UMOV, ULEA, USHF, USEL.
           TRIED 2026-08-28: low-word 32-bit add (`desc_add_lo`), what DESC_WALK's comment and
           CuTe both describe. Bit-exact on all 15 pins. ELIMINATED all 2,752,512 UIADD3.64 --
           and ptxas added EXACTLY 2,752,512 UMOV plus 4.1M UIADD3 to reconstruct the register
           pair. Net +7.3%, 1.67x -> 1.79x. REVERTED (bcb83ba).
           WHY: `HGMMA gdesc[URn]` reads a 64-bit uniform register PAIR, so every distinct
           descriptor needs its own pair with the high word COPIED into it. We precompute EIGHT
           descriptors per issue group => eight copies however the add is spelled.
           THE REAL LEVER: in-place accumulation in the issue loop, cuBLAS-style --
             HGMMA gdesc[UR16] ; UIADD3 UR16,UR16,0x80 ; HGMMA gdesc[UR16]
           That is a change to how issue_atoms SEQUENCES wgmma, and it conflicts with the
           current design that precomputes all offsets to keep acc[a][nc] compile-time indexed.
           NOT a knob -- a structural change. Sized but not attempted.

    [ ] 2. DIVERGENT CONTROL FLOW -- RESIZED 2026-08-29 (§H43), ~2.7M not 4.3M
           CORRECTION: predicate REGISTER NAMES are not comparable across compilers -- ptxas
           assigns them arbitrarily. "@!P1 1,184,068 vs cuBLAS ZERO" and "@P0 399,872 vs ZERO"
           were NAMING ARTIFACTS (cuBLAS has no @!P1 but has @!P0 doing the same job). Both come
           OUT of this group, and @!P1 is the mbarrier spin back-edge belonging to item 3.
           SOUND comparisons are OPCODES only: BSSY 51 vs 0, BSYNC 51 vs 0, SEL 9 vs 0.
           UNVERIFIED LEADS (a fan-out investigation lost 13 of 16 agents to a session limit and
           worked from SUBSTITUTE bare-config SASS -- record, do NOT believe, re-derive from the
           DISPATCHED config's own SASS first):
             - possibly NO spurious BSSY at all; all 51 may be ~13 real lane-election points each
               diverging at 1/4. CONTRADICTS the earlier "half are spurious" read. Settle this
               FIRST -- it decides whether item 2 is cheap or structural.
             - claimed split 0 spurious / 13 real / 38 never-executed
             - claimed uniform predication would be ~+211.8M instructions WORSE
             - claimed cuBLAS's zero BSSY comes from warp-uniform `BRA P,target`, not predication
             - claimed release_empty's lane guard is 65.8% of all BSSY, predicating it ~+47M worse
             - CUTLASS is NOT at external/cutlass; that -I path is dangling

    [X] 3. MBARRIER SPIN CONSTRUCT -- CLOSED. Two attempts, both fail (§H38, §H39).
           §H38: `_MBAR_SLEEP_TICKS` was a NO-OP -- ptxas silently discards the 3-operand
                 mbarrier.try_wait suspend hint; SASS byte-identical, NANOSLEEP never emitted.
           §H39: fixed it with explicit `nanosleep.u32` (commit b7c8e07). Mechanism ENGAGES --
                 NANOSLEEP 0 -> 4,096,310, spin BRA -95.9% -- and is STILL A LOSS:
                 poll count UNCHANGED (-0.7%), instructions +1.5%, throughput -1.1%.
                 DO NOT ADOPT _tk32/_tk128/_tk50k/_tk12k. Implementation kept (now correct),
                 configs compiled and undispatched.

    [ ] 3b. WE WAIT 156x MORE OFTEN -- ROOT CAUSE IS THE CONSUMER DRAIN, ALREADY DIAGNOSED
           try-wait attempts: ours 6,044,726 vs cuBLAS 38,618 (156x).
           The template header (§H27av note, ~line 7299) ALREADY records the mechanism:
             "consumer drains 12 wgmma to ZERO -> releases late -> producer parks on empty
              (24.57%) -> stage refills late -> consumer parks (3.90%) -> 3 stages give no
              slack to absorb it"
             "cuBLAS's samples land ON the HGMMA (~30%, pipe busy); ours land on the DEPBAR
              (~32%, parked). Identical wgmma work, opposite place."
             consumer DEPBAR ours 31.67% vs cuBLAS 9.08%
           So the spin was never fixable AT THE SPIN (§H38/§H39 confirmed that empirically).
           THE LEVER IS PIPELINE SLACK: deeper LOAD_PIPE_DEPTH, or releasing the stage before
           the full drain. Depth 4 was attempted and PARKED at 96.37 blocked on a PTX fence gap
           ([[project-sm90-depth4-parked]]) -- that item is now the front line, not a backlog item.
           SKEW / ACQ_FIRST / EPI_LATE_STORE_WAIT are prior attacks on the same drain; _lsw is
           already in the V3 dispatched config.

    [ ] 4. YIELD 459,886 (bare 0, cuBLAS 0)  +0.46M  2.6%   -- introduced by the knobs, unexplained
    [ ] 5. S2UR 428,672 (cuBLAS 4,096)  LEA 403,712  SHF.L.U32 394,496

## cuBLAS WEAKNESSES -- exploit, do not just match

    [ ] They issue 1.79x MORE tensor instructions (18.35M vs our 10.22M). OBSERVATION only,
        unexplained across differing tiles; do not build on it until it is.
    [ ] They stall MORE: issue on 5.34% of warp-cycles vs our 7.72%; gmma 20.50%,
        mio_throttle 13.96% vs our 4.08%/2.11% (H30).
    [ ] SHARED weakness, low K: V8 `16384x2048x512 nt` -- BOTH at 70.5%/75.1% of peak tensor.
        25-30% unclaimed by either library.

## STILL OPEN, NOT SUPERSEDED

    [ ] Conversion kernels 21.2% of total GPU time in SEPARATE kernels -- the ONLY lever big
        enough for 105% ([[project-tensor-ceiling]]).
    [ ] L2 1.312x mean, shape-specific ([[project-l2-read-excess-coop]]); its link to power
        is untested.

## SIZE

Removing the remaining 9.3M is ~28% of our instructions, ~2.3x the tuned change that returned
+0.41% => roughly **+0.95%, landing near 100%**. Linear scaling of a measured anchor, an
ESTIMATE. Puts us level-to-ahead of cuBLAS; does NOT reach 105%.
See [[project-tensor-ceiling]], [[project-kloop-unroll-future]], [[project-open-tasks-2026-08-28]].

## VARIANT SEARCH — first result (§H42, 2026-08-29)

Swept **251 configs on V8** (`16384x2048x512 nt`), the shape with the most idle tensor pipe
(70.5% ours / 75.1% cuBLAS). 0 FAIL, all bit-exact.

    128x256x64_2x1_h_ep64swz128_rot2_pdl_dh    562.40 TF   <- best, +1.72% over dispatched
    128x256x64_2x1_h_ep64swz128_rot2_pdl       552.89 TF   <- DISPATCHED
    128x176x64_2x1_h                           540.19 TF   <- a tile in NO pin

**NOT CONFIRMED**: one 400 ms run each, and the cuBLAS reference wobbled 571-594 TF across the
sweep — a 4% spread WIDER than the effect. Never quote that sweep's %cuB column. Repeat the top
three against the dispatched config before believing +1.72%.

**No variant claimed the ~29 idle points** — the whole top of the table sits in a ~10 TF band.
Wave quantisation at 128x256 is only 3.0%; the structural suspect is **8 k-iterations**
(K=512 / BK=64) leaving no steady-state window between pipeline fill and drain, which fits
cuBLAS being stuck at 75.1% too. **Next measurement on any candidate must be TENSOR %, not
TFLOPS** — TFLOPS cannot say whether headroom was actually claimed.
