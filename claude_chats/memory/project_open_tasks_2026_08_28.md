---
name: project-open-tasks-2026-08-28
description: Everything left open at end of 2026-08-28 in priority order - 1.7B is DEFERRED by the user, the 192x192 cluster flip is the highest-value lead, and the tuned-dispatch arm was never measured
metadata:
  type: project
---

**USER'S DECISION AT EOD 2026-08-28: "we will do on 1.7B later."** Everything below is a future
task. Nothing here is to be worked without first re-reading the retraction history.

**RE-READ FIRST:** [[feedback-backend-ab-snapshot]] — a full day of trainer measurement was void
because both arms ran cuBLAS. §H27bj and §H27bn are retracted (§H28a).

**HIGHEST VALUE (revised 2026-08-28 after §H30-H32)**

**0. THE POWER QUESTION — the correlation test is DEAD, use an INTERVENTION test instead.**
The per-pin sub-cap correlation was attempted and **could not be executed** (§H33/H34): `-lgc` is
not enforced on this box, so a back-to-back GEMM loop clamps at the 700 W cap and **12 of 15 pins
had power clamped**; the 3 survivors carry an instruction-ratio range of only 1.779-1.808 against
the 1.567-2.543 the test needed. NOT refuted — not run. Do not re-attempt it the same way.

**Replacement (causal, and runs on the only rig here that holds sub-cap):**

    arm A   trainer, current TUNED dispatch
    arm B   trainer, dispatch routed to the LOWEST-instruction configs available
    both    locked 1400 -- the TRAINER is sub-cap there (cap 0%, 523-538 W), unlike the bench
    read    trainer power at matched clock, 10 steps each

If instruction count drives the +2.69%, arm B draws measurably less. If power does not move,
instructions are refuted as the power driver, end to end. Power is still the only lever big
enough for the near-term goal: ~420 tok/s against ~170 for the entire remaining work gap.

**GOAL RESET BY THE USER 2026-08-28:** beat cuBLAS on the trainer by 100-200 tok/s
(28,220-28,320 from today's 27,862), i.e. ~103%. The 130% ambition is explicitly deferred to a
later, non-apples-to-apples strategy. See [[project-tensor-ceiling]] for why 105-109% is not
reachable on these tiles.

**PREVIOUSLY HIGHEST VALUE, NOW DOWNGRADED**

**1. Flip the 192x192 cluster axis to match cuBLAS (§H28e).** 73.3% of the trainer GEMM deficit
sits on two call sites whose cluster orientation is flipped; 7 of 8 call sites fit the pattern.
Dispatch-table edit plus a rebuild. See [[project-cluster-orientation-192]].

**2. Re-run the tuned-dispatch arm — it never produced a number.** It built cleanly (12 syms,
libtensor md5 `a07f1e3a389d` vs baseline `5d971abf1b56`), the baseline table was restored
(md5 `58d32d6db8e4`), the snapshot survives at `/root/libs_blublas_tuned`, and the box was
released during step 0. §H27bn's "tuned dispatch is 1.4% slower" is **void**, so the question is
simply open. Re-run: `/tmp/h28c.sh` (script also saved as `test_reports/h28b/…`).

**3. Does L2 traffic CAUSE the +2.69% power? (§H27bo — designed, still NOT run.)** The one
unproven arrow. The chain L2 excess -> power -> cap -> clock -> throughput is now confirmed from
the power end (§H28c) but the L2->power link is not. Test: 2x2 cluster cuts L2 requests 14.6% at
identical tensor work; measure power 2x1 vs 2x2 at locked 1200 (both sub-cap).

**4. Why 1.64x the L2 read requests? (§H27b5-bi)** **Five** explanations proposed, all five
retracted. Measured: requests 1.64x, hits 2.35x, misses 1.16x, DRAM 1.00x, identical TMA counts,
41.3 L2 requests per TMA transaction vs cuBLAS's 24.1. **Do not propose a sixth mechanism without
observing it directly.**

**NEW, UNEXPLAINED**

**5. `128x128 splitK` is +23.41% per call in the trainer** while standalone V1 measures 105.6%
locked. Contradiction — probably different shapes. Check what the trainer actually dispatches
before concluding anything.

**6. V2c is the only pin that gets WORSE when clocks are locked** (91.1% free -> 89.3% locked).
Every other pin gains +3 to +10 points. Unexplained.

**MEASUREMENT DEBT**

**7. Our own ncu counters were never captured.** `--launch-skip 3 --launch-count 1` landed on the
bench's cuBLAS reference kernel, so all four §H28 ncu reports profile cuBLAS. Useful as targets —
cuBLAS `192x192_64x3_1x2_h_coopB_NNN`: 1,572,183 cycles, 49.58M TEX/TMA L2 read requests, 83.34%
L2 hit, 549 MB DRAM, 109.42M smsp insts, 90.47% SM throughput, 168 regs, 188.6 KB smem, 14.82%
occupancy. Re-run with a launch index that selects OUR kernel.

**8. Standalone locked covered only 7 of 15 pins** before the box was released.

**8b. RUN THE REVERSED-ORDER CONTROL on the free-clock trainer pair.** cuBLAS ran FIRST in every
arm of §H28b, and [[feedback-lock-clocks-before-timing]] / §H27zD showed order alone produced the
standalone free-clock gap. Re-run free-clock with BluBLAS first: if the 30 MHz survives the swap
it is ours, if it flips sign it was order. Three things argue it is real (thermal trajectories
overlap; the two clocks move in OPPOSITE directions over the same ramp; the locked pair has no
order freedom and still shows +14.1 W) but the control itself was not run.

**9. V6 still carries the pins.sh defect (§H27bl)** — pins.sh names
`192x192x64_2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_wlsm`, the dispatch table routes V6's shape to
`128x256x64_2x1_h_ep64swz128_pdl`. Never re-measured.
See [[feedback-measure-against-dispatch-not-pins]].

**10. V2c row worth retargeting carefully.** `128x256x64_2x1_v_ep32s2_dh_shfl` beats the
DISPATCHED config by +1.4 free / +2.2 locked. Validate every shape routing through the row first
— that is the untested explanation for the §H27bm retarget's apparent loss.

**OLDER PARKED, UNCHANGED:** [[project-stmatrix-reopen]];
[[project-convert-kernels-future]]; [[project-pdl-everywhere-future]]; full 458-config sweep;
[[project-sm90-depth4-parked]].

**THEN: the 4B model**, same plan as 1.7B.
