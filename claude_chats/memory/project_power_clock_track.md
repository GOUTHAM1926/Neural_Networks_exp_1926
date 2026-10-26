> **SUPERSEDED 2026-08-27 — see [[project-power-clock-solved]].** The "power is 1.00x at free
> clocks" reading below is the CAP CLAMP, not parity: at free clocks both libraries sit pinned
> at 700 W so power *cannot* differ. Measured BELOW the cap at equal clock we draw
> **1.11-1.13x**. And the conclusion "cutting instructions is the lever" is REFUTED — removing
> 76% of the k-loop vector ops moved power 2.1%.

---
name: project_power_clock_track
description: PARKED future task - the power/clock track. Our kernels match cuBLAS on work; the production gap is that we issue 2.24x the instructions and so get less clock under the 700W cap.
metadata:
  type: project
---

**RESOLVED 2026-08-26 (§H27ag) — THE MECHANISM IS `P = C.V^2.f`, NOT POWER DRAW.**
At FREE clocks, measured on all 15 shapes with settled 10 s windows at 99 Hz (103,027 samples):
**power is 1.00x and BOTH libraries are 100% power-capped on every shape** — yet our clock is
lower, and the loss tracks the INSTRUCTION RATIO at **Pearson r = -0.742, slope -6.82% clock per
1.0x**. We do NOT draw more power; we cannot, we are both pinned at the ceiling. We spend the same
watts on a higher ACTIVITY FACTOR C (more switching per cycle) and are handed fewer cycles.

    issue 1.56-1.61x  ->  mean clock loss -0.37%
    issue 1.76-1.83x  ->  mean clock loss -0.75%
    issue 2.07-2.27x  ->  mean clock loss -4.55%

FALSIFIABLE PREDICTION: cutting the ratio returns ~6.82% clock per 1.0x. V1 2.07x->1.50x predicts
+3.9%. **Cut instructions, re-measure clock, check against the prediction** — if it does not move,
this is wrong.

RETIRED by this: "we draw more power than cuBLAS" (1.00x, fifteen shapes, settled). Also retired:
§H27af's suggestion that instruction-cutting is not the lever — it IS, via activity factor.

CAVEATS: r=-0.742 not 1.0, so something else is also in play; ncu says 2.54x on V1 where the nsys
issue counter says 2.07x, so the slope's absolute value is soft though the ranking is consistent.

**ORIGINAL ENTRY BELOW — its framing ("the governor gives us less clock under the 700 W cap") was
directionally right but the causal story was wrong: it is activity factor at a shared ceiling, not
us drawing more.** Independently reproduced on a DIFFERENT box
(`loud-tree-grows-fin-03`, H200) with per-kernel clock via `nsys --gpu-metrics-devices`, and the
pattern is consistent across shapes:

    shape                        lock    %cuB   ourClk  cubClk  clkGap
    V1  2048x512x16384           free   100.0     1479    1548   -4.5%
    V5  2048x11264x16384         free    98.0     1360    1388   -2.0%
    V5                           1400   100.1     1405    1407   -0.1%   LOCK HELD
    V6  16384x5632x2048          free    95.8     1382    1431   -3.4%
    V6                           1400    99.0     1396    1395   +0.1%   LOCK HELD
    V7  16384x2048x2048          free    98.2     1433    1467   -2.3%
    V7                           1400   100.3     1391    1389   +0.1%   LOCK HELD
    V7b 16384x2048x5632          free    96.0     1362    1419   -4.1%
    V7b                          1400   100.0     1401    1401   -0.0%   LOCK HELD

**AT EQUAL CLOCK WE ARE 99.0-100.3% OF cuBLAS. THE FREE-CLOCK DEFICIT TRACKS THE CLOCK DEFICIT
ALMOST ONE-FOR-ONE.** So the production gap is not kernel quality -- it is that we are handed less
clock. V1 is the sharpest case: 100.0% of cuBLAS while running 4.5% SLOWER in clock, i.e. per
cycle we are AHEAD.

**THE ACTIONABLE TASK: CUT THE INSTRUCTION COUNT.** Per-kernel counters on V1, same capture:

    ours / cuBLAS    tens 78.8/75.2    issue 31.4/15.3 (2.05x)    smact 95.1/90.2    dramR 32.0/31.7

Identical DRAM traffic, HIGHER tensor and SM activity, but 2.05x the instruction issue. Issue rate
burns fetch/decode/issue power, so the governor returns less clock under the same envelope. Fewer
instructions for the same tensor work -> more clock -> the free-clock gap closes. cuBLAS does the
same math in half the instructions, so it is achievable, not a law.

**WHAT IS ALREADY PROVEN.** At a held clock lock our kernels equal cuBLAS; at free clocks they do
not, and the whole difference is the SM clock the governor hands each kernel.
V2 2048x2048x16384, per-kernel clock measured with `nsys --gpu-metrics-devices` (not inferred):

    lock 1400 (held)   ours 1405 MHz   cuBLAS 1404 MHz   +0.1%   ->  99.8% of cuBLAS
    free clocks        ours 1389 MHz   cuBLAS 1457 MHz   -4.7%   ->  94.8%
    arithmetic closes: 99.8 x (1-0.047) = 95.1

**THE CAUSE, measured per kernel in the same trace.** Everything matches except one thing:

    Tensor Active  93.2 / 93.6      identical
    SMs Active     95.5 / 95.4      identical
    DRAM Read      14.5 / 14.5      identical
    SM Issue       24.7 / 11.0      2.24x  <-- ours

Independently corroborated by ncu: `smsp__inst_executed.sum` 35,946,060 vs 16,035,216 = 2.2417,
the same ratio from a different instrument. Stable at free AND locked clock, so it is structural.
So: identical tensor work, identical memory traffic, identical occupancy, but 2.24x the
instruction ISSUE rate -> the fetch/decode/issue pipeline burns the watts -> less clock under the
700 W cap -> ~5% slower in production.

**THIS REOPENS THE INSTRUCTION-COUNT WORK I CLOSED TOO EARLY.** Four cuts totalling -470M
(dw, dw+unroll, dh_ubd, uad) were all judged null -- but every one was measured by %cuB at FREE
clocks with a +-1.5 pt wobble. The right instrument is per-kernel **SM Issue** from nsys, which is
far more sensitive. UNTESTED PREDICTION: uad cut V2 35.9M->23.4M (-35%), so SM Issue should fall
24.7 -> ~16 against cuBLAS's 11.0, and the sustained clock should rise with it.

**THE FULL MATRIX STILL OWED:** all 15 shapes x 8 variants x the frequency ladder
(free/1200/1300/1400/1500/1600/1755), capturing OUR clock and cuBLAS's separately, plus the
REVERSED measurement order to confirm whether the order bias hits only us or cuBLAS too -- and if
cuBLAS is unaffected, what it does to hold its clock. Tooling exists: `tools/h27zw_clock_matrix.sh`
(per-kernel clock + tensor/issue/smact/dram), `tools/h27zv_ladder_variants.sh` (nvidia-smi
fallback). Valid band on this H200 is 1200-1400 MHz; the cap overrides the lock between 1400 and
1474. See [[feedback_lock_clocks_before_timing]].

## TWO CONCRETE LEADS FOR WHERE THE EXTRA INSTRUCTIONS ARE (both OPEN, both need DYNAMIC counts)

**LEAD 1 — WE USE THE PER-THREAD DATAPATH WHERE cuBLAS USES THE UNIFORM ONE.** Static SASS
opcode census, ours (`box_src/probe.sass`, 256x128 TN) vs cuBLAS
(`sweep_results_cublas/sass/shape_99_M4096_N11008_K2048.sass`, 192x192):

    OURS (per-thread datapath)        cuBLAS (uniform datapath)
    IMAD    757                       UIMAD   1702
    SHF     187                       USHF     575
    LEA     174                       ULDC    1771
    ISETP   121                       UISETP  1150
    IADD3    78                       UIADD3  3197

Same operations -- address arithmetic, predicates, shifts -- but the U-prefixed forms issue ONCE
PER WARP against the uniform register file instead of once per thread. Cheaper in issue slots and
in power, which is exactly the currency we are short of.

**LEAD 2 — STSM vs STS.** `STSM.16.M88.4 [R23+UR32], R24` moves a whole register block to shared
in one instruction; our `STS [R14+0x30000], R17` moves one register. Counts: ours STS 128 / STSM 0,
cuBLAS STS 483 / STSM 414. See [[project-stmatrix-reopen]] -- and note the REFRAME: the old
stmatrix test measured DURATION on a path worth 1.4% of the kernel and correctly found nothing.
The right metric now is INSTRUCTION COUNT, because that drives clock, which drives everything.

**THE CONFOUND ON BOTH, DO NOT SKIP IT.** These are STATIC counts from two kernels with different
unroll structure -- ours loops (16 static HGMMA), cuBLAS is heavily unrolled (552 static HGMMA) --
so static ratios do NOT translate to dynamic ones. Our IMADs could be one-time setup code. These
are LEADS, not results. The instrument that settles it is ncu dynamic instruction mix
(`smsp__inst_executed` broken out by category) on the SAME shape for both libraries.
See [[feedback-isolated-measurements-mislead]].

