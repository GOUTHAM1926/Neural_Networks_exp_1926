---
name: project_dense_fwd_gap_1017
description: "10-19 16:50 SCOREBOARD (same box, GPU 1, steady): best = PF; dense fwd 1.025x FA3 (morning 1.084), dense f+b 0.985, packed fwd 0.866, packed f+b 0.935; beats FA4/cuDNN everywhere. PF ncu: tile-0 K/V wait 1,498 -> 48. Left (~750 cyc/unit): our diagonal-LAST masked step (+635) + epilogue (+383); FA3 does diagonal FIRST"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-17T15:43:25.113Z
---

**K5q ncu (2026-10-17 20:46, 2x box, work doc §45.39; files profiling_reports/session_2026_10_17/box_runs/2x_1017/ncu_k5q_1017,
script deep/ncu_k5q_vs_fa3.py):**
- PACKED real_rows_752: K5q 0.670M cycles vs FA3 0.744M (0.90x), instructions 130.9M vs 142.2M. Ours already ahead.
- DENSE B8 18/6: K5q 1.538M vs FA3 1.423M (1.08x), instructions 322.4M vs 306.5M; tensor busy 77.1% vs 83.3% (same tensor
  work, more idle). Excess +8.1 points (FA3 = 100), tile counts matched:
  - steady loop +4.0 (a step costs 146.6 vs 139.0 per million steps): QK issue 72 vs 41 instructions (+5.5 per M: we
    rebuilt Q AND K descriptors for every HGMMA in ONE group gdesc[UR20]; FA3 keeps Q's 8 in 8 groups, built once per unit);
    softmax +5.9 per M with the same instruction count (MUFU "wait" +7.2: exps packed too close); PV issue -3.8 per M.
  - unit boundaries +4.0 (prologue tile 0 + diagonal tile + start/end): long-scoreboard waits for Q / tile-0 K/V +3.5.
- Clocks: ncu on the 2x box locks ~1,336 MHz, FA3's 10-13 reference ran ~1,620/1,730; F3c measured at both: +1.7% cycles
  dense / +0.5% packed at the higher clock, so cycle comparisons hold to ~1-2%.
- A local laptop nvcc 13.0 build gives the SAME SASS as the box build (only `.reuse` differs): check candidate SASS locally.

**Fix L1b** (candidates/fwd_L1b.cu 069772b7, switch L1_DESC, from the box's K5q source f35982af): Q descriptors built once,
K/V = stage-0 descriptor + stage offset. L1 alone (fwd_L1.cu) did NOTHING: ptxas rebuilt the "once" values in the loop from
the live smem base. Passing each Q descriptor half once through `__shfl_sync(.., 0)` stops that: QK issue 72 -> 42 (FA3 41),
168 regs, 0 spills. Same values -> bit-identical. NOT raced yet (first thing Monday, see [[project_state_2026_10_17]]).
Next after it: exp scheduling in the softmax; Q / tile-0 prefetch at unit start.

**Earlier (K1, §45.32):** same clock as FA3, +1.8% instructions, +18% cycles, 77% of the excess at pair start/end. K4b/K5q
fixed most of that (O through the last K buffer, split 16, one P set).

**REFUTED on the forward (§45.31; was memory project_mbar_wait_form):** our mbar_wait is try_wait without the CUTLASS
suspend hint (FA3/FA4/cuDNN: NANOSLEEP.SYNCS 0x989680); the step-start K/V wait is bimodal (100-150 / 275-300 cycles) but it
is SLACK: W1 (CUTLASS wait) -0.07% null, W1s +0.14/+0.44% worse, W2 (separate K / V barriers) -0.11%/+0.10% null (the wait
shrank 50-75 cycles, part A grew the same). Polynomial exp P8w +2.7/+3.4%, P4w +7.1/+7.4% (~0.035% per added instruction per
tile: the softmax phase is ISSUE-bound). Knock-outs: no exp -6.8%, no max -4%, no rescale -2.3%, all -11.2%.

**Why:** "we do the same as FA3" is true for the math, not for descriptor setup, exp placement and unit boundaries.
**How to apply:** do not retry wait-form / K-V split / polynomial exp; race L1b first; treat trace stamps as strong evidence
and the race as the verdict; dense and packed must stay bit-identical ([[feedback_same_numerics_dense_packed]]).

**10-19 RESULT (§45.41): L1b REJECTED.** New box CUDA 13.4: K5q's dense QK issue is already 59 (13.0: 72), packed 40 in 8
groups; L1b 43/43. Race l1b_1019: L1b vs K5q dense +0.36% (t 2.2), packed +0.90% (t 11.3). Lesson: the 45.39 split is
WARP time; the QK-issue excess was slack (warps wait for the baton/tensor pipe anyway). Judge the next lever by tensor-pipe
idle time (FA3 83.3% busy vs K5q 77.1%), not stall shares. 2-library ab_multi races run the candidate first every round.

**10-19 SAME-BOX SCOREBOARD (§45.43, free clocks, steady state, new box CUDA 13.4, ours = vj2_K5q):** ours fastest in 11 of
12 entries vs FA3 / FA4 / cuDNN. Only loss: dense forward vs FA3 1.084x (1,055 vs 974 us). Dense backward 0.965x FA3, dense
fwd+bwd 0.992x. Packed: forward 0.936x FA3, backward 0.952x, fwd+bwd 0.948x. FA4: ours faster everywhere (dense fwd 0.985).
Rivals env: /root/gautam_K/benv via session_2026_10_19/scripts/benv_setup.sh (needs pydev headers + GIT_CONFIG safe.directory).
nvidia-smi shows a stale 345 MHz for the first ~2 s of our C++ runs (artefact; kernels were faster then).
**10-19 ncu same box (§45.44): the dense gap = +150,561 tensor-IDLE cycles/SM (tensor-busy equal: 1.186M both).** Fit over
dense + packed: ours 3,905 idle cycles per unit boundary + 438 per tile step; FA3 9,422 per boundary + ~0 per step (its steady
step is tensor-bound). We win packed (cheap boundaries), lose dense (per-step overrun ~440 cycles of a 2,048-cycle tensor step).
Target = each group's per-step critical path (softmax + rescale/packs + issue) must fit the ~1,536-cycle window; FA3's fits.
**10-19 afternoon, ALL REJECTED for the dense gap:** R8/R16/R32 row-split softmax (dense +0.34/+0.76/+0.12%, R32 packed
+2.5%); KOkv = NO in-loop K/V wait at all: dense only -0.30%, packed -1.79% (3 rounds) -> FA3's split K/V pipeline (K released
right after QK, V one tile behind; confirmed in FA3 source mainloop:870-880 / 1175-1199, kStages 2) is NOT the dense lever
(max ~0.3%), only packed (~1.8%). Group 1's softmax alone (1,178 cycles) = FA3's (~1,186): our softmax code is not slower;
the dense loss is the two groups' softmax phases overlapping (shared issue slot + exp unit per SMSP).

**10-19 PM (§45.48 result, §45.49):** SKV = FA3's split K/V rings: 296/296 correct; race dense +0.20% (t 1.5), packed +0.87%
(t 9.1) -> REJECTED. ncu: dense -1.18% cycles (tensor idle -17.8k) but +2.2% instructions (second ring's waits/arrives).
**The forward is POWER-CAPPED at free clocks** (sb_1019: all 16 runs 691-695 W of 700, SM 1.56-1.76 GHz; dense fwd effective
~1.49 GHz ours / ~1.46 GHz FA3; driver sw_power_cap flag active in every sample): energy savings raise the clock (LH),
cycle savings can lower it when the forward runs alone (FA3 -3.8%). On this box ncu did NOT lock one clock (1.63-1.74 GHz).
**FA3 L2 hints (source mainloop:753/767/820 + SASS desc[UR] 0x14f00000/0x12f00000):** K/V EVICT_LAST, Q EVICT_FIRST, O no hint.
Ours none: dense L2 misses 1.30x, DRAM 520 vs 365 MB (packed equal). Work order already = FA3's (G5). Candidates LH (FA3's
hints, fwd_LH.cu 785bd68b) and LHO (+O evict-first, daadb501): SASS checked (LH0 == K5q), run_lh_1019.sh = race + ncu +
clock_power_check (K5q SKV LH LHO fa3) -- RAN 13:49-14:04, results below. Also in FA3, not copied: Q L2 prefetch (mainloop:668).

**10-19 14:05 (§45.50) LH/LHO RESULT + CLOCK CHECK.** Race lh_1019: LH dense -1.18% (8/8), LHO -1.49% (8/8); packed +0.56 /
+0.68% (0/8). ncu: dense cycles UNCHANGED (+0.1%), DRAM 520 -> 447 / 432 MB; packed DRAM unchanged, unit-start K/V wait grows.
Clock check (16 s forward-only runs, sw_power_cap active 100%): LH/LHO clock +0.9-1.0% vs K5q (time -0.44/-0.59%) = an ENERGY
gain. **CORRECTION: SKV's clock is NOT lower (+0.32%)** -- at the real ~1.55 GHz (ncu ~1.68) the K/V wait it removes is small,
its extra instructions cost +0.44% cycles (my earlier "paid back in clock" claim was wrong; fixed inline in §45.48/45.49).
FA3 forward-only: clock -3.8% vs ours, cycles -8.1%, time -4.5%. In the fwd+bwd scoreboard the ratio 1.084 ~ the cycle ratio
(clock set by the mix). In training the step mix sets the clock: cycle cuts convert fully, energy cuts partly.
**LHOd** (cands/fwd_LHOd.cu 8a7b4312, L2_HINT 2 + L2_HINT_DENSE 1): template-flag hints, off in the packed kernel; local SASS
dense == LHO, packed == K5q (0 lines; negative control 756). run_lhod_1019.sh (7321d9ed) = build + box SASS check + quick
suite + fwd+bwd race vs K5q: RAN 14:10-14:17 -> box SASS 0/0 lines, 296/296, dense fwd -0.34% (t -2.9) / bwd -0.33% (t -2.3,
code unchanged: shared clock), packed null. LHOd = new best; adoption = user's call. Next hunt: the ~8% dense CYCLE gap
(proposed: same-box FA3 per-step clock-stamp trace) or FA3's backward L2 hints (Q/dO evict_last, bwd mainloop:512-550).

**10-19 15:27 (§45.51) FA3 TRACED ON THE SAME BOX (the hunt's key measurement).** FA3 measurement copies (fa3_tl2/fa3_tl3 on
the box, patch session_2026_10_19/fa3_tl/, stamps only at sync points: a clock read inside the softmax moves FA3's exps) and
ours TL2/TL3 (= LHOd + identical stamps). Level 2: period FA3 2,367 vs ours 2,551 (+184 = +7.8%); group 0 chain 2,191 vs
2,429. FA3 level 3 perturbed (2,616) -> unused. ncu stall-reason split (deep/loop_wait_split.py; baton wait = "barrier"
samples, the BAR is DEFER_BLOCKING): HGMMA issue equal (~500 both); ours +108 other issue-phase instructions (stage via R2UR,
descriptors per HGMMA), softmax +185 (dependency waits +97, dispatch +100), waits -130. The 45.44 "FA3 tensor-bound step" fit
was WRONG. Laptop CUDA compiles need the glibc 2.43 workaround ([[project_laptop_glibc243_cuda13_break]]).

**10-19 16:20 (§45.52) PROVEN CAUSE #1 (PC sampling, same box):** same instruction counts as FA3 per step (66 exps, 68 FMNMX,
...); the +184 is stalls. Biggest: exps' "wait" +119 (FA3 248, ours 367): exps issued 0-2 instructions after the previous
exp wait 7-9 cycles; ours 25 such vs FA3 11. Source cause: FA3 softmax.h (max = one chain per row; exps row by row; sums
after) lets ptxas run row 0's exps while row 1's max chain runs; ours (4 max chains, both rows' exps mixed, inline sums)
packs the exps. SM3 (cands/fwd_SM3.cu 4ef6ec49) = FA3's structure: local SASS exps 0/13/22/7/23 (FA3 0/11/19/13/22, LHOd
4/21/13/5/22); local SASS == box SASS verified. run_sm3_1019.sh (b4042c5a) WAITING for the user's OK. Still open: F2FP/FMUL
dispatch +72, P.V HGMMA +48, issue-phase instr.
**SM3 RAN 15:52-16:00:** 296/296; exps' wait 367 -> 245 (FA3 247), ncu dense -3.8% cycles. Race fwd+bwd: dense fwd -3.05%
(8/8, t -21), packed fwd +0.45%, unchanged bwd +0.46% (power-shared clock); fwd-only dense -0.70%. Packed loss PROVEN: mask
bitmask (64 LOP3 per diagonal step) from the mixed-row mask loop. SM3m (fwd_SM3m.cu 86a2d73d, FA3 mask.h order): 0 bitmask
LOP3, packed 3,432 instr (LHOd 3,504) -> run_sm3m_1019.sh (00d63d34) WAITING for the user's OK.
**SM3m RAN 16:04-16:10 (§45.52):** 296/296; race fwd+bwd vs LHOd: dense fwd -3.51%, packed fwd -4.48% (t -60), bwd +0.56 / +0.17%
(unchanged code, power-shared clock); f+b dense -0.37%, packed -0.91%. ncu dense -4.0% / packed -6.3% cycles, mask bitmask 0.
Remaining vs FA3 per step: uniform +75 (descriptors per HGMMA, stage R2UR), int +34, baton waits ~+130, arith dispatch +71.
Adoption into the laptop master = user's call (master is still F3c; best chain F3c -> K5q -> LHOd -> SM3m).
**10-19 16:45 (§45.53):** after SM3m the PLAIN STEP = FA3's (2,282 vs 2,299 ncu); the rest is at UNIT BOUNDARIES (+2,560 cycles
per unit): the wait for tile 0's K/V = 1,498 cycles/unit (FA3 50) because our producer waits qempty (both groups' last QK)
before loading Q and then K/V(0); FA3 loads K first (mainloop:769), then QueryEmpty (:776), then Q (:780). PF (fwd_PF.cu
1888246a, KV0_FIRST) = FA3's order; protocol model clean + negative test caught; consumer SASS unchanged. run_pf_1019.sh
(eb7303ae) WAITING for OK. Next after PF: diagonal-last step (2,879) vs FA3's diagonal-first.
**PF RAN 16:39-16:44:** 296/296; vs SM3m dense fwd -2.60% (t -50), packed fwd -3.01% (t -35), bwd +0.2% (shared clock). New best.
**SCOREBOARD sb2_1019 (16:45-16:50, GPU 1, steady):** ours = PF. vs FA3: dense fwd 1.025, bwd 0.973, f+b 0.985; packed fwd 0.866,
bwd 0.955, f+b 0.935. vs FA4 / cuDNN: all < 1. PF ncu dense 1.018x FA3. Remaining per unit: diagonal-last step (+635 over a
plain step), epilogue +383 (FA3 2,059 vs ours 2,442), offset by faster plain steps and unit start. Next hunt: diagonal-first
structure (FA3 walks keys backwards) and the epilogue -- analyse with the same method before any candidate.
**10-19 17:30 (§45.55) remaining dense gap = 741 cycles/unit (PF 41,509 vs FA3 40,768, ncu):** epilogue +383, diagonal-last step
+635 over a plain step (mask ~347, rest waiting moved), start + tile 0 +66, our plain steps -26 each (~-400). Diagonal-first:
NOT proven positive (FA3 pays its mask in its first tile, 174 cycles), big change -> not proposed. Epilogue cause PROVEN (FA3
epilogue_fwd.hpp :250-264/:310-326): FA3 = stmatrix -> fence -> NamedBarrier arrive, one elected lane TMA-stores O, wait read,
release; ours = 8 LDS.128 + 8 STG.E.128 per lane + 2 fences. PFT (cands/fwd_PFT.cu 541a4f5e, TMA_O dense only; packed keeps
per-lane stores, FA3 has no TMA O for varlen either): local SASS dense 2 UTMASTG, 0 LDS/STG.128, 3,720 instr (PF 3,744), no
COLLECTIVE; packed == PF except param offsets. GOTCHA: `if (warp == 0)` made ptxas treat the kernel as divergent (+200 instr,
WARPSYNC.COLLECTIVE) -> use `__shfl_sync(~0u, warp, 0) == 0`. run_pft_1019.sh (6ce228b1, box SASS gate first, negative-tested)
WAITING for the user's OK. Expected <= 0.9% dense cycles.
**PFT RAN 17:08-17:12 (§45.56): REJECTED.** dense fwd +1.60% (t +25, 0/8), packed = (identical kernel); ncu +710 cycles/unit:
epilogue -484 (works) but unit start +668 (each group's warp 0 waits ~630 cycles on DEPBAR = wait_group.read before the slot
release; the next QK needs all 4 warps and the baton couples both groups) and steady loop +590 (+40/step: ptxas re-ordered the
loop; my "neutral" estimate from exp spacing was WRONG). FA3 (epilogue_fwd.hpp :249-327): ONE waiting warp = last warp of the
SECOND warpgroup, 288-thread barrier, one lane stores the whole 128-row tile. Next: PFT2 = that exact arrangement + HARD GATE:
steady-loop SASS identical to PF's (laptop + box build) before any GPU time. Box fully backed up 17:40 (§45.56).
**10-19 19:30 EXACT GAP (§45.57, one accounting for both kernels, deep/regions_fa3_pf.py + waits_fa3_pf.py):** dense fwd 1.025 =
cycles 1.018 (+741/unit) + ~0.3% shared clock (scoreboard nvidia-smi: ours -0.33%, noise larger). Cycles: unit start +366 (Q from
DRAM: FA3 prefetches Q to L2 when its producer STARTS the tile, mainloop :681 -- Q wait 286 vs 103; + 7 runtime divisions in
decode_u right before the Q wait, ~190, hidden until Q is early), diagonal LAST +271 net (FA3 masks its FIRST tile, :1150: mask
+285 there vs +583 in our pipelined last step). NOT causes: loop (2,084 vs 2,090/step), end of unit (+32: my 45.55 "+383" was an
accounting ERROR -- FA3's store-warp region 2302-2323 left out), producer (+68). Fix value: prefetch 0.45-0.7%, decode +0.1-0.45%,
diagonal-first ~0.7%; together ~parity.
**10-19 20:45 (§45.58) LAST CANDIDATES built + proven on the laptop:** patches/make_pfq.py from PF -> QP (Q L2 prefetch, FA3 :681),
QPU (+ producer-decoded unit, no consumer division), QPD (+ diagonal first, FA3 :1150), QPUD (all). QPoff == PF (0 lines). QP's
dense + packed steady loops IDENTICAL to PF's; QPU/QPD/QPUD reorder the loop (dense 143 / 57 / 132 instr out of order vs PFT's
197 that cost +40/step) -> cannot be priced locally -> race all 4 in ONE session (run_qp_1019.sh e039121d, GPU 4, ~30 min, box
SASS gate + negative test). Node PF SASS == Verda's (0 lines). WAITING for the user's OK.
**10-19 20:15 RUNNING on the 8x node, GPU 4 (user started run_qp_1019.sh after build_qp_1019.sh "GATE PASSED" 19:54; node libs PF
e9248d27, QP 46fb93a1, QPU 4956db18, QPD b4e82daf, QPUD 9e49098b; box SASS: QP loops == PF's, the others reordered as on the
laptop). Quick suites: ALL FOUR 296/296 (QPD / QPUD = diagonal first: correct, incl. packed == alone). Quick suites take ~4-5
min each on this node (Verda ~50 s: CPU references on shared CPUs). Results -> ~/gautam_k/sft_v2/runs/qp_1019 + qp_results.tgz;
pull to session_2026_10_19/box_runs/qp_1019, then race_stats.py (session_2026_10_17/deep) + kernel_map per candidate.
**RESULT 10-19 20:42 (§45.59): QPU = NEW BEST dense fwd.** Race vs PF: QPU dense fwd -0.81% (t -3.8, 7/8), packed -0.13% (ns), bwd
+0.24% (shared clock); f+b dense -0.20%. ncu: QPU -1.51% cycles (start -222, its reordered loop 2,065 vs 2,084/step), vs FA3 cycles
+0.31% (PF +1.85%). QP alone: Q wait 275 -> 171 but 0 time gain (divisions then exposed, as predicted). Diagonal-first (QPD/QPUD):
removes 333/unit after the loop but +121 first tile and its reordered loop +19/step -> dense +0.32% / -0.31%; packed -0.43%.
Cycles convert ~half to time (power cap). All 4 passed 296/296. Next: scoreboard on the node (QPU vs FA3/FA4/cuDNN), full suite,
adoption = user's call.
**SCOREBOARD 10-19 21:11 on the 8x node (§45.60), ours = QPU:** vs FA3 dense fwd 1.004 (984.2 vs 980.4 us), bwd 0.968, f+b 0.976;
packed fwd 0.865, f+b 0.933; vs FA4 / cuDNN all < 1 (dense fwd 0.919 / 0.863). Still 11 of 12. Of 1.025 (Verda PF) -> 1.004: QPU
-0.81% (same-node race) + box (node GPU 4 clocks 1-3.6% lower for all; FA3 lost +3.0% time vs ours ~+1.8%) -> PF here ~1.012.
Left in cycles: our masked diagonal step (+583 over a plain step); diagonal-first needs QPU's loop order kept (QPD's was +19/step).

**2026-10-22: colleague Uma Mahesh (Killi Uma Maheswara Rao) mailed "improving the fwd gqa" (21 Oct 23:18) for HIS GQA forward:**
loop rotation (QK(n) + PV(n-1) together), an mbarrier arrive before the PV wait that depends on the softmax row sum + a dummy
zero subtraction on the output, softmax split across the PV wait: -7.4% cycles (1,704,528 -> 1,578,448), tensor busy 77.6%.
These are exactly our K-series / QPU mechanisms (ours: loop rotation 10-16; order_after(rs0) pin + `sub.f32 zero, O_acc[0],
O_acc[0]` + K1_SPLIT 16, 10-17), on our branch dist_side_gautam_lottn since 10-21 ~16:10 (commits 15:58); the user had told him to
refer to our attention changes. His mail does not say where they came from. Speed vs ours not comparable from his numbers
(different kernel / measurement); a fair check = the same scoreboard on the same box.
