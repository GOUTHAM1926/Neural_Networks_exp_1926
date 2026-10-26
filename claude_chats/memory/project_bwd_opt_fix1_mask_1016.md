---
name: project_bwd_opt_fix1_mask_1016
description: "10-16 backward profile verdict (helpers equal, we lose only on clock = more instructions) + fix 1 mask block, A/B pending"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-16T13:25:24.695Z
---

2026-10-16 backward profile (work doc §45.13, records profiling_reports/bwd_scoreboard_2026_10_16/prof_1016 + bwd_opt_2026_10_16/):
- Helper kernels equal for all (ours 369-387 us, FA3/FA4/cuDNN 366-394). Gap = main kernel only.
- ncu locked 1.36 GHz: our main kernel has FEWEST cycles (dense 4.14M vs FA3 4.25, FA4 4.29; packed 1.73 vs 1.87/1.88).
- Free clocks we lose (dense 2633 vs 2551/2524 us): we run ~1575 MHz vs ~1665-1700 because we issue more instructions
  (705M vs FA3 626M, FA4 555M) -> more power under 700 W cap. Same story as GEMM [[project_power_clock_solved]].
- Fix 1: causal mask ran on EVERY step (predicated `if (diag)` inside unrolled loop; FSEL 38.9M = one per EX2). Now a
  separate block behind one branch (73 instr skipped per unmasked warp-step, ~83M fewer). FA3-style two-loop split SPILLED
  128 B/step -> dropped. Source md5 1b76873c; box libs: libs/vj2_old (2464c5b8) vs libs/vj2_maskblk (3b998bb6).
- Run 1 (14:06-14:13 IST): 296/296 pass; instr 705->617M (-12.5%, below FA3) but cycles +0.3% dense / +1.4% packed;
  race dense -0.08% (null), PACKED +0.58% 12/12 (diag steps slower: mask no longer hides in exp latency; packed ~30-40%
  diag steps). Fwd after it -0.4% (power). Not kept. Kernel is NOT issue-bound: instr cuts convert at zero again.
- Version 3 (md5 cc3a6125): two exp copies behind one branch (diag = old interleaved code). Test: ab_mask_1016.sh <gpu>
  vj2_maskv3 ab_maskv3_1016. Keep only if neither dense nor packed slower.
- Run 2 (V3, 14:19-14:26 IST): 296/296; instr -12.4%; bwd dense +0.11% (t 1.8), packed +0.22% (t 4.8); fwd after it
  -0.30/-0.62% -> per fwd+bwd pair a wash. Box lib/libtensor.so = V3 (f3557691). Keep-or-revert = USER'S CALL (pending).
- USER KEPT V3 (14:40 IST), moved to FORWARD. Fwd gap is real at same clock (ncu 1084 vs FA3 877 us, tensor busy 64 vs
  83%): FA3 runs softmax(n) WHILE P.V(n-1) is on the tensor cores; ours waits both GEMMs then softmax. Plan G1->F1->G5->G6.
- G1 fwd (P.V 8x m64n128k16, LBO 16384 B; fwd md5 86da0e1f, lib vj2_fwdg1 adce5abb): fwd -3.34% dense / -3.22% packed
  12/12, 296/296. KEPT. Next F1 = FA3 loop order (check spills: +32 regs live during softmax).
- F1 (FA3 loop order, fwd 79093fe2, lib a545b028): fwd -0.50% dense / -1.82% packed (gain = removed redundant QK;
  dense cycles unchanged). G5 (dynamic work order + 1 producer warp, fwd 8e3f54ca, lib vj2_fwdg5 25596273): fwd -2.63% /
  -3.55%, but dense L2 hit 92->83%, DRAM 254->526 MB (section 48 heads too big) -> next knob: section size 12/24.
- FWD TOTAL 10-16: dense -6.4%, packed -8.3%; locked-clock dense 1.226x FA3 (was 1.338x), = cuDNN, FA4 1.114x.
- DEEP ANALYSIS (§45.15, tools profiling_reports/fwd_deep_2026_10_16/): softmax same instr as FA3 but bad ORDER (vote+branch
  conditional O rescale -> exp runs + 29 F2FP in a row) + descriptors via R2UR (stage not uniform). F2 = always rescale +
  shfl-broadcast unit & stage (fwd b6c054a8, lib vj2_fwdf2 0e992f37): fwd -3.84% dense / -1.54% packed. KEPT.
- FWD after F2: dense -10.0%, packed -9.8% today; 1.18x FA3 locked. Left: unit end (64 STG, G6) +0.041M, unit start
  hand-over chain +0.051M, softmax exp latency +0.086M. Next = unit boundaries as ONE change (G6 + prefetch next unit).
- F3 (FA3 hand-over + 24/240) hung in packed (inactive group fetched before its tail; fixed F3b, proven by
  profiling_reports/fwd_deep_2026_10_16/protocol_sim.py). F3b: dense fwd +1.11% WORSE, packed -0.69%. F3c (40/232) untested.
- CROSS-CHECK 10-16 eve (§45.16): the 12:19 scoreboard compared our C++ bench (kernel+10us) with libs' PYTHON bench
  (kernel+83..324us) -> flattered us. Kernel-to-kernel free clocks: bwd DENSE ours +2.5% vs FA3 (not a tie), packed tie;
  fwd now est. dense ~1.24x FA3, packed ~1.15x. Dense fwd DRAM 524MB vs FA3 367 (FA3 uses EVICT_LAST K/V, EVICT_FIRST Q).
  Plan E1 L2 hints, E2 unit end (TMA store / 16B stores), E3 decoded hand-over, E4 softmax before PV wait, F3c; batch them.
- Power explanation NOT confirmed (fewer instr than FA3, still not faster). Untested alternative: memory-latency share
  does not scale with clock (ncu --clock-control none, or 5 s loops + nvidia-smi per contender).
- Next candidates: UMOV+UIADD3 163M (descriptor setup); FA4's 555M; libs' SASS CSVs still on the box.

**Why:** the "ncu faster but free-clock tie" question is answered: power cap, not helpers.
**How to apply:** judge every backward cut by instruction count AND free-clock time (>=4 s runs), not ncu cycles alone.
