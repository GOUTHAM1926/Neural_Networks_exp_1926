---
name: project-state-2026-10-10-v2
description: "STATE end of 10-10 (written 10-12): v2 steps 0-5 DONE (RoPE positions, harness, design B + design A fwd/bwd, sync fix); NEXT = Step 6 benchmarks vs cuDNN/FA3/FA4/FA2/SDPA/Flex, then Step 7 trainer"
metadata:
  type: project
---

Full write-up: work doc section 40 (`work_docs(... )/sft_work_documentation.md`, written 2026-10-12 because the 10-10 session ended before it). Build detail: [[project-v2-build-progress]].

**Done (10-10), all proven (SASS dense/B identical, laptop + box):**
- Step 1 RoPE positions 37/37; Step 2 harness (normal kernels 614/615); Step 3 B fwd G8 51/51; Step 4 B bwd G9 full 404/406 (7 segments = 2 known mechanisms of today's Vj1d, NOT packing); sync fix (+1 BAR, 289/289, no measurable cost); Step 5 A fwd G10 59/59 + A bwd G11 95/95, every segment bitwise = alone.
- Speed fwd+bwd vs dense on real rows: middle B 47.8% / A 45.6%, shortest 28.7% / 28.3%, longest 99.1% / 94.1%.

**Next:** Step 6 = same-session A vs B + cuDNN/FA3/FA4/FA2/SDPA/Flex on the same data, nsys/ncu (why time falls slower than work on short chats). Then choose A/B, then Step 7 (packing script, loader, autograd node, real packed SFT run vs v1).

**Open user decisions:** Step-6 env (proposed: ~/envv read-only + own env for FA3/FA4/TE); leftover rows (proposed: endless stream reshuffled per epoch, val scores all rows); end decisions fp32 P in dS and fp32 head sum for dK/dV.

**State:** nothing committed (user commits). Box ~/gautam_k/v2_work = laptop files; lib has A+B+fix; lib_nofix/ kept; GPU 7 was ours on 10-10 (ask before every run).

**Step 6 first run (2026-10-12, H200 GPU 7, 1980 MHz, 50 rounds, `profiling_reports/v2_step6_benchmarks_2026_10_12/`):** env `~/gautam_k/benv` (FA3 built from fa_src@94e22c9 bf16/hd128/sm90 in 4.5 min, FA4 pip; torch 2.14.1/cuDNN 9.24 reused from envv via .pth). All packed contenders agree with FA3 (bf16 noise).
- fwd+bwd ms, real middle rows: ours A 2.44, B 2.57, FA3 2.10, FA3-det 2.30, FA4 2.16, Flex 3.14, torch varlen (FA2) 3.73, cuDNN-with-mask 14.5. Short rows: ours A 1.49 BEST (FA3 1.60, FA4 1.62). Long rows: ours A 5.07 vs FA3 3.89.
- THE GAP IS OUR BACKWARD on long chats: uniform_4096 bwd ours 4.27 vs FA3 3.01 / FA4 3.02 / cuDNN full-row 3.28 (ours 30-42% slower at 18/6 B=8 T=4096). Forward ~10% behind FA3 on long chats, ties/wins on short. This contradicts the colleagues' "Vj1d beats cuDNN" table (other shapes) -> reconcile at their shapes before claiming anything.
- Noise floor: fa3 and fa3_det run the SAME forward kernel yet differ ~4% (order in the round) -> differences < 5% are not meaningful in this run.

**Step 6 later (2026-10-12 ~12:50-13:25), see `profiling_reports/v2_step6_benchmarks_2026_10_12/STEP6_ANALYSIS.md`:**
- A vs B DECIDED: A = main path (equal/faster almost everywhere).
- 6a dense at colleagues' shapes (13): our fwd beats cuDNN on 11/13; our BWD is SLOWER than cuDNN on all 13 (1.05-1.24x), FA3 1.32-1.42x, FA3-det 1.05-1.17x. Contradicts the colleagues' bwd table (method unknown).
- 6b nsys kernel-level (dense B8 18/6): fwd ours 1075 vs FA3 890 us; bwd main ours 3633 vs FA3 2495 / FA3-det 3283 / FA4 2482 / cuDNN 2836 us.
- 6c ncu: ours bwd 888M instr vs FA3 625M (+42%), smem wavefronts 662M vs 419M (+58%); hot spots = fences before each dQ TMA reduce (~16%) + dQ staging STS; cause = 64-key tiles (2x dQ reduces per key) + P round trip through smem + 1 warpgroup. Fix = FA3 structure (128-key tiles, producer + 2 consumers, P in registers -> also fixes bf16-P precision). Fwd: +17% instr; its long-scoreboard stall is mostly idle producer spin (harmless).
- Traps hit today: ncu 2026.3 writes `.ncu-repz` (not .ncu-rep); `/usr/local/cuda/bin/ncu` is a WRAPPER -> killing its PID leaves the real ncu running (kill the whole tree by PID); status text + tar on one ssh stdout corrupts the tar.
- Coverage audit: added G12 (T 1024/2048/8192/16384, 4096 one-token chats) and G13 (large inputs on packed rows, 60 rows per call). Full suite running as 6 parallel processes in `~/gautam_k/v2_work/logs/suite_1012_1319/` + chain (G13 then FA3 packed ncu).
- **FULL SUITE 2026-10-12 13:19-13:49 ALL PASS** (`profiling_reports/v2_step6_benchmarks_2026_10_12/suite_1012_1319/`): CLASSIC 605/605 (10 fewer than 10-10 = vacuous D6 "chats 0" lines now skipped), G8 223, G9 406, G10 235, G11 411, G12 134, G13 76; known items = exactly the listed ones. Packed ncu: ours A bwd 1466 vs FA3 1086 us (+35% instr, +57% smem). Doc section 41 written. Box clean after. NEXT: user decision on backward rework (FA3 structure) before/after Step 7.
