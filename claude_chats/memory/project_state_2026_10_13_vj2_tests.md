---
name: project_state_2026_10_13_vj2_tests
description: "10-13: fixed Vj2 (bbd977be) 1a 144/144 + math identical, 1b sanitizer clean, 2a 18-22% faster than Vj1d, 1.05-1.15x FA3, beats cuDNN 12/13; 2b done (2761 vs FA3 2486 us), 3 = 605/605 identical errors"
metadata:
  type: project
---

Fixed Vj2 (GQA_bwd_vj2_sm90.cu md5 bbd977be, box lib d2659657, tva_vj2 a68c366b), GPU 7. One test at a time, user's OK per test.

**Done 10-13 (logs on laptop `profiling_reports/v2_bwd_rework_research_2026_10_12/gpu_runs/vj2c_1013/`):**
- 1a quick CLASSIC: 144/144; all 88 error lines IDENTICAL to the first Vj2 (10-12 smoke), so the fix changed no math.
- 1b compute-sanitizer on `sanitize_vj2` (B1 6/2 T256): memcheck 0, synccheck 0, racecheck 0 hazards. nsys proved `gqa_bwd_vj2_kernel` really ran. Racecheck can't see TMA/wgmma smem accesses.
- 2a same-session dense speed, 13 shapes: Vj2/Vj1d 0.78-0.82 (13/13 pass the >=10% gate); Vj2/FA3 1.05-1.15 (mean 1.11); faster than cuDNN on 12/13 (B8 24/8 1.005); faster than FA3-det 13/13 (FA3-det also makes dQ bitwise, ours doesn't). Table: `vj2c_1013/2a/bwd_table.txt`.

**2b DONE 15:09-15:14** (nsys Vj1d/Vj2 from 14:36/14:38 + FA3 + ncu): main kernel B8 18/6 Vj1d 3623 / first Vj2 3197 / fixed Vj2 2761 / FA3 2486 us. ncu: instr 1,096M -> 840M (FA3 625M), generic LD/ST -> 0, tensor busy 51.4 -> 59.6%. Next leads: dQ-staging STS bank conflicts 2.34x ideal; LSE LDS loaded too late (short_sb 10%, 2 FFMA lines); dQ hand-off fence 4% + bar 3%. Record: `gpu_runs/vj2c_1013/RESULTS.md`.

**Test 3 DONE 15:17-15:39:** 605/605, all 435 error lines identical to the first Vj2 full run; G4 dQ max diff unchanged, more elements vary (B8 666 -> 4025).

**Phase 1.5 (user chose: fix first, one at a time, cumulative; analysis `gpu_runs/vj2c_1013/PHASE15_ANALYSIS.md`):**
- Box snapshots `~/gautam_k/v2_work/snapshots/{v2_bbd977be,f1}/lib/libtensor.so`; tests pick one via LD_LIBRARY_PATH (tva_vj2 has RUNPATH).
- Fix 1 DONE (dQ staging fragment order + 1-D bulk reduce, FA3 scheme; Vj2 file a4f7d5f6, lib 84c91617): 144/144 identical errors; same-session 5.7% faster (13/13); main kernel 2761 -> 2582 us = 1.039x FA3; tensor busy 65.2%. LSE wait NOT fixed by it (10.4%).
- Fix 2 (LSE load order) NULL (1.003x, LSE wait unchanged even with loads 70 instr earlier) -> REVERTED.
- Fix 4 (FA3 LPT block order, 8-pair L2 sections) KEPT: mean 0.967x vs f1, B2 -5..-10%, B8 ~flat (DRAM +4.5%); SM busy 98.8%.
- Fix 5 (epilogue SW128 staging + SWIZZLE_128B dK/dV store maps) KEPT: 0.989x vs f4, 12/13; store conflicts 1.79M = FA3.
- 4-pair sections (f4s4) tried, NOT kept (0.994x, B8 flat).
- FINAL = snapshot f5 (Vj2 file 826ae057, lib a07ca28b) = laptop file now; ncu B8 18/6 2.545 ms (~1.02x FA3 cross-session). Full 605 on f5: 605/605, all 435 errors identical (17:14-17:37); dQ run-to-run max diff 4.9e-4 -> 2.0e-3 (block order), atol 1.56e-2.
- Source copies: `code_snapshot/GQA_bwd_vj2_sm90.{f1_a4f7d5f6,f2_ab6e5cbb_reverted,f4_22c0a1e5,f5_826ae057}.cu`.

**Earlier option list:** decide: fix the 3 leads, or Phase 2 (A/B on Vj2). Related: [[project_bwd_rework_research_1012]], [[project_remind_1013_morning]].
