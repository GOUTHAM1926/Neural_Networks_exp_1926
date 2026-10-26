---
name: project_pingpong_facts_1026
description: "2026-10-26 checked facts on GEMM \"ping-pong\" — ours none, hybrid none (personal repo only), Kota's file dead in remote BLAS master ea666ab, CUTLASS details verified, colleague Harshini/Nova claim reviewed"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-26T10:38:53.591Z
---

**Checked 2026-10-26 (code + local CUTLASS 546c3ef + docs + `git ls-remote`).**

**The remote BLAS.**
- Chain: BluTrain → submodule Tensor-Implementations → submodule BluBridge-BLAS. `.gitmodules` URL is `kathir-23s/BluBridge-BLAS` (the same repo as `BlubridgeAI/BluBridge-BLAS`), branch master.
- Remote master = ea666ab: "Merge PR #3 from BlubridgeAI/production_blas", Kathir, 2026-08-31 14:35 IST. Its commits are by dhanakumarK777 = Kota Dhanakumar.
- Harshini's screenshot shows that same commit.
- There are 15 remote branches (plus 3 PR refs). None is a hybrid or Gautam branch. On every branch that has the ping-pong file (8 of 15), nothing includes it or uses its symbols, and no build file mentions it.
- The token for read-only ls-remote is `~/.config/blubridge/github_pat`, used through a credential helper so it is never printed.

**Kota's `Bgemm_sm90_pingpong_template.cuh`.**
- 680 lines, present at ea666ab. No file includes it.
- It is a real ping-pong design: alternating tiles, 24/240 registers, and mbarriers `mainloop_turn` / `epilogue_turn`.
- Its own header says (corrected 2026-08-24) "tested, failed (HUNG), fixed, NEVER re-run".
- Our hybrid deleted it in 713f0f7 (2026-09-02, §H65, line 26731: "VERDICT RECORDED: none").

**Our module.**
- The template is cooperative only (header lines 31-32).
- §H158 (line 31609) says "We have no Pingpong".
- R3 (line 16762) says ping-pong is dead for V5.
- The hybrid lives only in the personal `GOUTHAM1926/blublas`.

**Old cuBLAS research.**
- §H23d (line 6352) found that untagged nvjet = ping-pong. The evidence was register math (blocks/WG = grid; max R = blocks/WG·wgmmaN/2+26, 12/12), not a SASS opcode.
- §H24 (line 6409) found that the barrier mix is identical in all kernels: 46 BAR.SYNC, no BAR.ARV.
- §H25 (line 6493) is an opcode census of 18 kernels. Only 3 opcodes differ, and none tracks ping-pong vs cooperative.
- §H25 also shows USETMAXREG in 10/18 kernels, in both families, at max R >= 154. This corrects H24's "no SETMAXNREG" (H24 had grepped the PTX name).

**CUTLASS traps.**
- CUTLASS cooperative ALSO uses 40/232.
- The ping-pong order barrier is a `ClusterBarrier` = mbarrier (`sm90_pipeline.hpp:1255-1257`), NOT a named BAR.ARV.
- So "BAR.ARV 256 + USETMAXREG = ping-pong" is not a valid SASS test for CUTLASS or cuBLAS. Real marks: one tile per consumer + the order wait/arrive (pingpong.hpp 849-952).
- The PyTorch blog (Less Wright, Adnan Hoque, 2024-11-01) quotes 40/232 and "cooperative is largely equivalent to Ping-Pong, but both consumer groups work on the same C output tile".

**Our SASS check was blind.**
- H24 counted only BAR.* barriers.
- H25 compared which opcodes are present, not how many.
- CUTLASS's turn barrier is an mbarrier (SYNCS ops), and SYNCS ops are already in every kernel for TMA. So "not found" does not mean "not there".
- Not re-checkable locally: 0 of the 1,597 sm_90 cubins in libcublasLt 13.1 contain "nvjet". It needs an H200 SASS dump and a SYNCS count per kernel.

**Why we did not build it.** It was NOT "we didn't know":
- H17 (line 5054): undone, perf-matching only.
- H27e (line 6787): ping-pong tiles are only 25/163 of cuBLAS's picks.
- Line 9227: priority #4, behind the 10-14% gap.
- R3 and H158 (above).

**Kota's code vs CUTLASS.**
- His SM90 file matches all 3 CUTLASS parts:
  - tiles: 410-412 ↔ 558-562 and 952;
  - MMA turn: 460 and 514 ↔ 850 and 877;
  - epilogue turn: 521 and 600 ↔ 899 and 936.
- One difference: `wgmma_wait_group<0>` every k-step (line 502).
- On B300 (SM103 `cluster_dual_template` 24-31, 129-132), "ping-pong" means EPI_PIPE_DEPTH=2: two TMEM accumulator buffers. Same goal, different mechanism.

**REPLY CAME 2026-10-26** (`Neural_Networks_exp_1926/pingpong_crosscheck_answer_2026-10-26.md`). All numbers recomputed and they match.

Proven, so Nova's GEMM IS ping-pong:
- Owner guard `ordinal % turns == turn` (NovaGPUMapForallToGPU.cpp:418-459 @ac2ab83).
- 192 accumulator registers per consumer thread (whole 128x192 tile), max R215, 0 spills.
- SASS shows `BAR.SYNC R,0x100` / `BAR.ARV 0x100` and USETMAXREG 232/40.
- Cross-check: Nova coop 128x192 uses 122 registers, which equals cuBLAS's 122 for that tile (96+26).
- Not from Kota's file. Its records cite CUTLASS plus the FA3-style flash turn (Mahesh fbda5fe).
- Commits: Harshini's ping-pong is 961e4ff (10-23); c54a485 (10-24) is Aparna's and rewrote the lowering.

It admitted 6 errors, including that "−3.4% step" was really GEMM ncu cycles. The real step result:
- −1.19% (v3, 2 runs/arm; the ping-pong runs alone spread 0.67%);
- later build −0.37% (1 run each), which is noise.

Still open:
- numerics never run on the current (c54a485) lowering; the old check sampled 2 rows per tile;
- no compute-sanitizer run ever;
- per-GEMM A/B mixes the tile change (pp 128x192 vs coop 128x256) with the schedule change;
- barrier ID 1 = both the ping-pong turn and kSplitKExchangeBarrierId, with no guard shown;
- Harshini's BluBLAS reference can only be answered by her.

Nova vs cuBLAS on 10-10 (before ping-pong): 1.27x cycles.

**Harshini's kernel files** (`Neural_Networks_exp_1926/reports_harshini_ping_pong/`: pingpong and coop mlir/ptx/cubin/sass) were checked locally on 2026-10-26.
- My `ptxas` recompile of their PTX gives SASS identical to their cubin.
- All C1-C4 counts reproduce: 168/122 registers, max R215/R119, 16/4 HGMMA, BAR.ARV 2/0.
- The PTX shows the whole mechanism:
  - consumer vs producer split at line 81;
  - w = tid>>7 at line 86;
  - turn IDs r3 = w+1, r4 = 2-w at lines 92-93;
  - tile loop step 132 at line 130;
  - owner test ordinal&1 == w at lines 137-141;
  - hand-off at `rd141==1920` (after waiting step 31 of 36) when t <= 1403.
- **New finding:** Nova's COOPERATIVE kernel waits for all MMAs at every K step (`wgmma.wait_group 0`, coop.ptx:455), while ping-pong keeps 1 group in flight (`wait_group 1`, pingpong.ptx:446/544). So their pp-vs-coop gains mix the schedule, the drain depth and the tile size.
- Caveat: MY cooperative kernel also uses wait_group<0> every K step. My template line 7073 (§H16.7) says a wait_group<1> pipeline REGRESSED on H200, and lines 7121-7131 say cuBLAS also waits to zero, but drains at the top of the next iteration. So wait 0 is not wrong for cooperative. It hurts ping-pong because only one warpgroup issues MMAs during a turn.
- Kota's file also drains every step (line 502). That is the cause Nova recorded for their v1 being 17-20% slower.

**We never built or timed a ping-pong schedule.**
- No commit implements it. 569cdaa's "slot ping-pong" is only an epilogue double buffer.
- R3 is arithmetic only (288 > 232).
- The recorded reason for not building it is H56 (lines 26344-48): on the large shapes we were already at cycle parity with a 92-99% full tensor pipe, and "neither regsplit nor pingpong adds math throughput". Ping-pong was for small tiles / BM=64 (H158, H159).
- It was NOT the cuBLAS SASS: H23d concluded that cuBLAS does use ping-pong.

**DOCUMENTED 2026-10-26:** §H235 in `blublas/docs/analyzations_h200.md` (mirror synced) and §45.95 in `sft_work_documentation.md`, the same full text. Not committed; the user commits.

**Why:** the user must answer colleague Harshini (compiler team, "Nova"). She cites "blublas in remote BluTrain" for her ping-pong.

**Handoff doc for Harshini's Claude:** `work_docs(mark-down - (.md) files ) /pingpong_crosscheck_for_nova_2026_10_26.md` (note the trailing space in the folder name). It has questions A1-G1 and the answer format. When the answers come back, compare them item by item.

**How to apply:**
- Nova's code cannot be read from here.
- Ask her for the code lines that give each consumer its own tile, the turn barrier, and an A/B against her cooperative version.
- Do not accept a register split as proof.
- Related: [[project_multicast_vs_coop]].
