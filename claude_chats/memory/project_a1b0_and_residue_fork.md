---
name: project_a1b0_and_residue_fork
description: 2026-09-03 hybrid findings — A1B0 (alpha==1,beta==0) was built but switched off everywhere, now wired at launch and flips vector-instruction lead to 13/13 tiles; his residue design is padding-at-entry (complete, never fires in production), ours is in-kernel; RESIDUE_FREE launch-pick proposed, not decided
metadata:
  type: project
---

**A1B0 — the biggest single finding of the hybrid comparison (§H73).** Our `EPI_A1B0` (slot 37) removes the whole alpha multiply and C read-back and has been correct since §H27zzn — but it was a CONFIG slot whose precondition (alpha==1 && beta==0) no dispatcher can know, so 2 of 505 configs set it and ZERO dispatch rows reached one. Every shipped kernel carried a dead beta epilogue (~382 instr on 128x256x64). FIX (done, not committed): third kernel template param `A1B0_FORCE`, launcher compiles both and picks on `alpha == 1.0f && beta == 0.0f` — his `HasBeta` WIRING, not his code; ours also folds alpha. PROVEN: 477 general kernels byte-identical; fast kernel drops FMUL 256->0, FFMA 128->0, LDG 64->0, PRMT 64->0. RESULT: vector-instruction lead went from 2/13 tiles to **13/13** (his beta-false vs our A1B0, leanest each). COST: TU compile 245 s (~2x), 481->956 kernels. CAVEATS: static counts not speed; leanest-vs-leanest, the honest next check is DISPATCHED-vs-dispatched; TN TU only; nothing on hardware.

**Residue / padding — a design fork, NOT a bug (§H76).** All four of his TMA descriptors truncate a dim by `/ swizzle_w` (64) — the exact form our §H16.15 retired. He is safe because the three `dispatch_*` entry points PAD on the host (`N_padded`, `K_padded`, TN `M_padded` to 128) and route through `dispatch_padded_helper` = 3 sync cudaMalloc + 3 sync cudaFree + 3-4 memcpy2D per call. Gating is COMPLETE. The `sm90_tile_aligned`/`sm90_m_aligned` functions I once cited as his gate are DEAD CODE. Padding fires on **0 of 15 (1.7B), 0 of 18 (600M), 0 of 17 (4B)** production shapes — every M,N,K is a multiple of 64. So for the TRAINER his lean kernel is strictly better; for a GENERAL BLAS ours is. PROPOSAL (not decided): `RESIDUE_FREE` kernel template param picked at launch when M%BM==0 && N%BN==0 && K%BK==0 — same trick as A1B0. Note the 6-vs-2 TMA-per-k-step gap is the 3-D descriptor, not residue.

**DO NOT adopt his 4-D MN-major descriptor** even though it is 1 TMA vs our per-atom loop (§H74): it is the truncating form, CuTe never expresses the swizzle atom as a gmem dim, and it constrains MN to %64 where ours needs %8.

**2WG (§H74):** his always-M-split gives 1 atom/warpgroup on EVERY tile and threads grow to 768 (85 regs/thread at BM=320); `_2WG` on 256x128 just climbs back to our default. Knob still worth adding for **NCW=1** (BM=128 -> 2 atoms/WG, unlocks SKEW on 128-tiles). Earlier "51 of 53 blocked at BM>=256" was wrong twice — BM=192 was never blocked, and more warpgroups makes it worse.

**Method lesson, 4 self-corrections today:** opcode histograms said where to look; only reading code said why. "Port his HasBeta" (we had it), "his += desc is closer to cuBLAS" (SASS: our R2UR 10 vs his 37), "adopt his 4-D descriptor" (it's the retired bug form), "his gate is sm90_m_aligned" (dead code). See [[feedback_verify_claims_and_benchmark]], [[feedback_samples_not_static_counts]].

Related: [[project_hybrid_feature_verdicts]], [[project_warpgroup_knob_scope]], [[project_splitk_ours_vs_his]].
