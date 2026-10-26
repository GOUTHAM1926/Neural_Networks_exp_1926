---
name: project-v2-build-progress
description: "v2 (packing + varlen attention) BUILD progress from 2026-10-10: plan = build BOTH A and B as compile-time variants, test all data-load combos, keep the faster or both; Step 1 (RoPE positions) DONE with proofs; box copy ~/gautam_k/v2_work on GPU 7"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-10T12:39:52.369Z
---

**Decision (user, 2026-10-10):**
- Build BOTH designs as compile-time variants inside the existing kernels (v62 forward, Vj1d backward): A = per-chat offsets (cu_seqlens), B = packed row + block mask.
- Test every data-load combination, then keep the faster one, or both (like PyTorch).
- Every change gets a comment: what changed, what it was before, why, evidence. No name tags.
- Proof-first rule: read the reference sources, then our code, then implement. No assumptions.

**Step 1 DONE (2026-10-10): per-token RoPE positions.**
- 11 TI files changed, all uncommitted, laptop `~/Downloads/BluTrain_sft`:
  - `FusedRMSRoPEKernels.h`;
  - `FusedRMSRoPE_{fwd,bwd}_sm90{,_kernel}.cu` (template `HAS_POS`);
  - `Normalizations.h/.cpp` (validation helper `fused_rms_rope_positions_ptr`);
  - `NormalizationOps.h/.cpp`;
  - `NormalizationBackward.h/.cpp` (`positions_` member).
- New test: `Tests/kernel_tests/test_rope_positions.cu` (37 cases).
- **Proofs:**
  - SASS identical for 13/13 original kernels vs the new `HAS_POS=false` instantiations, on laptop CUDA 13.0 sm_90a AND on the box's real objects (CUDA 13.4 sm_90a). The positions-ON versions differ (+16 to +40 instructions, extra LDG), so the comparison is not vacuous.
  - 37/37 PASS on laptop RTX 3060, on H200 GPU 7 (dispatch = sm90), and on H200 with `BLUTRAIN_NO_SM90_QKNORM_ROPE=1` (portable).
- Records: `profiling_reports/v2_step1_rope_positions_2026_10_10/`.
- **Two test mistakes, both fixed with evidence:**
  1. The dgamma tolerance (1e-5) was below the portable kernel's atomicAdd noise; it is now 1e-4 for the atomic path only.
  2. The test inferred the dispatched kernel from the arch alone; it now calls the library's `get_arch()` + `disable_sm90_qknorm_rope()`.

**Box workflow (from 10-10):**
- The user said "you can use GPU 7".
- I work in a SEPARATE copy `~/gautam_k/v2_work` (a `cp -a` of BluTrain_sft, 570 MB) so the user's `_runner_` is untouched: a new signature would break its old library symbols.
- Build TI there with `make -j$(nproc) tensor SM_ARCH=90`. This is exactly BluTrain's own command (`build_sft.log:25`). Plain `make` FAILS on the box: libblublas has no sources there ("nvcc fatal: No input files").
- Copy files with tar over ssh, then compare md5 on both sides.
- Before every run: two `nvidia-smi` samples plus the process list for GPU 7. Pull the logs right after.

**Step 0 DONE (10-10):** the user ran `git ls-remote` on the box: GitHub dist_side_merge = 909ab750 (BluTrain) and d96b991a (TI), the same as our laptop base.

**Next:** Step 2 = attention test harness (reference + all data-load cases + timing), validated first on today's kernels. Then B (fwd, bwd), then A, then the matrix and the decision.

**Step 2 research DONE (10-10), design shown to the user, waiting for OK before writing the file.**
- Source rules:
  - FA3 `hopper/test_flash_attn.py:535-536,579,655-660`: error <= 2x the bf16-PyTorch error + 2x the fp32 rounding error, for out/dq/dk/dv.
  - FA2 `tests/test_flash_attn.py:1440` uses 2x for the output and `:1449-1451` uses 3x for dq/dk/dv.
  - PyTorch `test/test_varlen_attention.py:508-516,583-595` (downloaded to scratchpad): the same 2x rule, applied PER SEQUENCE.
  - TE `test_attention.py:327-336`: pad rows must be exactly 0.
  - FA does NOT gate the attention LSE. The 1e-5 LSE check at FA3 `:1216` is in the split-combine test only.
- Laptop CANNOT run any correct causal attention: on sm_86, causal hd64 falls back to non-causal `gqa_v2` (`GQA_fwd_generic.cu:269-274`, documented there), and causal hd128 throws. Kernel tests are H200-only; the laptop runs only the reference self-test.
- Entry points under test: `gqa_flash_attn_{forward,backward}_qkv` -> v62 (needs T%256) / Vj1d (needs T%64). The trainer allocates O/LSE/dQ/dK/dV with `Tensor::empty`, so the harness poison-fills the output buffers.
- Real data: `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/train`; chat length = last label index + 1. Production = 18/6 heads, hd 128, B=8, T=4096 (from the run.log).

**Step 2 harness WRITTEN (10-10):** `TI/Tests/kernel_tests/test_varlen_attention.cu`, 1564 lines, md5 30c12ae3. Records are in `profiling_reports/v2_step2_attention_harness_2026_10_10/`.
- Groups G1-G7.
- **References:** fp64 truth + PyTorch-bf16 floor, built per segment from tiled GEMMs that sum in index order. All rounding is spelled out because tests compile with `--use_fast_math` (`Makefile:142-143`).
- **Laptop:** G1 PASSES. GPU vs CPU full-row masked loop: truth 1.9e-16 rel; floor bitwise, 0 mismatches.
- **Negative control** (`-DVARLEN_SELFTEST_MUTANT`, plain causal ignoring chat borders): FAILS all 20 checks as it should.
- **Next:** box run on GPU 7 (G2-G7). Needs permission + the box labels path: `VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin`.

**H200 GPU 7 runs (10-10):**
- **Quick run:** 138/142. The 4 failures were 1-token chats (exact dQ = dK = 0, so the 2x rule allows 0). Fixed with a derived fp32 bound (`one_token_region`, u = 2^-23, c = 1.02), approved by the user.
- **Full run** (`G_full_h200_gpu7_1010_1632.log`, 1242 s): **614/615.**
  - 20,480 one-token chats pass at <= 0.5% of the bound.
  - G4: 1000 repeats bitwise for O/LSE/dK/dV.
  - D10: all 5000 chats pass O/LSE/dK/dV.
  - **1 FAIL:** D10 dQ, row 1022, chat@2552, L=1276, ratio 1.156 (err 2.87e-2 vs floor 1.24e-2). The next worst dQ over 89 lines is 0.812. dV runs close to the limit too: worst 0.956.
- **Hypothesis (not yet proven):** Vj1d's D = rowsum(dO * O_bf16) gives a coherent per-row dQ error -scale*dD*sum_j P_ij K_j. FlashAttention computes D the same way.
- **Diagnostic added, not yet run on the box:** `VARLEN_ONLY=D10 VARLEN_D10_FIRST=1016 VARLEN_D10_ROWS=8 VARLEN_DIAG_ROW=1022 VARLEN_DIAG_T0=2552`. It does 50 re-runs, finds the worst element, and fits cos(error, predicted). Percentiles were added to every check. File md5 733e08e6.
- **G7 at free clocks** varied 1.5-3% between the quick and full runs (18/6 fwd 1.14 vs 1.16 ms), so speed claims need same-session ABBA.

**STEP 2 DONE (10-10 ~17:20).** Harness md5 c464c762.
- **KNOWN list.** The user decided to KEEP Vj1d as is and observe the bf16-P issue in all G and B runs, deciding at the end. The harness's `kKnown` matches only row 1022 chat@2552 dQ, ratio <= 1.156*1.01; everything else fails as NEW. Verified on GPU 7: "ALL RUN CHECKS PASS (known issues observed: 1)".
- **D10 percentiles (5000 chats):**
  - dQ p99 0.68, max 1.156 (1 chat);
  - dV p99 0.74, max 0.956;
  - dK max 0.83; O max 0.54.
- **B1 env check (read-only):** system python has nothing. `~/envv` (shared operator user, made 10-03, owner unknown) has torch 2.14.1+cu130, cuDNN 9.24, triton 3.8, flex_attention and `torch.nn.attention.varlen.varlen_attn`. NO FA2, FA3, FA4, TE, xformers or flashinfer.
- **Proposal pending:** use envv read-only for B2, and build our own `~/gautam_k/benv` for FA3/FA4/TE.
- **BOX PAUSED 10-10 ~17:20:** a colleague is using all 16 GPUs. Do NOT run anything until the user says go. Nothing of ours is running.

**Cross-library run on the failing chat (10-10 18:07, `crosslib_row1022_1010_1807.log`):**
- **dQ ratio by backend:** ours 1.156 FAIL; cuDNN 0.527; FA2 0.450; memory-efficient 0.637; math 0.309; Flex 0.450. All others pass.
- **dQ at the worst element** (truth 3.674419): ours 3.703125 (2 bf16 steps off); cuDNN and mem-eff 3.6875 (1 step); FA2, math and Flex 3.671875 (correct).
- **O, dK, dV are at parity with everyone:** O 0.272 for all; dV 0.479 for ours, cuDNN, FA2 and mem-eff.
- **Source check:** FA2, FA3, FA4 and Flex use fp32 P in dS (`flash_bwd_kernel.h:591`; `mainloop_bwd...:893`; `flash_bwd_sm90.py:1569`; `flex_backwards.py.jinja:541/775`). PyTorch mem-efficient uses bf16 P like us (`kernel_backward.h:341-351`, `:1959-1960`) yet PASSED this chat. So one chat does NOT prove cuDNN's internal choice.
- **Offered, not run:** a decisive statistical test (thousands of 2-token chats, large dO/V; does cuDNN cluster with fp32-P or bf16-P kernels?).
- **Tool:** `crosslib_chat_check.py` (+ harness export `VARLEN_EXPORT_DIR`) is reusable for B-series per-chat checks.

**User added (10-10): full benchmark track A-Z vs ALL libraries.** It covers the cloned ones and others, kernel vs kernel: standalone timing, nsys, ncu, GPU power/clock behaviour, and memory. Gaps become the optimization backlog.
- Attention competitors: cuDNN (primary), FA3, FA4 (CuTe-DSL, has sm90 fwd/bwd + varlen + GQA), FA2, PyTorch SDPA/Flex/varlen_attn.
- Liger has NO standard flash attention; it is a competitor only for RoPE, RMSNorm, SwiGLU and CE.

**10-10 later: the user OK'd the Step 2 harness with research additions 1-12.** The additions are repeat x1000, large inputs, tiny chats, offsets, zero-length entries, pad segments, empty-row contract, rejections, MQA, refusal in non-varlen paths, tile-count check, and the FA FLOP formula.

**New plan item (proposed, waiting for the user's OK): head-to-head vs FA3/FA2/cuDNN/Flex on the same H200.**
- The harness exports the exact input bytes plus the fp64 truth, and a Python script runs the other libraries on them.
- FA3 = `flash_attn_interface` built from `hopper/`. Build bf16 hdim128 sm90 only, but keep ALL features for that config so FA3 picks its normal kernel.
- Kathir's `sft.py` uses torch, so a box env likely exists. Checking it is a box action, so ask first.
- No record exists that v62/Vj1d was ever compared with FA3 or cuDNN.

**Colleague's dense-attention benchmarks vs cuDNN (screenshots 10-10, H200, S=4096, D=128, causal):**
- Shapes: 12 total, B=2/4/8 x 12/4, 16/4, 24/8, 32/8 heads.
- **Backward:** Vj1d beats cuDNN on all 12, by +0.5% to +6.2% (their % = time saved vs cuDNN).
- **Forward:** v62 vs "sdpa" wins 8 and LOSES 4: 8x16 -0.9%, 8x24 -1.7%, 4x32 -0.4%, 8x32 -2.1%. So it is weak at the largest workloads.
- **Forward + backward:** we are faster on all 12, by +0.6% to +6.6%.
- **Checks:**
  - TF/s re-derived with FA's formula matches 11/12 rows; the 4x32 row prints 561.88 but the arithmetic gives 564.2.
  - The v62 header says 0.7075 ms at 8x12/4 but the table says 0.7403 ms, so different sessions differ ~4.6%. Never mix sessions.
- **Unknown:** whether "sdpa" was cuDNN or FA2.
- Our SFT shape 18/6 B=8 is NOT measured; the nearest shapes lose on forward.
- cuDNN is the primary bar: TE prefers FusedAttention (cuDNN) over FA on Hopper (`utils.py:1922-1928`), and Megatron builds on TE.

**Data-side analysis (10-10), 3 user decisions PENDING (the harness one is now answered):**
1. OK to write the Step 2 harness?
2. Write the packing prep script now, with Step 2? It writes a new folder; v1 stays untouched.
3. Leftover rows?

Facts behind them:
- **Packing:** whole chats only, first-fit decreasing, giving 1,517 rows (minimum possible is 1,511) and 1–12 chats/row. Shuffle the packed rows afterwards. This matches TRL's default "bfd".
- **Border labels:** ours are safe by design. Labels are shifted inside each chat before packing, same as torchtitan `text_datasets.py:221-224`.
  - HF and TRL instead mask the first label of every chat with -100 (`data_collator.py:1451`, `sft_trainer.py:516`).
  - Megatron relies on prompt masking, which fails in its "default" prompt format (`sft_tokenizer.py:178-181`).
  - Planned proof: every packed chat slice must equal its v1 row bit for bit, and the total stays 4,771,516 trained labels.
- **Leftover rows:** the loader never reshuffles (`SftDataLoader.h:109`), so the SAME rows are dropped every epoch.
  - Packed data: 5 rows (~16 chats) dropped at W=1; 13 rows at W=2 and W=4. Unshuffled, those would be the 47/119 short chats.
  - v1 already drops 8 train rows at W=2/4.
  - Val scored 496 rows at W=1/2 but 480 at W=4, so the 10-07 cross-GPU val losses used different row sets.
  - Recommended: train = endless stream + reshuffle per epoch (seed+epoch), like Megatron/torchtitan/PyTorch; val = all rows, last batch filled with all-pad rows.
- **A/B is not GQA- or FA3-specific.** It is about the token axis. HF picks A on FlashAttention backends and B on SDPA/eager/Flex for the same model. Linear attention (KDA) supports A only.

**Step 3 (design B forward) CODE DONE 10-10, GPU test pending.** `gqa_v62_sm90<PACKED>` + `seg_start` [B,T] int32 (last kernel param). Dense SASS identical 2304/2304 (laptop CUDA 13.0). Harness G8 (md5 at the time 8abad240). Records `profiling_reports/v2_step3_design_B_forward_2026_10_10/`.

**Step 4 (design B backward) CODE DONE 10-10 evening, laptop only, GPU test pending.**
- `gqa_bwd_vj1d<Br,Bc,D,PACKED>`: trip count per key tile = prefix count of `seg_start[q*64] <= kt*64+63` via `__syncthreads_count` (proof in the code comment); masked path when `q==kt || sSS[63] > k_row0`; `fused_p_stsm<Bc,MASK,SEG>` hides keys `< seg_start(row)`; `sSS[64]` rides with sLSE/sD in `issue()`; extra `__syncthreads()` after `issue(0)` (packed only).
- API: `gqa_flash_attn_backward_qkv(..., seg_start=nullptr)`; throws (message names seg_start) unless Hopper Vj1d path (not USE_FLASH / USE_DETERMINISTIC_DQ, T%64, hd128, causal).
- Proof: dense SASS IDENTICAL 1624/1624, helpers identical; packed 236 regs / 115472 B smem / 0 spill = still 2 CTAs/SM. Laptop lib build exit 0.
- Harness G9 added (md5 8acb3d26): per-segment dQ/dK/dV with random dO incl. pad tails, 1-token bound (one_token_region got a tk0 offset), tile-aligned = alone, isolation incl. dO, refusals, repeat x1000 (quick 50), tile-visit model + timing. Compiles sm_86 + sm_90a.
- FIXED a vacuous test: G8/G9 refusals now need "seg_start" in the message (T=320 would throw anyway on the generic path).
- FOUND (not fixed, user's call): dense Vj1d AND Vj1 read tile-0 `sLSE` with no barrier after `issue(0)` (SASS STS 0x1010 -> LDS 0x1ee0, no BAR). v44 is correct (producer_sync + d_ready). Fix = 1 `__syncthreads()` per CTA; changes the default kernel -> needs G re-run.
- Records: `profiling_reports/v2_step4_design_B_backward_2026_10_10/`.
- NEXT on box (needs go): copy 6 files + md5, build, SASS on CUDA 13.4, GPU 7: `VARLEN_QUICK=1 VARLEN_ONLY=G8` (~40 s est.), `VARLEN_ONLY=G9` (~1 min est.), 2 mode-refusal runs (seconds). Estimates from the measured quick run (230 s for G1-G7).

**H200 GPU 7 results 10-10 ~19:26-19:34 (box build 16 s, box SASS dense identical: fwd 2280/2280, bwd 1600/1600 on CUDA 13.4 objects):**
- G8 quick 51/51 PASS (47 s). G9 quick 93/94 (52 s); refusal runs USE_FLASH / USE_DETERMINISTIC_DQ PASS.
- The 1 FAIL = G9 D8 dQ chat row 1 @664 L=303 ratio 1.049. DIAG proved it is the KNOWN bf16-P issue, NOT packing: bf16-P model reproduces the kernel row exactly; fp32-P 0.74x floor; the SAME chat alone through today's dense kernels = 1.049 in 50/50 runs. Added to kKnown (harness md5 94764c14, laptop only so far).
- G9 timing D10 rows 0-7 (one session, 20 samples, free clocks): fwd packed 102.9% of dense with 93.0% of the tile work; bwd 98.2% with 92.5% -> per-tile cost of the packed path is higher; investigate in the B-series. D8 packed bwd work = 26.8% of dense.
- My duration estimate for the diag run was 20 s; it took 59 s (100 reruns). Estimate diag reruns properly next time.

**10-10 ~19:39-20:04 (GPU 7), all logs in `profiling_reports/v2_step4_design_B_backward_2026_10_10/` (+ `racefix/`):**
- G9 FULL (574 s): 404/406. The 2 fails = 3 dQ + 4 dV segments on real rows. DIAG (`G9_diag7_*`): for ALL 7 the same segment alone through today's dense kernels gives the SAME ratio (3 decimals) -> not packing. dQ = bf16-P mechanism (model 0 exact). dV = NEW 2nd mechanism of today's Vj1d: per-head bf16 dV partial before the fixed-order head sum (model A exact; fp32 head sum -> floor-level error, e.g. 6.3e-2 -> 9.1e-4). All 7 added to kKnown (harness 68adac3d, laptop only; box has 608624da). End-decision list now: fp32 P in dS (dQ) AND fp32 head sum for dK/dV partials.
- RACE FIX APPLIED (user OK): `__syncthreads()` after `issue(0)` in Vj1d (both instantiations) and Vj1. SASS laptop+box: +1 BAR.SYNC only (zeroing of R98-R103 reordered); packed identical. G quick with fix: 289/289 PASS, G4 x1000 bitwise. A/B (12 rounds, lib_nofix 089f18ef vs lib aee7d24a, ldd-verified): 12/4 bwd -0.04% t=-0.06; 18/6 bwd +1.71% t=2.01 BUT the untouched fwd null shows +1.39% t=2.02 -> overhead not measurable (<~2%).
- Vj1 (USE_FLASH) through the harness for the first time: 183/196; fails = bf16 atomic head-sum (non-deterministic dK/dV, 1 bf16 step) + extra rounding (MQA 8/1 dK 1.078, D5/D6 dV). Harness TODO: in non-Vj1d modes gate G4/G6c dK/dV with a dK/dV-based atol (currently the dQ atol, too tight).
- Box: libtensor = WITH race fix (aee7d24a); lib_nofix/ kept for A/B; backups in backup_pre_step3_4_1010/, backup_pre_racefix_1010/.

**Step 5 (design A) — research + forward DONE 10-10 ~20:40.** User insisted A is the standard (correct): TRL bfd/padding-free + HF FA (cu_seqlens), Megatron PackedSeqParams, TE thd = A; HF SDPA/Flex + torchtitan Flex = B; torchtitan has both. Lesson: I wrongly said "A not needed" — never call a method unnecessary without the framework evidence first.
- Design (FA2/FA3 method in our kernels): GqaChatPlan (host-built per batch: segments incl. pad tails must cover every row, builder throws), fwd units {b,t0,L,(cpl<<8)|hq} LPT-ordered, bwd tiles {b,t0,L,kt}. v62 template bool PACKED -> int MODE (0 dense, 1 B, 2 A); t0 offsets, segment-relative masks, rows past L not stored, q-tile past L -> my_nkv=0 (inactive-tail path, first time used).
- Proof: SASS dense 2304/2304 vs ORIGINAL v62 and B 2448/2448 vs G9-tested, laptop + box (2280/2432). G10 quick 59/59 (65 s): EVERY segment A == alone dense BITWISE (O/LSE), refusals 10/10.
- Forward time (8 real rows, free clocks): rows 0-7 B 101.9% / A 97.6% of dense; rows 752-759 B 49.4% / A 46.5%; rows 1504-1511 B 33.6% / A 28.5%. A beats B everywhere; time/work ratio grows with many short segments (per-unit fixed cost) -> later optimization.
- Harness 04980e3d (box = laptop). Records `profiling_reports/v2_step5_design_A_2026_10_10/`. NEXT: A backward (Vj1d + plan tile list: LSE=+inf/D=0 rows past L, tail-tile row-checked dK/dV store), then G11.
