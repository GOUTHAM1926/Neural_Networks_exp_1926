---
name: project_matmul_epilogue_audit_1024
description: "FIRST TASK MONDAY 2026-10-26 — matmul audit of the production pretraining trainer (C1-C7, \"use the fp32 accumulator directly\"); full text in sft_work_documentation.md §45.93"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-24T16:24:24.738Z
---

**The user decided (10-24 night):** this is the FIRST task on Monday morning, 2026-10-26, the immediate future task. Start there before anything else.

Full answer text, kept as the user asked: `work_docs(mark-down - (.md) files )/sft_work_documentation.md` §45.93. Note that the folder name ends with a space.

**Trainer:** lottn `train/production_zero/bluscript_zero_ckpt.cpp` with the production config:
- `BLU_ALL_BF16=1` + `BLU_BF16_LOGITS=1` (residual, logits and embedding all bf16, so every `to_dtype` around the GEMMs is a no-op);
- `grad_view=true`, `fp32_reduce=false` (micro-step grads accumulate in the bf16 wire buffer);
- Muon, ZeRO v2, BluBLAS.

**Library rule:**
- `matmul` output dtype = the input dtype (GenMatmul.cpp).
- The BluBLAS bf16 API has NO fp32-output GEMM.

**Already fused (proof):**
- Attention v62 forward: S, P and O stay in registers (GQA_fwd_sm90.cu:924-996, 1243).
- Vj2: S, dP and dQ are fine.
- Muon Newton-Schulz: `cuda_addmm` → BluBLAS SM90 addmm epilogue with a full-matrix bias.
- F2 (AutogradMeta.cpp:177-193) is a separate widen+add kernel, not an epilogue, and it is unused under grad_view.

**Exists but unwired:** the BluBLAS SM90 GEMM+SwiGLU epilogue (no TI caller; `cuda_matmul_swiglu` is gone; 08-14 NaN history).

**Open (estimates per micro-step):**
- **C1** dW beta=1 epilogue: −5.9 GB.
- **C2** fused QKV: −23.6 GB (caveat: Muon is per matrix).
- **C3** residual add in the epilogue: −15.7 GB, and it allows an fp32 residual at bf16 cost (addmm M·N bias; needs a 2-D flatten and an AddmmBackward check).
- **C4** lm_head+CE fused: 3.3 GB of logits.
- **C5** wire the SwiGLU epilogue.
- **C6** Vj2 dK/dV fp32 partials: +15.7 GB (a precision trade).
- **C7** Muon: the last NS GEMM writes fp32.


**SECOND PASS 2026-10-26 (work doc §45.96).** Read lottn trainer 3b80f15 + TI b1443a5; this is the newest code (the team's BLUCCL pin 02d0d185 is older).

Ranked scopes (estimated GB per micro-step):
- C2 fused QKV 23.6 (add kernels; RoPE untouched via a split-pointer epilogue, tiles line up at 2304/3072). The compiler team did a packed wQKV only in their Nova JIT script.
- C5 SwiGLU fwd in the gate_up epilogue: 20.9.
- C5b SwiGLU bwd in the down-dX epilogue (DGELU-style): 20.9.
- C4 lm_head+CE: 19.8 + 3.3 GB peak memory.
- C3 residual in the epilogue: 15.7 (also allows an fp32 residual).
- A2 wO head-major O/dO via rank-4 TMA: 15.7.
- C1 dW beta=1: 6.4. MatmulBackward.cu:647 hard-codes beta 0; the SM90 BLAS supports beta.
- A1 v head-major: 5.2.
- L1 lm_head fwd transpose copy: 0.5. cuda_matmul already takes OP_T; the trainer comment at :798-803 is outdated.
- L2 wte grad layout: 0.5.
- Muon, per optimizer step: M1 fp32-out last NS GEMM 5.9; M2 no transposes for tall matrices 7.4; M3 symmetric XX^T / AA.

Not matmul: the residual-grad engine adds, 23.6.

bf16-in/fp32-out is really needed only for C1-fp32 (Megatron main_grad), C3-fp32 residual and M1.

RoPE: list it, never change it (user 10-26).

Next: nsys on 95.133.254.29, GPU 0.


**MEASURED 2026-10-26 20:01-20:35 IST (work doc §45.97).** nsys on 2x H200 95.133.254.29 GPU 0; world 1; 71 micro-steps.

Run facts:
- 680.4 ms of GPU work per micro-step; idle 1.3%.
- GEMM 69.8%, attention 17.4%.
- Clock 1500 MHz median at 691 W (power-capped).

Item costs in ms per micro-step (sum ~58.7, i.e. 8.6%; realistic gain 35-45 ms):
- C5b 12.1, C5 7.1
- C2 adds 5.4 + ~6.5 from merging the GEMMs
- C3 5.5, residual-grad sums 5.4
- A1/A2 5.7
- C1 4.9 (fp32 path: GRAD_VIEW is DECLINED at world 1)
- C4 2.9, C6 2.8, L1 0.45

New items:
- G1: dW of wQ/wO at 58% of peak (wave quantization), ~4.5 ms.
- O1: optimizer has 331 host syncs/step, about 35% idle of a 280 ms window at world 1.
- RMSNorm runs at 41% of DRAM bandwidth (21 ms).

Files:
- `profiling_reports/matmul_audit_2026_10_26/` (box/ scripts, analyze_capture.py, box_runs/nsys_1026_2001 with .nsys-rep + analysis/).
- The box keeps `~/gautam_k/audit` (tree, pz, runs incl. sqlite).
- `BLU_NVTX=1` patch in BluTrain_lottn `Profiler/include/profiler/RecordFunction.h`; the user was given the commit command.


**BUILT + TESTED 2026-10-26 21:00-21:40 IST (work doc §45.98).**

Code:
- Laptop `~/Downloads/BluTrain_fuse` (not committed); box `~/gautam_k/audit/tree_new` + `pz_new`.
- Env switches, all off by default; `old` == `off` bit-identical at step 0:
  - `BLU_ADDMM_RS=1`: addmm uses the regsplit config for a full M x N bias.
  - `BLU_FUSE_RESID=1`: C3.
  - `BLU_FUSE_WGRAD=1`: C1.

C3 results:
- Standalone: -6.1% wO / -2.6% down.
- Precision: not-correctly-rounded 29% -> 0.1-0.4%.
- Trainer 1-GPU, CONFIRMED 22:15 IST: three pairs on steps 2-3 = -0.74% / -0.24% / -0.77% (both orders, both GPUs), **mean -0.58%**; +126 MiB peak; loss same band. §45.99.
- `compare_ab.py` averages every step >= 2: runs of different length give a FALSE gap (it printed old vs off +2.94%; same window = +0.23%). Always cut to a common last step.

C1 results:
- Precision: -18% gradient error.
- Speed: no gain. The beta=1 dW GEMM is 1.5-4.8% slower; it needs a TMA prefetch of C in the BLAS.
- Trainer (22:00 IST, world 1 with `BLU_GRAD_VIEW_W1=1` so the bf16 view engages): **+0.20% slower** (v 21,621/21,603 vs vc1 21,652/21,658 ms). Keep it OFF until the BLAS prefetches C.

SwiGLU: Kota's fused kernel is activation-only with a 128-block column pairing. Needs our own dual-output epilogue plus a bwd epilogue.

**How to apply:** Monday step 1 = one nsys run on the H200 timing the kernels behind C1–C3, before any code. Ask before the GPU run.

**Instance cut 2026-10-26 ~22:20 IST.** Everything copied to `profiling_reports/matmul_audit_2026_10_26/box_runs/` (rsync checksum dry-run = no difference; the 7 edited sources match `BluTrain_fuse` by md5). Not copied: 795 MB capA/capB sqlite (re-export from .nsys-rep), binaries. Next: SwiGLU dual-output epilogue, C-tile TMA prefetch for beta=1, C2 merged QKV.

**STATE end 2026-10-26 / RESUME 2026-10-27** (work doc §45.100):
- Done: 15 audit items all TIMED (nsys, §45.97); only 2 BUILT + tested (C3 win, C1 needs C-tile prefetch). SwiGLU: Kota's fused kernel only read, not usable for training. No real "fp32-output GEMM" variant was built (bytes argument only).
- Every fp32 op next to a matmul (SwiGLU, RoPE, residual add, grad add) = bf16 storage + fp32 math in registers; the epilogue removes the rounding of the matmul output before that math (checked in SwiGLUKernels_sm90.cu:107/200 and FusedRMSRoPE_bf16.cu:21-37).
- Uncommitted: `~/Downloads/BluTrain_fuse` (4 top-level files: ShardedMuon.cpp, bluscript_zero_ckpt.cpp + TI MatmulBackward.cu/.cuh, MatrixBackward.cpp, AutogradMeta.cpp + BLAS Bgemm_addmm_sm90.cu). BluTrain_lottn 16bce82 (BLU_NVTX) committed locally, NOT pushed (origin ref still 3b80f15 on 10-27). Exp repo docs §45.96-100 + box_runs + claude_chats sync uncommitted.
- Plan order: (1) C5 SwiGLU dual-output epilogue (gu + act from fp32 acc; gate/up column pairing per tile), (2) C5b SwiGLU bwd in down-dX epilogue, (3) C-tile TMA prefetch for beta=1 then re-run C1 A/B, (4) C2 merged QKV, (5) long precision run, A1/A2, C4, G1, O1; fix compare_ab.py common last step.
- No GPU box as of 10-27 morning (95.133.254.29 cut). SM90 epilogue code can only be COMPILED locally (sm_90a), runs need an H200.

**2026-10-27 (4x H200 95.133.252.206, GPU 1 locked till 20:00; work doc §45.101):**
- Trees in /dev/shm/gautam_k (symlinked in /root/gautam_k/audit): tree_new (C3/C1/C5), tree_dev (+beta TMA prefetch), tree_c5b (+C5b), tree_c2 (+C2 qkv K-grouped + engaged lines), tree_l1 (+L1/L2 linear_nt), tree_c4 (+C4 fused lm_head+CE with fp32-accumulate dW, +BLU_MEM_STATS). Binaries pz_new/pz_dev/pz_c5b/pz_c2/pz_l1/pz_c4. Laptop source of truth: ~/Downloads/BluTrain_fuse (NOT committed).
- Switches: BLU_FUSE_RESID+BLU_ADDMM_RS (C3), BLU_FUSE_SWIGLU (C5), BLU_FUSE_SWIGLU_BWD (+BLU_SWIGLU_BWD_BN=128 faster) (C5b), BLU_FUSE_WGRAD (C1), BLU_FUSE_QKV (C2), BLU_FUSE_LMHEAD (L1+L2), BLU_FUSE_LMHEAD_CE (+BLU_CE_CHUNKS) (C4), BLU_GRAD_VIEW_W1 (prod grad path at world 1), BLU_MEM_STATS (per-step [MEM]).
- Results: C5 gate_up -3.3..-3.6%/call, gu bit-identical, act not-correctly-rounded 34%->0.24%; C5b -3.3%/call (bn128), dgu 26%->0.14%; C1+prefetch bit-identical, break-even; L1 -1.74% fwd+bwd BIT-IDENTICAL; trainer (median steps 2-7): C5b alone -0.20%, C1+C3+C5+C5b(bn256) -0.84% = matches kernel prediction -0.80%. Block gradient error vs fp32 1-19% lower.
- WHY gains < 10-26 projection: fused epilogue keeps only 3-40% of the removed kernel's time (bytes cut 33-50% + epilogue serialized after MMAs). Next lever = ping-pong epilogue overlap (~-2% more). Memory: fusions remove transient buffers only; peak = saved activations. C5b +866 MiB nvidia-smi peak: suspected allocator cache (allocator trims cache + retries on OOM, so reserved-only is harmless) -- test_mem_mlp + [MEM] lines measuring.
- Tests crashed by TEST bugs today (not kernels): c5b autograd fp32-vs-bf16 checksum; c2 multi-root backward (engine runs each root separately -> 3-output node ran 3x). Use ONE root. Line-buffer stdout in tests.
- Box gotchas: step 6 slow in EVERY run (34-49 s) -> compare with median; startup ~65 s idle per launch -> run to step 7; queue scripts must not release the all-day lock (lock_gpu1.sh claims till 20:00, never releases).
- Deferred: A1/A2 (5-D TMA head-major A/C maps), R (engine/RMSNorm-bwd grad sum), M1/M2/M3 (Muon, tiny), C2-fwd merge, ping-pong epilogue overlap.

**10-27 afternoon (H200 GPU 1), CORRECTED 10-27 night (work doc §45.103):** ladder 2 vs production (1 GPU; median of steps
2-7 without step 6, mean of 2 ABBA runs): C3+C5+C5b+C1 -0.87%, +C2 -1.87%, +L1 -1.83%, +C4 -1.53% but -2.1 GB live /
-3.8 GB held (the earlier -1.09/-2.09/-1.78 are not reproducible from the logs). C5b does NOT raise memory. Per-op
precision vs fp64: C2 dh 45.4% -> 0.06% not-correctly-rounded (small), 45.2% -> 0.20% (production), C5 34.3% -> 0.04%,
C5b 26% -> 0.14% (production). Pipeline test step 0: fused grad norm ~2x closer to the fp32 reference (0.0010 vs 0.0019,
noise 0.0006), loss 7e-5 further.
G1 (dW tile) REFUTED: dispatcher already picks 128x192 (wQ/wO) / 128x128 (wK/wV); no tile faster. **C2F NGroup3** (q/k/v
as one N-grouped GEMM, tree_c2f, BluTrain_fuse): bit-identical fwd + dW (5 cfgs, beta 0/1); fwd +2% (OFF), dW -11.7%
standalone, trainer -0.29% (BLU_FUSE_QKV_DW=1 BLU_QKV_DW_CFG=0). O1 (Muon syncs) matters only at W=1.

**STOPPED 2026-10-27 19:57 IST (user: stop, copy, continue tomorrow; 4x box may close).** Validation run (n0 n3d mref,
40 steps) killed at start; GPU 1 claim released on the board. Copied to `profiling_reports/matmul_audit_2026_10_26/
box_runs_1027/` (runs/ incl. nsys_best_1027_1939 .nsys-rep + csv + sqlite, box_scripts/, audit_top/ build logs).
Laptop `~/Downloads/BluTrain_fuse` = superset of the box: 1218/1219 files of tree_c2f equal by md5, the 1 diff = the
measurement-only KO guards (laptop only). Local nsys 2025.3 is older than the box's 2025.6 -> use the copied sqlite.
Memory (allocator, MiB, 1 GPU B8xT4096): production used peak 88,139 / reserved peak 90,305 / end-of-step 20,197;
C3 C5 C5b C1 C2 = 0 change; +L1 (= best set n3d) -222 / -222; +C4 -2,138 / -3,830; nvidia-smi peak 90,899 / 90,675
(n3d) / 87,059 (n4d) (1903 run; filter the smi csv by GPU index -- it logs all 4 GPUs). Peak = saved activations.
Speed (§45.103 method): n3d -2.16% (21,317 -> 20,857 ms/step), n4d -1.80%. Not built: A1/A2, R, C6, M1-M3, O1, ping-pong, 8-GPU check.
Next: 40-step loss/precision validation (n0 n3d mref), nsys_best analysis, then R or A1/A2. Code NOT committed.
