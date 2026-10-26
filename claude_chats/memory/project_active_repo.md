---
name: project-active-repo
description: Active working repo for GPT-2 124M trainer + BLAS work. Which tree to edit, and the may27 repo/remote layout.
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-21T06:51:00.259Z
---

As of **2026-06-08**, active work moved to **`BluTrain_BestPrecisionEnv_may27_406k/BluTrain_BestPrecisionEnv/BluTrain/`** (called "B" / the may27_406k env) — the user now works on **BLAS with Kota**, mainly in the `BluBridge-BLAS/` and `Tensor-Implementations/` subtrees there. `BluTrain_best_precision/` ("A") is now a REFERENCE only (it held the newer reduction module + cublasLt fused path).

**Ported A→B and validated (this session):** the 5 reduction files (4 bug fixes + multi-CTA + vec4 + fastdivmod + bug-#4 trailing sync + cpu_demote_middle + nan-check removal + 32-bit guard) copied wholesale; the cuBLASLt EPILOGUE_BIAS fused-bias path injected into B's `GenMatmul.cu` (was a pure-addition, sits in the `#else`/default-cublas branch of `#ifdef WITH_MYBLAS`). Built clean on sm_86, `reduction_final_test` = **90/90** in B. Left B's newer `MatmulBackward.cu` (dual-stream) + `TensorOpsAdd.cu` (grid caching) + nvml v2 (RTX 3060) untouched. See [[project_reduction_bugs]], [[project_reduction_32bit_guard]].

**Repo/remote layout (changed 2026-06-08):** B was originally a multi-repo with submodules — BluTrain→`kiruthikkanna-m/BluTrain`, Tensor-Implementations→`kathir-23s/Tensor-Implementations`, dist/Data-Parallel→`kiruthikkanna-m/Data-Parallel`, BluBridge-BLAS→`kathir-23s/BluBridge-BLAS` (all on branch `master-instance-move-test`). The user **flattened it** (removed all nested `.git` + `.gitmodules`) and folded B's SOURCE into their own outer repo **`GOUTHAM1926/Neural_Networks_exp_1926`** (origin updated from old `GOUTHAM992003`). Heavy artifacts (checkpoints, *.bin/*.ckpt/*.pt, lib/, logs, nsys/ncu) are gitignored under the may27 path; tracked source = ~663 files / 7.6 MB. Goal: clone (not 14 GB scp) onto vast.ai **B300** instances — use `git clone --depth 1` to skip the 2.1 GB outer history.

**Full BluTrain→B sync verified (2026-06-08):** diffed parked `BluTrain/` (holds the sprint tracker `SPRINT_OPTIMIZATIONS_FULL_RECORD.md` + per-topic docs in `master_gau_latest_ada_6000_sm89/docs/`) against B — only 25 files differ; everything else byte-identical, so B already has: activations 3-layer restructure (math outside autograd; ActivationOps.cpp=179 lines, 0 math, delegates to gelu_forward/relu_forward), factory GPU-fills (TensorFactory.cpp), Tensor::contiguous() short-circuit (Tensor.cpp:283-287), ContiguousKernel 4-path, BatchedCat, allocator-async, sparse-CE, smart-reshape, Packed SDPA, weight-tying Gradient-Layout-Contract (AutogradMeta.cpp), Adam eps-in-denominator. **5 gaps closed this session (B only, build-clean):** (1) zero_grad multi-tensor fast path (Optim.cpp — was slow per-param loop; needs 1 training-step validation), (2) RELU branch-free (val+|val|)*0.5, (3) Types.h removed bogus bf16 `exponent>0x8E` inf-clamp (bf16 has fp32 range), (4) LayerNorm cudaMemset→cudaMemsetAsync ×3, (5) GELU fp16/bf16 templated bias fwd+bwd. Deliberately NOT changed: attention TF32 rounding (B uses biased `+0x1000u` round-half-up = 1 instr/elem; BluTrain RNE = ~4-5 instr/elem, differ only at exact ties → biased is fine per user). SizeClass buckets differ (tuning, ignored).

**Build:** nvcc is `/usr/local/cuda-13.0/bin/nvcc`; `make tensor` builds libtensor; header-dep gotcha still applies (rm reduction/GenMatmul .o after header edits — see [[project_reduction_build_gotcha]]).

**UPDATE 2026-07-21 — repo layout has drifted from the above, now THREE separate checkouts, easy to edit the wrong one:**
1. `~/Downloads/Neural_Networks_exp_1926/BLUBLAS_kota/BluBridge-BLAS` — **not its own git repo**, just a nested folder inside the outer monorepo described above. Playground only, edits here don't reach any BLAS remote on their own.
2. `~/Downloads/BLUBLAS/BLUBLAS_kota/BluBridge-BLAS` — a real, separate git repo, `origin = github.com/GOUTHAM1926/blublas.git` (personal). **This is the one actually used to push/pull for BLAS work now** (moved off the flattened monorepo referenced above).
3. `~/Downloads/BluBridge-BLAS_21_07_2026_kota_H200_setup/BluBridge-BLAS` — colleague Kota's handoff copy, `origin = github.com/Blubridge-Tech/BluBridge-BLAS` (the team/org repo) — looks like the actual eventual submission target ("main repo" / "others" the user needs to submit to).

**Open/unresolved:** how work pushed to #2 (personal `blublas.git`, where daily work happens) is meant to reach #3 (`Blubridge-Tech/BluBridge-BLAS`, the shared team repo) is NOT settled — no PR flow, fork-sync, or manual-copy convention confirmed yet, and the user expects this to change/get clarified later. Don't assume push-to-personal-origin is the finish line — check with the user whether a Blubridge-Tech PR (or some other sync step) is also required before treating any change as actually "shipped." Full detail in `analyzations_h200.md` §H2.5.
