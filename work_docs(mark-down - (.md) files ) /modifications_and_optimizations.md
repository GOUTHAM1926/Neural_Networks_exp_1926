# Modifications & Optimizations — Full Record (from the very start)

**Author:** Goutham Reddy (Gautam_1926)
**Purpose:** one consolidated, chronological record of *every* optimization,
restructure, and bug-fix shipped across the framework — folding together the
earlier `SPRINT_OPTIMIZATIONS_FULL_RECORD.md` (16 sprint items), the per-topic
docs in `master_gau_latest_ada_6000_sm89/docs/`, and the **post-sprint work**
done since (the BluTrain → may27_406k sync + the five gap-closing ports).

**Repos referenced (all under `/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/`):**

| Tag | Path | Role |
|-----|------|------|
| **B (active)** | `BluTrain_BestPrecisionEnv_may27_406k/BluTrain_BestPrecisionEnv/BluTrain/Tensor-Implementations` | the live BLAS+training repo we work in now (with Kota) |
| **A (reference)** | `BluTrain_best_precision/Tensor-Implementations` | precision fork; held the newer reductions + cuBLASLt fused-bias before the port |
| **BluTrain (parked)** | `BluTrain/Tensor-Implementations` | holds the sprint tracker + per-topic docs |
| **sm89 docs** | `master_gau_latest_ada_6000_sm89/docs/` | where the per-topic markdown docs live |

> **Status legend:** ✅ present & verified in **B**  ·  🆕 added/ported this session  ·  ⚪ deliberate no-op / left as-is

---

## Part A — The sprint optimizations (16 items)

Source of truth: `master_gau_latest_ada_6000_sm89/docs/SPRINT_OPTIMIZATIONS_FULL_RECORD.md`.
The sprint had two phases: **kernel-centric** (open a kernel, fix GPU idle/waste)
then **training-centric** (eliminate cross-kernel redundancy visible only from
the training-loop lens). Condensed below; all verified **byte-identical in B**
(md5-compared) unless noted.

### Kernel-centric phase

1. **Reductions module overhaul** (previous sprint's foundation) ✅
   - Replaced 4 copy-pasted kernels (~800 lines) with one `unified_reduce_kernel`
     (functor `ops_t`). PyTorch-style 3-path layout dispatch
     (`InnerContiguous` / `OuterContiguous` / `Generic`).
   - Multi-stage launch dispatcher (`nt ∈ {32,64,128,256,512}`), cascade (pairwise)
     sum for FP precision, Welford one-pass variance, O(1) `reduced_bitmap`
     replacing 22× `std::find`, packed `__grid_constant__` metadata, caching
     allocator for intermediates, NaN-deterministic argmin/argmax.
   - Files: `ReductionKernels.cuh`, `ReductionImpl.h`, `ReductionUtils.h`,
     `ReductionOps.h`, `ReductionImplGPU.cu`.

2. **GeLU full restructure (fwd+bwd, CPU+GPU)** ✅
   - Adopted the uniform **3-layer pattern**: `nn::GeLU.forward()` →
     `autograd::gelu()` (thin wrapper, saves tensor + attaches grad node) →
     `gelu_forward()` (pure math: CPU AVX2 or GPU dispatch). **Math lives OUTSIDE
     the autograd wrapper.** `ActivationOps.cpp` 400→179 lines;
     `ActivationBackward.cpp` 277→90 lines.
   - GPU: `fused_gelu_kernel_vectorized` (`float4` + `fast_tanh`); ReLU rewritten
     branch-free `(x+|x|)*0.5`; `fused_bias_gelu` templated for fp16/bf16.
   - Deleted buggy legacy `include/mlp/`, `src/mlp-blocks/`.

3. **Activation module restructure (ReLU, Dropout, Softmax, RMSNorm)** ✅
   - Same 3-layer pattern across the suite. RMSNorm via PyTorch's
     `bool rms_norm` template-parameter trick (same LayerNorm kernel reused with
     `if constexpr`, zero runtime cost).

4. **LayerNorm restructure + RMSNorm** ✅
   - CPU fwd 3→2 passes (Welford) + AVX2 + FMA; CPU bwd OMP + thread-local + AVX2;
     GPU `bool rms_norm` template fwd, `float4`-vectorized bwd; `__launch_bounds__(512)`
     on sm89. RTX 3060 numbers: CPU fwd 12.3×, CPU bwd 5.6×, GPU bwd ~20% faster.

5. **Memset cleanup — sync→async + redundant removal** ✅
   - All blocking `cudaMemset` → `cudaMemsetAsync(...,stream)`; removed memsets on
     buffers the next kernel fully overwrites (LN bwd ~25K memsets, attn dQ
     `zeros`→`empty` ~4K). Saved ~3.2 s/run.

6. **Stride-aware attention kernels** ✅
   - Kernels accept per-tensor strides; removed pre-kernel `.contiguous()` on
     Q/K/V/grad_out (~17K kernels, ~2 s).

7. **Contiguous kernel — 4-path hybrid dispatcher** ✅
   - `coalesce_dimensions` → (3a) DMA `cudaMemcpyAsync` for fully-contiguous,
     (3b) tiled shared-mem 2D transpose, (3c) vectorized `float4`/`uint2` inner
     copy, (3d) generic FastDivmod fallback. `ContiguousMeta` via
     `__grid_constant__`. FastDivmod ≈ 6× vs `%`/`/`. (A later Path-3d add covered
     `[B,S,H,D]→[B,H,S,D]`.)

8. **BatchedCat fused kernel** ✅
   - PyTorch-style `CatArrayBatchedCopy` — one launch for an N-way concat instead
     of N `cudaMemcpy2DAsync`. (`misc/BatchedCat.cu`.)

### Lens shift (§9) → training-centric phase

1. **Perspective shift** — from "is this kernel optimal?" to "is this kernel's I/O
   necessary given what the producer wrote and the consumer wants?"

2. **`Tensor::contiguous()` short-circuit** ✅
    - Was allocating + copying even when the input was already contiguous. Fix
      (`Tensor.cpp:283-287`):

      ```cpp
      if (is_contiguous() && storage_offset() == 0) { return *this; }
      ```

      Returns a view, not a copy.

3. **`multi_tensor_zero_grad` kernel + factory rewrites** ✅ (see Part C #1 for the
    Optim.cpp call-site, which needed re-enabling in B)
    - `multi_tensor_zero_sm89_kernel` zeros up to N tensors per launch (uint4 /
      STG.128 16-byte stores, 128 KB chunks, `block_to_tensor`/`block_to_chunk`
      metadata). `fill_cuda_launch<T>` (vec16/vec8/scalar) for ones/full.
    - Factory: `Tensor::zeros` → `cudaMemsetAsync` (no CPU round-trip);
      `ones`/`full` → `fill_cuda_launch`; bool byte-memset fix.

4. **CachingCudaAllocator integration** ✅
    - `cudaMallocAsync`-backed caching allocator on all compute allocations;
      removed two `cudaStreamSynchronize` from the free path (event-based reuse).

5. **Sparse cross-entropy save_max/save_sum** ✅
    - Forward saves `saved_max[B]`/`saved_sum[B]`; backward reads logits once
      (3×→2× the 3.3 GB logit traffic). ~half PyTorch's memory traffic.

6. **Matmul: cuBLASLt port + fused-bias epilogue** ✅ (see Part C / Part B for B status)
    - `CUBLAS_COMPUTE_32F_FAST_TF32` via `cublasLtMatmulAlgoGetHeuristic` →
      `cublasLtMatmul`; `CUBLASLT_EPILOGUE_BIAS` fuses bias into the GEMM
      (`broadcast_scale_kernel` ~17,280→1 per run). Backward DGELU_BGRAD fusion
      still **deferred**.

7. **Smart reshape (`compute_view_stride`)** ✅
    - Mirrors PyTorch `computeStride_impl`; view-eligible reshapes skip
      `.contiguous()` (killed 12,960 strided copies/run).

8. **Packed SDPA** ✅
    - Q/K/V read via strided pointer offsets into packed `[B,T,3C]`; output written
      as packed `[B,T,C]` (the layout `c_proj` wants) — eliminates the
      post-attention merge copy. `skip_grad_zero` flag for the packed backward.
      Env-gated (`USE_PACKED_SDPA=1`). Killed 15,840 generic + 3,840 cat copies.

**Sprint wall-time roll-up:** baseline 126.56 s, PyTorch 113.04 s; cuBLASLt port
(~10 s) + Packed SDPA (~3 s) + memset/stride/reshape close the gap and pass
PyTorch in some runs.

---

## Part B — Correctness fixes (cross-cutting)

- **Weight-tying scramble bug (permanent fix).** ✅ in B.
  Root cause: `TransposeBackward` returned a strided view →
  `AutogradMeta::accumulate_grad` adopted it as-is → stride-blind multi-tensor
  Adam walked `param[i]`/`grad[i]` flat → deterministic per-element permutation
  of `wte.weight` grad. Fix = **Gradient Layout Contract** in `AutogradMeta.cpp`
  (mirrors PyTorch `AccumulateGrad` Case 1.5: materialize a contiguous clone if
  the incoming grad doesn't obey the contract). `TransposeBackward.cpp` reverted
  to the simple form (v3 — the contract handles it generically).
  Verification metrics: wte cos 0.001→0.99999994 after fix.
  *(`AutogradMeta.cpp`, `TransposeBackward.cpp` byte-identical in B.)*
  Doc: `weight_tying_bug_fix.md`.

- **Reductions — the 4 bugs + extras.** ✅ in B.
  (1) CPU middle-axis demotion (`cpu_demote_middle_to_generic`), (2) InnerContiguous
  GPU, (3) GPU non-consecutive multi-axis (`setup_generic_path`), (4) vec4/multi-CTA
  `block_y_reduce` missing trailing `__syncthreads()`. Plus **NaN-check removal in
  `reduce_add`/`reduce_mul`** (`ReductionOps.h` — IEEE add/mul already propagate
  NaN, so the branch was redundant), **multi-CTA staging**, **vec4 output**, and
  **fastdivmod** (Granlund–Montgomery division-by-invariant) in the generic path.
  Also a **32-bit indexing guard** (`ensure_32bit_reduction_indexing`): GPU reduce
  indexes with uint32 + int counts → throws for numel > 2³²−1 or counts > 2³¹−1
  (PyTorch-style split is the deferred real fix). Validated **90/90** on sm_86.
  Docs: `reduction_bias_gradient_bug_fix.md`, `data_accessing_techniques.md`.

- **Adam `eps` placement (multi-tensor kernel).** ✅ in B.
  `denom = sqrtf(v_hat) + eps` — eps **inside** the denominator (not the old
  `rsqrtf(v_hat) + eps` which gave `rsqrtf(0)=+Inf → NaN` at v=0). Present in both
  the inline and macro forms (`MultiTensorKernels_sm89.cu`).

- **Types.h fp32→bf16 inf-clamp bug.** 🆕 fixed in B this session — see Part C #5.

---

## Part C — Post-sprint work (2026-06-08): BluTrain → B sync + gap-closing ports

Method: full `diff -rq` of parked **BluTrain** vs **B** → only **25 files** differ;
everything else byte-identical (so B already had the whole sprint). Triaged the 25
(token-normalized to strip clang-format noise) and closed the real gaps. Each port
built clean (`make tensor`, exit 0) on sm_86.

### 1. `multi_tensor_zero_grad` re-enabled in B's `Optimizer::zero_grad()` 🆕

- **Before (B):** `zero_grad()` looped `for (p:params) p.zero_grad();` — **one
  tensor zeroed at a time** (148 separate ops for GPT-2's 148 params). Scalar path.
- **After:** collects all GPU-resident grads into `std::vector<cuda::ZeroTensorInfo>`
  and calls `cuda::multi_tensor_zero_cuda(gpu_grads)` → `multi_tensor_zero_sm89_cuda`
  → **one batched kernel** zeroing many tensors/launch (uint4 STG.128, 128 KB
  chunks). ~4 launches instead of 148. Dtype-agnostic (bit-pattern 0 = value 0).
  CPU grads keep the per-param path.
- File: `src/nn/optimizer/Optim.cpp` (`zero_grad` body). The `multi_tensor_zero_*`
  kernel + wrapper already existed in B; only the call-site was the slow loop.
- ⚠️ **Validate with one training step** (only behavioral change in this batch).

### 2. RELU branch-free math 🆕

- `src/Kernels/cuda/activations/RELUKernels.cu`.
- fwd: `val > 0 ? val : 0` → **`(val + fabsf(val)) * 0.5f`** (the "x + |x| over 2"
  form; propagates NaN, unlike the ternary).
- bwd: `val > 0 ? grad : 0` → `grad * (float)(val > 0)` (branch-free mask).

### 3. GELU fused-bias epilogue — fp16/bf16 templated 🆕

- `src/Kernels/cuda/activations/GELUKernels.cu`.
- B previously had **float-only** `fused_bias_gelu` (fwd + bwd). Now **templated**
  for `float`/`__half`/`__nv_bfloat16` (storage type T, math in fp32 via
  `to_float`/`from_float`); added `launch_fused_bias_gelu` and
  `launch_fused_bias_gelu_backward` overloads for half/bf16 (grad_bias stays fp32).
  Float entry points unchanged → ABI-safe, no caller breaks.

### 4. LayerNorm/RMSNorm backward `cudaMemset` → `cudaMemsetAsync` 🆕

- `src/Kernels/cuda/norm/LayerNormKernels.cu` — 3 sites
  (`launch_layer_norm_backward` grad_gamma/grad_beta; `launch_rms_norm_backward` gg)
  switched from blocking `cudaMemset` to `cudaMemsetAsync(...,0)` so they don't
  serialize the default stream.

### 5. Types.h fp32→bf16 inf-clamp **bug** removed 🆕

- `include/dtype/Types.h`, `float_to_bfloat16`.
- **Bug:** an extra `if (exponent > 0x8E) return Inf;` clamped every `|x| ≥ 2¹⁶`
  (=65536) to infinity. But bf16 shares fp32's 8-bit exponent → its range is
  ~3.4e38; nothing in that range overflows. The clamp was an **fp16-style guard
  mis-applied to bf16** and wrongly flushed normal values ≥ 65536 to Inf.
- **Fix:** removed the clamp (bf16 needs only the `exponent == 0xFF` NaN/Inf
  passthrough). `float_to_float16` (fp16, 5-bit exponent, max ~65504) legitimately
  keeps its overflow handling and was left untouched.
- Latent for current runs (GPT-2 trains fp32), real for any bf16 path.

### 6. TensorImpl.cpp `zero_grad` dtype dispatch 🆕 (ported earlier this session)

- `src/core/TensorImpl.cpp`. Replaced `grad_tensor.fill<float>(0.0f)` (threw
  "Fill: Datatype mismatch" for fp16/bf16 grads on the CPU path) with
  `dispatch_by_dtype(...).fill<T>(static_cast<T>(0))`. Mirrors the multi-tensor
  byte-memset's dtype-agnostic behavior.

### 7. cuBLASLt fused-bias matmul (ported A → B earlier this session) ✅

- `src/Kernels/cuda/matmul/GenMatmul.cu` — the `CUBLASLT_EPILOGUE_BIAS` fused-bias
  path (A's inline version) injected into B's `#else` (default-cublas) branch of
  `#ifdef WITH_MYBLAS`. Pure addition (+247 lines), no other change.
  *(Note: `master_gau`'s `CHANGES_AND_FIXES_LOG.md` documents a separate
  `CublasLtHelper.cu`; B uses the inline-in-GenMatmul approach — functionally the
  fused-bias path is present either way.)*

---

## Part D — TF32 rounding: round-half-up (`+0x1000u`) vs RNE (analysis)

In the attention kernels (`AttentionForward.cu`, `AttentionForward_sm89.cu`,
`AttentionBackward.cu`, `AttentionBackward_sm89.cu`) and the SM89 matmul core, the
fp32→TF32 operand conversion uses a **biased round-half-up**: `reg += 0x1000u`
before the `mma.sync.tf32`. (BluTrain switched its attention kernels to RNE; **B
keeps `+0x1000u`** — a deliberate perf choice, documented here as intentional.)

**What `+0x1000u` is:** TF32 keeps the top 19 bits of fp32 (drops the low 13
mantissa bits). Adding `0x1000` = half of the 2¹³ truncation unit, then letting the
MMA drop the low 13 bits = **round-to-nearest with ties rounding up** ("round
half up", a.k.a. round-half-toward-+∞ in bit space ≈ round-half-away-from-zero for
positives). Without the bias, truncation alone is round-toward-zero (biased down).

**RNE (round-half-to-even)** uses a value-dependent bias:
`lsb = (r>>13)&1; r = (r + 0x0FFFu + lsb) & 0xFFFFE000u`. Ties round to the even
LSB — bit-exactly matching cuBLAS.

| | round-half-up `+0x1000u` | RNE |
|---|---|---|
| Instructions / element | **1** (`IADD`) | **~4–5** (shift, and, add, add, and) |
| When result differs | — | **only at the exact tie** (discarded bits == `0x1000`) |
| Matches cuBLAS at ties | no (tiny upward bias) | yes |

**Answers to the recurring questions:**

- *"Do the two differ only at 0.5?"* — **Yes.** At every non-tie value both round
  to the same nearest TF32; they disagree **only** at the exact midpoint.
- *"With RNE, do instructions increase only at 0.5, or always?"* — **Always.** RNE
  is *branchless* straight-line code; it runs the same ~4–5 ops for every element,
  tie or not. The instruction count is constant (and constantly higher than 1).
  What's tie-specific is the **result**, not the op count.
- *"Can RNE be 1 instruction?"* — **No.** Round-half-up uses a *constant* bias
  (`0x1000`) → single add. RNE's bias depends on the value's own bit-13 (to break
  ties to even), so it must be *computed* first → inherently ≥ 3–4 ops.
- **Conclusion:** ties are astronomically rare in real float products, so
  `+0x1000u` gives effectively identical numerics at 1 op vs ~4–5 — **strictly
  faster, same results in practice.** Keeping `+0x1000u` in B is correct for a
  throughput build. The only caveat: not bit-identical to cuBLAS at exact ties
  (a strict bit-parity test could show a handful of 1-ULP-TF32 diffs).

> Context: this replaces an earlier broken **"3×TF32"** path (which is why the
> SM89 GEMM comment still says "3xTF32 emulation" though the code now does a
> single TF32 MMA + rounding bias). True 3×TF32 would split each fp32 into 3 TF32
> limbs and issue 3–6 MMAs for near-fp32 accuracy; the single-MMA + round-half-up
> is the perf-favoring choice.

---

## Part E — Verification status in B (may27_406k)

| Item | Status in B | Where |
|------|-------------|-------|
| Reductions (4 bugs + nan-removal + multi-CTA + vec4 + fastdivmod + 32-bit guard) | ✅ (90/90 sm_86) | reduction files |
| Reductions — 16-bit InnerContiguous input-vectorization (VT0=8, Part F) | 🆕 (112/112 sm_86, BBP) | `ReductionKernels.cuh`, `ReductionImplGPU.cu` |
| Weight-tying Gradient Layout Contract | ✅ | `AutogradMeta.cpp` |
| Adam eps-in-denominator | ✅ | `MultiTensorKernels_sm89.cu` |
| Activations 3-layer restructure (math outside autograd) | ✅ | `ActivationOps.cpp` (179 lines, 0 math), kernels |
| Factory GPU-side fills | ✅ | `TensorFactory.cpp` |
| `Tensor::contiguous()` short-circuit | ✅ | `Tensor.cpp:283-287` |
| ContiguousKernel 4-path, BatchedCat, CachingAllocator, sparse-CE, smart-reshape, Packed SDPA | ✅ | byte-identical to BluTrain |
| cuBLASLt fused-bias matmul | ✅ | `GenMatmul.cu` (`#else` branch) |
| `multi_tensor_zero_grad` call-site | 🆕 re-enabled | `Optim.cpp` |
| RELU branch-free | 🆕 | `RELUKernels.cu` |
| GELU fp16/bf16 templated bias | 🆕 | `GELUKernels.cu` |
| LayerNorm/RMSNorm bwd memset→async | 🆕 | `LayerNormKernels.cu` |
| Types.h bf16 inf-clamp bug | 🆕 fixed | `Types.h` |
| TensorImpl zero_grad dtype dispatch | 🆕 | `TensorImpl.cpp` |
| Attention/matmul TF32 rounding | ⚪ round-half-up (intentional, faster) | attention + sm89 GEMM |
| `SizeClass.h` allocator buckets | ⚪ differ (tuning, not correctness) | `SizeClass.h` |

**Build:** `/usr/local/cuda-13.0/bin/nvcc`; `make tensor`. Header-dep gotcha:
after editing a header, `rm` the dependent `.o` (make doesn't track header deps).

---

## Part F — Post-sprint work (2026-06-11): 16-bit input-vectorization in the reduction kernel

**Repo:** `BluTrain_best_precision` (BBP). **GPU tested:** RTX 3060 (sm_86), CUDA 13.0.

**What & why.** Ported PyTorch's *"vectorize along input"* path (`Reduce.cuh`
`input_vectorized_thread_reduce_impl`) for **16-bit** dtypes on the
**InnerContiguous** (last-axis) reduction. Previously the kernel used scalar
strided loads with `VT0 = 4` accumulators for *all* dtypes/paths. Now, for fp16/bf16
InnerContiguous reductions, a thread issues **one 128-bit `VecLoad<scalar_t,8>`** per
step into **8 independent accumulators** — matching PyTorch's `input_vec_size = 8`
for 2-byte types (eight 16-bit values fill one 128-bit transaction).

**Files changed** (`Tensor-Implementations/`):

- `include/ops/helpers/ReductionKernels.cuh`:
  - `GpuReduceConfig`: new `int input_vec_size = 1` field.
  - `build_reduce_config<acc_t>(…, int input_vec_hint = 1)`: gate that sets
    `input_vec_size` and divides the block work-dim by the width. Fires only for
    `path == InnerContiguous && reduced_count >= 128 && reduced_count % hint == 0`.
    `num_inputs` stays in **element** units (so `MeanOps`' divisor and the multi-CTA
    math are unchanged); only the block geometry is sized in vector units.
  - `unified_reduce_kernel<…, int VT0, bool INPUT_VEC>`: new `INPUT_VEC` branch in
    the InnerContiguous case doing the contiguous 128-bit load into `acc[VT0]`.
  - Launcher: dispatches the `VT0=8, INPUT_VEC=true` kernel under
    `if constexpr (sizeof(scalar_t) == 2) { if (config.input_vec_size == 8) … }`,
    so the 8-wide kernel is only instantiated for 2-byte types.
- `src/UnaryOps/cuda/ReductionImplGPU.cu`: value-reduction and mean dispatchers pass
  `input_vec_hint = (sizeof(CudaT)==2) ? 8 : 1`. Index (argmin/argmax) and Welford
  variance dispatchers keep `hint = 1` (scalar path) — index ops need the per-element
  index, Welford is precision-sensitive.

**Design choices (honest scope):**

- **InnerContiguous only.** OuterContiguous (the bias-gradient case) reduces a
  *strided* axis, so a thread can't pull 8 contiguous values in one load — it stays
  on the `VT0=4` strided path (unchanged). fp32 also stays on `VT0=4` everywhere.
- **No head/tail peel.** Gating on `reduced_count % 8 == 0` guarantees every row base
  is 16-byte aligned, so the wide load is always aligned — simpler than PyTorch's
  unaligned-head handling. Non-multiples of 8 fall back to the scalar path.
- 4-accumulator ILP remains the **baseline for all paths**; 8 is the **only** 16-bit
  InnerContiguous exception.

**Validation.** `reduction_final_test` = **112/112** (90 prior + 22 new input-vec
cases: gate-on at 128/256/1024/2048/1000, gate-off fallbacks at 1001/130/<128,
multi-CTA at 262144, mean-divisor checks; f16 + bf16). New micro-benchmark
`reduction_ilp_benchmark.cpp` for before/after timings.

**Measured (RTX 3060, `sum(axis=-1)`, before → after):**

| case | f16 / bf16 | note |
|---|---|---|
| aligned large (e.g. 16384×1024) | +2–6% | already ~92% of ~360 GB/s — bandwidth-bound |
| `R = 1000` (= 8×125) | **~+25%** (241 → 300 GB/s) | scalar strided path was inefficient here |
| fp32, and non-÷8 sizes | unchanged | correctly stay on scalar path |

No regressions; the expected register-pressure occupancy drop from 8 accumulators did
not materialise.

**Caveats / not done:** small win on the 3060 because 16-bit reductions are already
bandwidth-bound there — likely larger on higher-bandwidth GPUs (A100/H100/B300),
**not yet benchmarked** on those. The formal technical report (`madhu.txt`/`.pdf`) was
**left unchanged**: its ILP paragraph + benchmarks are scoped to the bias-gradient
(OuterContiguous) path, which this change does not touch, and we have no
report-hardware (RTX 6000 Ada) numbers for the InnerContiguous path. Full conceptual
write-up (4-vs-8, Volkov/latency truth + sources) is in
`bias_gradient_reduce_sum_kernel_usage.md` §10.3.

---

## Source documents (for deeper detail)

In `master_gau_latest_ada_6000_sm89/docs/`:
`SPRINT_OPTIMIZATIONS_FULL_RECORD.md` (master), `gelu_deep_dive.md`,
`layernorm_modifications.md`, `gelu+layernorm changes in master_gau_1.md`,
`contiguous_kernel_architecture.md`, `CHANGES_AND_FIXES_LOG.md`,
`weight_tying_bug_fix.md`, `reduction_bias_gradient_bug_fix.md`,
`Optimizing Packed Attention Performance.md`, `template_programming_in_cpp_and_cuda.md`.

---
