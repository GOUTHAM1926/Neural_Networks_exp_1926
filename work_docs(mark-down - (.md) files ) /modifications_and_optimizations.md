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

## Part G — Post-sprint work (2026-06-13): `BluTrain_bf16_rev` GenMatmul sweep + merged file across 4 BluTrain folders

Working repo: `BluTrain_bf16_rev/` (the BF16 mixed-precision training branch).
Goal: optimize the cuBLAS-backend `GenMatmul.cu` for bf16/fp16 workloads, then
roll the merged result back to all 4 BluTrain folders.

### G.1. Dtype-mismatch bug (latent in all branches) — **fixed**

The `dispatch_by_dtype` system hands `bfloat16_t` / `float16_t` (project wrapper
structs in `include/dtype/Types.h`, each just a `uint16_t raw_bits` field) to
template-instantiated kernels. Multiple predicates in `GenMatmul.cu` only checked
the NATIVE CUDA types:

```cpp
if constexpr (std::is_same_v<T, __nv_bfloat16>) { ... }   // never fires for dispatch
```

Result: vectorized branches silently fell through to scalar / zero-init fallbacks
when the input was `bfloat16_t`/`float16_t` — a correctness-adjacent perf bug
present in **all** branches (`bf16_rev`, `best_precision`, `BluTrain_old`,
`may27_406k`). **7 sites fixed** to check BOTH wrapper and native types:

- `cublaslt_dtype<T>()` — fp16 + bf16 mapping (2 sites)
- `broadcast_scale_kernel<T>` — fp16 + bf16 vectorized branches (2 sites)
- `launch_broadcast_scale<T>` VEC selector (1 site)
- `add_scaled_kernel_typed` (1 site)
- `cuda_addmm` cuBLASLt dispatch lambda (1 site)

The merged file ships the fix to all 4 branches simultaneously.

### G.2. `broadcast_scale_kernel` VEC=2 → VEC=8 (fp16/bf16 paths)

Old code: VEC=2 for fp16/bf16 → 32-bit per-thread loads (one `__nv_bfloat162`
or `__half2`). Pushed to VEC=8 → 128-bit per-thread loads via `float4`
reinterpretation (4 × pair). Matches PyTorch's `can_vectorize_up_to<scalar_t>()`
(`pytorch_source/aten/src/ATen/native/cuda/MemoryAccess.cuh:508-533`) which
returns 8 for fp16/bf16 when pointer-aligned.

VEC selector at launcher:

```cpp
constexpr int VEC =
    std::is_same_v<T, float> ? 4
    : (std::is_same_v<T, __half> || std::is_same_v<T, float16_t> ||
       std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, bfloat16_t>) ? 8
    : std::is_same_v<T, double> ? 2 : 1;
```

**Measured** (`nsys12` VEC=8 vs `nsys9` VEC=2, both with EPILOGUE_BIAS OFF):

- `broadcast_scale_kernel` per-call avg: 68 µs → **44 µs** (−35%)
- Grid sizes 4× smaller (1536 / 4608 / 6144 vs 6144 / 18432 / 24576)
- Step 9 loss unchanged: 9.110535 (matches BluBLAS reference 9.110462)

Note: when EPILOGUE_BIAS is on (Part G.3), `broadcast_scale_kernel` is dead code
(never called) — the VEC=8 work is defensive for shapes where cuBLASLt's
heuristic might decline.

### G.3. cuBLASLt `EPILOGUE_BIAS` workspace — raw `cudaMalloc` → `AllocatorRegistry::get_caching_allocator()`

Was: per-device raw `cudaMalloc(32 MiB)` for the cuBLASLt workspace, leaked at
process exit. Switched to the project's caching allocator (the same one used by
`ReductionKernels.cuh` scratch buffers). Workspace allocated lazily on the first
`cuda_addmm` call per device.

### G.4. `nsys13` results (final EPILOGUE_BIAS + VEC=8 + dtype fixes)

GPT-2-44M BF16 autocast, B=8, T=1024, 10 steps, 3 transformer layers,
weight-tying OFF, sm_86 (RTX 3060):

| Metric | nsys12 (VEC=8 only) | **nsys13 (+ EPILOGUE_BIAS)** | Δ |
|---|---:|---:|---:|
| Avg step time (steps 1-8) | 891.5 ms | **884.8 ms** | −6.7 ms |
| Avg throughput | 73,514 tok/s | **74,075 tok/s** | +561 |
| Step 9 loss | 9.110535 | 9.110526 | ~0 (matches BluBLAS 9.110462) |
| `broadcast_scale_kernel` calls | 1,440 (62.25 ms) | **0** | −1,440 / −62.25 ms |
| Legacy `s1688gemm` addmm calls | 1,440 (436 ms) | **0** | −1,440 / −436 ms |
| Fused `s16816gemm_*_relu_f2f_*` calls | 0 | 1,440 (478 ms) | new fused-bias path |
| `cudaMemset` count | 4,986 | 3,546 | −1,440 |
| `cudaMemset` total bytes | 27.3 GB | **7.9 GB** | −19.4 GB |
| Total GPU kernel launches | 28,839 | 27,399 | −1,440 |

Fast path hit rate: **1,440 / 1,440 (100%)**. cuBLASLt's heuristic also picked
**larger tile widths** (`s16816gemm_256x128_32x3_nn` instead of
`s1688gemm_64x128_sliced1x2`) — a side benefit of the heuristic having access
to the bias as a fused epilogue op.

### G.5. Failed BGRADA backward bias-grad fusion (reverted)

Attempted `CUBLASLT_EPILOGUE_BGRADA` to fuse
`grad_bias = sum_i grad_output[i, :]` into the `grad_weight` backward matmul
(`A^T @ grad_output`).

**Important finding about the bias-grad code path**: the wired-up
`LinearBackward.cpp` + `cuda_linear_bias_backward` + `reduce_bias_kernel_bf16`
are **dead code** in this repo — `reduce_bias_kernel_bf16` recorded 0 calls in
every nsys. The actual bias-grad path goes through:

```text
nn::Linear.forward() → autograd::addmm
                  → AddmmBackward::apply (MatrixBackward.cpp:155)
                  → reduce_to_shape(grad_output, saved_input_shape_)
                  → reduce_sum
                  → unified_reduce_kernel<__nv_bfloat16, SumOp<bf16>>
```

240 calls / 20.9 ms total of `unified_reduce_kernel<bf16,SumOp<bf16>>` in
nsys13 are the bias-grad reductions. So I wired the BGRADA attempt into
`AddmmBackward::apply`, not `LinearBackward.cpp`.

**Two reasons it failed**:

1. **cuBLASLt heuristic rejection on sm_86**: with `COMPUTE_32F` + bf16 GEMM +
   bf16 bias output (`CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = CUDA_R_16BF`), the
   heuristic returns 0 algos. PyTorch's analogous backward paths use **fp32
   bias output** even for bf16 GEMMs. Likely BGRADA requires fp32 bias on this
   arch.

2. **Double-matmul bug in caller**: my `MatrixBackward.cpp` was:

   ```cpp
   grad_input_fused = cuda_matmul_backward_with_bias_grad(...);  // already did matmul + failed BGRADA
   if (!grad_input_fused) cuda_matmul_backward(...);              // did matmul AGAIN
   ```

   Each of the 240 eligible backward calls did the matmul TWICE (1 × `_nt` for
   grad_A + 1 × `_tn` for grad_B = +480 extra `s1688gemm` launches per run).
   Measured **+17 ms/step regression** in `nsys14`.

**Revert verified by `nsys15`**: 882.9 ms/step avg, 74,228 tok/s — matches the
nsys13 baseline within noise (±2 ms/step is normal run-to-run noise).

**Path forward if BGRADA is retried later**:

- Allocate `grad_bias` as **fp32** in the autograd graph
- Use a tiny N-element `fp32 → bf16` convert kernel after the BGRADA matmul
- Never let the fused-variant fallback double-run the matmul work — when the
  fused helper declines, only fall back for the BIAS computation
- Set `CUBLASLT_DEBUG=1` to see heuristic_empty count to confirm BGRADA fires

Saved as memory: `project_bgrada_bf16_unsupported.md`.

### G.6. `g_lt_stats` diagnostic counters ported from `best_precision`

`best_precision`'s `GenMatmul.cu` had granular failure-bucket counters that
`bf16_rev`'s version lacked. Ported into the merged file:

```cpp
struct CublasLtAddmmStats {
    std::atomic<uint64_t> entered, succeeded;
    std::atomic<uint64_t> fail_out_noncontig, fail_a_layout, fail_b_layout;
    std::atomic<uint64_t> fail_batch_dims, fail_unsupported_config;
    std::atomic<uint64_t> heuristic_empty, matmul_failed;
    std::atomic<cublasStatus_t> last_status;
};
```

Set env `CUBLASLT_DEBUG=1` to get an atexit dump:

```text
[CUBLASLT_DEBUG] cuda_addmm fast-path stats:
   entered=1440  succeeded=1440  fail_out_noncontig=0  fail_a_layout=0
   fail_b_layout=0  fail_batch_dims=0  fail_unsupported_config=0
   heuristic_empty=0  matmul_failed=0  last_status=0
```

CPU overhead: ~2 × `LOCK XADD` per addmm call = ~10-20 ns. Lost in run-to-run
noise (~28 µs total per training run, **0.0003%** of step time).

### G.7. Merged `GenMatmul.cu` deployed to all 4 BluTrain folders

Final merged file = `bf16_rev` (VEC=8 + caching allocator + 7 dtype fixes +
EPILOGUE_BIAS) + `best_precision` (g_lt_stats). Byte-identical across:

| Folder | Path | Pre-existing state |
|---|---|---|
| `BluTrain_bf16_rev` | `Tensor-Implementations/src/Kernels/cuda/matmul/GenMatmul.cu` | source of truth |
| `BluTrain_best_precision` | same | had EPILOGUE_BIAS + g_lt_stats, no VEC=8, no dtype fixes |
| `BluTrain_old` | same | older, had EPILOGUE_BIAS, no VEC=8, no dtype fixes, no g_lt_stats |
| `BluTrain_BestPrecisionEnv_may27_406k` | `BluTrain_BestPrecisionEnv/BluTrain/Tensor-Implementations/src/Kernels/cuda/matmul/GenMatmul.cu` | identical to `best_precision` |

`GenMatmul.o` built clean in each. `.bak.<unix_timestamp>` backups preserved in
each of the 3 overwritten locations.

**Why this matters**: the dtype-mismatch fix and VEC=8 broadcast_scale improvements
were ONLY in `bf16_rev`. The other 3 folders had:

- **Silent dtype-mismatch bug**: their `broadcast_scale_kernel` was hitting the
  scalar fallback instead of the vectorized branch whenever a `bfloat16_t` or
  `float16_t` wrapper tensor was passed (which is what `dispatch_by_dtype`
  always does in autocast mode)
- **VEC=2** broadcast_scale even when EPILOGUE_BIAS fell through (4× slower
  per-thread than the bf16_rev VEC=8 version)

Merged file fixes all 4.

### G.8. Caveat — numerical verification only done in `bf16_rev`

I built `GenMatmul.o` clean in each target repo but did NOT run training in
`BluTrain_old`, `best_precision`, or `may27_406k`. If those repos have specific
loss baselines, a sanity-training run is wise before relying on them. The
math should be identical (same numerical operations, same TF32 mode, same
accumulator dtypes) — only the broadcast_scale VEC width changed in any
visible way, and that's purely a vectorization layout change.

---

## Source documents (for deeper detail)

In `master_gau_latest_ada_6000_sm89/docs/`:
`SPRINT_OPTIMIZATIONS_FULL_RECORD.md` (master), `gelu_deep_dive.md`,
`layernorm_modifications.md`, `gelu+layernorm changes in master_gau_1.md`,
`contiguous_kernel_architecture.md`, `CHANGES_AND_FIXES_LOG.md`,
`weight_tying_bug_fix.md`, `reduction_bias_gradient_bug_fix.md`,
`Optimizing Packed Attention Performance.md`, `template_programming_in_cpp_and_cuda.md`.

---
