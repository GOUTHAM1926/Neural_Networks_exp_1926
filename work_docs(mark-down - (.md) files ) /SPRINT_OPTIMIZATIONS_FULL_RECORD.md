# Sprint Optimizations — Full Record (Reductions through Packed SDPA)

**Author**: Goutham Reddy (sprint owner)
**Scope**: every restructuring, kernel rewrite, redundancy elimination, and
architectural change shipped during this sprint, **in chronological order**.
The chronology is important — the sprint had two distinct phases divided
by one specific moment:

- **§1 to §8 — kernel-centric phase.** The lens was per-kernel: open a
  kernel, find where the GPU is sitting idle or under-utilised, fix it.
  Reductions (§1) was the previous sprint's foundation work; GeLU (§2),
  the activations module (§3), LayerNorm (§4), the memset cleanup (§5),
  stride-aware attention (§6), the contiguous-kernel 4-path dispatcher
  (§7), and BatchedCat / `cudaMemcpy2DAsync` removal (§8) were all this
  sprint's continuation in the same per-kernel lens.
- **§9 — the lens shift.** After BatchedCat shipped, while backtracking
  the next nsys report (specifically: "where are these residual
  `cudaMemcpyAsync` calls coming from?"), the perspective changed. The
  question stopped being *"is this kernel optimal?"* and became *"is this
  kernel's I/O necessary, given what the producer wrote and what the
  consumer wants?"* That moment is the lightbulb.
- **§10 onward — training-centric phase.** Cross-kernel redundancies that
  are invisible per-kernel but obvious in the training loop: silent
  redundant copies, shard-then-cat round-trips, layout-driven merge
  copies that no reshape rule can ever eliminate.

**Status**: each item is cross-verified against the live code in this
repository at the time of writing. Items where a colleague has since changed
something are flagged.

This document consolidates content from all the smaller per-topic markdown
files in this repo (`optimizations_and_code-changes_done_in_reductions.md`,
`contiguous_kernel_architecture.md`, `sparse_cross_entropy_optimization.md`,
`docs/CHANGES_AND_FIXES_LOG.md`, `docs/gelu_deep_dive.md`,
`docs/gelu+layernorm changes in master_gau_1.md`,
`docs/layernorm_modifications.md`, and the SDPA series) into one chronological
record so future-me (or anyone else) can re-derive the sprint's reasoning
without re-reading 11 files.

---

## Contents

**Kernel-centric phase** — open one kernel, find where the GPU is sitting idle or under-utilised, fix it. Per-kernel mindset throughout this phase:

1. [Reductions module — full-sprint overhaul (PREVIOUS sprint, foundation work)](#1-reductions)
2. [GeLU full restructure (forward + backward, CPU + GPU)](#2-gelu)
3. [Activation module restructure (ReLU, Dropout, Softmax, RMSNorm)](#3-activations-module)
4. [LayerNorm restructure + RMSNorm](#4-layernorm-rmsnorm)
5. [Memset cleanup — sync→async, redundant memsets removed](#5-memset-cleanup)
6. [Stride-aware attention kernels](#6-stride-aware-attention)
7. [Contiguous kernel — 4-path hybrid dispatcher](#7-contiguous-hybrid)
8. [BatchedCat fused kernel — eliminating `cudaMemcpy2DAsync`](#8-batchedcat)

**The lens shift** — backtracking nsys after §8 surfaced redundancies that no single-kernel view could see:

1. [The perspective shift — kernel-centric to training-centric](#9-perspective-shift)

**Training-centric phase** — eliminate cross-kernel redundancies that are only visible from the training-loop lens:

1. [`Tensor::contiguous()` short-circuit bug fix — first training-lens win](#10-contiguous-shortcircuit)
2. [`multi_tensor_zero_grad` kernel + factory function rewrites](#11-multi-tensor-zero)
3. [CachingCudaAllocator integration](#12-allocator)
4. [Sparse cross-entropy save_max/save_sum](#13-sparse-ce)
5. [Matmul: cuBLASLt port + fused-bias / GELU-bias epilogues](#14-matmul)
6. [Smart reshape (`compute_view_stride`)](#15-smart-reshape)
7. [Packed SDPA — final attention copy elimination](#16-packed-sdpa)

**Reference material:**

1. [Cross-cutting: folder layout, build, sm89 dispatch](#17-cross-cutting)
2. [Verification table — what is present in the code right now](#18-verification-table)
3. [Known divergences and follow-ups](#19-divergences)

---

<a id="1-reductions"></a>

## 1. Reductions module — full-sprint overhaul (PREVIOUS sprint, foundation work)

This was my **previous sprint's** primary work. Includes a complete rewrite of
the GPU reduction stack alongside CPU SIMD and a series of cross-framework
studies (PyTorch, TensorFlow, Eigen, CUB) before deciding what to adopt and
what to invent. Documenting it here at full depth so future-me has the
context without re-reading the 18+ analysis docs in the parent directory.

### 1.1 Pre-sprint state — what the codebase had

The legacy reduction stack was, candidly, **catastrophically naive** vs
PyTorch (per `ours_and_gpu_side_implementations_of_reductions_comparision.md`):

**4 separate kernels with ~800 lines of copy-paste:**

- `reduce_kernel` — sum / prod / min / max (the value reductions).
- `reduce_index_kernel` — argmin / argmax.
- `reduce_mean_kernel` — mean / nanmean.
- `reduce_variance_kernel` — variance / nanvariance (2-pass: mean first, then
  squared deviation).

Each was a separate template that re-implemented the same coordinate
mapping, shared-memory reduction, and accumulator logic.

**The naive offset algorithm — `unravel/ravel`:**
For each output element, the legacy kernel did:

```
for each output_index in [0, num_slices):
   out_coords = unravel(output_index, output_shape)         # O(ndim) divs
   for each i in [0, reduced_count):
      red_coords    = unravel(i, reduced_dims_shape)        # O(ndim) divs
      input_coords  = merge(out_coords, red_coords)
      input_offset  = ravel(input_coords, input_strides)    # O(ndim) muls
      acc          += input[input_offset]
   output[output_index] = acc
```

For a 4-D tensor with 1 M elements:

- Outer unravel: 1 × per output (negligible).
- **Inner unravel: O(ndim) divs/mods PER ELEMENT, for every input element**.
- For 4-D: 4 `%` + 4 `/` per element per thread × 1 M elements = **8 M
  expensive integer ops just for indexing**.

PyTorch in contrast computes a stride-based offset once per output via
`OffsetCalculator` and uses `base + i*stride` for the contiguous case — O(1)
per element.

**Other concrete problems:**

- **Fixed launch config**: `threads_per_block = 256`, `num_blocks =
  num_slices`. For a scalar reduction (full reduce → 1 output), this launches
  exactly 1 block = 256 threads. On an Ada RTX 6000 with 84 SMs × 1536
  threads = 129,024 thread slots, we used **0.2%** of the GPU.
- **Zero layout awareness on GPU**: the CPU side already had
  `compute_reduction_layout()` returning `InnerContiguous` /
  `OuterContiguous` / `Generic`. The GPU side ignored layout entirely —
  every case ran the same slow generic path.
- **No vectorized memory access**: each thread read 1 element (4 B) per
  transaction. PyTorch reads 4 elements (16 B) via `float4` / `int4`. Pure
  4× bandwidth left on the table.
- **22 × `std::find()` in OpenMP hot loops** on the CPU side, doing O(ndim)
  linear search per inner iteration to check "is this dim being reduced?".
- **Multiple `Device_Array` kernel args** for shapes / strides / axes /
  reduced_dims — N separate small PCIe pushes per launch.
- **Memory provenance**: every intermediate buffer (e.g., the mean tensor
  for variance's 2-pass) used `cudaMalloc` / `cudaFree` — driver round-trip
  per launch.
- **NaN bug in argmin/argmax**: two NaNs in input returned undefined index
  (whichever thread won the race). PyTorch's behavior: deterministic tie
  break returns lower index.

### 1.2 The cross-framework study (what I read before deciding)

Spread across the analysis docs in the parent directory:

**PyTorch** (`pytorch_reduction_deep_analysis.md`,
`pytorch_reduction_hierarchy.md`, `pytorch_cpu_reduction_variants.md`,
`binary_kernel_reduce_internals.md`):

- `TensorIterator` for both CPU and GPU.
- `OffsetCalculator<NARGS>` — pre-computes strides once, replaces unravel
  with stride-mul on the device side.
- `ReduceConfig` solver — picks block / grid / `vt0` (values per thread for
  ILP) / vectorize-input / output-vec-size dynamically per tensor shape.
- 4-stage GPU pipeline: `thread_reduce` → `block_x_reduce` (warp shuffle) →
  `block_y_reduce` (shared mem) → `global_reduce` (semaphore).
- Welford's online algorithm for variance (single pass).
- Cascade sum for FP precision without falling back to double.
- One generic kernel via functor — no copy-paste.

**TensorFlow** (`tf_reduction_deep_analysis.md`, `tf_cpu_reduction_core.md`):

- `_ReductionDims` resolves `axis=None` at graph build time → constant; the
  backend never sees None. Useful Python-level trick, not portable to our
  C++-direct path, but informed our `normalize_axes()` upfront-resolution.
- **`ReductionHelper::Simplify`** — collapses any N-D problem into a ≤3-D
  `[planes, rows, cols]` view, eliminating `unravel` / `ravel` math
  entirely. TF then uses 7 specialized 1-D / 2-D / 3-D kernels with simple
  `row * cols + col` indexing. We adopted the spirit of this via
  `ReductionLayout` (3 paths) but didn't go to 7 separate kernels.

**Eigen** (`eigen_gpu_reduction_mindmap.md`, `eigen_gpu_deep_dive.md`):

- Studied for CPU SIMD patterns. Eigen's `Tensor::reduction()` uses lazy
  evaluation + expression templates, which doesn't translate to our eager
  framework, but the SIMD packet-traits (AVX2 8-wide float lanes) approach
  was directly informative for our CPU path.

**CUB** (`cub/` source tree):

- Warp / block reduce primitives (`cub::WarpReduce`, `cub::BlockReduce`).
  Confirmed our hand-rolled `__shfl_down_sync` reduction matches CUB's
  shuffle-tree pattern. We chose to keep our hand-rolled version (no extra
  build dep) but the math is CUB-equivalent.

**Numerical stability** (`linear_double_vs_pairwise_float.md`):

- Linear sum in float32 over N ≥ 1e6 elements has ~ε·log N relative error.
- Pairwise (cascade) sum has ~ε·log log N — much tighter.
- Double-precision linear is ~ε_double·N — also fine, but 2× slower on
  consumer GPUs (no full-rate FP64). Decision: cascade in float32, skip
  double entirely.

### 1.3 The 5 reduction utilities (foundation, kept from earlier)

These are pure-CPU helpers, established in the earlier sprint and untouched
during the GPU overhaul. Documented here because they are shared by every
reduction op:

1. **`normalize_axes(input_dims, axes) → output_dims`** — converts the
   user's axes (which may contain duplicates and negative indices) into a
   clean sorted list of positive indices. Empty input axes vector means
   "reduce over all dims". Uses `std::set` for sort + dedup. Validates
   bounds against `ndim`.

2. **`calculate_output_shape(input_dims, normalized_axes, keepdim)`** —
   loops every input dim; if the dim is reduced and `keepdim=true`, push
   `1`; if reduced and `keepdim=false`, skip; otherwise push the dim size
   verbatim. Empty result → push `1` for scalar output.

3. **`calculate_reduced_count(input_dims, normalized_axes)`** — used only
   by `mean` / `nanmean`. Returns the divisor: `Π input_dims[d] for d in
   normalized_axes`. (Empty `normalized_axes` → product of *all* input
   dims, full reduction case.)

4. **`unravel_index(linear_idx, shape) → coords`** — successive division
   and modulo (C-order / row-major). Backward iteration over dims:
   `coords[i] = temp % shape[i]; temp /= shape[i]`. Used by the legacy
   reduction kernel; deprecated in the new GPU path but retained for the
   CPU `Generic` fallback.

5. **`ravel_index(coords, strides) → linear_idx`** — the inverse:
   `Σ coords[i] * strides[i]`. Used everywhere a multi-dim coordinate
   needs to be turned into a memory offset.

These live in
[`src/UnaryOps/cpu/ReductionUtils.cpp`](../src/UnaryOps/cpu/ReductionUtils.cpp)
and
[`include/ops/helpers/ReductionUtils.h`](../include/ops/helpers/ReductionUtils.h)
in the current code.

### 1.4 The 8 things that shipped this sprint

Listed in roughly the order they were applied:

#### 1.4.1 Layout bifurcation on GPU (PyTorch-style 3 paths)

Brought the GPU's layout intelligence up to par with the CPU. The dispatcher
now picks one of three paths:

- **`InnerContiguous`** — reduction axis is the fastest-striding dim
  (e.g. `sum` over the last dim of `[B, T, C]`). Maps `block.x` to the
  reduction dim → warp-shuffle reduction is natural and free.
- **`OuterContiguous`** — reduction axis is an outer dim (e.g. `sum` over
  the first dim of `[B, T, C]`). Maps `block.x` to the output dim →
  coalesced writes; reduction happens vertically across `block.y`.
- **`Generic`** — neither cleanly inner nor outer (mixed strides, multiple
  reduction axes that aren't memory-adjacent). Falls back to the
  stride-based offset calculator.

Path enum at
[`ReductionKernels.cuh:148`](../include/ops/helpers/ReductionKernels.cuh)
with the per-path branches at lines 361 (`InnerContiguous`) and 380
(`OuterContiguous`).

#### 1.4.2 Single unified `unified_reduce_kernel`

Replaced the 4 copy-pasted kernels with one templated kernel:
[`ReductionKernels.cuh:332`](../include/ops/helpers/ReductionKernels.cuh).
Template parameters:

```cpp
template <
    detail::ReductionLayout::Path PATH,    // InnerContiguous / OuterContiguous / Generic
    int NT,                                // num_threads_per_block
    int VT0,                               // values per thread (ILP)
    typename ops_t                         // SumOp, MinOp, ArgMinOp,
                                           // MeanOps, WelfordOps, ...
>
__global__ void unified_reduce_kernel(detail::ReduceOp<...> op);
```

The functor `ops_t` carries the operation semantics (combine, identity,
project) — same trick PyTorch uses. Net code: ~250 lines instead of
~800.

#### 1.4.3 Multi-stage launch dispatcher

Block size is no longer fixed at 256. The dispatcher at
[`ReductionKernels.cuh:486-501`](../include/ops/helpers/ReductionKernels.cuh)
picks `nt` from `{32, 64, 128, 256, 512}` based on the reduction's `nt_val`
and the chosen layout path. Smaller `nt` gives more blocks (more SMs
covered for small reductions); larger `nt` gives more shared-memory
reduction depth (better for huge inner reductions).

#### 1.4.4 Cascade sum (4-level bucket) for FP precision

`cascade_sum_kernel` at
[`ReductionImpl.h:210-219`](../include/ops/helpers/ReductionImpl.h)
implements pairwise summation in float32 — no double-precision needed.
Algorithm: maintain 4 (or N) accumulator buckets, fold pairs in tree style.
Achieves ε·O(log log N) error vs naive ε·O(log N), without the 2× cost of
FP64 on consumer GPUs.

#### 1.4.5 Welford one-pass for variance

Variance forward used to require 2 GPU launches (mean kernel, then squared
deviation kernel) plus an intermediate `[output_shape]` mean buffer. New
path uses Welford's online algorithm via a `WelfordOps` functor — single
pass, no intermediate, same numerics. Update rule:

```
n_new       = n + 1
delta       = x - mean
mean_new    = mean + delta / n_new
M2_new      = M2 + delta * (x - mean_new)
variance    = M2 / n
```

Kernel reuses `unified_reduce_kernel` with `WelfordOps` — no separate
variance kernel anymore (same trick the LayerNorm sm89 fwd uses, §4).

#### 1.4.6 O(1) bitmap lookup replacing 22 × `std::find()`

The legacy CPU dispatchers (`reduce_kernel`, `dispatch_mean_kernel`,
`dispatch_variance_kernel`) had 22 occurrences of `std::find(axes.begin(),
axes.end(), i)` *inside* OpenMP-parallel hot loops. For a tensor with
`ndim=8` reducing 4 axes, each find is up to 4 comparisons — and it
happens millions of times per call. Replaced with a precomputed
`bool reduced_bitmap[MAX_DIMS]` array — O(1) lookup per check. See
[`ReductionImpl.h:235`](../include/ops/helpers/ReductionImpl.h):
`bool reduced_bitmap[MAX_DIMS] = {false};` — populated once before the
loop, indexed `reduced_bitmap[d]` thereafter. Cut inner-loop overhead
significantly on multi-axis reductions.

#### 1.4.7 Packed metadata + caching allocator integration

- Replaced multiple `Device_Array` kernel args (one each for shape,
  stride, axes, reduced_dims, output_dims) with a single
  `Packed_Meta_Data` struct passed via `__grid_constant__`. One PCIe
  push per launch instead of N.
- Routed every intermediate buffer (variance's mean buffer, cascade's
  partial sums, etc.) through our **caching allocator** (§12). No more
  `cudaMalloc` / `cudaFree` per launch — driver round-trip eliminated.

#### 1.4.8 NaN-deterministic argmin/argmax + dead code removal

- **NaN tie-break**: input with two NaNs no longer returns whichever
  thread wins the race. Explicit check at the start of the comparator
  returns the **lower index** deterministically (matches PyTorch
  behavior and our CPU implementation).
- Removed unreachable `reduced_count == 0 && numel > 0` checks already
  validated upstream by `normalize_axes`.
- Removed `Tensor::zeros` allocations for buffers that the next kernel
  fully overwrites — same training-script-lens trick as §6.

### 1.5 What the current code looks like

Top-level files in the live repo:

| File | Role |
|---|---|
| [`include/ops/helpers/ReductionKernels.cuh`](../include/ops/helpers/ReductionKernels.cuh) | `unified_reduce_kernel:332`, dispatcher:486, `InnerContiguous`/`OuterContiguous` per-path branches:148/361/380 |
| [`include/ops/helpers/ReductionImpl.h`](../include/ops/helpers/ReductionImpl.h) | Host-side pipeline. `cascade_sum_kernel:210`, `ceil_log2:129`, `reduced_bitmap:235`, layout dispatch logic |
| [`include/ops/helpers/ReductionUtils.h`](../include/ops/helpers/ReductionUtils.h) | The 5 utility helpers (normalize_axes, output_shape, reduced_count, unravel, ravel) |
| [`include/ops/helpers/ReductionOps.h`](../include/ops/helpers/ReductionOps.h) | Public op-level decls (sum/prod/mean/min/max/var/argmin/argmax) |
| [`src/UnaryOps/cuda/ReductionImplGPU.cu`](../src/UnaryOps/cuda/ReductionImplGPU.cu) | GPU dispatcher (renamed from `ReductionImpl.cu`) |
| [`src/UnaryOps/cpu/Reduction.cpp`](../src/UnaryOps/cpu/Reduction.cpp) | CPU side (kept naive but with `reduced_bitmap` and `compute_reduction_layout`) |
| [`src/UnaryOps/cpu/ReductionUtils.cpp`](../src/UnaryOps/cpu/ReductionUtils.cpp) | The 5 utilities, definitions |
| [`src/autograd/operations/ReductionOps.cpp`](../src/autograd/operations/ReductionOps.cpp) | Autograd wrappers (3-layer pattern: nn → autograd → math engine) |
| [`src/autograd/backward/ReductionBackward.cpp`](../src/autograd/backward/ReductionBackward.cpp) | Backward nodes (broadcast forward grad to all input slots) |

### 1.6 What's deferred to future work

Per `new_implementation_plan_of_reduction_ops_on_gpu_side.md`, the **fully
PyTorch-equivalent stack** would also include:

- **`OffsetCalculator` for fully-strided tensors** — currently the
  `Generic` path still uses `unravel/ravel`. Replacing it with a
  pre-computed offset calculator would close the last O(ndim) divmod hot
  spot.
- **Semaphore-based global reduction** for huge tensors — single kernel
  launch with cross-CTA finalization. We currently do multi-launch (or
  single-block-with-large-nt) which is fine up to ~1 M-element reductions
  but not optimal beyond that.
- **Vectorized input loads (`float4`)** — the GPU paths still read 1
  element per thread per transaction. 4× bandwidth available.
- **Output vectorization for outer-reductions** — multiple outputs per
  thread when reducing inner dim of `[B, T, V]`-style shapes.

These were judged out-of-scope for the current sprint; the layout
bifurcation + unified kernel + cascade + Welford gave us the bulk of the
PyTorch-parity gains without the implementation surface area.

### 1.7 Forward / backward design philosophy

(From `sdpa_architecture_mindmaps.md`, applies to reductions:)

- **Forward** = "calculative worker / funnel" — many inputs collapse to
  one output. Reductions via `__shfl_down_sync`. Follows `f(x)`.
- **Backward** = "diagnostic accountant / paintbrush" — broadcasts one
  gradient to many inputs. No reduction math; just `dL/dx_i = dL/dy ·
  ∂y/∂x_i`.

For sum: forward warp-shuffles thousands of values into one; backward
broadcasts the incoming `grad_out` to **every** input slot (no reduction
needed). For mean: backward broadcasts `grad_out / N` instead. For
argmin/argmax: backward writes `grad_out` only to the index that won;
zeros elsewhere. The backward node files
([`src/autograd/backward/ReductionBackward.cpp`](../src/autograd/backward/ReductionBackward.cpp))
are tiny — most of the complexity is in the forward kernels.

---

<a id="2-gelu"></a>

## 2. GeLU full restructure (forward + backward, CPU + GPU)

### Problem

Activations module mixed autograd glue with math kernels. CPU paths were
naive (multiple passes per element, no SIMD). GPU paths existed but `nn::GeLU`
called the autograd op directly, so the math wasn't reusable. Two legacy bug
sources lived in `include/mlp/` and `src/mlp-blocks/` (e.g. wrong sign on
`0.044715`).

### What changed

Adopted a uniform **3-layer pattern** that is now used across the codebase:

```
nn::GeLU.forward()         (NN module — public entry)
   ↓
autograd::gelu()           (thin wrapper — saves tensor, attaches grad node)
   ↓
gelu_forward()             (pure math — CPU AVX2 OR GPU dispatch)
```

New files:

- [`include/Activations.h`](../include/Activations.h) (98 lines) — 8 fwd + 7 bwd decls.
- [`Activations.cpp`](../Activations.cpp) (~1170 lines) — AVX2 CPU paths and the GPU dispatcher.

Modified:

- [`src/autograd/operations/ActivationOps.cpp`](../src/autograd/operations/ActivationOps.cpp): 400 → 179 lines (autograd-only).
- [`src/autograd/backward/ActivationBackward.cpp`](../src/autograd/backward/ActivationBackward.cpp): 277 → 90 lines.
- `Vectorized.h`: added AVX2 `tanh()` (Cephes rational `P(z)/Q(z)` Horner with FMA, blendv ±1 clamping), AVX2 `abs()` via `andnot`, fp16/bf16 F16C helpers.
- `ActivationKernels.cu`: ReLU rewritten as `(x + fabsf(x)) * 0.5f` (branch-free, NaN-propagating); `fused_bias_gelu` templated for fp16/bf16.

Deleted: `include/mlp/`, `src/mlp-blocks/` (legacy with bugs).

### Eight kernel paths

| Path | Status | Key optimizations |
|---|---|---|
| `gelu_fwd` CPU | rewritten | single-pass fused, AVX2 8-wide, vectorized tanh, FMA, 2× unroll, OMP threshold 16384, fp16/bf16 F16C |
| `bias_gelu_fwd` CPU | new from scratch | (was throwing); same 7 optimizations |
| `gelu_fwd` GPU | kept (already good) | `fused_gelu_kernel_vectorized` with `float4` + `fast_tanh` PTX |
| `bias_gelu_fwd` GPU | extended | templated for fp16/bf16 (was fp32-only) |
| `gelu_bwd` CPU | extracted + AVX2 fused | single-pass fused |
| `bias_gelu_bwd` CPU | new from scratch | thread-local bias accumulators |
| `gelu_bwd` GPU | kept | `fast_tanh` + `unroll 4` |
| `bias_gelu_bwd` GPU | kept | two-pass: element-parallel grad_input + shared-mem bias reduction |

### Where it lives now

- sm89 variants: [`src/Kernels/cuda/activations/arch/GELUKernels_sm89.cu`](../src/Kernels/cuda/activations/arch/GELUKernels_sm89.cu)
  - `gelu_forward_sm89_kernel:61`, `gelu_backward_sm89_kernel:107`.
- Generic dispatch: [`src/Kernels/cuda/activations/`](../src/Kernels/cuda/activations/).

### Numbers (from `docs/gelu_deep_dive.md`)

At `[8,1024,1536]`: GeLU CPU 3.26ms (PT 0.88ms — still gap), GPU 0.33ms (PT 0.09ms).
At small `[8,1024,384]`: 5.7–14× faster than PyTorch on CPU, 19–48× vs TF;
GPU 1.2–2× faster than PT, 1.3–1.4× vs TF.

---

<a id="3-activations-module"></a>

## 3. Activation module restructure (ReLU, Dropout, Softmax, RMSNorm)

Same 3-layer pattern applied to the rest of the activation/normalization
suite. After GeLU shipped, ReLU, Dropout, and Softmax got the same
restructure (`Activations.cpp` houses them all). RMSNorm was implemented from
scratch using PyTorch's `bool rms_norm` template-parameter trick — the same
LayerNorm CUDA kernel reused with `if constexpr (rms_norm) { ... }` paths,
zero runtime cost (compiles to separate PTX).

Deliverables: `nn::ReLU`, `nn::Dropout`, `nn::RMSNorm` modules; full forward
and backward; CPU AVX2 plus GPU sm89. Verified at lines 159–180 of
[`GELUKernels_sm89.cu`](../src/Kernels/cuda/activations/arch/GELUKernels_sm89.cu)
and sibling files.

---

<a id="4-layernorm-rmsnorm"></a>

## 4. LayerNorm restructure + RMSNorm

### Before state

- **CPU forward** (lambda inside `NormalizationOps.cpp:93-124`): three scalar
  passes per row (mean, variance, normalize). No SIMD. No Welford. OpenMP
  only over rows.
- **CPU backward**: four scalar passes; gamma/beta serial (race risk),
  `std::vector<float>` accumulator.
- **GPU forward**: already good — fused 6-phase Welford one-pass with
  `float4`/`half2`/`bf16x4` vectorized loads, warp shuffle, shared-mem block
  reduce, `AccT=float` for fp16/bf16.
- **GPU backward**: 2 kernels structurally (column reduce vs row reduce
  cannot share). `gamma_beta` 2D block (32×8) with shared `s_dgamma[8][32]` +
  `s_dbeta[8][32]` and atomicAdd to global; `input` Pass A (sum1, sum2 via
  warp + block reduce) → Pass B (grad_x scalar, `#pragma unroll 4`). Backward
  did **not** have vectorized loads (PT proved 10–20% slower with them).

### What changed

- 3-layer split: [`Normalizations.h`](../include/Normalizations.h),
  [`Normalizations.cpp`](../Normalizations.cpp).
- **CPU forward**: 3 passes → 2 (Welford one-pass) + AVX2 SIMD + FMA.
- **CPU backward**: gamma/beta now OMP parallel with thread-local buffers +
  AVX2; input grad AVX2 vectorized.
- **GPU**: fused `bool rms_norm` template across forward kernel; vectorized
  GPU backward via `float4`; RMSNorm fully implemented (CPU+GPU, fwd+bwd, 3
  dtypes); `__launch_bounds__(512)` on sm89.
- Welford one-pass replaces the old two-pass; same numerics.

### Where it lives now

- [`src/Kernels/cuda/norm/arch/LayerNormKernels_sm89.cu`](../src/Kernels/cuda/norm/arch/LayerNormKernels_sm89.cu)
  - `layer_norm_forward_sm89_kernel:63`
  - `ln_backward_gamma_beta_sm89_kernel:250`
  - `ln_backward_input_sm89_kernel:289`
  - dtype dispatchers: 366–391.
- Generic: [`src/Kernels/cuda/norm/LayerNormKernels.cu`](../src/Kernels/cuda/norm/LayerNormKernels.cu) (with `bool rms_norm` template forward, float4 vec backward).

### Numbers (from `docs/layernorm_modifications.md`)

RTX 3060, fp32, `[8,1024,384]`:

- CPU fwd 1.728ms → 0.141ms (**12.3×**).
- CPU bwd 25.94ms → 4.637ms (**5.6×**).
- GPU fwd 0.113ms → 0.106ms (~same — same PTX emitted).
- GPU bwd ~20% faster from float4 vec.
- vs PyTorch: LN fwd GPU tied, LN bwd 1.16×, RMS fwd 3.7×, RMS bwd 4.0× faster.
- vs TensorFlow: LN fwd 4.4×, LN bwd 9.2×, RMS bwd 5.9× faster.

---

<a id="5-memset-cleanup"></a>

## 5. Memset cleanup — sync → async, redundant memsets removed

### Two related issues

1. **Sync `cudaMemset` blocking the launch queue.** Every blocking memset
   serialized the stream. Replaced everywhere with `cudaMemsetAsync(...,
   stream)`.
2. **Memsets on tensors that the next kernel was going to fully overwrite
   anyway.** Pure waste. Found via training-script lens.

### Specific fixes shipped

- LayerNorm backward `grad_input_kernel`: removed three redundant
  `cudaMemset` calls because the second LN-bwd kernel
  (`gamma_beta_kernel`) already zeros those gradient tensors before its
  reduction. Saved ~205ms over a run; ~25K memsets removed.
- Attention `dQ` tensor: `Tensor::zeros` → `Tensor::empty` because the
  unified backward kernel writes every byte. Saved ~240ms; ~4K memsets removed.
- LayerNorm scratch buffers: sync `cudaMemset` → `cudaMemsetAsync(stream)`.

### Current state

`src/` now contains exactly one `cudaMemset` token, in a *commented-out*
legacy block at
[`src/core/TensorFactory.cpp:211`](../src/core/TensorFactory.cpp). All 46+
active occurrences are `cudaMemsetAsync`. Verified by:

```bash
grep -rn "cudaMemset[^A]" src/    # → only the comment in TensorFactory.cpp
```

The Attention dQ fix is at
[`src/autograd/backward/AttentionBackward.cpp:107-109`](../src/autograd/backward/AttentionBackward.cpp).
Old `Tensor::zeros(...)` retained only as a comment at lines 39–41 for
historical context.

---

<a id="6-stride-aware-attention"></a>

## 6. Stride-aware attention kernels

### Problem

Attention forward and backward called `q.contiguous()` / `k.contiguous()` /
`v.contiguous()` before launching the kernel. For compute-bound ops like
attention, the GPU is already busy on the math — the latency from strided
loads is hidden by compute. The `.contiguous()` copy was pure
waste — ~17K kernels per nsys run, ~2 seconds of wall time.

### What changed

Modified the kernel signatures to accept stride parameters per tensor and
read with strided indexing. No more pre-kernel `.contiguous()`. Files
touched: `AttentionCommon.cuh`, `AttentionForward.cu`, `AttentionBackward.cu`,
`arch/AttentionForward_sm89.cu`, `arch/AttentionBackward_sm89.cu`,
`AttentionOps.cpp`, `AttentionBackward.cpp`, `gpt2_attn_navin.cpp`.

### Where it lives now

[`include/ops/helpers/AttentionKernels.h`](../include/ops/helpers/AttentionKernels.h):

- `mem_efficient_attn_forward(...)` line 17: takes `q_strideB, q_strideM,
  q_strideH` and the same triplet for K, V, O, LSE.
- `mem_efficient_attn_forward_tc(...)` line 29: tensor-core variant, same
  stride layout.
- `mem_efficient_attn_backward(...)` line 53: stride params for Q/K/V/O/dO
  *and* for dQ/dK/dV (`dq_strideB`, `dk_strideB`, `dv_strideB`, etc.).

The 4320 fwd / 3840 bwd kernel-count asymmetry seen in nsys is expected
(`20 val-steps × 12 layers × 2`), not a bug.

---

<a id="7-contiguous-hybrid"></a>

## 7. Contiguous kernel — 4-path hybrid dispatcher

(See `contiguous_kernel_architecture.md` for the full discussion.)

### Why a single path is wrong

Plain `generic_strided_copy_kernel` with FastDivmod is the best **generic
fallback** but it's expensive for cases that don't need it. Three common
patterns each have a specialized path that is dramatically faster:

| Pattern | Best path |
|---|---|
| Already fully contiguous post-coalesce | `cudaMemcpyAsync` D2D (DMA) |
| 2D transpose `[a,b]` with strides `[1,a]` | Tiled shared-mem kernel |
| Strided outer + dense inner row | Vectorized inner kernel |
| Anything else | Generic fallback |

### The 4-path dispatch (`src/Views/ContiguousKernel.cu`)

```
 contiguous_strided_copy_cuda
        │
        ├── coalesce_dimensions()        ── fold adjacent contig dims
        │
        ├── (3a) is_fully_contiguous → cudaMemcpyAsync D2D (~890 GB/s, 0 SMs)
        │
        ├── (3b) is_2d_transpose(rows>=16, cols>=16) →
        │        transpose_2d_tiled_kernel<T,32,8>
        │        (tile[32][33] +1-padded; coalesced reads AND writes)
        │
        ├── (3c) inner-stride==1 + 16-byte aligned + n_vec >= 128 →
        │        strided_inner_vec_copy_kernel<float4|uint2>
        │        (STG.128 16-byte loads; one block per outer index)
        │
        └── (3d) generic_strided_copy_kernel (FastDivmod, 4-way coarsening)
```

`ContiguousMeta` (~216 B) is passed as `__grid_constant__`. FastDivmod
replaces `%`/`/` (~40 cycles) with `__umulhi + shift + correction` (~6
cycles, ~6× speedup).

### vs PyTorch and TensorFlow

- vs PyTorch: tied for the generic + DMA + vec-inner cases; we add the tiled
  2D transpose path that PT calls into a separate util for.
- vs TensorFlow: we beat TF at runtime via FastDivmod + vec-inner; TF+XLA
  wins via AOT graph elimination (we don't try to compete on graph-level
  optimization).

### Where it lives now

[`src/Views/ContiguousKernel.cu`](../src/Views/ContiguousKernel.cu):

- `is_fully_contiguous:281`, `is_2d_transpose:256`, `coalesce_dimensions:233`.
- `strided_inner_vec_copy_kernel:138`, `generic_strided_copy_kernel:182`.
- Dispatch: `coalesce_dimensions:339`, memcpy fast path:348, transpose:359,
  vec copy:422, generic:459.

Saved ~380 ms (vec-inner) + ~4 ms (transpose) per nsys run.

### Bug history fixed during this work

- Broken 3D batched check (was always false).
- Dead `vectorized_contiguous_copy_kernel` removed (~90 lines).
- `MaxDims=12` template / struct=10 mismatch — silently dropped dims 10–11.
  Now consistent `kMaxContigDims=10` everywhere; loud-fail beats truncate.
- Vec kernel mis-calibration (12.5% utilization regression) gated by
  `n_vec >= InnerThreads(128)` so it never fires when it would lose.
- Grid-Y 65535 cap — outer index moved to gridDim.x.

---

<a id="8-batchedcat"></a>

## 8. BatchedCat fused kernel

### Problem

`Tensor::cat` was issuing one `cudaMemcpy2DAsync` (or `cudaMemcpyAsync`) per
input tensor — 3 calls for a 3-way QKV concat in attention backward. Each
goes through the driver and uses DMA, which is slower than just having
threads write to the destination directly when source and destination are on
the same device.

### What changed

Implemented `cat_batched_kernel` (PyTorch-style `CatArrayBatchedCopy`) that
concatenates many tensors in one launch.

### Where it lives now

[`src/Kernels/cuda/misc/BatchedCat.cu`](../src/Kernels/cuda/misc/BatchedCat.cu):

- `cat_batched_kernel` definition: line 48.
- Dispatched for `uint8/16/32/64` and `uint4` widths: lines 163–181.

(Note: file is in `misc/`, not `cat/`, despite some older docs implying
otherwise.)

This kernel is what later got eliminated entirely from attention backward
once Packed SDPA shipped (see §16).

---

<a id="9-perspective-shift"></a>

## 9. The perspective shift — kernel-centric to training-centric

Up through §8 (BatchedCat / `cudaMemcpy2DAsync` removal) the lens was still
**per-kernel**: open a kernel, find where the GPU is sitting idle or
under-utilised, fix it. Reductions (§1, prev sprint), GeLU (§2), activations
module (§3), LayerNorm (§4), memset cleanup (§5), stride-aware attention
(§6), the contiguous-kernel 4-path dispatcher (§7), and BatchedCat (§8) all
fit that lens — fill GPU idle gaps, avoid wasted compute, vectorize loads,
remove redundant memsets, etc. Each was a kernel-local optimization.

The lens shifted **after BatchedCat shipped**, while backtracking the next
nsys report. The question that triggered it was concrete: *"BatchedCat
killed all `cudaMemcpy2DAsync` calls — but `cudaMemcpyAsync` calls are
still showing up. Where are they coming from?"* Backtracking those calls
through the training-script call graph (not into a single kernel) led
straight to the silent bug in `Tensor::contiguous()` — a function that was
allocating a new buffer and copying data even when the input was already
contiguous (covered next, §10).

That moment was the lens shift. The question stopped being *"is this
kernel optimal?"* and became *"is this kernel's I/O necessary, given what
the producer wrote and what the consumer wants?"*

Many redundancies are invisible from a single-kernel view but obvious from
a training-loop view:

- A kernel that memsets a tensor that the *previous* kernel in the loop
  already wrote into.
- A `.contiguous()` copy whose result the consumer kernel could have read
  with strides anyway.
- A `Tensor::zeros` allocation whose every byte will be overwritten by the
  very next kernel.
- A `Tensor::cat` that recombines three gradients which were just split
  a moment earlier — when the consumer kernel could have read them via
  pointer offsets.

The rule of thumb that emerged from this lens:
**if a kernel's first write covers every byte of its output, the
allocator should hand it dirty memory (`Tensor::empty`), not zeroed
memory**. And more generally: don't ask "is this kernel optimal in
isolation?" — ask "is this kernel's I/O necessary, given what the
producer wrote and what the consumer wants?"

Almost every win from §10 onward came from that shift in lens, not from
any deeper kernel-level micro-optimization.

---

<a id="10-contiguous-shortcircuit"></a>

## 10. `Tensor::contiguous()` short-circuit bug fix — first training-lens win

### Problem (the discovery moment for the lens shift, see §9)

After §8 (BatchedCat) eliminated `cudaMemcpy2DAsync` calls, residual
`cudaMemcpyAsync` calls **still showed up in nsys** — surprising, since
BatchedCat had been fused. Backtracking those calls through the training
call graph (not into a single kernel — that's the new lens) led to
`Tensor::contiguous()` in `src/core/Tensor.cpp`. The old code allocated a
fresh tensor and copied data into it **even when the input tensor was
already contiguous**. Pure redundant copy. This was the first concrete
"training-perspective lens" win, and was what triggered the broader shift
described in §9.

### Fix

Added an early return: if `is_contiguous() && storage_offset() == 0`, return
`*this` (a view of the original) instead of allocating + copying.

### Where it lives now

[`src/core/Tensor.cpp:283-285`](../src/core/Tensor.cpp):

```cpp
if (is_contiguous() && storage_offset() == 0) {
    return *this;
}
```

Sits before the `contiguous_strided_copy_cuda` call at line ~337.

---

<a id="11-multi-tensor-zero"></a>

## 11. `multi_tensor_zero_grad` kernel + factory function rewrites

### Problem

Two issues feeding each other:

**(a)** `Optimizer::zero_grad` on GPU did a CPU round-trip:

```
allocate vector<float> on CPU
memset that to 0
cudaMemcpy(grad_gpu, cpu_zeros, size) ← per-parameter
```

For 124M parameters split across 200+ tensors, this is 200+ host allocations
- 200+ memsets + 200+ H2D copies. Awful.

**(b)** Even if we used `cudaMemsetAsync` instead, doing it per-tensor in a
loop has launch overhead for hundreds of small ops.

### What changed

Designed and shipped a multi-tensor approach in the spirit of the existing
`multi_tensor_adam_kernel`:

- **`multi_tensor_zero_sm89_kernel`** — one kernel launch zeros up to N
  tensors at once; metadata struct contains pointer + length per tensor.
- **`fill_cuda_launch<T>`** — generic GPU fill (used for `Tensor::ones`,
  `Tensor::full`).
- Three vectorized fill paths in
  [`src/Kernels/cuda/misc/FillKernel.cu`](../src/Kernels/cuda/misc/FillKernel.cu):
  - `fill_kernel_vec16` (line 52) — 16-byte (float4 / int4) writes.
  - `fill_kernel_vec8` (line 65) — 8-byte writes.
  - `fill_kernel_scalar` (line 76) — fallback.
  - Host launcher `fill_cuda_launch<T>:98` with template instantiations for
    float/double/half/bf16/int8-64/uint8-64.
- **Factory function rewrites** in
  [`src/core/TensorFactory.cpp`](../src/core/TensorFactory.cpp):
  - `Tensor::zeros` (line 172) uses `cudaMemsetAsync` directly on GPU
    (no CPU round-trip; line 184).
  - `Tensor::ones` (line 275) and `Tensor::full` (line 360) use
    `fill_cuda_launch`.
  - `set_data` / `set_grad` / `fill_grad` (declared at
    [`include/core/Tensor.h:240,261`](../include/core/Tensor.h),
    instantiated in `src/core/Tensor.cpp:1034+`) all do GPU-side fills.

### Optimizer integration

[`src/nn/optimizer/Optim.cpp`](../src/nn/optimizer/Optim.cpp):

- `Optimizer::zero_grad()` line 29.
- Loop at lines 43–53 collects GPU grads into `std::vector<cuda::ZeroTensorInfo>`.
- Line 55: `cuda::multi_tensor_zero_cuda(gpu_grads);` — single launch.
- CPU grads still use per-param `p.zero_grad()` (CPU-side, not the
  bottleneck).

### Kernel implementation pointers

- [`src/Kernels/cuda/optimizer/arch/MultiTensorKernels_sm89.cu`](../src/Kernels/cuda/optimizer/arch/MultiTensorKernels_sm89.cu):
  - `multi_tensor_zero_sm89_kernel` line 343, `__launch_bounds__(256, 2)`.
  - Wrapper `multi_tensor_zero_sm89_cuda` line 384, launched line 426.
- [`src/Kernels/cuda/optimizer/MultiTensorKernels.cu:314`](../src/Kernels/cuda/optimizer/MultiTensorKernels.cu) — public dispatcher `multi_tensor_zero_cuda`.
- [`include/ops/helpers/MultiTensorKernels.h:80,87`](../include/ops/helpers/MultiTensorKernels.h) — header decls.

### Bug found and fixed during this work

[`TensorFactory.cpp`](../src/core/TensorFactory.cpp) `ones()` for `bool`
dtype was writing `0x01` to *only the first byte* of the buffer — because
`std::vector<bool>::data()` doesn't exist (bitset specialization). Fixed by
special-casing `bool` to use `cudaMemsetAsync(grad(), value ? 1 : 0, ...)`
on the full buffer.

---

<a id="12-allocator"></a>

## 12. CachingCudaAllocator integration

### Problem

`cudaMalloc` and `cudaFree` are synchronous and round-trip the CUDA driver
every time. Even `cudaMallocAsync` / `cudaFreeAsync` have non-trivial driver
cost. Hot allocations every step are a known throughput killer.

### What changed

Routed every device allocation in compute paths through our own caching
allocator:

- [`include/device/CachingCudaAllocator.h`](../include/device/CachingCudaAllocator.h) (header at line 210 documents the wrapper).
- [`src/device/CudaCachingAllocator.cpp:370`](../src/device/CudaCachingAllocator.cpp): `cudaMallocAsync(&ptr, size, stream)`.
- [`src/device/AllocatorRegistry.cpp:5`](../src/device/AllocatorRegistry.cpp) — registration.
- [`src/device/CUDAAllocator.cpp:17`](../src/device/CUDAAllocator.cpp) — also `cudaMallocAsync`.
- No raw `cudaMalloc` (non-async) calls remain in the device layer.

The reduction module uses this allocator for intermediate buffers (§5).

### Tensor.cpp allocator cleanup

Removed two `cudaStreamSynchronize` calls from the free path; rely on
event-based reuse safety instead. Per `docs/CHANGES_AND_FIXES_LOG.md`.

---

<a id="13-sparse-ce"></a>

## 13. Sparse cross-entropy save_max/save_sum

(Full math discussion in `sparse_cross_entropy_optimization.md`.)

### Problem

Sparse cross-entropy needs `max` and `sum` per row (across vocab) for both:

- Forward: `loss = log(sum) + max - x_target`.
- Backward: `grad_j = (exp(x_j - max) / sum - 1{j==target}) * scale`.

Old layout:

- Forward (`sparse_ce_forward_kernel_vec`) computed `max` and `sum`,
  produced loss, **discarded** them.
- Backward did 2 kernels: `sparseCEReduce_kernel_optimized` recomputed
  `max`/`sum` (full vocab read again), `sparseCENormalize_kernel_optimized`
  read logits a third time and produced grad.

Total: **3 × 3.3 GB = ~9.9 GB of logit traffic per backward step**.

### What changed

- `sparse_ce_forward_kernel_vec_save_stats` writes `saved_max[B]` +
  `saved_sum[B]` (~131 KB extra) at zero compute cost (already had them).
- Reduce kernel deleted entirely.
- New `sparseCENormalize_from_stats` reads logits once.

Total: **2 × 3.3 GB = 6.6 GB**. Saved ~3.7 ms per micro-batch. Net ~1.1%
wall time over 32 micro-batches × 10 steps.

### vs PyTorch

PyTorch's `cross_entropy = log_softmax + nll_loss` must materialize the full
3.3 GB log_softmax tensor (because of the Python-level public API contract)
and save it for backward. Ours is a monolithic autograd node, so we save
only the stats — about half PyTorch's memory traffic. Same trick
FlashAttention and Apex use.

### Where it lives now

[`src/Kernels/cuda/loss/LossKernels.cu`](../src/Kernels/cuda/loss/LossKernels.cu):

- `sparse_ce_forward_kernel_vec_save_stats:188`
- `sparseCENormalize_from_stats:706`
- Host wrappers launch them at lines 913 and 953.

### Decisions noted in `sparse_cross_entropy_optimization.md`

- Saved tensors are shape `[B]` not `[V]` (reduction is across vocab).
- `sum_reduction_kernel` (2.7 µs) kept separate; replacing with
  `unified_reduce_kernel` adds 70 µs overhead, fusing via atomicAdd causes
  contention adding 50 µs. Total `sum_reduction_kernel` time across run is
  1.94 ms = 0.0018% — not worth optimizing further.

---

<a id="14-matmul"></a>

## 14. Matmul: cuBLASLt port + fused-bias / GELU-bias epilogues

(Full discussion in `docs/CHANGES_AND_FIXES_LOG.md`.)

### Problem

We were on legacy `cublasGemmEx` (`cutlass::Kernel`); PyTorch was on
`cublasLtMatmul` (`cutlass::Kernel2`). Two issues:

- Legacy lacks epilogue fusion — bias addition was a separate
  `add_kernel_nd_broadcast` launch after every matmul.
- Legacy has higher launch overhead (`cudaLaunchKernel` ~376 µs vs PT 9 µs).

### What changed

New helper:

- [`include/ops/helpers/CublasLtHelper.h`](../include/ops/helpers/CublasLtHelper.h)
- [`src/Kernels/cuda/matmul/CublasLtHelper.cu`](../src/Kernels/cuda/matmul/CublasLtHelper.cu)
  - per-device cached `cublasLtHandle_t`
  - 32 MiB workspace
  - `lt_gemm_impl` builds desc with `CUBLAS_COMPUTE_32F_FAST_TF32`,
    calls `cublasLtMatmulAlgoGetHeuristic` → `cublasLtMatmul`
  - returns `bool` so callers can fall back

Call-site ports:

- [`src/Kernels/cuda/matmul/GenMatmul.cu`](../src/Kernels/cuda/matmul/GenMatmul.cu)
  — 5 FP32-TF32 sites: fwd single, fwd strided batched, fwd 4D loop, addmm
  single, addmm batched. **Bias fusion**: `CUBLASLT_EPILOGUE_BIAS` set at
  line 1050 with `BIAS_POINTER` at 1054, full `cublasLtMatmul` call at 1088.
- `src/Kernels/cuda/matmul/MatmulBackward.cu` — 6 sites: dA + dB × 3
  variants. cuBLASLt-first with `cublasGemmEx` fallback.

GELU+bias fusion: **partial — not yet wired into the training path.** The
helper kernel `fused_bias_gelu` exists in
[`src/Kernels/cuda/activations/arch/FusedLinearGelu_sm89.cu`](../src/Kernels/cuda/activations/arch/FusedLinearGelu_sm89.cu)
with `CUBLASLT_EPILOGUE_GELU_BIAS` / `CUBLASLT_EPILOGUE_GELU` references at
lines 123, 208, 263, but the forward MLP path still calls Linear and GELU as
separate ops. Full integration (replacing `c_fc.forward(x) → gelu()` with a
single fused-epilogue call, plus the matching backward) is **deferred to a
later sprint**.

### Build bug fixed

The cuBLASLt include landed only inside the `#ifdef WITH_MYBLAS` branch.
Added to `#else` too (safe since `#pragma once`).

### Status notes

- **Backward GELU+bgrad fusion not implemented.** No
  `CUBLASLT_EPILOGUE_DGELU_BGRAD` in the codebase. Backward pass still does
  separate gelu_backward + bias-grad kernels. Open follow-up.
- **`broadcast_scale_kernel` retained as fallback only — not in the hot
  path.** The kernel still exists in
  [`GenMatmul.cu:294`](../src/Kernels/cuda/matmul/GenMatmul.cu) but
  `cuda_addmm` (line 1109) takes the cuBLASLt fused-BIAS fast path FIRST
  (lines 1120-1132) and `return`s on success. The `launch_broadcast_scale`
  fallback at line 1143 only fires when cuBLASLt fails or when
  `beta != 1.0` / `input.numel() != N` (i.e., non-bias addmm corner cases);
  the line 1154 path is the `beta == 0` zero-output edge case. nsys
  evidence: in the post-port runs, `broadcast_scale_kernel` shows up
  with 1 instance per run (vs ~17,280 pre-port) — confirming the fused
  epilogue absorbs all normal Linear-with-bias calls. Kept on purpose as
  a safety net and for non-bias scaling.

### Result on training-script wall time (closing the 13.5 s gap to PyTorch)

Per `CHANGES_AND_FIXES_LOG.md`, the cuBLASLt port was projected to close
~10 s of the 13.5 s gap. The other ~3.2 s came from items
(§6, §7, §10, §8, §13).

---

<a id="15-smart-reshape"></a>

## 15. Smart reshape (`compute_view_stride`)

### Problem

Many `reshape` calls in the model trigger an unnecessary `.contiguous()`
copy when the new shape can in fact be expressed as a metadata-only view
of the source. Example: QKV split `[B,T,3C] → [B,T,C] (sliced) →
[B,T,nh,hd]`. The inner 768-element row is dense within the `3C`
allocation; splitting it `[12, 64]` is a free view, but the old reshape
unconditionally called `.contiguous()`.

### What changed

Added `ViewUtils::compute_view_stride` mirroring PyTorch's
`at::detail::computeStride_impl`. It walks old dims back-to-front, tracks
memory-contiguous chunks, and drains view dims into each chunk. If all view
dims fit cleanly, returns a non-empty `Stride` (metadata-only view);
otherwise returns empty (caller must `.contiguous()`).

Wired into `Tensor::reshape` so view-eligible cases skip the copy.

### What it caught

12,960 `strided_inner_vec_copy_kernel` calls per nsys run — the forward QKV
reshape pattern. Eliminated.

### What it cannot catch (and why Packed SDPA was needed on top)

The merge after attention `attn_out.transpose(1,2).reshape([B,T,C])` is
**not** view-eligible — dims `nh` and `hd` are non-memory-adjacent after the
transpose, no stride-tuple can express the merge. Same in backward. PyTorch
hits the same dead end and falls back to `.contiguous()`. This is what
Packed SDPA (§16) eliminates by changing the kernel's output layout.

### Where it lives now

- [`include/core/Views/ViewUtils.h:43`](../include/core/Views/ViewUtils.h) — declaration.
- [`src/Views/ViewUtils.cpp:224`](../src/Views/ViewUtils.cpp) — definition.
- Wired into `Tensor::reshape` at
  [`src/Views/ViewOps.cpp:78-91`](../src/Views/ViewOps.cpp).

---

<a id="16-packed-sdpa"></a>

## 16. Packed SDPA — final attention copy elimination

This is the most recent change. Full detail in
[`docs/Optimizing Packed Attention Performance.md`](Optimizing%20Packed%20Attention%20Performance.md);
brief summary here for completeness.

### Three independent copy sources around attention

```
qkv ──reshape──→ q,k,v ──transpose──→ SDPA ──transpose──→ reshape ──→ c_proj
       (A)                (B)                  (C)
```

| Site | Forced by | Fix layer |
|---|---|---|
| (A) reshape into `[B,T,nh,hd]` | the reshape op | smart reshape (§15) |
| (B) SDPA reads strided Q/K/V | the kernel's read pattern | stride-aware kernels (§7) |
| (C) merge `[B,T,nh,hd] → [B,T,C]` after SDPA | `c_proj` (next cuBLAS GEMM, demands a single ld) | layout choice — Packed SDPA |

The smart-reshape and stride-aware-kernel fixes addressed (A) and (B).
(C) is a cuBLAS-driven copy that no reshape rule can avoid — the only fix is
to **not produce the layout that needs merging**.

### What Packed SDPA does

Forward (`sdpa_memory_efficient_packed`):

- No shard, no reshape, no transpose.
- Reads Q/K/V via strided pointer offsets `qkv_ptr + {0, C, 2C}` into the
  packed `[B,T,3C]` buffer — common strides `(strideB=T·3C, strideM=3C,
  strideH=hd, last=1)`.
- Calls `mem_efficient_attn_forward_tc` which is already stride-aware.
- Writes output as packed `[B,T,C]` contiguous (heads packed inner) — exactly
  the layout `c_proj` wants.

Backward (`PackedSDPABackward::apply`):

- Allocates one zeroed `dqkv [B,T,3C]` (`Tensor::zeros`).
- Three strided pointers into that one buffer; calls
  `mem_efficient_attn_backward(..., skip_grad_zero=true)`.
- Returns the single packed dqkv → no `Tensor::cat` needed downstream.

### `skip_grad_zero` flag

The backward kernel uses `atomicAdd` for dQ/dK/dV and therefore memsets them
to zero internally before launching the compute kernel. For packed mode, the
three buffers are interleaved pointer offsets into one allocation — a flat
contiguous memset would corrupt the layout. The flag (default `false`)
gates the internal memsets:

- `false` → existing behavior, kernel zeros internally (unfused path).
- `true` → kernel skips memset, caller has zeroed the whole `dqkv` buffer
  externally.

### Eight-edit integration

1. [`include/ops/helpers/AttentionKernels.h:66`](../include/ops/helpers/AttentionKernels.h) — added `bool skip_grad_zero = false`.
2. [`src/Kernels/cuda/attention/AttentionBackward.cu`](../src/Kernels/cuda/attention/AttentionBackward.cu) — param threaded through; both internal memsets (exp7 dQ, exp11 dK/dV) gated; sm89 dispatch forwards the flag.
3. [`src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu`](../src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu) — same param; exp12 dK/dV memsets gated.
4. [`include/autograd/backward/AttentionBackward.h:127`](../include/autograd/backward/AttentionBackward.h) — added `class PackedSDPABackward`.
5. [`src/autograd/backward/AttentionBackward.cpp:159`](../src/autograd/backward/AttentionBackward.cpp) — `PackedSDPABackward::apply`.
6. [`include/autograd/operations/AttentionOps.h:106`](../include/autograd/operations/AttentionOps.h) — declared `scaled_dot_product_attention_packed`.
7. [`src/autograd/operations/AttentionOps.cpp`](../src/autograd/operations/AttentionOps.cpp) — added `sdpa_memory_efficient_packed` static + public dispatch (line 452) + autograd wiring (350–401).
8. [`gpt2_attn_navin.cpp:180`](../gpt2_attn_navin.cpp) — 2-path env-gated branch (`USE_PACKED_SDPA=1` → packed; else original).

### Two paths only

Deliberately did **not** port the `USE_QKV_RESHAPE_FIRST` middle path that
appeared in the colleague's `gpt2_fmha_ddp.cpp`. It's a half-measure that
adds complexity without value.

### Default behavior table

| Env var state | Branch taken |
|---|---|
| not set | unfused (default — safe, identical to old behavior) |
| `USE_PACKED_SDPA=0` | unfused |
| `USE_PACKED_SDPA=1` | packed |
| anything else (e.g. `true`, `0x`, `01`) | unfused (only literal `1` matches) |

### Correct command syntax

```bash
# Inline env var, prefixed before the command that needs it:
CUDA_VISIBLE_DEVICES=6 USE_PACKED_SDPA=1 nsys profile --stats=true ./snippet_runner

# Or export once:
export USE_PACKED_SDPA=1
echo "$USE_PACKED_SDPA"          # must print "1"
CUDA_VISIBLE_DEVICES=6 nsys profile --stats=true ./snippet_runner

# For make:
USE_PACKED_SDPA=1 make run-snippet FILE=gpt2_attn_navin.cpp WITH_BLUBLAS=1
```

### Wrong syntax (env var silently dropped, runs unfused)

```bash
nsys profile --stats=true ./snippet_runner USE_PACKED_SDPA=1   # ← positional arg
make run-snippet FILE=gpt2_attn_navin.cpp USE_PACKED_SDPA=1    # ← make var, not env
USE_PACKED_SDPA=1
./snippet_runner                                                # ← assignment scope
```

### Verified result (report39)

- `cat_batched_kernel`: **0** (was 3,840 in unfused).
- `generic_strided_copy_kernel` (attention): **0** (was 15,840).
- Total kernel launches: 129,592 (was 149,272 — exactly 19,680 fewer = 15,840 + 3,840).
- Loss curve: bit-identical to unfused within < 1e-5 across all 10 steps.
- Wall-time per step: ~3–5% faster (~250–390 ms saved per step).

---

<a id="17-cross-cutting"></a>

## 17. Cross-cutting: folder layout, build, sm89 dispatch

### Folder layout adopted during the sprint

The team repo (`master_gau_latest_ada_6000_sm89`, sm_89) follows an
arch-subfolder convention:

```
src/Kernels/cuda/
├── activations/
│   ├── ActivationKernels.cu         (generic / dispatcher)
│   └── arch/
│       └── GELUKernels_sm89.cu      (Ada-tuned)
├── norm/
│   ├── LayerNormKernels.cu
│   └── arch/
│       └── LayerNormKernels_sm89.cu
├── attention/
│   ├── AttentionForward.cu
│   ├── AttentionBackward.cu
│   └── arch/
│       ├── AttentionForward_sm89.cu
│       └── AttentionBackward_sm89.cu
├── optimizer/
│   ├── MultiTensorKernels.cu
│   └── arch/
│       └── MultiTensorKernels_sm89.cu
└── misc/
    ├── BatchedCat.cu
    └── FillKernel.cu
```

Generic kernels live in the parent file; sm89-specific ones in `arch/`. The
top-level dispatch picks via `cuda::get_arch()` returning `ArchFamily::Ada`
(see `AttentionBackward.cu:675` for an example dispatch).

The local mirror `master_gau_latest_sm_86_RTX_3060` (sm_86) is
source-identical except for the Makefile arch flag.

### Build notes

- NVCC pinned to CUDA 13.0 in the Makefile (auto-detect was picking 11.5).
- After kernel signature changes (e.g. `skip_grad_zero` added), do
  `make clean` first — stale `.o` files compiled against the old signature
  cause linker errors or silent ABI mismatch.
- `WITH_BLUBLAS=1` switches to our in-house BLAS library (independent of
  cuBLAS).
- `WITH_MYBLAS` flag still flips between cublasGemmEx fallback and
  cuBLASLt-first paths.

---

<a id="18-verification-table"></a>

## 18. Verification table — what is present in the code right now

Cross-checked against the live tree at the time of writing.

| # | Item | Status | Anchor |
|---|------|--------|--------|
| 1 | Reductions: `unified_reduce_kernel`, packed metadata, bitmap | present | `include/ops/helpers/ReductionKernels.cuh:332,486` |
| 2 | GeLU sm89 fwd/bwd | present | `src/Kernels/cuda/activations/arch/GELUKernels_sm89.cu:61,107` |
| 3 | LayerNorm sm89 + RMSNorm template | present | `src/Kernels/cuda/norm/arch/LayerNormKernels_sm89.cu:63,250,289` |
| 4 | `cudaMemset` → `cudaMemsetAsync` | present | only one residual in a comment at `src/core/TensorFactory.cpp:211` |
| 5 | LN backward redundant memsets removed | present | autograd grads use `Tensor::empty`; kernel-internal accumulator memsets retained intentionally |
| 6 | Attention dQ `zeros` → `empty` | present | `src/autograd/backward/AttentionBackward.cpp:107-109` |
| 7 | Stride-aware attention kernels | present | `include/ops/helpers/AttentionKernels.h:17,29,53` |
| 8 | Contiguous 4-path hybrid | present | `src/Views/ContiguousKernel.cu` (lines per §8) |
| 9 | `Tensor::contiguous()` short-circuit | present | `src/core/Tensor.cpp:283-285` |
| 10 | `cat_batched_kernel` | present | `src/Kernels/cuda/misc/BatchedCat.cu:48` |
| 11 | `multi_tensor_zero_sm89_kernel` | present | `src/Kernels/cuda/optimizer/arch/MultiTensorKernels_sm89.cu:343` |
| 12 | `Optimizer::zero_grad` rewrite | present | `src/nn/optimizer/Optim.cpp:29-55` |
| 13 | Factory functions (zeros/ones/full/set_*/fill_grad) GPU-side | present | `src/core/TensorFactory.cpp:172,275,360` |
| 14 | `fill_cuda_launch` / `fill_kernel_*` | present | `src/Kernels/cuda/misc/FillKernel.cu:52,65,76,98` |
| 15 | CachingCudaAllocator | present | `src/device/CudaCachingAllocator.cpp:370` |
| 16 | Sparse CE save_max/save_sum | present | `src/Kernels/cuda/loss/LossKernels.cu:188,706,913,953` |
| 17a | cuBLASLt fused BIAS (Linear with bias forward) | present, hot path | `GenMatmul.cu:1050,1054,1088,1120-1132` |
| 17b | cuBLASLt fused GELU_BIAS (Linear+GELU forward) | **partial** | helper `fused_bias_gelu` exists in `FusedLinearGelu_sm89.cu:123,208,263`, not yet wired into MLP forward; deferred |
| 17c | DGELU_BGRAD backward fusion | **absent** | follow-up |
| 17d | `broadcast_scale_kernel` retired from hot path | present (fallback only) | `GenMatmul.cu:1143,1154` are non-hot-path; cuBLASLt absorbs all normal Linear-bias calls (~17,280 → 1 per nsys run) |
| 18 | Smart reshape (`compute_view_stride`) | present | `src/Views/ViewUtils.cpp:224` |
| 19 | Packed SDPA full stack | present | see §16 anchors |

---

<a id="19-divergences"></a>

## 19. Known divergences and follow-ups

### Open follow-ups

1. **`broadcast_scale_kernel` removal.** Still alive at three call sites in
   `GenMatmul.cu`. Originally planned to retire via cuBLASLt epilogue fusion;
   only partially done. Worth revisiting before the next round of matmul
   work.
2. **Backward GELU+bias-grad cuBLASLt epilogue.** No
   `CUBLASLT_EPILOGUE_DGELU_BGRAD` anywhere. Forward fuses bias and GELU into
   the GEMM; backward still does separate `gelu_backward_sm89_kernel` and
   bias-grad kernels. PyTorch eager doesn't fuse these either, so this is a
   "go beyond PT" item, not a parity gap.
3. **Reductions hardware optimizations section.** Reserved/empty in
   `optimizations_and_code-changes_done_in_reductions.md` — the float4 /
   `aligned_vector<float,4>` GPU vectorization pass for reductions is not
   yet shipped.
4. **GPU GELU backward not float4-vectorized.** Mentioned in
   `gelu_deep_dive.md` "Remaining" list.
5. **AVX-512 dispatch for activations.** Currently AVX2-only on CPU.
6. **Optimizer micro-ops on small models.** `multi_tensor_*` is well-tuned
   for 100M+ models; small models may want a different chunking.

### Things to watch in the team repo

- Colleague's `gpt2_fmha_ddp.cpp` introduced a third path
  (`USE_QKV_RESHAPE_FIRST`). I deliberately did not port it; if it shows
  up in the team repo, push back — it's a half-measure that adds branching
  complexity without distinct value.
- `BatchedCat.cu` lives in `misc/` (some old docs imply `cat/`); make sure
  any new `cat`-related work doesn't try to create a duplicate folder.

### How to keep this doc current

After any new optimization:

1. Add a section to this file under the right number (or appended).
2. Update §18 verification table.
3. Cross-check the relevant per-topic doc (e.g.
   `docs/CHANGES_AND_FIXES_LOG.md`) and link or fold its content here.
4. If a colleague changes something, update §18 status and add a row in
   §19 explaining what changed and why.

---

## Sprint roll-up

Wall-time impact (best estimates, single-GPU, 124M GPT-2 training script):

| Optimization | Saved per run |
|---|---|
| Memset cleanup (LN + Attention dQ + sync→async) | ~3.2 s |
| Stride-aware attention | ~2.0 s |
| ContiguousKernel hybrid (vec-inner + transpose) | ~0.4 s |
| BatchedCat fused | small |
| Tensor::contiguous() short-circuit | small |
| `multi_tensor_zero_grad` + factory rewrites | a few hundred ms |
| Sparse CE save_max/save_sum | ~1.1% wall = ~1.1 s |
| cuBLASLt port (TF32 + bias fusion + GELU_BIAS) | ~10 s (closes most of the 13.5 s PT gap) |
| Smart reshape (12,960 strided_inner_vec_copy gone) | ~0.4 s |
| Packed SDPA (15,840 generic_strided + 3,840 cat gone) | ~3 s over 10 steps |

Pre-sprint baseline: 126.56 s; PyTorch reference 113.04 s; gap 13.5 s.
Post-sprint: training script runs with all of the above stacked. The cuBLASLt
port and Packed SDPA together close the gap and push us past PyTorch in some
runs.

---

*End of record.*
