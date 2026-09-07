# Changes & Fixes Log — GPT-2 Training Throughput Optimization

Scope: everything done yesterday + today to close the wall-time gap
between our custom CUDA library and PyTorch on the GPT-2 training
workload. Active folder: `master_gau_latest_ada_6000_sm89/`
(server, sm_89 / Ada RTX 6000). Mirror folder for local RTX 3060:
`master_gau_latest_sm_86_RTX_3060/`.

---

## 1. Baseline we started from

- Ours wall time: **126.56 s**
- PyTorch wall time: **113.04 s**
- Gap: **~13.5 s**
- GPU utilization: ours 88.7 %, PyTorch 89.8 %
- `cudaLaunchKernel` avg: ours **376 µs**, PyTorch **9 µs**
  → stream back-pressure; PyTorch runs much further ahead of the GPU.
- nsys showed PyTorch dispatching `cutlass::Kernel2<...>` for FP32
  matmul, ours dispatching older `cutlass::Kernel<...>`. That is the
  cuBLASLt vs classic cuBLAS distinction. Hypothesized to account for
  ~6–10 s of the gap.

Step-0 baseline loss (must stay the same after every change):
**11.041492**.

---

## 2. Earlier session — memory-op cleanup (the first six optimizations)

Before the attention refactor, six "memory-op overhead" optimizations
were stacked. Profiled impact:

| # | Optimization                   | Kernel count change   | GPU time saved   | tok/sec Δ           |
| - | ------------------------------ | --------------------- | ---------------- | ------------------- |
| 1 | LayerNorm `zeros` → `empty`    | ~25K memsets removed  | –205 ms          | 0                   |
| 2 | Attn dQ `zeros` → `empty`      | ~4K memsets removed   | –240 ms          | 0                   |
| 3 | Loss `to_cpu` reorder          | 0                     | 0 (idle-shifted) | 0                   |
| 4 | Stride-aware attention (§3.1)  | ~17K kernels          | –2000 ms         | +2 % (within noise) |
| 5 | Contiguous bug fix (§3.3)      | ~12K memcpys          | –790 ms          | 0                   |
| 6 | Fused cat (§3.2)               | ~11K memcpy2Ds        | 0 (moved to SMs) | 0                   |

**Total GPU time saved: ~3.2 s across a 130-second run ≈ 2.5 %.**
Throughput moved by < 2 %, i.e. inside noise.

### 2.1 Why the total was so small — the diagnosis that drove today's work

All six optimizations targeted **memory-op overhead**, which was at
most ~8 % of total GPU time. The other 92 % of GPU time breaks down
roughly as:

```text
mem_efficient_bwd_unified_kernel_exp12   17.25 s   (15.4 %)   attention bwd
cutlass matmuls (8 variants)             ~66 s    (59 %)     MATMUL
fused_attn_forward_kernel_tc_sm89         9.37 s   (8.4 %)   attention fwd
```

**The matmuls alone eat ~60 % of GPU time. None of the six memory-op
optimizations touched those.** That is the reason today's session
went after the cuBLASLt port (§4) — it's the one change that actually
sits on top of the 60 % bucket.

### 2.2 The three "new" items (4-6 are documented in §3 below)

#### 2.2.1 LayerNorm `zeros` → `empty`

Files: LayerNorm fwd/bwd scratch allocations.

Bug: the kernel fully writes the output buffer, but we were allocating
it via `zeros()` — which issues a `cudaMemsetAsync` to clear the buffer
first. Pure waste: the memset data is thrown away before any consumer
reads it.

Fix: swap to `empty()` (uninitialized alloc). ~25K memsets / step
removed across the layernorm layers.

**Result:** –205 ms GPU time / run, no tok/sec movement (the memsets
were overlapped with other work).

#### 2.2.2 Attention dQ `zeros` → `empty`

File: attention backward — `dQ` gradient buffer allocation.

Same pattern: kernel fully writes dQ, but we were zeroing it first.

Fix: switch to `empty()`.

**Result:** ~4K memsets removed, –240 ms GPU time / run, no tok/sec
movement.

#### 2.2.3 Loss `to_cpu` reorder

File: training loop.

Problem: `loss.to_cpu()` (a D→H sync) was being called before the
backward/optimizer kernels launched for the next step. The sync
forced the CPU to wait on the forward pass before it could enqueue
backward, stalling the dispatch pipeline.

Fix: moved `loss.to_cpu()` to **after** the optimizer step. Backward
and optimizer enqueue first; the D→H sync then happens while the next
forward is already running.

**Result:** 0 GPU-time savings (no kernel removed), but stream
back-pressure was reduced — the sync cost got shifted into otherwise-
idle time.

---

## 3. Yesterday — reducing strided-copy overhead and clearing stalls

### 2.1 Attention fwd/bwd accept non-contiguous QKV tensors

Problem: attention forward and backward forced `.contiguous()` on Q, K,
V, grad_out before every call, launching redundant copy kernels every
step.

Files touched:
- `include/ops/cuda/attention/AttentionCommon.cuh`
- `include/ops/helpers/AttentionKernels.h`
- `src/Kernels/cuda/attention/AttentionForward.cu`
- `src/Kernels/cuda/attention/AttentionBackward.cu`
- `src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu`
- `src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu`
- `src/autograd/operations/AttentionOps.cpp`
- `src/autograd/backward/AttentionBackward.cpp`
- `gpt2_attn_navin.cpp`

Change: the sm89 kernels now read Q/K/V through strided pointers
(`strideQ_*`, `strideK_*`, `strideV_*` — head/seq/dim strides passed in
as kernel args) instead of assuming contiguous `[B, H, S, D]` layout.
Logic-level docs for this refactor live in
[ATTENTION_STRIDED_REFACTOR.md](../master_gau_latest_sm_86_RTX_3060/ATTENTION_STRIDED_REFACTOR.md).

**Result:** attention stopped paying the forced-contiguous copy on the
hot path. Forward-call count stayed the same (4320 fwd / 3840 bwd) but
the redundant preamble copies were eliminated.

Follow-up question from profiling: "why are fwd calls 4320 but bwd only
3840?" — **delta = 480 = 20 val steps × 12 layers × 2 validations**.
Validation runs forward-only, no backward — that's the entire
asymmetry, not a bug.

### 2.2 Fused cat kernel for attention (PyTorch-style)

New file: `src/Kernels/cuda/misc/BatchedCat.cu`

Problem: constructing the concatenated QKV projection tensor in attention
was launching one `memcpy`/strided copy per head per batch instead of a
single batched copy (PyTorch uses `CatArrayBatchedCopy`).

Change: added a fused batched-cat kernel modeled on PyTorch's
`CatArrayBatchedCopy` so the Q/K/V concat becomes one launch instead of
three strided copies. Functionality is additive — the previous cat path
is preserved.

**Result:** one launch instead of three per attention block; contributed
to the fwd/bwd call-count reduction.

### 2.3 ContiguousKernel Path 3d

File: `src/Views/ContiguousKernel.cu`
Doc: [CONTIGUOUS_KERNEL_MINDMAP](./CONTIGUOUS_KERNEL_MINDMAP) (commit
`a8b9b7e`).

Problem: the `contiguous_strided_copy_cuda` dispatcher had Paths 3a/3b/3c
but no cheap path for the common `transpose_for_attention`-style
permutation `[B, S, H, D] → [B, H, S, D]` with D contiguous.

Change: added Path 3d specialized for this permutation — warp-coalesced
load of the contiguous-D dim, no atomics, no scratch.

**Result:** the transpose-for-attention copy fires Path 3d instead of
falling back to the slow `generic_strided_copy_kernel`. Measured
remaining generic-copy calls dropped from their prior count down to the
~28,800 that are sourced from ReshapeBackward / Linear-backward reshape
chains (not attention).

Also investigated and confirmed: **Path 3a is not dead code.** It just
rarely fires in our training, because our transposed views don't
produce a stride pattern that coalesces into a single contiguous
linear copy — that was the user's concern; verified by reading the
dispatch logic.

### 2.4 TensorFactory — bool zero-init was wrong memset width

File: `src/core/TensorFactory.cpp`

Bug: `zeros()` on a `bool` tensor was going through the generic
`cudaMemsetAsync(ptr, 0, N * sizeof(T))` pattern, but `bool` sizeof is
1 on the host — fine — the issue was that the **true-value** path
(`fill(true)`) was writing `0x01` only to the first byte per element
instead of correctly setting every byte.

Change: `zeros()` for `bool` now does a plain byte memset of the full
byte count; `ones()` for `bool` uses `cudaMemsetAsync(ptr, 0x01, N)`.

**Result:** bool tensor init now correct and async, no blocking.

### 2.5 CudaCachingAllocator — removed two `cudaStreamSynchronize` calls

File: `src/core/Tensor.cpp` (allocator paths)

Bug: allocator was calling `cudaStreamSynchronize` twice on the free
path — once when reclaiming and once when returning the block — which
serialized kernel launches with the stream and directly contributed to
the 376 µs `cudaLaunchKernel` figure.

Change: removed both `cudaStreamSynchronize` calls. The event-based
reuse-safety mechanism is already sufficient (block is only reissued
once the recorded event completes on whichever stream had been using
it).

**Result:** alloc/free is fully async now; one of the contributors to
the stream back-pressure is gone.

### 2.6 LayerNorm memsets — sync → async

Files:
- `src/UnaryOps/cpu/Normalizations.cpp`
- (sm89 layernorm fwd kernel paths)

Bug: layernorm scratch buffers were being zeroed with
`cudaMemset` (synchronous, legacy-default stream), which serialized
with everything else on the user's stream.

Change: switched to `cudaMemsetAsync(..., stream)`.

**Result:** the memset cluster at step boundaries visible in nsys
dropped out of the critical path.

### 2.7 Folder sync (local ↔ server)

The user copied the server folder locally and diffed: local was
**behind** the server on the fixes in §2.4, §2.5, §2.6, and §2.3. The
user then renamed:
- `master_gau_latest_2` → `master_gau_latest_sm_86_RTX_3060` (local,
  RTX 3060, sm_86)
- server folder → `master_gau_latest_ada_6000_sm89` (Ada RTX 6000,
  sm_89)

Then copied six files from the ada folder into the sm_86 folder (the
four LayerNorm/TensorFactory/CachingAllocator/ContiguousKernel changes
plus the attention refactor files), **excluding the Makefile** (user's
explicit instruction — arch flag differs per folder).

**Result:** the two folders are now source-identical except for
`Makefile` (`-arch=sm_89` vs `-arch=sm_86`).

### 2.8 Investigated and rejected (did not change)

- **Pinning a reusable host buffer for the D→H loss copy.** The user
  pushed back: pinning a 4-byte float is pure overhead; the allocate-
  once-reuse version was proposed and then also rejected because the
  problem it was solving was not actually on the critical path.
- **Removing the NaN check on the loss.** Considered, then dropped —
  PyTorch does the same NaN check; it's not the differentiator.
- **Collapsing the step-boundary memset cluster.** User decided it
  wasn't worth the refactor after the cost was measured. Quote: "why
  to mess these small small things, now our old doubt that may be
  bridging that gap may increase throughput, but it didnt".

---

## 3. Today — cuBLASLt matmul port (the real ~10 s)

Diagnosis: nsys showed PyTorch dispatches `cutlass::Kernel2<...>`
templates via cuBLASLt's `cublasLtMatmulAlgoGetHeuristic`. We were
using `cublasGemmEx`, which uses a fixed default algorithm and the
older `cutlass::Kernel` templates. This closes an estimated ~6–10 s
of the remaining gap.

### 3.1 New files

**`include/ops/helpers/CublasLtHelper.h`** — header with
argument-order matching `cublasGemmEx` for a 1:1 port at every call
site. Declares:

- `cublasLtHandle_t get_cublaslt_handle(int device)`
- `void*  get_cublaslt_workspace(int device)`
- `size_t get_cublaslt_workspace_size()`
- `bool cublaslt_gemm_fp32_tf32(...)`
- `bool cublaslt_gemm_strided_batched_fp32_tf32(...)`

Both GEMM helpers return `bool` so the caller falls back to
`cublasGemmEx` if cuBLASLt's heuristic rejects the shape.

**`src/Kernels/cuda/matmul/CublasLtHelper.cu`** — implementation:
- Per-device cached `cublasLtHandle_t` (array of 8, mutex-protected
  lazy init).
- Per-device 32 MiB workspace (cached).
- `lt_gemm_impl` builds `cublasLtMatmulDesc` with
  `CUBLAS_COMPUTE_32F_FAST_TF32` + `CUDA_R_32F` scale, builds A/B/C
  layouts (cuBLASLt is column-major; our callers already do the N/M
  + opA/opB swap), sets `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`,
  calls `cublasLtMatmulAlgoGetHeuristic` for the best algo, then
  `cublasLtMatmul`. Returns `false` on heuristic rejection so the
  caller falls back.
- Handles batched GEMM via `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` +
  `CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET`.

### 3.2 Modified: `src/Kernels/cuda/matmul/GenMatmul.cu`

**5 FP32-TF32 call sites ported** (lines after edit):

| Site | Context | Ported to |
|------|---------|-----------|
| fwd, single batch | `cuda_matmul`, `tb==1` | `cublaslt_gemm_fp32_tf32` |
| fwd, strided batched | `cuda_matmul`, `tb>1` uniform | `cublaslt_gemm_strided_batched_fp32_tf32` |
| fwd, 4D loop | `cuda_matmul`, `on==4` non-uniform | `cublaslt_gemm_strided_batched_fp32_tf32` (per outer batch) |
| addmm, single batch | `cuda_addmm`, `tb==1` | `cublaslt_gemm_fp32_tf32` |
| addmm, strided batched | `cuda_addmm`, `tb>1` uniform | `cublaslt_gemm_strided_batched_fp32_tf32` |

Pattern at every site: try cuBLASLt helper first; on `false` return,
fall back to the existing `cublasGemmEx` / `cublasGemmStridedBatchedEx`
call unchanged.

### 3.3 Modified: `src/Kernels/cuda/matmul/MatmulBackward.cu`

**6 FP32-TF32 call sites ported** — `dA` (grad input) and `dB` (grad
weight) paths, each with three variants (single batch, batched,
non-uniform 4D loop).

### 3.4 Bug found during build — include landed in the wrong branch

Both `GenMatmul.cu` and `MatmulBackward.cu` are structured as:

```
#ifdef WITH_MYBLAS
   ... myblas (BluBridge-BLAS) path ...
#else
   ... cuBLAS / cuBLASLt path ...
#endif
```

The build default (no `WITH_KOBLAS=1`) compiles the `#else` branch.

Bug: my first pass added `#include "ops/helpers/CublasLtHelper.h"`
only at the top of the `#ifdef WITH_MYBLAS` branch. When compiling
default, the preprocessor skipped that branch — so the include was
never seen, and every reference to `cublasLtHandle_t` / `cuda::` in
the #else branch failed:

```
error: identifier "cublasLtHandle_t" is undefined
error: name followed by "::" must be a class or namespace name
           cublasLtHandle_t ltHandle = cuda::get_cublaslt_handle(dev_idx);
```

Fix: added the include to the `#else` branch's include block as well
(it's harmless in both branches because the header is `#pragma once`).

**Result:** clean compile of both files.

### 3.5 Non-bug confirmed on request

When the user builds with `WITH_KOBLAS=1` (myblas backend), the
`#ifdef WITH_MYBLAS` branch compiles and the `#else` branch is
excluded. That branch has **zero** cuBLAS/cuBLASLt calls — it uses
`mycublasSgemmStridedBatched` / `mycublasHgemmStridedBatched` /
`mycublasBgemmStridedBatched` / `mycublasDgemmStridedBatched` and
`mycublasSgemmAddmm_SM86`. So choosing myblas does not secretly
dispatch to NVIDIA libraries.

### 3.6 Build system

Makefile already uses `$(shell find $(SRCDIR) -name '*.cu')` so the
new `CublasLtHelper.cu` is auto-picked up. `LDLIBS` already includes
`-lcublasLt`. No Makefile change needed.

---

## 4. Bugs found, fixed, and outcome — cheat sheet

| # | Where | Bug | Fix | Outcome |
|---|-------|-----|-----|---------|
| 1 | attention fwd/bwd | forced `.contiguous()` on Q/K/V every step | sm89 kernels now accept strided Q/K/V | removed redundant copy kernels from attention hot path |
| 2 | misc | three strided copies to build QKV concat | fused batched-cat kernel (`BatchedCat.cu`) | 1 launch instead of 3 |
| 3 | `ContiguousKernel.cu` | no cheap path for `[B,S,H,D]→[B,H,S,D]` | added Path 3d | eliminated the transpose-for-attention falling into `generic_strided_copy_kernel` |
| 4 | `TensorFactory.cpp` | bool `ones()` wrote `0x01` to only the first byte per element | byte-wide memset | correct bool init, still async |
| 5 | `Tensor.cpp` (allocator) | two `cudaStreamSynchronize` on free path | removed, rely on event-based reuse safety | async alloc/free; removed one source of stream back-pressure |
| 6 | layernorm kernels | sync `cudaMemset` on scratch | `cudaMemsetAsync(..., stream)` | step-boundary memset cluster off the critical path |
| 7 | local vs server folders | local was behind on fixes #3–#6 and attention refactor | 6-file copy from server → local | folders source-identical (Makefile excluded) |
| 8 | `GenMatmul.cu` / `MatmulBackward.cu` | using `cublasGemmEx` (older `cutlass::Kernel`) instead of cuBLASLt | new `CublasLtHelper` + 11 call-site ports with `cublasGemmEx` fallback | PyTorch-matching `cutlass::Kernel2` dispatch on all FP32-TF32 matmuls |
| 9 | build | `#include "ops/helpers/CublasLtHelper.h"` landed only in `#ifdef WITH_MYBLAS` branch | added include to `#else` branch too | clean compile |

---

## 5. Investigated and rejected

- Pinning a reusable host buffer for the loss `D→H` copy.
- Removing the NaN check on the loss.
- Collapsing the step-boundary memset cluster.
- Refactoring for the remaining `generic_strided_copy_kernel` calls
  sourced from `ReshapeBackward` / Linear backward reshape chains
  (~28,800 calls, ~3.4 s) — deferred; lower priority than the matmul
  algorithm gap.

---

## 6. Current status

- Code changes: **done.**
- Build: **clean** (after the include-in-wrong-branch fix).
- Loss validation: **pending** — next run must show step-0 loss =
  `11.041492`.
- nsys validation: **pending** — expect `cutlass::Kernel2<...>`
  instead of `cutlass::Kernel<...>` on FP32 matmul, and wall time
  closer to PyTorch's 113.04 s.

## 7. File inventory

New files (in `master_gau_latest_ada_6000_sm89/`):
- `include/ops/helpers/CublasLtHelper.h`
- `src/Kernels/cuda/matmul/CublasLtHelper.cu`
- `src/Kernels/cuda/misc/BatchedCat.cu`

Modified files:
- `src/Kernels/cuda/matmul/GenMatmul.cu`
- `src/Kernels/cuda/matmul/MatmulBackward.cu`
- `src/Views/ContiguousKernel.cu`
- `src/core/Tensor.cpp`
- `src/core/TensorFactory.cpp`
- `src/UnaryOps/cpu/Normalizations.cpp`
- `src/Kernels/cuda/attention/AttentionForward.cu`
- `src/Kernels/cuda/attention/AttentionBackward.cu`
- `src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu`
- `src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu`
- `include/ops/cuda/attention/AttentionCommon.cuh`
- `include/ops/helpers/AttentionKernels.h`
- `src/autograd/operations/AttentionOps.cpp`
- `src/autograd/backward/AttentionBackward.cpp`
- `gpt2_attn_navin.cpp`

Docs produced along the way:
- `ATTENTION_STRIDED_REFACTOR.md` (in the sm_86 folder)
- `CONTIGUOUS_KERNEL_MINDMAP` (this folder, commit `a8b9b7e`)
- `CHANGES_AND_FIXES_LOG.md` (this file)
