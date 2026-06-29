# AtomicAdd Kernels in My BluTrain GPT-2 124M vs PyTorch — Full Research Record

> **What I'm documenting here.** This is my complete record of backtracking **every kernel
> that uses `atomicAdd` (or any atomic)** in my BluTrain GPT-2 **124M** training run on the Ada
> RTX 6000 (sm_89), comparing each one against PyTorch's equivalent kernel for the same op, and
> working out which of mine I can make **deterministic** (bit-reproducible) without losing speed.
> The bias-grad reduction piece also lives in [`bias_gradient_reduce_sum_kernel_usage.md`](bias_gradient_reduce_sum_kernel_usage.md) §8–12;
> this doc is the standalone "all atomic kernels vs PyTorch" survey I hadn't written down anywhere.
>
> **Why I cared.** While documenting my reduction kernel I established that **`atomicAdd` on
> floating-point values makes results bit-non-reproducible run-to-run** (FP add is
> non-associative, atomic arrival order is random each run). That made me skeptical: *which
> other kernels in my actual 124M training path use atomicAdd, and is that why my loss
> fluctuates more run-to-run than PyTorch's?* So I backtracked every launched kernel.
>
> **How I did it (no guessing — every atomic confirmed to its launched variant).**
>
> 1. I pulled all **32 distinct kernels** that actually launched in my 10-step run from
>    `may27_WITH_BLUBLAS.sqlite` (`CUPTI_ACTIVITY_KIND_KERNEL` joined to `StringIds`), with counts.
> 2. I mapped each kernel name → its source file/function.
> 3. I grepped every source for `atomic{Add,CAS,Max,Min,Exch}`, then confirmed the **enclosing
>    `__global__` kernel** and the **template/`SplitK` gating** so I only count atomics in the
>    *exact variants that ran*.
> 4. I repeated the whole thing for PyTorch against `pytorch_124M.sqlite` (96 launched kernels)
>    plus the cloned `pytorch_source/` tree.

---

## 0. TL;DR — the headline

| | My BluTrain 124M | PyTorch 124M |
|---|---|---|
| Kernels using atomics for **real gradient/data accumulation** | **6** | **1** (attention backward) |
| LayerNorm backward γ/β | **atomicAdd** (global) | **atomic-free** (block reduction over column-tiles) |
| LayerNorm backward input | atomicAdd on **shared** (block-scoped) | atomic-free (per-row warp reduction) |
| Embedding backward (the *launched* path) | **atomicAdd** scatter | **atomic-free** (sort + segment-reduce; atomic variant not launched) |
| Grad-norm (clip coefficient) | **atomicAdd** accumulator | atomic-free (two-pass `LpNorm` + cleanup) |
| Split-K GEMM | **atomicAdd** partials straight into C | separate `splitKreduce_kernel` pass (deterministic-style) |
| Attention backward dQ | **atomicAdd** | **atomicAdd** (`fmha_cutlassB`, officially non-deterministic) |
| Bias-gradient reduction | generic **reduce kernel** (staging+semaphore, atomic only on an integer counter → deterministic) | same algorithm (I ported it from them) |

**The crux I landed on:** **PyTorch is more deterministic than me.** They deliberately use
atomic-free algorithms in exactly the places I use atomicAdd — LN backward (γ/β + input),
embedding backward, grad-norm. The *only* op where both sides use atomics is **mem-efficient
attention backward dQ**, which PyTorch itself documents as non-deterministic. So I'm carrying
**~4–5 extra atomic-nondeterministic kernels** that PyTorch has already designed away.

> ⚠️ **Detection gotcha that nearly gave me a wrong answer:** PyTorch almost never writes a raw
> `atomicAdd`. They wrap it as **`gpuAtomicAdd`**, **`fastAtomicAdd`**, or
> `gpuAtomicAddNoReturn`. My first case-sensitive grep for `atomicAdd` missed all of them and
> almost reported PyTorch as fully atomic-free — which would have been wrong. I had to redo the
> whole scan with the wrapper names. Always grep for the wrappers too.

---

## 1. My 6 atomic kernels (124M, sm_89) — verified to launched variants

| # | Kernel | launches (10 steps) | atomic site | what it accumulates | scope |
|---|---|---|---|---|---|
| 1 | `sgemm_sm89_kernel<…,IsSplitK=1,…>` | **7,680** | [`Sgemm_core_template_sm89.cuh:312`](../mini_blutrain/BluTrain_BestPrecisionEnv_may27_406k/BluTrain/Tensor-Implementations/BluBridge-BLAS/src/level3/batched_gemm/FP32/SM89/Sgemm_core_template_sm89.cuh) | split-K GEMM partial products straight into C (some Linear backward dW/dX) | global float |
| 2 | `ln_backward_gamma_beta_sm89_kernel` | **8,000** | [`LayerNormKernels_sm89.cu:277-278`](../mini_blutrain/BluTrain_BestPrecisionEnv_may27_406k/BluTrain/Tensor-Implementations/src/Kernels/cuda/norm/arch/LayerNormKernels_sm89.cu) | grad_gamma / grad_beta summed across blocks | global float |
| 3 | `ln_backward_input_sm89_kernel` | **8,000** | [`LayerNormKernels_sm89.cu:330`](../mini_blutrain/BluTrain_BestPrecisionEnv_may27_406k/BluTrain/Tensor-Implementations/src/Kernels/cuda/norm/arch/LayerNormKernels_sm89.cu) | the two LN reduction sums (s1, s2) | **shared** (block-scoped, but still order-dependent) |
| 4 | `mem_efficient_bwd_unified_kernel_exp12` | **3,840** | [`AttentionBackward_sm89.cu:364`](../mini_blutrain/BluTrain_BestPrecisionEnv_may27_406k/BluTrain/Tensor-Implementations/src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu) | attention dQ accumulated across K-tiles/blocks | global float |
| 5 | `embedding_backward_kernel_optimized` | **640** | [`EmbeddingKernels.cu:143`](../mini_blutrain/BluTrain_BestPrecisionEnv_may27_406k/BluTrain/Tensor-Implementations/src/Kernels/cuda/embedding/EmbeddingKernels.cu) | token-embedding gradient scatter-add | global float |
| 6 | `multi_tensor_grad_norm_kernel` | **10** | [`MultiTensorKernels.cu:142`](../mini_blutrain/BluTrain_BestPrecisionEnv_may27_406k/BluTrain/Tensor-Implementations/src/Kernels/cuda/optimizer/MultiTensorKernels.cu) | per-block grad-norm² into one accumulator (feeds clipping) | global float |

So **6 kernel types break bit-reproducibility** — 5 are global float accumulations, 1
(`ln_backward_input`) is a shared-memory (block-scoped) atomic. The biggest contributors by
volume are **split-K GEMM (7,680)**, **LN backward γ/β + input (8,000 each)**, and
**attention dQ (3,840)**.

### 1.1 The atomic that is NOT a problem — my reduction kernel's election counter

| `unified_reduce_kernel_outer_vec4` | **15,680** | `atomicAdd(&semaphores…)` (**integer**) | election semaphore only — counts CTA arrivals to pick the combiner; the actual sum is combined in fixed order → **deterministic** |

This is the exact "staging-buffer instead of atomic-accumulate" design I built when I fixed the
bias-grad bug. The wise distinction I keep coming back to: **atomicAdd used for
*counting/sequencing* is fine and stays deterministic; only atomicAdd used for *accumulating FP
values* breaks reproducibility.**

### 1.2 Atomic sites that exist but did NOT launch (so I don't count them in the 6)

- `GELUKernels.cu:163` `atomicAdd` is inside `bias_grad_reduce_kernel` — **not launched** (my
  gpt-2 path uses the addmm reduce). The launched `gelu_forward`/`gelu_backward` are atomic-free.
- `GradNormKernels.cu` `grad_norm_squared_kernel` / `grad_norm_inf_kernel` (+ `atomicMaxFloat`
  via `atomicCAS`) — **not launched**; my grad-norm path is `multi_tensor_grad_norm_kernel`.
- `embedding_backward_kernel_cooperative` (my atomic-free ballot-dedup kernel, "inspired by
  PyTorch's `embedding_backward_feature_kernel`") exists but my launcher **always calls the
  atomicAdd version** → it's dead code (see §5 for why I left it that way).

### 1.3 The other ~25 launched kernels are atomic-free (I verified each)

GEMMs without split-K (`sgemm_addmm_sm89_kernel` 17,280; `sgemm_nt_256x128_sm89_2_k<false>`
15,720 — only the `<false>` non-split-K variant launched; `sgemm_sm89_scale_kernel` 7,680 plain
`C*=beta`); forward/elementwise (`layer_norm_forward`, gelu fwd/bwd, `fused_attn_forward`,
`mem_efficient_bwd_precompute_D`, vectorized adds, scalar mul/div, broadcast, transpose,
convert); loss (`sum_reduction_kernel` two-pass, all sparse-CE kernels — shared-mem block
reductions, 0 atomics); optimizer (`multi_tensor_adam`, `multi_tensor_scale`,
`compute_clip_coef`); `embedding_forward` (gather).

---

## 2. PyTorch's atomic usage (124M, same config, same 10 steps) — verified against cloned source

**PyTorch kernels that DO use atomics in this run:**

| PyTorch kernel | atomic? | verdict |
|---|---|---|
| `fmha_cutlassB…` (mem-eff **attention backward**) | **YES** | `kernel_backward.h`: `atomicAdd(&workspace_gq.counter,1)` (election, ~L2110) + `storeAtomicAdd`→`atomicAdd` (~L159) for gradQ. **Non-deterministic — PyTorch officially documents mem-eff attention backward as non-deterministic.** |
| `cublasLt::splitKreduce_kernel` | **unknown (closed)** | cuBLASLt is a closed-source `.so`, not in the repo, so I can't inspect it. But it's a **separate reduction pass** over split-K partials in a workspace (not atomic-into-C) — the *deterministic* split-K style. cuBLAS is generally deterministic for fixed shape+version. |

**PyTorch kernels that are atomic-FREE / deterministic — the ones that matter for my comparison:**

| Op | PyTorch kernel | verified |
|---|---|---|
| LayerNorm bwd γ/β | `GammaBetaBackwardCUDAKernelTemplate` | **0 atomics** — 2-D block reduction over a column-tile |
| LayerNorm bwd input | `layer_norm_grad_input_kernel_vectorized` | **0 atomics** — per-row warp/block reduction |
| Embedding bwd | `embedding_backward_feature_kernel` (shared-mem serialized accumulate) **and** sort-based `compute_grad_weight` + `sum_and_scatter` + cub-sort | **0 atomics**; the atomic variant `compute_grad_weight_atomic_accumulate` exists but was **NOT launched** |
| Grad-norm (clip) | foreach `LpNormFunctor` + `lpnorm_cleanup` | **0 atomics** — two-pass reduction |
| Reductions (bias-grad etc.) | `reduce_kernel` | **0 atomics** — staging buffer + integer semaphore (same as mine) |
| Softmax / cross-entropy | `SoftMax.cu`, `Loss.cu` | **0 atomics** |
| Elementwise, Adam (`FusedAdamMathFunctor`), copies, RNG | — | **0 atomics** |

Side-by-side: of my 6 atomic kernels, PyTorch matches me (also non-deterministic) in only
**one** — attention backward. For matmul, PyTorch reduces split-K in a separate clean pass while
I atomic-add into C. For the remaining four (LN γ/β, LN input, embedding, grad-norm) **PyTorch
is fully deterministic and I'm not.**

---

## 3. Can I make my 6 deterministic — fast and correct? (my per-kernel plan)

PyTorch's atomic-free kernel is a proven blueprint for each. Here's how I'd rank them by effort:

### ✅ Easy + clearly worth it (deterministic algorithm known, ~same speed)

**1. Grad-norm (`multi_tensor_grad_norm_kernel`) — easiest.** I'd replace
`atomicAdd(accumulator, blockpartial)` with a **two-pass reduction**: each block writes its
partial norm² to a scratch buffer; a tiny second kernel sums them (or I reuse my staging-buffer
reduction). This is exactly PyTorch's `LpNormFunctor`+`lpnorm_cleanup`. It's a 10-launch tiny
kernel → trivially deterministic, **no measurable speed cost.**

**2. LayerNorm backward input (`ln_backward_input_sm89`) — easy.** The non-determinism is a
**shared-memory** `atomicAdd(&s1,&s2)` doing a block reduction of two scalars. I'd replace it
with a **warp-shuffle + shared-memory tree** (which my `block_x/block_y_reduce` already
implement, and which PyTorch's `layer_norm_grad_input_kernel_vectorized` uses). Same speed,
deterministic.

**3. LayerNorm backward γ/β (`ln_backward_gamma_beta_sm89`) — moderate, high value (8000 launches).**
I'd replace the cross-block `atomicAdd(&grad_gamma/&grad_beta)` with PyTorch's
`GammaBetaBackwardCUDAKernelTemplate` scheme: a **2-D grid where each block owns a column-tile
and reduces rows with block reductions**, optionally writing per-block partials to a workspace
that a second pass sums (for very tall inputs). PyTorch ships this → comparably fast and deterministic.

### 🟡 Medium effort

**4. Split-K GEMM (`sgemm_sm89_kernel<…,IsSplitK=1>`, 7680 launches).** Today I `atomicAdd`
partials straight into C. The deterministic fix is the **cuBLASLt approach**: each split-K block
writes its partial to a **workspace**, then a **separate `splitKreduce`-style kernel** sums them
in fixed order (no atomics). It's real BLAS work but a standard algorithm. *Cheaper alternative:*
disable split-K for the affected shapes (small perf loss; the direct-store `IsSplitK=0` path
already exists and is deterministic).

### 🟠 Hard / lower priority

**5. Embedding backward (`embedding_backward_kernel_optimized`, 640 launches).** I'd replace the
global `atomicAdd(&grad_weight)` scatter with PyTorch's **sort + segment-reduce**
(`compute_grad_weight`+`sum_and_scatter`, needs a cub sort) or the **shared-mem serialized**
`embedding_backward_feature_kernel`. Both are deterministic and production-fast, but it's more
code. Worth it since embeddings tie to `lm_head` (large gradient). **Caveat from my own
experience:** my atomic-free `embedding_backward_kernel_cooperative` attempt was **16.9× slower**
on my wte shape (see §5) — so PyTorch's sort-based path is the right blueprint, not a naive
cooperative kernel.

**6. Attention backward dQ (`mem_efficient_bwd_unified_kernel_exp12`, 3840 launches) — hardest.**
PyTorch's `fmha_cutlassB` backward **also uses atomic/counter accumulation for dQ and is
officially non-deterministic.** A deterministic dQ would need a split-key workspace + separate
reduce (PyTorch's `kNeedsAccumGradQ` path still ends up non-deterministic). This is the one place
determinism is genuinely hard and **not even PyTorch provides it** — so it's my lowest priority.

### Already done

**Bias-gradient reduction** — I already made this deterministic with the staging-buffer +
integer-semaphore design (the reduce kernel's only atomic is the election counter). That's my
template for #1–#3.

**My honest bottom line:** I can make 4 of the 6 deterministic with little/no speed cost
(grad-norm, LN-input, LN-γ/β easily; split-K GEMM with more work). Embedding is doable
(sort-based) but more work. Attention dQ is the genuine hard case PyTorch doesn't solve either.

---

## 4. The bias-gradient reduction — the deep story (what kicked all of this off)

The full narrative is in [`bias_gradient_reduce_sum_kernel_usage.md`](bias_gradient_reduce_sum_kernel_usage.md);
here are the atomic-specific essentials.

### 4.1 The original bug — *not* an atomic, an unsynchronized overwrite race

Old code (`master_gau_latest_ada_6000_sm89/.../ReductionKernels.cuh:454`):

```cpp
if (is_leader) {                  // when !block_x_reduce, ALL blockDim.x lanes are leaders
    index_t slot = blockIdx.y + blockIdx.x * gridDim.y;   // SAME slot for all lanes
    cta_buf[slot] = value;        // PLAIN '=' — not atomic. Last write wins, partials lost.
}
```

My bias gradient came out scaled by `1/cpo` (the infamous "1/17", where 17 = `ctas_per_output`
for that shape on that GPU). It was a **data race**, not "atomic replace" — there was no atomic
at all.

### 4.2 The fix I shipped — PyTorch-faithful multi-CTA staging (NOT atomicAdd)

Staging buffer + `__threadfence` + **integer-semaphore `atomicAdd` for last-CTA election** +
the last CTA combines all `cpo` partials and writes `dst` once. I mirrored PyTorch's
`global_reduce` ([`Reduce.cuh:786-871`](../pytorch_source/aten/src/ATen/native/cuda/Reduce.cuh))
line-for-line. The semaphore atomicAdd is an integer counter — no float atomic on the gradient →
deterministic.

### 4.3 Why I keep staging instead of a direct `atomicAdd(&dst[col], partial)` — 4 reasons, ranked

My first instinct was "for small cpo they're the same speed, the only difference is determinism" —
that's *one* reason but not the main one. The full ranking:

1. **Generality (biggest).** `atomicAdd` only exists for *sum* (and only some dtypes). There's
   no `atomicMax`-for-argmax, no `atomicWelford`, no atomic for mean/variance. My reduction
   kernel has to do `sum, mean, min, max, prod, and/or, argmin, argmax, variance`. Staging does a
   normal `combine()` in the last block → **one code path works for every op.** atomicAdd would
   only ever handle sum.
2. **Determinism.** Staging reads partials in fixed order → reproducible bits; atomic order is
   random → non-reproducible bits.
3. **Precision.** Staging accumulates partials in **fp32** and writes once; a float atomicAdd
   accumulates in the **output dtype** — lossy for fp16/bf16 (historically not even supported).
4. **Speed at large cpo.** Contended float atomics get worse as cpo grows; at large cpo staging
   wins on speed too. (At *small* cpo they're comparable — the main element-reading loop dominates.)

### 4.4 How the last-CTA election actually works (nobody spins, nobody waits)

```cpp
int prev = atomicAdd(&semaphores[blockIdx.x], 1);   // returns the OLD value, atomically
s_is_last = (prev == ctas_per_output - 1);
__syncthreads();
if (!s_is_last) return;        // 16 of 17 blocks EXIT here — they do NOT spin/wait
```

`atomicAdd` hands out `0,1,2,…` and **never gives two blocks the same number** even if they
arrive together (that's the atomic guarantee). The block that draws `prev == cpo-1` is *by
definition* the last to arrive in real time (not the largest `blockIdx`). It stays, reads the
already-visible partials (made visible by `__threadfence`), and combines. Everyone else leaves.
**The election is per-BLOCK** — thread (0,0) bumps the counter on the block's behalf, then
`__syncthreads` + the shared `s_is_last` flag make the whole block continue/exit together. The
way I picture it: everyone drops their report in their own box-slot (parallel), grabs a numbered
ticket on the way out (the atomic dispenser), 16 go home, and whoever grabs ticket #16 stays and
totals all 17.

### 4.5 The atomicAdd experiment for the reduce kernel (I tried it, then reverted)

I tested replacing the staging machinery with a direct `atomicAdd(&dst[col], partial)`. My cost
model said it'd be **~16× less memory traffic**. But it **failed to compile**: `atomicAdd` has no
`int64_t` overload, and the kernel instantiates `out_scalar_t=int64_t` for int16/int32 SumOp (the
accumulator widens). It would need gating to float/half/bf16 outputs. I **reverted** it — the
diagnosis was solid, but between the dtype-gating quirks, the reasons in §4.3, and the bigger win
sitting on the table (BGRADB epilogue), it wasn't worth keeping. *One build gotcha that wasted my
time:* editing the `.cuh` header didn't recompile `ReductionImplGPU.o` (the Makefile doesn't track
transitive `.cuh` deps), so my first "test" ran stale code — I now always `rm` the `.o` + lib
after header edits.

### 4.6 The TensorFlow shared-mem-atomics alternative (I tried it, then reverted)

TF deliberately bypasses cub for bias gradient and uses `BiasGradNHWC_SharedAtomics` (shared-mem
accumulate → per-block global atomicAdd flush). My port came out **11% slower per call** (213 µs
vs 192 µs): my grid was under-populated (256 blocks on 142 SMs) and 2320 shared atomicAdds/thread
was too many. Reverted.

### 4.7 Who does what for bias-grad

- **PyTorch:** `nn.Linear` (3D) → `_flatten_nd_linear` → `at::addmm` ([`Linear.cpp:125`](../pytorch_source/aten/src/ATen/native/Linear.cpp));
  backward: addmm's `self`(bias) derivative passes `grad` un-reduced ([`derivatives.yaml:256`](../pytorch_source/tools/autograd/derivatives.yaml)),
  then `AccumulateGrad` → `reduce_grad` → `at::sum_to` → `sum_stub` → `gpu_reduce_kernel`. **No
  dedicated bias kernel, no atomicAdd-into-output. cuBLASLt `BGRADB` epilogue exists but PyTorch
  does NOT wire it into bias-grad** (they only fuse bias in *forward*).
- **Mine:** identical — `nn::Linear::forward` → `autograd::addmm` → `AddmmBackward` →
  `reduce_to_shape` → `reduce_sum` → `unified_reduce_kernel`. Same algorithm; I ported it.
- **My dead slow path:** `LinearKernels.cu:166 reduce_bias_kernel` (one thread/column, sequential
  16384-loop) is TP/tests only — gpt-2 never hits it.

### 4.8 My bias-grad performance vs PyTorch (apples-to-apples, 1568 calls each)

| shape (output) | MINE µs | PyTorch µs | winner |
|---|---|---|---|
| c_proj (768) | **35.2** | 62.7 | **MINE 1.78×** |
| c_attn (2304) | 481.3 | 223.7 | PyTorch 2.15× |
| c_fc (3072) | 573.0 | 293.4 | PyTorch 1.95× |
| pos_emb | 17.5 | 23.8 | MINE 1.36× |

The reason I lose on big outputs is block geometry. Mine is `32×16` (512 threads), PyTorch's is
`32×4` (128 threads) with more CTAs along the input-rows axis. PyTorch sets **`output_vec_size = 4`**
for outer-contiguous reductions when output dim % 4 == 0 → halves block height, 4× blocks-per-SM.
I had the field but never set it >1. My fix = `unified_reduce_kernel_outer_vec4` + float4 LDG/STG
- a host-side vec4 selector (shipped — that's the 15,680-launch kernel in §1.1).

---

## 5. Embedding backward — where atomic-free is *slower* on my shapes (the nuance I learned)

I assumed "atomic-free cooperative = always better", and I was **wrong for my wte/wpe shapes** —
I proved it to myself:

- My `embedding_backward_kernel_cooperative` (atomic-free, warp-ballot leader-election, modeled
  on PyTorch's `embedding_backward_feature_kernel`) ran **16.9× SLOWER** when I actually launched
  it (1.2 ms vs 71 µs per call, +72 ms/step). It was tuned for small N; my wte (V=50257) with
  heavy BPE-token duplication blew up the leader-election shared-mem path. I **reverted to atomicAdd.**
- My single atomicAdd kernel already **beats PyTorch's whole embedding-backward pipeline**:
  PyTorch splits it into **5 kernels** (192 launches, 7.87 ms) — a 2-pass sort-and-scatter for wte
  - an atomic feature kernel for wpe. I do it in **1 kernel, 64 launches, 4.40 ms → 44% faster.**

So PyTorch is more *deterministic* here, but my atomicAdd path is *faster* on these shapes. If I
want the deterministic win it has to come from PyTorch's **sort-based** path, not a naive
cooperative kernel.

> **Why atomics/RMW into global memory hurt (a micro-bench I ran):** "sum in a register, write
> once" vs "zero C then `C[i][j] +=` into global": on CPU it's ~a tie (the compiler
> register-promotes), but on **GPU the register version is 2.85× faster** — nvcc won't
> register-promote a global `+=` across a loop (aliasing), so every step is a global load+store.
> Same reason a global atomicAdd-into-output is expensive: it's memory traffic, not arithmetic.

---

## 6. What the atomics actually cost me — determinism & training impact

- FP addition is **non-associative**: `(a+b)+c ≠ a+(b+c)` in the last mantissa bits.
- atomicAdd arrival order is **random each run** → the bottom bits of every atomic-accumulated
  gradient change run-to-run. Staging-buffer reads (fixed order) are deterministic within a run.
- **The training symptom this explains:** my validation loss is stable only to ~the 2nd decimal
  and keeps fluctuating, while PyTorch stays stable to the 3rd decimal. Two same-seed runs slowly
  diverge as the tiny per-op differences (across *millions* of atomic adds × many steps) pile up.
  Carrying ~4–5 more atomic kernels than PyTorch is **very likely a real contributor** — I'll be
  honest that it's not the *only* source of run-to-run variation (DL has others), but it's a real one.
- Measured drift on my bias grads: ratio_median ≈ 1.00, all 48 biases within **±2%** element-wise,
  **cos > 0.99** vs PyTorch on every param; L2 norms deterministic to 4 sig figs. Step-0 parity:
  **148/148 dumps within 1 ppm** of PyTorch (~105 bit-equal, the rest 1–2 ULP).
- The persistent **~5–12% larger L2 vs PyTorch** on some params is a *separate* longstanding
  cpp↔pytorch numerical offset (cuBLAS algo choice, tf32 accumulation order), **not** from atomics
  — I verified that because my cpp↔cpp ratio_median is 0.98–1.00 and my forward is bit-deterministic
  (same loss every run).

### Where I could *beat* PyTorch on precision (haven't done it yet)

- Kahan/compensated summation on my critical reductions (loss, grad-norm, LN welford) — PyTorch
  uses it nowhere; ~1–2 extra digits at <1% perf cost.
- Deterministic reductions for Adam's `v` accumulator — PyTorch's atomic noise there hurts long-run
  Adam stability.

---

## 7. Cross-library context (how CUB / Eigen / TF / PyTorch do the cross-block combine)

| Library | Multi-CTA combine strategy | Atomics on the output? |
|---|---|---|
| **PyTorch** | staging buffer + integer-semaphore last-CTA election + combine | **No** (integer semaphore only) |
| **CUB** | strict **two-pass** (block partials → single-tile combine) for large reduce; one launch for small | **No** (lookback is for *scan*, not reduce — I confirmed by grep) |
| **Eigen** | atomic-only | Yes |
| **TensorFlow** | chains kernels / delegates to CUB; dedicated bias kernel uses shared-mem atomics | bias kernel: yes |
| **Mine** | PyTorch-faithful staging+semaphore (after my fix) | **No** on the generic reduce path |

A couple of my earlier claims I had to correct: CUB does **NOT** use decoupled lookback for plain
`DeviceReduce` (lookback lives in `agent_scan.cuh` / reduce-by-key only). No production library
does single-pass lookback reduce. My staging+semaphore single-kernel **is** CUB's two-pass inlined
into one launch (my staging buffer = CUB's `d_block_reductions`, my last-CTA-combine = CUB's
`SingleTileKernel`), so "switch to CUB two-pass" would be slightly slower (extra launch), no benefit.

---

## 8. The genuinely zero-overhead alternative for bias-grad — cuBLASLt `EPILOGUE_BGRADB`

For bias-grad the real win isn't a better reduce kernel — it's not having one. `CUBLASLT_EPILOGUE_BGRADB`
(cublasLt.h:1881) computes the bias gradient as a **free side effect** of the
`grad_weight = input.T @ grad_output` matmul I already do (it reduces B over the GEMM "k" axis →
`[N]` = exactly grad_bias). No separate launch, no memset, no staging, no shared mem. PyTorch has
it available but does **not** wire it into bias-grad backward. It's on my roadmap as the true
zero-overhead bias-grad path. (I already wired the cuBLASLt forward-bias epilogue fusion into
`GenMatmul.cu`.)

---

## 9. Appendix — my EOD writeup as I sent it (plain prose, team-facing)

> so first on my side, from my nsys report of the 10-step run (may27_with_blublas) i backtracked
> all 32 kernels that actually launched, opened each kernel's source and checked for atomic adds,
> and out of those, **six kernels use atomic add in a way that touches real gradient/data
> accumulation**: the split-k sgemm (atomic-adds partial products straight into the output matrix
> C), the layernorm backward gamma/beta kernel (atomic-adds grad_gamma and grad_beta across
> blocks), the layernorm backward input kernel (atomic-adds the two reduction sums — this one is
> on shared memory not global, so block-scoped but still order dependent), the attention backward
> kernel (atomic-adds the dQ gradient), the embedding backward kernel (atomic-adds the token
> gradients into grad_weight as a scatter), and the grad-norm kernel (atomic-adds each block's
> squared norm into one accumulator for the clipping coefficient).
>
> then on the pytorch side, from their nsys report (pytorch_124m) for the same config and same 10
> steps, i pulled all 96 launched kernels and backtracked each in their cloned source. one
> important correction i had to make: **pytorch never writes a raw atomicAdd — they wrap it as
> gpuAtomicAdd or fastAtomicAdd**, so my first search missed all of them and almost reported
> pytorch as fully atomic-free, which would have been wrong; i redid the search with the correct
> wrapper names.
>
> after doing it properly: **pytorch is deliberately using deterministic (no-atomic) algorithms
> in exactly the same places where i use atomic add.** their LN backward gamma/beta does a block
> reduction over column tiles with no atomic; their LN backward input does a per-row warp
> reduction with no atomic; their embedding backward uses a sort + segment-reduce path (and a
> shared-mem serialized accumulate kernel) and the atomic version was not even launched; their
> grad-norm uses a two-pass reduction (lpnorm + cleanup) with no atomic; and their reductions,
> softmax, cross-entropy, all elementwise ops and adam are atomic-free as well.
>
> the only two places pytorch itself uses atomic add: the memory-efficient attention backward
> (the fmha kernel) for dQ accumulation plus a counter — and pytorch officially documents this
> backward as non-deterministic — and the split-k gemm reduction (cublaslt splitKreduce kernel),
> which i couldn't fully verify because cublaslt is closed-source, but it runs as a separate
> reduce pass over a workspace (the deterministic style, not atomic-into-C), so most likely safe.
>
> putting both side by side: out of my six atomic kernels, pytorch matches me (also
> non-deterministic) in only one — attention backward; for matmul pytorch reduces split-k in a
> separate clean pass while i atomic-add into C; and for the remaining four (LN gamma/beta, LN
> input, embedding backward, grad-norm) pytorch is fully deterministic and i'm not. so effectively
> **i'm carrying around four-to-five extra kernels with atomic-add non-determinism that pytorch has
> already designed away.**
>
> the impact: atomic add lets whichever thread/block reaches first add first, so the order of FP
> additions changes every run, and FP add isn't associative, so the final value differs in the
> last few bits run-to-run. this is why my val loss is only stable to ~the 2nd decimal and keeps
> fluctuating while pytorch stays stable to the 3rd decimal, and across millions of these adds the
> tiny differences pile up — so my suspicion that non-deterministic accumulation contributes to my
> run-to-run loss divergence looks correct (not the only factor, but a real one).
>
> the most important point on my own reduction kernel: even though it also calls atomicAdd, it does
> NOT fall into this problem, because there the atomic add is only on a tiny **integer semaphore**
> to count how many blocks finished and elect the last block — a counting/sequencing use, not data
> accumulation. the actual summing is done afterwards by the elected last block reading the staged
> partials in a fixed order, so the result is the same every run. **the wise distinction: atomic
> add for counting is deterministic; only atomic add for accumulating FP values breaks
> reproducibility.** so in my reduction module i already escaped the problem by design, and my plan
> is to apply the same idea (scratch buffer + fixed-order combine, or a two-pass reduction like
> pytorch) to the other kernels for the b300 setup, then re-run 3–4 full runs to confirm the loss
> stabilizes, after discussing with the team.

---

## 10. Sources

- My working sessions (May 2026): the dedicated 6-atomic-kernel-vs-PyTorch backtrack + the
  determinism/semaphore deep-dive, and the main optimization/parity sessions (bias-grad bug,
  atomicAdd & TF experiments, embedding cooperative test).
- nsys reports: `may27_WITH_BLUBLAS.sqlite` (mine, 32 launched kernels) and `pytorch_124M.sqlite`
  (PyTorch, 96 launched kernels).
- Cloned source trees in this repo: `pytorch_source/`, `cub/`, `eigen/`, `cutlass/`, `tensorflow/`.
- Related docs: [`bias_gradient_reduce_sum_kernel_usage.md`](bias_gradient_reduce_sum_kernel_usage.md)
  (§8–12 reduce-kernel deep dive), [`reduction_algorithm_deep_dive.md`](reduction_algorithm_deep_dive.md),
  [`matmul_learning_journey.md`](matmul_learning_journey.md) (split-K + cuBLASLt epilogue).

> **Verify before I cite this.** The line numbers/counts are from the May 2026 tree + my 10-step
> nsys runs; the active repo is `may27_406k` flattened into the GOUTHAM1926 repo. I should re-grep
> `atomic` (including `gpuAtomicAdd`/`fastAtomicAdd` for PyTorch) and re-pull the nsys kernel list
> before quoting any number in a report. Two things I couldn't verify: the cuBLASLt
> `splitKreduce_kernel` internals (closed source) and the full internals of all 96 PyTorch library
> kernels (I only opened the direct analogs of my 6 by source).
