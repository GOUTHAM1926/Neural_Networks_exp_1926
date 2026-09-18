# BluTrain Precision Regression Investigation

**Date:** 2026-05-27
**Repos compared:**
- `BluTrain_BestPrecisionEnv_original/` — the older repo that achieved the **best validation loss** (better than PyTorch).
- `BluTrain_BestPrecisionEnv_modified_latest/` — the newer repo that **beats PyTorch on throughput** but has a **higher saturation validation loss**.

**Goal:** find every code change between these two folders that could explain the loss regression.

---

## 1. How the comparison was done

A full recursive `diff` was run between the two `BluTrain/` source trees. Anything that isn't actual source code (logs, csv files, `.nsys-rep` profiles, `.bin` checkpoints, `.o` object files, build outputs, `.git`, `__pycache__`, `_runner_*` binaries, lock files) was filtered out so we only look at things that change behaviour.

For each remaining differing file we generated a unified diff and then **separated real code changes from commented-out backup code**. A lot of the diff size is just old code being commented out for safekeeping — that has no runtime effect. Only the live `+`/`-` lines that aren't pure `//` comments matter.

---

## 2. Files that DID NOT change in any meaningful way

These files differ between the two folders, but the actual changes are cosmetic (renaming a log file, commenting out a print, adjusting trailing newlines). They cannot cause the precision drop.

| File | What actually changed | Verdict |
|---|---|---|
| `gpt2.py` | Commented out a few prints, changed log filename to `may27_bucket_check.*`, added a "Bucket cap: …" print | No effect on math |
| `gpt2_fmha_ddp.cpp` | Uncommented the validation / token-generation / checkpoint blocks (they were temporarily disabled for a 4-step benchmark), restored `WARMUP_STEPS=715` and `MAX_STEPS=19073`, changed a log filename, bumped print precision from 6 to 8 digits | No effect on math |
| `Makefile` | Output binary renamed from `_runner_44M` to `_runner_0` | No effect on math |
| `Tensor-Implementations/include/ops/helpers/ReductionOps.h` | Just a trailing newline added | No effect on math |
| `Tensor-Implementations/src/Kernels/cuda/matmul/MatmulBackward.cu` | Just a trailing newline added | No effect on math |
| `Tensor-Implementations/BluBridge-BLAS/.../Sgemm_core_template_sm89.cuh` | Just a trailing newline added | No effect on math |

**Plain English:** these are all "cleanup" diffs. They look big in `diff -u` output because of formatting, but if you look at the live code, nothing about the numbers being computed has changed.

---

## 3. Throughput-only changes — same numbers, faster execution

These are real code changes, but each one produces **bit-identical results** to the original. They only make training faster.

### 3.1 `dist/communication/include/CpuSync.hpp` — added `Work::rearm()`
In the original, every NCCL all-reduce call did `cudaEventCreateWithFlags(...)` and then `cudaEventDestroy(...)` for its own `Work` object. With ~20 buckets × every training step, that's a lot of CUDA event handle churn on the CPU side, which slows down kernel launches.

The new version adds a `rearm()` method that resets an existing `Work` object's state so the **same CUDA event handle** is reused across steps. Numerically, this is a no-op: a CUDA event handle is just a synchronization primitive; the actual NCCL all-reduce maths is unchanged.

### 3.2 `dist/communication/include/ProcessGroupNCCL.h` + `processGroupNccl.cpp` — added a Work-reuse overload
A second overload of `all_reduce_async()` was added that takes a `reuse` `Work` pointer. If `reuse` is non-null, the existing Work is re-armed instead of allocating a new one. This is the API that hooks into the change above.

The actual NCCL call (`ncclAllReduce(...)`) is identical — same buffer, same count, same dtype, same op. **Bit-identical output.**

### 3.3 `Tensor-Implementations/src/TensorOps/cuda/TensorOpsAdd.cu` — vectorized elementwise add
This is a big diff, but it's a pure rewrite for throughput:
- **Old:** scalar add — each thread does `output[i] = a[i] + b[i]` for one element.
- **New:** vectorized add — each thread loads a `float4` (or `int4` reinterpreted as 4×`half2` / 4×`bfloat162`) and does 4 (or 8) adds at once, with `UNROLL=2`.

For `fp16` and `bf16` the new code uses `__hadd2` (paired). Crucially, `__hadd2(a, b)` does each lane independently — exactly the same maths as two scalar `__hadd` calls. Same for `float4` add. So although the code looks completely different, **the result for every output element is exactly the same**.

It also caches the result of `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` in a `static` so we don't redo that query every call.

### 3.4 `Tensor-Implementations/BluBridge-BLAS/.../Sgemm_addmm_sm89.cu` — store batching + in-bounds fast path
Two real changes:
1. **Fast path for in-bounds tiles:** when the whole 16-byte load is inside both M/N and K bounds (`a_in_bounds` / `b_in_bounds`), the kernel now uses unpredicated `cp.async.cg.shared.global` instead of the predicated version. Same data is loaded, just with fewer instructions.
2. **Pair store:** instead of calling the `store(r, c, acc, bias)` lambda 4 times, the new code calls `store_pair(r, c0, acc0, acc1, b0, b1)` twice and uses a single `float2` write per pair. The actual FMA operations (`__fmaf_rn(alpha, f, ...)`, `__fmaf_rn(beta, dst, s)`) are **identical in every way** — same operands, same order, same rounding. Just the store-instruction width changed.

**Verdict:** both changes are throughput-only.

---

## 4. Changes that COULD affect precision — the actual suspects

This is the part that matters. The following changes alter either *what numbers are computed* or *what order they are summed in*. In floating-point arithmetic, summation order matters because `(a+b)+c` and `a+(b+c)` can differ in the last bit. Across millions of operations over thousands of training steps, those last-bit differences compound and show up as differences in the loss curve.

### Suspect 1 — Attention forward & backward kernels rewritten (BIGGEST suspect)
**Files:**
- `Tensor-Implementations/src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu`
- `Tensor-Implementations/src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu`

Both QK^T and PV matmuls in the forward kernel, **and** all three (dQ, dK, dV) matmuls in the backward kernel, were completely re-implemented. The old code used the higher-level CUDA `nvcuda::wmma` API. The new code talks directly to the Tensor Cores via inline PTX (`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`).

Both versions use TF32 precision with the same `+= 0x1000u` round-to-nearest-even bias trick before each mma. So the **per-mma operation is the same algorithm**. But the **accumulator structure changed**.

**Old accumulator pattern (per output tile):**
```
sub[0], sub[1], sub[2], sub[3]  — 4 separate accumulators
for k in 0..SCORE_K_TILES:
    sub[k & 3] += a_frag * b_frag    // round-robin into 4 partial sums

// At the end, tree-reduce:
sub[0] += sub[1]
sub[2] += sub[3]
sub[0] += sub[2]
```

**New accumulator pattern (per output tile):**
```
acc_l[4], acc_r[4]  — 2 accumulators, one for each n=8 half of the m16n16 tile
for k in 0..SCORE_K_TILES:
    mma(acc_l, a, b_left)    // straight in-place accumulation
    mma(acc_r, a, b_right)
```

Why does this matter? With `BK_TILE=64` and `HeadDim=64`, `SCORE_K_TILES = BK_TILE / WMMA_K = 64/8 = 8`. So each attention score is built from **8 sequential mma accumulations**.

- The **old code** distributed those 8 mmas across 4 partial sums (2 mmas each), then **tree-summed** the 4 partial sums. A tree sum is one of the most numerically stable ways to add many numbers — error grows as O(log N).
- The **new code** sums all 8 mmas directly into 1 accumulator. A linear sum has error that grows as O(N).

Plain English: imagine adding up a long column of numbers. If you sum them all into one running total, small rounding errors can accumulate in one direction. If you instead add them in pairs, then add the pairs, then add the pair-of-pairs (a "tree"), errors tend to cancel out better. The new attention kernel switched from the second method to the first method. **For a single mma the difference is tiny (a few ULPs in FP32). But these tiny errors flow through softmax — which is extremely sensitive to small input changes because of the exponential — and then through the value-times-attention matmul, and then through 12 transformer layers, and then through 19,073 training steps. Tiny drift becomes visible drift.**

**Why this is the prime suspect:** softmax in attention is the most numerically sensitive operation in the model. A 1 ULP change in the pre-softmax score can become a noticeably different probability after the exp. Any kernel change touching the attention accumulator is exactly where precision-vs-throughput tradeoffs are usually made.

### Suspect 2 — GEMM split-K dispatch threshold widened
**File:** `Tensor-Implementations/BluBridge-BLAS/src/level3/batched_gemm/FP32/SM89/Sgemm_sm89_wrappers.cu`

Three identical lines (probably one for each of forward NN, backward NT, backward TN) changed from:
```cpp
if (total_blocks * 2 < sm_count && K >= 512) { /* dispatch split-K kernel */ }
```
to:
```cpp
if (total_blocks * 4 < sm_count * 3 && K >= 512) { /* dispatch split-K kernel */ }
```

What this means in plain English:
- `total_blocks` is roughly "how busy is the GPU going to be with this GEMM".
- `sm_count` is how many SMs (streaming multiprocessors) the GPU has.
- **Old condition:** split-K only kicks in when `total_blocks < sm_count / 2`, i.e. when the GEMM is *very* under-utilizing the GPU (less than half the SMs are busy).
- **New condition:** split-K kicks in when `total_blocks < sm_count * 0.75`, i.e. whenever the GEMM is under 75% utilizing the GPU.

**What split-K does:** instead of computing each output tile by running one thread block through the entire K dimension, split-K splits the K dimension into multiple chunks, runs them in parallel across more thread blocks, and then **adds the partial sums together** in a separate reduction pass (typically using atomicAdd or a follow-up kernel).

**Why this affects precision:** the partial-sum reduction across K-splits **changes the order of FP32 additions** along the K dimension. A normal GEMM does `c = sum_{k=0..K-1} a[k] * b[k]` in one continuous accumulation. A split-K GEMM does `c = sum_{split} ( sum_{k in split} a[k]*b[k] )`. Same maths, different rounding pattern. A few ULPs of difference per output.

**Why this matters for the regression:** the new threshold means MORE GEMM shapes now go through split-K. Specifically, any shape where `0.5 * sm_count ≤ total_blocks < 0.75 * sm_count` was on the non-split-K path in the original, but is on the split-K path in the new code. Those shapes are getting a different accumulation order in the new code.

GPT-2 has a lot of medium-sized GEMMs (think: the QKV projection, the MLP up/down projection at smaller batch sizes), and quite a few of them may fall into exactly that newly-split-K band.

### Suspect 3 — DDP bucketing logic was substantially reworked
**Files:**
- `dist/Data-Parallel/include/DataParallel.hpp`
- `dist/Data-Parallel/src/DataParallel.cpp`

This is the biggest *behavioural* change in the diff, even though most of it is supposed to be "just" a throughput optimization. Several things changed at once:

#### 3a. A bug fix in `BucketInit::bucket_size`
The original had `size_t bucket_size;` — **uninitialized**. The default constructor was `BucketInit() : bucket_max_size(0) {}`, which set `bucket_max_size` but **left `bucket_size` holding stack garbage**. The new code initializes it to `0` and the header has a long comment explaining the consequence:

> *bucket_order_decide reads `(bucket_size + tensor.nbytes() <= bucket_max)` to decide coalescing on a freshly-constructed BucketInit. Left uninitialized it holds garbage, the test fails for every parameter, and each param lands in its own bucket → one tiny NCCL all-reduce per parameter (~148) instead of ~20 grouped 25 MB buckets, wrecking comm efficiency and backward/comm overlap.*

**Plain English:** in the original repo, because of the uninitialized variable, every single one of ~148 parameters was getting its own NCCL all-reduce call instead of being grouped into ~20 buckets. That's a massive throughput hit (which the new code fixes), but on the precision side it actually might not matter — NCCL's all-reduce is element-wise across ranks regardless of how the buffer is grouped, and the `avg` op divides by world_size at the end. So in principle the per-parameter gradient values after all-reduce should be identical either way.

**BUT** — there's a subtle question: when gradients are reduced one at a time in flight, the optimizer might end up using slightly different orderings or interactions with the gradient accumulator. This deserves a closer look.

#### 3b. `bucket_order_decide` always sorts now (was conditional)
**Old:**
```cpp
if(param_backward_order.has_value()) {
    std::sort(...);   // sort buckets by first-param-index
}
```
**New:**
```cpp
std::sort(...);   // always sort
```

Without sorting, `final_order_buckets` came out of an `unordered_map` in hash order — i.e. arbitrary. The new code forces the order to be by ascending first-parameter-index, which after the constructor's `std::reverse(...)` puts the *last-registered* params (typically the embeddings under weight tying) first or last depending on flow.

**Plain English:** this changes **which bucket fires first** in the all-reduce, but it doesn't change **what gradient values** end up summed. So this is likely throughput-only too — but it does mean the new code has deterministic bucket order while the old code had hash-arbitrary bucket order.

#### 3c. `apply_first_bucket_cap()` added — small leading bucket
A new method peels off a ~1 MB bucket from the end of the registration-order bucket list. After the reverse, this small bucket becomes bucket #0 in reduction order, so the first all-reduce launches as soon as the very last gradient is ready, filling the comm pipeline early.

**Plain English:** purely a launch-timing optimization. Doesn't change any numbers.

#### 3d. `pre_forward` now calls `rebuild_buckets()` — was commented out in original
**Old:**
```cpp
void DataParallel::pre_forward(){
    // if(rebuild_bucket){
    //     rebuild_buckets();
    // }
    ...
}
```
**New:**
```cpp
void DataParallel::pre_forward(){
    if(rebuild_bucket){
        rebuild_buckets();
    }
    ...
}
```

And `rebuild_buckets()` itself was rewritten to repack buckets in the order gradients were actually produced during the first backward pass, instead of going through `bucket_order_decide` (which would have re-sorted them by param index).

**Plain English:** on step 2 of training, the new code reorganizes which parameters go into which bucket based on what was learned about gradient-ready order in step 1. This is supposed to improve comm/backward overlap. The actual gradient values aren't changed by this — it just changes which parameters get all-reduced together. As long as NCCL all-reduce is bit-deterministic, this should be precision-neutral. **But it is a code path that is now active in the new repo and was inactive in the old one, so it's a candidate worth ruling out.**

#### 3e. `mark_param_ready` skips the D2D copy when `grad_ptr == bucket_ptr`
**Old:** always called `OwnTensor::device::copy_memory(...)` to copy each parameter's gradient into the bucket's flatten_gradient buffer.

**New:** checks if `parameters_[variable_index].grad()` already points to the same memory as the bucket view; if yes, skips the copy. The comment explains that `finalize_backward`'s `set_grad()` makes the param grad alias the bucket view from step 2 onward, so the copy was copying onto itself.

**Plain English:** if the grad and the bucket are literally the same buffer, skipping the memcpy is fine. The values are the same. This is throughput-only.

#### 3f. Unused-parameters set is now cached
**Old:** every backward, ran a full autograd-graph BFS to find unused parameters.
**New:** runs the BFS once on the first backward, caches the result, reuses it.

For a static model like GPT-2 the topology is identical every step, so this is safe and throughput-only. **But if the cache is somehow wrong on a graph where it actually does change, gradient sync for the "wrongly unused" param would be skipped → precision impact.** For GPT-2 this is almost certainly fine.

---

## 5. Putting it all together — what's the most likely culprit?

Ranking by likelihood of explaining the val loss regression:

### Tier 1 — Most likely
**The attention kernel rewrite (Suspect 1).** This is the only change in the whole diff that *unambiguously* alters the floating-point accumulation pattern of a math-critical operation. The switch from 4 tree-summed sub-accumulators to 2 linearly-summed accumulators measurably reduces numerical stability of every QK^T, PV, dQ, dK, and dV matmul in every attention block in every layer of every step. Softmax then amplifies these tiny differences.

If only one revert is going to be tried first, this is the one.

### Tier 2 — Plausibly contributing
**The GEMM split-K threshold widening (Suspect 2).** This changes the K-dimension accumulation order for any GEMM that falls into the `0.5*sm_count ≤ total_blocks < 0.75*sm_count` band. GPT-2 has many medium-sized GEMMs and some of them will land in this band. Split-K reductions are a known source of small FP32 non-determinism.

### Tier 3 — Possible but less likely to be the cause
**DDP bucketing changes (Suspect 3).** Most of these are supposed to be throughput-only, and on paper NCCL all-reduce results should be bit-identical regardless of bucket grouping. But this is the biggest behavioural change in the codebase, and a subtle interaction (e.g. the new `rebuild_buckets` rearrangement triggering different float accumulation in `mark_param_ready` or in the optimizer after sync) could be involved. Worth ruling out **after** the kernel-level reverts.

---

## 6. Recommended experiment order

Each step here is "revert just this thing in the latest repo, rebuild, train for ~250–500 steps, log val loss, compare to the original".

**Step 1 — revert attention kernels:**
- Copy `Tensor-Implementations/src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu` and `AttentionBackward_sm89.cu` from `original/` over the same files in `modified_latest/`.
- Rebuild via `make`.
- Run training, watch the val loss after ~500 steps.
- **Expected:** if this is the culprit, val loss will drop closer to the original's curve.

**Step 2 — revert split-K threshold:**
- In `modified_latest/.../Sgemm_sm89_wrappers.cu`, change all three occurrences of `total_blocks * 4 < sm_count * 3` back to `total_blocks * 2 < sm_count`.
- Rebuild, train, compare.

**Step 3 — disable rebuild_buckets in pre_forward:**
- In `modified_latest/dist/Data-Parallel/src/DataParallel.cpp`, in `pre_forward()`, comment out the `if(rebuild_bucket){ rebuild_buckets(); }` block again.
- Rebuild, train, compare. (Note: this will hurt throughput, that's fine — we're testing precision only.)

**Step 4 — combinations:**
- If no single revert fixes it, try combining Step 1 + Step 2.
- Then Step 1 + Step 3.
- Then all three.

**What to track in each experiment:**
- Train loss at steps 250, 500, 1000.
- Val loss at every VAL_FREQ checkpoint (default 250).
- Plot both against the original repo's `may19_final_check_*` curves and look for the inflection point where the curves diverge — that tells you whether the regression is from the start (suggests Tier 1) or builds up gradually (could be Tier 2 or 3).

---

## 7. What NOT to spend time on

To save investigation time, the following can be ruled out without running anything:
- `gpt2.py` changes — purely cosmetic.
- `gpt2_fmha_ddp.cpp` changes — purely commented-out code being uncommented.
- `Makefile` changes — output binary rename.
- `TensorOpsAdd.cu` rewrite — `__hadd2` vs `__hadd` and `float4` vs `float` produce identical results for elementwise add.
- `Sgemm_addmm_sm89.cu` store-pair change — same FMAs in same order, just batched stores.
- `CpuSync.hpp` `rearm()` — CUDA event reuse, no math.
- `ProcessGroupNCCL` Work-reuse overload — same `ncclAllReduce` call underneath.

---

## 8. Summary in one paragraph

Between the original (best-precision) repo and the latest (best-throughput) repo, the code changes fall into three buckets: (a) **cosmetic / log-name changes** with no runtime effect, (b) **throughput-only kernel and infrastructure rewrites** that produce bit-identical results, and (c) **a small number of changes that actually alter floating-point accumulation order**. The third bucket is what to investigate. By far the most likely cause of the val loss regression is the **rewritten SM89 attention kernels** — they switched from a 4-accumulator tree-summed mma pattern (numerically stable) to a 2-accumulator linearly-summed pattern (less stable). Secondary suspect is the **widened split-K threshold** in the FP32 GEMM dispatcher, which changes K-reduction order for medium-sized GEMMs. Third-tier suspect is the **DDP bucket rebuild logic** that is now active and wasn't before. Revert these in that order, rebuild, and watch the val loss curve to confirm.
