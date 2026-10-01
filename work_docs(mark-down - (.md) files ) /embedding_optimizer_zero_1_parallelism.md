# Embedding, ZeRO-1 optimizers and parallelism: what we learned on 2026-09-30

This file records everything from the 2026-09-30 session: every question asked, every answer, every finding, and
the F4 fix that was made and pushed. It is written from zero. Every claim points to the code line that proves it.
Section 17 adds the 2026-10-01 measurement of how much D1, D2 and D3 lose.

**Which code is meant.** Two team branches are involved. They are different, so each fact names its branch.
- `dist_side_merge`: our SFT branch. Local clone `~/Downloads/BluTrain_sft`, BluTrain `909ab75`, Tensor-Implementations
  (TI) `d96b991`. The trainer is `bluscript_zero_ckpt.cpp`. Unless a line says otherwise, "our trainer" means this one.
- `dist_side_merge_BLUCCL`: the colleagues' BluCCL branch. BluTrain `a3aba3e` (Navin, 2026-09-30) plus our 2 commits
  `149249d` and `3298e74`, pushed today. TI `b975be5`. The fused trainer is `bluscript_zero_ckpt_bluccl_fused.cpp`.

**Our model** (our trainer, `bluscript_zero_ckpt.cpp:127-140`, `:200-202`): 26 layers, 2,304 numbers per token
(`d_model`), a vocabulary of 50,304 tokens, weight tying on, B = 8 sequences of T = 4,096 tokens per micro-step
(32,768 positions), 1,048,576 tokens per optimizer step. At 1 GPU that is 1,048,576 / 32,768 = 32 micro-steps per
optimizer step (`:928`: `GRAD_ACCUM = global_batch / tokens_per_micro`).

**Background reference: `ddp_vs_zero_1.md`** (same folder). It explains from zero what DDP and ZeRO-1 are, how the
gradients and weights move between GPUs (all-reduce = reduce-scatter + all-gather), ZeRO-2/3, and the other kinds of
parallelism. Read it first if "reduce-scatter", "shard" or "all-gather" are unclear; this file assumes it.

**The two cases** used all through this file:
- **fp32 table:** the embedding table `wte` is stored in fp32 (the default; `BLU_BF16_EMBED=0`).
- **bf16 table:** `wte` is stored in bf16 (`BLU_BF16_EMBED=1`, which `BLU_ALL_BF16=1` switches on).

Contents:
1. fp32, bf16 and "rounding"
2. The embedding table and weight tying
3. The forward, one operation at a time
4. The backward, one operation at a time
5. Which rows of the table change at the optimizer step
6. The embedding-backward kernels (both branches)
7. Where `wte`'s gradient is rounded to bf16: D1, D2, D3
8. The send-dtype flags `BLU_BF16_REDUCE` and `BLU_FP32_REDUCE`
9. The ZeRO-1 optimizers: v1, v2 and Muon
10. The #2 fix (temporary buffers)
11. F1 (NaN/Inf gradients)
12. F4 (the 1-GPU bf16 round trip): found, fixed, tested, pushed
13. Navin's commit `a3aba3e`: what we found
14. What happened today, in order
15. Every question of the day, with its short answer
16. Open items for 2026-10-01
17. Measured on 2026-10-01: how much D1, D2 and D3 lose (local test)

---

## 1. fp32, bf16 and "rounding"

**fp32** stores a number in 4 bytes. It keeps 24 binary digits of the number.
**bf16** stores a number in 2 bytes. It keeps only 8 binary digits.

A computer can store only certain numbers exactly. Between 1 and 2:
- fp32 can store 8,388,608 different numbers. The gap between two of them is 0.00000012.
- bf16 can store only 128 different numbers: 1 + k/128. So near 1.0, the numbers bf16 can store are
  1.0, 1.0078125, 1.015625, 1.0234375, and so on. The gap is 0.0078125.

The gap grows with the size of the number. Near 0.003, the bf16 gap is only 0.0000153. So small numbers on their
own are stored well in bf16. The trouble comes only when a small number is added to a big one.

**Rounding.** When a result has to be stored in bf16, and the exact result is not one of the numbers bf16 can
store, the computer stores the nearest one. That change is called rounding. One rounding changes a number by at
most half the gap. That is at most 1/256 of the number, about 0.39%. In fp32 the most is 0.000006%.

**The half-gap rule.** Near 1.0 the gap is 0.0078125, so half the gap is 0.0039. If you add anything smaller than
0.0039 to 1.0 and store the result in bf16, the result rounds back to 1.0. The added value is lost completely.

Worked example (computed exactly):
- 1.0078125 + 0.0030060 = 1.0108185 (exact).
- The two bf16 numbers around it are 1.0078125 and 1.015625.
- The distance to 1.0078125 is 0.0030060. The distance to 1.015625 is 0.0048065.
- 1.0078125 is nearer, so bf16 stores 1.0078125, the same number as before the add. The 0.003 is gone.
- bf16 cannot store 1.01. bf16 does not keep "3 digits after the point"; it keeps 8 binary digits of the whole number.

**Widening bf16 to fp32 is exact.** A bf16 number is the top 16 bits of an fp32 number. Turning bf16 into fp32 adds 16
zero bits at the end. The value does not change at all. So widening never adds or loses anything.

**Adding two bf16 numbers.** The output is stored as bf16, but the exact sum of two bf16 numbers is usually not a
number bf16 can store. So the GPU computes the exact sum and stores the nearest bf16 number. That choice is the
rounding. It is part of the one add, not an extra step. The library's bf16 add does it with `__hadd2`
(`TensorOpsAdd.cu:195-205`) and with `float(a) + float(b)` converted back (`Types.h:525-527`). We tested 1,000,000
random bf16 pairs: "add in fp32, then round to bf16" gives exactly the same number as a correctly rounded bf16 add in
all of them (0 differences). So doing one add in fp32 and still storing bf16 changes nothing.

**Why one rounding is small but many roundings add up.** One rounding at the end of a long sum changes the result by
at most 0.39% of the result. But if the running total is rounded after **every** add, each small value can be lost
completely (the half-gap rule), and the losses pile up. If the total of 1.0 gets 300 more values of 0.003 each, the
exact total is 1.9. In bf16 with a rounding after every add, it can stay at 1.0.

---

## 2. The embedding table and weight tying

The embedding table `wte.weight` has shape [50,304 × 2,304]: one row of 2,304 numbers for every token id.

**Weight tying** (`weight_tying = true`) means the same table is used twice:
1. At the very start, as the **lookup** table: each position gets a copy of its token's row.
2. At the very end, as the **lm_head**: `logits = h × wteᵀ`, the score of every token at every position.

There is only **one** real weight. The lm_head has no weight of its own. The optimizer updates only `wte.weight`.

With `weight_tying = false`, the lm_head is its own separate `nn::Linear` (bf16, `bluscript_zero_ckpt.cpp:731`), with
its own starting values, its own gradient and its own optimizer state.

---

## 3. The forward, one operation at a time

Our trainer, `BLU::forward` and the blocks, with shapes and dtypes. Case 1 = fp32 table, case 2 = bf16 table.

**Before op 1:** the data loader reads the token ids from disk: `idx`, shape [8, 4096], one whole number per
position (for example 262 for ` the`). Nothing else happens before the lookup.

**Op 1. The lookup** (`wte.forward(idx)`, `:748`; `Embedding::forward` → `autograd::embedding`).
- Inputs: the table [50,304 × 2,304] and the ids [8, 4096].
- For each of the 32,768 positions it **copies** one row of the table: the row of that position's token id.
  Nothing is multiplied. (Mathematically a lookup equals a multiplication by a one-hot matrix, but that would be
  32,768 × 50,304 × 2,304 multiply-adds, almost all with zero, so it is done as a copy.)
- Output: [8, 4096, 2304]. Its dtype is the **table's** dtype (`EmbeddingOps.cpp`: output dtype = weight dtype).
- If ` the` sits at 300 positions, those 300 positions all get an exact copy of row 262.

**Op 2. Convert to the residual dtype** (`to_dtype(wte.forward(idx), cfg.resid_dtype())`, `:748`).
- `to_dtype` returns its input unchanged when the dtype already matches (`ReshapeOps.cpp:63-70`), and then no
  backward node is built at all.
- In both of our cases it does nothing: fp32 → fp32, or bf16 → bf16.
- The result `x` is the **start of the residual stream** (see below).

**The residual stream.** "Residual" means: a block adds its result to its own input. It does not replace the input.
- Attention block: `return autograd::add(x, proj);  // residual` (`:648`).
- MLP block: `return autograd::add(x, out);  // residual` (`:687`).
- Example with 3 numbers instead of 2,304: x0 = [0.5, −1.0, 0.2]; attention computes [0.1, 0.3, −0.2];
  x1 = x0 + that = [0.6, −0.7, 0.0]; the MLP computes [−0.2, 0.1, 0.4]; x2 = [0.4, −0.6, 0.4].
- The loop `x = (*attn[i])(x); x = (*mlp[i])(x);` (`:756-757`) runs 26 times, 52 blocks. So at the end,
  x_final = x0 + what block 1 added + … + what block 52 added. This one vector that goes through all layers and gets
  added to 52 times is the residual stream.

**The residual dtype and the table dtype are 2 separate switches** (`:189-195`, set at `:872-876`):
- `bf16_embed`: the dtype of `wte`. `bf16_resid`: the dtype of the residual stream and of the block RMSNorm weights.
- `BLU_ALL_BF16=1` turns both on (and the logits). Each also has its own flag that overrides it
  (`BLU_BF16_EMBED`, `BLU_BF16_RESID`).
- Our 2 cases happen to match (fp32 + fp32, or bf16 + bf16). They can differ: `BLU_BF16_EMBED=1` alone gives a bf16
  table with an fp32 residual, and then op 2 really converts. The code comment at `:862` lists
  `BLU_ALL_BF16=1 BLU_BF16_EMBED=0` = bf16 residual + fp32 table.
- `x` from op 2 is used twice in layer 1's attention block: by the RMSNorm (op 3, `:581`) and by the residual add
  (op 7, `:648`).

**Op 3. RMSNorm of layer 1's attention** (`norm.forward(x)`, `:581`).
- Out: `h1` [8, 4096, 2304]. For each position alone: r = 1 / √(mean of x² + 0.00001), then h1 = x × r × γ, where γ is
  the norm's own weight [2304].
- The norm is created in the residual dtype (`norm(c.d_model, c.rms_eps, c.resid_dtype())`, `:541`, `:666`, and
  `norm_f` `:715`). Case 1: input, γ and output fp32. Case 2: all bf16.
- **The maths inside is always fp32.** On H200 a bf16 input goes to `rms_norm_forward_sm90_cuda`. It widens each bf16
  number to fp32 (`RMS_NormKernels_sm90.cu:239`), sums the squares in fp32, computes r in fp32 (`:245`), multiplies in
  fp32, and rounds to bf16 only when it writes the output (`:269`, `:272`). There is no separate conversion kernel.
- The saved r (one number per position) is always fp32 (`Normalizations.cpp:873-877`).
- It uses only that one position's numbers, so the 300 ` the` positions still have identical values after it.

**Op 4. Convert `h1` to bf16** (`:582`). Case 1: a real fp32 → bf16 conversion, because the Q/K/V weights are bf16.
Case 2: nothing.

**Op 5. The Q, K, V matmuls** (`:594-596`). `nn::Linear` stores its weight as [in, out] and computes `y = x × W`.
- q = h × Wq, Wq [2304 × 2304] → [8, 4096, 2304] (18 heads × 128).
- k = h × Wk and v = h × Wv, each [2304 × 768] → [8, 4096, 768] (6 heads × 128).
- Still one position at a time, so the 300 ` the` positions still have identical q, k and v.

**Op 6. QK-norm + RoPE, then attention.**
- QK-norm weights `q_gamma` / `k_gamma` are always fp32 (`ModelConfig::qk_dtype()`), so q and k are widened to fp32
  before the fused QK-norm + RoPE kernel and narrowed to bf16 after it, in both cases (`:612-620`).
- RoPE rotates q and k by an angle that depends on the **position number**. This is the first operation where the
  300 ` the` positions stop being identical.
- Attention (causal): out[t] = Σ over s ≤ t of a[t,s] × v[s], with a[t,s] = softmax over s of (q[t]·k[s] / √128).
  Each position mixes in the positions before it, so every position ends up different.

**Op 7. Output matmul and residual add** (`:647-648`): proj = merged × Wo, then x1 = x + proj. Then layer 1's MLP,
then layers 2 to 26.

**Op 8. Final norm** (`norm_f`). Out: the hidden states `h` [8, 4096, 2304].

**Op 9. The lm_head** (`:786-788`), which uses the table a second time:
```cpp
Tensor w_bf   = autograd::to_dtype(wte.weight, Dtype::Bfloat16);        // [vocab, d_model]
Tensor w_t_bf = autograd::contiguous(autograd::transpose(w_bf, 0, 1));  // [d_model, vocab]
Tensor logits = autograd::matmul(x_bf, w_t_bf);
```
- `to_dtype(wte.weight, bf16)`: case 1 makes a new bf16 copy of the whole table; case 2 does nothing.
- `transpose` is a **view**: it only swaps 2 sizes and 2 strides and shares the same memory (`ViewOps.cpp:113-138`).
- `contiguous` then makes a **real copy** in the transposed shape (`Tensor.cpp:290-305`).
- The matmul: h [32,768 × 2,304] × [2,304 × 50,304] = logits [32,768 × 50,304], in bf16.

**Op 10. The loss**, from the logits.

**Summary of the forward:** the table is used at the start (a copy per position) and at the end (a matmul). The same
token gives the same vector until RoPE and attention; from there on, every position differs.

**A redundancy found here (future task).** `wte` changes only at the optimizer update, but op 9's bf16 copy (case 1)
and the transposed copy (both cases) are rebuilt in every micro-step: 31 of 32 rebuilds per step (63 of 64 in the
B = 4 harness) are identical to the previous one. They could be built once after each update. That needs a custom
backward node so the gradient still reaches `wte` (see `TiedLMHeadBackward` in the `TODO` at `Engine.cpp:330-339`, and
the comment at `:775-777`: the GEMM has no trans_b entry). Estimated, not measured: about 1.2 GB of memory traffic per
micro-step. Also, the comment at `:769-771` says `wte` is 294.8 MiB; for d_model 2304 it is 442 MiB.

---

## 4. The backward, one operation at a time

The backward runs the forward in reverse. **No backward step changes any weight.** The backward only computes
gradients. The only thing that changes the table is the optimizer update, once per optimizer step.

**Op 10 back.** The loss gives the gradient for the logits, [32,768 × 50,304].

**Op 9 back. The lm_head matmul's backward gives 2 gradients.** A matmul has 2 inputs, so its backward must say how the
loss changes with each of them. Each comes from the one incoming gradient multiplied by the **other** input, so the
backward of one matmul is 2 more matmuls:
1. Gradient for `h` = logits gradient [32,768 × 50,304] × table [50,304 × 2,304] = [32,768 × 2,304]. It goes back into
   the 26 layers.
2. Gradient for the transposed table = hᵀ [2,304 × 32,768] × logits gradient = [2,304 × 50,304]. The transposed copy's
   backward passes it through, and the transpose's backward turns it back into [50,304 × 2,304]. This is the
   **lm_head part** of the table's gradient. Inside the matmul each number is a dot product added up in fp32 by the
   tensor cores, then written **once as bf16**.
- The lm_head part gives **every one of the 50,304 rows** a gradient: at every position every token has some
  probability, and dlogits = softmax − (1 at the target), and softmax is never 0.
- Case 1 only: the conversion's backward (`ToDtypeBackward`, `ReshapeBackward.h:39-43`,
  `return { grads[0].as_type(src_dtype_) };`) widens the lm_head part to fp32. That widening is exact and adds nothing.
  Case 2 has no conversion node; the lm_head part stays bf16.

**Ops 8 to 3 back.** The final norm's backward, then layers 26 down to 1.
- Each residual add passes the **same** gradient to both of its inputs (`AddBackward::apply`,
  `src/autograd/backward/BinaryBackward.cpp:57-68`: `grad_a = grad_output, grad_b = grad_output`). So `x`'s gradient
  contains the gradient from the very top of the model, passed down unchanged 52 times ("path A").
- The RMSNorm backward computes, per position, u = γ × dh, then dx = r × (u − x̂ × mean(x̂ × u)) with x̂ = x × r
  ("path B"). The kernels (`rms_bwd_fused_*_sm90`) compute in fp32 and write the input's gradient in the input's dtype.
  γ's own gradient is **always** made in fp32 (`rms_norm_backward`: `wg_opts` Float32), and it goes into γ's fp32
  gradient buffer with a plain fp32 add. So the norm weights' gradients are never rounded to bf16.
- Attention's backward: dv[s] = Σ over t ≥ s of a[t,s] × d_out[t]. Position s gets gradient from every **later**
  position that looked at it. So each ` the` position gets a different gradient.
- The engine adds path A and path B for `x`. The result: the gradient at the lookup's output, [32,768 × 2,304], one
  different vector per position.

**Op 1 back. The lookup backward** (`EmbeddingBackward`).
- Inputs: the gradient at the lookup output [8, 4096, 2304] **and** the same token ids [8, 4096], saved from the
  forward. The ids are an **input**, used to know which row each vector belongs to.
- Output: **one** tensor, [50,304 × 2,304], the **lookup part** of the table's gradient. The ids get no gradient
  (they are whole numbers). Proof: `EmbeddingOps.cpp:116-118` connects the backward only to the weight
  (`set_next_edge(0, get_grad_edge(weight))`), and `EmbeddingBackward::apply` returns only `{grad_weight}`.
- Row 262 = the sum of the 300 vectors from the positions where ` the` was. A token seen once = its 1 vector. A token
  not in the micro-batch = zeros (the table starts as `Tensor::zeros`, `EmbeddingBackward.cpp:36`).
- **Why a sum:** in the forward, row 262 was copied into 300 places, and each copy affected the loss. Changing row 262
  changes all 300 copies at once, so row 262's effect on the loss is the sum of its effects through the 300 places.
  That is the chain rule for something used several times.

**Then the engine adds the 2 parts.** `wte.weight` got 2 gradients: the lm_head part and the lookup part. The engine
adds them (`Engine.cpp:341-344`: `sum += grads[i]`). That sum is added into `wte`'s fp32 gradient buffer, once per
micro-step. After the last micro-step, the optimizer updates the table once.

**Where the "learning" for each position happens.** Each of the 300 vectors is computed by the 26 layers' backward,
before the lookup backward, from that position's own context. The lookup backward computes none of them; it only
adds them up per token.

---

## 5. Which rows of the table change at the optimizer step

Every row changes, not only the rows of tokens in the batch. 3 reasons:
1. **The tied lm_head part** gives every row a gradient in every micro-step (section 4).
2. **Weight decay.** `wte` is 2-D, so it gets `weight_decay = 0.1` (`bluscript_zero_ckpt.cpp:204`); only 1-D weights are
   excluded (`:1190-1191`, `exclude_1d_from_weight_decay = true`). Every step, every number is multiplied by
   (1 − lr × wd) (`AdamKernels.cu:79`, `:116`: `p *= (1.0f - lr * weight_decay)`).
3. **Adam momentum.** m and v remember earlier steps, so a row whose gradient is 0 now still moves by lr × m̂ / (√v̂ + ε).

In an untied model reason 1 does not apply, but 2 and 3 still do.

---

## 6. The embedding-backward kernels

### 6.1 `dist_side_merge` (and every BLUCCL trainer except the fused one)

`EmbeddingBackward.cpp:33-36` makes the table's gradient in the **same dtype as the incoming gradient**. Then
`EmbeddingKernels.cu:411-445` picks the kernel:
- **fp32:** `embedding_backward_kernel_cooperative` (`:152-294`), when the gradient table is contiguous (it always is).
- **everything else, bf16 included:** `embedding_backward_kernel_optimized` (`:127-149`), which uses `atomicAdd`.

**The `atomicAdd` kernel (bf16).** One GPU thread per (position, number): 32,768 × 2,304 = 75,497,472 threads. All
threads of the 300 ` the` positions add into the same spot of row 262 at the same time
(`atomicAdd(&grad_weight[tok * stride_row + c * stride_col], grad)`, `:148`).
- Why `atomicAdd` is needed there: without it, thread A reads 0.5, thread B reads 0.5, A writes 0.5 + a, B writes
  0.5 + b, and A's value is lost. `atomicAdd` makes "read, add, write" one step that no other thread can interrupt.
- The spot holds a bf16 number, so **after every add the running total is rounded to bf16**: 300 roundings for row
  262. The order of the adds depends on thread scheduling and changes from run to run.
- Example, 4 values 1.0, 0.003, 0.003, 0.003 (exact sum 1.009): in the order 1.0 first, every +0.003 is below half the
  gap and is lost, result 1.0. In the order 0.003, 0.003, 0.003, 1.0: 0.009 + 1.0 = 1.009 → 1.0078125. Two different
  answers from the same values.

**The cooperative kernel (fp32).** It was written to remove `atomicAdd`. Its comment at `:385-388` says it was written
first, never used, and switched on later (for a determinism test), and only for fp32.
- The grid is 18 blocks (2,304 / 128). Each block owns 128 of the 2,304 numbers of every row; no other block touches
  them.
- Each block walks through all 32,768 positions, 32 at a time (1,024 groups), always in the same order.
- In each group: each position's value is copied into shared memory (fp32); `__ballot_sync` finds the positions with
  the same token; the first of them (the "leader") adds the others' values into its own and does one plain add into
  the table row. The next group starts only after `__syncthreads`.
- Only one thread writes a given spot at any moment, so no `atomicAdd` is needed, and the result is the same every run.

**Why bf16 does not use the cooperative kernel.** Only because it was written for fp32 only: its pointers and shared
memory are `float` (`:189-202`), and the dispatch sends only `float` there (`if constexpr (std::is_same_v<T, float>)`,
`:415`). Nothing in its method needs fp32. But making it accept bf16 alone is not enough: the running total lives in
the gradient table itself, which is bf16 in case 2, so the leader's add would still round to bf16 once per group.
Almost every ` the` lands in its own group of 32, so that is still about 260–300 roundings. Two separate things fix two
separate problems: the kernel fixes the random order; an fp32 total fixes the rounding.

### 6.2 `dist_side_merge_BLUCCL`: the fused kernel `EmbeddingBwdFused` (no `atomicAdd`)

File: `dist/communication/BluCCL/kernels/fusions/embedding_fused.cu` (1,375 lines, kiruthikkanna-m). It connects through
a hook that the BLUCCL branch added to TI: `EmbeddingBackward.cpp:122-139` calls the installed hook **before** the
normal kernels; if the hook returns true, the function returns and the old kernels never run.

**When it runs** (both must hold):
1. The trainer is `bluscript_zero_ckpt_bluccl_fused.cpp`. It is the only one that calls
   `install_fused_embedding_backward(B*T, d_model, vocab, -1, is_master)` (`:1154`, e6eaeba). `bluscript_zero_ckpt.cpp`,
   `bluscript_zero_ckpt_bluccl.cpp` and `bluscript_zero_ckpt_elastic.cpp` never call it.
2. `BLUCCL_FUSED_EMB=1` (`embedding_fused.cu:1337`; default 0). Until commit `e573458` (09-18) the Makefile did not pass
   `BLUCCL_*` variables to the MPI ranks; the Makefile's own comment (`:358`) says the fused backward "silently declined
   during a measurement campaign". Now the regex is `^(BLU|NCCL_)` (`Makefile:365`).
It also checks `supported()` (`:1104-1114`): V ≤ 65,535 and d_model a multiple of 4. Rank 0 prints
`[BluCCL] fused embedding backward ON (N=… C=… V=…, workspace … MB)` when it is installed (`:1359`).

**What it does each micro-step.**
- The sort part uses NVIDIA's **CUB** library (`#include <cub/cub.cuh>`, `:172`; headers only, it comes with the CUDA
  toolkit): `cub::DeviceRadixSort::SortPairs` sorts the positions by token id; `cub::DeviceRunLengthEncode::Encode`
  finds each run of equal ids (a "segment", e.g. ` the` → row 262, 300 positions); `cub::DeviceScan::ExclusiveSum`
  finds where each segment starts. Then `embedding_bwd_count_split_chunks` cuts segments longer than 32 positions into
  chunks of 32 (` the`: 300 → 10 chunks), `embedding_bwd_expand_split_chunks` records which chunk belongs to which
  segment, and `embedding_bwd_make_row_byte_offsets` works out where each position's gradient sits in memory.
- The sum part is their own code: `embedding_bwd_bounded_direct` sums a row used by ≤ 32 positions whole and writes it
  once; `embedding_bwd_bounded_chunks` sums long rows chunk by chunk into an fp32 scratch buffer (`float* partials`);
  `embedding_bwd_bounded_fold` adds a long row's partial sums in a fixed order (chunk 0, 1, 2, …) and writes it once.
- Two more kernels in the file (`embedding_bwd_rs_rowsliced`, `_featuresliced`, backward + reduce-scatter fused) are
  **not** used by the hook path, which calls `run()` (the 3 kernels above).

**Verified from the code: every add is fp32, for both fp32 and bf16 input.**
- Read: fp32 is read as fp32 (`io4<float>::load`, `:205-207`); bf16 is widened to fp32 on read (`__bfloat1622float2`,
  `:237-244`, exact).
- Add: every running total is a `float4 acc = 0`; `v4_acc` is 4 plain fp32 adds (`:180-182`); direct `:414-418`,
  chunks into fp32 `partials` `:444-448`, fold `:470-476`.
- Write: once, in Store mode (the hook passes `accumulate=false`, `:1305-1318`; `run_impl` → `WriteMode::Store`,
  `:1223`). fp32 output is written as is; bf16 output is rounded **once** with `__floats2bfloat162_rn` (`:245-251`).
- So repetition causes no extra rounding in this kernel. Example: 1.0 + 0.003 + 0.003 + 0.003 = 1.0090001 in fp32, one
  rounding → 1.0078125, the same every run. With an fp32 table the same kernel writes fp32: 1.0090001 is kept.
- The file's header gives these numbers (H200, fp32, against fp64; theirs, **not measured by us**): atomicAdd error
  1.83e-03 and not the same every run; cooperative 1.10e-03; this kernel 2.44e-04, same every run.

**One risk for SFT:** the hook ignores the token count it is given (`embedding_fused.cu:1300`, `(void)N`). It always
uses the count fixed at install (B × T). If a micro-batch had a different number of tokens, it would read the wrong
number of ids, with no error.

**CUB elsewhere:** Tensor-Implementations (`dist_side_merge`) does not use CUB anywhere. BluTrain `dist_side_merge`
uses NVIDIA Thrust (which runs on top of CUB) in one file, `dist/Tensor-Parallelism/dnn/EntropyKernels.cu`.

### 6.3 The forward kernels (both branches; the BLUCCL change is backward only)

`Tensor-Implementations/src/Kernels/cuda/embedding/EmbeddingKernels.cu`:
- fp32 table: `embedding_forward_kernel_vectorized` (runs when the row length is a multiple of 4 and the table is
  contiguous and 16-byte aligned, which holds: 2,304 / 4 = 576). It reads 4 fp32 numbers at once; each thread copies 16.
- bf16 table: `embedding_forward_kernel_optimized<__nv_bfloat16>`, one number per thread.
- Both only **copy** rows. No adds, no `atomicAdd`, no rounding, no CUB.

### 6.4 Which trainer runs which kernel (from the committed scripts)

- `run_arm.sh` / `run_nsys.sh` (BLUCCL, 09-16): `bluscript_zero_ckpt_bluccl_fused.cpp` with `BLU_ALL_BF16=1`,
  `BLUCCL_FUSED_EMB=1`, `BLU_OPTIMIZER=muon`, 8 GPUs → the fused kernel. `run_nsys.sh` also sets `BLU_GRAD_VIEW=1`.
- `run_bluscript.sh` (Navin, a3aba3e, his production launcher, `:1256-1264`): `bluscript_zero_ckpt.cpp` with
  `BLU_ALL_BF16=1 BLU_BF16_LOGITS=1 BLU_GRAD_VIEW=1 BLU_ZERO_V2=1`, Muon, 8 GPUs → that trainer never installs the hook,
  so **his run uses the old bf16 `atomicAdd` kernel**, and with `BLU_GRAD_VIEW=1` it also adds the micro-steps up in bf16.
- Our SFT branch `dist_side_merge` has no fused kernel: a bf16 table uses the `atomicAdd` kernel.

---

## 7. Where `wte`'s gradient is rounded to bf16: D1, D2, D3

The 3 places where the two cases differ are called D1, D2 and D3.
- **D1**: the lookup backward's sum (section 6).
- **D2**: the engine's add of the lookup part and the lm_head part.
- **D3**: the path from the micro-steps to the optimizer (the fp32 gradient buffer, the send, the switch).

How much each one loses was measured on 2026-10-01: section 17.

### 7.0 The explanation exactly as given in the chat (2026-09-30, 14:22 UTC)

You said about this answer: "actually I understood from this only clearly … remind this for documentation". It is kept
here word for word. Sections 7.1–7.3 below explain the same points in more depth.

> Sorry, I mixed the two cases in one paragraph. Here are the facts, one case at a time, from the code.
>
> **The fact that removes most of the confusion**
>
> **The micro-step gradient buffer is fp32 in both cases, even when the table is bf16.** For a bf16 weight, autograd
> keeps its gradient buffer in fp32 on purpose (`AutogradMeta.cpp:161-163`: "params store it upcast to fp32 so
> subsequent micro-steps accumulate in fp32").
>
> **What is sent follows the table's dtype, not the buffer's.** The default is SAME_AS_PARAM (`ZeroConfig.h:92-99`):
> - fp32 table: the fp32 buffer is sent **as fp32**. No conversion.
> - bf16 table: the fp32 buffer is **converted to bf16** first, then sent (`SymmetricShardedOptimizer.cpp:829-833`).
>   **That conversion is the rounding.**
>
> So "no need to round because it is bf16" is not right: the buffer is not bf16, it is fp32, so it has to be converted.
>
> The one exception is `BLU_GRAD_VIEW=1` with more than 1 GPU. Then the buffer really is bf16: every micro-step add
> rounds, and nothing is converted at the send.
>
> **ZeRO-1 over 8 GPUs, bf16 table: your chain, checked**
> 1. Send in bf16: **yes**.
> 2. The switch adds the 8 GPUs' values with an fp32 accumulator: **yes** (`.acc::f32`, `all_reduce_nvls.cuh:194`).
> 3. "The fp32 sums go to the optimizer": **no**. The switch instruction returns **bf16** (`bf16x2`). The fp32 sum is
>    rounded to bf16 as it comes back, and stored in a bf16 buffer (`shard_grad_`, `:578`).
> 4. Then it is widened to fp32 for AdamW (`:965-975`). This widening is exact.
>
> So the optimizer gets fp32 numbers, but their **values are the bf16-rounded sum**. That's for their BluCCL branch.
> On our `dist_side_merge` the sending library is NCCL, and I have not checked how NCCL adds bf16.
>
> **The full chain per case (their setup: `BLU_ALL_BF16=1`, 8 GPUs)**
>
> **fp32 table** (`BLU_BF16_EMBED=0`):
> - **D1:** the backward of Op 2 widens the gradient to fp32 (`ReshapeBackward.h:39-43`). So the table's gradient is
>   fp32 (`EmbeddingBackward.cpp:33-36`), and the new kernel adds in fp32 and **writes fp32**. No rounding.
> - **D2:** the lm_head part is widened to fp32, then fp32 + fp32 are added. No rounding.
> - **Micro-step buffer:** fp32 += fp32. No rounding.
> - **Send:** fp32. The switch adds in fp32 and returns fp32. No rounding.
> - **Total: 0 roundings to bf16.** (The lm_head part is born bf16 from its matmul, but that is the same in both cases.)
>
> **bf16 table** (`BLU_ALL_BF16=1`):
> - **D1:** adds in fp32, **writes bf16**. 1 rounding, every micro-step.
> - **D2:** bf16 + bf16, stored bf16. 1 rounding, every micro-step.
> - **Micro-step buffer:** fp32. No rounding, unless `BLU_GRAD_VIEW=1`.
> - **Send:** fp32 → bf16. 1 rounding, once per step.
> - **Switch:** fp32 add, returned as bf16. 1 rounding, once per step.
> - **At 1 GPU:** the same, minus the switch. The send conversion still happens even though nothing is sent; that is F4.
>
> Your chain started with "fp32 table" but then used the bf16-table steps. With an fp32 table, nothing is ever rounded
> to bf16.
>
> **D2: "same answer, so why change?"** Two different ideas:
> - **Idea A:** add in fp32 but still **store the sum as bf16**. This gives the **same** answer as today. Pointless.
>   You're right.
> - **Idea B:** **never store the sum as bf16**. Add both parts straight into the fp32 buffer. This gives a
>   **different, more exact** answer. Example: today the buffer receives 1.0078125. With idea B it receives
>   1.0078125 + 0.0030060 = 1.0108185.
>
> Whether idea B is **needed** depends on whether that error matters for training. My view: **test first, change nothing
> now.**

(At that time "the new kernel" meant the colleague's `EmbeddingBwdFused`, and the F4 conversion at 1 GPU still existed.
F4 was fixed later the same day, section 12.)

### 7.1 The full chain, one case at a time (their setup: `BLU_ALL_BF16=1`, fused trainer, 8 GPUs)

**fp32 table** (`BLU_ALL_BF16=1 BLU_BF16_EMBED=0`):
- **D1:** the backward of op 2 widens the gradient to fp32 (`ReshapeBackward.h:39-43`), so the table's gradient is
  fp32 (`EmbeddingBackward.cpp:33-36`), and the fused kernel adds in fp32 and **writes fp32**. No rounding.
- **D2:** the lm_head part is widened to fp32, then fp32 + fp32. No rounding.
- **Micro-step buffer:** fp32 += fp32. No rounding.
- **Send:** fp32 (the send dtype follows the weight's dtype). The switch adds in fp32 and returns fp32. No rounding.
- **Total: 0 roundings to bf16.** (The lm_head part is born bf16 from its matmul, but that is the same in both cases.)

**bf16 table** (`BLU_ALL_BF16=1`):
- **D1:** adds in fp32, writes bf16. 1 rounding, every micro-step.
- **D2:** bf16 + bf16, stored as bf16. 1 rounding, every micro-step.
- **Micro-step buffer:** fp32. No rounding, unless `BLU_GRAD_VIEW=1`.
- **Send:** fp32 → bf16. 1 rounding, once per step.
- **Switch:** fp32 add, returned as bf16. 1 rounding, once per step.
- **At 1 GPU:** the same, minus the switch. The send conversion used to happen even though nothing is sent: that is
  F4 (section 12), fixed today on BLUCCL.

With an fp32 table, nothing in `wte`'s gradient path is ever rounded to bf16. None of the bf16-table roundings depends
on how often a token repeats: the fused kernel already removed the repetition problem. (On `dist_side_merge`, D1 is the
`atomicAdd` kernel instead, and there it does depend on repetition.)

### 7.2 D2 in detail

With weight tying, `wte` gets 2 gradients. The lookup part is non-zero only for rows of tokens in the batch (all other
rows are exact zeros). The lm_head part is non-zero for every row. With weight tying off, there is no D2: the
embedding gradient is only the lookup part, and the separate lm_head weight has its own gradient.

D2 is **one** add per number per micro-step, so its error is at most one rounding: 0.39% of the result. In the example
above, the lm_head's 0.003 is lost for that number in that micro-step, which is 0.3% of 1.01. It does not grow with
repetition.

Two ideas to remove it:
- **Idea A:** add in fp32 but still store the sum as bf16. This gives exactly the same answer as today (the 1,000,000-
  pair test). Pointless.
- **Idea B:** never store the sum as bf16; add both parts straight into the fp32 buffer. This gives a different, more
  exact answer: today the buffer receives 1.0078125; with idea B it receives 1.0078125 + 0.0030060 = 1.0108185.
  - Option 1: the engine calls the existing F2 kernel once per part. My estimate for our model (one bf16 table 232 MB,
    one fp32 table 464 MB): today 696 MB (bf16 add) + 1,160 MB (F2 add) = 1,856 MB per micro-step; option 1 =
    2 × 1,160 = 2,320 MB, so +464 MB, about 0.14 ms on H200.
  - Option 2: one kernel that adds both parts into fp32 in one pass: 232 + 232 + 464 read + 464 write = 1,392 MB, so
    −464 MB compared with today. It removes the rounding and saves time. It needs the engine change plus a small new
    kernel (F2's kernel with 2 inputs).
- Decision: **measure first, change nothing now.** Whether 0.39% per micro-step matters is what the ablation measures.
- D1's final rounding can also go: the fused kernel adds in fp32 and writes bf16 only because the output table has the
  gradient's dtype. Writing fp32 needs a bf16-in / fp32-out version (+232 MB of writes), or the kernel's existing
  `WriteMode::Add` could add its fp32 sums straight into the fp32 buffer.

### 7.3 D3 in detail

**The micro-step gradient buffer is fp32 in both cases**, even for a bf16 weight (`AutogradMeta.cpp:161-163`: bf16
params "store it upcast to fp32 so subsequent micro-steps accumulate in fp32"; the comment at `:118-120` says summing
micro-step gradients in bf16 "loses small increments to swamping"). This buffer is what keeps the rounding at once
per step instead of once per micro-step.
- Exception: `BLU_GRAD_VIEW=1` with more than 1 GPU makes each weight's gradient a bf16 view into the send buffer
  (`SymmetricShardedOptimizer.cpp:760-790`; the trainer's warning at `:1356-1392`: "moves micro-step gradient
  accumulation from fp32 to bf16"). Then every micro-step add rounds.

**What is sent follows the weight's dtype, not the buffer's.** The default is `SAME_AS_PARAM` (`ZeroConfig.h`,
`resolve_reduce_dtype`). fp32 weight → sent as fp32, no conversion. bf16 weight → the fp32 buffer is converted to bf16
first (`SymmetricShardedOptimizer.cpp:829-833`, `stage_grad_into_flat`), and that conversion is the rounding.

**The 8-GPU sum on BLUCCL** (checked in the code):
1. `SymmetricShardedOptimizer::reduce_scatter_grads` (`:945-963`) calls `pg_->reduce_scatter` in bf16. `pg_` is BluCCL
   at more than 1 GPU (fused trainer `:1117-1118`).
2. `processGroupBluCCL.cu:122` calls `::comm::reduce_scatter<T>` without an accumulator argument, so the default
   `ACC::F32` is used (`collectives.h:383`).
3. The method: the first allowed rule wins (`init/algo_select.h:105-108`). `RS_INTERNODE` runs only with `BLU_IB_RS=1`.
   Next is `RS_NVLS`, "multicast default": on 8 H200s with multicast, this one. `BLUCCL_RS_P2P` defaults to 0
   (`collectives.h:66-67`).
4. `mm_reduce_window` (`reduce_nvls.cuh:49`) uses `multimem.ld_reduce … add.acc::f32 … bf16x2`
   (`all_reduce_nvls.cuh:194`): the NVSwitch adds the 8 GPUs' bf16 values with an **fp32 accumulator**. Their comment
   (`init/types.h:24-31`) says without `.acc::f32` the switch would add in bf16.
5. **But the instruction returns bf16** (`bf16x2`). The fp32 sum is rounded to bf16 once and stored in the bf16 buffer
   `shard_grad_` (`:578`). Then `cast_shard_grad_to_fp32` (`:965-975`) widens it (exact) for AdamW.
So the switch adds in fp32, but the result comes back as bf16. Not checked: the multi-node path (`BLU_IB_RS`), and how
NCCL adds bf16 on `dist_side_merge`.

---

## 8. The send-dtype flags `BLU_BF16_REDUCE` and `BLU_FP32_REDUCE`

(`bluscript_zero_ckpt_bluccl_fused.cpp:1444-1497`; our trainer has the same flags at `:1275-1326`.)
**The answer exactly as given in the chat (2026-09-30, 14:49 UTC)**, after you asked to remember these cases and
their consequences for the documentation. Kept word for word:

> On "everything collapses": **part of the benefit is lost, not all of it.** The per-case chain for 8 GPUs:
> - **fp32 table + `BLU_BF16_REDUCE=1`:** D1, D2 and the micro-step buffer stay fp32, with no rounding. The send rounds
>   once per step, and the switch returns bf16, one more rounding per step. **Total: 2 roundings per step.**
> - **bf16 table (default):** D1 and D2 each round once **every micro-step**, plus the same 2 roundings per step.
>
> So with that flag, the fp32 table loses its per-step advantage but keeps the per-micro-step one (D1 + D2). Whether that
> remaining difference matters is what the test will show. You're right that with this flag, the bf16 table gets much
> closer to the fp32 one.

(That same answer also recommended that, at 1 GPU, the F4 shortcut should keep today's path when one of these flags is
set. You then decided that the flags do not matter at 1 GPU, because nothing is sent there. So the final fix ignores
them at 1 GPU and prints one line saying so; see section 12.2.)

The facts behind it, checked in the code:

- Default (neither set): `SAME_AS_PARAM`, send dtype = weight dtype.
- `BLU_BF16_REDUCE=1`: `zcfg.reduce_dtype = BF16`, and Muon copies it (`mcfg.reduce_dtype = zcfg.reduce_dtype`, `:1640`).
  Every group sends bf16. For bf16 groups nothing changes; for fp32 groups the fp32 buffer is now converted to bf16.
- `BLU_FP32_REDUCE=1`: every group sends fp32, bf16 groups included.
- Both at once: the trainer stops the job (`:1480-1485`).
- The trainer's own comment on `BLU_BF16_REDUCE`: "for bisection, not for production"; it does not save memory
  (+147.6 MiB per GPU, a separate bf16 send buffer); it saves almost no time (~0.3 ms per step); it "costs precision
  exactly where the model chose not to spend it", i.e. `wte`.
- **AdamW v1 refuses both** whenever the send dtype would differ from the weight's dtype: it throws `ZeroConfigError`
  (`ShardedOptimizer.cpp:186-193`). The fused trainer always uses v2, so there the flags work.
- Their committed run scripts set neither flag; the colleague uses it in experiments.

**Chains with the flag (8 GPUs):**
- fp32 table, default: 0 roundings to bf16 (7.1).
- fp32 table + `BLU_BF16_REDUCE=1`: D1, D2 and the micro-step buffer stay fp32; the send rounds once per step; the switch
  returns bf16, one more per step. **Total: 2 roundings per step.**
- bf16 table, default: 2 per micro-step (D1, D2) plus 2 per step.

**The consequence (your point):** if everything is added up carefully in fp32 and then sent in bf16, part of the benefit
is thrown away. Not all of it: with the flag, the fp32 table still avoids the per-micro-step roundings D1 and D2; only
its per-step advantage is lost. So with this flag, a bf16 table is much closer to an fp32 one. The test decides whether
the rest matters.

**At 1 GPU, after F4:** nothing is sent, so the fix reads the fp32 gradient directly and ignores the send dtype. The
flags have no effect at 1 GPU, and v2 and Muon print one line saying so (section 12).

---

## 9. The ZeRO-1 optimizers: v1, v2 and Muon

**What ZeRO-1 is** (the full explanation from zero is in `ddp_vs_zero_1.md`, same folder): the optimizer state (fp32 masters, m, v) is split across GPUs; each GPU
updates only its piece, and the new weights are shared back. At 1 GPU the "piece" is everything.

**The three classes:**
- **v1** `ShardedOptimizer`: AdamW, reduce to the owner + broadcast. The default in `bluscript_zero_ckpt.cpp`.
- **v2** `SymmetricShardedOptimizer`: AdamW, reduce-scatter + all-gather. Switched on with `BLU_ZERO_V2=1`; the fused
  trainer forces it (`use_zero_v2 = true`, `:1059`). v2 is not "for Muon"; it is a second AdamW implementation.
- **Muon** `ShardedMuon`: Muon for the 2-D matrices of the 26 (or 36) layers (Q, K, V, O, MLP), switched on with
  `BLU_OPTIMIZER=muon`. It has an internal AdamW path, used only for non-Muon weights handed to it; in the trainers
  it is handed only Muon matrices.

**Muon does not update `wte`.** `wte` (and `lm_head`) are in `muon_excluded` (`bluscript_zero_ckpt.cpp:1147-1152`;
fused trainer `:1266-1271`). The fused trainer's comment (`:1256-1258`): "Muon for the transformer body's 2-D weight
matrices, AdamW for everything else … the token embedding and the output head". Non-Muon weights go to
`params_by_dtype` (`:1290-1292`) and then to v2 AdamW (`:1677-1681`). So in a Muon run, `wte` and the norm weights are
updated by AdamW. Under `BLU_ALL_BF16=1` the AdamW side has 2 dtype groups: bf16 (`wte` + the block norm scales) and
fp32 (`q_gamma` / `k_gamma`, which the fused QK-norm kernel pins to fp32).

**v1's buffers and its 6-step optimizer step** (one weight followed through, `wQ` of layer 0, 5,308,416 numbers):
- Before the step: the weight itself (bf16), its fp32 gradient buffer (the micro-steps added up), and inside the
  optimizer its fp32 master, m and v.
- v1 lines all weights of the group side by side in one long buffer, `flat_stage_`, in the **weights' dtype** (bf16),
  so it can send one buffer instead of hundreds. That is the first place bf16 comes from.
1. `gather_grads_into_flat`: each fp32 gradient → bf16 → its slice of `flat_stage_`.
2. `reduce_scatter_grads`: summed across GPUs into `shard_grad_` (bf16). At 1 GPU it was only a copy.
3. `cast_shard_grad_to_fp32`: bf16 → fp32 into `shard_grad_fp32_`, because Adam must calculate in fp32.
4. `local_adam_step`: Adam updates the fp32 master using `shard_grad_fp32_`, m, v.
5. `all_gather_params`: the fp32 master → bf16 → `shard_master_compute_` → every GPU's `flat_stage_`. The new weights
   must be bf16 because the model uses bf16 weights: the second place bf16 comes from.
6. `scatter_flat_into_params`: `flat_stage_` → each weight's own bf16 storage.

These steps run once per optimizer step, after the last backward, when the activations are already freed.

---

## 10. The #2 fix (temporary buffers; v1 only; commit `a264586`, 2026-09-25)

**Before**, each dtype conversion in steps 1, 3 and 5 was 2 operations:
```cpp
Tensor f = shard_grad_.as_type(Dtype::Float32);                       // a new full-size temporary
d2d(shard_grad_fp32_.data(), f.data(), S_ * sizeof(float), device_);  // then a copy into the real buffer
```
`as_type` allocates a brand-new tensor and writes the converted numbers into it; `d2d` copies them into the destination,
which already existed; the temporary is thrown away. The numbers are written twice.

**After**, one kernel writes the converted numbers straight into the destination:
`OwnTensor::convert_type_cuda_generic(src, src_dtype, dst, dst_dtype, n, stream);` The numbers are identical (0 bit
differences, 09-25 test).

**4 places** (step 1 exists twice): B′ `on_grad_ready` (the overlap path, `ShardedOptimizer.cpp:540`), B
`gather_grads_into_flat` (the 1-GPU / reference path, `:958`), C `cast_shard_grad_to_fp32` (`:998`), D the master →
bf16 conversion (`:1056`).
- Step 3 (site C) at 1 GPU: a 5,888,802,816-byte temporary → 0; 14 bytes of memory traffic per number → 6.
- Step 5 (site D): a 2,944,401,408-byte temporary → 0; 10 → 6 bytes per number.
- Step 1 (B / B′): one temporary per weight (10.6 MB for `wQ`, 2.94 GB in total, one after another) → none.
- Peak memory at 1 GPU did not change (the step runs after the activations are freed). At 2 GPUs: reserved memory
  −2,810 MiB per GPU, +0.13% tok/s (09-25).
- v2 and Muon's default path already wrote directly and never had #2. The 3 `as_type` calls left today: v1 `:357` and
  v2 `:616` are in `init_master_from_params` (run once at start), and Muon `launch_reduce` (`:816`), which runs only
  with `BLU_MUON_RB=1` (future task).

---

## 11. F1 (NaN/Inf gradients; fixed in all 3 optimizers, both branches)

**What NaN and Inf are.** Inf ("infinity") appears when a number is too big for its dtype. NaN ("not a number") appears
from things like Inf × 0 or Inf − Inf. Once a NaN is in a weight, every calculation that uses it gives NaN.

**The old bug.** Before the update the optimizer adds up the squares of all gradients (`nsq`, for clipping). One NaN
or Inf anywhere makes that sum NaN or Inf. The check that skips such a step was `if (sc && !isfinite(nsq))`: it ran only
with a LossScaler, and no trainer installs one. Clipping did not stop it (a NaN norm gave a clip factor of 1.0; an Inf
norm gave 0, and Inf × 0 = NaN). The trainer checks only the loss, before `step()`. So NaN went into the fp32 masters and
into m and v, and a checkpoint on that step would have saved it. Worst case, Muon's 2-D matrices: Muon writes the NaN
into its momentum first and only then rejects the update, so the weights stay finite but m keeps the NaN and every later
update of that matrix is rejected. The matrix stopped learning with no error (measured 09-26).

**The fix.** If `nsq` is not finite and there is no LossScaler, the optimizer throws `ZeroNonFiniteGradError` **before
touching the weights, m or v** (`ShardedOptimizer.cpp:1216-1217`; the same in v2 and Muon). The trainer catches it
(`bluscript_zero_ckpt.cpp:2409-2417`), prints `[FATAL] rank … step …: … stopped before the weight update; no checkpoint
written` and stops the job with exit code 4. With a LossScaler it skips the step as before. A normal step is
bit-identical. A NaN norm is now logged as `nan`, not `0.0000`.
- **How one number checks all of them:** NaN + anything = NaN, and Inf² = Inf, so one bad value anywhere makes the
  sum bad. The sum was already computed for clipping, so the check costs one comparison per step.
- **Where:** v1 `a264586` (09-25); v2 and Muon `23ae1e5` (09-26); on BLUCCL through our 09-29 port (`e6eaeba`). Today the
  error class appears in v1 4 times, v2 3 times, Muon 4 times, on both branches.
- **Tests:** 36/36 at 1 GPU (also re-run today on BLUCCL: 36/36 before and after F4). v1 also at 2 GPUs; v2 and Muon not
  at more than 1 GPU.

---

## 12. F4 (the 1-GPU bf16 round trip): found, fixed, tested, pushed

### 12.1 What it was

At 1 GPU nothing is sent, but all 3 optimizers still took the gradient through the send buffer: fp32 → **bf16** →
fp32, before the norm and the update read it. So every bf16 weight's gradient was rounded to bf16 once per step for no
reason. The root cause: the optimizers are built around a send buffer, and at 1 GPU the gradient still went through it.

On BLUCCL before our fix (a3aba3e):
- **v2:** fp32 → bf16 into `flat_grad_` (`SymmetricShardedOptimizer.cpp:857`, rounds); bf16 → fp32 (`:995`).
- **Muon:** fp32 → bf16 into `flat_grad_` (`ShardedMuon.cpp:1175`, rounds); the norm computed from the bf16 values;
  bf16 → fp32 (`:1322`).
- **v1:** fp32 → bf16 (`:950`, rounds), a copy (`:982`), bf16 → fp32 (`:994`).

Navin's commit `a3aba3e` ("in-place collectives") had already removed the plain copy in the middle for v2 and Muon at
1 GPU, and stopped allocating `shard_grad_` / `shard_param_` there. It did not remove the rounding. Before and after his
commit, the bug was steps 1 and 3 together.

### 12.2 The fix ("option B"), commit `149249d`

Every change sits behind `W_ == 1`, so nothing changes at more than 1 GPU. Each change carries a What/Was/Why comment
in the code.
- **v1 and v2:** a new function `stage_grads_direct_fp32()` copies each weight's fp32 gradient straight into
  `shard_grad_fp32_` (the buffer the norm, the clip and Adam read), zeroes the slices of weights with no gradient and the
  padding, and `reduce_grads_now()` calls it at W = 1 and skips the cast back. At 1 GPU the shard is the whole flat space
  (v2's `build_buckets` makes `shard_offset == begin`; v1's `shard_begin_ = 0`, `S_ = padded_numel_`), so a flat offset is
  also a shard offset.
- **Muon:** a new function `prepare_direct_grads()` keeps a contiguous handle on each owned weight's gradient. The 2
  readers, the norm (`accumulate_owned_grad_norm`) and the update (`owned_grad_fp32`), read from there at W = 1, in the
  gradient's own dtype, with Navin's own kernels (`kernels::sumsq_accumulate`, `kernels::cast_to_f32_scaled`, which
  accept fp32). Both Muon modes (RS/AG and `BLU_MUON_RB`) take this path, so they still agree at W = 1.
- **Flags:** at 1 GPU the fix ignores the send dtype (nothing is sent). If `BLU_BF16_REDUCE` / `BLU_FP32_REDUCE` is set at
  W = 1, v2 and Muon print one line saying it has no effect (only when `verbose`; the trainer sets it on rank 0). v1 still
  refuses those flags in its constructor, as before.
- **Unchanged:** Navin's in-place code, his kernels, the gradient-view code (`BLU_GRAD_VIEW` switches itself off at
  W ≤ 1), the overlap path (needs W > 1), the fused clip-AdamW (needs a BluCCL process group, none at 1 GPU), the
  parameter publish through `flat_stage_`, the F1 check (it now reads the exact fp32 values).
- Checked before choosing: Muon's bucket bookkeeping is reset at the start of every backward (`:2290`) and `rs_work` is
  waited only in `finalize_reduce_scatter`, which W = 1 now skips; nothing else reads the wire gradient buffers
  (`grad_wire_base()` / `flat_owned_grad_` are read only by those 2 readers and by the send).
- Why not "option A" (make each gradient a view into `shard_grad_fp32_`, zero copies): it would change the shared
  gradient-view code and how autograd stores gradients, which other people's code depends on. Option B changes only the
  optimizer files. Cost of B's one fp32 copy: about 0.3 ms per step for the `wte` group (estimate).
- The patch was applied with `patch_f4.py` (15 edits in 6 files, md5-checked against a3aba3e, every anchor matched
  exactly once); the pushed files have the same md5s as the tested ones.

### 12.3 Tests (1 × H200, BLUCCL a3aba3e + TI b975be5)

New test `test_f4_w1_exact_grad.cpp` (30 cases: v1, v2, Muon × bf16, fp32 × 5 checks). The gradients are
g = v × (1 + 2⁻¹⁰), with v a full-rank bf16 pattern, so bf16 rounds every g back to exactly v.
- `norm_exact`: `last_grad_norm()` (computed from the buffer the update reads) against the fp64 norm of g.
- `sees_exact`: two optimizers, one given g and one given bf16(g); after one step their states must differ.
- `autograd2`: 2 real backward passes (1.0 × c, then 2⁻¹⁰ × c) into a bf16 weight; the gradient buffers are float32.
- `nograd`: a weight with no gradient behaves exactly like a zero gradient.
- `flag_bf16`: the send dtype forced to bf16.

Results:
- **Old code: 17 FAIL, 13 PASS.** For bf16 the optimizer's norm was 1.65444779 = the fp64 norm of bf16(g)
  (1.65444767, error 7.6e-8); the true norm of g is 1.65606334; the error was 9.76e-4 = 2⁻¹⁰. Feeding g or bf16(g) gave
  byte-identical optimizer states. The fp32 controls passed (error ~1e-8).
- **New code: 30/30 PASS.** For bf16 the norm is 1.65606332 against 1.65606334 (error 1.3e-8); g and bf16(g) now give
  different states.
- F1 test: 36/36 before and after. Navin's `test_zero` (v1, W = 1): ALL PASSED before and after.
- Navin's `test_muon_rsag` (W = 1): fails at random before **and** after, with the identical signature (section 13).
- Not run: W > 1 (only 1 GPU); the trainer A/B (optional, open).

### 12.4 Pushed

`a3aba3e..3298e74` on BluTrain `dist_side_merge_BLUCCL` (the user ran the commands on the box):
- `149249d` "zero: at W == 1 the gradient no longer goes through the wire buffer (F4)": 6 files, +160 / −5.
- `3298e74` "Point Tensor-Implementations back to b975be5" (see section 13).
- Both authored by Goutham Kumar <goutham.kumar@blubridge.com>, no co-author lines. `git ls-remote` confirmed
  BluTrain `dist_side_merge_BLUCCL` = `3298e74` and TI `dist_side_merge_BLUCCL` = `b975be5` (so the pointer is on the latest).
- `dist_side_merge` was **not** patched (user's decision; maybe later, separately).
- Records: `profiling_reports/f4_2026_09_30/` (patch, diff, tests, runner scripts, logs, commit messages, `box_final/`).

---

## 13. Navin's commit `a3aba3e` (bruce-yeager, navin.kumar@blubridge.com, 2026-09-30): what we found

1. **It moved the TI pointer back** from `b975be5` to `38d0332` (`git diff e6eaeba HEAD -- Tensor-Implementations`).
   BluTrain does not contain TI's code, only a pointer to one TI commit; the pointer does not follow TI's latest by
   itself. `38d0332` is the commit just before our 09-29 port `b975be5`, so every BLUCCL build dropped F2 (fused gradient
   accumulation), the LinearBackward fix, the reduction fixes, the GPU conversion fixes and Bug D (15 files). Most likely
   his local TI was still at `38d0332` when he committed everything. Fixed by our `3298e74`. Colleagues must run
   `git submodule update` after pulling.
2. **An intermittent ShardedMuon failure at W = 1.** His own test `test_muon_rsag` (fp32 weights; it requires Muon's 2
   modes to give bit-identical results at W ≤ 2): always exactly 7,077 of 82,120 numbers differ (max_abs 2.679e-05,
   max_rel 1.719e-02), sometimes in "sync publish", sometimes in "staged refresh". Same test, same TI, same libraries,
   same compile line, with the ZeRO sources of `e6eaeba` (before his commit): 6 of 6 runs pass. With `a3aba3e`: 6 of 6
   runs have a failure. After our F4 fix: the same failure with the same signature (and F4 makes W = 1 Muon skip the
   shared gradient buffer entirely), so the bug is not in the gradient-read path but in the update or publish path.
   Checked and ruled out: a stream race in `d2d` (it is `cudaMemcpyAsync` on the current stream). A lead, not verified:
   his new `accumulate_owned_grad_norm` dropped the per-tensor `sync_compute()` the old code did, and the code's comments
   warn that `multi_tensor_*` stage their descriptors through one static pinned buffer and race without a sync.
3. **His test scripts no longer compile:** `dist/zero/build_and_test.sh` and `run_muon_rsag_test.sh` miss
   `-Idist/Distributed_Checkpointing/include`, which `ZeroOptimizerBase.h` now needs for `OptimShardSpec.h`. (We built
   his tests with the Makefile's own compile line instead; his files were not edited.)
4. **2 symlinks committed** to his machine: `checkpoints_bluscript1`, `checkpoints_bluscript_muon1` →
   `/data/checkpoints/gv_BluTrain/...`.
5. **Config changes in `bluscript_zero_ckpt.cpp`:** `rope_theta` 500000 → **10000**, `weight_tying` → true, B = 8,
   `global_batch` 1,048,576, checkpoint every 1,600 steps. **SFT risk:** our `dist_side_merge` uses 500000. If his
   step-8000 checkpoint was trained with 10000, our SFT must use 10000. The checkpoint stores only weights, so a wrong
   value would not be caught at load, only by a bad step-0 val loss.

**Message for Navin** (ready to paste; written for him):
```
Hi Navin, about a3aba3e on dist_side_merge_BLUCCL:

1. It moved the Tensor-Implementations pointer from b975be5 back to 38d0332 (probably an old
   local submodule when you committed), which dropped the 09-29 library fixes. I pushed
   3298e74 to point it back to b975be5. After pulling: git submodule update, then rebuild TI.

2. test_muon_rsag fails at random at W=1 since a3aba3e (1x H200, TI b975be5): always
   7077/82120 elements differ (max_rel 1.7e-2), sometimes in "sync publish", sometimes in
   "staged refresh". Same test, same TI and libs with e6eaeba's dist/zero: 6/6 runs pass.
   With a3aba3e: 6/6 runs have a failure. A lead, not verified: the new
   accumulate_owned_grad_norm dropped the per-tensor sync_compute() the old code did, and
   the comments say multi_tensor_* share one static pinned descriptor buffer.

3. Small things: build_and_test.sh and run_muon_rsag_test.sh no longer compile (they miss
   -Idist/Distributed_Checkpointing/include, needed for OptimShardSpec.h); the commit added
   2 symlinks to /data/checkpoints/gv_BluTrain/...; and bluscript_zero_ckpt.cpp now has
   rope_theta = 10000. Which rope_theta did your step-8000 checkpoint use? We start SFT from it.

Also: I pushed 149249d. At 1 GPU the 3 ZeRO optimizers no longer round the gradient to bf16
and back before the update (nothing is sent at W=1). W>1 is unchanged, and your in-place
change is untouched.
```

---

## 14. What happened today, in order (times are UTC)

- **05:37–10:20, questions about the tied embedding and the optimizers** (answers in sections 1–11):
  - the lm_head's view of `wte`, both dtypes; `atomicAdd` and "one rounding per occurrence"; transpose (view) vs
    contiguous (copy); the lm_head copy redundancy (noted as a future task);
  - the ZeRO-1 changes: the #2 fix step by step (B, B′, C, D), F1, the bf16 buffers, the 6-step flow;
  - F4 found in all 3 optimizers (not only AdamW); the Muon `launch_reduce` leftover noted; the user asked to leave the
    optimizers for later and focus on the embedding;
  - the forward/backward sequence around the table (ops 1–10), the lookup backward as a sum.
- **10:20–13:00, the embedding kernels and the BLUCCL branch:** why bf16 uses `atomicAdd`; the colleague's
  `EmbeddingBwdFused` on BLUCCL (CUB, fp32 adds, which trainer and flag), the forward kernels; the residual stream, the
  resid dtype vs the embed dtype, RMSNorm's fp32 maths; the D1/D2/D3 chain; Muon vs AdamW for `wte`.
- **13:00–14:30, precision of the send path:** the fp32 micro-step buffer, the switch's `.acc::f32` returning bf16, the
  1,000,000-pair add test, idea A vs B, F1 explained, the send-dtype flags and their consequences.
- **14:30–15:40, F4 on the BLUCCL box:** the user decided to fix F4 on `dist_side_merge_BLUCCL` only, on the 1 × H200
  (`round-earth-echoes-fin-03`), and push from there. The clone showed Navin's new commit `a3aba3e` (in-place collectives,
  TI pointer moved back). His code was read in full; the design was refined on top of it (option B).
- **15:40–16:30, build, test, push:** old-code build (8 min 37 s, cuBLAS, TI set to `b975be5` in the working copy); old
  suite (F4 17 FAIL as expected, F1 36/36, `test_zero` PASS, `test_muon_rsag` failing); the `e6eaeba` baseline (6/6 pass)
  and the repeats proving his commit introduced the flaky failure; patch applied (`--check`, then `--apply`); new suite
  (F4 30/30, F1 36/36, `test_zero` PASS, `test_muon_rsag` unchanged); 2 commits pushed by the user; `ls-remote` verified.
- **16:37:** everything copied from the box (`profiling_reports/f4_2026_09_30/box_final/`); the box can be released.

---

## 15. Every question of the day, with its short answer

1. *With a bf16 table and tying, is the lm_head gradient added to the embedding gradient; which tensor does the lm_head
   read?* One weight only. fp32 table: the lm_head reads a fresh bf16 copy, its gradient is widened to fp32 and added in
   fp32. bf16 table: it reads `wte` itself, both parts are bf16 and added in bf16 (sections 2, 4, 7).
2. *What is "one rounding per occurrence" and `atomicAdd`?* The bf16 lookup backward on `dist_side_merge` adds each
   position's vector into one bf16 spot, rounding after every add, in random order (6.1).
3. *Is transpose a view or a copy? Can the bf16 / transposed copy be made once?* Transpose is a view; `contiguous` copies.
   Not once at the start (the weights change every step), but once per optimizer step, yes: a future task (section 3).
4. *Are the embedding gradients bf16 padded to fp32 anyway in the fp32 case?* The lm_head part yes (widening is exact).
   The lookup part is summed in fp32 in the fp32 case (section 4, 6.1).
5. *One gradient row per token per micro-step, however often it repeats?* Yes, but that row is the sum of all its
   positions' vectors (section 4).
6. *What did we change in ZeRO-1: v1, v2, F1?* #2 in v1 only (section 10), F1 in all 3 (section 11).
7. *Where do the bf16 buffers come from, if the gradients are fp32?* The send buffer is in the weights' dtype, and the new
   weights must be bf16 (section 9).
8. *So we removed 3 temporaries in steps 1, 3, 5?* Yes: 3 steps, 4 places in the code (section 10).
9. *What is F4?* The 1-GPU bf16 round trip (section 12). *Only AdamW?* No: all 3 optimizers.
10. *Is the lm_head the same as the embedding at every step; the untied case?* The same table at every step; untied: a
    separate table, the copy task does not apply, the repetition issue does (sections 2, 3).
11. *The shapes, and why the embedding backward is a sum; is the lookup a matmul?* The lookup is a copy; the matmul is the
    lm_head at the end; the backward is a sum by the chain rule (sections 3, 4).
12. *Why `atomicAdd`, why not sum like fp32?* The cooperative kernel was written for fp32 only; nothing in its method
    needs fp32; an fp32 total is also needed to remove the rounding (6.1).
13. *The colleague's kernel without `atomicAdd`; kernel names; CUB; the forward kernel names?* 6.2, 6.3.
14. *Which kernel does the colleague's training script really use?* 6.4 (proven from the scripts and the call sites).
15. *Op 1's inputs and outputs; is op 2 compute; what is the residual; is RMSNorm fp32; is the residual dtype tied to the
    embed dtype?* Section 3.
16. *Does the lookup backward output 2 tensors, including [8, 4096]?* No, one: [50,304 × 2,304]; the ids are an input
    (section 4).
17. *Do only the rows of tokens in the batch change?* No, all rows (section 5).
18. *Does Muon use Adam?* Muon updates only the layers' 2-D matrices; `wte` and the norms use AdamW (section 9).
19. *Is the fused kernel really fp32 for both inputs?* Yes, verified line by line (6.2).
20. *With tying, is the lookup part zero for absent rows; without tying?* Yes in both; without tying there is no D2 (7.2).
21. *Why is the sum of 2 bf16 numbers rounded again; why is 0.003 lost when bf16 can hold 1.01?* bf16 cannot hold 1.01;
    the half-gap rule (section 1).
22. *D2 is only 2 numbers, why worry?* It is one rounding, at most 0.39%; measure before changing (7.2).
23. *Multi-GPU: sent in bf16, summed in fp32 in the switch, sent back as fp32?* Summed in fp32, returned as bf16 (7.3).
24. *Why keep an fp32 embedding gradient if it is sent in bf16 anyway?* With an fp32 table it is sent in fp32; with a bf16
    table the fp32 buffer keeps the rounding at once per step (7.3).
25. *Can D2 be done in fp32 then converted; is there overhead?* Same answer as today unless the sum stays fp32 (idea B)
    (7.2).
26. *"You say different things at different times"?* Two cases were mixed in one paragraph; the per-case chain (7.1)
    is the clean version.
27. *What is F1 actually?* Section 11.
28. *How should we remove F4, at the root?* At 1 GPU the gradient must not go through the send buffer (12.2).
29. *Does option A fight Navin's changes; do the flags matter at 1 GPU?* A touches the shared gradient-view code; with B
    the flags have no effect at 1 GPU (12.2, section 8).
30. *There is a flag that sends fp32 gradients as bf16?* `BLU_BF16_REDUCE` (section 8).
31. *Did Navin edit outside TI; does BluTrain still point to the latest TI?* He did not edit TI's code, but moved the
    pointer back; fixed in `3298e74` (section 13).
32. *Is the bug cleared now, partly?* Navin removed only the plain copy (step 2); the rounding stayed until our fix (12.1).
33. *Is the trainer the only test; how long; CPU or GPU?* The unit tests are the proof; the trainer A/B is an optional
    end-to-end check (~10–15 min GPU); builds are CPU, runs are GPU.
34. *F4 fixed and pushed; is the pointer on the latest TI?* Yes, both, confirmed with `ls-remote` (12.4).

---

## 16. Open items for 2026-10-01

The same list is in the loss doc §26 ("TOMORROW").
1. Send Navin the message in section 13.
2. Navin's intermittent ShardedMuon result at W = 1 (his bug; offer the evidence and the lead).
3. Ask Navin which `rope_theta` his step-8000 checkpoint used, before any SFT run.
4. Optional: the 1-GPU trainer A/B for F4 (fused trainer, flux; scripts ready in `profiling_reports/f4_2026_09_30/`; needs
   a new box, a clone and a ~9 min build). Also confirm there that the "no effect at W=1" line prints.
5. Fetch `dist_side_merge_BLUCCL` locally.
6. F4 on `dist_side_merge`: decide whether to port `149249d` there (separately).
7. The embedding ablation, re-scoped: measure D1-end, D2 and D3 for the bf16 table (fused setup), and the `atomicAdd`
   D1 on `dist_side_merge`; then decide idea B (option 2) and the SFT setup.
   **Done locally on 2026-10-01 (section 17)**, with real token ids, the real `dist_side_merge` kernels and made-up
   gradient numbers. Still open: the same test on real gradients from the trainer (H200), and Kruthik's real kernel
   (only emulated).
8. SFT embedding backward on `dist_side_merge`: port `EmbeddingBwdFused`, use an fp32 table, or keep; the fused hook
   ignores N, which must be handled before SFT with variable token counts.
9. Multi-GPU send dtype research: the flags' consequences, NCCL's bf16 add on `dist_side_merge`, BluCCL's multi-node path.
10. The Muon `launch_reduce` leftover (`BLU_MUON_RB=1` only).
11. Reuse the tied lm_head's bf16 transposed copy within an optimizer step.
12. SFT task 1 H200 test (`BLU_INIT_FROM`, SFT doc §35).
13. Commit the docs and records; rotate the GitHub token.

---

## 17. Measured on 2026-10-01: how much D1, D2 and D3 lose (local test)

This is item 7 of section 16, done locally on 2026-10-01. Section 7 says *where* `wte`'s gradient is rounded. This
section measures *how much* each place loses, with the real library code of `dist_side_merge`.

### 17.0 The short answer (as explained in the chat)

- **fp32 table:** almost nothing is lost anywhere. The worst error in all 50 lookup cases is 0.00006% (5.8e-7).
- **bf16 table (`dist_side_merge`):** frequent tokens lose part of their gradient.
  - On a real flux micro-batch, `,` `.` ` the` (about 1,000-1,300 repeats each) lose **4%** when their gradient
    vectors point in random directions.
  - They lose **47-55%** when the vectors point mostly the same way.
  - A token that appears once loses 0.
  - The more often a token appears, the more it loses.
- **Why:** bf16 keeps about 3 digits. In bf16, 256 + 1 = 256: the 1 is lost.
  - The bf16 kernel adds the K vectors one by one into a bf16 row (`atomicAdd`, `EmbeddingKernels.cu:148`).
  - Once the row is large, each new small vector is rounded away.
  - If the vectors share a direction, the row keeps growing, so more is rounded away. If the directions are random,
    the values cancel each other, the row stays small, and less is lost.
- **Not known yet:** how much real training gradients point the same way. The test used made-up gradient numbers, so
  the real loss for the top tokens is somewhere between ~4% and ~55%. Finding it needs a dump of real gradients from
  the trainer (H200).

### 17.1 What was real in the test and what was made up

| Part | Real or made up |
|---|---|
| Token ids: which token sits at which position, how many times it repeats | **Real**: 4 real flux micro-batches (`flux_train_000001.bin` at token 0, 1,000,000 and 50,000,000; `flux_train_000050.bin` at 7,000,000), 32,768 tokens each. Plus made-up id sets for edge cases (17.2). |
| The code that adds the vectors | **Real**: the library's own functions, `dist_side_merge` TI `d96b991` (clean tree; `lib/libtensor.so` built 2026-09-29, sm_86, cuBLAS) |
| The gradient numbers (2,304 per position) | **Made up**: random numbers in 9 styles (17.2). No model ran. |
| The correct answer | The exact sum, computed on the CPU in fp64 from the same numbers |

Every made-up value is rounded to bf16 first. So the fp32 arm and the bf16 arm add exactly the same numbers, and every
difference comes from the adding, not from the inputs.

### 17.2 How the test works

**File:** `Tensor-Implementations/Tests/kernel_tests/test_embedding_grad_precision.cu`. It is local-only: it's listed
in `.git/modules/Tensor-Implementations/info/exclude`, like the 2 loss tests. A copy and all outputs are in
`profiling_reports/embed_ablation_2026_10_01/`.
- **Build:** the Makefile's `run-cuda` nvcc line plus `-Xcompiler -fopenmp`.
- **Run:** `./embed_precision d1,d2,d3` on the RTX 3060. Wall time 1 min 07 s; peak host memory 4.1 GB.
- The predictions (P1-P7) were written in the file header before the first run.

**D1, the lookup backward alone.** The test calls `cuda::embedding_backward_cuda` directly. Arms:
- `fp32_cooperative`: the fp32 path. That is the cooperative kernel (`EmbeddingKernels.cu:415-424`), or fp32
  `atomicAdd` when `BLUTRAIN_EMB_BWD_COOPERATIVE=0` (`:391`). Run 2 times.
- `bf16_atomic`: the bf16 path, `atomicAdd` into a bf16 table (`:442`, `:148`). Run 5 times, because its order is
  not fixed.
- `sorted_fp32_once(emu)`: a **CPU emulation**, not the real kernel. It copies the arithmetic of BLUCCL's
  `EmbeddingBwdFused` (section 6.2): positions in ascending order; up to 32 summed whole; else chunks of 32 folded in
  order; all in fp32; one rounding to bf16 at the end.
- `floor`: the exact sum rounded to bf16 once. No bf16 table can do better than this.

**Token sets** (N = 32,768 unless noted):

| Set | What it is |
|---|---|
| `uniform` | random ids |
| `flux_a`-`flux_d` | the 4 real micro-batches |
| `zipf1.0`, `zipf1.3` | Zipf-distributed ids |
| `sft_padded` | rows of 4,096: a stand-in chat template + flux text + EOT; the rest padded with EOT (50256), pad positions get gradient exactly 0 |
| `sft_packed` | the same samples packed into each row |
| `repeat_doc` | one 512-token flux document repeated |
| `one_token` | token 262 at every position |
| `k_sweep` | 15 tokens with K = 1, 2, 4 … 16,384 |
| `flux_a_N8192`, `flux_a_N16384` | `flux_a` cut to B = 2 and B = 4 |

The stand-in template ids are not the real SFT template (none is chosen yet). Only the repetition pattern matters.

**Gradient styles:**
- `iid`: random numbers, σ = 1e-3.
- `alignedρ`: each vector = ρ × (one vector shared by all positions of that token) + √(1−ρ²) × (its own random vector),
  with ρ = 0.5, 0.9, 0.99, 1.0. At ρ = 0.9 the shared part is 0.9 and the own part is 0.44.
- `lognormal1.5`: random sizes, a few big and many small.
- `big+small`: the first occurrence is 1×, every later one is 0.003× the same vector.
- `iid_scale1e-8`, `iid_scale1e+2`: to check that the size of the numbers doesn't matter.

**Metrics:**
- Per row: the relative error ‖result − exact‖ / ‖exact‖. Rows are grouped by K (the number of non-zero
  occurrences), with the median and the max.
- "Global": the same formula over all present rows together. Big rows weigh more.
- Also: the 5 most frequent tokens, and the run-to-run bit differences.

**D2, the engine's add of the two parts, through real autograd.** The ops follow the trainer's order
(`bluscript_zero_ckpt.cpp:748`, `:786-788`):
1. `embedding(wte, ids) → to_dtype(bf16) → sum(x · G1)`
2. `matmul(H, contiguous(transpose(to_dtype(wte, bf16)))) → sum(logits · G2)`
3. add the two, `backward()`, read `wte.grad_view()`.

Details:
- `sum` of a bf16 tensor returns bf16 (`ReductionImplGPU.cu:119`), and its backward makes ones in that dtype
  (`ReductionBackward.cpp:28`). So G1 and G2 arrive exactly, in bf16, as in the trainer.
- N = 4,096 flux tokens (`flux_train_000004.bin` at token 3,000,000; 1,133 unique; max K 148).
- H is random bf16 with RMS 1. G2 is a realistic dlogits: softmax(random logits) − one-hot(the next flux token),
  divided by N.
- The size of the lookup part is swept against the lm_head part: s = 0.01, 0.1, 1, 10, 100. G1 is scaled by a power
  of 2, so it stays bf16-exact, and the actual s is printed.
- Each case runs 3 times: lookup part only (G2 = 0), lm_head part only (G1 = 0), and both.
- Errors are read on 256 sampled rows (64 most frequent, 128 random present, 64 absent), against fp64 references of
  both parts.
- "Lookup kept" = (both − lm_head only) vs the exact lookup part: how much of the lookup part survived the add.
  "lm_head kept" is the same the other way round.

**D3, the micro-steps and the send.** M = 4 and 32 micro-steps of N = 8,192 flux tokens (`flux_train_000005.bin`,
consecutive). A token keeps its shared direction across micro-steps. Arms:
- fp32 table: fp32 kernel + `acc += upd` (`AutogradMeta.cpp`, the fp32 branch).
- bf16 table: bf16 kernel + F2's `cuda_accumulate_lowp_into_fp32` (`AutogradMeta.cpp:193`).
- A bf16 running sum (library bf16 `+=`). This is my **assumption** of what `BLU_GRAD_VIEW=1` does; not checked,
  because BLUCCL isn't fetched locally.
- Then the send rounding (library `as_type(bf16)`): for the bf16 table (the default `SAME_AS_PARAM`; also at W = 1
  on `dist_side_merge`, where F4 is not fixed), and for the fp32 table with `BLU_BF16_REDUCE=1`.

**Self-checks.** If one fails, the numbers are not results. All passed: 234 in the full run, 152 in the fp32-atomic
run.

| Check | What it proves |
|---|---|
| SC1 | inputs ±2⁻¹⁰ with every token K = 4: every arm returns the exact sum |
| SC2 | every K = 1 row of the bf16 kernel equals its one input |
| SC3 | absent rows, and rows whose sum is exactly 0 (SFT pads), are exactly 0 |
| SC4 | the fp32 arm's error is < 1e-4 in every case |
| SC5 | the fp32 cooperative kernel gives the same bits in 2 runs |
| SC6 | D2: on every deterministic row of the whole table, (both) == round(lookup only + lm_head only), and the lookup part is exactly 0 on absent rows |
| SC7 | the CPU's bf16 rounding (used for `floor` and the emulation) equals the library's `as_type(bf16)` on 3M normal floats, incl. 1M exact ties |
| SC8 | F2 took every call (no fallback) |
| SC9 | D3: every arm is exact on rows seen once in the whole step |

### 17.3 D1 results

**`flux_a`, by K** (median error per row; the max is in the log):

| K | rows | fp32 | bf16, random | bf16, 90% same | emulation (BLUCCL-style) | floor |
|---|---|---|---|---|---|---|
| 1 | 2,796 | 0 | 0 | 0 | 0 | 0 |
| 2-3 | 1,449 | 0 | 0.18% | 0.20% | 0.18% | 0.18% |
| 4-15 | 1,099 | 0 | 0.32% | 0.28% | 0.17% | 0.17% |
| 16-63 | 211 | 1e-9 | 0.61% | 0.49% | 0.17% | 0.17% |
| 64-255 | 31 | 5e-9 | 1.3% | 1.2% | 0.17% | 0.17% |
| 256-1023 | 10 | 2e-8 | 2.7% | 17.5% | 0.17% | 0.17% |
| ≥ 1024 | 3 | 3e-8 | 4.2% | 52% | 0.17% | 0.17% |

**The 5 most frequent tokens of `flux_a`:**

| Token id | K | bf16, random | bf16, 90% same |
|---|---|---|---|
| 11 (`,`) | 1,290 | 4.2% | 55% |
| 13 (`.`) | 1,200 | 4.2% | 52% |
| 262 (` the`) | 1,056 | 3.9% | 47% |
| 290 | 829 | 3.3% | 37% |
| 198 | 784 | 3.3% | 34% |

The fp32 kernel's error is ≤ 3e-8 on all 5, and the emulation and the floor are 0.16-0.18% on all 5.

**`k_sweep` (one token per K):**

| K | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|---|
| bf16, random | 3.8% | 5.2% | 7.8% | 10.8% | 14.5% |
| bf16, 90% same | 45% | 70% | 84% | 91% | 95% |

The fp32 error stays ≤ 2.2e-7 and the floor stays 0.17% at every K.

**Every token set, bf16 kernel, global error:**

| Set | unique ids | max K | random | 90% same | few big, many small |
|---|---|---|---|---|---|
| `uniform` | 24,146 | 7 | 0.14% | 0.17% | 0.11% |
| `flux_a` | 5,599 | 1,290 | 1.9% | 40% | 1.9% |
| `flux_b` | 6,619 | 1,429 | 1.8% | 46% | 1.9% |
| `flux_c` | 6,629 | 1,276 | 1.8% | 42% | 1.5% |
| `flux_d` | 6,465 | 1,305 | 1.9% | 43% | 1.6% |
| `zipf1.0` | 9,499 | 2,853 | 2.4% | 65% | 2.0% |
| `zipf1.3` | 3,144 | 8,692 | 6.6% | 87% | 4.9% |
| `sft_padded` | 1,491 | 176 | 0.69% | 1.3% | 0.74% |
| `sft_packed` | 5,872 | 1,103 | 1.7% | 36% | 1.5% |
| `repeat_doc` | 274 | 1,600 | 2.3% | 45% | 1.8% |
| `one_token` | 1 | 32,768 | 21% | 98% | 11% |
| `k_sweep` | 16 | 16,384 | 12% | 94% | 7.4% |
| `flux_a_N8192` | 2,052 | 307 | 0.97% | 2.9% | not run |
| `flux_a_N16384` | 3,399 | 612 | 1.4% | 13% | not run |

The fp32 error is 0 to 5.8e-7 in every case. The floor is 0.10-0.17% in every case, with one exception:
`k_sweep` with 100% same direction. There every exact sum is bf16-exact (all vectors equal, K a power of 2), so the
floor is 0, but the bf16 kernel still loses 97%. The emulation's global error equals the floor's (to 3 digits) in every
case.

**Other gradient styles on `flux_a`** (bf16, global):

| Style | 50% same | 90% same | 99% same | 100% same | big + small | values 1e-8 | values 1e+2 |
|---|---|---|---|---|---|---|---|
| Error | 18% | 40% | 51% | 56% | 10.5% | 1.9% | 1.9% |

The size of the numbers doesn't change the error. The direction does.

**SFT sets:**
- `sft_padded` loses much less: 0.69% / 1.3%. Its rows are short (one sample each, then padding), so K stays ≤ 176.
  The pads add exact zeros, which round nothing.
- `sft_packed` is about the same as pretraining: 1.7% / 36%.

**Run to run:**
- The bf16 kernel gives different bits every run once K > 1. On `flux_a` (random), 39,857 of 12.9M elements differ.
- The fp32 cooperative kernel gave the same bits in every case (SC5).
- fp32 `atomicAdd` (`BLUTRAIN_EMB_BWD_COOPERATIVE=0`) differs in 40 of 50 cases (up to 2,058 elements). Its error is
  still ≤ 8.7e-7.

### 17.4 D2 results (the engine's add)

Median error on the 64 most frequent rows. N = 4,096 here, so max K is 148 and D1's own loss is small; most of what's
below comes from the add.

**fp32 table:**

| s (lookup / lm_head) | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| lookup kept (random) | 1.6e-6 | 7.6e-8 | 2.4e-9 | 9.4e-10 | 1.4e-9 |
| lm_head kept | 0.17% | 0.17% | 0.17% | 0.17% | 0.17% |

**bf16 table:**

| s (lookup / lm_head) | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| lookup kept (random) | **21%** | 2.8% | 0.64% | 0.61% | 0.61% |
| lookup kept (90% same) | **39%** | 2.6% | 0.62% | 0.50% | 0.50% |
| lm_head kept (random) | 0.23% | 0.24% | 0.31% | 1.4% | **11%** |
| lm_head kept (90% same) | 0.23% | 0.24% | 0.26% | 0.82% | **12.5%** |

(The actual s values were 0.0075-0.009, 0.07-0.12, 0.96-1.15, 7.7-9.2 and 74-123.)

What this shows:
- **The fp32 table keeps both parts.** The 0.17% on the lm_head part is the bf16 matmul output. Both tables have it,
  because the tied lm_head runs in bf16 (`:786-788`).
- **The bf16 table loses the smaller part.** When one part is 100× smaller, 21-39% of it (the lookup part) or 11-12.5%
  of it (the lm_head part) is lost in the add (`Engine.cpp:341-343`, `sum += grads[i]` in bf16).
- Absent rows have only the lm_head part, and both tables give 0.17% there.
- **A BLUCCL-style lookup kernel would not fix D2.** This add is in the engine, not in the lookup kernel. I checked
  this on `dist_side_merge` only.
- SC6 passed, so the result is exactly one rounding of (lookup part + lm_head part), as section 7.2 says.

### 17.5 D3 results (micro-steps, then the send)

Global error (random / 90% same):

| Arm | M = 4 | M = 32 |
|---|---|---|
| fp32 table (buffer = what is sent) | 1e-8 / 2e-8 | 3e-8 / 6e-8 |
| fp32 table + `BLU_BF16_REDUCE=1` | 0.16% / 0.17% | 0.16% / 0.17% |
| bf16 table, fp32 buffer (before the send) | 0.94% / 2.7% | 0.95% / 2.4% |
| bf16 table, after the send to bf16 | 0.95% / 2.7% | 0.96% / 2.4% |
| bf16 running sum (assumed `BLU_GRAD_VIEW`) | 0.96% / 2.7% | 1.1% / 2.4% |
| floor (round the exact total once) | 0.16% / 0.17% | 0.16% / 0.17% |

What this shows:
- F2's fp32 buffer adds no error.
- Almost all of the bf16 table's error is D1, carried from each micro-step.
- The send rounding adds almost nothing on top, because D1's error is bigger.
- For the fp32 table, `BLU_BF16_REDUCE` costs one rounding (0.16%).
- The 8-GPU switch rounding (section 7.3) is not in this test. It needs 8 GPUs.

### 17.6 Predictions vs results

| Prediction (written before the run) | Result |
|---|---|
| P1: fp32 ≈ 1e-7 to 1e-5 at every K | held, and better: 0 to 6e-7 |
| P2: the floor is ~1e-3 at every K, independent of K | held: 0.10-0.17% (0 where every exact sum is bf16-exact) |
| P3: the BLUCCL-style emulation is within 1.1× of the floor | held: equal in every case |
| P4: bf16 `atomicAdd` grows with K; random ~2⁻⁹·√K; same-direction loses tens of % once K ≫ 256 | held: 4.2% at K ≈ 1,000 (√K rule gives ~6%); 52% at K ≈ 1,000 with 90% same |
| P5: bf16 differs run to run, fp32 cooperative never | held |
| P6: the add costs ≤ 1 rounding of the sum; the smaller part loses most when the parts differ ≫ 256× | held; the loss starts already at 100× (21-39%) |
| P7: the send adds ~1 floor; `BLU_GRAD_VIEW` adds more with M | partly: the send adds ~1 floor only for the fp32 table; for the bf16 table it adds almost nothing, because D1's error is bigger |

### 17.7 What this does not answer yet

1. **The real gradients.** Two things are unknown: how much a token's real gradient vectors share a direction, and the
   real size ratio s between the lookup part and the lm_head part. Next: dump the gradient arriving at the embedding
   output (one micro-step) and both parts of `wte`'s gradient from the trainer on an H200, then run this test on them.
2. **Kruthik's real kernel.** Only its arithmetic was emulated, on the CPU. The real kernel runs only on BLUCCL.
3. **`BLU_GRAD_VIEW`.** That arm is my assumption, not the BLUCCL code.
4. **The 8-GPU switch rounding** (section 7.3, step 5). Not measured.
5. **D2 and D3 used smaller N** (4,096 and 8,192) than the trainer (32,768). So D1's part inside them is smaller than
   in real training. D1 at N = 32,768 is in 17.3.

### 17.8 Questions asked about the result, with the answers

- **"In real life the 1,290 vectors have different directions, so no problem?"** No. Random directions were the best
  case in the test, and `,` still lost 4.2%, while the fp32 table lost 0.0000028%. Real gradients are probably
  partly the same direction: the embedding of `,` does learn, so its 1,290 vectors must share some direction, or their
  sum would be noise. That is reasoning, not a measurement (17.7, item 1).
- **"How many times did `,` `.` ` the` repeat?"** 1,290, 1,200 and 1,056 times in one real flux micro-batch of
  32,768 tokens (17.3).
- **"What does 4% error mean?"** The correct row is the sum of the 1,290 vectors. The bf16 table gives a row that is
  4% away from it.
- **"Is the 52% real?"** The kernel really does this when the vectors point mostly the same way (90% shared). Whether
  real training gradients look like that is not known yet.
- **"What was the '1 million times worse'?"** It is just 4.2% ÷ 0.0000028% ≈ 1.5 million. It only means fp32 loses
  almost nothing and bf16 loses 4%. The two numbers are what matter.
- **"Did the test use only text?"** It has 2 inputs. The token ids come from real text (the flux files, already
  tokenized), plus made-up id sets. The gradient numbers are random numbers, not from text or a model. No model ran.
  The text gives how many times each token repeats; the numbers are made up.
- **"Is the test right?"** All 234 + 152 self-checks passed (17.2). For example, with inputs whose exact sum is known
  (SC1), every arm returns that exact sum.

### 17.9 Files (`profiling_reports/embed_ablation_2026_10_01/`)

- `test_embedding_grad_precision.cu`: a copy of the test.
- `embed_precision`: the binary (sm_86).
- `logs/run_1.log`: D1 + D2 + D3, cooperative fp32 path; ends with a SUMMARY block.
- `logs/run_fp32atomic.log`: D1 with `BLUTRAIN_EMB_BWD_COOPERATIVE=0`.
- `logs/embed_d1_buckets.csv`, `logs/embed_d1_buckets_fp32atomic.csv`, `logs/embed_d2.csv`, `logs/embed_d3.csv`.
