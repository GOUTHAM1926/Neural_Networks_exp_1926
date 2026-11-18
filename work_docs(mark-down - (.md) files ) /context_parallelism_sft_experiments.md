# Context parallelism for SFT training: everything we built and tested

Written 2026-11-12. One file with the whole story of context parallelism (CP) in our SFT trainer: why we needed it,
how it works, which model sizes and context lengths we tested, every result, the bug we found and fixed, what is not
tested yet, and where the code and the records are.

Sources: `sft_work_documentation.md` sections 45.124, 45.127, 45.177 to 45.186 and 45.192; the result files in
`profiling_reports/cp_2026_11_06/`, `cp_2026_11_07_1x/` and `cp_2026_11_07_2x/`; `BluTrain/sft/README.md` section 5.2;
`cp_sft_design.md`.

---

## 1. The short version

- **What it is:** with context parallelism, several GPUs share one training row. Each GPU holds one part of the row's
  tokens. Without it, one GPU must hold the whole row.
- **Why we need it:** a long row needs a lot of memory. The 1.6B model with a 65,536-token row does not fit on one
  H200 (out of memory at 143.1 GB). With CP on 2 GPUs it runs at 96.6 GB per GPU.
- **How it is built:** our own kernels are unchanged. Around the attention we call two all-to-all exchanges from
  Adhitya's Context_Parallelism module (the Ulysses method), used as they are.
- **Switch:** `BLU_SFT_CP=U` in the trainer (`SFT_CP=U` in the launcher). U is the number of GPUs that share a row.
  The default is 1, which means off.
- **What is tested:** CP on 2 GPUs (U = 2), on two model sizes (135M and 1.6B) and three row lengths (4,096, 65,536
  and 131,072 tokens). In every test, training with CP gives the same result as training without CP, within the
  normal difference between two identical runs.
- **Speed and memory at long rows (135M):** 1.58 times faster at 65,536 tokens and 1.65 times faster at 131,072
  tokens than one GPU, with 56 to 59% of the memory per GPU.
- **A bug found on the way:** the first version lost part of the gradient (7% lower gradient norm). The cause was in
  the tensor library's `autograd::contiguous`, not in the CP code. It is fixed and pushed.
- **Not tested yet:** CP together with data parallelism (more GPUs than U), checkpoint save and resume under CP,
  gradient accumulation, and 7 smaller cases (section 12).

---

## 2. Words used in this document

| Word | Meaning |
| --- | --- |
| row | one training example of fixed length T tokens. A row holds one or more chats packed together. |
| T | the row length in tokens (the context length): 4,096, 65,536 or 131,072 here. |
| B | rows per GPU in one micro-batch. |
| context parallelism (CP) | several GPUs share each row; each GPU holds T ÷ U tokens of it. |
| U | the number of GPUs that share one row. `BLU_SFT_CP=U`. |
| CP group | the U GPUs that share the same rows. |
| data parallelism (DP) | several GPUs each hold different whole rows. This is what our normal multi-GPU training does. |
| slice | one GPU's part of a row: tokens from r × T ÷ U up to (r + 1) × T ÷ U, for GPU number r of the group. |
| all-to-all | an exchange where every GPU of the group sends a part of its data to every other GPU of the group. |
| Ulysses | the CP method we use: exchange before attention so that each GPU has the whole row for some of the heads. |
| q heads / kv heads | the attention's query heads and key/value heads. 135M: 4 and 2. 1.6B: 18 and 6. |
| step-0 values | the loss and the gradient norm of the first step, before any weight has changed. Two runs from the same weights and the same rows must agree here. |
| gradient norm | the size of the whole gradient of a step, as one number. |
| trained tokens | the tokens that count in the loss (the answer tokens). Prompt tokens are not trained. |
| peak memory | the highest GPU memory used during the run, as `nvidia-smi` shows it. |

---

## 3. Why we needed it

### 3.1 The limit of one GPU

Memory grows with the row length. These are the one-GPU numbers measured on 2026-10-31 (135M model, H200 with
143.8 GB, real prepared data at each length):

| row length T | largest B that fits | tokens per step | peak memory |
| --- | --- | --- | --- |
| 4,096 | 42 | 172,032 | 138,184 MiB |
| 8,192 | 21 | 172,032 | 138,182 MiB |
| 16,384 | 10 | 163,840 | 132,066 MiB |
| 32,768 | 5 | 163,840 | 132,066 MiB |
| 65,536 | 2 | 131,072 | 107,608 MiB |
| 131,072 | 1 | 131,072 | 107,640 MiB |

So the small 135M model still fits one row of 131,072 tokens on one GPU. A larger model does not: the 1.6B model with
one row of 65,536 tokens ran out of memory on one H200 (section 9.5).

### 3.2 The question on 2026-10-31

The user asked: if one GPU is not enough later, can the SFT trainer call the attention of Adhitya's CP module instead
of writing context parallelism again?

What we found by reading his module (commit `cf1372d`):

| | our SFT trainer | his CP trainer |
| --- | --- | --- |
| data | packed SFT rows: inputs, labels (prompt masked), chat lengths and offsets | pretraining token shards, documents split at the end-of-text token |
| attention | our packed attention with a per-chat plan | his own CP kernels (forks of our older kernels), document ids from scanning for the end token |
| RoPE positions | restart at 0 for every chat | continue across the whole row |
| loss | trained tokens only, sum ÷ trained count | pretraining cross-entropy |

His attention could not be called as it is, for three reasons:

1. His document boundaries come from scanning for the end token. In our SFT data the end token and the pad token are
   both id 0, so the scan would split rows wrongly.
2. His RoPE positions continue across the row. Ours restart per chat. A CP run would not match a one-GPU run.
3. His loss weighs every GPU equally. In SFT the GPUs hold different numbers of trained tokens (section 6).

**Decision:** do not rewrite context parallelism and do not swap our kernels for his. Use the Ulysses method with our
own kernels, and take from his module only the process groups and the two all-to-all exchanges.

---

## 4. How CP works in our SFT trainer

### 4.1 One training step with CP

1. **Loading.** The U GPUs of one CP group read the same rows.
2. **Slicing.** Each GPU keeps only its own slice of every row: the tokens, their labels, and their RoPE positions.
   Each token keeps its true position inside its chat.
3. **Per layer:**

   | stage | where it runs | data shape on each GPU |
   | --- | --- | --- |
   | RMSNorm, the q / k / v projections, our fused QK-norm + RoPE | on the slice | all heads, T ÷ U tokens |
   | all-to-all "combine" (Adhitya's `ulysses_combine`) | exchange | [B, H, T ÷ U, D] becomes [B, H ÷ U, T, D] |
   | our packed attention, with the chat plan of the whole row | on the whole row | H ÷ U heads, all T tokens |
   | all-to-all "partition" (Adhitya's `ulysses_partition`) | exchange | [B, Hq ÷ U, T, D] becomes [B, Hq, T ÷ U, D] |
   | output projection and the MLP | on the slice | all heads, T ÷ U tokens |

   So before the attention each GPU gives up tokens and receives heads: it then holds the whole row, but only for its
   share of the heads. Attention needs every token of the row, and it treats each head on its own, so this gives the
   same result as one GPU.
4. **Loss.** The lm_head and the loss run on the slice (section 6).
5. **Backward.** The gradient of each exchange is the opposite exchange (his `UlyssesCombineBackward` and
   `UlyssesPartitionBackward` nodes). The gradients are then summed over all GPUs as in normal training.

### 4.2 What is ours and what is Adhitya's

| part | whose |
| --- | --- |
| the two exchanges (`ulysses_combine`, `ulysses_partition`) and their two backward nodes | Adhitya's module `dist/Context_Parallelism`, pinned at `cf1372d`, used as it is, nothing copied |
| QK-norm + RoPE kernels (per-chat positions) | ours, unchanged |
| packed attention kernels and the chat plan | ours, unchanged; the plan is built for H ÷ U heads |
| loss, loader, slicing, the CP group setup | ours, in `bluscript_sft.cpp` |

No kernel was changed for CP.

### 4.3 The CP group

- The CP group is made with `ncclCommSplit` from the NCCL group of all GPUs: GPUs with the same value of
  rank ÷ U (rounded down) form one group.
- The group uses a blocking CUDA stream, as Adhitya's own setup does. His exchanges do not wait for the stream our
  kernels compute on; a blocking stream is ordered after it, so the exchange always sees finished data.

### 4.4 The rules (the trainer stops with an error otherwise)

U must divide:

- the number of GPUs,
- the number of q heads,
- the number of kv heads,
- the row length T.

CP also cannot be used together with the trained-rows switch (`BLU_SFT_TRAINED_ROWS`), because its row lists are made
for whole rows.

**Which U each model allows:**

| model | q heads / kv heads | allowed U | on 8 GPUs |
| --- | --- | --- | --- |
| 135M | 4 / 2 | 1 or 2 | 2 |
| 1.6B | 18 / 6 | 1, 2, 3 or 6 | 2 (3 and 6 do not divide 8) |

U = 4 is not possible for either model. More CP GPUs than kv heads would need copies of the kv heads or Adhitya's
ring + Ulysses hybrid. Neither is wired in our trainer.

**Rows per step with CP** = B × GRAD_ACCUM × (GPUs ÷ U).

### 4.5 Every change we made

All the CP work is in the training script and its build files. BluTrain commit `125ae9f` changed 3 code files:
`bluscript_sft.cpp` (206 lines), `Makefile` (7 lines) and `sft/run_sft.sh` (6 lines). Everything is behind
`BLU_SFT_CP`; at the default 1 every CP line is skipped, so normal SFT runs the same code as before.

**In the trainer `bluscript_sft.cpp`:**

| # | change | what it does |
| --- | --- | --- |
| 1 | `struct SftCp` | holds U, this GPU's number inside its CP group, and the group's communication handles |
| 2 | `cp_make_group` | builds the CP group: `ncclCommSplit` from the group of all GPUs, plus a blocking CUDA stream (section 4.3) |
| 3 | `cp_combine` | calls Adhitya's `ulysses_combine` and attaches his `UlyssesCombineBackward` node, so the gradient flows back through the exchange |
| 4 | `cp_partition` | the same for `ulysses_partition` and `UlyssesPartitionBackward` |
| 5 | `cp_slice` | cuts this GPU's token slice out of a whole-row tensor (the ids, the labels, the RoPE positions); does nothing at U = 1 |
| 6 | the attention block of each layer | at U = 1 the old lines run. At U > 1: combine q, k and v, run our attention on the whole row, partition the result back to the slice |
| 7 | reading `BLU_SFT_CP` and the checks | U must divide the GPU count, the q heads, the kv heads and T; otherwise the trainer stops with an error |
| 8 | the loader | it gets the data-parallel rank (rank ÷ U) and the data-parallel size (GPUs ÷ U), so the U GPUs of a group read the same rows |
| 9 | the trained-rows switch | refused together with CP |
| 10 | the token counts | tokens per step and the trained-token count are divided by U, because the U GPUs of a group counted the same rows |
| 11 | the embedding backward's workspace | sized for B × T ÷ U tokens, the slice |
| 12 | the chat plan of the attention | built for q heads ÷ U |
| 13 | training and validation steps | the ids, labels and positions go through `cp_slice` before the model |
| 14 | log lines | the trainer prints the CP setup at the start |

**In the build files:**

| file | change |
| --- | --- |
| `Makefile` | adds the include path of `dist/Context_Parallelism` (his module is header-only) |
| `sft/run_sft.sh` | `SFT_CP` is passed to the trainer as `BLU_SFT_CP` |

**In the tensor library (Tensor-Implementations `d59740b`):** one fix in `ReshapeOps.cpp` (14 lines) and one new test
(66 lines). This is the `autograd::contiguous` fix of section 8. It is autograd code, not a kernel.

**Kernels changed: none.**

### 4.6 Which kernels normally change for context parallelism

There are two common CP methods, and they differ exactly in what they ask of the kernels.

**Ring method.** Each GPU keeps the queries of its own slice. The keys and values of the other slices travel around
the GPUs, one slice per ring step. So one attention call sees its own queries against another slice's keys. That is
not what a normal attention kernel does, so the kernels must change. Adhitya's module has these changed kernels:

| kernel | what has to change | his file |
| --- | --- | --- |
| attention forward | queries and keys of different lengths (T_q ≠ T_k); causal masking on or off per ring step; it must also return the log-sum-exp of each row, so the partial results of the ring steps can be merged | `GQA_causal_cp_fwd_sm90.cu` (a fork of our Hopper forward v62) |
| attention backward | the same, for the gradient | `GQA_causal_cp_bwd_sm90.cu` (a fork of our older backward v44) |
| merge of the partial attention results | a new step that combines the outputs of all ring steps with their log-sum-exp values | `SDPAMerger.h` |
| document mask | a document id per token, staged inside the attention kernel, so tokens of different documents do not attend to each other across slices | `DocMask.h`, `DocIdKernel.cu` |
| QK-norm + RoPE | a position offset per slice, so a slice that starts at token 32,768 gets positions from 32,768 on | `fused_qknorm_rope_cp_fwd.cu`, `fused_qknorm_rope_cp_bwd.cu`, `RopeDeltas.h` |
| fused attention variants | the same changes in the fused kernels | `GQA_fused_*_sm103_cp.cu` |

**Ulysses method.** Before the attention, an exchange gives each GPU the whole row for a share of the heads. The
attention kernel then sees a normal whole row, only with fewer heads. So the attention kernel needs no change. His
Ulysses path still uses a special fused kernel (QK-norm + RoPE inside the attention), which is his design choice and
not something the method needs.

Kernels that normally need no change in either method: the linear layers, RMSNorm, the MLP, the embedding and the
loss. They work token by token, so a slice is only a shorter input.

### 4.7 Why we changed no kernel

Two reasons together made it possible.

**Reason 1: we chose Ulysses, not the ring.** With Ulysses the attention gets a whole row, so the kernel does the
same work as on one GPU.

**Reason 2: our kernels were already built for SFT rows.** Each thing CP needs was already an input of the kernel,
so the trainer only has to pass a different value:

| kernel | what CP needs from it | why it already works | how we checked |
| --- | --- | --- | --- |
| packed attention, forward and backward | run on a whole row with H ÷ U heads | the kernels take any head count; the chat plan is an input, and the trainer builds it for H ÷ U heads | `test_varlen_attention` with heads 2/1 and 4/2: 515 cases pass |
| fused QK-norm + RoPE, forward and backward | run on a slice that can start in the middle of a chat, with each token's true position | the kernel already takes one position per token as an array (we built that for packed chats); the trainer passes the slice of that array | `test_rope_slice.cu`: dq and dk bit-identical to the whole-row run, gain sums equal |
| the norm gains' gradient | the sum over all tokens of the row | each GPU's gradient is the sum over its own tokens for all heads; the normal gradient sum over GPUs adds the parts. No extra rule is needed | the step-0 gradient norm of the gains group: 0.0458 with CP, 0.0458 without |
| RMSNorm, the linear layers (GEMM), the MLP | run on fewer tokens | they work token by token | the whole-run comparisons of section 9 |
| embedding, forward and backward | run on the slice's tokens | per token; only the workspace size is a trainer setting | same |
| lm_head + loss | divide by the trained-token count of the whole step, not of the slice | the loss kernel already takes the denominator from outside (we built that for multi-GPU SFT) | step-0 loss equal with and without CP |
| optimizer and the gradient reduce | sum the gradients of all GPUs | unchanged; it already reduces over all GPUs | same |
| the two exchanges | move data between the GPUs of a group | Adhitya's functions, used as they are; inside they call NCCL's all-to-all | his module; our 2-GPU runs |

**What this choice gives us:**

- Normal SFT is untouched: with CP off every CP line is skipped.
- All the speed work and all the tests of our kernels carry over to CP without repeating them.
- We use our current attention backward. The ring path would use his fork of our older backward (v44).
- A CP run matches a one-GPU run of the same data, because positions and chat boundaries are handled the same way.

**What this choice costs:**

- U must divide the kv heads, so the 135M can only use 2 GPUs per row and the 1.6B at most 6 (section 4.4). The ring
  has no such limit.
- Every layer does two exchanges: q, k and v go out, the attention result comes back.

**An option we looked at and did not take (2026-11-07):** putting QK-norm + RoPE inside the attention kernel, as
Adhitya does. It is possible with our per-chat positions. But it would mean rewriting our attention forward and
backward, and the RoPE kernels are only 2.4% of the kernel time of a step (96.3 ms of 4,051 ms in the 135M
pretraining profile). So it stays a separate speed project, only if a measurement shows it is worth it.

---

## 5. The models and the context lengths we tested

### 5.1 Models

| model | layers | d_model | q / kv heads | ffn | vocab | weights used in the tests |
| --- | --- | --- | --- | --- | --- | --- |
| 135M | 37 | 512 | 4 / 2 | 1,408 | 49,280 | the pretrained checkpoint `ckpt_A` (pretrained at context length 4,096) |
| 1.6B | 26 | 2,304 | 18 / 6 | 6,144 | 49,280 | random init (only the shape was tested, not a trained model) |

### 5.2 Context lengths, by model and by test

| model | row length T | B | CP off (1 GPU) | CP = 2 (2 GPUs) | data used |
| --- | --- | --- | --- | --- | --- |
| 135M | 4,096 | 4 | run | run | `m1_smol_A` (11-06), `data_4k` (11-07) |
| 135M | 65,536 | 1 | run (two times) | run | the long-source mix `data_long_T` |
| 135M | 131,072 | 1 | run (two times) | run | the long-source mix `data_long_T` |
| 1.6B | 4,096 | 4 | run | run | `data_4k` |
| 1.6B | 65,536 | 1 | out of memory | run | the long-source mix `data_long_T` |

Lengths we did **not** test with CP: 8,192, 16,384 and 32,768. (They were tested on one GPU without CP on
2026-10-31, section 3.1.)

### 5.3 Settings used in every CP test

- 30 steps per run.
- Optimizer Muon; segmented embedding backward; ZeRO-1 with the reduce overlap on; bf16 logits; packed rows; gradient
  clipping; validation at step 0.
- Learning rate 1e-4 for the 4,096 runs and 5e-5 for the long runs.
- U = 2 on 2 GPUs in every CP run. So the CP group was the whole job: no data parallelism next to it.

---

## 6. The loss with CP, with numbers

**The rule we use:** each GPU sums the loss of the trained tokens in its slice and divides by the number of trained
tokens of the whole step. So every trained token has the same weight, as without CP.

**Example.** 2 GPUs, U = 2, one row of 8 tokens with 4 trained tokens:

- GPU 0's slice holds 1 trained token, with loss 2.0.
- GPU 1's slice holds 3 trained tokens, with losses 1.0, 1.0, 1.0.
- The true loss is the mean over all trained tokens: (2.0 + 1.0 + 1.0 + 1.0) ÷ 4 = **1.25**.

**Ours:**

1. Count: each GPU counts the row's trained tokens, 4. The sum over the 2 GPUs is 8. It is divided by U = 2, because
   both GPUs counted the same row. So N = 4.
2. Per GPU: GPU 0's loss = 2.0 ÷ 4 = 0.5. GPU 1's loss = 3.0 ÷ 4 = 0.75.
3. Logged loss: 0.5 + 0.75 = **1.25**, the true loss.
4. Gradients: each GPU's backward starts at 2 (the number of GPUs). The gradient reduce averages over the 2 GPUs. The
   two cancel, so the result is the gradient of 0.5 + 0.75 = the gradient of 1.25.

**The rule that would be wrong for SFT (mean per GPU, then the average):**

- GPU 0's mean is 2.0. GPU 1's mean is 1.0. The average is **1.5**, not 1.25.
- The GPU with 1 trained token would weigh as much as the GPU with 3.
- In pretraining every GPU has the same number of tokens, so there the two rules agree. In SFT they do not.

---

## 7. The experiments in order

| date (IST) | what | where | result |
| --- | --- | --- | --- |
| 2026-10-31 | study: can we call Adhitya's CP? | laptop (reading his code) | no, not as it is; use Ulysses with our kernels (section 3.2) |
| 2026-11-06 | first version built and tested | P032, GPUs 2 and 3 | forward exact; backward loses gradient (section 8) |
| 2026-11-07 morning | root cause found and fixed | laptop, RTX 3060 | the bug is in `autograd::contiguous` (section 8) |
| 2026-11-07 11:38-11:43 | fix checked on H200; CP-off reference runs at 64k and 128k | 1x H200 box | fix confirmed; reference numbers (section 9.1) |
| 2026-11-07 12:39-12:59 | CP = 2 against CP off at 4k, 64k, 128k | 2x H200 box | CP = 2 equals CP off (sections 9.2, 9.3) |
| 2026-11-07 13:52-13:57 | the 1.6B shape with CP at 4k and 64k | 2x H200 box | equal at 4k; 64k runs only with CP (section 9.5) |
| 2026-11-07 | the library fix pushed | GitHub | Tensor-Implementations `d59740b`, BluTrain `8086e78` |
| 2026-11-11 | the CP code and its docs pushed | GitHub | BluTrain `125ae9f` |

---

## 8. The first version and the gradient bug

### 8.1 The first test (2026-11-06, P032)

**Setup:** 135M `ckpt_A`, data `m1_smol_A`, B 4, 4 rows per step, row length 4,096. Four runs from the same weights
and the same rows; the values are from step 0.

| run | setup | step-0 validation loss | step-0 gradient norm |
| --- | --- | --- | --- |
| A | CP off, 1 GPU | 1.424576 | 2.8002 |
| C | 2 GPUs, data parallel, no CP | 1.424576 | 2.8003 |
| B | CP = 2 | 1.424576 | 2.5912 |
| D | CP = 2, reduce overlap off | 1.424576 | 2.5917 |

**What it showed:**

- The validation loss is the same in all four. So the forward pass with CP is exact.
- The gradient norm with CP is 7.5% lower, and it is the same in runs B and D. So the backward pass lost a fixed part
  of the gradient.
- By parameter group: the group with the weight matrices was 7.8% low, and the group with the norm gains (gammas) was
  36% low.

**What we ruled out, each by a test:**

| suspect | test | result |
| --- | --- | --- |
| our attention kernel at the smaller head counts (2 q / 1 kv) | `test_varlen_attention` with heads 2/1 and 4/2 | 515 cases pass |
| our RoPE backward on a slice that starts in the middle of a chat | `test_rope_slice.cu` | dq and dk bit-identical, gamma sums equal |
| the gradient-reduce overlap | run D | same result with it off |
| the autograd engine's dependency counting | read in full | correct |
| the streams | read in full | correct |

### 8.2 The root cause (2026-11-07)

- Our fused QK-norm + RoPE operation has two outputs: q (output number 0) and k (output number 1). Both are already
  contiguous in memory.
- The CP code called `autograd::contiguous(k)` before the exchange.
- For a tensor that is already contiguous, the tensor function returns the same object, with no copy. That part is
  correct.
- But the autograd wrapper then attached a new "contiguous" backward node onto that same object, and kept its output
  number 1.
- That node has only one input slot, number 0. The engine delivered k's gradient to slot 1. The node found slot 0
  empty and returned nothing.
- **Result:** the gradient of k and the gradient of the k-norm gain were dropped in every layer. That matches the
  pattern of section 8.1 exactly: the gains group and the matrix group were the ones that lost gradient.

**Why Adhitya's CP runs never hit it:** his q / k / v come from a transpose, which is output number 0 and not
contiguous. So `autograd::contiguous` makes a real copy with a proper node.

### 8.3 The fix

In `ReshapeOps.cpp` of the tensor library: when the input is already contiguous and starts at offset 0,
`autograd::contiguous` now returns the input as it is, without a new node. This is the same rule as our
`autograd::to_dtype` and as PyTorch's `contiguous()`. The old code is kept as a comment.

**Checks of the fix:**

| check | where | result |
| --- | --- | --- |
| new regression test `test_contig_alias.cu`: gradient straight on q / k (A) against through `autograd::contiguous` (B) | RTX 3060 and H200 | without the fix B gives k = 0 and k gain = 0; with the fix A = B (k 156.991745, k gain 350.833604) |
| the library's autocheck | RTX 3060 and H200 | 43 of 43 pass |
| every other caller of `autograd::contiguous` | read | each passes a transpose view: unchanged |
| does the fix change normal SFT (CP off)? 30 steps, original library two times, fixed library once | 1x H200 | no effect (table below) |

Normal SFT with and without the fix (135M `ckpt_A`, `data_4k`, B 4 × 2 micro-batches; the largest difference over the
30 steps):

| comparison | train loss | gradient norm |
| --- | --- | --- |
| original against original (two identical runs) | 0.000630 | 0.0292 |
| fixed against original, run 1 | 0.000598 | 0.0108 |
| fixed against original, run 2 | 0.000604 | 0.0184 |

The step-0 train loss is 1.530339 and the peak memory 22,857 MiB in all three. The fix changes nothing when CP is off.

---

## 9. The results

**How to read the tables.** Every comparison is CP off against CP = 2 from the same weights and the same rows. Two
identical runs never agree to the last digit on a GPU (some kernels add numbers in a different order each run). So for
every case we also ran CP off two times, and the difference between those two is the "normal difference". CP = 2 is
correct when it is as close to CP off as two CP-off runs are to each other.

### 9.1 CP-off reference runs on the 1x H200 box (2026-11-07, 11:38-11:43)

**Setup:** 135M `ckpt_A`, the long-source mix, B 1, 1 row per step, 30 steps, fixed library.

| row length | step-0 validation loss | step-0 train loss | step-0 gradient norm | peak memory |
| --- | --- | --- | --- | --- |
| 65,536 | 3.216672 | 7.105492 | 398.4949 | 60,913 MiB |
| 131,072 | 3.216672 | 4.334521 | 34.7171 | 110,361 MiB |

The 131,072 train loss equals the 2026-10-31 one-GPU run exactly.

### 9.2 135M at 4,096 tokens (2x H200 box, 2026-11-07)

**Setup:** 135M `ckpt_A`, `data_4k` (a 480-row training set), B 4, 4 rows per step, learning rate 1e-4, 30 steps.
The step-0 validation loss is 1.424576 in all five runs.

| run | step-0 gradient norm | against CP off | largest train-loss difference over the run | step time | peak memory per GPU |
| --- | --- | --- | --- | --- | --- |
| A: CP off, GPU 0 | 3.0473 | reference | reference | 55.9 ms | 23.4 GB |
| A2: CP off, GPU 1 (the same run again) | 3.0473 | +0.000% | 0.000813 | 55.7 ms | 23.4 GB |
| C: 2 GPUs, data parallel | 3.0471 | −0.007% | 0.001163 | 38.4 ms | 21.1 GB |
| B: CP = 2, with the fix | 3.0473 | +0.000% | 0.000997 | 49.7 ms | 21.6 GB |
| Bn: CP = 2, without the fix (the control) | 2.8367 | −6.9% | 0.026596 | 50.2 ms | 21.6 GB |

- With the fix, CP = 2 equals CP off.
- The control run Bn shows the bug on the same box, the same day: 6.9% low without the fix.
- The reference here is 3.0473 and not the 2.8002 of section 8.1, because `data_4k`'s first 4 rows are other rows.

### 9.3 135M at 65,536 and 131,072 tokens (2x H200 box, 2026-11-07)

**Setup:** 135M `ckpt_A`, the long-source mix, B 1, 1 row per step, learning rate 5e-5, 30 steps. The step-0
validation loss is 3.216672 in all six runs.

| row length | run | step-0 gradient norm | against CP off run 1 | largest train-loss difference | step time | peak memory per GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 65,536 | CP off, GPU 0 | 387.05 | reference | reference | 250.7 ms | 61.5 GB |
| 65,536 | CP off, GPU 1 | 389.22 | +0.56% | 0.0165 | 247.9 ms | 61.5 GB |
| 65,536 | CP = 2 | 386.98 | −0.02% | 0.0083 | 158.2 ms | 36.3 GB |
| 131,072 | CP off, GPU 0 | 34.6985 | reference | reference | 492.9 ms | 111.0 GB |
| 131,072 | CP off, GPU 1 | 34.7813 | +0.24% | 0.0655 | 489.3 ms | 111.0 GB |
| 131,072 | CP = 2 | 34.7619 | +0.18% | 0.0467 | 297.9 ms | 62.2 GB |

- At both lengths CP = 2 is closer to CP off than the second CP-off run is.
- The step-0 train loss is the same too (7.105492 / 7.105493 at 65,536; 4.334521 at 131,072).

### 9.4 Speed and memory, 135M, CP off (1 GPU) against CP = 2 (2 GPUs)

| row length | step time, CP off | step time, CP = 2 | faster by | memory per GPU, CP off | memory per GPU, CP = 2 | CP = 2 as a share of CP off |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 | 55.9 ms | 49.7 ms | 1.12 times | 23.4 GB | 21.6 GB | 92% |
| 65,536 | 250.7 ms | 158.2 ms | 1.58 times | 61.5 GB | 36.3 GB | 59% |
| 131,072 | 492.9 ms | 297.9 ms | 1.65 times | 111.0 GB | 62.2 GB | 56% |

- The gain grows with the row length.
- CP = 2 at 131,072 tokens needs about the same memory per GPU as CP off at 65,536 (62.2 against 61.5 GB). That was
  predicted before the run (about 61 GB).
- At 4,096 tokens CP is slower than data parallelism on the same 2 GPUs (49.7 against 38.4 ms). CP is for long rows;
  at short rows, data parallelism is the right choice.
- The memory numbers include a 591 MiB holder process that kept the GPUs for us on the shared box.

### 9.5 The 1.6B shape (2x H200 box, 2026-11-07, 13:52-13:57)

**Setup:** 26 layers, d_model 2,304, 18 q / 6 kv heads (with CP = 2: 9 / 3 per GPU), ffn 6,144, vocab 49,280, random
init, 30 steps.

| run | step-0 validation loss | step-0 train loss | step-0 gradient norm | step time | peak memory per GPU |
| --- | --- | --- | --- | --- | --- |
| 4,096, B 4, CP off, 1 GPU | 11.232169 | 11.240100 | 41.0146 | 480.0 ms | 66.7 GB |
| 4,096, B 4, CP = 2 | 11.232170 | 11.240099 | 41.0215 (+0.017%) | 274.5 ms | 42.4 GB |
| 65,536, B 1, CP = 2 | 11.233593 | 11.194443 | 45.7685 | 861 ms | 96.6 GB |
| 65,536, B 1, CP off, 1 GPU | out of memory at 143.1 GB | | | | |

- At 4,096 tokens, CP = 2 gives the same training as one GPU: the train loss differs by at most 0.000737 over the 30
  steps. It is 1.75 times faster and uses 64% of the memory per GPU.
- At 65,536 tokens, one GPU runs out of memory and CP = 2 runs. So for the 1.6B, CP is what makes a 65,536-token row
  possible on H200.
- For the 65,536 case there is no CP-off number to compare with, because CP off does not fit. What was checked there:
  it runs 30 steps with no error.
- The losses near 11.2 are expected for random weights.

### 9.6 All CP cases in one table

| case | step-0 validation loss | step-0 gradient norm, CP = 2 against off | normal difference (two CP-off runs) | step time | memory per GPU |
| --- | --- | --- | --- | --- | --- |
| 135M, 4,096, B 4 | equal | +0.000% | 0.000% | 55.9 → 49.7 ms | 23.4 → 21.6 GB |
| 135M, 65,536, B 1 | equal | −0.02% | 0.56% | 250.7 → 158.2 ms | 61.5 → 36.3 GB |
| 135M, 131,072, B 1 | equal | +0.18% | 0.24% | 492.9 → 297.9 ms | 111.0 → 62.2 GB |
| 1.6B, 4,096, B 4 (random init) | equal | +0.017% | not measured | 480.0 → 274.5 ms | 66.7 → 42.4 GB |
| 1.6B, 65,536, B 1 | one GPU out of memory | none | none | 861 ms | 96.6 GB |
| control: 135M, 4,096, CP = 2 without the fix | equal | −6.9% | | | |

---

## 10. Other things the experiments showed

### 10.1 Loss spikes on long rows come from the checkpoint, not from CP

In the 65,536 and 131,072 runs some steps had a loss of 5 to 9 and a gradient norm up to 1,117. This happens with CP
off too. The reason:

- The 135M checkpoint was pretrained at context length 4,096. It has never seen a position past 4,096 inside a chat.
- Steps where no trained token sits past position 4,096 in its chat: loss 1.63 to 2.43.
- Steps where more than 20% of the trained tokens sit past position 4,096: loss 2.4 to 9.3.
- Fitted over the rows: loss about 1.70 before position 4,096 and about 6.9 after it.

So this is no problem for comparing CP = 2 with CP off (same rows in both). But real long-chat SFT needs a checkpoint
that was pretrained on long context first.

### 10.2 Two identical 131,072 runs differ by 0.24% at step 0

On 2026-11-07 the 131,072 step-0 gradient norm was 0.24% away from the 2026-10-31 run. The two CP-off runs on the 2x
box also differ by 0.24%. So that was the normal difference between two runs, not a change in the code.

### 10.3 The same rows on another H200

Against the 1x box's runs: at 131,072 the step-0 gradient norm differs by −0.05% and +0.13%; at 65,536 by −2.9%. On the
65,536 step-0 row, 91% of the trained tokens sit past position 4,096 (loss 7.1), so the gradients are very large and
sensitive. The forward values are identical everywhere.

---

## 11. The boxes we used

| box | GPUs | date | what ran | notes |
| --- | --- | --- | --- | --- |
| P032 | H200, GPUs 2 and 3 | 2026-11-06 | first CP version, the 4 runs of section 8.1 | |
| laptop | RTX 3060 | 2026-11-07 | the root-cause analysis and the regression test | |
| 1x H200 box (86.38.238.69) | 1 | 2026-11-07 | fix check and the CP-off references | closed by the user after the runs |
| 2x H200 box (86.38.238.112) | 2 | 2026-11-07 | all CP = 2 comparisons, 135M and 1.6B | shared with colleagues; we waited until both GPUs were free, held them with small holder processes, and released them after |

On the 2x box nothing was installed system-wide; `yaml-cpp` was unpacked into our own folder.

---

## 12. What is not tested yet

Listed on 2026-11-11, after the user asked whether CP is tested from every side. The answer was no: CP = 2 is proven
on the cases above, and these are open.

| # | case | why it matters | priority |
| --- | --- | --- | --- |
| 1 | CP together with data parallelism (for example 4 GPUs = 2 CP groups) | the loader split, the token count ÷ U and the gradient sum over more GPUs have only run with one group | high |
| 2 | checkpoint save and resume under CP | the save interval is 1,000 steps and every test ran 30 | high |
| 3 | gradient accumulation above 1 | every test used 1 micro-batch per step | medium |
| 4 | a long run (hundreds of steps) | every test ran 30 steps | medium |
| 5 | U = 3 or 6 on the 1.6B | only U = 2 was run | medium |
| 6 | the overlap-refresh option on | not run with CP | low to medium |
| 7 | validation in the middle of a run | only step-0 validation was run | low |
| 8 | AdamW | every test used Muon | low |
| 9 | a GPU whose slice has 0 trained tokens | not hit by the test rows | low |
| 10 | GPUs other than H200 | not run | unknown |

Also not tested with CP: the row lengths 8,192, 16,384 and 32,768, and a trained 1.6B checkpoint (the 1.6B tests used
random weights).

**Proposed before a real long-context SFT run:** cases 1 to 3, about 25 minutes on 4 GPUs.

---

## 13. Limits that are part of the design

- U must divide the kv heads. So the 135M allows only U = 2, and U = 4 is not possible for either model.
- CP cannot be used together with the trained-rows switch.
- At short rows (4,096) CP is slower than data parallelism.
- CP needs Tensor-Implementations `d59740b` or later. Before it, the gradient of k and of the k-norm gain is dropped
  under CP.

---

## 14. How to run it

    SFT_CP=2 NP=2 GPUS=0,1 ... bash sft/run_sft.sh

- `SFT_CP` sets `BLU_SFT_CP` for the trainer. `NP` is the number of GPUs and must be a multiple of `SFT_CP`.
- The other settings are the same as for normal SFT (`BluTrain/sft/README.md`, section 8).
- Rows per step = B × GRAD_ACCUM × (NP ÷ SFT_CP).

---

## 15. Where everything is

**Code (BluTrain, branch `dist_side_gautam_lottn`):**

| file | what |
| --- | --- |
| `train/production_zero/bluscript_sft.cpp` | the CP code, behind `BLU_SFT_CP` (default 1 = off): the group setup, the slicing, the two exchange wrappers, the loss count, the checks. Every old line is kept as a comment. |
| `Makefile` | the include path of `dist/Context_Parallelism` |
| `sft/run_sft.sh` | `SFT_CP` |
| `dist/Context_Parallelism` | Adhitya's module, a submodule pinned at `cf1372d` |
| `sft/README.md` section 5.2 | the user guide for CP |

**Commits:**

| repository | commit | what |
| --- | --- | --- |
| Tensor-Implementations | `d59740b` (2026-11-07) | the `autograd::contiguous` fix and the regression test `test_contig_alias.cu` |
| BluTrain | `8086e78` (2026-11-07) | points to the fixed library |
| BluTrain | `125ae9f` (2026-11-11) | the CP code, README section 5.2 and the docs |

All three were pushed by the user and checked on GitHub file by file. The branch `preproduction-135M` does not have
them yet.

**Records (laptop, not in git):**

| folder | what |
| --- | --- |
| `profiling_reports/cp_2026_11_06/` | the first tests on P032, `test_rope_slice.cu`, the analysis of Adhitya's module (`cp_module_analysis.md`), `test_cp_exchange.cpp` |
| `profiling_reports/cp_2026_11_07_1x/` | the 1x H200 box: scripts and 55 result files |
| `profiling_reports/cp_2026_11_07_2x/` | the 2x H200 box: scripts, 116 result files, the summary `box_results/results_2x.txt` |

**Other documents:**

| file | what |
| --- | --- |
| `cp_sft_design.md` / `.docx` | our design side by side with Adhitya's, written for his review, with 6 questions |
| `cp_ulysses_issues_for_adhitya.md` / `.docx` | 3 issues we found in his module (all in code our trainer does not call) |
| `sft_work_documentation.md` 45.127, 45.177 to 45.186, 45.192 | the dated work log |

---

## 16. Open points with Adhitya

- **His answer to our question about kv heads (2026-11-07):** "Dividing kv heads is not available in Ulysses; use the
  ring + Ulysses hybrid for that."
- **Our follow-up question, not answered yet:** his code does copy kv heads when there are fewer than U
  (`forward_ulysses_gqa`). Is that path fine for U larger than the kv heads, or is only the hybrid recommended?
- **His fixes from our issues document** were not pushed when we last looked (his repository had one branch at
  `73b3bdd`). When he pushes, we compare `73b3bdd` with the new commit. The 3 issues are in code our trainer does not
  call.
- **For SFT, the ring part of his hybrid** would need per-chat positions. His ring has a document mask but does not
  restart the positions per chat.
