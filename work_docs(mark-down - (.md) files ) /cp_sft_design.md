# Long-context SFT with context parallelism: how I use your Ulysses code (for your review)

I added context parallelism (CP) to my SFT trainer, `train/production_zero/bluscript_sft.cpp`. One long packed row (64k-128k tokens) can now be split across several GPUs.

**From your module:** I use your Ulysses all-to-all functions exactly as they are, from the Context_Parallelism submodule. BluTrain pins `cf1372d`; the CP code is identical in your `main`, `73b3bdd`.

**From my own SFT code:** my kernels stay as they are:
- QK-norm + RoPE with per-chat positions;
- the packed attention with the chat plan.

**What changed:**
- No kernel file was changed.
- The changes are in the training script, one include path in the Makefile and one line in the run script.
- All of it is behind a switch, `BLU_SFT_CP=U`. The default is 1 (off), and the trainer then runs exactly as before.
- While testing I found one bug in Tensor-Implementations' autograd (section 8). It is fixed there with one line.

I would like you to check the points in section 9. A separate document lists three problems I found in your module
(`cp_ulysses_issues_for_adhitya`).

## 1. The words I use

| word | meaning |
| --- | --- |
| row | one packed training sequence of T tokens; it holds several whole chats |
| chat | one conversation inside a row; its RoPE positions start at 0, and its tokens attend only to tokens of the same chat |
| U (CP size) | how many GPUs share one row |
| CP group | the U GPUs that share a row; each holds T/U of its tokens |
| slice | the T/U consecutive tokens of a row that one GPU holds: GPU r of a group holds tokens r·T/U to (r+1)·T/U - 1 |
| DP size | how many CP groups there are; each group reads different rows |
| world size | all GPUs = DP size × U |
| micro-batch | B rows processed together; several micro-batches make one optimizer step |
| ZeRO | the optimizer that splits the optimizer state over all GPUs and averages the gradients over all GPUs |

**Example:** 8 GPUs, U = 2, so the DP size is 4.
- The CP groups are GPUs {0,1}, {2,3}, {4,5} and {6,7}.
- GPUs 0 and 1 read the same rows: GPU 0 holds the first half of each row's tokens, GPU 1 the second half.
- GPUs 0 and 2 read different rows.

## 2. What I use from your module (unchanged)

| piece | file : lines | what I use it for |
| --- | --- | --- |
| `cp::ulysses_combine` | `UlyssesAttention.h:43-68` | Q, K, V before attention: [B, H, T/U, D] -> [B, H/U, T, D] |
| `cp::ulysses_partition` | `UlyssesAttention.h:72-97` | the attention output back to my slice: [B, Hq/U, T, D] -> [B, Hq, T/U, D] |
| `UlyssesCombineBackward`, `UlyssesPartitionBackward` | `HybridUlyssesBackward.h` | the backward of the two exchanges (partition and combine of the gradient) |
| the 12-line autograd hookup (`combine_ag` / `partition_ag`) | `ContextParallel.h:1474-1497` | the same lines, around your public functions and nodes (the two methods are private members of your class) |
| the CP-group recipe: its own communicator + a blocking stream | `device_mesh.cpp:~170-191` | I build my group the same way (ncclCommSplit + `cudaStreamCreate`), so your all-to-all runs after my kernels |

**Not used:**
- `forward_cp` and your attention kernels: RoPE positions 0..T-1 of the whole row, and no chat masking;
- the ring and the hybrid;
- `ulysses_gather_seq`;
- `head_repeat_interleave` / `head_group_reduce` (only needed when there are more GPUs than KV heads);
- your trainer.

## 3. What I changed

| repository · file | change |
| --- | --- |
| BluTrain · `Makefile` | `TRAIN_INCLUDES` gets `-I$(DIST_DIR)/Context_Parallelism`, last, so it never hides a header of the same name |
| BluTrain · `sft/run_sft.sh` | passes `SFT_CP` to the trainer as `BLU_SFT_CP` |
| BluTrain · `train/production_zero/bluscript_sft.cpp` | everything below, behind `BLU_SFT_CP=U` (1 = off) |
| Tensor-Implementations · `src/autograd/operations/ReshapeOps.cpp` | `autograd::contiguous` returns its input when the input is already contiguous (found during this work, section 8), plus the test `Tests/kernel_tests/test_contig_alias.cu` |
| any kernel file | **none** |

**In `bluscript_sft.cpp`:**
1. **Checks:** U must divide the world size, the query heads, the KV heads and T. The trained-rows switch is refused with CP.
2. **CP group:** U consecutive ranks. It is split off the world NCCL group (`ncclCommSplit`, colour rank / U, key rank % U)
   and gets its own blocking stream, as in your DeviceMesh.
3. **Rows:** picked by the data-parallel rank (rank / U of world / U), so the U GPUs of a group get the same rows. This is
   your `loader_rank = dp_rank`.
4. **Slicing:** each GPU keeps its slice of the token ids, the labels and the per-chat RoPE positions.
5. **Attention plan:** the chat plan is built for the **whole row**, for Nq/U heads.
6. **Attention block:** my QK-norm + RoPE on the slice -> `cp_combine` (Q, K, V) -> my packed attention for Nq/U query and
   Nkv/U KV heads -> `cp_partition`.
7. **Trained-token count of a step:** summed over all GPUs as before, then divided by U, because the U GPUs of a group
   counted the same rows. The same rule applies to validation.
8. **Embedding backward:** the segmented kernel gets B × T/U tokens per call. Its workspace is sized for exactly that.
9. **Logging:** tokens per step and tokens/s count world / U GPUs' rows.

## 4. Mind maps

### 4.1 The pieces

```
Long-context SFT with context parallelism (BLU_SFT_CP = U)
|
+-- Data  (unchanged)
|     +-- prepare_sft_data.py: packed rows of T tokens (T must divide by U)
|     +-- per row: token ids, labels (prompt tokens ignored), chat lengths
|     +-- per micro-batch, built for the WHOLE row: per-chat RoPE positions + chat plan
|
+-- GPUs
|     +-- world = DP size x U
|     +-- CP group = U consecutive GPUs: same rows, one slice each
|     +-- DP: different CP groups read different rows
|
+-- One layer (per GPU)
|     +-- RMSNorm, Q / K / V projections ............ my slice            (unchanged)
|     +-- my QK-norm + RoPE (fused_rms_rope_bf16) .... my slice, true per-chat positions  (unchanged kernel)
|     +-- your ulysses_combine ....................... -> whole row, my heads
|     +-- my packed attention + chat plan ............ whole row, Nq/U heads  (unchanged kernel)
|     +-- your ulysses_partition ..................... -> my slice
|     +-- output projection, RMSNorm, MLP ............ my slice            (unchanged)
|
+-- Loss: sum over my slice's trained tokens / trained tokens of the whole step (all GPUs, / U)
+-- Backward: the same steps in reverse; your backward nodes for the two exchanges
+-- Optimizer: ZeRO over all GPUs (DP x CP)  (unchanged)
```

### 4.2 One training step: normal SFT vs SFT with CP

```
NORMAL SFT (U = 1)                                   SFT WITH CP (U GPUs per row)
------------------------------------------------     -------------------------------------------------------
loader: my rows (by rank)                            loader: my group's rows (by rank / U)
count: trained tokens of the step (sum over GPUs)    count: same sum, / U
plan + positions for my rows                         plan (Nq/U heads) + positions for the whole rows
                                                      cut ids, labels, positions to my slice (T/U tokens)
embedding (T tokens per row)                         embedding (T/U tokens per row)
for each layer:                                      for each layer:
  norm, Q/K/V                                          norm, Q/K/V                      (my slice)
  QK-norm + RoPE (per-chat positions)                  QK-norm + RoPE (my slice's positions)
                                                        combine Q, K, V   -> whole row, my heads
  attention (all heads, chat plan)                     attention (Nq/U heads, chat plan)
                                                        partition         -> my slice
  output projection, norm, MLP                         output projection, norm, MLP     (my slice)
lm_head + loss on my rows                            lm_head + loss on my slice
backward (seed = world size)                         backward (seed = world size; exchanges reversed)
ZeRO: average gradients over all GPUs                ZeRO: average gradients over all GPUs
optimizer step                                       optimizer step
```

## 5. One layer: your fused Ulysses path and mine

```
Your fused Ulysses path (forward_ulysses_fused)
1. Q, K, V projections            on my token slice
2. ulysses_combine                -> whole row, my heads
3. ONE kernel: QK-norm + RoPE + attention        (RoPE positions 0..T-1 of the whole row)
4. ulysses_partition              -> my token slice
```

```
Mine
1. Q, K, V projections            on my token slice
2. fused_rms_rope_bf16            on my token slice, all heads: QK-norm + RoPE in ONE kernel,
                                  with each token's true position inside its chat
                                  -> two outputs: q_bf (output 0) and k_bf (output 1)
3. your ulysses_combine           -> whole row, my heads
4. my packed attention kernel     attention only, with the chat plan of the whole row
5. your ulysses_partition         -> my token slice
```

**Why mine puts QK-norm + RoPE before the exchange:**
- Packed SFT needs each chat's positions to start at 0. My RoPE op takes one position per token.
- QK-norm and RoPE are done per token and per head, so doing them before the exchange gives the same numbers and the same
  amount of work as doing them after.
- This is also the order my trainer already uses without CP.

## 6. Every other part, side by side

**Rows and GPU groups: the same idea**
```
Yours:  DeviceMesh {dp, cp}; loader_rank = dp_rank      (bluscriptCP.cpp:1711, 2062)
Mine:   CP group = U consecutive ranks; loader rank = rank / U, loader world = world / U
        -> in both, the GPUs of one CP group read the same rows
```

**Cutting the row: the same idea**
```
Yours:  token ids cut before the embedding (contiguous slices for Ulysses)   (bluscriptCP.cpp:744-750)
Mine:   token ids, labels and per-chat positions cut to my slice before the embedding
```

**Positions and masking: different (SFT needs chats)**
```
Yours (Ulysses):  positions 0..T-1 of the whole row, inside the attention kernel; causal over the whole row;
                  document mask only in the ring path, without restarting positions
Mine:             per-chat positions (restart at 0 in every chat), built on the CPU for the whole row, cut to my slice;
                  chat plan of the whole row: a token attends only to earlier tokens of its own chat
```

**Loss: different (SFT trains only answer tokens)**
```
Yours:  loss of a GPU = MEAN over its own tokens          (bluscriptCP.cpp:3354)
        backward seed 1 / GRAD_ACCUM; ZeRO averages over all GPUs
        logged loss = average of the GPUs' means          (bluscriptCP.cpp:3397-3404)
        -> right for pretraining: every GPU has the same number of tokens
Mine:   trained-token count N of the step = sum over all GPUs, / U   (computed before the step)
        loss of a GPU = SUM over its slice's trained tokens / N
        backward seed = world size; ZeRO averages over all GPUs (/ world size)  -> a plain sum
        logged loss = SUM of the GPUs' losses
        -> every trained token has the same weight, wherever it lands
```

**QK-norm gamma gradient: different, because QK-norm sits on a different side of the exchange**
```
Yours:  QK-norm runs AFTER the exchange (inside the attention kernel): a GPU sees only its heads, over all tokens.
        gamma is shared by all heads, so the gradient is summed over the CP GPUs. ZeRO then averages over the same
        GPUs, so you pre-divide: all-reduce with op_t::avg     (UlyssesFusedGQAAttentionBackward.h:129-161)
Mine:   QK-norm runs BEFORE the exchange, on my slice: a GPU sees all heads, over its own tokens.
        That is an ordinary per-token partial sum, like every other weight: ZeRO adds it up.
        -> no extra all-reduce and no pre-division
```

**KV heads: different**
```
Yours:  (nkv % U == 0) or (U % nkv == 0); KV heads copied when nkv < U     (DeepSpeed rule)
Mine:   U must divide the query heads AND the KV heads; no copying yet
        135M (4 query / 2 KV heads): U = 2;  1.6B (18 / 6): U = 2, 3 or 6
```

**Communication group and stream: same behaviour, built differently**
```
Yours:  CPProcessGroupNCCL per mesh axis, new ncclUniqueId per axis, its own blocking stream  (device_mesh.cpp:184)
Mine:   ncclCommSplit of the world NCCL group + its own blocking stream (cudaStreamCreate)
        -> in both, the all-to-all's stream waits for the kernels on stream 0 before it starts
```

**Backward hookup: your hybrid pattern**
```
Yours (MHA path):     ONE node for combine + attention + partition    (UlyssesAttentionBackward.h)
Yours (hybrid path):  one node per exchange: combine_ag / partition_ag
Mine:                 one node per exchange (your two nodes); my attention keeps its own backward node in between
```

**Data types**
```
Yours (fused):  q / k / v enter forward_cp as fp32 (to_master, bluscriptCP.cpp:361-363, 497-499); the fused kernel works
                in bf16 inside; its output is converted back to fp32 before the partition (ContextParallel.h:1948)
                -> the exchanges carry fp32
Mine:           bf16 end to end: Q, K, V, the attention output and their gradients go through the exchanges as bf16
                (half the bytes per exchange)
```

## 7. Loss and gradients with numbers

**Setup:**
- 2 GPUs, U = 2, so the DP size is 1.
- One row of 8 tokens. Its trained (answer) tokens: GPU 0's slice holds 1, with loss 2.0. GPU 1's slice holds 3, with
  losses 1.0, 1.0, 1.0.
- The true loss is the mean over all trained tokens: (2.0 + 1.0 + 1.0 + 1.0) / 4 = **1.25**.

**Mine:**
1. **Count:** each GPU counts the row's trained tokens, 4. The sum over the 2 GPUs is 8, divided by U = 2, so N = 4.
2. **Per GPU:** GPU 0's loss = 2.0 / 4 = 0.5. GPU 1's loss = 3.0 / 4 = 0.75.
3. **Logged:** 0.5 + 0.75 = **1.25**, the true loss.
4. **Gradients:**
   - each GPU's backward starts at 2 (the world size), so each GPU's gradient is 2 × the gradient of its own loss;
   - ZeRO averages over the 2 GPUs, so the result is the gradient of 0.5 + 0.75 = the gradient of 1.25.

**Mean per GPU, then the average (your trainer):**
- GPU 0's mean is 2.0. GPU 1's mean is 1.0.
- The average is **1.5**, not 1.25: the GPU with 1 trained token weighs as much as the GPU with 3.
- In pretraining every GPU has the same number of tokens, so the two rules agree there.

**Where "divide by the number of GPUs" happens in mine:**
- ZeRO's average over all GPUs (÷ world size) is cancelled by the backward seed (× world size).
- The count is divided by U, because the U GPUs of a group counted the same rows.
- Nothing else is divided.

## 8. Tests so far

**Setup:**
- 135M model from the 900B checkpoint, my SFT data, rows of 4,096 tokens, B = 4.
- Step 0 of runs on the same rows; H200, P032.

| run | validation loss | gradient norm |
| --- | --- | --- |
| no CP, 1 GPU | 1.424576 | 2.8002 |
| no CP, 2 GPUs (data parallel) | 1.424576 | 2.8003 |
| CP = 2 | 1.424576 | 2.5912 |
| CP = 2, ZeRO overlap off | 1.424576 | 2.5917 |

**The forward is exact.** The backward was low. Ruled out one by one:
- my attention kernel at 2 query / 1 KV head per GPU: the varlen attention test, 515 / 515 pass;
- my RoPE backward on a slice that starts in the middle of a chat: bit-identical to the full row;
- the 2-GPU path without CP: equal to 1 GPU;
- the ZeRO overlap.

**Cause: Tensor-Implementations' `autograd::contiguous`.**
1. My CP code passes `k_bf` through `autograd::contiguous` before `ulysses_combine`. `k_bf` is output 1 of my RoPE op
   (`NormalizationOps.cpp`, `set_output_nr(1)`).
2. `Tensor::contiguous()` returns the same object when the tensor is already contiguous (`Tensor.cpp:297-301`).
3. `make_unary_op` then stamped its new `ContiguousBackward` node onto that object, which is the input itself, and the
   input kept output number 1.
4. So the next link pointed to slot 1 of a node that has only slot 0. `ContiguousBackward::apply` returns nothing when
   slot 0 is empty. The K gradient was dropped in every layer, with no error.

**Fix:** `autograd::contiguous` now returns an already-contiguous input as it is, with no new node. This is the rule
`to_dtype` already uses, and PyTorch's.

**Checked on an RTX 3060** with the real RoPE op:

| gradient through `autograd::contiguous(k_bf)` | before the fix | after the fix |
| --- | --- | --- |
| k gradient norm | 0 | 156.991745 (= without the call) |
| k_gamma gradient norm | 0 | 350.833604 (= without the call) |

- The Tensor-Implementations gradient checker passes 43 / 43.
- **Your code does not hit this.** Your q / k / v come from `autograd::transpose` (`bluscriptCP.cpp:502-504`): output 0,
  and not contiguous, so `autograd::contiguous` makes real copies.
- **Still to run:** CP = 2 again on 2 GPUs. The gradient norm must be 2.8002. Then rows of 64k-128k: 135M with CP = 2 vs
  CP = 1, and 1.6B at 64k with CP = 2.

## 9. What I would like you to check

1. **The exchanges:** I call `ulysses_combine` / `ulysses_partition` with contiguous [B, H, T/U, D] bf16 tensors, on my own
   group (ncclCommSplit + a blocking stream). Is there anything else your functions expect?
2. **The hookup:** one backward node per exchange (your hybrid pattern) with my attention node in between, instead of one
   node for all three as in your MHA path. Is there a reason you used one node there that I should know about?
3. **Loss and gradients (section 7):** the trained-token sum / the global count, the seed = world size, the ZeRO average.
4. **The gamma gradient (section 6):** no extra reduce, because my QK-norm runs before the exchange.
5. **KV heads:** I require U to divide the KV heads. For more GPUs than KV heads (e.g. 135M with U = 4), I would use your
   replication. Is `head_repeat_interleave` the path you would recommend, or something else?
6. **Anything in your module** I should use instead for H200 (for example the cplocal path) or should not skip.
