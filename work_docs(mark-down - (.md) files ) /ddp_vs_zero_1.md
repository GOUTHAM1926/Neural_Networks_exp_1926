# DDP vs ZeRO-1, and the other kinds of parallelism

**Date:** 2026-09-29
**Why this exists:** while planning SFT task 1 (loading Navin's 4-GPU checkpoint on 1 GPU), the questions "how do 4 GPUs
train one model?", "what is the difference between DDP and ZeRO-1?" and "what gets sent between GPUs?" came up. This file
collects every question and every answer, with the real numbers of our model.

Our model in all examples: the 1.7B trainer `bluscript_zero_ckpt.cpp` on `dist_side_merge`, **1,588,229,888 weights**
(written "1.59B" below).

---

## 1. The words used in this file

| word | meaning |
|---|---|
| **GPU** | one graphics card. An H200 has 141 GB of memory |
| **micro-batch** | the text one GPU processes in one forward + backward pass: B rows × T tokens (e.g. 4 × 4,096 = 16,384 tokens) |
| **micro-step** | one forward + backward pass on one micro-batch |
| **gradient** | for every weight, how much the loss would change if that weight changed a little. Backward computes it |
| **gradient buffer** | where the gradients of several micro-steps are added up before the optimizer uses them |
| **gradient accumulation (GRAD_ACCUM)** | how many micro-steps are added up before one weight update |
| **optimizer step** | one update of all weights, using the added-up gradient. One printed `step` line in the training log is one optimizer step |
| **bf16 weight** | the working copy of a weight, 2 bytes. Forward and backward use it |
| **master weight** | the precise copy of the same weight, fp32, 4 bytes. The optimizer updates this one, then rounds it into the bf16 working copy |
| **m and v** | Adam's two running averages for every weight, fp32, 4 + 4 bytes |
| **optimizer state** | master weight + m + v together: 12 bytes per weight |

**GRAD_ACCUM** = global batch ÷ (B × T × number of GPUs). One printed step = GRAD_ACCUM micro-steps + 1 update.

| setup | GRAD_ACCUM | per printed step |
|---|---|---|
| our runs: 1 H200, B = 4, T = 4096, global batch 1,048,576 | 1,048,576 ÷ (4 × 4096 × 1) = **64** | 64 micro-steps + 1 update |
| Navin's pretraining: 4 GPUs, B = 8 (the trainer's default) | 1,048,576 ÷ (8 × 4096 × 4) = **8** | 8 micro-steps per GPU + 1 update |
| production: 8 GPUs, B = 8 | 1,048,576 ÷ (8 × 4096 × 8) = **4** | 4 micro-steps per GPU + 1 update |

`BLU_MAX_STEPS = 20` means **20 updates**. With GRAD_ACCUM = 64 that is 20 × 64 = 1,280 forward + backward passes.

---

## 2. Why use more than one GPU

There are only two reasons:

1. **Speed.** 4 GPUs process 4 times more text per second. Every GPU holds the whole model and works on different text.
   This is **data parallelism** (DDP, ZeRO-1, ZeRO-2, ZeRO-3 / FSDP are all versions of it).
2. **The model does not fit on one GPU.** The model itself is cut into pieces and each GPU holds a piece. This is
   **model parallelism** (tensor parallelism, pipeline parallelism, and others; section 11).

Our 1.59B model fits on one H200, so for our runs only reason 1 applies.

---

## 3. Data parallelism: the idea

- Every GPU holds the **whole model**.
- Every GPU gets **different text**.
- Every GPU computes gradients on its own text.
- Before each update, the GPUs **combine their gradients** so that every GPU uses the same total gradient.
- Every GPU therefore makes the same update, and all copies of the model stay **identical**.

What is combined is the **gradients**, not the weights. And every GPU gets the combined result, not only one GPU.

---

## 4. What each GPU stores

For our model (1.59B weights), per GPU, not counting activations:

| item | bytes per weight | size | DDP | ZeRO-1 |
|---|---|---|---|---|
| bf16 weights (forward and backward use them) | 2 | 3.2 GB | all, on every GPU | all, on every GPU |
| gradient buffer (fp32, adds up the micro-steps) | 4 | 6.4 GB | all, on every GPU | all, on every GPU |
| optimizer state (master + m + v) | 12 | 19 GB | **all, on every GPU** | **only 1/4 per GPU: 4.8 GB** |
| **total** | | | **28.6 GB** | **14.4 GB** |

**The only difference between DDP and ZeRO-1 is the last row.** Everything else is the same.

On 1 GPU our trainer prints `optimizer state/GPU: 18174.375 MB (vs 18174.375 MB unsharded)`: 12 bytes × 1.588B weights
= 19.06 GB = 18,174 MiB. With 1 GPU there is nothing to split.

---

## 5. One optimizer step: the micro-steps (the same in DDP and ZeRO-1)

Navin's setup: 4 GPUs, each micro-step reads 8 rows × 4,096 tokens, 8 micro-steps per step.

Each GPU works **alone, with no communication**:

1. GPU 0 reads its own 8 rows of text, GPU 1 reads 8 **other** rows, GPU 2 and GPU 3 read their own.
2. Each GPU runs forward → loss → backward on its rows and gets a gradient for **all 1.59B weights**.
3. It adds that gradient into its own gradient buffer.
4. This repeats 8 times.

After micro-step 8:
- 4 GPUs × 8 micro-steps = **32 different micro-batches** were used in this step: 32 × 8 × 4,096 = 1,048,576 tokens.
- Each GPU's buffer holds the sum over **its own** 8 micro-batches. The 4 buffers are **different**, because each GPU
  saw different text.

---

## 6. After micro-step 8: DDP

1. **Gradient sharing:** the GPUs combine their buffers, so every GPU ends up with the **sum of all 4 buffers**, for all
   1.59B weights. (Section 7 shows how the sending really works.)
2. **Update:** every GPU runs the optimizer on **all 1.59B weights**: it updates all master weights, all m and v, then
   rounds all master weights into new bf16 weights.
3. **Nothing is sent after the update.** All 4 GPUs did the same calculation with the same numbers, so they all already
   have the same new bf16 weights.
4. The gradient buffers are set to zero, and the next step starts.

In DDP all 4 GPUs do the same optimizer work, 4 times over, and each keeps the full 19 GB of optimizer state.

---

## 7. How gradients are really shared

The buffer being sent is called **S** below. (In our trainer the gradients travel as bf16, so S = 3.2 GB.)

### 7.1 The simple way: everybody sends to everybody

Each GPU sends its whole buffer to the other 3 and receives their 3 buffers, then does 3 full additions.
Per GPU: **send 3 S, receive 3 S**. This works, but it sends twice as much as needed.

### 7.2 Everything to one GPU, then back out

1. GPUs 1, 2, 3 each send their whole buffer to GPU 0.
2. GPU 0 receives 3 whole buffers and does 3 full additions.
3. GPU 0 sends the summed buffer back to GPUs 1, 2, 3.

| | sends | receives | additions |
|---|---|---|---|
| **GPU 0** | **3 S** | **3 S** | 3 full |
| GPUs 1–3 | 1 S | 1 S | none |

The result is correct, but **GPU 0 moves 6 buffers while the other 3 GPUs wait**. The step cannot continue until GPU 0
is done. It is roughly 3 times slower than the real way below.

### 7.3 The real way: every GPU collects one quarter ("all-reduce")

The buffer is cut into 4 quarters. Each quarter gets a **different** collecting GPU: quarter 0 → GPU 0, quarter 1 →
GPU 1, and so on.

- **Phase 1 (called reduce-scatter):** each GPU sends the 3 quarters it does not collect (one to each collector) and
  receives the other GPUs' copies of the quarter it collects. It adds them up. Now each GPU holds the **total** for one
  quarter. Per GPU: send 3/4 S, receive 3/4 S, 3 quarter-sized additions.
- **Phase 2 (called all-gather):** each GPU sends its finished quarter to the other 3 and receives their finished
  quarters. Now every GPU has the full summed buffer. Per GPU: send 3/4 S, receive 3/4 S.

**Per GPU in total: send 1.5 S, receive 1.5 S.** All 4 GPUs send and receive at the same time, so no GPU waits for
another. This is what NCCL's all-reduce does for DDP.

| way | sent per GPU | received per GPU | problem |
|---|---|---|---|
| 7.1 everybody to everybody | 3 S | 3 S | twice the traffic |
| 7.2 all to one GPU | 3 S on GPU 0, 1 S on the others | same | GPU 0 is the bottleneck |
| **7.3 all-reduce (2 phases)** | **1.5 S** | **1.5 S** | none |

---

## 8. After micro-step 8: ZeRO-1

First the 1.59B weights are divided into **4 quarters** of about 397M weights. GPU 0 **owns** quarter 0, GPU 1 owns
quarter 1, GPU 2 owns quarter 2, GPU 3 owns quarter 3. **Each GPU keeps master + m + v only for its own quarter.**
Every GPU still keeps **all** bf16 weights and a **full** gradient buffer (section 4).

After micro-step 8 (GPU 0 as the example):

1. **Gradient sharing (phase 1, the same as 7.3):** GPU 0 receives the other GPUs' quarter-0 gradients and adds them to
   its own. Now GPU 0 has the summed gradient for quarter 0. At the same time it sends its quarter-1, quarter-2 and
   quarter-3 gradients to their owners. Per GPU: send 3/4 S, receive 3/4 S.
2. **Update, only its quarter:** GPU 0 updates the master weights, m and v of quarter 0, then rounds the master weights
   into **new bf16 weights for quarter 0**. This is 1/4 of the optimizer work.
3. **Weight sharing (phase 2):** GPU 0 sends the **new bf16 weights of quarter 0** to GPUs 1, 2, 3, and receives the new
   bf16 weights of quarters 1, 2, 3 from their owners. Per GPU: send 3/4 of the bf16 weights, receive 3/4.
4. Now every GPU has the full new bf16 model. The gradient buffers are set to zero, and the next step starts.

**The master weights, m and v of quarter 0 never leave GPU 0.** They stay there for the next step. They are only written
out when a checkpoint is saved to disk.

### 8.1 What is sent, compared with DDP

| | DDP | ZeRO-1 |
|---|---|---|
| phase 1 | gradients: send 3/4 S, receive 3/4 S | the same |
| update | all 4 quarters, on every GPU | only the own quarter |
| phase 2 sends | the **summed gradients** (so every GPU can run the full optimizer) | the **new bf16 weights** of the own quarter (the optimizer already ran) |
| phase 2 amount | send 3/4 S, receive 3/4 S | send 3/4, receive 3/4 (of the bf16 weights, the same size as S in our trainer) |
| **total per GPU** | **send 1.5 S, receive 1.5 S** | **the same** |

**ZeRO-1 sends the same amount of data as DDP, but each GPU does 1/4 of the optimizer work and keeps 1/4 of the
optimizer memory.** The only difference is what travels in phase 2: summed gradients (DDP) or finished weights
(ZeRO-1).

---

## 9. A tiny example: a model with 8 weights, w1 … w8, on 4 GPUs

| | DDP | ZeRO-1 |
|---|---|---|
| bf16 weights on every GPU | w1 … w8 | w1 … w8 |
| optimizer state (master, m, v) | every GPU: w1 … w8 | GPU 0: w1, w2 · GPU 1: w3, w4 · GPU 2: w5, w6 · GPU 3: w7, w8 |
| after micro-step 8 | every GPU has its own g1 … g8 (from its own text) | the same |
| gradient sharing | every GPU gets the sum of g1 … g8 over all 4 GPUs | GPU 0 gets the sum of g1, g2 · GPU 1 of g3, g4 · GPU 2 of g5, g6 · GPU 3 of g7, g8 |
| update | every GPU updates w1 … w8 | GPU 0 updates w1, w2 · GPU 1 w3, w4 · GPU 2 w5, w6 · GPU 3 w7, w8 |
| weight sharing | nothing (everyone updated everything) | GPU 0 sends new w1, w2 to all · GPU 1 sends w3, w4 · GPU 2 sends w5, w6 · GPU 3 sends w7, w8 |
| result | every GPU has the new w1 … w8 | every GPU has the new w1 … w8 |
| master w1, w2 after the step | on every GPU | **only on GPU 0** |

---

## 10. Our trainer and Navin's checkpoint

### 10.1 Our ZeRO-1 has two versions

| version | switch | how the 2 phases are done |
|---|---|---|
| **v1** (default) | `BLU_ZERO_V2=0` | for each quarter: one "reduce to the owner" (phase 1) + one "broadcast from the owner" (phase 2) |
| **v2** | `BLU_ZERO_V2=1` (class `SymmetricShardedOptimizer`) | NCCL reduce-scatter + all-gather (the textbook form of 7.3) |

Both are ZeRO **stage 1** (only the optimizer state is split). "v2" is our name for the second implementation, **not**
ZeRO stage 2 (section 11). The trainer prints which one runs: `[ZeRO-1] variant: v1 reduce/broadcast`.

v1's "reduce to the owner, then broadcast from the owner" is the idea of 7.2 done **per quarter**, with 4 different
owners working in parallel. That is why it does not have 7.2's bottleneck.

With **Muon** (`BLU_OPTIMIZER=muon`, class `ShardedMuon`) the idea is the same, but a GPU owns **whole weight matrices**
instead of a flat quarter, because Muon's update needs the whole matrix.

**The gradients travel as bf16** in our trainer: the fp32 sum is rounded to bf16 for sending, then widened again. That
is the "F4" item in the loss doc (§19.7).

With **1 GPU** there is no sharing at all: that GPU owns all 4 "quarters", i.e. the whole optimizer state.

### 10.2 Why Navin's checkpoint has 4 files, and why 1 GPU is enough to load it

Navin trained on 4 GPUs with ZeRO-1 and Muon. When his run saved a checkpoint:
- **each GPU saved its own quarter of the optimizer state** (the master weights and optimizer numbers it owns);
- **the bf16 weights are the same on all 4 GPUs, so they were saved only once, in GPU 0's file.**

The file sizes show exactly that:

| file | bytes | contents |
|---|---|---|
| `data_rank_0.bin` | 6,469,008,896 | the bf16 weights (3,176,473,088) + GPU 0's optimizer share (3,292,535,808) |
| `data_rank_1.bin` … `data_rank_3.bin` | 3,292,535,808 each | that GPU's optimizer share |

The 4 optimizer shares add up to 13.17 GB. That matches Muon keeping master + momentum (8 bytes) for the 1,472,200,704
matrix weights (11.78 GB) plus AdamW keeping master + m + v (12 bytes) for the 116,029,184 embedding and norm weights
(1.39 GB).

**Do we need 4 GPUs to use his checkpoint? No:**
1. The model is not split into 4 pieces: all 262 weights are complete, in `data_rank_0.bin`.
2. This model already runs on 1 H200: our A-B-A-B runs on 2026-09-29 used about 90 GB at B = 4.
3. The loader (DCP) can load a checkpoint saved from 4 GPUs onto 1, 2 or 8 GPUs.
4. Navin used 4 GPUs for **speed** during pretraining (8.4 billion tokens). SFT data is much smaller.

For SFT we load **only the weights** and start a new optimizer, so the optimizer shares in files 0–3 are not needed.

---

## 11. The other kinds of parallelism

### 11.1 More memory saving inside data parallelism (the ZeRO stages)

| name | what is split across GPUs | what each GPU keeps in full | extra communication |
|---|---|---|---|
| **DDP** | nothing | weights, gradients, optimizer state | none |
| **ZeRO-1** | optimizer state | weights, gradients | none (section 8) |
| **ZeRO-2** | optimizer state **and gradients**: each GPU keeps the summed gradient only for its quarter, sending the rest away as soon as backward produces it | weights | about the same |
| **ZeRO-3 / FSDP** | optimizer state, gradients **and weights**: each GPU keeps only its quarter of the weights and fetches the other quarters just before each layer needs them (forward and backward), then drops them | nothing | about 1.5× DDP (the weights are fetched again for backward) |

All of these still give every GPU **different text** and keep the result identical to one big GPU.

### 11.2 Splitting the model itself (model parallelism)

| name | how the model is split | what the GPUs see | communication | when it is used |
|---|---|---|---|---|
| **Tensor parallelism** | each layer's weight **matrices** are cut: e.g. GPU 0 holds attention heads 1–4, GPU 1 heads 5–8; the MLP matrices are cut by columns / rows | the **same text** on all GPUs | inside **every layer**: the GPUs exchange their partial results | a model or a layer too big for one GPU; needs a fast link (NVLink) |
| **Pipeline parallelism** | whole **layers** go to different GPUs: GPU 0 = layers 1–7, GPU 1 = layers 8–14, … | the text flows from GPU to GPU | activations between neighbouring GPUs | very deep models; GPUs are idle at the start and end of each step (the "bubble") |
| **Sequence / context parallelism** | the **tokens of one long sequence** are split: GPU 0 gets tokens 1–4,096, GPU 1 tokens 4,097–8,192, … | parts of the same sequence | attention needs the other GPUs' keys and values (e.g. ring attention) | very long sequences |
| **Expert parallelism** | in mixture-of-experts models, different experts live on different GPUs | each token goes to the GPU of its expert | tokens are sent to and from the expert GPUs | MoE models |

Big trainings combine them, e.g. data parallelism × tensor parallelism × pipeline parallelism ("3D parallelism").

In BluTrain these live under `dist/`: `Data-Parallel/`, `zero/` (ZeRO-1 v1/v2, Muon), `FSDP/`, `Tensor-Parallelism/`
(DTensor), and the `Context_Parallelism` submodule.

---

## 12. The questions asked, and the answers

**Q: With 4 GPUs, does each GPU run its micro-steps on different data, and at the optimizer step do all GPUs send their
weights to one GPU, which adds them up?**
Different data per GPU and micro-steps without communication: yes. But what is combined is the **gradients**, not the
weights, and **every** GPU gets the combined result, not only one GPU (sections 3, 5, 6).

**Q: In DDP, does each GPU send its buffer 3 times, receive 3 buffers and do 3 additions?**
That would work, but it sends twice as much as needed. Real DDP does the 2 phases of section 7.3: each GPU sends
1.5 buffers and receives 1.5 buffers.

**Q: In ZeRO-1, does each GPU send 3/4 of its gradient buffer to the owners and receive 3/4? And is that less than DDP?**
Yes for the first part (phase 1). But DDP's phase 1 is exactly the same. The difference is the update (1/4 of the work)
and phase 2 (weights instead of gradients). In total both send 1.5 buffers per GPU (section 8.1).

**Q: So ZeRO-1 shares the same amount of data but does 1/4 of the compute?**
Yes. The same amount sent, 1/4 of the optimizer work per GPU, and 1/4 of the optimizer memory per GPU.

**Q: In DDP, can we add everything on one GPU and send it back to the others? Is that the same as ZeRO-1?**
The numbers would be correct, but GPU 0 would move 6 buffers while the others wait, about 3 times slower (section 7.2).
ZeRO-1 is that idea done **per quarter**, with a different owner for each quarter, so all GPUs work at the same time.

**Q: After the update, does ZeRO-1 send only the updated weights? Do the master weights stay on each GPU?**
Yes. Each GPU sends only the **new bf16 weights of its quarter**. The master weights, m and v stay on their owner and
are only written out when a checkpoint is saved (section 8).

**Q: In DDP, does each GPU have the full gradient buffer and also the full optimizer state and updated weights?**
Yes. Every GPU has everything and updates everything itself, so nothing is sent after the update (section 6).

**Q: Navin trained on 4 GPUs. Do we need 4 GPUs to use his checkpoint?**
No (section 10.2).

**Q: With GRAD_ACCUM = 8 and `BLU_MAX_STEPS = 20`, is it 20 × 8 forward/backward passes and 10 updates?**
20 × 8 = 160 forward + backward passes and **20** updates: one per printed step. In our runs GRAD_ACCUM was 64, so
1,280 passes and 20 updates (section 1).

---

## 13. Summary

| | DDP | ZeRO-1 |
|---|---|---|
| model on each GPU | full | full |
| text on each GPU | different | different |
| micro-steps | alone, no communication | the same |
| gradients after the last micro-step | summed, every GPU gets all of them | each owner gets the sum for its quarter |
| optimizer state per GPU | full (19 GB) | 1/4 (4.8 GB) |
| optimizer work per GPU | all weights (the same work on every GPU) | 1/4 of the weights |
| sent after the update | nothing | the new bf16 weights of the own quarter |
| data sent per GPU per step | 1.5 buffers | 1.5 buffers |
| master weights | on every GPU | only on the owner |
