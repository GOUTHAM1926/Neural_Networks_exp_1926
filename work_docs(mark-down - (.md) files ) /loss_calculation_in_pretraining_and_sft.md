# Loss calculation in pretraining and SFT: how the loss is averaged, how its gradient flows, and what SFT changes

- **Date:** 2026-10-06.
- **Code read** (read only, nothing was changed): BluTrain `dist_side_merge` at `909ab75`, with Tensor-Implementations (TI) at `d96b991`. This is the laptop clone `~/Downloads/BluTrain_sft`. Today's fetch failed because it needs the token, so newer remote commits, if there are any, were not checked.
- **Trainer line numbers** (`bluscript_zero_ckpt.cpp:…`) are from the committed file at `909ab75`. The laptop working copy also has the uncommitted task-2 edit (`BLU_INIT_FROM`), so there the same lines sit 77 to 79 lines lower.
- **Library line numbers:**
  - `LossOps.cpp` = `TI/src/autograd/operations/LossOps.cpp`;
  - `LossBackward.cpp` = `TI/src/autograd/backward/LossBackward.cpp`;
  - `LossKernels.cu` = `TI/src/Kernels/cuda/loss/LossKernels.cu`;
  - `LossOps.h` = `TI/include/autograd/operations/LossOps.h`;
  - optimizer files are in `BluTrain/dist/zero/src/`.
- **Other frameworks:**
  - laptop clones in `~/Downloads/sft_research`: torchtitan `1e4d33d`, Megatron-LM `8b2f660`, TRL `478624d`;
  - read on GitHub on 2026-10-06: Hugging Face Transformers `main`, nanoGPT `master`, PyTorch `main`. Line numbers on `main` can move later.
- **Data:** stage-0 SFT data, `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/train` (5,000 UltraChat rows, T = 4,096).
- **Status:**
  - the library already supports the SFT loss (verified by reading the code);
  - the trainer does not use it yet. That is task 6 (section 8).
  - **Update 2026-11-18:** task 6 is done, in the separate SFT trainer `train/production_zero/bluscript_sft.cpp`: it
    passes label 65535 as the ignore index and `sft_denom`, the step's global trained-token count (all micro-batches,
    all ranks), to the loss (lines 3008, 3045, 3548 and 3582 at BluTrain d9606c2). The pretraining trainer keeps
    Method 1, which is exact there. Sections 7.3 and 8 describe the state of 2026-10-06; the current state is in
    `sft/README.md` section 6.1.

---

## 0. The answer in eight sentences

1. Each micro-step makes **one loss number**: the sum of its token losses, divided by a count **N**.
2. **Pretraining:**
   - every micro-batch has the same N (8 × 4,096 = 32,768 tokens);
   - so "average each micro-batch, then average the 32 micro-batches" gives exactly the true average;
   - that is correct, and it stays as it is.
3. **SFT:**
   - only answer tokens and the end token count, and their number changes a lot from one micro-batch to the next;
   - in our stage-0 data it goes from 2,920 to 12,729, and up to **4.1×** within one optimizer step;
   - the same "average of averages" would give tokens in small micro-batches up to 4× more weight, for no reason.
4. **The fix is the global denominator:**
   - every micro-step divides by the same number: the answer tokens of the **whole step** (all micro-steps, all GPUs);
   - the backward seed becomes **W** (the number of GPUs) instead of 1/32.
5. **The 1/N is applied exactly once.**
   - The backward kernel computes `(softmax − one-hot) × seed ÷ N` for each token.
   - The loss's **value** is never an input to the backward (`LossKernels.cu:704`, `:813`).
6. **The library already does the global denominator correctly**: the same GPU tensor divides both the forward and the backward. But the trainer still calls the loss with 2 arguments (`bluscript_zero_ckpt.cpp:2164`), so the SFT path is **not wired yet** (task 6).
7. **The F1 NaN fix does not look at the loss.**
   - It checks the **gradient norm** inside the optimizer, before any weight changes, and stops with exit code 4.
   - A separate, older check looks at the loss number and stops with exit code 2.
8. **DFT and focal loss:**
   - they keep the same global N but change each token's weight;
   - a standalone test can prove they are built right; only training runs can show whether they help;
   - both are future tasks.

### Your questions, and where they are answered

| Question | Short answer | Section |
|---|---|---|
| "Count of 8 sequences ÷ 8, or valid tokens of 8 micro-steps ÷ 8?" | Neither. **Nothing is divided by 8.** N = all valid tokens of the whole step | 4, 5 |
| "Today: sum [8,4096] ÷ 8×4096; SFT: sum everything, divide once. Same intention?" | Yes. Same goal (the average per counted token). Pretraining is the special case where every token counts | 2, 4.9 |
| "Is one division at the end better?" | Same math. We divide inside each micro-step because that costs nothing extra | 5.3 |
| "Does one loss number make a gradient for every parameter?" | Yes | 6.6 |
| "Is the global N for the loss or for the gradients?" | It divides the **loss**. The gradients get the same ÷ N automatically | 5.4 |
| "Does PyTorch / HF / Megatron / torchtitan do this?" | torchtitan and HF: always. Megatron: with a flag. nanoGPT: no. Proof in section 9 | 9 |
| "What does the second average do in SFT? Visualize it" | It gives each micro-batch an equal share, whatever its size | 4.4 |
| "Equal share for all tokens: isn't that wrong too?" | No. Equal **multiplier**; each token keeps its own loss and its own push | 4.8 |
| "Is 1/N applied twice (kernel + the scalar)?" | **No.** The scalar's value is not used in the backward | 6 |
| "How far does the scaling carry?" | From the loss all the way to every weight, exactly once | 6.6 |
| "For SFT, is the global N what the loss backward uses? Did you check?" | **Library: yes, verified in code. Trainer: not yet (task 6)** | 7 |
| "Does the NaN fix check the final loss number?" | **No**, it checks the gradient norm. An older check looks at the loss | 10 |
| "Before the fix, where did a NaN stop the run?" | At the next step's loss check, one step late. Muon matrices: never | 10.5 |
| "DFT / focal with the global N: same or different? How to test?" | Different (per-token weights). Standalone test for correctness, training runs for usefulness | 11 |

---

## 1. Words you need

| Word | What it means here |
|---|---|
| **token** | One piece of text (a word or part of a word). The model reads tokens and predicts the next one. |
| **vocabulary** | All the tokens the model knows: 50,304 entries (`bluscript_zero_ckpt.cpp:135`). |
| **logits** | The model's raw scores: one score per vocabulary entry, at every position. |
| **p** | The probability the model gives to the **right** next token (the softmax of the logits). |
| **token loss** | −log p. Small when the model is right (p near 1), big when it is wrong (p near 0). |
| **label / target** | The right next token at a position. |
| **ignored position** | A position whose label is **65535**: prompt text and padding. It gives no loss and no gradient. |
| **valid token** | A position whose label is not 65535: the answer tokens, plus the end token (50256) after each answer. |
| **N** | The count the loss is divided by. |
| **micro-batch / micro-step** | The 8 rows the GPU processes at once (B = 8, T = 4,096, `:198-199`). |
| **gradient accumulation (GA)** | Running several micro-steps and adding their gradients before one weight update. GA = 32 on 1 GPU, 16 on 2 GPUs. |
| **optimizer step** | One weight update. |
| **W** | The number of GPUs. |
| **gradient** | For each weight: which way, and how much, to move it. |
| **backward** | The pass that computes the gradients, from the loss back to the first layer. |
| **seed** | The number backward starts with. Normally 1. Our trainer passes its own (`grad_scale`). |
| **chain rule** | How backward moves through the layers: each layer multiplies the gradient coming in by its own factor. |
| **push** | Our word for one token's gradient on its logits: (predicted probabilities − right answer). |
| **all-reduce SUM** | The GPUs exchange a number, and each one ends up with the total. |
| **ZeRO average** | How our optimizer combines the gradients of the GPUs: it averages them (÷ W). |
| **NaN / Inf** | "Not a number" / infinity: a broken value. It spreads to everything it touches. |
| **grad norm** | One number for the size of all the gradients together (the square root of the sum of their squares). |
| **clip** | If the grad norm is above 1.0, every gradient is scaled down so the norm becomes 1.0 (`grad_clip = 1.0f`, `:201`). |

---

## 2. Shapes: what one micro-step computes

```
input ids        [8, 4096]           uint16   8 rows of 4,096 tokens
   |  model.forward (26 transformer blocks, bluscript_zero_ckpt.cpp:128)
logits           [8, 4096, 50304]    one score per vocabulary entry, per position
   |  viewed as [32768, 50304]       (LossOps.cpp:269-281)
token losses     [32768]             fp32, one number per position (the d_losses buffer inside LossKernels.cu)
   |                                 an ignored position (label 65535) gives 0 and is skipped
   |  add all 32,768 up, then / N    (LossKernels.cu:166-191; the "/ N" is at :188)
micro loss       [1]                 fp32, ONE number
```

**Pretraining today:**
- **N** = the number of labels that are not the "ignore" value.
  - The trainer passes no ignore value, so the default −1 is used (`LossOps.h:41-43`), and no uint16 label can equal −1.
  - So N = **32,768**, every position (`LossOps.cpp:403-404`; the count kernel is `LossKernels.cu:102-121`).
- **GA** = global batch ÷ (B × T × W) = 1,048,576 ÷ 32,768 = **32** on 1 GPU (`bluscript_zero_ckpt.cpp:198-200`, `:918-926`).
- **The seed** is 1/32 (`:1796-1797`), passed in `loss32.backward(&grad_scale)` (`:2212`).
- So every token's gradient is multiplied by (1/32) × (1/32,768) = **1/1,048,576** = 1 ÷ (all tokens of the step). That is the true average.

**SFT:** same shapes. The only difference is in the labels: the prompt and padding positions hold 65535.

---

## 3. What the SFT data looks like

One conversation per row. Only the assistant's answer and the end token after it are trained:

```
<|user|>
What is 2+2?
<|assistant|>
2+2 is 4.<|endoftext|>   <- trained (real labels)
[padding up to 4,096]     <- not trained (label 65535)
```

Everything before the answer (the markers and the user's text) also has label 65535.

`valid_counts.bin` stores the number of valid labels in each row. It was written by `sft/prepare_sft_data.py`, and the loader re-checks it when it opens the files (`SftDataLoader.h:117-135`).

**Stage-0 numbers** (measured today from `valid_counts.bin`):

| | min | max | mean | of |
|---|---|---|---|---|
| valid tokens per row | 32 | 3,872 | 954.3 | 4,096 positions |
| valid tokens per micro-batch (8 rows) | 2,920 | 12,729 | 7,634 | 32,768 positions |

- Total: 4,771,516 trained tokens = **23.3%** of all positions.
- Within one optimizer step (32 micro-batches on 1 GPU), the biggest micro-batch has more valid tokens than the smallest:
  - **2.54×** more, as the median over the 19 steps of one epoch;
  - **4.13×** more in the worst step.

---

## 4. Two ways to average

- **Method 1, "average of averages"** (BluTrain today): each micro-batch divides by **its own** count, then the result is divided by the number of micro-steps (the 1/32 seed).
- **Method 2, "global denominator"** (the SFT plan): count the valid tokens of the **whole step** (all micro-steps, all GPUs) and divide by that one number.

The small examples below use 2 micro-steps to keep the numbers easy. The real trainer uses 32 (1 GPU) or 16 (2 GPUs).

### 4.1 Example 1: just the numbers

| | answer-token losses | sum | count |
|---|---|---|---|
| micro-batch 1 | 2, 2, 2, 2 | 8 | 4 |
| micro-batch 2 | 6 | 6 | 1 |
| **whole step** | | **14** | **5** |

- **The correct average** = 14 ÷ 5 = **2.8**.
- **Method 1:**
  - micro 1: 8 ÷ 4 = 2;
  - micro 2: 6 ÷ 1 = 6;
  - result: (2 + 6) ÷ 2 = **4.0** ❌. The **single token** of micro-batch 2 counted as much as **all 4 tokens** of micro-batch 1.
- **Method 2:** first count every answer token of the step: 4 + 1 = **5**. Then:
  - micro 1 computes 8 ÷ **5** = 1.6;
  - micro 2 computes 6 ÷ **5** = 1.2;
  - they are **added** (gradient accumulation adds): 1.6 + 1.2 = **2.8** ✅.

**That 5 is the global denominator.** Every micro-batch divides by the same 5, so every token gets the same weight, 1/5.

### 4.2 Example 2: the weight of each token

| | rows' valid tokens | micro-batch total |
|---|---|---|
| micro-step 1 | 100 + 300 | 400 |
| micro-step 2 | 50 + 50 | 100 |
| **step total** | | **500** |

| Method | weight of each token in micro-step 1 | weight of each token in micro-step 2 |
|---|---|---|
| **Method 1** (own count, then × 1/2) | 1/800 | **1/200: 4× more** |
| **Method 2** (both divide by 500, then add) | **1/500** | **1/500: equal** ✅ |

### 4.3 A common mix-up

"A micro-batch with fewer tokens gets a bigger average": **no.**
- An average shows how wrong the tokens are **on average**, whatever their number.
- A micro-batch's average is high only if its tokens are hard.
- What the token **count** changes is the **weight** in the second average. That is the whole problem.

### 4.4 Example 3: what the second average does (visual)

| | answer tokens counted | loss of each token | sum | 1st average |
|---|---|---|---|---|
| **micro-batch A** (long answers) | 8 | 1.0 each | 8 | 8 ÷ 8 = **1.0** |
| **micro-batch B** (one short answer) | 2 | 4.0 each | 8 | 8 ÷ 2 = **4.0** |
| **whole step** | **10** | | **16** | |

B's average is higher because **its tokens are harder** (loss 4.0), not because there are fewer of them.

**Method 1:** second average = (1.0 + 4.0) ÷ 2 = **2.5**. Each micro-batch gets **half** of the total, whatever its size:

| token | share of the final loss |
|---|---|
| each of A's 8 tokens | 50% ÷ 8 = **6.25%** |
| each of B's 2 tokens | 50% ÷ 2 = **25%** |

**Method 2:** (8 + 8) ÷ 10 = **1.6**:

| token | share of the final loss |
|---|---|
| each of A's 8 tokens | 1 ÷ 10 = **10%** |
| each of B's 2 tokens | 1 ÷ 10 = **10%** |

```
                    A tokens (8 of them)                 B tokens (2)
Method 2 (correct): ██ ██ ██ ██ ██ ██ ██ ██              ██ ██          -> loss 1.6
Method 1:           █▎ █▎ █▎ █▎ █▎ █▎ █▎ █▎              █████ █████    -> loss 2.5
```

**What goes wrong with Method 1 in SFT:**
1. **The update is skewed.**
   - The weights decide how hard the model is pushed to fix each token.
   - Each token of the short answer is pushed **4× harder** than each token of the long answers (25% vs 6.25%).
   - The model over-learns short answers and under-learns long ones.
2. **The reported loss is wrong:** 2.5 instead of the true 1.6, so the training curves mislead.

### 4.5 Example 4: real stage-0 numbers (step 0, 1 GPU)

| | weight of each answer token |
|---|---|
| **Method 2** | 1/243,536 for every token (243,536 = the step's valid tokens) |
| **Method 1**, token in the biggest micro-batch (11,827 valid tokens) | 1/(32 × 11,827) = 1/378,464 |
| **Method 1**, token in the smallest micro-batch (3,936 valid tokens) | 1/(32 × 3,936) = 1/125,952: **3.0× more** than the biggest |

So the problem is not theoretical. With our real data, Method 1 would weigh tokens 2.5× to 4.1× differently inside one step, only because of which micro-batch they landed in.

### 4.6 Why that matters per token: "fake" vs "real" reasons to count more

Each token's contribution = **its own loss × its multiplier**. Example 3 again:

| | token | own loss | × multiplier | = contribution |
|---|---|---|---|---|
| **Method 2** | each A token | 1.0 | × 1/10 | **0.1** |
| | each B token | 4.0 | × 1/10 | **0.4**: 4× more, because it is 4× more wrong ✅ |
| **Method 1** | each A token | 1.0 | × 1/16 | **0.0625** |
| | each B token | 4.0 | × 1/4 | **1.0**: 16× more = 4× (more wrong, real) × 4× (unequal multiplier, **fake**) ❌ |

With Method 2, **the only reason one token counts more than another is that it is more wrong.** Method 1 adds a fake factor that depends on which micro-batch the token landed in.

### 4.7 "Equal share" does not mean "equal update"

The global denominator gives every token the **same multiplier** (1/N), not the same update. The update follows each token's own error, because the cross-entropy gradient of a token on its logits is:

> **push = (the predicted probabilities) − (the right answer as one-hot)**

On the right token's logit, that is p − 1, so its size is (1 − p):

| how sure the model was of the right token (p) | token loss (−log p) | push on the right logit (1 − p) |
|---|---|---|
| 0.9 (almost right) | 0.105 | **0.1** (small) |
| 0.1 (quite wrong) | 2.303 | **0.9** (big) |
| 0.01 (very wrong) | 4.605 | **0.99** (it cannot go past 1) |

- A bigger loss means a bigger push, but **not in exact proportion**: the push tops out near 1, while the loss keeps growing.
- Backprop carries each token's push through every layer, so each weight's gradient = **the sum of all the tokens' pushes on it** (times the path factors of the layers in between).
- The optimizer then moves each weight a lot or a little based on that sum.

So the 1/N is only the **shared scale**, like the learning rate, and each token's own error decides how much it moves the weights.

### 4.8 Why pretraining is fine with Method 1

Every micro-batch has **exactly 32,768 tokens**, and none is ignored. Each micro-batch's share is split over the same number of tokens, so every token gets the same weight. Both methods give **exactly the same number**.

### 4.9 Method 2 also works for pretraining

- In pretraining, N = GA × W × 32,768 = all the tokens of the step. The result is mathematically identical to today's.
- So Method 2 is the **general** method, and pretraining is the special case where every token counts.
- torchtitan and Hugging Face use it for every run, pretraining included (section 9).
- We keep pretraining as it is: it is already right, and changing it would only risk tiny rounding differences in runs that are already validated.

---

## 5. How the global denominator works in code

### 5.1 The steps, once per optimizer step

1. **Before the first micro-step**, each GPU asks its loader for the valid tokens of its next GA micro-batches:
   - `valid_tokens_ahead(GA)` (`SftDataLoader.h:93-97`);
   - it adds up `valid_counts.bin` without consuming the batches.
2. The GPUs **all-reduce SUM** that number, so every GPU has the same total: the global denominator.
3. The total goes into a **1-element fp32 tensor on the GPU** (`denom`).
4. **Every micro-step** calls `sparse_cross_entropy_loss(logits, target, 65535, &denom)`.
5. The backward seed is **W** instead of 1/GA.

### 5.2 Why the total must be known before micro-step 1

- Micro-step 1 divides by N **while it runs**.
- Its backward then adds its gradients into the gradient buffer right away, so it cannot be changed later.
- So we count first. That is cheap, because the counts are already in `valid_counts.bin`.

### 5.3 Why every micro-step divides by the total, instead of "divide once at the end"

- **Mathematically they are identical:** (a + b) ÷ N = a ÷ N + b ÷ N.
- **But what accumulates across micro-steps is the gradients, not the loss numbers.**
  - After micro-step 1's backward, its gradients (one number per parameter, about **1.59 billion numbers**) are added into the buffer.
  - The micro-step's loss number has done its job and is only logged.
- **"Divide once at the end"** means dividing all 1.59 billion gradient numbers by N before the optimizer step. That is an extra pass over every gradient, plus new code in the optimizer path. Megatron's `--calculate-per-token-loss` does it this way (section 9).
- **Dividing inside each micro-step costs nothing extra.**
  - The loss kernel already divides by something (today 32,768); we just give it a different number.
  - The library's loss op was built for exactly this (the `denom` argument). torchtitan does the same.
- Precision is about the same either way.

### 5.4 The global N divides the loss, and the gradients follow by themselves

- It is used in the loss kernel: **micro loss = (sum of this micro-batch's valid-token losses) ÷ N**.
- Dividing by a fixed number passes straight through the backward:
  - if loss = sum ÷ N, then the gradient of the loss = (the gradient of the sum) ÷ N;
  - so every gradient automatically carries the same ÷ N, and nobody touches the gradients separately.

### 5.5 Two GPUs, and why the seed becomes W

| | sum of losses | answer tokens |
|---|---|---|
| GPU 0 | 14 | 5 |
| GPU 1 | 30 | 15 |
| **both** | **44** | **20** |

The correct average is 44 ÷ 20 = **2.2**.

1. The GPUs exchange their counts and add them: 5 + 15 = **20**, now known on both.
2. Each GPU divides by **20**: GPU 0 gets 14 ÷ 20 = 0.7, and GPU 1 gets 30 ÷ 20 = 1.5.
3. These should be **added**: 0.7 + 1.5 = **2.2** ✅.
4. **But ZeRO combines the GPUs by averaging:** (0.7 + 1.5) ÷ 2 = 1.1, which is **half** ❌.
5. So each GPU starts its backward with **W = 2**: 1.4 and 3.0. Then ZeRO's average gives (1.4 + 3.0) ÷ 2 = **2.2** ✅.

That multiplier is the **"seed = W"**.
- Today's seed of 1/GA existed because each micro-batch averaged itself, so the micro-batches needed averaging.
- With the global N, the total already covers all micro-steps and all GPUs, so the 1/GA must go.
- On 1 GPU, W = 1.

**All three of our optimizers average over the GPUs, so the seed is W for all three:**

| Optimizer | How it combines the GPUs | Proof |
|---|---|---|
| ShardedOptimizer (v1, the default) | `op=avg` | `ShardedOptimizer.cpp:580` (bucketed reduce to the owner), `:989` (reduce-scatter) |
| SymmetricShardedOptimizer (`BLU_ZERO_V2=1`) | `op=sum`, then × 1/W in its single scale pass | `SymmetricShardedOptimizer.cpp:751-759`, `:867` |
| ShardedMuon (`BLU_OPTIMIZER=muon`) | `op=avg`; in RS/AG mode `op=sum`, then × 1/W (`grad_prescale_`) | `ShardedMuon.cpp:840`, `:983`, `:1646` |

### 5.6 Two safety facts

- **The count is exact.**
  - It is stored as a 32-bit float, which holds every whole number up to 16,777,216 exactly.
  - One step has at most 1,048,576 positions.
- **There is no division by zero.**
  - The loader refuses any row without a trained token (`SftDataLoader.h:132`), so the total is at least 8 × GA × W.
  - The count kernel's own clamp to ≥ 1 (`LossKernels.cu:120`) only applies when the library counts by itself (`denom` = nullptr).

---

## 6. The backward: where the 1/N goes, and why it is not applied twice

### 6.1 The question

The loss value already contains 1/N, and the backward kernel also multiplies by 1/N. If backward "starts from" the loss value, doesn't every gradient get 1/N × 1/N?

**No.** The backward never uses the loss **value**. The code shows it.

### 6.2 What the backward kernel computes (the code)

In the kernel used during training (`sparseCENormalize_from_stats`, `LossKernels.cu:671`):

```cpp
// LossKernels.cu:692-697 -- an ignored position gets a ZERO gradient row
if (static_cast<int64_t>(targets[row]) == ignore_index) { ... row_grad_z[c] = 0; return; }

// LossKernels.cu:704 -- the only scale factor
const float f_scale = static_cast<float>(*grad_output) / (*denom);

// LossKernels.cu:739 and :751 -- for every vocabulary entry j of this position
row_grad[j] = (prob - indicator) * f_scale;   // indicator = 1 for the right token, else 0
```

The other path (no saved stats) is identical: `:777` (zero row), `:813` (`f_scale`), `:857` / `:871`. The CPU path does the same: `total_scale = grad_val * host_scale` with `host_scale = 1 / denom` (`LossBackward.cpp:291`, `:303`).

So per token: **gradient on its logits = (softmax − one-hot) × seed ÷ N.**

### 6.3 What goes into the backward kernel

The backward kernel receives these inputs (`LossBackward.cpp:351-376`):
- the logits;
- the labels;
- the saved max and sum of the softmax (from the forward);
- `grad_output`, which is **the seed**;
- `denom`, which is **N**;
- the ignore value.

**The loss value (for example 1.6) is not one of them.** It is never read.

### 6.4 What the seed is

- The seed is **"how much the final objective changes when this micro loss changes"**. It is **not** the loss value.
- In PyTorch, `loss.backward()` with no argument starts from 1.0.
- Our trainer passes its own start value: `grad_scale` = 1/GA today (`:1796-1797`, `:2212`).
  - Why 1/GA: today the step's objective = (1/GA) × Σ of the micro losses, so each micro loss affects it by 1/GA.
- The fp32 conversion node between them passes the seed through unchanged (`:2178-2181`; `LossBackward.cpp` converts `grad_output` to fp32 only if needed).

So today a token's gradient = **(1/GA) × (1/N) × (softmax − one-hot)**.
- There are two factors, but they are **different things**:
  - 1/GA averages over the micro-steps;
  - 1/N averages over the tokens inside one micro-batch.
- Neither is applied twice.
- In SFT it becomes W × (1/N_global) × (softmax − one-hot). ZeRO's ÷ W then cancels the W, so each token ends at **1/N_global**, exactly once.

### 6.5 A worked example (1 GPU, 1 micro-step, seed 1)

There are two valid tokens, so N = 2:

| token | p (right token) | token loss | gradient on the right logit = (p − 1) × 1/N | on each wrong logit j |
|---|---|---|---|---|
| a | 0.9 | 0.105 | (0.9 − 1) × ½ = **−0.05** | p_j × ½ |
| b | 0.1 | 2.303 | (0.1 − 1) × ½ = **−0.45** | p_j × ½ |

- The loss value = (0.105 + 2.303) ÷ 2 = **1.204**. It is printed, and **used nowhere** in the numbers above.
- If 1/N were applied twice, the gradients would be −0.025 and −0.225, half of the right answer. **The code does not do that.**

### 6.5a Check: PyTorch gives the same numbers (run on 2026-10-06)

I ran the example of 6.5 in PyTorch 2.14.0 (CPU, laptop venv `~/Downloads/tf_issue_venv`):
- two positions with p = 0.9 and p = 0.1 for the right token;
- `cross_entropy(..., reduction="mean")`, then `backward()` with no argument, which starts from 1.0.

```
loss value          : 1.204
grad on right logits: [-0.05, -0.45]
same, seed 1/32     : [-0.0015625, -0.0140625]  = above / 32
with 1 ignored row  : [-0.05, -0.45, 0.0]
```

- **The 1/N is applied once:** −0.05 and −0.45, as in 6.5. The loss value 1.204 is not in them.
- **A seed only scales them:** with 1/32 they are exactly 32 times smaller. That is our trainer's 1/GA.
- **An ignored position** (label −100 in PyTorch, 65535 for us):
  - its gradient row is 0;
  - N stays 2, because the ignored position is not counted.

PyTorch's design, which our kernel copies (comments at `LossKernels.cu:84-99`):
- `nll_loss` divides the forward by `total_weight`;
- it hands that **same** number to the backward, which computes `grad_output / total_weight`;
- so it is one division in the forward, and one in the backward, by the same number.

### 6.6 Where the 1/N is created (once) and how far it carries (all the way)

Two things are true at the same time.

**1. The 1/N is created in exactly one place: the backward of the averaging step.**
- The averaging is the step from "per-token losses" to "one number".
- Your idea, that the 1/N belongs to the step between the one number and the per-token losses, is right.
- In PyTorch the average and the per-token −log p can be separate graph nodes. In our library they are **one fused kernel**. Either way the 1/N enters once.

**2. Once created, it is carried down to every weight.**
- Every backward step after the loss only multiplies by more factors (the chain rule). It never divides by N again.

```
seed  (1/GA today; W in SFT)                <- start of backward. NOT the loss value
  |  x 1/N        backward of the average           \  one fused kernel:
  |  x (p - y)    backward of each token's -log p   /  LossKernels.cu:704 + :739/:751
grad_logits [32768, 50304] = seed x (p - y) / N   (0 for ignored positions)
  |  lm_head backward: its weight gradient and the hidden-state gradient
  |  are both proportional to grad_logits
block 26 ... block 1 ... embedding                (each step: x its own factor)
every weight's gradient = sum over tokens of (seed / N x that token's push x path factors)
  |  ZeRO: average over GPUs (/ W)
  |  clip (norm compared with 1.0), then AdamW / Muon
```

- So each of the 1.59 billion gradient numbers carries **seed/N exactly once**.
- Nothing upstream multiplies by 1/N again.
- One loss number per micro-step produces a gradient for **every** parameter. The GA micro-steps' gradients are added, and the gradient of a sum is the sum of the gradients, so the buffer ends up holding the gradient of the step's loss.

### 6.7 Starting backward "before the average" gives the same gradients

- Give each valid token a starting gradient of seed/N (and 0 to ignored tokens), and you get **exactly the same gradients**.
- Our kernel already works this way. Frameworks keep the scalar only as the handle you call `.backward()` on.
- If you start each token with **1** instead of 1/N (a sum instead of an average), every gradient becomes **N times bigger**. See 6.8 for what that does.

### 6.8 Does the overall size of the gradients matter? (a correction)

Earlier I said that starting each token with 1 instead of 1/N "acts like an N-times bigger learning rate". That is true for plain SGD. For our optimizers it is only **partly** true:

- **AdamW** divides each gradient by its own running size (√v), so a constant factor mostly cancels.
- **Muon** rescales each matrix's update to a fixed size, so it mostly cancels too.
- **It still matters in three places:**
  - the clip, which compares the norm with 1.0 (`:201`);
  - Adam's epsilon, a tiny number added to √v;
  - the logged grad norm.
- **The relative weights between tokens never cancel.** That is what the global denominator fixes.

---

## 7. For SFT, does the loss backward use the global N? (verified in code)

### 7.1 Library: yes

The **same** GPU tensor is used at every step of this chain:

| Step | What happens | Where |
|---|---|---|
| 1 | The caller passes `&denom`: a 1-element fp32 GPU tensor holding the global count | `LossOps.h:41-43` |
| 2 | The loss op uses that tensor itself (no copy, no counting) | `LossOps.cpp:405-407` |
| 3 | The forward's last reduction divides the sum by `*denom` | `LossOps.cpp:427`, `:436-443` → `LossKernels.cu:953-967` → `:188` |
| 4 | The **same** tensor is stored in the backward node | `LossOps.cpp:493-495`; constructor `LossBackward.cpp:198-205` |
| 5 | The backward hands it to the kernel | `LossBackward.cpp:221`, `:351-376` |
| 6 | The kernel computes `f_scale = seed / *denom`, then `grad = (p − one-hot) × f_scale` | `LossKernels.cu:704` + `:739` / `:751` (the training path); `:813` + `:857` / `:871` (the other path) |

- Because it is **one device tensor**, the forward and the backward cannot use different N values, and nothing waits for the CPU.
- When no `denom` is given, the library counts this micro-batch's own valid labels instead (`LossOps.cpp:408-426`). That is Method 1.

### 7.2 Ignored positions: no loss, no gradient, not counted

- **Forward:** the loss is 0 and the row is skipped (`LossKernels.cu:215-218` and `:350`).
- **Backward:** the whole gradient row is set to 0 (`:692-697` and `:777`).
- **Count:** ignored labels are not counted (`:110`).

### 7.3 Trainer: not yet (this is task 6)

(2026-11-19: done in the SFT trainer `bluscript_sft.cpp`; the status of every item is under the table of section 12.)

- `bluscript_zero_ckpt.cpp:2164` calls `sparse_cross_entropy_loss(logits, batch.target)` with **2 arguments**. So the ignore value is −1 and `denom` is nullptr, and each micro-batch divides by its own count, 32,768.
- The seed is 1/GA (`:1796-1797`).
- The validation loss (`:2102`) is the same, and it averages the per-batch means (`:2106`), which is Method 1.
- No SFT run has happened yet, so nothing has been trained wrongly.

### 7.4 Danger: the loader must not go in alone

Suppose `SftDataLoader` is plugged in, but the loss call still uses the default −1. Then the 65535 labels are **not** skipped:

- **Forward:** the kernel reads `row_logits[65535]`, but a row has only 50,304 entries, and there is no bounds check (`LossKernels.cu:318-319`, `:436-437`). It reads numbers from the next row, or from past the end of the buffer for the last row. The result is a garbage loss, or an illegal memory access.
- **Backward:** no vocabulary entry equals 65535, so every logit of that position is pushed down. The model learns from prompt and padding positions, in a broken way.

**So the loader switch and the `65535` loss call must go in the same change.** The loader cannot catch this itself: it is the trainer's call.

### 7.5 B300 warning

- The B300 (sm_103) forward kernel **ignores the ignore value** (the TODO at `LossKernels.cu:938-941`). On a B300 it would compute losses for the prompt and padding positions, and read out of bounds for 65535.
- Do not run SFT on a B300 until that is ported.
- The H200 (sm_90) uses the generic kernel, which is correct.

---

## 8. Task 6: the loss changes in the trainer

| # | What | Today (`909ab75`) | SFT |
|---|---|---|---|
| 1 | Data loader | `DataLoaderLite`: the target is the input shifted by one | `SftDataLoader`: the target comes from `labels.bin` |
| 2 | Before the micro-step loop | nothing | `local = loader.valid_tokens_ahead(GA)`; all-reduce SUM over the GPUs; write the total into a 1-element fp32 GPU tensor `denom` |
| 3 | Loss call (`:2164`) | `(logits, batch.target)` | `(logits, batch.target, 65535, &denom)` |
| 4 | Seed (`:1796-1797`) | 1/GA | W |
| 5 | Logged train loss (`:2218-2221`) | Σ micro losses ÷ GA, then all-reduce SUM ÷ W | Σ micro losses (no ÷ GA), then all-reduce SUM (no ÷ W). Otherwise the printed loss is GA × W times too small. This is logging only; the gradients do not use it |
| 6 | Validation loss (`:2102-2106`) | each batch's own mean, then averaged (Method 1) | same as training: count the val batches' valid tokens first, divide every batch by that total, and add (all-reduce SUM) |
| 7 | NaN checks | unchanged | unchanged (section 10.7) |

The global batch (1,048,576) counts **positions**. In stage-0 only about 23% of them are trained, so one step has about 244K trained tokens (step 0: 243,536). The SFT batch-size and LR decisions should think in trained tokens.

---

## 9. What other frameworks do (with proof)

Every claim below comes from the source code (versions in the header).

| Framework | Method | Proof (file:line) |
|---|---|---|
| **PyTorch itself** (`CrossEntropyLoss`) | averages **one call** (one micro-batch). Gradient accumulation is left to your training loop | `torch/nn/modules/loss.py:1278-1280`: *"ignore_index … the loss is averaged over non-ignored targets"* |
| **nanoGPT** (pretraining) | **Method 1** | `train.py:292`, `:301`: `for micro_step in range(gradient_accumulation_steps):` … `loss = loss / gradient_accumulation_steps` |
| **Megatron-LM, default** | **Method 1** | `core/pipeline_parallel/schedules.py:336-340`: `if not config.calculate_per_token_loss:` → `output_tensor /= num_tokens`, then `/= num_microbatches` |
| **Megatron-LM, `--calculate-per-token-loss`** | **Method 2**, dividing the **gradients once at the end** | `core/distributed/finalize_model_grads.py:697-716`: all-reduce `num_tokens`, `scaling = 1.0 / safe_num_tokens`, `model_chunk.scale_gradients(scaling)`. Required for sequence packing (`training/arguments.py:1737-1738`) |
| **torchtitan** (PyTorch's own trainer, used for **pretraining**) | **Method 2, always**, dividing the **loss** in each micro-step (like us) | `trainer.py:356-361` counts the valid tokens of all the micro-batches; `:375` sums them over GPUs; `:393` passes the same `global_valid_tokens` to every micro-batch; `components/loss.py:321-322`: `loss = loss / global_valid_tokens` |
| **Hugging Face Transformers Trainer** | **Method 2, for every run** (pretraining or SFT) whenever labels exist | `trainer.py:1855-1859` takes the micro-batches of one step; `:2318` `num_items_in_batch = sum(labels.ne(-100).sum() …)`; `:2325` sums it over GPUs; `:1892` passes it to every micro-step; `loss/loss_utils.py` `fixed_cross_entropy`: `reduction = "sum"` … `loss = loss / num_items_in_batch` |
| **TRL `SFTTrainer`** | **Method 2** | `trl/trainer/sft_trainer.py:161-163`: *"the loss is reduced as `sum / num_items_in_batch`"*; code at `:225-231` |

**How they handle "the GPUs average the gradients" (our seed = W):**

| Framework | What it does | Proof |
|---|---|---|
| Hugging Face | multiplies the loss by the **number of GPUs**, like our seed W | `trainer.py:2183-2190`: `loss *= loss_scale` (= `num_processes`) |
| Hugging Face | **drops the old ÷ gradient accumulation**, like removing our 1/GA | `trainer.py:2089-2094` |
| torchtitan | **turns off** FSDP's averaging | `distributed/fsdp.py:84-97`: `set_gradient_divide_factor(1.0)`, *"We handle gradient scaling ourselves in the training loop with global token count"* |
| Megatron (flag) | turns off DDP's averaging | `core/distributed/distributed_data_parallel.py:250-254`: `gradient_scaling_factor = 1.0` |

**Summary:**
- **Method 1 for pretraining is fine.** That is why nanoGPT, Megatron's default and today's BluTrain are all correct for pretraining.
- **torchtitan and Hugging Face use Method 2 for everything.** Megatron uses it only with the flag; nanoGPT never does.
- **Our arrangement** (pretraining unchanged, SFT with the global count) matches **Megatron's** (default vs flag).
- **Our way of dividing** (each micro-step divides its loss by the global count, seed = W) matches **torchtitan and Hugging Face**.

---

## 10. The NaN / Inf checks: what they look at, and where a run stops

### 10.1 The order of one training step

1. Zero the gradients (`bluscript_zero_ckpt.cpp:2155`).
2. Run the GA micro-steps: next batch, forward, loss, backward (`:2160-2213`).
3. Average the logged loss over the micro-steps and the GPUs (`:2216-2224`).
4. **Check 1, the loss number** (`:2270-2274`). On NaN/Inf it prints `ERROR: NaN/Inf loss at step s` and calls `MPI_Abort(…, 2)`: **exit code 2**.
5. Only with `BLU_GLOBAL_CLIP=1`: the clip across groups (`:2331-…`).
6. For each optimizer group, `opt->step()` (`:2410`):
   1. combine the GPUs' gradients (ZeRO);
   2. compute the grad norm: one number, the same on every GPU;
   3. **Check 2, F1:** if the norm is NaN or Inf, throw `ZeroNonFiniteGradError`. The throw is at `ShardedOptimizer.cpp:1285` (`:1217` on the global-clip path), and the same check is in `SymmetricShardedOptimizer.cpp:1045` and `ShardedMuon.cpp:1160`, `:1225`. The trainer catches it at `:2411-2418` and calls `MPI_Abort(…, 4)`: **exit code 4**;
   4. otherwise: clip, the AdamW/Muon update, and the all-gather of the new weights.
7. Save a checkpoint every `ckpt_freq` steps (`:2566-2591`).

### 10.2 The answer: F1 checks the gradient norm, not the loss

- **Check 1** (old) looks at the final loss number.
- **Check 2** (F1, the fix) looks at the **gradient norm**. That is the one number `step()` already computes for clipping, so checking it costs one `isfinite` per group per step.
- Check 2 runs **before** anything touches the weights, m or v.

### 10.3 Why the loss check is not enough

- The loss is made in the **forward**.
- A NaN can appear only in the **backward**, for example an overflow inside one layer's gradient.
- Then the loss is finite and check 1 passes, but the gradients are broken.

### 10.4 What happened before F1

From the code comment at `ShardedOptimizer.cpp:1262-1283`, and the H200 test on 2026-09-25 (v1, the default optimizer):

1. The grad norm became NaN (or Inf).
2. The log printed that norm as **0**, which hid it.
3. The clip factor: for NaN it came out as **1.0** ("don't clip"); for Inf it came out as 0, and Inf × 0 = NaN.
4. So the gradients went into AdamW unchanged (or became NaN).
5. AdamW wrote NaN into the fp32 master weights and into m and v. The all-gather then copied the NaN weights into the model.
6. If it was a checkpoint step, the save at the end of the step **wrote the NaN weights to disk**, and auto-resume would load them again.
7. The next step's forward used the NaN weights, so the logits were NaN and the loss was NaN. Check 1 then stopped the run with exit code 2, **one step late**.

**Test (H200, 2026-09-25):** a NaN put into the gradient at step 2 made the run print `NaN/Inf loss at step 3`, with the grad norm printed as `0.0000`.

**The other optimizers before F1** (standalone test, 2026-09-26):
- **v2:** the same; NaN went into the weights and m/v.
- **Muon, 1-D tensors:** NaN went into the weights.
- **Muon, 2-D matrices: silent.**
  - The NaN went into the momentum first, and then Muon's own check rejected the update.
  - The weights stayed finite, so the loss stayed finite, and **check 1 never fired**.
  - But the momentum kept the NaN, so every later update of that matrix was rejected too. The matrix **stopped learning, with no error, for the rest of the run**.
- `BLU_GLOBAL_CLIP=1` did not help.

### 10.5 "If it is not caught here, where does it stop?"

**Before F1:**
- **v1, v2 and Muon 1-D tensors:**
  - the NaN went **through the weight update**;
  - then **the next forward** turned it into a NaN loss;
  - then **check 1 of the next step** stopped the run (exit code 2).
- **Muon 2-D matrices:** **never**, as explained in 10.4.

### 10.6 Now (after F1)

- The run stops **at the same step**, before AdamW/Muon touches the weights, m or v, and before any checkpoint save. It exits with code 4 and one clear line: `[FATAL] rank r step s: … -- stopped before the weight update; no checkpoint written`.
- **Verified:**
  - H200, 2026-09-25: a NaN (T2b) and an Inf (T2c) both stop at step 2 with exit code 4, and finite steps are bit-identical to before;
  - 2026-09-26: v2 + Muon pass 36/36 cases.
- **Commits:** `a264586` (v1, 09-25) and `23ae1e5` (v2 + Muon, 09-26). Both are in `dist_side_merge` (ancestors of `909ab75`).
- With a LossScaler (fp16 loss scaling), the step is skipped instead. None is installed in our trainer.
- **Evidence:**
  - `profiling_reports/fix2_2026_09_25/` (audit);
  - `profiling_reports/f1_muon_v2_2026_09_26/` (test, patch, logs);
  - `loss_kernel_dtype_bugs.md` §19.7.

### 10.7 SFT does not change these checks

- Neither check depends on how the loss is averaged.
- The SFT logged loss must still be fixed (task 6, #5), because it is the number people read. But check 1 only asks "is it NaN/Inf?", so it works either way.

---

## 11. DFT and focal loss (future tasks)

### 11.1 What they are

| Loss | Per token | What it does | Paper |
|---|---|---|---|
| **plain cross-entropy** (our v1) | −log p | every token gets the same multiplier | — |
| **focal loss** (γ = 2) | −(1−p)^γ · log p | *"down-weights the loss assigned to well-classified examples"*, so hard tokens dominate. It was made for object detection, where *"the vast number of easy negatives"* would swamp training | Lin et al. 2017, arXiv 1708.02002 |
| **DFT** (Dynamic Fine-Tuning) | −sg(p) · log p. **sg** = stop gradient: p is used as a fixed number in the backward | *"rescaling the objective function with the probability of this token"*. **The opposite of focal:** tokens the model finds unlikely count **less**. The paper argues plain SFT's gradient *"implicitly encode[s] a problematic reward structure"* that hurts generalization, and reports better math and code results from *"a single-line change"* | Wu et al. 2025, arXiv 2508.05629 (code: github.com/yongliang-wu/DFT) |

### 11.2 Loss values: easy vs hard tokens

| | p = 0.9 (easy) | p = 0.1 (hard) | p = 0.01 (very hard) | hard ÷ easy |
|---|---|---|---|---|
| plain | 0.105 | 2.303 | 4.605 | **22×** |
| focal (γ = 2) | 0.00105 | 1.865 | 4.514 | **1,770×** |
| DFT | 0.0948 | 0.230 | 0.0461 | **2.4×** |

### 11.3 The push (the gradient on the right token's logit, before × seed/N)

| | p = 0.9 | p = 0.1 | p = 0.01 | hard (0.1) ÷ easy (0.9) |
|---|---|---|---|---|
| plain: (1 − p) | 0.100 | 0.900 | 0.990 | **9×** |
| focal (γ = 2) | 0.0029 | 1.102 | 1.061 | **380×** |
| DFT: p(1 − p) | 0.090 | 0.090 | 0.0099 | **1×** |

These numbers were computed on 2026-10-06 with a small Python check: finite differences for plain and focal, and the exact p(1 − p) for DFT.
- **Plain** pushes a hard token 9× more than an easy one.
- **Focal** pushes it **380×** more.
- **DFT** pushes them **equally**, and pushes a very unlikely token (p = 0.01) much **less**.

### 11.4 With the global denominator: same N, different weights

| | step loss |
|---|---|
| ours (plain) | Σ ( −log p ) ÷ N |
| focal | Σ ( (1−p)² · −log p ) ÷ N |
| DFT | Σ ( p · −log p ) ÷ N, with p treated as a constant in the backward |

- The ÷ N is the same for all three: the same global count, the same seed W, the same trainer code.
- The per-token weight is different, so the gradients are different, so the training is different.
- Only the loss kernel changes:
  - **DFT:** the forward multiplies each token loss by p, and the backward multiplies each push by p;
  - **focal:** the weight also has a gradient, so its push follows its own formula (11.3).

### 11.5 "If ours is correct, how can they be correct too?"

They answer **different questions**.
- **Our global average is correct** in the sense that it computes exactly the objective we chose: plain cross-entropy, every token equal, **without the accidental micro-batch bias**. Plain cross-entropy means "make the training answers as likely as possible". Tülu 3, Hugging Face and torchtitan all use it.
- **Focal and DFT deliberately choose a different objective.** That is a design choice with a goal: focal focuses on hard tokens, DFT aims for better generalization.
- They even pull in **opposite directions**. So they are trade-offs to test, not one "right answer".

### 11.6 How to test them

| What we want to know | How | Needs the training script? |
|---|---|---|
| **Is the implementation right?** (loss values and gradients match the formula; ignored tokens get 0; the global N is used) | a **standalone test**: random logits and labels, comparing the GPU kernel with a simple CPU reference. The same kind as our 136-case loss test; fine on the laptop GPU | ❌ no |
| **Does it give a better model?** | real **SFT runs** with the training script: the same data, LR and steps, comparing held-out loss and benchmarks for plain vs DFT vs focal | ✅ yes, nothing replaces this |

Standalone tests prove they are **built correctly**; training runs prove whether they **help**. Both come after v1 SFT works.

---

## 12. Open items found while reading

| # | Item | Effect | When |
|---|---|---|---|
| 1 | The trainer does not pass `65535` / `&denom` yet, and the seed is still 1/GA | SFT would use Method 1, and train on prompt tokens | **task 6** (section 8) |
| 2 | `SftDataLoader` + the default −1 = out-of-bounds read of `row_logits[65535]` | garbage loss and broken gradients | change both **together** in task 6 |
| 3 | The B300 (sm_103) forward ignores the ignore value (`LossKernels.cu:938-941`) | wrong SFT loss on a B300 | before any SFT run on a B300 |
| 4 | The forward's second reduction stage has 256 threads, but loads one partial sum per thread (`LossKernels.cu:953-967`, `:472-488`). So for B × T > 65,536 (more than 256 partials), the partials past 256 are dropped. Today B × T = 32,768 (128 partials), so this is fine | the loss **value** only; gradients are not affected | only if B × T ever exceeds 65,536. Found by reading; not run |
| 5 | The validation loss uses Method 1 (`:2102-2106`) | a biased SFT val loss | task 6 (section 8, #6) |

**Status on 2026-11-19:**

- #1 and #2: done together in the SFT trainer (`train/production_zero/bluscript_sft.cpp`): the loss gets label 65535
  and `sft_denom`, the step's trained-token count over all micro-batches and ranks; the backward seed is W.
- #3: still open. B300 does not take packed SFT rows at all (its causal attention has head size 64 only), so no SFT
  run goes there today (`sft/README.md` section 9).
- #4: fixed on 2026-10-30 (section 12b).
- #5: done: SFT validation adds all val losses and divides once by the val set's trained-token count
  (`sft/README.md` section 5, "SFT: the whole val set").

---

## 12b. The loss sum stopped at 65,536 rows (found and fixed 2026-10-30; pretraining and SFT)

**What the forward does.** The cross-entropy forward first writes one loss per row (one row = one token position;
ignored rows write 0). Then `sum_reduction_kernel` adds them in two launches:
1. one block per 256 rows writes one partial sum per block;
2. ONE block of 256 threads adds the partials.

**The bug.** In step 2 each thread read exactly one partial, so only partials 0-255 were added: 256 x 256 = 65,536
rows. The denominator (the count of trained rows) counted all rows. So the printed loss read about
true x 65,536 / N:

| rows in one loss call (B x T per GPU) | printed loss / true loss |
| --- | --- |
| up to 65,536 | 1 (correct) |
| 81,920 | 0.80 |
| 131,072 | 0.50 |
| 1,048,576 | 0.06 |

**What it did not touch.** The gradients: the backward uses the per-row max and sum saved by the forward and the
denominator, never the summed loss. So the weights trained correctly; only the printed train / val loss was wrong.
Checked in the trainer: the step-0 grad norm at 81,920 tokens was the same with and without the fix (2.331747 vs
2.331774, inside the run-to-run spread).

**Who it hit.** It was in the library since at least 2026-04, so pretraining too: any run with more than 65,536
tokens per GPU in one loss call (long context, or a large B x T). In our checks: Adhitya's long-context training
runs stayed at or below 32,768 tokens per call; one old LongRoPE search at 131,072 printed PPL 7.55 where the truth
was 59.87 (it had been blamed on a "2^32 elements" limit).

**The fix.** In step 2 each thread loops over the partials (64-bit index) when the grid does not cover all rows;
otherwise it keeps the single load. So every sum up to 65,536 rows is bit-identical to before. Chosen after measuring
ten ways on an H200 (the old kernel; an always-on loop; PyTorch's one-block loop; multi-pass; atomicAdd; CUB; a
last-block finish; a torch.sum-like reduction): the only one with today's bits, the same result every run, no new
memory or state, and today's GPU time. Results: the test (`Tests/kernel_tests/test_sparse_ce_ignore_index.cu`, group
H) 154/154 (16 failures before); the loss bits identical up to 65,536 rows; GPU memory and compute-sanitizer the same
as before; in the trainer at 81,920 tokens per micro-batch the val loss went from 1.300650 (wrong) to 1.606812, equal
to the numpy reference 1.606273. Full record: SFT work log §45.108 parts 12-16.

## 13. Sources

- **BluTrain** `dist_side_merge` `909ab75`:
  - `bluscript_zero_ckpt.cpp`;
  - `dist/zero/src/ShardedOptimizer.cpp`, `SymmetricShardedOptimizer.cpp`, `ShardedMuon.cpp`;
  - `sft/SftDataLoader.h` (new, not committed yet).
- **Tensor-Implementations** `d96b991`: `LossOps.h`, `LossOps.cpp`, `LossBackward.cpp`, `LossKernels.cu`.
- **Data:** `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/train/valid_counts.bin` and `meta.json` (built 2026-10-06 by `sft/prepare_sft_data.py`).
- **Frameworks:**
  - torchtitan `1e4d33d`, Megatron-LM `8b2f660`, TRL `478624d` (laptop clones);
  - Hugging Face Transformers `main`, nanoGPT `master`, PyTorch `main` (GitHub, 2026-10-06).
- **Papers:**
  - Lin et al. 2017, *Focal Loss for Dense Object Detection*, arXiv 1708.02002;
  - Wu et al. 2025, *On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification*, arXiv 2508.05629.
- **F1 evidence:**
  - `profiling_reports/fix2_2026_09_25/`, `profiling_reports/f1_muon_v2_2026_09_26/`;
  - `loss_kernel_dtype_bugs.md` §19.7;
  - commits `a264586`, `23ae1e5`.
