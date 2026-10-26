# Hyperparameters and configs: how to select them for model training, and what affects them

Written 2026-10-23. It collects every doubt I had about the batch size B, rows per step, GRAD_ACCUM, ranks, GPU memory,
and how to choose these settings, with the answers, in the order they came up (sections 2–7), plus the background I
did not ask about but need (sections 1, 8–10). Section 11 is the short summary to remember.

Every doubt section starts with a **Context** paragraph: what we were doing and what made me ask.

**The run these doubts are about, in one paragraph.** On 2026-10-22/23 we trained our 1.7B model (`fullchess`, 28
layers) with SFT on the Tülu 3 chat data, on **one H200 GPU**. The data was 741,413 chats, packed into 115,904 rows of
4,096 tokens (about 6.4 chats per row). The run used **B = 4 rows per pass, 5 passes per optimizer step = 20 rows per
step**, Muon with learning rate 5e-5, 2 epochs = 11,590 steps. The memory peak printed at the end of the log was
**74,064 MiB** (`[MEM] active_peak 74064 MiB`). The doubts came up while I was preparing the handover for the DPO team and
read "B 4 x GRAD_ACCUM 5" in the settings.

---

## 1. Parameters vs hyperparameters (the words first)

### 1.1 Parameters = the numbers the model LEARNS

- The model is **1,701,000,000 numbers** (1.701B): the weights of every layer (wQ, wK, wV, wO, gate_up, down, the
  norms, the embedding).
- **Training changes these numbers.** Every optimizer step moves each of them a little, so that the correct next tokens
  get higher probability.
- Example: one single parameter is one number inside the matrix wQ of layer 5, say `0.0132`. After a step it may be
  `0.0131`. Nobody chooses that value by hand; the training finds it.
- These are what a checkpoint is mainly for, and what DPO will continue to change.

### 1.2 Hyperparameters = the settings WE CHOOSE, which training does NOT change

- They are fixed **before** the run and stay the same while it runs (or follow a schedule we fixed in advance, like the
  learning rate going up and then down).
- Example: the learning rate `5e-5`, the batch size `20 rows per step`, `2 epochs`. The run never changes them by
  itself.
- They decide **how** the parameters are learned.

### 1.3 A third kind: optimizer state (not parameters, not hyperparameters)

- The optimizer keeps its own helper numbers: **Muon's momentum** (a running average of past gradients, one number per
  weight), **AdamW's two moments**, and the **fp32 master copy** of the weights.
- They change during training (like parameters), but they are **not part of the model**: you cannot run the model with
  them; they only help the optimizer make its steps.
- That is why our checkpoint is **17.5 GB** although the model's weights are only **3.4 GB**: the rest is optimizer state
  (fp32 master 6.8 GB + Muon momentum 6.3 GB + AdamW moments 0.9 GB). DPO starts from the weights only (`BLU_INIT_FROM`
  ignores the optimizer state), with a fresh optimizer.

### 1.4 So: are B, rows per step, GRAD_ACCUM and the learning rate hyperparameters?

Yes, all of them are chosen by us and not learned. But they are of **two different kinds**, and the difference matters
for how to choose them:

| Kind | Examples | Changes WHAT the model learns? | Chosen by |
| --- | --- | --- | --- |
| **Learning hyperparameters** | batch size per optimizer step (chats/rows/tokens), learning rate, its warm-up and decay, epochs, weight decay, grad clip, optimizer momentum | **yes** | a proven recipe or experiments (sweeps) |
| **System (hardware) settings** | B (rows per pass), GRAD_ACCUM (passes per step), number of GPUs (ranks), precision, how often to validate or save | **no**: they change speed, memory and logging, not the result | the GPU memory and arithmetic |

Example: **B 4 × 5 passes** and **B 5 × 4 passes** both give a step of 20 rows. The model learns the same thing (up to
tiny rounding differences); only the memory and speed differ. But **20 rows per step** vs **40 rows per step** is a
different training: half as many weight updates, each averaged over twice as many chats.

### 1.5 The architecture settings are hyperparameters too, but fixed for us

The number of layers (28), the width (2,304), the heads (18 query / 6 key-value, size 128), the MLP size (6,144), the
vocabulary (50,304), RoPE theta (50,000) and the context (4,096) were chosen when the model was **pretrained**. For SFT and
DPO we must use exactly the same values: the weights only fit that shape. (A wrong layer count stops with a shape
error; a wrong RoPE theta gives no error but a worse model.)

---

## 2. Doubt: "We tested B = 4, not 8? Can't we do 8 on one GPU? Why not? How do we calculate whether the memory is enough?"

**Context.** 2026-10-23, while preparing the DPO handover. The settings said `B 4 x GRAD_ACCUM 5` and I wondered why the
run used only 4 rows per pass on an H200, whether 8 would fit on one GPU, and how to calculate the memory limit ("storage
insufficiency").

### 2.1 The answer as given

We trained with **B = 4** (4 rows of 4,096 tokens per GPU pass). Two separate reasons it was not 8, one math, one memory:

**1. The math reason: B must divide the rows per step.** One step = **20 rows**, done as **B 4 × 5 passes** (gradient
accumulation). B = 8 would need 2.5 passes, which is impossible, so B = 8 means 16 or 24 rows per step. That changes the
recipe slightly (chats per step, number of steps). The result is mathematically the same kind of training (accumulation
just adds the passes up).

**2. The memory reason, and how to calculate it.** "Insufficiency" here means **GPU memory** (the H200's HBM), not disk.
The rule:

```
peak GPU memory  ≈  fixed part  +  B × (memory per row)
```

**Fixed part** (does not depend on B), taken from our own checkpoint:

| Item | Size |
| --- | --- |
| bf16 weights (1.70B × 2 bytes) | 3.4 GB |
| fp32 master weights (1.70B × 4) | 6.8 GB |
| Muon momentum fp32 (1.585B × 4) | 6.3 GB |
| AdamW moments (0.116B × 2 × 4) | 0.9 GB |
| *= the checkpoint size* | *17.5 GB* |
| + bf16 gradients (1.70B × 2) | 3.4 GB |
| **Fixed total** | **≈ 21 GB** |

**Per row** (activations kept for the backward pass: 28 layers × 4,096 tokens, plus the logits 4,096 × 50,304): our run's
**measured** peak at B = 4 was **74.1 GB** (log: `[MEM] active_peak 74064 MiB`):

```
per row ≈ (74.1 − 21) / 4 ≈ 13.3 GB
```

**Prediction:**

| B | 21 + B × 13.3 | Fits in an H200 (≈140 GB usable)? |
| --- | --- | --- |
| 4 | 74 GB | yes (measured) |
| **8** | **≈ 127 GB** | **yes, with ~13 GB spare (tight but OK)** |
| 9 | ≈ 141 GB | borderline |
| 16 | ≈ 234 GB | no |

**So B = 8 should fit on one H200**, but it is a calculation, not tested. The proof would be a short test (a few steps with
`BLU_SFT_B=8`, `BLU_SFT_ROWS_PER_STEP=16` or 24) and reading the `[MEM]` peak line. Speed-wise B = 8 would gain only a
little: at B = 4 each pass already handles 16k tokens and the GPU runs at 52% MFU.

### 2.2 The same calculation in exact units

The answer above wrote 74,064 MiB as "74.1 GB". MiB and GB are different units (1 MiB = 1,048,576 bytes, 1 GB =
1,000,000,000 bytes), so here is the same calculation in one unit, GiB (1 GiB = 1,024 MiB):

| | GiB |
| --- | --- |
| H200 memory (nvidia-smi: 143,771 MiB) | 140.4 |
| Our measured peak at B = 4 (74,064 MiB) | 72.3 |
| Fixed part: checkpoint 17,479,868,928 bytes = 16.3 GiB, + bf16 gradients 3.2 GiB | 19.5 |
| Per row: (72.3 − 19.5) / 4 | 13.2 |
| **B = 8: 19.5 + 8 × 13.2** | **125.1 (15.3 GiB spare)** |
| B = 9: 19.5 + 9 × 13.2 | 138.3 (only 2 GiB spare: too tight) |
| B = 10: 19.5 + 10 × 13.2 | 151.5 (does not fit) |

The conclusion is the same: **B = 8 fits, B = 9 is the edge, B = 10 does not.** (If the gradients were kept in fp32
instead of bf16, the fixed part would be 22.6 GiB and the per-row share 12.4 GiB; B = 8 would then need 121.8 GiB. Same
conclusion.)

### 2.3 More explanation: why memory grows with B and what the "per row" part is

- The **fixed part** is the same whatever B is: one copy of the weights, the gradients and the optimizer state.
- The **per-row part** is what the forward pass must **keep until the backward pass**: for every layer, the inputs it
  needs to compute its gradients (normalized inputs, q/k/v, attention outputs, the MLP's inner values), for every one of
  the 4,096 tokens of every row; plus the logits (4,096 × 50,304 numbers per row). Twice the rows = twice of all this.
- That is why the formula is "fixed + B × per row": a straight line in B. Measure one point (our B = 4 peak), know the
  fixed part (from the checkpoint size), and the line gives every other B.
- Leave spare room: memory is not used perfectly (the allocator reserves a bit more than it hands out; our log shows
  `reserved_peak 75728 MiB` vs `active_peak 74064 MiB`). That is why B = 9 (2 GiB spare) counts as too tight.
- A calculation is not a proof. The proof is to run a few steps with that B and read `[MEM] active_peak` in the log (or
  watch `nvidia-smi`).

---

## 3. Doubt: "Then why did we select B = 4, only 5 passes per step, 20 rows per step? Why that many chats per step? Is it an optimal number, or based on the total number of chats? Why not 4 × 6, or 4 × another number? 20 chats per micro-step?"

**Context.** 2026-10-23, right after section 2. Once B = 8 looked possible, I wanted to know where the numbers 4, 5 and 20
came from at all, whether 20 is special, and whether it was calculated from our number of chats.

### 3.1 The answer as given

First a small correction: it is **20 rows per optimizer step**, not 20 chats, and not per micro-step:

```
1 micro-step  = B = 4 rows    (one pass through the GPU)
1 step        = 5 micro-steps = 20 rows   (then the weights are updated once)
1 row         = 4,096 tokens holding about 6.4 packed chats (741,413 chats ÷ 115,904 rows)
→ 1 step ≈ 20 × 6.4 ≈ 128 chats
```

**Why 128 chats per step:** it was not calculated from our data size. **We copied the batch size of the published Tülu 3
SFT recipe: 128 chats per optimizer step.** Then:
- rows per step = 128 ÷ 6.4 chats per row ≈ **20**;
- **B = 4** was the largest per-pass size we had already **measured** to fit (74 GB);
- **5 passes** = 20 ÷ 4.

The total number of chats only decides **how many steps** an epoch has: 741,413 ÷ 128 ≈ **5,795 steps per epoch**. It does
not decide the batch size.

**Is 128 optimal? Not tuned for our model.** It is the standard recipe value, not something we tested against others.
Other choices would work too:

| Choice | Rows/step | Chats/step | Steps per epoch | Effect |
| --- | --- | --- | --- | --- |
| 4 × 4 | 16 | ~102 | ~7,244 | more, noisier updates |
| **4 × 5 (ours)** | **20** | **~128** | **5,795** | **Tülu 3's value** |
| 4 × 6 | 24 | ~154 | ~4,829 | fewer, smoother updates |

- **The total time per epoch is the same** either way: the same tokens.
- What changes is how often and how "averaged" the weight updates are. **Batch size and learning rate go together**:
  changing one usually means re-tuning the other. Our lr 5e-5 came from the 10-07 sweep (at 64 chats per step) and
  Moonlight's Muon SFT recipe, not from a sweep at 128. So 4 × 6 or 4 × 4 would be valid experiments, but they would need
  their own lr check.

**Why 4 × 5 and not another split of 20.** B must divide 20 evenly: 1, 2, 4, 5, 10 or 20.
- **B = 10 or 20** do not fit in memory (≈154 GB and more).
- **B = 5** (5 × 4) would also have fit (≈ 21 + 5 × 13.3 ≈ 88 GB) at about the same speed.
- **B = 4** was simply the setting already tested and measured (74 GB) in the checks before the run.

### 3.2 More explanation: what "more, noisier updates" and "fewer, smoother updates" mean

- One optimizer step uses the **average gradient** of all the chats in that step. The gradient of one chat points the
  weights towards "better on this chat"; the average over many chats points towards "better on chats in general".
- **Fewer chats per step** (e.g. 102): the average is over fewer chats, so it wobbles more from step to step ("noisier"),
  but there are **more** steps (7,244 per epoch), so the weights get updated more often.
- **More chats per step** (e.g. 154): a steadier average ("smoother"), but **fewer** updates (4,829 per epoch).
- Neither is "right" in general; it is a trade-off, and the best value depends on the learning rate. That is why recipes
  fix both together, and why changing one alone is an experiment, not a free change.

### 3.3 More explanation: why the total number of chats does not set the batch

- The batch size is about **how each update is made** (how many chats are averaged). That is the same question whether
  the dataset has 10,000 or 1,000,000 chats.
- The dataset size only sets **how many** updates you get: `steps per epoch = chats ÷ chats per step`.
- The one place the dataset size matters: the number of steps must not become **too small**. The learning rate warms up
  over the first 3% of the steps and then decays; with very few steps (for example a small 5,000-chat dataset packed
  into 64-row steps would give only ~24 steps per epoch) there is no room for that, and each step is a big jump.

---

## 4. Doubt: "B = 8 fits too, right? Then 128 chats per step at 6.4 × 8 = 51.2 chats per pass is 2.5 passes. What does the 0.5 pass mean? Like 8, 8, 4 = 20 rows, the third pass takes only 4 rows?"

**Context.** 2026-10-23, after section 3. If B = 8 fits, I wanted to keep 128 chats per step with it, and the division gave
2.5 passes. I asked how a half pass would work, guessing the third pass would just take the remaining 4 rows.

### 4.1 The answer as given

The math is right: with B = 8, one pass is 8 rows ≈ **51 chats**, and 20 rows would need **2.5 passes**.

**There is no half pass.** Our trainer **refuses to start** when the rows per step are not a whole multiple of B. From the
code (`bluscript_sft.cpp:1007-1009`):

```
if (rows_per_step % rows_per_micro != 0)  →  "ERROR: BLU_SFT_ROWS_PER_STEP (20) must be a positive multiple of B x ranks"
GRAD_ACCUM = rows_per_step / rows_per_micro      (whole number only)
```

So with B = 8 there are only two choices:

| Setting | Passes | Rows / step | Chats / step | Steps / epoch |
| --- | --- | --- | --- | --- |
| B 8 × 2 | 2 | 16 | ~102 | ~7,244 |
| B 8 × 3 | 3 | 24 | ~154 | ~4,829 |
| *ours: B 4 × 5* | *5* | *20* | *~128* | *5,795* |

**The "8, 8, 4" idea** (a smaller last pass) would be mathematically fine: our loss divides by the step's **total**
answer-token count, so uneven passes still give the correct average gradient. But our trainer only supports **equal**
passes: every pass has the same fixed shape [B, 4,096], with its buffers sized for that. It would need a code change and
would not gain anything, so it was never built.

One more detail: **6.4 chats per row is only an average.** A step is defined in **rows**, so the number of chats (and answer
tokens) per step varies: in the log, each step trains between about 49k and 62k answer tokens.

### 4.2 More explanation: why uneven passes would still be correct (and why that depends on the loss)

- Our loss for a step = (sum of the token losses of all answer tokens in all passes of the step) ÷ (total answer tokens
  of the step). That is one average over the whole step, whatever the passes look like.
- So 8 + 8 + 4 rows would give exactly the same gradient as 20 rows in one pass. The only reason it is not allowed is
  the program's fixed shapes, not the math.
- (A trainer that averaged **each pass on its own** and then averaged the passes would give a small pass the same weight
  as a big one; uneven passes would then be wrong. Ours does not do that.)

---

## 5. Doubt: "So the number of chats per step should be a multiple of (B × chats per row)? What are ranks, B × ranks? And GRAD_ACCUM means the number of micro-steps per one full optimizer step?"

**Context.** 2026-10-23, after section 4. The trainer's error message says "multiple of B x ranks", and I wanted the exact
rule, the meaning of "ranks", and to confirm what GRAD_ACCUM is.

### 5.1 The answer as given

**1. The rule is in rows, not chats.**

```
rows per step  =  B  ×  ranks  ×  GRAD_ACCUM
ours:    20    =  4  ×    1    ×     5
```

Chats per row is not a fixed number (6.4 is only the average: some rows hold 2 long chats, others 15 short ones), so the
trainer counts **rows**, and the chats per step just follow (about 20 × 6.4 ≈ 128, varying a little from step to step).

**2. A rank = one GPU training process.** With multi-GPU training, one copy of the trainer runs on each GPU; each copy is
called a rank.
- **Ours: 1 GPU, so ranks = 1.**
- In every pass, **each rank takes its own B rows at the same time**, so one pass covers **B × ranks** rows:

| GPUs (ranks) | B | Rows per pass | 20 rows per step allowed? |
| --- | --- | --- | --- |
| 1 (ours) | 4 | 4 | yes, 5 passes |
| 2 | 4 | 8 | no (20 ÷ 8 = 2.5) → use 16 or 24 |
| 4 | 4 | 16 | no → use 16 (1 pass) or 32 (2 passes) |
| 4 | 5 | 20 | yes, 1 pass |

**3. GRAD_ACCUM = the number of passes (micro-steps) per optimizer step. Yes.** "Gradient accumulation":
1. pass 1: compute the gradients of its rows and **add** them into a gradient buffer;
2. passes 2, 3, 4, 5: the same, each **adding** to the buffer;
3. after the last pass: the optimizer updates the weights **once**, using the total; then the buffer is cleared.

So 5 passes of 4 rows give exactly the same update as 1 pass of 20 rows. You just need memory for only 4 rows at a time.

### 5.2 More explanation: what happens with several GPUs in one step

With 4 GPUs, B = 5, GRAD_ACCUM = 1 (the 20-row step done in one pass):
1. Each of the 4 GPUs takes its own 5 rows (different rows: GPU 0 rows 0–4, GPU 1 rows 5–9, and so on).
2. Each computes the gradients of its 5 rows.
3. The GPUs **add their gradients together** over the network (an "all-reduce"), so every GPU now has the gradient of
   all 20 rows.
4. Each GPU makes the same weight update, so all copies of the model stay identical.

The training is the same as our 1-GPU run (20 rows per step); it just finishes each step faster. That is the reason to
keep **rows per step** the same when the number of GPUs changes: then the learning rate can stay too.

### 5.3 Words used in this document

| Word | Meaning |
| --- | --- |
| **row** | one training sequence of 4,096 tokens; with packing it holds several whole chats |
| **B** | rows per pass on ONE GPU (also called the micro-batch size) |
| **micro-step / pass** | one forward + backward over B rows on every GPU; adds gradients, does not update weights |
| **rank** | one GPU's training process; ranks = number of GPUs |
| **GRAD_ACCUM** | passes per optimizer step |
| **optimizer step** | the moment the weights change: once per GRAD_ACCUM passes |
| **rows per step** | B × ranks × GRAD_ACCUM (ours 20) — the batch size in rows |
| **chats per step** | rows per step × chats per row (ours ≈ 128) — varies a little per step |
| **trained tokens per step** | the answer tokens in those rows (ours ≈ 57k, from 49k to 62k) |
| **epoch** | one pass over all training rows (ours 5,795.2 steps) |

---

## 6. Doubt: "What should we select first, and based on what? The number of chats, then the rows after preprocessing, then B and the passes? But if we fix the chats per step first, we can't know whether it suits B. So what do we select first? Explain in detail."

**Context.** 2026-10-23, after section 5. I understood the pieces but not the order: whether to start from the data
(chats → rows), from B, or from the chats per step, given that a chats-per-step number fixed first might not divide by B.

### 6.1 The answer as given

The order matters. The rule: **first decide the training batch (it changes how the model learns), then pick B to fit the
GPU (it only changes speed and memory), then round the batch to fit B.** The batch can bend by about ±20%; the GPU memory
cannot.

**Step 1: choose the batch: how many chats (≈ tokens) per optimizer step.** This is a **learning** choice, not a hardware
one. Take it from a proven recipe or from your own experiments. For us: **Tülu 3's 128 chats per step**. It is **not**
calculated from the total chats; the total only gives how many steps you will have:

```
steps per epoch = total chats ÷ chats per step
ours: 741,413 ÷ 128 ≈ 5,795
```

so check the result is not too small (learning-rate warm-up and decay need room; e.g. on 10-07, 5,000 chats packed into
64-row steps would have left only ~24 steps per epoch — far too few).

**Step 2: turn chats into rows (after preprocessing).**

```
chats per row  = total chats ÷ total rows       = 741,413 ÷ 115,904 = 6.4   (from the packed data)
rows per step  ≈ chats per step ÷ chats per row = 128 ÷ 6.4 ≈ 20
```

**Step 3: choose B: the largest that fits in GPU memory.** This is a **speed** choice; it does not change what the model
learns. Measure it, or use the formula from section 2:

```
21 GB (fixed)  +  B × 13.3 GB  ≤  ~130 GB (leave some spare)   →   B ≤ 8
```

**Step 4: round the rows to a multiple of B × GPUs.**

```
rows per pass = B × ranks (GPUs)
GRAD_ACCUM    = rows per step ÷ rows per pass      (must be a whole number)
```

If 20 does not divide evenly, either choose a smaller B that divides it, or take the nearest multiple: batch size is not
exact magic — 102, 128 or 154 chats are all "about 128".

**Step 5: if the batch moved a lot, re-check the learning rate**, because the two are tuned together. Keep the batch about
the same, and the lr can stay.

**Our numbers, worked through:**

| GPUs | B | Rows per pass | Rows per step | GRAD_ACCUM | Chats per step | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 4 | 20 | 5 | ~128 | **what we ran**: exact, measured 74 GB |
| 1 | 5 | 5 | 20 | 4 | ~128 | exact too (~88 GB) |
| 1 | 8 | 8 | 16 or 24 | 2 or 3 | ~102 or ~154 | fits (~127 GB), but the batch moves ±20% |
| 4 | 5 | 20 | 20 | **1** | ~128 | exact, and one pass per step |
| 4 | 4 | 16 | 16 or 32 | 1 or 2 | ~102 or ~205 | 32 is too far from 128 |

**In one line: fix the batch first (~128 chats → 20 rows), then the biggest B that fits, then pick the B and GPU count
that divide 20 exactly; if none does, round the rows to the nearest multiple.**

### 6.2 More explanation: the same procedure for a NEW dataset, as a checklist

Say a new SFT dataset has 300,000 chats, and after packing into 4,096-token rows it gives 50,000 rows.

1. **Batch (learning choice):** keep the recipe's 128 chats per step (unless you have a reason to change it).
2. **Chats per row:** 300,000 ÷ 50,000 = **6.0** → **rows per step ≈ 128 ÷ 6.0 ≈ 21.3** → round to a number that will
   divide well later, e.g. 20 or 24.
3. **Steps per epoch:** 50,000 ÷ 20 = **2,500** (enough room for warm-up and decay; fine).
4. **B (memory):** same model and context as ours → B ≤ 8 on one H200. Pick a B that divides 20 → B = 4 (5 passes) or
   B = 5 (4 passes).
5. **GPUs:** with 4 GPUs → B = 5 gives 20 rows in one pass (GRAD_ACCUM = 1).
6. **Learning rate:** the batch stayed ≈128 chats → keep the same lr.
7. **Check:** run a few steps; read the memory peak and the step time in the log before starting the full run.

### 6.3 More explanation: chats or tokens as the batch unit?

- Recipes usually count the batch in **sequences** (chats). That is the unit Tülu 3 publishes (128).
- What the gradient really averages over is the **trained tokens** (answer tokens), because the loss is per token. Chats
  have very different lengths (ours: median 470 tokens, but up to 4,096), so "128 chats" is not a fixed amount of
  learning signal.
- With packing, a fixed number of rows per step gives a roughly fixed number of **positions** (20 × 4,096 = 81,920) and
  about 57k trained tokens per step (49k–62k). That is steadier than a fixed number of chats.
- So: choose the batch from the recipe in chats, convert it to rows once, and from then on think in rows / tokens.

---

## 7. Doubt: "Are these hyperparameters? Parameters that are trained vs settings that are not trained?"

**Context.** 2026-10-23, when asking for this document. I wanted to know the right name for these settings (B, the batch
size, the others), and whether "hyperparameters" are the ones we choose and "parameters" the ones that are trained.

### 7.1 The answer

Yes: **hyperparameters = chosen by us, not trained; parameters = the model's weights, trained.** The details, the third
kind (optimizer state) and the split into learning hyperparameters and system settings are in section 1.

### 7.2 All the hyperparameters of our SFT run, by kind

| Kind | Hyperparameter | Our value | Where it came from |
| --- | --- | --- | --- |
| Architecture (fixed by the base model) | layers | 28 | pretraining (`fullchess`) |
| | width / MLP size | 2,304 / 6,144 | pretraining |
| | heads | 18 query, 6 key-value, size 128 | pretraining |
| | vocabulary | 50,304 (GPT-2 50,257 padded) | pretraining |
| | RoPE theta | 50,000 | pretraining (must match) |
| | context length | 4,096 | pretraining |
| | norm eps | 1e-5 (blocks), 1e-6 (QK-norm) | pretraining |
| Data | max chat length | 4,096 tokens (longer chats dropped) | context length |
| | packing | First-Fit-Decreasing into 4,096-token rows | our packing study |
| | dropped part | Aya (multilingual) | base model is English-only |
| | val set | 2,000 held-out chats | our choice |
| Learning | batch | 20 rows ≈ 128 chats ≈ 57k answer tokens per step | Tülu 3 recipe (128 chats) |
| | epochs | 2 (11,590 steps) | Tülu 3 recipe |
| | optimizer | Muon (2-D weights) + AdamW (embedding, norms) | base model was pretrained with Muon |
| | learning rate | peak 5e-5 for both groups | 10-07 sweep + Moonlight's Muon SFT |
| | warm-up | 3% of the steps (348) | trainer default (as Tülu 3) |
| | decay | cosine down to 5e-6 (peak ÷ 10) | trainer default |
| | Muon momentum / Newton-Schulz steps | 0.95 / 5 | trainer default |
| | AdamW betas | 0.9 / 0.95 | trainer default |
| | weight decay | 0.1 (both) | trainer default |
| | gradient clip | 1.0 | trainer default |
| System | B (rows per pass) | 4 | measured to fit (74,064 MiB) |
| | GRAD_ACCUM | 5 | 20 ÷ 4 |
| | GPUs (ranks) | 1 | available hardware |
| | precision | all bf16, fp32 optimizer masters | `BLU_ALL_BF16=1` |
| | val every / checkpoint every | 200 / 1,000 steps | our choice |

---

## 8. Background I did not ask about: the learning rate and the batch size go together

- The learning rate says **how big each weight update is**. With a bigger batch, the averaged gradient is steadier, so a
  somewhat bigger step is often safe; with a smaller batch, a smaller step.
- Common rules of thumb when changing the batch size by a factor k: raise the learning rate by about k (linear rule)
  or by about √k (square-root rule, often used with Adam-type optimizers). They are **starting points, not guarantees**;
  the safe way is a short sweep.
- Example: going from 128 to 154 chats per step (k = 1.2) would suggest an lr between 5e-5 × 1.1 and 5e-5 × 1.2. Such a
  small change usually still works with the old lr; going from 128 to 512 (k = 4) would need a real check.
- What we know for our model: the 10-07 sweep (5,000 chats, 2 epochs, 64 chats per step) found that with Muon,
  lr 5e-5 kept improving in epoch 2, while 1.2e-4 overfitted in epoch 2. The Tülu run then used 5e-5 at 128 chats per
  step, and its val loss fell to the end (no overfitting).

---

## 9. Background: what the warm-up, the decay and the epochs do

- **Warm-up (first 3% = 348 steps):** the learning rate starts near 0 and rises linearly to the peak. At the start the
  optimizer's momentum is still empty and the model is meeting the chat format for the first time; full-size steps right
  away can make the loss jump.
- **Decay (cosine, from the peak at step 348 down to 5e-6 at the end):** smaller and smaller steps towards the end, so
  the weights settle instead of bouncing around.
- **Epochs (2):** how many times every training chat is seen. Our measurement on this run (work doc
  `overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md`, sections 15–17): epoch 2 gave a small
  real gain (val −0.0107) and about 0.06 nats of memorization, so a 3rd epoch would most likely add mostly memorization.

---

## 10. Background: the practical memory checklist before a run

1. **Fixed part:** the checkpoint size (weights + optimizer state) + the gradients (bf16: 2 bytes per parameter). For a
   1.7B model with Muon + AdamW and fp32 masters: ≈ 19.5 GiB.
2. **Per row:** measure once (peak at a known B minus the fixed part, divided by B). Ours: ≈ 13.2 GiB per 4,096-token row.
3. **Largest B:** (GPU memory − fixed − ~10% spare) ÷ per row. H200: (140.4 − 19.5 − 14) ÷ 13.2 ≈ 8.
4. **What changes the per-row number:** a longer context (more tokens per row), a bigger vocabulary (bigger logits), more
   or wider layers, fp32 instead of bf16 activations, activation checkpointing (saves memory by recomputing in the
   backward pass, at some speed cost; our SFT loop does not use it).
5. **What changes the fixed part:** the optimizer (Muon keeps 1 extra copy, AdamW 2), fp32 masters, and multi-GPU
   sharding (ZeRO splits the optimizer state over the GPUs, so each GPU holds only its share).
6. **Prove it:** a few steps with the chosen B, then read `[MEM] active_peak` in the log or watch `nvidia-smi`.

---

## 11. Summary: what to remember

| Question | Answer |
| --- | --- |
| Parameters vs hyperparameters? | parameters = the 1.7B weights, **learned**; hyperparameters = settings **we choose** (lr, batch, epochs, B, …) |
| Is B a hyperparameter? | yes, but a **system** one: it changes speed and memory, not what is learned (with gradient accumulation) |
| What is the batch that matters for learning? | **rows (≈ chats ≈ tokens) per optimizer step**: ours 20 rows ≈ 128 chats ≈ 57k answer tokens |
| Where did 128 chats come from? | the Tülu 3 SFT recipe; not tuned, not from the dataset size |
| What does the dataset size decide? | only the number of steps: chats ÷ chats per step (ours 5,795 per epoch) |
| Rule that must hold | rows per step = B × ranks × GRAD_ACCUM (whole numbers) |
| GRAD_ACCUM? | passes per optimizer step; gradients are added, weights updated once |
| Ranks? | GPUs (one training process each) |
| Can B be 8 on one H200? | yes by calculation (≈125 GiB of 140), not tested; then 16 or 24 rows per step |
| Order of choosing | batch first (learning) → rows (from chats per row) → B (memory) → round rows to a multiple of B × GPUs → re-check lr if the batch moved a lot |

---

## 12. Sources

- Trainer rule and error: `~/Downloads/BluTrain_sft/bluscript_sft.cpp:1003-1019` (`rows_per_step % rows_per_micro`,
  `ga = rows_per_step / rows_per_micro`); SFT defaults (warm-up 3%, min lr = peak ÷ 10): lines 982–1026; optimizer
  defaults (grad clip 1.0, weight decay 0.1, Muon momentum 0.95, 5 Newton–Schulz steps, AdamW betas 0.9/0.95): lines
  233–267, 1323–1324.
- Memory peak: the run log `post_training_research/tulu3_sft_1022/tulu3_sft_1022_full_run.log`, last lines
  (`[MEM] active_peak 74064 MiB | reserved_peak 75728 MiB`); H200 total 143,771 MiB from `nvidia-smi`.
- Checkpoint size: `blumodel_step_11590/data_rank_0.bin` = 17,479,868,928 bytes.
- Data numbers: `sft_data/tulu3_noaya/train/meta.json` (741,413 chats, 115,904 rows, 325,732,264 trained tokens).
- The 10-07 learning-rate sweep: `sft_work_documentation.md` §37.5; the run's overfitting/memorization measurements:
  `overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md` sections 11–20.
