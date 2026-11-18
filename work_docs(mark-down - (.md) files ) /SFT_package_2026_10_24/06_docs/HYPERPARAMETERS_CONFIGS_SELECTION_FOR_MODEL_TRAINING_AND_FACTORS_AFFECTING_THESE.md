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

---

## 13. Doubt: "A colleague packs differently: she fixes the chats per optimizer step (128) first, packs THOSE chats into rows, and only then decides B and the passes, so B and the passes change every step. Is that a real approach? Which libraries do it? Is ours efficient? Which is best for SFT, DPO and RL, and why?"

**Context.** 2026-10-24. In section 6 we fixed the batch in chats (~128), converted it once into rows (20) using the
average chats per row of the already packed dataset, and kept 20 rows per step. A colleague described another method:
take exactly 128 chats for every step, pack only those 128 into rows at that step, and let the number of rows (and so the
passes at a fixed B) change from step to step, because chat lengths differ. She said Megatron does it, and that RL (RLVR)
should be done that way. I asked whether this approach exists, which libraries use it, whether ours is efficient, whether
packing can be done another, more efficient way, and which is best for SFT, DPO and RL, with proofs and facts, not
opinions. Everything below is either read from the libraries' source code or docs, or measured on our own data.

### 13.1 The two methods, in our words

| | **A: offline packing (ours)** | **B: fixed samples per step, packing per step (colleague's)** |
| --- | --- | --- |
| When packing happens | once, before training, over the whole dataset (`prepare_sft_data.py --pack`) | at every step, over that step's samples only |
| What is fixed per step | the rows (20 × 4,096 positions) | the samples (128 chats, or 128 pairs, or N rollouts) |
| What varies per step | the number of chats in the step | the number of rows, and so the passes at a fixed B |

Both keep chats separate inside a row (block-diagonal attention, positions restart per chat); the difference is only
which quantity is held constant per optimizer step.

### 13.2 Who does what (read from source code and docs, 2026-10-24)

| Library | Method | The evidence |
| --- | --- | --- |
| NVIDIA NeMo (SFT, "packed sequence") | **A** | "the global batch size needs to be reduced by the average number of sequences per pack n … where n = num_sequences_in_dataset / num_packs … This ensures that each gradient iteration sees, on average, the same number of tokens"; "Micro batch size must be set to 1"; the packed dataset "is automatically prepared before training" (NeMo 25.04 SFT docs). Exactly our section-6 rule. |
| Hugging Face TRL `SFTTrainer`, `packing=True` | **A** | `pack_dataset(dataset, args.max_length, args.packing_strategy, ...)` on the dataset before training (`sft_trainer.py:1528`), Best-Fit Decreasing by default (`data_utils.py:837`). It runs as `dataset.map(..., batched=True)`, i.e. BFD inside chunks of the `map` batch (1,000 examples by default), not over the whole dataset as ours. |
| Krell et al. 2021 (packed BERT, the paper behind "packing without cross-contamination") | **A** | §3.3: "the primary consequence of packing is an increase in the effective batch size … A direct solution is to reduce the computational batch size by the packing factor (average number of sequences per pack) and keep all other hyperparameters the same"; with that, the learning curves "show a near-identical match" (§4.2). |
| Hugging Face Transformers `DataCollatorWithFlattening` (= TRL `padding_free=True` without packing) | **B** | flattens each mini-batch of N examples into one sequence with `cu_seqlens`; the blog: it "retains the minibatches and, hence, the same number of optimization steps"; 2x throughput on FLAN, 1.4x on OrcaMath vs padding, "no impact on train convergence". |
| AllenAI open-instruct (the Tülu 3 trainer), `--packing` | **B** | `collate_fn = TensorDataCollatorWithFlattening()` on a DataLoader with `batch_size=args.per_device_train_batch_size` (`finetune.py:664-688`); the batch is counted in examples (`total_batch_size = per_device_train_batch_size * dp_world_size * gradient_accumulation_steps`, `:754`). |
| verl (RL), `use_dynamic_bsz=True` | **B** | `rearrange_micro_batches`: `num_micro_batches = min(num_groups, ceildiv(total_seqlen, max_token_len))` per step, sequences spread over the micro-batches with Karmarkar–Karp for equal work, the count made equal across data-parallel ranks (`seqlen_balancing.py:348`); `force_group_size` keeps consecutive samples together ("for RM training"). |
| Megatron-LM RL (`megatron/rl/sequence_packing_utils.py`) | **B** | `SequencePacker.pack_sequences` packs the step's trajectories, longest first, into bins of `bin_size` (max 16 per bin); `update_microbatch_calculator` then reconfigures the number of micro-batches for that step: "our batch dimension shrinks as we move some trajs onto free space in sequence dimension". |
| NVIDIA NeMo-RL | **B** | sequence packing "reorders the elements in the batch and creates metadata" so each micro-batch holds sequences packed together (default `modified_first_fit_decreasing`, `train_mb_tokens`); "speedups in the ~2-3x range"; dynamic batching (the alternative) "is almost always slower than Sequence Packing". |
| OpenRLHF | packing flag in every trainer | "SFT / DPO / PPO / RM trainers all support --ds.packing_samples" (docs). |

So the colleague's method is real and common, and Megatron does it, **in its RL path**. Our method is also real and
common: it is NeMo's SFT recipe, TRL's packing, and the original packing paper's rule.

### 13.3 Measured on our own data (the 741,413 Tülu 3 chats we trained on)

Script and output: `post_training_research/packing_methods_1024/simulate_packing.py`, `simulate_packing_output.txt`
(input: the packed-row index files of the real training data, `seq_lengths.bin`, `seq_offsets.bin`, `valid_counts.bin`).
Chat lengths: mean 640.3, median 470, p95 1,608, max 4,096 tokens; 474,700,628 real tokens in total, so at least
115,894 rows of 4,096 are needed.

| | **A: ours** | **B: 128 chats per step, BFD per step** (2 shuffle seeds, identical to ±0.01%) |
| --- | --- | --- |
| Rows per epoch | **115,904** (only 10 above the minimum possible) | 119,080 (**+2.74%**) |
| Fill of the rows | **99.991%** | 97.32% |
| Steps per epoch | 5,795.2 | 5,792.3 |
| Chats per step | mean 127.9, sd 29.0 (min 67, max 390) | **exactly 128** |
| Real tokens per step | **81,913, sd 11.5** (fixed) | 81,955, sd 6,619 (min 60,704, max 106,235) |
| Trained tokens per step | 56,207, sd 2,885 (5.1%) | (not computable per chat from these files) |
| Rows per step | always 20 | 15 to 27 (mean 20.6) |
| Passes per step at B = 4 | always 5 | 4: 20 steps, 5: 2,874, 6: 2,824, 7: 74 |
| Row slots if every pass must hold a full B = 4 rows | 115,904 | 127,648 (**+10.13%**) |

What it costs in time at our measured 1.86 s per 20-row step (~3.0 h per epoch): **B needs ~2.7% more GPU time (~5 min
per epoch)** if a pass may hold fewer than B rows, and **~10% more (~18 min per epoch)** if every pass is a full B × 4,096
(our trainer's fixed micro-batch shape). For scale: no packing at all would need 741,413 rows = 6.4x A.

Why B packs worse: Best-Fit Decreasing needs many items to fill bins well. Over the whole dataset (A) there are 741,413
chats to choose from; inside one step (B) only 128, so the last rows of each step are left partly empty.

### 13.4 Which is better depends on what the loss averages over

- **Our SFT loss** is the sum of the trained tokens' losses divided by the step's trained-token count N (section 6.3; the
  same "token-mean over the global batch" as verl: `masked_sum(loss) / batch_num_tokens * dp_size`). Each token's weight
  in a step is 1/N.
  - With **A**, N is almost constant (real tokens sd 0.01%, trained tokens sd 5.1%), so every token in the run gets almost
    the same weight. That is the property Tülu 3 asked for: they switched to a **sum** loss because, with a mean over
    samples, "each example is weighted equally", while they found that weighting **every token equally** worked better
    (Tülu 3 paper §4.3.2; lr 5e-6, batch 128 sequences, length 4,096). A token-mean over a nearly constant N is a sum
    loss times a constant.
  - With **B**, the 128 chats are constant but N moves by about ±8% (sd 6,619 of 81,955 real tokens), so a token in a
    short step weighs more than a token in a long step: 1.30x between the 5% shortest and the 5% longest steps
    (92,931 / 71,573), 1.75x at the extremes (106,235 / 60,704).
  - **Verdict for SFT: A** (ours): 2.7–10% cheaper on our data, the same rows and shapes every step, and equal token
    weights. B is not wrong; nobody has shown a quality difference between the two (NeMo and the Hugging Face blog
    each report "no impact on convergence" only against padding).
- **DPO:** the loss is a mean over **pairs**: TRL `loss += per_sequence_loss.mean()` (`dpo_trainer.py:1580`), one value per
  pair. And the chosen and the rejected answer of a pair must be in **the same forward pass**: TRL puts them in one batch
  `[chosen, rejected]` and splits the log-probs with `logps.chunk(2, dim=0)` (`:1368`); verl's `force_group_size` exists to
  keep such groups in one micro-batch.
  - **Verdict for DPO: B**: a fixed number of pairs per step (the unit the loss averages over), packed per step, with each
    pair kept whole. (Packing pairs offline as whole units is possible too, but then pairs per step vary, and so does
    each pair's weight.) Note: TRL's DPO `padding_free` is currently disabled after a refactor (`:784-791`).
- **RL (GRPO / RLVR):** the samples (rollouts) do not exist before the step: they are generated by the current model at
  every step, so offline packing is impossible by construction. The number of samples per step is set by the algorithm
  (prompts × rollouts per prompt).
  - **Verdict for RL: B is the only option**, and every RL framework above does it (verl, Megatron-LM RL, NeMo-RL,
    OpenRLHF), with the loss normalized by the step's global token count so a varying number of micro-batches stays
    correct (verl `agg_loss`, token-mean).

### 13.5 Can our packing be made more efficient?

Not for SFT on this data: A is at **99.991%** fill, 10 rows above the theoretical minimum of 115,894 (0.009%). The only
padding left is the tail of those rows. The remaining waste in an SFT step is elsewhere: the untrained prompt positions
(trained tokens are 56,207 of 81,913 per step, 69%) and the attention work, which packing does not change.

### 13.6 In short

- The colleague's method (fixed samples per step, packing per step, varying passes) is real: open-instruct, the Hugging
  Face flattening collator, verl, Megatron-LM RL, NeMo-RL and OpenRLHF do it.
- Ours (offline packing, fixed rows per step) is real too: NeMo SFT, TRL packing and the packed-BERT paper prescribe it,
  including our rule rows per step = chats per step ÷ chats per row.
- On our data, ours uses 2.7% fewer rows (10% fewer row slots at a fixed B = 4) and keeps the tokens per step constant.
- Best choice by stage: **SFT → A** (the loss averages over tokens), **DPO → B with whole pairs** (the loss averages over
  pairs), **RL → B, the only option** (the data is generated during training).

### 13.7 Sources for this section

- NeMo Framework, Sequence Packing for SFT/PEFT (25.04): <https://docs.nvidia.com/nemo-framework/user-guide/25.04/sft_peft/packed_sequence.html>
- NeMo-RL, Sequence Packing and Dynamic Batching: <https://docs.nvidia.com/nemo/rl/nightly/design-docs/sequence-packing-and-dynamic-batching.html>
- Krell et al. 2021, Efficient Sequence Packing without Cross-contamination, arXiv 2107.02027 (§3.3, §4.2)
- Hugging Face blog, Improving Hugging Face Training Efficiency Through Packing with Flash Attention: <https://huggingface.co/blog/packing-with-FA2>
- TRL source: `trl/trainer/sft_trainer.py`, `trl/data_utils.py` (`pack_dataset`), `trl/trainer/dpo_trainer.py` (main branch, read 2026-10-24)
- verl source: `verl/utils/seqlen_balancing.py` (`rearrange_micro_batches`), `verl/trainer/ppo/core_algos.py` (`agg_loss`); docs: <https://verl.readthedocs.io/en/latest/perf/perf_tuning.html>
- Megatron-LM source: `megatron/rl/sequence_packing_utils.py` (main branch, read 2026-10-24)
- open-instruct source: `open_instruct/finetune.py` (main branch, read 2026-10-24)
- Tülu 3 paper, arXiv 2411.15124, §4.3.2 "Batch Aggregation", Table 11
- OpenRLHF docs: <https://openrlhf.readthedocs.io/en/latest/non_rl.html>
- Our data: `post_training_research/packing_methods_1024/simulate_packing.py` + output

---

## 14. Doubt: "In our runs the chats per step change. Doesn't the optimizer then see a different number of chats every step? Should it be based on tokens instead? Why did Tülu say 128 chats per step? What should we care about in SFT: chats, rows or tokens per step?"

**Context.** 2026-10-24, right after section 13 showed that with our method the chats per step vary from 67 to 390 (mean
128). I asked whether that hurts the update, whether a fixed number of tokens per step (we always process 20 × 4,096 =
81,920 positions) is what really matters, whether it is fine because every answer only sees its own prompt anyway (unlike
DPO, which needs the chosen and the rejected answer together), how Tülu's "128 chats per step" fits, and which of the
three (chats, rows, tokens) we should care about.

### 14.1 Why a changing number of chats is not a problem in SFT

- The SFT loss of a step is `L = (1/N) × (sum of −ln p over all trained tokens of the step)`, N = trained tokens of the step
  (section 6.3; `bluscript_sft.cpp` divides every micro-step by the step's global trained-token count).
- Each token's term depends only on its own chat: the model reads that chat's prompt and earlier answer tokens and nothing
  else (block-diagonal attention, positions restart per chat). There is **no term that links two chats**.
- So the gradient is `(1/N) × (sum of every trained token's own gradient)`. Chats are not a unit in this formula; a chat
  only contributes its tokens: a chat with 2,000 answer tokens contributes 2,000 terms, a chat with 20 contributes 20.
- DPO is different: its loss has one term per pair that needs both answers (section 13.4), so there the pair is a unit.

### 14.2 Which number our method holds constant, and which it does not

Measured over all 5,795 steps of our Tülu run (`post_training_research/packing_methods_1024/simulate_packing_output.txt`):

| Per step | Mean | Spread (sd) | Range |
| --- | --- | --- | --- |
| Rows | 20 | 0 | 20 |
| Positions processed (= rows × 4,096) | 81,920 | 0 | – |
| Real tokens (prompt + answer, 99.99% of the positions) | 81,913 | 11.5 (0.01%) | 81,510 – 81,920 |
| **Trained tokens (answers only: what N counts)** | **56,207** | **2,885 (5.1%)** | 42,600 – 65,180 |
| Chats | 127.9 | 29.0 (23%) | 67 – 390 |

- Rows and real tokens are the same thing for us (99.99% fill), and both are fixed.
- The trained tokens are NOT fixed: the prompt share of each chat differs (on average 69% of the positions are trained).
  They move by ±5% per step, far less than the chats (±23%).

### 14.3 Tülu's "128 chats per step"

- Tülu 3 trained **without packing**: open-instruct's default is `packing=False`, which collates with
  `DataCollatorForSeq2Seq(padding="longest")` (`finetune.py:664-668`): one chat per row, padded to the longest chat of the
  micro-batch. So in their setup **128 chats = 128 rows**, and the tokens per step vary with the chats' lengths.
- They also chose a **sum** loss so that every token has the same weight (Tülu 3 paper §4.3.2): what they cared about was
  the per-token weight, not the chat count. The "128" is the batch size their learning rate (5e-6, sum loss) was tuned for.
- We took their 128 only to size our step: 128 chats × 640 mean tokens ≈ 82k positions ≈ our 20 rows.

### 14.4 What to care about in SFT, in order

1. **Trained tokens per step** (N): this is what the gradient averages over, so it is the real batch size for learning.
   Keep it roughly constant and in a sensible range. Ours: 56k ± 5%.
2. **Positions (rows) per step**: this is the compute and the memory of a step. Ours: fixed, so the step time and memory
   peak are the same every step (1.86 s, 74 GB).
3. **Chats per step**: not used by the loss. It matters only indirectly: tokens of the same chat are related to each other,
   so a step of 67 long chats holds fewer independent examples than a step of 390 short chats with the same tokens. No
   recipe we read fixes this per step for SFT; the shuffle at data preparation spreads long and short chats over the steps.

So yes: in our SFT, packing efficiently (99.99%) and fixing the rows is the right choice; what we watch is the trained tokens
per step, and the chat count is allowed to move.

---

## 15. Doubt: "What does 'sum loss' mean? Is it over the valid-token count of the full step, not per micro-step? Does it change with padding or packing? Is our loss (option 2) the same as Tülu's, or only scaled?"

**Context.** 2026-10-24, after section 14, which said Tülu 3 chose a "sum" loss so that every token has the same weight.
I asked what a sum loss is exactly: is it divided by the valid-token count of the whole step or of each micro-step, and
does padding or packing change it. Then, after the three options below, I asked whether our loss (the mean over the full
step) is the same as Tülu's sum loss, since both give every token the same share, or only a scaled version of it.

### 15.1 Three ways to turn a step's token losses into one number

"Valid tokens" = the trained (answer) tokens. Prompt tokens and padding carry the "ignore" label (−100 in Hugging Face,
65535 in ours), so they are left out of the sum and out of every count below.

| Name | How one step's loss is made | Weight of one token |
| --- | --- | --- |
| **1. Mean per micro-step** (the old default of Hugging Face / PyTorch training loops) | each micro-batch: the sum of its tokens' losses ÷ **its own** valid-token count; then the micro-step losses are averaged (÷ GRAD_ACCUM) | 1 / (GRAD_ACCUM × that micro-batch's count): **different in every micro-batch** |
| **2. Mean over the full step** (**ours**) | the sum over **all** valid tokens of the step (all micro-steps, all GPUs) ÷ the step's total valid-token count N | 1/N: **the same for every token of the step** |
| **3. Sum loss** (Tülu 3) | the sum over all valid tokens of the step, **not divided by any token count** | 1: **the same in every step**; a step with more tokens gives a bigger gradient, so the learning rate must be smaller (Tülu 3: 5e-6) |

So yes: "sum loss" = the plain sum of the valid tokens' losses of the step, with no division by a token count (at most
by a fixed constant that does not depend on the data). Our option 2 divides by the valid count of the **full step**, not
per micro-step.

**Example:** one step of 2 micro-batches; micro-batch 1 has 1,000 valid tokens, micro-batch 2 has 9,000.

| | Token in micro-batch 1 | Token in micro-batch 2 |
| --- | --- | --- |
| Option 1 | 1/(2 × 1,000) = 1/2,000 | 1/(2 × 9,000) = 1/18,000 → **9x lighter**, only because of how the tokens were grouped |
| Option 2 | 1/10,000 | 1/10,000 |
| Option 3 | 1 | 1 |

Option 1 is the problem the Tülu 3 paper describes (§4.3.2): with gradient accumulation (or several GPUs) and padding,
"each example is weighted equally" instead of each token, and changing the accumulation changed their results.

### 15.2 Padding or packing does not change options 2 and 3

- Padding and prompt positions are ignored in the sum and in N, so only the set of valid tokens of the step matters.
- With option 2 or 3, the same chats in a step give exactly the same loss and gradient whether they were padded (one chat
  per row) or packed (several chats per row), and however they are split into micro-batches.
- Only option 1 depends on the grouping into micro-batches, so only option 1 changes with padding, packing or GRAD_ACCUM.

### 15.3 Our loss (option 2) against Tülu's sum loss (option 3)

**Inside one step: the same proportions, only scaled.** Every token gets the same share in both: ours = Tülu's sum × 1/N.
The gradient points in exactly the same direction; only its length differs.

**Across steps: not exactly the same,** because N changes a little from step to step and we divide each step by its own N.

| | Step A: N = 50,000 | Step B: N = 60,000 |
| --- | --- | --- |
| Ours: weight of one token | 1/50,000 | 1/60,000 → a token of step A weighs **1.2x** a token of step B |
| Tülu's sum: weight of one token | 1 | 1 → **equal** |

- Ours gives every step about the same total push, whatever its token count.
- Tülu's gives a step with more tokens a proportionally bigger push.

**How much it matters for us: very little.**
- Our N is 56,207 ± 2,885 trained tokens per step (5.1%, section 14.2), so token weights across steps differ by about ±5%
  normally, and by 1.5x only in the rare extreme steps (42,600 vs 65,180 trained tokens).
- Our optimizers largely ignore a constant scale of the gradient:
  - **AdamW** divides each update by the running size (RMS) of the gradients, so multiplying all gradients by the same
    1/N cancels out;
  - **Muon** orthogonalizes the update, which throws away its overall size; only the learning rate sets the step length.
- So the only real difference left is the ±5% variation of N between steps, which slightly changes how the momentum
  mixes consecutive steps.

**In short:** ours = Tülu's sum loss divided by a nearly constant number: the same proportions inside every step, and
only ±5% different between steps.

### 15.4 A note on the current open-instruct code

The Tülu 3 paper (§4.3.2) describes the sum loss. The current main branch of open-instruct (`finetune.py`, read
2026-10-24) computes `loss = outputs.loss` (the model's own loss of each micro-batch, a mean over that micro-batch's
valid tokens) and calls `accelerator.backward(loss)` per micro-batch, so its code has changed since the paper.
