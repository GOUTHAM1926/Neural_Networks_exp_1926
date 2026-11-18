# Learning-rate schedulers in our SFT trainer

Written 2026-11-12. Covers: what a scheduler is, the four schedulers our SFT trainer has, how each one works step by
step with worked examples, which one we used in every SFT run, what other labs use, what is tested and what is not,
and the questions asked on 2026-11-12 with their answers.

Code referred to: `BluTrain/train/production_zero/bluscript_sft.cpp` (branch `dist_side_gautam_lottn`) and the
launcher `BluTrain/sft/run_sft.sh`.

---

## 1. The short version

- **Every SFT run we have done used the same scheduler:** a linear warm-up over the first 3% of the steps, then a
  cosine fall down to one tenth of the peak learning rate.
- The trainer has **four** schedulers: `cosine`, `linear`, `constant`, `wsd`. They were added on 2026-10-28.
- All four share the same warm-up. They differ only in what happens after the warm-up.
- **None of them has an "angle" setting.** How steeply the learning rate falls comes out of three numbers: the peak
  learning rate, the end learning rate, and the number of steps the fall has.
- The three schedulers other than cosine are compiled but **no run is recorded with any of them.** The comparison of
  the four (stage 2 of the 135M grid) was never run.

---

## 2. Words used in this document

| Word | Meaning |
| --- | --- |
| step | one optimizer step: the weights are updated once. Steps are numbered from 0. |
| learning rate (LR) | the size of the update in one step. A larger LR changes the weights more per step. |
| scheduler | the rule that gives the LR for each step number. It is fixed before the run starts. |
| peak LR | the highest LR of the run (`BLU_SFT_LR`). |
| end LR | the LR at the last step. Default: peak ÷ 10 (`BLU_SFT_MIN_LR`). |
| max steps | the total number of steps in the run. |
| warm-up | the first part of the run, where the LR rises from near 0 to the peak. |
| stable phase | a part of the run where the LR stays at the peak. |
| decay | the part of the run where the LR falls from the peak to the end LR. |
| WSD | **W**arm-up, **S**table, **D**ecay: rise, stay at the peak, fall. |

---

## 3. The numbers every schedule is built from

The trainer works these out once, before step 0.

| Number | How the trainer gets it | Default |
| --- | --- | --- |
| max steps | epochs × micro-batches per epoch ÷ GRAD_ACCUM | from the data and `BLU_SFT_EPOCHS` (2) |
| warm-up steps | round(warm-up fraction × max steps), at least 1 | fraction `BLU_SFT_WARMUP` = 0.03 |
| peak LR | `BLU_SFT_LR` | 2e-5 |
| end LR | `BLU_SFT_MIN_LR` | peak ÷ 10 |
| stable steps | 0, except for `wsd` (section 5.4) | 0 |

The LR of a step depends only on the step number and these numbers. It does not depend on the loss or on anything
that happens during the run.

**Two optimizer groups, one shape.** With Muon, the trainer has two groups: Muon for the matrices and AdamW for the
embeddings and norms. Each group has its own peak (`BLU_SFT_LR` for AdamW, `BLU_SFT_MUON_LR` for Muon) and its own end
LR (peak ÷ 10 of that group, or `BLU_SFT_MIN_LR` / `BLU_SFT_MUON_MIN_LR`). Both groups use the same warm-up length,
the same stable length and the same fall, so they rise and fall at the same steps. Our launcher `run_sft.sh` sets both
peaks to the same value (`BLU_SFT_MUON_LR=$LR`).

---

## 4. The warm-up (the same in all four schedulers)

**Rule:** for a step number smaller than the warm-up steps,

    LR(step) = peak × (step + 1) ÷ warm-up steps

**What this gives:**

- The LR rises in a straight line.
- The rise per step is the same every step: **peak ÷ warm-up steps**.
- Step 0 does not have LR 0. It has one rise already: peak ÷ warm-up steps.
- The last warm-up step (step number warm-up steps − 1) has exactly the peak.

**Example, 135M stage-1 run** (6,996 steps, peak 1e-4):

- warm-up steps = round(0.03 × 6,996) = round(209.88) = **210**
- rise per step = 1e-4 ÷ 210 = **4.762e-7**

| step | LR | how |
| --- | --- | --- |
| 0 | 4.762e-7 | 1e-4 × 1 ÷ 210 |
| 1 | 9.524e-7 | 1e-4 × 2 ÷ 210 |
| 104 | 5.000e-5 | 1e-4 × 105 ÷ 210 (half of the peak) |
| 209 | 1.000e-4 | 1e-4 × 210 ÷ 210 (the peak) |

**Why a warm-up is used:** at step 0 the optimizer has no history yet (its running averages start at zero), so the
first updates are less reliable. Small steps at the start keep those first updates from moving the trained weights
far.

---

## 5. The four schedulers, one by one

All four start with the warm-up of section 4. The tables below describe what happens **after** the warm-up.

In the formulas, "position in the fall" is a number from 0 to 1:

    position = (step − first step of the fall) ÷ (number of steps in the fall − 1)

It is 0 at the first step of the fall and exactly 1 at the last step of the run. So the last step always lands
exactly on the end LR.

### 5.1 `cosine` (the default; the one we used in every run)

- **After the warm-up:** the LR falls from the peak to the end LR along a cosine curve. The fall takes all the
  remaining steps. There is no stable phase.
- **Formula:**

      LR = end + 0.5 × (1 + cos(π × position)) × (peak − end)

- **Shape of the fall:** slow at the start, fastest in the middle, slow again at the end.
- **Nothing extra to set.** The curve is fully fixed by the peak, the end LR and the number of steps.

Value of the factor `0.5 × (1 + cos(π × position))` at some positions:

| position in the fall | factor | LR for peak 1e-4, end 1e-5 |
| --- | --- | --- |
| 0 (start) | 1.000 | 1.000e-4 |
| 0.25 | 0.854 | 8.68e-5 |
| 0.50 | 0.500 | 5.50e-5 |
| 0.75 | 0.146 | 2.32e-5 |
| 1 (end) | 0.000 | 1.000e-5 |

### 5.2 `linear`

- **After the warm-up:** the LR falls from the peak to the end LR in a straight line. The fall takes all the remaining
  steps. There is no stable phase.
- **Formula:**

      LR = end + (1 − position) × (peak − end)

- **The drop per step is the same every step:**

      drop per step = (peak − end) ÷ (number of steps in the fall − 1)

- **There is no angle setting.** The slope is that drop per step, and it comes out of the peak, the end LR and the
  number of steps.

Example, 135M stage-1 numbers (6,996 steps, 210 warm-up steps, peak 1e-4, end 1e-5):

- steps in the fall = 6,996 − 210 = 6,786
- drop per step = (1e-4 − 1e-5) ÷ 6,785 = **1.326e-8**

**Difference from Tülu 3 and Moonlight:** their linear schedule falls to 0. Ours ends at peak ÷ 10 by default. To get
their shape, set `BLU_SFT_MIN_LR=0` (and `BLU_SFT_MUON_MIN_LR=0` with Muon). This setting has not been run.

### 5.3 `constant`

- **After the warm-up:** the LR stays at the peak until the last step. It never falls.
- **How the trainer does it:** it sets the end LR equal to the peak. The cosine formula then gives the peak at every
  step, because (peak − end) is 0.

### 5.4 `wsd` (Warm-up, Stable, Decay)

- **After the warm-up:** the LR stays at the peak (the stable phase), and then falls along the cosine curve over the
  last part of the run.
- **Length of the fall:** round(decay fraction × max steps), at least 2 steps. The decay fraction is
  `BLU_SFT_DECAY_FRAC`, default **0.2** (the last 20% of the steps).
- **Length of the stable phase:** max steps − warm-up steps − steps in the fall.
- **Where the fall ends:** at the end LR, peak ÷ 10, the same as cosine and linear. So in `wsd` both numbers apply:
  20% says **how long** the fall lasts, and 10% says **where** it ends.
- **Shape of the fall:** the same cosine formula as section 5.1, only over fewer steps, so it is steeper.

Example, 135M stage-1 numbers:

- steps in the fall = round(0.2 × 6,996) = round(1,399.2) = **1,399**
- stable steps = 6,996 − 210 − 1,399 = **5,387** (steps 210 to 5,596)
- the fall = steps 5,597 to 6,995

This is the shape our pretraining used (135M pretraining: 3.5e-4 → 3.5e-5, WSD).

### 5.5 The four side by side

| | warm-up | after the warm-up | LR at the last step |
| --- | --- | --- | --- |
| `cosine` | yes | cosine fall over all remaining steps | peak ÷ 10 |
| `linear` | yes | straight-line fall over all remaining steps | peak ÷ 10 |
| `constant` | yes | stays at the peak | peak |
| `wsd` | yes | stays at the peak, then cosine fall over the last 20% | peak ÷ 10 |

---

## 6. Worked examples

Each table gives the LR of the same step under each of the four schedulers. The numbers come from the trainer's own
formula (`get_lr`), computed for these settings.

### 6.1 Example A: a small run of 100 steps (easy to check by hand)

**Settings:** 100 steps, peak 1e-4, end 1e-5, warm-up 3%.

- warm-up steps = round(0.03 × 100) = 3 (steps 0, 1, 2); rise per step = 1e-4 ÷ 3 = 3.333e-5
- cosine and linear: the fall is steps 3 to 99 (97 steps); linear drop per step = 9e-5 ÷ 96 = 9.375e-7
- wsd: the fall = round(0.2 × 100) = 20 steps (steps 80 to 99); stable = 100 − 3 − 20 = 77 steps (steps 3 to 79)

| step | cosine | linear | wsd | constant | what is happening |
| --- | --- | --- | --- | --- | --- |
| 0 | 3.333e-5 | 3.333e-5 | 3.333e-5 | 3.333e-5 | warm-up, 1/3 of the peak |
| 1 | 6.667e-5 | 6.667e-5 | 6.667e-5 | 6.667e-5 | warm-up, 2/3 of the peak |
| 2 | 1.000e-4 | 1.000e-4 | 1.000e-4 | 1.000e-4 | warm-up ends at the peak |
| 3 | 1.000e-4 | 1.000e-4 | 1.000e-4 | 1.000e-4 | first step after the warm-up, still the peak |
| 10 | 9.882e-5 | 9.344e-5 | 1.000e-4 | 1.000e-4 | cosine has barely moved; linear is already 6.6% down |
| 25 | 8.883e-5 | 7.938e-5 | 1.000e-4 | 1.000e-4 | |
| 51 | 5.500e-5 | 5.500e-5 | 1.000e-4 | 1.000e-4 | middle of the fall: cosine and linear meet |
| 75 | 2.318e-5 | 3.250e-5 | 1.000e-4 | 1.000e-4 | cosine is now below linear |
| 80 | 1.842e-5 | 2.781e-5 | 1.000e-4 | 1.000e-4 | wsd starts its fall here |
| 85 | 1.464e-5 | 2.313e-5 | 8.548e-5 | 1.000e-4 | |
| 90 | 1.194e-5 | 1.844e-5 | 5.128e-5 | 1.000e-4 | wsd is near the middle of its fall |
| 95 | 1.038e-5 | 1.375e-5 | 1.949e-5 | 1.000e-4 | |
| 99 | 1.000e-5 | 1.000e-5 | 1.000e-5 | 1.000e-4 | last step: the three that fall all end at peak ÷ 10 |

### 6.2 Example B: the 135M stage-1 run (real settings)

**Settings:** 6,996 steps (2 epochs, about 128 chats per step), peak 1e-4, end 1e-5, warm-up 3%.

- warm-up steps = 210 (steps 0 to 209); rise per step = 4.762e-7
- cosine and linear: the fall is steps 210 to 6,995 (6,786 steps); linear drop per step = 1.326e-8
- wsd: stable = steps 210 to 5,596 (5,387 steps); the fall = steps 5,597 to 6,995 (1,399 steps)

| step | cosine | linear | wsd | constant | what is happening |
| --- | --- | --- | --- | --- | --- |
| 0 | 4.762e-7 | 4.762e-7 | 4.762e-7 | 4.762e-7 | first warm-up step |
| 104 | 5.000e-5 | 5.000e-5 | 5.000e-5 | 5.000e-5 | middle of the warm-up |
| 209 | 1.000e-4 | 1.000e-4 | 1.000e-4 | 1.000e-4 | warm-up ends at the peak |
| 1,000 | 9.702e-5 | 8.952e-5 | 1.000e-4 | 1.000e-4 | |
| 1,906 | 8.682e-5 | 7.750e-5 | 1.000e-4 | 1.000e-4 | 25% of the fall |
| 3,603 | 5.499e-5 | 5.499e-5 | 1.000e-4 | 1.000e-4 | 50% of the fall: cosine and linear meet |
| 5,299 | 2.318e-5 | 3.250e-5 | 1.000e-4 | 1.000e-4 | 75% of the fall |
| 5,597 | 1.910e-5 | 2.854e-5 | 1.000e-4 | 1.000e-4 | wsd starts its fall here |
| 5,946 | 1.520e-5 | 2.391e-5 | 8.686e-5 | 1.000e-4 | 25% of the wsd fall |
| 6,296 | 1.234e-5 | 1.927e-5 | 5.500e-5 | 1.000e-4 | 50% of the wsd fall |
| 6,646 | 1.059e-5 | 1.463e-5 | 2.314e-5 | 1.000e-4 | 75% of the wsd fall |
| 6,995 | 1.000e-5 | 1.000e-5 | 1.000e-5 | 1.000e-4 | last step |

### 6.3 Example C: the first 1.6B SFT runs (real settings)

**Settings:** 156 steps (2 epochs, 64 rows per step), peak 2e-5, end 2e-6, warm-up 3%.

- warm-up steps = round(0.03 × 156) = round(4.68) = 5 (steps 0 to 4); rise per step = 2e-5 ÷ 5 = 4e-6
- cosine and linear: the fall is steps 5 to 155 (151 steps); linear drop per step = 1.8e-5 ÷ 150 = 1.2e-7
- wsd: the fall = round(0.2 × 156) = 31 steps (steps 125 to 155); stable = 156 − 5 − 31 = 120 steps (steps 5 to 124)

| step | cosine | linear | wsd | constant | what is happening |
| --- | --- | --- | --- | --- | --- |
| 0 | 4.0e-6 | 4.0e-6 | 4.0e-6 | 4.0e-6 | warm-up |
| 1 | 8.0e-6 | 8.0e-6 | 8.0e-6 | 8.0e-6 | warm-up |
| 2 | 1.2e-5 | 1.2e-5 | 1.2e-5 | 1.2e-5 | warm-up |
| 3 | 1.6e-5 | 1.6e-5 | 1.6e-5 | 1.6e-5 | warm-up |
| 4 | 2.0e-5 | 2.0e-5 | 2.0e-5 | 2.0e-5 | warm-up ends at the peak |
| 20 | 1.956e-5 | 1.820e-5 | 2.0e-5 | 2.0e-5 | |
| 42 | 1.743e-5 | 1.556e-5 | 2.0e-5 | 2.0e-5 | about 25% of the fall |
| 80 | 1.100e-5 | 1.100e-5 | 2.0e-5 | 2.0e-5 | 50% of the fall: cosine and linear meet |
| 118 | 4.570e-6 | 6.440e-6 | 2.0e-5 | 2.0e-5 | about 75% of the fall |
| 125 | 3.719e-6 | 5.600e-6 | 2.0e-5 | 2.0e-5 | wsd starts its fall here |
| 140 | 2.440e-6 | 3.800e-6 | 1.100e-5 | 2.0e-5 | 50% of the wsd fall |
| 148 | 2.097e-6 | 2.840e-6 | 4.312e-6 | 2.0e-5 | |
| 155 | 2.0e-6 | 2.0e-6 | 2.0e-6 | 2.0e-5 | last step |

Only the cosine column of examples B and C was actually run. The other three columns are what the trainer would give.

---

## 7. How the four differ in total

**Cosine against linear:**

- They are equal at the start of the fall, in the middle, and at the end.
- In the first half cosine is higher (it keeps the LR near the peak for longer).
- In the second half cosine is lower (it spends more steps near the end LR).

**Average LR over the whole run, as a fraction of the peak** (computed over every step of each example):

| schedule | example A (100 steps) | example B (6,996 steps) | example C (156 steps) |
| --- | --- | --- | --- |
| cosine | 0.554 | 0.549 | 0.552 |
| linear | 0.553 | 0.549 | 0.552 |
| wsd | 0.900 | 0.895 | 0.898 |
| constant | 0.990 | 0.985 | 0.987 |

What this says:

- Cosine and linear give the same average LR. They differ only in **when** the larger steps happen.
- `wsd` and `constant` give a much higher average LR at the same peak (about 0.90 and 0.99 of the peak against 0.55).
  So at the same peak they move the weights more in total.
- This matters for any comparison of the four: our stage-1 result showed that a higher SFT LR lowers the validation
  loss but costs base knowledge (LR 1e-4: IFEval 0.405, ARC-easy 0.492 against the base's 0.543). A fair test of `wsd`
  or `constant` against cosine therefore needs its own peak LR, not only the same peak.

---

## 8. The settings

| Setting | Meaning | Default |
| --- | --- | --- |
| `BLU_SFT_SCHEDULE` | `cosine`, `linear`, `constant` or `wsd`. Any other word stops the trainer with an error. | `cosine` |
| `BLU_SFT_WARMUP` | warm-up length as a fraction of max steps | 0.03 |
| `BLU_SFT_LR` | peak LR (the AdamW group) | 2e-5 |
| `BLU_SFT_MIN_LR` | end LR (the AdamW group) | `BLU_SFT_LR` ÷ 10 |
| `BLU_SFT_MUON_LR` | peak LR of the Muon group. Required with Muon. | none |
| `BLU_SFT_MUON_MIN_LR` | end LR of the Muon group | `BLU_SFT_MUON_LR` ÷ 10 |
| `BLU_SFT_DECAY_FRAC` | `wsd` only: the fraction of the steps used for the fall | 0.2 |
| `BLU_SFT_EPOCHS` | epochs; together with the data it sets max steps | 2 |

In the launcher `sft/run_sft.sh` the same choices are `SCHEDULE` (default `cosine`), `WARMUP` (default 0.03), `LR`
and `EPOCHS`.

The trainer prints the schedule in its first lines, for example:

    [sft] ... | 2 epochs = 6996 steps | lr 0.0001 -> 1e-05, warmup 210 steps then cosine | ...

---

## 9. Limits and rules in the code

- **A fall of at least 2 steps is required.** The trainer stops with an error when
  warm-up steps + stable steps ≥ max steps − 1. This is because the position in the fall divides by
  (steps in the fall − 1). Example recorded in the work log: a run of 2 steps gives warm-up 1 and is refused; with 4
  steps it passes.
- **`constant` also goes through the same check**, because it is the cosine formula with the end LR set to the peak.
- **The warm-up is at least 1 step**, even when 3% of the steps rounds to 0.
- **The `wsd` fall is at least 2 steps**, even when 20% of the steps rounds to less.
- **No angle or slope setting exists.** To make a fall steeper or flatter, change one of: the peak, the end LR, the
  number of steps, or (for `wsd`) the decay fraction.

---

## 10. What other labs use for SFT

| Source | Schedule | Warm-up | Notes |
| --- | --- | --- | --- |
| Tülu 3 (Table 11) | linear | 0.03 | LR 5e-6 at 8B, 2 epochs, 128 chats per step |
| Moonlight (arXiv 2502.16982) | linear to 0 | | Tülu 3 mix, 2 epochs, LR 5e-5, Muon |
| SmolLM2 (alignment-handbook `config_smol.yaml`) | cosine | 10% | LR 1e-3 at 135M, 3e-4 at 1.7B |
| Qwen2.5-7B test quoted in our optimizer document | cosine 2e-5 → 2e-6 | | Tülu 3 mix packed to 8k |
| ours | cosine to peak ÷ 10 | 3% | every run so far |

**What we found:**

- Published SFT recipes use **cosine or linear**. Both fall from the first step after the warm-up.
- **No published SFT recipe we read uses WSD or constant.** This was the colleague's point (work log, 2026-10-28). We
  kept `wsd` in the trainer as one test arm only, because it is the shape of our pretraining.
- Our first plan (research document §3.11, September) said linear, copied from Tülu 3. The trainer was built with
  cosine (2026-10-06) and every run used cosine.

**A related paper that is about pretraining, not SFT:** Yano et al. (arXiv 2603.16127, ICLR 2026) pretrained models
with no decay, WSD, cosine and linear, then ran the same SFT on each. The models pretrained with no decay scored
higher after SFT (1B: 54.4 against 53.4; 8B: 55.0 against 53.9). For us this is the question of which base checkpoint
to start from (750B, before the decay, against 900B, after it). It is side experiment X2 in `plan_grid_search_sft.md`
and has not been run.

---

## 11. What we used, and what is tested

**Runs:**

| Runs | Schedule | Warm-up | LR |
| --- | --- | --- | --- |
| 1.6B SFT from step 8000 (from 2026-10-06) | cosine | 3% (5 of 156 steps) | 2e-5 → 2e-6 |
| 1.6B SFT on the Tülu mix (optimizer document; validation loss 1.1757 → 0.8501) | cosine | 3% (348 of 11,590 steps) | 5e-5 → 5e-6 |
| 135M stage-1 grid (2026-10-28) | cosine | 3% (210 of 6,996 steps) | swept 1e-6 to 3e-4, end = peak ÷ 10 |
| 135M format-study screen | cosine | 3% | stage-1 settings |
| 135M before/after and context-parallelism runs (from 2026-11-06) | cosine | 3% | 1e-4 → 1e-5 |

**Status of each scheduler:**

| Scheduler | Built | Run | Result |
| --- | --- | --- | --- |
| `cosine` | yes | yes, every run | the results above |
| `linear` | yes (2026-10-28) | no run recorded | none |
| `constant` | yes (2026-10-28) | no run recorded | none |
| `wsd` | yes (2026-10-28) | no run recorded | none |

**Open:**

- Stage 2 of the 135M grid (the best LR and batch × 4 schedulers × 2 arms) was planned and never run. Stage 1 was
  stopped on 2026-10-28 at 14:55 IST for the 8192 benchmarks, and the work then moved to the format study.
- The schedule is still undecided in the plan: factor 8 in `plan_grid_search_sft.md`, in one block with the LR, the
  chats per step and the epochs (factors 6-9).
- `linear`, `constant` and `wsd` need at least one real run each before they are used: a check that the printed LR per
  step matches the tables of section 6.
- A warm-up value correction: one chat answer on 2026-10-30 said "10% warm-up". The launcher used 3%
  (`WARMUP=0.03`). This is correction 20 in the plan file.

---

## 12. Questions asked on 2026-11-12, and the answers

**Q1. Which scheduler did we use for all our SFT training runs?**
Cosine, in every run: a linear warm-up over the first 3% of the steps, then a cosine fall to peak ÷ 10. It is the
default in the trainer and in the launcher, so any run that did not set a schedule got it. Section 11.

**Q2. What research did we do on schedulers?**
We read the SFT recipes of Tülu 3, Moonlight, SmolLM2 and the Qwen2.5-7B test, and the Yano et al. paper on
pretraining without decay. Section 10.

**Q3. Did we see that some schedulers are used by nobody and only some are used by everyone?**
Yes. Published SFT recipes use only cosine or linear. None of the ones we read uses WSD or constant.

**Q4. Are there four schedulers?**
Yes: `cosine`, `linear`, `constant`, `wsd`. Section 5.

**Q5. What is the full form of WSD?**
Warm-up, Stable, Decay. The LR rises, stays at the peak, then falls.

**Q6. In linear, at what angle does the LR fall?**
There is no angle setting. The LR drops by the same amount every step, and that amount is
(peak − end LR) ÷ (steps in the fall − 1). In the 135M stage-1 run it is 1.326e-8 per step. Section 5.2.

**Q7. In cosine, at what angle does it fall? Does it fall until that 10%?**
It falls from the peak to peak ÷ 10, so yes, it ends at 10% of the peak. Its steepness is not constant: slow at the
start, fastest in the middle, slow at the end. Section 5.1.

**Q8. So for cosine no angle or anything extra is needed?**
Correct. The peak, the end LR and the number of steps fix the whole curve. The same is true for linear.

**Q9. In WSD the fall is in the last 20%; so there is no 10%, it just falls along the cosine curve?**
It falls along the cosine curve, but the 10% still applies. The 20% is how long the fall lasts
(`BLU_SFT_DECAY_FRAC`), and the 10% is where it ends (peak ÷ 10). It does not fall to 0. Section 5.4.

**Q10. In the warm-up of 3%, the LR rises from 0 to the peak; is the rise per step (peak − 0) ÷ (3% of the steps)?**
Yes, the rise per step is peak ÷ warm-up steps, the same every step. One detail: step 0 does not have LR 0. It
already has one rise (peak ÷ warm-up steps), and the last warm-up step has exactly the peak. In the 135M stage-1 run:
210 warm-up steps, 4.762e-7 per step, step 0 = 4.762e-7, step 209 = 1e-4. Section 4.

**Q11. What is the "position" in the formulas? Is it the step's position in the whole run?**
No. It is the step's position **inside the fall only**, as a number from 0 to 1. The warm-up steps (and in `wsd` the
stable steps) are not counted.

    position = (step − first step of the fall) ÷ (steps in the fall − 1)

In the 135M stage-1 run with cosine, the fall is steps 210 to 6,995 (6,786 steps):

| step | position | how |
| --- | --- | --- |
| 210 | 0 | (210 − 210) ÷ 6,785: the first step of the fall |
| 1,906 | 0.25 | (1,906 − 210) ÷ 6,785 |
| 3,603 | 0.50 | (3,603 − 210) ÷ 6,785: the middle of the fall |
| 5,299 | 0.75 | (5,299 − 210) ÷ 6,785 |
| 6,995 | 1 | (6,995 − 210) ÷ 6,785: the last step of the run |

**Q12. In the cosine formula, why "1 + cos" and why the 0.5?**
The formula needs a factor that starts at 1 (full peak) and ends at 0 (end LR). The cosine alone does not give that,
so two small corrections are applied:

| piece | at position 0 | at position 0.5 | at position 1 | why it is there |
| --- | --- | --- | --- | --- |
| π × position | 0 | π/2 | π | uses exactly half a cosine wave: from its top to its bottom, so it only falls and never rises again |
| cos(π × position) | +1 | 0 | −1 | the cosine goes from +1 to −1 |
| 1 + cos(...) | 2 | 1 | 0 | adding 1 moves the range to 2 … 0, so it ends at 0 and is never negative |
| 0.5 × (1 + cos(...)) | 1 | 0.5 | 0 | halving makes the range 1 … 0: this is the factor |

Then `LR = end + factor × (peak − end)`: factor 1 gives the peak, factor 0 gives the end LR, factor 0.5 gives the
value exactly half-way between them.

One step worked out by hand (position 0.25, peak 1e-4, end 1e-5):

- cos(π × 0.25) = 0.707
- 1 + 0.707 = 1.707
- 0.5 × 1.707 = 0.854
- LR = 1e-5 + 0.854 × (1e-4 − 1e-5) = 1e-5 + 7.68e-5 = **8.68e-5**

**Why a cosine and not some other curve:** the cosine is flat at its top and flat at its bottom. So the LR leaves the
peak slowly and arrives at the end LR slowly, with no sudden change of speed at either end. It is a common choice in
published recipes, not something proven to be the best; linear is used just as much (section 10).

**Q13. Why the 0.1 (end LR = peak ÷ 10)?**
It is not part of the cosine formula. It is only our trainer's default for the end LR (`BLU_SFT_MIN_LR` = peak ÷ 10),
the same ratio as our pretraining (3.5e-4 → 3.5e-5). It is a chosen value and can be changed: Tülu 3 and Moonlight
end at 0, and the same formulas work with any end LR.

**Q14. Why does the formula have "min LR + ... × (peak LR − min LR)"? What is the idea of the whole formula?**

The formula as it is in the code (`get_lr`), for a step in the fall:

    decay_ratio = (step − (warmup + stable)) ÷ ((max_steps − (warmup + stable)) − 1)
    coeff       = 0.5 × (1 + cos(π × decay_ratio))
    lr          = min_lr + coeff × (max_lr − min_lr)

(`decay_ratio` is the "position" of this document, `coeff` is the "factor", `max_lr` is the peak.)

Each line answers one question:

| line | question it answers | range |
| --- | --- | --- |
| `decay_ratio` | how far through the fall is this step? | 0 (start) to 1 (end) |
| `coeff` | how much of the drop is still left to do? | 1 (all of it) to 0 (none) |
| `lr` | what is the actual LR for that? | peak down to min LR |

**Why (peak − min):** the LR must go from the peak down to the min LR, not down to 0. So the amount it has to lose in
total is peak − min. That is the whole drop.

**Why "min +":** the LR must never go below the min LR. So the min LR is always there, and only the drop part is
scaled by the factor:

    lr = min LR  +  (the part of the drop that is still left)
       = min LR  +  factor × (peak − min)

Check with peak 1e-4 and min 1e-5 (the whole drop is 9e-5):

| factor | min LR + factor × drop | LR | meaning |
| --- | --- | --- | --- |
| 1 | 1e-5 + 1 × 9e-5 | 1e-4 | nothing dropped yet: the peak |
| 0.854 | 1e-5 + 0.854 × 9e-5 | 8.68e-5 | 85.4% of the drop still left |
| 0.5 | 1e-5 + 0.5 × 9e-5 | 5.5e-5 | half of the drop done |
| 0.146 | 1e-5 + 0.146 × 9e-5 | 2.32e-5 | 14.6% of the drop still left |
| 0 | 1e-5 + 0 × 9e-5 | 1e-5 | the whole drop done: the min LR |

**What would go wrong without these two pieces:** with `lr = factor × peak`, factor 0 gives LR 0 at the last step, not
the min LR. With `lr = min + factor × peak`, factor 1 gives peak + min at the start, which is above the peak. Only
`min + factor × (peak − min)` gives exactly the peak at factor 1 and exactly the min LR at factor 0.

**The cosine only decides the factor.** The third line is the same in the linear schedule; only the second line
differs:

| schedule | factor | LR |
| --- | --- | --- |
| cosine | 0.5 × (1 + cos(π × position)) | min + factor × (peak − min) |
| linear | 1 − position | min + factor × (peak − min) |

Both factors go from 1 to 0. The linear factor loses the same amount every step. The cosine factor loses little at
the start, most in the middle, and little at the end.

**Q15. cos(0) = 1 and cos(π) = −1, so the factor goes from 1 to 0 and the LR from the peak to the min LR? Why three
formulas for one cosine fall; are all three needed; does everyone do it this way?**

Yes to the first part: cos(0) = +1 gives factor 1 and the peak; cos(π) = −1 gives factor 0 and the min LR.

It is **one** formula. The code writes it on three lines only to make it readable. On one line:

    lr = min_lr + 0.5 × (1 + cos(π × (step − first step of the fall) ÷ (steps in the fall − 1))) × (peak − min_lr)

None of the three pieces can be left out, because each one does a different job:

| piece | job | what breaks without it |
| --- | --- | --- |
| position (step → 0 … 1) | the cosine needs an input that runs from 0 to π over exactly the steps of the fall | the curve would not start at the first step of the fall or finish at the last step |
| 0.5 × (1 + cos) | cos gives +1 … −1; this turns it into 1 … 0 | the factor would go negative |
| min + factor × (peak − min) | turns the factor 1 … 0 into an LR between the peak and the min LR | the LR would end at 0 or start above the peak (Q14) |

Others write the same formula:

- **The paper it comes from** (Loshchilov and Hutter, "SGDR", 2017): `η = η_min + 0.5 × (η_max − η_min) × (1 + cos(π × T_cur ÷ T_max))`.
- **PyTorch `CosineAnnealingLR`:** the same expression, with `eta_min` as the min LR.
- **Hugging Face `get_cosine_schedule_with_warmup`:** `0.5 × (1 + cos(π × progress))` times the peak LR. This is the
  same formula with min LR = 0, so it ends at 0.

So ours is the standard cosine formula with a min LR of peak ÷ 10.

**Q16. If the min LR is 0, does the last step have LR 0?**
Yes, in our trainer. At the last step the position is exactly 1, the factor is exactly 0, so LR = min LR. With min
LR = 0 the last step has LR 0 and changes no weight; the run in effect has one step fewer. Only the last step is
exactly 0. The steps before it are small but not 0 (6,996 steps, peak 1e-4, min 0):

| step | cosine | linear |
| --- | --- | --- |
| 6,993 | 2.1e-11 | 2.9e-8 |
| 6,994 | 5.4e-12 | 1.5e-8 |
| 6,995 (last) | 0 | 0 |

With our default (min LR = peak ÷ 10) the last step has peak ÷ 10 (1e-5 here), not 0.

---

## 13. Where it is in the code

| What | Where |
| --- | --- |
| The LR formula (warm-up, stable, cosine or linear fall) | `bluscript_sft.cpp`, function `get_lr` (line 1038) |
| The SFT defaults: max steps, warm-up steps, peak, end LR | `bluscript_sft.cpp`, lines 1579-1584 |
| The Muon group's peak and end LR | `bluscript_sft.cpp`, lines 1589-1597 |
| The schedule choice (`BLU_SFT_SCHEDULE`) | `bluscript_sft.cpp`, lines 1599-1612 |
| The check for a fall of at least 2 steps | `bluscript_sft.cpp`, line 1679 |
| The LR applied to each optimizer group every step | `bluscript_sft.cpp`, lines 3697 and 3790 |
| The launcher's `SCHEDULE`, `WARMUP`, `LR` | `sft/run_sft.sh`, lines 46 and 87-88 |
| The settings table for users | `how_to_handle_sft.md`, lines 138-139 |
