# Overfitting check: train loss, val loss, dev suite and unseen suite

Written 2026-10-22. It collects every doubt I had about validation, overfitting, the dev suite and the unseen suite, with the answers,
in the order they came up. Section 9 is the short summary to remember.

Extended 2026-10-23, after the run finished: sections 11–20 give the finished run's numbers, charts and files, and every
doubt I had while reading them (memorization vs overfitting, what a "nat" is, the "4 of 5" step matching, the
memorization measure, the train–val gap, why the train loss jumps from step to step). Section 20 is their short summary.
Four statements (in sections 0, 5.2 twice, and 9) were made more precise on that day; each is marked "Corrected 2026-10-23".

Every doubt section starts with a **Context** paragraph: what the run was doing at that moment, what I was looking at,
and what made me ask, so the question still makes sense later. The run itself (model, data, settings, the three log
parts) is described once in section 11.1.

**The run these doubts are about, in one paragraph.** On 2026-10-22 we started the Tülu 3 SFT run of our 1.7B model
`fullchess` (28 layers, Muon, peak learning rate 5e-5, 2 epochs = 11,590 steps of 20 packed rows, val loss on 2,000
held-out chats every 200 steps) at 15:51 IST on GPU 0 of the box 86.38.182.101. Sections 2–8 were asked on that
afternoon while it ran in epoch 1. The run was stopped twice and continued from checkpoints 8,000 and 9,000, and
finished on 2026-10-23 at 14:42 IST on GPU 3 of the 4x box 95.133.252.52. Sections 11–19 were asked while reading its
finished loss curves.

---

## 0. The answer in five lines

1. Overfitting to the training data is detected with **two losses**: the **train loss** and the **val loss**.
   Both falling = the model is learning. **Train falling while val rises** = the model is memorizing its training chats.
   (Corrected 2026-10-23: more precisely, train falling while val rises is **overfitting**, the late stage.
   Memorization starts earlier and first shows as a **growing gap** between train and val while val is still falling;
   section 12.)
2. Validation every 200 steps **never changes the model**. It only reads the val chats; it never trains on them.
3. The **model never trains on the dev suite or the unseen suite**. The names are about **our** decisions,
   not about the model: we **use** the dev suite to make choices; we **hide** the unseen suite until the end.
4. **Choosing the checkpoint** (end of epoch 1 or end of epoch 2) = val loss + dev suite. **Then** the unseen suite,
   run **once** on the chosen checkpoint, gives the honest final score.
5. Dev score much higher than unseen score means **our choices were tuned to the dev suite**. It does not mean the
   model memorized the dev questions; it never saw them in training.

---

## 1. Basic terms

### 1.1 What one training step does

A training step has three parts:

1. **Forward:** the model reads the chats and gives a probability for every possible next token, at every position.
2. **Backward:** from those probabilities and the correct tokens, it computes how each weight should change so the
   correct tokens become more likely (the gradients).
3. **Optimizer step:** the weights are changed.

**Only part 3 changes the model.** The model "learns" or "memorizes" a chat **only** when that chat goes through
parts 2 and 3.

### 1.2 What the loss is

The loss at one position is **−log(probability the model gave to the correct next token)**.

- The model gives the correct token probability 0.9 → loss 0.105.
- Probability 0.5 → loss 0.693.
- Probability 0.1 → loss 2.303.

So **lower loss = the model gives the correct tokens higher probability**. The loss of a batch is the average over
all its trained positions.

### 1.3 What a "trained token" is

In SFT the model must learn to write **answers**, not the user's questions. So only the answer positions count.

A small example chat (simplified; real chats use our chat template, but the idea is the same):

```
position   0           1     2     3     4              5     6        7         8
token      <|user|>    \n    Hi    \n    <|assistant|>  \n    Hello    " there"  <end>
                       '-------- prompt (user turn) --------'  '------- answer -------'
```

The model reads the token at position t and must predict the token at t+1. The file `labels.bin` stores, for
every position, the token to predict, **but only when that next token belongs to the answer**; otherwise it stores
**65535**, which means "no loss here":

```
position   0        1      2      3      4              5        6         7
input      <|user|> \n     Hi     \n     <|assistant|>  \n       Hello     " there"
label      65535    65535  65535  65535  65535          Hello    " there"  <end>
                                                        '---- trained: 3 tokens ----'
```

- Positions 0–4 would teach the model to write the user's question. It should not learn that, so the label is 65535:
  no loss, no gradient.
- Positions 5–7 predict the answer, including the end token (so the model learns to stop). These **3 positions are
  the trained tokens** of this chat.

`prepare_sft_data.py` counts the trained tokens of every row and stores the count in `valid_counts.bin`.

### 1.4 The val set

- 2,000 chats from the Tülu 3 mix were **cut out before training** (`--skip-kept 2000` for the train split). They are
  never in the training data (checked: train and val do not overlap).
- They are packed into **320 rows** of 4,096 positions = **1,310,720 positions**.
- Of those, **898,276 are trained tokens** (answer tokens), 68.5%. The rest are prompt tokens and the empty tail at
  the end of each row (all label 65535).
- The training data has the same share: 68.6% of its positions are answer tokens. Val has about 449 answer tokens
  per chat, train about 439, so the val set looks like the training data.

---

## 2. Doubt: "Why is it always 898,276 trained tokens in every [val] line?"

**Context.** 2026-10-22, the first hour of the run (around steps 600–700). I was reading the watchdog file
`monitor.txt`, which copies every new val line of the log, for example
`VAL [val] step 600 | val_loss 0.969231 | 898276 trained tokens`. Every val line showed exactly the same
898,276, while the `trained` number in the normal step lines changed every step (51k–62k). So I asked whether that was
planned, and whether those 898,276 tokens were the whole val set, and whether they were answer tokens or prompt tokens.

Because the val set is **fixed**. Every validation (step 0, 200, 400, …) runs over **all 320 val rows**
("80 batches per rank" in the `[sft]` line = 80 micro-batches of 4 rows), so it always counts the same 898,276
answer tokens.

That is the point of a fixed val set: every val loss is measured on **exactly the same tokens with the same
denominator**, so when the val loss changes, the **model** changed, not the data.

The `trained` number in the **step** lines does change (51k–62k), because every training step gets different chats.
A step is 20 rows = 81,920 positions, of which about 57,400 (70%) are answer tokens.

---

## 3. Doubt: "Validation runs every 200 steps. Doesn't the model see the val chats every time and remember them?"

**Context.** 2026-10-22 afternoon, the run in epoch 1. We had set the run to measure the val loss every 200 steps
(`BLU_SFT_VAL_FREQ=200`), so by the end it would run over the same 2,000 val chats 58 times. I was thinking about how
to check overfitting after the run, with a "dev suite" and an "unseen suite", and asked whether the val chats count as
"seen" because the model goes over them again and again, and whether it could remember them.

**No.** Validation does **only the forward part** (section 1.1). The code (`bluscript_sft.cpp:2304-2329`):

```
if (sft && (step % cfg.val_freq == 0 || step == cfg.max_steps - 1)) {
    autograd::NoGradGuard no_grad;        // no gradients are recorded
    ...
    Tensor logits = model.forward(b.input);   // forward only
    ... the losses of the 898,276 answer tokens are summed and divided -> printed as val_loss
}                                          // no backward, no optimizer step
```

- No backward, no optimizer step → **the weights are exactly the same before and after a validation**.
- So the model cannot learn or remember anything from the val chats, whether we validate 58 times or 5,000 times.
- "The model sees a chat" in the training sense means "the chat changed the weights". Val chats never do.

---

## 4. Doubt: "If the model memorizes, it answers correctly, so the val loss should go DOWN, not up?"

**Context.** 2026-10-22 afternoon, the run in epoch 1. The answer to the overfitting question was "train loss down while
val loss up = memorization". That sounded backwards to me: if a model memorizes, it gives the right answers, so I
expected its loss to go down everywhere, val included. So I asked whether that answer was wrong, and whether the val loss
alone (the "dev suite") is enough or the "unseen suite" is needed to see memorization.

**No, because the val chats are not the chats it memorizes.** The val chats were never trained on (section 1.4).
Memorizing the training chats can only help on those exact training chats.

What memorizing does, in terms of the loss:

- In epoch 2 the model sees every training chat a second time. If it memorizes, it pushes the probability of the
  **exact wording of the training answers** towards 1. → the **train loss falls fast**.
- Probability is limited: making some tokens more likely makes the others less likely. On a **new** chat, the model
  now confidently expects the wording it memorized, and the token that actually comes next in the new answer often
  gets a **lower** probability than before. → the **val loss rises**.

Small example:

- A training answer: `The capital of France is Paris.`
- A val answer to a similar question: `Paris is the capital of France.`
- A memorizing model gives `The` a high probability as the first word. The val answer starts with `Paris`, so the
  model gave the correct token a lower probability, and the loss at that position goes up.

### Proof from our own runs (work doc §37.5, 2026-10-07)

5,000 chats, 2 epochs, Muon, base `blumodel_step_8000`. Epoch 2 started at step 78; step 80 is the first validation
in epoch 2.

| Run | val at step 80 (start of epoch 2) | val at the end of epoch 2 | train loss, epoch 2 mean | What happened |
| --- | --- | --- | --- | --- |
| Muon lr 5e-5 | 1.2920 | **1.2843** (went down) | 1.208 | still learning in epoch 2 |
| Muon lr 1.2e-4 | **1.2747** (the best of all runs) | 1.2820 (went **up**) | **1.084** (very low) | **overfitted in epoch 2** |

In the 1.2e-4 run, the train loss went very low while the val loss went up. That is memorization, measured.

---

## 5. Doubt: "So how exactly is overfitting detected?"

**Context.** 2026-10-22 afternoon, right after section 4. The run trains 2 epochs, so in epoch 2 the model sees every
training chat a second time, and overfitting was the main risk of the second epoch. I asked what exactly to watch to
detect it, whether the val loss every 200 steps is that check, and how the model could remember chats it only reads at
validation.

With **two losses, on the same model, at the same steps**:

| Loss | Computed on | What it measures |
| --- | --- | --- |
| **train loss** | the chats the model is **trained on** at that step | how well it fits its training data |
| **val loss** | 2,000 chats the model **never trains on** | how well it does on **new** chats of the same kind |

Watch both over the steps:

The meaning depends on the **epoch**, because the train loss means something different in each (section 5.1):

| Epoch | Train loss | Val loss | Meaning |
| --- | --- | --- | --- |
| 1 | going down | going down | **real learning** |
| 1 | going down | flat | **not memorization** (every train chat is new in epoch 1). The model gains on things the val set does not measure, or the gain is too small or noisy to show on val. Check with the dev suite (section 5.2) |
| 1 | going down | **going up** | **a problem**: duplicates in the training data (memorizing repeats = overfitting inside epoch 1), or train and val are different kinds of data, or the learning rate is too high (section 5.2) |
| 2 | going down | going down | **real learning continues** in epoch 2 |
| 2 | going down | flat | **most likely memorization starting**: the train gains are on chats already trained on once. Not proven yet (section 5.2) |
| 2 | going down | **going up** | **overfitting, proven**: memorization now makes new chats worse |
| any | flat | flat | nothing more to learn at this learning rate |

### 5.1 Doubt: "If the train loss falls and the val loss is flat, how is that learning? Is the train loss a guarantee?"

**Context.** 2026-10-22, while reading the first version of the table in section 5. One row said "train falling, val
flat = learning has mostly stopped". I pasted that row back and asked: if the val loss does not move, what guarantees
that the falling train loss is real learning and not memorization, and does a falling train loss guarantee anything?

It is **not** learning in the useful sense. (An earlier version of this table called it "learning has mostly stopped",
which was too soft.)

- **Val flat** = on **new** chats the model is no longer getting better (and not worse either).
- **Train still falling** = the model keeps getting better on **its own training chats only**. That extra drop does not
  carry over to new chats, so it is **fitting the training chats: memorization starting**. The gap between train and
  val grows. Val has not got worse yet, so it is not harmful yet, but it is the point where more steps stop helping.

**A falling train loss alone guarantees nothing**, with one exception:

- In **epoch 1**, each batch's loss is computed **before** the model trains on that batch, so every epoch-1 train loss
  value is on chats the model has not seen yet. It behaves like a val loss: a falling epoch-1 train loss **is** real
  learning.
- In **epoch 2** the chats were already trained on once. A falling train loss there can be memorization. Only the val
  loss can tell which.

**What gives the guarantee:**

| Question | What answers it |
| --- | --- |
| Is it learning things that carry over to new chats of this kind? | **val loss going down** (and the train loss in epoch 1) |
| Is it memorizing? | **train going down while val is flat or rising**: the train–val gap grows |
| Did it gain a **skill** (following instructions, stopping correctly, math)? | **dev suite benchmarks** |

**One honest limit of the val loss.** It is the average over all 898,276 answer tokens. A narrow skill (for example
ending the answer at the right place, or following a format) touches only a few of those tokens, so it can change
while the average hardly moves. So "val flat" means "no more **general** improvement on Tülu-style answers"; it does not
prove that **no** skill changed. That is why the dev suite benchmarks are run on both checkpoints before choosing.

### 5.2 Doubt: "Train down + val flat cannot be called memorization in epoch 1, right? And what if, in epoch 1, the train loss goes down but the val loss goes UP?"

**Context.** 2026-10-22, after section 5.1 said "train down + val flat = memorization starting" without naming the
epoch. Our run was still in epoch 1, where every training chat is new, so I pointed out that in epoch 1 a flat val cannot
mean memorization, that in epoch 2 it most likely does, and asked what it would mean if in epoch 1 the train loss went
down while the val loss went up, for example because the data contains the same chats again and again.

Correct: section 5.1 first said "train down + val flat = memorization starting" without naming the epoch. That is
only true in **epoch 2**. The table above is now split by epoch. The reasons, case by case:

**Two facts about our data first** (they decide which explanations are possible):

- The chats are shuffled before the val and train splits are cut (`prepare_sft_data.py:244`), so the 2,000 val chats
  are a **random sample of the same mix** as the training chats.
- The packed training rows are shuffled too (`prepare_sft_data.py:290`), and the loader reads them in that same file
  order **in both epochs** (`SftDataLoader.h:24`). So the training stream is a random mix, not one source after another.

#### Case 1: epoch 1, train going down, val flat → not memorization

- In epoch 1 every train-loss value is measured on chats the model has **not trained on yet** (the loss is computed
  before the update). A falling epoch-1 train loss is real improvement on new chats.
- So if val is flat at the same time, the model is not memorizing. Possible reasons:
  - it is improving on kinds of content that the 2,000 val chats contain little of (the val set is a sample; small
    categories may have only a few chats in it);
  - the train loss of single steps is noisy (every step has different chats), so a short "fall" can be noise.
    Always smooth it (average of ~100 steps) before reading a trend;
  - the improvement is real but too small to move the val average.
- What to do: nothing during the run. At the end, the dev suite benchmarks show whether something was learned that val
  does not measure.

#### Case 2: epoch 2, train going down, val flat → most likely memorization starting, not proven yet

- In epoch 2 the train chats are the same chats as in epoch 1, already trained on once. A train loss that keeps
  falling while val does not move means the gains are on **those chats only**: most likely the start of memorization.
- **It is not proven** until val actually rises. Val does not have to rise: if the memorization stays small, val can
  stay flat to the end. Near the end of this run the cosine schedule makes the learning rate small (down to 5e-6), so
  the updates get small and val may simply stay flat.
- If val **goes down again** later in epoch 2, then epoch 2 was still giving real learning, not memorization.
- If val **goes up**, that proves overfitting, and the end-of-epoch-1 checkpoint is the better model.

#### Case 3: epoch 1, train going down, val going UP → a problem, with three possible causes

In epoch 1 the train chats are new, and in our setup train and val are random samples of the same mix, so the two
losses should move together. If they move in opposite directions, one of these is happening:

1. **Duplicates in the training data** (the cause I guessed): the same or almost the same chats appear many times.
   Then the "new" chats of epoch 1 are really repeats, the model memorizes them even in epoch 1, the train loss falls
   and the val loss rises. **This is overfitting inside epoch 1.** How to check: count exact and near-duplicate chats
   in the training data.
2. **Train and val are different kinds of data**, for example val taken from other sources, or a training stream that
   is not shuffled and goes through one source at a time. Then "better on the current training chats" does not mean
   "better on val". This is not overfitting; it is a mismatch. In our run both are random samples of the same mix and
   the rows are shuffled, so this is unlikely.
3. **The learning rate is too high**: large updates fit the current batches but damage what the model knew, so it gets
   worse on everything else. Usually seen together with grad-norm spikes. This is also not memorization; it is
   damage.

(A small wobble of val in the very first steps can also come from the model switching from pretraining text to the
chat format. In this run val fell from step 0, so that did not happen.)

#### A direct memorization measure for epoch 2

Because both epochs read the rows in the same order, step s of epoch 2 trains on (almost) the same rows as step s of
epoch 1. Epoch 1 has 5,795.2 steps (28,976 micro-batches of 4 rows, 5 per step), so epoch 2 is shifted by one
micro-batch: 4 of the 5 micro-batches of a step line up. So:

- epoch-1 train loss of those rows = their loss **before** the model ever trained on them;
- epoch-2 train loss of the same rows = their loss **after** training on them once;
- the drop between the two, minus the drop of the val loss over the same time, is how much the model improved on
  **those exact chats beyond what it improved on new chats**. That extra part is memorization, measured directly.

Compare smoothed values (for example the averages of 100-step windows), not single steps.

(Corrected 2026-10-23: "step s of epoch 2" means the s-th step **counted from the start of epoch 2**, that is overall
step 5,795 + s. Epoch-2 step 5,795 + s trains on **16 of the same 20 rows** as epoch-1 step s; section 14 shows it row
by row from zero. The measure was run on the finished run with 200-step windows: sections 15–17.)

### Why the train loss drops at the start of epoch 2 (this explains "seen" and "not seen")

- **In epoch 1**, each chat's loss is computed **before** the model trains on it (forward first, then the update).
  So the epoch-1 train loss is on chats the model has **not seen yet**; it behaves almost like a val loss.
- **In epoch 2**, the same chats come again, now **already trained on once**. So the train loss **drops at the epoch
  boundary** (step ~5,795 in this run). That drop alone is normal and expected.
- From then on, the **gap between the train loss and the val loss** shows how much the model fits its own chats
  rather than new ones. A small gap with val still falling is fine. **A growing gap with val rising is overfitting.**

The train loss of single steps jumps around (1.0 to 1.3 in this run), because every step has different chats. To
read the trend, smooth it (for example the average of every 100 steps) or compare epoch means.

(Corrected 2026-10-23: "1.0 to 1.3" was the start of the run. Over the whole run, 90% of steps lie between 0.665 and
1.033 (lowest 0.469, highest 1.367); in epoch 2, 90% lie between 0.644 and 0.919. Section 19 proves that the jumps
come from the rows in each step, not from the model.)

---

## 6. Doubt: "What were the dev suite and the unseen suite? We discussed it before."

**Context.** 2026-10-22 afternoon, the run in epoch 1. I remembered that we had planned two kinds of evaluation for SFT
earlier (on 10-08) and called them "dev suite" and "unseen suite", and wanted those names and that plan brought back, so
that after the 2-epoch run we could tell from the losses and from those suites whether the model overfitted or really
learned.

Yes, on **2026-10-08**, when I asked how to evaluate SFT data (work doc §38.6–38.7, "Q7b: how to evaluate SFT data";
the 10-08 EOD mentions the dev suite too). The plan from that day:

| Suite | Benchmarks (our plan, 10-08) | Used for |
| --- | --- | --- |
| **dev suite** | answer val loss, end-token rate, IFEval, TruthfulQA MC2, PopQA, DROP, HellaSwag / ARC / WinoGrande (base vs SFT), GSM8K as a floor check | making choices: which checkpoint, which LR, which data |
| **unseen suite** | IFBench or IFEval-OOD, MMLU-Pro, AGIEval | one honest final number, never used for choices |

Settings agreed then: the SFT model is tested zero-shot in our chat template, the base model few-shot (OLMES style);
noisy generation tests are averaged over 3 runs. Running these needs an export to Hugging Face format and a
generation setup (post-run work).

What we agreed on 10-08 ("Dev vs unseen"):

- Every choice made by looking at the dev score also picks up **luck on those exact questions**. Example: the best of
  10 truly equal options lands about **+3.2 points** above its real level on IFEval.
- So the dev score of the final model is a bit too high.
- An unseen suite, designed separately and **run once on the frozen model**, gives the honest number.
- In SFT, dev and unseen are **separate benchmark lists**, not slices of the training data.

---

## 7. Doubt: "Is the val loss every 200 steps the dev suite? Do we tweak the run based on it?"

**Context.** 2026-10-22, right after section 6. The val loss is printed every 200 steps and we were watching it during
the run. I wanted to know whether that val loss is the "dev suite", and whether watching it means we change the run
(learning rate, data) based on it while it runs.

- The val loss is **one item of the dev suite** ("answer val loss" in the list above).
- **During the run we do not change anything based on it.** We only watch it.
- It acts as "dev" when we **make a decision** with it. At the end, "keep the epoch-1 checkpoint or the epoch-2
  checkpoint?" is such a decision. That is fine; that is what a dev measure is for.

---

## 8. Doubts: "Is the dev suite the chats the model trained on? Do we use the unseen suite to pick epoch 1 or 2?

## If it does well on dev and badly on unseen, did it memorize?"

**Context.** 2026-10-22, right after section 7. My own understanding at that point was: the unseen suite is run at the
end and its results pick the checkpoint (end of epoch 1 or end of epoch 2); the dev suite is made of chats the model has
already seen; and a model that does well on dev but badly on unseen has memorized. I asked whether that was right.

These three mixed two different ideas. Corrections:

### 8.1 The four kinds of data

The words "dev" and "unseen" are **not about what the model has seen**. **The model never trains on either suite.**
They are about **what we look at when we make choices.**

| Data | Does the model train on it? | Do **we** use its scores to make choices? | What it tells us |
| --- | --- | --- | --- |
| **training chats** (741,413) | **yes** | — | train loss |
| **val set** (2,000 held-out chats) | no | yes | with the train loss: is the model overfitting its training chats? |
| **dev suite** (IFEval, GSM8K, HellaSwag, …) | **no** | **yes**: we use it to pick the checkpoint, LR, data | which choice gives the best skills |
| **unseen suite** (IFBench, MMLU-Pro, AGIEval) | **no** | **no**: we do not look at it until the very end, then run it **once** | the honest final score |

So:

- **Dev suite ≠ chats the model trained on.** It is benchmark questions the model never trained on. It is called
  "dev" because **we** use its scores while developing: we try options and keep the one that scores best on it.
- **Unseen suite ≠ "used to pick epoch 1 or epoch 2".** If we picked the checkpoint with it, we would be using it to
  make a choice, and it would turn into a second dev suite. "Unseen" means **unseen by our decisions**: kept hidden
  until the choice is made, then run once.

**Picking epoch 1 vs epoch 2 = val loss + dev suite. Then the unseen suite, once, on the winner.**

### 8.2 What "good on dev, worse on unseen" means

The model never trained on the dev questions, so it is **not** memorization of those questions. It means **our
choices were tuned to the dev suite**:

- **Luck:** when we try many options and keep the best dev score, we also keep whichever option got lucky on those
  exact questions. From the 10-08 notes: the best of 10 equally good options looks about +3 points better on IFEval
  than it really is.
- **Benchmark-specific skill:** the model got good at the kind of task in the dev suite but not at the general skill.
  From the 10-08 notes: models tuned on IFEval's constraints fail many of IFBench's 58 new constraints.

---

## 9. Summary: three kinds of "too much", and how each is detected

| Problem | How to see it |
| --- | --- |
| **1. The model memorized its training chats** (overfitting to the training data) | **train loss down + val loss up**. This is the main overfitting check, and the only one that needs no benchmark. (Corrected 2026-10-23: memorization alone shows as a **growing train–val gap**, measured directly in sections 15–17; **train down + val up** is when that memorization has turned into **overfitting**. Section 12.) |
| **2. Our choices were tuned to the dev suite** | dev score clearly higher than the unseen score |
| **3. SFT made the model forget base knowledge** | base benchmarks (HellaSwag, ARC, WinoGrande) lower after SFT than before SFT |

And which tool answers which question:

| Question | Answered by |
| --- | --- |
| Is the model overfitting its training chats? | train loss vs val loss |
| Does the model work on new chats of the same kind? | val loss |
| Which checkpoint is better: end of epoch 1 or end of epoch 2? | val loss + dev suite, on both checkpoints |
| Did the model gain skills (instructions, math)? Did it forget base knowledge? | dev suite benchmarks, base vs SFT |
| How good is the model we finally picked, honestly? | unseen suite, run once on that checkpoint |

---

## 10. How all of this is used after a 2-epoch SFT run

1. **Keep a checkpoint from the end of epoch 1** as well as the final one, so the two can be compared.
2. **Plot** the train loss (smoothed, for example over 100 steps) and the val loss against the step, with a line where
   epoch 2 starts.
3. **Read the val loss in epoch 2** (section 5.2):
   - still falling → epoch 2 was still real learning;
   - flat while the train loss keeps falling → most likely memorization starting, not proven; let the dev suite decide;
   - rising → overfitting, proven; the end-of-epoch-1 checkpoint is the better model.
4. **Then** run the dev suite on both checkpoints to pick the better one, and the unseen suite **once** on that one.

For the finished Tülu 3 run, steps 1–3 are done (2026-10-23): sections 11–20. Step 4 is not done yet: the dev suite and
the unseen suite need an export to Hugging Face format, a generation setup in our chat template and a benchmark harness.

---

## 11. The finished run (2026-10-23): what was measured, where the files are

### 11.1 The run

- **Model:** the 1.7B base model `fullchess`: 28 layers, 18 query heads and 6 key/value heads, head size 128, RoPE
  theta 50,000, context 4,096.
- **Data:** the Tülu 3 SFT mix without the Aya part: 741,413 training chats packed into **115,904 rows** of 4,096
  positions, **325.7M answer tokens** per epoch. Val = the 2,000 held-out chats of section 1.4 (898,276 answer tokens).
- **Training:** 2 epochs = **11,590 optimizer steps**. One step = **20 rows** (4 rows at a time, 5 times; section 14).
  Muon optimizer, peak learning rate 5e-5 (it rises over the first 348 steps, then falls along a cosine to 5e-6 at
  the end). Val is measured every 200 steps and at the last step.
- The run was stopped twice and continued from a checkpoint, so the log is in **three parts**:

| Part | Log file | Steps in it | Where and when | How it ended |
| --- | --- | --- | --- | --- |
| 1 | `run.log` | 0 – 8,723 | old box, GPU 0, 10-22 15:51–20:43 IST | stopped |
| 2 | `run_resume1.log` | 8,000 – 9,194 | same box, continued from checkpoint 8,000, 10-22 20:48–21:30 IST | stopped |
| 3 | `run_resume_4x.log` | 9,000 – 11,589 | 4x box, GPU 3, continued from checkpoint 9,000, 10-23 13:15–14:42 IST | finished, exit 0 |

- **The steps of the final model:** 0–7,999 from part 1, 8,000–8,999 from part 2, 9,000–11,589 from part 3. After each
  continue, the steps after the checkpoint were trained again, so the end of parts 1 and 2 is repeated in the next part.
- **Check that the three parts join exactly.** At each point where the run was continued, the stopped run and the
  continued run print the same numbers for the same step:
  - step 8,000: val **0.856308** and train **0.653908**, in part 1 and in part 2;
  - step 9,000: val **0.854744** and train **0.567569**, in part 2 and in part 3.

  So the joined step-by-step history of the final model is exact.

### 11.2 The files

All in `post_training_research/tulu3_sft_1022/` (the raw logs are also on the laptop in
`~/Downloads/box_backup_2026_10_22/tulu3_run_final/`):

| File | What it is |
| --- | --- |
| `tulu3_sft_1022_full_run.log` | the three logs joined as they are, nothing removed or changed, with a 7-line header saying which part covers which steps (12,776 lines = 12,763 log lines + 13 header and separator lines) |
| `trajectory.csv` | the final model's 11,590 steps: train loss, val loss where measured, learning rate, grad norm, trained tokens, tokens per second, MFU, and which log each step came from |
| `loss_train_val.png` | chart: train loss (every step, and the mean of the last 100 steps) and val loss, with a line where epoch 2 starts |
| `loss_val.png` | chart: val loss alone, the whole run and zoomed in from step 3,000 |
| `memorization_check.png` | chart: the memorization measure of section 15 |
| `memorization_windows.csv` | the numbers behind that chart: 28 windows of 200 steps |
| `analyze_run.py`, `analysis_output.txt` | the script that makes everything above, and its full printout |

![Train and val loss](../post_training_research/tulu3_sft_1022/loss_train_val.png)

![Val loss, whole run and zoomed](../post_training_research/tulu3_sft_1022/loss_val.png)

### 11.3 The numbers

"Val at step N" is measured at the start of step N, before that step's update: it is the model after N updates, the
same weights as checkpoint `blumodel_step_N` (`bluscript_sft.cpp:2304` and `:2857`).

| | Epoch 1 (steps 0 → 5,795) | Epoch 2 (steps 5,795 → 11,589) |
| --- | --- | --- |
| Val loss | 1.1757 → **0.8608** (down **0.315**) | 0.8608 → **0.8501** (down **0.0107**) |
| Train loss, mean of 500 steps | 1.0399 (first 500) → 0.8431 (last 500) | 0.8122 (first 500) → 0.7737 (last 500) |
| Gap: val − train, at the end of the epoch | **+0.018** | **+0.076** |
| Val going up | never | 3 small rises out of 58 val points, the largest +0.0007 (step 6,200) |

Val along the run:

| Step | 0 | 200 | 400 | 600 | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | 5,800 | 6,000 | 7,000 | 8,000 | 9,000 | 10,000 | 11,000 | 11,589 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Val | 1.1757 | 1.0489 | 0.9928 | 0.9692 | 0.9433 | 0.9096 | 0.8910 | 0.8768 | 0.8669 | 0.8607 | 0.8599 | 0.8590 | 0.8563 | 0.8547 | 0.8528 | 0.8508 | **0.8501** |

- In epoch 2, val went down by 0.0009, 0.0027, 0.0016, 0.0019, 0.0020 in each block of 1,000 steps. **The lowest val is
  the last one.**
- Speed: 43,062 tokens/s on the old box (parts 1 and 2), 44,024 tokens/s on the 4x box (part 3); MFU 51.3% → 52.5%.
- **A note on reading the log.** The first line of each part says `[precision] bf16 logits ON ... the REPORTED loss is
  bf16 and quantizes to ~0.06 steps`. For this SFT trainer that warning is out of date: the printed losses are **not**
  rounded to bf16. Proof from the log: step 9,001 prints train 0.792157 in part 2 and 0.792117 in part 3, only 0.00004
  apart. Near 0.79, two neighbouring bf16 numbers are 0.0039 apart, so both values cannot be bf16 numbers. So even small
  val changes like 0.0003 are real measurements.

### 11.4 The result in short

1. **No overfitting.** Val went down all the way through both epochs, and its lowest value is the last one.
2. **Epoch 2 gave little real learning.** On chats the model never trained on, epoch 2 gained only 0.0107 nats, which
   is 3.4% of epoch 1's 0.315.
3. **Memorization is there, and it was measured directly:** about **0.06 to 0.07 nats**, steady through epoch 2
   (sections 15–17). At the end, memorization (0.057) is about **4.7 times** the real learning of the same time (0.012).
4. **Which checkpoint is the better model** for real use (following instructions, math, facts) cannot be read from the
   loss; the dev suite decides (section 10, step 4), on `blumodel_step_6000` and the final `blumodel_step_11590`.
   Checkpoint 6,000 is only 205 steps into epoch 2 (val 0.8599, against 0.8608 at the end of epoch 1), so it stands for
   "the end of epoch 1".

---

## 12. Doubt: "You say no overfitting, but some memorization. How can you tell? And if the memorization keeps growing and the val loss goes up, that is overfitting, right? So memorization is the start of overfitting?"

**Context.** 2026-10-23, about 15:00 IST, right after the run finished and `analyze_run.py` printed the numbers of
section 11. The answer I got then read: "No overfitting by val (val never rose, its best value is the last one). But
epoch 2 added little real learning (−0.0107 vs −0.315). Memorization is clearly there, measured directly; the gap grew
from +0.018 to +0.076." So I asked: you say no overfitting, but some memorization (a little overfitting?), how can you
tell? Later, once the measure was clear to me, I added: if this memorization keeps growing and the val loss goes up,
that is overfitting, so memorization is the starting cause of overfitting, and that was not written in this document yet.

Yes. Memorization is the first stage, and overfitting is what it becomes when it starts to hurt new chats.

### 12.1 The two words

- **Memorization:** the model gets better on the **exact chats it trained on** by **more** than it gets better on new
  chats. It is measured by the train–val gap (sections 15–16).
- **Overfitting:** memorization that has started to **hurt new chats**: the **val loss goes up**.

So memorization can be there without overfitting: the model fits its own chats extra well, and new chats still get
better too. That is our run.

### 12.2 How memorization turns into overfitting, stage by stage

| Stage | Train–val gap | Val loss | What is happening |
| --- | --- | --- | --- |
| 1 | stays the same | going down | **real learning only**: every chat improves the same, seen or not |
| 2 | **growing** | still going down | **memorization has started**, but new chats still improve. Not overfitting. **Our run in epoch 2** |
| 3 | growing | flat | memorization keeps growing; new chats stopped improving. The edge |
| 4 | growing | **going up** | memorization now **hurts** new chats: **overfitting** |

Why stage 4 happens (section 4 in more detail): probability is limited, it adds up to 1 over all tokens. When the model
pushes up the exact wording of its training answers, other wordings get less probability. On a new chat whose answer
uses another wording, the correct token now gets a lower probability than before, so the val loss goes up.

Proof from our own runs, the 10-07 sweep (section 4): with Muon learning rate 1.2e-4 the train loss went very low (1.084)
while val went **up** (1.2747 → 1.2820) = stage 4, overfitting. With 5e-5, val was still going down in epoch 2.

**One caution.** Val can also go up **without** memorization, for example when the learning rate is too high (damage),
or when train and val are different kinds of data (section 5.2, case 3). So to call it overfitting, check both: **val
going up AND the gap growing.**

### 12.3 Our run is at stage 2

- The gap grew from **+0.018** (end of epoch 1) to **+0.076** (end of epoch 2): memorization.
- Val still went down, **0.8608 → 0.8501**, and never really rose: not overfitting.

---

## 13. What a "nat" is, the unit of our loss

**Context.** 2026-10-23, the same answer as section 12 gave the memorization as "+0.065 nats" without saying what a nat
is. I asked what "nats" means. The first explanation was a short table (loss → e^(−loss) for our start and end values)
and was not clear enough, so I asked for it in detail, with many examples, including why ln is used.

### 13.1 The loss of one token

At every answer position the model gives a probability to every possible next token. The probabilities add up to 1.
Say the correct next token got probability **p**. Then the loss at this position is

```
loss = −ln(p)
```

| Probability of the correct token (p) | Loss = −ln(p) | In words |
| --- | --- | --- |
| 1.0 | 0 | completely sure, and right |
| 0.99 | 0.0101 | almost sure |
| 0.9 | 0.1054 | quite sure |
| 0.75 | 0.2877 | |
| 0.5 | 0.6931 | a coin flip between two tokens |
| 0.427 | 0.851 | about our final average |
| 0.309 | 1.174 | about the base model's average at step 0 |
| 0.25 | 1.3863 | |
| 0.1 | 2.3026 | the correct token got only 10% |
| 0.01 | 4.6052 | |
| 0.001 | 6.9078 | nearly ruled out the correct token |

Higher probability for the correct token → lower loss. The loss is never below 0.

### 13.2 What "ln" is

ln is the **natural logarithm**. ln(x) answers: "**e** to which power gives x?", where e = 2.71828…

- ln(1) = 0, because e⁰ = 1.
- ln(e) = 1, because e¹ = e.
- ln(2) = 0.6931, because e^0.6931 = 2.
- ln(10) = 2.3026, because e^2.3026 = 10.
- For numbers below 1, ln is negative: ln(0.5) = −0.6931, ln(0.1) = −2.3026.

### 13.3 Why the minus sign

A probability is at most 1, so ln(p) is 0 or negative. The minus sign turns it into a positive number, so that
"0 = perfect" and "bigger = worse".

### 13.4 Why a logarithm at all, and not simply "1 − p"

**Reason 1: a whole answer.** The probability that the model writes a whole answer is the probability of the first
token times the probability of the second token times …. A logarithm turns "times" into "plus":

- An answer of 3 tokens with p = 0.5, 0.8 and 0.25. The whole answer gets 0.5 × 0.8 × 0.25 = **0.1**.
- −ln(0.1) = **2.3026**.
- The three token losses: −ln(0.5) + −ln(0.8) + −ln(0.25) = 0.6931 + 0.2231 + 1.3863 = **2.3026**. The same number.

So the loss of an answer is simply the **sum** of its token losses, and dividing by the number of tokens gives a fair
**average per token** that can be compared between answers of different lengths. With "1 − p" this does not work.

**Reason 2: sure mistakes are punished hard.**

| p | 1 − p | −ln(p) |
| --- | --- | --- |
| 0.5 | 0.5 | 0.693 |
| 0.1 | 0.9 | 2.303 |
| 0.01 | 0.99 | 4.605 |
| 0.001 | 0.999 | 6.908 |

With "1 − p", going from 0.01 to 0.001 hardly changes anything (0.99 → 0.999). With −ln(p) it costs 2.3 more. So
training pushes hardest exactly where the model is most wrong about the correct token.

### 13.5 Why ln (base e) and not log2 or log10

- Any base gives the same ordering: a lower loss in one base is a lower loss in every base. Changing the base only
  multiplies every loss by the same number. For base 2 (the unit is then called a **bit**): 1 nat = 1.4427 bits, so
  our final 0.8501 nats = 1.226 bits.
- ln is used because the model turns its scores into probabilities with **e to the power** (the softmax:
  p = e^(score of the correct token) / sum over all tokens of e^(score)). The ln and the e cancel, and the loss becomes
  simple: **loss = ln(sum of e^(scores)) − (score of the correct token)**. That is exactly what our loss kernel
  computes: `loss = logf(final_sum) + final_max - target_logit` (`LossKernels.cu:410`, `logf` = ln in fp32).
- A loss measured with ln is said to be in **nats** ("natural units"). That is all the word means.

### 13.6 The printed loss is an average, so e^(−loss) is the geometric mean of the probabilities

The printed loss is the average over all trained tokens. Going back from the average loss to a probability gives the
**geometric mean** of the token probabilities, not the plain average:

- 2 tokens with p = 0.9 and p = 0.1. Losses 0.1054 and 2.3026; average 1.2040.
- e^(−1.2040) = **0.300** = the square root of 0.9 × 0.1. The plain average would be 0.5.
- The 3-token answer of 13.4: average loss 2.3026 / 3 = 0.7675, e^(−0.7675) = **0.464** = the cube root of 0.1.

So "average probability of the correct token" below always means this geometric mean.

### 13.7 Our numbers

| Loss | e^(−loss): average probability of the correct token | e^(loss): "perplexity" |
| --- | --- | --- |
| 1.1757 (val at step 0, the base model) | 0.309 | 3.24 |
| 0.8608 (val at the end of epoch 1) | 0.423 | 2.37 |
| 0.8501 (val at the end) | 0.427 | 2.34 |

Perplexity = e^(loss) = 1 / (average probability). A perplexity of 2.34 means the model is, on average, as unsure as if
it had to pick among 2.34 equally likely tokens at every answer position.

### 13.8 What a drop of d nats means

A loss drop of d nats multiplies the average probability of the correct token by e^d:

| Drop d | e^d | Meaning |
| --- | --- | --- |
| 0.693 | 2.00 | the correct token's probability doubles |
| 0.315 (val, epoch 1) | 1.370 | 37% higher |
| 0.1 | 1.105 | 10.5% higher |
| 0.065 (memorization, section 15) | 1.067 | 6.7% higher |
| 0.0573 (memorization at the end of the epochs, section 16) | 1.059 | 5.9% higher |
| 0.0122 (real learning at the end, section 16) | 1.012 | 1.2% higher |
| 0.0107 (val, all of epoch 2) | 1.011 | 1.1% higher |
| 0.01 | 1.010 | 1% higher |

---

## 14. Doubt: "What does '4 of 5' mean? What are 'step k' and 'step k+1'?"

**Context.** 2026-10-23, the same answer as section 12. To measure memorization, epoch-1 steps were compared with
epoch-2 steps, and the answer said "epoch-2 step 5795+k trains on 4 of the same 5 micro-batches as epoch-1 step k". I
asked what "4 of 5" means and what "step k" and "step k+1" are. The first explanation still used "k" and
"micro-batch" without saying what they are, so I asked again for it from zero, in simple words. This is that version.

From zero.

### 14.1 The data is a list of rows

- The training data is **115,904 rows**. One row = 4,096 tokens, holding several chats packed together.
- The rows are numbered row 0, row 1, row 2, …, row 115,903.
- This order was fixed once, when the data was prepared. The trainer reads the rows in this same order every time:
  row 0 first, then row 1, and so on. There is no new shuffle per epoch (`sft/SftDataLoader.h:25-26`).

### 14.2 The GPU takes 4 rows at a time

The trainer gives the GPU **4 rows at once** (our setting B = 4). A group of 4 rows is a **micro-batch**:

```
micro-batch 0 = rows 0, 1, 2, 3
micro-batch 1 = rows 4, 5, 6, 7
micro-batch 2 = rows 8, 9, 10, 11
...
```

115,904 rows ÷ 4 = **28,976 micro-batches**, numbered 0 to 28,975. Reading all of them once = **one epoch**.

### 14.3 One step = 5 micro-batches

The model changes its weights once per **step**. In one step the trainer runs 5 micro-batches (20 rows) through the
model, adds up their gradients, and then updates the weights once:

```
step 0 = micro-batches  0,  1,  2,  3,  4
step 1 = micro-batches  5,  6,  7,  8,  9
step 2 = micro-batches 10, 11, 12, 13, 14
step 3 = micro-batches 15, 16, 17, 18, 19
...
```

### 14.4 The end of epoch 1 does not fit evenly

- 28,976 micro-batches ÷ 5 per step = **5,795.2 steps**, not a whole number.
- Steps 0 to 5,794 (5,795 steps) use micro-batches 0 to 28,974.
- Only **one** micro-batch of epoch 1 is left: number 28,975 (the last 4 rows of the data).
- Step 5,795 needs 5 micro-batches. It takes that 1 left-over micro-batch, and then the trainer starts again at the
  beginning of the data for the other 4:

```
step 5,794 = micro-batches 28,970  28,971  28,972  28,973  28,974     (all epoch 1)
step 5,795 = micro-batch   28,975  + micro-batches 0, 1, 2, 3 again   (epoch 2 starts inside this step)
step 5,796 = micro-batches  4,  5,  6,  7,  8                         (epoch 2)
step 5,797 = micro-batches  9, 10, 11, 12, 13                         (epoch 2)
step 5,798 = micro-batches 14, 15, 16, 17, 18                         (epoch 2)
```

### 14.5 Epoch 2 next to epoch 1

```
epoch 1:  step 1     = micro-batches  5,  6,  7,  8,  9
epoch 2:  step 5,796 = micro-batches  4,  5,  6,  7,  8
                                          ===========
                                          5, 6, 7, 8 are in both = 4 of the 5 micro-batches

epoch 1:  step 2     = micro-batches 10, 11, 12, 13, 14
epoch 2:  step 5,797 = micro-batches  9, 10, 11, 12, 13
                                          ===============
                                          10, 11, 12, 13 are in both = 4 of 5 again
```

It is the same for every step: **step 5,796 matches step 1, step 5,797 matches step 2, step 6,795 matches step 1,000**,
always **5,795 steps apart**. Each matching pair shares **4 of its 5 micro-batches = 16 of its 20 rows**. That is all
"4 of 5" means; "step k" in the earlier text was just "any step number", like step 1, step 2 or step 1,000.

**Why it matters:** the train loss printed at step 5,797 is mostly measured on the same rows as the train loss at
step 2. So the same rows can be compared on their **first visit** (epoch 1) and their **second visit** (epoch 2).

---

## 15. Doubt: "In epoch 2 the val loss went down by x and the train loss by y. So x − y is memorization and y is real learning?"

**Context.** 2026-10-23, after section 14. I summarized the memorization measure in my own words to check that I had it:
"val loss decreased by x and train loss decreased by y in epoch 2, so x − y nats is memorization and y is real learning,
or am I wrong?"

Close, but the two were swapped. **Real learning = x**, the val drop. **Memorization = y − x**, the same-rows drop
minus the val drop. And y is not "the train loss in epoch 2"; it is the drop of the **same rows** between their first
and second visit.

### 15.1 Step by step, with real numbers (epoch-1 steps 1,000–1,199 against epoch-2 steps 6,795–6,994)

1. **First visit.** At steps 1,000–1,199 (epoch 1) the model sees some rows for the first time. The train loss of a step
   is computed **before** that step's update, so it is the loss of rows the model **never trained on**. Average:
   **0.9143**.
2. **Second visit.** At steps 6,795–6,994 (epoch 2) the model sees mostly the **same rows** again (16 of 20 per step,
   section 14). Their train loss is now **0.7705**. The same rows dropped **y = 0.9143 − 0.7705 = 0.1438**.
3. That drop has **two reasons**:
   - **(a) General learning.** Between step 1,100 and step 6,895 the model got better at **all** chats. Val measures
     this, because val chats are never trained on: val was 0.9389 at step 1,100 and 0.8593 at step 6,895, so
     **x = 0.0797**.
   - **(b) Having trained on these exact rows** once before. Only these rows get this part.
4. So **y = x + memorization**, which gives **memorization = y − x = 0.1438 − 0.0797 = 0.0642**.

| | First visit | Second visit | Drop |
| --- | --- | --- | --- |
| **Same rows** (train loss) | 0.9143 | 0.7705 | **y = 0.1438** |
| **Never-seen chats** (val loss, at steps 1,100 and 6,895) | 0.9389 | 0.8593 | **x = 0.0797** |
| **Extra on the seen rows = memorization** | | | **y − x = 0.0642** |

- **Real learning** (it works on new chats too) = **x = 0.0797**.
- **Memorization** (it works only on rows already trained on) = **y − x = 0.0642**.
- If the model had learned only general things, the same rows would have dropped about as much as val (~0.08). They
  dropped 0.064 more.

### 15.2 Why averages of 200 steps

The train loss of one step jumps between about 0.6 and 1.0, because each step has different chats (section 19). The
average of 200 steps (4,000 rows) removes that noise. Val is measured every 200 steps, so its value at the middle of
each window is read off the straight line between the two nearest val points.

### 15.3 All windows

Done for all 28 windows of 200 steps in epoch 2 (`memorization_windows.csv`, chart `memorization_check.png`):
on average, the same rows dropped **0.120** while val dropped **0.056** over the same times: memorization **+0.065**.
In the second half of epoch 2: 0.088 and 0.022, memorization **+0.067**. Section 17 has every window.

![Memorization check](../post_training_research/tulu3_sft_1022/memorization_check.png)

In the chart, the blue line is y (the same rows), the orange line is x (val), and the green band between them is y − x,
the memorization.

---

## 16. Doubt: "The gap at the end of epoch 1 against the end of epoch 2: what does it mean if it grows, stays the same, or shrinks?"

**Context.** 2026-10-23. The analysis printed the train–val gap at the end of each epoch: +0.018 at the end of epoch 1,
+0.076 at the end of epoch 2. I asked: since the gap grew, epoch 2 mostly memorized and did not learn in general, right?
If the gap had shrunk below 0.018, would that mean more real learning than memorization, and if it had stayed the same,
steady learning? And when the answer said a shrinking gap means the model improved less on its own training rows, I
asked why that would be bad: doesn't a smaller gap sound good?

### 16.1 The gap is the same calculation as section 15

- Gap = val loss − train loss, at the end of each epoch (train = the average of the last 500 steps of the epoch).
- **End of epoch 1:** train 0.8431 (rows seen for the **first** time, so they behave like val chats), val 0.8608,
  gap **+0.018**.
- **End of epoch 2:** train 0.7737 (the **same** rows, seen for the **second** time, so they are easier), val 0.8501,
  gap **+0.076**.
- How much the gap grew = (how much the train loss of those rows dropped) − (how much val dropped) = **y − x** =
  memorization. The two ends of the epochs are exactly a pair of matching windows (section 14): epoch-1 steps
  5,295–5,794 and epoch-2 steps 11,090–11,589:

| | End of epoch 1 | End of epoch 2 | Drop |
| --- | --- | --- | --- |
| Same rows (train) | 0.8431 | 0.7737 | y = 0.0695 |
| Val (at the middle of each window) | 0.8626 | 0.8505 | x = 0.0122 |
| Memorization | | | **y − x = 0.0573** |

(Read with val at the exact end of each epoch instead of the window middle, the gap grows by 0.058. Same calculation,
same answer.)

### 16.2 Real learning does not change the gap

Real learning helps **every** chat, seen or not, so it moves val and the train rows down **by the same amount**. The gap
only changes when the seen rows and the new chats change **differently**. Three cases, with made-up numbers:

| Case | Val drop (x) | Same rows drop (y) | y − x | What it means |
| --- | --- | --- | --- | --- |
| Gap stays the same | 0.05 | 0.05 | 0 | **all of it is real learning**, no memorization |
| Gap grows (**our run**) | 0.05 | 0.11 | +0.06 | real learning 0.05 **plus** memorization 0.06 |
| Gap shrinks | 0.05 | 0.02 | −0.03 | the rows it **trained on** improved **less** than chats it **never saw** |

### 16.3 Why a shrinking gap is not good

- Training on a row should help that row **at least** as much as a chat the model never saw.
- If the rows it trained on improve **less** than new chats, something works against those rows. Examples: the model
  forgot what it learned from them (later training overwrote it), or those rows are unlike the val chats (for example
  noisier data).
- **More real learning never shows up as a shrinking gap.** It shows up as a **bigger val drop (x)**.
- So a shrinking gap is a **warning sign to look into**, not a good sign. (An earlier answer said "it got worse on its
  own training rows"; that was too strong. The rows can still improve, just by less than new chats.)

---

## 17. Doubt: "Is the memorization about 0.06 in every 200-step window? How big is 0.06?"

**Context.** 2026-10-23. The measure had been shown for one example window (memorization 0.064) and as an average
(0.065). I asked whether it is about 0.06 for every 200-step window, whether we can then say "overall 0.06 nats is
memorization and the rest is learning", how to judge whether 0.06 is big or small (only by comparing it with the real
learning x of the same time?), and asked for all the numbers.

### 17.1 Every window

| Epoch-1 steps → epoch-2 steps | Same rows: 1st → 2nd visit | y | Val at the two moments | x (real learning) | y − x (memorization) |
| --- | --- | --- | --- | --- | --- |
| 0–199 → 5,795–5,994 | 1.1114 → 0.8321 | 0.2794 | 1.1123 → 0.8603 | 0.2520 | **+0.0274** |
| 200–399 → 5,995–6,194 | 0.9990 → 0.8004 | 0.1986 | 1.0209 → 0.8602 | 0.1606 | **+0.0380** |
| 400–599 → 6,195–6,394 | 0.9641 → 0.7876 | 0.1765 | 0.9810 → 0.8603 | 0.1207 | +0.0558 |
| 600–799 → 6,395–6,594 | 0.9480 → 0.7865 | 0.1615 | 0.9618 → 0.8600 | 0.1018 | +0.0596 |
| 800–999 → 6,595–6,794 | 0.9292 → 0.7780 | 0.1512 | 0.9488 → 0.8597 | 0.0891 | +0.0621 |
| 1,000–1,199 → 6,795–6,994 | 0.9143 → 0.7705 | 0.1438 | 0.9389 → 0.8593 | 0.0797 | +0.0642 |
| 1,200–1,399 → 6,995–7,194 | 0.9203 → 0.7813 | 0.1390 | 0.9308 → 0.8587 | 0.0720 | +0.0669 |
| 1,400–1,599 → 7,195–7,394 | 0.9167 → 0.7830 | 0.1337 | 0.9236 → 0.8581 | 0.0655 | +0.0682 |
| 1,600–1,799 → 7,395–7,594 | 0.9055 → 0.7757 | 0.1299 | 0.9176 → 0.8577 | 0.0599 | +0.0700 |
| 1,800–1,999 → 7,595–7,794 | 0.8935 → 0.7681 | 0.1254 | 0.9124 → 0.8574 | 0.0550 | +0.0704 |
| 2,000–2,199 → 7,795–7,994 | 0.8882 → 0.7671 | 0.1210 | 0.9072 → 0.8566 | 0.0505 | +0.0705 |
| 2,200–2,399 → 7,995–8,194 | 0.8830 → 0.7650 | 0.1180 | 0.9029 → 0.8563 | 0.0466 | +0.0714 |
| 2,400–2,599 → 8,195–8,394 | 0.8787 → 0.7629 | 0.1157 | 0.8992 → 0.8562 | 0.0431 | **+0.0727** (highest) |
| 2,600–2,799 → 8,395–8,594 | 0.8859 → 0.7734 | 0.1125 | 0.8957 → 0.8557 | 0.0400 | +0.0725 |
| 2,800–2,999 → 8,595–8,794 | 0.8728 → 0.7645 | 0.1083 | 0.8925 → 0.8552 | 0.0373 | +0.0710 |
| 3,000–3,199 → 8,795–8,994 | 0.8729 → 0.7661 | 0.1068 | 0.8892 → 0.8549 | 0.0343 | +0.0725 |
| 3,200–3,399 → 8,995–9,194 | 0.8683 → 0.7664 | 0.1018 | 0.8858 → 0.8545 | 0.0313 | +0.0705 |
| 3,400–3,599 → 9,195–9,394 | 0.8578 → 0.7581 | 0.0997 | 0.8828 → 0.8540 | 0.0288 | +0.0709 |
| 3,600–3,799 → 9,395–9,594 | 0.8626 → 0.7653 | 0.0973 | 0.8801 → 0.8537 | 0.0264 | +0.0709 |
| 3,800–3,999 → 9,595–9,794 | 0.8622 → 0.7681 | 0.0942 | 0.8779 → 0.8533 | 0.0246 | +0.0696 |
| 4,000–4,199 → 9,795–9,994 | 0.8533 → 0.7617 | 0.0916 | 0.8758 → 0.8529 | 0.0229 | +0.0687 |
| 4,200–4,399 → 9,995–10,194 | 0.8583 → 0.7696 | 0.0887 | 0.8737 → 0.8526 | 0.0211 | +0.0676 |
| 4,400–4,599 → 10,195–10,394 | 0.8516 → 0.7667 | 0.0849 | 0.8716 → 0.8522 | 0.0194 | +0.0655 |
| 4,600–4,799 → 10,395–10,594 | 0.8597 → 0.7770 | 0.0827 | 0.8695 → 0.8518 | 0.0177 | +0.0650 |
| 4,800–4,999 → 10,595–10,794 | 0.8566 → 0.7767 | 0.0799 | 0.8677 → 0.8515 | 0.0162 | +0.0637 |
| 5,000–5,199 → 10,795–10,994 | 0.8570 → 0.7799 | 0.0771 | 0.8660 → 0.8511 | 0.0149 | +0.0623 |
| 5,200–5,399 → 10,995–11,194 | 0.8437 → 0.7706 | 0.0731 | 0.8644 → 0.8508 | 0.0136 | +0.0595 |
| 5,400–5,599 → 11,195–11,394 | 0.8510 → 0.7804 | 0.0705 | 0.8630 → 0.8505 | 0.0124 | +0.0581 |
| **5,295–5,794 → 11,090–11,589** (the ends of the epochs, 500 steps) | 0.8431 → 0.7737 | 0.0695 | 0.8626 → 0.8505 | 0.0122 | **+0.0573** |

### 17.2 What the table says

- **Memorization is steady, about 0.06 to 0.07 nats,** for every window from epoch-1 step 400 on.
- The first two windows are lower (0.027 and 0.038). Their rows were trained during the warm-up (steps 0–347), when the
  learning rate was still small, which is the likely reason they left a smaller mark. This reason is a reading; it was
  not tested separately.
- **x is very different from window to window, and it should be.** x is the real learning **between the two visits**.
  For the first window, that time includes the big learning at the start of epoch 1 (val 1.112 → 0.860), so x = 0.252.
  For the last window, the time between the visits is almost exactly epoch 2, so **x = 0.012 is the real learning that
  epoch 2 gave**.

### 17.3 How big is 0.06? Only next to x

| Windows | Real learning x | Memorization y − x | Which is bigger |
| --- | --- | --- | --- |
| first (0–199 → 5,795–5,994) | 0.252 | 0.027 | real learning, 9 times |
| 1,000–1,199 → 6,795–6,994 | 0.080 | 0.064 | real learning, a little |
| 1,600–1,799 → 7,395–7,594 | 0.060 | 0.070 | memorization, from here on |
| the ends of the epochs | **0.012** | **0.057** | **memorization, 4.7 times** |

So for the time of epoch 2 itself, the model gained about **4.7 times more on its own training rows than on new chats**.
Epoch 2 was mostly memorization, with a small real gain and no harm to new chats (stage 2 of section 12.2).

---

## 18. Doubt: "What are 'train' and 'val' in the step lines? Are both measured every step?"

**Context.** 2026-10-23, while reading the step lines of the log and the charts. Every step line shows both a `train`
and a `val` number, so I asked whether "train" is the training loss and "val" the validation loss, and whether both are
measured at every step.

- **train** (train loss): measured **every step**, on **that step's 20 training rows** (about 50,000–60,000 answer
  tokens), before that step's update.
- **val** (val loss): measured **every 200 steps** and at the last step, on the **fixed set of 2,000 held-out chats**
  (898,276 answer tokens): the same chats every time, never trained on (sections 1.4 and 3).
- The step lines in between **repeat the last val number**. That is why, for example, `val 0.850392` appears on every
  line from step 11,400 until the next val at step 11,589.

---

## 19. Doubt: "Why does the train loss jump between about 0.6 and 1.0 from one step to the next? The model changes very little between two steps."

**Context.** 2026-10-23. To explain why the memorization measure uses averages of 200 steps, the answer said "a single
step's loss jumps between about 0.6 and 1.0, since each step has different chats". I did not accept that without proof:
the model changes very little from one step to the next, so the loss of neighbouring steps should differ very little.
How can it jump that much? I asked for proofs from the steps themselves, not a story.

Right: the **model** changes very little between two steps. What changes a lot is the **data**. Between two steps:

- **the model** gets one small weight update (learning rate between 5e-6 and 5e-5);
- **the data** is 20 completely different rows, about 130 different chats.

### 19.1 Proof 1: the model barely changes

Val is the same 2,000 chats every time. It moves only about **0.0003 per 200 steps**: 0.850836 (step 11,000) →
0.850685 (11,200) → 0.850392 (11,400) → 0.850100 (11,589). The same model on the same data gives almost the same number.

### 19.2 Proof 2: the train loss still jumps, because the data changes

```
step 11,543: 0.812   (53,436 trained tokens)
step 11,544: 0.781   (59,379)
step 11,545: 0.786   (57,002)
step 11,546: 1.022   (52,193)   +0.237
step 11,547: 0.753   (51,873)   −0.269
step 11,548: 0.745   (49,585)
```

In epoch 2 the median change from one step to the next is **0.079**; 90% of the changes are below 0.194; the largest is
0.525. If the model itself changed that much, val would jump just as much, and it does not.

### 19.3 Proof 3, the decisive one: the same rows give the same jumps, 5,795 steps later

From section 14, epoch-2 step 5,795 + n trains on 16 of the same 20 rows as epoch-1 step n:

```
epoch-1 steps 2,873 .. 2,877:   0.843  0.848  1.196  0.865  0.677
epoch-2 steps 8,668 .. 8,672:   0.669  0.724  1.169  0.729  0.549
                                               =====
                                 the spike sits at the matching step, 5,795 steps later

epoch-1 steps 5,749 .. 5,753:   0.928  0.770  1.135  0.763  0.834
epoch-2 steps 11,544 .. 11,548: 0.781  0.786  1.022  0.753  0.745
```

Over the whole of epoch 1, with the slow downward trend removed (each step minus the average of the 201 steps around
it), I measured the **correlation** of the ups and downs. A correlation of 1.0 means the two lists go up and down
together exactly; 0 means no relation.

| Pairing | Rows shared | Correlation |
| --- | --- | --- |
| step n with step 5,795 + n | 16 of 20 | **0.804** |
| shifted by 1 step (5,795 + n + 1) | 4 of 20 (1 micro-batch) | 0.169 |
| shifted by 2, 7 or 50 steps | none | −0.022, −0.023, −0.005 |

The more rows two steps share, the more their ups and downs match; with no shared rows there is no match at all. So the
jumps come from **which rows are in the step**, not from the model.

### 19.4 Why some rows have a much higher loss than others

The loss of an answer token says how hard it was to predict. Answers that copy or reformat text from the question, or
follow fixed patterns, are easy to predict (low loss). Open-ended answers (stories, long explanations, opinions) have
many possible next words (high loss). A step that happens to get more of the hard kind has a higher average.

What is proven is that **it is the rows**. Which kinds of chats sit in a spike step like 2,875 or 8,670 was not checked;
it can be checked by decoding those rows.

---

## 20. Summary of sections 11–19

| Question | Answer for this run |
| --- | --- |
| Is the joined history of the three log parts exact? | Yes: same val and train at steps 8,000 and 9,000 in both parts |
| Did val ever go up for real? | No. 3 tiny rises out of 58 points (largest 0.0007); the lowest val is the last one (0.8501) |
| Overfitting? | **No** (val never rose) |
| Memorization? | **Yes, about 0.06–0.07 nats**, steady through epoch 2 (stage 2 of section 12.2) |
| Real learning in epoch 2? | Small: val down 0.0107 (3.4% of epoch 1's 0.315) |
| Memorization vs real learning at the end | 0.057 vs 0.012: memorization about 4.7 times bigger |
| What a nat is | the unit of −ln(probability of the correct token); a drop of d nats multiplies that probability by e^d |
| Why the train loss jumps between steps | each step has different rows; proven by the 0.804 correlation of matching steps |
| Which checkpoint to use (6,000 or 11,590)? | not decided by the loss: the dev suite decides (section 10, step 4) |

---

## 21. Sources

- Validation code: `bluscript_sft.cpp:2304-2329` (`NoGradGuard`, forward only) in `~/Downloads/BluTrain_sft`.
- Labels and trained-token counts: `sft/prepare_sft_data.py` (65535 = no loss; `valid_counts.bin`).
- The 10-07 Muon learning-rate sweep: `sft_work_documentation.md` §37.5.
- The dev suite and unseen suite plan (2026-10-08): `sft_work_documentation.md` §38.6–38.7.
- Data numbers: the Tülu 3 data preparation of 2026-10-22 (train 741,413 chats → 115,904 rows, 325.7M answer tokens
  of 474.7M positions per epoch; val 2,000 chats → 320 rows, 898,276 answer tokens).
- Sections 11–20 (2026-10-23): the run's three logs (laptop `~/Downloads/box_backup_2026_10_22/tulu3_run_final/`),
  and everything made from them in `post_training_research/tulu3_sft_1022/` (`analyze_run.py`, its printout
  `analysis_output.txt`, `tulu3_sft_1022_full_run.log`, `trajectory.csv`, `memorization_windows.csv`, the three charts).
- Same row order in every epoch, no shuffle: `sft/SftDataLoader.h:25-26`. Val measured at the start of a step, before
  its update: `bluscript_sft.cpp:2304`; checkpoints named after the number of updates: `bluscript_sft.cpp:2857`.
- The loss is in nats (natural log): `LossKernels.cu:410`, `loss = logf(final_sum) + final_max - target_logit`.
