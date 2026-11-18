# Packing chats into rows: First-Fit, Best-Fit, chats longer than a row, TRL's three strategies, varlen attention, and how packing changes the number of training steps

Written 2026-10-14. Every number in this document is computed by a script, not by hand. The scripts and their outputs
are in `profiling_reports/q10_packing_sim_2026_10_08/`:

- `packing_examples.py` -> `packing_examples_output.txt` (examples 1-13 and 17, and section 7.3);
- `ffd_vs_bfd.py` -> `ffd_vs_bfd_output.txt` (First-Fit vs Best-Fit on our data);
- `group_packing.py` -> `group_packing_output.txt` and `packing_stats_detail.py` -> `packing_stats_detail_output.txt`
  (examples 15 and 16, and section 9);
- `ffd_vs_bfd_study.py` -> `ffd_vs_bfd_study_output.txt` (section 13: First-Fit vs Best-Fit on 1,673 tests, and speed);
- `ffd_vs_bfd_full_study.py` -> `ffd_vs_bfd_full_study_output.txt` (section 13: the larger study, about 25,000 random tests,
  every small input, and careful timing);
- `subset_sum_packing_1511.py` and `check_packing_1511.py` -> `subset_sum_packing_1511_output.txt` (example 15, step 6);
- `trl_units_groups_output.txt` (example 16, in TRL's own units).

A few small numbers are plain arithmetic, shown where they appear (for example 1,510 x 4,096 = 6,184,960).

**How to read this document.** Section 0 explains every word. Section 1 says what problem "packing" solves. Section 2
explains attention from zero, because packing changes what attention sees, and 2.8 says where document masking is used. Sections 3-6 are the two packing rules with
17 worked examples. Section 7 is about chats longer than a row. Section 8 is TRL's three packing strategies. Section 9 is
how packing changes the number of training steps (a future task). Section 10 is how the attention kernel keeps packed
chats apart. Section 12 lists every correction. Section 13 compares First-Fit and Best-Fit and says which one to use.

---

## 0. Words used in this document

| word | meaning |
| --- | --- |
| token | one piece of text the model reads (a word or a part of a word) |
| chat | one conversation: system text, user turns, assistant answers, and an end token at the end |
| document | the general word for one independent piece of text. In SFT, one document = one chat |
| chunk (piece) | a part of a long chat, made by cutting the chat (TRL's `bfd_split`, section 8.3) |
| row | one training sequence the GPU processes. In our trainer every row has the same number of slots |
| T (`max_seq_length`, `max_length`) | the number of slots in a row. Ours is 4096. The small examples use T = 8, 10 or 20 so every row can be drawn |
| slot / position | one place in a row. It holds one token. Positions are numbered from 0 |
| pad | a slot with no real token in it. It costs GPU work and teaches nothing |
| packing | putting several chats into one row so that fewer slots are pads |
| boundary | the place where one document ends and the next one begins (section 8.4) |
| micro-batch | the rows the GPU processes in one forward and backward pass. Ours: 8 rows (B = 8) |
| gradient | for every weight of the model, how it should change to lower the loss |
| gradient accumulation | adding up the gradients of several micro-batches before changing the weights. Ours: 8 micro-batches (GRAD_ACCUM = 8) |
| optimizer step (step) | one change of the model's weights, made from the added-up gradients. Ours: 8 micro-batches x 8 rows = 64 rows per step |
| epoch | one pass over all the training data |
| learning rate | how big each change of the weights is. Ours rises to 2e-5 (0.00002) during warmup, then falls to 2e-6 |
| warmup | the first steps of training, where the learning rate grows from small to full. Ours: the first 3% of the steps |
| forward pass | one run of the model over a micro-batch, from the input tokens to the loss |
| attention | the part of each layer where a token takes information from other tokens (section 2) |
| causal attention | a token takes information only from itself and the tokens before it (section 2.2) |
| plain causal attention | the causal rule applied to the whole row, without knowing where chats start (section 2.3) |
| cross-document attention | a token takes information from a token of a different chat in the same row (section 2.5) |
| document masking (intra-document causal masking) | causal attention inside each chat only (section 2.7) |
| kernel | a program that runs on the GPU. The attention kernel is the program that computes attention |
| tile | a block of 64 or 128 positions that one GPU work item handles at once inside a kernel |
| varlen attention | the attention-kernel form of document masking: the kernel gets the start of every chat (section 10) |
| `cu_seqlens` | the list of positions where each chat starts, plus the end of the last one. It is what varlen attention reads |
| `seq_lengths` | the list TRL keeps of the length of every chat (or chunk) inside a packed row |
| positions (RoPE) | each token is given its position number; RoPE is the method our model uses to put the position into the token. In a packed row the positions restart at 0 for every chat |
| padding-free | TRL's mode where the chats of a batch are joined into one long list with no pads and no fixed row length |
| collator | the code that turns rows into one training batch (it adds the pads, the positions and the labels) |
| map batch (a "group" in this document) | in TRL's data preparation, how many chats the packing code sees at once (1,000 by default). It is NOT a training batch (example 16) |
| ablation | a test where only one setting changes and everything else stays the same, to see what that one setting does |

**How many slots a chat needs.** A chat of N tokens needs N - 1 slots. Example: a chat of 3 tokens "Hi", "there",
`<end>`. The model reads "Hi" and must predict "there"; it reads "there" and must predict `<end>`. So it reads 2 tokens
(2 slots) and predicts 2 tokens. The last token (`<end>`) is only predicted, never read. In all examples below, "chat
length" already means slots.

**What "decreasing" means in Best-Fit-Decreasing(BFD) and First-Fit-Decreasing(FFD) algorithms.** Both packing rules first sort the chats from longest to shortest, then place them one by
one.

---

## 1. The setup: what problem are we solving

You have a training dataset of chats. Each chat is a different length: some are 300 tokens, some are 2,000, some are
5,500. The model's `max_seq_length` is 4096.

In our trainer every training row is exactly 4096 slots long, because our GPU kernels work on tensors of one fixed shape,
[rows, 4096]. (Not every trainer works like this. TRL's padding-free mode joins all the chats of a batch into one long
list with no fixed row length, and varlen attention kernels accept chats of any length; see section 10.)

The question: **how do you arrange these chats of different lengths into rows of one fixed length?**

**What a row looks like without packing (v1, what we train with today).** One chat per row; the rest of the row is pads.
A small row of 10 slots with one chat of 4 tokens:

```
slot:   0  1  2  3  4    5    6    7    8    9
token:  A0 A1 A2 A3 PAD  PAD  PAD  PAD  PAD  PAD      4 real tokens, 6 pads: 40% of the row is useful
```

**Our stage-0 data in numbers.** 5,000 chats from UltraChat. Each chat fills between 134 and 4,069 slots; the average is
1,238 and the middle value (median) is 1,154. An average chat in a 4,096-slot row leaves 4,096 - 1,238 = 2,858 pads.

| way | what it does | on our stage-0 data |
| --- | --- | --- |
| no packing (v1, today) | one chat per row, the rest of the row is pads | 5,000 rows; only 30.2% of the slots are real tokens; 69.8% of the slots are pads, and they take 72.3% of the GPU work (example 15) |
| packing (First-Fit or Best-Fit, sections 3-6) | several whole chats per row | 1,517 rows; 99.6% of the slots are real tokens |
| `wrapped` (section 8.1) | join all chats into one stream and cut it every 4096 tokens | no pads, but chats are cut at row ends and read each other |

The questions this document answers:

- How to choose which chats share a row (sections 3-6, with 17 examples).
- What to do with a chat longer than the row (section 7).
- How TRL does it, with its three strategies (section 8).
- How to stop the chats in one row from reading each other (sections 2, 8.4 and 10).
- What packing does to the number of training steps (section 9).

---

## 2. Attention, causal attention, and cross-document attention (read this first)

### 2.1 What attention does

In every layer of the model, every token builds a new version of itself by taking information from other tokens in the
same row. "Token i attends to token j" means: token i takes information from token j.

How much it takes from each one:

1. For token i and every token j it is allowed to look at, the model computes a score: how useful j is for i.
2. The scores of token i are turned into weights that are all positive and add up to 1 (this step is called softmax).
3. Token i's new version = the weighted sum of what the allowed tokens j offer.

A small example with numbers. Token i may look at three tokens, with scores 2, 1 and 0:

```
step 1, scores:            2        1        0
step 2, e^score:           7.39     2.72     1.00      (e = 2.718...)   sum = 11.11
        weight = e^score / sum:
                           0.665    0.245    0.090     (they add up to 1)
step 3, new version of i = 0.665 x (what token 1 offers) + 0.245 x (what token 2 offers) + 0.090 x (what token 3 offers)
```

A **mask** decides which pairs (i, j) are allowed. A pair that is not allowed gets weight 0: token i takes nothing from
token j. In the example, if the mask blocks the third token, only the first two share the weight (0.731 and 0.269).

### 2.2 What "causal" means

The model is trained to predict the next token. When the model is used, it writes text one token at a time, so when it
writes token i, the tokens after i do not exist yet. In training it must therefore never use them either.

So the rule is: **token i may attend to tokens 0, 1, ..., i (itself and everything before it), and never to tokens
i+1, i+2, ...** This rule is called causal attention. Drawn as a table of allowed pairs, it is a triangle.

### 2.3 Plain causal attention

"Plain causal attention" means the causal rule applied to the **whole row**, with only one condition: j <= i. It knows
nothing about chats. It treats the whole row as one text.

### 2.4 One chat per row: plain causal attention is correct

Row of 10 slots holding one chat A of 6 tokens and 4 pads after it:

```
position:  0  1  2  3  4  5  6   7   8   9
token:     A0 A1 A2 A3 A4 A5 PAD PAD PAD PAD
```

- A3 attends to A0, A1, A2, A3. All of them are in chat A. Correct.
- A5 attends to A0 ... A5. All in chat A. Correct.
- The pads come after the chat, so no chat token ever attends to a pad. The pads' own outputs are never trained.

**There is no cross-document attention here, because the row holds only one document.** This is what we do today
(v1: one chat per row).

### 2.5 Several chats per row: plain causal attention CAUSES cross-document attention

Row of 10 slots packed with three chats: A (4 tokens), B (3 tokens), C (3 tokens):

```
position:  0  1  2  3  4  5  6  7  8  9
token:     A0 A1 A2 A3 B0 B1 B2 C0 C1 C2
```

With plain causal attention (only the rule j <= i):

- B0 (position 4) attends to positions 0-4: A0, A1, A2, A3 and B0. **B0 takes information from chat A.**
- B2 (position 6) attends to all of A and B0, B1, B2.
- C0 (position 7) attends to all of A, all of B, and C0. **C0 takes information from chats A and B.**
- C2 (position 9) attends to all 10 tokens.

**Cross-document attention is exactly these pairs: a token of one chat attending to a token of a different chat**
(B to A, C to A, C to B).

Counting the pairs in this row:

- plain causal allows 55 pairs (1 + 2 + ... + 10);
- pairs inside one chat: A 10 (1+2+3+4), B 6, C 6, which is 22 in total;
- cross-document pairs: 55 - 22 = **33**.

So:

| row holds | attention rule | cross-document attention? | correct for SFT? |
| --- | --- | --- | --- |
| one chat (pads after it) | plain causal | no | yes |
| several chats | plain causal | **yes** | no |
| several chats | causal inside each chat (document masking / varlen) | no | yes |

**Plain causal attention is the rule. Cross-document attention is what that rule produces when a row holds more than one
chat.** They are not the same thing: plain causal attention on a one-chat row has no cross-document attention at all.

### 2.6 Why cross-document attention is wrong for SFT

1. **The model is trained in a situation that never happens when it is used.** When a person uses the model, their
   conversation is alone: the model never sees another person's conversation before theirs. If, in training, chat B's
   tokens can read chat A, the model learns to use text that will not be there later.
   - Example: chat A is "Translate 'cat' into French." -> "chat". Chat B is "What is the capital of France?" -> "Paris".
     With plain causal attention, while learning to write "Paris", the model can also read chat A's text about French.
     Part of what it learns for "Paris" then comes from text that will not exist when a user asks about France.
2. **A wrong target at every boundary** (when the rows are built as one stream, as in `wrapped`): the last token of
   chat A (its end token) is trained to predict the first token of chat B. A's end and B's start have nothing to do with
   each other, so this target teaches nothing true.
   - In our design this target never exists. Each chat's inputs and labels are built before packing (labels = the
     inputs moved by one token), so the last target of a chat is its own end token.
3. **How much it hurts, from sources we can check:** Hugging Face (the blog a colleague shared) saw little change on
   short-context tasks and found document masking crucial when going from 4k to 64k context. TRL's docs cite the
   Qwen3-Coder-Next technical report: breaking sequence continuity hurts performance.

### 2.7 How cross-document attention is stopped: causal attention inside each chat

The rule becomes: **token i may attend to token j only if j <= i AND j is in the same chat as i.** This is called
intra-document causal masking, or document masking. It is the right-hand picture in the blog.

Same row as 2.5 (A = positions 0-3, B = 4-6, C = 7-9); `x` = may attend:

```
plain causal (one big triangle, 55 pairs)     document masking (one triangle per chat, 22 pairs)
key:   0 1 2 3 4 5 6 7 8 9                     key:   0 1 2 3 4 5 6 7 8 9
A0     x                                       A0     x                       chat A
A1     x x                                     A1     x x
A2     x x x                                   A2     x x x
A3     x x x x                                 A3     x x x x
B0     x x x x x                               B0             x               chat B
B1     x x x x x x                             B1             x x
B2     x x x x x x x                           B2             x x x
C0     x x x x x x x x                         C0                   x         chat C
C1     x x x x x x x x x                       C1                   x x
C2     x x x x x x x x x x                     C2                   x x x
```

- B0 (position 4) now attends only to itself. Position 4 cannot see position 3.
- The 33 cross-document pairs are gone, so the attention also does less work. On stage-0 the whole packed run is 27.8%
  of today's work with document masking and 30.3% without it (`pack_sim.py`).
- Positions (RoPE) also restart at 0 for every chat: A0-A3 get positions 0-3, B0-B2 get 0-2, C0-C2 get 0-2. So each
  chat looks to the model as if it starts at the beginning of a row.

How this is built in a kernel (varlen attention, `cu_seqlens`) is section 10. Our two designs (A and B) both do it.

### 2.8 Where document masking is used: SFT, pretraining, long-context training, DPO and RL (with sources)

#### 2.8.1 The rule

Document masking is needed **whenever more than one document sits in the same row**. If every row holds exactly one
document (and pads after it), there is nothing else in the row to read, and plain causal attention is already correct
(2.4).

| what is in one row | example | plain causal attention gives | document masking needed? |
| --- | --- | --- | --- |
| one document + pads after it | `A0 A1 A2 A3 PAD PAD` | A's tokens read only A | no |
| several whole documents | `A0 A1 A2 A3 B0 B1 B2 C0 C1 C2` | B reads A; C reads A and B | yes, if the documents are unrelated |
| parts of several documents, cut at the row ends (`wrapped`) | `A7 A8 B0 B1 B2 B3 C0 C1 C2 C3` | B reads the end of A; C reads A and B | yes, if the documents are unrelated |

Whether a training stage uses it therefore depends on how that stage builds its rows. The sections below go stage by
stage.

#### 2.8.2 SFT

- **Without packing (our v1 today):** one chat per row, pads after it. No document masking is needed or used. Our
  current SFT runs train this way.
- **With packing:** document masking is the standard. Every packed-SFT tool we checked keeps the chats of a row apart:

| tool | how it keeps packed chats apart | where (checked in the source) |
| --- | --- | --- |
| TRL (`bfd`, `bfd_split`) | positions restart at 0 for every chat; FlashAttention turns that into `cu_seqlens` (varlen); SDPA / eager get a same-chat mask from `transformers` for models that pass the positions | `trl/trainer/sft_trainer.py:1180`; `transformers/modeling_flash_attention_utils.py:480-515`; `transformers/masking_utils.py:734-763, 976-979` (section 8.4) |
| TRL (`wrapped`) | **none**: no chat starts are recorded, so the chats read each other | `trl/data_utils.py:829-843`; `sft_trainer.py:1180, 1188-1192` (section 8.1) |
| torchtitan | FlexAttention with a "block_causal" mask: causal AND same document | `torchtitan/models/common/attention.py:467` (`get_document_mask_mod`) |
| Megatron-LM | the start of every sequence in the pack (`cu_seqlens`, passed as `PackedSeqParams`) | `megatron/core/packed_seq_params.py:10` |
| ours (v2) | design A (the start of every chat in each row) or design B (a block mask); positions restart at 0 for every chat | our forward v62 and backward kernels (SFT work doc §40) |

Why SFT always needs it when packing: each chat is a separate conversation, and when the model is used, a person's
conversation is alone, with no other conversation before it (2.6). Training a chat while it can read another chat
teaches the model to use text that will not be there.

#### 2.8.3 Pretraining

In pretraining, the data is long plain text (web pages, books, code). The usual way to build rows is to join the
documents one after another, with an end token between them, and cut the stream into rows of the training length.
That is the same as TRL's `wrapped` (8.1), so a row usually holds the end of one document and the start of the next.
Real models made different choices here:

**GPT-3 (OpenAI, 2020): no document masking.** From the paper's Appendix B ("Details of Model Training"): *"During
training we always train on sequences of the full nctx = 2048 token context window, packing multiple documents into a
single sequence when documents are shorter than 2048, in order to increase computational efficiency. Sequences with
multiple documents are not masked in any special way but instead documents within a sequence are delimited with a
special end of text token, giving the language model the information necessary to infer that context separated by the
end of text token is unrelated. This allows for efficient training without need for any special sequence-specific
masking."* In plain words: the documents do read each other, and the model is expected to learn that text before an
end token belongs to a different document.

**Llama 3 (Meta, 2024): document masking used.** From the paper's model architecture section (3.2), among the small
changes from Llama 2: *"We use an attention mask that prevents self-attention between different documents within the
same sequence. We find that this change had limited impact during in standard pre-training, but find it to be
important in continued pre-training on very long sequences."*
**SmolLM3 (Hugging Face, 2025): document masking used.** From the blog a colleague shared (section "document masking",
which follows Hugging Face's SmolLM3 training playbook):

- *"Causal masking means that for unrelated files A and B in the same batch, the tokens in B can attend to the tokens
  in A, which degrades performance. With intra-document masking, the attention mask is modified so tokens can only
  attend to previous tokens within the same document."*
- *"When implementing document masking, Hugging Face saw small improvements on PIQA but otherwise no noticeable impact on
  short context tasks. But in line with aforementioned research, they observed that it became crucial for scaling from
  4k to 64k tokens."*
- On model size: *"For smaller models, some implementations choose to omit intra-document masking, finding that the
  additional complexity and potential reduction in cross-document learning doesn't justify the benefits at those
  scales. However, for larger models, intra-document masking becomes more critical as the model's capacity to learn
  from cross-document attention patterns diminishes relative to the benefits of cleaner document boundaries."*
- SmolLM3 adopted document masking because it *"provides the foundation for long-context handling"*.

**Our BluTrain pretraining: no document masking.**

- Our pretraining shards separate documents with the end token 50256 (`flux_train_000001` has 74,264 end tokens per 100
  million tokens, one before each document; SFT work doc §32.2), and rows are cut from that stream.
- Our pretraining attention kernels have no document mask: causality is done by skipping tiles plus a mask on the
  diagonal tile, with no attention, padding or document mask, no `cu_seqlens` and no position ids (SFT work doc §32,
  the table of what our kernels do).
- So our pretraining works the GPT-3 way.

**Why the effect is small at short context and large at long context (as the sources report):**

- At short context (2k-4k rows), a token mostly attends to nearby text. The end token marks the switch to a new
  document, and the model learns that text before it is usually unrelated (GPT-3's reasoning). Reading a little of the
  previous document costs little.
- At long context (the 4k-to-64k step, Llama 3's long continued pretraining), the point of training is to learn real
  long-distance links inside one long document. If a 64k row is full of unrelated documents and they can all read each
  other, the model sees many long-distance links that mean nothing. Both Llama 3 and Hugging Face found the mask
  important there.
- The blog also lists *"shortening the average context length"* among the effects: with the mask, a token can only
  read back to the start of its own document, so the stretch of text it learns from is, on average, shorter than the
  whole row.

#### 2.8.4 Mid-training and long-context extension

This is the stage where the sources found document masking most important: Llama 3's "continued pre-training on very
long sequences", and Hugging Face's step from 4k to 64k context. A model that is meant to handle long inputs well is
trained on long rows, and the mask stops unrelated documents in those long rows from reading each other.

#### 2.8.5 Preference tuning (DPO) and RL

The same rule applies: it is needed only if several examples are packed into one row.

- In TRL's DPO trainer today, the padding-free (packed) mode is switched off: *"`padding_free=True` is temporarily
  unavailable after a refactor and is currently disabled. Falling back to standard padding"*
  (`trl/trainer/dpo_trainer.py:708-713`). So TRL's DPO trains one example per row with padding, and no document mask is
  needed there.
- RL training (for example GRPO) and other trainers follow the same rule. Whether they pack depends on the tool, and we
  have not checked them.

#### 2.8.6 What document masking costs

- **With a kernel that skips blocked tiles** (FlashAttention varlen, FlexAttention, our designs A and B), it removes work:
  on stage-0, the packed run is 27.8% of today's work with the mask and 30.3% without it (2.7).
- **With a full mask table** (PyTorch SDPA given an explicit mask), the GPU still computes every tile and only throws
  the blocked results away, so there is no saving, and it is slower than a varlen kernel (SFT work doc §40.2).

#### 2.8.7 Summary

| stage | how rows are usually built | document masking | sources |
| --- | --- | --- | --- |
| SFT, no packing (our v1) | one chat per row + pads | not needed | 2.4 |
| SFT, packed | several whole chats per row | **always used** (TRL `bfd`/`bfd_split`, torchtitan, Megatron-LM, ours) | 2.8.2 |
| SFT, TRL `wrapped` | one stream cut every T tokens | not possible (no chat starts recorded) | 8.1 |
| pretraining, short context | documents joined with an end token, cut into rows | **a choice:** GPT-3 and our BluTrain do not use it; Llama 3 and SmolLM3 do; small effect at short context | 2.8.3 |
| long-context training (mid-training) | long rows with many documents | **used, and reported important** (Llama 3, SmolLM3) | 2.8.4 |
| DPO / RL | depends on the tool | needed only when examples are packed; TRL DPO does not pack today | 2.8.5 |

#### 2.8.8 Sources for this section

- GPT-3, Appendix B: [Brown et al., "Language Models are Few-Shot Learners", arXiv 2005.14165](https://arxiv.org/abs/2005.14165)
- Llama 3, section 3.2: ["The Llama 3 Herd of Models", arXiv 2407.21783](https://arxiv.org/abs/2407.21783)
- SmolLM3 / Hugging Face: [the blog, section "document masking"](https://djdumpling.github.io/2026/01/31/frontier_training.html#document-masking),
  which follows [Hugging Face's smol-training-playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- TRL: [`sft_trainer.py:1180-1192`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L1180-L1192),
  [`data_utils.py:829-843`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L829-L843),
  [`dpo_trainer.py:708-713`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/dpo_trainer.py#L708-L713)
- transformers: [`masking_utils.py:734-763`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/masking_utils.py#L734-L763),
  [`modeling_flash_attention_utils.py:480-515`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/modeling_flash_attention_utils.py#L480-L515)
- torchtitan: [`models/common/attention.py:467`](https://github.com/pytorch/torchtitan/blob/1e4d33de4b02f6a6f2ea8d58e65f4d3ce33f1df8/torchtitan/models/common/attention.py#L467)
- Megatron-LM: [`core/packed_seq_params.py:10`](https://github.com/NVIDIA/Megatron-LM/blob/8b2f660bb89e38b6210575dea7984fd05899adf0/megatron/core/packed_seq_params.py#L10)
- Ours: SFT work doc §32 (our attention kernels have no document mask) and §32.2 (end token 50256 between pretraining
  documents); §40 (our packed designs A and B).

---

## 3. The two packing rules, step by step

**First-Fit-Decreasing (FFD).** This is what our simulation and our attention test data use.

1. Sort the chats longest first.
2. Take the next chat. Look at the rows from the first row to the last. Put the chat into the **first** row that still
   has enough empty slots.
3. If no row has enough empty slots, start a new row and put the chat there.
4. Repeat until every chat is placed.

**Best-Fit-Decreasing (BFD).** This is TRL's default (`trl/trainer/sft_config.py:230-231`; the code is `trl/data_utils.py:739-826`).

1. Sort the chats longest first.
2. Take the next chat. Look at **all** rows that still have enough empty slots. Out of those, put the chat into the row
   that will have the **fewest** empty slots left after the chat goes in (the tightest fit).
3. If no row has enough empty slots, start a new row.
4. Repeat until every chat is placed.

The only difference is step 2: First-Fit takes the first row that fits; Best-Fit takes the tightest row that fits.
(In computer science, putting items of different sizes into as few fixed-size containers as possible is called "bin
packing"; here a row is the container.)

**The smallest possible number of rows** is (sum of all chat lengths) / T, rounded up. No rule can use fewer rows,
because fewer rows do not have enough slots. Sometimes even this number cannot be reached (example 8).

**How to read the examples.** Each example gives the row size T and the chat lengths, walks through every chat in order
("6 -> row 1 (2 empty)" means: the chat of length 6 goes into row 1, and row 1 then has 2 empty slots), shows the final
rows, and ends with what the example teaches.

---

## 4. Examples

### Example 1 - why we pack at all (T = 8; chats 6, 3, 5, 2)

**Without packing**, every chat gets its own row:

```
row 1: [6 + 2 pads]   row 2: [3 + 5 pads]   row 3: [5 + 3 pads]   row 4: [2 + 6 pads]
4 rows, 32 slots, 16 real tokens = 50%
```

**With packing** (both rules give the same). Sorted longest first: 6, 5, 3, 2.

```
6 -> new row 1 (2 empty)
5 -> row 1 has 2 empty, too few -> new row 2 (3 empty)
3 -> First-Fit: row 1 (2) too few, row 2 (3) fits -> row 2 (0 empty)
     Best-Fit:  only row 2 fits -> row 2
2 -> row 1 (2) fits -> row 1 (0 empty)
rows: [6, 2] [5, 3] -> 2 rows, 16 of 16 slots real = 100%
```

**What it teaches:** packing needs half the rows here, so half the GPU work, for the same 16 real tokens.

### Example 2 - rows of 4000; chats 4000, 3000, 2500, 300, 200, 100

| chat | rows that still have room | First-Fit picks | Best-Fit picks |
| --- | --- | --- | --- |
| 4000 | none | new row 1 (0 empty) | same |
| 3000 | none (row 1 is full) | new row 2 (1000 empty) | same |
| 2500 | none (row 2 has 1000) | new row 3 (1500 empty) | same |
| 300 | row 2 (1000), row 3 (1500) | row 2, the first that fits (700 empty) | row 2 leaves 700, row 3 would leave 1200 -> row 2 |
| 200 | row 2 (700), row 3 (1500) | row 2 (500 empty) | row 2 (500 vs 1300) |
| 100 | row 2 (500), row 3 (1500) | row 2 (400 empty) | row 2 (400 vs 1400) |

Result: `[4000] [3000, 300, 200, 100] [2500]`: 3 rows, 10,100 real of 12,000 slots = 84.2%. The smallest possible is
ceil(10100 / 4000) = 3 rows (2 rows hold only 8,000 slots), so this is the best any rule can do.

**What it teaches:** both rules agree here because at every choice the first row with room (row 2) is also the row
with the least room. (With 150 instead of 100, the last chat also goes to row 2 under both rules, leaving 350 empty.)

### Example 3 - the rules choose different rows, same count (T = 4096; chats 3796, 3196, 700, 150)

```
3796 -> new row 1 (300 empty)
3196 -> row 1 has 300, too few -> new row 2 (900 empty)
 700 -> only row 2 fits -> row 2 (200 empty)
 150 -> row 1 (300 empty) and row 2 (200 empty) both fit
        First-Fit -> row 1 (the first that fits): rows [3796, 150] (150 empty) [3196, 700] (200 empty)
        Best-Fit  -> row 2 (leaves 50, fewer than 150): rows [3796] (300 empty) [3196, 700, 150] (50 empty)
```

**What it teaches:** both use 2 rows, but the empty space is spread differently. Best-Fit keeps one bigger gap (300)
open, which a later chat of up to 300 slots could still use. First-Fit leaves two medium gaps (150 and 200).

### Example 4 - Best-Fit needs FEWER rows (T = 20; chats 15, 9, 7, 4, 3, 2)

```
First-Fit-Decreasing                         Best-Fit-Decreasing
15 -> new row 1 (5 empty)                    15 -> new row 1 (5 empty)
 9 -> new row 2 (11 empty)                    9 -> new row 2 (11 empty)
 7 -> row 2 (4 empty)                         7 -> row 2 (4 empty)
 4 -> row 1 is first that fits (1 empty)      4 -> row 1 leaves 1, row 2 leaves 0 -> row 2 (0 empty)
 3 -> row 1 (1) no, row 2 (4) -> (1 empty)    3 -> row 1 (2 empty)
 2 -> no row has 2 empty -> new row 3         2 -> row 1 (0 empty)
rows [15,4] [9,7,3] [2]: 3 rows, 66.7%       rows [15,3,2] [9,7,4]: 2 rows, 100%
```

**What it teaches:** Best-Fit put the 4 into the row it filled exactly, so row 1 kept 5 slots for the 3 and the 2.
First-Fit put the 4 into row 1 because row 1 came first, and that left two gaps of 1 slot that nothing could use.

### Example 5 - First-Fit needs FEWER rows (T = 20; chats 15, 9, 7, 3, 2, 2, 2)

```
First-Fit-Decreasing                         Best-Fit-Decreasing
15 -> new row 1 (5 empty)                    15 -> new row 1 (5 empty)
 9 -> new row 2 (11 empty)                    9 -> new row 2 (11 empty)
 7 -> row 2 (4 empty)                         7 -> row 2 (4 empty)
 3 -> row 1, first that fits (2 empty)        3 -> row 1 leaves 2, row 2 leaves 1 -> row 2 (1 empty)
 2 -> row 1 (0 empty)                         2 -> row 2 has 1, too few -> row 1 (3 empty)
 2 -> row 2 (2 empty)                         2 -> row 1 (1 empty)
 2 -> row 2 (0 empty)                         2 -> no row has 2 empty -> new row 3
rows [15,3,2] [9,7,2,2]: 2 rows, 100%        rows [15,2,2] [9,7,3] [2]: 3 rows, 66.7%
```

**What it teaches:** this time Best-Fit is the one that loses. It put the 3 into the tighter row 2 and left 1 slot
there that no remaining chat (all of length 2) could use. **Examples 4 and 5 together show that neither rule always
wins.** Each one can beat the other by a row on some inputs.

### Example 6 - why sort longest first (T = 10; chats arrive as 3, 3, 3, 7, 7, 7)

```
First-Fit WITHOUT sorting (arrival order)    First-Fit-Decreasing (sorted: 7, 7, 7, 3, 3, 3)
3, 3, 3 -> row 1 (1 empty)                   7, 7, 7 -> rows 1, 2, 3 (3 empty each)
7 -> new row 2 (3 empty)                     3 -> row 1 (0 empty)
7 -> new row 3 (3 empty)                     3 -> row 2 (0 empty)
7 -> new row 4 (3 empty)                     3 -> row 3 (0 empty)
4 rows, 75% real                             3 rows, 100% real
```

**What it teaches:** without sorting, the three short chats fill row 1 first, and then every long chat needs a row of
its own with a 3-slot gap that nothing fills. With sorting, the long chats come first and each gap of 3 gets exactly one
short chat. Sorting longest first matters more than the choice between First-Fit and Best-Fit.

### Example 7 - a chat exactly T long (T = 10; chats 10, 5, 5)

```
10 -> new row 1 (0 empty)        the row is full with one chat
 5 -> row 1 has 0 empty -> new row 2 (5 empty)
 5 -> row 2 (0 empty)
rows: [10] [5, 5] -> 2 rows, 20 of 20 slots real = 100%
```

**What it teaches:** a chat of exactly T slots fills a row on its own; no other chat can join it. That costs nothing:
the row is 100% real tokens. Both rules give the same here.

### Example 8 - the smallest possible number cannot be reached (T = 10; chats 6, 6, 6, 6)

```
6 -> new row 1 (4 empty)
6 -> row 1 has 4 empty, too few -> new row 2 (4 empty)
6 -> rows 1 and 2 have 4 empty each, too few -> new row 3 (4 empty)
6 -> new row 4 (4 empty)
rows: [6] [6] [6] [6] -> 4 rows, 24 of 40 slots real = 60%
```

By division the smallest possible is ceil(24 / 10) = 3 rows. It cannot be reached: 6 + 6 = 12 does not fit in 10, so
no two chats can ever share a row. The division number assumes chats could be cut, and we never cut a chat.

**What it teaches:** a row can end with a gap that no chat is short enough to fill, and then no rule can reach the
division number. In our real data (example 15), First-Fit and Best-Fit also leave such gaps (our shortest chat fills 134
slots, so a gap under 134 slots can never be filled), but there a smarter method avoids making them and does reach the
division number.

### Example 9 - many one-token chats (T = 10; 25 chats of length 1)

```
chats 1-10  -> row 1 (after the 10th chat: 0 empty)
chat 11     -> row 1 is full -> new row 2; chats 11-20 fill row 2
chat 21     -> rows 1 and 2 are full -> new row 3; chats 21-25 -> row 3 (5 empty)
rows: [1 x 10] [1 x 10] [1 x 5] -> 3 rows, 25 of 30 slots real = 83.3%
```

**What it teaches:** a row can hold many chats. Row 1 here holds 10 separate chats, so with document masking its
attention is 10 separate triangles of 1 slot each. Our real rows hold 1 to 12 chats (example 15). Very short chats also
waste a little inside the attention kernel, because a kernel tile is 64 or 128 positions wide (section 10.6).

### Example 10 - all chats exactly half a row (T = 10; 7 chats of length 5)

```
5, 5 -> row 1 (0 empty)
5, 5 -> row 2 (0 empty)
5, 5 -> row 3 (0 empty)
5    -> rows 1-3 are full -> new row 4 (5 empty)
rows: [5, 5] [5, 5] [5, 5] [5] -> 4 rows, 35 of 40 slots real = 87.5%
```

**What it teaches:** with an odd number of half-row chats, one row is always half empty, whatever the rule. The minimum
by division is ceil(35 / 10) = 4 rows, so here the division number is reached.

### Example 11 - empty chats and chats without an answer

These never reach packing, because there is nothing in them to train:

- TRL removes chats of length 0 before packing (`_pack_bfd`, the `non_empty_mask` at `trl/data_utils.py:750`).
- Our preprocessing (`sft/prepare_sft_data.py:38-51`) drops a chat, before any row is made, when:
  - the row is empty (`empty_row`);
  - a turn has a role other than system / user / assistant (`unknown_role`);
  - there is no assistant turn at all (`no_assistant_turn`): nothing for the model to learn to say;
  - the last turn is not the assistant's (`last_turn_not_assistant`), for example a chat that ends with a user question
    and no answer;
  - an assistant answer is empty (`empty_answer`).
- Each drop is counted with its reason in `meta.json`. Stage-0 had none of these; its only drops were 10 chats that were
  too long (section 7.5).

**What it teaches:** packing only ever sees chats that have something to train.

### Example 12 - one chat longer than the row (T = 10; chats A = 13, B = 4, C = 3)

A needs 13 slots, but a row has only 10. Each method handles it differently:

**Drop (ours, torchtitan).** A is removed before packing. B and C are packed:

```
row 1: [B B B B C C C . . .]      1 row; all 13 slots of A are lost
```

**TRL `bfd` (cut the end).** A is cut to its first 10 slots; its last 3 are thrown away. Then Best-Fit:

```
row 1: [A A A A A A A A A A]      A's slots 1-10 (A's end, including its end token, is gone)
row 2: [B B B B C C C . . .]      2 rows; 3 slots of A lost
```

**TRL `bfd_split` (cut into chunks of 10).** A becomes chunk A1 (A's slots 1-10) and chunk A2 (A's slots 11-13). Each
chunk is packed like a separate chat. Sorted: A1 (10), B (4), A2 (3), C (3):

```
row 1: [A1 A1 A1 A1 A1 A1 A1 A1 A1 A1]      chunk A1
row 2: [B  B  B  B  A2 A2 A2 C  C  C ]      B, chunk A2, C: 2 rows, nothing lost
```

A2 is attended on its own: it never sees A1 (section 8.3).

**TRL `wrapped` (one stream, cut every 10).** The chats are joined in arrival order A, B, C into one stream of 20 and cut
every 10:

```
row 1: [A A A A A A A A A A]      A's slots 1-10
row 2: [A A A B B B B C C C]      A's slots 11-13, then B, then C: 2 rows, nothing lost
```

In TRL, row 2 is one sequence: B reads A's end, and C reads A's end and all of B (section 8.1).

| method | rows | slots of A lost | what goes wrong |
| --- | --- | --- | --- |
| drop (ours, torchtitan) | 1 | all 13 | chat A is not trained at all |
| TRL `bfd` | 2 | 3 | A is trained without its end, so it is never shown stopping |
| TRL `bfd_split` | 2 | 0 | A's last 3 slots are trained without A's beginning |
| TRL `wrapped` | 2 | 0 | A is cut; B and C read A's end (cross-document attention) |

### Example 13 - one chat far longer than the row (T = 10; chats A = 25, B = 2)

**Drop:** A removed; `[B B . . . . . . . .]` -> 1 row, all 25 slots of A lost.

**`bfd`:** A cut to 10 (15 slots lost); rows `[A x 10]` and `[B B . . . . . . . .]` -> 2 rows.

**`bfd_split`:** A becomes chunks of 10, 10 and 5. Sorted: 10, 10, 5, 2:

```
row 1: [A1 x 10]          A's slots 1-10
row 2: [A2 x 10]          A's slots 11-20
row 3: [A3 x 5, B B, . . .]   A's slots 21-25, then B: 3 rows, nothing lost
```

Each chunk is attended alone: A2 never sees A1, A3 never sees A1 or A2.

**`wrapped`:** stream A (25) + B (2) = 27, cut every 10:

```
row 1: [A x 10]   row 2: [A x 10]   row 3: [A x 5, B B, + 3 pads when batched]   3 rows
```

**What it teaches:** for a chat many times longer than a row, `bfd` loses most of the chat, `bfd_split` keeps every
token but splits it into pieces that cannot see each other, and drop loses all of it.

### Example 14 - the one-slot boundary (T = 4096)

A chat of N tokens needs N - 1 slots (section 0):

| chat length in tokens | slots needed | fits in 4096? | what we do |
| --- | --- | --- | --- |
| 4,096 | 4,095 | yes, 1 slot spare | kept |
| 4,097 | 4,096 | yes, exactly | kept |
| 4,098 | 4,097 | no, 1 slot too many | dropped |

Our check is `len(res[0]) - 1 > T`, where `res[0]` is the chat's list of token ids (`sft/prepare_sft_data.py:167-168`):
drop when the slots needed are more than T.

**What it teaches:** "longer than the row" means more than 4,096 slots, which is more than 4,097 tokens.

### Example 15 - our real stage-0 data: why First-Fit and Best-Fit give 1,517 rows, and how 1,511 can be reached (T = 4096; 5,000 chats)

**The data.** 5,000 chats from UltraChat. Each chat fills between 134 slots (the shortest) and 4,069 slots (the
longest); the average is 1,238 slots (1,239 tokens) and the median is 1,154 slots.

**Without packing (v1, today): 5,000 rows.** One chat per row:

```
slots:        5,000 rows x 4,096 = 20,480,000
real tokens:  6,188,911
real share:   6,188,911 / 20,480,000 = 30.2%      -> 69.8% of the slots are pads
```

Counted as GPU work instead of slots (with `pack_sim.py`'s work model), the pads take 72.3% of today's work, because
attention work grows with the square of the row length (`ffd_vs_bfd_study_output.txt`).

**Step 1. Count the real tokens.** Adding up the lengths of all 5,000 chats gives **6,188,911** slots of real tokens.

**Step 2. The smallest possible number of rows.** Each row holds 4,096 slots.

```
6,188,911 / 4,096 = 1,510.96
```

A part of a row does not exist, so round up to **1,511**. No packing method can use fewer rows:

```
1,510 rows hold 1,510 x 4,096 = 6,184,960 slots  <  6,188,911  -> too few (3,951 tokens would not fit)
1,511 rows hold 1,511 x 4,096 = 6,189,056 slots  >= 6,188,911  -> just enough, 145 slots spare
```

So 1,511 rows are possible only if **all the rows together** have no more than 145 empty slots. Those 145 empty slots
can be spread over any rows (step 6 shows a packing that does exactly this).

**Step 3. What First-Fit-Decreasing and Best-Fit-Decreasing give: 1,517 rows** (both rules give the same count on this
data, `ffd_vs_bfd_output.txt`).

```
slots:              1,517 x 4,096 = 6,213,632
real tokens:                         6,188,911
empty (pad) slots:                      24,721 = 0.40% of all slots  -> 99.60% real
empty per row:      24,721 / 1,517 = 16.3 slots on average
```

24,721 = 6 x 4,096 + 145: the 6 extra rows' worth, plus the 145 spare slots that even the smallest possible packing
(1,511 rows) has.

**Step 4. Where the empty slots are.** Most rows are completely full; the empty slots sit in a few rows. The two rules
give the same row count but slightly different rows (1,395 of the 1,517 rows are the same in both;
`ffd_vs_bfd_study_output.txt`):

| empty slots at the end of the row | rows, First-Fit-Decreasing | rows, Best-Fit-Decreasing |
| --- | --- | --- |
| 0 (completely full) | 1,039 (68.5% of the rows) | 1,060 (69.9%) |
| 1 to 16 | 310 | 292 |
| 17 to 64 | 48 | 45 |
| 65 to 256 | 114 | 114 |
| more than 256 | 6 (the largest gap is 3,222 slots, in one of the last rows) | 6 (largest 3,222) |

The median row has 0 empty slots; the average of 16.3 comes from the few rows with big gaps.

**Step 5. Why First-Fit and Best-Fit stop at 1,517.** We never cut a chat across two rows, so a gap at the end of a row
can only be filled by a chat short enough to fit in it.

- The shortest chat in stage-0 fills **134 slots**, so a gap smaller than 134 slots can never be filled by any chat.
  - In the First-Fit layout, **403 rows** end with such a gap (6,906 empty slots in total).
  - In the Best-Fit layout, **382 rows** do (6,790 slots).
- The other **75 rows** with a gap end with 134 or more empty slots (First-Fit 17,815 slots, Best-Fit 17,931). They could
  only have been filled by a chat of a fitting size that was still unplaced at that moment.
- First-Fit: 1,039 + 403 + 75 = 1,517 rows. Best-Fit: 1,060 + 382 + 75 = 1,517 rows.

The reason is how both rules work: each chat is placed once, longest first, into a row chosen only by how much room the
rows have at that moment, and a placed chat is never moved. A row that ends up with a small gap keeps it. These gaps come
from the order of placement, not from the data: a method that chooses chats differently does not create most of them
(step 6).

**Step 6. 1,511 rows can be reached** (found by a review on 2026-10-14; an earlier version of this document wrongly said
it could not). The method:

1. Open a new row with the longest chat that is not placed yet.
2. Fill the rest of that row with the group of unplaced chats whose lengths add up as close as possible to the free
   space, without going over. Finding such a group is a known problem called "subset sum"; here it is solved with a
   table of every total from 0 to 4,096 that the chats can make.
3. Repeat until every chat is placed.

Result on stage-0 (`subset_sum_packing_1511.py`, checked by the separate `check_packing_1511.py`; output in
`subset_sum_packing_1511_output.txt`):

- **1,511 rows**, every chat placed exactly once, no row over 4,096 slots.
- **1,502 rows are full to the last slot.** All 145 empty slots sit in 9 rows:

  | chats in the row (slots) | empty slots |
  | --- | --- |
  | 4,069 (the longest chat, alone) | 27 |
  | 3,933 + 152 | 11 |
  | 3,919 + 169 | 8 |
  | 3,923 + 169 | 4 |
  | 3,862 + 232 | 2 |
  | 3,837 + 258 | 1 |
  | 3,796 + 299 | 1 |
  | 3,791 + 304 | 1 |
  | 19 short chats (303 down to 134) | 90 |
  | total | 145 |

- A stronger lower bound from the bin-packing literature (Martello and Toth's "L2" bound, computed in the same script)
  is also 1,511, so 1,511 is the true minimum for these chats.
- It took 3.3 seconds for 5,000 chats. As written, every new row searches all chats not yet placed, so it gets much
  slower on large datasets (not measured).
- The 4,069-slot chat's row keeps 27 empty slots in every possible packing, because no chat is 27 slots or shorter.

**Step 7. How many chats share a row.** On average 5,000 / 1,517 = 3.3 with First-Fit and Best-Fit, and 5,000 / 1,511 =
3.3 with the subset-sum method:

| chats in the row | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | more than 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rows, First-Fit-Decreasing | 1 | 474 | 527 | 290 | 129 | 49 | 25 | 11 | 6 | 2 | 2 | 1 | 0 |
| rows, Best-Fit-Decreasing | 4 | 474 | 529 | 282 | 127 | 53 | 25 | 12 | 6 | 2 | 1 | 2 | 0 |
| rows, subset-sum method | 1 | 343 | 817 | 193 | 22 | 82 | 21 | 1 | 16 | 1 | 6 | 3 | 5 (14, 14, 14, 16 and 19 chats) |

The only First-Fit row with a single chat holds the longest chat (4,069 slots). Best-Fit has 4 single-chat rows (the
chats of 4,069, 3,933, 3,919 and 3,862 slots).

**Step 8. What it means.**

| | rows | real tokens | work (pack_sim.py, attention + matrix work) |
| --- | --- | --- | --- |
| no packing (1 chat per row, v1 today) | 5,000 | 30.2% | 100% |
| packing, First-Fit-Decreasing or Best-Fit-Decreasing | 1,517 | 99.6% | 27.8% |
| packing, subset-sum method (the minimum) | 1,511 | 99.998% | not computed |

- First-Fit and Best-Fit are 6 rows (0.4%) above the true minimum. A slower method saves those 6 rows: a 0.4% saving
  on our data.
- Compared with today, packing needs 3.3 times fewer rows and about 3.6 times less work. (The work is an estimate of the
  GPU work, not a measured speed.)

### Example 16 - TRL packs in groups of 1,000 chats: what that means (stage-0)

**What TRL does.** TRL does not pack the whole dataset at once. Its packing code runs through the `map` function of
the Hugging Face `datasets` library with `batched=True` (`trl/data_utils.py:910-925`). `map` hands the packing code the
chats **in groups: 1,000 chats at a time** (the `datasets` default, `Dataset.map(batch_size=1000)`). TRL's SFT trainer
does not change that number; it passes only the number of worker processes and a progress label
(`trl/trainer/sft_trainer.py:1475-1477, 1681-1682`; `map` gets these settings at `:1692`). The packing code sees only the
1,000 chats of its group, packs them into rows, and returns those rows. The next group starts with new, empty rows.

One more detail: this holds for TRL's default `dataset_num_proc=None` (one worker). With P workers, `datasets` first
cuts the data into P equal parts and makes the groups inside each part (`datasets` `arrow_dataset.py:3479, 3546-3553`),
so a group is then at most min(1,000, chats / P) chats. For example, 5,000 chats with 8 workers give groups of 625.

So a chat from group 1 and a chat from group 2 can never share a row, even when group 1's last row still has space.

**Is this "batch" the same as the training batch, the micro-batch, or gradient accumulation? No.** It only has the same
name. It happens once, while the data is prepared, before training starts. This document calls it a "group" to keep it
apart from the training batches:

| name | when it is used | what it counts | TRL setting | ours |
| --- | --- | --- | --- | --- |
| `map` batch (a "group" here) | once, while preparing the data, before training | how many chats the packing code sees at once; only chats of one group can share a row | `batch_size` of `datasets.map`, default 1,000; TRL does not set it | no groups: we pack all 5,000 chats at once |
| micro-batch | in training, every forward and backward pass | rows processed together on one GPU | `per_device_train_batch_size` | B = 8 rows |
| gradient accumulation | in training, every optimizer step | micro-batches whose gradients are added up before one weight update | `gradient_accumulation_steps` | GRAD_ACCUM = 8 |
| rows per optimizer step | in training | micro-batch x accumulation x number of GPUs | product of the above | 8 x 8 x 1 = 64 |

The group size only decides **which chats may share a row**. It does not change how many rows go into a training step.

**Which TRL strategy does this apply to? All three.** `bfd`, `bfd_split` and `wrapped` all run through the same
`map(batched=True)` call (`trl/data_utils.py:910-925`):

- `bfd` and `bfd_split`: each group of 1,000 chats is sorted and packed with Best-Fit on its own.
- `wrapped`: each group of 1,000 chats is joined into its own stream and cut every 4,096 tokens; the last row of each
  group is shorter.

**A tiny example (rows of 10; chats 6, 6, 4, 4).**

```
all four chats in one group (sorted 6, 6, 4, 4):
  6 -> new row 1 (4 empty)
  6 -> row 1 has 4, too few -> new row 2 (4 empty)
  4 -> row 1 (0 empty)
  4 -> row 2 (0 empty)
  rows: [6, 4] [6, 4] -> 2 rows, 100% real

two groups, group 1 = [6, 6] and group 2 = [4, 4]:
  group 1:  6 -> new row 1 (4 empty);  6 -> too few -> new row 2 (4 empty)      rows [6] [6]
  group 2:  4 -> new row 3 (6 empty);  4 -> row 3 (2 empty)                      row  [4, 4]
  rows: [6] [6] [4, 4] -> 3 rows, 20 of 30 slots real = 66.7%
```

The 4s could have filled the gaps next to the 6s, but they were in a different group, so they never met.

**Our measurement on stage-0.** I shuffled the 5,000 chats (seed 1234), cut them into groups of G chats in that order,
and packed every group on its own. (TRL shuffles before packing only when `shuffle_dataset` is on, and its default is
off, `trl/trainer/sft_config.py:219-221`. The results in TRL's default order are given below the table.)

- The "Best-Fit per group" column is TRL's `bfd`. It is also TRL's `bfd_split` here, because no stage-0 chat is longer
  than 4,096 slots, so nothing is cut.
- The "`wrapped` per group" column joins each group into one stream and cuts it every 4,096 slots.

| chats per group (G) | groups | Best-Fit per group: rows | real tokens | rows with more than 256 empty | `wrapped` per group: rows | real tokens |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 500 | 1,813 | 83.34% | 878 | 1,763 | 85.70% |
| 100 | 50 | 1,556 | 97.11% | 112 | 1,537 | 98.31% |
| 1,000 (TRL default) | 5 | 1,521 | 99.34% | 30 | 1,513 | 99.87% |
| 5,000 (all at once, ours) | 1 | 1,517 | 99.60% | 6 | 1,511 | 100.00% |

**In TRL's default order (no shuffle)** (`ffd_vs_bfd_study_output.txt`, E4): groups of 1,000 give the same 1,521 rows
(99.34% real), with 21 rows ending with more than 256 empty slots instead of 30. First-Fit gives the same 1,521 rows in
both orders.

**In TRL's own units.** TRL packs whole token lists, one token more per chat than our slots (it does not shift the
labels before packing). Counted that way, with no shuffle (`trl_units_groups_output.txt`): groups of 1,000 give 1,521
rows (99.42% real, 18 rows with more than 256 empty); all 5,000 at once give 1,518 rows (99.62% real).

**Why smaller groups need more rows.** Every group ends with its own partly empty rows, and inside a small group there
are fewer chats to choose from when a gap needs filling. With TRL's default of 1,000, stage-0 needs 1,521 rows instead
of 1,517 (4 more), and 30 rows end with more than 256 empty slots instead of 6. With groups of 10 the loss is large:
1,813 rows.

`wrapped` loses less to groups, because only the last row of each group is short. But `wrapped` cuts chats and lets
them read each other (section 8.1), so its extra fill is not usable for SFT.

**What it means for us.** We pack all 5,000 chats at once, so we have no group effect. A TRL user with the default
settings gets 99.34% real tokens on this data instead of 99.60%: a small loss, 0.26 percentage points.

### Example 17 - five chats, rows of 4096, Best-Fit step by step (used again in sections 7, 8 and 10)

Chats: A = 1500, B = 800, C = 3000, D = 500, E = 2000 tokens (7,800 in total).

BFD does not join the chats and cut them (that is `wrapped`, section 8.1). It packs whole chats into rows.

**Step 1: sort all chats by length, longest first.**

```
Chat C: 3000 tokens    <- longest
Chat E: 2000 tokens
Chat A: 1500 tokens
Chat B:  800 tokens
Chat D:  500 tokens    <- shortest
```

**Step 2: one by one, put each chat into a row.** Go through the sorted list. For each chat, find the row that has the
**least remaining space** but still has enough room for this chat. If no row has room, open a new row.

```
Processing chat C (3000 tokens):
  No rows exist yet. Open row 0.
  row 0: [C1 ... C3000 | 1096 empty]

Processing chat E (2000 tokens):
  Row 0 has 1096 free - not enough for 2000. Open row 1.
  row 1: [E1 ... E2000 | 2096 empty]

Processing chat A (1500 tokens):
  Row 0 has 1096 free - not enough for 1500.
  Row 1 has 2096 free - enough. Put it here (Best-Fit = tightest fit that works).
  row 1: [E1 ... E2000 | A1 ... A1500 | 596 empty]

Processing chat B (800 tokens):
  Row 0 has 1096 free - enough.
  Row 1 has 596 free - not enough.
  Put it in row 0 (the only option).
  row 0: [C1 ... C3000 | B1 ... B800 | 296 empty]

Processing chat D (500 tokens):
  Row 0 has 296 free - not enough.
  Row 1 has 596 free - enough.
  row 1: [E1 ... E2000 | A1 ... A1500 | D1 ... D500 | 96 empty]
```

Final layout:

```
row 0: [C (3000) | B (800) | PAD (296)]            -> 92.8% of the row used
row 1: [E (2000) | A (1500) | D (500) | PAD (96)]   -> 97.7% of the row used
```

**What it teaches:** First-Fit-Decreasing gives the same two rows here, because at every step at most one row has room
(so there is nothing to choose between). Without packing these 5 chats would need 5 rows; with packing, 2.

---

## 5. What the examples show

1. Both rules sort longest first; they differ only in which row gets each chat when more than one row fits
   (example 3).
2. Neither rule always wins: Best-Fit saves a row in example 4, First-Fit saves a row in example 5. Both usually give the
   same count (examples 1, 2, 6-10, 17, and our real data, example 15).
3. Sorting longest first matters more than the choice of rule (example 6).
4. Division gives the smallest possible row count, but it is not always reachable, because chats are never cut
   (example 8).
5. On our data the choice between the two rules does not matter: 1,517 rows either way. Both are 6 rows above the true
   minimum of 1,511, which a slower subset-sum method reaches (example 15, step 6).
6. Packing whole datasets at once fills rows best; packing in small groups costs rows (example 16).

---

## 6. Which rule we use

- Our simulation (`pack_sim.py`, 2026-10-08) and the attention test data D10 use **First-Fit-Decreasing**. (The 10-08
  table in the SFT work doc §38.5 called it Best-Fit; the code was First-Fit. The label was fixed on 2026-10-14.)
- TRL uses Best-Fit-Decreasing. Hermes-4 (per the blog) uses First-Fit-Decreasing.
- On our stage-0 data both give 1,517 rows, so the choice does not change anything for us.
- The packing script for real training is not written yet; it is Step 7 of the v2 plan, and the plan says
  First-Fit-Decreasing, whole chats, rows shuffled.
- Which rule is better, in theory and measured, is answered in section 13.

---

## 7. Chats longer than the row (N > T): what can be done, what others do, what we do

### 7.1 The options, on one small chat

A small chat for the drawings: T = 10 slots; the chat has a prompt of 4 tokens (P1-P4), an answer of 7 tokens (R1-R7)
and the end token (E): 12 tokens, so it needs 11 slots. Each slot holds one token the model reads and, under it, the
token it must predict (`-` = not trained, because it is part of the prompt):

```
the whole chat (11 slots, one more than a row holds):
reads:     P1  P2  P3  P4  R1  R2  R3  R4  R5  R6  R7
predicts:  -   -   -   R1  R2  R3  R4  R5  R6  R7  E
```

**Option 1: drop the chat (ours).** Nothing from this chat is trained.

**Option 2: cut the end (keep the first 10 slots; TRL's `keep_start`, `bfd`, Megatron):**

```
reads:     P1  P2  P3  P4  R1  R2  R3  R4  R5  R6
predicts:  -   -   -   R1  R2  R3  R4  R5  R6  R7       the last slot (R7 -> E) is gone
```

The model learns R1-R7 but is never trained to predict E after this answer. It is never shown where this answer stops.

**Option 3: cut the start (keep the last 10 slots; TRL's deprecated `keep_end`):**

```
reads:     P2  P3  P4  R1  R2  R3  R4  R5  R6  R7
predicts:  -   -   R1  R2  R3  R4  R5  R6  R7  E        P1 is gone
```

The answer and E are kept, but the start of the question is missing, so the model learns to answer a question it did
not fully see.

**Option 4: cut into chunks (TRL's `bfd_split`):**

```
chunk 1 (10 slots):  reads P1 ... R6, predicts - - - R1 ... R7
chunk 2 (1 slot):    reads R7,        predicts E            packed into some other row
```

Nothing is lost, but chunk 2 is attended on its own: when the model predicts E, it sees only R7, not the question and
not the rest of the answer.

**Option 5: a longer row (bigger T).** With T = 11 or more the whole chat fits. For our model this is not free: it was
pretrained on 4,096-token rows, positions past 4,096 were never trained, so a long-context training stage is needed
first. Our script refuses T > 4096 (`prepare_sft_data.py:141`).

**Option 6: cut whole later turns of a multi-turn chat.** If a chat has two question-and-answer turns and only the first
turn fits, keep the first turn (with its complete answer, followed by an end token) and drop the second turn. The model
then sees one complete question and one complete answer that stops properly. None of TRL, torchtitan or Megatron does
this by default.

**The same options on a real-size chat:** T = 4000; one chat = a 500-token prompt + a 3,999-token answer + the end
token = 4,500 tokens.

| option | what training sees | what goes wrong |
| --- | --- | --- |
| **1. drop the chat** (ours) | nothing from this chat | the chat is lost; if many chats are long, the model sees few long answers |
| **2. cut the end** (keep the first 4000) | the prompt + the first 3,500 answer tokens | the last 499 answer tokens **and the end token** (500 tokens in all) are gone; this answer is never shown stopping |
| **3. cut the start** (keep the last 4000) | the whole answer and the end token, and none of the prompt (the last 4,000 tokens are exactly the 3,999 answer tokens + the end token) | the model learns to answer without seeing any of the question |
| **4. cut into chunks** | chunk 1 = the first 4000 tokens; chunk 2 = the last 500, packed elsewhere | chunk 2 is attended on its own (section 8.3), so the end of the answer is learned without the question |
| 5. a longer row (bigger T) | the whole chat | our model was pretrained on 4096-token rows; positions past 4096 were never trained, so a long-context stage is needed first; our script refuses T > 4096 (`prepare_sft_data.py:141`) |
| 6. cut whole later turns of a multi-turn chat | the first turns, each answer complete with its end token | the later turns are lost; none of TRL, torchtitan or Megatron does this by default |

### 7.2 What happens if a chat is longer than 4096 under BFD

Take example 17 again, but say chat C was actually 5,500 tokens. It cannot fit in any row (the most a row holds is
4096). **BFD (TRL's `bfd`) cuts it to 4096. The last 1,404 tokens are thrown away. They are never seen during training.
This is the data loss.**

### 7.3 The five chats with C = 5,500, under every method (T = 4096; computed)

| method | rows | what happens to C |
| --- | --- | --- |
| drop C, then Best-Fit | row 0: E (2000) + A (1500) + D (500), 96 empty; row 1: B (800), 3296 empty | all 5,500 tokens lost |
| `bfd`: cut C to 4096, then Best-Fit | row 0: C (4096); row 1: E + A + D, 96 empty; row 2: B, 3296 empty | the last 1,404 tokens are thrown away and never seen in training |
| `bfd_split`: C -> C1 (4096) + C2 (1404), then Best-Fit | row 0: C1 (4096); row 1: E + A + D, 96 empty; row 2: C2 (1404) + B (800), 1892 empty | kept, but C2 is attended without C1 (section 8.3) |
| `wrapped` (arrival order A, B, C, D, E) | row 0: A 1-1500, B 1-800, C 1-1796; row 1: C 1797-5500, D 1-392; row 2: D 393-500, E 1-2000, 1988 empty | kept, but cut at a row end; C and D are each split over two rows |

### 7.4 What others do (checked in their source code)

| framework | what it does with an over-long chat | where |
| --- | --- | --- |
| TRL SFT trainer | option 2: cuts at `max_length` (default 1024), `truncation_mode="keep_start"`; option 3 (`keep_end`) is deprecated | `trl/trainer/sft_config.py:203-216` |
| TRL packing `bfd` | option 2 (cut the end) | `trl/data_utils.py:754-755, 860-869` |
| TRL packing `bfd_split` | option 4 (chunks of `max_length`) | `trl/data_utils.py:756-769` |
| TRL packing `wrapped` | the whole group is one stream cut every `max_length` tokens (section 8.1) | `trl/data_utils.py:829-843` |
| torchtitan | option 1, drop, "avoiding training on truncated responses"; a TODO notes cutting as a possible change | `torchtitan/hf_datasets/text_datasets.py:145-147, 210-217` |
| Megatron-LM SFT | option 2: when a packed row overflows, cuts the end | `megatron/training/datasets/sft_dataset.py:163-172` |
| **ours** | **option 1, drop**, the only method (a split was built on 2026-10-30 and removed on 2026-10-31, section 7.8) | `sft/prepare_sft_data.py` (`_examine`: `reason = "too_long"`) |

### 7.5 Why we drop

- Decided 2026-09-25 (SFT work doc §32.2, "EOT must be a target"): the end token must be a training target at the end of
  every answer, otherwise the model never learns to stop. Cutting the end removes that target (option 2 in 7.1). So:
  drop, do not cut.
- Same rule as torchtitan (SFT work doc §34.2, item 4: "drop over-long examples instead of truncating").
- Stage-0: 10 of the 5,010 sampled chats were too long and were dropped (0.2%; `meta.json`: `drops: {too_long: 10}`).

### 7.6 The one thing that none of these can fix

If a conversation is 8,000 tokens and the row is 4096, no strategy can let the model attend from token 5,000 back to
token 1,000 in one forward pass: the two tokens are never in the same sequence. That requires a longer
`max_seq_length`. Packing strategies only control how you arrange the data into fixed-length rows; they cannot extend the
attention window. For our model, a longer row also needs a long-context training stage first (option 5).

### 7.7 What we have not done yet

The drop rate on long datasets is not measured. UltraChat is short (mean 1,238 slots, about 1,239 tokens). Reasoning data (long thinking)
and agentic data (long tool traces) could lose a large share under the drop rule. For any new dataset, the first check is
the `too_long` count in `meta.json`. If it is more than a few percent, choose between option 6 (cut whole later turns) and
option 5 (a long-context stage first, then a bigger T).

**Update 2026-10-30 / 10-31:** measured on the 10-28 mixes (23,253 chats over 4,096 tokens) and decided: drop, and a
longer row as the fix (section 7.8).

---

### 7.8 Long chats: what we built, tested, researched and decided (2026-10-14 to 2026-10-31)

**Final decision (the user, 2026-10-31): a chat longer than the row is dropped. That is the only method in the code.**
There is no option and no second algorithm. The split code pushed on 2026-10-30 (`--long-chats split_trl | split_turns`,
`--ctx-frac`, `seq_complete.bin`, `sft/test_long_chat_split.py`) was removed again on 2026-10-31. The data files are
byte-identical to the pushed drop mode (section 7.8.7).

#### 7.8.1 Why we drop (the user's reasoning)

- **What SFT must teach.** The model must learn to answer a question fully and to stop at the right place: the end
  token is a training target at the end of every answer (section 7.5). So every training example must be a whole,
  correct chat. More tokens or more chats do not help when the examples are broken. SFT uses little compute, so saving
  compute by cutting chats into pieces is not the goal.
- **A long answer cut into pieces confuses the model.** The first piece ends in the middle of the answer, without an
  end token. The next piece starts in the middle of the answer. If the question is repeated in front of it (our
  `split_turns`), the model sees the same question followed by an answer that starts halfway. If the question is not
  repeated (TRL's `bfd_split`), the piece has no question at all, and the model learns to write text without the
  question it answers.
- **A multi-turn chat cannot be cut into single turns.** Every turn depends on the turns before it:

  ```
  <|user|> What is 2+2?  <|assistant|> 4<end>   <|user|> Times 3?  <|assistant|> 12<end>
  ```

  "Times 3?" means nothing without turn 1. Real examples from WildChat: "yes please I need further sections to be
  translated", "yes please continue and translate further sections", "满意" ("satisfied"). Cutting such a chat into
  separate single turns gives pieces without their context.
- **So we drop, as torchtitan does** ("avoiding training on truncated responses", section 7.4). Megatron-LM does not
  drop; it cuts the end of a row that overflows, which removes the end token (the worst option for SFT).
- **The permanent fix is a longer row**, after a long-context checkpoint (section 7.6; `sft/README.md` section 11).
  Dropping keeps that need visible instead of hiding it behind broken pieces.

#### 7.8.2 The only split we would consider later

Only if needed: **exactly TRL's `bfd_split`** (section 8.3: the chat's tokens are cut into blocks of T + 1 tokens,
nothing is added at a cut, every block is packed as a chat of its own), **with one simple rule added: if ANY prompt of
the chat (the system message + the question of ANY turn, first or later) is as long as a whole row or longer, the whole
chat is dropped**, single-turn and multi-turn alike. Every other chat is split by `bfd_split` as it is. Nothing else.

Examples (T = 4,096, so a row holds 4,097 tokens; counts rounded, the role markers counted with the question):

| case | the chat | result |
| --- | --- | --- |
| single turn, long question | a 10,000-token question (a long document + "summarise it"), a 500-token answer | the prompt is longer than a row: **whole chat dropped** |
| multi-turn, long FIRST question | turn 1: a 6,000-token pasted document + "what does it say?", a 300-token answer; turn 2: 100 + 400 | turn 1's prompt is longer than a row: **whole chat dropped**, turn 2 too |
| multi-turn, long LATER question | turn 1: 300 + 700; turn 2: a 9,000-token pasted log + "what is wrong?", a 400-token answer | turn 2's prompt is longer than a row: **whole chat dropped**, turn 1 too |
| multi-turn, long answer | turn 1: 200-token question + 6,000-token answer; turn 2: 300 + 500 | no prompt is a whole row: **bfd_split as it is** (block 1: the question + about 3,900 answer tokens, no end token; block 2: the other ~2,100 answer tokens, no question, + turn 2) |

The `split_trl` function of commit 334ecbb (`git show 334ecbb:sft/prepare_sft_data.py`) is `bfd_split` without this
rule (it dropped only the blocks with no trained token and kept the rest of such a chat). Even with the rule, a long
answer is still cut (a block that ends with no end token, a block without its question), which is why nothing is split
now. Our own `split_turns` and the 2026-10-31 variants below are not to be used.

#### 7.8.3 Timeline

1. **2026-10-14 / 10-15: research, keep dropping.** Sections 7.1-7.7 and 8 were written from the code of TRL,
   torchtitan and Megatron-LM. The user wanted TRL's `bfd_split` later; the open points were written down (SFT work doc
   §45.4): where to cut, the missing end token in every piece but the last, targets in a piece that starts inside an
   answer, the context lost between pieces, an ablation drop vs split.
2. **2026-10-30: measured, built, tested, pushed.**
   - Re-checked: TRL `bfd_split` = "splits overflow sequences ... may break conversation structure in SFT datasets";
     Megatron-LM cuts (it does not split); torchtitan drops.
   - Measured every chat over 4,096 tokens in the 10-28 mixes (section 7.8.4).
   - Built `split_trl` (TRL's `bfd_split`, prompt-only pieces left out) and `split_turns` (ours): cut between turns;
     a turn longer than a row cut inside its answer, with the question and the last answer tokens repeated, untrained,
     in front of the rest; a turn whose question does not fit half a row left out with every later turn.
   - Tests: hand-made chats and 5,000 random chats per template, every piece checked (each answer token trained once,
     after its true previous token; end tokens only at real ends). A bug found by the real data: `--num` counted pieces,
     not chats, so the val set took 124,638 chats instead of 80; fixed before the push.
   - GPU gates (P022 H200): the trainer's step-0 val loss on split data equalled the numpy reference (1.780719 vs
     1.779986; on rows holding every case 1.717223 vs 1.716187).
   - Pushed in BluTrain 334ecbb (default still `drop`).
3. **2026-10-31 morning: case 4 judged too strict, about 40 variants studied (CPU only).** 11,734 of the 23,253 long
   chats have a turn whose question is longer than half a row (2,072 where the question still fits one row, 9,662
   where it does not). We studied every way we could think of to keep them, on all 23,253 chats, scored by **hidden
   copies**: an answer phrase of 8 tokens that appears earlier in the chat (in the question or an earlier turn) but is
   not in the answer's own piece. Each one trains the model to write text it cannot see. The best variant kept the
   question's first 64 and last 192 tokens plus the 256-token chunks the answer uses most, chose the earlier-turn
   context the same way, and left out a turn with more than 100 hidden copies per 1,000 trained tokens. It was built and
   tested, not pushed.
4. **2026-10-31: the user rejected it as too complex and confusing**, and proposed a simpler split: drop only when the
   prompt fills the whole row, cut between turns, otherwise `bfd_split`. Measured (section 7.8.5): the turns that start
   a new piece after a cut between turns lose their context badly when nothing is put in front of them.
5. **2026-10-31: decision: drop, nothing else** (section 7.8.1). The split code was removed from the branch; the
   2026-10-31 code is kept outside the repository as a patch (work records).

#### 7.8.4 The long chats, and what dropping costs

The 10-28 mixes have 23,253 chats longer than 4,096 tokens (the length of the template text): WildChat 18,653
(median 8,012 tokens, max 215,158; 14,713 multi-turn), LongAlign 3,559 (a long document then a question; median
10,448), smol-summarize-5k 522, evol_codealpaca 337, flan_v2 111, others 71.

What would fit whole with a longer row (no split needed):

| row length T | chats that fit whole (of 23,253) |
| --- | --- |
| 4,096 | 0 |
| 8,192 | 10,915 |
| 16,384 | 19,538 |
| 32,768 | 22,043 (every LongAlign chat) |
| 65,536 | 22,998 |

On the 10-28 long-source mix (LongAlign, smol-summarize, openhermes, WildChat; T 4,096): drop keeps 132,316 chats with
90.7 M answer tokens and drops 22,744 chats holding about 102 M answer tokens.

#### 7.8.5 Everything we measured (all 23,253 long chats, T 4,096)

"Hidden copies per 1,000" = per 1,000 trained answer tokens (section 7.8.3, item 3). Lower is better. "Tokens per
trained token" = all tokens in the pieces / trained answer tokens (the compute cost).

| method | chats used | answer tokens trained | tokens per trained token | hidden copies per 1,000 |
| --- | --- | --- | --- | --- |
| drop (what we do) | 0 | 0 | - | - |
| TRL `bfd_split` (prompt-only pieces left out) | 23,253 | 103.4 M | 1.65 | 45.3 (up to 103.6 for long questions) |
| `split_turns` as pushed 2026-10-30 | 12,847 | 79.0 M | 1.85 | 25.7 |
| the 2026-10-31 best variant (with its rule) | nearly all (train mix: 22,630 of 22,738) | 97.1 M | 2.06 (train mix) | 11.6 (every kind at or below 20.4) |
| the user's simple split, nothing in front of a new piece | 14,702 | 87.7 M | 1.43 | 41.3 |
| the user's simple split, earlier whole turns in front | 14,702 | 87.7 M | 1.89 | 31.6 |

Other 2026-10-31 findings:
- Long questions: LongAlign puts the instruction at the end of the document; WildChat at the start or the end. Keeping
  only the end of the question lost the instruction in WildChat chats; chunks chosen by the answer alone kept the
  question's start in only 22% of pieces.
- Repeating the whole question in every piece gave the fewest hidden copies (5.3) but 5.4 tokens per trained token,
  and is impossible when the question is longer than a row.
- Turns that start a new piece after a cut between turns (24,168; 4,572 with a question of 20 tokens or less): 91.9
  hidden copies per 1,000 with nothing in front, 55.0 with the earlier whole turns in front.
- No end token is ever added at a cut (it would teach the model to stop halfway). In every split method the end token
  is trained once, in the last piece. Truncation (keeping only the start) loses it completely.
- None of these numbers says which broken piece hurts the model least; only an evaluation could. The decision does not
  depend on that: SFT data should be whole chats.

#### 7.8.6 Facts checked along the way

- The system message is never trained; a chat whose system message alone is longer than half a row: 0 of the 23,253.
- A piece with no trained token gives exactly zero learning: the loss is only on trained tokens, and in a packed row
  each chat attends only to itself, so nothing reads that piece. TRL keeps such pieces (39% of the tokens of its split
  of our long chats); they only cost compute.
- The row length: `prepare_sft_data.py --seq-len` writes `T` into `meta.json`, the trainer takes `BLU_SFT_T`, and the
  data loader stops if they differ. Until 2026-10-31 the script refused T > 4,096 (a hard-coded limit; the 8,192 kernel
  benchmarks of 10-28 used rows repacked from 4,096 data because of it). Now it accepts multiples of 256 up to 131,072,
  its length histogram reaches T, and `run_sft.sh` takes T from the data's `meta.json` (one place to set it).

#### 7.8.7 The removal (2026-10-31)

Removed from `sft/prepare_sft_data.py`: `--long-chats`, `--ctx-frac`, `split_trl`, `split_turns`, the message spans,
`seq_complete.bin` and the `long_chats` part of `meta.json`. Removed from `sft/SftDataLoader.h`: the optional
`seq_complete.bin` (every chat must again end with its end token). Removed: the `seq_complete.bin` copy in
`sft/ref_val_loss.py` and `sft/test_long_chat_split.py`. Kept: the 135M tokenizer and templates, `--mix`,
`--part-cache`, `--workers`. Proof (CPU, 2026-10-31): the pushed script (drop mode) and the new script on the same data
write byte-identical `inputs.bin`, `labels.bin`, `valid_counts.bin`, `seq_lengths.bin` and `seq_offsets.bin` (train and
val): the long-source mix (132,316 + 80 chats) and a 6,000 + 80 chat mix in tulu-plain-v1 and chatml-v1; 1 worker and 8
workers give the same files; `meta.json` differs only in the two removed fields (`long_chats`,
`packing.seq_complete`); `--self-test` passes for GPT-2 and the three 135M templates.

Records: SFT work doc §45.4, §45.108 parts 15-18, §45.109-45.112; plan file Q&A 137-148; the scripts and logs in
`profiling_reports/long_chat_split_2026_10_30/` and `profiling_reports/long_prompt_variants_2026_10_31/` (work records
repository).

## 8. TRL's three packing strategies in detail

**Long chats (section 7.8):** we drop. TRL's `bfd_split` behaviour and our own `split_turns` were built, tested and
pushed on 2026-10-30 and removed again on 2026-10-31. If a split is ever needed, it is exactly `bfd_split`, with a
chat dropped whole when any of its prompts is as long as a row (section 7.8.2).

### 8.1 `wrapped` - exactly how it works

`wrapped` is the simplest strategy. It has two steps and nothing else.

**Step 1: join (TRL calls this concatenate).** Take the chats and write them one after another into one single long
stream of tokens.

- Each chat still ends with its own end token (the chat template puts it there). What is missing is any record of where
  each chat starts: after joining, nothing says "a new chat starts here". It is just one token after the next.
- TRL does this per group of 1,000 chats (example 16), not for the whole dataset at once, and it shuffles the chats first
  when `shuffle_dataset` is on, "to avoid correlations between sequences packed together"
  (`trl/trainer/sft_trainer.py:1686-1689`).

Say you have 5 chats (the chats of example 17):

```
Chat A:  1500 tokens
Chat B:   800 tokens
Chat C:  3000 tokens
Chat D:   500 tokens
Chat E:  2000 tokens
```

After joining you have one stream of 7,800 tokens:

```
[A1 A2 ... A1500  B1 B2 ... B800  C1 C2 ... C3000  D1 ... D500  E1 ... E2000]
 <---- 1500 ----> <--- 800 ----> <----- 3000 ----> <-- 500 --> <-- 2000 -->
```

**Step 2: cut the stream into rows of `max_seq_length`.** Cut it into pieces of exactly 4096 tokens each.

```
row 0: positions 0-4095 of the stream
       [A1 ... A1500 | B1 ... B800 | C1 ... C1796]
        <- chat A ->   <- chat B ->  <- first 1796 tokens of chat C ->

row 1: positions 4096-7799 of the stream (3,704 tokens, padded to 4096 when the batch is built)
       [C1797 ... C3000 | D1 ... D500 | E1 ... E2000 | PAD PAD ... PAD]
        <- last 1204 of C -> <- chat D -> <- chat E ->   <- 392 pads ->
```

TRL's `_pack_wrapped` (`trl/data_utils.py:829-843`) leaves the last row of each group shorter (3,704 here); the pads are
added later by the collator, when it builds the batch.

That is it. That is the whole algorithm. There is no decision-making in it: no sorting, no fitting, no attempt to keep
chats whole. Join, then cut.

**What this means.** Look at row 0. It contains:

- all 1,500 tokens of chat A;
- all 800 tokens of chat B;
- the first 1,796 tokens of chat C.

Three different conversations sit next to each other in one row. Chat C got **cut in two**: its first 1,796 tokens are
in row 0, its remaining 1,204 tokens are in row 1. Note that C (3,000 tokens) is shorter than a row: in `wrapped`, any
chat can be cut at a row end, not only long ones.

**Why this is a problem for attention.** During training, the model computes attention over each row on its own. With
plain causal attention (section 2.3), position i attends to all positions 0, 1, ..., i in the same row. So in row 0:

- Token B1 (at position 1500) attends to tokens A1 through A1500. But chat B has nothing to do with chat A. These are
  unrelated conversations. B1 should not be reading A's tokens. This is **cross-document attention** (chats leaking into
  each other, section 2.5).
- Token C1 (at position 2300) attends to all of chat A, all of chat B, and itself (it is the first C token). Again, C
  should not be attending to A or B.

Without any fix, the model trains on wrong attention patterns: tokens learn from unrelated context (section 2.6 says how
much this hurts, from sources we can check).

Three more effects of `wrapped`:

- **Wrong targets:** the last token of A (its end token) is trained to predict B1, and the last token of B to predict C1.
  The chats are unrelated, so these targets teach nothing true.
- **Cut chats:** C1797 ... C3000 sit at the start of row 1 without C's beginning (the question).
- **Positions:** positions run 0-4095 over the whole row; they do not restart at each chat.

**The same thing on a tiny row (example 12: T = 10; A = 13, B = 4, C = 3, in that order):**

```
stream:  A A A A A A A A A A A A A B B B B C C C
row 1:   A A A A A A A A A A          (A's tokens 1-10)
row 2:   A A A B B B B C C C          (A's tokens 11-13, all of B, all of C)
positions in row 2: 0 1 2 3 4 5 6 7 8 9   (one sequence, no restart)
```

What training does with row 2:

- **Attention:** plain causal over the whole row. B's tokens see A's last 3 tokens; C's tokens see A's end and all of B.
- **Targets:** the last token of A is trained to predict B's first token, and B's last token to predict C's first token.
- **Cut chat:** A's tokens 11-13 sit at the start of row 2 without A's beginning (the question).

**Can varlen attention fix this in TRL? No.** Varlen attention (section 10) needs to know where every chat starts.
`_pack_wrapped` records no chat starts (it writes no `seq_lengths`). So:

- TRL does not turn padding-free / varlen mode on for `wrapped` (`sft_trainer.py:1180`), and logs a warning when a user
  forces it (`:1188-1192`). Forcing it does not help: without recorded chat starts, each row is still treated as one
  sequence.
- So in TRL, `wrapped` always trains with plain causal attention over the row, and the cross-document attention above
  happens.
- TRL's docs: `wrapped` "may mix unrelated examples", and breaking sequence continuity "hurts performance", citing the
  Qwen3-Coder-Next technical report (`reducing_memory_usage.md:85`).
- What varlen would do for `wrapped` if a different implementation recorded the chat starts is in section 10.5.

Why `wrapped` exists: zero padding and no sorting. That suits pretraining on plain text, where documents are long and
many pretraining setups accept the mixing.

### 8.2 `bfd` (TRL's default) - how it differs from `wrapped`

BFD does not join and cut. It uses the Best-Fit-Decreasing packing rule (section 3), shown step by step for the five
chats in **example 17**: sort the chats longest first, then put each chat into the row with the least remaining space
that still fits it, opening a new row when none fits. The result for the five chats:

```
row 0: [C (3000) | B (800) | PAD (296)]            -> 92.8% of the row used
row 1: [E (2000) | A (1500) | D (500) | PAD (96)]   -> 97.7% of the row used
```

**What happens if a chat is longer than 4096?** Say chat C was actually 5,500 tokens. It cannot fit in any row (the most
is 4096). BFD cuts it to 4096. The last 1,404 tokens are thrown away. They are never seen during training. This is the
data loss (`trl/data_utils.py:754-755`).

**Key property of BFD.** Each chat sits in exactly one row as one unbroken block: all its tokens side by side. Several
chats share a row, but they sit next to each other, never mixed into each other. **No chat is ever split across two
rows.** The cost: long chats get cut.

**How TRL keeps the chats in a row apart.** TRL records the length of every chat in every row (`seq_lengths`). From it,
the positions restart at 0 at the start of every chat, and TRL turns on padding-free mode (`sft_trainer.py:1180`), in
which a FlashAttention implementation runs varlen attention, so each chat attends only to itself (sections 8.4 and
10.5).

### 8.3 `bfd_split` - what "preserving boundaries per chunk" means

`bfd_split` is BFD with one extra step before packing.

**Step 0 (the extra step): split long chats before packing.** Before running BFD, go through every chat. If a chat is
longer than `max_seq_length`, cut it into chunks of at most `max_seq_length`: full chunks of `max_seq_length` tokens,
then one shorter last chunk (`trl/data_utils.py:756-769`). Each chunk becomes a separate item to be packed.

Say chat C is 5,500 tokens:

```
Before split:  chat C = 5500 tokens (too long for 4096)
After split:   chunk C1 = 4096 tokens,  chunk C2 = 1404 tokens
```

Now chunk C1 and chunk C2 are treated as **two separate, independent documents**. Each gets its own entry in
`seq_lengths`, its own positions starting at 0, and goes into the BFD sorting and packing just like any other chat.

**Then: run BFD as normal on the chunks.** BFD packs the chunks and chats into rows. Chunk C2 (1,404 tokens) can end up in
the same row as other chats, for example chat D (500 tokens) and chat B (800 tokens):

```
row X: [C2 (1404) | D (500) | B (800) | PAD (1392)]
```

With exactly the six items of this example (C1 4096, E 2000, A 1500, C2 1404, B 800, D 500), Best-Fit gives a slightly
different layout (computed, section 7.3): D fits more tightly into the row with E and A (596 free, leaves 96) than into
the C2 row (would leave 1,392). The actual rows:

```
row 0: [C1 (4096)]
row 1: [E (2000) | A (1500) | D (500) | PAD (96)]
row 2: [C2 (1404) | B (800) | PAD (1892)]
```

**What "preserving boundaries per chunk" means.** Inside row 2, chunk C2 and chat B each take their own unbroken region
of the row. They do not overlap. The attention (varlen, section 10) keeps them apart: C2's tokens attend only to other C2
tokens, and B's tokens only to B tokens. The **boundary** of each chunk is kept: the tokens of one chunk do not mix with
the tokens of another chunk in the same row during attention. And a chunk is never cut again: each chunk lies whole
inside one row. (Section 8.4 compares this with `bfd` and `wrapped`.)

**The same on a tiny row (example 12: T = 10; A = 13, B = 4, C = 3):** A becomes A1 = A's tokens 1-10 and A2 = A's
tokens 11-13. The rows are `[A1]` and `[B, A2, C]`. In row 2, A2's tokens see only A2: not A1 (another chunk), not B, not
C.

**What you gain and what you lose.** No data is lost: all 5,500 tokens of chat C are trained on. But chunk C2 (tokens
4,097-5,500 of the original chat) **cannot attend back to chunk C1** (tokens 1-4,096). The reason is not that they sit in
different rows (rows of one batch go through the same forward pass). The reason is that TRL gives each chunk its own
length entry, so the attention treats each chunk as a separate sequence; they would not see each other even inside one
row. The model processes chunk C2 with no access to the first 4,096 tokens of that conversation. The later part of the
chat has lost its earlier context (the question and the start of the answer).

Compare:

- **BFD:** throws away tokens 4,097-5,500 entirely. Simple. You lose data.
- **BFD_split:** keeps tokens 4,097-5,500 but processes them without the context of tokens 1-4,096. You keep data but
  lose context for the later chunks.

More facts:

- TRL's own description of `bfd_split`: it "may break conversation structure in SFT datasets"
  (`trl/data_utils.py:866-867`).
- If every chat is no longer than `max_length`, `bfd` and `bfd_split` are identical, because nothing is cut
  (`trl/docs/source/reducing_memory_usage.md:88`; `trl/data_utils.py:754-765`). That is our case, since we drop chats over 4096 first.
- The idea comes from the paper "Fewer Truncations Improve Language Modeling" (arXiv 2404.10830), which packs document
  chunks with Best-Fit-Decreasing so that fewer documents are cut than with `wrapped`.

### 8.4 Boundaries: kept per chat in `bfd`, per chunk in `bfd_split`, not kept in `wrapped`

**What a boundary is.** A boundary is the place where one document ends and the next one begins. In a packed row, the
boundaries are the positions between two chats (or between a chat and the pads after it).

**What "the boundary is kept" (preserved) means.** Three things must all be true for a document:

1. **Placement:** the document lies whole inside one row. Its first token and its last token are in the same row; a row
   end never cuts it.
2. **Record:** the trainer writes down where the document starts and ends. In TRL this is `seq_lengths`; from it the
   positions restart at 0 for every document, and FlashAttention gets `cu_seqlens`. In ours it is the chat-start list
   (`seg_start` / `doc_start`).
3. **Use:** the attention uses that record, so tokens attend only inside their own document, and the first token of a
   document is never used as the target of the last token of the document before it.

If point 1 fails, a document is cut across two rows. If point 2 or 3 fails, the documents in one row read each other
(cross-document attention, section 2.5).

**Which unit is kept whole decides "per chat" or "per chunk":**

- **`bfd`: boundaries are kept per chat.** Every chat is one unit. It lies whole in one row, its length is recorded in
  `seq_lengths`, its positions restart at 0, varlen attention keeps it apart from its neighbours, and its first token is
  not a target (`sft_trainer.py:516`). The only thing lost is the true end of an over-long chat: the part after
  `max_length` is thrown away, so that chat ends early and loses its end token.
- **`bfd_split`: boundaries are kept per chunk.** The unit is the chunk, not the chat.
  - A chat no longer than `max_length` is one chunk, so for it this is the same as per chat.
  - A longer chat becomes several chunks, and each chunk is its own unit: whole in one row, recorded on its own,
    positions restarting at 0, attended on its own.
  - So the cut between two chunks of the same chat is treated as a boundary between two different documents. The link
    inside that chat is not kept: chunk C2 never sees chunk C1.
- **`wrapped`: no boundaries are kept.** None of the three points holds:
  1. Placement: rows are cut every `max_length` tokens wherever that lands, so chats are cut at row ends (chat C in
     section 8.1).
  2. Record: `_pack_wrapped` writes no `seq_lengths`. The end token of each chat is still in the stream as a token, but
     nothing tells the trainer "a new chat starts here".
  3. Use: positions run 0 .. T-1 over the whole row, the attention is plain causal over the row (cross-document
     attention), and the last token of each chat is trained to predict the first token of the next.
- **Ours: boundaries are kept per chat** (like `bfd`). Because over-long chats are dropped, every chat is also complete,
  with its own end token.

The tiny example in each strategy (T = 10; chats A = 13, B = 4, C = 3). `|` marks a recorded boundary:

```
bfd (A cut to its first 10 tokens; A's tokens 11-13 thrown away):
row 1: A A A A A A A A A A
       positions 0 1 2 3 4 5 6 7 8 9
row 2: B B B B | C C C | PAD PAD PAD      boundaries recorded: B | C | pads
       positions 0 1 2 3 | 0 1 2
       B sees only B; C sees only C

bfd_split (A cut into chunk A1 = A's tokens 1-10 and chunk A2 = A's tokens 11-13):
row 1: A A A A A A A A A A                 chunk A1
       positions 0 1 2 3 4 5 6 7 8 9
row 2: B B B B | A A A | C C C             boundaries recorded: B | A2 | C
       positions 0 1 2 3 | 0 1 2 | 0 1 2
       A2 is a separate document: it sees only A2 (not A1 in row 1, not B, not C)

wrapped (one stream, cut every 10 tokens; nothing recorded):
row 1: A A A A A A A A A A                 A's tokens 1-10; the row end cuts chat A
       positions 0 1 2 3 4 5 6 7 8 9
row 2: A A A B B B B C C C                 A's tokens 11-13, all of B, all of C
       positions 0 1 2 3 4 5 6 7 8 9       (no restart)
       B sees A's end; C sees A's end and all of B
```

| | unit kept whole in one row | start and end recorded | attention keeps units apart | positions restart | boundary target blocked | an over-long chat |
| --- | --- | --- | --- | --- | --- | --- |
| `bfd` | each chat | yes, per chat (`seq_lengths`) | yes: varlen with FlashAttention, or a same-chat mask with SDPA / eager (see below) | yes, per chat | yes (`sft_trainer.py:516`) | cut to `max_length`; its true end is lost |
| `bfd_split` | each chunk (a short chat is one chunk) | yes, per chunk | yes, per chunk | yes, per chunk | yes, per chunk | split; its chunks are separate documents |
| `wrapped` | nothing: rows are cut every `max_length` tokens | no | no: plain causal over the row | no: 0 .. T-1 over the row | no | cut at row ends, like any other chat |
| ours | each chat | yes, per chat (chat-start list) | yes (design A or B) | yes | yes (labels built per chat) | dropped before packing |

**Which attention code reads the record in TRL.**

- **FlashAttention:** `transformers` turns the restarting positions into `cu_seqlens`. Every position equal to the
  smallest value (0) starts a new document (`modeling_flash_attention_utils.py:480-515`). A batch counts as packed when
  it is one sequence whose positions do not rise in one single run (`:558-571`).
- **PyTorch SDPA or the plain "eager" attention:** in this version of `transformers`, the mask builder also finds the
  chats from the restarting positions and adds a same-chat rule to the causal mask (`masking_utils.py:734-763`, called at
  `:857-865`, added at `:976-979`; the same builder serves `sdpa` and `eager`, `:720-727`). This works for models whose
  code passes the positions to the mask builder; Llama does (`models/llama/modeling_llama.py:380-386`).
- **TRL still warns** that packing without FlashAttention "may lead to cross-contamination between samples"
  (`sft_trainer.py:1247-1255`). Two reasons the warning still matters: a model whose code does not pass the positions to
  the mask builder would let the chats read each other; and SDPA with a full mask computes every tile instead of skipping
  the blocked ones, so it is slower (SFT work doc §40.2).
- (An earlier version of this section said only FlashAttention keeps chats apart. That was wrong; see section 12.)

### 8.5 Summary of the three strategies (and ours)

| | `bfd` | `bfd_split` | `wrapped` | ours (v2 plan) |
| --- | --- | --- | --- | --- |
| **algorithm** | sort by length, pack with Best-Fit | split long chats into chunks, then sort + Best-Fit | join all, cut into fixed rows | sort by length, pack with First-Fit |
| **long chats** | cut (data lost) | split into chunks (data kept, context lost between chunks) | cut at row ends (data kept, context lost across rows) | dropped before packing (section 7.8) |
| **other chats** | always whole in one row | always whole in one row | any chat can be cut at a row end | always whole in one row |
| **what a row holds** | several complete chats | several chunks and chats | parts of several chats | several complete chats |
| **boundaries kept (8.4)** | per chat | per chunk | none | per chat |
| **needs varlen?** | yes, to keep the packed chats apart; TRL turns it on | yes, to keep the packed chunks apart; TRL turns it on | it would need it to stop the mixing, but TRL cannot use it (no chat starts recorded) | our design A or B |
| **packed together** | groups of 1,000 chats (example 16) | groups of 1,000 chats | groups of 1,000 chats | all chats at once |
| **pads** | low (BFD packs well) | low (BFD packs well) | almost none (only the last row of each group) | 16.3 per row on stage-0 |
| **how hard to build** | medium (sorting + packing) | medium (splitting + sorting + packing) | trivial (join + cut) | medium |

---

## 9. Packing changes how many optimizer steps we take (future task: research and an ablation before the first packed SFT run)

The user asked on 2026-10-14 to remember this, to research it, and to test it before the first packed SFT run. The
user's note: real runs may use much bigger datasets, where this matters less, but with stage-0 (5,000 chats) it matters
now.

### 9.1 How one training step is built in our trainer

These are the settings of our SFT runs (`bluscript_zero_ckpt.cpp:201` for B, `:920-966` for the SFT settings, `:801-818` for the learning-rate schedule;
`sft/SftDataLoader.h:16, 56`):

1. **Micro-batch: 8 rows** (B = 8). The GPU runs the model forward over 8 rows, computes the loss, and runs backward. That
   gives a gradient: for every weight, how it should change.
2. **Gradient accumulation: 8 micro-batches** (GRAD_ACCUM = 8). The trainer adds up the gradients of 8 micro-batches
   without changing the weights yet.
3. **Optimizer step: one weight update** made from the added-up gradient. One step therefore uses 8 x 8 = **64 rows**
   (`BLU_SFT_ROWS_PER_STEP` = 64). The trainer requires the rows per step to be a whole number of micro-batches: a
   multiple of B x number of GPUs (`bluscript_zero_ckpt.cpp:947-952`). With B = 8 on 1 GPU: 8, 16, 24, ..., 64.
4. **The loss of a step:** all trained (answer) tokens of the 64 rows are added up and divided by their total count once
   (the "global denominator"), so every trained token in the step counts the same.
5. **Epoch:** one pass over all rows. The loader makes rows / 8 micro-batches per epoch (rounded down); rows left over
   after the last full micro-batch are not used in that epoch.
6. **Number of steps for the run** = epochs x micro-batches per epoch / GRAD_ACCUM, rounded down.
7. **Learning rate schedule:** warmup for 3% of the steps (the learning rate grows in a straight line up to 2e-5), then
   it falls along a cosine curve down to 2e-6 (one tenth) at the last step.

**Today (v1, one chat per row):** 5,000 rows -> 625 micro-batches per epoch -> 2 epochs x 625 / 8 = **156 steps**,
5 warmup steps, **64 chats in every step**.

### 9.2 Why packing changes the number of steps: a tiny example

12 chats of 3 slots each; rows of 10 slots; one step = 4 rows.

**Without packing:** one chat per row (3 real slots, 7 pads) -> 12 rows.

```
step 1: rows 1-4   (chats 1-4)
step 2: rows 5-8   (chats 5-8)
step 3: rows 9-12  (chats 9-12)
-> 3 steps per epoch, 4 chats per step: the weights are changed 3 times per epoch
```

**With packing:** 3 chats per row (9 real slots, 1 pad) -> 4 rows.

```
step 1: rows 1-4   (chats 1-12)
-> 1 step per epoch, 12 chats per step: the weights are changed once per epoch
```

Same data, same chats seen per epoch, but **3 times fewer weight changes**, and each change is made from 3 times more
chats. To keep 4 chats per step with packing, a step would need 4 / 3 = 1.33 rows, which is not a whole number of rows.
So with packing the step has to be counted in chats or tokens, not in rows.

### 9.3 Our numbers (stage-0, the trainer's own formula, 1 GPU, 2 epochs)

From `packing_stats_detail_output.txt`. With packing there are 1,517 rows holding 3.3 chats and 3,145 trained tokens per
row on average:

| setup | B (rows per micro-batch) | rows per step | micro-batches per epoch (rows unused) | GRAD_ACCUM | steps for 2 epochs | warmup steps | chats per step | trained tokens per step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 today (no packing) | 8 | 64 | 625 (0) | 8 | **156** | 5 | 64 | ~61,075 |
| v2 packed, step kept at 64 rows | 8 | 64 | 189 (5) | 8 | **47** | 1 | ~211 | ~201,303 |
| v2 packed | 8 | 24 | 189 (5) | 3 | 126 | 4 | ~79 | ~75,489 |
| v2 packed | 8 | 16 | 189 (5) | 2 | 189 | 6 | ~53 | ~50,326 |
| v2 packed | 4 | 20 | 379 (1) | 5 | 151 | 5 | ~66 | ~62,907 |
| v2 packed | 8 | 20 | - | - | refused by the trainer: 20 is not a multiple of 8 | - | - | - |

**How every number in the table is computed** (the trainer's own steps, `bluscript_zero_ckpt.cpp:943-961`,
`sft/SftDataLoader.h:56`):

```
step 1  micro-batches per epoch = rows / B, rounded DOWN (a part of a micro-batch is not used)
step 2  rows left unused        = rows - (micro-batches x B)
step 3  GRAD_ACCUM              = rows per step / B          (micro-batches added up per weight update)
step 4  steps per epoch         = micro-batches per epoch / GRAD_ACCUM
step 5  steps for the run       = (epochs x micro-batches per epoch) / GRAD_ACCUM, rounded DOWN
step 6  warmup steps            = 3% of the steps, rounded to the nearest whole step (at least 1)
step 7  chats per step          = rows per step x chats per row (packed: 5,000 / 1,517 = 3.3)
```

Worked through for the first three rows of the table:

```
v1 today (no packing): 5,000 rows, B = 8, 64 rows per step
  step 1  5,000 / 8 = 625 micro-batches per epoch          (625 x 8 = 5,000, so 0 rows unused)
  step 3  64 / 8    = 8   -> GRAD_ACCUM 8
  step 4  625 / 8   = 78.1 steps per epoch
  step 5  2 x 625 / 8 = 1,250 / 8 = 156.25 -> 156 steps for 2 epochs
  step 6  3% of 156 = 4.68 -> 5 warmup steps
  step 7  64 rows x 1 chat = 64 chats per step

v2 packed, step kept at 64 rows: 1,517 rows, B = 8, 64 rows per step
  step 1  1,517 / 8 = 189.6 -> 189 micro-batches per epoch
  step 2  1,517 - 189 x 8 = 1,517 - 1,512 = 5 rows unused per epoch (about 16 chats)
  step 3  64 / 8    = 8   -> GRAD_ACCUM 8
  step 4  189 / 8   = 23.6 steps per epoch
  step 5  2 x 189 / 8 = 378 / 8 = 47.25 -> 47 steps for 2 epochs
  step 6  3% of 47 = 1.41 -> 1 warmup step
  step 7  64 rows x 3.3 = about 211 chats per step

v2 packed, B = 4, 20 rows per step:
  step 1  1,517 / 4 = 379.25 -> 379 micro-batches per epoch (1 row unused)
  step 3  20 / 4    = 5   -> GRAD_ACCUM 5
  step 5  2 x 379 / 5 = 758 / 5 = 151.6 -> 151 steps for 2 epochs
  step 6  3% of 151 = 4.53 -> 5 warmup steps
  step 7  20 rows x 3.3 = about 66 chats per step
```

The 16- and 24-row rows of the table follow the same steps with GRAD_ACCUM 2 and 3: 2 x 189 / 2 = 189 steps, and
2 x 189 / 3 = 126 steps.

What the table shows:

- **Keeping 64 rows per step with packing gives 47 steps instead of 156**: 3.3 times fewer weight changes, only 1 warmup
  step instead of 5, and about 211 chats in each step instead of 64.
- **"About 20 packed rows per step" (which would keep about 64 chats per step) is not allowed with B = 8**, because 20 is
  not a multiple of 8. The closest choices:
  - B = 8 with 24 rows per step: 126 steps, about 79 chats per step;
  - B = 8 with 16 rows per step: 189 steps, about 53 chats per step;
  - B = 4 with 20 rows per step: 151 steps, about 66 chats per step. Smaller micro-batches may use the GPU less well;
    not measured;
  - a token budget per step: the loader would fill each step up to a target number of trained tokens. This is a new
    loader feature, not built.
- The loader leaves the last rows that do not fill a whole micro-batch unused in each epoch: 5 packed rows (about 16
  chats) with B = 8, 1 row with B = 4. Which rows those are depends on the row order (the open "leftover rows" decision
  of v2 Step 7, SFT work doc §40.15).
- Each step's loss is still divided by that step's own trained-token count, so inside a step every token counts the
  same; with more tokens per step, the gradient of each step is an average over more tokens.

### 9.4 What the blog reported

From the blog's section on sequence packing (it cites Hugging Face's SmolLM3 training playbook): "Despite yielding up to a
33x tokens/batch/optimization step, for a fixed token budget, packing alters training dynamics since the more data means
fewer gradient updates. This especially hurts small datasets where each sample matters more. An effective batch size of
128 hurt evals like IFEval by up to 10%; for effective batch sizes larger than 32, there was an average drop in
performance (for SmolLM3 and the dataset). But for large datasets, packing is almost always beneficial."

- **"Effective batch size"** usually means the rows (sequences) in one optimizer step: micro-batch x accumulation x
  number of GPUs. The blog does not define it further. In our terms, v2 at 64 rows per step is an effective batch of 64
  packed rows, about 211 chats.
- **IFEval** is a benchmark that checks whether the model follows format instructions such as "answer in exactly 3
  bullet points"; a program checks the answer. It needs the model to generate text, which BluTrain cannot do yet.

### 9.5 What we know and what we do not know yet

**Known (from the numbers above):**

- With packing and the same 64 rows per step, the model's weights change 3.3 times less often over the same data.
- Each change is made from more chats and more tokens, so it is an average over more examples.
- The warmup shrinks to 1 step, because it is 3% of a smaller number of steps.

**Not known for us:**

- Whether fewer, larger steps lower the quality of our model on our data. The blog's numbers are for SmolLM3 and its
  dataset. Measuring this for us is what the ablation in 9.6 is for.
- Whether the learning rate should change when the step holds more chats.

**The user's point about bigger datasets.** With many chats, the number of steps stays large even with packing. For
example, all of UltraChat's training split (207,865 chats), packed at the same 3.3 chats per row, would be about 207,865 / 3.3 =
63,000 rows, and at 64 rows per step about 63,000 / 64 = 985 steps per epoch. (This is an estimate from the stage-0 average, not a packing
run.)

### 9.6 The plan (before the first packed SFT run, v2 Step 7)

1. **Research.** Read how others set the step size when they pack, in their papers and source code:
   - TRL / SmolLM3: is the "effective batch" counted in packed rows, and which value did they pick?
   - Tülu 3, OLMo, Hermes-4: do they count steps in rows, chats or tokens?
   - How the learning rate is changed when the batch size changes.
2. **Ablation on stage-0.** The same base model (step-8000 checkpoint), the same data, the same 2 epochs and the same
   learning rate schedule. Only the step size changes:

   Updated 2026-10-14 with measured numbers (`packing_step_options.py` -> `packing_step_options_output.txt`):

   | run | rows | how a step is built | chats per step | rows per step | steps for 2 epochs | real tokens | trainer change needed |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | 1 | v1, one chat per row (padding) | 64 rows | 64 | 64 | 156 | 30.2% | none (today) |
   | 2a | packed | take 64 chats, pack only those into rows | exactly 64 | 17-23 (mean 20.2) | 156 | 95.9% | varying rows per step |
   | 2b | packed | pack all chats once, then fill each step with rows until it holds today's average trained tokens (61,075) | 51-82 (mean 65.9) | 18-22 (mean 20.0) | about 150 (75 per epoch) | 99.6% | varying rows per step (a token budget) |
   | 2c | packed | B = 4, 20 rows per step | about 66 | 20 | 151 | 99.6% | none |
   | 3 | packed | 64 rows | about 211 | 64 | 47 | 99.6% | none |
   | 4 | packed | 64 rows, lower learning rate (the blog's advice; for example half, 1e-5) | about 211 | 64 | 47 | 99.6% | none |

   - Runs 2a, 2b and 2c are three ways to build "about 64 chats per step". 2c needs no trainer change; 2a and 2b need
     the trainer to accept a varying number of rows per step (a part micro-batch at the end of a step).
   - 2a counts the step in chats; 2b counts it in tokens. Comparing them shows whether the unit matters.
   - Compare: validation loss on answer tokens (the same 500 validation chats), likelihood benchmarks (HellaSwag-style,
     which BluTrain can run), the base-vs-SFT forgetting check (the same benchmarks on the base model and the SFT
     model), and training time. IFEval needs text generation, which BluTrain does not have yet.
   - Repeat every run once with another seed, and only trust differences bigger than the gap between those two runs.
3. **Precautions in the loader.**
   - Define the step by chats or tokens, not by rows (already the plan).
   - Log the chats, the rows and the trained tokens of every step.
   - Keep the rows per step a whole number of micro-batches (the trainer's check, 9.1), or add a token budget.
   - Decide the leftover rows (9.3).

The same plan is written in the SFT work doc §44.1 and in memory as a future task.

---

## 10. Varlen attention: what it actually is and how it works

### 10.1 The core problem

When you pack several documents into one row, the GPU sees one row of 4,096 tokens. It has no idea that positions
0-1499 are chat A, positions 1500-2299 are chat B, and positions 2300-4095 are part of chat C. To the GPU, it is just
4,096 positions.

A plain causal mask (section 2.3) says: position i can attend to positions 0, 1, ..., i. This is wrong for packed rows,
because it lets chat B attend to chat A (cross-document attention, section 2.5).

### 10.2 What you actually want

Using row 0 of the `wrapped` example in section 8.1, you want the attention to work like this:

- Positions 0-1499 (chat A): position i attends to positions 0, 1, ..., i: causal inside chat A only.
- Positions 1500-2299 (chat B): position j attends to positions 1500, 1501, ..., j: causal inside chat B only.
  **Position 1500 cannot see position 1499.**
- Positions 2300-4095 (the part of chat C): position k attends to positions 2300, 2301, ..., k: causal inside this part
  of chat C only.

Each document has its own causal mask. Documents in the same row are completely invisible to each other. (Section 2.7
draws this for a 10-slot row.)

### 10.3 How varlen attention does it

You give the attention kernel one extra input: `cu_seqlens` ("cumulative sequence lengths", meaning a running total of
the lengths). It is a list that tells the kernel where each document starts and ends.

For the row above:

```
cu_seqlens = [0, 1500, 2300, 4096]
```

This means:

- document 0: positions 0 to 1499 (length 1500);
- document 1: positions 1500 to 2299 (length 800);
- document 2: positions 2300 to 4095 (length 1796).

Each number is the previous one plus the next document's length: 0, then 0 + 1500 = 1500, then 1500 + 800 = 2300, then
2300 + 1796 = 4096.

The kernel reads `cu_seqlens` and, for each query position, computes attention only over key positions that belong to
the same document. It does not build a full 4096 x 4096 table of allowed pairs (that would waste memory). Instead it
uses the document starts and ends directly while it computes the attention scores.

In FlashAttention, this function is called `flash_attn_varlen_func`. The "varlen" part of the name means **variable
length**: each document in the row has a different length (chat A is 1,500, chat B is 800, and so on), and the kernel
handles these different lengths directly.

How TRL gets `cu_seqlens`: TRL does not pass the list itself. It restarts the positions at 0 for every chat, and the
FlashAttention code in `transformers` builds `cu_seqlens` from the places where the positions restart (section 8.4).
TRL also marks the first token of every chat or chunk as "not a target" (`sft_trainer.py:516`), so the last token of the
chat before it is never trained to predict it.

**Varlen attention does not do cross-document attention; it prevents it.**

### 10.4 What happens without varlen

If you pack several documents into one row and use plain causal attention (no varlen), every token in the row can attend
to every earlier token in the row, whatever document it came from. Chat B's tokens learn from chat A's tokens. The model
trains on wrong context and can learn false links between unrelated conversations. Training still runs; nothing
crashes. How much it hurts is in section 2.6.

### 10.5 How varlen works with each of the three packing strategies

**Varlen + BFD.** BFD packs several complete chats into one row. Varlen keeps them apart.

```
row 0: [chat C (3000) | chat B (800) | PAD (296)]
cu_seqlens = [0, 3000, 3800]    (the pad positions belong to no chat and are not trained)
```

- Chat C's tokens attend only to chat C's tokens.
- Chat B's tokens attend only to chat B's tokens.
- Chat B cannot see chat C even though they share the same row.

This is the cleanest setup: no data loss (when every chat fits), no context loss, no cross-document attention.

**Varlen + BFD_split.** Same as BFD, but some of the "documents" in a row are chunks of a longer chat, not complete
chats. Varlen treats each chunk as its own document.

```
row X: [chunk C2 (1404) | chat D (500) | chat B (800)]
cu_seqlens = [0, 1404, 1904, 2704]
```

- Chunk C2's tokens attend only to chunk C2's tokens.
- Chat D's tokens attend only to chat D's tokens.
- Chat B's tokens attend only to chat B's tokens.

(Row X is the general example from section 8.3; for these exact six items Best-Fit puts D in the row with E and A.)

Chunk C2 does **not** attend to chunk C1. Varlen does not help here, and it is not a masking problem: TRL made C1 and C2
separate documents on purpose, so they are kept apart even if they sit in the same row. The context lost between chunks
comes from how `bfd_split` builds its chunks.

**Varlen + wrapped.** `wrapped` joins and cuts without caring about chat boundaries. In TRL it cannot use varlen, because
no chat starts are recorded (sections 8.1 and 8.4). If a different implementation recorded where every chat part starts
in every row, varlen would remove the mixing:

```
row 0: [A1 ... A1500 | B1 ... B800 | C1 ... C1796]
cu_seqlens = [0, 1500, 2300, 4096]
```

- Chat A's tokens attend only inside chat A.
- Chat B's tokens attend only inside chat B.
- The first 1,796 tokens of chat C attend only among themselves.

Even then, two problems stay:

- C's last 1,204 tokens in row 1 still never see C's first 1,796 tokens (the chat is still cut at the row end).
- The wrong target at each chat boundary (A's last token predicting B1) must be blocked separately, as TRL does for
  `bfd` with `sft_trainer.py:516`.

Without varlen, `wrapped` packing mixes chats everywhere. **To stop the mixing, `wrapped` needs varlen plus a record of
the chat starts, and TRL's `wrapped` has neither.**

**`wrapped` WITHOUT varlen (what TRL actually runs).** The model sees one row of 4,096 tokens and applies one causal
triangle. Token B1 attends to all of chat A. Token C1 attends to all of chat A and chat B. The gradients are computed as
if these unrelated conversations were one continuous text. The model still trains (it does not crash), but it learns
from wrong context. (The merged explanation also said that people have observed this "hurts performance on multi-turn
tasks and can introduce hallucinations". It gave no source, and we could not check it; section 2.6 lists what the
sources we can check say.)

### 10.6 The varlen kernel: what is different inside the GPU

**Standard FlashAttention** (`flash_attn_func`): the kernel processes one sequence per batch element. Every position in
that sequence can attend to every earlier position. The causal mask is one triangle.

**Varlen FlashAttention** (`flash_attn_varlen_func`):

- **How the input is laid out:** there is no row dimension. All documents of the batch are joined into one long list of
  tokens, and `cu_seqlens` says where each document starts and ends in that list. (TRL's padding-free mode builds exactly
  this: it joins the batch into one sequence, `sft_trainer.py:495-500`.)
- **How the work is split:** the kernel treats every document as its own attention problem. Each GPU work item handles
  one tile of query positions (64 or 128 positions) of one document and one head, and reads that document's start and
  end from `cu_seqlens`. It does not need to search which document a tile belongs to: the work item is made for one
  document.
- **How the mask works:** key positions outside the document are never loaded, so those tiles are skipped entirely (not
  computed at all, instead of computed and thrown away). The causal mask becomes **several small triangles**, one per
  document:

```
standard causal mask (one document, 4096 tokens):

  ##
  ####
  ######
  ########
  ##########
  ############
  ##############
  ################      <- one big triangle

varlen causal mask (three documents packed: 1500 + 800 + 1796):

  ##                    <- chat A's triangle (1500 x 1500)
  ###
  ####
      ##                <- chat B's triangle (800 x 800)
      ###
          ##            <- chat C's triangle (1796 x 1796)
          ###
          ####          everything else: blocked (attention weight 0), never computed
```

- **Cost:** packed varlen attention costs about the same as running each document separately, but with the speed of
  doing them all in one kernel launch.
- **Not exactly zero waste:** tiles are 64 or 128 positions wide, so the last tile of each document is partly empty,
  and a document shorter than a tile still uses a whole tile. With many very short chats this waste grows.

**Ours.** Both of our packed attention designs block cross-document attention:

- **Design A** (the varlen form): the start of every chat in each row, inside our forward v62 and backward kernels.
- **Design B** (a block mask): each query sees key j only if j is inside its own chat.
- Our rows stay fixed at 4096. In design B a tile can hold the end of one chat and the start of the next; the masked
  part of such a tile is computed and thrown away (SFT work doc §39.8: "part of a key tile is wasted at each chat start").
- Positions restart at 0 for every chat.
- Each chat's inputs and labels are built before packing (labels = the inputs moved by one token), so the last target of
  a chat is its own end token, never the first token of the next chat.

---

## 11. Sources

Every source below was opened and checked on 2026-10-14. Code links point to the exact version (commit) we read, and
the line numbers match that version, so they keep working even if the projects change later.

### 11.1 Papers, blog and web pages

| what it supports | source (link) |
| --- | --- |
| the blog a colleague shared: sequence packing, First-Fit vs Best-Fit, the SmolLM3 effective-batch result | [frontier_training, section "sequence packing"](https://djdumpling.github.io/2026/01/31/frontier_training.html#sequence-packing) |
| the blog: the learning-rate advice for packing | [frontier_training, section "learning rate and epochs"](https://djdumpling.github.io/2026/01/31/frontier_training.html#learning-rate-and-epochs) |
| the blog: document masking, the SmolLM3 results, small models sometimes skipping it | [frontier_training, section "document masking"](https://djdumpling.github.io/2026/01/31/frontier_training.html#document-masking) |
| Hugging Face's SmolLM3 training playbook (the blog's source for the SmolLM3 numbers) | [smol-training-playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) |
| GPT-3: packed documents are not masked; an end token separates them (Appendix B, "Details of Model Training") | [Brown et al., "Language Models are Few-Shot Learners", arXiv 2005.14165](https://arxiv.org/abs/2005.14165) |
| Llama 3: a mask between documents; limited impact in standard pretraining, important in long continued pretraining (section 3.2) | ["The Llama 3 Herd of Models", arXiv 2407.21783](https://arxiv.org/abs/2407.21783) |
| the idea behind `bfd_split`: Best-Fit-Decreasing on document chunks, fewer cut documents | ["Fewer Truncations Improve Language Modeling", arXiv 2404.10830](https://arxiv.org/abs/2404.10830) |
| TRL's docs: breaking sequence continuity hurts performance (cited for `wrapped`) | [Qwen3-Coder-Next technical report, arXiv 2603.00729](https://huggingface.co/papers/2603.00729) |
| First-Fit-Decreasing never uses more than 11/9 x best + 6/9 rows, and this is tight (Dósa 2007; full proof Dósa, Li, Han and Tuza, Theoretical Computer Science 510, 2013) | [Wikipedia: First-fit-decreasing bin packing](https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing) |
| Best-Fit-Decreasing has the same 11/9 ratio (Johnson's 1973 thesis) | [Wikipedia: Bin packing problem](https://en.wikipedia.org/wiki/Bin_packing_problem) |
| the "L2" lower bound used in example 15 | Martello and Toth, "Knapsack Problems: Algorithms and Computer Implementations" (book, 1990) |

### 11.2 Source code of other projects (exact versions)

| what it supports | source (link to the exact lines) |
| --- | --- |
| TRL Best-Fit-Decreasing, `bfd` / `bfd_split` | [`trl/data_utils.py:739-826`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L739-L826) (empty chats removed [`:750`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L750), cut the end [`:754-755`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L754-L755), chunks [`:756-769`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L756-L769), each chunk's length [`:774-775`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L774-L775)) |
| TRL `wrapped` | [`trl/data_utils.py:829-843`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L829-L843) |
| TRL strategy descriptions | [`trl/data_utils.py:860-869`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L860-L869); [`docs/source/reducing_memory_usage.md:81-88`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/docs/source/reducing_memory_usage.md?plain=1#L81-L88) |
| TRL packs per `map` batch (1,000 chats) | [`trl/data_utils.py:910-925`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/data_utils.py#L910-L925); the `datasets` library 5.1.0, `Dataset.map(batch_size=1000)` |
| `datasets` cuts the data into one part per worker before grouping | `datasets` 5.1.0 (installed package), `datasets/arrow_dataset.py:3479, 3546-3553` |
| TRL default strategy `bfd`; shuffle off by default | [`trl/trainer/sft_config.py:230-231`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_config.py#L230-L231); [`:219-221`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_config.py#L219-L221) |
| TRL truncation settings | [`trl/trainer/sft_config.py:203-216`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_config.py#L203-L216) |
| TRL joins a batch into one sequence (padding-free) | [`trl/trainer/sft_trainer.py:495-500`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L495-L500) |
| TRL "not a target" at each chat start | [`trl/trainer/sft_trainer.py:516`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L516) |
| TRL padding-free on for `bfd` / `bfd_split` only; warning for `wrapped` | [`trl/trainer/sft_trainer.py:1180-1192`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L1180-L1192) |
| TRL warning for packing without FlashAttention ("cross-contamination between samples") | [`trl/trainer/sft_trainer.py:1247-1255`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L1247-L1255) |
| TRL's SFT trainer passes only `num_proc` and a label to `map` | [`trl/trainer/sft_trainer.py:1475-1477`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L1475-L1477), [`:1681-1692`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L1681-L1692) |
| TRL shuffle before packing | [`trl/trainer/sft_trainer.py:1686-1689`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/sft_trainer.py#L1686-L1689) |
| TRL DPO: packed (padding-free) mode currently disabled | [`trl/trainer/dpo_trainer.py:708-713`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/trainer/dpo_trainer.py#L708-L713) |
| transformers: restarting positions -> `cu_seqlens` (FlashAttention); packed-batch check | [`modeling_flash_attention_utils.py:480-515`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/modeling_flash_attention_utils.py#L480-L515), [`:558-571`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/modeling_flash_attention_utils.py#L558-L571) |
| transformers: same-chat mask for SDPA / eager from the positions | [`masking_utils.py:720-727`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/masking_utils.py#L720-L727), [`:734-763`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/masking_utils.py#L734-L763), [`:857-865`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/masking_utils.py#L857-L865), [`:976-979`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/masking_utils.py#L976-L979); Llama passes the positions: [`models/llama/modeling_llama.py:380-386`](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/models/llama/modeling_llama.py#L380-L386) |
| torchtitan drops over-long chats | [`torchtitan/hf_datasets/text_datasets.py:145-147, 210-217`](https://github.com/pytorch/torchtitan/blob/1e4d33de4b02f6a6f2ea8d58e65f4d3ce33f1df8/torchtitan/hf_datasets/text_datasets.py#L145-L217) |
| torchtitan packing: First-Fit, streaming, 8 open rows | [`torchtitan/components/data/packing.py:225-279`](https://github.com/pytorch/torchtitan/blob/1e4d33de4b02f6a6f2ea8d58e65f4d3ce33f1df8/torchtitan/components/data/packing.py#L225-L279) |
| torchtitan block-causal (document) mask | [`torchtitan/models/common/attention.py:467`](https://github.com/pytorch/torchtitan/blob/1e4d33de4b02f6a6f2ea8d58e65f4d3ce33f1df8/torchtitan/models/common/attention.py#L467) |
| Megatron-LM cuts the end of an overflowing pack | [`megatron/training/datasets/sft_dataset.py:163-172`](https://github.com/NVIDIA/Megatron-LM/blob/8b2f660bb89e38b6210575dea7984fd05899adf0/megatron/training/datasets/sft_dataset.py#L163-L172) |
| Megatron-LM packed sequence starts (`cu_seqlens`) | [`megatron/core/packed_seq_params.py:10`](https://github.com/NVIDIA/Megatron-LM/blob/8b2f660bb89e38b6210575dea7984fd05899adf0/megatron/core/packed_seq_params.py#L10) |
| TRL's docs: the paper behind `bfd_split` | [`docs/source/paper_index.md:1294`](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/docs/source/paper_index.md?plain=1#L1294) |

### 11.3 Our own code and records

| what it supports | where |
| --- | --- |
| our drop rules and row-length limit | `BluTrain_sft/sft/prepare_sft_data.py:38-51, 141, 167-168` |
| our SFT step settings (rows per step, epochs, warmup, the multiple-of-B check, learning-rate schedule) | `BluTrain_sft/bluscript_zero_ckpt.cpp:201, 920-966, 801-818` |
| our loader: micro-batches per epoch, unused rows | `BluTrain_sft/sft/SftDataLoader.h:16, 56` |
| our pretraining: end token 50256 between documents; no document mask in our attention kernels | SFT work doc §32.2 and §32 |
| every computed example and study in this document (scripts and their outputs) | [`profiling_reports/q10_packing_sim_2026_10_08/`](https://github.com/GOUTHAM1926/Neural_Networks_exp_1926/tree/main/profiling_reports/q10_packing_sim_2026_10_08) |

---

## 12. Corrections made to the merged explanation

The second explanation (merged on 2026-10-14) was right on the main points: the examples, `cu_seqlens`, and what each
strategy does. Its text is kept above. These statements in it were corrected:

| # | it said | corrected to | why |
| --- | --- | --- | --- |
| 1 | every training row must be exactly 4096 tokens because the GPU processes fixed-size tensors | true for our trainer by design; TRL's padding-free mode has no fixed rows, and varlen kernels take any lengths | `sft_trainer.py:495-500` joins the batch into one sequence |
| 2 | `wrapped` glues chats with "no separators, no markers" | each chat still ends with its own end token; what is missing is any record of where chats start | the chat template adds the end token |
| 3 | `wrapped` makes one stream of the whole dataset; the last piece is padded | one stream per group of 1,000 chats; the last row of each group is shorter and is padded when the batch is built | `data_utils.py:829-843, 910-925` |
| 4 | for `wrapped`, "this is exactly where varlen attention comes in" | TRL's `wrapped` records no chat starts, so TRL never uses varlen with it and warns if forced | `sft_trainer.py:1180, 1188-1192` |
| 5 | `bfd_split` row [C2, D, B, PAD 1392] | kept as a general example; for these exact items Best-Fit gives [C2, B] with 1,892 empty, because D goes to the tighter row | computed, section 7.3 |
| 6 | C2 cannot attend to C1 because they are "in different rows, processed in different forward passes" | rows of one batch share a forward pass; the reason is that each chunk is its own document for the attention, even inside one row | `data_utils.py:756-769` cuts the chunks; `:774-775` records each chunk's own length |
| 7 | "Varlen + wrapped" with `cu_seqlens = [0, 1500, 2300, 4096]`; wrapped "essentially requires varlen to function correctly" | correct only for an implementation that records chat starts; TRL does not; even then chats stay cut at row ends | section 10.5 |
| 8 | without varlen, people observed it "hurts performance on multi-turn tasks and can introduce hallucinations" | kept in section 10.5 but marked unverified; the sources we can check are the HF result (small effect on short tasks, crucial for long context) and the Qwen3-Coder-Next report cited by TRL | no source was given for the claim |
| 9 | without varlen "the gradients are wrong" / "garbage attention patterns" | the model trains on wrong context; how much it hurts depends on the task (point 8) | same |
| 10 | the varlen kernel "processes one packed row per batch element" and "looks up which document each tile belongs to" | there is no row dimension: all documents are one list, each document is its own work item, and it reads its start and end from `cu_seqlens` | how `flash_attn_varlen_func` takes its input |
| 11 | varlen has "no wasted compute" | close, but the last tile of each document is partly empty; short chats waste more | tiles are 64-128 positions |
| 12 | for `wrapped`, long chats are "cut at row boundaries" | any chat can be cut at a row end, not only long ones (chat C of 3,000 in example 17) | computed, section 8.1 |
| 13 | for `bfd`, varlen keeps the packed chats apart (no condition) | FlashAttention uses varlen; SDPA and eager get a same-chat mask from `transformers` for models that pass the positions (Llama does); TRL still warns for non-FlashAttention (section 8.4). (This row first said "only FlashAttention", which was wrong.) | `masking_utils.py:734-763, 857-865, 976-979`; `sft_trainer.py:1247-1255` |

Plain words used instead of the second explanation's harder words: "contiguous" -> "one unbroken block, all its tokens
side by side"; "interleaved" -> "mixed into each other"; "spurious correlations" -> "false links"; "degenerate case" ->
"`wrapped` WITHOUT varlen"; "natively" -> "directly"; "inherent" -> "comes from how `bfd_split` builds its chunks";
"leakage" -> "cross-document attention (chats leaking into each other)"; "tradeoff" -> "what you gain and what you
lose"; "no intelligence, no optimisation" -> "no decision-making: no sorting, no fitting".

### 12.2 Corrections to this document's own earlier text (review of 2026-10-14)

Two review agents re-computed every number and re-opened every cited source line. They found these mistakes in what I
had written. Each one is fixed in place:

| # | it said | corrected to | how it was found |
| --- | --- | --- | --- |
| 1 | 1,511 rows can never be reached without cutting chats (example 15; also section 3, example 8 and section 5) | 1,511 is reachable: a subset-sum method packs all 5,000 chats into 1,511 rows (1,502 completely full); First-Fit and Best-Fit stop at 1,517 | a reviewer built the packing; I re-ran it and checked it with a separate checker (example 15, step 6) |
| 2 | correction 13 said TRL's chat boundaries work only with FlashAttention | `transformers` also builds a same-chat mask for SDPA and eager from the positions | read in `masking_utils.py` (section 8.4) |
| 3 | 6,188,911 / 4,096 = 1,510.97 | 1,510.96 | arithmetic |
| 4 | "6.04 rows' worth, exactly the 6 extra rows" | 24,721 = 6 x 4,096 + 145 | arithmetic |
| 5 | cutting the end of the 4,500-token chat loses "the last 500 answer tokens and the end token" | the last 499 answer tokens and the end token (500 in all) | arithmetic |
| 6 | cutting the start keeps "the end of the prompt + the whole answer" | it keeps the answer and the end token and none of the prompt | arithmetic |
| 7 | "69.8% of the GPU's work goes to pads" | 69.8% of the slots are pads; they take 72.3% of the work | `pack_sim.py`'s work model |
| 8 | "our learning rate starts at 2e-5" | it rises to 2e-5 during warmup, then falls to 2e-6 | `bluscript_zero_ckpt.cpp:801-818` |
| 9 | "mean 1,238 tokens" | 1,238 slots (about 1,239 tokens) | section 0's definition |
| 10 | example 16: "TRL also shuffles before packing" | only when `shuffle_dataset` is on, and the default is off; no-shuffle numbers added | `sft_config.py:219-221` |
| 11 | example 17: "or (for A) the first row that fits is also the tightest" | at every step at most one row has room | the example's own steps |
| 12 | some citations: TRL's default, `map` settings, our step settings, §40 for a §39.8 quote, the §34.2 item, `reducing_memory_usage.md` lines, `data_utils.py:774-775`, `prepare_sft_data.py:167-168` | fixed in place and in section 11 | source check |

---

## 13. First-Fit or Best-Fit: which one is better? (research and measurements, 2026-10-14)

The question: both rules sort the chats longest first. First-Fit takes the first row that has room and does not search
any further. Best-Fit searches all rows for the tightest one. Is Best-Fit's search just wasted work, or does it pack
better?

### 13.1 The idea behind each rule

- **First-Fit:** go down the rows from the top; stop at the first row that has enough room.
- **Best-Fit:** look at every row that has enough room; pick the one that will be left with the fewest empty slots.
  The idea: use up small gaps with chats that fit them tightly, and keep rows with big gaps open for later chats that
  need a big gap. Example 3 shows it: Best-Fit keeps a 300-slot gap open where First-Fit splits the space into gaps of
  150 and 200.

### 13.2 What is proven (theory)

- **First-Fit-Decreasing** never uses more than 11/9 x (the smallest possible number of rows) + 6/9 rows, on any input.
  This bound is tight: there are inputs where it is reached. Johnson proved the 11/9 ratio in his 1973 thesis (with a
  larger extra constant); Dósa proved the 6/9 in 2007, with a full proof in Dósa, Li, Han and Tuza, Theoretical
  Computer Science 510 (2013).
- **Best-Fit-Decreasing** has the same worst-case ratio, 11/9 (Wikipedia's "Bin packing problem" page states at most
  11/9 x best + 1 for both rules). I could not confirm a smaller extra constant for Best-Fit.
- What 11/9 means: on the worst possible input, where the best packing needs 900 rows, either rule may need up to about
  1,100. Our data is far from that worst case: 1,517 rows against the minimum of 1,511 (0.4% more).
- So in theory neither rule is proven better, and neither always wins (examples 4 and 5).

### 13.3 What we measured

Two studies, both on the laptop CPU, both checking that their fast versions give exactly the same rows as the simple
ones:

- `ffd_vs_bfd_study.py` -> `ffd_vs_bfd_study_output.txt` (1,673 tests, quick timing);
- `ffd_vs_bfd_full_study.py` -> `ffd_vs_bfd_full_study_output.txt` (written by a review agent, run once in 406 s:
  about 25,000 random tests plus every possible small input, and careful timing up to 1,000,000 chats).

**Number of rows, our kind of data** (no difference anywhere):

| test | tests | First-Fit fewer rows | same | Best-Fit fewer rows |
| --- | --- | --- | --- | --- |
| our stage-0 data (5,000 chats) | 1 | 0 | 1 (1,517 rows each) | 0 |
| random sets drawn from our chat lengths, 1,000 / 5,000 / 20,000 chats (full study) | 5,000 + 1,000 + 300 | 0 | all | 0 |
| TRL-style groups of 1,000 chats, 1,000 different shuffles (full study) | 1,000 | 0 | 1,000 | 0 |

**Number of rows, where differences do appear** (all from the full study):

| test | tests | First-Fit fewer rows | same | Best-Fit fewer rows |
| --- | --- | --- | --- | --- |
| random sets of 100 chats drawn from our lengths | 5,000 | 0 | 4,999 | 1 |
| TRL-style groups of 100 chats, 1,000 shuffles | 1,000 | 0 | 989 | 11 |
| made-up lengths spread evenly over 1-4,096, 1,000 chats | 2,000 | 0 | 1,993 | 7 |
| the same, 10,000 chats | 200 | 1 | 199 | 0 |
| made-up lengths spread evenly over 1-2,048, 1,000 chats | 2,000 | 0 | 1,994 | 6 |
| EVERY possible input with rows of 20 slots and 2 to 8 chats of 1-20 slots | 3,108,084 | 19 (0.001%) | 3,107,769 (99.990%) | 296 (0.010%) |

Seven other made-up length mixes (2,000 + 200 sets each: lengths over 1-1,024; over 1,024-2,048; 50% very short + 50%
very long; 80% short + 20% long; and two mixes of lengths just over 1/2, 1/3 and 1/4 of a row) showed no difference at
all.

- When the two rules differ, it is by exactly one row, and **Best-Fit is the one with fewer rows about 15 times more
  often** than First-Fit (296 against 19 on every small input). Examples 4 and 5 are two such cases.
- On data like ours (our real chats, or random sets of 1,000 or more chats drawn from them), they never differed.
- The rows themselves do differ a little. On stage-0, 1,395 of the 1,517 rows are the same in both layouts. Best-Fit has
  more completely full rows (1,060 against 1,039), but also more rows holding a single chat (4 against 1). The
  attention work per row (the sum of the squared chat lengths in the row) has the same average (6.23 million) and the
  same spread.
- TRL's own `_pack_bfd` function, run on all 5,000 chats, gives the same 1,517 rows.

### 13.4 Is Best-Fit's search wasted work? (speed)

Careful timing from the full study (one CPU core, Python, best of several runs, the shared sorting step timed
separately):

| chats | simple First-Fit | simple Best-Fit | fast First-Fit | fast Best-Fit | TRL's Best-Fit code |
| --- | --- | --- | --- | --- | --- |
| 5,000 | 0.060 s | 0.077 s | 0.003 s | 0.006 s | 0.009 s |
| 50,000 | 6.4 s | 8.0 s | 0.030 s | 0.046 s | 0.075 s |
| 200,000 | 103 s | 129 s | 0.13 s | 0.18 s | 0.29 s |
| 1,000,000 | not run (about 43 minutes, estimated) | not run (about 54 minutes, estimated) | 0.67 s | 0.87 s | 1.44 s |

- **Simple versions** (look at the rows one by one): First-Fit stops at the first row with room; Best-Fit looks at every
  open row. First-Fit looks at only 12% fewer rows (on stage-0: 915.7 against 1,034.7 per chat), because the first rows
  fill up early and First-Fit walks past almost all of them anyway. It is 1.27 times faster. Both grow with
  (chats x rows), so both become far too slow for large datasets: 200,000 chats take about 2 minutes.
- **Fast versions** (a search tree): First-Fit uses a tree over the rows that finds the first row with enough room;
  Best-Fit uses a tree over the possible "room left" values (0 to 4,096) that finds the tightest row. First-Fit is 1.3
  to 1.9 times faster, but both pack 1,000,000 chats in under 1 second.
- So Best-Fit's extra search is real, but small: under 0.3 seconds per million chats when written well. Packing runs
  once, before training.

(The quick timing in `ffd_vs_bfd_study_output.txt` had simple Best-Fit slightly faster than simple First-Fit. That was
measurement noise from single runs; the careful timing above is the one to use.)

### 13.5 What other tools use

| tool | rule | where |
| --- | --- | --- |
| TRL | Best-Fit-Decreasing (default `bfd`), with a search tree | `trl/data_utils.py:739-826`; `sft_config.py:230-231` |
| torchtitan | First-Fit without sorting: chats arrive one by one and only 8 rows are open at a time (`num_packing_bins = 8`), through Google's `grain` library | `torchtitan/components/data/packing.py:225-279` |
| Hermes 4 | First-Fit-Decreasing ("more than 99.9% batch efficiency", per the blog) | the blog; not checked in Hermes 4's own report |
| "Fewer Truncations Improve Language Modeling" (arXiv 2404.10830) | Best-Fit-Decreasing on document chunks | TRL `docs/source/paper_index.md:1294` |
| ours (simulation and tests so far) | First-Fit-Decreasing | `pack_sim.py`, attention test data D10 |

### 13.6 Verdict

- **How many rows: practically a tie, with a very small edge for Best-Fit.** On our data the two rules always give the
  same number of rows. In the rare inputs where they differ, Best-Fit is about 15 times more often the one with one row
  fewer. Both are 6 rows (0.4%) above the true minimum on stage-0.
- **Speed: First-Fit is a little faster, but it does not matter.** Written with a search tree, First-Fit is 1.3-1.9
  times faster, and both pack 1,000,000 chats in under 1 second. Written the simple way, both are too slow for large
  datasets.
- **Your idea was partly right.** First-Fit is simpler and does a little less work. But Best-Fit's search costs well
  under a second, so it is not a waste of time, and it is not true that after sorting both rules always leave the same
  gaps (examples 3, 4 and 5, and the cases above).
- **For us, either rule is fine.** On our data they give the same rows. Two reasonable choices:
  - **Best-Fit-Decreasing with a search tree** for the real packing script (Step 7): it is very slightly better at
    packing, costs under a second, and is what TRL uses.
  - **First-Fit-Decreasing with a search tree:** a little simpler, and it is what our simulation and tests use today.
- **Whichever we pick, write it with a search tree.** The simple loop takes minutes at 200,000 chats.
- **If we ever want the last 0.4%:** the subset-sum method of example 15 reaches the true minimum. It is slower, and on
  our data it saves 6 rows of 1,517, so it is not needed now.
