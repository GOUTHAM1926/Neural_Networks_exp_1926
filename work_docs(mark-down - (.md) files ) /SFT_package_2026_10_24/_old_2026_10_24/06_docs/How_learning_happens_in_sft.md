# How learning happens in SFT

SFT Q&A question 9: how the model learns only from the answer tokens and not from the prompt, what masking really does, and how gradients and weights change for prompt tokens and answer tokens over one step and over the whole training.

Written 2026-10-08.
- **Where the numbers come from:** a small test model built like ours (4 layers, a tied embedding table with 50,304 rows, RMSNorm, causal attention, SwiGLU), in fp64 on the laptop CPU.
- **The example:** one real chat, tokenized by our own `sft/prepare_sft_data.py` (template `tulu-plain-v1`).
- **Run records:** `profiling_reports/q9_gradient_flow_2026_10_08/` (`q9_flow.py`, `q9_sideeffect.py`, `q9_results.json`).

---

## 0. The short answer

**Question:** "In the loss we mask the prompt positions, so they get no loss. Then their gradients are zero, so do the prompt tokens' weights stay random for the whole training?"

**No.** The prompt tokens' weights do get updated, and they were never random:

1. **They were never random.** SFT starts from Navin's pretrained weights (step 8000). The "France" row of the embedding table is already a trained vector before SFT starts.
2. **Masking removes only the prompt token's *prediction job*, not the token itself.** "France" is still an input. The answer reads it through attention, so gradient comes back to it.
3. **Most weights are shared by all positions** (Q, K, V, O, MLP, norms). One update per step changes them for every token.

---

## 1. A token has two jobs. Masking removes only one

The example chat: the user asks "What is the capital of France?" and the answer is "Paris." It becomes 24 tokens. Inputs = the tokens without the last one, labels = the tokens shifted by one, then padding up to T = 28.

```
position:     10      11        12     ...   20      21       22     23
input:        " of"   " France" "?"    ...   "\n"    "Paris"  "."    eot (pad)
label:        65535   65535     65535  ...   "Paris" "."      eot    65535
```

- **23 real input positions:** positions 0–19 are the prompt, and 20–22 are trained.
- **5 padding positions:** 23–27.
- **N = 3.** Only 3 positions are trained: the one *before* "Paris", plus "Paris" and ".".

"France" has two jobs:

```
job 1  PREDICTOR : at position 11, guess the next token ("?")   → label 65535 → MASKED
job 2  CONTEXT   : be read by later positions (20, 21, 22)       → still active
```

Masking means "don't score position 11's guess". It does **not** mean "ignore France". When position 20 guesses "Paris", it looks back at "France". That look is how gradient reaches "France".

---

## 2. Forward pass: only attention mixes positions

```
token ids (28 positions)
   │
[embedding lookup]           row by row: id → vector
   │
   ┌──────────── one layer, repeated L times ───────────────────┐
   │ RMSNorm                      row by row                    │
   │ Q, K, V = x·Wq, x·Wk, x·Wv   row by row                    │
   │ ATTENTION    ← the ONLY op that mixes positions:           │
   │                row t = weighted sum of V rows 0..t         │
   │ O = attn·Wo                  row by row                    │
   │ + residual, RMSNorm, MLP, + residual   row by row          │
   └────────────────────────────────────────────────────────────┘
   │
[final RMSNorm]              row by row
[logits = h · Embᵀ]          row by row, 50,304 scores each (same table as the lookup: tied)
[loss]                       only rows with label ≠ 65535 → rows 20, 21, 22
```

Information moves between positions only in attention, both forward and backward. Everything below follows from that.

---

## 3. A 2-position toy: why the prompt still gets gradient

Two positions: **p** (prompt, masked) and **a** (answer, scored). Say attention at a puts 0.4 on p and 0.6 on itself:

```
out_p = x_p                      (p can only see itself)
out_a = 0.4·x_p + 0.6·x_a        (a reads p and itself)
loss  = L(out_a)                 (out_p is masked, not in the loss)

backward:
∂loss/∂out_p = 0                 ← masking: p's own guess gives nothing
∂loss/∂out_a = g
∂loss/∂x_a   = 0.6·g
∂loss/∂x_p   = 0.4·g             ← NOT zero: it comes through a's read of p
```

So the "France" vector still gets 40% of the gradient here, because the answer used it. Real attention also has Q, K and V matrices, but the logic is the same.

---

## 4. Backward, layer by layer (measured)

✓ = non-zero, 0 = exactly zero (fp64, so a zero is a true zero).

```
                                   prompt (0–19)   answer (20–22)   pad (23–27)
loss → logits                           0               ✓               0
layer 4 output (top)                    0               ✓               0
   layer 4 attention: answer rows read K,V of every earlier row ─┐
layer 3 output                          ✓  ◄────────────────────┘ ✓      0
layer 2 output                          ✓               ✓               0
layer 1 output                          ✓               ✓               0
embedding output (input vectors)        ✓               ✓               0
```

- **Top:** prompt rows are 0, because their own guesses aren't scored.
- **One layer down:** prompt rows become non-zero, because the answer read their K and V.
- **Below that:** every op has gradient at the prompt rows.
- **Pad:** 0 everywhere. Pads come *after* the answer, and attention only looks back, so nobody reads them.

**The gradient at the logits (measured):**
- Rows 20–22 have a non-zero value in all 50,304 columns.
- On the right token the value is −0.3333. The formula (p − 1) ÷ N = (≈0 − 1) ÷ 3 gives the same.
- The other 50,303 values add up to +0.3333, because a softmax gradient row always sums to 0.
- Rows 0–19 and 23–27 are all exactly 0.

---

## 5. Which weights change, and from where

A shared weight's gradient is the sum over positions of (input × gradient) at each position. A position with 0 gradient adds nothing. This counts the positions that add to each weight (measured):

```
              last layer (4)               layers 1–3
weight     prompt  answer  pad        prompt  answer  pad
Wq            0       3     0           19*     3      0
Wk           20       3     0           20      3      0
Wv           20       3     0           20      3      0
Wo            0       3     0           20      3      0
MLP           0       3     0           20      3      0
* position 0 can only look at itself, so its query never gets gradient
```

- **Last layer:** the prompt feeds only Wk and Wv, the "be read" part.
- **Lower layers:** the prompt feeds everything, because what it computes there is read again higher up.

**Embedding table:** one table of 50,304 rows, used twice because it's tied.

```
                          input side (lookup)              output side (logits)
"France", "capital", …    ✓ updated (the answer read them) ✓ small push down
"Paris", "."              ✓ updated (inputs at 21, 22)      ✓ pushed UP where they are the
                                                              right answer
tokens not in this chat   ✗ not touched                     ✓ small push down (all rows,
                                                              every step)
eot / pad (id 50256)      ✗ 0 (only used as padding)        ✓ pushed UP as the end-token target
```

Measured:
- the input side touched **16 rows**, the 16 different tokens of the prompt and answer;
- the output side touched **all 50,304 rows**;
- the pad row's input-side gradient was **exactly 0**;
- the two sides add up to PyTorch's own gradient exactly (largest difference 0.0).

---

## 6. One full step, end to end

```
FORWARD   all 28 positions computed ──► loss from rows 20, 21, 22 only (÷ N = 3)
BACKWARD  rows 20–22 at the top
            → layer 4 attention sends gradient to prompt rows 0–19
            → layers 3, 2, 1 carry gradient at rows 0–22
            → embedding: input rows of the 16 chat tokens
                         + output side: all 50,304 rows
          pad rows 23–27: 0 the whole way
UPDATE    every shared matrix changes once → affects every token in every later step
```

**Two checks (measured):**

| Change | Result | What it shows |
|---|---|---|
| The 5 padding tokens replaced by random tokens (" Boss", "ymph", " interfered", " Aub", " posture") | the largest change in any weight gradient was **0.0**, bit-for-bit the same | padding has no effect at all, with no attention mask |
| One prompt word changed: France → Japan (answer still "Paris.") | the loss moved from 10.9265 to 10.9011, and gradients changed (largest change 0.379) | the answer's gradient depends on the prompt |

---

## 7. Over the whole training

- **Prompt-token rows** (like "France") change every time that token appears in a prompt. They are tuned to be **read well**, so they carry what the answer needs. They are never trained to be **predicted**.
- **Answer-token rows** get both: tuned to be read, and pushed up as targets in the right context.
- **Layer weights** change every step, from every non-pad position.
- **What the model learns:** "answer, given the prompt" gets better. "Predict the next prompt token" is not trained.

**The 60-step test.** The same small model was trained on this one chat (AdamW, LR 3e-3), once with our SFT labels and once with every position trained (pretraining style). −log p of the true next token, mean over positions:

| | answer positions | prompt positions |
|---|---|---|
| SFT labels | 10.93 → **0.065** | 10.81 → 9.00 |
| every position trained | 10.93 → 0.64 | 10.81 → 0.47 |

The prompt loss fell from 10.8 to 9.0 even though it was never trained. To find out why, at the prompt positions after 60 SFT steps, I measured −log p of:

| token | −log p | meaning |
|---|---|---|
| the true next prompt token | 9.00 | the same as below: the model does **not** know which prompt token comes next |
| any other token of this chat | 9.08 | |
| tokens that never appear in the chat | 15.56 | pushed down from 10.8 |
| the answer tokens (Paris . eot) | 2.11 | the model now wants to say the answer everywhere |

So the drop is not learning the prompt. The tokens that never appear were pushed far down, so all 17 chat tokens look a bit more likely, all by about the same amount. The model also learned to say "Paris ." almost everywhere, which is the one-example overfit you'd expect.

**In our real model,** the skill of predicting prompt-like text comes from pretraining. SFT doesn't train it, so it may drift a little. That's why we compare base vs SFT on general tests (MMLU, HellaSwag) later.

---

## 8. Where this happens in BluTrain

| step | where | what it does |
|---|---|---|
| labels | `sft/prepare_sft_data.py`, `build_row` | label = next token on answer and end-token positions, 65535 everywhere else |
| loss, forward | `LossKernels.cu:215`, `:350` | a row with label 65535 gets loss 0 and skips the 50,304-wide softmax |
| count N | `LossKernels.cu:110` | rows with 65535 are not counted |
| loss, backward | `LossKernels.cu:692`, `:777` | a row with label 65535 gets an all-zero gradient row |
| attention | causal GQA, unchanged | answer rows read prompt rows; nothing reads the padding |
| denominator | `bluscript_zero_ckpt.cpp`, SFT mode | N = trained tokens of the whole step, across all GPUs |

Line numbers are from Tensor-Implementations `d96b991`.

**What this measurement can and cannot show:** the test model is small and starts from random weights, so the *sizes* of its numbers mean nothing for our 1.5B model. *Which* values are zero and which are not comes from the structure (masked labels, causal attention, a tied table), and that structure is the same in our model.

---

## 9. Further reading (trusted sources)

**Start with these:**
- Nathan Lambert (Ai2), *RLHF Book*, chapter "Instruction Fine-Tuning".
  - It says SFT "uses the same autoregressive loss function used in pretraining… but with substantially different data and masking".
  - It also says the prompt is masked "so the model isn't learning to accurately predict user queries – just responses".
  - https://rlhfbook.com/c/09-instruction-tuning
- Hugging Face TRL docs, `SFTTrainer`: token-level cross-entropy with a one-token shift, label −100 for ignored positions, completion-only loss by default for prompt/answer data, and `assistant_only_loss`. https://huggingface.co/docs/trl/main/en/sft_trainer
- Llama 2 paper (Meta), §3.1: "we zero-out the loss on tokens from the user prompt, so as a result, we backpropagate only on answer tokens". https://arxiv.org/abs/2307.09288
- URIAL (Ai2 + UW, ICLR 2024): base and aligned models pick the same token at most positions, and differ mainly on style tokens and safety disclaimers (about 5–8% of positions). https://arxiv.org/abs/2312.01552

**Official docs:**
- PyTorch `CrossEntropyLoss`: `ignore_index` (default −100) "does not contribute to the input gradient", and the mean is over non-ignored targets. https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- torchtune: with `train_on_input=False`, prompt tokens are replaced with −100. https://docs.pytorch.org/torchtune/0.4/generated/torchtune.datasets.instruct_dataset.html

**Explainers:**
- Cameron Wolfe, "Understanding and Using Supervised Fine-Tuning (SFT) for Language Models" (2023). https://cameronrwolfe.substack.com/p/understanding-and-using-supervised
- Chip Huyen, "RLHF: Reinforcement Learning from Human Feedback" (2023): SFT as behavior cloning. https://huyenchip.com/2023/05/02/rlhf.html
- Andrej Karpathy, "State of GPT" (Microsoft Build 2023): the 4 training stages. https://www.youtube.com/watch?v=bZQun8Y4L2A
- Stanford CS224N, "Prompting, Instruction Finetuning, and RLHF" (slides). https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture10-prompting-rlhf.pdf
- Sebastian Raschka, "LLM Research Insights: Instruction Masking and New LoRA Finetuning Experiments" (2024). https://magazine.sebastianraschka.com/p/llm-research-insights-instruction

**Research on how the learning happens:**
- Ren & Sutherland, "Learning Dynamics of LLM Finetuning" (ICLR 2025): how training on one answer changes predictions on other inputs, including why models reuse phrases from one answer elsewhere. https://arxiv.org/abs/2407.10490
- Ghosh et al., "A Closer Look at the Limitations of Instruction Tuning" (ICML 2024): LoRA SFT mostly learns how to start a response and its style, while full fine-tuning can degrade knowledge. https://arxiv.org/abs/2402.05119
- LIMA (Meta, 2023): the "superficial alignment" idea. https://arxiv.org/abs/2305.11206
- Chu et al., "SFT Memorizes, RL Generalizes" (ICML 2025). https://arxiv.org/abs/2501.17161

**Is masking the prompt always best?**
- Shi et al., "Instruction Tuning With Loss Over Instructions" (NeurIPS 2024): adding loss on the prompt helps when prompts are long and answers short, or when there are few examples. https://arxiv.org/abs/2405.14394
- Huerta-Enochian & Ko, "Instruction Fine-Tuning: Does Prompt Loss Matter?" (EMNLP 2024): a small prompt-loss weight helps with short answers, and there is no effect for medium or long answers. https://arxiv.org/abs/2401.13586
- For us: UltraChat answers are long, so masking the prompt is the right choice.

I read the TRL, PyTorch and torchtune docs, the RLHF Book chapter and the explainer pages on 2026-10-08. For URIAL, Ren & Sutherland, Ghosh et al., Shi et al. and Huerta-Enochian & Ko, only the abstracts were read. The Llama 2, LIMA and Chu et al. quotes were checked earlier for the SFT report.
