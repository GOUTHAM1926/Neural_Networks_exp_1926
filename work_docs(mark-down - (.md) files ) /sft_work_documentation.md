# SFT (Supervised Fine-Tuning) — Work Documentation

**Author:** Goutham Reddy (Gautam_1926)
**Team:** Tensor & Ops
**Started:** 2026-09-17
**Status:** Research / scoping phase. No implementation yet.

**Context:** Assignment changed at a meeting on 2026-09-17. Moved off pretraining kernel work
(BluBLAS / matmul, which is now on hold at 99.80% of cuBLAS) onto **SFT research in
post-training**. This document is the record of everything established on day 1.

---

## Table of contents

1. [What SFT is — and the one thing everyone gets wrong](#1)
2. [The full training pipeline — three stages, not two](#2)
3. [Mid-training — the stage most people do not know about](#3)
4. [What is inside post-training](#4)
5. [Order — SFT first or RL first, and the evidence](#5)
6. [Types inside SFT — four independent axes](#6)
7. [Types inside RL / preference tuning](#7)
8. [The library landscape — what exists and what is dead](#8)
9. [The layering — PyTorch vs HuggingFace vs us (verified from source)](#9)
10. [Why a whole SFT module exists if the loss is unchanged](#10)
11. [How DPO and GRPO differ from SFT, mechanically and in the kernels](#11)
12. [Datasets](#12)
13. [Papers to read](#13)
14. [Hyperparameters](#14)
15. [Efficiency levers — where our background applies](#15)
16. [Evaluation](#16)
17. [BluTrain gap analysis — what we have and what we must build](#17)
18. [Side investigation — the Wave Network paper](#18)
19. [Open decisions](#19)
20. [The three layers — engine vs platform vs recipes](#20)
21. [Framework landscape — verified, and which one we pick](#21)
22. [How three reference implementations do masked CE](#22)
23. [The global denominator — a bug caught before writing code](#23)

---

<a name="1"></a>
## 1. What SFT is — and the one thing everyone gets wrong

**SFT trains a pretrained model on (prompt -> response) pairs so it answers the way we want.**

The loss:
```
L_SFT(θ) = − Σ_t log p_θ( y_t | y_<t )
```

**This is exactly the pretraining loss.** Token-level cross-entropy, next-token prediction.

I first described SFT as "modifying the cross-entropy loss". That is wrong, and the correction
matters: **SFT does not modify the loss at all.** It is the same `F.nll_loss` PyTorch has
always had. What changes is **the labels fed into it** — tokens we do not want counted are set
to an ignore value, and the loss function skips them.

So SFT is a **label-construction** change, not a loss change.

| | pretraining | SFT |
|---|---|---|
| loss function | cross-entropy | **cross-entropy — identical** |
| optimizer | AdamW | AdamW |
| forward / backward | same | same |
| where the label comes from | the raw text itself | a human or a stronger model wrote the answer |
| loss computed on | **every** token | **only the answer tokens** |
| data volume | trillions | thousands to millions |

---

<a name="2"></a>
## 2. The full training pipeline — three stages, not two

```
PRETRAINING   ->   MID-TRAINING   ->   POST-TRAINING
~trillions          ~hundreds of        ~millions of tokens
of tokens           billions            (SFT -> preference/RL)
```

| | PRETRAINING | MID-TRAINING | POST-TRAINING |
|---|---|---|---|
| tokens | trillions | hundreds of billions | millions |
| data | broad web text, quantity | curated, quality-first | human/synthetic demos |
| goal | language + world knowledge | specialise, extend, prepare for RL | behaviour, format, alignment |
| loss | next-token CE | next-token CE (same) | next-token CE (SFT), then preference/RL |

**Pretraining is SELF-supervised, not unsupervised.** Unsupervised means no labels at all
(clustering, PCA). Self-supervised means labels exist but come free from the data — for
"The sky is ___" the label is "blue", taken from the text. There IS a cross-entropy loss
against a real label; no human wrote it.

---

<a name="3"></a>
## 3. Mid-training — the stage most people do not know about

The term reportedly first appeared in an **OpenAI job description in mid-2024**. It is now a
standard stage.

**What it is:** continue pretraining on a **much better data mixture**. Same loss, same
mechanics. The shift is quantity -> quality.

**What actually happens in it:**
1. **Data quality shift** — upsample STEM, code, maths, reasoning; downsample generic web text
2. **Long-context extension** — 4k -> 32k -> 128k -> 256k (Composer 2 went 32K -> 256K here)
3. **Domain / language expansion**
4. **Large-scale synthetic data** — at a scale post-training cannot afford
5. **Capability additions** — e.g. Multi-Token Prediction layers for speculative decoding
6. **Preparing for RL** — mid-training measurably changes how well later RL works

**Why it exists as its own stage:** pretraining is too expensive to redo, post-training is too
small to teach real capability. Mid-training is cheap enough to repeat and big enough to change
what the model knows.

**Related terms:** CPT (Continued Pre-Training) is mostly the same idea. "Annealing" is the
final high-quality phase at the end of pretraining.

---

<a name="4"></a>
## 4. What is inside post-training

Six things, not two:

```
POST-TRAINING
├── 1. SFT                     <- teach format and behaviour
├── 2. Preference tuning / RL  <- teach which answer is better
├── 3. Distillation            <- learn from a stronger teacher
├── 4. Model merging           <- combine specialists, no retraining
├── 5. Safety tuning           <- refusals, harmlessness
└── 6. Quantization / QAT      <- shrink for deployment
```

1 and 2 are the main path. 3-6 support them.

---

<a name="5"></a>
## 5. Order — SFT first or RL first, and the evidence

**Short answer: it depends on whether the base model can already follow the prompt format.**
The flat rule "SFT must always precede RL" that I wrote on day 1 is **refuted by published
ablations**. Corrected below with the actual numbers.

### 5.1 The DeepSeek-R1 stage ablation (official, Table 3 of arXiv 2501.12948)

DeepSeek published every intermediate checkpoint of the pipeline. This is the real evidence:

| Benchmark | R1-Zero (pure RL) | Dev1 (+cold-start SFT) | Dev2 | Dev3 | R1 (final) |
|---|---|---|---|---|---|
| **AIME 2024** | **77.9** | **59.0**  (-18.9) | 74.0 | 78.1 | 79.8 |
| **CNMO 2024** | **88.1** | **58.0**  (-30.1) | 73.9 | 77.3 | 78.8 |
| GPQA Diamond | 75.8 | 66.1  (-9.7) | 70.7 | 71.2 | 71.5 |
| SimpleQA | 30.3 | 17.8  (-12.5) | 28.2 | 24.9 | 30.1 |
| **IF-Eval** | 46.6 | **71.7**  (+25.1) | 72.0 | 78.1 | 83.3 |
| **AlpacaEval 2.0** | 24.7 | **50.1**  (+25.4) | 55.8 | 62.1 | 87.6 |
| ArenaHard | 53.6 | 77.0  (+23.4) | 73.2 | 75.6 | 92.3 |
| LiveCodeBench | 50.0 | 57.5 | 63.5 | 64.6 | 65.9 |
| Codeforces (rating) | 1444 | 1534 | 1687 | 1746 | 2029 |

Their own wording (§3):

> "due to the limited size of the cold-start dataset, **Dev1 exhibits a partial degradation in
> reasoning performance** compared to DeepSeek-R1-Zero, **most notably on the AIME benchmark**."

**What this actually proves — read it carefully:**

1. Cold-start SFT made **reasoning worse**, not better (AIME -18.9, CNMO -30.1).
2. What SFT bought was **usability** (IF-Eval +25.1, AlpacaEval +25.4).
3. **R1-Zero's raw reasoning was already near-final**: 77.9 vs 79.8 AIME. The entire
   multi-stage pipeline added **+1.9 points of reasoning**. Everything else was formatting,
   helpfulness and readability.

So the correct claim is: **RL alone was enough for reasoning. It was not enough for a
shippable product.** That is a different and much weaker claim than "RL alone doesn't work".

The readability fix itself also costs accuracy (§B.6 ablation):

> "ablation experiments in Supplementary B.6 show that such alignment **results in a slight
> degradation in the model's performance**, this reward aligns with human preferences,
> making it more readable."

### 5.2 Counter-evidence: SFT-first can LOWER the ceiling (SimpleRL-Zoo, arXiv 2503.18892)

HKUST/TikTok ran the controlled experiment across 10 base models (Llama3-8B, Mistral-7B/24B,
DeepSeek-Math-7B, Qwen2.5 0.5B->32B):

> "While high-quality CoT data can rapidly enhance a base model's performance through
> imitation, we find that it **significantly limits the model's ability to explore freely
> during RL**. This constraint **diminishes post-RL performance** and suppresses the
> emergence of advanced reasoning capabilities."

Mistral-24B, final accuracy after identical RL:

| RL starting point | result |
|---|---|
| Base model, no SFT | **best** |
| After 100 SFT steps | lower ceiling |
| After 500 SFT steps | lowest ceiling |

> "these limitations become **increasingly pronounced as the number of initial SFT steps grows**"

500-step SFT models showed "significant declines in 'enumeration' and 'verification' behaviors."

### 5.3 The variable that reconciles them (Chu et al., ICML 2025, arXiv 2501.17161)

Section heading, verbatim:

> **"SFT is necessary for RL training when the backbone model does not follow instructions."**
>
> "Figure 9 shows that **without SFT, all end-to-end RL runs fail to improve**. More
> specifically, we observe that without SFT, the base model suffers from poor instruction
> following capability."

And SimpleRL-Zoo explains why they saw the opposite:

> "Qwen2.5 models, due to their extensive use of synthetic data during pretraining,
> **already exhibit instruction-following abilities and reflective behaviors**"

**THE RULE (corrected):** SFT before RL is needed **only if the base model cannot follow the
prompt format well enough for RL to get a usable reward signal.** Qwen2.5 base can -> skip it.
Llama3.2 base cannot -> RL fails outright without it.

### 5.4 Is SFT alone enough?

Chu et al. also give the strongest argument *against* SFT-only — out-of-distribution:

| task | RL | SFT |
|---|---|---|
| GeneralPoints-L (OOD) | +3.5% | **-8.1%** |
| V-IRL-L (OOD) | +11.0% | **-79.5%** |

Nearest real evidence *for* SFT-only is **LIMA** (arXiv 2305.11206) — 65B LLaMa, SFT on
**1,000 hand-curated examples, no RL at all**: preferred or equal to GPT-4 in **43%** of human
comparisons, Bard **58%**, DaVinci-003 **65%**. But: "while LIMA held its ground on examples
similar to training, it was **thrown off by out-of-distribution prompts or adversarial
examples**" — which matches Chu's -79.5% exactly.

**Verdict: SFT-only is strong in-distribution and brittle outside it.** Fine for a narrow
internal tool. Not fine for an open-ended assistant.

> **Retracted:** an earlier draft of this doc claimed "SFT alone is enough for ~80% of business
> applications." **There is no source for that number — it was invented.** Treat it as a rule of
> thumb from practice, never cite it.

---

<a name="6"></a>

## 6. Types inside SFT — four independent axes

Pick one from each.

### Axis A — what parameters you update
| type | updates | memory | when |
|---|---|---|---|
| Full fine-tuning | all weights | highest | plenty of GPUs and data |
| **LoRA** | small low-rank matrices, base frozen | 10-20x less | the default |
| **QLoRA** | LoRA on a 4-bit base | least | 65B on one 48GB GPU |
| adapters / prefix / prompt tuning | small added modules | least | older |

### Axis B — where the data comes from
| type | source | note |
|---|---|---|
| Human-written | people write answers | best quality, does not scale |
| **Distillation** | a stronger model writes answers | **has largely won for instruction data** |
| Self-improvement | the model generates its own data | Self-Instruct, Alpaca |
| **Rejection-sampling FT (RFT)** | generate N, keep the good ones, train on those | between SFT and RL; also called STaR |

"SFT with sampling" = rejection sampling fine-tuning. Bigger teacher -> distillation; same
model -> self-improvement. Same mechanism, different teacher.

### Axis C — what you are teaching
instruction tuning · chat/conversational · domain adaptation · reasoning/CoT · tool-calling ·
multimodal (VLM) · safety

### Axis D — the loss variant (where the open research is)
| variant | idea |
|---|---|
| standard NLL | plain cross-entropy |
| **chunked NLL** | same maths, chunked CE + skip ignored tokens -> far less peak memory. **TRL's default** |
| **DFT** (Dynamic Fine-Tuning) | rescales the learning signal by model confidence (arXiv 2508.05629, in TRL as `loss_type="dft"`) |
| **PSFT** (Proximal SFT) | PPO-style clipping on token probability ratios; prevents policy drift and entropy collapse |
| ASFT | anchors to a reference model for stability |
| ProFiT | emphasises high-value supervision tokens |

**Axis D is closest to our kind of work** — loss-function and numerical-stability changes,
not data changes.

---

<a name="7"></a>
## 7. Types inside RL / preference tuning

Two families. The split: **does the model have to generate during training?**

### Family 1 — true RL (generates rollouts, gets scored)
| method | how | cost |
|---|---|---|
| **PPO (classic RLHF)** | train a separate reward model, then PPO with a KL penalty | highest — reward model is a second model of similar size; sample-inefficient |
| **GRPO** | drops the value network; scores a group of samples and normalises within the group | much cheaper — what DeepSeek used |
| **RLVR** | no reward model — a **program** checks the answer (maths result, unit test) | cheap and objective, only where correctness is checkable |

### Family 2 — direct preference (no rollouts, no reward model)
| method | how it differs |
|---|---|
| **DPO** | reparameterises the RLHF objective so no reward model and no PPO are needed. One loop on static (chosen, rejected) data. **~3x less compute than RLHF** |
| IPO | fixes DPO overfitting; regularises harder toward the reference |
| **KTO** | works with **unpaired** thumbs-up/down feedback — no matched pairs needed |
| ORPO | merges SFT and preference into one loss, skipping the separate SFT stage |
| SimPO | reference-model-free variant |

TRL exposes **15+ DPO loss variants** in one trainer (`loss_type=` sigmoid, hinge, ipo,
exo_pair, nca_pair, robust, bco_pair, sppo_hard, aot, apo_zero, apo_down, discopop, sft,
sigmoid_norm...).

**Choosing:**
```
Can a program verify correctness? (maths, code, tests)
  YES -> RLVR
  NO
   |- paired (chosen, rejected) data? -> DPO   (IPO if it overfits)
   |- only thumbs up/down?            -> KTO
   |- need to exceed the data?        -> GRPO  (PPO only with a reason)
```

**CAVEAT WORTH REMEMBERING.** A 2026 controlled study across model scales (arXiv 2603.19335)
found **scale-dependent ranking inversions** — which method wins **changes with model size**.
Any "X beats Y" claim from a blog is only valid at the scale it was measured. **Measure at our
scale, do not inherit a ranking.** Same lesson as the locked-1200 vs free-clock finding in
BluBLAS.

---

<a name="8"></a>
## 8. The library landscape

| library | status | notes |
|---|---|---|
| **HuggingFace TRL** | **ALIVE — use this** | v1.13.0. SFT, DPO, GRPO, KTO, RLOO, reward, distillation |
| **torchtune** (PyTorch) | **DISCONTINUED 2025** | Many 2026 blogs still recommend it. They are out of date |
| **torchforge** (PyTorch) | **PAUSED 2026** | *not* the successor. "Development paused... consolidated in torchtitan". See §21 |
| **torchtitan** (PyTorch) | **ALIVE — the real PyTorch one** | pretraining platform that ADDED SFT. Closest analogue to BluTrain. See §20 |
| **Unsloth** | alive | 2x faster, up to 70% less VRAM, custom Triton kernels. **Closest to our kind of work** |
| **Axolotl** | alive | YAML config, production scale |
| **LLaMA-Factory** | alive | GUI, widest model coverage |
| **NVIDIA NeMo / NeMo-Aligner** | alive | Megatron-Core underneath |

**Supporting:** PEFT (LoRA/QLoRA) · bitsandbytes (4-bit) · Liger Kernel (fused Triton:
+20% throughput, -60% memory) · FlashAttention · DeepSpeed / FSDP

**torchtune being dead is the most useful single fact from day 1** — it is what most guides
still recommend. Verified on the official wind-down issue (#2883) and the torchforge
announcement, not from a blog.

**Framework speed** (Llama-3.1 8B, single A100 40GB, same config): Unsloth 3.2h ·
LLaMA-Factory 3.4h · torchtune 4.7h · Axolotl 5.8h. Indicative only — I would re-measure.

---

<a name="9"></a>
## 9. The layering — verified from source, not assumed

**PyTorch core has NO SFT module.** Checked the local clone (2.10.0a0):
```
dirs named sft/finetune/rlhf/post_train under torch/  -> (blank)
files named *sft*.py / *dpo*.py under torch/          -> (blank)
```
Everything in `torch/` is infrastructure: autograd, nn, optim, distributed, cuda, fx, amp.

**What PyTorch DOES provide — the building blocks:**
| SFT needs | PyTorch file |
|---|---|
| the loss | `torch/nn/modules/loss.py` — `CrossEntropyLoss` |
| **the masking** | same file **line 245: `ignore_index: int = -100`** |
| optimizer | `torch/optim/adamw.py` |
| gradient checkpointing | `torch/utils/checkpoint.py` |
| flash attention | `torch/nn/functional.py` -> `scaled_dot_product_attention` |
| multi-GPU | `torch/distributed/fsdp` |
| bf16 | `torch/amp` |

**The `-100` that TRL uses is not TRL's invention — it is PyTorch's default `ignore_index`.**

**Proof TRL sits on PyTorch** (from `trl/trainer/sft_trainer.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
...
chunk_loss = F.nll_loss(log_p, lbl, ignore_index=-100, reduction="sum")
labels[output["attention_mask"] == 0] = -100
```

**The stack:**
```
        HUGGING FACE                        PYTORCH
        TRL (SFT, DPO, GRPO, ...)           torchtune  <- DISCONTINUED
              |                             torchforge <- successor
        transformers.Trainer                     |
              |                                  |
              +---------> PyTorch <--------------+
                    (tensors, autograd, nn, optim, cuda)
```

**TRL is POST-TRAINING ONLY.** Its `trl/trainer/` directory contains sft, dpo, grpo, kto,
rloo, reward, distillation — **and no pretraining trainer**. Same for torchtune (recipes are
standalone scripts, all post-training). **Nobody ships a pretraining library**; pretraining is
custom code at each lab.

**And this maps onto us:**
```
PyTorch core    ≈  BluTrain's Tensor + autograd + dtypes + kernels   <- what our team writes
TRL / torchtune ≈  the layer nobody on our team has built yet        <- the new work
```

---

<a name="10"></a>
## 10. Why a whole SFT module exists if the loss is unchanged

I read `trl/trainer/sft_trainer.py`:

| | lines | share |
|---|---|---|
| **data pipeline** | **~1,100** | **~88%** |
| loss computation | ~150 | ~12% |
| total | ~1,850 | |

**The module exists for the data pipeline, not the loss.**

What that 88% does, by function name:

| function | what it does | does pretraining need it? |
|---|---|---|
| `maybe_convert_to_chatml()` | convert dataset into message format | no |
| `apply_chat_template()` | messages -> `<\|im_start\|>user…` token string | no |
| `add_eos()` | append the stop token | no — pretraining just streams text |
| `build_labels()` | **write `-100` on prompt/user tokens** | no — pretraining trains on everything |
| `truncate()` | keep_start / keep_end | sort of |
| `pack_dataset()` | best-fit-decreasing bin packing | different problem |
| `get_position_ids_from_packed_seq_lengths()` | **reset position IDs at each packed boundary** | no |
| `prepare_multimodal_messages()` | image handling | no |
| `DataCollatorForLanguageModeling.torch_call()` | padding, attention masks | different |

**That position-ID function is the good example of why this is not trivial.** Pack 5 short
conversations into one 4096-token sequence and the position IDs must restart at each
conversation. If they do not, the model treats it as one long document and RoPE gives
conversation 3 completely wrong positional encoding. **Silent, subtle, and wrong.**

---

<a name="11"></a>
## 11. How DPO and GRPO differ from SFT — mechanically and in the kernels

**The one-line version:**

| | what you give it | what it does |
|---|---|---|
| **SFT** | prompt + **the right answer** | "copy this answer" |
| **DPO** | prompt + **two answers**, one better | "prefer this over that" |
| **GRPO** | prompt **only** (no answer) | "generate 8 yourself, I score them, do more of the good ones" |

**The loss must differ because the data differs.**

**SFT** — you know the correct token, so plain likelihood works. No new loss needed.

**DPO** — there is no single right answer. If you SFT on the chosen answer you throw away the
information that the rejected one was bad. DPO keeps both and widens the gap:
```
L_DPO = − E[ log σ( β ( log π_θ(y⁺|x)/π_ref(y⁺|x) − log π_θ(y⁻|x)/π_ref(y⁻|x) ) ) ]
```
Two new things: a **sigmoid** (binary classification, not CE over the vocab) and **π_ref**, a
second frozen model — the copy from before DPO started, which stops the model drifting into
gibberish that happens to score well.

**GRPO** — you have no answers at all, only prompts and a scorer:
```
Â_i = (r_i − mean(r)) / std(r)          <- "group relative"
L_GRPO = PPO-style clipped surrogate on Â
```
Generate 8 completions, score each, push up the above-average ones. The "group relative" trick
replaces PPO's value network by comparing the 8 samples against each other.

### What actually runs on the GPU

**SFT step:** 1 forward -> logits `[B,T,V]` -> CE -> backward. **1 model, 1 forward.**

**DPO step:** policy on chosen, policy on rejected, ref on chosen, ref on rejected -> combine
4 scalars -> sigmoid -> backward through the 2 policy passes. **2 models, 4 forwards.**

**GRPO step:** generate 8 completions (autoregressive decode) -> score -> advantages ->
forwards -> backward.

**The kernel point that matters to us:** GRPO puts an **inference workload inside the training
loop**.

| | training forward | autoregressive decode |
|---|---|---|
| matmul shape | `[B×T, K] × [K, N]` — big GEMMs | `[B, K] × [K, N]` — **skinny, one token** |
| bound by | **compute** (tensor cores) | **memory bandwidth** |
| KV cache | none | **grows every token** |

That is why GRPO needs **vLLM**. And it creates a real problem: vLLM uses different kernels
than training, so the logprobs disagree — TRL corrects for it with importance sampling
(`vllm_importance_sampling_correction=True` by default). **A numerical mismatch caused purely
by two kernel paths computing the same thing differently. Very much our kind of problem.**

### Summary

| | SFT | DPO | GRPO |
|---|---|---|---|
| data | prompt + answer | prompt + chosen/rejected | **prompt only** |
| loss | cross-entropy (**unchanged**) | sigmoid on log-ratio margins | clipped surrogate + advantage |
| models in memory | **1** | **2** | 1-3 (+ vLLM copy) |
| forwards/step | 1 | 4 | 8 generations + 8 forwards |
| generates during training? | no | no | **yes** |
| default LR | 2e-5 | **1e-6** | 1e-6 |

DPO and GRPO default **20x lower LR than SFT** — that alone says how much more fragile they are.
**SFT is the cheap, stable one.**

---

<a name="12"></a>
## 12. Datasets

| dataset | size | note |
|---|---|---|
| **Tulu 3** (AllenAI) | 939k | 4-axis LLM-judge filtered; strongest open recipe |
| **SmolTalk / SmolTalk2** (HF) | ~1.1M | aimed at small models |
| OpenHermes-2.5 | 1,001,551 | multi-source conversations |
| **Magpie** | 4M -> filtered 300k | shows filtering beats volume |
| Alpaca | 52k | historical |
| **LIMA** | 1,000 | the quality-over-quantity proof |
| Capybara (`trl-lib/Capybara`) | small | TRL's demo set, good for smoke tests |

**The finding that matters:** an unfiltered synthetic dataset is **worse** than a smaller
filtered one. TuluTalk reached better benchmarks with **-23% vs SmolTalk and -14% vs Tulu** by
filtering. **Data curation is the research, not the training loop.**

Curated lists: `jianzhnie/awesome-instruction-datasets` · Maxime Labonne's `llm-datasets`

---

<a name="13"></a>
## 13. Papers to read

**Foundational, in order:**
1. **InstructGPT** (2022) — arXiv 2203.02155 — SFT + RLHF, the origin
2. **FLAN** — instruction tuning across many tasks
3. **Self-Instruct** (2022) — arXiv 2212.10560
4. **Alpaca** (2023) — 52k synthetic examples
5. **LIMA** (2023) — arXiv 2305.11206 — **1,000 examples is enough.** The "superficial
   alignment hypothesis": the base model already has the knowledge, SFT only teaches format
   and style. **The most important idea in SFT.**
6. **Tulu 3** (2024) — arXiv 2411.15124 — the fully open modern recipe

**Survey:** Instruction Tuning for LLMs: A Survey — arXiv 2308.10792

**Current:** Packing with Flash Attention (2407.09105) · DFT loss (2508.05629) ·
Post-training data quality (2506.06522) · SFT vs RL (2603.13985) · SFT-then-RL (2604.23747) ·
Does prompt loss matter (2401.13586) · Post-training algorithms scale inversions (2603.19335)

---

<a name="14"></a>
## 14. Hyperparameters

| knob | value | why |
|---|---|---|
| epochs | **1-2** | >2 is not beneficial and worsens catastrophic forgetting; optimum reported near 1.5 |
| LR (full FT) | **1e-5 to 2e-5** | 2e-5 is TRL's default |
| LR (LoRA) | **~1e-4** | only the new small params learn |
| scheduler | cosine or linear with warmup | |
| batch | as large as memory allows, via grad accumulation | |
| precision | **bf16** | |
| max_length | 1024-4096 | longer = more memory; use packing |

**Catastrophic forgetting** is the main risk. Mitigations: fewer epochs, lower LR, LoRA instead
of full FT, mix in general data, NEFTune noise (`neftune_noise_alpha=5`), model merging.

**Always evaluate the base model too.** SFT that improves IFEval but drops MMLU 5 points is
forgetting, not a win.

---

<a name="15"></a>
## 15. Efficiency levers — where our background applies

| lever | effect |
|---|---|
| **Packing + FlashAttention** | token utilisation 10% -> 98% |
| **Liger Kernel** (fused Triton: RMSNorm, SwiGLU, RoPE, fused CE) | +20% throughput, -60% memory, 4x longer context |
| **Chunked cross-entropy** | peak activation memory stops scaling with `vocab × seq_len` |
| Gradient checkpointing | ON by default in TRL |
| Activation offloading | activations to CPU |
| QLoRA (4-bit) | 65B on one 48GB GPU |
| Padding-free batching | needs FA2/FA3 |

**The `vocab × seq_len` logits tensor is the memory monster in SFT.** At vocab 50k+ and
seq 4096 the logits alone dominate — **exactly the sparse-CE / chunked-CE problem I already
worked on in BluTrain.** Directly transferable.

---

<a name="16"></a>
## 16. Evaluation

| benchmark | measures | notes |
|---|---|---|
| **IFEval** | instruction following | 500+ prompts, 25 atomic instruction types, checked by **deterministic programs — no LLM judge, so no judge bias**. Best first metric |
| MT-Bench | multi-turn quality, 8 categories | GPT-4 as judge |
| AlpacaEval 2.0 | pairwise win rate | 805 prompts, length-controlled to reduce length bias |
| Arena-Hard | 500 hard queries | highest correlation with human Chatbot Arena |
| **lm-evaluation-harness** | MMLU, GSM8K, TruthfulQA | use this to check **what the model FORGOT** |

**Method: always run the base model on the same benchmarks.** Two-sided measurement, same as
our kernel work.

TRL logs during training: `loss`, `entropy`, `mean_token_accuracy`, `num_tokens`, `grad_norm`,
`learning_rate`.

---

<a name="17"></a>
## 17. BluTrain gap analysis — audited against the real code

Audited `blutrain_ours/Tensor-Implementations` + `bluscript_zero.cpp`. Every line checked, not
assumed.

### What we already have
training loop · GPT architecture (RMSNorm, GQA, RoPE, SwiGLU) · all 81 forward/backward CUDA
kernels · autograd engine + backward nodes · AdamW + WSD schedule + grad clip · sparse CE
(fwd + bwd, vectorised, save-stats) · 18 dtypes + promotion · memory allocator · data loader
(mmap uint16 shards) · checkpoint save/load · ZeRO-1 v2 + DDP · BluBLAS · scoring-based eval

**The entire compute path is done. SFT needs no new maths.**

### GAP 1 — cross-entropy has NO label masking. THE BLOCKER.
```cpp
void sparse_cross_entropy_forward_cuda(
    const T* logits, const T_idx* targets, T* loss_output,
    int64_t batch_size, int64_t vocab_size, cudaStream_t stream);
```
No mask parameter. Grepping `ignore_index|ignore_idx|-100` across `src/` and `include/` gives
**zero hits**. And the trainer calls it with two tensors:
```cpp
Tensor loss = autograd::sparse_cross_entropy_loss(logits, b.target);   // bluscript_zero.cpp:1003
```
**Without this, SFT trains on the prompt as well as the answer.**

### GAP 2 — data loader carries only (input, target)
Both `[B,T]` uint16 from one mmap'd stream with a +1 offset. No place for a mask.

### GAP 3 — no tokenizer, no chat-template machinery in C++
Zero hits for `Tokenizer|BPE|tiktoken|SentencePiece`. Data arrives pre-tokenized as uint16
`.bin`. **This is fine** — templating and tokenisation belong offline in Python.

### GAP 4 — RoPE takes a single scalar `pos_offset`, not per-token position IDs
```cpp
void fused_qk_norm_rope_forward(..., int B, int Hq, int Hkv, int T, int hd,
                                int pos_offset, float eps, bool interleaved, ...);
```
One `int` for the whole batch. **Blocks packing.**

### GAP 5 — attention has no varlen / `cu_seqlens`
Zero hits for `cu_seqlens|varlen|doc_mask|block_mask`. With packing, conversation 3 would
attend to conversation 1's tokens. **Silently wrong; loss still descends. Blocks packing.**

### GAP 6 — `Module::forward` takes exactly one tensor
```cpp
virtual Tensor forward(const Tensor& input) = 0;     // include/nn/NN.h:21
```
No attention mask, no position_ids, no labels.

### GAP 7 — no HuggingFace checkpoint import
Zero hits for `safetensors|from_pretrained`. `convert_ckpt.py` goes **ours -> PyTorch**
(`torch.save`, line 151), not the reverse. SFT-ing our own checkpoint works today via
`load_latest()`; SFT-ing someone else's base model needs this built.

### GAP 8 — no text generation / KV cache
`hellaswag_eval.h` scores by forward + log-prob, it does not generate. Fine for loss and
scoring evals; **not** enough for IFEval / MT-Bench / AlpacaEval.

### The minimum path (4 things, one of them a kernel)
1. **Offline Python script** — chat template -> tokenize -> write two uint16 `.bin` streams
   (tokens, labels) with the prompt masked.
   **Use `IGNORE = 65535`** — targets are uint16 and vocab is 50304, so 65535 is free. No dtype
   change, no extra tensor, no need for -100.
2. **Add `ignore_index` to the CE kernels** — the only kernel work.
   - forward: `if (targets[row] == ignore) { loss[row] = 0; return; }`
   - backward: zero that gradient row
   - **divide by `n_valid`, not `batch_size`** — get the denominator wrong and every gradient
     is silently scaled
   - touches `LossKernels.h/.cu`, `LossOps.cpp`, `LossBackward.cpp`
3. **Data loader reads two streams** instead of one.
4. **Trainer** — pass the ignore index, load the pretrained checkpoint, LR **2e-5** not 3e-4,
   1-2 epochs.

**Skip packing for v1** — it needs GAP 4 + GAP 5, and packing without varlen attention is a
silent correctness bug. Pad instead; it wastes compute but is correct.

### The verification that cannot pass by accident
- Two identical batches differing **only in the prompt text**. **Gradients must be identical.**
- Print `n_valid` vs `B*T` on batch one. If prompts are half the tokens, `n_valid` should be
  ~half. If it equals `B*T`, masking never fired.
- Loss should start near the pretrained model's loss, not at `ln(50304) = 10.8`.

---

<a name="18"></a>
## 18. Side investigation — the Wave Network paper (arXiv 2411.02674)

Came up by accident; investigated and closed.

**Claim:** a 2.4M-parameter "ultra-small language model" using complex vectors reaches
BERT-comparable accuracy with -77% memory and -86% training time.

**Method:** each token becomes `G · e^{iα}` where
`G_k = ‖[w_1k…w_nk]‖₂` (L2 norm down each dimension across **all** tokens) and
`α_jk = arctan2(√(1−(w_jk/G_k)²), w_jk/G_k)`. Via Euler: real part = `w_jk` (the token itself),
imaginary part = `√(G_k² − w_jk²)` (the rest of the context). Updates by complex **addition**
(interference) or **multiplication** (modulation) instead of dot products — **O(n·d) instead of
attention's O(n²·d)**.

**Why it does not apply to us:**
1. **It is not a language model.** Every experiment is text classification (AG News 4-class,
   DBpedia 14-class, IMDB 2-class). Zero mentions of causal / autoregressive / next-token /
   perplexity in the paper. No generation anywhere.
2. **It breaks causality.** `G` sums over **all** tokens including future ones. For next-token
   prediction that is feeding the model the answer.
3. **Structural bottleneck.** `G` is **one vector shared by every token**; only the phase
   differs per token. Fine for classification (one label per document), but generation needs a
   **different** context view per position. Attention gives that; this cannot.
4. **The killer number:** **fastText (bag-of-words, 2016) gets 92.44% on AG News. Wave Network
   gets 91.66%.** It loses to a bag-of-words baseline from eight years earlier. The "2.4M vs
   100M" framing only looks dramatic because AG News never needed a language model.
5. Their single-transformer-layer baseline scores 71.68% — not a real baseline.
6. Two years old (Nov 2024), no independent reproduction; only the authors' own follow-up.

**If we ever wanted it:** it would replace **attention only** (`src/Kernels/cuda/attention/`
plus `AttentionBackward.cpp`). New primitives needed: prefix scan along the sequence axis,
`arctan2`, complex add/mul — **and we already have complex32/64/128 and the reduction module.**

**Benefit ceiling:** from our own 4B nsys profile, attention is **16.6% of GPU time**. So 16.6%
is the absolute maximum gain, and only if the wave op were free. Not worth it at T=4096.

**Where the maths changes:** attention is O(n²), so its share grows with sequence length. At
32k-128k context an O(n) attention replacement becomes worth real money — **but then use
Mamba/SSM, which is causal by construction, has a per-position state (fixing the shared-`G`
bottleneck), and is proven on generation at scale.**

**Verdict: read, understood, not pursuing.**

---

<a name="19"></a>
## 19. Open decisions (before implementation starts)

1. **Which base model?** Our own pretrained checkpoint (works today via `load_latest`) or
   someone else's (needs GAP 7 first). **This decides the timeline.**
2. **Which dataset**, and do we have its tokenizer?
3. **Is BluTrain the target**, or do we prototype in TRL first to learn the shape and port
   after? A day in TRL would de-risk the BluTrain build.
4. **GPU** — no box is currently allocated.
5. **General instruct model or domain-specific?**
6. **Do we own the data pipeline too, or only the training?**

---

<a name="20"></a>
## 20. The three layers — engine vs platform vs recipes

This was a real point of confusion, so it is written down explicitly. **torchtitan is not
PyTorch.** It is Python code that *calls* PyTorch. Verified against the local checkout.

| layer | what it actually is | PyTorch world | **BluTrain equivalent** |
|---|---|---|---|
| **1. Engine** | C++/CUDA kernels, autograd graph, tensor memory | `pytorch/pytorch` | **`Tensor-Implementations/`** (submodule) |
| **2. Platform** | parallelism, checkpointing, the training loop, data loading | `torchtitan` (pure Python) | **`bluscript_zero.cpp`, `dist/`, `Makefile`** |
| **3. Recipes** | chat templates, dataset formats, trainer wrappers | `TRL`, Axolotl, LlamaFactory, ms-swift | *(we have none — this is what we must build)* |

**Proof that `pytorch_source/` is the engine, not torchtitan** (local, 216 MB):

```
pytorch_source/aten/          <- ATen, the C++ tensor library     FOUND
pytorch_source/aten/src/ATen  <- kernel implementations           FOUND
pytorch_source/c10/           <- core abstractions                FOUND
pytorch_source/torch/csrc/    <- C++/Python bindings              FOUND
pytorch_source/caffe2/                                            FOUND
```

**Why this matters for where our work lands:**

- The **CE `ignore_index` change is an ENGINE-layer change.** Its references are
  PyTorch ATen `Loss.cu`, Liger-Kernel, and Megatron's fused kernels — **not** TRL.
- The **trainer / dataloader / denominator change is a PLATFORM-layer change.** Its
  references are Megatron `pretrain_gpt.py` and torchtitan.
- The **data preprocessing (chat template -> masked token stream) is a RECIPE-layer
  change.** Its reference is TRL. This is the layer we have nothing for.

Most SFT tutorials online only ever touch layer 3. **We are the only ones of the three
who have to write layer 1 ourselves**, because we own our engine.

---

<a name="21"></a>
## 21. Framework landscape — verified 2026-09-18, not from blogs

### 21.1 Status, checked against the repos themselves

| repo | stars | size | last push | status |
|---|---|---|---|---|
| `NVIDIA/Megatron-LM` | 17,939 | 137 MB | 2026-09-18 | alive |
| `pytorch/torchtitan` | 5,747 | 179 MB | 2026-09-18 | **alive — the PyTorch consolidation point** |
| `linkedin/Liger-Kernel` | 6,619 | 28 MB | 2026-09-17 | alive |
| `huggingface/trl` | 19,339 | 28 MB | 2026-09-18 | alive |
| `hiyouga/LlamaFactory` | 74,851 | 13 MB | 2026-09-14 | alive |
| `unslothai/unsloth` | 76,353 | 360 MB | 2026-09-18 | alive (multi-GPU is paid) |
| `verl-project/verl` | 23,485 | 29 MB | 2026-09-18 | alive |
| `modelscope/ms-swift` | 15,667 | 83 MB | 2026-09-18 | alive |
| `axolotl-ai-cloud/axolotl` | 12,487 | 49 MB | 2026-09-18 | alive |
| `OpenRLHF/OpenRLHF` | 10,015 | 4 MB | 2026-09-17 | alive |
| `apple-aiml-research/ml-cross-entropy` | 618 | <1 MB | 2026-09-11 | alive |
| `meta-pytorch/torchtune` | 5,812 | 63 MB | 2026-09-09 | **DISCONTINUED** |
| `meta-pytorch/torchforge` | 702 | 416 MB | 2026-09-09 | **PAUSED** |

Verbatim README notices:

> **torchtune:** "Torchtune is no longer actively maintained: torchtune development wound
> down in 2025 — see The future of torchtune (issue #2883)."

> **torchforge:** "Development paused: Development in Forge has paused. **LLM training at
> PyTorch is being consolidated in torchtitan.**"

**Correction to §8 of this document:** it called torchforge "the successor to torchtune."
That is now wrong — PyTorch started two post-training libraries and stopped both. The live
one is **torchtitan**, which now lists *"Supervised Fine-Tuning (SFT) with chat-formatted
datasets"* as a feature.

### 21.2 Repositories that were RENAMED (old clone URLs are stale)

| old | current |
|---|---|
| `hiyouga/LLaMA-Factory` | `hiyouga/LlamaFactory` |
| `volcengine/verl` | `verl-project/verl` |
| `apple/ml-cross-entropy` | `apple-aiml-research/ml-cross-entropy` |
| `NVIDIA-NeMo/NeMo` | redirects to `NVIDIA-NeMo/Speech` — the monorepo was split; LLM post-training moved to `NVIDIA-NeMo/Megatron-Bridge` (918 stars) and `NVIDIA-NeMo/RL` (2,025 stars) |

### 21.3 Selection — one main, two cross-references

| role | choice | why |
|---|---|---|
| **MAIN architectural reference** | **`NVIDIA/Megatron-LM`** | The only one with our shape: own C++/CUDA kernels, tensor+data parallel, trains from scratch, and **one script does pretraining and SFT**. Its `pretrain_gpt.py` docstring is literally `"""Pretrain and SFT GPT."""` |
| **Cross-ref 1 — kernel** | **PyTorch ATen `Loss.cu`** (already local) + **Liger-Kernel** | The exact kernel we are writing. ATen is the correctness reference, Liger the performance one |
| **Cross-ref 2 — data/recipe** | **`huggingface/trl`** | What the ecosystem uses. Defines chat templates, packing, and the `-100` convention. We must interop with it whether we like it or not |
| secondary | `pytorch/torchtitan` | A pretraining platform that *added* SFT — the same transition we are making. Read its SFT dataset + PR #2769 |
| benchmark-only | LlamaFactory, Unsloth, Axolotl, ms-swift, verl | For throughput comparison later. Do not copy design from them |

**Why Megatron over HuggingFace as "the main one", despite HF being more popular:** we are not
*using* any of these. We own our engine. TRL sits at layer 3 and assumes PyTorch underneath —
none of its code is portable to us. Megatron sits at layers 1+2 and its design decisions
*are* portable. Popularity decides the data format we interop with; architecture decides
what we copy.

---

<a name="22"></a>
## 22. How three reference implementations do masked CE — read from source

Three different designs solve the same problem. All three verified by reading the code,
not documentation.

### 22.1 PyTorch ATen — `aten/src/ATen/native/cuda/Loss.cu` (local, `pytorch_source/`)

**Design: `ignore_index` inside the kernel, early-out per row.**

```cpp
// nll_loss_forward_no_reduce_cuda_kernel
CUDA_KERNEL_LOOP(index, batch_size) {
    index_t cur_target = target[index];
    if (cur_target == ignore_index) {
      output[index] = static_cast<scalar_t>(0);
      continue;                               // <-- skips all work for this row
    }
    CHECK_INDEX_IN_CLASS(cur_target, n_classes);
    auto cur_weight = weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    output[index] = -cur_weight * input[index][cur_target];
}
```

And the reduce kernel accumulates the denominator **in the same pass** as the loss:

```cpp
for (int i = threadIdx.x; i < nframe; i += blockDim.x) {
    index_t t = target[i];
    if (t != ignore_index) {
      sh_inputs[threadIdx.x]  -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;        // <-- this is n_valid
    }
}
// ... tree reduction on BOTH arrays ...
if (threadIdx.x == 0) {
    *total_weight = static_cast<scalar_t>(acc_weight[0]);
    if (size_average) {
      *output = static_cast<scalar_t>(sh_inputs[0] / acc_weight[0]);   // <-- / n_valid
    } else {
      *output = static_cast<scalar_t>(sh_inputs[0]);                   // <-- unnormalized
    }
}
```

**Three things to copy from this:**
1. The early-out shape is *exactly* what we planned — confirmed against the reference.
2. `total_weight` is a **returned output**, not an internal. That is what makes global
   renormalization possible downstream.
3. There is a `size_average` switch — the caller can ask for the **unnormalized sum**.
   We need that switch for the distributed case (see §23).

Also note their NaN guard, which we need for the all-masked edge case:

```cpp
// If we try to normalize a zero then we return a NaN
if (cur_weight == 0) { *output = std::numeric_limits<scalar_t>::quiet_NaN(); }
```

### 22.2 Megatron-LM — `pretrain_gpt.py` (the closest match to BluTrain)

**Design: no `ignore_index` at all. A float mask multiplied after the fact.**

The file's own docstring: `"""Pretrain and SFT GPT."""` — **one script does both jobs.**

```python
loss_mask  = loss_mask.view(-1).float()
loss       = torch.sum(losses * loss_mask)          # UNNORMALIZED sum
num_tokens = loss_mask.sum().clone().detach().to(torch.int)
...
return loss, num_tokens, report                     # both returned, division happens later
```

**They never divide inside the loss function.** The sum and the count are returned
separately so the framework can all-reduce both across data-parallel ranks and divide
once, globally. This is the single most important design decision we found.

Trade-off vs ATen's approach: no kernel signature change at all, but every masked token
still pays the full vocab-wide softmax before being multiplied by zero.

### 22.3 Liger-Kernel — `ops/cross_entropy.py`, `ops/fused_linear_cross_entropy.py`

**Design: `ignore_index` in the kernel (default `-100`), plus chunking to avoid
materializing logits.**

Avoids the large logits tensor by fusing the final linear layer into the CE computation
and computing the gradient during the forward pass. Reported ~4x memory reduction vs
native CE; for vocab 128,000 at seq 2048 / batch 4 that is roughly 2 GB per batch saved.

Our vocab is **50,304** and our logits are `[B, T, V]` = `[2, 4096, 50304]` — this is a
real lever for us later, but it is a *separate* optimisation from correctness. Do not mix
the two changes.

### 22.4 Side-by-side

| | PyTorch ATen | Megatron | Liger |
|---|---|---|---|
| mechanism | `ignore_index` in kernel | float `loss_mask`, multiply after | `ignore_index` in kernel |
| kernel signature change | yes | **none** | yes |
| skips work on masked rows | **yes** | no | **yes** |
| denominator | `acc_weight`, same pass | `num_tokens`, returned | returned |
| division happens | in kernel (optional) | **outside, globally** | outside |
| proven at 100B+ scale | — | **yes** | yes |

---

<a name="23"></a>
## 23. The global denominator — a bug caught before writing any code

### 23.1 What the earlier plan said

> "divide by `n_valid`, not `batch_size` — get the denominator wrong and every gradient is
> silently scaled"

Correct about `batch_size` being wrong. **Wrong about where the division goes.**

### 23.2 Why per-rank `n_valid` is wrong

The 4B config runs **2 ranks x grad_accum 32 = 64 micro-batches per optimizer step.**
In SFT every micro-batch has a *different* number of unmasked tokens, because completions
have different lengths. If each micro-batch divides by its own local `n_valid` and the
results are then averaged:

- micro-batch with 50 valid tokens  -> each of its tokens weighted `1/50`
- micro-batch with 500 valid tokens -> each of its tokens weighted `1/500`

**Tokens in short completions get weighted 10x more than tokens in long ones.**

This is invisible in every normal check: loss is finite, loss descends, `maxRelErr` is 0,
gradients are not NaN. The model simply learns the wrong objective. **It never appears in
pretraining** because there every token is valid, so the count is uniform across
micro-batches and the bug cancels exactly.

### 23.3 The correct form

```
per micro-batch:  emit  sum_loss_i  and  n_valid_i     (NO division)
accumulate:       S = sum_i(sum_loss_i)   N = sum_i(n_valid_i)   over all 64 micro-batches
all-reduce:       S and N across both ranks
divide ONCE:      loss = S / N
```

This is exactly why Megatron returns `(loss, num_tokens)` and why ATen exposes
`total_weight` as an output and a `size_average` switch. Both frameworks arrived at the
same answer independently.

### 23.4 Corrected touch points

| # | file | change |
|---|---|---|
| 1 | `LossKernels.cu:188` forward | early-out on ignored rows; write `losses[row]=0`, `saved_max[row]=0`, **`saved_sum[row]=1.0f`** (not 0 — `1/saved_sum` would be `inf`, and `inf*0` = NaN) |
| 2 | `LossKernels.cu` reduce | accumulate `n_valid` **in the same pass** as the loss sum (ATen pattern) |
| 3 | `LossKernels.cu:635` backward | zero the gradient row for ignored targets |
| 4 | `LossOps.cpp:349` | **do not divide here.** Return `(sum_loss, n_valid)` |
| 5 | `LossBackward.cpp:213` | `host_scale = 1/N_global`, not `1/batch_size_` and not `1/n_valid_local` |
| 6 | trainer / `dist` | accumulate `S` and `N` across grad_accum **and** all-reduce across ranks; divide once |
| 7 | guard | `N == 0` (every token masked in a micro-batch) must not divide by zero |

Item 6 is new and is the one that was missing. Items 1-3 and 5 are unchanged in shape but
now have a verified reference implementation behind each.

### 23.5 Design decision still open

Take **Megatron's semantics** (mask + global denominator, proven at scale) and **keep
ATen's in-kernel early-out** (masked rows skip the whole 50,304-wide softmax, so masking
becomes a ~2x speedup on a typical SFT batch rather than pure overhead). The two are
compatible — one decides correctness, the other performance. Confirm with both codebases
open before writing.

---

## The seven ideas to carry

1. **The objective is the same as pretraining.** Token-level cross-entropy. Data, masking and
   hyperparameters change — not the maths.
2. **Data quality beats quantity, by a lot.** LIMA got there with 1,000 examples.
3. **The base model already knows the content** (superficial alignment hypothesis). SFT teaches
   format and behaviour, not knowledge.
4. **Loss masking is nearly free and worth 5-15%.**
5. **1-2 epochs, not more.**
6. **Synthetic data has largely won** for instruction data.
7. **SFT is the foundation, not the finish.** SFT -> DPO/RLHF -> (RLVR).

---

# Appendix A — original note: full training pipeline order

*Written 2026-09-17. Superseded by §2-§7 above, which are more complete and carry the
corrected evidence (see §5). Kept verbatim so nothing from the original research is lost.*

# The full LLM training pipeline — every stage, in order
**For: Gautam | 2026-09-17**

---

# PART 1 — HOW MANY STAGES ARE THERE?

Not two. **Three.**

```
PRETRAINING   ->   MID-TRAINING   ->   POST-TRAINING
~trillions         ~hundreds of         ~millions of tokens
of tokens          billions of tokens   (SFT -> preference/RL)
```

| | PRETRAINING | MID-TRAINING | POST-TRAINING |
|---|---|---|---|
| tokens | trillions | hundreds of billions | millions |
| data | broad web text, quantity | curated, quality-first | human/synthetic demos |
| goal | learn language + world knowledge | specialise, extend, prepare for RL | behaviour, format, alignment |
| loss | next-token CE | next-token CE (same) | next-token CE (SFT), then preference/RL |

Note the loss row: **pretraining, mid-training and SFT all use the same next-token
cross-entropy.** Only data and masking change. The objective only changes at preference/RL.

---

# PART 2 — IS PRETRAINING "UNSUPERVISED"?

**No — it is SELF-supervised.** People get this wrong.

- **Unsupervised** = no labels at all (clustering, PCA)
- **Self-supervised** = labels exist, but come free from the data itself

For "The sky is ___" the label is "blue", taken from the text. There IS a label and there IS a
cross-entropy loss against it. **No human wrote it — the data supplied it.**

So pretraining vs SFT is not "unsupervised vs supervised":

| | pretraining | SFT |
|---|---|---|
| loss | next-token CE | next-token CE — **same** |
| label source | the raw text itself | a human or a stronger model wrote the answer |
| loss computed on | every token | **only the answer tokens** (masking) |
| volume | trillions | thousands to millions |

---

# PART 3 — MID-TRAINING (yes, it is real)

Term reportedly first appeared in an **OpenAI job description, mid-2024**. Now a standard stage.

**What it is:** continue pretraining on a **much better data mixture**. Same loss, same
mechanics. The shift is quantity -> quality.

**What happens in it:**
1. **Data quality shift** — upsample STEM, code, maths, reasoning; downsample generic web text
2. **Long-context extension** — 4k -> 32k -> 128k -> 256k (Composer 2 went 32K -> 256K here)
3. **Domain / language expansion**
4. **Large-scale synthetic data** — at a scale post-training cannot afford
5. **Capability additions** — e.g. Multi-Token Prediction layers for speculative decoding
6. **Preparing for RL** — mid-training changes how well later RL works (arXiv 2506.20512)

**Why a separate stage:** pretraining is too expensive to redo, post-training is too small to
teach real capability. Mid-training is cheap enough to repeat, big enough to change what the
model knows.

**Related terms:** CPT (Continued Pre-Training) is mostly the same idea. "Annealing" = the
final high-quality phase at the end of pretraining.

---

# PART 4 — WHAT IS INSIDE POST-TRAINING?

Not just SFT and RL. **Six** things:

```
POST-TRAINING
├── 1. SFT                     <- teach format and behaviour   (I am here)
├── 2. Preference tuning / RL  <- teach which answer is better
├── 3. Distillation            <- learn from a stronger teacher
├── 4. Model merging           <- combine specialists, no retraining
├── 5. Safety tuning           <- refusals, harmlessness
└── 6. Quantization / QAT      <- shrink for deployment
```

**1 and 2 are the main path.** 3-6 are supporting techniques used around them.

---

# PART 5 — WHAT ORDER? SFT FIRST OR RL FIRST?

## Answer: **SFT first, then RL.** There is hard evidence.

### The natural experiment: DeepSeek-R1

DeepSeek ran both orders and published both:

| model | recipe | outcome |
|---|---|---|
| **R1-Zero** | **pure RL, no SFT** | Works - reasoning emerges. **But:** poor readability, mixes languages mid-answer, unstable training, degenerate outputs |
| **R1** | **small "cold-start" SFT, then RL** | Fixed all of it |

### Why RL alone struggles
The base model has **poor task-following ability**. Its rollouts score overwhelmingly low ->
reward signal is nearly all zeros -> unstable updates -> collapse. Even when it converges,
exploration is enormously expensive because the model searches blindly.

**SFT's job before RL:** get the model into a region where some outputs earn reward. Then RL
has a gradient to work with. Hence **"cold start"** - warming the model so RL can take over.

### Simply
- **SFT** = show it the answer. Cheap, stable, fast. Teaches *format and style*.
- **RL** = let it try, then score it. Expensive, unstable, powerful. Teaches *quality*.
- SFT can only imitate what is in the data. **RL can exceed the data** - that is the point.

### Is SFT alone enough?
**Often yes.** Alpaca, FLAN-T5, LIMA are SFT-only and useful. For "follow instructions in our
format / our domain", SFT alone is usually sufficient. **Start here.**

### Is RL alone enough?
Technically possible (R1-Zero proved it) but you pay in compute and get a degraded model.
Nobody ships it.

### 2026 position
**"SFT-then-RL outperforms mixed-policy methods for LLM reasoning"** (arXiv 2604.23747). The
two-stage order is holding up.

---

# PART 6 — TYPES INSIDE SFT

Four independent axes. Pick one from each.

## Axis A - what parameters you update
| type | updates | memory | when |
|---|---|---|---|
| **Full fine-tuning** | all weights | highest | you have the GPUs and lots of data |
| **LoRA** | small low-rank matrices, base frozen | 10-20x less | the default |
| **QLoRA** | LoRA on a 4-bit base | least | 65B on one 48GB GPU |
| adapters / prefix / prompt tuning | small added modules | least | older, less used |

## Axis B - where the data comes from
| type | source | note |
|---|---|---|
| **Human-written** | people write answers | best quality, does not scale |
| **Distillation** | a stronger model writes answers | **has largely won for instruction data** |
| **Self-improvement** | the same model generates its own data | Self-Instruct, Alpaca |
| **Rejection-sampling FT (RFT)** | generate N answers, keep the good ones, train on those | between SFT and RL; also called STaR |

**"SFT with sampling" = rejection sampling fine-tuning.** Bigger teacher -> distillation.
Same model -> self-improvement. Same mechanism, different teacher.

## Axis C - what you are teaching
| type | purpose |
|---|---|
| **Instruction tuning** | follow instructions (the classic) |
| **Chat / conversational** | multi-turn dialogue, roles |
| **Domain adaptation** | medical, legal, code, our own domain |
| **Reasoning / CoT SFT** | long chain-of-thought traces (feeds R1-style RL) |
| **Tool-calling SFT** | emit function calls in the right schema |
| **Multimodal / VLM SFT** | images + text |
| **Safety SFT** | refusals, harmlessness |

## Axis D - the loss variant (the research-y part)
| variant | idea |
|---|---|
| **standard NLL** | plain cross-entropy |
| **chunked NLL** | same maths, chunked CE + skip ignored tokens -> far less peak memory. **TRL default** |
| **DFT** (Dynamic Fine-Tuning) | rescales learning signal by model confidence - arXiv 2508.05629, in TRL as `loss_type="dft"` |
| **PSFT** (Proximal SFT) | PPO-style clipping on token prob ratios; prevents policy drift and entropy collapse |
| **ASFT** | anchors to a reference model for stability |
| **ProFiT** | emphasises high-value supervision tokens |

**Axis D is where the open research is** - and it is closest to our kind of work: loss-function
and numerical-stability changes, not data changes.

---

# PART 7 — TYPES INSIDE RL / PREFERENCE TUNING

Two families. The split: **does the model have to generate during training?**

## Family 1 - True RL (model generates rollouts, gets scored)
| method | how | cost |
|---|---|---|
| **PPO (classic RLHF)** | train a separate reward model, then PPO with KL penalty to the reference | highest - reward model is a second model of similar size; sample-inefficient; hyperparameter-sensitive |
| **GRPO** | drops the value network; scores a *group* of samples, normalises within the group | much cheaper - what DeepSeek used |
| **RLVR** (Verifiable Rewards) | no reward model - a **program** checks the answer (maths result, unit test) | cheap and objective, only where correctness is checkable |

## Family 2 - Direct preference optimisation (no rollouts, no reward model)
| method | how it differs |
|---|---|
| **DPO** | the key trick: the RLHF objective can be **reparameterised** so you never train a reward model or run PPO. One loop on static (chosen, rejected) data. **~3x less compute than RLHF** for comparable results |
| **IPO** | fixes DPO overfitting on preference pairs; regularises harder toward the reference |
| **KTO** | works with **unpaired** binary feedback (thumbs up/down) - no matched pairs needed. Much easier data |
| **ORPO** | merges SFT and preference into **one** loss - skips the separate SFT stage |
| **SimPO** | reference-model-free variant |

There are **20+ DPO variants** now.

## How to choose
```
Can a program verify correctness? (maths, code, tests)
  YES -> RLVR
  NO
   |- have paired (chosen, rejected) data? -> DPO  (IPO if it overfits)
   |- only thumbs up/down?                 -> KTO
   |- need to exceed the data (reasoning)? -> GRPO   (PPO only with a reason)
```

## Honest caveat
A 2026 controlled study across model scales (arXiv 2603.19335) found **scale-dependent ranking
inversions** - which method wins **changes with model size**. Any "X beats Y" claim from a blog
is only valid at the scale it was measured. **We should measure at our scale, not inherit a
ranking.** Same lesson as our locked-1200 vs free-clock finding in BluBLAS.

---

# PART 8 — THE FULL PICTURE, IN ORDER

```
1. PRETRAINING       self-supervised next-token, trillions of tokens
   v
2. MID-TRAINING      same loss, better data; long-context; domain; prepares RL
   v
3. SFT               <- I am here
   |- pick: full FT / LoRA / QLoRA
   |- pick: human / distilled / self-generated / rejection-sampled data
   |- pick: instruction / chat / domain / reasoning / tool-calling
   |- pick: NLL / chunked-NLL / DFT / PSFT
   v
4. PREFERENCE TUNING  DPO family (cheap) or RL family (GRPO/PPO/RLVR)
   v
5. (optional) more RL  reasoning RL with verifiable rewards
   v
6. (optional) merging, distillation, quantization/QAT, safety passes
   v
   SHIPPED MODEL
```

---

# PART 9 — HOW I THINK WE SHOULD PLAN IT

**Do not try to do all of it.** Sensible order for us:

1. **SFT only, end to end.** One model that follows instructions. This alone is a complete,
   useful result and covers most real use cases.
2. **Measure it properly** - IFEval for instruction following, plus MMLU/GSM8K to catch what it
   forgot. Two-sided, same as our kernel work.
3. **Then DPO**, if we need quality beyond what the SFT data contains. Cheapest next step, no
   reward model, no rollouts.
4. **GRPO / RLVR only if** we have a verifiable-reward task (maths, code) and a real reason.

**Why this order:** each stage is a working model on its own. Stop after step 1 and we still
have something. Start at step 4 and we have nothing for months, and it probably collapses.

---

# SOURCES
- Mid-training notes - https://cameronrwolfe.substack.com/p/midtraining-notes
- OctoThinker: Mid-training Incentivizes RL Scaling - https://arxiv.org/pdf/2506.20512
- A Survey on Post-training of LLMs - https://arxiv.org/pdf/2503.06072
- DeepSeek-R1 recipe - https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1
- SFT-then-RL outperforms mixed-policy - https://arxiv.org/pdf/2604.23747
- Do Post-Training Algorithms Actually Differ? (scale inversions) - https://arxiv.org/pdf/2603.19335
- RLHF vs DPO vs IPO vs KTO - https://dev.to/tech_nuggets/rlhf-vs-dpo-vs-ipo-vs-kto-which-alignment-method-should-you-use-ggm
- DPO vs PPO decision guide - https://www.spheron.network/blog/dpo-vs-ppo-rlhf-algorithm-production-llm-alignment/
- New LLM Pre-training and Post-training Paradigms (Raschka) - https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training
- RLHF Book (Nathan Lambert) - https://rlhfbook.com/

---

# Appendix B — original note: BluTrain SFT gap analysis

*Written 2026-09-17. Superseded by §17 and §23 above — in particular the denominator
fix in §23 corrects Step 2 of this note's minimum path. Kept verbatim.*

# SFT in BluTrain — gap analysis against the real code
**Audited: 2026-09-17 | against `blutrain_ours/Tensor-Implementations` + `bluscript_zero.cpp`**

Every line below was checked against the code, not assumed.

---

# PART 1 — WHAT WE ALREADY HAVE (and it is most of it)

| piece | status | where |
|---|---|---|
| training loop (fwd/bwd/step) | **have** | `bluscript_zero.cpp` |
| GPT architecture (RMSNorm, GQA, RoPE, SwiGLU) | **have** | `bluscript_zero.cpp` + `nn/` |
| all forward + backward CUDA kernels | **have** | `src/Kernels/cuda/` (81 .cu) |
| autograd engine + backward nodes | **have** | `src/autograd/` |
| AdamW + WSD LR schedule + grad clip | **have** | `nn/optimizer/` |
| sparse cross-entropy (fwd + bwd, vectorised, save-stats) | **have** | `LossKernels.h`, `LossKernels.cu` |
| dtypes (18) + promotion | **have** | yours |
| memory allocator, pinned, caching | **have** | yours |
| data loader (mmap uint16 shards, per-rank offsets) | **have** | `dist/dataloader_dist.h` |
| checkpoint save / load_latest | **have** | `Distributed_Checkpointing` |
| ZeRO-1 v2, DDP, multi-GPU | **have** | `dist/` (other team) |
| BluBLAS GEMM backend | **have** | ours |
| eval by scoring (hellaswag) | **have** | `hellaswag_eval.h` — forward + logprob, no generation needed |

**The entire compute path is done.** SFT does not need one new kernel for the maths.

---

# PART 2 — WHAT WE DO NOT HAVE (the actual gaps)

## GAP 1 — cross-entropy has NO label masking. **This is the blocker.**

Current signature (`include/ops/helpers/LossKernels.h`):
```cpp
template<typename T, typename T_idx>
void sparse_cross_entropy_forward_cuda(
    const T* logits, const T_idx* targets, T* loss_output,
    int64_t batch_size, int64_t vocab_size, cudaStream_t stream);
```
There is **no mask / ignore_index parameter anywhere in the codebase** — I grepped
`ignore_index|ignore_idx|-100` across `src/` and `include/`: zero hits.

And the trainer calls it with two tensors only:
```cpp
Tensor loss = autograd::sparse_cross_entropy_loss(logits, b.target);   // bluscript_zero.cpp:1003
```

**Without this, every SFT run trains on the user's prompt as well as the assistant's answer.**
That is the 5-15% we discussed, and it is the single highest-value change.

## GAP 2 — the data loader carries only (input, target). No third channel.

`dist/dataloader_dist.h`: both `input_tensor_` and `target_tensor_` are `[B,T]` uint16,
read from one mmap'd stream with a +1 offset. There is no place to put a mask.

## GAP 3 — no tokenizer, and no chat-template machinery, in C++.

Zero hits for `Tokenizer|BPE|tiktoken|SentencePiece`. Data arrives pre-tokenized as uint16
`.bin`. **This is fine** — it means chat templating and tokenisation happen offline in Python,
which is where they belong anyway.

## GAP 4 — RoPE takes a single scalar `pos_offset`, not per-token position IDs.

```cpp
void fused_qk_norm_rope_forward(..., int B, int Hq, int Hkv, int T, int hd,
                                int pos_offset, float eps, bool interleaved, ...);
```
One `int` for the whole batch. **Packing needs a per-token position vector** so positions
restart at each conversation boundary. Blocks packing.

## GAP 5 — attention has no variable-length / `cu_seqlens` support.

Zero hits for `cu_seqlens|varlen|doc_mask|block_mask`. With packing, conversation 3 would
attend to conversation 1's tokens. **Silently wrong output, loss still descends.** Blocks packing.

## GAP 6 — `Module::forward` takes exactly one tensor.

```cpp
virtual Tensor forward(const Tensor& input) = 0;     // include/nn/NN.h:21
```
No attention mask, no position_ids, no labels. Any of those needs a signature change or a
side-channel.

## GAP 7 — no HuggingFace checkpoint import.

Zero hits for `safetensors|from_pretrained`. `convert_ckpt.py` goes **ours -> PyTorch**
(`torch.save` at line 151), not the other way. **To SFT someone else's pretrained model we
need the reverse direction.** To SFT our own pretrained checkpoint, `load_latest()` already works.

## GAP 8 — no text generation / KV cache.

`hellaswag_eval.h` scores by forward + log-prob, it does not generate. Fine for loss and for
scoring-style evals; **not** fine for IFEval / MT-Bench / AlpacaEval, which need real generation.

---

# PART 3 — THE MINIMUM PATH (what to actually build)

Tomorrow's target: **SFT that trains, with correct loss masking, on our own pretrained
checkpoint.** No packing, no LoRA, no generation. Pad instead of pack.

### Step 1 — offline preprocessing script (Python, ~150 lines)
Input: a conversational dataset. Output: two aligned uint16 `.bin` streams.
```
for each conversation:
    text     = apply_chat_template(messages)      # "<|im_start|>user\n...<|im_end|>\n..."
    tokens   = tokenizer.encode(text)
    labels   = tokens.copy()
    labels[: len(prompt_tokens)] = IGNORE          # mask the prompt
    labels[padding positions]    = IGNORE
    pad both to T=4096
    write tokens -> sft_tokens_000001.bin
    write labels -> sft_labels_000001.bin
```
**Use the tokenizer we already tokenise our pretraining data with** — same vocab (50304), same
uint16 format, so nothing downstream changes.

**Sentinel choice:** targets are uint16 and vocab is 50304, so **65535 is free**. Use
`IGNORE = 65535` rather than -100. No dtype change, no extra tensor.

### Step 2 — add ignore to the CE kernels (the only kernel work)
Forward — skip the sample entirely:
```cpp
void sparse_cross_entropy_forward_cuda(..., int64_t vocab_size,
                                       T_idx ignore_index,        // NEW
                                       int64_t* n_valid,          // NEW: count for the mean
                                       cudaStream_t stream);
// in-kernel:  if (targets[row] == ignore_index) { loss[row] = 0; return; }
```
Backward — zero the gradient row:
```cpp
// if (targets[row] == ignore_index) { memset grad_logits row to 0; return; }
```
Then divide the total loss by `n_valid`, **not** by `batch_size`. Getting that denominator
wrong silently scales every gradient.

Touches: `LossKernels.h`, `LossKernels.cu`, `LossOps.cpp`, `LossBackward.cpp`.

### Step 3 — data loader reads two streams
`dataloader_dist.h`: open `sft_labels_*.bin` alongside `sft_tokens_*.bin`, mmap both, and fill
`target_tensor_` from the labels stream instead of `input+1`. Everything else is unchanged.

### Step 4 — trainer
```cpp
Tensor loss = autograd::sparse_cross_entropy_loss(logits, b.target, IGNORE);
```
Plus: load the pretrained checkpoint at start (`ckpt.load_latest`), set LR to **2e-5**
(not 3e-4), epochs 1-2, and **turn checkpointing back on** for the final model.

### Step 5 — verify (do not skip)
1. **Masking works:** run one batch, print `n_valid` vs `B*T`. If prompts are ~half the
   tokens, `n_valid` should be roughly half. If it equals `B*T`, masking is not firing.
2. **Gradient check:** two identical batches differing only in the prompt text. Gradients
   must be **identical**. That is the check that cannot pass by accident.
3. Loss should start around the pretrained model's loss and descend, not start at ln(50304)=10.8.

---

# PART 4 — WHAT TO ADD LATER (in order of value)

| # | item | value | cost |
|---|---|---|---|
| 1 | **packing** (needs GAP 4 + GAP 5) | token utilisation 10% -> 98% | high — per-token position_ids in RoPE + varlen attention |
| 2 | **generation + KV cache** | needed for IFEval / MT-Bench | high |
| 3 | **HF checkpoint import** | lets us SFT other people's base models | medium |
| 4 | **LoRA** | fit bigger models / faster experiments | medium |
| 5 | chunked CE | cuts the `[B,T,50304]` logits memory | you already know this problem |

**Do 1 only after the unpacked version trains correctly.** Packing without varlen attention is
a silent correctness bug, and those are the ones that cost us weeks.

---

# PART 5 — DECISIONS NEEDED BEFORE STARTING

1. **Which base model?** Our own pretrained checkpoint (works today via `load_latest`), or
   someone else's (needs GAP 7 first)? **This decides whether tomorrow is possible.**
2. **Which dataset?** If we already have a tokenizer for flux data, the same script can
   tokenise an SFT set.
3. **Is BluTrain the target at all**, or do we prototype in TRL first to learn the shape and
   port after? Prototyping in TRL costs a day and de-risks the BluTrain build.
4. **GPU?** No box is currently allocated.

---

# PART 6 — THE HONEST SUMMARY

**SFT needs no new maths and no new kernels for the forward/backward path.** We have all of it.

What we are missing is **four small things**, and only one of them is a kernel:
1. an `ignore_index` on the CE kernels — **the only real kernel work**
2. a second data stream for labels
3. an offline chat-template + tokenise + mask script
4. a pretrained checkpoint to start from

Everything else (packing, generation, LoRA, HF import) is optional for a first working run.

**That is genuinely a few days of work, not weeks** — because the hard part (the framework)
is already built.
