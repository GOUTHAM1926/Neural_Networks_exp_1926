# SFT (Supervised Fine-Tuning) — Work Documentation

**Author:** Goutham Reddy (Gautam_1926)
**Team:** Tensor & Ops
**Started:** 2026-09-17
**Status:** Research / scoping phase. No implementation yet.

**Context:** Assignment changed at a meeting on 2026-09-17. Moved off pretraining kernel work
(BluBLAS / matmul, which is now on hold at 99.80% of cuBLAS) onto **SFT research in
post-training**. This document is the record of everything established on day 1.

**Status 2026-09-26:** no SFT code yet. The day went to library correctness fixes found on the way, all pushed
to `dist_side_merge`:
- Bug D and the bf16/fp16 conversions;
- F1 (NaN/Inf stop) for Muon and ZeRO-v2;
- 7 reduction bugs;
- the GPU `as_type` fix.

They are documented in `loss_kernel_dtype_bugs.md` §17.3 and §22–§25. The SFT tasks 1–7 and every other pending
item are listed in that file's §26.

**Status 2026-09-28/29:** still no SFT code. Two more library fixes, tested and pushed to `dist_side_merge`:
- F2, fused gradient accumulation: bit-identical, +0.52% tok/s at W=1 (2 runs per arm);
- LinearBackward: a 3-D input gave the weight no gradient.

They are in `loss_kernel_dtype_bugs.md` §27–§29. On 09-29 everything was also ported to the colleague's
`dist_side_merge_BLUCCL` branch (§30). The SFT tasks 1–7 are still in its §26.

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
24. [Session log — 2026-09-18](#24)
25. [Session log — 2026-09-21: the loss-kernel implementation](#25)

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

---

<a name="24"></a>
## 24. Session log — 2026-09-18

### 24.1 Framework landscape settled (details in §20-§22)

- **Both PyTorch post-training libraries are dead.** torchtune "no longer actively maintained,
  wound down in 2025"; torchforge "Development paused... consolidated in **torchtitan**".
  Verified from the repo READMEs, not blogs. My earlier advice "clone torchforge, skip
  torchtune" was wrong and is corrected in §21.
- **Chosen references:** MAIN = `NVIDIA/Megatron-LM` (same architecture as us: own CUDA
  kernels, tensor+data parallel, and `pretrain_gpt.py` whose docstring is literally
  `"""Pretrain and SFT GPT."""`). Cross-refs = PyTorch ATen `Loss.cu` (already local in
  `pytorch_source/`) + `Liger-Kernel`. Data/interop = `TRL`. Secondary = `torchtitan`.
- **Liger-Kernel audited:** it has **no SFT module** — no trainer, no data handling. Its
  `chunked_loss/` is all preference losses (DPO/CPO/ORPO/SimPO/KTO/GRPO). What it does give us
  is the CE kernel with `ignore_index`, plus a `megatron/` integration showing how a custom CE
  gets swapped into a Megatron-style trainer. Its `NOTICE` credits 7 upstream sources
  (Unsloth, Triton, mgmalek/efficient_cross_entropy, FlashAttention, AutoAWQ, llm.c);
  **nothing is taken from PyTorch** — these are Triton kernels written to replace it.

### 24.2 SFT-before-RL: my day-1 claim was too strong (full evidence in §5)

Re-checked against the papers. DeepSeek-R1 Table 3 shows cold-start SFT **cost** reasoning
(AIME 77.9 -> 59.0, CNMO 88.1 -> 58.0) and **bought** usability (IF-Eval +25.1,
AlpacaEval +25.4). R1-Zero's raw reasoning was already near-final (77.9 vs 79.8). SimpleRL-Zoo
shows SFT-first *lowers* the RL ceiling, monotonically worse with more SFT steps. Chu et al.
give the reconciling condition: *"SFT is necessary for RL training **when the backbone model
does not follow instructions**."* Also retracted in writing: the "80% of business applications"
figure was invented and has no source.

### 24.3 The denominator bug — caught before writing any code (full detail in §23)

Megatron never divides inside its loss function: it returns `sum(losses * loss_mask)` and
`num_tokens` **separately**, so the framework can all-reduce both and divide once, globally.
PyTorch ATen does the same thing — `total_weight` is a returned output, and `size_average` is
a switch.

Our config is **2 ranks x grad_accum 32 = 64 micro-batches**, each with a different unmasked
token count. Dividing per micro-batch and averaging weights a 50-valid-token batch's tokens
**10x more** than a 500-token one. Loss stays finite, the curve descends, maxRelErr is 0 — the
model just learns the wrong objective. Invisible in pretraining because there every token is
valid and the counts are uniform. Touch points went from 5 to 7.

### 24.4 Production BluTrain cloned

    ~/Downloads/sft_research/BluTrain              153 MB, branch master, 0095fee
    └── Tensor-Implementations                     pinned a4c80c1  (372 tracked files)
        └── BluBridge-BLAS                         pinned 35e51a0

All four SFT files confirmed present: `LossKernels.cu` 1660 lines, `LossOps.cpp` 385,
`LossBackward.cpp` 361, `LossKernels.h` 228.

**Cloning gotchas hit, in order:**
1. `--recurse-submodules` with a token in the parent URL does NOT pass the token to submodule
   clones — they use the plain `.gitmodules` URL and fail with "Repository not found".
   Fix: `git submodule set-url <name> https://<TOKEN>@github.com/<org>/<repo>.git` per level.
2. There are **three** levels, not two. BluTrain -> Tensor-Implementations ->
   `kathir-23s/BluBridge-BLAS` (a colleague's personal repo, not the org's).
3. One clone landed in `~/.local/share/Trash/...` because the shell's CWD was inside a
   directory that had been deleted to Trash. Always `cd` with an absolute path first.
4. Tokens end up in `.git/config` and in submodule URLs. `.gitmodules` is the one that gets
   COMMITTED — it was verified clean (0 tokens) before finishing.

### 24.5 Branch plan — why TWO branches

Proven on a throwaway repo and re-confirmed against the real one. BluTrain stores
`160000 commit a4c80c17...` for Tensor-Implementations — a **pointer**, not files. It tracks
**0** files inside the submodule; Tensor-Implementations tracks 372, including all four of
ours. So a branch in the parent alone cannot hold our kernel changes: committing from the
parent reports "no changes added to commit", and committing inside a detached submodule
orphans the commit, which the next `git submodule update` silently discards.

    cd ~/Downloads/sft_research/BluTrain            && git checkout -b sft_gautam
    cd Tensor-Implementations                       && git checkout -b sft_gautam
    # BluBridge-BLAS: leave alone, SFT does not touch it

Push order at the end is **submodule first, then parent** — otherwise nobody can resolve the
commit the parent points at. Merging is likewise two PRs: Tensor-Implementations, then the
pointer bump in BluTrain.

Where each change lands:

    Tensor-Implementations : LossKernels.cu, LossKernels.h, LossOps.cpp, LossBackward.cpp
    BluTrain               : dataloader (masked targets), trainer (ckpt/LR/epochs), Makefile

### 24.6 Working conventions set today

- **Guide-only mode.** The user runs every mutating command; I print commands and do read-only
  inspection. `git config` counts as a mutation, including `--global`.
- **Identity:** global stays personal (`goutham64840@gmail.com`). No commits at all until the
  work is finished; at that point set `user.email=goutham.kumar@blubridge.com` **in BluTrain
  and in Tensor-Implementations separately** (a submodule does not inherit it), then stage,
  commit, push. Because the first commit happens after the switch, no history rewriting is
  needed. Merge permission comes from the **PAT used to push**, not the commit email.
- **Layout:** `~/Downloads/sft_research/` sits outside any repo — deliberately, so a
  `git clean -fdx` in `Neural_Networks_exp_1926` cannot delete it.

### 24.7 Open — first thing tomorrow

1. Create the two `sft_gautam` branches (commands in §24.5).
2. **Decide `ignore_index` vs `loss_mask`.** Recommendation: take Megatron's semantics (mask +
   global denominator) AND PyTorch's in-kernel early-out, so masked rows skip the whole
   50,304-wide softmax — masking becomes a ~2x speedup rather than pure overhead.
3. Apply the 7 touch points from §23.4, including the `N == 0` guard.
4. Clone the 4 reference repos into `~/Downloads/sft_research/`.
5. Still unresolved: a PAT from the BluBridge account is required for the final merge.

**BluBLAS/matmul work remains ON HOLD.**

---

<a name="25"></a>
## 25. Session log — 2026-09-21: the loss-kernel implementation

First day of actual code. Branch `sft_gautam` created in **Tensor-Implementations only**
(the submodule); BluTrain the parent stays on `master` until we touch the dataloader/trainer.
~38 individual edits, every one verified by reading the code back. **Nothing compiled yet.**

### 25.1 Design decision #1 — the `sft` bool was removed

First implementation added `bool sft = false` alongside `int64_t ignore_index = -1`. The user
challenged whether that was the robust design. Checked the references:

- **PyTorch ATen** `nll_loss_forward_no_reduce_cuda_kernel` takes `int64_t ignore_index` and
  compares unconditionally. No flag.
- **Liger** `liger_cross_entropy_kernel(..., ignore_index, ...)`, public API
  `ignore_index: int = -100`. No flag.

Two independent production implementations, neither has a boolean. The bool was removed from
all 5 kernels.

**Why it was wrong, not merely redundant:** `ignore_index = -1` already means "off", so a
separate flag creates two sources of truth and makes ILLEGAL STATES REPRESENTABLE —
`sft=true, idx=-1` (on but masking nothing) and `sft=false, idx=65535` (set but ignored).
It also had a silent-swap hazard: `kernel(..., 65535, true)` compiles clean (65535 -> bool
true, true -> int64 1) and masks token id 1 out of every batch.

### 25.2 The sentinel: why 65535 and not -100

Targets are `UInt16` (`UInt16ShardView` in the dataloader), vocab is 50304.

    uint16 range     0 .. 65535
    real token ids   0 .. 50303
    unused           50304 .. 65535   -> sentinel 65535

`-100` is PyTorch's convention because ITS targets are `int64`, a signed type that can hold it.
Writing -100 into a uint16 stores **65436**, so `65436 == -100` is false forever and masking
silently does nothing.

The parameter default `-1` is a "match nothing" guard: `static_cast<int64_t>(targets[row])` is
always >= 0 for uint16, so -1 never matches. If someone enables SFT and forgets to set the
index, the result is pretraining behaviour, not wrongly-masked data.

### 25.3 The 5 kernels (all in `LossKernels.cu`)

| # | kernel | role | what the guard does |
|---|---|---|---|
| 1 | `sparse_ce_forward_kernel_vec` | fwd, no stats (benchmarking) | `losses[row]=0`, return |
| 2 | `sparse_ce_forward_kernel_vec_save_stats` | fwd, training | `losses=0`, `saved_max=0`, **`saved_sum=1.0f`** |
| 3 | `sparseCEReduce_kernel_optimized` | bwd no-stats, speed only | `d_partial_max=0`, **`d_partial_sum=1.0f`** |
| 4 | `sparseCENormalize_kernel_optimized` | bwd no-stats, **correctness** | zero the grad row |
| 5 | `sparseCENormalize_from_stats` | bwd with-stats, **correctness** | zero the grad row |

**The 1.0f sentinel is mandatory.** Both consumers compute `1.0f/sum`; a 0.0f sentinel gives
`inf`, then `inf*0` = NaN which spreads through the entire gradient tensor. Traced the merge in
kernel #4 by hand: two partials of (max=0, sum=1) merge to sum=2, `inv_sum=0.5`, finite.

**Other properties checked on every guard:** placed after the row-bound check and before any
work; `return` outside any `tid==0` scope so all threads exit; the condition is block-uniform
(depends only on `blockIdx.x`) so no `__syncthreads()` deadlock is possible; zeroing uses a
plain `T` loop, NOT the `float4` path, because kernels #4/#5 are also instantiated for
float16/bfloat16 where a float4 write would run past the end of the row. Coverage simulated:
50,304 / 50,304 elements, no gaps, no OOB.

**The forward early-out is CORRECTNESS, not optimisation.** Kernels #1/#2 later do
`row_logits[target_idx]`. With target 65535 and vocab 50304 that reads 15,231 elements out of
bounds. Verified the guard precedes it: kernel #1 guard at line 92 vs read at 187; kernel #2
guard at 218 vs read at 297.

### 25.4 Wrappers, headers, macros

4 host wrappers threaded (`impl_vec`, `impl_vec_save_stats`, `backward_cuda_vec`,
`backward_cuda_vec_with_stats`), 5 kernel launch sites, 3 instantiation macros = 10 explicit
instantiations (3 + 4 + 3).

Two C++ rules that govern this and will break the build if missed:

1. **A default argument may be written exactly once.** Header declaration gets `= -1`; the
   `.cu` definition must NOT repeat it.
2. **Defaults must be trailing.** Every wrapper ends with `cudaStream_t stream`, so
   `ignore_index` goes AFTER it. Before it, every existing call site fails.

Instantiation macros must list the new type or the code compiles and fails at LINK with an
undefined symbol for that dtype only. `INSTANTIATE_TWO_KERNEL_VEC` has **4** instantiations
(int8/int16/int32/int64), the other two have 3.

### 25.5 Denominator — researched against three references, decision recorded

The question: what do we divide the summed loss by?

    Option A  constant batch_size (today)      uniform weighting, scaled by mask density
    Option B  GLOBAL valid-token count          exact
    Option C  LOCAL n per micro-batch           NON-UNIFORM -> wrong

Option C distorts token weighting by **13.5x** between micro-batches in simulation (our config
is 2 ranks x grad_accum 32 = 64 micro-batches, each with a different valid count).

**What each reference does — read from source, not assumed:**

- **Megatron default** (`calculate_per_token_loss=False`): `output_tensor /= clamp(num_tokens,
  min=1); output_tensor /= num_microbatches` — this is **Option C**. Their default is the
  distorting one.
- **Megatron `calculate_per_token_loss=True`**: accumulates `total_num_tokens`, then in
  `finalize_model_grads.py:697-711` does `broadcast` + `all_reduce(num_tokens, dp_cp_group)`
  and scales the GRADIENTS by `1/clamp(num_tokens,1)` AFTER the reduce-scatter.
- **torchtitan**: loss is `reduction="sum"` (the type is `Literal["sum","none"]` — mean is not
  even expressible). The trainer PREFETCHES all micro-batches for the step, sums
  `microbatch.num_valid_tokens` from the **collator**, all-reduces once
  (`dist_sum_tensor`), and passes one `global_valid_tokens` device tensor into every
  micro-batch. `BaseLoss.__call__`: `if global_valid_tokens is not None: loss = loss /
  global_valid_tokens`.
- **Liger / PyTorch F.cross_entropy**: local count only — they see one call, they cannot do
  better. Liger uses `.item()`, a device->host sync per call.

**Chosen: torchtitan's.** It is the most robust of the three because the count is known BEFORE
the forward runs (no post-hoc gradient rescaling, no interaction with ZeRO reduce-scatter
ordering), it comes from the data loader which already built the mask, it never syncs to CPU,
and it composes with auxiliary losses for free.

Implemented as an optional trailing parameter so pretraining is untouched:

    Tensor sparse_cross_entropy_loss(const Tensor& logits, const Tensor& targets,
                                     int64_t ignore_index = -1,
                                     const Tensor* denom = nullptr);
    // denom == nullptr -> divide by batch_size  (pretraining, byte-identical)
    // denom != nullptr -> divide by *denom      (SFT, global count, device tensor)

`Tensor*` not `int64_t` deliberately — keeps the count on device. torchtitan's own comment:
*"Keep the global token count on device so loss normalization does not introduce a CPU
synchronization in the training path."*

### 25.6 fp32 accumulator — APPLIED BUT ON HOLD

Separate pre-existing issue found while doing the above: the loss chain was typed `T`, so a
bf16 caller squashed to bf16 three times (per-row write, block partial write, final store).

Changed: `d_losses`/`d_partial` to `float*`, kernels #1/#2 and the sm103 kernel take
`float* losses`, and `sum_reduction_kernel` was split to
`template <typename TIn, typename TOut = TIn>` so the partial pass is `<float,float>` and the
final pass `<float,T>` — one cast, matching ATen `Loss.cu:266`
`*output = static_cast<scalar_t>(sh_inputs[0] / acc_weight[0])`. The `TOut = TIn` default keeps
all 22 existing call sites compiling unchanged.

**Then measured it, and the benefit is narrower than first claimed.** Isolating each squash
over 8192 rows:

    squash 1 (per-row -> bf16)      0.28%
    squash 2 (partials -> bf16)     0.35%
    squash 3 (final store -> bf16)  0.58%   <- dominates
    all three combined              0.58%

The final store dominates because bf16 error is RELATIVE, so casting 20443 or 2.4955 costs the
same. Per-row and partial errors are random and cancel (~0.4%/sqrt(8192) = 0.004%). Removing
the final store is impossible: the returned loss dtype must stay `T` because
`LossBackward.cpp:336-341` REINTERPRETS `grad_output` as the logits dtype (unchecked — returning
fp32 while logits are bf16 would give garbled numbers, not a type error).

**And the loss value never reaches the gradient.** `f_scale = static_cast<float>(*grad_output)
* host_scale` — `grad_output` is the incoming gradient (1.0 / grad_scale), not the loss. So the
bf16 loss affects LOGGING only. An earlier claim in this session that it cost "0.31% on every
gradient" was wrong and is retracted.

What the change does still buy: the accumulation is correct by construction rather than by
luck, and it makes the SFT `denom` division safe (fp32 sum ÷ a 241,341 token count has no
rounding). **Parked pending the user's decision.**

Also corrected: an earlier claim that "4 other loss functions in this file already accumulate
in fp32, so we are aligning with convention" is FALSE. Those four
(`categorical_cross_entropy`, `mse_loss`, `mae_loss`, `bce_loss`) are not templated at all —
they are hard-coded `float`-only and never faced the choice. **Sparse-CE is the only templated
loss in the file**, hence the only one where the accumulator question arises.

### 25.7 Two branches, and why — proven, not assumed

BluTrain stores `160000 commit a4c80c17...` for Tensor-Implementations — a POINTER, not files.
It tracks **0** files inside the submodule; Tensor-Implementations tracks 372, including all
four loss files. Demonstrated on a throwaway repo: editing a submodule file and committing from
the parent reports *"no changes added to commit"*, and committing inside a detached submodule
orphans the commit, which the next `git submodule update` silently discards.

Push order is submodule first, then parent, or nobody can resolve the commit the parent points
at. Merging is likewise two PRs.

### 25.8 Cloning gotchas (for the record)

1. `--recurse-submodules` with a token in the parent URL does NOT pass the token to submodule
   clones; they use the plain `.gitmodules` URL and fail "Repository not found". Fix:
   `git submodule set-url <name> https://<TOKEN>@github.com/<org>/<repo>.git` per level.
2. There are **three** levels: BluTrain -> Tensor-Implementations -> `kathir-23s/BluBridge-BLAS`
   (a colleague's personal repo, not the org's).
3. One clone landed in `~/.local/share/Trash/...` because the shell's CWD was inside a
   directory that had been trashed. Always `cd` with an absolute path first.
4. `.gitmodules` is the file that gets COMMITTED — verified token-free before finishing.

### 25.9 Status at end of day

    5 kernels             DONE   ignore_index + masked-row early-out
    4 wrappers            DONE   header default / .cu none / trailing
    3 macros              DONE   10 instantiations
    LossOps .h/.cpp       DONE   API, CPU mask, 2 launches, 2 divides
    fp32 accumulator      APPLIED, ON HOLD
    LossBackward          NOT STARTED   (ignore_index + host_scale)
    global n_valid        DESIGNED, NOT BUILT
    dataloader + trainer  NOT STARTED   (BluTrain repo)
    sm103 masking         SKIPPED       TODO in code; masking is a no-op on B300
    COMPILED              NO

Branch `sft_gautam`, Tensor-Implementations only.

### 25.10 Tomorrow

1. **Build it.** Everything so far is static analysis. Likely failure points: the 10
   instantiations and the `float*`/`T*` boundaries.
2. Decide on the fp32 accumulator change.
3. `LossBackward`: thread `ignore_index`, decide `host_scale`.
4. BluTrain: dataloader emits `num_valid_tokens`; trainer prefetches the step, sums, all-reduces,
   passes one `global_valid_tokens`.

---

## 26. Session log — 2026-09-22: test suite, backward wiring, and the denominator

Three pieces landed today. All validated by one new file,
`Tensor-Implementations/Tests/kernel_tests/test_sparse_ce_ignore_index.cu`
(1085 lines, 136 cases), run on an RTX 3060 (sm_86).

**Final state: 136 cases, 0 SFT failures, 32 diagnostic XFAILs.**

### 26.1 The test file

Built with `make run-cuda FILE=Tests/kernel_tests/test_sparse_ce_ignore_index.cu`.
Same three-arm methodology as `test_RMSNorm_fwd.cu`: (A) fp64 reference, (B) dtype-T
noise floor, (C) the kernel. Pass = `err_kernel <= 2*err_noisefloor + floor`.

Groups: A masking-off regression (9 dtype combos) · B masking-on forward (6 patterns)
· C masking-on backward (both paths, `grad_output != 1`, `host_scale != 1/B`)
· D shapes · E numerical stress · F sentinel semantics · G op level + CPU/CUDA parity
· W wiring gate · X diagnostics for pre-existing defects.

**Anti-vacuity is the design centre.** A masking test that only checks
"masked result == masked reference" passes when masking does nothing, because both
arms are identical in that case. So every masked case also runs a control arm with
masking off and asserts the difference equals *exactly* the loss the masked rows
carried. Grad buffers are poisoned with 7.0 before every launch, so "row is zero"
proves the kernel wrote zero. Masked rows must be **bit-zero**, not within tolerance.

Two of my own diagnostics were vacuous on the first run and had to be fixed:

- `batch_65537` passed while the bug was present. 65537 = 256*256+1, so exactly ONE
  row falls in the dropped partial, and it may itself be masked. Replaced with
  B=131072, which drops half the batch (measured err 5.008e-01).
- `vocab_50303` passed at `ek = 3.723e-06` against a `9.304e-09` noise floor — 400x
  the floor, obviously a defect, but swallowed by a `floor = 1e-5`. Tightened to 1e-7.

### 26.2 Backward wiring — `ignore_index` never reached the gradient

`grep -c ignore_index src/autograd/backward/LossBackward.cpp` returned **0**. The
three backward kernel guards were correct but the caller never passed the value, so
every backward took the `-1` default and masked nothing. Measured before the fix:
`dirty=1609728` = 32 masked rows x 50304 columns, i.e. *every element of every masked
row* received a gradient it should not have.

Fixed by threading it through: `LossOps.cpp` -> the `SparseCrossEntropyLossBackward`
node (`ignore_index_` member, both constructors) -> 6 CUDA call sites + the CPU loop.
`sparse_cross_entropy_backward_cuda` had no such parameter at all and gained one; it
forwards to `sparse_ce_backward_cuda_vec`, which already carried the guards.

The CPU path **zeroes** the masked row rather than skipping it — `grad_logits_2d`
comes from `Tensor::empty`, so skipping would leave garbage.

Group W is now a real gate (not diagnostic): if it goes red, SFT is silently training
on prompt tokens.

### 26.3 The denominator — cross-checked against all three references

Read from local source, not from memory:

| | denominator | scales the gradient too? | target bounds check |
|---|---|---|---|
| PyTorch ATen | `batch_size - num_ignored` (`LossNLL.cpp:288`) | yes — forward emits `total_weight` (`:304`), backward reads it (`:384`) | `TORCH_CHECK_INDEX` CPU, `CUDA_KERNEL_ASSERT` CUDA |
| Liger | `n_non_ignore` (`ops/cross_entropy.py`) | yes, same value both | — |
| torchtitan | `global_valid_tokens` (`components/loss.py:322,357`) | yes — the divide is a tracked autograd op | — |
| ours (before) | `batch_size` | **no** | **none** |

All three divide the loss AND the gradient by the count of non-ignored tokens. We
divided the loss by `batch_size` and the gradient by `batch_size` unconditionally, so
passing a `denom` would have produced gradients scaled wrong by `denom/batch_size`.

**Implemented PyTorch's mechanism exactly:**

- `count_valid_targets_kernel` writes the non-ignored count as a **device scalar**,
  clamped to >= 1 (Megatron guards the same all-masked case with
  `torch.clamp(num_tokens, min=1)`, `pipeline_parallel/schedules.py:339`).
- The final `sum_reduction_kernel` divides by it (stage-1 partials deliberately do
  not, or the denominator would be applied `reduce_blocks` times).
- The backward kernels take `const float *denom` and compute
  `*grad_output / (*denom)` **on device** — the same signature as
  `aten/src/ATen/native/cuda/Loss.cu:441`, `:449`. No host sync.
- The autograd node carries the denominator tensor, so the two passes cannot disagree.
- `denom != nullptr` overrides with the caller's globally all-reduced count
  (torchtitan's `loss = loss / global_valid_tokens`).

**The invariant that made this safe to switch on by default:** with `ignore_index = -1`
and uint16 targets nothing can ever match, so `n_valid == batch_size` exactly and
**every pretraining run is bit-identical**. The change can only alter behaviour where
masking is actually on.

Measured effect: `G/denom_null_masked/cuda` moved `5.769925 -> 11.539849` and now
lands exactly on `denom_tensor_32` (32 of 64 rows masked, so it divides by 32).
CPU and CUDA agree bit-for-bit: `cpu=11.365391 cuda=11.365391`, err `0.000e+00`.

### 26.4 Three pre-existing defects found (verified against `a7da79d^`)

None introduced by the SFT work. Each has a permanent XFAIL case so it cannot be
silently reintroduced.

1. **fp16/bf16 forward faults unconditionally.** `sparse_ce_forward_kernel_vec` checks
   that the ROW BASE is 16-byte aligned, then issues `cp_async_16(&row_logits[j])`
   with `j = tid*4`. The *global address* must also be 16B aligned:
   `j*sizeof(T) % 16 == 0` holds for every tid at 4 bytes, but only for EVEN tid at 2
   bytes. Row 0 of a `cudaMalloc` buffer is always aligned, so the float4 branch is
   always taken — this kernel can never have run for a 2-byte dtype. The GPU proved it
   ("misaligned address"). Same flaw in `sparseCENormalize_kernel_optimized` and, in a
   different form, in `sparseCENormalize_from_stats` (casts `T*` straight to `float4*`).
   **Latent** because the trainer upcasts first:
   `bluscript_zero_wsd.cpp:739,:771` `sparse_cross_entropy_loss(autograd::to_dtype(logits, Dtype::Float32), ...)`.
   torchtitan does the same (`pred.float()`), so the upcast is standard practice.
   *Unmeasured lead:* part of the convert cost may be forced by this bug rather than chosen.

2. **`batch_size > 65536` silently drops loss.** `sum_reduction_kernel` loads exactly
   one element per thread; stage 2 is `<<<1,256>>>` over `ceil(B/256)` partials, so
   partials 256+ are never read. 7 sites in the file. Only corrupts the *logged* loss —
   the backward re-derives from `saved_max`/`saved_sum` and never reads `loss_output`.
   Production runs 8192-16384 tokens per micro-batch, so it does not fire.

3. **`vocab % 4 != 0` double-counts.** On 16-byte-aligned rows the float4 branch
   overruns into the tail loop's range. At vocab 50303 that is 400x the noise floor; at
   vocab 7 it is 6.7e-02. Our vocab is 50304, so it does not fire.

### 26.5 Process note

A patch script used `s.replace(a, b, 1)` with **no assert** on three edits to
`LossOps.cpp`. That section is 2-space indented (it sits outside the `#ifdef WITH_CUDA`)
not the 4-space assumed, so all three silently did nothing — and the script still
printed "CPU path done". A success message that verifies nothing is the same vacuous
pattern the test exists to catch, just in the tooling.

The test named the cause immediately: both failures were exactly 2x, which is 16/8 —
the signature of dividing by `batch_size` instead of `n_valid`. Rule going forward:
assert the pattern matched AND matched exactly once, then grep the file back.

### 26.6 Files changed

| file | change |
|---|---|
| `include/ops/helpers/LossKernels.h` | `count_valid_targets_cuda` declared; `denom` on 2 forward + 3 backward wrappers |
| `src/Kernels/cuda/loss/LossKernels.cu` | count kernel; `sum_reduction_kernel` optional denominator; 2 live backward kernels take `const float*`; 4 instantiation macros |
| `include/autograd/backward/LossBackward.h` | `ignore_index_` and `denom_` members, both ctors |
| `src/autograd/backward/LossBackward.cpp` | ctors, CPU mask + CPU denominator, 6 CUDA call sites |
| `src/autograd/operations/LossOps.cpp` | denominator built on both paths, node carries it |
| `Tests/kernel_tests/test_sparse_ce_ignore_index.cu` | new, 1085 lines |

Only the two live backward kernels moved to the device denominator; the two dead ones
(`sparseCENormalize_kernel`, `sparse_ce_backward_kernel_typed`) keep `float host_scale`.
`INSTANTIATE_TWO_KERNEL` was wrongly edited mid-session and reverted — it instantiates
`sparse_ce_backward_cuda`, which has no such parameter.

---

## 27. The three open items — analysis and conclusions (2026-09-22)

### 27.1 Trainer-side global valid-token count

**The honest position first: today's default is Megatron's default, and that is not
the ideal one.** Worth stating plainly because it is a side effect of §26.3 that is
easy to miss.

Per-step gradient scale chain in `bluscript_zero_wsd.cpp`:

```
per micro-batch:  1 / n_valid_i        <- the loss op (was 1/batch_size before today)
                x 1 / GRAD_ACCUM       <- grad_scale passed to loss.backward()
      summed over GRAD_ACCUM micro-batches
      summed over W ranks              <- ncclReduceScatter, op = sum
                x 1 / W                <- SymmetricShardedOptimizer.cpp:798 "undo op=sum"
```

Three possible normalizations, and what each actually weights equally:

| | formula | weights equally | who does this |
|---|---|---|---|
| old (before today) | `Sum_i sum_i / (batch_size * GA)` | every **token**, masked ones counted as zero-loss | ours until 2026-09-22 |
| current | `(1/GA) * Sum_i (sum_i / n_valid_i)` | every **micro-batch** | Megatron default (`calculate_per_token_loss=False`) |
| exact | `Sum_i sum_i / Sum_i n_valid_i` (all micros, all ranks) | every **valid token** | torchtitan; PyTorch for a single batch |

So the old behaviour had the *wrong denominator* but the *right relative weighting*
between tokens; the current one has the right denominator per micro-batch but gives a
micro-batch with few valid tokens the same weight as one with many. Neither is exact.

**Magnitude.** The distortion scales with the variance of `n_valid_i` across
micro-batches. At B*T = 4096-16384 tokens per micro-batch the valid fraction is
fairly stable, so it is small — the 13.5x figure recorded earlier was for pathological
micro-batches, not this regime. **At `GRAD_ACCUM == 1` and `world_size == 1` the
current behaviour is exactly PyTorch-correct**, because the local count *is* the
global count.

**What the exact fix needs — three changes that only work together:**

1. **Prefetch every micro-batch of the step** before the first backward, sum
   `n_valid` over them, all-reduce across ranks, pass the one number as `denom`.
   This is what torchtitan does and why it prefetches.
2. **`grad_scale = 1.0`** instead of `1/GRAD_ACCUM` — the global denominator already
   spans all micro-batches, so dividing again double-counts.
3. **Cancel the optimizer's `1/W`** — the global denominator already spans all ranks.
   Either multiply the denominator by `W`, or change the optimizer scale (which would
   also affect pretraining, so prefer the former).

Doing any one or two of these alone gives a *wrong* scale, not a partially-better one.

**Conclusion.** Not a bug, unbuilt work, and it belongs with the dataloader piece that
produces `num_valid_tokens` in the first place. The `denom` parameter on
`sparse_cross_entropy_loss` is the finished hook. **Do it as part of the dataloader
work, not before.** Until then: single-GPU / `GRAD_ACCUM=1` SFT is already exact;
multi-micro-batch SFT is Megatron-default, which is shipped behaviour in a major
framework and acceptable as an interim.

### 27.2 Target bounds check

**Reachable — confirmed by tracing upstream, not assumed.** The chain is:
`.bin` file -> `mmap` (uint16) -> `cudaMemcpyAsync` -> kernel `row_logits[target_idx]`.
`dist/dataloader_dist.h` validates file size, divisibility by 2, shard bounds and B/T,
but **never** that a token id is `< vocab_size`. There is no check anywhere between
the file and the indexed read.

So a `.bin` containing any id >= 50304 reads past the row. Concretely: our SFT sentinel
is 65535, and on a 50304-wide row that is 61 KB past it. If `ignore_index` is left at
the `-1` default while the labels carry 65535, the guard does not fire and the kernel
indexes out of bounds. I hit exactly this while writing the test — it is why the
control arm had to be skipped for out-of-vocab sentinels.

This is **not** a redundant defensive check: an upstream caller can still produce the
bad input, which is the bar for adding one.

Cost: one integer compare per row, against a vocab-wide online softmax. Immeasurable.

PyTorch does it on both paths — `TORCH_CHECK_INDEX(t >= 0 && t < n_classes)` on CPU
and `CUDA_KERNEL_ASSERT(index >= 0 && index < n_classes)` on CUDA. The CUDA form
aborts the kernel and kills the context, which is the correct trade: a loud death
beats silent memory corruption and a plausible-looking loss curve.

**Conclusion. Do it — roughly 5 lines, highest value per line of anything left.** It
converts the most likely SFT misconfiguration (sentinel written but `ignore_index` not
set, or a vocab/tokenizer mismatch) from silent corruption into an immediate, named
error. Put the assert in the forward kernels right after the masking guard, and the
`TORCH_CHECK`-style throw on the CPU path.

### 27.3 sm103 / B300 masking

`src/Kernels/cuda/loss/arch/CrossEntropyKernels_sm103.cu` is **163 lines, one kernel,
forward only** (`sparse_ce_fwd_kernel_sm103_save_stats`). The backward is shared, so
it already masks correctly.

The port is a single guard in the same place as kernel #2 — after
`if (row >= batch_size) return;`, writing `losses[row]=0, saved_max[row]=0,
saved_sum[row]=1` — plus one parameter on the kernel, one on the launcher, and three
instantiations. Roughly 15 lines.

Routing: `LossKernels.cu` branches on `get_arch() == ArchFamily::Blackwell_B300` (103).
**Only B300 takes this path.** H200 is `Hopper` (90) and A10/3060 is `Ampere_A10` (86),
so both run the generic kernel that already masks.

The risk if left: on B300 the forward silently does **not** mask while the backward
does. Loss and gradient would disagree, and nothing would report it.

**Conclusion. Cheap enough to just do, and the failure mode is a silent wrong answer,
which is the worst kind.** Not blocking anything on H200. Do it in the same pass as
the bounds check (§27.2), since both touch the same forward kernels.

### 27.4 Should we validate on the 1x H200?

**For the kernel work: no, it would re-run identical code.** The only arch branch in
the loss path is `== Blackwell_B300`. H200 resolves to `ArchFamily::Hopper`, so it
takes the same generic kernels, line for line, as the sm_86 box the 136 cases already
passed on. An H200 run of the same test proves nothing new about correctness.

Three things an H200 run *would* add, in order of value:

1. **Production-shape validation.** B*T = 16384 at vocab 50304 in fp32 is 3.3 GB of
   logits plus 3.3 GB of gradient — tight on the 3060's 11.6 GB, trivial on H200.
   Worth one run to confirm behaviour at the real shape, and it also exercises the
   two-stage reduction at production scale. Low risk, ~5 minutes.
2. **An sm_90 compile check.** The code is arch-independent, but it has only ever been
   compiled `-arch=sm_86`. Cheap to rule out.
3. **The end-to-end SFT trainer run.** This is the high-value one — and it is blocked
   on the dataloader / global-count work (§27.1), so it cannot happen yet.

**Conclusion. Not needed now.** The productive order is: finish §27.2 and §27.3 (both
small, both on the same files), then build the dataloader + global count, and *then*
use the H200 for the end-to-end run where it actually tells us something. If the H200
is idle and free, one production-shape run of the existing test is a reasonable
5-minute confirmation — but it is confirmation, not discovery.

### 27.5 Recommended order

| # | item | size | why this position |
|---|---|---|---|
| 1 | target bounds check (§27.2) | ~5 lines | highest value per line; prevents the most likely misconfiguration |
| 2 | sm103 masking guard (§27.3) | ~15 lines | same files as #1; silent-wrong-answer risk |
| 3 | dataloader emits `num_valid_tokens` | new work | unblocks everything below |
| 4 | trainer global count (§27.1) | 3 coupled changes | needs #3 |
| 5 | end-to-end SFT run on H200 | — | needs #3 and #4 |

---

## 28. Session log — 2026-09-23: bf16 vector path, dead code, fp32 loss scalar

Three pieces landed, all validated by `test_sparse_ce_ignore_index.cu`.
**Final state: 136 cases, 0 SFT failures, 2 diagnostic XFAILs** (down from 32).

**None of this is SFT-specific.** All of it fires on pretraining too — the only
SFT-only work is the `ignore_index` masking and the valid-token denominator from
§26. Worth stating plainly because it changes who benefits: production carries every
one of these defects today.

### 28.1 Bug B — the bf16/fp16 vector path

The 16-byte vector accesses hardcoded **4 elements** regardless of `sizeof(T)`. That
is correct only for a 4-byte type. For a 2-byte type it did two things wrong at once:

- **misalignment** — the per-thread address is `row_base + j*sizeof(T)` with
  `j = tid*4`, so at 2 bytes it lands on `tid*8`, not `tid*16`.
  `cp.async.cg.shared.global ... 16` requires 16-byte alignment and the kernel died
  with "misaligned address". The guard `(row_logits & 0xF) == 0` was **vacuous**: it
  checked the ROW BASE while the access is per-thread.
- **type pun** — the staging buffer was `extern __shared__ float`, so 16 bytes of
  bf16 (8 values) were read back as 4 floats. Garbage even if aligned.

Fix: `LossVec<T>` with `WIDTH = 16 / sizeof(T)` — 4 for fp32, 8 for bf16/fp16, 2 for
fp64 — typed staging, `alignas(16)`, and `static_assert`s so the compiler checks every
instantiation is exactly 16 B. Deriving the count from the dtype makes the per-thread
byte offset `v*16` for every type, so **one change fixed both faults**; they were the
same mistake seen from two angles.

Applied to all 5 live kernels: `sparse_ce_forward_kernel_vec`,
`sparse_ce_forward_kernel_vec_save_stats`, `sparseCEReduce_kernel_optimized`,
`sparseCENormalize_kernel_optimized`, `sparseCENormalize_from_stats`.

Reference for the pattern: our own `AddVecTraits` in
`src/TensorOps/cuda/TensorOpsAdd.cu` (VEC=4 float, VEC=8 half/bf16) and PyTorch's
`template <typename scalar_t, int vec_size> struct alignas(sizeof(scalar_t)*vec_size)
aligned_vector { scalar_t val[vec_size]; }` (`MemoryAccess.cuh:182`), plus
`can_vectorize_up_to<scalar_t>()` (`:509`) which derives the width from the actual
pointer at runtime. Both keep 16 bytes constant and vary the element count.

### 28.2 vocab % 4 — same root cause

The loops ran to `vocab_size` while unconditionally reading 4 slots, so the remainder
was **double-counted** in the forward and, in the two normalize kernels, the trailing
gradient columns were **never written at all** (`sparseCENormalize_from_stats` had no
tail; `sparseCENormalize_kernel_optimized` skipped them behind
`if (j + 3 < vocab_size)`). `sparseCEReduce_kernel_optimized` silently dropped the
remainder from the max/sum. Fixed with an explicit scalar tail in every kernel.

Measured: `vocab_50303` went from `3.723e-06` (400x the noise floor) to `2.337e-08`
(ratio 1.00); `vocab_7` from `6.705e-02` to `7.774e-09`, and its `removed` figure from
`92.0434/89.5021` to an exact `89.5021/89.5021`.

### 28.3 Dead code — 469 lines

Five exported wrappers and the four kernels only they launched:
`sparse_cross_entropy_forward_cuda`, `..._impl` (+ `sparse_ce_forward_kernel_typed`),
`..._vec`, `sparse_ce_backward_cuda` (+ `sparseCEReduce_kernel`,
`sparseCENormalize_kernel`), `..._backward_cuda_impl` (+
`sparse_ce_backward_kernel_typed`). Plus 3 instantiation macros and 4 header decls.

Not just clutter: `sparse_cross_entropy_forward_cuda` had the plainest name in the
header, so it was the most likely thing for someone to call, and it routed to the
3-squash kernel with **no `ignore_index` parameter** and **no `denom`** — three silent
regressions from calling the most natural-looking function.

**The near-miss worth remembering:** `sparse_cross_entropy_forward_cuda_impl_vec`, the
function `LossOps.cpp` actually calls, had **no explicit instantiation**. It existed
only as a side effect of the dead `..._vec` wrapper calling it. Deleting that wrapper
removes the symbol `LossOps.o` needs. Caught by grepping for
`template void ..._impl_vec<` and finding nothing. Added `INSTANTIATE_FWD_IMPL_VEC`
for the 9 combos LossOps dispatches. **The live production forward had been linking
by accident for as long as that dead code existed.** Before deleting an exported
symbol, check what its removal de-instantiates, not just who calls it.

### 28.4 fp32 loss scalar — and the bug it removed

`LossBackward.cpp` bound `grad_output`'s C++ type from the LOGITS:

    using LogitT = ...decltype(logits_ptr)...;
    ..._with_stats<LogitT, ...>(..., grad_out_ptr, ...);

The trainer hands `backward()` an fp32 `grad_scale`. With bf16 logits the loss is
bf16, so that fp32 value was read as bf16 — silently, because `Tensor::data<T>()` is a
raw `reinterpret_cast` with no dtype check. Wrong gradient scale, nothing reported.

This **predates the SFT work** — all three ingredients are in `a4c80c1`
(`LossBackward.cpp:302-303`, `LossOps.cpp:295`, and the fp32 `grad_scale`). Bug B was
masking it: bf16 crashed in the forward before the backward could ever produce a wrong
scale. Fixing Bug B made an older bug reachable.

Fix: `loss_output` and `grad_output` are now **fp32 by declaration** for every logits
dtype. One element each, so fp32 costs nothing, and the type is in the signature
rather than inferred from a neighbouring tensor — a mismatch is a **compile error**.
Plus a boundary conversion (`as_type`) in `LossBackward` if a caller hands in a
non-fp32 gradient, since the trainer supplies `grad_scale` itself.

### 28.5 The two valid designs, and which we chose

| | design | who |
|---|---|---|
| (1) | assume the type, but **check** it | **PyTorch** |
| (2) | **don't assume** | **Liger, Megatron** |

**PyTorch** — `Loss.cu:567-580` binds `scalar_t` from `input.scalar_type()` and then
reads `grad_output.const_data_ptr<scalar_t>()`. Structurally identical to what we did,
but `const_data_ptr` calls `check_type()` (`TensorMethods.cpp:22`) which **throws**.
Their engine also validates a user-supplied gradient against the output's dtype
(`torch/autograd/__init__.py`, `out_dtype` at :119, enforced at :180/:209).

**Liger** — `_input * grad_output.unsqueeze(dim=1)` (`ops/cross_entropy.py:483`) and
`tl.load(grad_output_ptr)` (`ops/utils.py:135`). Triton reads the pointer's own dtype;
nothing is asserted.

**Megatron** — `grad_input.mul_(grad_output.unsqueeze(dim=-1))`
(`core/tensor_parallel/cross_entropy.py:19`). Plain tensor op.

We were doing **neither** — assuming without checking, the one combination nobody
ships. Declaring the two scalars as `float*` puts us in category (2), reached through
C++ types instead of a dynamic framework.

Chose (2) because **(1) detects the bug at runtime; (2) makes it unrepresentable at
compile time.** Also (2) is free, testable by the existing suite, and delivers the
precision and memory wins; (1) delivers neither.

### 28.6 Measured result of the fp32 scalar

bf16 loss error dropped **139x**, because the final store no longer rounds:

| row | before | after |
|---|---|---|
| `A/off_baseline bf16` | `1.565e-03` (ratio 1.00) | **`1.127e-05`** (ratio 0.01) |
| `A/off_baseline f16` | `2.007e-04` (ratio 1.00) | **`8.629e-06`** (ratio 0.04) |
| `B/mask_dtypes bf16` | `removed=370.0000` | **`369.1480`** vs expected `369.1512` |

Before, `369.1512` was rounded to `370.0000` — a whole bf16 ulp. Group G and all 26
group-C backward rows are byte-identical to the previous run, which is the check that
the change did not disturb anything else.

### 28.7 Trainer upcast removed

Both call sites in `bluscript_zero_wsd.cpp` now pass logits at their native dtype.
The `to_dtype(logits, Dtype::Float32)` existed **only** because the kernel could not
read 2-byte types.

Expected gain, recorded in the code comment and flagged there as
**CALCULATED FROM TENSOR SIZES, NOT MEASURED** — at B*T=16384, vocab 50304:

| | now | after |
|---|---|---|
| peak memory (saved_logits + grad_logits + the fp32 copy) | 9.211 GB | **3.070 GB** |
| loss-path traffic per micro-step | 18.42 GB | **4.61 GB** |

**Precision is unaffected** — bf16→fp32 is a widening conversion, so reading a bf16
logit in-kernel is bit-identical to pre-converting the tensor.

**These numbers have NOT been measured.** The test calls the loss op directly and
never touches the trainer, so the upcast removal is validated only by the build.
A real training run on the H200 is required before quoting them.

### 28.8 Still open

| item | note |
|---|---|
| measure the upcast removal | needs an H200 training run; the numbers above are arithmetic |
| `batch_size > 65536` reduction cap | stage 2 is `<<<1,256>>>` over `ceil(B/256)` partials; at 131072 half the loss is dropped. Production runs 8192-16384 |
| target bounds check | PyTorch asserts on both paths (`CHECK_INDEX_IN_CLASS`, which drops the `>= 0` half for unsigned targets via `if constexpr`); we have none |
| sm103/B300 masking | one guard, ~15 lines, `TODO(SFT)` in the code |
| option 1 — `check_type` in `data<T>()` | logged as a RESEARCH/FUTURE TASK in `LossKernels.h` with the 4 steps: alias table for the 38 `__nv_bfloat16`/`__half` sites first, hoist the 59 in-loop `data<T>()` calls, keep it out-of-line like PyTorch, gate on a full-repo run |

---

## 29. What is missing to run SFT — environment gap analysis (2026-09-23)

Question asked: can we do a sample SFT run using an open-source HuggingFace model?
Then narrowed to the realistic version: **our model, our tokenizer, HF only as a source
of SFT text.** Both answered below from the actual repo, not from assumption.

### 29.1 An open-source HF MODEL is not feasible, and SFT is not the reason

Our architecture is Llama-shaped but the vocabulary is not any HF model's:

    d_model 2048, n_layers 36, GQA 16q/4kv, SwiGLU 5632, RoPE NeoX, weight-tied
    vocab_size = 50304          <- GPT-2 tiktoken (50257) padded

A Llama-style body with a GPT-2 tokenizer. No HF checkpoint has that combination —
TinyLlama is 32000, Qwen ~151936, Llama-3 128256 — so the embedding and lm_head are
shape-incompatible with anything downloadable.

And there is no import path at all. Greps across the repo (excluding
Tensor-Implementations): `safetensors` 0 files, `from_pretrained` 0, `gguf` 0. The
four `huggingface` hits are a **dataset cache path**
(`/root/.cache/huggingface/hub/datasets--navingv--flux/...`), not model loading.
`Tensor-Implementations/convert_ckpt.py` maps OUR dtype enum to torch/numpy — it is an
exporter for our checkpoints, not an HF importer.

Fine-tuning a real open-source model is therefore a separate model-import project
(safetensors reader, tokenizer, vocab/architecture mapping, numerical parity check
against HF's own forward). Independent of SFT and much larger.

### 29.2 Our model + our tokenizer + HF text IS feasible — here is the delta

**How pretraining works today** (`dist/dataloader_dist.h::next_batch`):

    const u_int16_t* tokens = shard_.data_ptr() + pos_;   // flat uint16 stream, no header
    memcpy(pinned_x_, tokens,     BT * 2);                // input  = tokens[0 .. BT)
    memcpy(pinned_y_, tokens + 1, BT * 2);                // target = tokens[1 .. BT+1)
    pos_ += BT;                                           // walk the stream, no boundaries

One flat token river; `target` is `input` shifted by one; nothing marks where a
document starts or ends. `list_shards` globs `*.bin` and the only file validation is
"size divisible by 2".

**The six parts that change:**

| # | part | state | note |
|---|---|---|---|
| 1 | tokenizer | **missing entirely** | zero `*token*` files; `blu16.py` only `np.memmap`s pre-tokenized shards. Must use the SAME GPT-2 BPE that produced the pretraining shards or every id is meaningless |
| 2 | on-disk format | **decision not made** | see below |
| 3 | preprocessing script | **missing** | HF dataset -> chat template -> tokenize -> write shards |
| 4 | dataloader | **must change shape** | two parallel streams instead of one shifted view; step by exactly T (no `+1`); add `num_valid_tokens` to `Batch`; validate ids `< vocab_size` once per shard |
| 5 | trainer | **0 refs** to `ignore_index` or `denom` | 2 lines for a single-GPU run |
| 6 | loss kernels | **DONE** (§26-§28) | — |

### 29.3 The format decision (part 2) — recommended: one example per sequence, padded

The flat stream has no document boundaries, and SFT examples are variable-length and
must not bleed into each other. Simplest correct option for a first run:

    input:   [prompt tokens][answer tokens][pad ...]
    target:  [65535 x len(prompt)][answer shifted][65535 x pad]

**Right-padding is safe with the attention we already have.** Attention is causal, so a
padded position can only be attended BY LATER positions — which are also padding, and
whose loss is masked. No attention mask is needed.

Packing several examples per sequence would need a block-diagonal attention mask, which
we do not have. Padding wastes compute but is correct, and correctness is the point of a
first run. Packing is an optimisation for later.

Note `target` is no longer `input` shifted by one, which is exactly why the dataloader
needs two streams rather than one.

### 29.4 What this does NOT need

No HF model loading, no safetensors reader, no vocab remapping, no architecture mapping.
All of that disappears because we fine-tune our own pretrained checkpoint. HuggingFace
supplies text and nothing else.

### 29.5 Honest position

Roughly **2 of 8 pipeline layers** are done, and they are the bottom two (loss op +
kernels). Everything from "which dataset" down to "the dataloader hands over a mask" has
not started. **SFT cannot run today** — the trainer calls
`sparse_cross_entropy_loss(logits, target)` with two arguments, so `ignore_index`
defaults to -1, nothing is masked, and an "SFT run" would be bit-identical to
pretraining with nothing reporting it.

---

## 30. Session log — 2026-09-24: the loss fixes validated on the H200 production trainer

### 30.1 In one paragraph

The three loss-kernel fixes (Bugs A, B, C) and the removal of the redundant logits cast were
ported onto the colleagues' latest branch, `BluTrain@dist_side_merge`, and measured on a Verda H200
with their production trainer, `bluscript_zero_ckpt.cpp`.

- Their code crashes at step 0 with `BLU_BF16_LOGITS=1`; ours runs.
- With the cast off, the live peak drops by **4,716 MiB at B=4 (≈9.2 GiB at B=8)**, loss-path DRAM
  traffic drops **74.97%**, and throughput rises **+0.60%**.
- Training loss shows **no detectable change**.

Two new findings came out of it: **Bug D**, a bf16 overflow in `Types.h`, and **the trainer is not
reproducible run to run**. Full detail, every command, and every change made on the box are in
`loss_kernel_dtype_bugs.md` §15–§20 and its appendices.

### 30.2 Timeline

**Morning — analysis, no code changes.** Each fact below was confirmed in code before being stated.
- `grad_scale` is created by the trainer as fp32 (`bluscript_zero_ckpt.cpp:1657`, and the
  `TensorOptions()` default in `Tensor.h:24`). It is used exactly once, in the loss kernel
  (`LossKernels.cu:704`).
- **Bug C's root cause:** the old `LossKernels.h` signature used one template parameter `T` for three
  things — logits, grad_logits and the scalar `grad_output`.
- The dtype path from loss to optimizer: `ToDtypeBackward` recasts the gradient
  (`ReshapeBackward.h:41-44`), and the optimizer accepts `float*` only (`MultiTensorKernels.h:9-12`).
- **The cast adds no information.** The matmul output takes its input's dtype (`GenMatmul.cpp:89`), so
  the logits are already bf16, and widening them is zero-padding.
- Weight gradients make a bf16 round trip (`MatrixBackward.cpp:73`). Weight tying does not change the
  logits tensor.
- Wrote `loss_kernel_dtype_bugs.md`, first version, 828 lines.

**Midday — the box.** Verda H200 `cold-bear-flows-fin-02`.
- Cloned `dist_side_merge`. **The branch had moved** since the local copy: `03b94cf`, with submodule
  `8808f89`. The smoke gate re-confirmed all three bugs at that tip.
- Built with sm_90a and BluBridge-BLAS.
- Arm 0: their code with bf16 logits crashed at step 0. **compute-sanitizer named the kernel**:
  `sparse_ce_forward_kernel_vec<bfloat16_t>`, 16-byte reads at 8-aligned addresses, 50,253 errors.
- Arm 1 at B=8 OOMed on one GPU (ZeRO shards nothing at world_size=1), so the A/B moved to B=4.
- The port first failed to link: 6 files were shipped, and `LossOps.h` was missing because committed
  files don't appear in `git status`. With 7 files it linked.
- The first bf16/fp32 runs were void: a co-tenant job took 43 GB, and one run silently resumed from a
  checkpoint with 0 steps.

**Afternoon — controlled measurement.**
- Wrote `test_cast_equivalence.cu` and ran it on the RTX 3060: 0 predictions violated.
- A host test of `float_to_bfloat16` found **Bug D**.
- Ran one exclusive GPU hour with `gpu_hour.sh`:
  - **Block 1** repeated the precision test on the H200, with identical output.
  - **Block 2** ran 4 clean training runs (fp32, bf16, fp32, bf16) with allocator memory stats.
  - **Block 3** measured DRAM bytes with ncu.

**Evening:** documentation (this section, and `loss_kernel_dtype_bugs.md` §16–§20), plus the memory note
`project_bf16_logits_h200_validated.md`.

### 30.3 Results

| | cast ON (fp32 logits) | cast OFF (bf16 logits) | effect | evidence |
|---|---|---|---|---|
| their code | runs | **crashes at step 0** | — | sanitizer |
| our code | runs | **runs, 20/20 steps** | — | 3 separate runs |
| live peak memory, B=4 | 95,038 MiB | 90,322 MiB | **−4,716 MiB** (= 3 × 1,572) | allocator, identical in both reps |
| reserved peak, B=4 | 104,904 MiB | 99,028 MiB | −5,876 MiB | allocator |
| DRAM traffic per micro-step | 19.714 GB | 4.935 GB | **−74.97%** | ncu, 2,500 kernels per arm |
| kernel launches per micro-step | 4 on the loss path | 2 | −2 | ncu |
| throughput | 41,792 / 41,760 tok/s | 42,095 / 41,955 tok/s | **+0.60%** | n=2 per arm |
| gradient into the matmul | — | — | 99.999% bit-identical, max 1 ulp | test B, RTX 3060 = H200 |
| training loss | — | — | **no detectable change** | fp32 vs bf16 ≤ fp32 vs fp32 |

### 30.4 What changed, where

- **`Tensor-Implementations`, branch `sft_gautam`:** new `Tests/kernel_tests/test_cast_equivalence.cu`
  (untracked). **No library code changed today**; the 7 loss files date from 09-21 to 09-23.
- **The box copy of `dist_side_merge`, not in git:**
  - the 7 loss files swapped in;
  - trainer harness edits: B=4, the `[MEM]` allocator print, and the `BLU_VAL_STEPS` override.
  - **Never commit these.** The exact recipe to redo it on any branch is `loss_kernel_dtype_bugs.md` §20.
- **Docs:** `loss_kernel_dtype_bugs.md` is now 1,713 lines.
  - §0, §5.4, §10, §11, §13 and §15.6 are corrected in place, with superseded claims marked.
  - §9 now has the provenance of the false `VEC_ELEMS` claim.
  - §16–§20 and Appendices A–C (`gpu_hour.sh`, `parse_ncu.py`, `watch_hour.sh`) are new.
- **Local artefacts:** `profiling_reports/bugtest_2026_09_24/` holds every log, both ncu CSVs, the
  scripts and the 7-file tarball. This directory is **gitignored**, so it is not pushed.

### 30.5 What this means for SFT

- **The loss side of SFT now runs on the production trainer with both fp32 and bf16 logits.** Caveat:
  `ignore_index` stays at its default of −1 in pretraining, so masking was **not** exercised in these
  runs. Masking itself was validated by the standalone kernel suite on 09-22/23.
- `BLU_BF16_LOGITS=1` is safe with our fixes and frees ~9.2 GiB of live memory per rank at B=8. That
  headroom helps an SFT run's batch or context budget.
- **SFT A/Bs on this trainer need repetitions** (§18 of the loss doc). Single-run early losses are noise.
- The remaining SFT work is unchanged from §29: data path, tokenizer, dataloader with
  `num_valid_tokens`, and a trainer that passes `ignore_index` and `denom`.

### 30.6 Pending after today

- ~~Commit and push `sft_gautam`~~ — **done**: `1ed2ac3` on `BlubridgeAI/Tensor-Implementations` (§30.7).
- **Ask the `dist_side_merge` owners whether a local, unpushed `VEC_ELEMS` fix exists**
  (loss doc §9). If it does, reconcile it with this port before merging.
- Bug D fix (`Types.h:85`), with its own full-repo test run.
- The port as a proper PR to `dist_side_merge`, recommending `BLU_BF16_LOGITS=1`.
- Items in `loss_kernel_dtype_bugs.md` §13.

### 30.7 Evening addendum — push done, and the three casting follow-ups analysed

- **Pushed:** `sft_gautam` → `BlubridgeAI/Tensor-Implementations`, `a7da79d..1ed2ac3`, author
  `goutham.kumar@blubridge.com`. It holds the 7 loss files plus `test_sparse_ce_ignore_index.cu` and
  `test_cast_equivalence.cu`. Checked file by file against what ran on the H200.
- **Applying the fix on a colleague's system:** the 7 file paths and the before/after flag 0/1 run matrix,
  with expected results, are in loss doc §21. **No training-script edits are needed.** The flag
  `BLU_BF16_LOGITS=1` is enough, and it must reach every rank (`mpirun -x`).
- **With the flag on, the cast line costs nothing on the GPU.** `to_dtype` returns early
  (`ReshapeOps.cpp:63-67`), and ncu showed no convert kernel (loss doc §5.5).
- **Four immediate future tasks** — three casting follow-ups and one pending measurement (loss doc §19;
  nothing changed yet):

  | # | what | status |
  |---|---|---|
  | 1 | `wte` is fp32, but the lm_head uses a bf16 copy, so its gradient goes fp32 → bf16 → fp32. Switch: `BLU_BF16_EMBED=1` (`:189`, `:193`, `:709-710`, `:874`). | it changes precision: needs a code check, then a short A/B, then a long run |
  | 2 | `cast_shard_grad_to_fp32` makes a ≈5.9 GB fp32 temporary per step (W=1), then copies it | small fix; same pattern already fixed in `cast_master_range` |
  | 3 | `data<T>()` never checks dtype (827 sites) | research; option 1 = `check_type()` |
  | 4 | speed gain +0.60% vs 0.79–0.94% predicted from the bytes saved; ~1 ms per micro-step unexplained | measurement pending: per-kernel durations from an unprofiled nsys run (loss doc §19.6) |

- **Correction:** an earlier draft said #1 and #2 "pull in opposite directions". **That was wrong.** They
  touch different weight groups. G0 = 1,472,200,704 bf16 Linear params; G1 = 116,029,184 fp32
  (`wte` + norms). Both counts match the log exactly.
- **Order:** the SFT work comes first. Next is testing the version-0 SFT setup with Hugging Face SFT
  datasets, starting from the model checkpoints.
- **#2 is done on 2026-09-25**, because it is a small fix whose pattern already exists in the same file
  (`cast_master_range`, `ShardedOptimizer.cpp`).
- **⚠️ Bug D — HIGH PRIORITY, also looked into on 2026-09-25.** `float_to_bfloat16` (`Types.h:85`) turns
  every value ≥ 65,536 into Infinity. It is `__device__ __host__`, so it runs **inside GPU kernels** as
  well as on the CPU. Plan (loss doc §17.1): find where it can hit, fix it, test all 2³² floats, then a
  clean rebuild and a full test run.
- **Right after the SFT work**, #1, #3 and #4 are done as training-script optimisations. #1 starts
  with a code-only check of how the embedding backward sums gradients for a bf16 weight; if it sums in
  bf16, stop there.

---

## 31. Session log — 2026-09-25

### 31.1 #1 (`wte` fp32) — code-checked, verdict provisional

- **Question:** the lm_head only ever uses a bf16 copy of `wte`, so is the fp32 `wte` needed at all?
- **Answer from the code** (loss doc §19.3.1):
  - For the lm_head, no. Both ways give the matmul identical bits.
  - But with a bf16 `wte`, the embedding backward sums every occurrence of a token with bf16 `atomicAdd`
    (`EmbeddingKernels.cu:147-148`), while fp32 gets a cooperative fp32 kernel (`:415-426`).
  - Emulated error: ~4% at 1,000 contributions, 7.5–20% at 4,000.
  - The halves sum is one rounding (`Engine.cpp:342`). Micro-step accumulation is already fp32
    (`AutogradMeta.cpp:112-116`).
  - #1 would save only ≈0.08% (estimate).
- **Provisional verdict:** keep `wte` fp32 and do not flip `BLU_BF16_EMBED`.
- **Parked for a deeper re-analysis** (user's call; loss doc §19.3.2). The verdict rests on unmeasured
  assumptions: the gradient model, token frequencies, hardware rounding, and which kernel actually runs.
- **Side findings on the way:**
  - the atomicAdd kernel's comment is wrong;
  - an out-of-range token gives silent garbage on the GPU. This is SFT-relevant: 65,535 is our target ignore
    marker, and it must never reach the input ids;
  - the cooperative kernel runs with only 18 blocks.
- **Emulation source:** `profiling_reports/wte_analysis_2026_09_25/emb_bf16_sum.cpp` (gitignored).

### 31.2 Next

#2 (`ShardedOptimizer::cast_shard_grad_to_fp32`), then Bug D.

---

## 32. Does SFT need attention changes? — researched 2026-09-25

**Method:**
- 4 researchers: our kernels (library `8808f89`, BluTrain `03b94cf`); TRL; torchtitan + Megatron (local clones); the web.
- Then a synthesis that re-read every code claim.
- Raw report: `Neural_Networks_exp_1926/profiling_reports/sft_attention_research_2026_09_25.json`.
- **No numeric test has been run yet.**
- **Question:** colleagues (who build the post-training environment) said attention must be masked so that one prompt
  cannot see another.

### 32.1 Verdict

- **v1 (one example per sequence, RIGHT-padded): no attention change is needed.** The colleagues' requirement already
  holds.
- **The colleagues are right for packing** (several examples in one row), padding-free/flattened batches, and
  left-padded RL/generation batches. There a block-diagonal ("document") mask is **required**.

Why v1 is safe, from the code:

| fact | evidence |
|---|---|
| The trainer calls one attention op: GQA `scaled_dot_product_attention(q,k,v, is_causal=true)`, `[B,H,T,hd]` bf16. On H200 that is forward **v62**, backward **Vj1d** by default (Vj1 with `USE_FLASH=1`, v44 with `USE_DETERMINISTIC_DQ=1`) | `bluscript_zero_ckpt.cpp:639-641`; `GQA_fwd_generic.cu:333-338`; `GQA_bwd_generic.cu:697-704` |
| **Batch rows never attend to each other**: Q offset `(b*Hq+hq)*S`, K/V `(b*Hkv+hkv)*S` | `GQA_fwd_sm90.cu:389,397,434`; `GQA_bwd_vj1d_sm90.cu:424,447,469` |
| Causality = tile skipping + a diagonal-tile element mask. **No** attention/padding/document mask, no `cu_seqlens`, no position ids, no sliding window anywhere | `GQA_fwd_sm90.cu:436,498`; `GQA_bwd_vj1d_sm90.cu:465,496,257-262`; a grep finds none |
| With right padding a real token (i < L) only reads keys j ≤ i < L, so it **never sees a pad**. No fully-masked rows, so no NaN | same lines |
| Pad rows give **exactly zero** gradient (the loss backward writes 0 at ignored rows), as long as pad activations are finite | derivation `dS = P·(dP − D)`; the loss backward zeroes ignored rows |
| RoPE: one scalar `pos_offset` (0) for the whole batch, so each example starts at position 0 | `bluscript_zero_ckpt.cpp:614-616`; all RoPE kernels use `t + pos_offset` |

External evidence (web):
- Every framework checked (TRL, HF, NeMo, LongAlign, Llama 3) adds document masking **only when packing**.
- TRL's SFT collator pads **right**. Left padding is for generation only.

### 32.2 New hazards found (act on them in v1)

1. **Input pad id: use `50256` (EOT).**
   - `65535` only works by accident (`padding_idx=-1` → `(uint16)-1 == 65535` gives a zero row).
   - Ids **50304–65534** make the embedding kernel skip the row and leave **uninitialised memory** (`EmbeddingKernels.cu:43-50`, `:96-107`).
   - So any input id ≥ 50304 must be a hard error in the loader.
   - 65535 is a *label* sentinel only.
2. **EOT must be a target** at the end of each completion (the target of the last answer token = 50256). Otherwise the
   model never learns to stop.
   - The local shards separate documents with 50256: `flux_train_000001` has 74,264 EOT per 100M tokens, one at index 0,
     so EOT goes *before* each document.
   - Not checked against the production shards.
   - Truncating a long example drops its EOT target, so drop or cap such examples.
3. **Never left-pad in training:** no pad mask exists, and it creates fully-masked rows.
4. **Zero valid tokens:** a batch with 0 valid targets gives 0/0 = NaN under a per-micro-batch denominator. Drop such
   examples.
5. **B300 cannot run this model today.**
   - There is no causal attention kernel for `head_dim=128` on sm103. The forward and backward throw "causal attention is
     only implemented for head_dim 64" (`GQA_fwd_generic.cu:340-359`, `GQA_bwd_generic.cu:706-727`).
   - The B300 loss forward also ignores `ignore_index` (the `TODO(SFT)` at `LossKernels.cu:938`).
   - **SFT is H200-only for now.**
   - Found by reading the code; not run on a B300.
6. **Shape limits:** the v62 forward needs `T % 256 == 0` (backward `T % 64`) and `T ≤ 4096` (the RoPE cache length).

### 32.3 Why packing matters (the reason the colleagues want the mask)

- **Padding waste** (web): the average SFT example is ~90–1,800 tokens (Alpaca ~92, FLAN ~387, UltraChat ~568, Tulu-v2 mix
  ~1,759).
- Padding to T = 4096 therefore wastes ~57–98% of the GEMM work, and more of the attention work (useful ≈ (L/T)²).
- **Reported packing speedups:** ~2× (HF: FLAN, 1.4× OrcaMath), ~1.9× (IBM), 3.9–6.7× (Packing Analysis, 70B), up to 6×
  (NeMo).
- **Cost of NOT masking when packing:**
  - pretraining (Zhao et al., ACL 2024, 1.3B): masking cut perplexity 9.17 → 8.41 and raised in-context-learning accuracy
    63.5 → 70.5;
  - Llama 3: the mask had a "limited impact during standard pre-training" but was "important in continued pre-training on
    very long sequences";
  - for SFT, no clean ablation was found.

### 32.4 Plan

| stage | what | attention change | notes |
|---|---|---|---|
| **v1** | one example/row, right-padded with 50256, labels 65535 on prompt+pad, EOT target, ignore_index + global denom wired in the trainer | **none** | preprocess ~150–250 lines, SFT loader ~150–300, trainer wiring ~40–80 (estimates). Refuse SFT on B300 |
| **v1.5** | length bucketing: each micro-batch padded only to its longest example, rounded up to 256 | none | ~2× (LongAlign). Changes GEMM shapes; check our GEMM dispatch |
| **v2** | packing with a **block-diagonal mask**: query i sees key j only if `doc_start[i] ≤ j ≤ i` (per-token `doc_start` [B,T] from the loader). Pack already-shifted (input, target) pairs, so each example's last target = 65535 | forward v62: mask every boundary tile + a **−inf guard** (a later document can see a fully-masked first tile, which is NaN today). Backward Vj1d: a runtime-masked version for boundary tiles. Other kernels throw. Optional per-token RoPE positions (the mask alone isolates; RoPE depends only on i−j) | ~300–500 + 100–200 lines (tile skipping; the producer/consumer tile counts must agree or it hangs). Retune batch/LR |

**Tests that cannot pass by accident:**
- v1:
  - **pad invariance**: an example at T=L (rounded up to 256) vs padded to 4096 gives the same loss/grads;
  - pad-content invariance (bitwise, with `USE_DETERMINISTIC_DQ=1`);
  - **row leakage**: changing row 1 leaves row 0's logits bitwise unchanged;
  - causality;
  - denominator (GRAD_ACCUM 4 vs 1, 2 ranks vs 1).
- v2:
  - kernel vs an fp32 masked reference (boundaries at tile edges / mid-tile / 1-token documents / a document starting
    after the first tile);
  - `doc_start` all-zero is bitwise equal to today's kernel;
  - **document leakage**: changing document A leaves document B's logits bitwise unchanged;
  - packed vs unpacked equivalence.

---

## 33. Immediate future task: remove the redundant lm_head matmul on ignored rows (decided 2026-09-29)

**The waste.** The lm_head (`hidden [N × 2304] × W [2304 × 50,304] → logits [N × 50,304]`) runs on every row, but
the scores of a row whose label is 65535 (prompt, padding) are never used. Our loss kernel already skips the softmax
of those rows (`LossKernels.cu:211-219`); the matmul before it is still computed, and in backward its 2 matmuls
multiply the zero gradient rows.
- lm_head per token: forward 2 × 2304 × 50,304 = 0.232 GFLOP, forward + backward 0.695 GFLOP = **6.3%** of the
  model's 11.0 GFLOP per token (the trainer's own MFU line).
- Logits + their gradient: 2 × N × 50,304 values (bf16 at N = 16,384: 1.65 GB + 1.65 GB).
- **Not the same as the 26 layers:** a prompt token must still go through every layer (answer tokens attend to it).
  Only its lm_head row + loss are waste. Padding is waste in every layer (removed by v1.5 / v2, §32.4).

**What the references do** (local clones, 2026-09-17/18):

| library | lm_head on ignored rows | where |
|---|---|---|
| **TRL SFT** | **dropped before the matmul, by default** (`loss_type="chunked_nll"`) | `trl/trainer/sft_trainer.py:122-232`, `sft_config.py:106-114` |
| Liger-Kernel fused linear CE | computed (chunks save memory, not the ignored rows' matmul) | `ops/fused_linear_cross_entropy.py:156-168` |
| Megatron-LM | computed, then `losses * loss_mask` | `pretrain_gpt.py:255-258` |
| torchtitan | computed, then CE with `ignore_index=-100` | `components/loss.py:33-60` |

TRL's method: `valid = labels != -100`; move valid rows to the front (`argsort`, a static-shape op); run the lm_head
only on chunks of 256 valid rows; each chunk's logits live only during its own forward/backward (gradient
checkpointing, so the lm_head forward runs twice for valid rows); divide by the global valid count
(`num_items_in_batch`); one device-to-host sync per call for the chunk count.

**Plan for BluTrain:**
- Option A first: gather the non-ignored rows of `hidden` → lm_head on `[M × 2304]` → our existing loss → scatter the
  gradient back (zeros elsewhere). Round M up to a multiple of 256 (fill with label 65535) to bound the number of
  GEMM shapes BluBLAS must handle.
- Option B (TRL-style chunks + recompute) only if logits memory becomes the limit.
- Same math, not bit-identical (summation order changes).

**Measure it in all 3 stages, with and without: 6 cases.**

| stage | without | with |
|---|---|---|
| v1: pad every row to 4096 | case 1 | case 2 |
| v1.5: length bucketing | case 3 | case 4 |
| v2: packing + block-diagonal mask | case 5 | case 6 |

The ignored share (and so the saving) is largest in v1 (prompt + padding) and smallest in v2 (prompt only).
Per case: **answer tokens per second** (raw tok/s is misleading, since padding differs per stage), step time, peak
active / reserved memory, lm_head GEMM time (nsys), loss equal within noise. At least 2 runs per arm: the trainer is
not reproducible run to run (loss doc §18).

**When:** right after the first SFT run (task 7), before v1.5.

---

## 34. SFT reference survey: what the best implementations do, and what we were missing (2026-09-29)

**Method:** read from code in the local clones (TRL `478624d`, torchtitan `1e4d33d`, Megatron-LM `8b2f660`,
Liger-Kernel `19d92e7`, all 2026-09-17/18, and the local PyTorch source); Axolotl, LLaMA-Factory, Unsloth and Apple
CCE from their docs. NeMo, ms-swift and OpenRLHF were not read.

### 34.1 Feature by feature

| feature | TRL | torchtitan | Megatron-LM | Axolotl / LLaMA-Factory | BluTrain |
|---|---|---|---|---|---|
| prompt masking (ignore label) | completion-only / assistant-only (`{% generation %}` masks) | `ChatProcessor` masks the prompt | `loss_mask` | yes | **done** (65535) |
| global-count denominator | `num_items_in_batch` | yes | `--calculate-per-token-loss` | yes | in the API; **trainer not wired** (task 5) |
| lm_head only on answer rows | **default** `loss_type="chunked_nll"` (`sft_trainer.py:122-232`) | no | no | — | §33 |
| no full logits in memory | chunks of 256 + recompute | no | no | via Liger | no (bf16 logits halve it) |
| length bucketing | no | no | no | — | v1.5 |
| packing | `bfd`, `bfd_split`, `wrapped` | concat-then-split, first-fit (`components/data/packing.py`) | per-sample packing (`training/datasets/sft_dataset.py`) | multipack / `neat_packing` | v2 |
| document mask when packed | FlashAttention varlen via `position_ids` (padding-free) | FlexAttention `block_causal` (`get_document_mask_mod`, `models/common/attention.py:467`) | `cu_seqlens` (`PackedSeqParams`) | FlashAttention varlen | v2 (v62 / Vj1d) |
| positions reset per document | yes | yes | yes | yes | "optional" (§32.4) |
| label bounds check | — | "labels must be -100 or in [0, V)" (`components/loss.py:139-142`) | — | — | task 6 |
| fused kernels | via Liger | partly | yes | via Liger | fused RMSNorm + RoPE |
| activation checkpointing / offload | both | yes | yes | yes | none |

PyTorch provides both packing building blocks: `torch.nn.attention.varlen.varlen_attn` (`cu_seq_q`, `cu_seq_k`) and
FlexAttention block masks.

### 34.2 Missing from our plan until now

| # | item | reference | why | when |
|---|---|---|---|---|
| 1 | tokenizing the prompt alone must be an exact prefix of the full tokenization; otherwise error | torchtitan `_require_token_prefix` | BPE can merge across the prompt/answer boundary and move the mask by a token | task 3 |
| 2 | exactly one EOT at the end, never duplicated | torchtitan `ChatProcessor` | a doubled EOT teaches the wrong ending | task 3 |
| 3 | packing: no target across documents (the last token of A must not predict the first token of B) | TRL `sft_trainer.py:516` | a silent wrong target at every boundary | v2 |
| 4 | drop over-long examples instead of truncating | torchtitan drops (TRL truncates `keep_start`) | truncation removes the EOT target | task 3 (already our rule, §32.2) |
| 5 | log token accuracy and entropy | TRL | standard SFT health signals, cheap | task 5 |
| 6 | no full logits in memory (chunked or fused CE: TRL, Liger, Apple CCE) | TRL / Liger / CCE | frees ~3 GB at B = 4 for a bigger B | after §33 |
| 7 | activation checkpointing | all | memory for a longer T or bigger B | only if memory limits |
| 8 | loss variants (DFT, NEFTune noise) | TRL / transformers | recipe experiments, not waste | after v1 works |

The global denominator is confirmed by an industry bug: in October 2024 Unsloth and Hugging Face found that
averaging per micro-batch gives wrong losses and gradients under gradient accumulation, and fixed it by dividing by
the whole step's token count (`num_items_in_batch`, transformers PR #34191). That is our `denom`.

### 34.3 The waste, in one list

| waste | status |
|---|---|
| softmax / loss on ignored rows | **removed** (`LossKernels.cu:211-219`) |
| lm_head matmul on ignored rows | §33, 6 measured cases |
| all layers on padding | v1.5 (bucketing), v2 (packing) |
| attention tiles for padding queries | removed by v1.5 / v2 |
| full logits memory `[N × 50,304]` | chunked / fused CE, later (34.2 #6) |
| prompt tokens in the 26 layers | **not waste**: the answer attends to them |

### 34.4 Not verified

- NeMo, ms-swift, OpenRLHF: not read.
- Apple CCE: confirmed that it never materializes the logits and skips negligible gradient elements; whether it
  drops ignored rows before the matmul is not stated on its page.

**Conclusion:** the plan (v1 → v1.5 → v2, plus §33) matches the best references. The new must-haves are 34.2
items 1–5; they fit into tasks 3 and 5 and v2, with no extra stage.

**Sources (web):** [Axolotl multipack](https://docs.axolotl.ai/docs/multipack.html) ·
[LLaMA-Factory PR #4224](https://github.com/hiyouga/LlamaFactory/pull/4224) ·
[Unsloth gradient accumulation](https://unsloth.ai/blog/gradient) ·
[HF: fixing gradient accumulation](https://www.huggingface.co/blog/gradient_accumulation) ·
[transformers PR #34191](https://github.com/huggingface/transformers/pull/34191) ·
[Apple Cut Cross-Entropy](https://github.com/apple/ml-cross-entropy)

---

## 35. SFT task 1: load only the weights of Navin's checkpoint (2026-09-29, code applied locally, not tested)

**Where the work happens:** on `dist_side_merge` itself, in the local clone `~/Downloads/BluTrain_sft` (BluTrain
`909ab75`, library `d96b991`). `sft_gautam` was compared file by file first: its 7 loss files are identical to
`dist_side_merge`; its 2 loss tests were copied into the local clone only (hidden with `info/exclude`) and pass there,
136/136 and 0 predictions violated (`profiling_reports/sft_loss_tests_2026_09_29/`); its reduction notes are comments
only and were not copied. `sft_gautam` was then deleted on GitHub and kept as the tag `archive/sft_gautam` (`1ed2ac3`).

**The checkpoint:** `navingv/curriculum_8btokens` (public), folder `blumodel_step_8000`: step 8000 (8.4B tokens),
loss 2.009818, `last_val_loss` 1.866113, saved by a 4-GPU run with Muon.
- `dump_ckpt_meta.cpp` (uses the library's own `CheckpointMetadata::deserialize`, and reads the 129,127-byte
  `metadata.bin` to the last byte) lists **262 weights, 1,588,229,888 numbers**: exactly our model (d_model 2304,
  26 layers, 18/6 heads, ffn 6144, tied embedding). Order: 26 layers × 10 tensors (q-norm, k-norm, norm, wQ, wK, wV,
  wO, norm, gate_up, down), then `wte`, then `norm_f`, the same as our model's registration order.
- 54 weights are **bf16** (the embedding + 53 RMSNorm scales): Navin trained with `BLU_ALL_BF16=1`. **Decision: SFT
  runs with `BLU_ALL_BF16=1` too**, so every dtype matches and no conversion is needed.
- **All 262 weights are in `data_rank_0.bin`**: its 6,469,008,896 bytes = the weights (3,176,473,088) + the same
  optimizer share every other rank file has (3,292,535,808). Loading the weights needs only `metadata.bin` +
  `data_rank_0.bin` (6.5 GB), not all 16.35 GB.

**The code (`profiling_reports/sft_task1_2026_09_29/patch_init_from.py`, applied: `bluscript_zero_ckpt.cpp` +80 −1,
md5 `ad7c5f6803dd64f252bf7fd9ef6dd690`, diff in `bluscript_init_from.diff`):**
- `BLU_INIT_FROM=<checkpoint dir>`: DCP `load()` with no optimizer, placed right after the model is built and **before
  the optimizers are built**. All 3 ZeRO optimizers copy the model weights into their fp32 masters in their constructor
  (`init_master_from_params`), so the masters start from the loaded weights. The step, epoch, data position and the
  checkpoint's optimizer state are not loaded. A later resume of the run's own checkpoint overwrites everything
  (torchtitan's `initial_load_path` + `initial_load_model_only`).
- No pre-flight compatibility check (the user's decision): DCP already stops on a missing weight or a different shape
  (`[Architecture Mismatch]`, `LoadPlanning.cpp:33-41`), and `sync_back_to_model()` stops on a different dtype
  (`Tensor::copy_`). The requirement is written as a comment.
- `BLU_INIT_VERIFY=1` (test switch): compares every loaded weight in GPU memory with the CRC32 the checkpoint stored.
- The checkpoint block reuses the single-rank process group created for the load at W = 1.

**Next (not done): the H200 test.**
1. Download `metadata.bin` + `data_rank_0.bin` on the box.
2. rsync the clone **without** the laptop's sm_86 build outputs (`Tensor-Implementations/lib/`, `Profiler/lib/`,
   `dist/lib/`, `dist/build/`, `_runner_`); apply the box-only B = 4 harness diff; build with `WITH_BLUBLAS=1`.
3. Run 4 steps with `BLU_INIT_FROM=... BLU_INIT_VERIFY=1 BLU_ALL_BF16=1 BLU_CKPT=0`.
4. Expect `262/262 weights match the checkpoint's CRC32`, and a **step-0 validation loss of about 1.87**, not ~11.

### 35.1 Result: task 2 PASSED (2026-10-06, H200 box `ssh -p 2224`, GPU 5)

**How it was run.** The user ran every command; I gave the commands.
- **Code on the box:** a fresh clone of BluTrain `dist_side_merge` (`909ab75`, TI `d96b991`), identical to GitHub. The laptop's edited `bluscript_zero_ckpt.cpp` was copied in with `scp` (md5 `ad7c5f6803dd64f252bf7fd9ef6dd690`); `git diff --stat` showed only that file, +80 −1.
- **Build:** `make -j dist SM_ARCH=90`, which builds profiler → tensor → dist. That avoids `make all` and its system-wide `ensure-blurun`. Backend cuBLAS, `sm_90a`. The trainer was compiled with the Makefile's own `build-mpi` line.
- **Checkpoint** (HF `navingv/curriculum_8btokens/blumodel_step_8000`, public):
  - `metadata.bin` is 129,127 bytes, md5 `5119b20d…`, the same file as our 09-29 copy;
  - `data_rank_0.bin` is 6,469,415,424 bytes, md5 `62804353…`.
  - The file is 406,528 bytes larger than the shard records add up to; ranks 1-3 are each 32,768 bytes larger. That is padding between records, and the CRC check below proves the data itself matches.
- **Data:** the trainer reads the mixed layout (`BLU_MIX_ROOT/{commoncrawl,code,math}`). There is no team mix data on this box, so the three folders are links to our 4 `flux_*.bin` files. All three domains are therefore the same flux data.
- **Flags:** `CUDA_VISIBLE_DEVICES=5 BLU_INIT_FROM=… BLU_INIT_VERIFY=1 BLU_ALL_BF16=1 BLU_CKPT=0 BLU_MAX_STEPS=4 BLU_MIX_ROOT=… BLU_TOKEN_MAX=50304`, with the default AdamW, B=8, T=4096, `grad_accum` 32.

**Results:**

| Check | Result |
|---|---|
| CRC of every loaded weight | **`262/262 weights match the checkpoint's CRC32`** |
| Load message | `weights only from …/blumodel_step_8000 (its step 8000, loss 2.009818)` |
| Step-0 validation loss (flux val, before any update) | **2.4538**; step-0 train loss 2.3274. Random weights give ~10.8 |
| Out-of-range token ids (`BLU_TOKEN_MAX=50304`) | none |
| Steps 1-3 | train 8.56 → 10.76 → 13.30, val at step 3 is 13.38, no NaN |
| Peak memory (`[MEM]`) | active 125,086 MiB, reserved 127,256 MiB, at B=8 with all bf16 |

**What it means:**
- **The weights-only load works.** The loss is close to Navin's level, so rope_theta 500000 matches. It is not exactly 1.866 because this is flux validation data, not his validation set.
- **The loss jump in steps 1-3 is expected.** A fresh AdamW at the PRETRAINING learning rate (1.2e-3, warmup 1 step) wrecks trained weights. SFT must use its own small LR: §3.10 plans 5e-6 to 2e-5.
- **Memory is the same model** (1,588,229,888 params, 18,176 MB of optimizer state, as in the 09-25 runs). The 09-25 run peaked at 90,322 MiB, but at B=4 with fp32 residual/embedding. Today's B=8 is twice the activations.
- **The same model at B=8 used to not fit:** 147,528 MiB with fp32 logits (loss doc §15.5). Now it fits with ~18 GB spare. Which flag saves how much (bf16 logits vs the bf16 residual stream vs the embedding, ~0.8 GiB) was not measured; an A/B is left for later, as curiosity only.

**Status:** the code is NOT pushed yet. It stays in the laptop's `~/Downloads/BluTrain_sft` (`dist_side_merge`) until the SFT work is complete. The working method: edit only on the laptop; each day the box gets a fresh clone of `dist_side_merge` plus the changed files.

---

## 36. Notes for SFT from 2026-09-30 (full explanations in `embedding_optimizer_zero_1_parallelism.md`)

1. **`rope_theta` must match Navin's checkpoint.** His commit `a3aba3e` (on `dist_side_merge_BLUCCL`) sets
   `rope_theta = 10000` in `bluscript_zero_ckpt.cpp`; our `dist_side_merge` trainer uses 500000. The DCP checkpoint stores
   only weights, so a wrong value loads without any error; only the step-0 validation loss (expected ~1.87, task 1 above)
   would show it. Ask Navin which value his `blumodel_step_8000` run used, before the task 1 test.
2. **Which embedding backward SFT would use.** SFT runs `BLU_ALL_BF16=1` (a bf16 `wte`, to match the checkpoint). On
   `dist_side_merge` a bf16 `wte` uses the old `atomicAdd` kernel: every occurrence of a token in a micro-batch rounds the
   running total to bf16, in a random order. SFT data repeats template tokens and EOT in every example, so repetition is
   high. The colleague's `EmbeddingBwdFused` (only on BLUCCL, only in the fused trainer with `BLUCCL_FUSED_EMB=1`) adds
   every occurrence in fp32 and rounds once. Decide: port it, use an fp32 `wte` (needs a bf16 → fp32 widening when
   loading the checkpoint), or keep; the re-scoped ablation measures it.
3. **If the fused kernel is ported:** its hook ignores the token count it is given (`embedding_fused.cu:1300`,
   `(void)N`) and always uses the B × T fixed at install. With variable-length SFT micro-batches that would read the wrong
   number of ids, silently. It must take the real count first.
4. **F4 is fixed on BLUCCL only** (`149249d`): at 1 GPU the ZeRO optimizers no longer round the gradient to bf16 and back.
   `dist_side_merge` (our SFT branch) still has the round trip at W = 1 until `149249d` is ported.

---

## 37. The SFT pipeline: built, run and tested (2026-10-06 and 2026-10-07)

**In one paragraph:** the whole SFT path now works in our own C++ trainer on the H200. The path is preprocessing → SFT dataloader → trainer SFT mode, starting from Navin's step-8000 weights.
- A full 2-epoch run took the validation loss from **1.4464 to 1.2839**.
- Resume, 2 GPUs and 4 GPUs reproduce the 1-GPU numbers.
- The pretraining path is unchanged, checked bit-for-bit at step 0.
- A Muon learning-rate sweep shows **Muon needs a higher LR than AdamW**. At 5e-5 it about matches AdamW, and at 1.2e-4 it gives the best val of all runs, then overfits in epoch 2.
- Not done yet: evaluation of the model's answers, and the push. The user decided on 10-07 not to push yet.

Code: laptop `~/Downloads/BluTrain_sft` (`dist_side_merge`, base `909ab75`, TI `d96b991`). It is not pushed; a backup is in `profiling_reports/sft_stage0_2026_10_06/code/`, pushed to the docs repo in `fc8e20c`. Logs:
- `profiling_reports/sft_stage0_2026_10_06/` (the 10-06 smoke run and the box logs);
- `profiling_reports/sft_stage0_full_2026_10_07/` (every 10-07 run, the two queue scripts and the Muon sweep).

### 37.1 Data (tasks 3 and 4)

- **Dataset decision (the user's):** use one English-only dataset instead of filtering languages out of a mixed one.
  - `HuggingFaceH4/ultrachat_200k` at revision `8049631c`: MIT licence, English, used for Zephyr's SFT.
  - `train_sft` has 207,865 chats; `test_sft` has 23,110.
- **`sft/prepare_sft_data.py`** (new, 236 lines) does the following:
  - Applies template `tulu-plain-v1`: `<|system|>\n…\n<|user|>\n…\n<|assistant|>\n{answer}<|endoftext|>`.
  - Uses the GPT-2 tokenizer. **No new tokens:** the markers are plain text (6–7 normal tokens each), and `<|endoftext|>` is id 50256.
  - Builds `inputs = ids[:-1]` and `labels = ids[1:]`. Every label that is not an answer token or the end token after an answer is set to 65535 (ignored).
  - Pads inputs with 50256 to T = 4096. A chat longer than T is **dropped, not cut**.
  - Writes `inputs.bin`, `labels.bin` (uint16 [N, 4096]), `valid_counts.bin` (uint32 [N]) and `meta.json`.
- **Stage-0 set:** `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/`.
  - **train:** 5,000 random `train_sft` chats (seed 1234; 10 too long were dropped). It has 4,771,516 trained tokens, which is 23.3% of the positions. Mean length 1,238 tokens, max 4,069.
  - **val:** 500 `test_sft` chats, 444,233 trained tokens.
- **Measured from `valid_counts.bin`:**
  - Per micro-batch of 8 rows, the trained tokens range from 2,920 to 12,729.
  - Inside one optimizer step, the biggest micro-batch has 2.54× (median) to 4.13× (max) the trained tokens of the smallest.
  - This is why the global denominator is needed; see `loss_calculation_in_pretraining_and_sft.md`.

### 37.2 SFT dataloader (task 5)

**`sft/SftDataLoader.h`** (new, 174 lines). The only real change from `DataLoaderLite`: the target comes from `labels.bin`, at the same position, instead of the input shifted by one. What it does:
- Takes one chat per row and gives each rank its rows: micro-batch g of rank r is rows `((g % M)·W + r)·B …`.
- `valid_tokens_ahead(n)` reports the trained tokens of the next n micro-batches, which is what the global denominator needs.
- Checks every file when it opens them: ids below the vocab size, labels below the vocab size or equal to 65535, row counts equal to `valid_counts.bin`, and no row without a trained token.

**`sft/test_sft_dataloader.cpp`** passed **5/5** on the laptop RTX 3060 in 0.8 s. Log: `code/loader_test_2026_10_06.log`.

| Check | Result |
|---|---|
| Data | 625 batches copied back from the GPU are byte-equal to the files. Every trained label is the next input token, and every row ends with 50256. Control: the pretraining target differs at 15,703,484 positions |
| Counts | total 4,771,516 = files = meta. Per-batch count and look-ahead wrong in 0 of 633 and 0 of 625 (including across the epoch end) |
| 2 ranks | 2,496 rows each, 0 overlap, the last 8 rows unused. Step-0 global count: 1 GPU = 2 GPUs = **60,263** |
| Epochs | wraps to row 0 at epoch 1; `reset()` and skip land on the right rows |
| Bad files | 9 cases (wrong T, bad rank, too few rows, missing file, short file, input ≥ vocab, label ≥ vocab, wrong count, row with no trained token) all stop with a message naming the row |

### 37.3 Trainer changes (task 6)

`bluscript_zero_ckpt.cpp`, +154 / −23 on top of task 2. The task-2 + task-6 diff against `909ab75` is in `code/bluscript_zero_ckpt_task2_task6.diff`.

| Piece | Today (pretraining) | SFT mode (`BLU_SFT_DATA=<dir>`) |
|---|---|---|
| Loader | `MixedDataLoaderLite` (not created in SFT mode) | `SftDataLoader` for train and val |
| Loss call | `(logits, target)` | `(logits, target, 65535, &denom)` |
| Denominator | each micro-batch's own count (32,768) | the step's trained tokens over all micro-steps and GPUs (look-ahead, then all-reduce SUM) |
| Backward seed | 1/GRAD_ACCUM | W (ZeRO averages over GPUs) |
| Logged train loss | ÷ GRAD_ACCUM, then ÷ W | a plain sum over micro-steps and GPUs |
| Val loss | per-batch mean, then averaged | the whole val set, summed once and divided once |
| Settings | config | `BLU_SFT_ROWS_PER_STEP` 64, `BLU_SFT_EPOCHS` 2, `BLU_SFT_LR` 2e-5 (min LR/10), `BLU_SFT_WARMUP` 0.03 then cosine, `BLU_SFT_VAL_FREQ` 10, `BLU_SFT_MUON_LR` (required with Muon) |
| Checkpoints | `checkpoints_bluscript` | `checkpoints_sft` by default, so it never resumes a pretraining checkpoint |

Checks before the first run:
- the compile check (`mpic++ -fsyntax-only`, Makefile flags) is clean, and a control copy with a broken call fails as it should;
- the ZeRO path adds no hidden 1/GRAD_ACCUM (`grad_accum_steps` = 1; `reduce_grads_now` only averages).

### 37.4 Runs on the H200 (node `innmi1srh2-p032`; the user ran every command or OKed each queue)

All runs start from `blumodel_step_8000`. Settings: B = 8, 64 rows per step, 2 epochs = 156 steps, LR 2e-5 → 2e-6 (5 warmup steps, then cosine), `BLU_ALL_BF16=1`.

| Run (date, GPUs) | What it tests | Result |
|---|---|---|
| Smoke (10-06, GPU 5, stopped at step 20) | does SFT run at all | val 1.4464 → 1.3528 → 1.3317; step-0 count **60,263** and val count **441,382**, both exact |
| Full run (10-07, GPU 4) | a full 2 epochs plus checkpoint saves | val **1.4464 → 1.2839**, falling to the end (no overfit). Train epoch means 1.316 / 1.239. Max norm 0.918. 5.3 s/step, 53% MFU, peak 125,026 MiB. Checkpoints at 78 and 156 (22.2 GB each) |
| Resume (GPU 4) | restart from the step-78 checkpoint | step 78 **exact** (train 1.264598, 58,076 tokens); later steps within ~2e-4; final val 1.283921 vs 1.283946 |
| Pretraining check (GPU 4) | is the pretraining path unchanged | step 0 **exact** against 10-06 (train 2.327407, val 2.453832); steps 1–3 track it |
| 2 GPUs (4,5) | count all-reduce, seed W = 2 | step 0 identical, **including grad norm 0.7442**; exact counts to step 77; final val 1.283918; 2.68 s/step (2×) |
| 4 GPUs (3–6) | seed W = 4 | step 0 identical (train 1.450003, norm 0.7442); 1.35 s/step (4×); 102 GB per GPU. Val uses 480 rows (426,157 tokens): 1.4432 → 1.2809, the same 0.162 drop as on 1 GPU |
| Muon 2e-5 (GPU 4) | Muon with today's LR | runs fine (no rejected NS updates), but val 1.3192, slower than AdamW |

**Run-to-run noise:** step 0 is bit-identical between runs; after the first update the val values differ by about 2e-5.

**Why the 2-GPU and 4-GPU runs differ from 1 GPU from step 78:** the loader makes each GPU's epoch `N / (B·W)` micro-batches.
- On 1 GPU an epoch is 625 micro-batches, not a whole number of 8-micro-batch steps, so epoch 2 starts one micro-batch into step 78.
- On 2 and 4 GPUs, epoch 2 starts exactly at step 78.

This is by design, not a bug. It is fixable later by making every epoch a whole number of steps.

### 37.5 Muon learning-rate sweep (10-07, GPUs 2–5)

Muon and the AdamW part get the **same** LR, like Kathir's `sft.py`. The reference is the 4-GPU AdamW run (same 480 val rows). Script: `run_muon_sweep_1007.sh`.

| Run | val @10 | val @40 | val @80 | best val (step) | final val | train epoch 1 / epoch 2 |
|---|---|---|---|---|---|---|
| AdamW 2e-5 | 1.3497 | 1.3011 | 1.2860 | 1.2809 (155) | **1.2809** | 1.316 / 1.239 |
| Muon 2e-5 | 1.4364 | 1.3757 | 1.3330 | 1.3162 (155) | 1.3162 | 1.384 / 1.295 |
| Muon 5e-5 | 1.4121 | 1.3192 | 1.2920 | 1.2843 (155) | 1.2843 | 1.340 / 1.208 |
| Muon 1.2e-4 | 1.3584 | 1.2972 | **1.2747** | **1.2747 (80)** | 1.2820 | 1.314 / **1.084** |

What it shows:
- Muon needs about **2.5–6× AdamW's LR**. The likely reason: our Muon scales each update to RMS 0.2 × LR, like a settled AdamW, while a fresh AdamW takes much bigger first steps. This is not proven.
- At 1.2e-4 the best val of any run comes at the end of epoch 1. In epoch 2 it overfits: train falls to 1.08 while val goes back up.
- On 5,000 chats: **2 epochs at about 5e-5–8e-5** for now. The 1.2e-4 best (step 80) is a mid-run point with the LR still at 6.6e-5, not a finished 1-epoch run, so **a real 1-epoch run at about 1.2e-4 is the next test**. Note: in the sweep the AdamW part (embeddings, norms) used the same LR as Muon, and each run decays to one tenth of its LR.
- Caveat: one seed, small data, val loss only (no benchmarks).

### 37.6 Muon or AdamW for SFT: what the papers actually say (checked in the full text, 10-07)

| Paper | Pretraining optimizer | SFT optimizer, as written |
|---|---|---|
| Moonlight, arXiv 2502.16982 (Moonshot) | Muon | §3.5, Table 6: **Muon-pretrained + Muon-SFT is the best** of the 4 combinations (MMLU 55.7 / HumanEval 57.3 / MBPP 55.6 / GSM8K 68.0; Muon→AdamW 50.2 / 52.4 / 55.2 / 64.9). SFT on the Tülu 3 mix, 2 epochs, LR 5e-5, linear to 0 |
| Kimi K2, arXiv 2507.20534 | MuonClip (15.5T tokens) | §3.1: "We employ the Muon optimizer in our post-training and recommend its use for fine-tuning with K2"; Muon for RL too |
| GLM-4.5, arXiv 2508.06471 | Muon (not for embeddings, biases or RMSNorm; update RMS 0.2) | not stated |
| Qwen2 / Qwen2.5 / Qwen3 | **not stated** | not stated (Qwen2.5 SFT: >1M examples, 2 epochs, LR 7e-6 → 7e-7; DPO with an "Online Merging Optimizer") |
| Llama 2, Llama 3, DeepSeek-V3, OLMo 2 | AdamW | not named in their SFT sections |
| LIMA, Zephyr, InstructGPT | — | AdamW, AdamW, Adam (stated) |
| Tülu 3 | — | the paper names no optimizer |
| Stanford, arXiv 2509.02046 | Muon vs AdamW, carefully tuned | the speedup shrinks with size: 1.4× at 0.1B, **1.1× at 1.2B** |
| Essential AI, arXiv 2505.02222 | Muon vs AdamW | Muon expands the compute–time frontier and keeps data efficiency at large batch sizes |

Conclusions:
- "All SFT papers use AdamW" is not true. Many do not name the SFT optimizer. Kimi K2 uses Muon for SFT, and Moonlight shows it is best when the base was also pretrained with Muon.
- **Our base was pretrained with Muon, so the evidence points to Muon SFT.**
- Our earlier SFT report said Llama 2 used AdamW for SFT. Its SFT section names no optimizer; AdamW is named only for pretraining and RLHF.

### 37.7 Kathir's PyTorch `sft.py` vs our trainer (read 10-07)

The file is now at `Neural_Networks_exp_1926/kathir_sft/sft.py`. It was moved out of `work_docs`, where it had been put by mistake.

| | Ours (BluTrain C++) | Kathir (PyTorch) |
|---|---|---|
| Same | base step 8000; answer-only loss; summed loss ÷ the whole step's trained tokens; Muon for matrices + AdamW for embeddings and norms; betas (0.9, 0.95), eps 1e-8, weight decay 0.1, clip 1.0 on the AdamW part; 3% warmup; 2 epochs | same |
| Data | UltraChat, 5,000 chats | OLMo 2 SFT mix, 839,168 chats |
| Batch | 64 chats per step, fixed [8, 4096] rows (about 70% padding) | 512 chats per step, sorted by length, cut into micro-batches of ≤ 16,384 padded tokens (little padding) |
| LR, schedule | separate LRs; cosine to 10% | **one** LR (5e-5); linear to 0 (the same recipe as Moonlight) |
| Logits | all positions | **only trained positions, fp32** (Q11 already done in PyTorch) |
| Shuffle | the same order every epoch | a new shuffle every epoch |
| Activation checkpointing | off | on |
| Output | DCP checkpoints | an HF checkpoint at each epoch end (for inference) |
| Eval | val loss | val loss plus manual inference on 2 prompts (base vs epoch 1 vs epoch 2); no benchmarks |

Worth adopting:
1. logits only at trained positions;
2. length-sorted micro-batches (T per micro-batch in steps of 256);
3. a new shuffle every epoch;
4. an HF export for evals.

Needed from Kathir:
- his **prep script** (template, special tokens, max length);
- `zero_muon.py` (Muon scaling);
- `blu_model.py`;
- his **DCP → HF converter**. Ours (`TI/convert_ckpt.py`) reads the old `TNS1` format, not DCP.

### 37.8 Traps found on the box

1. **`mpirun` on one node passes the whole shell environment to the job, whatever `-x` lists.** On 10-06 an old `BLU_CKPT=0` (and `BLU_MAX_STEPS=4`, `BLU_INIT_VERIFY=1`) leaked in from the morning's shell and switched checkpoints off. The queue scripts now unset every `BLU_*` before each run.
2. **A started queue whose parent script dies leaves an orphaned `mpirun`.** The first Muon-sweep start went to the default GPUs 3–6 and lost its parent, so the queue would not have continued. The user OKed killing it by its exact PID, and it was restarted on 2–5.
3. **The val set depends on the GPU count:** 496 rows on 1 or 2 GPUs, 480 on 4. Only compare val numbers across runs with the same GPU count.
4. **Disk:** each SFT checkpoint is 22.2 GB (weights + optimizer state). The duplicates were deleted, and only `stage0_adamw_full/ckpt/blumodel_step_156` is kept, for evals.
5. **Steps 1 and 6 take ~35 s instead of 5.3 s** because of the trainer's `empty_cache()` at steps 0 and 5. It is a colleague's old experiment, and he will remove it.

### 37.9 Status at the end of 10-07

| Part | Tested? |
|---|---|
| Prep script, loader, weights-only load | ✅ |
| SFT training: 1 GPU, full 2 epochs, checkpoint save, resume | ✅ |
| 2 GPUs, 4 GPUs | ✅ |
| Pretraining unchanged | ✅ (step 0 bit-exact) |
| Muon in SFT mode | ✅ works; the LR needs tuning (37.5) |
| The model's real answers and benchmarks | ❌ needs the HF converter and inference (ask Kathir) |

Next:
1. Get Kathir's converter and files.
2. Run his 2 prompts and real benchmarks on `blumodel_step_156`.
3. Decide the SFT optimizer and LR from evals.
4. Logits only at trained positions, and length-sorted batching.
5. More data (stage 1).

**Push:** not yet (the user's decision, 10-07).

### 37.10 The SFT report now lives in the shared post-training doc (10-07)

- **Where it is now:** the SFT report is the "SFT - what it is" tab of the shared Google Doc *PostTraining - Research and Planning*. The latest download is `work_docs/PostTraining - Research and Planning.docx`.
  - The 10-05 `sft_report.md` export and its PDF were deleted from `work_docs` by the user.
  - The tab keeps sections 1–6, plus the doubts (now numbered 7) and the sources.
  - It drops the old sections 7 (what I did) and 8 (what's next).
- **New section 8 for that tab:** `profiling_reports/sft_report_2026_10_07/sft_doc_section8_1007.md`, pasted by the user. It covers:
  - 8.1 what was built;
  - 8.2 tests and results;
  - 8.3 the Muon sweep;
  - 8.4 the papers;
  - 8.5 what was learned;
  - 8.6 Kathir vs ours;
  - 8.7 the plan;
  - new sources.
- **Pasted on 10-07 (download 21:11):** 8.1–8.5 and the new sources. Not pasted: 8.6 (Kathir vs ours) and 8.7 (the plan). The user had also removed the old summary items and the old 5.5 plan table, so the old 5.6 is now 5.5 and 8.7's references were updated to match.
- **Still out of date in that tab at 21:11:**
  - the settings-table optimizer row (still "open", still says Llama 2 used AdamW);
  - the "(7.5)" and "(7.6)" references in that table, which point to sections that no longer exist;
  - the 6.3 row "a validation split";
  - a stray copy of the last 8.2 table row, pasted as a line below the table;
  - the headings: "3.10 The dataloader design" sits over the preprocessing-script list, and "3.11 The preprocessing script" sits over the settings table.
- **Listed before the user's own edits (most now handled):**
  - Summary: "Where it stands" still says SFT has not run, and all 5 "What I need to decide" items are now decided.
  - Settings table: the optimizer row is still "open"; the batch row says 128 (we use 64); the schedule row says linear (we use cosine).
  - 3.7: "Still to do on the trainer side" is done.
  - 3.9: the dataloader is done, except a per-epoch shuffle and bucketing.
  - 5.5: stage 0 is done, with UltraChat 5k.
  - 6.3: the held-out SFT loss now runs.
- **Heading fix in that tab:** "3.9 The dataloader" appears twice.
  - The second one should be "3.10 The preprocessing script".
  - The table now headed "3.10 The preprocessing script" should be "3.11 Training settings". The text already points to Section 3.11 for the optimizer and NaN rows.

---

## 38. 2026-10-08: SFT Q&A work. Q9 measured; Q7/Q8 researched and written into the shared doc; Q10 packing measured

**In one paragraph:** the plan changed to the user's own 13 SFT Q&A questions. Q9 (how SFT learns only from the answer) was measured on a small model built like ours and written up as `How_learning_happens_in_sft.md`. Q7 (kernel, SFT-data and SFT-model benchmarks) and Q8 (checkpoints, overfitting, dev vs unseen) were researched from the papers and pasted into the shared Google Doc. The three ways of putting chats into rows (v1 / v1.5 / v2, which is Q10) were measured on the stage-0 data. No GPU-box runs that day; everything ran on the laptop CPU.

### 38.1 Plan changes (user decisions)

- **The plan is the 13 questions** in the shared doc's SFT Q&A, answered in depth. Order: Q9, then Q7 + Q8.
- **#5 (evals as a separate task) and the comparison with Kathir's PyTorch script were dropped** ("the questions will cover it"). Evals come up only inside Q7/Q8.
- **Explanations stay in the chat**, with ASCII diagrams and tables, and no analogies. I had published one private Artifact page for Q9; the user rejected that way of explaining.
- **Batching (Q10):**
  - the user first said "skip v1.5, do v2", then revised it the same day to **do v1.5, then v2**;
  - the embedding backward will be **Kruthik's fused kernel**, which the user says is the mandated one, after a fix so it takes the real token count (38.4).

### 38.2 Q9: how learning happens only on the answer tokens (measured)

**Setup:**
- a small model built like ours: 4 layers, d = 64, a tied embedding table of 50,304 rows, RMSNorm, causal attention, SwiGLU;
- fp64 on the laptop CPU;
- one real chat ("What is the capital of France?" → "Paris.") tokenized by `sft/prepare_sft_data.py` (`tulu-plain-v1`), padded to T = 28: 23 real positions, 3 trained, 5 padding.

| Check | Result |
|---|---|
| Loss | only positions 20–22 are scored; N = 3; loss = (10.984 + 10.850 + 10.946) ÷ 3 = 10.927 |
| Logit gradient | rows 20–22 non-zero in all 50,304 columns (−0.3333 on the right token = (p−1)/N); every other row exactly 0 |
| Gradient through the layers | the top layer has only the 3 trained rows; every layer below also has the prompt rows (the answer reads their K, V); padding is 0 in every layer |
| Weight contributions, last layer | the prompt feeds only Wk and Wv (20/20 positions); Wq, Wo and the MLP get 0 from the prompt |
| Weight contributions, layers 1–3 | the prompt feeds every weight (Wq 19/20: position 0 can only look at itself) |
| Embedding table | input side touches 16 rows (the chat's distinct tokens); output side touches all 50,304 rows; the pad row's input side is 0 |
| Pads → random tokens | largest change in any weight gradient: **0.0** |
| Pads → tokens from the chat (" France", "Paris", …) | weight gradients change by **0.0**; logits at chat positions change by 0.0; the pad positions' own logits change by 0.36 (unused) |
| One prompt word changed (France → Japan) | loss 10.9265 → 10.9011; gradients change (largest 0.379) |
| 60 steps with SFT labels vs every position trained | answer 10.93 → 0.065 vs → 0.64; prompt 10.81 → 9.00 vs → 0.47 |
| Why the prompt dropped to 9.00 without being trained | the true next prompt token scores 9.00, the same as any other chat token (9.08); tokens never in the chat were pushed down to 15.56; the answer tokens rose to 2.11 everywhere. So the order of the prompt was not learned |

**Points explained in the chat that day:**
- **A token has two jobs.** Masking removes only its prediction job, not its job as context. Gradient reaches a prompt token only through attention, the only operation that mixes positions.
- **Gradient belongs to a position, not to a token id.** A pad slot adds 0 whatever token it holds.
- **Padding is fully computed but wasted.** Pads read the whole chat, but nothing reads them, so their gradient is exactly 0. Left padding would need a pad mask.
- **Weight decay.** It is 0.1 on all 2-D weights, wte included; only 1-D weights are excluded (`bluscript_zero_ckpt.cpp:1267`). A token used only as padding never learns an input meaning, but its row still moves: it gets the softmax push-down on the output side, plus weight decay. The 47 spare rows (50257–50303) are in the same state.

**Files:**
- doc `How_learning_happens_in_sft.md` (master `~/Downloads/sft_research/` + `work_docs/` mirror), with trusted sources: Lambert's RLHF Book chapter, the HF TRL docs, Llama 2 §3.1, URIAL, the PyTorch CrossEntropyLoss docs, torchtune, Ren & Sutherland ICLR 2025, Ghosh et al. ICML 2024, Shi et al. NeurIPS 2024, Huerta-Enochian EMNLP 2024;
- run records in `profiling_reports/q9_gradient_flow_2026_10_08/` (`q9_flow.py`, `q9_sideeffect.py`, `q9_padsame.py` (re-created and re-run on 10-09, same numbers), the results JSON, and the Artifact page source).

### 38.3 Q7a: which kernels need SFT benchmarks, and the rivals

**Scope** (checked in code):

| Group | Kernels |
|---|---|
| Changed for SFT | the loss kernel (ignore 65535 at `LossKernels.cu:215/:350`, zero gradient row at `:692/:777`, count at `:110`; global denominator; fp32 loss value; bf16 vector path and vocab%4 fixed) |
| Unchanged, but SFT gives them different input | the embedding backward (38.4) |
| New when the features are built | v1.5: new GEMM shapes M = 8·T. v2: block-diagonal attention mask in forward and backward, RoPE restart per chat, a guard for fully-masked tiles. Q11: a gather of trained rows and an lm_head GEMM with irregular M (BluBLAS gives wrong numbers silently when M/N is not a multiple of 64 with a transposed operand). Long-context SFT: fused linear + CE. DFT/focal loss variants |
| No new benchmark needed | RMSNorm, RoPE (v1), SwiGLU, GEMMs at today's shapes, AdamW/Muon, all-reduce; the denominator has no kernel (CPU count + one MPI all-reduce; the library's GPU count kernel at `LossKernels.cu:104` is unused) |

**Where the work goes in one v1 micro-batch** (arithmetic from d 2304, 26 layers, ffn 6144, V 50,304, B × T = 8 × 4096; not measured):

| Part | Work | Wasted |
|---|---|---|
| linear layers | ~289 TFLOP (80%) | ~70% (pad rows) |
| attention | ~48 TFLOP (13%) | up to ~90% (pads sit at the end of each row) |
| lm_head | ~23 TFLOP (6%) | ~77% (prompt + pad) |

- About 73% of the work is wasted.
- The logged 53% MFU counts pads (`tokens = B·T·GA·W`, `bluscript_zero_ckpt.cpp:2666`); on real tokens it is about 14%.
- The loss kernel's backward writes ~3.3 GB of mostly-zero rows per micro-batch, about 0.5 ms out of ~660 ms. For that kernel, correctness and memory matter more than speed.

**Rivals** (their own claims; sources checked on 10-08):

| What | Rivals |
|---|---|
| Loss kernel | Apple Cut Cross-Entropy (Apple research, Triton, Ampere+; Gemma 2 2B loss memory 24 GB → 1 MB); Liger (fused linear CE: >4× memory cut at a 128k vocab; +20% throughput, −60% memory overall); PyTorch `F.cross_entropy` (ignore_index −100, mean over non-ignored targets) |
| Attention | FlashAttention-3 (H100 BF16 up to 840 TFLOP/s; `flash_attn_varlen_func` with forward + backward exists in `hopper/flash_attn_interface.py`, checked); FlexAttention (90% of FA2 forward / 85% backward on A100; +71% packing throughput in torchtune); PyTorch SDPA |
| Embedding backward | PyTorch's embedding backward (sorts the ids for large batches) |
| Whole SFT step | NVIDIA NeMo / Megatron + Transformer Engine ("up to 6×" training time with packing); HF TRL + FlashAttention ("up to 2×" with padding-free packing); torchtitan |

**Framework facts:**
- **torchtitan** is a separate PyTorch-team project, built on top of PyTorch and installed on its own. Its README lists "SFT with chat-formatted datasets".
  - Its loss calls `F.cross_entropy(reduction="sum")`, then divides by the global token count.
  - So the kernel comes from PyTorch, and the averaging recipe from torchtitan (the recipe our trainer follows).
  - Its README doesn't mention packing.
- **torchtune** is no longer maintained ("wound down in 2025").
- **torchforge** (PyTorch's RL post-training library) uses torchtitan underneath.

### 38.4 Embedding backward facts (SFT runs on `dist_side_merge`)

- **Which kernel the SFT runs used:**
  - `BLU_ALL_BF16=1` gives a bf16 wte, which falls to the atomicAdd kernel (`EmbeddingKernels.cu:127-148`); the cooperative atomic-free kernel is fp32-only (`:415`).
  - The trainer passes `padding_idx = -1` (`bluscript_zero_ckpt.cpp:713`).
  - So per micro-batch, about 22,900 pad positions × 2,304 columns ≈ 53M atomic adds of 0.0 land on the same addresses.
  - The template tokens repeat in every chat, which makes the bf16 rounding issue bigger (see the 10-01 D1 test).
- **The user's direction (10-08):** Kruthik's `EmbeddingBwdFused` (BLUCCL, the fused trainer, `BLUCCL_FUSED_EMB=1`) is the kernel the team uses and mandates. It sorts the ids, adds in fp32, rounds once, and uses no atomics.
- **The fix it needs:** the hook ignores the real N (`embedding_fused.cu:1300`, `(void)N`), and `run()` sorts `N_`, fixed at install to B × T (`:1189`; install at `bluscript_zero_ckpt_bluccl_fused.cpp:1154`).
  - That is correct for v1 and v2 (always 8 × 4096 tokens), and silently wrong for v1.5 or padding-free batching.
  - Pass the real N, and size the workspace for the largest N.
- **Do not mention the atomicAdd kernel in EODs or reports** (the user's rule).

### 38.5 Q10: v1 / v1.5 / v2 measured on stage-0

Script: `profiling_reports/q10_packing_sim_2026_10_08/pack_sim.py`, output saved. Re-created and re-run on 10-09, same numbers. Data: 5,000 UltraChat chats; mean 1,238 tokens, median 1,154, p90 2,021, longest 4,069.

| Method | Real tokens | Work vs v1 | Speed-up (upper bound) |
|---|---|---|---|
| v1: one chat per row, padded to 4096 | 30.2% | 100% | ×1.00 |
| v1.5 realistic: shuffle, sort each 64-chat step, pad to the longest ×256 | 79.8% | 35.2% | ×2.84 |
| v1.5 ideal: whole set sorted (not usable) | 90.5% | 30.7% | ×3.26 |
| v2: best-fit-decreasing packing + block-diagonal mask | 99.6% | 27.8% | ×3.60 |
| v2 without the mask | 99.6% | 30.3% | ×3.30, and wrong (chats see each other) |

- **v2:** 1,517 rows, an average of 16.3 pad tokens per row, **1 to 12 chats per row**.
  - Correction: the chat on 10-08 and the shared doc's kernel section said "1–5"; the measured range is 1–12.
  - Never split a chat across rows; chats longer than 4096 are dropped.
- **Row length:** it changes per micro-batch in v1.5 and stays 4096 in v2. "Padding-free" (one flat stream with `cu_seqlens`) is not needed.
- **Fused embedding kernel with fixed N:** works with v1 and v2, silently wrong with v1.5 and padding-free until the real-N fix.

### 38.6 Q7b: how to evaluate SFT data (research summary)

- **The method every lab uses is an ablation:** same base model and settings, only the data changes, same benchmarks.
  - Tülu 1 ("How Far Can Camels Go?", NeurIPS 2023): 12 datasets, no single best.
  - Tülu 3: leave one category out (without math: GSM8K 76.2 → 64.1; without safety: 93.1 → 74.7; without persona IF: IFEval 72.8 → 53.6), a size curve (TruthfulQA falls with more data), and mix vs mix (Tülu 2 mix 48.3 vs Tülu 3 mix 60.1).
  - OLMo 3: add one category at a time; adding math lowered IFEval 61.7 → 54.0.
  - OpenThoughts: 1,000+ experiments.
  - "Fixing It in Post" (NeurIPS 2025): Tülu 3 mix vs SmolTalk on 14 benchmarks, giving TuluTalk (14–23% smaller, same or better).
  - SmolLM2 (1.7B): small ablations plus stage-by-stage mix updates.
- **Fairness rules:**
  - the same number of examples, and the same trained tokens if answer lengths differ;
  - at least 2 runs per arm;
  - a dev suite plus an unseen suite;
  - few-shot for the base model, zero-shot in the chat template for the SFT model.
  - Run-to-run variance: Madaan et al. 2024, with 10 seeds. Benchmark-size noise (one standard error at 50%): IFEval 541 prompts ±2.1, GSM8K 1,319 problems ±1.4.
- **Scoring methods** (AlpaGasus 9k/52k, 5.7× faster; Deita 6k; IFD 10%; LESS 5%; Long Is More, 1,000 longest):
  - "Rethinking Data Selection at Scale" (EMNLP 2025 Findings): at about a million examples they are ≈ random; variety matters more; length filtering helps.
  - So always compare against random and longest.
- **Small proxy models:** not needed (DataDecide's 150M → 1B ranking agreement is ~80%; our runs are cheap: about 4 min per 5k-chat run on 4 H200s).
- **Cleaning:**
  - decontamination: an example is removed if it holds >50% of a test question's tokens in matching 8-grams; a dataset is removed or filtered if >2% of a benchmark's questions overlap;
  - dedup, licence, correctness.

### 38.7 Q7c / Q8: how to evaluate the SFT model and pick checkpoints (research summary)

- **Two kinds of scoring:**
  - **likelihood (cloze):** works for base models; OLMES says small models need the cloze form;
  - **generation:** the SFT skills.
  - OLMo 3 has a separate base suite (OlmoBaseEval, mostly bits-per-byte) and post-training suite.
- **Tülu 3 suites:**
  - **dev:** MMLU 0-shot CoT, PopQA 15-shot, TruthfulQA 6-shot MC2, BBH 3-shot CoT, DROP 3-shot F1, MATH 4-shot CoT flex, GSM8K 8-shot CoT, HumanEval(+) pass@10, IFEval prompt-loose, AlpacaEval 2 LC;
  - **unseen:** AGIEval, MMLU-Pro, GPQA, DeepMind Math, BigCodeBench, IFEval-OOD, HREF;
  - **safety:** six tests.
  - 5 SFT seeds scored 59.8–60.1; the best seed ≈ the best soup.
- **OLMo 3 suite** (Table 16):
  - all zero-shot; temperature 0.6, top-p 0.95, 32,768 tokens; AIME 32 samples, code 10; think traces stripped; mean of 3 runs;
  - evaluation was 10–20% of compute;
  - standard deviation across 3 runs: GPQA 1.48, AlpacaEval 1.24, IFEval 0.88 … MMLU 0.22, PopQA 0.16;
  - BoolQ was removed (too noisy).
- **OLMo 3 SFT selection:** 2 epochs "to avoid overfitting", an LR sweep (4 values at 32B), candidates chosen by the eval suite, then "vibe-tests", then a linear merge of two checkpoints trained with different LRs.
- **Stage tables:**

| Model | Benchmark | SFT → DPO → final |
|---|---|---|
| Tülu 3 8B | average | 60.6 → 64.7 → 65.1 |
| Tülu 3 8B | IFEval | 72.8 → 81.1 → 82.4 |
| Tülu 3 8B | AlpacaEval 2 | 12.4 → 33.5 → 34.5 |
| OLMo 3 Instruct 7B | MATH | 65.1 → 79.6 → 87.3 |
| OLMo 3 Instruct 7B | IFEval | 81.7 → 82.0 → 85.6 |
| OLMo 3 Instruct 7B | AlpacaEval LC | 21.8 → 43.3 → 40.9 |
| OLMo 3 Instruct 7B | PopQA | 16.5 → 20.7 → 14.1 |
| OLMo 3 Instruct 7B | Safety | 89.5 → 89.9 → 87.6 |
| OLMo 2 1B | average | 36.9 → 40.6 → 42.7 |

  - OLMo 3 Think 7B, average (Table 22): SFT 70.1, SFT+DPO 72.7, SFT+RLVR 71.9, SFT+DPO+RLVR 74.1.
  - Other ~1–1.7B models' AlpacaEval (Ai2 table): 5.8 (SmolLM2 1.7B) to 20.4 (Gemma 3 1B).
  - A colleague's 480M model: TruthfulQA, PopQA and DROP rose base < SFT < DPO; MMLU, BBH, GSM8K, MATH and HumanEval stayed at the floor.
- **Forgetting:** run the base model and the SFT model on the same likelihood tests.
  - "LoRA Learns Less and Forgets Less" used HellaSwag, WinoGrande and ARC-C; full fine-tuning forgets more.
  - Tülu 3: TruthfulQA falls with more data.
  - URIAL: SFT mostly changes style tokens.
- **Factors that change scores:**
  - the chat template (Tülu 3: 51.6–53.0 across templates);
  - prompt wording (lm-eval, "Lessons from the Trenches");
  - shots and CoT;
  - answer extraction (Tülu 3 "flex": up to +10 on MATH);
  - decoding, seeds and hardware ("A Sober Look": up to 8%; AIME, 30 questions, moves 3.3 points per question);
  - signal vs noise (Ai2 "Signal and Noise", 30 benchmarks × 465 models);
  - judge biases (MT-Bench: position, verbosity, self-preference, >80% agreement with humans; AlpacaEval LC: 0.94 → 0.98 correlation with Chatbot Arena; Arena-Hard: 500 prompts, 3× separation, 98.6%, ~$20);
  - contamination;
  - the floor at small size;
  - IFBench: models overfit IFEval's constraints and fail its 58 new ones.
- **Dev vs unseen** (explanation agreed with the user):
  - Every choice made by the dev score also picks up luck on those exact questions. Example: the best of 10 truly equal options is about +3.2 points above its real level on IFEval.
  - So the dev score of the final model is too high.
  - An unseen suite, designed separately and run once on the frozen model, gives the honest number.
  - In SFT, dev and unseen are separate benchmark lists, not slices of the training data.
- **Plan for our model:**
  - **dev:** answer val loss, end-token rate, IFEval, TruthfulQA MC2, PopQA, DROP, HellaSwag/ARC/WinoGrande (cloze, base vs SFT), GSM8K as a floor check;
  - **unseen:** IFBench or IFEval-OOD, MMLU-Pro (cloze), AGIEval;
  - **settings:** SFT model zero-shot in `tulu-plain-v1`, base model OLMES few-shot; average 3 runs for noisy generation tests;
  - **depends on** an HF export plus generation.

### 38.8 Shared Google Doc (Q7/Q8 tab)

- **Tab:** `https://docs.google.com/document/d/1nJ4utik9Jm7RT9YuaWD8UXcPpJRAj2WUrXnEdjJ_jlg/edit?tab=t.jzwfpg8vqqer` (readable through its export link).
- **Pasted that day, all in "I" voice:**
  1. SFT-DATA EVALUATION PLAN (Steps 1–9);
  2. SFT – KERNEL BENCHMARKS (Steps 1–10, including v1 / v1.5 / v2 and the rivals with torchtitan);
  3. SFT – MODEL EVALUATION PLAN (Steps 1–9).
- **Final check:** 86 links, 11 tables, no "we/our" (except the user's own question and a Tülu 3 quote).
- **Snapshot:** `profiling_reports/sft_docs_snapshot_2026_10_08/` (docx + txt, downloaded on 10-09).
- **Still to fix in the tab:** the kernel section, Step 5 ("Two small things that change with v2"), says "1–5" chats per row; the measured range is 1–12.

### 38.9 EOD 2026-10-08

- **Content:** the kernel, SFT-data and SFT-model benchmarks, checkpoints and overfitting, dev vs unseen, and the decision to go to v2.
- **Rules learned:**
  - cover only the topics the user's draft lists (no Q9);
  - never write about the atomicAdd embedding kernel in reports;
  - no unexplained paper or method names;
  - plain words.
- **Verda:** no box runs.

### 38.10 Next

1. Kernel benchmarks: the loss kernel vs PyTorch, Liger and Apple CCE at 77% ignored rows (laptop GPU, then H200).
2. v1.5 bucketing (loader, GEMM-shape check, Kruthik's kernel with the real-N fix), then v2 packing (mask forward and backward, RoPE restart, packing loader, correctness vs SDPA).
3. Q11 after v2.
4. The remaining Q&A questions (Q4, Q5, Q6, Q11, Q12, Q13).
