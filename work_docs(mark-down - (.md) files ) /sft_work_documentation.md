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

## 39. 2026-10-09: v2 step 1 (how every library does variable-length attention), attention background, and the loss-kernel benchmark on H200

**In one paragraph:**
- I documented everything up to 10-08 (§38).
- I started v2 (packing + variable-length attention) with step 1: reading the code of every library we cloned to see how they handle packed chats.
  - Every library uses one of two designs: A (each chat as its own sequence) or B (one row plus a block mask).
- Background questions answered from the code:
  - what each library is;
  - whether causal attention skips or masks the upper triangle;
  - the attention types from 2014 to now;
  - FlashAttention vs memory-efficient attention and their precision;
  - which kernels our SFT run uses.
- A colleague asked for numbers on our modified SFT loss kernel, so I wrote a standalone benchmark and ran it on H200 GPU 5. I wrote a separate document for her.

### 39.1 The libraries (what each one is)

All clones are under `~/Downloads/sft_research/`, except PyTorch, which is at `~/pytorch_cloned/pytorch` (commit `5f76830b728`, 2025-12-05, older than the rest).

| Library | Made by | What it is | Built on | Own attention kernel? |
|---|---|---|---|---|
| PyTorch | Meta / PyTorch Foundation | the base: tensors, autograd, CUDA ops, distributed | itself | yes: math, flash (FA2 compiled from the `third_party/flash-attention` submodule), memory-efficient (xFormers), cuDNN, FlexAttention |
| flash-attention (`94e22c9`) | Dao-AILab | a kernel library: FA2 `csrc/`, FA3 `hopper/`, FA4 `flash_attn/cute/` | PyTorch C++/CUDA extension (`setup.py:821`) | yes |
| Liger-Kernel (`19d92e7`) | LinkedIn | Triton kernel library: losses, norms, RoPE, SwiGLU | PyTorch + Triton | no varlen attention (0 files with `cu_seqlens`) |
| Transformer Engine (`39c30c5`) | NVIDIA | fused layers and FP8/FP4 for PyTorch and JAX | PyTorch / JAX | calls cuDNN (closed source) or FlashAttention |
| Megatron-LM (`8b2f660`) | NVIDIA | training framework (parallelism, models, data) | PyTorch (`torch>=2.6.0`; its own `torch.autograd.Function`s, no own tensor or autograd) | no; packed attention only through TE |
| transformers (`4cc2aa8`) | Hugging Face | model definitions + Trainer | PyTorch | no; dispatches to eager / SDPA / FA2 / FA3 / Flex (`modeling_utils.py:5120-5123`) |
| TRL (`478624d`) | Hugging Face | "Transformers Reinforcement Learning": SFT/DPO/GRPO trainers | transformers + accelerate + datasets | no (one loss kernel). Not related to Transformer Engine (0 files mention it) |
| torchtitan (`1e4d33d`) | PyTorch org | "PyTorch native platform for training" | PyTorch | no; calls PyTorch varlen / Flex / SDPA |

### 39.2 v2 step 1: the two designs, from the code

**Variable-length attention is not a new kind of attention.** The math is the same: softmax(Q·Kᵀ/√d)·V, causal, GQA. The new input is only where each chat starts and ends. Each library added it kernel by kernel. Paths without it either refuse packed input (Megatron's own attention: "Packed sequence is not supported by DotProductAttention", `core/transformer/dot_product_attention.py:161`) or undo the packing (TE's unfused path converts packed back to padded, `backends.py:545-563`).

**Design A: each chat is its own sequence (`cu_seqlens` = start offset of each chat).**
- **FA2:**
  - the grid is (ceil(max_len/BlockM), chats, heads) (`flash_fwd_launch_template.h:63-64`);
  - each block reads its chat's offset and length (`block_info.h`) and exits early past the chat end (`flash_fwd_kernel.h:88`);
  - causal is per chat (`:91-94`);
  - rows past the chat end are not loaded, and their scores are set to −∞ (`mask.h:96-100`);
  - the backward's dQ buffer is padded by 128 rows per chat (`flash_api.cpp:1145-1155`).
- **FA3 (Hopper):**
  - one TMA descriptor for the whole packed tensor, shifted per chat (`domain_offset`, `mainloop_fwd_sm90_tma_gmma_ws.hpp:663-665`);
  - it returns before loading K when a chat has no work (`:612-617`);
  - **TMA output stores are switched off for varlen**, so a tile cannot overwrite the next chat (`epilogue_fwd.hpp:37, 331`; dK/dV `epilogue_bwd.hpp:31`);
  - a persistent scheduler maps a global tile number to (chat, tile, head) with a warp prefix sum (`tile_scheduler.hpp:493, 600-680`), with an optional longest-first sort (`flash_prepare_scheduler.cu:163-179`);
  - dQ is accumulated with atomicAdd into a padded fp32 buffer (`mainloop_bwd…:621, 964`).
- **FA4:** the same structure (`flash_attn/cute/seqlen_info.py`, `interface.py:682-813`), plus block sparsity.
- **PyTorch:**
  - `varlen_attn` → `aten._flash_attention_forward` = the FA2 kernels. Its cuDNN path is switched off (`varlen.py:21-23, 75-84`);
  - nested tensors use the same kernels (`nested/_internal/sdpa.py:352, 793, 830`);
  - the memory-efficient kernel takes `seqstart` offsets (`kernel_forward.h:211-235`).
- **Transformer Engine:** `qkv_format="thd"`.
  - The cuDNN graph gets ragged offsets (`fused_attn_f16_arbitrary_seqlen.cu:124-155`) and a causal band (`:183-186`).
  - Or it calls FA2/3/4 varlen (`backends.py:130, 191, 246-248`).

**Design B: one long row plus a block mask.**
- **FlexAttention:**
  - `create_block_mask` evaluates the mask on every (q, kv) pair, cuts it into 128×128 blocks and labels each block skip / full / partial (`flex_attention.py:979-1030`);
  - the kernel runs a loop over partial blocks (mask applied) and a loop over full blocks (no mask) (`flex_attention.py.jinja:129-180`; `common.py.jinja:67-81`);
  - splitting out the full blocks is "about a 15% speedup" for causal;
  - `BACKEND='FLASH'` passes the same block mask to FA4.
- **torchtitan** uses **both** designs. It picks one per model with a config string (`models/common/config_utils.py:96-115`):
  - `"varlen"` = design A, PyTorch's `varlen_attn` (`attention.py:39-41, 132+`);
  - `"flex"` = design B, the Triton kernel;
  - `"flex_flash"` = design B, FA4, Hopper/Blackwell only;
  - `"sdpa"` is rejected for language models, because it "cannot consume per-document positions".
  - Model defaults: varlen for Qwen3 8B and the gpt-oss debug model; flex for DeepSeek-V3, Kimi and muse_glimmer. Kimi needs flex: "Kimi QK clipping requires the FlexInnerAttention backend" (`models/kimi_k2_7/__init__.py:196-198`).
  - Masks for flex: a chat id from `positions == 0`, and the fast causal mask `kv_idx >= chat_start[chat_id[q_idx]]` (`attention.py:467-530`).
- **PyTorch itself provides both designs:** `varlen_attn` (A) and FlexAttention (B). torchtitan is a user of PyTorch, not a separate kernel source.
- torchtitan calls `varlen_attn(..., window_size=...)`, which our PyTorch clone (2025-12-05) does not have. So torchtitan targets a newer PyTorch nightly, and the kernel behind `varlen_attn` there is not visible in our clone. In our clone it is FA2.

**The frameworks only prepare data and call the kernels above:**
- **Megatron:**
  - SFT packing appends chats in order, pads each to a multiple, and truncates overflow (`training/datasets/sft_dataset.py:120-200`);
  - `PackedSeqParams(qkv_format="thd")` goes to TE (`pretrain_gpt.py:341-365`);
  - RoPE takes per-chat positions from `cu_seqlens` (`rope_utils.py:208-275`).
- **HF transformers:**
  - detects packing from `position_ids` (`modeling_flash_attention_utils.py:558-572`);
  - builds `cu_seqlens` from `position_ids == 0` and calls FA varlen (`:480-517, 843-847`);
  - for SDPA it builds a dense same-chat + causal mask (`masking_utils.py:183-190, 522`).
- **TRL:**
  - best-fit-decreasing packing (`data_utils.py:739`);
  - padding-free collator with positions restarting per chat (`sft_trainer.py:1180, 528-552`);
  - sets the label of each chat's first token to −100 (`:516`), because HF shifts labels inside the loss. Our labels are already shifted per chat, so we don't need this.

**For us, varlen is a feature to add to the kernels the SFT run uses, not a new attention.** On H200, SFT (`bluscript_zero_ckpt.cpp:642-644`) runs:
- **v62 forward** (`GQA_fwd_generic.cu:330-336`);
- **Vj1d backward** by default (`GQA_bwd_generic.cu:126-130`; `USE_FLASH=1` picks Vj1, `USE_DETERMINISTIC_DQ=1` picks V44).

Every other attention path must either get the feature or refuse packed input with an error. Silently ignoring the chat boundaries would let chats see each other with no error.

### 39.3 Causal attention: computed and masked, or skipped?

**Both, depending on the tile.**
- **Tiles fully above the diagonal are never loaded or computed** in every optimized kernel:
  - FA2 `flash_fwd_kernel.h:91-94` (backward `flash_bwd_kernel.h:318-320`);
  - FA3 `block.h:23-33`;
  - memory-efficient `kernel_forward.h:282-290`;
  - FlexAttention (only listed blocks);
  - our v62 `GQA_fwd_sm90.cu:436` (`my_nkv`), with the producer loading only `pair+1` key tiles (`:397-401`).
- **Tiles crossing the diagonal are computed fully, then the entries above the diagonal are set to −∞**:
  - FA2 `mask.h:96-100`;
  - FA3 `mask.h:89-100`;
  - v62 `:479-498`.
- **The naive path computes everything and masks.** PyTorch math SDPA builds a full lower-triangle mask, computes all of Q·Kᵀ and adds −∞ (`attention.cpp:60-71`). This is also how the 2017 paper defines it ("masking out (setting to −∞)").
- **FA2 paper:** skipping about half the blocks gives "around 1.7-1.8× speedup compared to attention without the causal mask"; only 1 block per row needs the mask.
- **Our v62 at T = 4096** (tile 64 × 128): 2,048 tiles in the full grid, **1,056 computed (51.6%)**. 64 diagonal tiles are about half masked, which is about 3% of the computed work.
- **Unequal work per row:**
  - FA2 loops key tiles in reverse (`flash_fwd_kernel.h:138-140`);
  - FA3 uses longest-processing-time-first (`tile_scheduler.hpp:222-224, 309-310`);
  - v62 pairs an early query tile with a late one (`decode_u`).
- **−∞ and NaN:** exp(−∞) = 0. A fully masked row would give −∞ − (−∞) = NaN. FA guards this (`softmax.h:73-76`). Plain causal never fully masks a row; packing with pad rows can.
- **The KV cache only works because of causal attention.** At decode, no mask is needed. When seqlen_q ≠ seqlen_k, FA2 aligns the mask to the bottom right (README "2.1").

### 39.4 Attention types (background, answered in the chat)

The history from Bahdanau 2014 to 2026, grouped by what each type changes:
- **the score function:** additive, multiplicative, scaled dot;
- **which pairs are allowed:** bidirectional, causal, prefix-LM, sliding window, fixed sparse (Sparse Transformer, Longformer, BigBird, Reformer), learned sparse (NSA, MoBA, DeepSeek DSA, DeepSeek-V4 CSA+HCA);
- **the K/V storage:** MHA, MQA, GQA, MLA;
- **replacing softmax:** linear attention, Performer, Linformer; then RetNet, Mamba/Mamba-2, GLA, DeltaNet, Gated DeltaNet, KDA, and the hybrids Jamba, MiniMax-01, Qwen3-Next, Kimi Linear. MiniMax went back to full attention for M2;
- **the computation only:** online softmax, memory-efficient attention, FlashAttention 1–4, PagedAttention, FlexAttention.

Facts checked on 10-09:
- the DeepSeek-V4 model card: at 1M context, 27% of single-token FLOPs and 10% of the KV cache of V3.2;
- the MiniMax M2 post (2025-10-30): efficient attention "still has some way to go".

**Our model is GQA (18 query heads, 6 KV heads, head_dim 128), causal, RoPE, with an exact FA3-style kernel. v2 does not change this; it only changes which pairs are allowed (block-diagonal causal).**

### 39.5 FlashAttention vs memory-efficient attention, precision, and what we run

**PyTorch (`aten/.../cuda/sdp_utils.cpp`): the two are different kernels with different rules, not only different dtypes.**
- **Flash** = FA2:
  - fp16/bf16 only (`:662-664`);
  - no attention mask (`:777`);
  - GQA supported (`:801`);
  - sm80 and newer.
- **Memory-efficient** = xFormers on CUTLASS:
  - fp16/bf16/fp32 (`:830-831`);
  - any additive mask;
  - **no GQA**: K/V must have as many heads as Q (`:871`);
  - sm50 and newer.
- **On H100/H200 the order is cuDNN, then flash, then memory-efficient, then math**, when cuDNN ≥ 9.9 (`:78-108`; `Context.h:457-462`).
- FA3 is not in PyTorch. FA4 is opt-in (`activate_flash_attention_impl("FA4")`, sm90/sm100 only, `_fa4.py:105-106`).
- **The forward and backward always come from the same backend** (`derivatives.yaml:2882-2908`).

**Precision (bf16 inputs).** FA2, PyTorch memory-efficient and our v62 use the same scheme:

| Step | Precision |
|---|---|
| Q, K, V | bf16 |
| Q·Kᵀ | **fp32 accumulate** |
| softmax max, exp, sum | **fp32** |
| P | rounded to bf16 for the P·V matmul |
| O | **fp32 accumulate**, written as bf16 |
| LSE | **fp32** |

Code:
- FA2: `kernel_traits.h:26, 33`, `softmax.h:131`, `flash_fwd_kernel.h:194, 310, 354, 443`;
- memory-efficient: `kernel_forward.h:99`, `mma_from_smem.h:1526`;
- v62: `GQA_fwd_sm90.cu:278, 440-441, 511, ~540, 605-609`.

**"Flash = everything bf16" is false. Softmax is fp32 everywhere.** With fp32 inputs, PyTorch memory-efficient uses 3×TF32 (`gemm_kernel_utils.h:112-123`).

**Our BluTrain paths:**
- **Current trainers (single, zero, zero_ckpt = SFT, production):** `FlashAttention` backend, bf16, GQA, v62 + Vj1d on H200. Before attention, QK-norm + RoPE run in fp32 (`qk_dtype()`, `bluscript_zero_ckpt.cpp:198`).
- **Vj1d backward:**
  - D = rowsum(dO·O) in fp32;
  - dQ summed in an fp32 buffer (TMA reduce-add);
  - dK/dV: per-query-head bf16 partials, then the 3 heads of a group summed in fp32 (`GQA_bwd_vj1d_sm90.cu:283, 373, 395-413, 646`). FA2 does the same for GQA (`flash_api.cpp:935-1004`).
- **Our old `MemoryEfficient` backend:**
  - fp32 only, MHA only, causal or none, no dropout, no varlen;
  - matmuls in **1×TF32** (`AttentionOps.cpp:89-90`; `AttentionForward.cu:727-733`);
  - used only by `gpt2_fmha.cpp` and an old DDP test.
  - It is a real memory-efficient kernel (same idea as flash), but not PyTorch's xFormers kernel.
- **v62 header:** H200, B=8, Hq=12, Hkv=4, S=4096 → 0.7075 ms, 579 TFLOPS, "~1.6% faster than PyTorch SDPA's cuDNN path" (`GQA_fwd_sm90.cu:14-15`).

**Attention rivals on H200, forward and backward:**
- cuDNN (PyTorch's default);
- FA3, FA4, FA2 (PyTorch's fallback);
- FlexAttention;
- ThunderKittens. Its paper says the forward is level with FA3 and the backward is 10–40% faster than FA3 on H100; I have not cloned it.

Liger is a rival for the loss kernel, not for attention.

### 39.6 Loss-kernel standalone benchmark on H200 (for a colleague)

**Everything about it is in `profiling_reports/sft_loss_bench_2026_10_09/SFT_loss_kernel_benchmark_H200.md`:** the plan, design, data, settings, every result, a glossary and a checklist for running other kernels the same way.

**The test:**
- the loss kernels alone (no model);
- H200 GPU 5;
- B=8 × T=4096 = 32,768 positions, V = 50,304, bf16 logits;
- labels from `stage0_ultrachat_5k_T4096/train/labels.bin` (answer 23.30%, prompt 6.92%, pad 69.78% of positions);
- random logits from a fixed formula;
- 6 cases: 4 real batches (first, min, median, max trained share), masking off, and all masked;
- 5 timed operations: F (forward with saved stats), B (backward with stats), F+B, Fv (forward without stats, used by validation), Fv+Bo (no-stats pair);
- 5 warm-up + 50 timed calls with CUDA events.

**Results (median batch, 23.1% trained):**

| Operation | Time | % of 4.8 TB/s peak |
|---|---|---|
| F | 0.276 ms | 57.5% |
| B | 1.600 ms | 52.9% |
| **F+B** | **1.878 ms** | |

- **Per step on 1 GPU:** 8 × 1.878 = **15.0 ms**, about 0.3% of the ~5.3 s step.
- **Masking:** 3.2× faster than no masking on the same rows (1.676 vs 5.376 ms).
- **Saved stats:** 17–22% faster than recomputing them (1.878 vs 2.258 ms; 5.376 vs 6.884 ms).
- **The backward is 85% of the pair.**
  - Writing the full gradient as zeros runs at 83% of peak (all-masked case).
  - The trained-row part of the backward runs at only about a third of peak (nomask: 31.6%). This is the slowest part; the cause is not measured yet.
- **Time is linear in the trained share:** F+B ≈ 0.87 + 4.51 × share (ms).
- **All checks PASS:**
  - every trained position computed, every skipped one skipped;
  - skipped gradients exactly zero;
  - the no-stats path identical;
  - the loss equal to the theory value (12.746 for this logit distribution).

**Run state:**
- source md5 `f8a5bf8c98b2…`; binary on the box `~/gautam_k/loss_bench/bench_sparse_ce_sft`; TI `d96b991`; CUDA 13.4; `sm_90`; free clocks;
- the GPU was empty before and after;
- the raw log `bench_run1.log` is on the box and still to be copied back;
- nsys per-kernel run prepared, not run.

### 39.7 Found and parked: the B300 loss forward ignores ignore_index

- On B300, `sparse_cross_entropy_forward_cuda_impl_vec_save_stats` calls `sparse_ce_fwd_sm103_launch`, which has no `ignore_index` parameter (`arch/CrossEntropyKernels_sm103.cu:140-143`).
- So on B300 the masked positions are computed as trained, and the target 65535 is read past the row. The loss value and the saved stats are wrong. The backward still zeroes the masked rows.
- H200 is not affected.
- Parked: the user said "forget B300, we are doing H200". Fix it before any SFT run on B300.

### 39.8 Which design for v2: A or B

PyTorch has both: `varlen_attn` (A, the FA2 kernels) and FlexAttention (B). A is what FlashAttention and TE use, and the most general. **For our kernels I lean to B. The final choice comes after reading Vj1d (step 2).** Reasons:

1. **The rows stay 4096 tokens** (v2 fills rows; on average 16 pad tokens per row). So v62 and Vj1d still see B rows × S = 4096, and their constraints hold: S % 256 = 0, head_dim 128, causal, a fixed 132-CTA persistent grid, TMA loads.
2. **Output stores stay as they are.** In B a tile never writes into another row's memory. In A, FA3 had to switch off TMA output stores so a tile at a chat end does not overwrite the next chat.
3. **The change is local:**
   - forward: per query tile, start the key loop at the chat start of the tile's first row, and set to −∞ the keys of an earlier chat on that first key tile;
   - backward: the mirror, so per key tile, stop at the end of that key's chat.
4. **No NaN case:** if pad tokens form their own segment, every token sees at least itself, so no row is fully masked.

**Costs of B:**
- part of a key tile is wasted at each chat start, because tiles are aligned to the row, not to the chat. A wastes part of a query tile at each chat end instead. The exact difference on our packed rows is to be measured in step 3;
- the work per query tile is no longer the plain triangle, so v62's early/late pairing no longer balances exactly. A has the same problem.

**What A has that B does not:** any chat lengths without a fixed row size, which is what FA/TE need. We do not need it, because we drop chats longer than 4096.

### 39.9 Records

- `profiling_reports/sft_loss_bench_2026_10_09/`: `bench_sparse_ce_sft.cu`, `SFT_loss_kernel_benchmark_H200.md`. `bench_run1.log` and the nsys output are to be added.
- The library clones: `~/Downloads/sft_research/{flash-attention, TransformerEngine, transformers, trl, torchtitan, Megatron-LM, Liger-Kernel}` and `~/pytorch_cloned/pytorch`.
- **Verda:** H200 GPU 5 for about 10 seconds (the loss benchmark); no other box GPU use.

### 39.10 Next

1. **v2 step 2:** read Vj1d, RoPE and the loader, then choose A or B.
2. **v2 step 3:** the design, with the exact tile cost of A vs B measured on our packed rows.
3. **v2 step 4:** build it, test against a PyTorch reference with an explicit mask, benchmark against FA3 varlen / FlexAttention / cuDNN.
4. Optional: the nsys run for the loss benchmark. Copy `bench_run1.log` back.
5. The shared doc tab: change "1–5" chats per row to "1–12" (kernel section, Step 5).
6. Later: benchmark the loss kernel vs PyTorch / Liger / Apple CCE; Q11; Q4, Q5, Q6, Q12, Q13.

## 40. 2026-10-10: v2 steps 0–5 — per-chat RoPE positions, the attention test harness, design B and design A inside our kernels (forward + backward), a backward sync fix, precision findings, speed

### 40.1 What v2 needs, and the decisions taken

- **v1** puts one chat in each 4096-token row and fills the rest with padding: 5000 stage-0 chats = 5000 rows.
- **v2** packs several whole chats into one row: the same 5000 chats (6,188,911 tokens) fit in **1517 rows** (the minimum possible is 1511), 1–12 chats per row, mean 3.3. That is 3.3x fewer rows per epoch.
- Only two parts of the model must change:
  1. **attention:** a token may see only the earlier tokens of its own chat, never another chat in the same row (block-diagonal causal);
  2. **RoPE positions:** they must restart at 0 at every chat start.
- The industry has two ways to tell the attention kernel where the chats are (same math, different speed and code):
  - **design A, per-chat offsets** (the `cu_seqlens` / "varlen" way): the kernel gets a list (row, start, length) and cuts its work per chat;
  - **design B, packed row + chat mask:** the kernel gets one number per token, the start of that token's chat (`seg_start`), cuts its work per row and masks chat borders.
- Decisions (user, 10-10):
  - build **both** inside our own kernels as compile-time variants, test them on the same data, keep the faster one or both (as PyTorch and torchtitan do);
  - the normal (dense) kernels must stay exactly as they are: proven by comparing compiled SASS before/after;
  - every change is checked against the reference sources and proven before moving on;
  - known precision limits of today's backward are kept for now, watched in every run, and decided at the end;
  - no GPU run without the user's OK, with the expected duration stated first.

### 40.2 Who uses which design (checked in the source, our clones under `~/Downloads/sft_research/`)

| Framework | Packed-SFT attention | Evidence |
|---|---|---|
| TRL (SFT trainer) | A. The default packing `bfd` turns on padding-free; positions restart per chat; TRL warns unless the attention is FlashAttention | `trl/trainer/sft_trainer.py:1180-1202`, `489-550` |
| HF transformers | A on the FlashAttention path (`cu_seqlens` built from positions); B on the SDPA / eager / Flex paths (same-chat mask per token) | `modeling_flash_attention_utils.py:480-517`; `masking_utils.py:183-189, 734-763` |
| Megatron-LM | A: `PackedSeqParams` with `cu_seqlens_q`, `cu_seqlens_q_padded` | `megatron/core/packed_seq_params.py:10-46` |
| TransformerEngine (cuDNN) | A: `qkv_format="thd"` + `cu_seqlens` | `dot_product_attention.py:470, 550, 2445` |
| torchtitan | both: `VarlenInnerAttention` (A) and `FlexInnerAttention` with a document mask (B) | `models/common/attention.py:132, 177, 228, 467, 598` |

- Why A is the common choice: the fast kernels these frameworks call (FlashAttention, cuDNN) take per-sequence offsets. The "B" they get through SDPA is a full mask that computes every tile; the fast B is Flex, which skips fully masked blocks.
- Our B skips tiles like Flex, so it already gets the speed benefit. A was still built because it is the industry-standard form and it can be faster (no tile shared by two chats, per-chat scheduling).

### 40.3 Step 0: the base

- The user ran `git ls-remote` on the box: `dist_side_merge` = 909ab750 (BluTrain) and d96b991a (Tensor-Implementations), the same as my laptop base.
- All v2 work is on the laptop in `~/Downloads/BluTrain_sft` (uncommitted; the user commits). The box copy is `~/gautam_k/v2_work` (a separate copy, so the user's own `~/gautam_k/BluTrain_sft` and its running library are untouched). Box build: `export PATH=/usr/local/cuda/bin:$PATH; make -j$(nproc) tensor SM_ARCH=90` (box CUDA 13.4).

### 40.4 Step 1: per-token RoPE positions (done)

- What I changed: the fused RMSNorm+RoPE kernels take an optional per-token `positions` array (template `HAS_POS`), so positions can restart at every chat start. 11 Tensor-Implementations files: `FusedRMSRoPEKernels.h`, `FusedRMSRoPE_{fwd,bwd}_sm90{,_kernel}.cu`, `Normalizations.h/.cpp` (validation helper `fused_rms_rope_positions_ptr`), `NormalizationOps.h/.cpp`, `NormalizationBackward.h/.cpp` (`positions_` member).
- Proof the old path is unchanged: SASS of 13/13 original kernels identical to the new `HAS_POS=false` instantiations (laptop CUDA 13.0 and the box's real objects, CUDA 13.4). The positions-on versions differ (+16 to +40 instructions, extra loads), so the check is not vacuous.
- Test `Tests/kernel_tests/test_rope_positions.cu`, 37 cases: 37/37 on the laptop RTX 3060, on H200 GPU 7 (sm90 dispatch) and on H200 with `BLUTRAIN_NO_SM90_QKNORM_ROPE=1` (portable path).
- Records: `profiling_reports/v2_step1_rope_positions_2026_10_10/`.

### 40.5 Step 2: the attention test harness (done)

File: `Tensor-Implementations/Tests/kernel_tests/test_varlen_attention.cu` (final md5 7c2e5fee). Build + run from `Tensor-Implementations/`: `make run-cuda FILE=Tests/kernel_tests/test_varlen_attention.cu`.

**References** (built from the same bf16 inputs the kernel gets, per segment = per chat or pad tail):
- **truth:** fp64 attention inside the segment, causal;
- **floor:** plain PyTorch bf16 math, rounding at the same points as FlashAttention's `attention_ref(upcast=False, reorder_ops=True)`;
- both are built from tiled GEMMs that add their terms in index order, with every rounding spelled out (the tests compile with `--use_fast_math`, so the references use explicit IEEE intrinsics).

**Pass rule** (per chat, for O, dQ, dK, dV): `max|kernel - truth| <= 2 * max|floor - truth| + 2 * max|(x + 0.3) - 0.3 - x|` — FA3 `hopper/test_flash_attn.py:535-536, 579, 655-660`; PyTorch `test/test_varlen_attention.py:508-516, 583-595`. LSE gets the same rule (my addition; FA does not gate LSE). 1-token chats have an exact dQ = dK = 0, so for them dQ/dK are checked against a derived fp32 rounding bound instead (`one_token_region`, u = 2^-23, c = 1.02).

**Known list (`kKnown`):** explained deviations kept by decision. Each entry matches one exact case, tensor and chat, up to the recorded ratio + 1%; anything else fails as NEW.

**Data loads:** D1 real v1 rows; D2 2048+2048; D3 1000+3096; D4 1+4095 / 4095+1; D5 64x64, 32x128; D6 4096 one-token chats (D6q: 512 + pad); D7 63/65/127/129/255/257 repeated; D8 random lengths with pad tails 0/1/16/63/64/65/255; D9 an all-pad row; D10 the real stage-0 rows packed first-fit decreasing (1517 rows); D11 the same 300-token chat at offsets 0/1/63/64/65/127/128/3796.

**Groups:**

| Group | What it checks |
|---|---|
| G1 | the references themselves vs a CPU full-row masked loop (+ a mutant control that must fail) |
| G2 | today's kernels on dense rows, 3 lengths x 5 head layouts x 3 batch sizes |
| G3 | v1 SFT rows (chat + pad) |
| G4 | repeat 1000x (race test): O/LSE/dK/dV bitwise, dQ within FA's atol |
| G5 | large inputs (FA2's overflow test) |
| G6 | every chat run alone through today's kernel vs the per-chat reference; G6c filler independence |
| G7 | timing |
| G8 | design B forward |
| G9 | design B backward |
| G10 | design A forward |
| G11 | design A backward |

Head layouts: 18/6 (production), 12/4, 8/8, 32/8, 8/1. Quick mode (`VARLEN_QUICK=1`) runs 18/6 only and 16 D10 rows.

**Environment switches:** `VARLEN_QUICK=1`, `VARLEN_ONLY=D10|G7|G8|G9|G10|G11`, `VARLEN_CASE=<layout name part>`, `VARLEN_D10_FIRST`, `VARLEN_D10_ROWS`, `VARLEN_REPEATS`, `VARLEN_DIAG_ROW/T0/REPEAT/CASE`, `VARLEN_DIAG_LIST="row:t0,..."` (G9 per-segment diagnostic), `VARLEN_EXPORT_DIR`, `VARLEN_STAGE0_LABELS` (box: `/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin`).

**Result on today's kernels (v62 forward + Vj1d backward), H200 GPU 7:** full run **614/615** (1242 s). D10 percentiles over 5000 chats: dQ p99 0.68, max 1.156 (1 chat); dV p99 0.74, max 0.956; dK max 0.83; O max 0.54.

Records: `profiling_reports/v2_step2_attention_harness_2026_10_10/`.

### 40.6 The one dQ case on today's kernel: bf16 P in dS (kept, decide at the end)

- The case: D10 row 1022, chat at 2552, length 1276, dQ ratio 1.156 (error 2.87e-2 vs floor 1.24e-2).
- The cause: Vj1d builds dS from P after rounding P to bf16 (`fuse_dS_ldstsm` re-reads P from shared memory as bf16). FA2 (`flash_bwd_kernel.h:591`), FA3 (`mainloop_bwd_sm90_tma_gmma_ws.hpp:893`), FA4 (`flash_bwd_sm90.py:1569`) and Flex (`flex_backwards.py.jinja:541/775`) use fp32 P there. PyTorch's memory-efficient kernel uses bf16 P like us (`kernel_backward.h:341-351, 1959-1960`).
- Proof: a model of the kernel's arithmetic with bf16 P reproduces the kernel's dQ row exactly; the fp32-P model gives 1.09x the floor. It hits rows with few keys and a large dP − D.
- The same chat through the other libraries (`crosslib_row1022_1010_1807.log`):

| Backend | dQ ratio |
|---|---|
| ours (Vj1d) | 1.156 |
| cuDNN | 0.527 |
| FA2 | 0.450 |
| PyTorch memory-efficient | 0.637 |
| PyTorch math | 0.309 |
| Flex | 0.450 |

- Decision (user): keep Vj1d as it is, watch the issue in every run, decide at the end with the speed cost measured.

### 40.7 Step 3: design B forward (done)

File `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu` (kernel v62), API `gqa_flash_attn_forward_qkv(..., is_causal, seg_start)`.
- `seg_start[b*T + t]` = where the segment of token t starts (a chat, or the row's pad tail as its own segment). Contract: `seg_start[t] <= t`, and it is t or `seg_start[t-1]`.
- What the kernel does differently:
  - the K/V loop of a 128-row query pair starts at tile `n0 = seg_start[first row of the pair] / 128`; earlier tiles are never loaded (Flex does the same with its block lists);
  - besides the diagonal tile, a tile is masked when some row's chat starts after the tile's first key (`border`); key < seg_start(row) gets −inf;
  - the online softmax gets the −inf guard Flex and FA use (a row that has seen no key yet keeps max −inf; the rescale uses 0 instead, so no NaN).
- The API refuses `seg_start` on any path that would ignore it (non-causal, head_dim ≠ 128, T % 256 ≠ 0, not Hopper, `BLUTRAIN_NO_SM90[_GQA]`).
- Proof: dense SASS 2304/2304 identical to the original v62 (laptop), 2280/2280 (box CUDA 13.4).
- G8 quick on H200: **51/51** (47 s): O/LSE of every segment vs the reference, tile-aligned chats bitwise = the chat alone, isolation, refusals.

### 40.8 Step 4: design B backward (done)

File `src/Kernels/cuda/attention/arch/GQA_bwd_vj1d_sm90.cu` (kernel Vj1d), API `gqa_flash_attn_backward_qkv(..., is_causal, seg_start)`.
- **Trip count:** a key tile visits only the query tiles of the chats that touch it. Query tile q has a visible element iff `seg_start[q*64] <= last key of the tile`; this is true for a prefix of q (seg_start never decreases), so the CTA counts it in one `__syncthreads_count` pass.
- **Mask:** the masked path runs on the diagonal tile and on any tile where the last row's chat start is after the tile's first key; keys before the row's chat start get P = 0.
- `seg_start` of the current 64 rows rides in shared memory next to LSE and D (+256 B).
- Cost: 236 registers, 115,472 B shared memory, no spills; still 2 CTAs per SM.
- Proof: dense SASS 1624/1624 identical (laptop), 1600/1600 (box).
- The backward refuses `seg_start` on Vj1 (`USE_FLASH=1`), v44 (`USE_DETERMINISTIC_DQ=1`) and every non-Hopper path.
- G9 quick **93/94**; G9 full **404/406** (574 s; 5 head layouts, all synthetic layouts, all 1517 real rows). Real-row work: the backward visits 40.2% of the dense tiles.

**The 7 over-limit segments of G9 full** (out of ~5,478; diagnostic `G9_diag7_D10_*.log`):

| Segment | Tensor | Packed | Same segment alone, today's dense kernel | Model = kernel | Cause |
|---|---|---|---|---|---|
| pad row 492 @4094 L=2 | dQ | 1.062 | 1.062 | yes | bf16 P in dS |
| chat row 791 @2998 L=1098 | dQ | 1.121 | 1.121 | yes (exact) | bf16 P in dS |
| chat row 1102 @3537 L=559 | dQ | 1.096 | 1.096 | yes | bf16 P in dS |
| pad row 455 @4057 L=39 | dV | 1.031 | 1.031 | yes (exact) | per-head bf16 partial |
| chat row 1140 @2283 L=1141 | dV | 1.171 | 1.171 | yes (exact) | per-head bf16 partial |
| chat row 1262 @1005 L=1005 | dV | 1.033 | 1.033 | yes (exact) | per-head bf16 partial |
| chat row 1359 @1714 L=857 | dV | 1.014 | 1.014 | yes (exact) | per-head bf16 partial |

- Packing adds nothing: on identical inputs, packed and alone give the same ratio to 3 decimals.
- **Second mechanism (new, also in today's kernel):** to make dK/dV bit-reproducible, Vj1d stores each query head's dK/dV as a bf16 partial and then sums the heads in fixed order (`gqa_dkdv_greduce`), so the value is rounded per head and again after the sum. FA3 sums the heads in fp32 and rounds once (`hopper/epilogue_bwd.hpp`, the GQA epilogue). The model of Vj1d's arithmetic reproduces the kernel's worst dV element exactly; summing in fp32 brings the error to the floor's (chat 1140: 6.3e-2 → 9.1e-4).
- All 7 (and the G9 D8 dQ chat, ratio 1.049, same bf16-P mechanism) are on the known list. **End-decision list:** (1) fp32 P in dS for dQ, (2) fp32 head sum for dK/dV — both after measuring their speed cost.

### 40.9 A sync fix in today's backward (applied, user-approved)

- **The issue:** in Vj1d (and Vj1), the first query tile's LSE values are written into shared memory by threads 0–63 (`issue(0)`) and read by all 4 warps before any barrier. The only wait in between is the TMA mbarrier for Q/dO, and only thread 0 arrives on it (before its own store). Tiles ≥ 1 are covered by the store barrier after `issue(it + 1)`; tile 0 had nothing. SASS of the old kernel: the store at 0x1010, the first read at 0x1ee0, no `BAR.SYNC` between. It worked only because the 256-byte LSE load always lands before the 32-KB Q/dO load plus the first GEMM. v44 orders the same data correctly (`producer_sync` + the `d_ready` mbarrier, `GQA_bwd_sm90.cu:594-596, 622`).
- **The fix:** one `__syncthreads()` after `issue(0)`, in Vj1d (both instantiations) and Vj1. Once per CTA, not per tile.
- **Proof:** SASS = +1 `BAR.SYNC` only (the same registers R98–R103 are zeroed, just reordered around it), laptop and box; the packed kernel is identical (it already had the barrier).
- **Tests:** G quick with the fix **289/289**; G4 1000 repeats at both shapes bitwise for O/LSE/dK/dV.
- **Overhead:** same-session A/B, 12 rounds, ABBA, the old library (`lib_nofix/`, md5 089f18ef) vs the new one, libraries checked with `ldd`. 12/4 backward −0.04% (t = −0.06). 18/6 backward +1.71% (t = 2.01), but the untouched forward (null control) moved +1.39% (t = 2.02). So no measurable overhead; the method cannot see anything below ~2%.

### 40.10 Vj1 (`USE_FLASH=1`) checked for the first time (not used in training)

- G quick in Vj1 mode: 183/196. Vj1 sums the heads' dK/dV with bf16 atomic adds, so dK/dV differ run to run by one bf16 step (max 3.1e-2 at |x| in [4, 8)) and carry extra rounding: MQA 8/1 dK 1.078, D5/D6 dV over the limit.
- The harness now gates Vj1's dK/dV in G4/G6c with FA's atol computed on dK/dV (it was the dQ atol, too tight).
- Keep Vj1 off for SFT.

### 40.11 Step 5: design A (done)

**Plan object** (`include/ops/helpers/AttentionKernels.h`): `GqaChatPlan`, built once per batch with `gqa_chat_plan_build(plan, seg_b, seg_t0, seg_len, n_seg, B, S, Hq)` and freed with `gqa_chat_plan_free`.
- The segments (chats and each row's pad tail) must cover every token of every row exactly once; the builder throws otherwise, because an uncovered row would keep garbage outputs that the next layers read.
- Forward work list `{b, t0, L, (cpl << 8) | hq}`: per segment, the query pairs coupled like dense v62 (pair c with pair nP−1−c), longest segment first (as FA3's varlen scheduler orders work).
- Backward list `{b, t0, L, kt}`: one entry per 64-key tile, most work first.
- Pass the same plan to `gqa_flash_attn_forward_qkv(..., seg_start = nullptr, chats = &plan)` and `gqa_flash_attn_backward_qkv(..., nullptr, &plan)`.

**Forward** (v62, template mode 0 dense / 1 B / 2 A):
- rows offset by t0, tile indices relative to the segment, so the diagonal mask is exactly dense causal with S = L;
- keys past L appear only in diagonal tiles, where causal already hides them;
- rows past L are computed but not stored (FA3 does the same: `epilogue_fwd.hpp:303, 331, 358`);
- a query tile entirely past L gets no active tiles and runs only the existing inactive-tail handshake (the first mode to use that path).

**Backward** (Vj1d, same modes):
- one CTA per (plan key tile, query head);
- rows past L get LSE = +inf and D = 0, so P = 0 and dS = 0 (FA2 does this, `flash_bwd_kernel.h:411-413`); their dQ rows get exact zeros from the reduce-add;
- full key tiles keep the TMA store; the segment's last, partial tile stores only rows < L, so it never overwrites the next segment's partials (FA3 stores all varlen dK/dV that way, `epilogue_bwd.hpp:31, 263-268`).

**Refusals:** a plan together with `seg_start`; a plan on paths v62/Vj1d cannot take; a plan built for another (B, T, Hq); the builder's gap / overlap / out-of-row / uncovered-row checks.

**Proofs:**

| Kernel | Dense | B | A |
|---|---|---|---|
| forward (v62) | 2304/2304 identical to the original v62 (box 2280/2280) | 2448/2448 identical to the G8/G9-tested B (box 2432/2432) | 2224 instructions, 168 registers, no spills |
| backward (Vj1d) | 1624/1624 identical to the race-fixed dense (box 1600/1600) | 1776/1776 identical (box 1736/1736) | 219 registers, 115,216 B shared memory, no spills, 2 CTAs/SM |

**Tests on H200:**
- G10 quick **59/59** (65 s) and G11 quick **95/95** (53 s); mode refusals pass.
- The key result: **every segment computed by A is bit-identical to the same segment run alone through today's dense kernels** — O, LSE, dK and dV. That covers chats on and off tile edges, 513 one-token chats, odd lengths and pad tails. dQ stays within the atomics atol. A cuts its tiles at the segment start exactly as the dense kernel cuts them at row 0, so it does exactly per-chat attention.
- Isolation including dO and the 50x repeat are bit-identical.
- G11 runs exactly G9's inputs (same seeds), so G9's known entries apply to the same segments (the D8 dQ chat shows as KNOWN).

### 40.12 Speed on real packed rows (8 rows each, median of 20, free clocks, H200 GPU 7)

Forward (G10 run):

| Rows | Work vs dense (B / A) | Dense | B | A |
|---|---|---|---|---|
| 0–7 (longest chats) | 93.0% / 92.8% | 1.111 ms | 101.9% | 97.6% |
| 752–759 (middle) | 39.7% / 37.1% | 1.084 ms | 49.4% | 46.5% |
| 1504–1511 (shortest) | 19.2% / 15.5% | 1.083 ms | 33.6% | 28.5% |

Backward (G11 run):

| Rows | Work vs dense (B / A) | Dense | B | A |
|---|---|---|---|---|
| 0–7 | 92.5% / 92.6% | 4.179 ms | 98.4% | 93.2% |
| 752–759 | 37.1% / 36.1% | 4.190 ms | 47.4% | 45.3% |
| 1504–1511 | 15.1% / 14.0% | 4.120 ms | 27.4% | 28.2% |

- Forward + backward (the two runs added, so approximate): middle rows B 47.8% / A 45.6%; shortest B 28.7% / A 28.3%; longest B 99.1% / A 94.1%.
- So packed attention is about 2x faster on typical rows and about 3.5x on short-chat rows. A is slightly ahead overall; B wins only the shortest-chat backward.
- Time falls more slowly than work when there are many short chats: each work unit pays a fixed cost (Q load, pipeline fill, epilogue). A also gives every tiny chat its own tile (D6q's 512 one-token chats cost 101% of the dense backward work). To profile in Step 6.
- These numbers come from single sessions at free clocks; Step 6 repeats them properly.

### 40.13 Files changed (laptop, uncommitted; the user commits)

Tensor-Implementations:
- Step 1: the 11 RoPE/normalization files in 40.4; new test `Tests/kernel_tests/test_rope_positions.cu`.
- Steps 3–5 and the sync fix: `include/ops/helpers/AttentionKernels.h` (md5 5470501b), `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu` (698d2705), `GQA_fwd_generic.cu` (b71b6dc5), `GQA_bwd_vj1d_sm90.cu` (deadc500), `GQA_bwd_generic.cu` (04255500), `GQA_bwd_vj1_sm90.cu` (26f6468c).
- New test `Tests/kernel_tests/test_varlen_attention.cu` (7c2e5fee).
- The box copy `~/gautam_k/v2_work` has the same files. Its library has A + B + the sync fix; `lib_nofix/` keeps the pre-fix library for A/B timing. Backups: `backup_pre_step3_4_1010/`, `backup_pre_racefix_1010/`, `backup_pre_A_fwd_1010/`, `backup_pre_A_bwd_1010/`. Test binaries: `tva_step34` (harness 7c2e5fee), `tva_diag`.
- The trainer (BluTrain) is not changed yet.

### 40.14 Records

- `profiling_reports/v2_step1_rope_positions_2026_10_10/`
- `profiling_reports/v2_step2_attention_harness_2026_10_10/` (G-series logs, diagnostics, cross-library scripts and logs)
- `profiling_reports/v2_step3_design_B_forward_2026_10_10/` (SASS laptop + box)
- `profiling_reports/v2_step4_design_B_backward_2026_10_10/` (G9 logs, the 7-segment diagnostic, `racefix/` with the A/B timing, SASS)
- `profiling_reports/v2_step5_design_A_2026_10_10/` (G10/G11 logs, SASS, final sources)

### 40.15 Next

1. **Step 6, benchmarks** (needs the environment decision): A vs B in one session, plus cuDNN, FA3, FA4, FA2, PyTorch SDPA and Flex on the same data and the same GPU; nsys/ncu to see why time falls more slowly than work on short chats. Proposed environment: `~/envv` read-only for torch / cuDNN / Flex / `varlen_attn`, and our own env for FA3/FA4/TE.
2. Choose A or B, or keep both.
3. **Step 7:** packing script (whole chats, first-fit decreasing, shuffled rows), loader output (tokens, labels, positions, `seg_start` or the plan), the attention autograd node passing it to forward and backward, then a real packed SFT run against v1. Open decision: leftover rows (proposed: endless stream reshuffled every epoch; validation scores all rows).
4. End decisions: fp32 P in dS, fp32 head sum for dK/dV.
5. Small: the G11 mode-refusal lines print a "G10" label (cosmetic).

## 41. 2026-10-12: v2 Step 6 — benchmarks against every library, nsys/ncu profiling, the coverage audit, and the full test suite

Details, all tables and the method: `profiling_reports/v2_step6_benchmarks_2026_10_12/STEP6_ANALYSIS.md`. All runs on H200 GPU 7 at a steady 1980 MHz.

### 41.1 Environment

- `~/envv` (read-only, not ours): torch 2.14.1+cu130, cuDNN 9.24, Triton 3.8, Flex attention, `varlen_attn`. No FA2/FA3/FA4/TE.
- My env `~/gautam_k/benv` reuses envv's torch read-only through a `.pth` file. FA3 is built from FlashAttention `94e22c9` `hopper/` for bf16 + head_dim 128 + sm90, with every feature for that configuration kept (4.5 min build). FA4 = `flash-attn-4` from the same commit. FA2 needs no build: PyTorch's SDPA flash backend and `varlen_attn` use FA2 kernels.
- Note: torch 2.14's `varlen_attn` has no `is_causal`; causal = `window_size=(-1, 0)`, GQA needs `enable_gqa=True`.

### 41.2 What I measured

1. **Packed attention** on 7 layouts (3 sets of 8 real packed rows, and rows of equal 512/1024/2048/4096 chats), forward and backward, 50 interleaved rounds: ours dense / B / A vs FA3, FA3-deterministic, FA4, PyTorch `varlen_attn` (FA2), Flex with a document mask, SDPA with a block-diagonal mask (cuDNN and memory-efficient backends), and SDPA over the full row. All packed contenders agree with FA3 to bf16 noise on the same inputs.
2. **Dense attention at the colleagues' shapes** (B 2/4/8 x 12/4, 16/4, 24/8, 32/8 heads, plus 18/6 B8).
3. **nsys** kernel traces and **ncu `--set full`** on the main forward and backward kernels (ours, FA3, FA4, cuDNN), dense and packed.

### 41.3 Results

- **A vs B:** A is equal or faster almost everywhere (its forward always), so A is the main path.
- **Packed, forward + backward:** ours is fastest on short chats (real shortest rows 1.49 ms vs FA3 1.60, FA4 1.62). FA3/FA4 are faster on middle and long chats (real middle rows: ours A 2.44 vs FA3 2.10; one 4096 chat per row: 5.43 vs 4.06).
- **Dense at the colleagues' shapes:** our forward beats cuDNN on 11 of 13 shapes; our backward is slower than cuDNN on all 13 (1.05–1.24x), than FA3 by 1.32–1.42x and than FA3-deterministic by 1.05–1.17x. This does not match the colleagues' earlier backward table; their measurement method is unknown to me.
- **Kernel level (nsys, dense B8 18/6):** forward ours 1075 us vs FA3 890; backward main kernel ours 3633 us vs FA3 2495, FA3-det 3283, FA4 2482, cuDNN 2836.
- **Why (ncu):** our backward executes 42% more instructions than FA3 (888M vs 625M) and moves 58% more data through shared memory (662M vs 419M wavefronts). About 16% of its stall samples sit in the fences before each dQ TMA reduce, plus the dQ staging stores. With 64-key tiles, this per-tile dQ work happens twice as often per key as in FA3 (128-key tiles), and P makes a round trip through shared memory. Our forward has 17% more instructions than FA3; its large long-scoreboard stall is mostly the idle producer warpgroup spinning (harmless).
- **Packed (real middle rows), kernel level:** forward ours A 501 us vs FA3 432; backward 1466 vs 1086 (FA3-det 1362), with +35% instructions and +57% shared-memory traffic, the same pattern as dense.
- **What would close the backward gap:** FA3's structure: 128-key tiles with a producer plus 2 consumer warpgroups, P kept in registers (this also removes the bf16-P precision issue), warp-specialised loads. That is a new backward kernel version. Not done; decision pending.

### 41.4 Test-coverage audit and the full suite

New groups:
- **G12:** packed rows at T 1024, 2048 and 8192; a 16384-token chat (the first test of the 2nd pass of B's backward prefix loop); a row of 4096 one-token chats, for A and B.
- **G13:** large inputs (q*5, k*3, v*3) on packed rows, gated like G5 with FA2's rule; 60 packed rows in one call.
- New modes: `VARLEN_ONLY=CLASSIC` (G1–G7), `BENCH`, `BENCHDENSE`; `VARLEN_CASE` filters one layout for profiling.

Full suite, every group at full size (5 head setups, all 1517 real rows, 1000x repeats), run as parallel processes on GPU 7 (`suite_1012_1319/`):

| Group | Checks |
|---|---|
| CLASSIC (G1–G7) | 605/605 |
| G8 (B forward) | 223/223 |
| G9 (B backward) | 406/406 |
| G10 (A forward) | 235/235 |
| G11 (A backward) | 411/411 |
| G12 (new edge cases) | 134/134 |
| G13 (large inputs, 60 rows) | 76/76 |

- Known items observed: exactly the listed ones. G6 D10 row 1022 dQ; the 7 D10 segments in G9; the same 7 in G11, with identical ratios.
- CLASSIC has 605 checks where the 10-10 run had 615. The 10 missing lines were vacuous: D6's regular dQ/dK checks with 0 chats (all its chats are 1-token and are checked by the 1-token bound, which is present: 4096 chats per head setup, all within 0.5% of the bound). The harness has skipped empty checks since the G8 work.
- The timing lines inside these groups ran with 6 processes sharing the GPU and are not used; the real timings are the exclusive Step 6 runs.
- Still not covered: two streams at once (Vj1d uses static scratch buffers, so it is not safe on two streams at once; this predates v2), cuDNN varlen (measured the same evening, see 41.7; it does not need Transformer Engine), and the end-to-end model-gradient test (Step 7).

### 41.5 Traps hit (and avoided next time)

- ncu 2026.3 saves `.ncu-repz`, not `.ncu-rep`.
- `/usr/local/cuda/bin/ncu` is a wrapper: killing its PID leaves the real ncu running, and it restarted the target. Stop the whole process tree by PID.
- Status text and a tar archive on the same ssh stdout corrupt the tar: send text to stderr, or fetch files separately.
- The scratch folder in /tmp was cleaned over the weekend; keep everything needed in records.
- `nsys stats` never overwrites an existing CSV ("SKIPPED: output file ... exists"), so a re-run silently shows the old capture. Always pass `--force-overwrite=true`.
- `nsys --capture-range=cudaProfilerApi` on the torch script recorded no kernels at all. NVTX ranges around the timed calls (`nvtx_kern_sum`) work.
- A name filter that is too clever can silently drop the thing under test: after each capture, print how many lines and kernels of it were recorded.

### 41.6 Next

1. Decision: rework the backward (FA3 structure) before Step 7, or after the trainer works.
2. Step 7: packing script, loader output (tokens, labels, positions, the A plan), the attention autograd node, the end-to-end gradient test (packed batch vs the same chats as v1 rows), then a real packed SFT run vs v1.
3. Open: leftover-rows policy; the two precision items (the backward rework fixes the first).

### 41.7 cuDNN packed attention (evening)

**Why it was missing.** I first thought cuDNN's packed (varlen) attention was only reachable through Transformer Engine. That was wrong: torch 2.14 calls it directly with per-chat offsets (`torch.ops.aten._cudnn_attention_forward` / `_cudnn_attention_backward`). But torch's public `varlen_attn` never uses it for causal attention: it turns causal into a window, and any window, or any GQA, goes to FA2 (`torch/nn/attention/varlen.py:62-65, 336`). So I called the op exactly the way torch's own code does (`varlen.py:118-134, 559-576`).

**How I checked it.** On every layout, before timing, cuDNN's O, dQ, dK and dV were compared with FA3 (relative error limit 2e-2):
- cuDNN errors were about 1e-3 on O and 5–8e-3 on the gradients, and all 21 checks passed. For comparison, FA4 vs FA3 is 0.4e-3 to 2.7e-3.
- A deliberately wrong head mapping failed on all 7 layouts (errors around 1.2–1.5), so the check can catch that mistake.
- Passing our 6 KV heads straight into cuDNN works: cuDNN sums the heads itself with its own `fmha_reduce_head_ragged` kernel. I ran it first in a separate process because torch never sends GQA down this path.
- The kernel names prove cuDNN ran.

**Results** (8 rows × 4096, 18/6 heads; ms per call; "cuDNN" = 6 KV heads passed directly, the real GQA number):

| layout | fwd ours A | fwd cuDNN | fwd FA3 | bwd ours A | bwd cuDNN | bwd FA3 |
|---|---|---|---|---|---|---|
| real rows, longest chats | 1.126 | 1.165 | 1.009 | 3.977 | 3.446 | 3.074 |
| real rows, middle | 0.540 | 0.534 | 0.533 | 1.927 | 1.724 | 1.586 |
| real rows, shortest chats | 0.326 | 0.341 | 0.342 | 1.171 | 1.149 | 1.227 |
| one 4096 chat per row | 1.169 | 1.227 | 1.087 | 4.312 | 3.711 | 3.182 |

- **Forward:** ours and cuDNN are within 5% of each other on all 7 layouts.
- **Backward:** equal on short chats. cuDNN is 9–16% faster on middle and long chats.
- **Backward order on middle/long chats:** FA3 ≈ FA4 < cuDNN < FA3 deterministic < ours A < ours B.
- **Kernel time only** (nsys, real middle rows): forward ours A 519 µs vs cuDNN 457 vs FA3 451; backward ours A 1825 vs cuDNN 1547 vs FA3 1509. At the API level the forward looks equal, because cuDNN's Python call adds about 0.08 ms.

This matches the dense picture: our forward is competitive, our backward is the gap. The FA3-style backward rework (41.6 item 1) is the fix.

Records: `profiling_reports/v2_step6_benchmarks_2026_10_12/cudnn_varlen_1012/` (`CUDNN_RESULTS.md`, `tables.txt`, all logs and nsys CSVs). Scripts: `bench_step6_libs.py`, `run_cudnn_varlen.sh`.

Three mistakes in my scripts were caught and fixed before any number was used: the empty capture-range report, the cuDNN filter that dropped cuDNN from one capture, and stale nsys CSVs. All three are listed in 41.5. The main timing run was not affected.

## 42. 2026-10-12 (evening): A/B decision, who calls which kernel, why our backward is slower, the new backward kernel Vj2, scope, and one bug found

Records: `profiling_reports/v2_bwd_rework_research_2026_10_12/`.
- `RESEARCH.md`: the evidence behind 42.3 to 42.5.
- `DESIGN_VJ2.md`: the design, proofs and results of 42.6 to 42.8.
- `gpu_runs/`: every log and profile CSV.
- `code_snapshot/`: today's code, with md5.

### 42.1 Design A or B: both kept, picked per call

**Measured on our own kernels (same code, 7 packed layouts):**
- Forward: A is 2-16% faster than B on all 7 layouts.
- Backward: A is 3-5% faster than B on middle and long chats; B is 2-4% faster than A only when chats are very short.
- Kernel time on real middle rows: forward A 501 µs vs B 538 µs; backward A 1466 µs vs B 1537 µs.

**Why A wins on real rows:**
- In B, a 128-row tile that holds the end of one chat and the start of the next computes both, and then masks half away.
- In A, every chat starts its own tiles, so no tile mixes two chats.
- When chats end exactly on tile edges (all 512 / 1024 / 2048 long), that difference disappears.

**Decision:** keep both, with A as the default.

**How one is picked:**
- No compiler flag. Both versions are compiled into the library.
- Each call picks by its arguments: nothing = normal causal, `seg_start` = B, a `GqaChatPlan` = A, both = refused.
- In Step 7 this becomes one training-config setting (for example `attn_pack_mode: A | B | none`). The loader builds the plan or `seg_start` that goes with it.
- Switching then needs no rebuild, and A and B can be compared in the same binary.

### 42.2 Who calls which attention kernel, and what I measured

The training frameworks do not have their own attention kernels; they call these:

| framework | how it packs | kernel inside | measured? |
|---|---|---|---|
| Hugging Face, `flash_attention_2` | A (cu_seqlens) | FlashAttention 2 varlen | yes, through PyTorch's copy of FA2 (`varlen_attn`), not the `flash-attn` package |
| Hugging Face, Flex | B (document mask) | PyTorch FlexAttention | yes |
| Hugging Face, SDPA + mask | full mask | cuDNN / memory-efficient | yes: 6-10× slower than all others |
| TRL (padding-free) | uses Hugging Face's attention | FA2 | through the FA2 numbers; TRL itself not run |
| Transformer Engine / Megatron | A (THD) | cuDNN fused attention | through cuDNN packed called from PyTorch; TE not installed |

PyTorch's public `varlen_attn` never uses cuDNN for causal attention (`varlen.py:62-65, 336`), so cuDNN was called directly (41.7). FA1 was not run, since FA2 replaced it.

Packed rows (8 × 4096, 18/6 heads), ms per call. Ours, FA2, FA3, FA4 and Flex are from the morning run; cuDNN from the evening run (ours repeated within about 1%):

| | ours A | ours B | FA3 | FA4 | cuDNN packed | Flex | FA2 |
|---|---|---|---|---|---|---|---|
| forward, short chats | 0.325 | 0.385 | 0.360 | 0.452 | 0.341 | 0.648 | 0.675 |
| forward, middle | 0.533 | 0.573 | 0.532 | 0.605 | 0.534 | 0.958 | 0.921 |
| forward, long | 1.119 | 1.170 | 0.980 | 1.074 | 1.165 | 1.780 | 1.864 |
| backward, short | 1.169 | 1.143 | 1.244 | 1.166 | 1.149 | 1.382 | 1.752 |
| backward, middle | 1.907 | 1.996 | 1.569 | 1.552 | 1.724 | 2.183 | 2.807 |
| backward, long | 3.951 | 4.162 | 2.909 | 2.846 | 3.446 | 4.187 | 5.668 |

- **Against Flex and FA2 (what HF and TRL use):** we win everywhere. Forward 1.6-2× faster; backward 6-18% (Flex) and 1.4-1.5× (FA2) faster.
- **Against cuDNN packed (TE's backend):** forward is a tie; backward is a tie on short chats and takes 11-15% longer on middle and long chats.
- **Against FA3 / FA4:** forward wins on short chats, ties on middle, and takes 14% longer on long. The backward takes 22-39% longer on middle and long chats.

### 42.3 Why our backward (Vj1d) is slower than FA3: measured, not guessed

ncu on dense B8 18/6 (10-12 captures):

| | Vj1d | FA3 | cuDNN |
|---|---|---|---|
| kernel time | 3.626 ms | 2.494 ms | 2.788 ms |
| tensor-core busy time | 1.66 ms | 1.74 ms | 1.73 ms |
| L2 read requests | 81.0M | 40.9M | 41.9M |
| dQ add-back requests | 76.7M | 43.6M | 77.9M |
| shared-memory stores | 41.9M | 10.9M | 10.3M |
| instructions | 888M | 625M | 809M |

**What the numbers show:**
- The tensor-core work is the same in all three, so the whole gap is time when our tensor cores sit idle.
- Our L2 reads are 1.98× FA3's. That matches the loop count exactly: 64-key blocks visit the 64-query tiles 299,520 times, FA3's 128-key blocks 152,064 times.
- cuDNN does as many dQ add-backs as we do and is still 23% faster. So the waiting around the add-backs costs more than their volume.

**Where our kernel waits (per-instruction samples):**
- ~14%: `__syncthreads` + fences before the dQ add-back (`GQA_bwd_vj1d_sm90.cu:609-611, 621-627`);
- ~10%: waiting for GEMMs;
- ~7%: the P/dS round trip through shared memory;
- ~4%: LSE/D loads;
- ~4%: dQ staging stores;
- ~14%: the softmax math itself.

**What FA3 does differently (from its code):**
- 128 keys per block, 1 loader warpgroup plus 2 compute warpgroups, registers 24/240 (`flash_bwd_launch_template.h:351`, `flash_bwd_kernel_sm90.h:64-65`);
- loads issued ahead of time (`mainloop_bwd…:511-555`);
- a separate dQ-writer warp (`:645-654, :953-957`);
- P kept in registers (`:279`).

128 keys need 256 registers per thread for dK/dV alone in one warpgroup, over the 255 limit. That is why the larger block size needs FA3's 3-group layout.

### 42.4 Precision: what dS from fp32 P changes (CPU emulation)

**Setup:** `bwd_precision_sim.py`, one KV head shared by 3 query heads, causal, T = 2048. The fp64 truth was checked against finite differences (error ≤ 1.5e-9).

**Result (RMS error as a multiple of the unavoidable bf16 rounding):**
- Normal inputs, ours today vs FA3's arithmetic: dQ 1.74-1.81 → 1.44-1.48; dK 2.05-2.07 → 1.45-1.47; dV 1.71-1.77 → 1.40-1.41.
- Large inputs: about 4-23% lower.

The gain is real but modest. Most of what remains comes from rounding P and dS to bf16 before the GEMMs, which every library does.

### 42.5 Scope: what the backward kernel serves

- **Trainer path:** `AttentionBackward.cpp:237` calls `gqa_flash_attn_backward_qkv`. The fast Hopper path is taken when the call is causal, head dim is 128, T % 64 == 0, the GPU is an H100/H200 and `BLUTRAIN_NO_SM90` is unset (`GQA_bwd_generic.cu`).
- **Same path for both kinds of training:** pretraining and SFT use the same path, so any backward speedup helps both.
- **Head ratio does not matter:**
  - G is a run-time number; the only rule is that query heads are a multiple of KV heads.
  - The test suite covers 18/6, 12/4, 8/8, 32/8 and 8/1.
  - The gap to FA3 was 1.32-1.42× on all 13 dense shapes (G = 3 and 4).
- **Head dim:** only 128 has the fast kernel.
- **Sequence length in practice:** the forward needs T % 256 == 0 (`gqa_v62_shape_ok`), so training always sends multiples of 256.
- **B300 is not affected:** head dim 128 causal is refused there (`GQA_bwd_generic.cu`, the `is_causal` throw).

### 42.6 The new backward kernel Vj2 (FA3 layout)

**File and switch:**
- New file `src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu`. Vj1d is unchanged and stays the default.
- Opt-in with `USE_VJ2=1`, a new mode in `GQA_bwd_generic.cu`.
- Dense only; packed rows still go to Vj1d. S % 128 != 0 also goes to Vj1d (training never sends it, and the forward+backward tests cannot reach it).

**Design (FA3's layout, built only from pieces already proven in our code):**
- 128 keys per block and 384 threads. Warp 0 of the loader issues TMA loads of K/V once and Q/dO/LSE/D two steps ahead. Warp 1 is the dQ writer. Two compute warpgroups own 64 keys each. `setmaxnreg` 24/240.
- The score GEMMs are swapped (Sᵀ = K·Qᵀ), so P and dS stay in registers for the dV and dK GEMMs. Only dS goes to shared memory, for the dQ GEMM.
- dS is computed from fp32 P.
- The compute groups hand dQ to the writer warp with named barriers; there is no block-wide barrier in the loop.

**Kept from Vj1d:**
- the TMA maps and swizzle;
- the fp32 dQ reduce-add;
- the one-writer bf16 dK/dV partials and the fixed-order head sum, so dK/dV stay bit-identical every run;
- the dQ convert.

**Proofs before the GPU:**
- 0 spills on both CUDA 13.0 and 13.4.
- SASS has exactly 40 HGMMA per step (16 Sᵀ/dPᵀ, 8 dQ, 16 register-A dV/dK) and exactly the 5 fences and 4 waits that were written.
- Vj1d, Vj1 and the forward files are md5-identical. The router's other kernels have identical SASS, 9,360 lines.
- Harness: `bwd_mode_name` knows `USE_VJ2`, and G4 checks Vj2's dK/dV bitwise.

**Results (H200, GPU 7):**
- Correctness: quick 144/144; full CLASSIC **605/605**, the same checks Vj1d passes. G9/G11 correctly refuse packed rows under `USE_VJ2=1`. dK/dV are bit-identical over all repeats.
- Precision against Vj1d on identical inputs (worst ratio per check):
  - dQ is lower in 57 of 84 checks, higher in 16, median 0.91×;
  - dK is lower in 61 of 84, higher in 16, median 0.91×;
  - the one known over-limit case (G6 D10 row 1022 dQ, 1.156) is now 0.891.
- Speed: the backward per call, run Vj1d/Vj2/Vj1d/Vj2 in the same session, then the libraries:

| backward, ms | Vj1d | Vj2 | FA3 | FA3 deterministic | cuDNN |
|---|---|---|---|---|---|
| B2 12/4 | 0.764 | 0.695 | 0.580 | 0.726 | 0.729 |
| B8 18/6 (our SFT model) | 4.326 | 3.852 | 3.079 | 3.717 | 3.468 |
| B8 32/8 | 7.549 | 6.744 | 5.365 | 6.533 | 6.117 |

- Across all 13 shapes:
  - Vj2 is 5.2-13.0% faster than Vj1d (mean 10.6%);
  - against FA3 it is 1.20-1.28× (was 1.32-1.43×);
  - against FA3 deterministic it is 0.96-1.05×;
  - against cuDNN it is 0.95-1.11×.
- Kernel time alone (nsys, B8 18/6): 3197 µs vs Vj1d 3615 µs (−11.6%). FA3 is 2495 µs. The helpers are about 0.3 ms in all three.
- The provisional bar (at least 10% on every shape) is met on 9 of 13. It misses on the small ones: B2 12/4, B2 16/4, B2 24/8, B4 12/4.

### 42.7 Profiling of Vj2 and the fix (not yet measured on the GPU)

**What ncu showed:**
- Data movement now matches FA3: L2 reads 42.7M (FA3 40.9M), dQ reduce 38.9M, L2 busy 48%.
- The block-wide waits are gone.
- What is left is instruction count: 1,096M vs FA3's 625M. It has two causes in the code:
  1. Every shared-memory read and write except stmatrix compiled to generic LD/ST (0 LDS/STS, `lg_throttle` stalls, ~11% of samples). The base was rounded to 1024 B through an integer cast, so the compiler lost the address space.
  2. GEMM descriptors were rebuilt from pointers for every wgmma: 5.4 extra instructions per HGMMA, 23% of all instructions.

**The fix:**
- the base is rounded by pointer arithmetic;
- base descriptors are made once, and siblings come from constant adds (the forward's `desc_add_lo`).

**SASS after the fix:**
- 2520 → 1816 lines; generic LD/ST 161 → 0; LDS/STS 0 → 96;
- IMAD 219 → 89, LOP3 433 → 233;
- the same 40 HGMMA, 0 spills.

It is built into the box library (file md5 `bbd977be`) but has not run on the GPU yet.

### 42.8 Bug found: causal head dim 64 on H100/H200 runs without the causal mask

- **Forward:** causal + head dim 64 + not B300 goes to `gqa_v2`, which is documented as non-causal (`GQA_fwd_generic.cu:59, 279-286`; the `_qkv` path at `:374-387`).
- **Backward:** it goes to `launch_gqa_backward`, which has no mask (`GQA_bwd_sm103.cu:400`). Only the B300 kernels mask.
- **Reachable:** yes. `bluscript.cpp:73` defaults to `head_dim = 64`. The guard at `:588` and `Config.h:270` assume head dim 64 has a causal fallback, which is true only on B300.
- **Production today:** the yaml sets 128 (`bluscript_config.yaml:14`), so production is fine. The GPT-2 trainer uses a different kernel.
- **Status:** not yet proven on the GPU. The fix is to refuse that case, as head dim 128 already is.

### 42.9 Where the SFT preprocessing lives

- **Script:** `~/Downloads/BluTrain_sft/sft/prepare_sft_data.py`, on the laptop only. It takes ultrachat_200k (pinned revision), uses the tulu-plain-v1 template and GPT-2 tiktoken, and writes `inputs.bin`, `labels.bin` (65535 = ignore), `valid_counts.bin` and `meta.json`.
- **Loader:** `sft/SftDataLoader.h`, and its test `sft/test_sft_dataloader.cpp`.
- **Data:** `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/` on the laptop and `~/gautam_k/sft_data/...` on the box. The box `meta.json` script hash equals the laptop script's SHA-256 (`2add2080…`).
- **Risk:** `sft/` is untracked in git, so it exists on the laptop only.

### 42.10 Files changed today

All in the laptop `~/Downloads/BluTrain_sft/Tensor-Implementations`, uncommitted, plus the 10-10 v2 files:

| file | md5 | what |
|---|---|---|
| `src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu` (new) | bbd977be | the Vj2 kernel |
| `src/Kernels/cuda/attention/arch/GQA_bwd_generic.cu` | 9274489e | the `USE_VJ2` mode; S % 128 != 0 goes to Vj1d |
| `Tests/kernel_tests/test_varlen_attention.cu` | 5a988de0 | Vj2 mode label; G4 checks Vj2 bitwise |

The box `~/gautam_k/v2_work` has the same files, and its library was rebuilt at 21:00. Backups of the files they replaced are in `backup_pre_vj2_1012/`.

### 42.11 Where the backward work stands, and the task list for 2026-10-13

| step | done? | result |
|---|---|---|
| 1. profile the old backward (Vj1d) to find where it lags | yes | the tensor cores sit idle. 64-key blocks reload Q/dO and add back dQ twice as often as FA3; one thread group does everything and waits around the dQ add-back |
| 2. first optimization: new backward Vj2 (FA3 layout) | yes, tested on the GPU | 605/605; 5-13% faster than Vj1d. Against cuDNN: faster on the 2 smallest dense shapes, 5-11% slower on the other 11. Against FA3: 20-28% slower. Within about 5% of FA3's deterministic mode |
| 3. profile Vj2 | yes | data movement now matches FA3. What is left is extra instructions (generic memory path, descriptors rebuilt for every wgmma) |
| 4. second optimization: fix those two | built on the box (md5 `bbd977be`) | 28% fewer instructions in the compiled kernel. **Not tested on the GPU yet** |

Tasks for 2026-10-13, in order. Every GPU run happens only after the user's OK, on GPU 7, with absolute paths in the scripts.

1. **Run 1:** quick correctness (CLASSIC quick) of the fixed Vj2, ~3.5 min.
2. **Run 2:** only if run 1 passes. Dense speed with Vj1d/Vj2 alternating on the 13 shapes, plus nsys and ncu on B8 18/6 (~5 min), to see whether the instruction cut turns into time.
3. **Ask the user about run 3:** the full CLASSIC, 605 checks, ~23 min.
4. **Stage 3:** move design A and B onto Vj2. A's plan needs 128-key backward tiles. Test with G8-G13. If Vj2 loses on short chats, keep Vj1d's 64-key path for them.
5. **Benchmark again** on the packed layouts and the dense shapes against FA2, FA3, FA4, cuDNN (packed and masked), PyTorch (varlen, Flex, SDPA) and **Meta xFormers**: its own packed mode, forward and backward, installed into `~/gautam_k/benv` only after the user's OK.
6. **The head-dim-64 bug (42.8):** a 1-minute proof test, then the refusal fix.
7. **Step 7:** the `attn_pack_mode` config setting, the packing script, the loader, the attention node, the end-to-end gradient test, and a real packed SFT run vs v1.
8. **Chat template decision (option A now, option B later):** give `<|system|>`, `<|user|>`, `<|assistant|>` and possibly a separate end-of-turn token their own ids in the 47 spare vocab rows. To be written up as a decision; the background is in the SFT report §3.2-3.4 (Tülu 3 Table 13: 51.6-53.0 across templates).

## 43. 2026-10-13: the fixed Vj2 tested, Phase 1.5 (one fix at a time), FA3/FA4/cuDNN compared line by line, the G-series, and the SFT pipeline walkthrough

All GPU runs were on the H200 box, one test at a time and each only after the user's OK: GPU 7 until 18:40, then GPU 2.
Unless a line says otherwise, numbers are for our model's attention shape (batch 8, 18 query heads, 6 KV heads, 4096
tokens, head dim 128 = "B8 18/6"). "Same session" means the versions were run alternately (A, B, A, B) in one sitting,
so clock changes hit both; every speed ratio below names what it is compared against.

Records: `profiling_reports/v2_bwd_rework_research_2026_10_12/gpu_runs/{vj2c_1013, prof_1013, g2_1013, g12_1013, g123_1013}/`
(each holds the run scripts, logs, CSV exports and a RESULT/RESULTS file); the analysis is in
`gpu_runs/vj2c_1013/PHASE15_ANALYSIS.md`; every kernel version is in `code_snapshot/`; the SFT proofs are in
`profiling_reports/sft_walkthrough_2026_10_13/`.

### 43.1 Testing the fixed Vj2 (yesterday's second optimization, file md5 `bbd977be`)

| test | what it checks | result |
|---|---|---|
| quick CLASSIC (144 checks) | forward + backward values against the reference | 144/144; all 88 error values identical to the first Vj2's run of 10-12, so the fix changed no math |
| compute-sanitizer memcheck / synccheck / racecheck, new driver `sanitize_vj2.cu` (B1, 6/2 heads, T256) | out-of-bounds, barrier misuse, shared-memory races | 0 / 0 / 0. nsys of the same program shows `gqa_bwd_vj2_kernel` really ran. Racecheck does not see TMA or wgmma accesses; those are covered by the value checks |
| speed, 13 dense shapes, Vj1d / Vj2 alternating + libraries | | Vj2 takes 18-22% less time than Vj1d (mean 20%); 1.05-1.15x FA3's time (mean 1.11x); faster than cuDNN on 12 of 13; faster than FA3 deterministic on 13 of 13 |
| nsys, B8 18/6 | kernel time | 2761 us (first Vj2 3197, Vj1d 3623, FA3 2486) |
| ncu, B8 18/6 | instructions, stalls | instructions 1,096M -> 840M; generic loads/stores 44.9M -> 0; tensor cores busy 51.4% -> 59.6% |
| full CLASSIC (605 checks) | adds G6 = all 5000 real chats alone, G4 = 1000 repeats, G5 = large inputs | 605/605; all 435 error values identical to the first Vj2's full run |

FA3 and FA3 deterministic differ in one template flag (`Deterministic`, the 5th bool of `CollectiveMainloopBwdSm90`,
proven from the kernel names in our nsys). With it on, the blocks add their dQ (and, for GQA, dK/dV) in a fixed order, so
the output is bit-identical every run but slower (B8 18/6: 3.28 ms vs 2.50 ms). Our Vj2 is in between: dK/dV bitwise
(one writer + fixed-order head sum), dQ by fp32 atomic adds. Plain FA3 is the fair speed target.

### 43.2 Phase 1.5: why the fixed Vj2 still lost to FA3 (research before any code)

Sources: our ncu of the fixed Vj2, FA3's ncu from 10-12 (same shape), and the FA3 source (commit `94e22c9`, the one
we benchmark; local clone `~/Downloads/sft_research/flash-attention`). FA3's kernel name shows it uses the same work
split as Vj2 (64 queries x 128 keys, swapped score GEMM, 2 compute warpgroups of 64 keys, dQ split by head columns,
P/dS in registers), so a line-by-line comparison is fair.

| | FA3 | fixed Vj2 |
|---|---|---|
| kernel time | 2.494 ms | 2.778 ms |
| shared-memory loads | 19,464,192 | 19,464,192 (identical) |
| shared-memory stores | 10.9M | 21.8M |
| store bank conflicts | 1.8M | 56.3M |
| tensor cores busy | 68.8% | 59.6% |
| stall samples above FA3 | | +36k, of which "shared-memory queue full" +24k and "waiting for a shared-memory result" +16k |

1. **dQ staging stores (root cause):** 16 `STS.64` per thread per step in the SW128 layout of the 2-D TMA reduce; in each
   half-warp rows r and r^1 hit the same 8 banks, so 4 wavefronts instead of 2. Checked exactly: 19,464,192 x 4 +
   2,359,296 x 8 = 96,731,136 wavefronts = ncu's number. FA3 (`mainloop_bwd_sm90_tma_gmma_ws.hpp:195-198, 645-654,
   951-957`) writes dQ in thread order with 128-bit stores (no conflicts) and sends 16 KB with one 1-D bulk reduce-add.
2. **LSE values arrive late:** the first FFMA that uses each loaded LSE value stalls (9.7% of samples).
3. **dQ hand-off:** the fence and barrier after the dQ stores wait for those stores (7%).
4. **Block order (tail):** SMs busy 96.9% of elapsed cycles (FA3 98.6%); FA3 uses `SingleTileBwdLPTScheduler`
   (`tile_scheduler.hpp:368-469`): heaviest key blocks first, 8 heads per L2 section.
5. **Epilogue staging:** the bf16 dK/dV staging writes plain 128-B rows: 8-way bank conflicts (2.2% of samples).

### 43.3 Phase 1.5 fixes, one at a time (each built on the one before, measured against it in the same session)

| fix | what I changed | correctness | speed vs the version before (13 shapes) | profile change (B8 18/6, ncu) | kept? |
|---|---|---|---|---|---|
| 1 dQ path | FA3's scheme: thread-order 128-bit stores (8 `STS.128` instead of 16 `STS.64`), one 1-D `cp.reduce.async.bulk` per group, a global fp32 accumulator in fragment order, and a new convert kernel `convert_dq_frag_vj2` (padded shared memory, coalesced writes) | 144/144, identical | 0.932-0.968, mean 0.943, faster on 13/13 | 2.778 -> 2.599 ms; store conflicts 56.3M -> 18.3M; tensor busy 59.6 -> 65.2%; convert kernel 106.2 -> 105.9 us | yes |
| 2 LSE order | LSE loads moved between the S and dP GEMM issues | 144/144, identical | mean 1.003 (4/13) | the wait did not move | no, removed (solved later by G3) |
| 4 block order | 1-D grid, heaviest key blocks first, 8 (batch, head) pairs per L2 section (FA3's sizing: 40 MB / 4 MB per head); a script checked every block runs exactly once on 336 shapes | 144/144, identical | 0.904-1.010, mean 0.967, faster on 10/13 (batch 2: -5 to -10%; batch 8 about flat) | SMs busy 96.9 -> 98.8%; DRAM 713 -> 745 MB | yes |
| 5 epilogue | bf16 dK/dV staging in the 128-B swizzle, dK/dV TMA store maps `SWIZZLE_128B` | 144/144, identical | 0.962-1.023, mean 0.989, faster on 12/13 | 2.590 -> 2.545 ms; store conflicts 18.3M -> 1.79M (FA3 1.81M); tensor busy 66.7% | yes |
| 4-pair sections | fix 4 with a 20 MB budget | (not needed) | mean 0.994, batch 8 flat | DRAM 745 -> 724 MB | no |

Full CLASSIC on the result (snapshot f5, Vj2 `826ae057`): **605/605**, all 435 error values identical to the 15:17 full
run. The block order made dQ's run-to-run differences larger (G4 B8: max 4.9e-4 -> 2.0e-3, 1.9% of elements differ;
limit 1.56e-2), because more blocks now add into the same dQ tile at the same time. dK/dV stay bitwise.

### 43.4 FA3, FA4, cuDNN, FA2: who makes them and what runs on the H200 (proofs)

- **FA3:** `Dao-AILab/flash-attention`, folder `hopper/`, C++ on NVIDIA CUTLASS, BSD-3, built from commit `94e22c9`.
  Paper arXiv 2407.08608 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao).
- **FA4:** same repository, folder `flash_attn/cute/`, package `flash-attn-4` (author: Tri Dao), written in Python CuTe
  DSL. Its README says "for Hopper and Blackwell GPUs". On compute capability 9.x it picks its own Hopper kernels
  (`interface.py:1629-1631` forward, `:2883` backward). Its Blackwell features are only in the sm100 files: `tcgen05`
  (20 / 15 uses in the forward / backward sm100 files, 0 in sm90), tensor memory (117 / 143 vs 0), 2-CTA MMA, and the
  software exp2 (`flash_fwd_sm100.py:104-109`). So on the H200 FA4 runs a Hopper version of FA3's design.
- **On the H200 FA4 loses to FA3:** forward 1.12-1.17x FA3's time on all 13 shapes; backward about equal (mean 1.01).
- **FA4's software exp2 would not help us on Hopper:** the exp unit is busy 14% (our backward), 34% (our forward) and
  43% (FA3's forward) of the time, nowhere near its limit.
- **cuDNN:** NVIDIA's own closed library (9.24), called through PyTorch SDPA; its kernels are named
  `cudnn_generated_fort_native_sdpa_sm90_flash_bprop/fprop_wgmma...`, not FA code.
- **FA2:** PyTorch's built-in copy (`pytorch_flash::...` kernels).

### 43.5 Profiling round against every library (GPU 2, 18:48-19:38)

**Dense, 13 shapes, same session** (`prof_1013/1_dense_scoreboard.txt`; ours = v62 forward + Vj2 f5):

| ours / library, time ratio | FA3 | FA4 | cuDNN | FA3 det | FA2 |
|---|---|---|---|---|---|
| forward, mean (we win on) | 1.08 (4/13) | 0.96 (6/13) | 0.93 (7/13) | 1.16 (2/13) | 0.54 (13/13) |
| forward, B8 18/6 | 1.21 | 1.08 | 1.03 | 1.25 | 0.57 |
| backward, mean | 1.01 (5/13) | 1.00 (5/13) | 0.85 (13/13) | 0.83 (13/13) | 0.47 (13/13) |
| backward, B8 18/6 | 1.04 | 1.05 | 0.92 | 0.86 | 0.48 |
| forward + backward, mean | 1.03 (4/13) | 0.99 (5/13) | 0.87 (13/13) | 0.90 (13/13) | 0.48 (13/13) |
| forward + backward, B8 18/6 | 1.08 | 1.06 | 0.95 | 0.94 | 0.50 |

**SFT packed rows, 7 layouts of `segments_1012.txt`, same session** (`prof_1013/2_packed_scoreboard.txt`; best of our A/B;
our packed backward is still Vj1d because Vj2 refuses packed calls):

| best ours / library | FA3 varlen | FA4 varlen | cuDNN varlen | Flex | torch varlen (FA2) |
|---|---|---|---|---|---|
| forward, mean | 0.99 (3/7) | 0.85 (6/7) | 0.88 (7/7) | 0.57 (7/7) | 0.54 (7/7) |
| backward (Vj1d), mean | 1.18 (2/7) | 1.21 (2/7) | 1.08 (2/7) | 0.91 (7/7) | 0.66 (7/7) |
| forward + backward, mean | 1.13 (2/7) | 1.11 (2/7) | 1.03 (2/7) | 0.80 (7/7) | 0.63 (7/7) |

At the API level the packed forward looks level with FA3, but kernel to kernel (ncu, typical real rows `real_rows_752`)
ours (design A) is 502 us vs FA3's 430 us: FA3's Python call adds about 0.1 ms per call.

**Per-instruction profiles (ncu, B8 18/6):**

| | ours v62 fwd | FA3 fwd | FA4 fwd | cuDNN fwd | ours Vj2 bwd (f5) | FA3 bwd | FA4 bwd | cuDNN bwd |
|---|---|---|---|---|---|---|---|---|
| time (us) | 1079 | 877 | 952 | 1009 | 2545 | 2481 | 2478 | 2770 |
| instructions | 360M | 306M | 316M | 369M | 796M | 625M | 555M | 809M |
| tensor cores busy | 64.4% | 82.9% | 74.4% | 67.9% | 66.7% | 68.3% | 68.9% | 60.0% |
| SMs busy (of elapsed) | 95.7% | 99.6% | 99.0% | 96.5% | 98.8% | 98.4% | 96.9% | 97.0% |

Trap: ncu reports DRAM bytes in Mbyte for some kernels and Gbyte for others (FA3's backward reads 983 MB, ours 744 MB),
and the packed run reports time in us instead of ms. Every table above is converted.

**Gap list (each from SASS or source):**

| # | kernel | gap | proof | fix |
|---|---|---|---|---|
| G1 | fwd + bwd | register-A GEMMs half width: forward P.V and backward dV/dK as 2 x `m64n64k16` | fwd 16 per key block vs 8 `m64n128k16` in FA3/FA4/cuDNN (`GQA_fwd_sm90.cu:305-330`); bwd 16 per step vs 8 | one `m64n128k16` with a B descriptor spanning both head halves |
| G2 | fwd + bwd | GEMM descriptors built in per-thread registers | R2UR fwd 40.2M vs 0.7M, bwd 59M vs 2M | warpgroup index broadcast with `__shfl_sync` (FA3 `warp_uniform`, `hopper/utils.h:621-623`) |
| G3 | bwd | LSE loads queued behind GEMM instructions | our exp FFMAs wait 11.8% vs FA3 5.5%; FA3 issues them after its first S HGMMA | load LSE before the S GEMM |
| G4 | bwd | dQ scaled every step | FMUL 80M vs 41M; FA3 scales in its convert kernel | scale once at the end (parked: only removes instructions) |
| G5 | fwd | tail | SMs busy 95.7% dense / 93.7% packed vs 99.6% / 98.3%; v62 runs 2304 equal units on 132 SMs = 17.45 rounds | smaller units, heaviest first |
| G6 | fwd | output written with per-thread global stores | 2.1% (dense) / 4.2% (packed) of stall samples | shared memory + TMA store |
| G7 | packed bwd | still Vj1d | 1.18x FA3 | packed A/B modes in Vj2 (Phase 2) |

### 43.6 The G-series so far (backward; each measured against the version before it)

| step | change | correctness | speed (mean of 13 / B8 18/6) | ncu B8 18/6 | kept? |
|---|---|---|---|---|---|
| G2 | `c = __shfl_sync(0xffffffff, threadIdx.x / 128, 0) - 1` in v62 and Vj2 | 144/144 identical | fwd 0.990 / 0.993, bwd 0.999 / 1.008 | bwd R2UR 59.1M -> 0.66M, IMAD 74 -> 41M, but uniform-datapath instructions 137 -> 212M and time 2545 -> 2547 us | yes (removes real work, no cost) |
| G1 | dV and dK as one `m64n128k16` per k-step; B descriptor LBO = 8192 B (CUTLASS `Major::MN` B128 layout) | 144/144 identical (dK/dV bitwise) | 0.997 / 0.997 | 2547 -> 2532 us; 8 + 24 HGMMA per step = FA3's mix | yes |
| G3 | the 8 LSE `LDS.64` before the first S HGMMA | 144/144 identical | 0.994 (9/13) / 0.995 | 2532 -> 2512.5 us; LSE wait 11.3% -> 3.1% (FA3 5.5%); tensor busy 67.0% | yes |

What G2 proved: removing 59M instructions changed nothing in time, so neither kernel is limited by instruction issue;
the time is lost while the tensor cores wait. After G3 the largest wait in the backward is on the tensor cores
themselves ("warpgroup arrive" 16.8%).

### 43.7 Where the backward started today and where it is now

Backward kernel, step by step (B8 18/6). The kernel-time columns are the main kernel alone, measured by ncu at fixed
clocks; FA3 appears only in the "x FA3" column. The same-session column says which version of OUR kernel each row was
raced against (alternating runs, mean over 13 shapes); the last column multiplies those races together, so every row
is against Vj1d:

| step | kernel time (us) | x FA3's time (2481 us) | same-session race (mean of 13 shapes) | running total vs Vj1d's time |
|---|---|---|---|---|
| Vj1d, old default (measured 10-12) | 3626 | 1.46x | - | 1.00x |
| first Vj2 (10-12) | 3195 | 1.29x | 0.89x Vj1d's time | 0.89x |
| + memory path and descriptor fixes (43.1) | 2778 | 1.12x | 0.80x Vj1d's time | 0.80x |
| + fix 1 dQ path | 2599 | 1.05x | 0.943x the 43.1 version's time | 0.757x |
| + fix 4 block order | 2590 | 1.04x | 0.967x the fix-1 version's time | 0.732x |
| + fix 5 epilogue | 2545 | 1.03x | 0.989x the fix-4 version's time | 0.724x |
| + G2 | 2547 | 1.03x | 0.999x the fix-5 version's time | 0.723x |
| + G1 | 2532 | 1.02x | 0.997x the G2 version's time | 0.721x |
| + G3 (current) | **2512** | **1.01x** | 0.994x the G1 version's time | **0.717x** |
| FA4 / cuDNN | 2478 / 2770 | 1.00x / 1.12x | - | - |

Rows 2 and 3 are raced against Vj1d because that is how those runs were done: the fixed Vj2 was never raced against the
first Vj2 in one session. The kernel-time column is our model's shape only; the race columns are means over 13 shapes and
include the small helper kernels, so the two differ a little.

Start of day vs now (API time per call, ms, same-session runs; "now" = 18:48, before the G-series):

| case | start of day | now | FA3 | cuDNN | now / FA3 | now / cuDNN |
|---|---|---|---|---|---|---|
| dense backward, B8 18/6 | 4.330 (Vj1d) | 3.185 | 3.076 | 3.466 | 1.04 | 0.92 |
| dense backward, mean of 13 (ratios) | 1.38 x FA3, 1.19 x cuDNN | | | | 1.01 | 0.85 |
| dense forward, B8 18/6 (unchanged code) | 1.166 | 1.197 | 0.993 | 1.158 | 1.21 | 1.03 |
| dense forward + backward, B8 18/6 | 5.495 | 4.382 | 4.069 | 4.623 | 1.08 | 0.95 |
| dense forward + backward, mean of 13 (ratios) | 1.30 x FA3, 1.11 x cuDNN | | | | 1.03 | 0.87 |
| packed forward, real_rows_752 (unchanged) | 0.535 | 0.535 | 0.532 | 0.587 | 1.01 | 0.91 |
| packed backward, real_rows_752 (Vj1d, unchanged) | 1.908 | 1.908 | 1.561 | 1.740 | 1.22 | 1.10 |
| packed forward + backward, real_rows_752 | 2.443 | 2.443 | 2.092 | 2.327 | 1.17 | 1.05 |

### 43.8 SFT pipeline walkthrough (morning, before the GPU was free)

- **Flow:** `sft/prepare_sft_data.py` downloads ultrachat_200k (pinned revision) with Hugging Face `datasets` (its own
  Arrow cache), applies the `tulu-plain-v1` template, tokenizes with tiktoken GPT-2, builds 4096-token rows in memory
  and writes `inputs.bin`, `labels.bin` (uint16), `valid_counts.bin` (uint32) and `meta.json` with `.tofile()`
  (`:192-194`). There are no pickle files anywhere in the SFT path. The C++ loader (`sft/SftDataLoader.h:150-158`)
  memory-maps the `.bin` files and copies one batch to the GPU per step (`:81-83`). Python reads the same files with
  `np.memmap` (Kathir's `sft.py:81` does).
- **Labels:** `labels = where(mask[1:], ids[1:], 65535)`; inputs = `ids[:-1]` then padding with 50256 (GPT-2's only
  special token). 65535 is not a token id (vocab 50257, rows padded to 50304); the loss kernel counts the valid labels
  for the denominator, skips the softmax for ignored rows in the forward and writes zero gradient rows for them in the
  backward. The end-of-text after the answer appears only as a label (the target of the last answer token).
- **Padding id does not matter (proof, `pad_grad_proof.py`, float64, tiny 2-layer causal transformer):** pad = EOS vs pad =
  a separate id: max gradient difference over all weights 0.0 (single and two-turn rows); pad vs no padding: <= 1.8e-15.
  Reason: padding is on the right, attention is causal, and padded positions have ignored labels.
- **Prompt tokens and the answer loss (`prompt_grad_check.py`, `prompt_seeds.py`):** the answer loss changes the
  embeddings of the prompt words too (e.g. 1.22 vs 1.90 under an all-token loss), because the answer attends to them. It
  does not teach predicting the prompt: over 20 seeds, one SFT step lowered the probability of the next prompt word in 14
  of 20 seeds (mean change -0.032 and -0.040 at two positions) and raised the first answer word in 18 of 20 (mean +0.35);
  an all-token step raised the next prompt word in 16 and 13 of 20.

### 43.9 Methods used today

- **Library snapshots on the box** (`~/gautam_k/v2_work/snapshots/<name>/lib/libtensor.so`): the test binary uses
  RUNPATH, so `LD_LIBRARY_PATH` picks the snapshot (`ldd` checked each time). Same-session A/B = two snapshots alternated.
- **Rule from the user (G2):** keep a change that removes real redundant work when it is correct and not slower; revert
  only if it is slower, wrong, or adds complexity. Every change is still measured alone, built on top of the kept one.
- **Library profiling:** `BENCH_WARM=0 BENCH_ITERS=1`, `--kernel-name-base demangled`, regex per kernel family; ncu
  replays each kernel ~40 times, so a library capture takes 8-9 minutes.

### 43.10 Code state (laptop `~/Downloads/BluTrain_sft/Tensor-Implementations` = box `~/gautam_k/v2_work`, uncommitted)

| file | md5 | change today |
|---|---|---|
| `src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu` | a956e789 | fixes 1, 4, 5 + G2 + G1 + G3 (from `bbd977be`) |
| `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu` | 060e6946 | G2 (from `698d2705`) |
| `Tests/kernel_tests/sanitize_vj2.cu` (new) | d23c09e3 | compute-sanitizer driver |
| `src/Kernels/cuda/attention/arch/GQA_bwd_generic.cu` | 9274489e | unchanged today |
| `Tests/kernel_tests/test_varlen_attention.cu` | 5a988de0 | unchanged today |

Box snapshots: `v2_bbd977be`, `f1`, `f2`, `f4`, `f4s4`, `f5`, `g2`, `g12`, `g123` (current). The full 605 check has not
run on g123 yet (it passed on f5; g2, g12, g123 passed the quick 144 with identical errors).

### 43.11 Clarifications asked today (names, what applies where, how to read the tables)

**Two different "v2"s.** "v2" (our SFT plan) = packing several chats into one 4096-token row. "Vj2" = the name of a
backward kernel version (the Vj series). They are unrelated names.

**The backward kernels for H200** (`GQA_bwd_generic.cu`, the mode list and the comments above it):

| kernel | file | chosen by | design | repeat-exactness | packed SFT rows |
|---|---|---|---|---|---|
| v44 | `GQA_bwd_sm90.cu` | `USE_DETERMINISTIC_DQ=1` | one block per (batch, KV head, 64-key tile); the block sums the G query heads itself | dQ bit-identical (slowest) | no |
| Vj1 | `GQA_bwd_vj1_sm90.cu` | `USE_FLASH=1` | one block per (batch, query head, 64-key tile), 128 threads, 2 blocks per SM; dK/dV of the G query heads summed by atomic TMA reduce-adds in global memory | dK/dV change by ~1 bf16 ULP run to run | no |
| Vj1d | `GQA_bwd_vj1d_sm90.cu` | default | Vj1's design, but each query head writes its dK/dV once and a small kernel sums the G heads in a fixed order | dK/dV bit-identical (as cuDNN) | yes (A and B since 10-10) |
| Vj2 | `GQA_bwd_vj2_sm90.cu` | `USE_VJ2=1` | FA3 layout: 128 keys per block, 384 threads, 1 loader + 2 compute warpgroups | as Vj1d | not yet (G7) |

Order they were made: v44 -> Vj1 (3x more blocks, so small batches fill the GPU) -> Vj1d (dK/dV made repeat-exact, made
the default) -> Vj2 (10-12). The forward is one kernel, v62, with modes 0 (normal), B and A (packed).

**What today's work applies to.** Every backward fix of today is inside Vj2, so it speeds up the backward of normal
causal rows: pretraining, and SFT v1 (one chat per row + padding, which is a normal causal row). It does not reach SFT v2
packed rows yet: Vj2 refuses them and they run Vj1d. G7 adds modes A and B to Vj2 (packing only changes where each chat
starts and adds the chat-border masks; the main loop is the same), after which the fixes apply to packed rows too and Vj2
can replace v44, Vj1 and Vj1d. Today's benchmarks covered both normal and packed rows (43.5). Vj2 is still opt-in
(`USE_VJ2=1`); making it the default is a later step.

**Fix (b) of 43.1 and G2 are two different things.** Fix (b) changed how many instructions make a descriptor; G2 changed
in which kind of register it is made. The tensor-core instruction reads its descriptor only from a uniform register (one
value per warp), so a descriptor made in normal per-thread registers needs a copy (`R2UR`) before every multiplication.
Counts are executed instructions per backward run, B8 18/6:

| version | how the descriptor is made | made in | instructions spent making descriptors | copies before each multiplication | total instructions |
|---|---|---|---|---|---|
| first Vj2 (10-12) | rebuilt from a pointer, ~5.4 instructions per multiplication | normal registers | ~250M (23% of all) | 58.5M | 1096M |
| after fix (b) (made 10-12, tested 10-13) | base made once, then 1 add per multiplication | normal registers | ~5.4 -> 1 per multiplication | still 58.5M | 840M (fix a + fix b) |
| after fixes 1, 4, 5 (before G2) | same | normal registers | same | 59.1M | 796M |
| after G2 | same | uniform registers (once per warp) | same | 0.7M | 743M |

G2 also cut IMAD 74M -> 41M; part of the work moved to the uniform unit (137M -> 212M), so the total fell by 53M.

**LSE and exp.** exp(x) = e^x (e = 2.718). LSE (log-sum-exp) is one number per query row that the forward saves: the log
of that row's softmax sum. The backward rebuilds every probability as p = exp(score - LSE) without re-adding the row; each
thread reads the LSE of its 16 query rows per step (8 x `LDS.64`). The GPU's exp unit computes 2^x, so the scores are
multiplied by log2(e) = 1.4427 first and the saved LSE values are pre-multiplied by 1.4427 (one multiply fewer per score).

**How to read the step table in 43.7.** FA3 appears only in the "x FA3" column. The same-session column names which version
of our kernel each row was raced against (rows 2-3 against Vj1d, from row 4 against the version in the row above); the last
column multiplies the races, so every row is against Vj1d.

### 43.12 Next

1. Full CLASSIC (605) on g123.
2. Forward, where the gap is (1.21x FA3 on B8 18/6): G1 for P.V (`m64n128k16`, V descriptor LBO = 16384 B), G5 tail,
   G6 output through shared memory + TMA store; same method (quick check, same-session speed, ncu, keep what is correct
   and not slower).
3. G7 / Phase 2: design A and B in Vj2, tests G8-G13, packed benchmark incl. xFormers (install only with OK).
4. Later: the head-dim-64 bug (42.8), Step 7 (training config switch, packing script, loader, packed SFT run), chat
   template option B, and FA4's Blackwell methods for our B300 kernels.
