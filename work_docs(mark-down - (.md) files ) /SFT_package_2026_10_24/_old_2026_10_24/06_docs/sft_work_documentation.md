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
| `pack_dataset()` | Best-Fit-Decreasing bin packing | different problem |
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
| packing | `bfd`, `bfd_split`, `wrapped` | concat-then-split, First-Fit (`components/data/packing.py`) | per-sample packing (`training/datasets/sft_dataset.py`) | multipack / `neat_packing` | v2 |
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
| Batch | 64 chats per step, fixed [8, 4096] rows (about 70% padding) | 512 chats per step, sorted by length, cut into micro-batches of ≤ 16,384 padded tokens (little padding). (10-14: the copy we have, `kathir_sft/sft.py:54`, defaults to `global_batch` 128; where 512 came from is not recorded, probably his run config.) |
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
| v2: First-Fit-Decreasing packing + block-diagonal mask (10-14: the code was First-Fit, the label said Best-Fit; Best-Fit gives the same 1,517 rows) | 99.6% | 27.8% | ×3.60 |
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
  - Best-Fit-Decreasing packing (`data_utils.py:739`);
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

**Data loads:** D1 real v1 rows; D2 2048+2048; D3 1000+3096; D4 1+4095 / 4095+1; D5 64x64, 32x128; D6 4096 one-token chats (D6q: 512 + pad); D7 63/65/127/129/255/257 repeated; D8 random lengths with pad tails 0/1/16/63/64/65/255; D9 an all-pad row; D10 the real stage-0 rows packed First-Fit-Decreasing (1517 rows); D11 the same 300-token chat at offsets 0/1/63/64/65/127/128/3796.

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
3. **Step 7:** packing script (whole chats, First-Fit-Decreasing, shuffled rows), loader output (tokens, labels, positions, `seg_start` or the plan), the attention autograd node passing it to forward and backward, then a real packed SFT run against v1. Open decision: leftover rows (proposed: endless stream reshuffled every epoch; validation scores all rows).
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

## 44. 2026-10-14

### 44.1 Future task: packing changes how many optimizer steps we take (research + ablation before the first packed SFT run)

**Where it comes from.** A blog a colleague shared (`djdumpling.github.io/2026/01/31/frontier_training.html`, section
"sequence packing", which cites Hugging Face's SmolLM3 training playbook): packing puts more chats into each optimizer
step, so the same data gives fewer steps. This hurt small datasets: in SmolLM3 an effective batch of 128 lowered IFEval
by up to 10%, and above 32 there was an average drop. For large datasets packing was "almost always beneficial".

**What it means for us (stage-0: 5,000 chats in 1,517 packed rows = 3.3 chats per row):**

| | rows per step | chats per step | steps per epoch |
|---|---|---|---|
| v1 today (one chat per row) | 64 | 64 | 78 |
| v2 packed, step kept at 64 rows | 64 | ~211 | ~24 |
| v2 packed, step kept at ~64 chats | ~19-20 | ~64 | ~78 |

The real runs may use much bigger datasets, where fewer steps matter less; with stage-0 (5,000 chats) it matters now.

**Plan (before the first packed SFT run, Step 7):**
1. **Research:** how TRL / SmolLM3, Tülu 3, OLMo and Hermes-4 set the step size when packing (chats, tokens or rows per
   step); what "effective batch" means in each; how the learning rate is set when the batch changes.
2. **Ablation on stage-0** (same base model and settings, only this changes):
   - (a) v1, no packing, 64 chats per step;
   - (b) v2 packed, ~64 chats per step;
   - (c) v2 packed, 64 rows per step (~211 chats).

   Compare: validation loss on answer tokens, likelihood evals (HellaSwag-style), and base-vs-SFT forgetting checks.
   IFEval needs text generation, which BluTrain does not have yet.
3. **Precaution in the loader:** define the step by chats or tokens, not rows (already the plan); log chats and trained
   tokens per step.
   - Detail: with 8 rows per micro-batch, ~20 rows per step is not a multiple of 8, so the loader needs a token budget
     per step or variable micro-batches. Decide in Step 7.

**Exact numbers from the trainer's own formula** (added 2026-10-14; `bluscript_zero_ckpt.cpp:943-958`: steps = epochs x
micro-batches per epoch / GRAD_ACCUM, warmup = 3% of the steps; the trainer refuses rows per step that are not a multiple
of B x GPUs, `:947-952`), 1 GPU, 2 epochs, stage-0 packed into 1,517 rows:

| setup | B | rows per step | steps (2 epochs) | warmup | chats per step |
|---|---|---|---|---|---|
| v1 today | 8 | 64 | 156 | 5 | 64 |
| v2, step kept at 64 rows | 8 | 64 | 47 | 1 | ~211 |
| v2 | 8 | 24 | 126 | 4 | ~79 |
| v2 | 8 | 16 | 189 | 6 | ~53 |
| v2 | 4 | 20 | 151 | 5 | ~66 |
| v2 | 8 | 20 | refused (not a multiple of 8) | - | - |

Full explanation, from zero, with a tiny worked example: `work_docs/sft_packing_algorithms_bfd_ffd.md` section 9
(script `profiling_reports/q10_packing_sim_2026_10_08/packing_stats_detail.py`).

**What the blog says, and what it does not say** (re-checked word for word on 2026-10-14):

- Section "sequence packing" (`#sequence-packing`): "An effective batch size of 128 hurt evals like IFEval by up to 10%;
  for effective batch sizes larger than 32, there was an average drop in performance (for SmolLM3 and the dataset)."
  These are Hugging Face's SmolLM3 results (the blog's figure links to their smol-training-playbook), not a paper. "This
  especially hurts small datasets where each sample matters more" is a separate, general sentence: the blog does not
  say the 10% drop was measured on a small dataset.
- Section "learning rate and epochs" (`#learning-rate-and-epochs`): "When packing is enabled, it's safer to decrease the
  learning rate further due to the larger effective batch size and getting fewer updates for the same token budget."
  That is the blog's only advice for packing. **Add to the ablation:** run (d) = packed, 64 rows per step, lower learning
  rate.
- The blog does **not** suggest changing the micro-batch, the gradient accumulation or the step size by the number of
  chats per row. Counting each step in chats or tokens is **our** plan (from 10-08), not the blog's.
- "Effective batch size" = rows (sequences) per optimizer step = rows per micro-batch x accumulation steps x GPUs (ours
  64). It is not the number of chats per row (stage-0: 3.3). Chats per step = effective batch size x chats per row.

**Ablation plan, final form (10-14; measured with `profiling_reports/q10_packing_sim_2026_10_08/packing_step_options.py`):**
run 1 = v1 padding, 64 chats/step, 156 steps; run 2a = pack each step's 64 chats (17-23 rows, 95.9% real, 156 steps);
run 2b = pack once + fill steps to 61,075 trained tokens (51-82 chats, 18-22 rows, ~150 steps); run 2c = B=4 x 20 rows
(~66 chats, 151 steps, no trainer change); run 3 = 64 rows (~211 chats, 47 steps); run 4 = run 3 + lower LR. Every run twice (two seeds). Full table: packing doc section 9.6.

### 44.2 Packing rules, over-long chats, TRL's three strategies (full detail: `work_docs/sft_packing_algorithms_bfd_ffd.md`)

- **First-Fit-Decreasing vs Best-Fit-Decreasing.** Both sort the chats longest first. First-Fit puts each chat into the
  first row that has room; Best-Fit puts it into the row that will have the fewest empty slots left.
  - Neither always wins. With rows of 20, chats 15, 9, 7, 4, 3, 2 need 2 rows with Best-Fit and 3 with First-Fit;
    chats 15, 9, 7, 3, 2, 2, 2 need 2 with First-Fit and 3 with Best-Fit.
  - On stage-0 both give **1,517 rows** (99.60% real tokens); the minimum possible is 1,511 (6,188,911 tokens / 4096,
    rounded up), and a slower subset-sum method does reach it (1,502 rows completely full; found by a review on
    10-14, which corrected my earlier claim that 1,511 could not be reached). Sorting longest first matters more than
    the choice of rule.
  - **First-Fit vs Best-Fit, measured (packing doc section 13; two studies, ~27,000 random tests + every small input):**
    on data like ours they always give the same number of rows; where they ever differ (by one row), Best-Fit is the
    one with fewer rows about 15 times more often (296 vs 19 on all 3.1 million small inputs). Speed with a search tree:
    First-Fit 1.3-1.9x faster, both under 1 s for 1,000,000 chats; the simple loop takes ~2 minutes at 200,000 chats.
    Theory: the same worst case, 11/9 x best (+ 6/9 for First-Fit, tight). Either is fine for us; write it with a
    search tree. Best-Fit has the tiny packing edge (and is TRL's choice); First-Fit is what our tests use.
- **Which one we use:** First-Fit-Decreasing, in `pack_sim.py` (10-08) and in the attention test data (D10). The 10-08
  table in §38.5 said "Best-Fit"; the code was First-Fit. Label fixed today.
- **TRL packs per group of 1,000 chats** (the `datasets` `map` default). On stage-0 that gives 1,521 rows (99.34%) vs 1,517
  for the whole set at once.
- **Over-long chats:** we drop them (`prepare_sft_data.py:167`; decided 09-25 in §32.2 because cutting removes the end-token
  target). torchtitan drops; TRL and Megatron cut the end; TRL `bfd_split` cuts into pieces. Stage-0 dropped 10 of
  5,010. Not yet measured: the drop rate on long reasoning or agentic datasets. Check `too_long` in `meta.json` first.
- **TRL strategies:**
  - `bfd` cuts over-long chats, then Best-Fit; each chat whole in one row; boundaries recorded.
  - `bfd_split` cuts them into pieces of `max_length`; every piece is packed and attended as its own chat, so pieces of
    one chat cannot see each other.
  - `wrapped` writes all chats into one stream and cuts it every `max_length` tokens. No boundaries are recorded, so the
    attention is plain causal over the row (cross-document attention) and the last token of one chat is trained to
    predict the first token of the next.
- **Varlen attention prevents cross-document attention**; it does not do it. TRL uses it for `bfd` and `bfd_split`, not
  for `wrapped`. Our designs A and B both block it.
- A second explanation of TRL's strategies and varlen (from another chat) was merged into that doc in full the same day;
  13 of its statements were wrong or unverified and are corrected in place, listed in the doc's section 12 (main ones:
  TRL's `wrapped` cannot use varlen; a `bfd_split` row Best-Fit does not produce; the varlen kernel has no row
  dimension; TRL's packing keeps chats apart with FlashAttention (varlen) and, in this `transformers` version, also with SDPA /
  eager for models that pass the positions to the mask builder; TRL still warns for non-FlashAttention).
- The doc also explains, from zero: attention, causal attention, and why **plain causal attention on a packed row causes
  cross-document attention** (one chat per row: none; 3 chats in a 10-slot row: 33 of 55 pairs cross chats), and what
  "boundaries kept" means: per chat in `bfd`, per chunk in `bfd_split`, none in `wrapped`, per chat in ours.

### 44.3 LoRA and QLoRA (asked today; we chose full fine-tuning on purpose)

- **LoRA:** the original weight table W is frozen. Two small tables are added, A (r x d) and B (d x r), and the layer
  computes W·x + s·B·(A·x). Only A and B are trained.
  - B starts at zero, so the model starts unchanged. A starts random; if both were zero, neither would ever get a
    gradient.
  - For one of our 2304 x 2304 tables with r = 16: 73,728 trained numbers instead of 5,308,416 (1.4%).
  - No gradient or optimizer state for W, and the dW GEMM is skipped. After training, W + s·B·A is computed once.
- **QLoRA:** the same, with W stored in 4 bits and turned back into bf16 before each GEMM.
- **Relation to SFT:** SFT decides what we train on (chats, answer-only loss); LoRA decides which weights change.
- **Why full fine-tuning:** our model (1,588,229,888 numbers) fits on one H200 with full fine-tuning, and full
  fine-tuning learns more ("LoRA Learns Less and Forgets Less" also shows LoRA forgets less). Already in §6, §14, §15
  and the "add later" list (item 4). Nothing built so far would be wasted by adding LoRA later.

### 44.4 From the frontier-training blog a colleague shared

- **Document masking:** a packed row with plain causal attention lets one chat see another; intra-document masking stops
  it. Hugging Face found it mattered little on short tasks and became crucial from 4k to 64k context. Our v2 does it,
  and positions restart per chat.
- **General reasoning SFT vs agentic SFT** (Intellect-3's two SFT stages):
  - General reasoning: math, code, science, chat and instruction data with step-by-step answers written by a stronger
    model; 9.9B tokens.
  - Agentic: multi-step tasks with tool calls and tool outputs; the context was raised from 65K to 98K tokens.
  - Our stage-0 (UltraChat) is plain chat SFT, the step before both.
- **Upstream vs downstream evals** (the blog uses only "downstream"):
  - Upstream: the training task on held-out data, i.e. validation loss.
  - Downstream: task benchmarks. Either scored by likelihood (HellaSwag, ARC, MMLU; we can run these) or by generation
    (IFEval, GSM8K, MT-Bench; these need text generation, which BluTrain does not have yet).

### 44.5 Backward B-2a: FA3's wait placement between the dQ and dK GEMMs (records `gpu_runs/b2a_1014/RESULT.md`)

- **Change:** one wait before the dK GEMM (wait until dV is done; dQ keeps running), as FA3 does.
- **Test 1:** 144/144, every error value identical.
- **Test 2 (6 alternating rounds):** backward 1.9% less time than G1+G2+G3 (t = -8.7, faster on 13/13); forward +
  backward 1.4% less (13/13).
- **Test 3 (ncu on our model's shape):** main kernel 2526.3 -> **2478.0 us**, FA3 2481.4 = equal.
  - Warps waiting for each other at the dK GEMM start: 215.5 -> 148.2 us-equivalent (FA3 141.1). This was the predicted
    cause, and it is confirmed.
  - The new wait costs 50; exp and dS lines now wait longer for shared-memory values (+64).
- **Remaining gaps to FA3:** shared-memory-queue stalls +93 (B-3), dispatch +51, warps waiting at the dP / dV / S
  starts +44 (B-2b), helper kernels +8 (B-1).

### 44.6 Future task, FIRST THING on 2026-10-15 morning: special tokens or plain-text role markers, and chat templates

The user asked to start 2026-10-15 with this, in parallel with the attention-kernel work. Nothing below is decided yet.
Everything marked **(checked 10-14)** was run or read today; everything marked **(to check)** must be confirmed from a
source (paper, model card, tokenizer file or source code at an exact commit) before it goes into a decision.

#### 44.6.1 The questions to answer

1. **Add or not:** should the role markers (`<|system|>`, `<|user|>`, `<|assistant|>`) and the end-of-turn marker be
   added to the tokenizer as new special tokens (one id each), or should we keep writing them as plain text that the
   GPT-2 BPE splits into pieces (`<|assistant|>` -> `<`, `|`, `ass`, `istant`, `|`, `>`)?
2. **If we do not add them:** how does the model learn where a chat, a turn and an answer start and end? (What is
   learned from the plain-text pieces, what is learned from the end token, and what the inference code must do.)
3. **If we do add them:** when (at pretraining, at SFT, or in a separate short stage between them), and what is the
   full process (tokenizer, embedding rows, output rows, their starting values, which steps train them, every tool
   that must know the new ids)?
4. **How others do it, and how it is going for them:** which models use plain-text markers, which use real special
   tokens, which ones reserved the ids at pretraining, and what problems people reported (for example new rows that
   were never trained).
5. **Chat templates:** what a chat template is, where it lives, how it is applied for training and for generation,
   how it decides which tokens get loss, and what ours should be.

#### 44.6.2 What we do today (checked 10-14)

- **Tokenizer:** GPT-2 BPE through `tiktoken.get_encoding("gpt2")`. `n_vocab` = 50,257. The only special token is
  `<|endoftext|>` = id **50256** (`enc.special_tokens_set` = {'<|endoftext|>'}).
- **Model vocab rows:** 50,304 (`VOCAB_ROWS = 50304`, `sft/prepare_sft_data.py:32`; 50,257 rounded up). So ids
  **50,257 to 50,303 (47 ids) exist as rows in the model but are not used by the tokenizer**.
- **Template "tulu-plain-v1"** (`sft/prepare_sft_data.py:10-14, 33`):
  `<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...`
  - Each piece is encoded on its own with `enc.encode_ordinary(...)` (`:52-63`): the marker, then the content, then
    either the end id (after an answer) or `"\n"` (after a user or system turn). So BPE never merges a marker with the
    words next to it.
  - After each answer the real id 50256 is appended (`ids.append(PAD_ID)`, `:60`) and it is a **target** (the model
    must learn to stop). Marker tokens are not targets (`mask += [False] * len(marker)`, `:54`).
  - Text that looks like `<|endoftext|>` inside a message is encoded as plain text (7 pieces), never as id 50256
    (asserted at `:114-115`). So a user message cannot fake an end token.
  - Rows are padded with 50256 too (`PAD_ID = 50256`, `:30`); pads are not targets.
- **How the markers split today** (tiktoken gpt2, `encode_ordinary`, run 10-14):

| text | tokens | ids | pieces |
| --- | --- | --- | --- |
| `<\|system\|>\n` | 6 | 27, 91, 10057, 91, 29, 198 | `<` `\|` `system` `\|` `>` `\n` |
| `<\|user\|>\n` | 6 | 27, 91, 7220, 91, 29, 198 | `<` `\|` `user` `\|` `>` `\n` |
| `<\|assistant\|>\n` | 7 | 27, 91, 562, 10167, 91, 29, 198 | `<` `\|` `ass` `istant` `\|` `>` `\n` |
| `assistant` (no space) | 2 | 562, 10167 | `ass` `istant` |
| ` assistant` (with a space) | 1 | 8796 | ` assistant` |
| `<\|endoftext\|>` as plain text | 7 | 27, 91, 437, 1659, 5239, 91, 29 | `<` `\|` `end` `of` `text` `\|` `>` |
| `<\|endoftext\|>` as the special id | 1 | 50256 | (one token) |
| `<\|im_start\|>assistant\n` (ChatML, for comparison) | 10 | 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198 | `<` `\|` `im` `_` `start` `\|` `>` `ass` `istant` `\n` |

  The markers cost about 20 tokens per chat, against an average chat of 1,238 tokens on stage-0.
- **Tülu 3 does the same as us (checked 10-14** in `allenai/Llama-3.1-Tulu-3-8B`, `tokenizer_config.json`):
  - `<|user|>`, `<|assistant|>`, `<|system|>` are **not** registered tokens; they appear only as plain text inside the
    chat template, which starts `{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' +
    message['content'] + '\n' }}` and adds `<|assistant|>` as the generation prompt.
  - `eos_token` = `<|end_of_text|>` (128001), `bos_token` = `<|begin_of_text|>`, `pad_token` = `<pad>` (128256).
  - The only registered tokens are Llama 3's own (`<|begin_of_text|>`, `<|end_of_text|>`, `<|finetune_right_pad_id|>`,
    `<|start_header_id|>`, `<|end_header_id|>`, `<|eom_id|>`, `<|eot_id|>`, `<|python_tag|>`, the
    `<|reserved_special_token_N|>` ids) plus `<pad>` (128256), which Tülu added.
  - So our "tulu-plain-v1" follows Tülu 3: plain-text role markers + the tokenizer's real end token after each answer.

#### 44.6.3 What changes if we add special tokens (to check, step by step, tomorrow)

1. **Tokenizer:** register the new strings with fixed ids. With tiktoken this means building a new `Encoding` from the
   GPT-2 ranks plus extra `special_tokens` (for example ids 50257, 50258, ...), and encoding with those strings
   allowed. The tokenizer then finds the exact string `<|assistant|>` first and gives it one id; BPE only runs on the
   text between such strings. The word "assistant" in normal text still splits as before. (to check: the exact
   tiktoken API, and what Hugging Face's `add_special_tokens` + `resize_token_embeddings` do.)
2. **Embedding rows (input side):** the new ids need rows. We have 47 unused rows (50,257-50,303), so no resize is
   needed. But these rows were **never an input in pretraining**, so they hold their starting values (plus whatever
   weight decay did). Options to check: leave them, set them to the mean of all trained rows, or set them to the mean
   of the pieces they replace (for `<|assistant|>`: the rows of `<`, `|`, `ass`, `istant`, `|`, `>`).
3. **Output rows (prediction side):** if the new ids are targets (for example a new end-of-turn token), the output
   layer's rows for them matter too. (to check in our code: is the output layer tied to the embedding table? Does our
   softmax / loss cover all 50,304 rows or only 50,257? If it covers all 50,304, pretraining pushed the scores of the
   47 unused ids down on every step, so a new target id starts with a very low score.)
4. **Which steps train them:** in full fine-tuning every row trains, so SFT learns them. (With LoRA the embedding and
   output layers are usually frozen and must be listed as trainable; we use full fine-tuning, §44.3.)
5. **Every tool that touches ids must know the new ids:** `prepare_sft_data.py`, the label mask, the converter to
   Hugging Face format, the eval scripts, and any generation / chat code later.
6. **Risk to measure:** new rows start untrained, so the first SFT steps may be noisier; with plain text, every piece
   already has a trained row.

#### 44.6.4 How others do it (to check tomorrow, with exact sources)

| model / tool | role markers | end of turn | ids reserved at pretraining? | status |
| --- | --- | --- | --- | --- |
| Tülu 3 (on Llama 3.1) | plain text `<\|user\|>` ... | the real `<\|end_of_text\|>` | n/a (plain text) | **checked 10-14** (44.6.2) |
| ours ("tulu-plain-v1") | plain text | the real 50256 | n/a | **checked 10-14** |
| Llama 3 instruct | `<\|start_header_id\|>user<\|end_header_id\|>` (special ids) | `<\|eot_id\|>` | reserved in the tokenizer (`<\|reserved_special_token_N\|>`) | to check: were these rows trained in the base model, or only in instruct tuning? (reports exist that some were untrained in the base model) |
| Qwen (ChatML) | `<\|im_start\|>user` ... (special ids) | `<\|im_end\|>` | to check | to check |
| Zephyr / HF alignment-handbook | plain text `<\|user\|>` ... (the format Tülu copied) | `</s>` | n/a | to check |
| TRL | `setup_chat_format` / `clone_chat_template` (adds ChatML-style tokens and resizes the embeddings) | | | to check in TRL at our pinned commit `478624d` |
| SmolLM / SmolLM3, OLMo 2, Mistral, Gemma, DeepSeek | | | | to check |

Also to check: what problems were reported when tokens were added at SFT time (untrained rows, wrong eos in generation
configs, a template's eos differing from the model's eos), and how people solved them.

#### 44.6.5 Chat templates (to cover tomorrow)

- What a chat template is: the rule that turns a list of messages (`system`, `user`, `assistant`, later `tool`) into
  one text / one token list. In Hugging Face it is a Jinja string stored in `tokenizer_config.json` (`chat_template`),
  applied with `tokenizer.apply_chat_template(messages, add_generation_prompt=...)`. (to check exact behaviour.)
- The **generation prompt:** at inference the template ends with the assistant marker (`<|assistant|>\n` for Tülu)
  so the model starts writing an answer; it stops when it writes the end token.
- **Which tokens get loss:** only the assistant answers and their end token (ours: §44.6.2; TRL: "assistant only loss"
  with `{% generation %}` blocks in the template, to check).
- Multi-turn, system prompts, tool calls (agentic SFT, §44.4), and whether the training template and the inference
  template are exactly the same (a mismatch is a known source of bad generations, to check).
- Ours: "tulu-plain-v1" is written in Python inside `prepare_sft_data.py`; there is no Jinja template file yet. To
  decide: do we also store it as a Hugging Face `chat_template` when we convert the model, so that eval / chat tools use
  exactly the same format?

#### 44.6.6 Plan for 2026-10-15 morning

1. Read the sources for 44.6.4 and 44.6.5 (model cards, `tokenizer_config.json` files, papers, TRL / transformers
   source at exact commits) and fill the table with checked facts and links.
2. Check our own code for 44.6.3 step 3 (tied output layer? loss over 50,304 or 50,257 rows?).
3. Write the answer to each question in 44.6.1 in simple English, with a recommendation: keep plain text (today) or add
   special tokens, and if adding, when and how.
4. If the answer is "add", plan an ablation (plain text vs special tokens, same data, same steps) before changing the
   real SFT data.

---

## 45. 2026-10-15

### 45.1 The chat template: what Tülu 3 really uses, what we use, and the decision (answers the chat-template part of §44.6)

The special-tokens part of §44.6 (questions 1-4 of 44.6.1) is still open; this section answers question 5 (chat
templates) and records the decision taken today.

#### 45.1.1 What a chat template is (from zero)

- An SFT example is a list of messages. Each message has a role (`system`, `user` or `assistant`) and its text.
- The model does not read a list. It reads one row of tokens.
- The **chat template** is the fixed rule that turns the list into one text (and then tokens): which marker goes
  before each turn, where the newlines go, and where the end token goes.
- The model learns this exact layout during SFT. Later, every place that gives the model a chat (evaluation, DPO, RL,
  a chat program) must build the text with the same rule, or the model sees a layout it was never trained on.

#### 45.1.2 What the Tülu 3 paper measured (Section 4.3.1, Table 13)

Source: [Tülu 3 paper, arXiv 2411.15124](https://arxiv.org/abs/2411.15124), Section 4.3.1 "Chat Template Variation",
Table 13. They trained the same early SFT setup (Llama 3.0 base, an intermediate SFT mix) with 5 templates:

| chat template (row name in Table 13) | average score |
| --- | --- |
| Tülu (replace `\n` w/ eos) | 53.0 |
| Zephyr | 52.9 |
| Tülu 3 (no `\n`) - the one they used | 52.8 |
| Tülu 2 template | 52.6 |
| Llama 3 template | 51.6 |

Their words (quoted exactly):

- Text: *"We made a small change to the chat template used in previous Tülu versions, specifically removing the new
  line at the end of the template (before the model response)."*
- Text: *"We found that replacing the newlines at the end of assistant messages with an eos token resulted in the best
  performance, but we opted not to use this to avoid generation inconsistency with later steps in our post-training
  pipeline."*
- Table 13 caption: *"While replacing the newline does best, we instead opted for simply removing the newline to
  avoid complexity."*

What the table shows:

1. **Which template you pick matters only a little.** All 5 are within 1.4 points (53.0 - 51.6). This is 1.4
   points of an average score, not 1.4%.
2. **They took the 3rd-best one (52.8) on purpose, so that it matched their later stages** (DPO and RL). They gave up
   0.2 points for that.
3. The paper does not say "use the same template and no problem will come". It says consistency with the later stages
   mattered more to them than the best score.

#### 45.1.3 Figure 27 (Appendix B.3) is cut off in the PDF: the full template, line by line

In the PDF, every line of Figure 27 that shows only `"{%` is a Jinja control line (`for` / `if` / `else` / `endif`)
whose text was cut off. The full template is in Tülu 3's `tokenizer_config.json` (field `chat_template`), downloaded
and printed on 2026-10-15 from
[allenai/Llama-3.1-Tulu-3-8B at 6669437](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B/blob/666943798adbde0b1aff34626007e26986a3c107/tokenizer_config.json).
It has exactly the 16 lines of Figure 27:

```
 1  {% for message in messages %}
 2  {% if message['role'] == 'system' %}
 3  {{ '<|system|>\n' + message['content'] + '\n' }}
 4  {% elif message['role'] == 'user' %}
 5  {{ '<|user|>\n' + message['content'] + '\n' }}
 6  {% elif message['role'] == 'assistant' %}
 7  {% if not loop.last %}                                                  <- this answer is NOT the last message
 8  {{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}         <- end token, then \n
 9  {% else %}                                                              <- this answer IS the last message
10  {{ '<|assistant|>\n' + message['content'] + eos_token }}                <- end token, NO \n
11  {% endif %}
12  {% endif %}
13  {% if loop.last and add_generation_prompt %}
14  {{ '<|assistant|>\n' }}
15  {% endif %}
16  {% endfor %}
```

`allenai/Llama-3.1-Tulu-3-8B-SFT` (the SFT-stage model) has exactly the same `chat_template` string (compared
2026-10-15).

#### 45.1.4 What "no `\n`" means: only ONE newline was removed

"No `\n`" does **not** mean "no newline anywhere". All these newlines stay in Tülu 3:

- after each marker: `<|system|>\n`, `<|user|>\n`, `<|assistant|>\n`;
- after system and user text (lines 3 and 5);
- after the end token of an earlier answer (line 8).

**Only the newline after the end token of the LAST answer was removed** (line 10 compared with line 8). Tülu 2 had
only line 8's form, so every answer, the last one too, ended with end token + `\n`. Tülu 3 added lines 7, 9 and 10.

The code change is
[open-instruct PR 376, "Remove newline at the end of tulu template"](https://github.com/allenai/open-instruct/pull/376)
(merged 2024-10-10). Its diff, exactly:

```
         "{% elif message['role'] == 'assistant' %}"
+        "{% if not loop.last %}"
         "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
+        "{% else %}"
+        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
+        "{% endif %}"
         "{% endif %}"
```

The paper's own phrase "(before the model response)" does not match this code clearly; the code above is what they
actually used.

On a real chat (`<eos>` = the end token):

| | one turn |
| --- | --- |
| Tülu 2 | `<\|user\|>\nWhat is the capital of France?\n<\|assistant\|>\nParis.<eos>\n` |
| Tülu 3 | `<\|user\|>\nWhat is the capital of France?\n<\|assistant\|>\nParis.<eos>` |

| | two turns |
| --- | --- |
| Tülu 2 | `...<\|assistant\|>\nA1<eos>\n<\|user\|>\nU2\n<\|assistant\|>\nA2<eos>\n` |
| Tülu 3 | `...<\|assistant\|>\nA1<eos>\n<\|user\|>\nU2\n<\|assistant\|>\nA2<eos>` |

The only difference is the final `\n`.

#### 45.1.5 The `\n` between turns in Tülu 3: in the text, read by attention, never trained

In a multi-turn chat, Tülu 3 keeps the `\n` after an earlier answer's end token. That `\n` is an **input** token
(later tokens attend to it), but it is **never a target** (the model is never asked to write it).

Each position has an input token and a target (the next token). "Trained" = the loss is computed on that target.
The boundary between answer 1 and user turn 2:

**Tülu 3:**

| input token | target | trained? |
| --- | --- | --- |
| `Paris.` (last word of answer 1) | `<eos>` | **yes**: the model learns to stop |
| `<eos>` | `\n` | **no** (masked) |
| `\n` | `<` (start of `<\|user\|>`) | no |
| ... user turn 2 ... | ... | no |
| `<\|assistant\|>\n` | first word of answer 2 | yes |

**Ours:**

| input token | target | trained? |
| --- | --- | --- |
| `Paris.` | `<eos>` | **yes** |
| `<eos>` | `<` (start of `<\|user\|>`) | no |
| ... user turn 2 ... | ... | no |
| `<\|assistant\|>\n` | first word of answer 2 | yes |

So in Tülu 3 the model never learns to write `\n` after `<eos>`; it only learns to read past it (answer 2's tokens
attend to it). In both formats the only thing learned at the boundary is "after the answer, write `<eos>`".

**Why that `\n` is not trained in Tülu 3.** PR 376's description: *"we can stick with the original tulu chat template
that doesn't have a newline in the end. Meanwhile, this also masks the loss of newlines for intermediate assistant
messages."* In simple words:

- First sentence: the chat text now ends right after the last answer's end token, with nothing after it.
- Second sentence: a side effect - the `\n` after an earlier answer's end token now gets no loss either. Before this
  change it was trained.
- How the side effect happens (their `mask_labels`,
  [open-instruct `dataset_transformation.py:1307-1368` at 95a84bb](https://github.com/allenai/open-instruct/blob/95a84bb5781c69e58ba5a4f9b265f4b504ce4f2b/open_instruct/dataset_transformation.py#L1307-L1368)):
  to find where user turn 2 starts, the code renders the chat only up to answer 1. In that shorter text answer 1 is
  the last message, so it is rendered with line 10 (`<eos>`, no `\n`). So user turn 2's masked span starts right after
  `<eos>`, and in the full chat the `\n` that sits there falls inside user turn 2's span and gets label -100 (no loss).
  Before PR 376, line 8's form (`<eos>\n`) was used even in the shorter text, so the `\n` belonged to the answer and
  `<eos> -> \n` was a trained target.

| | `\n` after an earlier answer's `<eos>` | is `<eos> -> \n` trained? |
| --- | --- | --- |
| Tülu 2 | yes, in the text | yes |
| Tülu 3 | yes, in the text | **no** (masked, as above) |
| ours | **not in the text** | n/a |

#### 45.1.6 Our template, and which of the 5 it is

Ours ("tulu-plain-v1", `sft/prepare_sft_data.py`) on a 2-turn chat:

```
<|system|>\nSYS\n<|user|>\nU1\n<|assistant|>\nA1<eos><|user|>\nU2\n<|assistant|>\nA2<eos>
```

All the `\n` of Tülu 2's text, numbered:

```
<|system|>\n  SYS \n  <|user|>\n  U1 \n  <|assistant|>\n  A1<eos>\n  <|user|>\n  U2 \n  <|assistant|>\n  A2<eos>\n
          (1)     (2)        (3)    (4)             (5)          (6)        (7)    (8)             (9)         (10)
```

| `\n` | where it is | Tülu 2 | Tülu 3 | 53.0 variant | ours |
| --- | --- | --- | --- | --- | --- |
| 1, 3, 5, 7, 9 | after a role marker | yes | yes | yes | yes |
| 2, 4, 8 | after system or user text | yes | yes | yes | yes |
| **6** | after an earlier answer's end token | yes | yes | **no** | **no** |
| **10** | after the last answer's end token | yes | **no** | **no** | **no** |
| inside the text | a user's or answer's own line breaks | yes | yes | yes | yes |

- Tülu 3 removed only (10). The 53.0 variant ("the newlines at the end of assistant messages" replaced) removed (6)
  and (10). Ours has neither (6) nor (10). **So on a literal reading, ours is the 53.0 format.**
- It is NOT "replace every `\n` with eos": the newlines after markers and inside the text are part of the prompt and
  the answers; an end token there would mean "stop" in the middle of a turn.
- **Why "most likely" and not certain:** their exact 53.0 template was never published - not in the paper (one
  sentence only), not in open-instruct's code, and not in open-instruct's commit history (searched 2026-10-15: the only
  change to this newline is PR 376, which is Tülu 3's final template). Read word by word against Tülu 2's text,
  "replace the `\n` with eos" would turn `A1<eos>\n` into `A1<eos><eos>` (two end tokens), which makes no sense; the
  sensible meaning is "the answer ends with the end token and nothing after it", which is ours.

All 5 rows of Table 13, and the same 2-turn chat in each format we know exactly:

| score | row | what it is | exact text known? |
| --- | --- | --- | --- |
| 53.0 | Tülu (replace `\n` w/ eos) | a test variant: no `\n` after any answer's end token (our reading) | no, one sentence only |
| 52.9 | Zephyr | every turn (system, user and assistant) ends with end token + `\n` | yes (open-instruct `"zephyr"`) |
| 52.8 | Tülu 3 (no `\n`) | Tülu 2 minus the `\n` after the last answer = Tülu 3's final | yes |
| 52.6 | Tülu 2 | end token + `\n` after every answer, the last one too | yes |
| 51.6 | Llama 3 template | Llama 3's own header-token format | not checked |

```
Tülu 2  (52.6)   ...<|assistant|>\nA1<eos>\n<|user|>\nU2\n<|assistant|>\nA2<eos>\n
Tülu 3  (52.8)   ...<|assistant|>\nA1<eos>\n<|user|>\nU2\n<|assistant|>\nA2<eos>
Zephyr  (52.9)   ...<|user|>\nU1<eos>\n<|assistant|>\nA1<eos>\n<|user|>\nU2<eos>\n<|assistant|>\nA2<eos>\n
ours             ...<|assistant|>\nA1<eos><|user|>\nU2\n<|assistant|>\nA2<eos>
53.0 (literal reading of their sentence, not confirmed)
                 ...<|assistant|>\nA1<eos><|user|>\nU2\n<|assistant|>\nA2<eos>      <- same as ours
```

(Tülu 2 and Tülu 3 rendered from their Hugging Face `tokenizer_config.json` files with Jinja `trim_blocks` and
`lstrip_blocks`, as `transformers` does; Zephyr from open-instruct's `CHAT_TEMPLATES["zephyr"]` at 95a84bb.)

#### 45.1.7 Checked in our code and in our real stage-0 data (2026-10-15)

**Code** (`sft/prepare_sft_data.py`, `tokenize_conversation`):

- line 52: each turn starts with its marker, `<|{role}|>\n`;
- lines 59-61: after an **answer**, only the end token is added (`ids.append(PAD_ID)`, 50256, a target) - nothing
  after it;
- lines 62-65: a `\n` is added only after a **user or system** turn (not a target);
- the docstring (lines 10-14) states the same template.

**Data** (`~/Downloads/sft_data/stage0_ultrachat_5k_T4096`, every row scanned):

| | train | val |
| --- | --- | --- |
| chats (rows) | 5,000 | 500 |
| multi-turn chats | 4,998 | 500 |
| answers (end tokens that are targets) | 15,845 | 1,555 |
| after an earlier answer's end token, the next tokens are `<` `\|` `user` (ids 27, 91, 7220) | 10,845 | 1,055 |
| after the last answer's end token, the next token is padding (50256) | 5,000 | 500 |
| "end token then `\n`" (ids 50256, 198) anywhere in the inputs | **0** | **0** |

So v1 (padding) SFT was trained on the 53.0 format, and since almost every chat is multi-turn, the missing `\n` shows
up 10,845 times in training.

#### 45.1.8 Why Tülu 3 did not use the 53.0 one, and whether that reason applies to us

- Their reason (45.1.2): *"to avoid generation inconsistency with later steps in our post-training pipeline"* and
  *"to avoid complexity"*. Their pipeline is SFT -> DPO -> RL, then evaluation. The later steps also build chats with a
  template (in RL and evaluation the model writes answers from template-built prompts), and those steps were built
  around the Tülu template. Switching only SFT to the 53.0 format would have made SFT use one format and the later
  steps another. The gain was only 0.2 points (53.0 vs 52.8) in one early run, so they kept the smaller change.
- Their code shows the same idea: [open-instruct PR 350](https://github.com/allenai/open-instruct/pull/350) (merged
  2024-09-30) made both their SFT script (`finetune.py`) and their DPO script (`dpo_tune.py`) read the template from
  one shared table, `CHAT_TEMPLATES`, with the default `"tulu"`.
- The paper does not say which exact step or tool would have been inconsistent; those two sentences are all it gives.
- **For us the reason does not apply now:** we have only SFT so far - no DPO, no RL, no generation-based evaluation.
  We can build all of them on our format from the start, so there is nothing to be inconsistent with. What carries
  over is the rule behind their choice: every later stage must use exactly our SFT format.
- Do not expect a gain from the 53.0 choice itself: 0.2 points over 52.8 in one early run is too small to count on.

#### 45.1.9 DECISION (2026-10-15): keep our template, and use it in every stage

The user decided: keep ours (Tülu 3's template minus the `\n` after an earlier answer's end token = the 53.0 format
on a literal reading). No code change, no data change.

"The same template in every stage" (SFT, DPO, RL, evaluation, inference) means these five things are the same
everywhere:

1. **The same markers:** `<|system|>\n`, `<|user|>\n`, `<|assistant|>\n`.
2. **The same newlines:** `\n` after system and user text; **no** `\n` after an answer's end token.
3. **The same end token:** 50256 after every answer, and generation stops at 50256.
4. **The same generation prompt:** the prompt ends with `<|assistant|>\n`, and the model writes from there.
5. **The same tokenizing:** each piece encoded on its own, as in `prepare_sft_data.py` (45.1.10 shows why).

Training is not affected by the missing `\n` (Tülu 3 never trains it either, 45.1.5). It matters only for
consistency: our multi-turn evaluation and chat code must build `A1<eos><|user|>` with no `\n`. So Tülu 3's
`chat_template` file must never be reused as it is (it adds that `\n`).

#### 45.1.10 The same text can give different tokens (tested 2026-10-15, tiktoken `gpt2`)

The same text can become different tokens depending on whether it is encoded in one go or piece by piece:

| text | piece by piece (ours) | in one go | same? |
| --- | --- | --- | --- |
| a normal chat: `<\|user\|>\nWhat is the capital of France?\n<\|assistant\|>\n` | 21 tokens | the same 21 tokens | yes |
| `"\n" + "Paris"` | [198, 40313] | [198, 40313] | yes |
| `"\n" + "<\|user\|>\n"` | [198, 27, 91, 7220, 91, 29, 198] | the same | yes |
| a message ending with a newline, `"Write a poem.\n"`, then the template's `"\n"` | `.` `\n` `\n` = [13, 198, 198] | `.` `\n\n` = [13, 628] | **no** |

GPT-2 has one token for `\n\n` (628). So a tool that encodes the whole chat in one go gives a different token here
than our script. That is why item 5 of 45.1.9 says "the same tokenizing", not only "the same text".

Also expected but not tested yet: a user message that contains the text `<|endoftext|>`. Our script encodes it as 7
plain tokens; a Hugging Face GPT-2 tokenizer would probably turn it into the real end token 50256.

#### 45.1.11 Our own code vs outside tools: when a chat_template file and a token check are needed

| later stage built with | what we need |
| --- | --- |
| our own code (BluTrain + our Python) | call the same function as `prepare_sft_data.py`; nothing else |
| an outside tool (Hugging Face `transformers` / its evaluation tools, vLLM, TRL) | our own `chat_template` file + a token-by-token check |

- Our template exists today only as Python code inside `prepare_sft_data.py`.
- Outside tools cannot call that function. They read the template from `tokenizer_config.json` (field
  `chat_template`, written in Jinja) and call `tokenizer.apply_chat_template(messages)`.
- Only if we use such a tool, we write our own `chat_template` for our format (the same as Tülu 3's, except the answer
  line has no `\n` and no if/else). Not tested yet:

```
"{% for message in messages %}"
"{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}"
"{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
"{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}"
"{% endif %}"
"{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
"{% endfor %}"
```

- Then the check: take all 5,500 stage-0 chats; make tokens our way and the tool's way; compare token by token; every
  chat must match before any later stage uses the file. The two cases of 45.1.10 (a message ending with `\n`, and the
  text `<|endoftext|>` inside a message) are expected to differ and must be fixed on one side first.
- Whether we will use outside tools is not decided; it will be decided when evaluation and DPO are built.

#### 45.1.12 Tülu 3's tokenizer facts (checked 2026-10-15 in `allenai/Llama-3.1-Tulu-3-8B-SFT`)

- `eos_token` = `<|end_of_text|>` (id 128001); `pad_token` = `<pad>` (id 128256, added by Tülu).
- `added_tokens_decoder` holds only Llama 3's own tokens - 128000 `<|begin_of_text|>`, 128001 `<|end_of_text|>`,
  128004 `<|finetune_right_pad_id|>`, 128006 `<|start_header_id|>`, 128007 `<|end_header_id|>`, 128008 `<|eom_id|>`,
  128009 `<|eot_id|>`, 128010 `<|python_tag|>`, and the `<|reserved_special_token_N|>` ids - plus 128256 `<pad>`.
- `<|user|>`, `<|assistant|>`, `<|system|>` are **not** registered tokens. Tülu 3 writes them as plain text, the same
  as our option A.
- With Tülu 3's own tokenizer (Llama 3 BPE, rebuilt from its `tokenizer.json` with tiktoken) each marker is **5**
  tokens: `<` `|` `user` `|` `>\n` (and the same with `assistant` and `system`). With our GPT-2 tokenizer they are 6, 7
  and 6 tokens.
- Tülu 3's end token (`<|end_of_text|>`, 128001) is Llama 3's; ours is GPT-2's `<|endoftext|>` (50256). Same job,
  different token.

#### 45.1.13 Sources for 45.1

- Tülu 3 paper: [arXiv 2411.15124](https://arxiv.org/abs/2411.15124) - Section 4.3.1 and Table 13; Appendix B.3,
  Figure 27.
- Tülu 3 chat template as released:
  [allenai/Llama-3.1-Tulu-3-8B `tokenizer_config.json` at 6669437](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B/blob/666943798adbde0b1aff34626007e26986a3c107/tokenizer_config.json)
  (field `chat_template`); the same string in
  [allenai/Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT/blob/main/tokenizer_config.json).
- Tülu 2 chat template: [allenai/tulu-2-7b `tokenizer_config.json`](https://huggingface.co/allenai/tulu-2-7b/blob/main/tokenizer_config.json).
- [open-instruct PR 376](https://github.com/allenai/open-instruct/pull/376) (remove the last newline; also masks the
  between-turn newline) and [PR 350](https://github.com/allenai/open-instruct/pull/350) (one template table for SFT and
  DPO; the old Tülu 2 Python template it replaced).
- open-instruct at commit 95a84bb:
  [`CHAT_TEMPLATES` ("zephyr", "tulu")](https://github.com/allenai/open-instruct/blob/95a84bb5781c69e58ba5a4f9b265f4b504ce4f2b/open_instruct/dataset_transformation.py#L160-L277)
  and [`mask_labels`](https://github.com/allenai/open-instruct/blob/95a84bb5781c69e58ba5a4f9b265f4b504ce4f2b/open_instruct/dataset_transformation.py#L1307-L1368).
- Our code: `sft/prepare_sft_data.py` (lines 10-14, 52, 59-65); our data: `~/Downloads/sft_data/stage0_ultrachat_5k_T4096`.

### 45.2 The SFT doc as markdown: `work_docs/post_training_research_sft.md` (2026-10-15)

The Google Doc "PostTraining - Research and Planning" (tab "SFT : what it is") now has a clean markdown copy in
work_docs: **`post_training_research_sft.md`**, with its 5 images in `post_training_research_sft_images/`. It was
built from the live doc (Google's markdown and plain-text exports, downloaded 2026-10-15 ~13:00) and has all the
fixes below. The user will update the Google Doc from this file by hand later.

**Why by hand, not with Gemini.** Over 4 rounds today, Gemini in Google Docs:

- rewrote paragraphs in its own words instead of pasting the given text;
- added its own wrong facts: Llama 3 token ids "128016 to 128019" (real: `<|eot_id|>` 128009, `<|start_header_id|>`
  128006, `<|end_header_id|>` 128007), "6, 7 and 6 Llama tokens" (real: 5 each), a wrong `\n<|user|>\n` example, an
  unseen-suite row "scored by log-likelihood and classifiers", and a wrong Tülu 2 paper title;
- rewrote the doc's title, intro, summary and 3 "why it's needed" bullets without being asked (removing the exact
  quotes from the papers);
- in the last round it pasted the text exactly, but merged the line breaks and added double bullets.

**How it was built and checked.**

1. Export cleanup (formatting only): Google's backslash escapes removed; special tokens put in code spans; `|` inside
   table cells escaped (Google's export left `<|user|>` cells unescaped, so those rows split into extra columns);
   formulas, the pipeline line, the 3.1 flow, the 3.6 pseudo-code, the 3.8 input/target rows, the JSON/ChatML
   examples and the 4.1 examples put in code blocks; headings cleaned; the 5 embedded images saved as PNG files.
2. **Word-by-word check against the live doc's plain-text export:** 17,555 words in the doc, and every difference is
   one of the edits listed below (no word lost). The restored top matches the pre-Gemini text word for word
   (289/289).
3. 29 tables, every row with the same number of cells; code fences balanced; no `<tag>` that a markdown viewer would
   hide.
4. **Every link checked:** 52 arXiv ids exist and 51 titles matched the doc (the Tülu 2 one is fixed below); the 8
   other links (GitHub, Hugging Face) return HTTP 200.

**The content changes, compared with the live Google Doc (to apply to the Google Doc later):**

| # | where | change |
| --- | --- | --- |
| 1 | title, intro, Summary, 3 bullets | restored to the user's own text from before Gemini's rewrite |
| 2 | Summary, 4th bullet | "open-soure" -> "open-source" |
| 3 | 1.4 example paragraph | says the table shows each marker as one entry and leaves the newlines out; the real row is 24 GPT-2 tokens (`<\|user\|>\n` = 6, question = 7, `\n`, `<\|assistant\|>\n` = 7, `Paris` `.`, 50256) |
| 4 | 1.5 "less data vs more data" | closed the bracket after "filtered beat unfiltered" |
| 5 | 1.6 | made a real heading (it was bold text, so the markdown export lost it as a heading) |
| 6 | 1.7 | "8.5B tokens" -> "8.4B tokens" (1.2, 6.4 and the checkpoint all say 8.4B: "step 8000 (8.4B tokens)", recorded when the checkpoint was inspected) |
| 7 | 2.3 | "(DPO paper)Tülu 3" -> "(DPO paper) , Tülu 3" |
| 8 | 2.4 | "task->for ex" -> "task -> for ex" |
| 9 | 3.2 first sentence | restored to the user's own wording (Gemini had reworded it) |
| 10 | 3.2 Tülu 3 template | split into lines; both template strings on their own lines as code (Tülu 3's two rules, Tülu 2, ours) |
| 11 | 3.2 Table 13 | split into its 3 paragraphs (Table 13 / "Which one I use" / "The same template in every stage") |
| 12 | 3.2 Tülu 3 tokens paragraph | the token example written as code: `<` `\|` `user` `\|` `>\n` |
| 13 | 3.3 intro | corrected: with option A the role markers are plain text split into ordinary tokens, and only the end token is one special id, which the script adds as an id (the old text said the tokenizer "recognizes them and gives each one its id") |
| 14 | 3.3 "One thing to watch" | the examples written as code spans |
| 15 | 3.4 LIMA sentence | "itt" -> "it" |
| 16 | 3.4 plan | restored to the user's own wording, with the Tülu 3 fact corrected (Tülu 3 does NOT have dedicated role tokens) |
| 17 | 3.8 | "REGARIDNG" -> "REGARDING" |
| 18 | 4.3 | "expeeriemtnations" -> "experimentations", and the bracket closed |
| 19 | 6.2 | heading moved above its table (as in the 09-28 report); the empty heading removed; "MATH  BBH" -> "MATH , BBH" |
| 20 | 6.2 table, unseen-suite row | corrected from Tülu 3 Tables 3 and 24: MMLU-Pro, GPQA, AGIEval English, Deepmind Mathematics, BigCodeBench, IFEval-OOD, HREF; scored by exact match, unit tests (pass@10), instruction checks (pass@1), win rate (HREF); all generate |
| 21 | 6.4 | "willl", "testingwith", "8.4 B", "pretrainign", "experimentaitons", "seee e", "theseablation", "tooo" fixed |
| 22 | 7 | "iIt's" -> "it's"; "excperimetnaitons ... onthis" -> "experimentations ... on this" |
| 23 | 7, "longer than 4096 tokens?" | corrected: packing with varlen attention does not let an example longer than the row in; only a longer context length can (packing doc §7.6); the old text said "variable-attention ... can interpret long sequences" |
| 24 | 7 source line | stray ":" removed |
| 25 | 8.1 template bullet | ours = Tülu 3's template minus the between-turn `\n` (the 53.0 one) + the stage-0 check (4,998 of 5,000 multi-turn, 15,845 answers) |
| 26 | 8.2 | a stray copy of the table's last row ("Muon , 3 learning rates , 4 GPUs ... see 8.3") removed |
| 27 | Sources | the 3 chat-template links: double bullets removed, links clickable |
| 28 | Sources | Tülu 2 title corrected: "Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2" (arXiv 2311.10702) |
| 29 | Sources | "Adamw" -> "AdamW"; the 7 Muon papers as a list; "post-trainign" -> "post-training" |

Tools (scratchpad, not kept): `build_md.py` (export cleanup) and `edits.py` (the content changes, each asserted to
match exactly once).

### 45.3 Plan (2026-10-15): the full packed SFT pipeline first, kernel optimisation paused

The user changed the order: attention-kernel optimisation pauses; first a complete SFT pipeline with packing (v2),
then optimisation on top of it. Status checked in the code today (laptop master `~/Downloads/BluTrain_sft`):

| piece | status | evidence |
| --- | --- | --- |
| loss (masked CE, global trained-token count) | ready, no change for packing | labels are built per chat before any packing (`sft/prepare_sft_data.py` `build_row`: inputs = ids[:-1], labels = ids[1:]) |
| chat template (Tülu 3 minus the between-turn `\n`, the 53.0 layout) | ready, decided 10-15 | §45.1 (code + stage-0 data) |
| RoPE with per-token positions (restart at 0 per chat) | ready in kernel and autograd | `autograd::fused_rms_rope(..., positions)` (`NormalizationOps.h`, saved for the backward) |
| attention forward, packed rows (v62, designs A and B) | ready in the kernel | `AttentionKernels.h:110-161` (`GqaChatPlan` for A, `seg_start` [B,T] for B) |
| attention backward, packed rows | ready in Vj1d only (the default) | `GQA_bwd_generic.cu:728`: a packed call with `USE_VJ2=1` (or `USE_FLASH`/`USE_DETERMINISTIC_DQ`) throws, so chats can never silently mix |
| autograd attention op passes packing info | **missing** | `AttentionOps.h`: no `seg_start` / plan parameter; the backward node stores none |
| trainer passes positions + packing info to every layer | **missing** | `bluscript_zero_ckpt.cpp:617-619` (`pos_offset=0`, no positions), `:642` (attention, no packing) |
| dataloader carries per-row chat segments | **missing (small)** | `sft/SftDataLoader.h` reads only inputs/labels/valid_counts/meta and returns the pretraining `Batch` |
| preprocessing packs chats (First-Fit-Decreasing) | **missing** | one chat per row today |
| end-to-end tests "packed == the same chats unpacked" | **missing** | only kernel-level packed tests exist (`test_varlen_attention.cu`, 605/605) |
| SFT and DPO use the same tokens | **no: mismatch** | `prepare_dpo_data_tokenize.py:52-78, 114, 119` (below) |

Kernel selection facts: the attention kernels live in `Tensor-Implementations/src/Kernels/cuda/attention/arch/`
(`GQA_fwd_sm90.cu` = forward v62, `GQA_bwd_vj1d_sm90.cu` = Vj1d, `GQA_bwd_vj2_sm90.cu` = Vj2), not in `LossKernels.cu`.
The backward is chosen at run time by environment variables in `GQA_bwd_generic.cu` (default Vj1d; `USE_VJ2=1` = Vj2,
dense only); there is no compile-time flag. The model ties `wte` and the output layer (`bluscript_zero_ckpt.cpp:702-767`)
and the loss runs over all 50,304 rows (`vocab_size = 50304`, line 138).

**The DPO preprocessing does not match our SFT** (`/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/prepare_dpo_data_tokenize.py`,
seen 10-15; its comments say "same as SFT" four times):

- it registers `<|user|>` = 50257, `<|assistant|>` = 50258, `<|system|>` = 50259 as tiktoken special tokens, so each
  marker is ONE new id; our SFT writes them as plain text (6 / 7 / 6 GPT-2 tokens);
- it encodes message CONTENT with `allowed_special` (line 119), so a message containing the text `<|endoftext|>` or
  `<|user|>` becomes a real control token (our SFT uses `encode_ordinary`: plain text);
- it says its data must match the SFT checkpoint `stage0_adamw_full`, which is OUR plain-text SFT run: DPO would feed
  that model 3 ids it never saw as inputs (with tied embeddings and a 50,304-wide softmax in pretraining, those rows
  were only ever pushed DOWN as outputs);
- the layout (no `\n` after an answer's end token) and the end token 50256 do match;
- it mentions an `sft_tokenizer.py` that is not on the laptop (to ask the DPO team; their
  `compute_dpo_reference_logprobs.cpp` sits untracked in the box's `~/gautam_k/BluTrain_sft`).

**Phase 0, decisions (before coding):**

- D1 one tokenization for SFT and DPO: (a) plain-text markers everywhere (DPO switches to `encode_ordinary`; no SFT
  re-run) or (b) special ids 50257-50259 everywhere (SFT re-run first so those rows are learned). Recommendation: (a)
  now (our SFT model already learned it, Tülu 3 does the same), (b) only if the ablation shows a gain. The
  `allowed_special` on content must be fixed in either case.
- D2 end-of-answer token: keep 50256 (as Tülu 3 reuses Llama's `<|end_of_text|>`) or a dedicated end-of-turn id in a
  spare row (as Llama 3 `<|eot_id|>`). Recommendation: keep 50256 for the pipeline; test a dedicated id later, once a
  generation loop exists (stopping can only be measured by generating).
- D3 packing design: wire design A (the faster main path, §41.3); keep B behind a debug switch.
- D4 packing rule: First-Fit-Decreasing (same rows as Best-Fit on our data, packing doc §13).
- D5 steps / LR: first run keeps 64 rows per step; then the step-count ablation (§44.1).

**Phase 1, library:** the autograd attention op and its backward node take an optional packing argument (design A
plan, or B `seg_start`); empty = dense = bit-identical to today. Tests: dense output unchanged bit for bit; packed via
autograd = the kernel-level packed tests; a packed row = the same chats as separate padded rows (outputs, LSE, dQ/dK/dV
within tolerance, positions reset per chat).

**Phase 2, data:** `prepare_sft_data.py --pack ffd`: chats built exactly as today, packed First-Fit-Decreasing into
rows, plus `seg_start.bin` (uint16 [N, T]; positions = t - seg_start), meta (rule, rows, fill %, chats per row);
self-tests (every chat once, no label crosses a chat, seg_start contract, the pad tail is its own segment). Val packed
too, and the v1 val kept for the cross-check. `SftDataLoader`: optional `seg_start.bin`; per micro-batch uploads
positions [B, T] int32 and builds one design-A plan (host segment list -> `gqa_chat_plan_build`), shared by all layers
and the backward; without the file it behaves exactly as today.

**Phase 3, trainer:** thread positions + plan from the batch to every block (`fused_rms_rope(positions)`,
`scaled_dot_product_attention(plan)`); pretraining untouched; log fill rate and chats per step.

**Phase 4, proof runs (1 GPU, short):** pretraining step 0 bit-identical; val loss on the packed val set = val loss on
the v1 val set (same chats; checks masking + positions end to end); step 0 packed = step 0 padded on the same chats;
then the full stage-0 packed run vs the v1 run (val curve per token, throughput).

**Phase 5, after the pipeline:** DPO alignment check with the DPO team, the special-token / end-token ablation (needs
generation), the step-count ablation, then the paused optimisations (Vj2 packed modes = G7, Vj2 as default, the
forward v62 work, B-1 test 2).

### 45.4 Future task: split over-long chats like TRL `bfd_split` (decided 2026-10-15: not now)

Today `prepare_sft_data.py` DROPS every chat longer than the row (T = 4096): 10 of 5,010 sampled chats on stage-0 (0.2%;
packing doc §7.5). The user wants, later, TRL's `bfd_split` behaviour (packing doc §8.3): cut an over-long chat into
pieces of at most T tokens and pack each piece as its own segment, with its own start (positions restart at 0) and end,
so no data is thrown away. For now we keep dropping. Points to settle when we do it:

- where to cut (TRL cuts every `max_length` tokens, even inside a turn) and whether a piece may end in the middle of an
  answer: the end token 50256 would then be missing from every piece except the last, which is what our "EOT must be a
  target" rule (§32.2) protects against;
- which tokens of each piece are targets (a piece that starts inside an answer has no prompt, so its first tokens are
  trained without the question they answer);
- the context lost between pieces (a later piece cannot see the earlier one; packing doc §7.6) and whether to repeat
  the prompt at the start of each piece instead;
- how many chats are affected on the real data mix (stage-0: 0.2%; long-answer data such as reasoning traces: many more);
- an ablation: drop vs split, same data, same steps.

### 45.5 The packed SFT pipeline: built and proven end to end (2026-10-15)

> **Update (same evening):** the pipeline below used packing design B (`seg_start`). It was switched to design A
> (the per-chat list, the cu_seqlens way) everywhere and B was removed: see §45.7. The results below are kept as measured.

Kernel optimisation paused (§45.3); the full packed (v2) SFT pipeline was built in one day and passes its one
mandatory end-to-end test. Only the mandatory tests were run (GPU time is short); the others are listed under "later".

**What was built (laptop master `~/Downloads/BluTrain_sft`, box copy `~/gautam_k/sft_v2/BluTrain_sft`):**

| step | change | file (md5) | check |
| --- | --- | --- | --- |
| A1 | Vj2 backward learns packed rows (design B, `seg_start`): per key block only the query steps whose chat can see it (a prefix, Vj1d's proof), the step's 64 `seg_start` loaded with LSE/D, mask `key < seg_start[query]`; packed smem 199,232 B | `GQA_bwd_vj2_sm90.cu` (ced7d3d8) | box `test_varlen_attention` G9 under Vj2: **94/94**; dense Vj2 SASS **identical** to b1 (1,728/1,728 lines) |
| A2 | Vj2 is the DEFAULT Hopper backward; `USE_VJ2=0` = Vj1d; design A plans run on Vj1d; the `T % 128` fallback now passes the packing on | `GQA_bwd_generic.cu` (c81c9909) | CLASSIC 144/144 (Vj2); G9 94/94 and G11 95/95 with `USE_VJ2=0` |
| A3 | autograd `scaled_dot_product_attention(..., seg_start)`: checked like `positions`, passed to the forward, kept in `GQAFlashAttentionQKVBackward` for the backward | `AttentionOps.h/.cpp` (438f6177 / ea40d371), `AttentionBackward.h/.cpp` (124f8b91 / f8163997) | compiles; covered by the e2e test |
| D1 | `prepare_sft_data.py --pack ffd`: chats labelled alone, First-Fit-Decreasing with a max segment tree, rows shuffled, `seg_start.bin` (uint16 [N,T], pad tail = own segment); packing self-checks | `sft/prepare_sft_data.py` (c1f2f3b2) | self-test (tree FFD == plain FFD on 300 sets); **v1 output byte-identical** to the old stage-0 files |
| D2 | `SftDataLoader`: optional `seg_start.bin` (contract checked when opened); per micro-batch `last_seg_start()` and `last_positions()` (t - seg_start) as int32 [B,T] | `sft/SftDataLoader.h` (ed727f72) | covered by the e2e test |
| T1 | `bluscript_sft.cpp` = copy of `bluscript_zero_ckpt.cpp` (unchanged) + global `g_packed` set before every SFT forward, read by RoPE (`positions`) and attention (`seg_start`); `BLU_SFT_B` (rows per micro-batch); `g_packed` emptied after the training loop | `bluscript_sft.cpp` (05dede59) | e2e test; the exit fix is checked on the first real run |

**Packed stage-0 data** (`~/Downloads/sft_data/stage0_ultrachat_5k_T4096_packed`, box `~/gautam_k/sft_data/...`):
train 5,000 chats -> **1,517 rows** (minimum possible 1,511), real tokens 99.6% of the slots (v1: about 30%), 3.3 chats
per row (max 12); val 500 chats -> **147 rows** (min 146), 99.0%. Same chats and same 15,845 / 1,555 end-token targets
as v1.

**The mandatory test (box GPU 1, 2026-10-15 15:44-15:58):** the same 500 chats (the val set used as the train split),
once as 500 v1 rows and once as 147 packed rows; `BLU_SFT_B=1` (no row dropped), one optimizer step over all rows
(`BLU_SFT_ROWS_PER_STEP` = 500 / 147, 3 steps because the schedule needs a decay tail), AdamW 2e-5, from step 8000.

| | v1 (500 rows) | packed (147 rows) |
| --- | --- | --- |
| step 0 val loss | 1.445038 | 1.445084 (+0.003%, bf16 rounding) |
| step 0 train loss | 1.445038 | 1.445084 |
| **step 0 grad norm** | **0.7935** | **0.7935** (group norms 0.7929 / 0.0321 in both) |
| step 1 train loss / norm | 1.423608 / 0.6496 | 1.423671 / 0.6502 |
| val after step 2 | 1.381130 | 1.381213 |
| trained tokens | 444,233 | 444,233 |
| time per step (steps 0 / 1 / 2) | 53.6 / 49.0 / 43.9 s | 21.2 / 18.3 / 13.3 s (**2.5-3.3x faster**) |

The equal step-0 loss proves the data, the loader, the per-chat positions and the forward masking; the equal grad norm
proves the backward (Vj2 packed) through the whole model.

**Bug found and fixed:** the packed run printed "Training complete" and then hung (futex wait, 57.8 GB held on the GPU)
while the v1 run exited. `g_packed` is a global made before `main`, so it was destroyed after the caching allocator (a
static made later) and freeing its two GPU tensors then blocked. Fix: `g_packed = PackedRows{}` right after the training
loop. The box blocks debuggers (`ptrace_scope=1`), so this is from the code, not from a trace.

**How to run (box):** data `prepare_sft_data.py --pack ffd`; build `make -j all SM_ARCH=90`, then
`bash build_sft_cmd.sh` (the run-mpi compile line with `-o bluscript_sft_bin`); run as the 10-07 runs, with
`BLU_SFT_DATA=<dir with packed train>`; a split without `seg_start.bin` trains as v1.

**Later (not run, by decision):** Vj2 vs Vj1d pretraining comparison, compute-sanitizer on packed Vj2, CLASSIC error-value
diff vs b1, packed speed per kernel, design A in Vj2, the step-count ablation (§44.1), the drop-vs-split ablation (§45.4).

**First real packed run, `packed64` (box GPU 1, 2026-10-15 16:13-16:19; `bluscript_sft_bin` 05dede59):** stage-0 packed
train (1,517 rows) + the v1 val set (496 chats on 1 GPU, so the numbers compare directly with 10-07), AdamW 2e-5 ->
2e-6, B=8, 64 rows per step (about 211 chats per step), 2 epochs = 47 steps.

| | v1 (10-07, 1 GPU) | packed64 |
| --- | --- | --- |
| rows (chats) per step | 64 (64) | 64 (~211) |
| optimizer steps, 2 epochs | 156 | 47 |
| val at step 0 | 1.4464 | 1.4464 |
| val at the end | 1.2839 | 1.3002 |
| time per step / 2 epochs | 5.3 s / ~14 min | 5.2 s / ~6.3 min (sum of step times 379.5 s, first step 58.4 s) |

Val curve: 1.4464 (0), 1.3433 (10), 1.3180 (20), 1.3057 (30), 1.3014 (40), 1.3002 (46). The process exited by itself
after "Training complete", so the `g_packed` exit fix works. The end val is 0.016 higher than v1 with 3.3x fewer
optimizer steps at the same LR schedule: the step-count effect of §44.1 (correctness is shown by the same-chats test
above). `packed16` (16 rows per step, about 53 chats per step, 189 steps) was started to separate that effect and was
stopped by the user at step 5; to re-run when GPU time allows.

### 45.6 Special tokens or plain-text role markers: what others do, and the decision to make (research, 2026-10-15)

(Answers questions 1-4 of the §44.6 future task; the chat-template part is §45.1.)

**Today:** our SFT data writes `<|system|>`, `<|user|>`, `<|assistant|>` as plain text (6 / 6 / 7 GPT-2 tokens; the word
"assistant" is the pieces `ass` + `istant`), and only the end token 50256 is a special id. The DPO team's
`prepare_dpo_data_tokenize.py` makes the three markers special ids 50257 / 50258 / 50259 (and encodes message content
with special tokens allowed). Our SFT checkpoints never saw those ids, so the two do not match.

**What others do (checked 2026-10-15):**

| model / tool | role markers | end of a turn | when the ids exist | checked in |
| --- | --- | --- | --- | --- |
| Tülu 3 (Llama 3.1 base) | plain text `<\|user\|>` ... | Llama's `<\|end_of_text\|>` | n/a (plain text) | its `tokenizer_config.json` (§45.1) |
| Zephyr-style (open-instruct `"zephyr"`) | plain text | the model's eos (`</s>`) | n/a | open-instruct `CHAT_TEMPLATES` @95a84bb |
| Llama 3 instruct | special ids `<\|start_header_id\|>`role`<\|end_header_id\|>` | `<\|eot_id\|>` (128009) | reserved in the tokenizer before pretraining | Tülu 3's tokenizer (Llama 3's tokens) |
| Llama 3 **base** | (same ids) | | the rows of `<\|eot_id\|>`, `<\|start_header_id\|>`, `<\|end_header_id\|>` are **zero** in the base model: "The original Llama 3 8b (base) special token weights are zero, which might cause NaN gradients" | [imone/Llama-3-8B-fixed-special-embedding](https://huggingface.co/imone/Llama-3-8B-fixed-special-embedding) (also reported by [Unsloth](https://unsloth.ai/blog/llama3), page not machine-readable) |
| SmolLM2-1.7B base | special `<\|im_start\|>`, `<\|im_end\|>` (ChatML) | `<\|im_end\|>` in chat | ids **1 and 2**, right after `<\|endoftext\|>` = 0: made with the tokenizer | [tokenizer_config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/blob/main/tokenizer_config.json) |
| Qwen2.5-0.5B base | special `<\|im_start\|>`, `<\|im_end\|>` | `<\|im_end\|>` | 151644 / 151645 in the base tokenizer | [tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/tokenizer_config.json) |
| OLMo-2-0425-1B base | special `<\|im_start\|>`, `<\|im_end\|>` | `<\|im_end\|>` | 100264 / 100265 in the base tokenizer | [tokenizer_config.json](https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/tokenizer_config.json) |
| OLMo 3 (paper) | chat special tokens put into MIDTRAINING data made the model print them (GSM8K 49.43 -> 0); the same layout as plain text kept 46.02: "Leave special tokens for SFT stage" | | | [arXiv 2512.13961](https://arxiv.org/abs/2512.13961) (our report §2.1) |
| TRL `clone_chat_template` | takes a base model WITHOUT chat tokens (its example: Llama-3.2-1B) and, at SFT time, adds another model's special tokens (Qwen3's), switches the eos to that end-of-turn token and resizes the embeddings | | added at SFT | [trl@478624d chat_template_utils.py:45](https://github.com/huggingface/trl/blob/478624dfa96e35451ee68b6d6299fdd10940154f/trl/chat_template_utils.py#L45) |
| transformers `resize_token_embeddings` | new rows start from the mean (and covariance) of the old rows by default (`mean_resizing=True`), "to reduce the kl-divergence between the next token probability before and after adding the new embeddings" | | | [transformers@4cc2aa8 modeling_utils.py:2634](https://github.com/huggingface/transformers/blob/4cc2aa84301c9aa210b5513dab9fefae03981f6e/src/transformers/modeling_utils.py#L2634), citing [Hewitt](https://nlp.stanford.edu/~johnhew/vocab-expansion.html) |

No published head-to-head comparison of "special ids vs the same markers as plain text" for SFT was found. Both are used
by strong recipes (Tülu 3 plain text; Llama 3 / Qwen / SmolLM / OLMo special ids).

**The answers:**

1. **Is adding them to the tokenizer enough?** No. Three parts: (a) the tokenizer maps the exact string to one id (easy;
   the DPO script does it); (b) the MODEL must learn the id: our ids 50257-50259 have rows (50,304 rows) but were never
   an input, and because the input table is tied to the output layer and pretraining's softmax covered all 50,304 rows,
   those rows were only pushed down ("never predict me"); they need a starting value (the mean of the trained rows, as
   transformers / Unsloth / the imone fix do) and SFT trained with them; (c) every stage and tool uses the same ids
   (SFT, DPO, eval, chat, HF converter).
2. **Must they be added when the tokenizer is trained?** No. BPE training only learns merges of ordinary text; special
   tokens are a separate list of exact strings matched BEFORE BPE and mapped to fixed ids, so they can be added later.
   Reserving them at tokenizer creation (as SmolLM2 did) is free and keeps the ids stable; we cannot retrain our
   tokenizer (it is GPT-2's; every id would change), but we have 47 spare ids, so adding them changes no existing id.
3. **When does the model learn them?** Reserving an id is not training it (Llama 3 base: reserved but zero rows). OLMo 3
   found that putting chat special tokens into midtraining hurt and advises SFT. Our pretraining is finished, so for this
   model SFT is the only place. For a future base model: reserve the ids in the tokenizer from the start, keep them out
   of pretraining/midtraining data, learn them in SFT.
4. **Which is better for us?**
   - Plain text (A): works (our SFT runs, Tülu 3), no new rows, no initialisation risk. Weak point: a user can type
     `<|assistant|>` and it becomes exactly the same tokens as the real marker (the model cannot tell them apart).
   - Special ids (B): 1 id per marker (about 20 tokens per chat saved, ~1.6% of an average chat), a user's typed
     `<|assistant|>` stays plain text (when content is encoded with `encode_ordinary`), clean role/stop parsing, the
     modern default, and it matches the DPO team. Cost: 3 rows to initialise and learn, an SFT re-run (now ~6 minutes
     packed on 1 GPU for stage-0), every tool aware of the ids.
   - **Recommendation (changed from §44.6/§45.3, which said plain text):** move to B for SFT and DPO together, with the 3
     rows initialised to the mean of the trained rows (ids 0-50256), the end token kept at 50256, and content always
     encoded as plain text; decide it with one A-vs-B ablation (same packed data and steps; compare answer-token val loss
     now, and stopping / role leaks once generation exists) before the DPO team builds its final data. Reasons for the
     change: the DPO team already uses B, the injection weak point of A, every checked modern base tokenizer reserves
     chat tokens, TRL's own path adds them at SFT, and an SFT re-run is now cheap.
   - The end-of-turn token stays 50256 in both (Tülu 3 reuses its eos; a separate end-of-turn id would be a fourth new
     row and is tested only after generation exists).

**Not done yet:** the A-vs-B ablation (needs GPU); measuring the current norms of rows 50257-50259 in the checkpoint; the
DPO team's `sft_tokenizer.py` and which SFT checkpoint their DPO starts from.

### 45.7 Packed rows: design A (the per-chat list) everywhere, design B (seg_start) removed (2026-10-15, evening)

**Why this change.** On 2026-10-12 we decided that packing design A is the default (§41.3), because it was measured
faster. The pipeline of §45.5 was still built on design B (one `seg_start` number per token) in the morning. That went
against the decision, and the speed cost was not flagged. The user decided: use only the list way ("only seqlens"),
remove B from every kernel, and make A the pipeline's way.

The two designs answer the same question ("which keys may this query see?") in two ways:

| | design B (removed) | design A (kept) |
| --- | --- | --- |
| what the kernel gets | one number per token: `seg_start[t]` = where token t's chat starts | a list of chats per row: (row, start, length) |
| how work is cut | in fixed row tiles; a tile that holds a chat border mixes two chats and needs the extra mask | per chat; every chat starts on its own tiles, only the normal causal mask is left |
| who uses it | FlexAttention's document mask; HF transformers on its SDPA / eager paths | FlashAttention 2/3/4 (`cu_seqlens`), TE `thd` / Megatron `PackedSeqParams`, PyTorch `varlen_attn`, HF on its FlashAttention path, TRL |

Measured on our kernels (2026-10-10/12, real packed rows): forward A is 2-16% faster than B; backward A is 3-5% faster
on middle and long chats, B is 2-4% faster only on very short chats. Middle rows: forward A 501 µs vs B 538 µs,
backward A 1,466 µs vs B 1,537 µs. (The source links for "who uses it" are in the research of 2026-10-15: FA2
`flash_attn_interface.py:1391`, FA3 `hopper/flash_attn_interface.py:890`, FA4 `flash_attn/cute/interface.py:4167`,
TE `dot_product_attention.py:470, 550`, Megatron `packed_seq_params.py:10-46`, PyTorch `varlen.py:62-65`, torchtitan
`attention.py:132, 177, 467`, HF `modeling_flash_attention_utils.py:480-517` and `masking_utils.py:734-763`, TRL
`sft_trainer.py:1180-1202`.)

**Why the forward kernel (v62) had both, and how one was chosen.** On 2026-10-10 both were built into the same kernels
to measure them (Step 3 = B, Step 5 = A). They were compiled as two modes of one template kernel, next to the dense
mode. There was no flag or environment variable: every call chose by its arguments. A `seg_start` pointer meant B, a
plan meant A, neither meant dense, and both together were refused. After 2026-10-12 nothing removed B, and the
pipeline then called the B form. Now the B mode is gone from v62, Vj1d and Vj2, and the `seg_start` argument is gone from
the API, so there is one packed path left to test.

**How design A works, step by step.** Example row, T = 4096 tokens: chat 1 has 1,000 tokens, chat 2 has 2,500 tokens,
and the last 596 positions are padding (the "pad tail").

1. **The data file stores the list.** `prepare_sft_data.py --pack ffd` writes, per row, the lengths of its chats in row
   order: `[1000, 2500]` (`seq_lengths.bin`, int32, all rows one after the other; `seq_offsets.bin`, int64 [N+1],
   says where each row's list starts). This is TRL's `seq_lengths` column (`trl/data_utils.py:775`, TRL commit 478624d).
   The positions after the last chat are the pad tail. 5,000 chats take 20 KB (the old per-token `seg_start.bin` was
   12.4 MB).
2. **The loader turns the list into segments and positions**, for every micro-batch (`SftDataLoader::next_batch`):
   segments (row, start, length) = (0, 0, 1000), (0, 1000, 2500), (0, 3500, 596): every chat, then the pad tail as a
   segment of its own (the kernels need every token covered). The RoPE positions restart at 0 in every segment:
   0..999, 0..2499, 0..595, as TRL's collator builds `position_ids` from `seq_lengths` (`sft_trainer.py:478-489, 528-541`).
3. **The trainer builds one plan per micro-batch** (`packed_rows_of` in `bluscript_sft.cpp` calls
   `gqa_chat_plan_make(rows, starts, lengths, n, B, T, Hq)`). The plan is a few small lists of work items, made on the
   host and uploaded with ONE asynchronous copy (the host does not wait for the GPU). All 26 layers use the same plan,
   in the forward and in the backward. It is freed when the last user lets go of it (a shared handle: the attention
   backward nodes keep it until their backward has run).
4. **Forward (v62).** One work unit = (one segment, one couple of its 128-row query pairs, one head). For the example
   row with 18 query heads: chat 1 has ceil(1000/128) = 8 query pairs = 4 couples, chat 2 has 20 pairs = 10 couples,
   the pad tail 5 pairs = 3 couples, so (4 + 10 + 3) x 18 = 306 units. Every index inside a unit is counted from the
   segment start, so a unit of chat 2 starts reading keys at token 1000 and never loads a key of chat 1. The mask
   inside a unit is the plain causal mask. Rows past the segment end are computed but never stored.
5. **Backward (Vj2, the default).** One block = (one 128-key tile of a segment, one query head): chat 1 has
   ceil(1000/128) = 8 tiles, chat 2 has 20, the pad tail 5, so 33 x 18 = 594 blocks. A block of key tile kt visits only
   its own chat's 64-row query steps, from the tile's diagonal (step 2kt) to the chat's last step: tile 0 of chat 1
   visits 16 steps, tile 7 visits 2. Blocks run longest first, in sections of (row, head) pairs, as the dense grid does
   (FA3's LPT scheduler; for rows that hold one chat each, the plan's order is exactly the dense order).
   Two things are new compared with dense, and both are FlashAttention-3's way:
   - *Padded slots.* A chat can start at any token, but Vj2 reads LSE and D in 256-byte bulk copies and adds dQ in
     fixed 64 x 128 tiles, which need 64-row alignment. So every chat gets its own 64-row-aligned slot per head in
     those three buffers (FA3 varlen's `offset_padded`): ceil(1000/64) + ceil(2500/64) + ceil(596/64) = 16 + 40 + 10 = 66
     tiles per head instead of 64 (3% more). A small prepare kernel fills the slots; slot rows past a chat's end get
     LSE = +inf and D = 0, so they add exact zeros (P = exp(-inf) = 0, dS = 0; FlashAttention 2 does the same,
     `flash_bwd_kernel.h:411-413`). A convert kernel writes each slot back to its chat's token rows.
   - *The chat's last key tile.* Keys past the chat end sit only in its last tile; causal hides them. Their dK/dV rows
     belong to the next segment, so that tile stores only its rows below the chat end, with plain stores (FA3 stores all
     varlen dK/dV this way, `hopper/epilogue_bwd.hpp:31, 263-268`).
   `USE_VJ2=0` runs Vj1d instead, which has had design A since 2026-10-10 (64-key tiles, `bwd_tiles` of the same plan).
6. **Why B was slower:** with `seg_start`, key tile 7 of the example row holds keys 896-1023, that is keys of chat 1
   (896-999) AND of chat 2 (1000-1023). It must run the masked path and visit the query steps of both chats. Every
   tile that holds a chat border pays that. In A every tile belongs to one chat.

**What changed (laptop `~/Downloads/BluTrain_sft`, 14 files; the morning's B versions were never committed, a copy was kept only in the session's scratch folder):**

| file | change | md5 |
| --- | --- | --- |
| `GQA_fwd_sm90.cu` | v62: design B mode removed (modes now dense = 0, chats = 1); launcher takes only the plan's units | eff8cdb6 |
| `GQA_bwd_vj1d_sm90.cu` | Vj1d: design B mode removed; launcher takes only the plan's tiles | 302ece10 |
| `GQA_bwd_vj2_sm90.cu` | Vj2: B replaced by A (`template<bool CHATS>`, block list from the plan, padded slots, `bwd_prep_vj2_chats`, `convert_dq_frag_vj2<true>`, plain-store epilogue for a chat's last tile); `gqa_vj2_section_pairs(S)` shared with the plan builder | 7667deec |
| `GQA_fwd_generic.cu` | forward API without `seg_start`; plan builder adds the Vj2 block list and the padded dQ slots, one allocation + one async copy; `gqa_chat_plan_make` (shared handle) | c993599a |
| `GQA_bwd_generic.cu` | backward API without `seg_start`; Vj2 mode runs the plan on Vj2 when T % 128 == 0, else Vj1d | 2dbe59eb |
| `AttentionKernels.h` | `GqaChatPlan` + `vj2_blocks`, `dq_tiles`, `storage`; `GqaChatPlanPtr`; `seg_start` parameters removed | a95a7c63 |
| `AttentionOps.h/.cpp` | autograd `scaled_dot_product_attention(..., const cuda::GqaChatPlanPtr& chats = nullptr)` | 1453a83f / 5ea8541e |
| `AttentionBackward.h/.cpp` | the QKV backward node keeps the plan (`chats_`) instead of `seg_start_` | 0be1184e / e46df1f2 |
| `test_varlen_attention.cu` | G8/G9 (the B groups) deleted, -808 lines; G11 now also runs under Vj2; timings dense vs A | 612b7d53 |
| `sft/prepare_sft_data.py` | `--pack ffd` writes `seq_lengths.bin` + `seq_offsets.bin`; new check: every chat's last position predicts its own end token, and the pad tail trains nothing | f9dc7a42 |
| `sft/SftDataLoader.h` | reads the lists; per micro-batch: positions (GPU) + segment lists (host); refuses an old folder that has only `seg_start.bin` | 83c88d41 |
| `bluscript_sft.cpp` | `g_packed = {positions, chats}`; `packed_rows_of()` builds the plan per micro-batch | 204587bd |

Packed stage-0 data regenerated (laptop `~/Downloads/sft_data/stage0_ultrachat_5k_T4096_packed`): `seq_lengths.bin` /
`seq_offsets.bin` / `meta.json` = train 2e23ee29 / 207168e2 / c2feaf3a, val fc96ebae / 068c7987 / c86824ba;
`seg_start.bin` removed. Same 1,517 + 147 rows.

**Checks done on the laptop (no GPU):**

| check | result |
| --- | --- |
| machine code (SASS), dense v62 before vs after | identical, 2,280 instructions |
| SASS, v62 design A before vs after | identical except two parameter offsets (the removed pointer), 2,168 |
| SASS, dense Vj1d / Vj1d design A | identical (1,624) / identical except parameter offsets (1,784) |
| SASS, dense Vj2 vs the first Vj2 (b1) | identical, 1,760 instructions |
| Vj2 design A kernel | 1,688 instructions, 168 registers, 0 spills (same as dense) |
| plan builder, the real source compiled on the CPU with a fake allocator, 254 layouts (one chat per row, random chats, 4,096 one-token chats, border cases) | 0 failures: every (chat, head, key tile) exactly once, every step's slot = its own rows, LPT order in sections, one-chat rows = the dense order exactly. Two planted bugs were caught (9.2 M and 108 K failures), so the check is not vacuous |
| data regenerated with the new script | inputs / labels / valid counts byte-identical to the morning's packed files; the chat boundaries from the lists = the old `seg_start.bin` exactly (train 5,000 chats, val 500) |
| loader, its real new code on the CPU over every row and micro-batch | 0 check failures; positions = t - old seg_start for every token; 5,458 train + 604 val segments = the old chat runs. A planted bug was caught (358,314 wrong positions) |
| v1 (unpacked) output of the script | still byte-identical (fdb9ef90 / eec5e38b / d57b5ac5) |
| compile | all 5 kernel files (nvcc 13.0, sm_90a), autograd files, the test, `sanitize_vj2.cu`, the trainer (`mpic++ -fsyntax-only`): 0 errors |

**Box (2026-10-15 18:14):** the 14 files and the new data copied (checksums verified, the box's `seg_start.bin` removed);
`make -j all SM_ARCH=90`, the trainer (`build_sft_cmd.sh`) and the test binary `tva_a` all built with exit 0. GPUs were
not available, so nothing was run.

**Pending (needs the GPU, run only with permission):**
1. `test_varlen_attention` G11 under Vj2 (quick: all layouts with 18/6 heads, 16 real rows, repeat 50): design A
   backward vs the fp64 reference, every chat = the chat run alone (dK/dV bit for bit), isolation, refusals.
2. The same-chats end-to-end test of §45.5 again (500 v1 rows vs 147 packed rows, step 0): val loss and grad norm must
   match as before (v1: 1.445038 / 0.7935).

Later: G11 with `USE_VJ2=0` (Vj1d's design A, unchanged code), compute-sanitizer on Vj2 design A, speed of A on Vj2 vs
the dense rows, and the `packed16` run (§45.5).

### 45.8 Everything still to do: tasks and tests (written 2026-10-15 evening)

One place for every open task and test. When an item is done, tick it and write the date and the result next to it;
do not delete lines. Sources are given for every item, so each can be checked.

**Standing rules for every run (the user's instructions):** no GPU run without the user's OK, and the user names the GPU
index (this morning: GPU 1); the user runs the box commands; one log per run, with a summary; say how long a run takes;
only mandatory tests now, the rest are "later"; edit on the laptop and copy to the box (never edit on the box); the
user does all commits and pushes.

**State at the end of 2026-10-15:** packing design A (the per-chat list) is in every kernel and in the whole pipeline,
design B is removed (§45.7). Laptop: done and checked without a GPU. Box (`~/gautam_k/sft_v2/BluTrain_sft`): files
and data copied (checksums verified, the box's `seg_start.bin` removed), library + trainer + test binary built at 18:14,
all exit 0. **Nothing of the design-A change has run on a GPU yet.**

**2026-10-16 morning, first:** the learning session of §45.10 (how blocks compute masked work, segstart vs seqlens).

#### 45.8.1 Next GPU window: the two mandatory tests (in this order)

- [x] **Test 1 - G11 quick under Vj2** (about 5-10 min, not measured before). **DONE 2026-10-16, PASS: 94/94, 12 s (2x H200 box, GPU 0).** Checks the new Vj2 design-A backward: dQ/dK/dV
  of every chat and pad tail against the fp64 reference, every chat = the same chat run alone (O/LSE/dK/dV bit for bit,
  dQ within FlashAttention's tolerance), isolation (changing one chat changes no other chat), the refusals, a repeat
  test, and a dense-vs-A backward time.
  ```bash
  GPU=1; cd ~/gautam_k/sft_v2/BluTrain_sft/Tensor-Implementations && nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader
  CUDA_VISIBLE_DEVICES=$GPU VARLEN_QUICK=1 VARLEN_ONLY=G11 VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin LD_LIBRARY_PATH=lib:../Profiler/lib:$LD_LIBRARY_PATH ./tva_a > ~/gautam_k/sft_v2/runs/G11_quick_vj2_A.log 2>&1; echo "exit=$?"; grep -E 'backward mode|FAIL|checks passed|ALL RUN' ~/gautam_k/sft_v2/runs/G11_quick_vj2_A.log | head -30
  ```
  Pass: `backward mode: Vj2 (default)`, `checks passed N, failed 0`, `ALL RUN CHECKS PASS`.
- [x] **Test 2 - the same-chats end-to-end run, packed, design A** (about 3 min). **DONE 2026-10-16, PASS: step-0 val 1.445038 = v1 exactly, grad norm 0.7935 (GPU 1).** The same 500 chats as §45.5 (147
  packed rows, `BLU_SFT_B=1`, one optimizer step over all rows, from step 8000). The v1 side does not need a re-run: its
  code path did not change (dense machine code byte-identical), and its numbers are known.
  ```bash
  G=~/gautam_k; R=$G/sft_v2/BluTrain_sft; BIN=$R/bluscript_sft_bin; INIT=$G/ckpt/blumodel_step_8000; GPU=1
  LDP=$(make -C $R --no-print-directory print_run_ld_path); nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader
  d=$G/sft_v2/runs/e2e_packed_A; mkdir -p $d && cd $d
  for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  env CUDA_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$LDP BLU_SFT_DATA=$G/sft_data/e2e_packed BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0 BLU_OPTIMIZER=adamw BLU_SFT_B=1 BLU_SFT_ROWS_PER_STEP=147 BLU_SFT_EPOCHS=3 \
    mpirun -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA -x BLU_INIT_FROM -x BLU_ALL_BF16 -x BLU_CKPT -x BLU_OPTIMIZER -x BLU_SFT_B -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS $BIN < /dev/null > run.log 2>&1
  echo "== packed A exit=$?"; grep -E "SftLoader|\[val\] step 0 |^step +0 |Training complete|ERROR" run.log | cut -c1-200
  ```
  Pass: the loader line says `PACKED rows (seq_lengths.bin: 500 chats, each attends only to itself)`; step-0 val loss
  about 1.4450 (v1: 1.445038; design B gave 1.445084); step-0 grad norm 0.7935 (group norms 0.7929 / 0.0321); the
  process exits by itself after "Training complete". With A every chat's attention is cut exactly as in its own v1
  row, so the loss should match v1 to at least 4 digits.
- [ ] After both: pull the two logs to the laptop (`profiling_reports/`), write the results into §45.7, tick here.

If a test fails: stop, keep the log, do not change anything on the box; the fix is made on the laptop.

#### 45.8.2 Then: finish the SFT pipeline phase (user's order, §45.3 phases 4-5)

| # | task | needs | source |
| --- | --- | --- | --- |
| 2.1 | [ ] Re-run the stage-0 sample runs on the packed data: AdamW 2e-5, and the Muon runs (2e-5 / 5e-5 / 1.2e-4 from the 10-07 sweep), same settings as 10-07, the v1 val set kept for direct comparison | GPU, ~6 min per run packed on 1 GPU | user's plan 10-15; v1 runs §37.4-37.5 (AdamW 2e-5 end val 1.2809 on 4 GPUs; Muon best at 1.2e-4) |
| 2.2 | [ ] `packed16` again (16 rows per step, 189 steps): separates the step-count effect from packing itself (it was stopped at step 5) | GPU, about 10-15 min (estimate) | §45.5 |
| 2.3 | [ ] The step-count ablation: runs 1 (padding, 156 steps), 2a (pack per step, 64 chats), 2b (pack once + 61,075-trained-token budget per step), 2c (B=4 x 20 rows, 151 steps), 3 (64 rows, 47 steps), 4 (= 3 with a lower LR); two seeds each; compare answer-token val loss | GPU; 2a and 2b need a loader/trainer change (a step counted in chats or tokens, not rows) | §44.1, packing doc §9, memory "packing cuts optimizer steps" |
| 2.4 | [ ] Special tokens decision with the DPO team (D1/D2): recommendation = special ids 50257 `<\|user\|>`, 50258 `<\|assistant\|>`, 50259 `<\|system\|>` for SFT AND DPO, the 3 rows initialised to the mean of the trained rows 0-50256, end token stays 50256, message content always encoded as plain text (`encode_ordinary`) | meeting with the DPO team | §45.6, §45.3 (D1/D2) |
| 2.5 | [ ] Before 2.4: measure the current norms of rows 50257-50259 in the checkpoint; ask the DPO team for their `sft_tokenizer.py` and which SFT checkpoint their DPO starts from; their script encodes content with `allowed_special` (`prepare_dpo_data_tokenize.py:119`): must change to plain text in either option | CPU + the DPO team | §45.3, §45.6 |
| 2.6 | [ ] Tokens ablation: plain-text markers vs special ids, same packed data and steps; compare answer-token val loss now, stopping and role leaks once generation exists | GPU; a `prepare_sft_data.py` option for the special ids + the row initialisation in the trainer | §45.6 |
| 2.7 | [ ] Pretraining step 0 with today's library: Vj2 is now the default backward for pretraining too, so compare a pretraining step 0 with `USE_VJ2=0` (Vj1d, the old default) against the default (Vj2): the loss must be equal, the grad norm close (different rounding order) | GPU, short | §45.3 phase 4; "Vj2 vs Vj1d pretraining comparison" (§45.5 later list) |

#### 45.8.3 Later tests of the attention kernels (listed, not run, by decision)

| # | test | how | why |
| --- | --- | --- | --- |
| 3.1 | [ ] G11 on Vj1d's design A | `USE_VJ2=0 VARLEN_QUICK=1 VARLEN_ONLY=G11 ./tva_a` | Vj1d's A code is unchanged (SASS identical) but has not run with today's library |
| 3.2 | [ ] G11 refusals in the other modes | `USE_FLASH=1` and `USE_DETERMINISTIC_DQ=1`, each with `VARLEN_ONLY=G11` (seconds) | the refusal must hold where packing is not supported |
| 3.3 | [ ] the full (not quick) packed groups under Vj2: G10, G11, G12 (other row lengths, a 16,384-token chat, 4,096 one-token chats), G13 (large inputs, 60 rows in one call) | `./tva_a` with no `VARLEN_ONLY` runs everything (long: tell the time first) | the quick run covers one head setting and 16 real rows |
| 3.4 | [ ] compute-sanitizer memcheck + racecheck on Vj2 design A | on the G11 quick layouts | new memory code: padded slots, plain-store epilogue |
| 3.5 | [ ] speed of design A on Vj2: `VARLEN_ONLY=BENCH` (dense vs A, forward + backward, real and uniform rows), then against FA3 varlen as on 10-12 | GPU | A on Vj2 must beat A on Vj1d; the 10-12 numbers were Vj1d |
| 3.6 | [ ] Vj2 vs Vj1d in a pretraining run (speed + loss curve) | GPU, longer | Vj2 became the default on 10-15 |
| 3.7 | [ ] full CLASSIC (605 checks) on the current Vj2 | `VARLEN_ONLY=CLASSIC` without `VARLEN_QUICK` | the last full 605 was on snapshot f5 (§43); later changes ran 144-check quick runs |
| 3.8 | [ ] CLASSIC error values vs b1 (optional) | diff of the error columns | proves identical numbers, not only PASS |
| 3.9 | [ ] B-1 test 2: speed race g4 vs b1 | `gpu_runs/b1_1014/t2_speed.sh`, about 4 min | pending since 10-14 (all GPUs were taken) |

#### 45.8.4 Paused kernel optimisation (after the pipeline; goal: beat FA3 / FA4 / cuDNN on H200, §43)

- [ ] Forward v62 (1.21x FA3 on B8 18/6): G1 for P.V (`m64n128k16`, V descriptor LBO 16,384 B), G5 tail (wave
  quantisation: 2,304 units / 132 CTAs = 17.45 rounds), G6 the output through shared memory + TMA store (§43.12).
- [ ] Vj2 design A speed (the old G7 / Phase 2 item, now on A only): profile packed rows, per-kernel time.
- [ ] Packed-row tile waste (§45.9): design A throws away 11.5% of its computed pairs (7.5% diagonal tiles, 3.8% empty
  slots at chat ends, 0.2% pad tails). Ideas: smaller tiles or one shared masked tile for the ends of several chats;
  skip the pad-tail segments (write zeros; their outputs feed no loss).
- [ ] Final scoreboard against FA3, FA4, cuDNN (dense and packed).
- [ ] Head-dim-64 bug (§42.8), and FA4's Blackwell methods for our B300 kernels (§43.12 item 4).

#### 45.8.5 Future tasks (decided, not now)

| task | source |
| --- | --- |
| [ ] **Deterministic dQ for Vj2** (dense and packed; decided 2026-10-16: not now, only if bit-identical runs are needed). Today O/LSE (v62) and dK/dV (Vj2: one writer per query head + fixed-order head sum) are bit-identical run to run; only dQ varies (atomic adds of up to 32 key blocks per dQ tile; max 4.9e-4 vs FA's tolerance 1.6e-2 in the 10-16 repeat test). Plan: ordered adds, one counter per dQ tile, fixed order = highest key tile first (the order the blocks reach a tile anyway in our LPT sections, so little waiting expected), done by the writer warp; a switch like `USE_DETERMINISTIC_DQ`. The other two ways cost more: v44's per-block dQ copies + fixed-order sum (up to 32 copies of dQ), or a separate dQ pass (about 7 GEMMs per step instead of 5). Reference cost: FA3's deterministic mode = 1.22x FA3's backward time on 10-16, but it also has to order dK/dV, because FA3's normal GQA backward adds dK/dV with atomicAdd into fp32 buffers (`hopper/epilogue_bwd.hpp:323, 458-491, 480`; `flash_bwd_launch_template.h:92-94, 138-153`, FlashAttention 94e22c9) | chat 2026-10-16, §45.12 |
| [ ] Split over-long chats like TRL `bfd_split` instead of dropping them (10 of 5,010 on stage-0), with the drop-vs-split ablation | §45.4, packing doc §7.7 |
| [ ] DFT and focal loss vs plain cross-entropy (a change to the loss op only) | memory "DFT + focal loss" (10-06) |
| [ ] The remaining SFT Q&A questions: Q4, Q5, Q6, Q11 (logits only at trained positions: a gather + an lm_head GEMM with irregular M; BluBLAS gives wrong numbers silently for a transposed operand when M/N is not a multiple of 64, so that needs a guard first), Q12, Q13 | §38, memory "SFT plan = the 13 Q&A questions" |
| [ ] Kruthik's fused embedding backward (`EmbeddingBwdFused`, the team's mandated kernel) is not used by `bluscript_sft.cpp`; with packing the tokens per micro-batch stay B x T, so it would work without its N fix: ask the user whether to wire it in | memory "v2 packing + Kruthik fused emb" |
| [ ] B300: the sm103 loss forward has no `ignore_index` (masked rows are computed as trained in the loss value); fix before any SFT run on B300 | memory "B300 loss fwd ignores ignore_index" |
| [ ] Casting follow-ups: #1 bf16 `wte` (parked, keep fp32), #3 `data<T>()` is an unchecked reinterpret (827 sites), #4 the unexplained ~1 ms of the bf16-logits change | memory "4 immediate tasks after SFT" |
| [ ] Shared-memory race warnings in TI's `matmul_bf16_optimized` (20 racecheck warnings) | memory "racecheck warnings in TI matmul_bf16_optimized" |
| [ ] Re-explain `ignore_index` clearly (why -100 cannot work with uint16 targets; why the default is -1) | memory, parked by the user 2026-09-21 |

#### 45.8.6 Housekeeping

- [ ] **Commit + push (the user does it). Nothing has been committed in either repo since 2026-09-28** (TI `d96b991`,
  BluTrain_sft `909ab75`, both on `dist_side_merge`): all the SFT work since then exists only as uncommitted files on
  the laptop (and partly on the box). `git status` on 2026-10-15 evening:
  - Tensor-Implementations, modified: `AttentionBackward.h/.cpp`, `AttentionOps.h/.cpp`, `AttentionKernels.h`,
    `NormalizationBackward.h/.cpp`, `NormalizationOps.h/.cpp`, `UnaryOps/Normalizations.h` + `cpu/Normalizations.cpp`,
    `FusedRMSRoPEKernels.h`, the 4 `FusedRMSRoPE_{fwd,bwd}_sm90{,_kernel}.cu`, `GQA_bwd_generic.cu`,
    `GQA_bwd_vj1_sm90.cu`, `GQA_bwd_vj1d_sm90.cu`, `GQA_fwd_generic.cu`, `GQA_fwd_sm90.cu`; new (untracked):
    `GQA_bwd_vj2_sm90.cu`, `Tests/kernel_tests/{sanitize_vj2,test_rope_positions,test_varlen_attention}.cu`.
  - BluTrain_sft, modified: `bluscript_zero_ckpt.cpp` (+235 / -24 lines, the SFT loader integration of 10-06/07) and the
    Tensor-Implementations submodule pointer; new (untracked): `bluscript_sft.cpp`, `sft/` (`prepare_sft_data.py`,
    `SftDataLoader.h`, `test_sft_dataloader.cpp`; it still compiles with today's loader), and
    `bluscript_zero_ckpt.cpp.orig_init` (a backup: decide whether to keep it out of the commit).
  - This repo (`Neural_Networks_exp_1926`): the work docs and records.
  The design-B version was never committed, so a commit now holds design A only. Ask me for the push block.
- [ ] Any other binary in `~/gautam_k/sft_v2/BluTrain_sft` that uses the attention op (for example a pretraining
  trainer built there) must be rebuilt before use: the op's signature changed on 10-15. `bluscript_sft_bin` is rebuilt.
- Old-format packed data (`seg_start.bin`) is gone from the box and from the laptop's data folder; nothing needs it
  (`seq_lengths.bin` reproduces it exactly, checked).

### 45.9 Packed rows and attention tiles: how much computed work is thrown away (2026-10-15, question from the user)

**The question.** Attention kernels work in tiles (blocks of queries x keys). With packed rows and a per-chat causal
mask, part of some tiles is masked: those query-key pairs are computed and then thrown away. How much, and does it hurt?

**Where the waste can come from.** The tensor cores always compute a whole tile; masked pairs are computed and then
given weight exactly 0 (forward: score set to -inf before the softmax, so its exp is 0; backward: P = 0, so dS = 0).
That is why masking cannot change the result, only the time. The waste depends on where the tile grid sits:

- **Grid fixed to the row (design B, FlexAttention's document mask, the user's picture):** a tile can hold the end of
  one chat and the start of the next. Example, a 16-token row with chats a (6), b (7), c (3) and 4-token tiles:

  ```
  tokens:  0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
  chat:    a a a a | a a b b | b b b  b  | b  c  c  c
                     ^^^^^^^ one tile holds the end of chat a AND the start of chat b
  ```

  Such a tile is masked for both reasons (causal, and other chat), and the key loop of chat b's rows must also visit
  chat a's tiles that share a tile with b.
- **Grid restarted at every chat (design A, ours; FlashAttention varlen does the same):** no tile ever holds two chats:

  ```
  chat a (6):   [0 1 2 3]  [4 5 . .]        . = slot past the chat end (computed, never used)
  chat b (7):   [6 7 8 9]  [10 11 12 .]
  chat c (3):   [13 14 15 .]
  ```

  What is left: the causal triangle inside each chat's diagonal tiles (the same as any causal attention), the empty
  slots of each chat's last tile, and the row's pad tail (processed as a segment of its own).

**In our code** (laptop `~/Downloads/BluTrain_sft`, `Tensor-Implementations/src/Kernels/cuda/attention/arch/`):
forward v62 tiles are 64 queries x 128 keys (`GQA_fwd_sm90.cu:47`); in design A a work unit reads its chat's start and
length (`:476-477`), visits key tiles 0 .. the diagonal tile of its own chat only (`:488`), skips a 64-row query tile
that starts past the chat end (`:491`), masks only the diagonal tile (`:535`, `:554`), and never stores rows past the
end (`:657`). Backward Vj2 blocks are 128 keys x 64-query steps (`GQA_bwd_vj2_sm90.cu:68-69`); a block takes its chat
from the plan (`:452`), visits only that chat's query steps (`:468`), masks the diagonal steps (`:604`, `:613`), gets
LSE = +inf for slot rows past the chat end so their P is 0 (`:409`), and stores dK/dV only for rows inside the chat
(`:703-714`). The plan gives each chat its own work items (`GQA_fwd_generic.cu:455, 475, 494`).

**Counted on the stage-0 packed data** (1,517 rows, 5,000 chats, 478 pad tails), following the kernels' tile rules
exactly; per head; real = key <= query in the same chat (script and output:
`profiling_reports/tile_waste_2026_10_15/`):

| layout | forward pairs computed | x real | backward pairs computed | x real |
| --- | --- | --- | --- | --- |
| v1 (one chat per padded 4096-token row) | 43,253,760,000 | 9.15 | 43,253,760,000 | 9.15 |
| packed, no chat mask (wrong: chats see each other) | 13,123,190,784 | 2.78 | 13,123,190,784 | 2.78 |
| packed, design B (row grid; removed 10-15) | 5,665,947,648 | 1.20 | 5,512,896,512 | 1.17 |
| **packed, design A (grid per chat; ours)** | **5,345,517,568** | **1.13** | **5,345,517,568** | **1.13** |

Real pairs: 4,728,665,163. Design A's forward throws away 11.54% of what it computes: 7.48% inside the diagonal tiles
(key after the query, including keys past a chat's end), 3.83% in query rows past a chat's end in its last 64-row
tile, 0.23% in the pad-tail segments. For comparison, one causal sequence of 4,096 tokens with the same tiles throws
away 3.01%: packed rows have more diagonal tiles (one set per chat) and one partly filled last tile per chat.

**Does it hurt?** Not the result (the thrown-away pairs get weight exactly 0; G10/G11 check every chat bit for bit
against the same chat run alone). For speed: design A computes 8.1x less attention work than v1 and about 6% less
forward / 3% less backward than design B, which matches the measured A-vs-B gain (forward 2-16%, backward 3-5% on
middle and long chats, 2026-10-10/12). Design B wins only when chats are much shorter than a tile (several tiny chats
fit in one tile there, while A gives each chat at least one whole tile); our chats average 1,238 tokens.

**What could still be cut (not now; listed in §45.8.4):** the diagonal-tile waste is the same as in FlashAttention
(tile-based causal attention computes the whole diagonal tile); the empty slots at chat ends (3.8%) could be reduced by
giving chat ends smaller tiles or by putting the ends of several chats into one masked tile; the pad tails (0.2%)
could be skipped by writing zeros instead (their outputs feed no loss).

### 45.10 Tomorrow morning FIRST (2026-10-16): learn how attention blocks compute masked work, segstart vs seqlens

**Decided by the user on 2026-10-15 evening:** this is the first task of 2026-10-16 morning, before anything else. It
continues the question the user asked with a picture before the EOD (a packed row with 4 documents d0-d3 cut into
fixed square blocks: inside each block some pairs are real work and the rest is masked; "won't that waste affect us?").
The explanations below were given in chat on 10-15 and were NOT yet clear to the user, so tomorrow they are explained
again from zero, step by step, with a small drawn grid first (8-token blocks, every pair drawn), and only then with the
real block sizes and the numbers. The 10-15 EOD stays as it was sent (the user decided not to change it).

**The facts to explain (all counted with the kernels' own block rules; scripts in `profiling_reports/tile_waste_2026_10_15/`):**

1. **What a block is.** The GPU computes attention in fixed blocks: in our forward kernel one block is 64 queries x
   128 keys = 8,192 pairs; the backward uses 128 keys x 64-query steps. A block is always computed whole; pairs that
   must not count (a key after the query, or a key of another chat) are computed and then set to weight 0.
2. **The rule behind all the waste.** A query must look at every key of its own chat, from the chat's first token up to
   itself. So the work of one query block is a whole line of key blocks, from the key block where its chat starts to
   the key block holding the query itself. Even a query block in the middle of a long chat reaches back to the
   chat's first token.
3. **segstart (design B, removed):** the blocks sit at fixed places in the row (multiples of 64 / 128 tokens). A chat
   almost never starts exactly at a block edge, so its first key block is shared with the previous chat, and EVERY
   query block of the chat (start, middle and end) must go through that shared, mostly masked block. Example (one
   4096-token row: chat 1 = tokens 0-999, chat 2 = 1000-3999, padding 4000-4095): the middle query block of chat 2
   (rows 2048-2111) needs key blocks 896-1023 ... 2048-2175 = 10 blocks, and the first one holds 104 keys of chat 1
   (masked) and 24 of chat 2. The padding rows share a block pair with chat 2's last rows, so they also go through all
   of chat 2's key blocks (25 blocks instead of 1). Whole row: segstart 720 blocks.
4. **seqlens (design A, ours):** every chat's blocks start at its own first token. The same middle query block needs
   chat-2 keys 1000-2151 = 9 blocks, all of chat 2. The only extra is the empty part of each chat's last block.
   Whole row: seqlens 650 blocks (segstart computes 10.8% more blocks than seqlens on this row).
5. **Why the gap depends on chat length** (rows filled with chats of one length, forward blocks):

   | chat length | seqlens blocks | segstart blocks | segstart vs seqlens |
   | --- | --- | --- | --- |
   | 2000 | 546 | 576 | +5.5% |
   | 1000 | 290 | 336 | +16% |
   | 300 | 123 | 168 | +37% |
   | 100 | 82 | 124 | +51% |
   | 64 | 64 | 64 | equal |
   | 20 | 205 | 114 | -44% (segstart wins) |

   segstart's extra is about one block per query block of a chat, while a chat's total work grows with its length
   squared, so the extra is a small part of a long chat and a big part of a middle one.
6. **Why segstart wins on very short chats.** (a) A chat shorter than a block still gets a whole block with seqlens (a
   20-token chat fills 20 of its 64 query rows), while segstart puts about 3 such chats into one block. (b) Measured on
   2026-10-12 (§40.12, real rows with the shortest chats, backward): seqlens computed slightly LESS work than segstart
   (14.0% vs 15.1% of a dense row) but took MORE time (28.2% vs 27.4% of the dense time), because it splits such rows
   into many more small work pieces and every piece has a fixed cost (loading its keys, starting up, writing results)
   however little real work it holds. That is the "segstart is 2-4% faster than seqlens only on very short chats".
7. **The correct one-line summary** (the 10-15 EOD sentence put a reason next to the wrong fact): masked work at chat
   borders is why seqlens is faster than segstart on middle and long chats; whole blocks for tiny chats and the
   per-piece fixed cost are why segstart is faster than seqlens on very short chats. Our stage-0 chats average 1,238
   tokens, so seqlens is the right choice (1.13x the real work against segstart's 1.20x on the real packed data, §45.9).

**Plan for the morning session:** (1) draw a 16-token row with 4-token blocks and two chats, count every pair by hand;
(2) the same with the chat start moved off a block edge, to see the shared block appear for every query block;
(3) a tiny-chat row, to see why segstart wins there; (4) only then the real sizes, the tables above and §45.9;
(5) then the GPU tests of §45.8.1 when a GPU is free.

### 45.11 2026-10-16: design A (seqlens) proven on the GPU (2x H200 box)

**Where:** the 2-node box was busy, so the work moved to a 2x H200 box (`root@86.38.238.80`, Ubuntu 24.04, CUDA 13.0).
Set up: Open MPI 4.1.6 installed (`openmpi-bin libopenmpi-dev zlib1g-dev`; the same apt line upgraded the box's NCCL
2.28.9 -> 2.32.3); the laptop's public key added to the box's `authorized_keys`; the laptop master copied with rsync
(252 MB code incl. git history, the laptop's sm_86 build left out; 132 MB SFT data), no clone and no push; the step-8000
checkpoint downloaded from HF `navingv/curriculum_8btokens` with curl (`metadata.bin` 129,127 B md5 5119b20d,
`data_rank_0.bin` 6,469,415,424 B). Build from scratch about 10 min: library, trainer and `tva_a` all exit 0, the 3
checked files have the laptop's md5 (7667deec / 83c88d41 / 204587bd).

**Test 1, `test_varlen_attention` G11 quick under Vj2 (GPU 0): 94 of 94 checks, 0 failed, 12 s.** Design A's
backward on Vj2 matches the fp64 reference on every layout; every segment equals the same segment run alone (O, LSE,
dK, dV bit for bit, dQ within FlashAttention's tolerance); isolation holds. (The 10-15 Vj1d run had 95 checks: the
"plan + seg_start" refusal no longer exists.) Repeat x50 on real rows 0-7: O/LSE/dK/dV mismatched elements 0, dQ max
difference 4.883e-4 against the tolerance 1.562e-2. Backward time, same run (8 rows x 4096, 18/6 heads, free clocks):

| real rows | work vs dense (64-key tile visits) | dense backward | design A backward | A time vs dense |
| --- | --- | --- | --- | --- |
| 0-7 (longest chats) | 92.6% | 3.0556 ms | 2.9010 ms | 94.9% |
| 752-759 (middle) | 36.1% | 2.8978 ms | 1.4740 ms | 50.9% |
| 1504-1511 (shortest) | 14.0% | 2.9389 ms | 1.0225 ms | 34.8% |

On 2026-10-12 (another box and GPU, so only a rough comparison) Vj1d's design A took 3.895 / 1.898 / 1.162 ms on the
same rows (§40.12), so Vj2's design A is about 26% / 22% / 12% faster than Vj1d's. The time falls more slowly than the
work (middle rows 36% of the work, 51% of the time): fixed costs per block and the partly filled last blocks; this is
the packed-speed item of §45.8.4.

**Test 2, the same-chats end-to-end run, packed, design A (GPU 1):** the same 500 chats as §45.5 as 147 packed rows,
B = 1, one optimizer step over all rows, from step 8000:

| | v1 (500 rows, 10-15) | packed, design B (10-15) | **packed, design A (10-16)** |
| --- | --- | --- | --- |
| step 0 val loss | 1.445038 | 1.445084 | **1.445038** |
| step 0 grad norm | 0.7935 | 0.7935 | **0.7935** (groups 0.7929 / 0.0321) |
| step 0 time | 53.6 s | 21.2 s | 13.6 s (another box; 44,396 tok/s, mfu 49.4%) |

With design A the step-0 loss equals v1's to all 7 printed digits, where design B differed in the 5th: A computes each
chat's attention with the same blocks, in the same order, as the chat's own v1 row (each chat starts its own blocks at
its first token, like the v1 row starts at token 0), so the numbers are the same; B cut the blocks on the row grid,
which changes the rounding a little. The process exited by itself after "Training complete" (exit 0).

Steps 1 and 2 of the same runs:

| | v1 (500 rows) | packed, design B | packed, design A |
| --- | --- | --- | --- |
| step 1 train loss / grad norm | 1.423608 / 0.6496 | 1.423671 / 0.6502 | 1.423602 / 0.6498 |
| val loss after step 2 | 1.381130 | 1.381213 | 1.381214 |
| time per step (steps 0 / 1 / 2) | 53.6 / 49.0 / 43.9 s | 21.2 / 18.3 / 13.3 s | 13.6 / 13.0 / 12.9 s (other box) |

Why step 0 is exact and steps 1-2 are only close: step 0's loss is a forward pass on the starting weights, and every
token's forward numbers are the same in v1 and design A. From step 1 on, the weights have been updated with a gradient
that is summed differently: v1 adds up 500 micro-batches of 1 row, the packed runs 147 micro-batches, and each weight
gradient (inputs x output gradients) sums over the tokens of a micro-batch, so the fp32 roundings of those sums differ
(and dQ is summed with atomics, in a run-to-run order). So after an update the losses differ in the 5th-6th digit.
Design A and design B give the same val loss after step 2 (1.381214 vs 1.381213) because they share that micro-batch
structure, while v1 differs by 0.006%: the difference comes from how the gradient is grouped, not from the attention
design. Logs on the laptop: `profiling_reports/sft_seqlens_gpu_tests_2026_10_16/` (G11 log, run.log, sft_adamw.csv).

**Result: the packed SFT pipeline with design A is proven on the GPU.** Logs on the box: `~/gautam_k/sft_v2/runs/`
(`G11_quick_vj2_A.log`, `e2e_packed_A/run.log`).

### 45.12 2026-10-16: scoreboard on the 2x H200 box, backward first (dense and packed), Vj1d added as a rival

**Decision (user, 10-16):** optimise the attention kernels of the finished SFT pipeline, backward (Vj2) first, dense
("normal") and packed (per-chat mask), then the forward (v62). Rivals: our old Vj1d, FA3 (+ deterministic), FA4, cuDNN,
FA2, Flex (packed); xFormers to be added. The learning session of §45.10 is moved to later.

**Setup:** the rivals' environment was copied from the old box to the same paths (`/home/operator/envv`: torch 2.14.1+cu130,
cuDNN 9.24; `/home/operator/gautam_k/benv`: FA3 built from FlashAttention `94e22c9`, FA4 from the same commit; Python
3.12.3 on both boxes; 6.2 GB, about 11 min). Scripts `sb_dense.sh` / `sb_packed.sh` (records
`profiling_reports/bwd_scoreboard_2026_10_16/`), GPU 1 (GPU 0 was a colleague's training run), free clocks, 12:19-12:21 IST (the 2x H200 box's own clock is UTC,
5:30 behind; all times in these notes are IST).
Ours ran twice, before and after the libraries; the two rounds differ by 0.98% (median), 3.9% (max). All exits 0; the
packed row sets are identical to 10-12; 14 of 14 correctness gates pass, the packed libraries agree with FA3 to bf16
noise. Full tables: `tables_1016.md` (made by `tables.py`).

**Backward, time ratio ours Vj2 / rival (below 1 = ours faster), mean over the shapes (ours faster on):**

| | Vj1d | FA3 | FA3 det | FA4 | cuDNN | FA2 | Flex |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense, 13 shapes | 0.687 (13/13) | 0.986 (7/13) | 0.805 (13/13) | 0.975 (6/13) | 0.785 (13/13) | 0.444 (13/13) | - |
| packed (seqlens), 7 row sets | 0.783 (7/7) | 0.939 (6/7) | 0.805 (7/7) | 0.967 (4/7) | 0.878 (7/7) | 0.504 (7/7) | 0.694 (7/7) |

Our model's shapes: dense B8 18/6 backward 3.072 ms vs FA3 3.073, FA4 3.042, cuDNN 3.632; packed typical rows
(`real_rows_752`) 1.511 ms vs FA3 varlen 1.560, FA4 1.539, cuDNN 1.730, Vj1d 1.983. Where we lose: dense B4 24/8 (1.096x
FA3), B8 32/8 (1.031x FA3, 1.041x FA4); packed long chats (uniform_4096 1.019x FA3, real_rows_0-7 1.018x FA4).

**Forward (v62, unchanged), same runs:** dense 1.135x FA3 (B8 18/6 1.233x), 1.034x FA4, 0.984x cuDNN; packed 1.059x FA3
varlen, 0.932x FA4, 0.975x cuDNN. The forward is the bigger gap (the known G1/G5/G6 items).

**Two things to check next:** (1) the 10-14 profiler had our backward main kernel 3.0% faster than FA3's (fixed clocks), but at
free clocks the whole call is equal at B8 18/6: either the helper kernels or the 700 W power cap (a faster kernel draws
more power and the clock drops) eats it; to measure with per-kernel times plus clock and power. (2) The same forward kernel
measured 5% slower in the Vj2 runs than in the Vj1d runs (mean 1.050): the forward is timed right after the backward, so
the faster, more power-hungry Vj2 backward lowers the clock for the next forward; forward speed work must time the forward
on its own.

**Box rule seen 10-16:** the box now has a GPU-sharing board (`/root/gpu_board/gpuboard status | claim <gpu> --note .. |
release <gpu>`, rules `/root/gpu_board/PROTOCOL.md`): claim before every run, never start on a refused claim, never kill
others' processes.

### 45.13 2026-10-16: the backward profiled kernel by kernel (ours vs FA3 / FA4 / cuDNN), and fix 1 (the causal mask ran on every step)

**What ran:** `prof_bwd.sh` (records `profiling_reports/bwd_scoreboard_2026_10_16/`, data in `prof_1016/`), GPU 1, 13:04-13:08 IST,
plus our two ncu steps again at 13:44-13:45 IST. Part A: nsys (`--trace=cuda`) at free clocks, with an nvidia-smi clock/power
log every 100 ms. Part B: ncu `--set full` (ncu locks the clock at 1.36 GHz). Shapes: our model, dense B8 18/6 T4096, and
packed typical rows `real_rows_752` (seqlens plan). Analysis scripts in the same folder: `bwd_breakdown.py` (one backward
call kernel by kernel), `pipe_mix.py` (instructions per pipe), `sass_mix.py` (instructions per SASS opcode), `seq_names.py`.

Three run problems, all fixed: (1) an output paste into the box shell started a second copy at 12:59 IST; both were killed,
the clean run at 13:04 IST overwrote every file. (2) Our ncu steps first profiled nothing: the filter was
`gqa_bwd_vj2_kernel<false>`, but ncu writes the template flag as `gqa_bwd_vj2_kernel<(bool)0>`; filter fixed to
`<.bool.0>` / `<.bool.1>`, and `prof_bwd.sh <gpu> ncu_ours` reruns only those two steps. (3) The library dense run also timed
FA3-deterministic and PyTorch's flash path (FA2): `BENCH_ONLY` keeps names by prefix, and `fa3` also keeps `fa3_det`.
Harmless: the kernels are told apart by name. Pulling: only the CSVs are needed (the `.sqlite`, `.nsys-rep`, `.ncu-rep` files
and the libraries' per-instruction CSVs stay on the box: `rsync --exclude`), the link to the box is slow (~500 KB/min).

**1. One backward call, kernel by kernel (nsys, free clocks, median over the steady calls, first 10 calls skipped, µs).**
"Helpers" = every kernel of the call except the main one; "gaps" = idle time between them.

| dense B8 18/6 | whole call | main kernel | helpers | gaps | what the helpers are |
| --- | --- | --- | --- | --- | --- |
| ours (Vj2) | 3007 | 2633 | 369 | 5 | prep 157 (D = rowsum(dO*O), LSE copy, dQ zero), dQ convert 105, dK + dV head sums 107 |
| FA3 | 2933 | 2551 | 374 | 8 | zero-fills 45, prep 150, dQ convert 105, dK + dV converts 75 |
| FA3 deterministic | 3592 | 3206 | 375 | 11 | same as FA3 |
| FA4 | 2905 | 2524 | 379 | 3 | zero-fills 45, prep 152, dQ + dK + dV post 182 |
| cuDNN | 3308 | 2903 | 366 | 39 | prep 150, dQ convert 117, dK + dV head sums 99 |
| FA2 (PyTorch flash) | 6517 | 5483 | 1027 | 7 | K/V repeated per query head 431, prep 156, dQ convert 144, dK/dV head sums 296 |

| packed real_rows_752 | whole call | main kernel | helpers | gaps | what the helpers are |
| --- | --- | --- | --- | --- | --- |
| ours (Vj2, plan A) | 1495 | 1103 | 387 | 5 | prep 174, dQ convert 106, dK + dV head sums 107 |
| FA3 varlen | 1495 | 1101 | 386 | 8 | zero-fills 50, prep 154, dQ convert 106, dK + dV converts 76 |
| FA3 varlen deterministic | 1790 | 1393 | 387 | 10 | same |
| FA4 varlen | 1484 | 1085 | 394 | 4 | zero-fills 49, prep 156, post 189 |
| cuDNN varlen | 1583 | 1174 | 392 | 16 | prep 156, dQ convert 116, dK + dV head sums 117 |

**Answer to question (1) of §45.12:** the helper kernels are NOT the problem (ours 369-387 µs, the others 366-394 µs).
The whole difference is in the main kernel.

**2. At the same clock our main kernel is the fastest (ncu, clock locked at 1.36 GHz):**

| | ours | FA3 | FA4 | cuDNN | FA3 det |
| --- | --- | --- | --- | --- | --- |
| dense: cycles (millions) | **4.14** | 4.25 | 4.29 | 4.93 | 5.18 |
| dense: instructions issued (millions, warp level) | 705 | 626 | **555** | 809 | 634 |
| packed: cycles (millions) | **1.73** | 1.87 | 1.88 | 2.05 | 2.31 |
| packed: instructions issued (millions) | 275 | 248 | **225** | 358 | 253 |

All use 168 registers per thread at launch and about 199-216 KB of shared memory per block; ours, FA3 and FA4 use the same
grid (4608 blocks dense: 32 key tiles x 18 heads x 8; cuDNN is persistent, 132 blocks).

**3. At free clocks we lose the lead, because our kernel runs at a lower clock.** Main kernel, steady state: dense ours 2633 µs
vs FA3 2551, FA4 2524; packed ours 1103 vs FA3 1101, FA4 1085. Over one run our main kernel took 2393-2404 µs for the
first ~12 calls and then stepped up to ~2630 µs (the clock dropped); FA3 / FA4 ran 2460 → ~2530-2560 µs. Cycles (ncu) divided
by free-clock time gives the clock each kernel ran at, roughly: ours ~1575 MHz, FA3 ~1665, FA4 ~1700, cuDNN ~1700 (dense);
packed ours ~1570, FA3 ~1695, FA4 ~1730. This is an estimate: the memory part of a kernel does not scale with the SM clock,
so cycles at 1.36 GHz are not exactly cycles at 1.7 GHz; all four kernels have a similar memory profile (DRAM 7-10% of
peak, L2 hit 90-92% dense), so the comparison holds. The nvidia-smi log could not show it: our runs last under one second
(only 2 busy samples), and the power reading is an average over a window (H200: about 4 s, see the 4 s power-window rule).
**Why the clock is lower:** we execute more instructions (705M vs 626M / 555M). Every instruction costs energy; under the
700 W power cap more energy per second means a lower clock. **Answer to question (1) of §45.12: it is the power cap.**

**4. Where our extra instructions are (dense; pipe = the unit that executes them; work relative to ours = 1.00):**

| pipe | what runs there | ours % of peak | FA3 | FA4 | cuDNN |
| --- | --- | --- | --- | --- | --- |
| alu | compare, select, logic (ISETP, FSEL, LOP3) | 15.3% | 0.83 | 0.78 | 1.36 |
| fma | float multiply / add, integer multiply-add (FMUL, FFMA, FADD, IMAD) | 7.7% | 0.78 | 0.77 | 1.75 |
| uniform | one value per warp: addresses, loop counters, tensor-core descriptors | 12.0% | 0.86 | 0.89 | 0.50 |
| xu | exp (MUFU.EX2) | 14.5% | 0.98 | 0.98 | 0.98 |
| lsu | shared-memory loads / stores | 7.6% | 0.98 | 1.00 | 1.10 |
| tensor (gmma) | the matrix multiplies | 7.1% | 1.00 | 1.00 | 0.88 |

So the exp, the loads and the matrix multiplies are the same for everyone; we do 20-28% more ALU work, 28-30% more FMA work
(packed: 37%) and 12-16% more uniform work. Per SASS opcode (ours, dense, 720.7M executed): UMOV 89.1M, UIADD3 74.2M,
FMUL 41.3M, F2FP (fp32 → bf16 pack) 41.3M, FFMA 38.9M, MUFU.EX2 38.9M, **FSEL 38.9M**, FADD 38.9M, LOP3 37.3M,
HGMMA 64x64 29.2M, ULOP3 29.0M, ISETP.GT 21.9M, IMAD.MOV 19.6M.

**FSEL = 38.9M = exactly one per exp.** FSEL is the select that zeroes P where key > query (the causal mask). The mask is
only needed on the first two query steps of each key block (the steps that hold the diagonal), but it ran on every step.
The source (`GQA_bwd_vj2_sm90.cu`, step loop) tested `if (diag)` inside the unrolled per-element loop; ptxas turned that
into predicated code (the compare result `P0` from the uniform flag `UP0` decides the outcome, but the instructions run
anyway): per element the key/query index math (LOP3, VIADD, IMAD.SHL), the compare (ISETP.GT) and the select (FSEL), on
every step. In the hot loop (545 instructions per warp per step, each executed 1,216,512 times = 4608 blocks x 8 compute
warps x 33 steps on average) the mask instructions are part of the 545.

**5. Fix 1 (laptop, 2026-10-16): the mask is its own block after the exp loop, behind one branch.**

    for (nt, e) S_[4nt+e] = exp2(S * scale2 - LSE2);            // every step, no mask code
    if (diag) for (nt, e) { key = ...; qry = ...; if (key > qry) S_[4nt+e] = 0; }   // only the 2 diagonal steps

`diag = (q0 < k0 + BN)` is the same as `i < 2` (q0 = (2kt + i)·64, k0 + BN = (2kt + 2)·64), in dense and packed mode (packed
indices are segment-relative). The values and their order are unchanged, so O / dK / dV should be bit-identical (dQ is
summed with atomic bulk adds, so its last bits can differ from run to run, as before).
Checked on the laptop (nvcc 13.0, `-gencode arch=compute_90a,code=sm_90a`, 2 s per build):
* SASS: `@P0 BRA` jumps over a block of 73 instructions (32 FSEL, 18 ISETP, 2 LOP3, 21 others) on the unmasked steps;
  168 registers, 0 bytes spilled (same as before); the other five backward kernels (prep, prep_chats, both dQ converts,
  head sum) are SASS-identical to before.
* Expected: 73 instructions x 1,142,784 unmasked warp-steps = **about 83M fewer, 705M → ~622M (FA3: 626M)**.
* First attempt (dropped): FA3's way, two loops (masked steps first, then unmasked) with their own copy of the step. ptxas
  spilled 32 registers (128 B) in the unmasked copy: it stored half of the dK accumulator to local memory and loaded it back
  on every step (with `#pragma unroll 1` too). The one-loop version keeps the old register layout.
* Source md5 1b76873c (was 7667deec). Box: old library kept as `~/gautam_k/libs/vj2_old/libtensor.so` (2464c5b8), new as
  `~/gautam_k/libs/vj2_maskblk/libtensor.so` (3b998bb6).

**Test (`profiling_reports/bwd_opt_2026_10_16/ab_mask_1016.sh`, GPU 1, started after 14:00 IST):** (1) the quick suite on the
new library (G2 dense, G3-G6, G10/G11 packed, G7); (2) a same-session race old vs new library, 12 rounds, the order
alternating (old-new, new-old, ...), dense B8 18/6 and packed `real_rows_752`, 1000 iterations per run (at least 4 s of GPU
work, so the clock settles under the power cap as in training); (3) ncu of the new main kernels. The script checks with `ldd`
that the test binary really loads the library of the folder it is pointed at (else the race compares a library with
itself), and stops if the quick suite fails.

**Result of run 1 (GPU 1, 14:06-14:13 IST, data `bwd_opt_2026_10_16/ab_mask_1016/`):**
* Quick suite on the new library: 296 checks passed, 0 failed (71 s).
* Instructions really dropped (ncu): dense 705M → 617M (-12.5%; FA3 626M), packed 275M → 240M (-12.7%; FA3 248M).
* But cycles at the locked clock did not drop: dense 4.144M → 4.158M (+0.3%), packed 1.730M → 1.754M (+1.4%). "Issue
  active" fell 12.8%, and the waiting grew by the same amount (stall per instruction: wait +56%, barrier +57%, gmma +9%).
  So the kernel is not limited by issuing instructions: the removed ones were sitting in time the warps spent waiting
  anyway (on the tensor cores, the barriers and the loads).
* Race, new vs old, per round (12 rounds, 1000 iterations per run, free clocks):

| | mean | new faster in | t | verdict |
| --- | --- | --- | --- | --- |
| dense backward | -0.08% | 7/12 | -1.2 | no change |
| packed backward | **+0.58%** | **0/12** | +7.8 | **slower in every round** |
| dense forward (same kernel in both libraries) | -0.40% | 12/12 | -7.1 | faster |
| packed forward (same kernel in both libraries) | -0.34% | 11/12 | -4.3 | faster |

* Why packed got slower: on the 2 diagonal steps the mask used to be mixed in between the 32 exps (it hid in their
  latency); now it runs after them, so those steps take longer. Dense rows have about 6% diagonal steps (2 of ~33 per key
  block); packed rows have many short chats, so about 30-40% of their steps are diagonal (a 512-token chat: 8 of 20 steps).
* The forwards ran faster although their kernel is the same: they are timed right after the backward, and the new backward
  issues 12.5% fewer instructions (less power), so the next forward gets a slightly higher clock. This is the only sign of
  the power effect; the backward itself did not get faster, so the "more instructions → lower clock" explanation of point 3
  is not proven by this run (the same as the GEMM work: instruction cuts converted at zero).
* Decision: not kept as it is (packed is the SFT path).

**Fix 1, version 3 (laptop, source md5 cc3a6125):** two copies of the exp loop behind one branch: the diagonal steps run
the original interleaved exp + mask code, all other steps run the plain exp (32 FFMA + 32 EX2, no mask code). Local SASS:
168 registers, 0 spills; the plain copy is 65 instructions; FSEL only in the masked copy. Expected: the instruction cut of
run 1 on the non-diagonal steps, and the diagonal steps as before (not certain: in the old code the compiler could also mix
the mask with the code around it; the branch now stops that). Test: the same script, `ab_mask_1016.sh <gpu> vj2_maskv3
ab_maskv3_1016` (old library vs version 3). Kept only if the quick suite passes and neither dense nor packed is slower.

**Next candidates after fix 1 (not started):** (a) UMOV + UIADD3 = 163M (23% of our instructions): building the tensor-core
descriptors and addresses per HGMMA; to compare with FA3 / FA4 we need their per-instruction CSVs
(`ncu_libs_dense.source.csv`, `ncu_libs_packed.source.csv`, left on the box). (b) FA4 does the same work in 555M
instructions: find what it skips. (c) After each cut, check the clock again: fewer instructions should raise the free clock.

**Result of run 2, version 3 (GPU 1, 14:19-14:26 IST, data `bwd_opt_2026_10_16/ab_maskv3_1016/`, stats by `race_stats.py`):**
quick suite 296 passed, 0 failed. Instructions: dense 705M → 618M (-12.4%), packed 275M → 240M (-12.6%). Cycles at the
locked clock: dense +0.5%, packed +1.1%.

| new vs old, per round (12 rounds) | run 1 (mask block after the exps) | run 2 (version 3, two exp copies) |
| --- | --- | --- |
| dense backward | -0.08% (7/12 faster, t -1.2) | +0.11% (3/12 faster, t +1.8) |
| packed backward | +0.58% (0/12, t +7.8) | +0.22% (1/12, t +4.8) |
| dense forward (same kernel) | -0.40% (12/12) | -0.30% (11/12) |
| packed forward (same kernel) | -0.34% (11/12) | -0.62% (12/12) |

What this shows:
* Even with the diagonal steps running the original code (version 3), the backward does not get faster; the other steps
  lose 12% of their instructions and the cycles stay the same (+0.5%). The two compute warpgroups then wait longer for
  each other at the named barriers (run 1: barrier stall per instruction +57%, so total barrier waiting about +37%). The
  backward is limited by its chain of tensor-core GEMMs, loads and barriers, not by issuing instructions.
* Per forward + backward pair (as in training, where the two alternate), version 3 is a wash: dense backward +3.4 µs vs
  forward -3.7 µs; packed backward +3.4 µs vs forward -3.6 µs. The forward gain is a power effect (fewer instructions in the
  backward leave more power for the next kernel); in training the next kernel after the attention backward is a GEMM, so
  the same effect would go to it, but that is not measured.
* So the clock explanation of point 3 ("we lose at free clocks because we execute more instructions") is NOT confirmed: with
  fewer instructions than FA3 our backward is still not faster at free clocks. Second explanation, not yet tested: our kernel
  spends more of its time waiting on memory (long_scoreboard 1.68 stall cycles per instruction); that waiting is fixed in
  nanoseconds, so it does not shrink when the clock goes up, and a kernel that is fastest at 1.36 GHz can lose at ~1.7 GHz
  without drawing more power. Test to tell the two apart: ours vs FA3 under ncu with `--clock-control none` (cycles at the real
  clock), or each one in a 5 s loop with an nvidia-smi log (clock and power per contender).

**Decision (user, 2026-10-16 ~14:40 IST):** keep version 3 of the backward fix (fewer instructions, neutral over a forward +
backward pair; box library `f3557691`); the backward is at FA3's level, move to the forward now, come back to the backward
later to beat the others.

### 45.14 2026-10-16: the forward (v62) — where the gap is, and the plan G1 → F1 → G5 → G6

**Kernel against kernel (nsys, 10-16 traces, free clocks):** dense B8 18/6 ours 1,202 µs vs FA3 874, FA4 967, cuDNN 1,035;
packed `real_rows_752` ours 554 µs vs FA3 432, FA4 491, cuDNN 457. (The scoreboard's 1.233x FA3 includes the libraries'
Python launch overhead; kernel against kernel it is 1.38x dense, 1.28x packed. These forwards were timed right after a
backward, so the clock is not the forward's own.)

**Same clock (ncu 10-13, records `v2_bwd_rework_research_2026_10_12/gpu_runs/g2_1013/ncu_g2_fwd.*` for ours after G2 and
`prof_1013/ncu_libs_fwd.*` for the libraries), dense B8 18/6:**

| | ours v62 | FA3 | FA4 | cuDNN |
| --- | --- | --- | --- | --- |
| time at the locked clock | 1,084 µs | 877 | 952 | 1,009 |
| instructions | 349M | 307M | 316M | 369M |
| tensor cores busy | **64.1%** | **82.9%** | 74.4% | 67.9% |
| SMs busy (10-13 table) | 95.7% | 99.6% | 99.0% | 96.5% |

Unlike the backward, the forward's gap is real kernel work at the same clock.

**A misleading number, corrected:** averaged over all warps, our forward shows 2.75 "long scoreboard" stall cycles per
instruction vs FA3's 0.30, and 33% of our stall samples sit on mbarrier wait lines. The barrier addresses
(`0x38018 + 8·stage` = `empty[stage]`, `0x38040/48` = `qempty[0/1]`) show these are the LOADER warps waiting for a free
buffer: our loader warpgroup runs its loop in all 4 warps and they idle there, which costs no compute time (FA3 lets 3 of
its 4 loader warps exit). Restricted to the compute warps' code, the stall mix of ours and FA3 is similar (wait 32% vs 26%,
barrier 16% vs 12%, long scoreboard 3.9% vs 3.8%).

**The real difference: the order of the steps in the loop** (ours `GQA_fwd_sm90.cu:527-622`, FA3
`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1170-1207`, IntraWGOverlap):

    OURS  (per key tile n, one compute group)        FA3  (per key tile n, one compute group)
    1. wait: QK(n) finished                          1. start QK(n) and P.V(n-1) on the tensor cores
    2. softmax(n): this group has NOTHING on         2. wait: QK(n) finished
       the tensor cores                              3. softmax(n), while P.V(n-1) is STILL RUNNING
    3. start P.V(n), then QK(n+1)                    4. wait: P.V(n-1) finished; rescale O
    4. wait: P.V(n) finished (QK(n+1) still runs)

Both use ping-pong between the two compute groups, and both start a GEMM early, but ours overlaps only QK(n+1) with the
wait for P.V(n) (a short window), while FA3 overlaps the whole softmax with P.V(n-1). Our earlier statement "our forward
already has FA3's structure" (from reading only the file's header comments) was too quick.

**Plan (one change at a time; each: local build + SASS / spill check, then one ~10 min GPU run = quick suite + 12-round race
against the current library + ncu):**
1. **G1:** P.V as 8 full-width `m64n128k16` per key tile instead of 16 half-width `m64n64k16` (FA3/FA4/cuDNN all use the full
   width; the same change in the backward on 10-13 was bit-identical). The tile does NOT change (64 queries x 128 keys per
   group, 128 x 128 per block, as FA3): only the instruction. Half as many P.V instructions, and the P fragments are read once.
2. **F1:** FA3's loop order (softmax(n) during P.V(n-1)); about 32 more registers per thread during the softmax (P(n-1)
   must stay live while its GEMM runs); FA3 fits it in 240 registers per thread, ours has 232: check spills first.
3. **G5:** the tail: 2,304 equal units over 132 SMs = 17.45 rounds, the last round 45% full; heaviest first (FA3: LPT).
4. **G6:** the output through shared memory + TMA store.

**G1 done on the laptop (`GQA_fwd_sm90.cu` md5 86da0e1f, was eff8cdb6):** new `wgmma_pv128_x8` (8 x
`m64n128k16`, A = P from registers, trans-b = 1) and `make_pv_desc` (the V tile's SW128 descriptor with LBO = 16,384 B, the
distance between its two head-half blocks of [128 keys][64]; the backward's `desc_lbo` uses 8,192 B because its tiles hold 64
rows). The m64n128 accumulator layout equals the two old halves side by side (o[0..31] = head cols 0-63, o[32..63] = 64-127),
so the softmax rescale and the epilogue are unchanged. Local SASS (dense and packed kernels): P.V = 8
`HGMMA.64x128x16 ... R152 (P from registers), gdesc.tnspB` per key tile (was 16 `HGMMA.64x64x16`), 168 registers, 0 spills,
WARPGROUP.ARRIVE / DEPBAR 3 / 3 as before (no per-wgmma sync), R2UR 81 → 65.
Test: `ab_mask_1016.sh <gpu> vj2_fwdg1 ab_fwdg1_1016 vj2_maskv3` (old = the current library with backward version 3).

**G1 result (GPU 1, 14:49-14:56 IST, data `bwd_opt_2026_10_16/ab_fwdg1_1016/`; old = library f3557691 (backward version 3),
new = adce5abb (+ forward G1)):** quick suite 296 passed, 0 failed.

| G1 vs current, per round (12 rounds) | mean | new faster in | t |
| --- | --- | --- | --- |
| **dense forward** (B8 18/6) | **-3.34%** (1.2398 → 1.1972 ms) | **12/12** | -61.8 |
| **packed forward** (real_rows_752) | **-3.22%** (0.5750 → 0.5564 ms) | **12/12** | -46.4 |
| dense backward (kernel unchanged) | +0.19% | 1/12 | +3.0 |
| packed backward (kernel unchanged) | +0.10% | 5/12 | +2.0 |

ncu dense forward (vs the 10-13 profile after G2): cycles 1.916M → 1.840M (-4.0%), instructions 349M → 326M, tensor
cores busy 64.1% → 66.6%. The unchanged backward got slightly slower: the same power effect as in §45.13 in reverse (the
forward is timed before the backward; a faster forward leaves the next kernel a little less power). Per forward + backward
pair: dense -41 + 6 = -35 µs, packed -18.6 + 0.1 = -18.5 µs. **Kept.** Box library with G1: `~/gautam_k/libs/vj2_fwdg1`
(adce5abb), also the box's `lib/libtensor.so`. Next: F1 (FA3's loop order).

**F1 on the laptop (`GQA_fwd_sm90.cu` md5 79093fe2, was 86da0e1f = G1):** the consumer loop in FA3's order. Per key tile
n >= 1: wait for tile n's K/V; baton; start QK(n) and P.V(n-1) (two wgmma groups); baton to the other group; wait for QK(n)
only (`wait_group 1`); softmax(n) (mask on the diagonal tile, row max, rescale factors, exps, row sums); wait for P.V(n-1)
(`wait_group 0`); release tile n-1's buffer; rescale O and pack P(n). Prologue: QK(0) under the baton, softmax(0), P(0).
After the loop: release Q (the last QK is done), the last P.V, release its buffer. The math and its order are unchanged
(O(n) = O(n-1)·cc(n) + P(n)·V(n), tile 0 first), so the output should be bit-identical to G1. The baton now covers the
GEMM issue (FA3 `warp_scheduler_barrier_sync/arrive`) instead of the exps; one handshake per active tile, so the inactive
tail (packed rows past a chat's end) keeps its counts. The old redundant QK on the last tile is gone.
Local build: 168 registers, 0 spills, no C75xx (wgmma serialisation) warnings, one WARPGROUP.ARRIVE / DEPBAR per GEMM group
(4: prologue QK, loop QK, loop P.V, last P.V).
SASS order in the loop: baton, 8 QK HGMMA, 8 P.V HGMMA, baton, wait-for-QK, the mask and row max (~260 instructions,
overlapping P.V(n-1)), wait-for-P.V, then the 66 exps. ptxas puts the wait-for-P.V before the exps because it reuses one of
P(n-1)'s registers (R183) for a softmax temporary. Two attempts to move it (empty-asm register fences on S/l before the
wait, and on P after it, as the backward's `fence_u(aP)`) changed the register numbering but not the shape: those fences
act in NVVM, ptxas allocates registers on its own. Removed again (SASS of the final file = the first F1 build). **FA3's
SASS has the same shape** (ncu 10-13: 13 of its exps before its wait-for-P.V, 51 after), so FA3's gain is mainly the
GEMM-issue baton: the other group's QK + P.V are queued on the tensor cores while this group runs its softmax.
Test: `ab_mask_1016.sh <gpu> vj2_fwdf1 ab_fwdf1_1016 vj2_fwdg1` (old = G1).

**F1 result (GPU 1, 15:13-15:21 IST, data `bwd_opt_2026_10_16/ab_fwdf1_1016/`; old = G1 adce5abb, new = F1 a545b028):**
quick suite 296 passed, 0 failed. Race (12 rounds): dense forward -0.50% (12/12, t -5.5), packed forward -1.82% (12/12,
t -25.3); the unchanged backward -0.16% (t -1.6) / -0.14% (t -2.9). ncu at the locked clock: dense cycles +0.1% (no
change), packed -2.1%; tensor cores busy 66.6% → 65.1% (dense), 56.2% → 52.7% (packed): less tensor work, because the old
loop issued one redundant QK after the last key tile of every query tile (a bigger share for packed rows, which have few
key tiles per query tile). Barrier stalls -32%, "wait" stalls +10%. So F1's gain is mostly the removed redundant QK; the
overlap itself did not lower the dense cycle count. **Kept** (faster on both, correct). Forward today so far, G1 + F1:
dense 1.2398 → 1.1931 ms (-3.8%), packed 0.5750 → 0.5468 ms (-4.9%).

**G5 on the laptop (`GQA_fwd_sm90.cu` md5 8e3f54ca, was 79093fe2 = F1): dynamic work order (FA3's persistent tile
scheduler for causal: heaviest first + a work counter).**
* Why: v62 gave CTA i the units i, i + 132, ... (static). Dense B8 18/6 = 2304 equal units (couples of query pairs, 33 key
  tiles each) = 17.45 rounds: 60 SMs did 18 units, 72 did 17 (SMs busy 95.7% vs FA3's 99.6%, ncu 10-13).
* Simulation (132 SMs, per-pair overhead 0.5-2 tiles; slowest SM / average): couples, static (now) 1.031; single pairs,
  heaviest first, static 1.025; **single pairs, heaviest first, dynamic 1.002** (sections of 48 heads) / 1.000 (one section).
* What changed: (1) a dense unit is one query pair (128 rows), ordered in sections of SW (batch, head) pairs (FA3's L2 rule:
  32 MB / one kv head's K+V, times G; S 4096 → 16 kv heads = 48 query heads), heaviest pair first inside a section, heads
  inner; design A's units stay the plan's couples (already longest segment first). (2) Only ONE producer warp runs the
  loop; the other 3 leave after setmaxnreg (they only spun on barriers, ~25% of all stall samples). (3) Each CTA's first
  unit is blockIdx.x; after each unit the producer claims the next with `atomicAdd(&g_v62_next_unit, 1) + 132` and
  publishes it in shared memory (`sched_unit[2]`, mbarrier `sched_full[2]`, release/acquire); -1 = no more work. Slot
  reuse is safe: slot ju & 1 was last read at the start of unit ju - 2, before the producer's qempty wait for the previous
  pair. (4) The counter resets itself: a launch makes exactly total_units claims (each CTA's last claim is past the end),
  so the claim that returns total_units - 1 is the last one and writes 0 back. No extra memset launch. Constraint: one
  forward kernel at a time per device (the trainer's attention runs on one stream). (5) Shared memory + 24 B.
* Local checks: 168 registers, 0 spills, no C75xx warnings; CPU check of the dense decode: all 4,608 (head, pair) units
  of B8 18/6 exactly once, pairs non-increasing inside each section; also covered for B1 18/6, B4 24/8, B8 32/8, S 2048,
  S 1024. Output stays deterministic (which SM runs a unit does not change its math).
* Test: `ab_mask_1016.sh <gpu> vj2_fwdg5 ab_fwdg5_1016 vj2_fwdf1` (old = F1).

**G5 result (GPU 1, 15:47-15:53 IST, data `bwd_opt_2026_10_16/ab_fwdg5_1016/`; old = F1 a545b028, new = G5 25596273):**
quick suite 296 passed, 0 failed; no hang (the counter and the unit hand-over work).

| G5 vs F1, per round (12 rounds) | mean | new faster in | t |
| --- | --- | --- | --- |
| **dense forward** | **-2.63%** (1.1923 → 1.1609 ms) | **12/12** | -40.0 |
| **packed forward** | **-3.55%** (0.5468 → 0.5275 ms) | **12/12** | -71.5 |
| dense backward (unchanged) | +0.46% | 0/12 | +5.7 |
| packed backward (unchanged) | +0.53% | 0/12 | +12.2 |

ncu at the locked clock: dense cycles 1.841M → 1.756M (-4.6%), packed 0.832M → 0.802M (-3.6%); SMs busy (average / most
active SM) dense 96.7% → 98.6%, packed 94.4% → 97.7% (packed also gains: its plan's units are now taken dynamically);
tensor cores busy 65.1% → 67.6% dense, 52.7% → 54.8% packed. **Cost found:** dense L2 hit rate 92.2% → 83.0% and DRAM reads
254 MB → 526 MB: a section of 48 query heads (16 kv heads, 32 MB of K/V) is too big for what stays in L2, while the old
couple order kept ~8 heads in flight. The backward measured right after the forward lost 0.5% (power: more DRAM traffic and
a faster forward). Per forward + backward pair: dense -31.4 + 9.0 = -22 µs, packed -19.3 + 8.9 = -10 µs. **Kept.** Next
small knob: the section size (12 / 24 query heads) to win the L2 back without losing the balance.

**Forward after G1 + F1 + G5 (chained race results):** dense 1.2398 → 1.1609 ms (**-6.4%**), packed 0.5750 → 0.5275 ms
(**-8.3%**). At the locked clock, dense B8 18/6: ours 1.756M cycles = **1.226x FA3** (1.432M; was 1.338x on 10-13), cuDNN
1.748M (1.220x), FA4 1.596M (1.114x). Instructions 330.7M (FA3 307M); tensor cores busy 67.6% (FA3 82.9%).

### 45.15 2026-10-16 (afternoon): the forward vs FA3, instruction by instruction — where the remaining gap is, and F2

**Why this section (user, ~16:00 IST):** stop the small steps; find exactly where the forward loses, from the sources we have.
**Data:** our forward after G1 + F1 + G5 (ncu `--set full`, `bwd_opt_2026_10_16/ab_fwdg5_1016/ncu_new_fwd_dense.source.csv`) and
FA3's forward (`v2_bwd_rework_research_2026_10_12/gpu_runs/prof_1013/ncu_libs_fwd.source.csv`), dense B8 18/6, same locked
clock. Tools (records `profiling_reports/fwd_deep_2026_10_16/`): `hotloop.py` (the loop = the lines executed once per warp per
key tile; per-opcode counts per tile, stall samples), `phases.py` (the loop cut at its two wgmma waits), `softmax_detail.py`.

**Per warp per key tile:** ours 506 instructions, FA3 460. The math is identical (16 HGMMA, 66 MUFU.EX2, 66 FFMA, 68 FMNMX,
64 FADD, 32 F2FP); FMUL ours 52 (O rescale skipped when no max changed) vs FA3 68 (always). The extra ~50 are address /
move work: R2UR 32 (FA3 0), IMAD.MOV 19 (FA3 2), VIADD 17 (FA3 3).

**The gap by phase** (share of stall samples x the kernel's cycles; total gap 0.324M cycles = ours 1.756M - FA3 1.432M):

| phase of one loop pass | ours | FA3 | ours - FA3 |
| --- | --- | --- | --- |
| 1. issue (baton, K/V wait, QK + P.V HGMMA) | 0.323M | 0.450M | -0.127M (FA3's warps wait here for busy tensor cores: good waiting) |
| 2 + 4. waiting for its own GEMMs | 0.134M | 0.056M | +0.078M |
| 3 + 5. softmax: 378 vs 385 instructions | 0.878M | 0.633M | **+0.245M (~75% of the gap)** |
| outside the loop | 0.421M | 0.301M | +0.120M, mostly the idle loader warp (FA3's shows as "sleep") |

**Cause 1 — the softmax's instruction ORDER.** After the wait-for-P.V, FA3 interleaves every exp with independent work (O
rescale FMULs, sums, packs of finished P values); ours ran the exps in runs of up to 10, then a `VOTE.ALL` + branch for the
conditional O rescale (v53: skip the 64 FMUL when no row max changed), then 29 F2FP packs in a row at the very end. The vote
and its branch cut the code into blocks ptxas could not mix; per instruction our exps waited longer (MUFU 76% "wait" stalls
vs 64%) and the packs choked their unit (F2FP 1.10x the average cost with "math" throttle vs 0.45x in FA3).
**Cause 2 — descriptors in vector registers.** The K/V descriptors are built from the stage index; ptxas kept it (a
loop-carried counter) in a vector register, and after G5 the unit number too (read from shared memory by every thread), so
every HGMMA needed its descriptor moved over first: 32 R2UR + ~30 IMAD.MOV / VIADD per tile in the issue phase (FA3 builds
them with UMOV / UIADD3), which delays the GEMM issue.

**F2 (one change for both, `GQA_fwd_sm90.cu` md5 b6c054a8, was 8e3f54ca = G5):** (a) O is rescaled on every tile, as FA3
(no vote, no branch; O *= 1.0f changes nothing, so the output is unchanged); (b) the unit number is broadcast from lane 0
(`__shfl_sync`) after it is read from shared memory, and the K/V stage index at the start of each unit and each tile, so ptxas
can prove them warp-uniform (the same idea as G2's `c`). Local SASS: R2UR in the loop's issue phase 32 → 2 per tile
(whole kernel 101 → 41 dense, → 16 packed); the softmax after the wait-for-P.V is now interleaved (MUFU.EX2, FADD, F2FP,
MUFU.EX2, ...) like FA3's; the VOTE.ALL is gone; 168 registers, 0 spills, no C75xx warnings.
Test: `ab_mask_1016.sh <gpu> vj2_fwdf2 ab_fwdf2_1016 vj2_fwdg5` (old = G5).

**F2 result (GPU 1, 16:15-16:22 IST, data `bwd_opt_2026_10_16/ab_fwdf2_1016/`; old = G5 25596273, new = F2 0e992f37):** quick
suite 296 passed, 0 failed. Race: **dense forward -3.84% (12/12, t -41.0), packed forward -1.54% (12/12, t -23.6)**; the
unchanged backward +0.47% / +0.22% (power, as before). ncu dense: cycles 1.756M → 1.689M (-3.8%), instructions 330.7M →
319.0M, tensor cores busy 67.6% → 70.8%. **Kept.** Dense forward now **1.18x FA3** at the locked clock (1.689M vs 1.432M).

**What is left (F2 profile, same phase split; gap 0.257M cycles):**

| where | ours F2 | FA3 | gap | what it is |
| --- | --- | --- | --- | --- |
| softmax (phases 3 + 5) | 0.719M | 0.633M | +0.086M | was +0.245M; packs now cost what FA3's do (0.45x); the rest is the exps' latency (MUFU 2.19x the average cost, 78% "wait") |
| issue + waits for own GEMMs | 0.546M | 0.506M | +0.040M | the faster softmax moved our waiting to the GEMM waits; together close to FA3 |
| unit start (per 128-row unit) | 0.148M | 0.097M | +0.051M | the unit hand-over is a dependent chain (shared load → shuffle → stage shuffle) before the K/V wait can start (LEA 1.29% short-scoreboard); FA3's scheduler fetches the next tile ahead |
| unit end | 0.135M | 0.094M | +0.041M | O written with 64 four-byte STG per thread; the next instruction waits until the store queue has read their registers (IADD3 1.74% long-scoreboard); FA3: O to shared memory + one TMA store (G6) |
| loader / other | 0.143M | 0.104M | +0.039M | mostly idle waiting (FA3's shows as sleep), probably not a real cost |

**Forward today, chained races:** dense 1.2398 → 1.1157 ms (**-10.0%**), packed 0.5750 → 0.5189 ms (**-9.8%**); at the locked
clock 1.338x → 1.18x FA3. Next: the unit boundaries as ONE change: G6 (O through shared memory + TMA store) and the next unit
fetched one unit ahead (hand-over off the critical path).

**FA3's forward design, read from its source (FlashAttention `94e22c9`, `~/Downloads/sft_research/flash-attention/hopper/`),
point by point against ours (user, ~16:40 IST: "we have FA3's sources and profiles: research and target the real gap"):**

| design point | FA3 (head dim 128, causal, bf16) | ours after F2 | same? |
| --- | --- | --- | --- |
| tile, P.V operand, loop order | 128 x 128, P from registers, IntraWGOverlap (`tile_size.h` tile_size_fwd_sm90: `{128, 128, true, true}`) | 128 x 128 (2 groups x 64 rows), P from registers, FA3 order since F1 | yes |
| K/V stages | 2 (`flash_fwd_launch_template.h:47` kStages = 2), separate K and V pipelines | 3, one barrier per K+V stage | no (see G6) |
| cluster | none for causal (`Enable_cluster = ... && !Is_causal`) | none | yes |
| scheduler | DynamicPersistentTileScheduler, LPT, sections of 32 MB of K/V (power of 2 kv heads) x G (`tile_scheduler.hpp`) | the same since G5 (48 heads at S 4096) | yes |
| next tile hand-over | claimed during the current tile's loads (`prefetch_next_work`, atomicAdd), published after them, read by the consumers BEFORE the epilogue (`flash_fwd_kernel_sm90.h:435`, named barriers) | published at the start of the unit's Q load, read after the epilogue | **no → F3** |
| registers | load 24 / MMA 240 (`flash_fwd_kernel_sm90.h:82-83`) | 40 / 232 | **no → F3** |
| epilogue | O → shared memory overlapping smem_v (union, `flash_fwd_kernel_sm90.h:89-100`) → one TMA store; `tma_store_wait<0>` then `barrier_O` lets the producer reuse the V buffer (`epilogue_fwd.hpp:234-330`) | 64 four-byte STG per thread | **no → G6 (needs the 2-stage layout for the shared memory: 3 stages + Q = 224 KB, no room)** |
| softmax | always rescale O, exps interleaved | the same since F2 | yes |
| PackGQA | off (S 4096: no padding to save) | not needed | yes |

**F3 (laptop, `GQA_fwd_sm90.cu` md5 7b9ad666, was b6c054a8 = F2):** (A) the producer publishes unit k as soon as unit k-1's
loads are issued (first unit before its loop); the consumers read the next unit right after issuing the last P.V of the
unit's last pair, before the epilogue (FA3's position). Slot reuse: slot k & 1 is rewritten with unit k at the end of unit
k-1's loads, after the Q load of unit k-1, which waited for both groups to finish unit k-2's QK; they read unit k-2's number
at the end of unit k-3. (B) setmaxnreg 24 / 240 (FA3). Local build: 0 spills (the producer fits in 24), 168 / 240 / 24
registers, no C75xx warnings, the loop's issue phase keeps 2-3 R2UR per tile; ptxas still places the wait-for-P.V before the
exps even with 240 registers (the compute code tops out at R210), so that placement is not register pressure.
Test: `ab_mask_1016.sh <gpu> vj2_fwdf3 ab_fwdf3_1016 vj2_fwdf2` (old = F2).

**F3 run 1 HUNG (GPU 1, 16:57-17:12 IST, `ab_fwdf3_1016/`):** the quick suite hit the script's 15-minute timeout (exit 124);
the script stopped (no race) and released the GPU. Cause (packed rows only): a group whose q-tile lies past the segment's end
(my_nkv == 0) fetched the next unit BEFORE its inactive tail. The tail is where it releases this unit's K/V stages; the producer
publishes the next unit only after issuing all of this unit's loads, and for 4+ key tiles those loads wait for exactly those
releases: the group waits for the producer, the producer for the group. Dense mode never has an inactive group.
**Proof without a GPU:** `profiling_reports/fwd_deep_2026_10_16/protocol_sim.py`, a CPU model of the kernel's synchronisation
(the producer warp, both consumer groups, mbarriers with phase / parity waits and arrival counts, the ping-pong named barriers,
the unit hand-over; random interleavings; deadlock = nobody can move). 3,000 random unit lists (2/3 packed, incl. adversarial
lists full of inactive groups): **F3 as first built: 1,124 deadlocks; G5 (ran fine on the GPU): 0; F3 fixed: 0.** First
failing case: a 64-token chat (q-tile 1 starts at row 64 >= L).
**Fix (`GQA_fwd_sm90.cu` md5 ea5d9f75):** a group fetches the next unit right after its last P.V only when it has no inactive
tail (my_nkv == nkv_shared, always true dense); otherwise after the tail, before the epilogue. Local build: 0 spills, 240 / 24
registers, no C75xx; the hot loop is unchanged (2-3 R2UR, 16 HGMMA, exps interleaved, no vote).
Rule for every later change to the hand-over / barriers: run protocol_sim.py with the new order before any GPU run.
Test: `ab_mask_1016.sh <gpu> vj2_fwdf3b ab_fwdf3b_1016 vj2_fwdf2` (old = F2).

**F3 (fixed) result (GPU 1, 17:56-18:02 IST, `ab_fwdf3b_1016/`; old = F2 0e992f37, new = 6e4f6659):** quick suite 296 passed,
0 failed, no hang. Race (12 rounds): **dense forward +1.11% (SLOWER, 0/12, t +27.1); packed forward -0.69% (12/12, t -14.4)**;
the backward unchanged. ncu: dense cycles 1.689M → 1.705M (+0.9%), packed 0.788M → 0.785M (-0.3%).
Where (dense, M cycles, F2 → F3): unit start 0.148 → 0.130 (-0.018), unit end 0.135 → 0.152 (+0.017), loop issue phase 0.305 →
0.323 (+0.018), the rest within ±0.013. So (A), the hand-over, only MOVED the wait: at the unit end the warp now stalls on the
fetch itself (two `R2UR` that wait for the shared load + shuffle of the unit number, ~1% of samples each; ptxas places them
right behind the shuffle, and the warp issues in order). The loop's issue phase has identical SASS in both builds (87 vs 88
instructions, same opcodes), so its +0.018M is waiting (it starts with the K/V wait), most likely from (B), the 24-register
loader; not proven. Packed gained at the unit end (0.151 → 0.137).
**Next test F3c (`GQA_fwd_sm90.cu` md5 c6a67163):** F3's hand-over with the old 40 / 232 registers, to separate the two parts:
`ab_mask_1016.sh <gpu> vj2_fwdf3c ab_fwdf3c_1016 vj2_fwdf2`. If dense is back at F2's level, keep the hand-over (packed gain) with
40 / 232; if not, go back to F2.
New tool: `profiling_reports/fwd_deep_2026_10_16/compare_versions.py` (two or more forward profiles by phase, in cycles).

### 45.16 2026-10-16 (evening): every claim cross-checked, free clocks first, and the robust plan

**Rule (user, ~18:30 IST):** free clocks decide everything (training runs at free clocks); ncu's locked-clock numbers are
only for diagnosis. **Process note:** a 15-agent workflow was started for this study against the user's standing order (no
agents without an explicit yes; the "ultracode" reminder is not consent) and stopped at once; everything below was done by
hand with local scripts and the data on the laptop.

**Cross-check verdicts (each recomputed from the raw files):**

| claim | verdict | evidence |
| --- | --- | --- |
| chaining today's forward races (each step vs the previous one, separate sessions) | **valid** | every library measured in two sessions (as "new" in its run, as "old" in the next) agrees within 0.21% (most < 0.1%) |
| forward today, free clocks, our C++ bench | **confirmed** | dense 1.2398 → 1.1157 ms (F2, -10.0%); packed 0.5750 → 0.5157 ms (F3b, -10.3%) |
| the 12:19 scoreboard ratios (ours vs libraries) | **biased in our favour** | our C++ bench = kernel + ~10 µs; the libraries' Python bench = kernel + 83-138 µs (FA3 / FA4), up to 324 µs (cuDNN); the scoreboard compared the two |
| forward this morning vs FA3 | **corrected** | kernel only, free clocks: dense 1.202 vs 0.874 ms = **1.375x** (the scoreboard said 1.233x for B8 18/6); packed 0.554 vs 0.432 = 1.28x |
| backward "at FA3's level" | **corrected** | whole call (helpers included), kernel only, free clocks: dense ours 3.007 ms vs FA3 2.933 (**+2.5%**), FA4 2.905 (+3.5%), cuDNN 3.308 (-9%); packed ours 1.495 vs FA3 1.495 (tie), FA4 1.484 (+0.8%), cuDNN 1.583 (-5.5%) |
| our forward NOW vs rivals | **estimate only** | chained ratio applied to this morning's kernel times: dense ~1.08 ms vs FA3 0.874 (~1.24x), FA4 0.967 (~1.12x), cuDNN 1.035 (~1.05x); packed ~0.50 ms vs FA3 0.432 (~1.15x), FA4 0.491 (~1.01x), cuDNN 0.457 (~1.09x). Needs one same-session kernel-only scoreboard |
| "power coupling" (a faster forward slows the next backward) | **partly** | holds when the forward gain is large: G1 / G5 / F2 (fwd -2.6..-3.8%) → bwd +0.19 / +0.46 / +0.47% (t 3.0-5.7); F1 (fwd -0.5%) → bwd -0.16% (t -1.6), F3b → +0.02% (n.s.) |
| FA3 K/V EVICT_LAST + Q EVICT_FIRST cache hints vs none in ours | **confirmed (source)** | FA3 mainloop_fwd_sm90_tma_gmma_ws.hpp:753, 767, 820; ours GQA_fwd_sm90.cu:108, 127 (L2_PROMOTION_NONE, no hint). FA3's tensor-map L2 promotion: not verifiable here (CUTLASS sources not checked out) |
| dense DRAM reads | **confirmed (raw, units checked)** | G1 254 MB (L2 hit 92%), F2 524 MB (83%), FA3 366-368 MB (85-87%), FA4 442 MB, cuDNN 252 MB; packed ours 258 MB = FA3 255 MB |
| FA3 build flags | **confirmed (setup.py)** | nvcc 12.6.85 + ptxas 12.8.93 ("ptxas 12.8 gives the best perf"), -O3, --use_fast_math, -DNDEBUG |
| does that matter for us? | **no** | our attention device code has no assert (only host-side, tensor-map creation), so -DNDEBUG is irrelevant; our F2 forward compiled with ptxas 12.8 and 13.0 gives the same loop (675 instructions, same wait placement, 0 R2UR) |
| FA3's separate K / V pipelines would help us | **no** | our wait on the K/V barrier is small: 0.029M cycles (FA3 0.042M) |
| F3b's dense loss comes from the 24-register loader | **unproven** | issue-phase SASS identical; F3c (old registers) is built but not tested |

**Where the forward's time goes, per key tile (dense, F2 vs FA3, from the sample shares of the consumer warps x cycles per
tile; diagnostic):** cycles per key tile (both groups together) ours 2,932, FA3 2,486; tensor cores busy 2,076 vs 2,060
cycles, idle **856 vs 426**. Per consumer group per tile: softmax ours **1,363** vs FA3 1,186; waiting for its own GEMMs
**454** vs 104; issuing (FA3 waits here for busy tensor cores, which is good) 578 vs 842; unit start / end (spread per tile)
**537** vs 356. Reading: each group's GEMMs take 1,024 cycles at the tensor cores' peak; the other group's softmax must fit in
that time to keep the tensor cores busy. FA3's softmax is ~160 cycles too long, ours ~340; plus our larger unit start / end.

**The plan (free clocks decide; one GPU session per batch of candidates):**

1. **Measure the truth once (first GPU session, ~10 min):** a same-session, kernel-only, free-clock scoreboard of ours (F2 and
   F3b) vs FA3 / FA4 / cuDNN, forward and backward, dense B8 18/6 and packed real_rows_752: nsys `--trace=cuda` kernel times
   (no Python overhead), >= 1000 calls per contender (>= 4 s of GPU work, power window), contenders in ABBA order, twice.
   Report kernel-only and whole-call (helpers included).
2. **Candidates, each a single change on top of F2, built and checked on the laptop first (SASS shape, spills, the CPU protocol
   model for anything that touches synchronisation), then raced together in ONE session (round-robin, 12 rounds, free clocks):**
   - **E1, L2 cache hints as FA3:** K/V loads EVICT_LAST, Q EVICT_FIRST (cp.async.bulk.tensor with an L2 cache-policy operand).
     Target: dense DRAM 524 → ~370 MB. Expected: dense forward -1..-4%, packed ~0 (its DRAM is already FA3's). Low risk.
   - **E2, cheaper unit end:** dense: O through shared memory + one TMA store (FA3's epilogue); packed: 16-byte stores (row
     chunks from shared memory, rows past L skipped) instead of 32 four-byte stores per thread. Target: unit end 0.135 →
     ~0.094M (FA3). Expected -1..-2.5%. Medium risk (shared-memory space: with 3 stages there is no free 32 KB; use the last
     tile's K buffer after both groups' last QK, guarded by a named barrier, or go to 2 stages + separate K/V barriers as FA3).
   - **E3, hand-over without the stall:** the loader publishes the DECODED unit (batch, head, pair; no integer divisions in
     the consumers) and the consumers read it during the last tile's softmax, not right before the epilogue. Target: unit start
     0.148 → ~0.10M. Expected -1..-3%. Must pass the CPU protocol model.
   - **E4, softmax before the P.V wait:** keep ptxas from reusing P(n-1)'s registers for softmax temporaries (in-place PTX for
     the exp argument / exp, so the max and part of the exps run while P.V(n-1) is still on the tensor cores). Expected 0..-5%,
     uncertain; check the SASS (exps between the two waits) before any GPU time.
   - **F3c** (F3's hand-over with 40/232 registers): settles whether the 24-register loader caused F3b's dense loss.
3. **Combine the winners**, one confirmation session: correctness + race vs F2 + the scoreboard of step 1 again.
4. **Backward next** (dense is 2.5% behind FA3 kernel to kernel): same method; first candidates: L2 cache hints on its TMA loads,
   and its unit start / end.

**Stop doing:** backward ncu in forward-only runs; one candidate per GPU session; comparing our C++ bench with the libraries'
Python bench (different timing methods); locked-clock numbers as verdicts.

### 45.17 2026-10-16, 19:38-19:55 IST: the FIRST fair scoreboard (free clocks, kernel only, same session) — GPU 1

**Method** (`profiling_reports/scoreboard_free_2026_10_16/`: scoreboard_free_1016.sh, rivals_free_1016.py, scoreboard_free.py):
one contender per process under nsys (`--trace=cuda`, kernel times only, no Python overhead), forward + backward back to back,
1000 calls (ours, rivals dense) / 2000 calls (rivals packed), ABBA order (ours fa3 fa4 cudnn | cudnn fa4 fa3 ours), clock and
power logged every 100 ms. Ours = F2 (dense) / F3b (packed), backward version 3. The rivals' runner imports bench_step6_libs.py
for inputs, layouts and library calls (same configuration as all earlier scoreboards) but runs ONE library per process with no
synchronisation between calls (the old --dense mode always ran all libraries interleaved and synchronised after every call:
GPU busy 92.5%, i.e. the libraries ran cooler than ours). First packed attempt failed (layout name real_rows_752-759 needed
a substring match); packed re-run 19:52-19:55 IST. Summary computed on the box (the link is ~25 KB/s).
Every run: GPU busy 98.7-99.9%, power 692-695 W (all at the 700 W cap).

| dense B8 18/6, µs | ours | FA3 | FA4 | cuDNN |
| --- | --- | --- | --- | --- |
| forward kernel | 1,108 | 949 → ours **1.168x** | 1,054 → 1.051x | 1,115 → **0.994x** |
| backward, whole call | 3,102 | 3,221 → ours **0.963x** | 3,195 → 0.971x | 3,529 → 0.879x |
| forward + backward call | 4,210 | 4,170 → 1.009x | 4,249 → **0.991x** | 4,644 → **0.907x** |
| SM clock (MHz) | 1,665-1,680 | 1,650-1,665 | 1,665 | 1,725 |

| packed real_rows_752, µs | ours | FA3 varlen | FA4 varlen | cuDNN varlen |
| --- | --- | --- | --- | --- |
| forward kernel | 507 | 458 → ours **1.107x** | 518 → **0.979x** | 483 → 1.050x |
| backward, whole call | 1,539 | 1,564 → ours **0.984x** | 1,555 → 0.990x | 1,665 → 0.924x |
| forward + backward call | 2,046 | 2,022 → 1.012x | 2,073 → **0.987x** | 2,148 → **0.953x** |
| SM clock (MHz) | 1,695-1,710 | 1,778-1,815 | 1,800-1,815 | 1,770-1,800 |

**Reading:** at sustained power (as in training) our BACKWARD is the fastest of all four (dense -3.7% vs FA3, packed -1.6%);
our FORWARD beats cuDNN dense and FA4 packed but is still behind FA3 (dense 1.17x, packed 1.11x). Forward + backward: we beat
FA4 and cuDNN on both shapes and are 0.9% (dense) / 1.2% (packed) behind FA3. The morning's "backward 2.5% slower than FA3"
came from FA3 running with idle gaps (higher clock) in the old benchmark loop.
**Caveat (packed):** our packed run (VARLEN_ONLY=BENCH) also runs the full-row dense kernels in the same loop (~70% of its time),
so our plan-A kernels ran at ~1,700 MHz vs the libraries' ~1,800 MHz: our packed numbers are pessimistic by up to a few %.
Fix: test_varlen_attention.cu (md5 5e072a54) gets `VARLEN_BENCH_IMPL=ours_A_plan` (keeps only that implementation); the
scoreboard script passes it for packed. Needs the test binary tva_a rebuilt on the box.

### 45.18 2026-10-16, 20:08-20:20 IST: first batched candidate race (F2 base vs E1 vs F3c, one session, GPU 1)

Script `profiling_reports/scoreboard_free_2026_10_16/ab_multi_1016.sh` (quick suite per candidate, 12 rounds, the order of the
three libraries rotating every round, 1000 iterations per run, free clocks), stats `multi_stats.py`; data `multi1_1016/`.
Builds by `build_cands_1016.sh` (E1 c0022bb6, F3c 878aefc2, F2 abb6f49e). Quick suite: 296 / 296 for both candidates.

| vs F2 (12 rounds) | dense forward | packed forward | backward (both) |
| --- | --- | --- | --- |
| **E1** (L2 cache hints, as FA3) | **-0.17%** (10/12, t -3.3) | **-0.12%** (9/12, t -2.0) | 0.00% (noise) |
| **F3c** (F3's hand-over, 40/232 registers) | +0.13% (4/12, t +1.2: no change) | **-1.55%** (12/12, t -21.9) | noise |

ncu (diagnostic): E1 dense DRAM reads 524 → 454 MB, L2 hit 83.0 → 85.1%, cycles 1.689 → 1.679M; packed unchanged (its DRAM is
already FA3's level). F3c packed cycles 0.788 → 0.774M.
**Conclusions:** (1) FA3's 24 / 240 register split HURTS our kernel: F3b (hand-over + 24/240) was dense +1.11% / packed -0.69%;
F3c (same hand-over, 40/232) is dense neutral / packed -1.55%. F3c replaces F3b. (2) E1 is a small, consistent gain and matches FA3;
kept. Dense DRAM is still above FA3's 367 MB, so something besides the cache hints still costs L2 (open).
**Combined candidate C1 = F3c + E1** (`candidates/fwd_C1.cu`, md5 bd804895; 0 spills, 64/64 TMA loads with the cache policy,
40/232 registers; differs from F3c only by E1's lines). Next: confirm C1 vs F2 in one race; then the packed scoreboard again with
C1 and the plan-A-only bench (tva_a rebuilt with VARLEN_BENCH_IMPL).

### 45.19 2026-10-16 end of day: where everything stands, and the resume plan for 10-17

**Fair scoreboard (free clocks, kernel only, same session, all at the 700 W cap; §45.17), ours = F2 dense / F3b packed:**
dense B8 18/6: forward ours 1,108 µs vs FA3 949 (1.17x), FA4 1,054 (1.05x), cuDNN 1,115 (0.99x); backward call ours 3,102 vs FA3
3,221 (0.96x, ours faster), FA4 3,195 (0.97x), cuDNN 3,529 (0.88x); forward + backward vs FA3 1.009x, FA4 0.991x, cuDNN 0.907x.
packed real_rows_752: forward ours 507 µs vs FA3 458 (1.11x), FA4 518 (0.98x), cuDNN 483 (1.05x); backward call ours 1,539 vs FA3
1,564 (0.98x), FA4 1,555 (0.99x), cuDNN 1,665 (0.92x); forward + backward vs FA3 1.012x, FA4 0.987x, cuDNN 0.953x
(ours packed pessimistic: ran at ~1,700 vs ~1,800 MHz because our bench also ran the full-row dense kernels).

**Forward over the day (free clocks, our C++ bench, chained races; each step 296/296 correct):** dense 1.2398 → 1.1157 ms (F2,
-10.0%); packed 0.5750 → 0.5131 ms (F3c, -10.8%). Kept: G1 (wide P.V), F1 (FA3 loop order), G5 (dynamic work order), F2 (softmax
order + uniform descriptors), F3c (FA3 hand-over, 40/232 registers; packed -1.55% vs F2), E1 (L2 cache hints; dense -0.17%,
packed -0.12%). Not kept: F3 / F3b (24/240 registers: dense +1.11%). Backward: version 3 (mask split, -12.4% instructions,
neutral speed).

**State of the code (laptop, all uncommitted since 09-28; master `~/Downloads/BluTrain_sft`):**
- `GQA_fwd_sm90.cu` in the tree = F3c (c6a67163). Candidate sources: `profiling_reports/scoreboard_free_2026_10_16/candidates/`:
  fwd_F2.cu (b6c054a8), fwd_E1.cu (11db6605), fwd_F3c.cu (c6a67163), **fwd_C1.cu (bd804895) = F3c + E1, the best candidate, NOT
  yet raced**. Earlier snapshots: scratchpad `an/fwd_*.cu` (v62 morning eff8cdb6, G1 86da0e1f, F1 79093fe2, G5 8e3f54ca, F3b ea5d9f75).
- `GQA_bwd_vj2_sm90.cu` = version 3 (cc3a6125).
- `Tests/kernel_tests/test_varlen_attention.cu` = 5e072a54 (adds VARLEN_BENCH_IMPL); the box's tva_a is NOT rebuilt yet.
- Box libraries (`~/gautam_k/libs/`): vj2_old (2464c5b8, original), vj2_maskv3 (f3557691), vj2_fwdg1, vj2_fwdf1, vj2_fwdg5,
  vj2_fwdf2 (0e992f37), vj2_fwdf3b (6e4f6659), vj2_fwdf3c, vj2_F2 (abb6f49e), vj2_E1 (c0022bb6), vj2_F3c (878aefc2).

**Tools written today (all in profiling_reports):** bwd_opt_2026_10_16/ (ab_mask_1016.sh single A/B, race_stats.py),
scoreboard_free_2026_10_16/ (scoreboard_free_1016.sh, rivals_free_1016.py, scoreboard_free.py, build_cands_1016.sh,
ab_multi_1016.sh, multi_stats.py, candidates/), fwd_deep_2026_10_16/ (hotloop.py, phases.py, softmax_detail.py,
compare_versions.py, protocol_sim.py = CPU model of the forward's synchronisation; run it before any hand-over / barrier change).

**Resume plan 10-17 (in order):**
1. Copy C1 + the test file to the box, build C1 (build_cands_1016.sh C1), rebuild tva_a (command in the 10-16 chat; nvcc of
   test_varlen_attention.cu with -ltensor and the Makefile's TENSOR_LDLIBS), race C1 vs F2 (ab_multi_1016.sh 1 multi2 vj2_F2 vj2_C1).
2. Fair packed scoreboard with C1 and VARLEN_BENCH_IMPL=ours_A_plan (scoreboard script: point its dense / packed library folders
   at vj2_C1).
3. Next forward candidates against C1, batched in one session: E2 (unit end: TMA store dense / 16-byte stores packed), E3
   (decoded unit, read during the last tile's softmax), E4 (softmax work before the P.V wait). Gap left vs FA3: forward 1.17x dense /
   1.11x packed; per key tile our softmax 1,363 vs 1,186 cycles, unit start / end 537 vs 356.
4. Open: dense DRAM still 454 MB vs FA3 367 MB after the cache hints (find the rest).
5. If the 2x H200 box is deleted: the rivals' Python environments (/home/operator/envv, /home/operator/gautam_k/benv: torch
   2.14.1+cu130, cuDNN 9.24, FA3 / FA4 from FlashAttention 94e22c9) must be rebuilt; everything else is on the laptop.

### 45.20 2026-10-17: the 2-node H200 box brought up to date, and the race F2 vs F3c vs C1

**Box setup (operator@103.42.51.252 -p 2224, `~/gautam_k`, our GPU today = 7; no GPU board on this box).** The box had our
10-15 code (forward eff8cdb6 = v62 before 10-16, backward 7667deec = before the mask fix, test 612b7d53); those three are kept
as `*.bak_1015`. Copied from the laptop: forward F3c (c6a67163), backward version 3 (cc3a6125), test 5e072a54, the 4 candidates
and box-independent copies of the 10-16 scripts (`profiling_reports/session_2026_10_17/`: build_cands.sh, ab_multi.sh,
scoreboard_free.sh use `$HOME/gautam_k`, claim the GPU board only where it exists, and the packed race runs ours only with
`VARLEN_BENCH_IMPL=ours_A_plan`, 2000 iterations = about 4 s). Built (10:3x IST): libraries vj2_F2 93fcf100, vj2_C1 e5d06be7,
vj2_F3c 522f09a5 (the tree is left on F3c = the laptop), test binary tva_a efe49685. The library md5s differ from the 2x box's
(different compiler install); every comparison below is inside one session on this box.

**Race `multi_1017` (GPU 7, 10:51-11:26 IST, free clocks, 12 rounds, order rotating; records `session_2026_10_17/box_runs/multi_1017`).**
Quick suite: F3c 296/296, C1 296/296. Paired per-round differences (median of the run; cross-checked with p10 and with
rounds 1-6 vs 7-12, all agree in sign unless said):

| pair | dense fwd | dense bwd | packed fwd | packed bwd |
|---|---|---|---|---|
| F3c vs F2 | -0.10% (t -0.7; p10 -0.28%, t -5.2) | +0.07% (null) | **-0.84% (t -11.8, 12/12)** | +0.09% (t 1.9) |
| C1 vs F2 | -0.18% (t -1.7) | +0.09% (null) | -0.79% (t -10.1, 12/12) | +0.27% (t 6.1, 0/12) |
| C1 vs F3c (= what E1 adds) | -0.07% (null) | +0.02% (null) | +0.05% (null) | **+0.18% (t 4.0, 0/12; p10 t 8.4)** |

Medians: F2 dense fwd 1.1240 ms, F3c 1.1197, C1 1.1216; packed fwd F2 0.5000, F3c 0.4959, C1 0.4961; bwd dense ~3.101,
packed F2 1.4934, F3c 1.4950, C1 1.4981.

**ncu, forward main kernel (diagnostic; this box's ncu runs it at ~1.69-1.73 GHz, so not comparable to 10-16's ncu):**
dense F3c 1,720,716 cycles, DRAM read 522 MB, L2 hit 83.2%; C1 1,718,548 cycles (-0.13%), DRAM read **449 MB (-14%)**, L2 hit
85.6%; instructions equal (318.9M), tensor busy 69.7 vs 69.8%. Packed: DRAM 258 MB both, cycles -0.15%. The hints work (DRAM
drops), but cutting 73 MB of DRAM gives ~0.1% of time: the dense forward is NOT memory-bound. Open item 4 of §45.19 (DRAM 454
vs FA3 367 MB) is therefore not a speed lever; closed.

**Verdict.** F3c is real on packed (-0.84%, robust in every check) and a small gain on dense (p10 only). E1 (the L2 cache hints,
inside C1) gives nothing to the forward and makes the NEXT kernel, the backward (identical code in all three libraries),
+0.18% slower on packed. The bench runs forward then backward back to back, so the likely cause is the L2 state the forward
leaves (Q loaded evict_first, K/V marked evict_last), not proven. Per packed forward+backward pair C1 costs ~2.9 us more than
F3c; dense is a wash. Keep F3c (the laptop tree already is F3c); E1 not taken. Next forward candidates are built on F3c.

**Fix to ab_multi.sh:** this box's ncu writes `.ncu-repz` (zstd); the script imported `.ncu-rep`, so the on-box CSVs were empty.
Now it imports whichever exists. Locally: `zstd -d` the .ncu-repz first (laptop ncu 2025.3.1 cannot read .ncu-repz).

### 45.21 2026-10-17: where the dense forward loses to FA3, from the profiles we already had (no GPU run)

The 60-70 min measurement run (deep_1017.sh: trace build, knock-outs, scoreboard) was NOT run: the user asked for analysis from the
sources and existing profiles first. Data: F3c ncu from this morning's race (`session_2026_10_17/box_runs/multi_1017`, reports are
zstd `.ncu-repz`: decompress before importing locally) and FA3's ncu from 10-13 on the SAME box (`v2_bwd_rework_research_2026_10_12/
gpu_runs/prof_1013/ncu_libs_fwd.*`, `ncu_fa3_packed_fwd.*`). Tools: `session_2026_10_17/deep/unit_blocks.py` (code cut into blocks at
every wait, cycles = sample share x kernel cycles, executed instructions per tile / per unit), `deep/region_detail.py`.

**Equal clock (ncu, same box):** dense F3c 1,720,716 cycles vs FA3 1,431,902 (+20.2%; tensor cores busy 69.7 vs 83.3%, the same
~1.195M tensor cycles in both); packed 780,913 vs 747,987 (+4.4%; we execute fewer instructions, 125.8M vs 142.2M). The free-clock
packed 1.11x of 10-16 was mostly our lower clock in that bench (~1,700 vs ~1,800 MHz). The big gap is DENSE.

**Dense gap, M cycles (ours - FA3 = +0.288):** loop +0.127, unit end +0.073, unit start +0.047, loader / other +0.040.
- Loop, per key tile (executed instructions per tile: ours 487, FA3 460). FA3's source order (mainloop_fwd_sm90_tma_gmma_ws.hpp
  1170-1206) is the same as ours since F1. The difference is the SASS: AFTER the P.V wait ours runs 328 instructions (569k cycles),
  FA3 258 (371k), +0.198M. FA3 finishes the whole row max (66 FMNMX) and starts 13 exps (33 FFMA) BEFORE the wait; ours finishes 49
  of 68 FMNMX and no exp before it. Likely causes: (a) our `if (diag)` branch inside softmax_tile splits the softmax into blocks
  ptxas cannot schedule across (FA3: separate masked / unmasked loops, no branch in the unmasked one); (b) 16 partial row sums
  (sA/sB) + 16 partial maxes vs FA3's 2 + 2 registers: register pressure makes ptxas move the P.V wait up to reuse P(n-1)'s 32
  registers. Before the wait we are equal or better (issue + wait QK + max: -0.093M).
  Loop top: both our groups wait for K/V every tile (31.6k + 26.3k); FA3 only warpgroup 0 waits (line 1175/1180), +0.022M.
- Unit end: our epilogue 109k cycles (32 STG.E of 4 B per thread: stalls long_sb 31% = waiting for stored registers, lg 29% =
  store queue full) vs FA3's ~32k (FMUL, F2FP, 8 STSM into shared memory, one named barrier, TMA store).
- Unit start: prologue softmax(0) 69k vs FA3 39k (ours latency-bound, FA3 issue-bound; ours also rescales O = 0 with 64 FMUL);
  last P.V + next-unit read 55.6k vs 32.3k.

**Fix list (all checkable on the laptop in SASS before any GPU run):** loop: peel the diagonal tile out of the loop (mask-free
loop body), row sums / maxes in few registers as FA3, only group 0 waits for K/V; unit end: O through shared memory (stmatrix) +
wide stores / TMA store (needs 16 KB per group: shared memory is full with 3 K/V stages, so reuse the last tile's V stage or go to
2 stages as FA3); unit start: no O rescale in the prologue. Target: the +0.288M.

### 45.22 2026-10-17: K1 (the loop) and K2 (the O store): what was built, and the ptxas finding

**The loop problem, in SASS** (tool `session_2026_10_17/deep/sass_shape.py`: per opcode, what sits between the wait for QK(n) and
the wait for P.V(n-1), and after it). F3c: 61 instructions before the P.V wait (part of the row max), 328 after (rest of the max,
all 64 exps, the packs, the O rescale). FA3 (its 10-13 profile): 127 before (the whole max, 33 FFMA, 13 exps), 258 after.

**What did NOT move the wait** (each compiled and checked on the laptop, CUDA 13.0 and 12.8 alike): the diagonal tile peeled
out of the loop (mask-free body); one / four / eight max chains; row-by-row softmax; 4 + 4 registers for max / sum instead of
32; two P register sets; the empty arrive moved; an independent arrive; a never-taken branch before the wait (that one SPILLED
400 B and made ptxas put a wait after every single HGMMA). Registers were not the limit (184 of 232 used in the loop). The PTX
always had the whole softmax before the wait: ptxas moves the wait up on its own.

**What moves it:** an mbarrier arrive placed before the wait whose ADDRESS depends on a value computed at the end of the
softmax. ptxas keeps the wait after it. K1 uses two barriers of count 1 that nobody waits on (`order_bar[2]`, 16 B of shared
memory; every arrive just completes a phase), index `(bits(l0 + l1) ^ pa[0] ^ pa[31]) == 0xffffffff`: both choices harmless.
Result: before the wait 314 instructions (68 FMNMX, 66 MUFU, 66 FFMA, 65 FADD, 32 F2FP = the whole softmax and the packs),
after it 95 (the 64-FMUL O rescale). FA3's SASS shows the same mechanism by accident: its K-release arrive (mainloop :1185) is
scheduled late and its P.V wait right after it.

**K1 = F3c +** (source `session_2026_10_17/candidates/fwd_K1.cu`): the order arrive; the diagonal tile peeled (the loop body
has no mask branch; the diagonal is always a group's last active tile: q-tile 2*pair + c, my_nkv - 1 = pair); two P register
sets (P(n) packed while P.V(n-1) reads the other set; loop unrolled by two, +32 registers); only group 0 waits for the K/V of
tile n (FA3 :1175 / :1180; group 1 follows group 0's baton); no O rescale in the prologue (O is 0). 0 spills, no warnings.
Variants: K1s (one P set: 279 before / 144 after), K1n2 (2 K/V stages, as FA3).

**K2 = K1n2 + the O store through shared memory** (as FA3's epilogue): normalise + convert, 8 stmatrix.x4 per thread into the
group's 16 KB block (rows of 256 B, 16-byte chunk index XOR (row & 7): conflict-free), __syncwarp, then 8 x (16-byte shared load +
16-byte global store) per thread, each warp only its own 16 rows (no barrier). Needs 32 KB of shared memory -> only with 2 K/V
stages (static_assert). SASS: 8 STSM, 8 LDS.128, 8 STG.E.128 (+2 LSE) instead of 32 STG.E. A plain uint4 store was split by the
compiler into four 4-byte stores; an explicit st.global.v4.b32 fixed it. Packed instance: 8 B spill outside the loop (setup /
epilogue), dense 0.

**Race k1_1017** (GPU 7, started 12:30:14 IST): F3c vs K1 vs K1s vs K1n2, quick suite on K1, 8 rounds, ncu after.
Note: two builds of the same source give different object md5s but identical SASS (nvcc output is not bit-reproducible),
which is why vj2_F3c's library md5 changed (522f09a5 -> d00d7209).

### 45.23 2026-10-17 afternoon: race k1_1017, the K1 timing trace, and the split softmax (K3)

**Race k1_1017** (GPU 7, 12:30-12:58 IST, 8 rounds, free clocks; K1 quick suite 296/296): vs F3c, K1 dense fwd **-0.95%**
(t -7.5, 8/8; p10 -1.26%), packed fwd 0.00%, bwd +0.28% / +0.07% (weak). K1s (one P set) +0.94% / +2.81% worse. K1n2 (2 K/V
stages) **+39% / +22% worse** (bwd -4.2% / -1.8%: a lighter forward leaves the backward more clock). Equal clock (ncu): K1 dense
1.698M cycles vs F3c 1.721M, tensor busy 70.6% vs 69.7%; FA3 1.432M / 83.3%. Packed: K1 0.781M = F3c, FA3 0.748M (+4.4%).
Two stages fail for us because one stage holds K AND V of a tile and stays busy two steps (V(n) is read in step n+1); FA3 has
separate K and V pipelines (producer: K(n) then V(n-1), mainloop :871-874), so its 2+2 stages have the slack of our 3. K2 (which
needed 2 stages for 32 KB of O staging) is therefore dropped as built. FA3 stages O in the V buffer (smem_o aliases smem_v,
`barrier_O`, :848-852).

**Stall reasons, K1 vs FA3 (dense, M cycle-equivalents):** wait +0.110, long_sb +0.095, short_sb +0.059, mio +0.054. The
long_sb excess is the PRODUCER warp spinning on empty[] / qempty (barrier offsets 0x38018 / 0x38040, out-of-line try-wait loops):
idle by design; FA3's producer sleeps (NANOSLEEP.SYNCS: its 'sleep' 0.037).

**Timing trace of K1** (library vj2_TRK1 = K1 + clock stamps, deep/make_trace.py, same SASS shape; GPU 7 14:12-14:13 IST;
records `session_2026_10_17/box_runs/trace_1017/`, reader deep/trace_read.py; the two launches per shape agree to a few cycles;
the trace build is ~3-5% slower than K1). SM clock under this load: **~1.40-1.45 GHz** on this box. Dense, per group per tile:
wait K/V + baton 118 | issue + wait QK 813 | **softmax (+pack) 1,413** | wait P.V 8 | rescale 104 | between steps 75 = **2,531**
per step; tensor-bound would be 2,048. Group 0's softmax 1,533, group 1's 1,294. Per pair: prologue 2,101 (4.5%), loop 39,245
(83.8%), last P.V + fetch 925, **epilogue 4,230 (9.0%; packed 20.8%)**, gap between pairs mean 1,862; tail 0.4% (packed 2.3%).
An in-order issue model (deep/sass_timing.py, same latencies for both) gives K1's softmax ~750 cycles ALONE (MUFU: 66 x 8) and
FA3's softmax + rescale ~850: the code is not slower alone; the measured 1,413 is interference (the two groups' softmaxes
overlap on the special-function unit / MIO; the other group's GEMM issue). The ncu issue share confirms it: K1's softmax lines
run at 22% issue (~1,400 cycles per tile), FA3's after-wait block at 34% with its 64 rescale FMULs between the exps.
A step must satisfy softmax + rescale + K/V/baton + overhead <= ~1,536 for the loop to be tensor-bound: we are at ~1,710.

**K3 = K1 + split softmax** (switch K1_SPLIT = exps before the wait): part A (mask, max, rescale factors, the first K1_SPLIT exps)
before the P.V wait, pinned by the order arrive on part A's row sums; part B (the other exps) after it, pinned there by a
bias + (O_acc[0] - O_acc[0]) (an O register, readable only after the wait: ptxas had pulled part B in front of the wait in one of
the two unrolled loop copies). Part B, the 64-FMUL rescale and the packs then interleave as in FA3. Exact splits in all 5 loop
copies, 0 spills: K3h (32) 34 MUFU before / 32 after, K3q (16) 17 / 49, K3e (8) 9 / 57. Model: K3h before ~460 (< P.V's ~512)
+ after ~310; K3q / K3e push more behind the wait. Next: race K1 vs K3h / K3q / K3e + a trace of K3h.

**O store (unit end, 4,230 cycles per pair): the plan.** Stage O through the LAST TILE'S K BUFFER: after a group's last P.V,
wait for a 'K(last) free' handshake from the other group (its QK(last) done: one mbarrier arrive after its wait-QK of the last
step), stmatrix O there, 16-byte stores out, and only then release that stage (empty arrive). 32 KB with no change to the 3
stages. Packed's inactive groups need care (a group with no active tile has no epilogue).

### 45.24 2026-10-17 15:00 IST: moved to a new 2x box; K4 (O staged in the last K buffer) and its hang

**Box:** the 8x node is busy with colleagues for ~5 h; work moved to a NEW 2x H200 box root@86.38.238.36 (tall-door-completes-fin-02,
not yesterday's .80): 80 GB free of 193 GB, CUDA 13.0, 88 cores, 334 GB RAM, H200 700 W / 1980 MHz, no GPU board; only GPU 0 may be
used. Setup = source tree without .git / lib (68 MB), labels.bin, scripts, candidates; full build ~10 min with -j88. Timings here
are compared only within this box (K1 raced again as the base).

**K4** = K1 + the O store through shared memory (K2's stmatrix + 16-byte stores), staged in the group's half of the LAST tile's K
buffer (keeps the 3 stages); that stage is released (empty arrive) only after the epilogue has read it back (proxy fence + group
barrier `bar.sync 3+c, 128`); each group signals 'my last QK of this pair is done' and waits for the other group's signal before
writing. K4h = K4 + the half split. SASS: 8 STSM + 8 LDS.128 + 8 STG.E.128 (+2 LSE) instead of 32 STG.E; spills only outside the
loop (K4 16 / 64 B).

**The hang (run kall_1017, GPU 0, 14:45 IST): K4h's quick suite timed out (exit 124, 600 s) and was dropped.** Cause: the first
handshake was an mbarrier with parity waits. An inactive group (A) signals at the very start of its pair and can run a whole pair
ahead; its second arrival completes the phase the other group is about to wait for, parity says 'not done', and that group waits
for ever (then nobody can move). The CPU protocol model (session_2026_10_17/deep/protocol_sim_k4.py, extended from 10-16's with K1
and K4 rules; 1000 random unit lists each): F3fix 0 / 0 deadlocks (dense / packed), K1 0 / 0, **K4 (mbarrier) 9 / 385**, K4b
(monotonic counter) 0 / 0. **K4b**: the signal is `st.release.cta.shared` of kcount[c] = pairs done, the wait spins on
`ld.acquire.cta.shared` until kcount[1-c] >= j + 1 (then a proxy fence): a count that only grows cannot be overtaken. K4b /
K4hb / TRK4b compiled, same loop SASS as K1 / K3h. The race kall_1017 still contains the old K4 (it may hang; each hang = 600 s).

**Race kall_1017 results** (2x box GPU 0, 14:56-15:01 IST, 8 rounds, free clocks; K4h dropped: quick suite hung = the K4
handshake bug; K3h / K3q / K4 had no quick suite; all 64 race logs complete). Forward vs K1 (all 8/8 rounds):
| | dense fwd | packed fwd | dense bwd | packed bwd |
|---|---|---|---|---|
| **K4** (O through the last K buffer) | **-5.30%** (t -30) | **-7.33%** (t -80) | +0.20% | +0.19% |
| K3h (split 32) | -1.26% (t -13) | -1.97% (t -12) | +0.10% | +0.15% |
| K3q (split 16) | -1.45% (t -24) | -0.86% (t -3) | +0.11% | +0.09% |
Medians: dense fwd K1 1.1123 ms, K3h 1.0970, K3q 1.0958, K4 1.0511; packed K1 0.4925, K3h 0.4833, K3q 0.4873, K4 0.4567. K3q vs
K3h: dense -0.19% (null), packed +1.13% (worse) -> the half split stays. The backward after a faster forward is +0.1-0.2% (power).
Expected K4hb (K4b + split 32): about -6.5% dense / -9% packed vs K1 (to be raced: K4b / K4hb with full quick suites).

**K3h trace** (vj2_TRK3h, dense, 1.44 GHz; 3 of 4 trace files copied: this box's link is ~50 KB/s): per step 2,399 cycles (K1
2,457 on the 8x box): issue + wait QK 839 | part A (max + 32 exps) 825 | wait P.V 8 | part B + rescale + packs 601 | K/V + baton
125. Softmax + rescale 1,426 vs K1's 1,517; still ~1.8x the single-warp model (the two groups' softmaxes overlap ~400 cycles per
step on the same SM sub-partitions). Per pair: epilogue 3,864 dense (8.4%) / 4,874 packed (22.5%), prologue 4.3% / 9.2%, gap
between pairs 1,262 / 1,841 of which fetch_unit 568 (dense): the producer publishes the next unit only after issuing all loads
of the current one (and those wait for stage releases); FA3 claims its next tile earlier -> next lever after K4.

### 45.25 2026-10-17 15:10-15:16 IST: K4b (fixed handshake) is the new best forward

Run k4b_1017 (2x box, GPU 0; sources K4b 936de1d7, K4hb 877a9a72, TRK4b 57f8e0d2; records box_runs/2x_1017/k4b_1017,
k4b_results.tgz): **quick suite K4b 296/296, K4hb 296/296** (40 s / 37 s on this box), no hang. 8 rounds, free clocks:
| vs K1 | dense fwd | packed fwd | dense bwd | packed bwd |
|---|---|---|---|---|
| **K4b** | **-5.26%** (t -37, 8/8) | **-7.37%** (t -73, 8/8) | +0.51% (t 3.0) | +0.01% |
| K4hb (K4b + split 32) | -2.58% | **+8.38% worse** | +0.36% | -0.89% |
Medians: dense fwd K1 1.1144 ms, K4b 1.0558, K4hb 1.0861; packed K1 0.4962, K4b 0.4599, K4hb 0.5383. K4b reproduces K4
(-5.30 / -7.33% in kall_1017) with a correct handshake. The split does not stack on K4 yet: K4hb spills ~6x more than K4b
(108 / 76 B vs 16 / 64 B) at every unit's start and end, and packed (many short units) pays +17% vs K4b. Per forward + backward
pair: dense -58 us fwd, +16 us bwd (-1.0% of fwd+bwd), packed -36 / +1 us (-1.8%).
**Chain today (forward, free clocks):** F3c -> K1 -0.95% dense / 0% packed (8x box) -> K4b -5.26% / -7.37% (2x box): about
**-6.2% dense / -7.4% packed vs F3c**. Not yet measured against FA3 on one box (the rivals' environment is on the 8x box).
Trace TRK4b: like TRK4 it spills in exactly the epilogue it should measure (100-120 B), so its epilogue (4,028 / 4,447 cycles) is
not K4b's; the loop is unchanged (2,429 cycles per step, softmax 1,373). An uninstrumented ncu of K4b would show the real unit end.
**Next, in order:** (1) the user's call: K4b into the laptop master; (2) cut K4's register pressure at the unit start / end so
the split (-1.3 / -2.0% alone) stacks; (3) unit boundaries: the next unit's index published earlier, packed: prefetch the unit
descriptor during the last P.V; (4) a fair scoreboard vs FA3 / FA4 / cuDNN once the 8x box is free.

### 45.26 2026-10-17 15:33-15:41 IST: K5 / K5q / K6 (single P set + split on top of K4b) -> K5q is the new best

Run k5_1017 (2x box GPU 0; libs K5 f3a7f34a, K5q 8dc36782, K6 534c44cc; quick suite 296/296 each; 8 rounds; records
box_runs/2x_1017/k5_results.tgz). K5 = K4b + split 32 + ONE P register set (with the split all of P(n) is packed after the P.V wait,
so the second set only cost registers: K5 / K5q 0 B spills vs K4b 16 / 64 B, K4hb 108 / 76 B; no 2x unrolled loop). K5q = split 16.
K6 = K5 + design A: the next unit's descriptor loaded during the last P.V.
| vs K4b | dense fwd | packed fwd | dense bwd | packed bwd |
|---|---|---|---|---|
| **K5q** | -0.36% (t -2.4, 7/8) | **-4.33%** (t -64, 8/8) | -0.04% | +0.20% |
| K5 | +0.84% (t +6.4) | -4.63% (t -30) | -0.23% | +0.18% |
| K6 | +0.95% | -4.20% (+0.46% vs K5: prefetch not kept) | -0.19% | +0.16% |
Medians: dense fwd K4b 1.0582, K5 1.0681, K5q 1.0552, K6 1.0684 ms; packed K4b 0.4606, K5 0.4389, K5q 0.4406, K6 0.4411. In dense
the smaller split is better (K5q vs K5 -1.20%, t -13.6); packed gains come mostly from the removed spills (many short units).
**Chain, forward:** K5q vs K1 about -5.6% dense / -11.4% packed; vs F3c about -6.5% / -11.4%.

**Spill-free trace (TRK5, 0 B dense):** per pair dense: epilogue **1,411 cycles (3.2%)** (K1-era 4,230 = 9%), prologue 2,029 (4.6%),
loop 38,869 (88.5%), last P.V + fetch 910, gap 935; packed: epilogue 2,058 (11.2%, was 22.5%), prologue 1,949 (10.6%), loop
71.0%, gap 512. Loop step: dense 2,401 cycles (kv/baton 118 | issue + wait QK 892 | part A 792 | wait P.V 8 | part B + rescale +
packs 591), packed 2,445; tensor-bound would be 2,048. The O-store fix removed ~2/3 of the unit end; **the dense gap to FA3 is now
the loop step** (88.5% of the time; equal-clock estimate K5q 1.60M cycles vs FA3 1.43M). Next: cut the post-QK work per step
(FA4-style conditional rescale: skip the 64-FMUL O rescale unless a row max grows by > 8 in log2 units; a smaller split; the
per-step stage broadcast on the critical path at the loop top).

### 45.27 2026-10-17 15:59-16:08 IST: split 8 and FA4's conditional rescale (K5e / K7 / K7e)

Run k7_1017 (2x box GPU 0; quick suite 296/296 for K5e, K7, K7e; 8 rounds vs K5q; records box_runs/2x_1017/k7_results.tgz).
K7 = K5q + FA4's conditional rescale (a warp keeps its row maxima unless one grows by > 8 in log2 units; P then <= 2^8; the
64-FMUL O rescale and its 2 exps run only when needed; LSE exact with the max used). K5e / K7e = split 8.
| vs K5q | dense fwd | packed fwd |
|---|---|---|
| K5e | +1.22% (t +18) | +0.39% |
| K7 | -0.49% (t -3.8, 7/8) | +0.48% (t +3.2) |
| **K7e** | **-0.81%** (t -13.6, 8/8) | +0.63% (t +5.1) |
Backward unchanged (within +-0.24%). Medians: dense K5q 1.0476, K5e 1.0615, K7 1.0436, K7e 1.0403 ms; packed K5q 0.4384,
K5e 0.4396, K7 0.4405, K7e 0.4412. The conditional rescale pays in dense (long rows: the max settles) and costs in packed (short
chats: maxima keep growing; the vote and branch remain). Smaller than expected: dropping the 64 FMULs barely shortens the step's
critical path. Best per mode: dense K7e, packed K5q (a per-MODE build: about -0.8% dense, 0 packed).

### 45.28 2026-10-17 16:21-16:24 IST: K8 / K9 / K9h rejected (they break the packed == alone bit-exact checks)

Run k9_1017 (records box_runs/2x_1017/k9_results.tgz): quick suite **K8 278/296, K9 277/296, K9h 274/296 -> all dropped**; the race
ran K5q only. K8 = per-mode build (dense split 8 + conditional rescale = K7e's SASS, packed split 16 = K5q's SASS, both verified
instruction for instruction on the laptop); K9 / K9h = K8 + every 4th / 2nd exp by a degree-3 polynomial (max rel. error 7.5e-5).
**Every failed check is a BIT-EXACT one:** G10 / G11 "every segment bitwise = alone" (a chat in a packed row must give exactly the
O / LSE / dQ / dK / dV of the same chat run alone through the DENSE kernel), and for K9h also G6c "filler independence". The
differences are rounding-sized (worst dQ diff 0.25-0.5 of atol), not wrong results. Cause: with per-mode settings the dense
reference and the packed kernel compute the softmax differently, so packed != alone bit for bit; the polynomial adds a few more
position-dependent differences (G11 D5, G6c). **A per-mode numerics split is incompatible with the bit-exact guarantee that
proves packing correct; kept rule: dense and packed must use the same softmax arithmetic.** Usable (296/296): K5q (best for packed
SFT: -4.3% packed vs K4b, dense about equal) and K7e (dense -0.8% vs K5q, packed +0.6%).

### 45.29 2026-10-17 16:46-16:52 IST: what limits the dense forward step (knock-outs + per-step timeline)

Run ko_1017 (2x box GPU 0; records box_runs/2x_1017/ko_results.tgz, unpacked to ko_1017/ and trace_1017_vj2_TLK5q/). Knock-out
builds = K5q minus one part of the softmax (results WRONG on purpose, so forward-only timing, VARLEN_BENCH_DIR=fwd, no quick suite),
6 rounds, 4000 dense / 8000 packed forwards per run, every candidate faster in 6 of 6 rounds:

| removed from K5q | dense fwd | packed fwd |
|---|---|---|
| the 64 exps per tile (KOexp: P = the exp's argument, no MUFU) | **-6.76%** (t -41) | **-6.83%** (t -38) |
| the row max (KOmax: no FMNMX, no shuffles) | -4.12% (t -18) | -3.52% (t -53) |
| the O rescale (KOres: no 64 FMUL) | -2.38% (t -15) | -2.18% (t -29) |
| all three (KOall) | **-11.18%** (t -73) | **-11.58%** (t -97) |

Medians (ms): dense K5q 1.0867, KOexp 1.0135, KOmax 1.0396, KOres 1.0599, KOall 0.9637; packed 0.4571 / 0.4260 / 0.4409 / 0.4472 /
0.4040. The parts do not add up (6.8 + 4.1 + 2.4 = 13.3 > 11.2): when one is removed, the next limit shows. So the softmax IS on
the critical path; if it were hidden under the other group's GEMMs, removing it would change nothing.

Timeline TLK5q (K5q + clock stamps at 6 points of every loop step, both groups, first pair of every CTA; 0 spills; SM clock 1.45
GHz during the traced launch = power-limited). Steady state, dense, cycles per step (period 2,706 in this build):

| | wait K/V + baton | issue + wait QK | softmax part A | wait P.V | part B (exps + rescale + pack) |
|---|---|---|---|---|---|
| group 0 | 301 | 675 | 640 | 8 | **856** |
| group 1 | 203 | 1079 | 623 | 8 | 529 |

What it shows:
1. **The P.V wait is 8 cycles**: P.V(n-1) is finished long before part A ends. The tensor cores have no queued work ~24% of the loop.
2. **Group 1 gets the baton 716 cycles after group 0**: issuing the 16 HGMMAs blocks (the tensor-core queue holds about one 8-HGMMA
   group), so group 0's baton arrive comes only when its QK is nearly done. Same structure as FA3 (its arrive is also after both issues).
3. **Group 0's part B takes 856 cycles, group 1's 529**: group 0's part B runs while group 1's part A runs (both use the MUFU unit
   and the issue slots of the same 4 sub-partitions); group 1's part B runs alone. Part B is mostly MUFU time even alone (49 MUFU x
   8 cycles = 392 of 529).
4. **The step-start wait is bimodal**: group 0 waits either ~100-150 or ~275-300 cycles and almost never in between (dense: the long
   one on 3,799 of 3,972 steps; packed: 193 short / 130 long, nothing in 150-275). The values repeat to +-1 cycle on every SM.
   A wait for memory would spread; a fixed cost after a FAILED first check of the barrier would not.
   Caveat: the stamps are S2UR clock reads, which can themselves be delayed in a busy queue; strong evidence, not proof.

Static check of the softmax code itself (deep/sass_timing.py, one warp alone, same latency model for both): K5q part A 132
instructions -> 350 cycles, part B 250 -> 426 (sum 776); FA3's unmasked loop: 127 -> 403 and 257 -> 447 (sum 850). **Our
softmax instruction stream is now as short as FA3's**; the remaining gap is in how the two groups' work overlaps and in the waits.

**The wait code differs from FA3's.** Our mbar_wait = try_wait WITHOUT a suspend-time hint; the SASS is a TRYWAIT loop.
CUTLASS (cutlass/arch/barrier.h ClusterBarrier::wait, used by FA3's pipelines) passes 0x989680; the SASS is TRYWAIT, then
`NANOSLEEP.SYNCS 0x989680` (a sleep the barrier event ends), then a re-check. FA3, FA4 (cute DSL) and cuDNN all use that form
(ncu 2026-10-13 SASS: 21 / 14 / 6 NANOSLEEP.SYNCS). **FA3's in-loop K and V waits ran 285,696 first checks and never took the retry
path**: its data is always there when it looks. Ours misses ~95% of the time in dense. FA3 also keeps K and V on SEPARATE
barriers (pipeline_k / pipeline_v): its QK(n) waits for K(n) only, while ours waits for all 64 KB of K(n) + V(n) although V(n) is
first read one step later.

Correction to 45.28: K9h's G6c failure has a concrete cause: ex2_poly clamped x to -126, so a MASKED key got P = 2^-126 instead of
0 and the filler's V leaked into O (by ~1e-38). Fixed in the master: x <= -127 returns exactly 0 (checked in Python: -inf -> 0,
max rel. error on [-120, 8] still 7.5e-5). The bit-exact failures of K8 / K9 are still the per-mode split (45.28).

Next run (run_w.sh, built and SASS-checked on the laptop, 0 spills, K5q's SASS unchanged by the new switches):
- **W1** = K5q + CUTLASS's wait (W1_WAIT 1: try_wait + 0x989680 -> NANOSLEEP.SYNCS, as FA3)
- **W1s** = K5q + test_wait spin (W1_WAIT 2: never sleeps)
- **W2** = W1 + separate K and V barriers (W2_KV 1): group 0 waits only for K(n) at the step start; both groups wait V(n-1)
  between the QK and P.V issues (FA3's place, hidden under QK). The producer still loads K and V of a tile together and the stage
  is still released after P.V, so the producer has no new wait (no new deadlock); every place that waited full[s] before
  releasing stage s also waits full_v[s]. SASS: 8 QK HGMMAs, the V check, 8 P.V HGMMAs, groups intact. +24 B shared memory (229,536 B)
- **P8w / P4w** = W1 + every 8th / 4th exp by the fixed polynomial (MUFU per tile 66 -> 58 / 50; same code in dense and packed,
  so packed == alone stays bit-exact by construction; the quick suite checks it)
- **TLW2** = W2 + timeline (spills 44 B only at the pair start and in the epilogue, none in the loop)

### 45.30 2026-10-17: checkpoint questions (CRC32, our 22.2 GB SFT checkpoint, what DPO / a resume needs)

- **CRC32** = cyclic redundancy check, 32 bits: a fingerprint of a tensor's bytes. DCP stores one per tensor in metadata.bin
  (dist/Distributed_Checkpointing/include/Checksums.h). BLU_INIT_VERIFY=1 recomputes it on the weights after they land in GPU
  memory: 262/262 matched for Navin's step 8000 (any flipped bit changes it).
- **Our SFT checkpoint** stage0_adamw_full/ckpt/blumodel_step_156 (10-07 run: 5,000 chats, 2 epochs, 156 steps, AdamW, 1 GPU):
  same DCP format as Navin's: metadata.bin + ONE data_rank_0.bin (1 GPU). Holds the weights (same 262 tensors) + the full AdamW
  state (fp32 master + m + v = 12 B per parameter) + step and loss = 22.2 GB. Navin's is 16.35 GB because Muon keeps less state
  per parameter. The step-78 checkpoint was deleted; step 156 is the END of that run (156 of 156).
- **DPO or any next stage needs the weights only.** BLU_INIT_FROM=<step folder> loads only the weight tensors (param_NNNNNN) by
  name from metadata.bin and reads only their bytes; the optimizer records stay on disk unread (DCP load with optimizer = nullptr:
  LoadPlanning.cpp build_load_items returns before the optimizer part). No preprocessing or stripping is needed. It also restores
  the saved random-number-generator state (DistributedCheckpointing.cpp load). The new run starts at step 0 with a new AdamW;
  its fp32 master copy is built from the bf16 weights (each weight differs from the SFT run's master by at most one bf16 rounding).
  DPO uses the SFT weights twice: the trained policy and the frozen reference (its log-probs can be computed once before training).
  bluscript_sft.cpp has no DPO loss yet.
- **Continuing the same SFT run needs weights + optimizer state + step.** That is NOT BLU_INIT_FROM: it is the trainer's
  auto-resume. BLU_CKPT_DIR=<folder that holds blumodel_step_N> -> the newest complete checkpoint is loaded with its optimizer
  state and step, the bf16 weights are rebuilt from the fp32 masters, and the chats already used are skipped
  (bluscript_sft.cpp: auto-resume block). A wrong folder gives no error: the run silently starts at step 0.
- **Trap for a new stage**: give it a NEW, empty BLU_CKPT_DIR. If its BLU_CKPT_DIR holds an older checkpoint, auto-resume loads
  that one (weights + optimizer + step) and overwrites what BLU_INIT_FROM loaded.
- Resuming step 156 with the same settings runs 0 steps (the run is complete). A 3rd epoch (BLU_SFT_EPOCHS=3, 234 steps) re-stretches
  the cosine: the LR jumps from 0.10x peak (where the run ended) to 0.33x peak at step 156.

### 45.31 2026-10-17 17:57-18:12 IST: the wait form, separate K/V barriers and the polynomial exp -- all rejected

Run w_1017 (2x box GPU 0; records box_runs/2x_1017/w_results.tgz). Quick suite **296/296 for all five** (W1, W1s, W2, P8w, P4w:
the fixed polynomial keeps packed == alone bit-exact, as expected with the same code in both modes). Forward-only race, 8 rounds:

| vs K5q | dense fwd | packed fwd | verdict |
|---|---|---|---|
| W1 = CUTLASS's wait (try_wait + hint -> NANOSLEEP.SYNCS, as FA3) | -0.07% (t -0.8, 4/8) | -0.04% (t -0.8) | null |
| W1s = test_wait spin | +0.14% | **+0.44%** (t +5.5, 0/8) | worse |
| W2 = W1 + separate K / V barriers | -0.11% (t -0.6; p10 -0.56%) | +0.10% | null |
| P8w = W1 + every 8th exp by the polynomial | **+2.69%** (t +15) | **+3.40%** (t +53) | worse |
| P4w = W1 + every 4th exp by the polynomial | **+7.07%** (t +37) | **+7.44%** (t +104) | worse |

Medians (ms): dense K5q 1.1008, W1 1.1010, W1s 1.1019, W2 1.0993, P8w 1.1289, P4w 1.1772; packed 0.4583 / 0.4582 / 0.4607 /
0.4594 / 0.4744 / 0.4926.

What the trace of W2 (TLW2) shows: group 0's step-start wait did get shorter (bimodal ~100-125 / **~225-275**, was ~100-150 /
~275-300), but the time moved into the next phases (part A 640 -> 748, issue + wait QK 675 -> 711 cycles) and the period stayed
the same. **So the step-start wait of 45.29 is slack, not the cause**: something else sets the loop period, and the wait only
shows where the spare time ends up. The 45.29 hypothesis (fix the wait -> faster step) is refuted by measurement.

The polynomial result says what the limit is: each polynomial exp is ~11 FP32 / integer instructions instead of 1 MUFU. Replacing
8 MUFU per tile with ~88 instructions costs +2.7%; 16 with ~176 costs +7.1% (~0.03-0.04% per added instruction per tile, the
same as the rescale knock-out: 64 FMUL = 2.3%). **On H200 the loop is limited by instruction issue in the softmax phases (the two
consumer groups share each sub-partition's one-instruction-per-cycle issue), not by the exp unit.** FA4's trick pays on B200, whose
tensor cores are twice as fast relative to the exp unit; not here. What remains for the dense forward: fewer instructions per
tile (the softmax stream is already as short as FA3's in the static model, 45.29; our loop carries ~40 more non-math
instructions per tile: order_after, loop control, descriptors), and a same-box FA3 measurement to fix the real size of the gap
(the ~7-8% is chained across boxes and days).

### 45.32 2026-10-17 evening: why the dense forward still loses to FA3 (data we already had)

1. **Not the clock.** Same-box fair scoreboard 10-16 (8x box, free clocks, sbfree_1016): ours 1,665-1,680 MHz at 692 W, FA3
   1,650-1,665 MHz at 693 W. Same clock, same power cap: the 1.17x of that day was extra CYCLES. (The two boxes are equivalent:
   K1 dense fwd 1.1127 ms on the 8x box, 1.1123 on the 2x box.) Chained to today: K5q / F2 = 0.939 -> about 1.10x FA3's time.
2. **Not extra work.** ncu at locked clock (dense B8 18/6): FA3 (10-13) 306.5M warp instructions, 1.42M cycles; our K1 (10-17)
   311.9M instructions (+1.8%), 1.68M cycles (+18%). Same work, more waiting: issue active 41.1% (FA3) vs 35.4% (K1) of cycles,
   tensor pipe busy 83.3% vs 70.6%.
3. **Where K1's warps wait more** (stall samples, in units where FA3's whole kernel = 100): fixed-latency waits +7.5, memory-result
   waits (long scoreboard) +6.5, exp / shuffle / shared-memory results (short scoreboard) +4.0, MIO queue full +3.7, global-store
   queue full +1.8, named barriers +1.3; FA3 waits more at warpgroup_arrive (-4.8) and sleeps (-2.6).
4. **Where in the code**: by executed-count groups (same sampling, so samples = warp-time), K1's main loop held 72,023 samples
   vs FA3's 68,742 (+5%), but the per-pair code (pair start, prologue, diagonal tile, epilogue, waits) held ~31,300 vs ~18,700
   (+67%): **~77% of K1's excess was at pair start / end, not in the loop.** Hottest single spots in K1: the instruction after the
   epilogue's 32 four-byte O stores per thread (2,011 samples: the store queue was full and that instruction had to wait until
   the stores had read its register), the pair-start stage shuffle stuck behind that store traffic (1,092), the epilogue's row-sum
   shuffles (1,507). K2/K4b (stmatrix + 8 sixteen-byte stores per thread) targeted exactly this and gave -5.3% dense.
5. **Still different from FA3 (K5q):** (a) FA3 writes O of a dense tile with ONE TMA store issued by one thread (asynchronous; the
   warps go on at once) and keeps per-row stores only for varlen; ours still issues 8 x 16 B stores + 8 shared loads per thread;
   (b) our pair start waits for Q (~160 cycles) and tile 0's K/V (~430) per pair (trace TLK5q); (c) ~44 more non-math instructions
   per loop step (descriptor setup, the ordering arrive, two predicated barriers instead of one).
6. K4b / K5q have not been profiled with ncu yet; the next measurement is one ncu run of K5q (dense + packed, ~3 min) to repeat
   this split for the current kernel before choosing the fix (TMA store for the dense O, earlier Q / tile-0 prefetch, fewer
   instructions per step). Changing how O is stored does not change any value, so packed == alone stays bit-exact.

### 45.33 2026-10-17 18:34-18:41 IST: our SFT loss vs PyTorch / torch.compile / Liger / Apple CCE

Run loss_1017 (2x box GPU 0; records profiling_reports/sft_loss_vs_rivals_2026_10_17/: scripts, box_runs/loss_results.tgz;
tables: `python3 loss_compare.py box_runs/loss_1017`). Ours = the trainer's pair (forward save_stats + backward with stats),
bench_sparse_ce_sft.cu of 10-09 unchanged, 2000 warm-up + 500 timed calls per pass. Others = loss_rivals_1017.py: torch
2.14.1 + cu130, triton 3.8.0, liger-kernel 0.8.4 (its Triton kernels; the new CuTe-DSL backend was not installed),
cut-cross-entropy 25.1.1; >= 4 s warm-up, then 200 (part 1) / 30 (part 2) timed calls each. SM clock median 1,980 MHz in both
phases, power median 579 W (ours) / 624 W (others): memory-bound, not power-limited.

**Same inputs, same answers.** The logits are the C++ bench's hash fill (checked bit for bit at 2,004 points) and the micro-batches
are ours (0 / 287 / 458 / 415): PyTorch fp32's loss equals ours to 6 decimals in all 5 trained cases (median 12.740235 both).
torch.compile and Liger gradients = PyTorch fp32's at the checked row (max diff 0); PyTorch bf16 4e-6.

Part 1, loss only (logits given), F+B in ms (the like-for-like number; ours / theirs in brackets):

| case (trained rows) | ours | Liger | torch.compile | PyTorch bf16 | PyTorch fp32 (HF) |
|---|---|---|---|---|---|
| real_min (8.9%) | 1.232 | 1.206 (ours 1.02x) | 3.896 | 6.226 | 21.137 |
| real_first (18.7%) | 1.683 | 1.426 (1.18x) | 3.897 | 6.230 | 21.138 |
| **real_median (23.1%)** | **1.885** | **1.525 (1.24x)** | 3.917 | 6.231 | 21.140 |
| real_max (38.9%) | 2.597 | 1.886 (1.38x) | 3.921 | 6.234 | 21.151 |
| nomask (100%) | 5.387 | 3.279 (1.64x) | 3.964 (1.36x) | 6.266 | 21.171 |
| all_masked (0%) | 0.859 | 1.004 (0.86x) | 3.896 | 6.220 | 21.135 |

Memory above the logits: ours keeps 0.25 MiB from F to B (row max + sum), peak +3,144 MiB (the separate gradient buffer);
Liger 0 / +0.3 MiB (it writes the gradient INTO the logits during the forward); torch.compile 0.3 MiB (it also saves only the
row LSE) / +3,144; PyTorch bf16 3,144 / 9,432; PyTorch fp32 6,288 / 18,864.

Verdict: ours beats every PyTorch variant at SFT masking (2.1-3.2x torch.compile, 3.3-5x bf16, 8-17x fp32 at 9-39% trained) and
is the fastest with everything masked, but **Liger is faster in every trained case (median batch 1.525 vs 1.885 ms, -19%)**, and
with no masking torch.compile is also faster (3.96 vs 5.39 ms).

Why, with the same bytes: Liger moves exactly what ours moves (trained rows read twice, every gradient row written once: 4.82 GB
at the median batch) in 1.525 ms = 3.2 TB/s; ours needs 1.885 ms = 2.6 TB/s. The loss is in our BACKWARD on trained rows: no-mask
B = 4.36 ms for 6.6 GB = 1.5 TB/s (31% of HBM peak), while torch.compile's backward does the same reads and writes in 2.01 ms
(3.3 TB/s). The code (LossKernels.cu sparseCENormalize_from_stats): one 256-thread block per row, a loop that loads one 16-byte
vector, computes, stores, and only then loads the next: one load in flight per thread, too few to keep HBM busy. Masked rows are
zeroed with 2-byte stores. The forward is fine (no-mask 1.03 ms for 3.3 GB = 3.2 TB/s; zero-writing alone runs at 4.0 TB/s).

Part 2, lm_head (X [32768, 2304] x W^T, d_model 2304) + loss, F+B in ms (ours = cuBLAS GEMMs measured in the same run + ours'
loss pair; our trainer uses our BLAS at about cuBLAS speed). cuBLAS GEMMs alone: F 11.80 (644 TFLOPS), B 22.89, F+B 34.44.

| case | ours | Liger fused linear CE | CCE (default) | CCE exact | PyTorch (GEMM + fp32 CE) |
|---|---|---|---|---|---|
| real_median (23.1%) | 36.32 | 33.67 (-7%) | **31.68 (-13%)** | 31.23 | 49.82 (+37%) |
| real_max (38.9%) | 37.03 | 35.66 (-4%) | 52.37 (+41%) | 51.68 | 50.43 (+36%) |
| nomask (100%) | 39.82 | 39.01 (-2%) | 135.58 (+240%) | 133.03 | 52.63 (+32%) |

Peak memory above X and W: ours 6,653 MiB (logits + their gradient + dX + dW), Liger fused 563, CCE 367, PyTorch 18,864.
CCE wins at the median batch only because it skips the masked tokens INSIDE the lm_head: its forward is 3.9 ms vs cuBLAS's
11.8 ms for all rows; its own kernels are about half cuBLAS's speed (backward ~315 TFLOPS), which is why it loses badly at
39% / 100% trained. **Biggest SFT lever found: run the lm_head + loss only on the trained positions** (gather the hidden states of
the trained tokens, GEMM with our BLAS, loss, scatter dX back). Estimate at the median batch: GEMMs 23% of 34.4 = ~8 ms + loss
~1 ms + gather / scatter ~0.2 ms = ~9.5 ms instead of 36.3 ms per micro-batch, ~0.2 s per 8-micro-batch step (roughly 4-5% of
a step, estimate; not measured).

Fix list, in order of value (none started; the user decides): (1) trainer: lm_head + loss on trained positions only;
(2) backward kernel: several 16-byte loads in flight per thread (unroll 4-8) and 16-byte zero stores, target ~3.3 TB/s like
torch.compile's (median B 1.61 -> ~1.1 ms, F+B ~1.4 ms, below Liger's 1.525); (3) write the gradient into the logits buffer
(-3.1 GiB peak, Liger's memory). Not compared: Liger's CuTe-DSL backend (needs nvidia-cutlass-dsl).

### 45.34 2026-10-17: decisions on the loss findings + the order of the remaining work

**FUTURE TASK (after the v1 SFT pipeline; user decision 10-17): logits only at the trained positions (= Q11, planned in §33).**
Today the lm_head GEMM computes logits for all 32,768 positions of a micro-batch and only the loss kernel skips the masked
rows, so the GEMM work for prompt (and, in v1, pad) positions is spent and thrown away: ~77% of the lm_head work at the median
stage-0 batch (§37 arithmetic). Packing (v2) removes the pads, but the prompt positions' logits are still wasted. Plan (§33,
unchanged): gather the hidden states of the trained positions -> lm_head GEMM on [M x 2304] -> our loss -> scatter dX back
(zeros elsewhere); round M up to a multiple of 256 (filler label 65535) because BluBLAS silently gives wrong numbers for a
transposed operand when M or N is not a multiple of 64 (memory: FUTURE BluBLAS transposed GEMM at M/N not %64). New evidence
(45.33): Apple CCE does exactly this skip and beats our GEMM + loss by 13% at the median batch although its own kernels run at
about half of cuBLAS's speed; with our BLAS the estimate is ~36 -> ~9.5 ms per micro-batch. Not bit-identical to today (the sums
run in a different order); measure in all 3 stages, with and without (6 cases, §33).

**Loss kernel, to do after the attention work (user decision 10-17):**
- (2) backward `sparseCENormalize_from_stats`: keep 4-8 independent 16-byte loads in flight per thread (today one: the loop
  loads, computes, stores, then loads the next) and zero the masked rows with 16-byte stores (today 2-byte). This is NOT the
  §28 vectorisation (that made each access 16 bytes, for correctness); it is how many accesses wait at once. Never tried
  before (10-09 noted "~33% of HBM peak, never profiled"). Same arithmetic per element, so the gradient must stay bit-identical:
  check bitwise against today's kernel. Target ~3.3 TB/s (torch.compile's backward): median B 1.61 -> ~1.1 ms, F+B ~1.4 ms.
- (3) gradient in place: write dL/dlogits into the logits buffer instead of a new one. Today LossBackward.cpp allocates
  `grad_logits_2d = Tensor::empty(logits_2d.shape(), opts)`: during the loss backward both the logits (3.1 GiB) and the gradient
  (3.1 GiB) exist, at the moment the step's memory is highest (all saved activations are still alive). Each gradient value needs
  only its own logit (+ the row's max, sum, target) and nothing reads the logits after the loss backward (the lm_head backward
  needs X, W and the gradient), so overwriting is safe; -3.1 GiB peak, no speed change. Liger does this (45.33: peak +0.3 MiB vs
  ours +3,144 MiB). Conditions: no metric may read the logits after the backward; logits and gradient same dtype (bf16 both).

**Agreed order (10-17 evening):** 1. attention forward: the K5q ncu profile -> fix -> race; 2. same-box attention benchmark
(forward, backward, forward + backward vs FA3 / FA4 / cuDNN); 3. loss kernel (2) then (3), measured with the 45.33 scripts;
4. wrap up. Item (1) above after the v1 pipeline.

### 45.35 2026-10-17 ~19:15 IST: loss backward fix (item 2) -- the real cause was 2-byte memory accesses

Done on the laptop (Tensor-Implementations/src/Kernels/cuda/loss/LossKernels.cu, md5 27d06bef; git diff +~70 lines), box test pending.
- **Root cause (SASS, nvcc 13.0, sm_90a):** in `sparseCENormalize_from_stats<bfloat16_t, uint16_t>` -- the backward the trainers run
  -- every "16-byte vector" access was EIGHT 2-byte accesses: 8 x `LDG.E.U16` and 8 x `STG.E.U16` per vector, no `LDG.E.128` at all.
  `bfloat16_t` (and `float16_t`) has a user-written copy constructor (Types.h:451), so `LossVec<bfloat16_t>` is not trivially
  copyable and the compiler copies it element by element. The float instantiation was one `LDG.E.128` / `STG.E.128`. So the §28
  vectorisation never took effect for 2-byte types in this kernel (§28 fixed correctness; nobody checked the SASS for the width).
- **Fix:** `loss_ld16` / `loss_st16` move the 16 bytes as one `uint4` (memcpy only moves registers); masked rows are zeroed with
  `uint4` stores; the trained-row loop is unrolled by `LOSS_BWD_UNROLL` (default 4) so several loads are in flight per thread.
  SASS after: 5 `LDG.E.128` + 6 `STG.E.128` in the bf16 kernel, no local memory. The arithmetic of every element is unchanged.
- **Bit-identical, checked on the laptop GPU (RTX 3060):** old kernel (git HEAD) vs new, extracted into one test
  (scratchpad lossk/test_bwd.cu): 20 configurations -- bf16 and fp32; V = 50,304 / 50,257 (unaligned rows) / 4,096 / 13;
  0%, ~23%, 100% trained; masking on / off -- 0 differing bytes in all. RTX 3060 speed: fully trained bf16 V 50,304 0.912 -> 0.624 ms
  (1.46x, 63% -> 92% of that GPU's bandwidth); masked-heavy cases about equal (zero writes were already at its bandwidth); V = 13
  0.8x (tiny rows; not our case).
- **Who gains:** every trainer that calls `autograd::sparse_cross_entropy_loss` (bluscript_sft, bluscript_zero_ckpt,
  bluscript_single): SFT and pretraining, bf16 and fp32 logits. Largest kernel gain where every row is trained (pretraining):
  the H200 no-mask backward was 4.36 ms vs torch.compile's 2.01 ms. In a whole training step the loss is small (~2-5 ms of a
  ~600 ms micro-batch), so the step-level gain is below 1%; the kernel-level gain is large.
- **H200 test (run_loss_v2.sh):** rebuild the library with the new file, then bench_sparse_ce_sft_v2.cu (= the 10-09 bench + a
  checksum of the trainer pair's gradient per case) with OLD (libs/vj2_K5q) and NEW library: old, new, old, new; summary = B and
  F+B per case + whether all 4 gradient checksums are equal.
- **Same type problem elsewhere (later, not now):** the bf16 forward `sparse_ce_forward_kernel_vec_save_stats` copies 16 bytes into
  shared memory with cp.async (fine) but reads them back as 8 x `LDS.U16`; it already runs at 66% of HBM peak.

### 45.36 2026-10-17 19:19-19:27 IST: the loss backward fix on the H200 -- bit-identical, 1.6x at the median batch, now ahead of Liger

Run loss_v2_1017 (2x box GPU 0; records sft_loss_vs_rivals_2026_10_17/box_runs/loss_v2_results.tgz, corrected summary in
loss_v2_1017/summary_fixed.txt). Library rebuilt from the tree with the new LossKernels.cu (27d06bef) -> libtensor cd0bb2a9; old =
libs/vj2_K5q (8dc36782). Same bench binary (bench_sparse_ce_sft_v2.cu = the 10-09 bench + a checksum of the trainer pair's gradient
per case), old / new / old / new, 2000 warm-up + 500 timed calls per pass; SM 1,968 MHz average, power 601 W average (not capped).

**Gradient bit-identical in all 6 cases, all 4 runs** (each case's 4 checksums equal; all_masked = 0, all zeros). The status line
"5 of 6" was a bug in the summary script, not in the kernel: the bench prints a case's checksum just BEFORE that case's header, the
parser gave each checksum to the previous case and crashed on the last (fixed in run_loss_v2.sh; the per-case pairing was rechecked).

| case (trained) | B old -> new (ms) | B faster | B bandwidth old -> new | F+B old -> new (ms) | F+B faster | Liger F+B | torch.compile F+B |
|---|---|---|---|---|---|---|---|
| real_min (8.9%) | 1.099 -> 0.883 | 1.24x | | 1.233 -> 1.013 | 1.22x | 1.206 | 3.896 |
| real_first (18.7%) | 1.450 -> 0.965 | 1.50x | | 1.685 -> 1.196 | 1.41x | 1.426 | 3.897 |
| **real_median (23.1%)** | **1.607 -> 1.008** | **1.60x** | 2.53 -> 4.03 TB/s | **1.883 -> 1.281** | **1.47x** | 1.525 | 3.917 |
| real_max (38.8%) | 2.169 -> 1.148 | 1.89x | 2.11 -> 3.99 TB/s | 2.596 -> 1.571 | 1.65x | 1.885 | 3.921 |
| nomask (100%, pretraining) | 4.362 -> 1.645 | **2.65x** | 1.51 -> 4.01 TB/s | 5.386 -> 2.719 | **1.98x** | 3.279 | 3.964 |
| all_masked (0%) | 0.815 -> 0.745 | 1.09x | 4.04 -> 4.43 TB/s | 0.859 -> 0.788 | 1.09x | 1.004 | 3.896 |

The forward is unchanged (same kernel: median 0.2768 vs 0.2771 ms). **With the fix our pair is faster than Liger in every case**
(Liger now needs 19-27% more time; before, it was 19% faster at the median batch) and 1.46x-4.9x faster than torch.compile. The
backward now moves its bytes at ~4.0 TB/s (~84% of the 4.8 TB/s HBM peak) on trained rows (was 1.5-2.5) and 4.4 TB/s for zero rows.
Remaining loss items: (3) gradient into the logits buffer (memory, next); the forward's 8 x LDS.U16 reads (45.35; later).

### 45.37 2026-10-17 evening: item 3 -- the loss gradient written over the logits (opt-in) + a double allocation found

Written on the laptop (BluTrain_sft, not committed; the user commits), compile-checked for sm_90a, box test pending.
- **Found while writing it:** `SparseCrossEntropyLossBackward::apply` (LossBackward.cpp) allocated the gradient TWICE on the CUDA
  path: `Tensor::empty(logits shape)` before the CPU / CUDA branch (only the CPU path used it) and again inside the CUDA branch.
  `Tensor::empty` allocates at once (Tensor ctor -> allocator), and the first buffer is freed only when the second is assigned,
  so for a moment the loss backward held the logits plus TWO gradient-sized tensors (3 x 3,144 MiB at 8 x 4096 x 50,304 bf16).
  This fits the "3-tensor model" of the 09-24 bf16-logits measurement (-4,716 MiB = 3 x 1,572 at B=4). Now each path allocates
  once (CPU in its branch; CUDA once, or not at all with the switch on).
- **The switch:** `autograd::sparse_cross_entropy_loss(..., bool grad_into_logits = false)` -> the node's `grad_into_logits_` ->
  on the CUDA stats path the returned gradient shares the logits' storage and the new kernel
  `sparseCENormalize_from_stats_inplace` writes each gradient over its logit (read once, then written once, by the same thread).
  The row code is shared (`normalize_from_stats_row`): the out-of-place kernel keeps its `__restrict__` parameters and its SASS
  is IDENTICAL to item 2's (0 differing lines, bf16 and fp32); the in-place kernel uses plain (not read-only-cache) 16-byte loads.
  Trainer: `BLU_LOSS_GRAD_INPLACE=1` (default 0) passes it in the training step only (bluscript_sft.cpp; validation has no
  backward). Safe there: the logits feed only the loss and are not read after backward().
- **Checks (run_item3.sh):** standalone `Tests/kernel_tests/test_loss_grad_inplace.cu` = the trainer's lm_head + loss through
  autograd (X [8,4096,2304] x W [2304,50304] bf16, real stage-0 targets, micro-batch 458), one process per arm: OLD library
  (libs/loss_v2, item 2 only, legacy 4-argument API) / NEW switch off / NEW switch on; same loss and W / X gradient checksums
  expected in all, two reps each (determinism), allocator peaks per arm. Then the SFT trainer on a tiny stage-0 set (64 / 16 rows,
  2 steps of 8 micro-batches, random init), switch off / on / off / on, its [MEM] allocator peaks and step losses.
- Expected: 'on' one logits-sized tensor (3,144 MiB) below 'off' wherever the loss backward sets the peak; OLD up to one more.
  If the peak sits elsewhere in a run (e.g. the matmul backward or the optimizer), the drop is smaller; that says where the peak is.
- **Trainer A/B gets the OLD binary too (user, 10-17):** so the whole 6.1 GiB shows in one session, on one box, on one data set.
  `old_trainer/bluscript_sft_old.cpp` = today's bluscript_sft.cpp with ONLY the item-3 lines removed (diff checked; git HEAD is
  from 09-28 and lacks the SFT / packing work, so it is not the "old" trainer), built with the pre-item-3 `LossOps.h`
  (`old_trainer/old_inc`, 4-argument declaration) and linked to libs/loss_v2 (item 2, double allocation); it calls the 4-argument
  loss op (nm). Runs: old / new-off / new-on, twice. Expected at the loss-backward moment: old 3 logits-sized tensors, new-off 2,
  new-on 1 (3,144 MiB each).

### 45.38 2026-10-17 20:06 IST: item 3 checked -- standalone: bit-identical, -5,926 MiB (old -> new switch on)

Run item3_1017 (2x box GPU 0; started by me at the user's request while the user was on a break, both phases as one command
`run_item3.sh 0 all`, after checking that all 11 files on the box were byte-identical to the laptop's and that both GPUs were
idle). Library with item 3: libs/loss_v3 (465cfa56); old = libs/loss_v2 (cd0bb2a9, item 2 only).

Standalone, test_loss_grad_inplace.cu (lm_head + loss through autograd: X [8,4096,2304] x W [2304,50304] bf16, real stage-0
targets of micro-batch 458 = 7,577 trained rows; one process per arm, forward + backward twice per process):

| arm | loss | W.grad (fp32) checksum | X.grad (fp32) checksum | reps identical | allocated peak | vs old |
|---|---|---|---|---|---|---|
| old library (double allocation), gradient into a new tensor | 11.336594 | 883fe7a5e3b70000 | b017f1eacd9f0000 | yes | 10,536 MiB | |
| new library, switch off | 11.336594 | 883fe7a5e3b70000 | b017f1eacd9f0000 | yes | 7,754 MiB | -2,782 |
| new library, switch on (gradient over the logits) | 11.336594 | 883fe7a5e3b70000 | b017f1eacd9f0000 | yes | **4,610 MiB** | **-5,926** |

- **Bit-identical in all arms and reps** (loss and both gradient checksums).
- **The switch saves exactly 3,144 MiB** (7,754 -> 4,610 = one 8 x 4096 x 50,304 bf16 tensor).
- **The double-allocation fix saves 2,782 MiB here**, not the full 3,144: the old peak was at the loss backward (X, W, their fp32
  gradients from rep 1 = 730 MiB, logits + 2 gradient buffers: 365 + 730 + 9,432 = 10,527 ~ 10,536); with the fix the peak moves
  to the matmul backward (logits + gradient + ~370 MiB of the matmul's new gradient temporaries: 7,754 = 7,383 + 371), and the
  in-place peak is the same moment minus the gradient buffer (4,610 = 7,754 - 3,144).
- The backward times in these logs are single cold calls (16-52 ms, jumping between reps): not a speed measurement (that was
  45.36's job).
- Reserved peaks (what the allocator holds from the GPU, close to what nvidia-smi shows): old 12,952 -> new switch off 10,174 ->
  new switch on **4,618 MiB (-8,334 MiB)**. With the switch on, reserved = allocated (4,618 vs 4,610): no logits-sized block is
  freed and cached any more, so nothing big sits unused in the cache.

**Trainer phase, try 1 (20:07-20:09 IST): FAILED before step 0. The cause was my script, not item 3.** Both trainers built fine
(new bluscript_sft_bin_item3 1c2fbb7a, old bluscript_sft_bin_old 383f4f23, each loading the intended library: new 465cfa56,
old cd0bb2a9). All 6 runs then stopped in about 3 s with exit 134:
`SftDataLoader: /root/gautam_k/sft_data/tiny_item3/train/inputs.bin has 0 bytes, meta.json says it should have 524288`.
- Why: the script made the tiny set on the box with `head -c` from `stage0_ultrachat_5k_T4096/{train,val}/`. This box has
  only `train/labels.bin` of stage 0 (copied for the loss benchmark), so `head` failed on the missing files and its error went
  to /dev/null. The redirect still created the files empty. The script checked only that labels.bin exists, and it kept going
  after each failed run instead of stopping.
- A second problem the run never reached: 64 rows at 64 rows per step and the default 2 epochs give max_steps = 2 and
  warmup = max(1, round(0.03 x 2)) = 1. The trainer refuses that (`warmup + stable >= max_steps - 1`, bluscript_sft.cpp:1080,
  "leaves no decay tail"). So even with good data, try 1 would have stopped.
- Fixes (laptop, run_item3.sh md5 fdf019ec, item3_summary.py 685ecd76):
  1. The tiny set is made on the laptop from the same files: `~/Downloads/sft_data/tiny_item3/` = the first 64 train rows and
     the first 16 val rows of stage 0, meta.json `{"N": .., "T": 4096}` (the loader reads only N and T). Checked with a
     plain-Python copy of the loader's checks: sizes, valid_counts = non-65535 labels in every row, rows byte-identical to the
     full set (the full labels.bin is byte-identical to the box copy, md5 eec5e38b). Trained tokens: train 60,263, val 11,808.
     The check fails on a changed label and on an empty inputs.bin (tested). The box script no longer makes the set; it
     checks all 6 file sizes and both N values before the builds, and prints the md5 of the set (laptop: 90d39b16).
  2. `BLU_SFT_EPOCHS=4` -> 4 steps of 8 micro-batches, warmup 1 (1 < 3, passes the decay-tail check).
  3. The script now stops at the first trainer run that exits non-zero or prints no [MEM] line.
- Try-1 artefacts on the laptop: box_runs/item3_1017_try1/ (+ item3_results_try1.tgz).
- Expected time for the trainer phase alone: ~2 min of builds + 6 runs x ~1-1.5 min (the stage-0 full run took 5.3 s per
  step on one H200, 37.4) = ~10 min. Not run yet: copying the tiny set and the script to the box, and re-running
  the trainer phase, wait for the user's OK.

**Trainer phase, try 2 (20:42-20:52 IST, GPU 0): PASSED.** The user copied the tiny set and the fixed script and ran
`run_item3.sh 0 trainer` (one more stop at 20:41:34: the set was not there yet; the new size check caught it before any build).
Trainers rebuilt (new 1c2fbb7a, old 383f4f23, same as try 1); all 6 runs exit 0 with 4 step lines. Results in
`sft_loss_vs_rivals_2026_10_17/box_runs/item3_1017/` (summary: `python3 item3_summary.py box_runs/item3_1017`).

| run | allocated peak | reserved peak | step 0 train loss | step 0 grad norm | step 3 train loss |
|---|---|---|---|---|---|
| old r1 / r2 | 125,046 / 125,046 MiB | 126,986 / 126,986 | 11.236371 / 11.236371 | 33.8707 / 33.8711 | 9.830540 / 9.831098 |
| new, switch off r1 / r2 | 122,264 / 122,264 (**-2,782**) | 123,312 / 123,312 | 11.236371 / 11.236371 | 33.8712 / 33.8718 | 9.829689 / 9.831923 |
| new, switch on r1 / r2 | 119,376 / 119,376 (**-2,888** more) | 121,866 / 121,866 | 11.236371 / 11.236371 | 33.8728 / 33.8720 | 9.830827 / 9.831545 |

- **Memory: -5,670 MiB allocated peak (-4.5%) and -5,120 MiB reserved peak in the real trainer** (B 8 x T 4096, bf16 logits),
  the same in both reps. The double-allocation fix alone gives -2,782 MiB (exactly the standalone number); the switch gives
  -2,888 MiB more. That is 256 MiB less than one logits tensor (3,144): with the switch on, the step's peak moves to another
  moment of the backward that holds 256 MiB more than "the old peak minus the gradient buffer".
- **Numbers: the step-0 loss is identical in all 6 runs** (the forward is unchanged). The grad norms differ in the 5th digit,
  but they differ by the same amount between two runs of the SAME binary (old r1 33.8707 vs old r2 33.8711; steps 1-3 drift a
  little more because the random-init model has grad norms up to 159). That is run-to-run noise in the trainer: the attention
  backward's default dQ accumulation uses atomic adds (there is a separate USE_DETERMINISTIC_DQ=1 mode), so the order of the
  sums changes between runs. The loss backward itself is bit-identical with and without the switch (standalone test above).
- The old trainer's peak (125,046 MiB) matches the 10-07 full stage-0 run (125,026 MiB, 37.4), so the tiny run shows the same
  peak as a real run.
- Decision for the user: keep BLU_LOSS_GRAD_INPLACE opt-in (default 0) or make it the default in bluscript_sft.cpp. It is safe
  in this trainer because nothing reads `logits` after backward(); any future code that reads the logits after the backward
  (for example a token-accuracy metric) must run before it or turn the switch off.
- Changed files to commit (user commits): Tensor-Implementations/src/Kernels/cuda/loss/LossKernels.cu (cf947bf4),
  include/ops/helpers/LossKernels.h, include/autograd/backward/LossBackward.h, src/autograd/backward/LossBackward.cpp (c59dffd7),
  include/autograd/operations/LossOps.h, src/autograd/operations/LossOps.cpp (81e27dfd), Tests/kernel_tests/
  test_loss_grad_inplace.cu (40010b31, new), bluscript_sft.cpp (6f94d3ab).

### 45.39 2026-10-17 20:46-21:20 IST: ncu of K5q vs FA3 -- packed already ~10% ahead, dense ~8% behind; where, and fix L1b

ncu run `ncu_k5q_1017` on GPU 1 (user ran it, twice: 20:46 and 20:47; both copies on the laptop, numbers equal to 0.1%):
`AB_QUICK=0 AB_ROUNDS=0 ab_multi.sh 1 ncu_k5q_1017 vj2_K4b vj2_K5q`. Locked clock here was ~1,336 MHz; FA3's reference (10-13)
ran at ~1,620 (dense) / ~1,730 MHz (packed). F3c at both clocks: +1.7% cycles dense / +0.5% packed at the higher clock, so the
cycle comparisons below hold to ~1-2% (FA3 would be ~1% fewer cycles at 1,336 MHz). Script: session_2026_10_17/deep/
ncu_k5q_vs_fa3.py (tables in box_runs/2x_1017/ncu_k5q_1017/analysis_vs_fa3.txt). A local nvcc 13.0 build of the box's
fwd_K5q.cu (f35982af) gives the SAME SASS as the profiled library (3,672 / 3,504 lines, only `.reuse` flags differ), so
candidate SASS can be checked on the laptop before any GPU run.

| | FA3 | K5q | K5q / FA3 |
|---|---|---|---|
| packed real_rows_752: cycles | 0.744M | **0.670M** | **0.90 (ours ~10% faster)** |
| packed: instructions | 142.2M | 130.9M | 0.92 |
| dense B8 18/6: cycles | 1.423M | 1.538M | **1.08** |
| dense: instructions | 306.5M | 322.4M | 1.05 |
| dense: tensor pipe busy | 83.3% | 77.1% | cycles x busy = 1.186M both: same tensor work, more idle time |

**Where the dense +8.1 points go** (cycle-equivalents = share of stall samples x cycles, FA3 = 100; tile counts matched: both
kernels process 608,256 warp-tiles; FA3's loop holds 571,392 of them, K5q's 535,680, the rest sit in per-unit code):
1. Steady loop, **+4.0**: a K5q step costs 146.6 vs FA3 139.0 per million steps (+5.5%). By phase (per million steps):
   - QK issue (baton -> last QK HGMMA): FA3 41 instructions 26.1 | K5q 72 instructions 31.7 = **+5.5**. K5q rebuilt BOTH
     descriptors of every QK HGMMA every step (4 uniform instructions each, one 4-register group gdesc[UR20] for all 8);
     FA3 keeps Q's 8 descriptors in 8 groups (UR16..UR48, built once per unit) and writes only K's half (2 per HGMMA).
   - PV issue: FA3 41 instructions 28.8 | K5q 28 instructions 25.0 = **-3.8** (FA3's warps wait longer at warpgroup arrive).
   - softmax (QK wait -> loop end): FA3 387 / 84.1 | K5q 385 / 89.9 = **+5.9**, same instruction count. By opcode: MUFU
     +7.5 (of it "wait" +7.2: K5q's 66 exps sit closer to their inputs / each other than FA3's, which are spread among FMA
     work), FMNMX -1.7, the WARPGROUP waits -4.3.
2. Unit boundaries (prologue tile 0, diagonal tile, unit start / end), **+4.0**: mostly long-scoreboard waits on Q and on
   tile 0's K/V (+3.5 vs FA3) = the "earlier Q / tile-0 prefetch" item of 45.32.
So the 45.32 picture (77% of K1's excess at pair start/end) has changed: K4b/K5q fixed most of that; half of K5q's excess is
now in the loop step.

**Fix 1 = L1b** (candidates/fwd_L1b.cu 069772b7; from the box's K5q source f35982af; switch L1_DESC). Q's 8 descriptors are
made once per consumer group (its Q buffer never moves); a K or V tile's descriptor = the stage-0 descriptor + the stage
offset (one add). First try L1 (fwd_L1.cu 19c0c3fb) did NOT change the SASS: ptxas rebuilt the "once" values inside the loop
from the smem base (still live there for K/V). L1b passes each Q descriptor half through `__shfl_sync(.., 0)` once: ptxas can
no longer rebuild it and keeps it uniform. Local SASS: steady QK issue **72 -> 42 instructions** (FA3 41), Q descriptors in 8
groups like FA3 (gdesc[UR40], [UR12] .. [UR36]), 168 registers, 0 spills, dense 3,648 instructions (K5q 3,672), packed 3,504.
Same descriptor values (every smem address < 0x40000: the 14-bit field never carries) -> bit-identical results; the quick
suite (296 checks, incl. packed == alone bit-exact) must confirm. Expected gain: the QK-issue phase's +5.5 per million steps
is ~2.7% of the dense kernel; how much of it goes is for the race to say (estimate 1-2% dense, small packed).
Next items after L1b: the exp scheduling in the softmax (+3.2 points), the unit-boundary waits (+4.0).

### 45.40 2026-10-17 ~21:15 IST: END OF DAY -- where we stopped, all numbers, and the plan for Monday

The 2x H200 box (root@86.38.238.36) is being closed by the user. Before that, everything of ours was copied to
`profiling_reports/box_2x_final_2026_10_17/` (622 files, md5 identical to the box; README.md there lists what is and is not
in it). The box's source tree was identical to the laptop master (1,496 of 1,496 source files same md5). Not copied, because
they are rebuilt: the 35 libraries (1.9 GB; md5 list kept) and the loss-rivals venv (5.6 GB; pip freeze kept).

**WHERE WE STOPPED (exactly):**
- Attention forward: candidate **L1b** (candidates/fwd_L1b.cu 069772b7) is written and its SASS checked on the laptop
  (45.39), but **NOT built on a box and NOT raced**. The next action is the race L1b vs K5q.
- Loss kernel: items 2 and 3 are DONE and verified on the H200 (45.36, 45.38). Waiting for 2 user decisions: (a) make
  BLU_LOSS_GRAD_INPLACE=1 the default in bluscript_sft.cpp or keep it opt-in; (b) commit the 8 changed files.

**All numbers of the day (details in the sections named):**
- Attention forward (base at the start of the day = F3c, the laptop master):
  - 45.20-45.28: race F2 / F3c / C1 -> F3c kept (E1's L2 hints made the packed backward +0.18%); K1 (FA3 loop order) -0.95%
    dense; K4b (O staged in the last K buffer, fixed handshake) -5.26% dense / -7.37% packed vs K1; **K5q** (K4b + split 16 +
    one P set) = best overall; K7e (split 8 + FA4 conditional rescale) -0.81% dense but +0.63% packed; per-mode builds K8 / K9
    rejected (they break packed == alone bit-exact). **K5q vs F3c: about -6.5% dense, -11.4% packed.**
  - 45.29: knock-outs: no exp -6.8%, no max -4%, no rescale -2.3%, all three -11.2% (the softmax is on the critical path).
  - 45.31: the wait form (W1, W1s), separate K/V barriers (W2) and the polynomial exp (P8w +2.7%, P4w +7.1%): all rejected.
  - 45.32 + 45.39: ncu. **Packed: K5q ~10% FEWER cycles than FA3 (0.670M vs 0.744M) and 8% fewer instructions. Dense: K5q
    +8% cycles (1.538M vs 1.423M).** Dense excess = steady loop +4.0 points (QK descriptors +3.0, exp scheduling +3.2, PV
    issue -2.0) + unit boundaries +4.0 (Q / tile-0 K/V waits +3.5).
  - Estimated position vs FA3 in time (free clocks, chained from the 10-16 scoreboard; NOT yet measured on one box): dense
    about 1.10x FA3, packed about level or better.
- Attention backward: unchanged today; beats everyone (10-16 scoreboard: dense 0.963x FA3, packed 0.984x).
- Loss kernel:
  - 45.33: vs PyTorch / torch.compile / Liger / Apple CCE at our SFT shape: ours beat every PyTorch variant; Liger was 19%
    faster at the median batch (our backward ran at 1.5 TB/s).
  - 45.35-45.36: **item 2** (bf16's user copy constructor made LossVec<bf16> load/store as 8 x 2-byte accesses; now uint4 +
    4 loads in flight): bit-identical, backward 1.607 -> 1.008 ms (1.6x) at the median batch, forward + backward 1.883 ->
    1.281 ms; **ours now beats Liger in every case**.
  - 45.37-45.38: **item 3** (opt-in gradient over the logits + the double allocation removed for everyone): standalone
    bit-identical, -5,926 MiB; **real trainer -5,670 MiB allocated peak (125,046 -> 119,376), -5,120 MiB reserved, loss
    identical**.
  - Item 1 (lm_head + loss on trained positions only): FUTURE task after the v1 SFT pipeline (45.34).
- Checkpoints (45.30): DPO / next stage needs only the weights (BLU_INIT_FROM loads weights only, automatically); continuing
  SFT = auto-resume from BLU_CKPT_DIR (weights + AdamW state + step); a new stage needs a new empty BLU_CKPT_DIR.

**Designs decided / found today (keep):**
- K4b: the dense O goes through the last K buffer (stmatrix + 16-byte stores) with a monotonic counter handshake
  (protocol_sim_k4.py: 0 deadlocks in 1000; K4's handshake deadlocked 385/1000 packed).
- K5q: softmax split 16 (16 exps before the P.V wait) + one P register set.
- ptxas tricks: (1) a data-dependent mbarrier arrive pins the P.V wait after the softmax (45.22); (2) NEW (45.39): ptxas
  rebuilds "computed once" wgmma descriptors inside the loop when it can rebuild them cheaply from a live value; passing
  each value once through `__shfl_sync(.., 0)` stops that and keeps it in a uniform register (L1b: QK issue 72 -> 42).
- Local check: an nvcc 13.0 build of a candidate on the laptop gives the same SASS as the box build (45.39), so a candidate's
  SASS can be checked before any GPU run.
- Loss: LossVec loads / stores through uint4 + memcpy (never through the bf16 copy constructor); the gradient-in-place switch
  is safe only while nothing reads the logits after backward().

**Uncommitted code on the laptop (the user commits; ~/Downloads/BluTrain_sft is the master):**
- Tensor-Implementations: 27 modified + 5 new files (attention fwd/bwd, loss items 2+3, norm/RoPE, tests). The tree's
  GQA_fwd_sm90.cu is still **F3c**; K5q / L1b live only in profiling_reports/session_2026_10_17/candidates/ until the user
  decides to adopt one.
- BluTrain: bluscript_sft.cpp and sft/ are untracked; bluscript_zero_ckpt.cpp modified.
- This work has not been committed since 09-28 (it exists only on the laptop): commit + push soon.

**MONDAY: set up again and continue from the stop point (in this order):**
1. New box: claim our GPU(s) as the user says. Copy the laptop master tree (`~/Downloads/BluTrain_sft`) to the box, build
   the library (`make -j$(nproc)` in Tensor-Implementations), build the test binary tva_a (build_cands.sh `tva` option, or
   the nvcc line in build_cands.sh), copy `stage0_ultrachat_5k_T4096/train/labels.bin` (the packed bench case needs it).
2. Copy `box_2x_final_2026_10_17/scripts/` to `~/gautam_k/scripts/` (it holds cands/ with fwd_K5q.cu f35982af and F3c) and
   add `candidates/fwd_L1b.cu`. Build: `bash build_cands.sh K5q L1b F3c` (F3c last: puts the tree back to the master's file).
3. **Race L1b vs K5q** (forward only, 8 rounds, quick suite first: 296 checks incl. packed == alone bit-exact):
   `AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 bash ab_multi.sh <gpu> l1b vj2_K5q vj2_L1b`
   (~10 min; ends with an ncu of L1b). Expected: dense -1..-2%, packed small.
4. Next forward fixes, from 45.39: (a) the exp scheduling in the softmax (MUFU "wait" +7.2 per million steps vs FA3);
   (b) the unit-boundary waits for Q and tile 0's K/V (+3.5 points). Each: build -> SASS check on the laptop -> race.
5. The same-box scoreboard vs FA3 / FA4 / cuDNN (forward, backward, forward + backward, dense + packed, free clocks):
   needs the rivals' environment on the new box (FA3 / FA4 from flash-attention 94e22c9, cuDNN through torch 2.14.1+cu130;
   loss_env_setup.sh is the starting point).
6. Loss: the user's two decisions (default switch, commit); later the forward loss kernel's 8 x LDS.U16 (same bf16 copy issue
   as item 2's backward); item 1 after the v1 pipeline.
7. The user decides whether K5q (or L1b if it wins) replaces F3c in the laptop master GQA_fwd_sm90.cu.
- Analysis scripts of 45.39 (run from profiling_reports/): session_2026_10_17/deep/ncu_k5q_vs_fa3.py (4 views),
  k5q_loopdiff.py (loop bodies by opcode), k5q_skeleton.py (loop skeleton: GEMM issue, barriers, waits), k5q_pergroup_stalls.py
  (per code group by stall reason), k5q_loop_phases.py (QK issue / PV issue / softmax per million steps); loop skeletons in
  box_runs/2x_1017/ncu_k5q_1017/skel_*.txt; local SASS + ptxas logs of K5q (box source), L1, L1b in deep/l1_sass/.
  sft_loss_vs_rivals_2026_10_17/check_tiny_set.py = the tiny-set check of 45.38.

### 45.41 2026-10-19 (Monday): new 2x H200 box set up; race L1b vs K5q -> L1b REJECTED

**New box:** root@31.56.109.89 (dull-tree-fades-fin-02), 2x H200, driver 580.178.04, **CUDA 13.4** (old box 13.0), 88 cores,
113 GB free. Our folder = `/root/gautam_K` (capital K, the user's choice; a gautam_k link was made and removed: one folder
only). Others on the box: /root/harshini, /root/Ananya (BluTrain trainers; both GPUs were theirs until ~10:30 IST), Documents.
Scripts switched to gautam_K in a fresh copy `profiling_reports/session_2026_10_19/{scripts,loss_bench}` (12 files; the
10-17 archive is untouched). Copied (user ran the commands): the laptop master tree (no .o/.so/.a), scripts (cands/ with
fwd_K5q.cu f35982af + fwd_L1b.cu 069772b7), loss_bench, stage0 data + tiny_item3; md5-checked: 3,589 / 54 / 13 / 16 files
identical. Built: top-level `make -j48 all SM_ARCH=90` exit 0; `build_cands.sh tva K5q L1b F3c` -> K5q a804ea04, L1b 2e2d2544,
F3c d12fe6cb, tva_a c267b5db (test source b7c752b3). The user said GPU 0 is ours; at launch both GPUs were at 0 MiB twice.

**CUDA 13.4 compiles K5q differently** (cuobjdump of the box libraries): steady QK issue (baton -> last QK HGMMA) dense K5q
**59** (13.0: 72) vs L1b 43; packed K5q **40, already in 8 descriptor groups like FA3** vs L1b 43. So all comparisons from
now on are against libraries built on this box.

**Race l1b_1019** (GPU 0, 10:35-10:39 IST, forward only, 8 rounds, 4000 dense / 8000 packed forwards per run, free clocks;
quick suite L1b 296/296; results in session_2026_10_19/box_runs/l1b_1019):

| | K5q median | L1b median | L1b vs K5q (paired) | p10 | rounds L1b faster |
|---|---|---|---|---|---|
| dense B8 18/6 | 1.0905 ms | 1.0934 ms | **+0.36%** (t 2.2) | +0.55% (t 4.4) | 2/8 |
| packed real_rows_752 | 0.4570 ms | 0.4608 ms | **+0.90%** (t 11.3) | +0.79% (t 7.3) | 0/8 |

**Verdict: L1b rejected.** Order note: with only 2 libraries ab_multi.sh's rotation and its odd-round reversal cancel, so L1b
ran first in every round (status lines "(vj2_L1b vj2_K5q)"); any first-position advantage went to L1b, so the verdict
stands. (For future 2-library races: use 3 libraries or fix the order rule.)

**What it teaches:** the 45.39 split counted WARP time (stall samples), not tensor-pipe idle time. The QK-issue phase's
+5.5 per million steps was time the warps would have waited anyway (for the baton / the tensor pipe), so removing ~16
instructions there did not shorten the step; the same caution applies to the other two items (exp scheduling, unit-start
waits). The next lever must be judged by what it does to the tensor pipe's idle time (dense: FA3 busy 83.3% vs K5q 77.1%),
not by stall shares. Same lesson as the 4B BLAS work (memory: instruction cuts converted at zero).

### 45.42 2026-10-19: item 2 re-checked -- the loss module never used the library's CUDA-type mapping

The user asked whether our GPU kernels use CUDA's own types (as the reduction module does) and whether item 2's cause and
gain are real. Checked from source, fresh compiles and the raw logs (scratchpad item2_recheck):
- **The library rule exists:** `include/dtype/CudaTraits.h:17-22` maps `bfloat16_t -> __nv_bfloat16`, `float16_t -> __half`
  (`CudaNativeType<T>`). Reduction uses it (`ReductionImplGPU.cu:130` `using CudaT = CudaNativeType<T>;`, `:153`
  reinterpret_cast of the data pointers, then the kernel launch); so do dropout, embedding, attention, LayerNorm, conversion.
- **The loss module does not:** `LossKernels.cu` has 0 uses of CudaNativeType / __nv_bfloat16 (now and at git HEAD ad10e7d);
  its kernels are instantiated with our `bfloat16_t` (lines 1192-1269) and the compiled names say `OwnTensor::bfloat16_t`.
  Same for the B300 file `loss/arch/CrossEntropyKernels_sm103.cu`. These are the only GPU files with our bf16 and no mapping.
- **Why it matters (test typetest.cu, sm_90a):** our `bfloat16_t` / `float16_t` are NOT trivially copyable (user copy
  constructor, `Types.h:451`); copying a 16-byte struct of 8 of them = 8 x LDG.E.U16 + 8 x STG.E.U16. With `__nv_bfloat16` /
  `__half` (trivially copyable) the same copy = 1 x LDG.E.128 + 1 x STG.E.128. Following the rule would have avoided it.
- **Old vs new loss backward, fresh compile:** old bf16 kernel 0 x 16-byte loads/stores, 10 x 2-byte loads + 10 x 2-byte
  stores; the SAME code for fp32 had 16-byte accesses; new kernel 5 x 16-byte loads + 6 x 16-byte stores.
- **Gain (raw logs loss_v2_1017, H200, old/new/old/new, 2000 warm-up + 500 timed calls, all 4 gradient checksums equal in
  all 6 cases):** median batch backward 1.6072 / 1.6075 -> 1.0077 / 1.0075 ms; max batch 2.169 -> 1.147; no mask 4.361 -> 1.645.
  Per training step this is ~0.6 ms per micro-batch (~5 ms of a ~5.3 s step); not measured inside a trainer (all 10-17
  trainer arms already had item 2).
- **Library-wide audit (today's box build d12fe6cb, 5,960 kernels, 600 built with our bf16/fp16):** the elementwise kernels
  (scalar_*, add/sub/mul/div, compare) do 1 value per thread per step with neighbours reading neighbours, so their 2-byte
  accesses combine across the warp by design (checked: ScalarOps.cu scalar_add_inplace, 4x unrolled grid-stride). The
  split-vector problem only hits code that packs several values into one struct of our type: the loss kernels. Still open
  there: the forward `sparse_ce_forward_kernel_vec_save_stats` (8 x LDS.U16 reads, 45.35) and the B300 forward
  `sparse_ce_fwd_kernel_sm103_save_stats` (14 x 2-byte global accesses, 2 wide; not testable without a B300).
- **Cleaner fix, proposed:** launch the loss kernels with `CudaNativeType<T>` like reduction; plain LossVec copies then become
  16-byte by themselves and the forward's 2-byte shared-memory reads should go too. Bit-identity expected: since 09-26 our
  bf16/fp16 GPU conversions call CUDA's intrinsics, checked bit-for-bit on every non-NaN input (Types.h notes). Must be
  re-verified with the 20-config test + the H200 old/new bench before it replaces item 2's uint4 helpers.

**FUTURE (user decision 10-19): all pending loss-kernel work waits until the attention work is complete.** In order:
1. Loss module -> the library rule `CudaNativeType<T>` (forward + backward; the B300 file too), keeping 4 loads in flight;
   verify bit-identical (the 20-config laptop test) + the H200 old-vs-new bench (~10 min GPU) before it replaces item 2's
   uint4 helpers (45.42).
2. Full-length SFT validation run with BLU_LOSS_GRAD_INPLACE=1 (stage 0, 156 steps from the step-8000 checkpoint) against the
   10-07 reference (val 1.4464 -> 1.2839, 5.3 s/step, peak 125,026 MiB; expect ~119.4k MiB). Needs the 6.5 GB checkpoint.
3. User decisions: BLU_LOSS_GRAD_INPLACE default on or opt-in; commit the 8 files of 45.38.
4. The forward loss kernel's 2-byte shared-memory reads (likely fixed by 1); the B300 forward (needs a B300).
5. After the v1 pipeline: logits + loss only at the trained positions (45.34).

### 45.43 2026-10-19 11:29-11:33 IST: same-box scoreboard on the new H200 -- ours vs FA3 / FA4 / cuDNN, forward, backward, both

**Setup.** Box root@31.56.109.89 GPU 0 (GPU 1 had a colleague's run; separate 700 W limits). Rivals' environment rebuilt
today with `session_2026_10_19/scripts/benv_setup.sh` (57f0bc3a -> 883a5468 -> 346f625b): `/root/gautam_K/benv` = torch
2.14.1+cu130, cuDNN 9.24.0 (92400), FA3 `flash_attn_3 3.0.0` built from flash-attention 94e22c9 `hopper/` (bf16 + head dim
128 + sm90, all features of that configuration; CUTLASS submodule 7127592; built with the box's nvcc 13.4, the same compiler as
ours), FA4 `flash-attn-4 4.0.0b34` (nvidia-cutlass-dsl 4.8.1, quack-kernels 0.6.5). Three fixes on the way: git refused the
rsync'd clone ("dubious ownership": a per-run `GIT_CONFIG_*` safe.directory, no global change); the box has no Python
headers (python3.12-dev / libpython3.12-dev 3.12.3-1ubuntu0.17 downloaded and UNPACKED into /root/gautam_K/pydev, used via
CPATH; nothing installed); the import check ran inside hopper/ (whose source flash_attn_3/ shadowed the installed one).
Ours = vj2_K5q (a804ea04: forward K5q, backward version 3) for both shapes. Method = the 10-16 scoreboard
(`scoreboard_free.sh 0 sb_1019 vj2_K5q vj2_K5q`, 12c2f6ba): free clocks, one contender per process under nsys (kernel times,
no Python overhead), forward + backward back to back, 1000 (dense) / 2000 (packed) iterations, ABBA order (ours fa3 fa4 cudnn |
cudnn fa4 fa3 ours), clock/power sampled every 100 ms. All 16 runs exit 0. Results:
`session_2026_10_19/box_runs/sb_1019/` (summary.txt; summary_steady.txt).

**Clock-log artefact (not a slowdown):** nvidia-smi reported 345 MHz (the idle clock) for the first ~2 s of OUR runs while the
same samples show 100% utilisation and ~700 W; our C++ bench starts GPU work within ~1 s of launch, the rivals after ~5 s of
Python start-up. The nsys trace shows our kernels were in fact a bit FASTER in those 2 s (dense forward 1,036.9 vs 1,052.6 us
after; packed 431.2 vs 443.9), i.e. a cooler GPU. With the stale samples removed, our clocks are 1,635-1,755 MHz like the
rivals'. To rule out any cool-phase advantage the table below uses the STEADY state only (calls starting >= 2 s after the first
forward; `scripts/scoreboard_steady.py`); it differs from the full-run medians by <= 0.016 in every ratio.

| steady state, kernel time in us (mean of r1/r2) | ours | FA3 | FA4 | cuDNN |
|---|---|---|---|---|
| dense B8 x 4096, 18/6 heads: forward | 1,055.2 | **974.0** | 1,070.9 | 1,139.2 |
| dense: backward call (helpers incl.) | **3,177.6** | 3,291.7 | 3,229.9 | 3,600.3 |
| packed real_rows_752: forward | **442.9** | 473.2 | 526.1 | 494.0 |
| packed: backward call | **1,524.5** | 1,601.5 | 1,573.9 | 1,693.4 |

| ours / rival (below 1 = ours faster) | forward | backward call | forward + backward |
|---|---|---|---|
| dense vs FA3 | **1.084** | 0.965 | 0.992 |
| dense vs FA4 | 0.985 | 0.984 | 0.984 |
| dense vs cuDNN | 0.926 | 0.883 | 0.893 |
| packed vs FA3 | **0.936** | 0.952 | 0.948 |
| packed vs FA4 | 0.842 | 0.969 | 0.937 |
| packed vs cuDNN | 0.897 | 0.900 | 0.899 |

**Where we stand (measured on one box, not chained):** ours is the fastest of the four in 11 of 12 entries; the ONLY loss is
the dense forward vs FA3 (8.4% more time). SFT-packed (the case we train with): ours beats FA3 by 6.4% forward, 4.8% backward,
5.2% both together. Versus the 10-16 scoreboard (forward 1.17x / 1.11x FA3, fwd+bwd 1.009x / 1.012x): the 10-17 forward work
moved dense to 1.08x and packed to 0.94x, and fwd+bwd is now ahead of FA3 in both shapes (0.992x / 0.948x). The packed
backward ratio also improved (0.98x -> 0.95x) with the same backward code: on 10-16 our packed bench also ran full-row dense
kernels in the same process (noted as pessimistic then); today it runs only the packed path.
Next: the dense forward gap to FA3 (8.4%), judged by tensor-pipe idle time (45.41).

### 45.44 2026-10-19 11:41 IST: ncu FA3 vs K5q on the same box -- the dense gap is idle tensor time in every LOOP STEP

Run `scripts/ncu_fa3_k5q.sh 0` (a010c18f; GPU 0; ncu 2026.3.1 --set full, 4th launch; FA3 = benv flash_attn_3 3.0.0, ours =
vj2_K5q a804ea04, both built with nvcc 13.4; inputs as in the 45.43 scoreboard). Files: `session_2026_10_19/box_runs/
ncu_fa3_k5q_1019/` (raw + source CSV; the .ncu-repz need ncu >= 2026.3, the laptop has 2025.3.1).

| dense B8 18/6 forward | ours K5q | FA3 | ours / FA3 |
|---|---|---|---|
| time (ncu clock: ours 1.68, FA3 1.63 GHz) | 934.9 us | 871.3 us | 1.073 |
| cycles per SM | 1,574,797 | 1,423,158 | 1.107 |
| instructions | 322.0M | 306.5M | 1.050 |
| tensor pipe busy | 75.3% | 83.3% | |
| tensor-BUSY cycles (cycles x busy) | 1,186,028 | 1,184,951 | 1.001 (same work: 579 tiles x 2,048) |
| tensor-IDLE cycles | **388,769** | **238,208** | **+150,561 = the whole gap** (151,639) |
| DRAM read | 520 MB | 365 MB | 1.43 |
Packed real_rows_752 (we win): cycles 685,282 vs 745,727 (0.919), tensor-idle 235,614 vs 303,581 (-67,967).

**Where the idle time goes (fit over dense + packed):** idle per SM = a x units + b x tiles, with units = 128-row blocks
(dense 4,608 = 34.9/SM; packed: 24 chats of 1,024-1,536 tokens x 18 heads = 4,734 = 35.9/SM) and causal tiles (dense 76,032 =
576/SM; packed 28,782 = 218/SM, matching the tensor-busy cycles / 2,048 = 216-220). Ours: **a = 3,905 cycles per unit
boundary, b = 438 cycles per tile step**; FA3: **a = 9,422, b = -158 (~0: its steady step is tensor-bound)**.
- Dense: ours 136K (boundaries) + 252K (steps) = 389K; FA3 329K + ~0 = 238K. Packed: ours 140K + 96K = 236K; FA3 338K + ~0.
- So: our unit boundaries are 2.4x cheaper than FA3's (K4b's O path, the prologue work of 10-16/17), which is why we win
  packed (many short units); but every steady step leaves the tensor pipe idle ~438 cycles (of ~2,490), which is why we lose
  dense (16.5 steps per unit). Consistent with the 10-17 trace (TRK5): step 2,401 cycles vs the 2,048 tensor-bound minimum,
  per-pair non-loop time ~5,300 cycles.
- Mechanism: with the two consumer groups in ping-pong, a group's work between its GEMM issues (QK wait, softmax part A,
  P.V wait, part B + rescale + packs, baton) must fit in the window in which the tensor pipe runs its own P.V(n-1) and the
  other group's QK + P.V (~1,536 cycles). FA3's softmax (1,186 cycles per tile, 10-16) fits; ours (1,363 then; part A 792 +
  part B 591 in TRK5) plus the baton and issue overruns it by ~440 cycles per step.
- This also explains 45.41: the QK-issue instructions L1b removed were not on this overrun path, so they did not convert.
**Target:** shorten each group's per-step critical path by ~440 cycles (softmax + rescale/packs + issue), measured as tensor
idle per step. Next: a fresh per-step timeline on this box (CUDA 13.4 changed the SASS) to see which part overruns.

### 45.45 2026-10-19 11:48 IST: per-step timeline of K5q on the new box -- what fills the ~440 idle cycles per dense step

Build `build_cands.sh TLK5q F3c` (TLK5q = K5q + clock stamps, source 7297dfbb -> library 97765067; F3c back in the tree,
c791440a: the same F3c source as this morning's d12fe6cb, the library bytes differ between builds). Trace `trace_1017.sh 0
vj2_TLK5q` (GPU 0, free clocks, launches 900 and 950, dense + packed). Files: session_2026_10_19/box_runs/trace_1019_vj2_TLK5q.
Read with deep/timeline_read.py (steady steps of each CTA's first pair, both groups, 3,048 steps, 132 CTAs).

| dense loop step, cycles | group 0 | group 1 |
|---|---|---|
| s0->s1: K/V wait (group 0 only) + baton | 299 | 195 |
| s1->s2: issue QK(n) + P.V(n-1), wait for QK(n) | 673 | 1,057 (queued behind group 0's GEMMs) |
| s2->s3: softmax part A (mask, max, 16 exps) | 640 | 643 |
| s3->s4: wait for P.V(n-1) | 8 | 8 |
| s4->s5: part B (48 exps + sums) + O rescale + P pack | **853** | 535 |
| period (incl. ~250 of trace bookkeeping between s5 and the next s0) | 2,699 | 2,699 |
Launch 950 identical (2,700); the 10-17 trace of the same source with CUDA 13.0 identical too (2,706): CUDA 13.4 did not
change the step. Packed: no steady steps in the first pair (units too short), as expected.

- Without the ~250 cycles of trace bookkeeping (the timeline stores + accumulators sit between s5 and the next s0), the step
  is ~2,465 cycles, matching the ncu fit of 45.44 (2,048 + 438 = 2,486) from an independent measurement.
- The chain that sets the period is group 0's: QK 673 + part A 640 + 8 + part B 853 + K/V wait 299 = 2,473 vs the 2,048 a
  tensor-bound step allows: **~430 cycles too long**. Two pieces stand out:
  1. **Group 0's part B is 853 vs group 1's 535** for the same work: it runs while group 1 does its part A (group 1's QK is done
     at 1,771, its part A runs 1,771-2,414; group 0's part B runs ~1,321-2,174, ~400 cycles of overlap). Two softmax warps per
     SMSP then share the issue slot and the exp unit. Root: our softmax (A + B = 1,493) is longer than the other group's GEMM
     window (1,024), so the two softmaxes overlap; FA3's softmax (~1,186, 45.32) overlaps far less.
  2. **Group 0's K/V wait at the step start (299)** = the TMA load of tile n+1 not yet landed (group 1 waits only for the
     baton, 195, and the baton from group 1 is available long before). The 10-17 experiments (W1 / W2, 45.31) showed this
     wait alone is slack (shortening it made part A longer by the same amount), consistent with point 1: the groups are
     coupled through the shared pipes.
- So the lever is the softmax length per group (fewer / better-scheduled exp-phase instructions: our exps carry +7.2 per
  million steps of MUFU "wait" stalls vs FA3's, 45.39), judged by tensor idle per step in this trace, then raced.

### 45.46 2026-10-19: the softmax schedule, ours vs FA3 (same-box ncu SASS), and the row-split candidates R8 / R16 / R32

Script `session_2026_10_19/deep/softmax_sched.py` on the 45.44 source CSVs (steady loop, from the QK wait to the loop end;
M = exp, x = row-max FMNMX, f FFMA, a FADD, m FMUL, p F2FP pack, s SHFL, W wait):
- FA3 (387 instr, 66 exps): `W x*32 s x . x s x m a f f x f f f x f m M x x f f M x x M x x f f M x f x M ...`: after row 1's
  max (32 FMNMX + shuffles) it runs row 1's exps WHILE computing row 2's max (2 FMNMX between exps); 13 exps before its P.V
  wait; exps with 0-1 other instructions before the next: 11 of 66; median gap 3; the rescale FMULs and packs fill the gaps
  after the wait (`M p m m s m m m M p m m m m m m M ...`).
- Ours K5q (385 instr): `W x*47 s x*15 s x s x x x m s f*8 M M x M x m M f*6 M f f M a M M a M M a M M ...`: BOTH rows' max
  first (62 FMNMX + 4 shuffles), then part A's 16 exps nearly back to back; 25 of 66 exps have 0-1 instructions before the
  next (one exp keeps the exp unit busy ~8 cycles per warp, 10-17 note), so the warp waits: exps carry 14.2% of the kernel's
  stall samples, 71% of it "wait" (FA3 12.9%, 64%). Part B is interleaved like FA3's except its tail (`M a p M a p ...`).
- Cause in the source: part A = registers [0, K1_SPLIT), which hold BOTH rows (reg % 4 < 2 = row 0), so its first exp needs
  both maxes.
**Candidates (switch K10_ROWSPLIT, from the box K5q f35982af):** part A = the first K1_SPLIT scores of ROW 0 only; the ordering
arrive also depends on row 1's max (this_bb1, cc1) so that max stays before the wait; each row sum still adds its registers in
increasing order and every exp is the same expression -> bit-identical to K5q expected. R8 (817be23a), R16 (35ed6226), R32
(1276f48b). Local SASS (nvcc 13.0): 168 registers, 0 spills each; R16 part A = `x*32 s x*10 s x*20 m x f x f f f x f f M x f f f
x f f M f s f f f f M M M a x s M a M M a x M a x a M ...` (row 1's max partly mixed in; 0-1 gaps 22 vs 25); R8 22; R32 31
(its tail `M a M a ...` is worse). ptxas still puts most of row 1's max first, so FA3's "2 FMNMX per exp" pattern is not
reproduced; the race decides. Run: `scripts/run_r_1019.sh` (b54fa916): build, quick suite, forward race vs K5q (8 rounds,
4 libraries, rotating order), ncu of each, packed into r_results.tgz.
**R race result (r2_1019, 12:06-12:14 IST, GPU 0, 8 rounds, 4 libraries rotating; the first attempt r_1019 was stopped at
12:05 because Mahesh's 2-GPU job (/root/mahesh/BluTrain, mpirun -np 2) started on GPU 0 13 s after our race; its quick suites
had passed):** quick suite 296/296 for R8 (be25425b), R16 (d99ddebe), R32 (9f5e6f46). vs K5q: R8 dense +0.34% (t 1.8, 2/8),
packed -0.07% (t -0.8); R16 dense **+0.76%** (t 4.3, 0/8), packed 0.00%; R32 dense +0.12% (t 0.7), packed **+2.54%** (t 27,
0/8). K5q's own rounds 1.077-1.090 / 0.455-0.457 ms (= this morning): no interference. **Rejected**: changing which exps sit
before the wait does not shorten the step. Note: group 1's softmax alone (643 + 535 = 1,178 cycles in 45.45) is already about
FA3's (~1,186): the softmax code is not slower than FA3's; the extra comes from group 0's ~299-cycle K/V wait and the overlap.

### 45.47 2026-10-19: what FA3 really does with K and V (read from its source, flash-attention 94e22c9 hopper/)

- Head dim 128, causal: tile 128 x 128, P.V from registers, IntraWGOverlap on (tile_size.h:31-33), **kStages = 2**
  (flash_fwd_launch_template.h:47), **separate pipelines for K and V** (MainloopPipelineK / MainloopPipelineV,
  mainloop_fwd_sm90_tma_gmma_ws.hpp:284-285).
- Loader (mainloop:860-880): per tile `load_K(n_block, smem_pipe_write)` then `load_V(n_block_prev, smem_pipe_write_v)`
  ("write_v is always 1 step behind"): K(n), then V of the PREVIOUS tile; V runs one tile behind K.
- Consumer fwd_step (mainloop:1170-1206): group 0 alone waits K(n) (1175) -> baton sync -> QK(n) -> group 0 alone waits
  V(prev) (1180) -> P.V(prev) -> baton arrive -> warpgroup_wait<1> -> **pipeline_k.consumer_release: K(n) freed right after
  QK(n), before the softmax (1185)** -> mask, max, online softmax -> warpgroup_wait<0> -> pipeline_v.consumer_release (1199)
  -> convert P -> rescale O.
- Ours (K5q): 3 stages of K+V together; group 0 waits for K(n) AND V(n) before QK(n) although V(n) is used one step later;
  the K+V slot is released only after P.V, one step + part A later than FA3 releases K.
- Caution: with 3 slots our K has MORE lead time (in steps) than FA3's 2-stage K ring; what differs is that we wait for 64 KB
  (K+V) instead of 32 KB, and our loads miss L2 more (DRAM 520 vs 365 MB, 45.44). W2 (45.31: waiting V separately) only
  trimmed 50-75 cycles. So first the knock-out below: does the step get faster with NO K/V wait at all?
**Knock-out KOkv** (cands/fwd_KOkv.cu 8820517e, MEASUREMENT ONLY, results wrong on purpose): K5q without group 0's in-loop
K/V wait. Run `scripts/run_kokv_1019.sh` (3565dd99). Also fixed: `ab_multi.sh` (8e859127) reverses the order on even rounds
only for 3+ libraries, so a 2-library race now alternates A B / B A (before, the candidate ran first in every round).
**KOkv result (kokv_1019, 12:32-12:34 IST, GPU 0; stopped by the user after 3 of 8 rounds; the order alternated KOkv K5q /
K5q KOkv as intended).** First launch 12:31 failed at build_cands' check (fwd_KOkv.cu not yet copied). Verified that the
knock-out is in the binary: dense kernel 3,632 vs 3,648 instructions, 44 vs 48 mbarrier TRYWAIT (one in-loop wait removed);
the build log shows GQA_fwd_sm90.cu compiled (this box compiles the one file in seconds; "libtensor is up-to-date" is the
Makefile's closing summary). Per round, KOkv vs K5q: dense -0.19 / -0.45 / -0.24% (**mean -0.30%**), packed -1.97 / -1.64 /
-1.76% (**mean -1.79%**). Even with NO K/V wait at all (the upper bound of any K/V pipeline change, FA3's included) the
dense forward gains only ~0.3%: the wait is slack there (its time moves into the overlapping softmax phases, as W2 showed
on 10-17). **FA3's split K/V pipeline is not the lever for the dense gap (8.4%)**; it could give up to ~1.8% on packed, where
we already beat FA3 (0.936x). Remaining dense explanation: the two groups' softmax phases overlap and share each SMSP's
issue slot and exp unit (45.45); FA3 reaches a tensor-bound step with the same instruction counts.

### 45.48 2026-10-19: SKV -- FA3's split K / V pipeline in K5q (user's choice A), protocol checked on the CPU first

**Design** (`session_2026_10_19/scripts/cands/fwd_SKV.cu` d05c0b23, switch `SKV`, from the box K5q f35982af; FA3 45.47):
- Barriers: `full` / `empty` become K's ring; new `full_v` / `empty_v` = V's ring (3 slots each, same slot index per tile;
  +48 B shared memory). With SKV 0, full_v / empty_v alias full / empty.
- Loader: per tile K(n) (32 KB, its own barrier), then V(n-1); V(last) on the last pass (FA3 mainloop:860-880).
- Loop step: group 0 waits K(n) -> baton -> QK(n) -> group 0 waits V(n-1) -> P.V(n-1) -> baton -> wait QK(n) -> **release
  K(n)** (FA3 :1185) unless it is the unit's last tile (K4: O is staged in that K buffer, released after the O store as
  before) -> softmax -> wait P.V(n-1) -> release V(n-1). Prologue: K(0) released after QK(0) (unless last). Tail: both groups
  wait V(last) before the last P.V and release it after; K(last) after the O store (K4). Inactive tail (packed): wait K(next)
  and V(cur), release both.
**CPU protocol model** (`session_2026_10_19/deep/protocol_sim_skv.py`, extended from 10-17's protocol_sim_k4.py): + SKV, +
CORRUPTION checks (every K / V read at QK / P.V issue, again at its release, and the K4 O staging must find the expected tile
still in the slot). 3,000 dense + 3,000 packed random unit lists each: K4cnt (today's K5q) clean; **SKV clean (no deadlock, no
corruption)**; SKV_noVwait (inactive tail without the V wait) also clean (the per-tile baton already paces it; the wait is
kept as cheap insurance); **SKV_lastK (K(last) released after its QK, ignoring K4) is CAUGHT** in ~half the runs ("O staged
in K(tile 53): slot 2 holds tile 56"): the checks are not vacuous. (A first SKV run deadlocked everywhere: a model bug, the
K4 counter signal did not know the new version name; fixed.)
**Local build (nvcc 13.0):** SKV / K5q both 168 registers, 0 spills; **SKV with the switch off compiles to the SAME SASS as
K5q** (3,672 / 3,504 lines identical); SKV: 66 vs 48 mbarrier TRYWAIT, 33 vs 24 arrives (V's ring). Expected results
bit-identical to K5q (only the synchronisation changes). Run: `scripts/run_skv_1019.sh` (build, quick suite, race, ncu).
**SKV race result (skv_1019, 12:46-12:51 IST, GPU 0, free clocks, 8 rounds, order alternating SKV K5q / K5q SKV).** Built SKV
(source d05c0b23 -> library 841bb583; F3c back in the tree). Quick suite **296 / 296** (packed == alone bit-exact included).
| SKV vs K5q, forward only | mean of 8 paired rounds | t | faster rounds | medians K5q / SKV (ms) |
|---|---|---|---|---|
| dense B8 18/6 | +0.20% | 1.5 | 1 / 8 | 1.0825 / 1.0865 |
| packed real_rows_752 | **+0.87%** | 9.1 | 0 / 8 | 0.4555 / 0.4598 |
K5q's own rounds 1.0775-1.0884 / 0.4548-0.4569 ms (as in the morning races: no interference). **SKV is REJECTED**: correct,
but no faster on dense and slower on packed (KOkv's bound was -0.30% / -1.79%).
**Why (ncu, same box, `session_2026_10_19/deep/ncu_skv_vs_k5q.py` fd3139ba; K5q = ncu_fa3_k5q_1019, SKV = skv_1019):**
| ncu, one launch | dense K5q | dense SKV | packed K5q | packed SKV |
|---|---|---|---|---|
| cycles (average SM) | 1,574,797 | 1,556,227 (**-1.18%**) | 685,282 | 690,168 (+0.71%) |
| tensor-idle cycles | 388,769 | 370,966 (-17,803) | 235,614 | 240,826 (+5,212) |
| warp instructions | 322.0M | 329.0M (+2.2%) | 130.7M | 133.3M (+1.9%) |
| average SM clock during the launch | 1,685 MHz | 1,678 MHz | 1,718 MHz | 1,727 MHz |
- Dense: SKV does what FA3's split is for: group 0 starts QK(n) as soon as K(n) is in (stall time at the mbarrier waits -25,682
  cycle-equivalents per SM; the step-start K/V wait site 46,056 -> 16,201), and 4.6% of the tensor-idle time goes. But the
  free-clock race shows no gain (see 45.49: the forward runs at the 700 W power cap, so removed waiting is paid back in clock
  unless energy per forward also drops; SKV adds 2.2% instructions). **[CORRECTED in 45.50: the measured clock of SKV is NOT
  lower (+0.3% vs K5q). At the real power-capped clock (~1.55 GHz; ncu ran ~1.68 GHz) a memory wait is ~8% fewer cycles, so
  the K / V wait SKV removes is mostly gone already, and SKV's extra instructions make it +0.4% cycles per forward.]**
- The extra instructions are the second ring's bookkeeping, per tile and group: BRA +1.2M, BSSY / BSYNC +1.5M (the
  `if (lane_wg == 0) mbar_arrive` blocks), ISETP / LEA / ULEA / SHF (V barrier addresses), SYNCS.ARRIVE +0.6M, TRYWAIT +0.34M.
- Packed: short units, so the per-tile extra work is not covered: +0.71% cycles at ncu, +0.87% in the race.
**Protocol model, parked variant:** `protocol_sim_skv.py` (bcd9479e) also has SKV2 (V(n) loaded right after K(n), K released
after QK, V after P.V): clean on 3,000 dense + 3,000 packed lists; its wrong twin SKV2_lastK is caught (corruption, ~half the
runs; the negative-test switches now match by suffix, so a new version name cannot skip them). Not built: SKV's result says
a second ring costs more instructions than it saves at free clocks.

### 45.49 2026-10-19 13:10 IST: the forward runs at the 700 W power cap; FA3 tells the L2 to keep K / V, we do not (candidate LH)

**1. The forward is power-capped at free clocks.** Same-box scoreboard sb_1019 (nvidia-smi every 100 ms, samples 2 s after the
start of GPU work): every one of the 16 runs (dense / packed x ours / FA3 / FA4 / cuDNN x 2) has a median power of **691-695 W**
(limit 700 W) and an SM clock of **1,560-1,755 MHz** (the H200's maximum is 1,980). Effective clock of the dense forward alone =
ncu cycles / free-clock kernel time: K5q 1,574,797 / 1,055 us = **~1.49 GHz**, FA3 1,423,158 / 974 us = **~1.46 GHz** (under ncu,
single launches with gaps, the same kernels averaged 1.685 / 1.633 GHz). Consequences:
- At the cap, time = energy per forward / ~700 W. Removing idle cycles helps only as far as the clock does not drop to pay for
  the higher power per cycle; extra instructions or memory traffic cost time even if the cycles do not change. FA3 converts its
  9.6% fewer cycles into 7.7% less time and runs ~2% lower clock than ours; SKV's 1.2% fewer cycles converted to nothing.
  **[CORRECTED in 45.50 by direct measurement: SKV's clock is not lower; its ncu cycle saving does not exist at the real clock.
  FA3, forward alone at the cap: 3.8% LOWER clock than K5q and 8.1% fewer cycles per forward.]**
- Correction to 45.44 / 45.47: on this box ncu did NOT run every kernel at one locked clock: the average SM clock per profiled
  launch is 1,633-1,735 MHz depending on the kernel (FA3 dense 1,633, K5q dense 1,685, SKV dense 1,678, packed 1,718-1,735).
  The cycle comparisons stand (cycles are counted per kernel), but ncu clocks are not "the locked base clock".
- Next measurement (in run_lh_1019 below): the steady clock each forward actually gets, per library, with power and the
  driver's "power cap active" flag (`clocks_event_reasons.sw_power_cap`).

**2. FA3 tells the L2 to keep K and V and to drop Q first; we give no hint.** Source (flash-attention 94e22c9
hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp): K and V TMA loads `TMA::CacheHintSm90::EVICT_LAST` (753, 767), Q `EVICT_FIRST`
(820, when not split); O = TMA store without a hint (epilogue_fwd.hpp:48 / 322). FA3's SASS (ncu source page): `UTMALDG.4D
[..], [..], desc[UR35]` with UR35 = 0x14f00000.. (8 + 1 sites) or 0x12f00000.. (1 site, Q). Ours: `UTMALDG.2D [..], [..]`, no
policy. ncu, dense B8 18/6, same box:
| L2 / DRAM, dense | K5q | FA3 | ours / FA3 |
|---|---|---|---|
| L2 sectors with evict_last / evict_first / normal | 0 / 0.02M / 156.1M | 145.6M / 7.1M / 14.3M | |
| L2 lookup misses (sectors) | 29.9M | 23.1M | **1.30x** |
| L2 hit rate | 80.6% | 86.6% | |
| DRAM reads | 520 MB | 365 MB | **1.43x** |
| cross-partition L2 fabric sectors | 16.3M | 13.9M | 1.18x |
Packed (real_rows_752): DRAM 258 vs 255 MB, misses 13.6M vs 13.5M: no difference there (short segments, K / V reused at once).
The work ORDER is already FA3's (our G5: sections of 48 (batch, head) pairs = 16 kv heads x 2 MB = 32 MB of K / V, longest
query pair first, a work counter; FA3 DynamicPersistentTileScheduler, tile_scheduler.hpp:218-310, the same 48), so the hint
is the remaining L2 difference: without it, Q (151 MB, read once) and O (151 MB, written once) push K / V (reused by every
query pair of the section) out of the L2. At the power cap, the extra ~155 MB of DRAM reads and the misses cost energy.
Also seen in FA3, not copied yet: an L2 PREFETCH of each new Q tile at the start of the producer's load (mainloop:668,
`UTMAPF.L2`), before the Q buffer is free.

**3. Candidates (from the box K5q f35982af, switch `L2_HINT`; `session_2026_10_19/scripts/cands/`):**
- **LH** (fwd_LH.cu 785bd68b, L2_HINT 1): FA3's hints exactly: Q loads evict-first, K and V loads evict-last; policy values =
  CUTLASS `CacheHintSm90` (EVICT_FIRST 0x12F0000000000000, EVICT_LAST 0x14F0000000000000; FA4's copy_utils.py uses the same).
- **LHO** (fwd_LHO.cu daadb501, L2_HINT 2): LH + the O stores (`st.global.v4`) evict-first.
- LH0 (fwd_LH0.cu 5baf9b5c, L2_HINT 0, check only).
Local build (nvcc 13.0, sm_90a): **LH0 = K5q's SASS exactly** (7,178 lines, 0 differ); LH: the 64 `UTMALDG.2D` now carry the
policy operand `desc[URx]` (the same form as FA3's), +128 lines = `UMOV`s of the policy values in the producer warp; LHO: also
`STG.E.128 desc[UR12]` (the policy descriptor) for the O stores; all 168 registers, 0 spills. Only the L2 policy changes, the
values cannot: results bit-identical to K5q expected (the quick suite checks).
**Run** `scripts/run_lh_1019.sh` (3987f98e, GPU 0, ~17 min): build LH LHO (F3c back in the tree), quick suite each, forward-only
race K5q / LH / LHO (8 rounds, rotating order), ncu of each, then `scripts/clock_power_check.sh` (b0dca7ab): K5q, SKV, LH, LHO
and FA3, dense + packed forwards, ~16 s per run, order A B C D E E D C B A, nvidia-smi every 500 ms (clocks.sm,
power.draw.instant, sw_power_cap flag, temperature, utilisation) + each run's start / end; read with
`deep/clock_power_read.py` (4ea49733; checked on a synthetic run: it recovers the planted clocks). For FA3 forward-only,
`rivals_free_1016.py` got an opt-in switch RIVAL_FWD_ONLY=1 (de42be66, was fbd31e90; default unchanged = forward + backward).

### 45.50 2026-10-19 14:05 IST: LH / LHO result -- the L2 hints help dense through ENERGY (same cycles), hurt packed; LHOd

**Run lh_1019** (`scripts/run_lh_1019.sh`, GPU 0, 13:49-14:04 IST). Built LH (785bd68b -> library e7ba3123), LHO (daadb501 ->
544d4e5e), F3c back in the tree (a0908d99). Quick suites **296 / 296** both. Files: session_2026_10_19/box_runs/lh_1019.
**Race** (forward only, 8 rounds, 3 libraries in rotating order, 4,000 dense / 8,000 packed forwards per run = ~4.3 s):
| vs K5q, forward | dense | packed |
|---|---|---|
| LH (Q evict-first, K / V evict-last = FA3's) | **-1.18%** (t -5.5, faster 8/8) | +0.56% (t 6.9, 0/8) |
| LHO (LH + O stores evict-first) | **-1.49%** (t -6.3, 8/8) | +0.68% (t 8.7, 0/8) |
| LHO vs LH | -0.31% (t -3.4, 7/8) | +0.12% (t 1.3) |
Medians (ms): dense K5q 1.0909, LH 1.0772, LHO 1.0726; packed K5q 0.4567, LH 0.4592, LHO 0.4598.
**ncu** (`deep/ncu_cand_vs_k5q.py` 0b4b775a = ncu_skv_vs_k5q.py for any candidate; same SKV output):
| dense | K5q | LH | LHO | FA3 |
|---|---|---|---|---|
| cycles (average SM) | 1,574,797 | 1,576,997 (+0.14%) | 1,575,826 (+0.07%) | 1,423,158 |
| DRAM reads | 520 MB | **447 MB** | **432 MB** | 365 MB |
| L2 misses (sectors) / hit rate | 29.9M / 80.6% | 26.0M / 82.8% | 24.5M / 84.1% | 23.1M / 86.6% |
| L2 sectors evict-last | 0 | 138.3M | 137.4M | 145.6M |
| cross-partition fabric sectors | 16.3M | 14.3M | 13.6M | 13.9M |
Packed: cycles 685,282 / 688,953 (+0.54%) / 691,573 (+0.92%); DRAM 258.2 MB in all three (nothing to save); the extra time is
at ONE place, each unit's wait for its first K / V tile (`full[]`, 37,872 executions): 18,813 -> 23,144 (LH) / 21,382 (LHO)
cycle-equivalents per SM. Why the first tile arrives later with the hints is not known (misses unchanged, fabric +3%).
**Clock / power check** (`scripts/clock_power_check.sh` b0dca7ab, read by `deep/clock_power_read.py` 4ea49733; 13:58-14:04 IST;
each entry twice, mirrored order, ~16 s of forwards per run, samples from 4 s after the start; the driver's sw_power_cap
flag was active in 100% of the samples of every run, power 671-695 W):
| forward alone, steady state | SM clock | time per forward | cycles per forward (clock x time) |
|---|---|---|---|
| dense K5q | 1,552.9 MHz | 1.1024 ms | = 100 |
| dense SKV | +0.32% | +0.11% | +0.44% |
| dense LH | **+0.98%** | **-0.44%** | +0.55% |
| dense LHO | **+0.88%** | **-0.59%** | +0.30% |
| dense FA3 | **-3.77%** (1,494) | **-4.53%** (1.0524) | **-8.14%** |
| packed K5q | 1,637.7 MHz | 0.4633 ms | = 100 |
| packed SKV / LH / LHO | +1.06 / +0.43 / +0.82% | +1.07 / +0.59 / +0.95% | +2.13 / +1.03 / +1.78% |
| packed FA3 | +2.39% | +5.91% | +8.45% |
(Our time = the harness's median per forward; FA3's = wall time / 16,400 or 34,000 forwards from rivals_free_1016.py with
RIVAL_FWD_ONLY=1. Clock x time is 8.7% (ours) / 10.5% (FA3) above ncu's cycle counts: launch gaps, sampled vs effective
clock, different timers; compare within one harness.)
**What this shows:**
1. **Dense: the hints save ENERGY, not cycles.** Same cycles (ncu +0.07-0.14%), 14-17% fewer DRAM reads; at the 700 W cap the
   GPU then runs ~0.9-1.0% faster clock. In 4.3-s races that is -1.2 / -1.5%; in 16-s steady runs -0.44 / -0.59% (2 runs each,
   noisier; the first seconds of a run are not yet at the steady power: H200 power settles after ~4 s).
2. **Packed: the hints cost cycles** (+0.5 / +0.9% ncu, +1.0 / +1.8% at free clocks) and save nothing -> no hints in packed.
3. **SKV, corrected:** its clock is not lower (+0.32%); at the real clock (~1.55 GHz vs ncu's ~1.68) a memory wait is fewer
   cycles, so the K / V wait SKV removes is mostly gone, and its extra instructions cost +0.44% cycles (inline corrections
   in 45.48 / 45.49).
4. **FA3, forward alone at the cap: 3.8% LOWER clock than K5q, 8.1% fewer cycles, 4.5% faster.** Its busier tensor pipe draws
   more power per cycle. In the fwd + bwd scoreboard (45.43) the forward ratio was 1.084, near the cycle ratio: consistent
   with the clock being set by the fwd + bwd MIX there, not by the forward alone. In training the clock is set by the whole
   step (GEMMs, attention, ...): cycle savings in the forward convert ~fully, energy savings (like LH) only partly.
5. **The dense gap is cycles: FA3 needs ~8% fewer per forward** (45.45: ~420 cycles per step on group 0's chain).
**LHOd** (`cands/fwd_LHOd.cu` 8a7b4312, from LHO + switch `L2_HINT_DENSE` 1): `tma_load_2d<HINT>` / `stg128<HINT>` take the hint
as a template flag, `HINT_LD` / `HINT_O` = on except in the packed kernel. Local nvcc 13.0, per kernel: **LHOd dense == LHO
dense (0 differing SASS lines), LHOd packed == K5q packed (0)**; checks fwd_LHOd_off.cu (036db397, L2_HINT 0) == K5q in both
kernels, fwd_LHOd_all.cu (f1fea0e0, L2_HINT 2) == LHO in both; negative control dense LHOd vs K5q = 756 lines. 168 registers,
0 spills. So LHOd = LHO's measured dense speed + K5q's packed speed, by construction.
**Run** `scripts/run_lhod_1019.sh` (7321d9ed, ~8 min): build LHOd (F3c back), the same per-kernel SASS check on the
box's CUDA 13.4 build (cuobjdump of libs/vj2_LHOd vs vj2_LHO / vj2_K5q; aborts if a dump is empty), quick suite, then a
FORWARD + BACKWARD race vs K5q (8 rounds; the scoreboard's regime, where the backward shares the clock), no ncu.
**LHOd result (lhod_1019, 14:10-14:17 IST, GPU 0).** Built LHOd (8a7b4312 -> library 00be51bd; F3c back, 5eb66f27). Box SASS
check (CUDA 13.4 build): **dense LHOd vs LHO 0 differing lines (3,713), packed LHOd vs K5q 0 (3,481)**. Quick suite **296 / 296**.
Forward + backward race, 8 rounds, A B / B A (1,000 dense / 2,000 packed iterations of forward + backward per run):
| LHOd vs K5q | forward | backward (code unchanged) |
|---|---|---|
| dense B8 18/6 | **-0.34%** (t -2.9, faster 6/8) | **-0.33%** (t -2.3, 6/8) |
| packed real_rows_752 | +0.03% (t 0.6, 3/8) | -0.00% (t 0.0, 3/8) |
Medians (ms): dense fwd K5q 1.0532 / LHOd 1.0483, bwd 3.1568 / 3.1376; packed fwd 0.4394 / 0.4393, bwd 1.5006 / 1.5009.
- In this regime the backward shares the clock: the forward's DRAM saving lowers the average power, the clock rises for both
  kernels, so the UNCHANGED backward is 0.33% faster too (the expected sign of an energy gain). The forward's gain is
  smaller than in the forward-only race (-1.49%), as 45.50 point 4 predicted. Packed: identical code, null, as designed.
- (The forward runs ~3.5% faster in this regime than in forward-only loops: dense K5q 1.0532 vs 1.0909 ms, packed 0.4394
  vs 0.4567: alone, the forward is the most power-dense kernel and sets a lower clock.)
- **LHOd is the new best forward** (K5q + FA3's L2 hints in the dense kernel; packed = K5q exactly). Adoption into the laptop
  master GQA_fwd_sm90.cu (still F3c) is the user's decision.
**Also found:** FA3's BACKWARD uses L2 hints too (mainloop_bwd_sm90_tma_gmma_ws.hpp: Q and dO loaded EVICT_LAST 512 / 533 / 540
/ 550, the dQ accumulation bulk-reduce EVICT_LAST 647; epilogue_bwd.hpp: dK / dV accumulation EVICT_LAST 471 / 502); in the
backward Q and dO are the operands re-read by every K / V block. Our backward has no hints. Not measured yet (needs an ncu
of our backward vs FA3's DRAM / L2 first).

### 45.51 2026-10-19 15:20 IST: measuring FA3's own per-step timeline on the same box (the dense CYCLE gap), plan + checks

**Why.** After 45.50 the dense gap is cycles: FA3 needs ~8% fewer per forward. Our per-step timeline (45.45) shows WHERE we
lose (group 0's chain ~420 cycles over the 2,048-cycle tensor step), but FA3's schedule was so far only inferred from its
source and ncu, and the fixes built on those inferences (L1b, R8-R32, SKV) all failed. So: measure FA3's step the same way.
**FA3 measurement build** (`session_2026_10_19/fa3_tl/`: mainloop_fwd_sm90_tma_gmma_ws.hpp 5572bf6f, patch base 2ed79549 =
flash-attention 94e22c9; instantiations flash_fwd_hdim128_bf16_sm90_tl2.cu f5711129 / _tl3.cu 4a1faea9, base 1ecaa31c):
clock stamps in FA3's fwd_step (mainloop :1170-1207) for each consumer warpgroup's FIRST tile in every CTA, written to a
device buffer read back through an exported `fa3_tl_read` (ctypes); file layout = our V62_TRACE dumps, so
deep/timeline_read.py reads FA3's and ours the same way. Points: s0 step start | s1 scheduler barrier passed (+ K landed
for warpgroup 0) | s2 QK(n) done (level 3 only) | step end = the next step's s0.
**Placement rules found by checking the SASS** (laptop nvcc 13.0, FA3's pinned CUTLASS 7127592 downloaded to the scratch
area, FA3's setup.py nvcc flags + torch's default flags): FA3's own loop puts **13 of its 66 exps before the wait for
P.V(n-1), 53 after** (mixed with the 64 rescale FMULs), the same as the box SASS of 45.46. A clock read anywhere inside the
softmax made ptxas put all 66 before the wait; stamps kept in registers to the step end: 23; so stamps are taken only at
synchronisation points and each is stored the moment it is read. **Level 2** (s0, s1): 16 / 66 before the wait, the loop's
instructions otherwise FA3's (482 vs 483 in the steady loop, one PLOP3); **level 3** (+ s2): 27 / 66 (cross-check only).
Compute and sync instructions (HGMMA, MUFU, FMNMX, FMUL, FFMA, FADD, F2FP, SHFL, BAR, WARPGROUP, SYNCS) unchanged at both
levels; 168 registers.
**Ours with the same stamps:** `cands/fwd_TL2.cu` (c08c978b) / `fwd_TL3.cu` (91d6f0c5) = LHOd + switch `V62_TL` 2 / 3, the
same points and rules, file via V62_TRACE_FILE / V62_TRACE_AT (as trace_1017.sh). Local SASS: exps before / after the P.V
wait 17 / 49 at both levels = LHOd exactly (our split is pinned by the order_bar trick), compute / sync instructions
unchanged, 168 registers, 0 spills; host + device compile checked.
**Laptop toolchain note:** the laptop got a package upgrade today 14:56-15:00 IST (glibc 2.39 -> 2.43, libc6-dev
2.43-2ubuntu2.4, linux-libc-dev 7.0.0-38). glibc 2.43 declares `rsqrt` / `rsqrtf` (C23), which CUDA 13.0's
crt/math_functions.h also declares -> EVERY local CUDA compile now fails ("exception specification is incompatible with
that of previous function rsqrt"). Local checks use a shadow copy of bits/mathcalls.h without that one line
(-I<scratch>/sysfix); proven harmless: LHOd built with it = the pre-upgrade SASS, 0 lines differ. The box (CUDA 13.4) is
not affected.
**Box scripts:** `scripts/fa3_tl_build.sh <2|3>` (1ba5cb48: copies /root/gautam_K/flash-attention to fa3_tl<level>, checks the
patch base md5s, builds IN PLACE, benv's FA3 untouched; switches: benv's + no split / paged / softcap / PackGQA / append-KV /
local / varlen / cluster, none of them in the dense causal kernel), `scripts/build_tl_1019.sh` (5ee9352f: both FA3 levels in
parallel + build_cands TL2 TL3 F3c, CPU only), `scripts/run_tl_1019.sh` (51b089dd: GPU 0, ~3 min: trace_1017.sh for vj2_TL2 /
vj2_TL3, `scripts/fa3_tl_run.py` (67a7eadd) for FA3 levels 2 / 3, all in the forward + backward regime, launches 900 / 950,
nvidia-smi every 200 ms). Comparison: the period, each group's wait for K / baton (s0 -> s1), the offset between the groups,
the chain s1 -> next s0, and (level 3) QK latency vs softmax; levels 2 vs 3 show whether the s2 stamp changes the timing.
**RESULT (tl_1019, 15:22-15:27 IST, GPU 0).** Build (build_tl_1019.sh): ours TL2 c08c978b -> library b3dd6667, TL3 91d6f0c5 ->
12e6b4f7 (F3c back, ceb62227); FA3 level 2 _C.abi3.so f99994dd, level 3 a5f569e9 (13 backward kernels each; benv's FA3 files
unchanged). Read-only checks on the box: dense kernel clock reads TL2 4 / TL3 6 / LHOd 0; FA3 .so clock reads 6 / 9.
Run: our traces with trace_1017.sh (forward + backward bench, launches 900 / 950: TL2 fwd med 1.0876 / 1.0892 ms, TL3 1.0981 /
1.0983), FA3 with fa3_tl_run.py (1,000 forward + backward: level 2 4.3055 ms per iteration, level 3 4.3325). Each trace 2,916
steady steps over 132 CTAs (deep/timeline_read.py); launches 900 and 950 identical to the cycle.
| dense, steady loop step (cycles), level 2 | ours (LHOd + stamps) | FA3 (+ the same stamps) |
|---|---|---|
| period | **2,551** | **2,367** |
| group 0: wait for K + baton (s0 -> s1) | 122 | 176 |
| group 1: wait for the baton | 229 | 117 |
| group 0: baton -> next step (issue, QK, softmax, P.V wait, rescale, packs, loop) | **2,429** | **2,191** |
| group 1: the same | 2,322 | 2,250 |
| group 1 gets the baton after group 0 | 914 | 852 |
| "no GEMM queued" estimate | 19.0% | 13.1% |
- **Our step is 7.8% longer than FA3's: the whole dense gap (~8% cycles) sits in the steady loop step**, on group 0's chain
  (+238). The 45.44 fit ("FA3's step is tensor-bound, ~0 idle per step") was wrong: FA3 too idles ~300 cycles per step; the
  2-point fit had no residual to show the error.
- Level 3: ours 2,551 again (the s2 stamp changes nothing in ours: issue + QK wait 870 / 1,020, the rest 1,556 / 1,254 for
  groups 0 / 1); FA3 2,616 (+249 per step: the s2 stamp moves FA3's exps, as the SASS check predicted) -> FA3's level-3
  split is not used.
**Where one step's time goes, from ncu's stall sampling (no instrumentation; same-box ncu of 45.44 / 45.50; share of the
steady-loop samples x the traced period; both groups' average):** `deep/loop_wait_split.py` (by stall REASON: the baton
BAR.SYNC is DEFER_BLOCKING, its wait shows as "barrier" samples on the next instructions, so a split by position mislabels it).
| cycles per step | FA3 | ours (LHOd) | ours - FA3 |
|---|---|---|---|
| baton wait ("barrier" samples) | 239 | 173 | -66 |
| K / V wait (group 0's TRYWAIT) | 88 | 24 | -64 |
| issue phase (loop top .. last HGMMA, other samples) | 716 | 834 | +118 |
|   of it the 16 HGMMA instructions | 489 | 499 | +10 |
|   of it the other instructions (loop control, stage, descriptors) | 227 | 335 | **+108** |
| QK wait (+ between issue and it) | 7 | 18 | +11 |
| softmax before the P.V wait | 544 | 650 | **+106** |
| P.V wait | 2 | 1 | -1 |
| after the P.V wait (exps, rescale, packs, loop) | 771 | 850 | **+79** |
| period | 2,367 | 2,551 | +184 |
- Our warps WORK ~300 cycles more per step and WAIT ~130 less; FA3's warps run out of work and wait for the tensor pipe
  (an unlabeled stall reason on its first P.V HGMMA = 204 cycles per step, ours 78 there): FA3 is closer to tensor-bound.
- The HGMMA issue itself is not the difference (both ~500 cycles per step: issuing 16 HGMMAs is paced by the tensor pipe).
  FA3 issues its 8 QK HGMMAs back to back (8 descriptor registers prepared before), ours with 4 descriptor instructions
  between each (all through UR20); L1b made our QK HGMMAs back to back too (QK stalls 116 -> 73) but the P.V issue absorbed
  it (394 -> 422): issue time is set by the pipe, not by the instruction count (why L1b did not help).
- The +108 of other issue-phase instructions: after the baton our stage lives in a regular register and goes through R2UR
  into the uniform datapath before the descriptors (R2UR, ULEA, ULOP3, ULEA.HI chains with dispatch stalls); FA3 keeps its
  pipeline state in uniform registers (UIADD3 / USEL / ULOP3) and its descriptors ready.
- The softmax (+185 in total): the same instruction count (385 vs 383; 66 exps each); ours has more fixed-latency
  dependency waits (+97) and more dispatch stalls (+100: register-file / dispatch conflicts in the FMUL / FFMA / FADD
  stretches), +24 not-selected. Not the row max: FA3's max is ONE linear chain per row (depth 36, `deep` check), ours two
  interleaved chains (depth 21).
- `deep/loop_region_reasons.py` (stall reasons per region, instruction mix), `deep/loop_regions_ncu.py` (first, position-
  based version: kept for the record, its "baton" figure is wrong for the reason above).
**Next levers (cheap first):** (1) consumers' register budget FA3's way (setmaxnreg 24 / 240 instead of 40 / 232; tried on
F3 10-16 = +1.1%, but the code has changed since and the dispatch stalls point at register-file conflicts), (2) loop state
and descriptors in uniform registers, prepared before the baton (the +108), (3) the softmax's dependency waits.

### 45.52 2026-10-19 16:20 IST: FA3's loop vs ours, instruction by instruction (PC sampling) -> the exp packing; candidate SM3

**User's direction (16:00):** no more candidates built on assumptions: FA3's source is local; find exactly what we do
differently, with proof, then change only that. Method: ncu PC sampling (= the "Warp Stall Sampling" columns of the ncu
source page: the stall reason of each warp at each instruction, no instrumentation) of FA3's real kernel and ours, same box
(fa3_dense / ncu_vj2_LHO_dense, LHO dense == LHOd dense), the steady loop body (instructions run once per step per warp),
times scaled by the traced period (45.51: FA3 2,367, ours 2,551).
**1. Same work.** `deep/loop_diff_fa3.py` (6cf7ee4c): per step and warp, both run 66 exps, 68 FMNMX, 66 FFMA, 68 FMUL, 32
F2FP, 4 SHFL, 16 HGMMA (FADD 64 vs 70, uniform 46 vs 65, integer 17 vs 15). The extra 184 cycles are STALLS on the same
instructions: exps +125, F2FP +38, FMUL +34, HGMMA P.V +48 / QK -37, FADD +13, FFMA +14, FMNMX -32, waits -130.
**2. The exps: +120 cycles of "wait" stalls** (ours 367, FA3 247 per step). Not their input: the FFMA producing an exp's
argument is far away in both (median 80 / 109 instructions; `deep/exp_producer_distance.py` 95237995). It is the exp unit's
throughput: an exp issued 0-2 instructions after the previous one waits ~7-9 cycles (`deep/exp_spacing.py` 56520ca9):
| exps issued ... after the previous exp | FA3: count / "wait" cycles | ours: count / "wait" cycles |
|---|---|---|
| 0 instructions between | 0 / 0 | 4 / 37 |
| 1 | 11 / 92 | 21 / 198 |
| 2 | 19 / 125 | 13 / 111 |
| 3-4 | 13 / 30 | 5 / 14 |
| 5+ | 23 / 1 | 23 / 7 |
| total | 248 | 367 (+119) |
**3. The cause in the source.** FA3 hopper/softmax.h: `thread_reduce_` = the row max as ONE dependent chain per row from the
previous max (columns in order; measured chain depth 36 incl. shuffles); `scale_apply_exp2` = exps ROW BY ROW; `reduce_sum`
after the exps, one chain per row from row_sum * scale. Its compiled loop: row 0's max (32 FMNMX), then row 0's exps WHILE
row 1's max chain runs (`M x x M x x M x x f f M`): 2 FMNMX between exps. Ours (K5q / LHOd softmax_tile): the max in 4
short interleaved chains (depth 21, finished fast), the exps of BOTH rows mixed (`reg % 4`), the sums right after each exp:
after the max nothing is left to overlap, so the exps go nearly back to back (`M M x M x m M ... M a M M a M M a`). (Not the
row max itself: ours is the faster one, -32 cycles.)
**4. Local SASS = box SASS for this measure** (`deep/sass_exp_pattern.py` d39e8e11 on cuobjdump listings, the mask-free step):
LHOd local 4 / 21 / 13 / 5 / 22 = box 4 / 21 / 13 / 5 / 22; FA3 local (CUTLASS 7127592) 0 / 11 / 19 / 13 / 22 = box. So a
change can be checked on the laptop before any GPU run.
**Candidate SM3** (`cands/fwd_SM3.cu` 4ef6ec49 = LHOd + switch `SM3`): FA3's structure exactly: max = one chain per row from the
previous max, columns in order, then quad_max; exps row by row (part A = row 0's first K1_SPLIT = 16 exps); sums = FA3's
order (l * cc, then the row's values in column order; row 0's first 16 adds in part A, which also pins the P.V wait through
order_after(rs0), the rest + row 1 in part B). Local SASS: **exps 0 / 13 / 22 / 7 / 23** (FA3 0 / 11 / 19 / 13 / 22), the
pattern `x*32 s ... M x x M x x M x x M` = FA3's, 17 exps before the P.V wait (FA3 13), same instruction counts, 168
registers, 0 spills; fwd_SM3off.cu (c366a999, switch off) = LHOd's SASS in both kernels (0 lines differ). Numerics: the row
sums are added in FA3's order (rounding differs from LHOd at the last bits); dense and packed run the same code, so packed ==
alone stays bit-exact (the quick suite checks it). Expected: part of the +119 exp wait (by the per-distance costs above,
~30-100 cycles per step = ~1-4% dense); measured by the run, not assumed.
**Run** `scripts/run_sm3_1019.sh` (b4042c5a, GPU 0, ~9 min): build, quick suite, forward race vs LHOd + ncu of SM3 (PC sampling:
did the exp waits drop?), forward + backward race vs LHOd.
**Still open (from the same table, next after SM3):** F2FP / FMUL dispatch stalls (+72: same instructions, more dispatch
stalls), the P.V HGMMA issue (+48), the issue-phase instructions (stage through R2UR, descriptors per HGMMA).
**SM3 RESULT (sm3_1019 forward-only + sm3fb_1019 forward + backward, 15:52-16:00 IST, GPU 0).** Built SM3 (4ef6ec49 -> library
a54c3ff8; F3c back, 9e2e6349). Quick suite **296 / 296**.
| SM3 vs LHOd, 8 rounds each | dense forward | packed forward | dense backward | packed backward |
|---|---|---|---|---|
| forward-only race | **-0.70%** (t -5.0, 8/8) | +0.48% (t 5.9, 0/8) | | |
| forward + backward race (training-like) | **-3.05%** (t -21.1, 8/8) | +0.45% (t 4.8, 0/8) | +0.46% (t 3.3, 1/8) | +0.16% (t 2.2) |
Medians, forward + backward race (ms): dense fwd LHOd 1.0447 / SM3 1.0139, bwd 3.1320 / 3.1578; packed fwd 0.4393 / 0.4407.
- **The prediction holds** (ncu PC sampling of SM3, same method): the exps' "wait" stalls 367 -> **245** cycles per step
  (FA3 247), exps 15.6% of the loop's samples (FA3 16.2%, LHOd 19.9%); ncu cycles dense **-3.8%**, packed -0.9%.
- Forward-only, the forward sets its own clock at the 700 W cap and gives back most of the cycle cut (-0.70%); with the
  backward in the loop (the scoreboard's and the trainer's regime) it shows (-3.05%). The unchanged backward is +0.46% in the
  same race: the faster forward draws more power, the shared clock drops a little (the mirror of LHOd's -0.33%; both t ~3,
  consistent with 45.50's power model); dense forward + backward together about -0.1%.
- **Packed +0.45%, cause proven (PC sampling, packed kernel):** exps -32k cycle-equivalents per SM, but integer / predicate
  +28k: in SM3 the diagonal (masked) tiles build the causal mask as a BITMASK (`@P6 LOP3.LUT R2, R2, 0x2000, ...`, 64 per
  masked step, executed 33,912 times): our mask loop kept the mixed-row order (`reg % 4`) while the max now goes row by row,
  and ptxas parked the 64 conditions in a register. Packed has a diagonal tile in every (short) unit, dense rarely.
**SM3m** (`cands/fwd_SM3m.cu` 86a2d73d = SM3 + FA3's mask order, hopper/mask.h causal: row by row, each element's column offset
(a constant) against ONE per-row limit; same condition): local SASS: mask-bitmask LOP3 **64 -> 0**; packed kernel **3,432**
instructions (LHOd 3,504, SM3 3,688), dense 3,672 (LHOd 3,736); steady-loop exp spacing unchanged (0 / 13 / 22 / 7 / 23); 168
registers, 0 spills. **Run** `scripts/run_sm3m_1019.sh` (00d63d34, ~9 min): build, quick suite on SM3m, forward + backward race
LHOd / SM3 / SM3m (8 rounds, rotating), ncu of SM3 and SM3m.
**SM3m RESULT (sm3m_1019, 16:04-16:10 IST, GPU 0; GPU 1 had a colleague's process, 656 MiB, separate GPU and power limit).**
Built SM3m (86a2d73d -> library b9815e38; F3c back, 096123e0). Quick suite **296 / 296** (packed == alone bit-exact incl.).
Forward + backward race, 8 rounds, LHOd / SM3 / SM3m in rotating order:
| vs LHOd | dense fwd | packed fwd | dense bwd (code unchanged) | packed bwd (code unchanged) |
|---|---|---|---|---|
| SM3 | -3.30% (t -25.1, 8/8) | +0.27% (t 2.7, 1/8) | +0.27% (t 2.1) | -0.02% |
| **SM3m** | **-3.51%** (t -31.3, 8/8) | **-4.48%** (t -60.0, 8/8) | +0.56% (t 5.4, 0/8) | +0.17% (t 3.1) |
Medians (ms): dense fwd LHOd 1.0487 / SM3 1.0123 / SM3m 1.0110, bwd 3.1371 / 3.1467 / 3.1592; packed fwd 0.4395 / 0.4405 / **0.4198**,
bwd 1.5016 / 1.5016 / 1.5036. Forward + backward per iteration, SM3m vs LHOd: dense 4.1858 -> 4.1702 ms (**-0.37%**), packed 1.9411
-> 1.9234 ms (**-0.91%**). The unchanged backward is slower again by the shared clock (power cap, as for SM3).
ncu (same box): dense cycles -4.02% vs LHOd (instructions -1.07%), packed **-6.28%** (instructions -2.09%); the mask bitmask:
SM3 64 LOP3 executed 2,170,368 times, SM3m 0. vs FA3 (ncu cycles): dense 1.107x -> **1.063x**, packed 0.927x -> **0.869x**.
Chained to the 45.43 scoreboard (forward, fwd + bwd regime): dense ~1.084 x 0.9966 x 0.9649 = **~1.042x FA3**, packed ~0.936 x 0.955 =
**~0.894x FA3** (estimates until a new scoreboard measures them directly).
**What is left vs FA3 (SM3m's PC sampling, `loop_diff_fa3.py` with SM3m; SM3m's period estimated 2,449 = 2,551 x 0.960):** exps
+1 and row max +2 cycles per step (= FA3 now); remaining: uniform datapath +75 (65 vs 46 instructions per step: descriptors
built per HGMMA, the stage through R2UR), integer +34, branch / control +109 and named barrier +21 (mostly the deferred baton
wait of group 1), FMUL / FFMA / FADD / F2FP +71 (dispatch stalls); offset by less waiting on the tensor pipe (P.V HGMMA -146,
QK wait -66).

### 45.53 2026-10-19 16:45 IST: after SM3m the loop step equals FA3's; the rest is at UNIT BOUNDARIES (the first K / V tile); PF

**Note (user's question):** the backward code is byte-identical in all these libraries (only GQA_fwd_sm90.cu differs); its
+0.2-0.6% in the forward + backward races is the shared clock at the 700 W cap. A backward-only benchmark shows no change.
**1. Where the remaining gap is: the whole kernel, not just the loop** (ncu PC sampling, same box; consumer warps only, the
producer warp's samples left out; `deep/kernel_map.py` f581bc56 = blocks by executed count, `deep/unit_budget.py` da64b86a):
| per unit / per plain step (ncu clock) | FA3 | LHOd | SM3m |
|---|---|---|---|
| plain loop step | 2,299 | 2,402 | **2,282** |
| unit start (Q wait etc.) | 619 | 74 | 73 |
| first tile (QK + softmax; FA3: the diagonal block, masked) | 2,449 | 4,874 | **4,843** (tile 0, plain) |
| diagonal step (ours runs it LAST, separately) | (in its first tile) | 2,985 | 2,879 |
| last P.V + epilogue | 2,059 | 2,308 | 2,368 |
Units per SM 34.91; plain steps per consumer warp per SM FA3 541.1, ours 507.3 (FA3 has one more plain step per unit: its
diagonal is the first tile). SM3m's plain step = LHOd's - 120 = exactly the exp-wait fix (367 -> 245), and = FA3's (-17).
Per unit: FA3 ~40,760 cycles, SM3m ~43,320 (+2,560 = the +6.3% of ncu cycles): **the remaining dense gap is at the unit
boundaries**, mainly ours' first tile + diagonal step (7,722) vs FA3's first tile + one plain step (4,748): +2,974; unit start
-546; epilogue +309.
**2. What: one wait** (`deep/boundary_reasons.py` a3f510ae): SM3m's unit start + first tile has 1,961 cycles per unit of
long-scoreboard stalls (FA3 155), on ONE site: `SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R25+URZ+0x38000]` + its retry branch =
the wait for tile 0's K / V (full[stage]): **1,498 cycles per unit** (FA3's wait for its first K: 50; Q's wait: ~3 in both).
= ~50k cycles per SM = ~57% of the remaining gap.
**3. Why (source):** FA3's producer for a new tile (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp): an L2 prefetch of Q (:668),
`load_K` of the first block as soon as a K slot is free (:769), THEN the QueryEmpty barrier (consumers done with the old Q,
:776), then Q (:780). Ours: `if (j > 0) { mbar_wait(&qempty[0]...); mbar_wait(&qempty[1]...); }` -> Q -> K / V(0): the first
tile's 64 KB were requested only after both groups' last QK of the previous unit (qempty is arrived after the last loop
step), i.e. one unit-tail + epilogue (~2,400 cycles) before they are needed, too late for a 64 KB DRAM fetch while all SMs
fetch (Q, 16 KB, arrives in time).
**PF** (`cands/fwd_PF.cu` 1888246a = SM3m + switch `KV0_FIRST`): the producer loads the unit's first K / V tile (waiting for
its slot) BEFORE the qempty waits, then Q, then the other tiles = FA3's order; nothing else. Proofs before any GPU run:
(a) protocol model `deep/protocol_sim_pf.py` (9890281f): PF clean on 3,000 dense + 3,000 packed random unit lists (no deadlock,
every K / V / O-staging read finds its tile); its twin PF_noempty (tile 0 loaded without the slot wait) is CAUGHT in ~89% of
runs; (b) local SASS: the producer's first events SM3m `wait qempty x2, TMA x4 (Q), wait slot, TMA x4 (K / V 0)` -> PF
`wait slot, TMA x4 (K / V 0), wait qempty x2, TMA x4 (Q)`; consumer code: packed identical to SM3m, dense 3 setup
instructions reordered, the steady loop identical (exp spacing 0 / 13 / 22 / 7 / 23); 168 registers, 0 spills. Expected: up to
~1,450 cycles per unit (~3% of dense in ncu; packed has more, shorter units). Run `scripts/run_pf_1019.sh` (eb7303ae, ~7 min):
build, quick suite, forward + backward race vs SM3m, ncu of PF.
**Rest of the boundary gap (next, same method):** ours runs the diagonal block as a separate LAST step (2,879) while FA3
makes it the first tile (no extra step); epilogue +309.
**PF RESULT (pf_1019, 16:39-16:44 IST, GPU 0).** Built PF (1888246a -> library 6653f9f2; F3c back, b33a8a53). Quick suite **296 / 296**.
Forward + backward race vs SM3m, 8 rounds, A B / B A: **dense forward -2.60%** (t -50.1, faster 8/8), **packed forward -3.01%**
(t -34.5, 8/8); backward (code unchanged) +0.19% (t 1.5) / +0.16% (t 2.2). Medians (ms): dense fwd SM3m 1.0060 / PF 0.9809, bwd
3.1353 / 3.1461; packed fwd 0.4194 / 0.4069, bwd 1.5038 / 1.5058. **PF is the new best forward.** Chained vs FA3 (forward):
dense ~1.040 x 0.974 = **~1.013x**, packed ~0.894 x 0.970 = **~0.867x** -> a direct same-box scoreboard on GPU 1 next (sb2_1019).

### 45.54 2026-10-19 16:50 IST: same-box scoreboard with PF (GPU 1) -- dense forward 1.025x FA3 (was 1.084x), everything else won

**Run sb2_1019** (`scripts/scoreboard_free.sh 1 sb2_1019 vj2_PF vj2_PF`, GPU 1, 16:45-16:50 IST, free clocks, nsys kernel times, one
contender per process, forward + backward back to back, ABBA order; all 16 runs exit 0; power 691-695 W = the cap in every
run). Steady state (calls >= 2 s after the first forward, `scripts/scoreboard_steady.py`), ratio = ours (PF) / rival:
| | vs FA3 | vs FA4 | vs cuDNN |
|---|---|---|---|
| dense forward | **1.025** (10-19 morning, K5q: 1.084) | 0.928 | 0.872 |
| dense backward call | 0.973 | 0.987 | 0.888 |
| dense forward + backward | **0.985** | 0.972 | 0.884 |
| packed forward | **0.866** (morning 0.936) | 0.778 | 0.823 |
| packed backward call | 0.955 | 0.971 | 0.898 |
| packed forward + backward | **0.935** | 0.923 | 0.881 |
Kernel times (us, steady, r1 / r2): dense fwd ours 971.4 / 978.2, FA3 952.4 / 950.5, FA4 1051.0 / 1050.3, cuDNN 1113.2 / 1122.8;
packed fwd ours 400.8 / 401.0, FA3 463.3 / 462.2. (This ran on GPU 1, the morning's on GPU 0: FA3's own dense forward is 2.3%
faster here, so compare ratios within a run.) The chained estimate (1.013) was a little optimistic; 1.025 is the measured one.
**PF's ncu (same box):** the wait for tile 0's K / V **1,498 -> 48 cycles per unit** (FA3 50); dense cycles 1,449,041 (SM3m
1,512,445, FA3 1,423,158: **1.018x**). Per unit (`deep/unit_budget.py` with PF): plain step 2,273 (FA3 2,299); unit start +
first tile 3,134 (FA3 3,068, its first tile = the masked diagonal); our diagonal step 2,908 = a plain step + 635; last P.V +
epilogue 2,442 (FA3 2,059). The remaining ~750 cycles per unit (1.8%) = our separate diagonal-last step and the epilogue.
**Today's chain (dense / packed forward vs FA3):** K5q 1.084 / 0.936 -> LHOd (L2 hints) -> SM3 (FA3 softmax structure) -> SM3m
(+ FA3 mask order) -> PF (+ FA3 producer order) 1.025 / 0.866; each step proven first (source + SASS + PC sampling), then raced.

### 45.55 2026-10-19 17:20 IST: the remaining boundary gap item by item -- the diagonal step (no proven win yet) and the epilogue (PFT)

**Diagonal step** (`deep/step_vs_diag.py`, PF dense ncu, per execution): our masked last step costs 2,565 + 343 (its start: stage,
group 1's baton) = 2,908 per unit vs a plain step's 2,273 (+635). Of that, the mask itself (64 ISETP + 64 FSEL) = ~347; the rest
is waiting moved around (baton +265 at the QK wait, tensor back-pressure +163 on P.V, less elsewhere). FA3 does the same mask
work (68 select / compare + integer) in its first tile (the diagonal block), and its start + first tile (3,068) is about our
start + unmasked first tile (3,134, `deep/boundary_jobs.py`). Moving our diagonal block first (FA3 walks the keys backwards)
would only save what the mask costs in the pipelined last step minus its cost in the prologue: NOT proven to be positive; a
large change (reverse key order in producer and consumers, K4's O staging, numerics order) -> not proposed now.
**Epilogue** (`deep/boundary_jobs.py` 05d275ab; `deep/step_vs_diag.py` 693c6cc0 for the diagonal step above; per unit): ours 2,442 vs FA3 2,059 (+383): memory 28 vs 11 instructions
(+239 cycles: 8 x 16-byte shared load + 8 x 16-byte global store per lane, short-scoreboard waits), FENCE.VIEW.ASYNC 2 vs 1
(+125: long-scoreboard waits), the group bar.sync (+49), integer / branch +396; offset by less waiting at the last P.V wait (-438)
and uniform (-270). FA3 (hopper/epilogue_fwd.hpp :250-264, :310-326, dense = Use_TMA_O): every thread stmatrix -> O in smem,
fence_view_async_shared, NamedBarrier::ARRIVE (non-blocking); only the last warp syncs and one elected lane issues the TMA store,
tma_store_wait<0> (smem read done), then barrier_O.arrive (smem free). Varlen: no TMA store (other sequences' rows).
**PFT** (`cands/fwd_PFT.cu` 541a4f5e = PF + switch `TMA_O`, dense only): stmatrix in the TMA SW128 layout (two 64-column halves
of 8 KB, as Qt), fence.proxy.async, warps 1-3 bar.arrive and go on, warp 0 (index broadcast from lane 0, as FA3's warp_idx_sync)
bar.sync, lane 0: 2 x cp.async.bulk.tensor store (O's map = make_tma_q64(O): box 64 x 64, SW128), commit + wait_group.read 0,
then the K slot's empty arrive; packed keeps the per-lane stores. First build used `if (warp == 0)` directly: ptxas compiled the
whole kernel as possibly divergent (WARPSYNC.COLLECTIVE fallbacks, +200 instructions, the loop rescheduled) -> fixed by the
broadcast. Local SASS (final): dense 2 UTMASTG, 0 LDS.128 / STG.128 for O (PF 8 + 8 per lane), 3,720 instructions (PF 3,744), no
collective fallbacks, steady-loop exp spacing 1 / 13 / 18 / 10 / 23 (PF 0 / 13 / 22 / 7 / 23: by the per-spacing costs about
-17 cycles, neutral); packed kernel identical to PF except the parameter offsets (0 of 3,504 lines); TMA_O 0 (fwd_PFToff.cu) =
PF's SASS exactly; 168 registers, 0 spills. Expected: up to ~380 cycles per unit (~0.9% dense, ncu); packed unchanged.
Run `scripts/run_pft_1019.sh` (6ce228b1, GPU 0, ~10 min): build PFT (F3c back in the tree), the same SASS check on the box
build (CUDA 13.4; stops unless dense UTMASTG 2 / STG.E.128 0 / COLLECTIVE 0 and packed PFT == PF with the parameter offsets
ignored; dry run on the local cubins: passes, and fails on PF vs SM3m packed with 1,120 lines, so the check is not vacuous),
quick suite, forward + backward race vs PF (8 rounds, A B / B A), ncu of PFT (dense + packed).

### 45.56 2026-10-19 17:15-17:45 IST: PFT RESULT -- dense forward SLOWER (+1.60%); why, from ncu + FA3's source; box backup

**Run** (`box_runs/pft_1019`, GPU 0, 17:08-17:12 IST; run_pft_1019.sh 6ce228b1): build PFT b29aa706 (F3c back in the tree); box SASS
check passed (box CUDA 13.4: dense PF 3,720 instructions, PFT 3,696, UTMASTG 2, STG.E.128 0, LDS.128 0, COLLECTIVE 0; packed PFT ==
PF, 0 of 3,480 lines); quick suite 296/296; race forward + backward vs PF, 8 rounds A B / B A (`session_2026_10_17/deep/race_stats.py`):

| PFT vs PF | % change | t | rounds faster |
|---|---|---|---|
| dense forward | **+1.60%** (0.9819 -> 0.9970 ms) | +25.3 | 0 / 8 |
| dense backward | -0.25% (unchanged code: shared clock) | -2.7 | 8 / 8 |
| packed forward | -0.02% (identical kernel) | -0.2 | 3 / 8 |
| packed backward | -0.04% | -0.6 | 4 / 8 |

ncu (same box; PF from pf_1019, PFT from pft_1019): dense 1,449,041 -> 1,473,819 cycles per SM (+1.71%; per unit 41,509 -> 42,219,
+710), packed -0.28% (identical kernel). Instructions +0.03%.
**Where** (`deep/kernel_map.py` regions, paired in order; `deep/align_pf_pft.py`, `deep/regions_pf_pft.py`; per unit = cycles per
SM / 34.9):

| region | PF | PFT | change |
|---|---|---|---|
| epilogue (stmatrix .. release; PFT incl. the store warp's path + LSE) | 1,363 | 879 | **-484** (the intended gain, > the 383 expected) |
| next unit's start + tile 0 (part 1) | 1,915 | 2,583 | **+668** |
| steady loop (481 instructions, 14.53 steps per unit) | 30,286 | 30,875 | **+590** (+40.5 per step) |
| diagonal step / loop exit / last P.V / rest | | | about -64 |

1. **Unit start +668: the store warp's wait.** PFT SASS: lane 0 of warp 0 (each group): 2 x UTMASTG.2D, UTMACMDFLUSH, DEPBAR.LE
SB0 0x0 (= cp.async.bulk.wait_group.read 0), then SYNCS.ARRIVE (empty[s_last]): 140 cycles per unit of samples stall on that
arrive (long scoreboard = the DEPBAR), i.e. about 630 cycles of real waiting per store warp per unit (2 store warps of ~9 sampled
warps). The next unit's first QK (HGMMA) needs all 4 warps of the group, and the baton barriers (bar 1 / 2, 256 threads) couple
both groups, so the late warp holds back both: the wait for Q (mbarrier 0x38030) 286 -> 610 (+324, part of it is warps 1-3
arriving earlier because they skip the old read-back) and the baton sync before the first QK 15 -> 248 (+233).
2. **Steady loop +590: the compiler re-ordered the loop** (same 481 instructions, different order; box SASS). Per step: dispatch
+61, wait +40, mio +36 (exps bunched), not_selected +15, baton -94. The laptop check (45.55) saw the changed exp spacing
(PF 0/13/22/7/23 -> PFT 1/13/18/10/23) and estimated "-17 cycles, neutral": WRONG. Lesson: a candidate that does not target the
loop must leave the loop's SASS IDENTICAL to the base's (0 differing lines), checked on the laptop and again on the box build
before any GPU time; an estimate from the exp spacing is not a proof.
**FA3's source** (hopper/epilogue_fwd.hpp, 94e22c9): :249 named_barrier_sync(NumEpilogueThreads) (all 8 MMA warps), :251-262 every
thread stmatrix -> fence_view_async_shared -> NamedBarrier::arrive(NumEpilogueThreads + 32) (non-blocking), :277-301 LSE stores,
:317-327 ONLY the last warp (warp_idx_sync == NumEpilogueThreads / 32 - 1 = the last warp of the SECOND MMA warpgroup) syncs on
that barrier (288 = 256 arrivals + its own 32), one elected lane stores the whole 128-row O tile (32 KB), tma_store_wait<0>, then
barrier_O.arrive; the producer waits barrier_O only before the next tile's V (mainloop :865; O aliases V,
flash_fwd_kernel_sm90.h :90-101). So FA3 has ONE waiting warp, in the warpgroup that goes second; PFT has one in EACH group, and
group 0 (first at the next unit: baton bar 1, SASS `@!P2 BAR.SYNC 0x1, 0x100`) waits for its own warp 0.
**Next candidate (PFT2, proposed):** FA3's exact arrangement: both groups stmatrix + fence + bar.arrive on one barrier of 288; only
group 1's last warp syncs, one lane stores all 4 boxes (both groups' 64 rows), waits for the read, arrives empty[s_last] twice
(count 2, one per group as before); + the hard gate above (steady loop SASS == PF's on the laptop and on the box build) + the
protocol model for the new barrier.
**Box backup** (instance may close, credits ending; 17:20-17:40 IST): every run folder on the box is now byte-identical on the
laptop (rsync --checksum dry run: 0 differences): new pft_1019 (46 files), the 16 scoreboard nsys reports of sb2_1019 (24 MB, left
out before), and trace_1017_vj2_TLK5q of 10-19 11:48 IST (a re-run on this box under the 10-17 name, library 97765067; the laptop
had only the 10-17 run in box_2x_final_2026_10_17/) -> session_2026_10_19/box_runs/. `profiling_reports/box_backup_2026_10_19/`
(6.3 MB): logs/ (20 build / setup logs incl. benv_pip_freeze.txt), step6/, loss_bench/, box_scripts_snapshot/ (96 files; all equal
to the laptop's except __pycache__ and the fa3_tl patch, which is in session_2026_10_19/fa3_tl/, same md5s). Outside git:
~/Downloads/box_backup_2026_10_19_bin/flash_attn_3/ (the scoreboard's FA3 build, _C.abi3.so 67 MB: skips the long FA3 build on a
new box with the same benv). Checked equal, not copied: the code tree (sft_v2/BluTrain_sft, checksums), sft_data (md5s = the
laptop's ~/Downloads/sft_data). Not copied (rebuildable): libs/ (18 candidate builds x 55 MB: build_cands.sh from the local
cands/), benv (5.9 GB; benv_setup.sh + the pip freeze), flash-attention source (94e22c9) and fa3_tl2/3 builds, pydev, *.tgz
(= the run folders).

### 45.57 2026-10-19 18:40-19:30 IST: the remaining dense gap, exact -- one consistent accounting, every wait labelled, FA3's source line for each cause

**Why again:** the user (after PFT failed): one last chance, find the EXACT culprit with proofs, no assumptions. All from data already
on the laptop: the same-box ncu source pages (FA3 `box_runs/ncu_fa3_k5q_1019`, PF `box_runs/pf_1019`), the scoreboard clock logs
(`box_runs/sb2_1019/*.clocks.csv`), FA3's source (94e22c9). Scripts (deep/): waits_fa3_pf.py 19288f63, regions_fa3_pf.py bd944979, align_pf_pft.py b2074201, regions_pf_pft.py d56c374e.

**Stats (scoreboard sb2_1019, ours = PF):** vs FA3 dense fwd **1.025** (974.8 vs 951.5 us; the ONLY loss), bwd 0.973, f+b 0.985;
packed fwd 0.866, bwd 0.955, f+b 0.935. vs FA4 and cuDNN every entry < 1. Best forward = PF (chain F3c -> K5q -> LHOd -> SM3m -> PF);
PFT rejected (45.56).
**Time vs cycles vs clock:** ncu cycles PF / FA3 = 1,449,041 / 1,423,158 = **1.018**; scoreboard SM clock (nvidia-smi, busy samples
from 2 s on, both runs): ours 1,646.9 vs FA3 1,652.3 MHz = **-0.33%** (run-to-run: ours 1,664 / 1,630, FA3 1,663 / 1,642: the noise
is larger than the difference). So the dense forward gap is CYCLES (1.8%) + ~0.3% shared clock + noise. (Forward alone FA3 runs at
a LOWER clock than ours, 45.50.)

**One accounting for both kernels** (`regions_fa3_pf.py`: the same cut points in a unit's life, cycles per unit = share of ALL
samples x SM cycles / 34.9 units per SM):

| region | FA3 | PF | difference |
|---|---|---|---|
| unit start (after the previous epilogue .. first K landed) | 503 | 869 | **+366** |
| first tile (FA3: the diagonal, MASKED; PF: tile 0, unmasked) | 2,226 | 1,941 | -285 |
| steady loop (FA3 15.5 steps, PF 14.53) | 32,397 (2,090 / step) | 30,286 (2,084 / step) | -2,111 |
| diagonal (PF only: loop exit + the masked last step) | 0 | 2,667 | +2,667 |
| end (last P.V + epilogue + FA3's TMA-store warp) | 2,207 | 2,240 | +32 |
| producer warp + prologue | 3,434 | 3,506 | +72 |
| total | 40,768 | 41,509 | +741 (1.8%) |

**CORRECTION of 45.55:** "epilogue +383" was a measuring error: FA3's region was cut at instruction 2302 and left out FA3's own
TMA-store warp (2302-2323, 208 cycles / unit) and its exit code, and the two kernels were normalised differently. With the same
cut points the end of a unit costs the same (+32). PFT (45.56) targeted a non-problem.
**Cause 1 -- the unit start, +366 (49% of the cycle gap)** (`waits_fa3_pf.py`, each mbarrier wait followed to its retry branch):
- **Q comes from DRAM.** Consumers' wait for Q: PF 286 cycles / unit (BRA after `TRYWAIT [0x38030]` = qfull), FA3 103 (barrier_Q,
  0x28800). Both release the old Q at the same point (FA3 mainloop_fwd_sm90_tma_gmma_ws.hpp:1237 NamedBarrier::arrive QueryEmpty
  after the loop; ours fwd_PF.cu:1072 qempty after the last QK); both producers are already WAITING for that release (PF 172 / unit
  on qempty 0x38040/48, FA3 117 on QueryEmpty), so both issue the Q load at the same moment. The one difference: **FA3 prefetches
  the tile's Q into L2 when its producer STARTS the tile** (:681 `prefetch(params.tma_load_Q, tQgQ)`, before load_K :769 and the
  QueryEmpty sync :776), so its Q load hits L2; ours reads DRAM.
- **7 runtime integer divisions per unit in the consumer** (decode_u, fwd_PF.cu:529-535, called :704 just before the Q wait :716):
  two I2F.RP / MUFU.RCP / ... sequences in the SASS, ~190 cycles / unit. FA3 decodes with precomputed constants (ULDC
  c[0x0][0xdd0] ...: FastDivmod), no division. Today this is hidden inside the Q wait; it shows as soon as Q arrives earlier.
**Cause 2 -- the diagonal tile LAST, +271 net (37%):** PF's masked diagonal step costs 2,667 = a plain step (2,084) + 583; FA3
processes the diagonal FIRST (mainloop :1150 mask.apply on the first tile, then fwd_step from n_block_max - 2 down, causal 128 x 128:
no other masked tile), where the same mask costs only +285 (2,226 vs our unmasked 1,941): there the mask overlaps the other
group's first QK instead of sitting on the pipelined step's critical path. Net (583 - 285) - our faster loop = +271.
**Not causes:** the steady loop (2,084 vs 2,090 per step), the end of unit (+32), the producer (+68; it mostly waits for free slots).
Instructions: PF +4.1% (69,275 vs 66,521 warp instructions per unit), almost all uniform (ULEA +1,342, UMOV +884), inside the loop,
which is still as fast as FA3's.
**Expected value of each fix (cycles per unit, of 41,509):** Q L2 prefetch -183 .. -286 (0.45-0.7%); division-free consumer decode
-47 .. -190 (only together with the prefetch); diagonal first ~-298 (0.7%). Together ~-530 .. -770 (1.3-1.9%): about parity.

### 45.58 2026-10-19 19:30-20:45 IST: the last forward candidates (QP / QPU / QPD / QPUD), laptop proofs, the loop gate; node setup

**Node (8x H200, innmi1srh2-p032, GPU 4):** one folder /localstorage/gautam_k (~/gautam_k is a link), tree = the laptop master
(rsync checksum: 0 files differ), build OK (PF 1888246a -> lib e9248d27, PFT 541a4f5e -> f071bd1d, F3c c6a67163 -> 42641cb5,
tva_a b7c752b3 -> 74f3cdc3). **PF's SASS built here == Verda's** (dense 3,720 / packed 3,480 instructions, 0 differing lines,
box_runs/node_setup_1019/ vs box_runs/pft_1019/): the node's CUDA 13.4 (compiler 38657139, Verda 38855100) compiles our kernels
identically, so every Verda measurement holds here.
**Candidates** (`patches/make_pfq.py`: anchored edits of PF, each anchor exactly once; files: patches/make_pfq.py fbc5f7d8, deep/loop_sass.py a99f7937, deep/loop_order_diff.py 5f6a25bd, scripts/run_qp_1019.sh e039121d, scripts/cands/fwd_QP.cu f345c965, scripts/cands/fwd_QPU.cu 7e4197ea, scripts/cands/fwd_QPD.cu 34640bc5, scripts/cands/fwd_QPUD.cu 1899e6a7, scripts/cands/fwd_QPoff.cu 9621c861). Three switches, each
one proven cause of 45.57:
- **Q_PREF** (FA3 mainloop :681): `cp.async.bulk.prefetch.tensor.2d.L2.global` (CUTLASS SM90_TMA_LOAD_2D::PREFETCH) of the pair's
  2 x 2 Q boxes when the producer starts the pair, before its first K / V and before qempty; both kernels (packed waits on Q too:
  1.27% of its samples, dense 0.80%).
- **DEC_PUB**: the producer publishes (b, hq, pair) of each dense unit in a 3-int slot next to sched_unit (same mbarrier); the
  consumers read their unit's slot (ju & 1) at its top (safe: the producer rewrites slot k & 1 only after both groups' last QK of
  unit k-2, the G5 rule), no decode_u. First version (A) carried next_* registers through the epilogue and moved the dense loop
  (49 changed blocks after register renaming); version B (read at the top) is the one built.
- **DIAG_FIRST** (FA3 :1150): the producer's n-th tile is tile nkv - 1 - n (the stage / slot / mbarrier sequence is unchanged), the
  prologue masks with nt = my_nkv - 1, no later tile is masked, the last step keeps the K4 count. Both kernels (packed == alone
  stays bit-exact); the online-softmax order changes (results not bit-identical to PF's).
**Laptop proofs** (nvcc 13.0, PF's flags; `deep/loop_sass.py` finds the steady loop = the smallest backward-branch range with 16
HGMMA and >= 64 exps; `deep/loop_order_diff.py` renames registers canonically): all 168 registers, 0 spills. **QPoff (all off) ==
PF**: 0 differing lines of 3,744 / 3,504. **QP: dense AND packed steady loops identical to PF's (even the register numbers)**,
4 UTMAPF (producer), +16 instructions. QPU: the consumers' division sequences gone (I2F.RP before the loop 3 -> 1; the producer
+6 for publishing), packed loop identical after renaming. QPD: the mask moved (FSEL after the loop 68 -> 4, the prologue's 64
kept). How far each moves the steady loop (instructions out of PF's order, opcode alignment after renaming) vs **PFT (the known
+40 cycles / step, 45.56): 197 dense**:

| | dense | packed |
|---|---|---|
| QP | 0 | 0 |
| QPU | 143 | 0 |
| QPD | 57 | 23 |
| QPUD | 132 | 23 |

A reorder cannot be priced on the laptop (the exp-spacing estimate failed for PFT), so each fix is RACED on its own: QP (fix 1
alone, loop-clean), QPU (+ fix 2), QPD (+ fix 3), QPUD (all), one session vs PF, with ncu of all five (PF too, same day).
**Run** `scripts/run_qp_1019.sh` (GPU 4, ~30 min: builds ~8 min CPU, then ~20 min GPU): build, box SASS gate (stops unless QP's
two loops equal PF's and all 8 candidate kernels hold the 4 prefetches; dry run on the laptop cubins: passes; with QPD's cubin
in QP's place it fails, so the gate is not vacuous), quick suite per candidate, forward + backward race (8 rounds, rotating),
ncu. **Expected (45.57):** QP dense -183 .. -286 cycles per unit (0.45-0.7%); QPD about -300 more if its 57-instruction reorder
is free; QPU -47 .. -190 more once Q is early. Together about parity with FA3 (dense fwd 1.025 today).
**Split (user, 20:50 IST: CPU work first, the GPU only when needed, colleagues share the node):** `scripts/build_qp_1019.sh`
(11031e0e, CPU only, ~8 min: build + the SASS gate + the libraries' md5s in qp_1019/gate_libs.md5, "GATE PASSED") and
`scripts/run_qp_1019.sh` (5e68bdaf, GPU, ~20 min: refuses without "GATE PASSED" and if any library's md5 changed since the
gate; then quick suites, race, ncu incl. PF). Dry-tested on the laptop cubins in a fake ~/gautam_k: gate passes (2 of 2, 8 of
8), the GPU script refuses a library changed after the gate, starts with the gated ones. The dry run found that cuobjdump needs
a writable TMPDIR (else it prints only the header and the gate fails closed); the build script now creates ~/gautam_k/tmp.
`scripts/loop_sass.py` (a99f7937) = deep/loop_sass.py (reads only the two forward kernels with -fun; names libraries by folder).

### 45.59 2026-10-19 19:54-20:42 IST: RESULT of the last candidates (8x node, GPU 4) -- QPU is the new best dense forward

**Run** (`box_runs/qp_1019`, pulled 127 files): build_qp_1019.sh 19:53-19:54 (gate passed on the node build: QP loops == PF's,
4 prefetches in all 8 kernels; libs PF e9248d27, QP 46fb93a1, QPU 4956db18, QPD b4e82daf, QPUD 9e49098b), run_qp_1019.sh:
quick suites **296/296 for all four** (QPD / QPUD with the diagonal first included; each suite took 224-294 s here vs ~50 s on
Verda: the CPU references on the node's shared CPUs), race 8 rounds x 5 libraries (80 runs, 20:15-20:39), ncu of all five incl.
PF. GPU 4 held only our processes (one watcher alarm 20:16 was a pid that had already exited: ours; the watcher then alerted only
on a non-ours process seen twice; none).
**Race** (`session_2026_10_17/deep/race_stats.py`, paired per round vs PF, forward + backward back to back; median ms PF: dense
fwd 0.9918, bwd 3.1817, packed fwd 0.4179, bwd 1.5341):

| candidate | dense fwd | packed fwd | dense bwd | packed bwd |
|---|---|---|---|---|
| QP (Q L2 prefetch) | +0.08% (t +0.8, 2/8) | -0.24% (t -2.6) | +0.01% | -0.16% |
| **QPU** (+ unit decoded by the producer) | **-0.81% (t -3.8, 7/8; p10 -1.27%, t -10.4)** | -0.13% (t -1.3) | +0.24% (t +1.9) | -0.05% |
| QPD (+ diagonal first) | +0.32% (t +3.4, 1/8) | -0.43% (t -4.2, 8/8) | -0.07% | -0.14% |
| QPUD (all three) | -0.31% (t -4.6, 7/8) | -0.44% (t -6.2, 8/8) | +0.01% | -0.16% |

Forward + backward totals (medians): dense PF 4.1735 ms, QPU 4.1650 (-0.20%), QPUD 4.1694 (-0.10%); packed PF 1.9520, QPU 1.9511
(-0.05%), QPUD 1.9463 (-0.29%). The backward code is unchanged in all (its shifts = the shared clock).
**ncu, same session** (`deep/qp_ncu.py`; dense per unit, regions: consumer code before the steady loop, the loop, after it, the
producer warp; Q wait = qfull's retry branch). PF here 1,449,507 cycles/SM = Verda's 1,449,041 (same SASS, same cycles):

| dense | kernel cycles | before loop | loop per step | after loop | Q wait |
|---|---|---|---|---|---|
| PF | 1,449,507 | 2,770 | 2,084.2 | 4,951 | 275 |
| QP | -0.17% | 2,661 | 2,084.0 (loop identical) | 4,988 | 171 |
| QPU | **-1.51%** | 2,548 | **2,065.4** (reordered loop, 19 faster) | 4,854 | 308 |
| QPD | -0.12% | 2,782 | 2,103.0 (reordered loop, 19 slower) | 4,618 | 167 |
| QPUD | -0.51% | 2,614 | 2,097.8 | 4,703 | 327 |

Packed (cycles/SM): PF 626,033; QP -0.35%, QPU -0.17%, QPD -0.20%, QPUD -0.23% (Q wait 7,287 -> 6,091 QP).
**Reading, exactly as 45.57 predicted for the start:** the prefetch makes Q land early (wait 275 -> 171 per unit, FA3 103-144),
but alone it gains nothing in time because the 7 divisions were then on the path; with them removed (QPU) the unit start is 222
cycles cheaper (the Q wait rises to 308 because the consumers arrive earlier; that is moved time). QPU's compiler-reordered loop
turned out FASTER (2,065 vs 2,084 per step: -273 per unit). **The diagonal-first fix works where it acts** (after the loop
-333 per unit: the masked last step is gone) **but its masked first tile costs +121 and its reordered loop +19 per step (+275 per
unit)** -> no net gain in dense; packed gains a little (race -0.43%). Cycles convert to time only about half here (QPU: -1.51%
cycles, -0.81% time; effective clock from cycles / race time ~0.5% lower than PF's): under the 700 W cap a faster kernel draws
more power per unit time. **QPU vs FA3 in cycles: 1,427,607 vs 1,423,158 = +0.31% (PF was +1.85%)**; in time, estimated from the
race, dense forward ~1.017x FA3 (PF 1.025x): to be measured by a scoreboard on this node. What is left in cycles: our masked
diagonal step (+583 over a plain step) against our now-faster loop (-19 per step vs PF, -25 vs FA3's 2,090) and start.

### 45.60 2026-10-19 20:49-21:11 IST: scoreboard on the 8x node with ours = QPU (`box_runs/sb_qpu_1019`)

`scripts/scoreboard_free.sh` (8cd5ecb0: GK = ~/gautam_k, else = the Verda script) on GPU 4: 16 runs (2 shapes x 2 rounds, ABBA x 4
contenders), all exit=0; ours = vj2_QPU (forward QPU, backward Vj2 unchanged); rivals from ~/gautam_k/benv (torch 2.14.1+cu130,
cuDNN 9.24, FA3 / FA4 from step6/fa_src 94e22c9; step6 bench files = the laptop's, md5 4d667815 / 93367592). Each run ~80 s, of
which ~4.5 s on the GPU (caught with nvidia-smi -lms 500: 100%, 692 W). Summary `scoreboard_steady.py` (calls >= 2 s after the
first forward):

| ours / rival | dense fwd | dense bwd | dense f+b | packed fwd | packed bwd | packed f+b |
|---|---|---|---|---|---|---|
| FA3 | **1.004** (984.2 vs 980.4 us; r1 +0.44%, r2 +0.34%) | 0.968 | 0.976 | 0.865 | 0.953 | 0.933 |
| FA4 | 0.919 | 0.989 | 0.972 | 0.781 | 0.971 | 0.923 |
| cuDNN | 0.863 | 0.891 | 0.884 | 0.827 | 0.899 | 0.883 |

**Same 11 of 12 wins; the dense forward vs FA3 is now 1.004 (Verda, PF: 1.025).** Split of that move: QPU's own gain is -0.81%
(same-node race, 45.59); the rest is the box: this node's GPU 4 runs every kernel at a lower clock at the same ~690 W (dense: ours
-2.2%, FA3 -2.3%, FA4 -0.8%, cuDNN -0.9%; packed -3.1 .. -3.6%), and FA3's dense forward lost more (+3.04% time vs Verda) than ours
would have (PF-equivalent ~+1.8%), so PF here would be ~1.012 (984.2 x 1.0082 = 992 us vs 980.4). FA3's build here is its own
(same source, the node's compiler build); ours compiles to the same SASS on both boxes (45.58).

### 45.61 2026-10-19 21:45 IST: end of day -- what is saved where, and how to resume (8x node or a new Verda box)

**Saved on the laptop:** every run of the day (`session_2026_10_19/box_runs/`: the 19 Verda runs, `qp_1019`, `sb_qpu_1019`,
`node_setup_1019` with the node's build logs and PF's SASS dumps; the node's run folder checked equal by an rsync checksum dry run),
the candidates (`scripts/cands/fwd_QP*.cu`, made by `patches/make_pfq.py`), the tools (`deep/loop_sass.py`, `loop_order_diff.py`,
`qp_ncu.py`, `regions_fa3_pf.py`, `waits_fa3_pf.py`), the scripts. The colleague package `~/Downloads/attention_kernels_2026_10_19.zip`
(PF forward, Vj2 backward, dispatch, API, README with shapes and the speed table).
**Made box-independent tonight:** `scripts/benv_setup.sh` (fd09e69f: GK = $HOME/gautam_k), new `scripts/run_full_suite.sh`
(223b9f3d: full suite on any library, `run_full_suite.sh <gpu> <lib>`, 4 h limit: the node runs the CPU references ~5x slower).
**Resume:** `session_2026_10_19/RESUME_2026_10_20.md` (state, next tasks, exact commands for A = the 8x node, nothing to set up,
and B = a new Verda box: folders, rsyncs incl. the FA clone ~/Downloads/sft_research/flash-attention 94e22c9 and the step-6
files, build, the SASS check against the node's known counts, benv_setup.sh). Next: full suite on QPU, adoption (user), then
the diagonal tile first with QPU's loop order kept.

### 45.62 2026-10-21: pretraining hand-over -- branch `dist_side_gautam_lottn` (BLUCCL + our attention and loss work), tested, pushed

**Why:** the colleagues start pretraining (the user calls it the 1435M model) on `dist_side_merge_BLUCCL`. Only what helps
pretraining goes on a NEW branch from the latest BLUCCL; SFT-only work (packed RoPE positions, logits only at trained
positions, CudaNativeType in the loss) comes later with the SFT push.

**What went in (15 Tensor-Implementations files; BluTrain needed no code change):**

| change | files | measured on the 1x H200 |
|---|---|---|
| QPU attention forward (45.58-45.60) | `GQA_fwd_sm90.cu` 7e4197ea | forward +18.9% vs BLUCCL's v62 |
| Vj2 attention backward as the default (`USE_VJ2=0` = Vj1d) | `GQA_bwd_vj2_sm90.cu` (new), `GQA_bwd_generic.cu`, Vj1d / Vj1, `GQA_fwd_generic.cu`, the API headers / ops / backward node | backward +40.5% |
| loss: the gradient allocated once (it was allocated twice) | `LossBackward.cpp` | part of the memory below |
| loss: gradient written OVER the logits, the ONLY training path (no switch) | `LossKernels.cu/.h`, `LossBackward.cpp`, `LossOps.h` (contract comment) | -5.8 GiB allocated peak (bf16 logits) |
| loss: 16-byte loads / stores + 4 in flight, backward | `LossKernels.cu` | bf16 backward kernel 4.35 -> 1.64 ms |
| loss: 16-byte loads straight to registers, forward | `LossKernels.cu` | bf16 forward kernel 1.03 -> 0.93 ms |

Left out on purpose: the 11 norm / RoPE files (checked line by line: ONLY the per-chat positions plumbing for packed SFT rows),
the SFT trainer / loader / schedule, our test files. Same-day edits made in the master too (`BluTrain_sft`): the always-true
`if (inplace)` folded away in `LossBackward.cpp`, Vj1d's stale "this is the DEFAULT" comment fixed, the master forward = QPU (F3c
kept as `scripts/cands/fwd_F3c.cu` c6a67163).

**Branch creation (laptop, `~/Downloads/BluTrain_lottn`; frozen OLD copy `~/Downloads/BluTrain_lottn_old`):** copy of the master,
checked out at BLUCCL `afc462d` (10-10) with TI `02d0d18`, BluBLAS `ea666ab`, Context_Parallelism `661b6ef`; upstream tracking
removed so a bare `git push` cannot hit the colleagues' branch. The colleagues changed 16 TI files since our base `d96b991`, none
of them attention or loss, so the port is a copy (md5 = master). Traps met: the laptop has no git-lfs but `~/.gitconfig` makes the
LFS filter required, and BLUCCL's only LFS file (`dist/Orchestrator/BluRun/deploy/obs/dcgm/counters.csv`, a DCGM counter list)
broke the checkout -> the copies carry `filter.lfs.process=""`, `filter.lfs.required=false` (the file stays a pointer; not used by
build or training); `set -e` is ignored by this shell, so every step is chained with `&&`. Compile check: the 22 TUs that changed or
include a changed header, sm_90a, the BLUCCL tree: 0 errors. Laptop glibc is 2.39 again (no rsqrt workaround needed).
In-place safety checked in the production trainer `train/production_zero/bluscript_zero_ckpt.cpp` (also after the colleagues'
4f45a88): `logits` feeds only the loss and goes out of scope after `loss32.backward`; validation runs no backward.

**Box:** 1x H200 `root@86.38.238.55` (GPU 0, CUDA 13.2, GCC 13.3, 44 cores, 167 GB RAM), `~/gautam_k/lottn/{old,new,ab,runs}`,
data `~/gautam_k/pt_data/mix/{commoncrawl,edufineweb,code,math}` = links to the laptop's 4 edufineweb train shards + the val shard.
Build `make -j22 profiler tensor dist SM_ARCH=90` per tree (not `all`: `ensure-blurun` / `ensure-tmux` touch PATH / apt on a shared
box): libtensor OLD d5423cd3 (55,994,520 B), NEW 77af65f9 (56,649,024 B). yaml-cpp was missing: built 0.8.0 from GitHub into
`~/gautam_k/local` (static, `-include cstdint`), trainers by `CPATH=... LIBRARY_PATH=... make _ablation_compile MODE=production_zero
ABLATION_OUT=...` -> `pz_old` 0e197a4f, `pz_new` b1faa246. `libblublas.so` is found only through `LD_LIBRARY_PATH` (libtensor's
RUNPATH is `$ORIGIN/lib`), as the Makefile's own `RUN_LD_PATH` does.
**The production config, read by `--print-config` (exits before MPI / GPU):** 2304 wide, 26 layers, 18 / 6 heads, hd 128, ffn 6144,
vocab 50,304 (tied), T 4096, B 8, 1,048,576 tokens per step, Muon, 8000 steps, mix c2; `mixed_precision` -> `BLU_ALL_BF16=1`,
`BLU_BF16_LOGITS=1` (bf16 logits). ~1.59B parameters with the tied embedding (the user's "1435M" not confirmed). It also found a
**BLUCCL bug**: the YAML's `monitoring.production` (added in dba5cf4, 10-10) is unknown to `ConfigBridge.h`, so the trainer stops
at startup with its own default YAML ("unknown key"); still there in 4f45a88. Fix = one line in `env_table()`:
`{"monitoring", "production", "BLU_PRODUCTION", Kind::Bool},` (told to the user for the colleague). `BLU_PARALLELISM` is
`parallelism.enabled` (an int), never a path; the parallelism YAML is resolved relative to the main YAML.

**Tests (scripts `profiling_reports/pretrain_branch_2026_10_21/ab/`, results `.../box_runs/`):**
- **A loss** (`run_loss_ab.sh`, `loss_ab_1021_1444`, old / new / old / new, one binary, `ldd` checked per run): every loss bit
  pattern and gradient checksum BIT-IDENTICAL in all 4 runs, fp32 and bf16 logits, kernel level and through autograd (W.grad,
  X.grad). bf16: forward kernel 1.032 -> 0.931 ms, backward kernel 4.349 -> 1.640 ms (1.5 -> 4.0 TB/s, 84% of HBM peak), lm_head +
  loss backward in steady state (3rd repeat) 23.82 -> 21.09 ms per micro-step (OLD's 2nd repeat was still slower: allocator
  warm-up), allocated peak 10,536 -> 4,610 MiB (-5.8 GiB), reserved 12,954 -> 4,620 MiB. fp32: forward -4.6%, backward unchanged
  (fp32 already moved 16 bytes), allocated peak -8.9 GiB (not -12.3: the peak moves to the fp32 -> bf16 gradient cast; the MiB
  accounting adds up for all four cases). Clock 1965-1980 MHz, <= 640 W.
- **B attention** (`run_attn_ab.sh quick`, `attn_ab_quick_1021_1508`; the first try `..._1449` failed at link: `-lblublas` was
  missing in my build line): B 8, 18 / 6 heads, T 4096, hd 128: forward 1.215 -> 1.022 ms (507 -> 605 TFLOP/s), backward 4.296 ->
  3.058 ms (360 -> 506 TFLOP/s), forward + backward -26% (+35.1% faster); O, LSE, dK, dV bit-identical run to run in both, dQ not
  (by design in both). Quick suite on NEW 296 / 296 (31 s).
- **C production trainer** (`ab_trainer.sh 12`, `trainer_ab_1021_1515`): the colleagues' trainer and YAML, ONLY `validation_freq`
  250 -> 4 and the broken `monitoring.production` line removed; env: data path, no checkpoints / monitoring / CSV; the real LR
  schedule (no `BLU_MAX_STEPS`), each run stopped after step 12; old / new / old / new, ~7 min each.

| | OLD r1 / r2 | NEW r1 / r2 | |
|---|---|---|---|
| step time, steps 3-11 without validation | 22,827 / 22,833 ms | 21,463 / 21,467 ms | **-5.98%** (0.9402 in both pairs) |
| tok/s | 45,936 / 45,923 | 48,855 / 48,845 | **+6.4%** |
| MFU / TF per GPU (step 12) | 51.1% / 505 | 54.3% / 537 | |
| peak GPU memory (nvidia-smi) | 103,893 MiB | 96,725 MiB | **-7.0 GiB** (same in both pairs) |
| SM clock while busy, power | 1,526 MHz, 692 W | 1,493-1,497 MHz, 691-692 W | NEW slowed 2% by the 700 W cap |

  Precision: step 0 (forward only, same weights) 11.288473 (both OLD) vs 11.288469 (both NEW): each side reproducible, the 4e-6 is
  the attention forward's rounding (the loss is bit-identical). Over steps 0-12 the largest |NEW - OLD| (train 1.1-1.7e-4, norm
  2.6-3.6e-3, val 0.4-1.0e-4) matches OLD's own run-to-run spread (1.2e-4, 3.9e-3, 0.9e-4). The trainer's `[MEM]` / `[ALLOC]`
  lines print only after "Training complete", so the memory number is nvidia-smi's (reserved + context).
- **D full attention suite** (`run_attn_ab.sh full`, `attn_ab_full_1021_1551`): **1249 / 1249** in 546 s incl. the 1000-repeat
  race test at 8 x 4096 (the QPU adoption gate); the bench repeat: forward +18.6%, backward +40.9%.

**Commits (the user's; company identity; one per change, each compile-checked ALONE on BLUCCL's TI + the stages in order, the
tip byte-identical to the tested files; stage files in `profiling_reports/pretrain_branch_2026_10_21/commits/c1..c6`):** TI on
`02d0d18`: e857c77 single gradient allocation, 43a5c3f backward 16-byte, 37f2196 in-place only, fc4459b forward 16-byte, c435578
attention (Vj2 default, packed-row plan, v62 forward up to F3c: these three share headers and were never built apart), b1443a5 QPU.
BluTrain: a987851 (TI pointer), then the merge of BLUCCL **4f45a88** (the colleague's multinode BluCCL commit, pushed 10-21 15:55,
AFTER we had branched from afc462d; it does not touch the TI pointer; 0 conflicts, checked first with `git merge-tree`) ->
**3b80f15**, which differs from 4f45a88 by exactly one line, the TI pointer (Context_Parallelism -> their cf1372d).
**Pushed** with a new PAT kept in `~/.config/blubridge/github_pat` (chmod 600, outside every repo; pushes read it with `$(cat ...)`);
GitHub verified: TI `dist_side_gautam_lottn` = b1443a5, BluTrain = 3b80f15.
**For the colleague:** `git fetch origin && git checkout dist_side_gautam_lottn && git submodule update --init --recursive`, then
`make -j$(nproc) all SM_ARCH=90`; plus the `monitoring.production` fix above.
**Also re-checked (pretraining fixes from earlier, inside what the colleagues run):** Muon V1n (a9bc08c is an ancestor of 02d0d18,
0 racy `atomicAdd(acc`), the F1 non-finite-gradient fix in both ZeRO optimizers, the optimizer grad cast converting straight into
`shard_grad_fp32_` (no 5.9 GB temp), F4 (W = 1); the GEMM library is Kota's BluBLAS `ea666ab` (accepted for BluTrain, newer than
the master's 35e51a0).

### 45.63 2026-10-21: FUTURE TASKS for pretraining (user: "we will do one by one") + what the colleague's DPO loss kernel uses

1. **Casts on H200: ALREADY FAST (checked the same day; this item was wrong as first written).** `ConversionKernelsSM90.cu`
   (kathir-23s, d8506d5, 2026-08-06) gives H100 / H200 their own vector casts: fp32 -> bf16 `sm90::vec8_convert_to_bf16` (8 per
   thread), bf16 -> fp32 `sm90::vec4_convert_to_fp32` (4 per thread), 128-bit accesses, bit-exact round-to-nearest-even.
   `ConversionKernels.cu:198 / :229` take them by default when both pointers are 16-byte aligned (off only with
   `BLUTRAIN_NO_SM90_CONVERT` / `BLUTRAIN_NO_SM90`); B300 has its own vec8 path (`:183 / :213`); the scalar `convert_type_kernel`
   (one element per thread) is only the fallback (other dtype pairs, unaligned data). My 08-26 note (23.6% casts on the 4B
   trainer, scalar kernel) was older than this path. Left (low priority): an nsys run to confirm the production casts take the
   vector path and what they cost now. **Wiring verified the same day:** `to_dtype` (no kernel when the dtype already matches)
   -> `as_type` -> `convert_type_cuda_generic` -> `convert_type_cuda<float, bfloat16_t>` (and back) -> sm90 vec8 / vec4; the cast's
   backward is `as_type` too; the ZeRO optimizers call `convert_type_cuda_generic` directly; `BLUTRAIN_NO_SM90*` is set only by the
   older `train/production/bluscript.cpp`, never by production_zero; leftovers are handled inside the vector kernel; both kernels
   are in the box's built OLD and NEW `libtensor.so` (cuobjdump -sass -fun). Only runtime condition: 16-byte aligned pointers. Casts happen in `as_type` / `autograd::to_dtype` (trainer: around RoPE, Linears and the
   residual, embedding, wte for the lm_head, logits, loss; many are no-ops in the all-bf16 config) and in the ZeRO optimizers.
2. **PDL (programmatic dependent launch) on the remaining kernels** (convert, attention, norm, SwiGLU, add, copy). 08-25: 0.06%
   measured under nsys = a lower bound (the profiler spreads kernels apart); the decisive test is unprofiled tok/s on vs off.
3. **Attention: we already beat FA3 / FA4 / cuDNN everywhere except ONE tie** (corrected the same day: 0.968 is ours / FA3 TIME,
   below 1 = ours faster; `session_2026_10_19/box_runs/sb_qpu_1019/summary_steady.txt`). Dense ours / FA3: forward 1.004 (986 vs
   982 us = the tie), backward 0.968, f+b 0.976; ours / FA4: 0.919, 0.989, 0.972; ours / cuDNN: 0.863, 0.891, 0.884; packed: ours
   faster than all three in every column (vs FA3 0.865 / 0.953 / 0.933). The only open idea is for that tie: diagonal tile first
   (QPD, 45.59, lost to a compiler loop reorder, +19 cycles per step) retried with QPU's loop order kept; gate `scripts/loop_sass.py`.
4. **GEMM hybrid** (our 967-line pin table + Kota's grid-keyed dispatcher): PAUSED since 09-17 when the work moved to SFT; only on
   the user's word. Production already uses Kota's BluBLAS `ea666ab`.
- Correctness to-do (not speed): racecheck warnings inside TI's `matmul_bf16_optimized` (used by Muon's Newton-Schulz); never a wrong
  result in 12,000 calls (10-05), still unexamined.

**The colleague's DPO loss kernel** (`DPOLossKernels(1).cu`, 546 lines; this one file answers the type question -- her header only
declares the same functions): instantiated for OUR types (`float`, `bfloat16_t`); memory moves as raw `uint4` (16 bytes, the same
idea as our `loss_ld16` / `loss_st16`); bf16 -> fp32 by a bit shift of the raw word; fp32 -> bf16 by the CUDA hardware intrinsics
`__floats2bfloat162_rn` (two values per instruction, vectors) and `__float2bfloat16_rn` (scalar tail). Her comment says our
`bfloat16_t`'s "software conversion" made her kernels issue-bound: true for TIs before **b975be5 (09-29)**, which made our type use
the hardware conversion on the GPU (`Types.h:111-114`; bf16 -> fp32 was always a shift). Our pretraining / SFT loss kernels use our
types (`LossVec<bfloat16_t>` through `uint4`), no CudaNativeType. **Corrected the same day, from the SASS:** there is NO
conversion difference. ptxas pairs our per-element `__float2bfloat16_rn` (Types.h) into `F2FP.BF16.F32.PACK_AB` with TWO source
registers: the bf16 backward (in-place and out-of-place) has 21 F2FP, 20 of them two values, 1 with RZ (the scalar tail) -- the
same instruction her `__floats2bfloat162_rn` gives. Two values per instruction is the sm90 maximum (one 32-bit result); 8 values
= 4 instructions in both kernels. So CudaNativeType in our loss kernels would change the code's spelling, not its instructions.
Task 1 is a different kernel: the trainer's whole-tensor cast `convert_type_kernel` converts ONE element per thread (2-4 byte
accesses); the B300 vec8 kernel reads 2 x float4 per thread, converts in pairs (`__float22bfloat162_rn`), writes 16 bytes.

### 45.64 2026-10-21 17:24-18:36 IST: SFT kernel stats against the other libraries (Kathirvel's ask), fresh on the 1x H200

**Ask** (Kathirvel's email, 10-21): "need the kernels stats for SFT kernels against the peers". User: fresh, robust numbers;
standalone benchmarks of every modified kernel against the other frameworks and against our pretraining kernels; padding vs
packing; full SFT runs on the 5,000 chats (padding and packing) with a loss-curve graph. GPU use approved for **the 1x H200
root@86.38.238.55 only** (not the 8x node); I launched and monitored the queue myself under that approval.

**Report for Kathirvel** (my voice, "I", no we/our): `profiling_reports/sft_benchmarks_2026_10_21/sft_kernel_benchmarks.md`
-> `sft_kernel_benchmarks.docx` (python-docx; the laptop has no LibreOffice / pandoc now), figure
`box_runs/sft_padding_vs_packing.png`, all raw logs under `box_runs/` (+ `sftb_results.tgz` from the box).

**Scripts** (all in `profiling_reports/sft_benchmarks_2026_10_21/`, box copies in `~/gautam_k/sftb/`): `PLAN.md` (126203c0),
`setup_sftb.sh` (fa8f1d16: CPU-only setup in parallel -- library + trainer + `tva_a` build of the laptop master, the loss rivals'
env, the attention rivals' env, the base checkpoint; 17:24:58-17:32:06 = 7 min), `run_all_sftb.sh` (22a6bc58: the GPU queue, a
GPU-free check before every step, a status line per step in `sftb/run_status`), `run_extra_sftb.sh` (744714df: A3b),
`bench_rope_sft.cu` (d6d4a5dd), `bench_rope_rivals.py` (3d122dd8), `bench_sparse_ce_sft_v3.cu` (2d644d88 = the 10-17 v2 bench +
the trainer's in-place backward), `plot_sft_runs.py` (25185260), `summarize_sftb.py` (87c143b9), `build_report_docx.py` (f78bb3f0).
Library under test: master build `libtensor` b5c9844c (LossKernels.cu 064590b4, GQA_fwd_sm90.cu 7e4197ea = QPU, Vj2 default),
`libblublas` d15d3c4b; OLD = BLUCCL lib `d5423cd3` (`~/gautam_k/lottn/old`); trainer `bluscript_sft_bin` 99db45fc.
Box: CUDA 13.2, driver 580.178.04, torch 2.14.1+cu130, Triton 3.8.0, FA3 3.0.0, FA4 4.0.0b34, cuDNN 9.24 (92400), Liger 0.8.5,
CCE 25.1.1; H200 700 W, free clocks.

**Timeline (box `run_status`)**: S1 17:43:58-17:44:30 | A3 -17:51:13 | A5 -17:52:08 | B1 -17:54:02 | A1 -17:57:31 (16 runs x ~12 s,
not the 21 min I planned) | A4 -18:04:21 | B2 -18:06:46 | C padded 18:06:46-18:24:48, packed -18:30:59 | A3b 18:31-18:36:05.
GPU free afterwards (no process of mine left).

**S1 attention quick suite (master build):** 296 / 296, 31 s, backward mode Vj2 (default).

**A3 loss vs the rivals** (bf16 logits [32,768 x 50,304], real stage-0 labels; `box_runs/A3_loss_rivals`, table by
`sft_loss_vs_rivals_2026_10_17/loss_compare.py`), forward + backward ms, theirs / ours:

| case (trained of 32,768) | ours | Liger | torch.compile | PyTorch bf16 | PyTorch fp32 |
| --- | --- | --- | --- | --- | --- |
| real_min (2,920) | 1.003 | 1.304 (1.30x) | 3.875 (3.86x) | 6.227 (6.21x) | 21.14 (21.1x) |
| real_first (6,112) | 1.173 | 1.523 (1.30x) | 3.874 (3.30x) | 6.231 (5.31x) | 21.14 (18.0x) |
| real_median (7,577) | 1.255 | 1.623 (1.29x) | 3.878 (3.09x) | 6.232 (4.96x) | 21.14 (16.8x) |
| real_max (12,729) | 1.530 | 1.968 (1.29x) | 3.900 (2.55x) | 6.237 (4.08x) | 21.16 (13.8x) |
| nomask (32,768) | 2.603 | 3.603 (1.38x) | 3.942 (1.51x) | 6.266 (2.41x) | 21.18 (8.14x) |
| all_masked (0) | 0.789 | 1.097 (1.39x) | 3.876 (4.91x) | 6.220 (7.89x) | 21.14 (26.8x) |

On 10-17 (before the 16-byte backward) Liger was FASTER than ours (0.61-0.98x); now ours wins every case. Backward at 83-85%
of 4.8 TB/s. Extra memory: ours 0.2 MiB with the in-place trainer path (the bench's out-of-place entry adds the 3,144 MiB
gradient buffer), Liger 0.3, compile 3,144, bf16 9,432, fp32 18,864 MiB. Loss = PyTorch fp32 within 1e-6; 6 / 6 checks pass.
**Part 2, lm_head + loss** (forward + backward ms, theirs / ours): median 34.93 vs PyTorch 49.01 (1.40x), Liger FLCE 33.19
(0.95x), CCE 31.86 (0.91x), CCE exact 31.37 (0.90x); real_max 35.20 vs 49.53 / 34.50 (0.98x) / 52.66 (1.50x) / 51.89; nomask 36.28
vs 52.39 / 38.85 (1.07x) / 135.21 (3.73x) / 132.97. **The one place others win:** on the median SFT micro-batch CCE is 9% and
Liger FLCE 5% faster, and they peak at 367 / 563 MiB (ours keeps the 3,144 MiB bf16 logits): our lm_head GEMMs compute logits for
the 77% masked tokens. CCE's time scales with trained tokens (31.9 ms at 23% -> 135 ms at 100%); FLCE drops 38.9 -> 33.2 ms
(mechanism not checked). Fix = the recorded future item "logits only at trained positions" (needs the BluBLAS M/N%64 guard).

**A3b in-place backward** (the trainer's path since 10-21) vs the out-of-place entry the benches call
(`box_runs/A3b_loss_inplace`): in / out 0.983-1.007 over 6 padded + 6 packed micro-batches; F+B in-place 0.4-1.3% lower; the
gradient checksum bit-identical in 12 / 12.

**A4 loss OLD (BLUCCL) vs NEW**, same binary, old / new / old / new (`box_runs/A4_loss_old_new/summary.txt`), F+B new / old:
real_min 0.813, real_first 0.697, real_median 0.666 (1.882 -> 1.254 ms; B 1.605 -> 1.007), real_max 0.589, nomask 0.485
(5.376 -> 2.610), all_masked 0.911; gradients bit-identical in all 4 runs, 6 / 6 cases.

**B2 loss on the packed labels** (1,517 rows, 189 micro-batches, mean trained 76.8%; `box_runs/B2_loss_packed`): median packed
micro-batch F+B 2.203 ms (real_first 2.245, real_min 1.966, real_max 2.397, nomask 2.607, all_masked 0.787); 6 / 6 PASS. Per
epoch: 625 x 1.255 = 0.78 s padded vs 189 x 2.203 = 0.42 s packed -> packing about halves the loss kernel time (our forward
skips masked rows, the backward only zero-writes them, so padded rows are already cheap for the loss).

**A5 QK RMSNorm + RoPE** (q [8,18,4096,128], k [8,6,4096,128] fp32, theta 10,000, eps 1e-6; `box_runs/A5_rope`):
`test_rope_positions` 37 / 37 PASS. Ours fwd / bwd ms: nopos 0.2087 / 0.5080, identity 0.2105 / 0.5207, packed (rows 0-7 of the
packed data, 29 chats) 0.2102 / 0.5209; identity == nopos BIT-IDENTICAL (q_out, k_out, dq, dk). Positions cost +0.7% fwd /
+2.5% bwd. Forward 3.8 TB/s (80% of peak). Rivals (packed; identity within 6%): torch.compile 0.2852 / 0.7860 (F+B call 0.9163),
Liger 0.7795 / 1.7442 (2.3376), HF eager 2.4829 / 5.0130 (7.3208). Ours / theirs: compile fwd 0.74, bwd 0.66; Liger 0.27 / 0.30;
eager 0.08 / 0.10. Kept for backward: ours 3 MiB (rstd), compile 3, Liger 3, eager 387 MiB; all rivals vs fp64 ~1e-6. Liger's
cutedsl backend is not installed -> its Triton backend ran (warning in rivals.log).

**A1 attention scoreboard** (`scoreboard_free.sh 0 sb_sft_1021 sft_new sft_new`, steady state; `box_runs/A1_scoreboard`), ours /
rival time: dense vs FA3 fwd 1.006, bwd call 0.975, f+b 0.982; vs FA4 0.918 / 0.988 / 0.970; vs cuDNN 0.828 / 0.888 / 0.873.
Packed (real_rows_752) vs FA3 0.863 / 0.956 / 0.935; vs FA4 0.776 / 0.972 / 0.923; vs cuDNN 0.821 / 0.898 / 0.881. ms: dense ours
0.983 / 3.190 / 4.173, FA3 0.977 / 3.271 / 4.248, FA4 1.071 / 3.230 / 4.301, cuDNN 1.188 / 3.593 / 4.781; packed ours 0.408 /
1.527 / 1.934, FA3 0.472 / 1.597 / 2.069, FA4 0.525 / 1.571 / 2.096, cuDNN 0.496 / 1.700 / 2.196. Same as the 10-19 node
scoreboard within 0.5% (dense fwd 1.004, packed f+b 0.933) -> holds on a second machine. Only open item: the dense-fwd tie.

**B1 attention, 7 layouts** (`tva_a VARLEN_ONLY=BENCH` + `step6/bench_step6_libs.py`; segments export == segments_1012.txt;
`box_runs/B1_attn_layouts`). Chats per layout from the test's own FFD packing (recomputed in `summarize_sftb.py`: 5,000 rows ->
1,517 packed rows): real_rows_0-7 15, 752-759 24, 1504-1511 72, uniform 512/1024/2048/4096 = 64/32/16/8. Packed f+b, theirs /
ours: FA3 1.05-1.34x real, 1.00-1.35x uniform; FA3-det 1.13-1.32 / 1.09-1.33; FA4 1.04-1.31 / 1.02-1.31; cuDNN 1.15-1.23 /
1.12-1.21; cuDNN+expsum 1.24-1.49 / 1.20-1.50; Flex doc mask 1.65-1.80 / 1.59-1.62; torch varlen 2.08-2.22 / 2.04-2.25; SDPA
cuDNN mask 3.89-11.48 / 3.58-12.04; SDPA efficient mask 9.38-27.78 / 8.60-29.11; SDPA flash full row 2.18-6.38 / 2.00-6.68 (flash
refuses a mask: the 7 "error" lines). Backward alone: FA4 0.6-2.4% ahead on real_rows_0-7, 752, uniform_2048; FA3 / FA4 4.5% /
4.2% ahead on uniform_4096. **Caveat found today:** in B1 our process ran at a median 1,635 MHz (691 W, power-capped) and the
rivals' Python process at 1,980 MHz (622 W, pauses between calls) -- `B1/clocks.csv` -- so B1 favours the rivals; quote A1.
GATE lines: 21 / 21 PASS (FA4 and cuDNN vs FA3), 7 / 7 negative controls FAIL as they must.
Padding vs packing for the same chats (ours, f+b; padded = chats x dense time per row): real 0-7 7.69 vs 3.86 ms (0.50), 752
12.57 vs 2.01 (0.16), 1504 36.95 vs 1.29 (0.035); uniform 512 0.038, 1024 0.099, 2048 0.30, 4096 1.003. Plan overhead on full
chats (uniform_4096 A vs dense): fwd +1.5%, bwd 0%.
**Our older attention, same box today** (test B of 45.62): fwd 1.215 -> 1.022 ms (0.841), bwd 4.296 -> 3.058 (0.712), f+b 0.740.

**C full SFT runs** (from `blumodel_step_8000`, AdamW 2e-5, warmup 0.03, cosine to 2e-6, 2 epochs, val every 10 steps, BLU_ALL_BF16,
1 GPU; `box_runs/C_P_padded`, `box_runs/C_K_packed`; `plot_sft_runs.py` -> `sft_padding_vs_packing.png` / `.txt`):

| | padding (B 8 x 8 = 64 rows/step) | packing (B 4 x 5 = 20 rows/step) |
| --- | --- | --- |
| steps | 156 (78/epoch) | 151 (~75.5/epoch) |
| trained tokens seen | 9,531,228 | 9,498,277 |
| step time, median | 5.014 s | 1.629 s (0.325) |
| trainer clock, whole run | 16.99 min | 5.14 min (0.302) |
| wall clock incl. process start + 6.5 GB checkpoint load | 18.0 min | 6.2 min |
| chats/s; trained tok/s; all tok/s | 12.78; 12,185; 52,282 | 40.65; 38,608; 50,281 |
| MFU median | 58.1% | 55.9% |
| train mean epoch 1 / 2 | 1.3158 / 1.2391 | 1.3171 / 1.2403 |
| val start -> end (drop) | 1.446384 -> 1.283890 (0.1625) | 1.449444 -> 1.286579 (0.1629) |
| peak GPU memory | 123,233 MiB | 85,757 MiB |
| max grad norm | 0.919 | 0.946 |

Same learning, 0.30x the time. The ~0.003 val gap is already there at step 0 (same weights): the loader drops the last
incomplete micro-batch, so the padded val uses 441,382 of 444,233 val tokens (B 8: 4 rows dropped) and the packed val 435,043
(B 4: 3 rows dropped). For an exact val comparison next time: val B must divide the row count (500 rows: B 4/5/10; 147 rows: B
1/3/7), or B 1 as in the 10-16 e2e test. Packing's 4% lower all-tokens/s: fixed per-step work (optimizer) is a bigger share of a
1.6 s step + the smaller micro-batch -- not separated. Lower peak memory = mostly the smaller micro-batch (4 vs 8 rows), not a
packing gain. The padded run repeats 10-07 (final val 1.2839, epoch means 1.316 / 1.239) at 5.0 s/step (was 5.3) and 58% MFU (was
53%); no empty_cache slow steps any more (all steps ~5.0 s).

**Lessons recorded:** (1) a single-process multi-library sweep can hide a 17% clock difference -- always log clocks per process;
(2) `loss_compare.py` counts ours' peak with an out-of-place gradient -- with the in-place trainer path it is 0.2 MiB; (3) the
`head -c` cut of `segments_1012.txt` ("153" = "1536") looked like a packing mismatch -- it was not (the file has 1536/1536/1024).

**45.64 addendum (19:00-19:15 IST): why the padding / packing curves differ, and the exact-comparison run (prepared, not run).**
User's question: the kernels give the same numbers padded or packed, so why do the per-step train losses of the two runs differ
(figure panel a), why the small val gap (b), and what (c) shows. Answer, measured (`step_composition.py` -> `box_runs/step_composition.txt`;
every padded chat found in the packed file by token hash, 5,000 / 5,000; every step of both runs rebuilt and checked against the
trainer's trained_tokens column, equal for all 156 / 151 steps):
- the runs train on DIFFERENT chats at the same step: the padded file is one random chat order (seed 1234), step k = chats
  64k..64k+63; the packed file is FFD + its rows shuffled (seed 1236), step k = 20 rows. Padded step k and packed step k share
  0.75 of 64 chats on average (max 3). Packed steps hold 54-78 chats (padded 64), 57,543-67,333 trained tokens (padded 49,618-74,892).
- proof: step 0 has IDENTICAL weights in both runs, yet train loss 1.449952 (60,263 tokens) vs 1.403362 (64,520 tokens) -> only the
  chats differ. One step's train loss moves ~0.08 from step to step (std of differences 0.084 / 0.089). Last steps differ (1.262 /
  1.317, different chats); epoch means agree (1.3158 / 1.3171, 1.2391 / 1.2403).
- (b): the loader drops the last incomplete micro-batch -> padded val (B 8) = val chats 0-495 (496), packed val (B 4) = 144 of 147
  packed rows = 490 chats; 486 in both, 10 only padded, 4 only packed -> the constant ~0.003 gap, present at step 0 with identical
  weights. Same drop (0.1625 / 0.1629). (c) = (b) against minutes.
- What proves equal numbers: kernel tests (packed == alone bit-exact: attention G10/G11, RoPE test_rope_positions) + the 10-16 same-500-
  chats test (step-0 val 1.445038 both, grad norm 0.7935 both). A whole run with the same chats per step is the missing proof.
**Prepared exact run** (`build_samesteps.py`, laptop, CPU only): `~/Downloads/sft_data/stage0_5k_samesteps_as_padded` (77 MB):
train = 156 groups x 28 rows = 4,368 rows, group s = exactly the 64 chats of padded step s (incl. the epoch-wrap step 78), worst-fit
decreasing so every row has >= 1 chat (the loader refuses a row without a trained token, SftDataLoader.h:218; empty filler rows are
impossible); chat slices copied from the packed file; val = padded val rows 0-495 (= what the padded run used), dense. Checks
passed: loader rules, every group == the padded step's chats by token hash, trained tokens per step == the padded run's column.
Run (NOT started; needs the user's OK + a free GPU): `BLU_SFT_DATA=<that folder> BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=28 BLU_SFT_EPOCHS=1`
(max_steps = 1 x 1,092 / 7 = 156, same LR per step as the padded run), rest as phase C; ~11 min; compare step by step with
`box_runs/C_P_padded` (trained tokens must be equal; train loss, grad norm, val expected equal to 3-4 digits).
Report updated (sections 5-7) and rebuilt; root copies `sft_kernel_benchmarks.md/.docx` (image path fixed for the root). Antigravity:
both docx viewers were already installed; added `workbench.editorAssociations` `*.docx -> docxViewer.DocxEdit` (Docx Renderer).

**45.64 addendum 2 (19:16-19:27 IST): test 2 RUN -- the same chats in every step, padding vs packing REPRODUCE.**
User's OK ("start the run now"); GPU free at 19:14 (the colleague's job had ended). Data `stage0_5k_samesteps_as_padded`
copied to the box (md5 equal, 10 files), `run_samesteps.sh` (ad6dd325): B 4, 28 rows per step, 1 epoch, val = the padded run's
496 val rows (dense; `packed_rows_of` returns an empty plan for a non-packed loader, so a packed train + dense val folder works).
Run 19:16:02-19:26:46, exit 0, 156 steps, 9,531,228 trained tokens (= the padded run), peak 85,757 MiB, 2.18 s/step, 9.69 min.
Against `C_P_padded` (`compare_samesteps.py` -> `box_runs/sft_padding_vs_packing_same_chats.txt` / `.png`):
- trained tokens equal at all 156 steps; step 0 train 1.449952 both (diff 0), grad norm 0.7446 / 0.7445, val 1.446384 both;
- train loss |diff| mean 8.27e-05, max 3.08e-04 (step 21); by range 0-9 7.3e-05, 10-49 7.7e-05, 50-99 8.3e-05, 100-155 8.8e-05
  (does not grow); grad norm |diff| mean 1.6e-04, max 1.35e-03 (0.67% relative); epoch means 1.3158 / 1.2391 in BOTH;
- val at all 17 checks within 9.4e-05 (end 1.283890 / 1.283943); last train 1.262134 / 1.261958.
Remaining difference = order of additions (8 x 8-row vs 7 x 4-row micro-batches, GEMMs with other row counts, atomic dQ in the
attention backward), bf16 last bits. Verdict: packing trains the same model as padding; test 1's per-step differences came only
from the step composition. Report section 5 rewritten (5.1 test 1 setting / figure 1 / results, 5.2 figure 1 graph by graph =
the user's doubts, 5.3 why a second test, 5.4 test 2 setting + how the steps were built + figure 2 + results, 5.5 figure 2 graph
by graph, 5.6 result), correctness table + summary updated, the old "next" item removed; master + root copies rebuilt (docx
2 figures, 14 tables). GPU free after the run.

**45.64 addendum 3 (~19:45 IST): 1x H200 box backed up before it may be closed.** `profiling_reports/box_1x_h200_backup_2026_10_21/`
= `~/gautam_k` minus venvs / checkpoint / git clones / build trees (all rebuildable); 1.4 GB, 485 files, checksum-verified
(0 differences); plus the tested libraries of the pretraining A/B (`lottn/old_libs`, `lottn/new_libs`). Index: `README_BACKUP.md`.

### 45.65 2026-10-22 morning: the colleague's attention mail, the SFT data for the 1.7B model, the RoPE positions zip

**The colleague's attention mail (Uma Mahesh).** His forward changes are the same mechanisms as our QPU forward (K1 / K5 /
SM3 / QPU: the P.V wait ordered after the softmax with `order_after(rs0)`, the `sub.f32 zero` trick, K1_SPLIT 16), which we
pushed on 10-21 (TI b1443a5, 15:58). Recorded in memory (`project_dense_fwd_gap_1017`).

**SFT data for the DPO team's checkpoint** (they asked for an SFT checkpoint trained on robust data, not the 5,000-chat set).
Compared from the official papers only:
- Tülu 3: 939,344 prompts; used for Llama 3.1 8B / 70B / 405B; LR 5e-6 / 2e-6; 2 epochs; batch 128; max length 4,096.
- OLMo 2 1B: SFT on the **0225 variant of the Tülu 3 mix (`allenai/tulu-3-sft-olmo-2-mixture-0225`, 866,138 chats)**; average
  score 36.9 (base) -> 40.6 (SFT) -> 42.7 (after the later stages). OLMo 2's tokenizer is cl100k-based (§2.2).
- OLMo 3: Dolci Think 2,268,178 / Instruct 2,152,112 chats, 45.4B / 3.4B tokens, 32K context, 7B / 32B only -> does not fit our
  T 4096 or our model size.
- **Chosen: the Tülu 3 OLMo-2 mix** (the mix used for a 1B model), revision `d91a0785ade02942520280fb484866fce41e448f`.
  **Aya dropped** (`ai2-adapt-dev/tulu_v3.9_aya_100k`, 99,989 chats): our base is trained on English only and the GPT-2
  tokenizer needs 2-4x more tokens for other languages. **2,000 chats held out** as the validation set. The tokenizer stays
  GPT-2 tiktoken (the model's embedding uses GPT-2 ids; it can encode any language, it is just inefficient for non-English).

**RoPE positions zip for a colleague:** `~/Downloads/rope_positions_2026_10_22/` (+ `.zip`, 172 KB, 82 files): `README.md` (443
lines: why positions restart, the benefits, every change line by line, the tests), `sft_branch/` (our files), `for_bluccl/`
(merged into BLUCCL keeping Kruthik's bf16 code: 0 conflicts, compiles 9/9), `patches/` (306 changed lines), `tests/`,
`sft_branch_trainer_and_loader/`, `evidence/`.

### 45.66 2026-10-22: Kruthik's bf16 fused RMSNorm + RoPE in the SFT branch, with per-chat positions (stages A, B, C)

**Why.** The SFT trainer used the fp32 fused kernel: bf16 q/k -> transpose -> copy -> widen to fp32 -> kernel -> narrow to bf16,
with extra copies and fp32 tensors kept for the backward. The production trainer (BLUCCL `885ff4d`,
`train/production_zero/bluscript_zero_ckpt.cpp:627-643`) uses Kruthik's bf16 kernel by default (`BLU_QKROPE_FP32=0`): it reads
token-major bf16, writes head-major bf16, fp32 math inside, deterministic dgamma. It had no per-chat positions. User's plan:
port it as it is, test, add positions, test, switch the trainer, test, then resume the SFT run.

**Stage A (as it is).** Into the laptop master `~/Downloads/BluTrain_sft/Tensor-Implementations`: `src/Kernels/cuda/norm/
FusedRMSRoPE_bf16.cu` from BLUCCL 02d0d18 (md5 6f360b51, saved as `profiling_reports/rope_bf16_port_2026_10_22/kruthik_original/`)
+ only Kruthik's additions in 7 wiring files (+22/+25/+12/+30/+24/+39/+107 lines, -0); the files before the port are in
`backup_sft_before/`. Test `test_rope_bf16_vs_fp32.cu` (a9d63b29; the fp32 path exactly as bluscript_sft.cpp ran it vs
`fused_rms_rope_*_bf16_tm`), H200: **all 8 outputs bit-identical** (q_out, k_out, q_rstd, k_rstd, dq, dk, dgamma_q, dgamma_k) at
B 4 and 8; time fp32 path 0.7449 ms vs bf16 0.2462 ms (0.330) at B 4, 1.5133 vs 0.5425 (0.358) at B 8.

**Stage B (positions).** The same scheme as the fp32 kernels: a compile-time `HAS_POS` template flag in the forward and backward
kernels, `positions` appended LAST through every layer (`FusedRMSRoPEKernels.h`, `Normalizations.h/.cpp` with the existing
`fused_rms_rope_positions_ptr` check, `NormalizationOps.h/.cpp`, `FusedRMSRoPEBf16Backward` with a `positions_` member);
the launcher picks `HAS_POS=true` only when the pointer is not null. Kernel file now md5 5338bd7b; diff vs Kruthik's in
`stage_b_kernel_vs_kruthik.diff`. **Pretraining unchanged: 9 / 9 kernels SASS-identical** to Kruthik's original (`sass_compare.py`,
`orig.sass` / `new.sass`); the `HAS_POS=true` kernels are different code (+8 to +32 instructions), so the comparison is not empty.
Test `test_rope_bf16_positions.cu` (dad694c1), H200 GPU 0, B 4 and 8: check 1 identity positions == none, check 2 bf16 packed ==
fp32 path packed (and packed differs from none in 30.0M / 37.7M values, so positions are used), check 3 one chat inside a packed
row == the same chat alone: **all bit-identical**. Time fwd+bwd one layer: B 4 bf16 no positions 0.2427 ms, positions 0.3295
(+35.8%), fp32 path with positions 0.7578 -> the new SFT path is 0.435x the old; B 8 0.5409 / 0.6907 (+27.7%) / 1.5294 (0.452).

**Stage C (the trainer).** `bluscript_sft.cpp` (now md5 f2024ab2): `fused_rms_rope_bf16(q, k, Nq, Nkv, q_gamma, k_gamma,
cos_sin_cache, 0, qk_eps, false, g_packed.positions)` by default, `BLU_QKROPE_FP32=1` = the old fp32 path (as production).
Check script `stagec_check.sh` (0168c3d8): two short runs (4 min each) on the real Tülu data from the real base checkpoint.
- First attempt: both runs aborted in the checkpoint load: `[Architecture Mismatch] Shape mismatch for tensor 'param_000260'`.
  `ckpt_shapes.py` (eba478fe) read `metadata.bin`: **fullchess has 282 weights = 28 layers x 10 + embedding + final norm**; our
  trainer builds 26 layers (param 260 = our embedding = their layer 27's first weight). Every other shape equal (d 2304, 18/6
  heads, ffn 6144, vocab 50304, tied). Added `BLU_N_LAYERS` and `BLU_ROPE_THETA` overrides (defaults unchanged 26 / 500000)
  right after `ModelConfig cfg;` -- a checkpoint stores weights, not the config, and **a wrong theta does not error**.
- theta: our SFT trainer had 500000 (from commit 7e2264f, the navin-sep1 cherry-pick), the production trainers 10000. Measured
  (28 layers, same data, same weights, step 0): theta 10000 val 1.230082 / train 1.180083 / grad norm 0.40; theta 500000 val
  1.259555 / train 1.211322 / norm 1.01. Then **Navin's pretraining log (`bluscript_zero_ckpt_022.log`, repo root) gave the real
  config: 28 layers, theta 50000** (NeoX), T 4096, resid/logits/embed bf16, 1,701,485,824 params, Muon + AdamW peak 1.2e-3 ->
  1.2e-4 WSD (warmup 700, stable 17,380, decay 4,520), 22,600 steps x 1,048,576 tokens = 23.7B tokens, final val 1.3088.
  `rms_eps` 1e-5 and `qk_eps` 1e-6 are not in the log; every trainer copy uses these.
- bf16 vs fp32 at theta 10000: **step 0 identical** (train 1.180083, val 1.230082, norm 0.4017); steps 1-5 equal to 4-5 digits;
  step time 1.664 s vs 1.715 s (**3.0% faster** than the fp32 path). The trainer is not bit-reproducible run to run in the
  backward: two identical bf16 runs gave norm 0.4019 vs 0.4017 at step 0 and 1.267043 vs 1.267130 at step 1 (open item).
  Logs: `box_logs/stagec/` (`bf16_theta10000.log` there is the bf16-vs-fp32 run; it overwrote the theta-test log of that name).
- Why the restart of positions per chat is kept, its cost, and what every library does: `kernel_diff_sft_pretraining.md`.

### 45.67 2026-10-22: the Tülu 3 SFT run (box 86.38.182.101, 4x H200, GPU 0)

**Box setup** (the user typed every command): `~/gautam_k/{BluTrain_sft (laptop master), local (yaml-cpp), prepenv (datasets
5.1.0, tiktoken 0.14.0, numpy 2.5.4, huggingface_hub 2.2.0), ckpt/fullchess, sft_data, runs, rope_opt}`; Python 3.12.3, CUDA
13.0.88, driver 580.178.04, Open MPI 4.1.6, gcc 13.3.0 (all in `~/Downloads/box_backup_2026_10_22/env_versions/`).

**Data** (`sft/prepare_sft_data.py`, md5 d4d8f8b1: added `--num 0` = all, `--skip-kept N`, `--drop-source` repeatable; tested
on fake data: val and train disjoint, union = all English chats, no Aya). Val `--num 2000`: 2,000 chats -> 320 packed rows
(898,276 trained tokens). Train `--num 0 --skip-kept 2000`: 741,413 chats -> **115,904 packed rows** (6.4 chats per row, 474.7M
positions, **325.7M trained tokens per epoch**); dropped too long 22,020, Aya 99,989, empty answer 716. Base checkpoint
`navingv/fullchess` (8-rank save; `metadata.bin` 180,039 B + `data_rank_0.bin` 5,191,315,968 B hold the weights).

**Settings** (`run_tulu3_sft.sh`, 0db5bdad): packed rows only (decided 10-22: no padding anywhere from now on), 20 rows per step
(B 4 x GRAD_ACCUM 5 x W 1, ~128 chats), 2 epochs = **11,590 steps**, **Muon (matrices) + AdamW (embedding, norms) at peak lr
5e-5** for both, warmup 3% (348 steps), cosine to 5e-6, `BLU_ALL_BF16=1`, `BLU_N_LAYERS=28`, `BLU_ROPE_THETA=50000`, bf16 RoPE,
validation every 200 steps, checkpoint every 1000 steps, newest 2 kept. Why Muon 5e-5: the base was pretrained with Muon;
Moonlight (arXiv 2502.16982, §3.5, Table 6) found Muon-pretrained + Muon-SFT the best of 4 combinations on the Tülu 3 mix at
lr 5e-5; our 10-07 sweep (§37.5) had Muon 5e-5 still falling at the end and 1.2e-4 overfitting in epoch 2.

**Run** (started 15:51:33 IST): step-0 val **1.175723** (below 1.2301 at theta 10000 -> theta 50000 confirmed on the real data),
**1.88 s per step, 43.5k tokens/s, 52% MFU**, 70% of positions are trained tokens, `nsrej 0/0` and `nsratio` 0.98-1.02 (Muon's
Newton-Schulz health), grad norm 0.13-0.35. Val: 1.175723 (0) -> 1.048921 (200) -> 0.992842 (400) -> 0.969231 (600) -> ...
-> 0.856308 (8000) -> 0.856344 (8200) -> 0.855941 (8400) -> **0.854744 (9000)**. Checkpoints are 17 GB each (bf16 weights 3.4 GB
+ Muon fp32 master and momentum 12.7 GB + AdamW state 1.4 GB). Every save goes into its own folder `blumodel_step_<N>`; after a
save the trainer deletes all but the newest 2 (`prune_old_checkpoints`, `BLU_CKPT_KEEP=2`), so the end-of-epoch-1 checkpoint was
copied aside: **`keep_blumodel_step_6000`** (step 6000 = 205 steps after epoch 1 ends at step ~5,795; an exact step-5,795 save is
not possible in a running process). Watchdog `monitor_tulu3_sft.sh` (bc0f720d): read-only, one line every 5 minutes into
`monitor.txt` with a live ETA, ALERT lines for crash text, nan/inf, trainer gone, no step for 15 min, val up > 0.02, disk < 20 GB.

**Stop and resume.** 20:43:35 IST stopped at step 8723 to move to 2 GPUs (the code allows it: DCP re-splits Muon's state for a
different GPU count, `ShardedMuon.h:315-321`; with B 2 x W 2 every step reads the same rows, `SftDataLoader.h:177`; untested
for Muon) -- the other GPUs were taken, so it was **resumed on GPU 0 at 20:48:49** with the exact settings
(`run_tulu3_sft_resume1.sh`, 081b6d2c; log `run_resume1.log`, status `status_resume1.txt`). Resume check: `[resume] loaded DCP
checkpoint at step 8000 (loss 0.694324)`; **step 8000 identical to the first run** (train 0.653908, grad norm 0.1476, val
0.856308); steps 8001-8004 within 1.5e-4 (the known backward non-determinism). **Stopped at 21:30:37 IST at step 9194**, after
the step-9000 checkpoint was complete (`metadata.bin` present, 17 GB). Tomorrow: resume from 9000 (2,590 steps, ~1.5 h).
Loss curves = `run.log` steps 0-7999 + `run_resume1.log` from 8000. Note: the resumed process rewrote `sft_muon.csv`, which now
holds steps 8000-9194 only; steps 0-8723 are in `run.log`'s step lines.

**Saved.** Laptop `~/Downloads/box_backup_2026_10_22/` (43 MB): every log, CSV and status file, the watchdog log, GPU CSVs, the
run's exact trainer binary (`bluscript_sft_bin.used`), `stagec/`, `rope_opt/` (benchmark logs + ncu report + sources), all
scripts and build logs, `env_versions/`, and `BluTrain_sft_box_only/sft_adamw.csv`. Checked: the box's `BluTrain_sft` source is
identical to the laptop master (rsync checksum dry run; the 2 test files there are md5-equal to the laptop copies). The three
checkpoints (6000, 8000, 9000) stay on the box disk (`/dev/vda1`, 290 GB), which the user will attach to a new instance tomorrow;
if that disk is not kept, they must be uploaded first (the laptop link was 5.4 MB/s; a partial 4.9 GB copy of step 9000 on the
laptop is incomplete). Not copied (reproducible): step 8000, `sft_data` (same script + revision + seed), `fullchess` (HF),
the environments, the build trees, `tmp` (Open MPI session files).

### 45.68 2026-10-22: optimizing Kruthik's bf16 fused RMSNorm + RoPE -- measured, profiled, compared with every library

All files in `profiling_reports/rope_bf16_port_2026_10_22/`, box copy `~/gautam_k/rope_opt` (built standalone: the benchmarks
compile the kernel file directly, no libtensor, so they never touch a library a run is using). **Timing caveat:** the first
runs were on GPU 2, which ran everything ~10-14% slower than GPU 0 (originals too); the GPU-0 re-runs at the end are the clean
numbers, and from now on timing runs go on GPU 0 only. All numbers B 4, T 4096, 18/6 heads, hd 128.

**1. Kernel-only benchmark** (`bench_rope_bf16.cu`, a1ccc5b7; each kernel separately, 2 s warm-up, no-positions and positions
alternated, median of 9 rounds of 200 launches): GPU 2 forward 78.4 us (81.0 with positions), backward 120.9 (**210.9**), dgamma
reduce 58.4; forward and backward at 53-56% of the 4.8 TB/s peak. The positions cost is all in the backward (+74.5%).

**2. ncu, all 12 kernels** (`rope_bf16_ncu_1022.ncu-rep`, `--set full --import-source yes`, built `-lineinfo`), per-instruction
stall samples (SASS order):
- forward: the 4 tokens of a warp run one after another -- token k+1's loads are issued only after token k's store (each
  `PRMT` that waits on a load: 9.0 / 10.9 / 12.9 / 17.9% of stalls); 51% of stalls wait on loads.
- backward: one warp per block walks 128 tokens; each token = 13 small loads (2-byte `LDG.E.U16`) then one wait (`IMAD.U32`
  bf16->fp32 50.6%); the grid fills 0.7 waves; 65% of stalls wait on loads.
- positions: `IMAD.WIDE pos*128` waits 23.1% (the position) and the cos/sin consumers 40.3% (the row it then loads) -> two
  chained waits per token; 81% of stalls wait on loads, DRAM use drops 52% -> 30%.
- reduce: 1 block of 128 threads, 0.0 waves, DRAM 0.3%.

**3. How the others do it** (source read in `~/Downloads/sft_research/`, commits in `kernel_diff_sft_pretraining.md` §7 + vllm
f424213, TensorRT-LLM 4d2c821, sglang afd0d7b, unsloth 8a7d528, apex 575968b, quack 77a750c): vLLM / TensorRT-LLM / SGLang have a
fused QK-norm + RoPE, **forward only**, in place on a fused QKV tensor with `position_ids`; vLLM's advanced version handles
several heads per warp with cos/sin in shared memory via `cp.async` (`fused_qknorm_rope_kernel.cu:320-437`). Unsloth's fused
Triton kernel has a backward but **no gamma gradient** (falls back to separate kernels when the norm weights are trainable,
`qk_norm_rope.py:416-438`). Liger, FlashAttention (Triton), TransformerEngine, Quack: separate RMSNorm and RoPE, dgamma from
partial rows reduced by a full-GPU sum or a finalize kernel. **Nobody else fuses QK-norm + RoPE with the dgamma backward.**

**4. Round 1** (`opt/FusedRMSRoPE_bf16_opt.cu`, 3eb65982; `opt/bench_rope_opt.cu`, 182a0a3b, checks all 10 outputs bit for bit
against the original before timing): F1 forward = all 4 tokens' loads first; B1 backward = the next token's loads one step
ahead; B2 positions = lane l loads the positions of 32 tokens at once, each token takes its own from a shuffle; R1 reduce = q
and k in one launch, one warp per 32 columns, the same addition order, 32 rows of loads in flight. **Bit-identical at B 1/2/4/8,
with and without positions.** One rule found: the forward rotation must be written exactly as the original compiles it,
`__fmaf_rn(a, own, lo ? -__fmul_rn(b, partner) : __fmul_rn(b, partner))`; left to the compiler, the second half was paired the
other way and 85 of 9.4M outputs came out 1 bf16 step off.

**5. Round 2** (`opt/FusedRMSRoPE_bf16_v2.cu`, 23f15e7d; `opt/bench_rope_v2.cu`, 342cd231; every variant checked bit for bit):
fwd2 = one warp per token reading cos/sin once for 2-24 heads; bwd2 = one block per group of 8-24 heads with cos/sin in shared
memory in stages of 8-16 tokens; reduce3 = 32-128 rows in flight. Idea: ncu showed the L2 at ~70% busy (cos/sin rows re-read for
every head, ~200 MB per call). **All variants bit-identical, but fwd2 0.56-0.74x and bwd2 0.62-0.79x of the original's speed
without positions -> the L2 theory is refuted** (cutting cos/sin traffic 24x made it slower; L2 activity was a symptom). Both
forward attempts made each warp do more work one after another and lost to the original's many small warps. The real limit of
the forward and the no-positions backward is still unproven.

**6. GPU-0 clean results** (`box_logs/bench_opt_h200_1022_gpu0.log`, `bench_v2_h200_1022_gpu0.log`), us at B 4:

| kernel | original (Kruthik) | round 1 | round 2 best | kept |
| --- | --- | --- | --- | --- |
| forward | 68.9 (70.5 positions) | 94.3 | 95.8 (HPW 6) | original |
| backward, no positions | 110.1 | 111.8 | 142.2 | original |
| backward, positions | 194.9 | **118.3** (B2) | 141.5 | round 1 |
| dgamma reduce | 48.2 | 36.0 | **30.1** (UNROLL 128) | round 2 |

**Best bit-identical combination: pretraining 227.2 -> 209.1 us (1.09x), SFT 313.5 -> 218.9 us (1.43x)** at B 4; the same
ranking at B 1, 2 and 8. Not ported yet.

**7. Rival benchmark, ready, not run** (`rivals/rival_bench.py` 53b063d1, `rivals/rope_capi.cu` d2f42d4d = C entry points so
Python calls our kernel through ctypes, `rivals/vendor/` = FlashAttention's Triton RMSNorm + RoPE files from commit 94e22c9):
ours, HF eager, torch.compile, Liger, FlashAttention, Quack RMSNorm + FlashAttention RoPE; pretraining and packed-SFT cases; B
1/2/4/8; 18/6 and 32/8 heads; forward and backward timed separately (CUDA events + summed kernel time from torch.profiler);
outputs, input gradients and gamma gradients checked against fp64 first. Box venv `~/gautam_k/rivalenv` (torch
2.14.1+cu132, triton 3.8.0, liger 0.8.5, quack 0.6.5, cutlass-dsl 4.8.1). The first run failed in every Triton rival:
`Python.h: No such file or directory` -> needs `apt-get install python3.12-dev` (and `pip install numpy`); run with `python -u`.

### 45.69 2026-10-22 end of day: what is where, and what is next

- **SFT run:** stopped at step 9194; resume tomorrow from `blumodel_step_9000` with `run_tulu3_sft_resume1.sh`'s settings (a new
  log name, since that script refuses an existing `run_resume1.log`); then hand the final checkpoint to the DPO team, and compare
  `keep_blumodel_step_6000` (epoch 1) with the final one (val + dev suite, see `overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md`).
- **RoPE:** run the rival benchmark on GPU 0 (after `python3.12-dev`), profile the fastest rival against ours side by side, then
  port the proven combination (original forward + round-1 backward with positions + round-2 reduce) into the SFT library.
- **Docs written today:** `kernel_diff_sft_pretraining.md` (what changes between pretraining and SFT, kernel and non-kernel, and
  what 10 libraries change), `overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md` (train vs val, dev suite vs unseen suite).
- **Noted for later:** the loss kernel has a B300-only forward (`loss/arch/CrossEntropyKernels_sm103.cu`, routed at
  `LossKernels.cu:1093-1100`) that does not support `ignore_index` yet; the SFT tree's Muon sum-of-squares still has the old
  float `atomicAdd` (MuonKernels.cu md5 f6d821c2 = before the 10-05 V1n fix); the backward is not bit-reproducible run to run.

### 45.70 2026-10-23 morning: the boxes, the old disk, the checkpoints to Hugging Face

- The new **4x H200 box** `root@95.133.252.52` (`short-data-lies-fin-03`) came up with a **fresh disk**: no `~/gautam_k`. The
  old box's IP `86.38.182.101` now answers with a different host key (another machine).
- The **old disk** was attached to a **1x RTX PRO 6000 Blackwell box** `root@95.133.252.123` (`clean-name-flourishes-fin-03`):
  `/root/gautam_k` was intact (checkpoints 9000 and `keep_blumodel_step_6000`, 8000, all logs; the log ends at step 9194).
- **Hugging Face bucket** `hf://buckets/gautam1926/checkpoints_sft` (public, made by me): checkpoints 9000, 6000 and later
  11590 uploaded with `hf sync` (prepenv hf 2.2.0). The `hf` login on that shared root account was a teammate's
  (`MADHU_h200`); our uploads used my own token passed to the command only. Token file: laptop `~/.config/blubridge/hf_token`
  (chmod 600, account gautam1926; value never written into a doc).
- Checkpoint md5 (laptop `~/Downloads/box_backup_2026_10_22/checkpoint_md5_1023.txt`): 9000 `404c2809…` / `c14e3418…`,
  6000 `548c8a2b…` / `80aba476…`, 11590 `171a164e…` / `0f9cdf52…` (`data_rank_0.bin` / `metadata.bin`).
- Everything except the checkpoints was copied **1x → 4x** (rsync, 14.32 GB, 37,297 files + 9 links); the content hash of all
  files was identical on both boxes (`fe9545e8`). Step 9000 was downloaded from the bucket to the 4x box and its md5 matched.
- Plain download links work for a public bucket:
  `https://huggingface.co/buckets/gautam1926/checkpoints_sft/resolve/<checkpoint>/<file>` (tested: metadata.bin md5 matched).

### 45.71 2026-10-23: the Tülu 3 SFT run finished on the 4x box (GPU 3)

- Script `run_tulu3_sft_resume_4x.sh` (`profiling_reports/rope_bf16_port_2026_10_22/`): the same settings as 10-22 and the same
  trainer binary (md5 `2e98b56b`); two changes that do not touch the math: **CUDA 13.0 libraries first** in `LD_LIBRARY_PATH`
  (the box's `/usr/local/cuda` points to 13.4, installed by a teammate that morning; proven with ldd: all 5 CUDA libraries then
  load from 13.0) and **`BLU_CKPT_KEEP=1`** (the shared disk was filling). New log names: `run_resume_4x.log`,
  `status_resume_4x.txt`; watchdog `monitor_4x.txt` (now also alerts on a failed checkpoint save and on < 25 GB disk).
- Started **13:15:02 IST** (with my OK on GPU 3). **Step 9000 matched 10-22 exactly** (train 0.567569, grad norm 0.1390, val
  0.854744); steps 9001+ differed by 4e-5 to 9e-5, less than 10-22's own same-box resume (1.3e-4 at steps 8001–8005), i.e. the
  known run-to-run non-determinism of the backward, not the box.
- Checkpoints 10000 and 11000 saved and pruned as planned. **Finished 14:42:58 IST, exit 0, 0 watchdog ALERTs.** Last step
  11589: val **0.850100**. Final checkpoint `blumodel_step_11590` (md5 above), uploaded to the bucket.
- Speed (log summary): steady 44,021 tokens/s, MFU 52.47%, 2,585 steps in 80.2 min; memory peak 74,064 MiB.

### 45.72 2026-10-23: reading the finished run (loss curves, overfitting, memorization)

- Files: `post_training_research/tulu3_sft_1022/` (`analyze_run.py`, `tulu3_sft_1022_full_run.log` = the three logs joined
  as they are, `trajectory.csv`, `loss_train_val.png`, `loss_val.png`, `memorization_check.png`, `memorization_windows.csv`).
  The three log parts join exactly (val and train identical at steps 8000 and 9000).
- Val 1.1757 (base) → 0.8608 (end of epoch 1) → **0.8501** (end, the lowest). **No overfitting.** Epoch 2 real learning
  0.0107 nats; **memorization measured directly at 0.06–0.07 nats** (epoch-2 step 5795+n shares 16 of 20 rows with epoch-1
  step n, no shuffle; same rows dropped 0.120 vs val 0.056 on average); at the end memorization 0.057 vs real learning 0.012.
- Every doubt and answer of that discussion (memorization vs overfitting, nats, "4 of 5", y − x, the gap, the step-to-step
  jumps with the 0.804 correlation proof) is in `overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md`
  sections 11–21 (I renamed that file on 10-23), each with a Context paragraph.

### 45.73 2026-10-23: evaluation of checkpoints 6000 and 11590 (dev suite, unseen suite)

- **Export to Hugging Face format** (`post_training_research/tulu3_sft_1022/eval/export_to_hf.py`): our model is exactly
  Qwen3ForCausalLM (Llama-style + per-head QK RMSNorm before RoPE, NeoX RoPE theta 50000, tied embedding, vocab 50304); only
  the QK-norm eps differs (ours 1e-6, Qwen3 config 1e-5: effect 0.00001 on the val loss). Our Linear stores W as [in, out]
  (transposed on export); `parameters()` lists a module's own parameters first (per layer: q_gamma, k_gamma, norm, wQ, wK,
  wV, wO, MLP norm, gate_up, down) — a wrong order assumption was caught by a local metadata test before any GPU time.
- **Proof** (`verify_export.py`: the 2,000 val chats one by one through the HF model): 11590 → 0.850077 vs trainer 0.850100;
  6000 → **0.859864 vs 0.859895** (same weights). Padded ids ≥ 50257 never above 0.12% probability.
- The 4x disk was 98% full (teammates), so exports and caches went to RAM (`/dev/shm`, 335 GB). Harness: lm-evaluation-harness
  0.4.13 in `rivalenv` (transformers 5.20.2). A `max_new_tokens: 2048` in my exported `generation_config.json` overrode the
  benchmarks' own limits (found in the smoke test, fixed).
- **Dev suite** (chat template, same settings for both; paired per-question test, `paired_compare.py`):

| Task | 6000 | 11590 | Change | Paired z |
| --- | --- | --- | --- | --- |
| IFEval prompt-strict | 0.357 | 0.355 | −0.002 | −0.1 |
| GSM8K (8-shot CoT) | 0.169 | 0.184 | +0.015 | +1.7 |
| TruthfulQA MC2 | 0.401 | 0.392 | **−0.009** | **−4.1** |
| HellaSwag acc_norm | 0.500 | 0.511 | **+0.011** | **+6.0** |
| ARC easy / challenge | 0.581 / 0.304 | 0.578 / 0.314 | −0.003 / +0.010 | −0.8 / +1.6 |
| WinoGrande | 0.518 | 0.523 | +0.006 | +0.5 |
| End-token rate (200 val prompts) | 0.88 | 0.90 | | |

  DROP in chat format is invalid (its generation stops at the first "."; full-sentence answers are cut: F1 0.046), and
  multiple-choice tasks in chat format are depressed (WinoGrande ≈ chance); the plain-format pass and the base-model
  evaluation were stopped by me as not needed. About 10% of greedy answers loop until the length limit (median answer length
  257 = real 256).
- **Chosen: 11590** (lower val, HellaSwag +1.1, the rest equal; TruthfulQA −0.9 fits the measured memorization).
- **Unseen suite on 11590:** AGIEval-English 0.183 (chance level); MMLU-Pro on a random subset of 150 questions per subject
  (2,100, seed 1023; the full 12,032 projected ~5 h) **0.0971 ± 0.0065** (chance 0.10; answers not written as "The answer is
  (X)" also count as wrong, not checked). Both at chance for a 1.7B model.
- All 79 result files: `post_training_research/tulu3_sft_1022/eval/box_results_1023/`; the box copies were deleted.

### 45.74 2026-10-23: handover to the DPO team

- `post_training_research/tulu3_sft_1022/handover_dpo_tulu3_sft_1022.zip` (1.1 MB): README (download, checksums, how to load,
  model, tokenizer and chat format, data, settings, results), the joined log, `trajectory.csv`, loss charts,
  `run_tulu3_sft.sh`, `prepare_sft_data.py`, `bluscript_sft.cpp`.
- Must-know for loading: `BLU_INIT_FROM=<dir> BLU_N_LAYERS=28 BLU_ROPE_THETA=50000` (a wrong theta gives no error, only a
  worse model).
- Download command length: Navin's model is a Hugging Face **model repo** (`hf download navingv/curriculum_8btokens`); ours is in a
  **bucket** (`hf sync hf://buckets/…/<checkpoint> <dir>` or the curl links). A model repo `gautam1926/tulu3-sft-1.7b` would make
  it `hf download gautam1926/tulu3-sft-1.7b` (offered, not done).

### 45.75 2026-10-23: the colleague's forward-attention EOD ("improving the fwd kernel", Uma Mahesh)

- His 8 changes were **stacked** (each % is the gain over the previous row; recomputed from his cycle table, matches within
  0.1–0.3 points except "baton" at 8×12(4): −0.21% computed vs −0.5% written). Single runs, hand-edited PTX/SASS for fixed
  shapes, cycles not time, no correctness or packed test reported.
- "Vg1" = our QPU forward (pushed 10-21 on `dist_side_gautam_lottn`). All 8 are already in it or were tested by us: uniform Q
  descriptor (G2 10-13; L1b rejected 10-19), raw max + FA3 bias (no −inf guard, `ex2(fma(S, scale2, −m·scale2))`, SM3),
  sections + longest-first (G5) + no consumer divisions (QPU), persistent + dynamic scheduler + Q-free barrier + K/V before Q
  (PF) + O in the last K slot (K4), per-warpgroup 64-row Q/O, continuous baton, diagonal peeled out of the loop (K1, SM3m).
  Not tried: `elect.sync` in the producer; one epilogue barrier with a register id. Vg1 is already 7–13.6% faster than his
  starting kernel; his prototype is 1.0–1.9% ahead of Vg1, except 2×18(9) (−7.0%: worth a look on our side).

### 45.76 2026-10-23: P032 node (old 8x node, being replaced) cleaned

- Our `/localstorage/gautam_k` (12 GB) checked file by file against the laptop (name + size): 13,742 of 14,429 files already
  on the laptop; the rest was public (git history, FlashAttention source, the HF checkpoint `navingv/curriculum_8btokens`),
  rebuildable (build files, test binaries, envs) or older code copies. Kept on the laptop
  `~/Downloads/p032_backup_2026_10_23/` (50.6 MB): `compute_dpo_reference_logprobs.cpp` (untracked, author unknown), the kernel
  snapshots `v2_work/snapshots/*`, logs and results. Then `rm -rf /localstorage/gautam_k` (run by me); colleagues' files
  (`kk/…goutham…`) and shared `/dev/shm` segments not touched.

### 45.77 2026-10-23: hyperparameters (B, rows per step, GRAD_ACCUM, ranks, GPU memory)

- New doc `HYPERPARAMETERS_CONFIGS_SELECTION_FOR_MODEL_TRAINING_AND_FACTORS_AFFECTING_THESE.md`: parameters vs
  hyperparameters vs optimizer state; why B = 4 × 5 = 20 rows ≈ 128 chats (Tülu 3's batch); rows per step = B × ranks ×
  GRAD_ACCUM (the trainer refuses anything else, `bluscript_sft.cpp:1007-1009`); memory = fixed ≈ 19.5 GiB + B × 13.2 GiB
  per 4,096-token row, so **B = 8 fits one H200 by calculation (≈125 of 140 GiB), not tested**; the order to choose them.

### 45.78 2026-10-23: Kruthik's bf16 fused RMSNorm + RoPE — port, proof, rivals, ncu

- **Port into the SFT library** (`~/Downloads/BluTrain_sft/Tensor-Implementations/src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu`,
  md5 `6ec34cc9`): backward **with positions at hd 128** = the pipelined kernel `fused_qknorm_rope_bwd_pf_bf16_tm` (positions
  32 at a time via a shuffle + next token's loads issued early); dgamma reduce = one launch for q and k, 128 rows in flight
  (`reduce_dgamma_qk_bf16_tm<128>`); forward and the no-positions backward = Kruthik's code unchanged.
- The 10-22 bit-identity benches were built **without `--use_fast_math`** (the library uses it), so the port was re-proven with
  the library's flags: **SASS gate 17/17** (all pretraining kernels unchanged; the new kernels equal the measured ones;
  `port_1023/port_sass_compare.py`) and the **GPU port test** (`port_1023/test_rope_port.cu`, all 8 outputs bit for bit).
  First run: hd 128 all identical, but **hd 64 with positions differed** (a few hundred dq/dk values, ~29 of 64 dgamma columns:
  ptxas pairs the multiply-adds of the pipelined loop differently at hd 64). Fix: hd 64 with positions keeps the original
  kernel. Second run: **36 of 36 cases bit-identical.**
- Speed (GPU 3, production shape hd 128, 18/6 heads, T 4096; forward + backward + reduce):

| B | Pretraining | SFT (packed positions) |
| --- | --- | --- |
| 1 | 122.1 → 114.9 µs (1.06x) | 208.4 → 120.1 µs (1.73x) |
| 2 | 159.1 → 146.7 µs (1.08x) | 247.6 → 153.4 µs (1.61x) |
| 4 | 237.2 → 213.6 µs (1.11x) | 322.7 → 223.7 µs (1.44x) |
| 8 | 533.8 → 468.9 µs (1.14x) | 684.2 → 488.4 µs (1.40x) |

- **Rival benchmark** (`rivals/rival_bench.py`, ours = the ported kernel built with the library flags; HF eager, torch.compile,
  Liger, FlashAttention Triton, Quack + FA RoPE; 18×6 and 32×8 heads, B 1/2/4/8, pretraining and packed SFT):
  - **forward: ours is the fastest in 15 of 16 cases** (1.02x–1.40x faster than the best rival, always torch.compile);
    the 16th (32×8, SFT, B4) is a tie by wall time (compile 112.5 vs ours 112.7 µs): there the kernel-time sum caught only
    1 of compile's 2 forward kernels (55.8 µs, below its own B2 time), so its kernel time is not valid;
  - **accuracy: ours is the most exact** (error 1.00 = bf16 rounding of the exact result; dgamma 1e-6); compile 1.5–2.7,
    eager and Liger ≈ 3.3–3.8, FlashAttention / Quack ≈ 2.0–2.2;
  - **backward: two rivals are faster than ours, torch.compile and Quack + FlashAttention RoPE, each in 14 of 16 cases**
    (FlashAttention alone in 2, at B1). Backward kernel time in µs from `rival_results_port_4x.json` (ours / compile / Quack):

| Heads | Case | B1 | B2 | B4 | B8 |
| --- | --- | --- | --- | --- | --- |
| 18×6 | pretraining | 92.3 / 51.7 / 54.9 | 107.8 / 88.3 / 96.5 | **141.8** / slower / slower | 332.0 / 298.3 / 278.0 |
| 18×6 | SFT | 97.8 / 52.6 / 53.9 | 113.6 / 88.3 / 87.3 | 152.1 / 149.6 / slower | 348.5 / 298.0 / 277.6 |
| 32×8 | pretraining | 103.5 / 72.7 / 76.3 | 131.1 / slower / 130.6 | 313.8 / 243.0 / 236.6 | 555.9 / 462.1 / 446.1 |
| 32×8 | SFT | 109.2 / 72.6 / 76.4 | 140.2 / 128.0 / 133.0 | 327.1 / 242.5 / 235.7 | 574.6 / 464.7 / 451.0 |

    Ours is the fastest backward only at 18×6 B4 pretraining. At our SFT training size (18×6, B4, packed) compile is 1.6%
    faster (149.6 vs 152.1 µs). Quack's backward is less exact than ours (error ≈ 2.0–2.2 vs 1.00).
  - A harness bug was found and fixed: Liger's RoPE backward rotates its incoming gradient in place, and the shared gradient
    tensor corrupted the accuracy check of the rivals after it (FlashAttention/Quack showed dx error 1307; each piece alone
    was exact, `debug_fa_bwd.py`); every rival now gets its own copy.
- **ncu, one backward call, ours vs torch.compile** (`rivals/ncu_bwd.py`, `run_ncu_bwd.sh`, reports `ncu_bwd_*.ncu-repz`):
  ours runs **one warp per block, each warp walking 128 tokens one after another**: occupancy **9% (B1) / 36% (B4)**, DRAM
  throughput **18% / 57%**. torch.compile splits the backward into a RoPE-backward kernel with 36,864–65,536 blocks of 128
  threads (occupancy 87–90%) and a per-row norm-backward kernel, reaching **72–79% of DRAM bandwidth**. Cause proven: too little
  parallel work in our backward. (Quack was not profiled with ncu; its kernels are the next to read.)
- **Open decision (next session):** more parallel work = fewer tokens per warp; dq/dk would stay bit-identical, but dgamma's
  float sum would be added in a different grouping (last-bit changes, still deterministic). Detailed analysis first, then the
  choice (allow the dgamma rounding change, or keep strict bit-identity).

### 45.79 2026-10-23 end of day: what is where, and what is next

- **Checkpoints:** bucket `gautam1926/checkpoints_sft` (6000, 9000, 11590, public); 4x box `~/gautam_k/runs/tulu3_sft_1022/checkpoints/blumodel_step_11590`.
- **Laptop backups:** `~/Downloads/box_backup_2026_10_23/`: `rope_opt/` (all port, rival and ncu results; 80 files, md5
  identical to the box), `gautam_k_logs/` (the 4x run script, the watchdog script, the 4 Hugging Face transfer logs),
  `sft_data_small/` (the val split and the small train index files) and `sft_data_tulu3_noaya_md5.txt` (md5 of all 12 data
  files). The 3 big train files (`inputs.bin`, `labels.bin`, `meta.json`, 1.98 GB) were **not** copied: the laptop link ran at
  about 100 KB/s (over 5 hours); they can be rebuilt with `prepare_sft_data.py` and checked against the saved md5. The run logs
  were already in `~/Downloads/box_backup_2026_10_22/tulu3_run_final/` (same sizes as the box). Also
  `post_training_research/tulu3_sft_1022/` (analysis, eval code, eval results, handover zip) and
  `~/Downloads/p032_backup_2026_10_23/`.
- **Next:** the backward of the RoPE kernel (detailed analysis of torch.compile's design vs ours, then the bit-identity decision);
  rebuild the library on a box with the port and check the trainer (step-0 loss identical, step time); commit/push of the SFT
  tree (my call); results to the DPO colleague; the 2×18(9) forward-attention gap from Uma's table.

### 45.80 2026-10-24: why our RoPE backward loses to torch.compile and Quack (analysis on the laptop, no GPU run)

**Goal (agreed 10-23):** a detailed analysis before any edit of the backward. Inputs, all on the laptop:
- the 6 ncu reports of 10-23 (one backward call, ours vs torch.compile, 18×6 B1, 18×6 B4, 32×8 B4, **pretraining case**, no
  positions). They are zstd-compressed (`.ncu-repz`); decompressed copies import fine with ncu 2025.3.1;
- the SASS with per-instruction stall samples, taken from those reports;
- the SASS of the 10-22 bench binary (`bench_rope_opt`, the depth-1 prefetch "B1") and of the port library
  (`libropeours_port.so`, the pipelined positions kernel), via `cuobjdump -sass`;
- Quack's RMSNorm backward source (`quack/rmsnorm.py`, `quack/rmsnorm_config.py`, GitHub main, read only) and
  FlashAttention's vendored Triton RMSNorm backward (`flash_attn/ops/triton/layer_norm.py`).
Scripts and outputs: `profiling_reports/rope_bf16_port_2026_10_22/rivals/analysis_1024/`.
(The 10-23 `summarize_ncu.py` printed times and MB too small: the CSV is already in µs and MB. `summ2.py` reads the units row.)

#### 45.80.1 Per kernel, under ncu (ncu fixes the clock and flushes caches, so the times are higher than in a normal run; compare ratios)

| Case | Ours: main kernel | Ours: dgamma reduce | Ours total | compile: RoPE-bwd + norm-bwd kernels | compile: dgamma sum + cast | compile total |
| --- | --- | --- | --- | --- | --- | --- |
| 18×6 B1 | 82.0 µs (72.5 MB, DRAM 18%) | 17.7 µs | 99.7 | 46.7 µs (k: 3 kernels, q: 2) | 15.0 µs | 61.7 |
| 18×6 B4 | **108.1 µs (298 MB, DRAM 57%)** | **64.3 µs** | 172.4 | **135.2 µs (466 MB, DRAM 57–79%)** | **26.8 µs** | 162.0 |
| 32×8 B4 | **206.4 µs (504 MB, DRAM 51%)** | **112.6 µs** | 319.0 | **226.8 µs (806 MB)** | **29.4 µs** | 256.1 |

- **At B4 our main kernel is already faster than compile's kernels** (108 vs 135 µs, 206 vs 227 µs) and moves less data (298
  vs 466 MB): ours reads dz and x and writes dx once; compile writes the un-rotated gradient to memory and reads it back.
- **At B4 we lose only because of the dgamma reduce:** 64.3 vs 26.8 µs and 112.6 vs 29.4 µs. Ours runs 8 blocks of 32 threads
  on the whole GPU (occupancy 1.6%, 25 GB/s); compile's sum (`at::reduce_kernel`) is a tree over 24–64 blocks.
- **At B1 our main kernel also loses** (82.0 vs 46.7 µs): there are only 768 warps for 132 SMs.

#### 45.80.2 Our main kernel: one token in flight per warp

- Launch: 1 warp per block, one warp per (batch, head, chunk), each warp walks 128 tokens (t0, t0+32, …). H200 holds at most
  32 blocks per SM, so 1-warp blocks cap at 32 warps per SM = **50% theoretical occupancy**; the grid is 768 × B warps =
  **0.18 waves at B1, 0.73 at B4** (achieved occupancy 9% and 36%).
- SASS of the loop (97 instructions per token): the token's 13 loads at the top (8 × 2-byte bf16, 5 × 4-byte fp32), then the
  first bf16→fp32 convert (`IMAD.U32 R30, R29, 0x10000`) waits for them. **56% of all stall samples sit on that one
  instruction**, 15% on the next convert. So each warp waits a full memory round trip per token, 128 times in a row.
- Proof from the times: 4 times the data costs only 1.3 times the time (**82 µs at B1, 108 µs at B4**): the time is set by
  128 round trips per warp, not by bandwidth.
- Instruction mix (q + k, B4): 13.4 M global memory instructions (6.3 M 2-byte loads, 4.0 M 4-byte loads, 3.1 M 2-byte stores).
  compile issues about 2.2 M for the same work (16-byte loads and stores). Not the limit now (≈13 µs of LSU time at B4), but
  it is why compile needs far fewer warps for the same bandwidth.

#### 45.80.3 Why the 10-22 prefetch ("B1", 84.3 → 84.3 µs at B1) and the port's pipelined kernel did not overlap anything

- Both issue the next token's loads before computing the current token, as designed. But in both SASS (10-22 bench binary,
  `sass_opt_B1_bwd_nopos_1022.txt`; port library, `sass_port_pf_bwd_1023.txt` lines with `LDG.E.U16` followed directly by
  `SHF.L.U32` / `IMAD.U32 …, 0x10000`), **the bf16→fp32 converts of the prefetched values come right after their loads**,
  inside the block guarded by "is there a next token" (`@P0 BRA`). A convert needs the loaded value, so the warp waits for
  the next token's data before it computes the current one: still one token in flight.
- Cause in the source: the prefetch struct holds **floats** (`BwdTokBf16::dza[]` etc. are `float`), so the convert is part of
  the load step, and the load step is inside the `if (next token exists)` branch.
- The pipelined positions kernel still gained 1.65–1.90x on 10-22 because it removed the second wait (position → cos/sin
  row), not because of the prefetch.

#### 45.80.4 How the rivals do it

- **torch.compile (Inductor), per tensor (q, then k):** (1) a pointwise RoPE-backward kernel, 128-thread blocks, 8 elements per
  thread, 16-byte loads (`LDG.E.128`), 5.8–23 waves, occupancy 80–90%; (2) a per-row RMSNorm-backward kernel, 1 warp per
  block, `cp.async` 16-byte loads into shared memory (`LDGSTS.E.BYPASS.128`), 72 registers, low occupancy (17–40%) but
  DRAM 63–79%, it writes dx and one dgamma partial row per block; (3) `at::reduce_kernel`: tree sum of the partial rows;
  (4) a 1-block cast/copy. Its weak point: the un-rotated gradient goes to memory and back (extra 2 passes).
- **Quack (CuTe DSL) RMSNorm backward + FlashAttention's Triton RoPE backward:** persistent grid = SMs × 16 = 2,112 blocks
  of 128 threads at N = 128 (`_get_sm_count_hopper`), 16 threads per row (8 elements, 16-byte loads), 2-stage double buffer
  (cp.async / TMA), dw accumulated in registers across all rows a block walks, then reduced in shared memory, one dw row per
  block, final `dw_partial.sum(0)` over 2,112 rows. RoPE backward is a separate pass.
- **FlashAttention's Triton RMSNorm backward:** grid = 8 × SMs (1,056 programs), each walks M / 1,056 rows, dw in registers,
  `_dw.sum(0)` over 1,056 rows.
- **Common pattern:** many bytes in flight per warp (wide loads, double buffering) and a tree-shaped final dw sum. Ours has
  neither: 2-byte loads, one token in flight, and a sequential final sum on 8 warps.

#### 45.80.5 Options

| # | Change | dq / dk | dgamma | Expected effect (estimate, to be measured) |
| --- | --- | --- | --- | --- |
| 1 | **Real prefetch** in both backward kernels: keep the next token(s)' data as raw bf16 bits and convert them only when that token is computed; load the next token unconditionally (clamped index) so no branch holds the converts; depth 2–3 tokens | identical | **identical** (same per-token math, lane mapping and token order) | main kernel B1 84 → ~30–45 µs; B4 110 → ~80 µs (the DRAM floor: 298 MB at 75–80% of 4.8 TB/s) |
| 2 | **Pipelined reduce:** issue the next batch of rows' loads before adding the current batch; same sequential order per column | – | **identical** | B4 18×6: ~30 → ~5 µs (floor = 2,304 dependent adds ≈ 4.7 µs at 1.98 GHz, assuming 4 cycles per dependent add); B8 ≈ 10 µs |
| 3 | **Tree-shaped final dgamma sum** (like compile / Quack / FlashAttention) | – | last-bit changes, still the same on every run | reduce ~2 µs at any batch: only ~3 µs more than option 2 at B4, ~8 µs at B8 |
| 4 | Shorter token walks (more warps per head) | identical | last-bit changes | probably not needed after option 1; it also multiplies the partial rows, which makes a sequential reduce slower |
| 5 | 16-byte loads staged through shared memory (compile / Quack style), same compute mapping | identical (if the mapping is kept) | identical | only if option 1 does not reach the DRAM floor |

**Estimate at our SFT training size (18×6, B4, packed):** today ours 152.1 µs vs compile 149.6 µs (rival bench, kernel time);
with options 1 + 2 about 85–90 µs, i.e. about 1.7x faster than compile, **with every output still bit-identical**. These
are estimates from the ncu numbers, not measurements.

#### 45.80.6 Reading and next step

- The question of 10-23 ("allow a dgamma rounding change to get more parallel work?") is **not needed for the big gains**:
  options 1 and 2 keep every output bit-identical. The rounding change only buys option 3's last few µs.
- Next (needs a GPU): write options 1 + 2 into the laptop library, build, run the port test (36 cases, bit-identical vs
  Kruthik's original), time B1–B8 pretraining + SFT, rerun the rival bench, and confirm the overlap in ncu (the first-convert
  stall must drop).

### 45.81 2026-10-24: options 1 + 2 written (laptop library), checked by compiling only (no GPU yet)

**File:** laptop `~/Downloads/BluTrain_sft/Tensor-Implementations/src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu`, md5 **07953145**
(10-23 port: 6ec34cc9). Kruthik's original backward kernel is unchanged and still runs at hd 64.

**Option 1, new backward `fused_qknorm_rope_bwd_pp_bf16_tm<IS_NEOX, HD, HAS_POS, DEPTH>`**, used at hd 128 with and without
positions (replaces the 10-23 pipelined kernel `fused_qknorm_rope_bwd_pf_bf16_tm`, deleted):
- each warp keeps DEPTH = 3 tokens in flight (`BF16_BWD_DEPTH`) in a ring of register slots; a slot holds the loads as
  loaded (bf16 bits for dz and x, fp32 cos/sin and rstd); a token is converted only when it is computed; then its slot is
  refilled with the token DEPTH places later;
- positions: each slot also loads the position of the token it will hold next, so a refill's cos/sin address never waits.
  Two first attempts were fixed after reading the SASS: (a) a per-32-token batch of positions rotated by a shuffle made
  ptxas copy the fresh batch at the branch join (`MOV R56, R59` 25 instructions after the load: one stall per 32 tokens);
  (b) `m < n_tok ? load : 0` made ptxas load into a temporary and copy it into the slot (`MOV R15, R60` 10 instructions
  after the load, every round). Now the position load is unconditional with a clamped index and lands in the slot;
- the per-token math uses explicit-rounding intrinsics in the exact form the original compiles to at hd 128 (read from its
  SASS): `dy_a = fma(dza, cs, dzb*sn)`, `dy_b = fma(dzb, cs, -(dza*sn))`, row sum `0 + fma(g_a, xh_a, g_b*xh_b)` per pair
  then the xor tree, `dx = rstd * fma(-xh, S, g)`, `dgamma = fma(dy, xh, dgamma)`; same lane mapping and token order.

**Option 2, dgamma reduce `reduce_dgamma_qk_bf16_tm<HD, U, STAGES>`:** the same sequential sum per column in row order;
rows go through STAGES = 3 register buffers of U = 64 rows (`BF16_RED_STAGES`, `BF16_RED_U`): while one buffer is added,
the next two are loading. Fixed after reading the SASS: (a) with read-only (`.nc`) loads ptxas dropped a buffer's first
3 rows and loaded them again right before their adds -> `__ldcg` (L2) loads; (b) the per-row clamp `min(r0+u, n_rows-1)`
and 64-bit index math cost ~10 integer instructions per add -> HD is a template parameter (each load = column pointer +
constant offset) and the partial buffers carry `BF16_RED_PAD_ROWS` = 2·STAGES·U zeroed rows (`ensure_part`, zeroed with
`cudaMemsetAsync` on the kernels' stream), so no load needs a bound check; full buffers add without a per-row test.

**Checks on the laptop (nvcc 13.0, library flags, sm_90a, compile only):**

| Check | Original (Kruthik) | 10-23 pipelined kernel | New (all 4 variants: NeoX / interleaved × positions / none) |
| --- | --- | --- | --- |
| Loads per loop, distance load -> first use | 13 per token, 22–27 instructions | 14, **13–18** for the bf16 loads | 39–42 per 3 tokens, **196–305** instructions (~2 tokens of compute) |
| Per-token FP instructions (FFMA / FFMA with negated operand / FMUL / FMUL by 1/128 / FADD / FADD with zero / bf16 pack / shuffle) | 8 / 6 / 18 / 1 / 6 / 1 / 4 / 5, all .FTZ | – | **identical counts and forms**, all .FTZ |
| Registers, spills | 40, 0 | – | 72, 0 |
| Reduce: load -> first use | (load, wait, add per batch) | – | ≥ 723 instructions (≈ 2 buffers = 128 adds ahead), 244 registers, 0 spills |

Identical instruction forms are strong evidence of bit-identity, not proof: the proof is the GPU port test.

**GPU test package:** `profiling_reports/rope_bf16_port_2026_10_22/port_1024/` (`run_port_test.sh <Tensor-Implementations dir>
[-D flags]` copies the current library file, builds with the library flags, compares all 8 outputs against Kruthik's stage-B
file, then times both). 44 cases: the 36 of 10-23 plus T 40 (warps with fewer tokens than DEPTH) and 5/1, 7/1 heads (reduce
row counts that are not a multiple of 192, so the last partial buffer runs). Builds on the laptop.

**Next, on a GPU (~15–20 min):** (1) port test: 44/44 bit-identical expected, timing vs stage B; (2) variants by flag:
`-DBF16_BWD_DEPTH=2/4`, `-DBF16_RED_U=32 -DBF16_RED_STAGES=4`; (3) rival bench with the new library (`rivals/rope_capi.cu`);
(4) ncu of one backward (the first-convert stall must be gone, DRAM % up); (5) library rebuild + trainer step-0 check.

### 45.82 2026-10-24 ~16:00 IST: the 10-24 RoPE backward PASSES on H200 (P032, GPU 4), DEPTH 3 kept

Box: the old 8x node P032 (`innmi1srh2-p032`, 8x H200, nvcc 13.4), GPU 4 (all 8 GPUs idle, 0 MiB), for ~20 minutes.
Package: laptop `~/Downloads/rope_1024.tgz` (library file md5 07953145 + its headers + Kruthik's stage-B file + the
44-case test), unpacked to `/localstorage/gautam_k/rope_1024`. Logs: `profiling_reports/rope_bf16_port_2026_10_22/port_1024/
port_test_p032_gpu4.log` (DEPTH 3, the default), `..._depth2.log`, `..._depth4.log`.

**Correctness: 44 of 44 cases bit-identical in all 8 outputs, at DEPTH 2, 3 and 4** (so the pinned per-token math holds
whatever the unroll depth).

**Speed, backward + dgamma reduce, µs per call, vs Kruthik's original (same run, same GPU):**

| Case | Original | DEPTH 2 | **DEPTH 3 (kept)** | DEPTH 4 |
| --- | --- | --- | --- | --- |
| B1 none | 99.8 | 88.8 | **56.0 (1.78x)** | 88.8 |
| B1 positions | 182.6 | 89.4 | **90.2 (2.02x)** | 90.3 |
| B2 none | 120.9 | 100.5 | **68.2 (1.77x)** | 100.5 |
| B2 positions | 204.1 | 101.6 | **101.9 (2.00x)** | 101.8 |
| B4 none | 168.2 | 128.1 | **119.6 (1.41x)** | 204.5 |
| B4 positions | 249.0 | 130.4 | **131.2 (1.90x)** | 207.0 |
| B8 none | 395.6 | 275.1 | **250.7 (1.58x)** | 338.3 |
| B8 positions | 535.7 | 282.4 | **275.1 (1.95x)** | 337.6 |

Forward unchanged (1.00x). Totals (fwd + bwd + reduce) at DEPTH 3: SFT B1/B2/B4/B8 1.84x/1.74x/1.59x/1.63x, pretraining
1.58x/1.51x/1.26x/1.37x. Against the 10-23 port (other box, the original measured the same there: 237.2 vs 235.8 µs at B4):
B4 total SFT 223.7 -> 199.9 µs, pretraining 213.6 -> 187.2 µs.

**Why DEPTH 4 is slower at B4/B8 (registers, compiled locally):** DEPTH 2: 63–64 registers -> 32 one-warp blocks per SM,
B4 grid = 0.73 waves; DEPTH 3: 72 -> 28 blocks, 0.83 waves; **DEPTH 4: 89 (none) / 116 (positions) registers -> 23 / 17
blocks per SM -> 1.01 / 1.37 waves**: the grid no longer fits in one wave, so a second, almost empty wave runs.

**Open (next session):** (1) at B1/B2, DEPTH 3 without positions reaches 56.0 µs but DEPTH 3 with positions (90.2) and
DEPTH 2/4 (~89) do not: find out why in SASS / ncu; (2) the rival bench with this library (torch.compile backward at
18×6 B4 SFT was 149.6 µs on 10-23 with another harness; ours now 131.2 here; at B1 compile 52–53 µs vs ours 56.0 / 90.2);
(3) library rebuild + trainer step-0 check. GPU 4 on P032 released after the runs.

### 45.83 2026-10-24 afternoon: why B 1 with positions stays at ~90 µs (ncu on the new 1x H200 box, laptop SASS decode)

Box: new 1x H200 `root@86.38.238.224` (`kind-home-falls-fin-02`, driver 580.178.04, CUDA 13.0). Set up with
`profiling_reports/rope_bf16_port_2026_10_22/setup_1024/` (`setup_1_check.sh`, `setup_2_env.sh` = rivalenv with the 10-23
versions, `setup_3_build.sh`); ncu with `run_ncu_new.sh` (`test_rope_port --ncu`: one backward per case).
Report: `port_1024/ncu_bwd_new_1024.ncu-rep`; SASS exports `port_1024/sass_new_B1_{none,pos}.csv`; control-bit decoder
`rivals/analysis_1024/sass_ctrl_bits.py`.

**ncu, one backward + reduce (clock fixed by ncu, caches flushed):**

| | B1 none | B1 positions | B4 none | B4 positions |
| --- | --- | --- | --- | --- |
| Backward kernel | 56.2 µs, DRAM 25%, 68.6 MB | **91.7 µs, DRAM 15%, 67.9 MB** | 103.2 µs, DRAM 59.5% | 122.1 µs, DRAM 50% |
| dgamma reduce | 10.7 µs | 10.7 µs | 32.7 µs | 33.1 µs |

Same bytes at B1, 35 µs more time: the positions kernel waits, it does not work more.

**Stall samples (B1):** without positions 39% of all samples on ONE first-convert (line 258) and the other two slots'
first converts almost free; with positions the first convert of **every** slot stalls (21.8% + 21.5% + 19.9%).

**Cause, from the control bits of the SASS (decoded from the instruction words, `sass_ctrl_bits.py`):**
- ptxas tags **every** global load of every slot with the same scoreboard counter, **SB5** (`wb=5` on all 39 loads
  without positions, all 42 with positions).
- A wait on SB5 drains **all** outstanding SB5 loads, including a refill issued a few instructions earlier.
- With positions: the first convert of each slot waits on SB5 (line 421 `wait=[1,5]`, line 518 `wait=[1,3,5]`), right
  after the previous slot's refill → **three full drains per round of 3 tokens** ≈ one memory round trip per token.
- Without positions (DEPTH 3 only): **one** SB5 wait per round (line 258); the other two first converts wait on other
  counters (`wait=[0]`, `wait=[1,2]`) → three tokens share one round trip → 56 µs.
- So the 56 µs was a lucky schedule, not something the source guarantees; DEPTH 2 and 4 (88.8 µs) and every positions
  variant (~90 µs) got the per-slot drains.

**Fix options (decision next session):**
1. **cp.async ring in shared memory** (`LDGSTS` + `cp.async.commit_group` / `cp.async.wait_group<DEPTH-1>`): each wait
   then covers only the oldest token's group, by construction (compile's norm kernel uses `LDGSTS` + `LDGDEPBAR`);
   the per-token math, lane mapping and dgamma order stay the same (bit-identical by design); ~1 KB per token per warp
   (3 KB per warp at DEPTH 3).
2. **One convert point per round in the source** (convert all DEPTH slots at the start of the round, then compute and
   refill): reproduces the lucky "one drain per round" shape on purpose, but still relies on ptxas's scheduling.
Expected (estimate): B1 positions ~90 -> ~56 µs or better; B4 positions 131 -> ~120 µs.

### 45.84 2026-10-24 ~17:00 IST: rival benchmark with the 10-24 library (new 1x H200 box)

`run_rivals_new.sh` (same `rival_bench.py` and harness as 10-23; kernel = summed GPU time per call from torch.profiler).
Tables: `port_1024/rival_new_1024_tables_from_paste.txt` (and `rival_results_new_1024.json` on the box). Liger was
skipped this time (`No module named 'packaging'` in the new env; `pip install packaging` then `--only liger` adds it).

**Backward (kernel time, µs): ours vs the best rival of each case**

| Heads | Case | B1 | B2 | B4 | B8 |
| --- | --- | --- | --- | --- | --- |
| 18×6 | pretraining | 54.3 vs compile 51.8 (0.95x) | **66.5** vs 87.7 (1.32x) | **119.3** vs 148.4 (1.24x) | **247.8** vs quack 276.8 (1.12x) |
| 18×6 | SFT | 90.2 vs compile 51.3 (0.57x) | 101.3 vs quack 87.4 (0.86x) | **128.5** vs quack 150.4 (1.17x) | **271.2** vs quack 276.2 (1.02x) |
| 32×8 | pretraining | **65.1** vs compile 72.4 (1.11x) | **99.6** vs 128.0 (1.29x) | **215.3** vs quack 236.1 (1.10x) | **414.4** vs quack 447.0 (1.08x) |
| 32×8 | SFT | 99.9 vs compile 72.3 (0.72x) | **119.6** vs 127.4 (1.07x) | 256.6 vs quack 236.1 (0.92x) | **434.0** vs quack 451.4 (1.04x) |

- **Ours fastest backward in 11 of 16 cases (10-23: 2 of 16); fastest forward + backward total in 13 of 16** (18×6
  pretraining B1 a tie: 73.0 vs 73.2 µs).
- **Our SFT training size (18×6, B4, packed): backward 128.5 µs vs Quack 150.4 / torch.compile 150.9 (1.17x faster);
  total 197.3 vs compile 245.6 µs (1.24x).**
- Accuracy unchanged: ours err 1.00 / 1.00, dgamma 7e-7–1.2e-6; compile 1.5–1.7 / 2.55–2.66; Quack/FlashAttention ~2.0 / 2.1.
- The 5 losses are 4 SFT (positions) cases at B1/B2 and 32×8 B4, i.e. the per-slot scoreboard drains of §45.83, and
  18×6 pretraining B1 (−5%). Next: the cp.async ring (§45.83 option 1), then this bench again.

### 45.85 2026-10-24 evening: the four causes of the remaining losses, and what changed since 10-23

**The dgamma reduce has the same scoreboard problem (control bits of `reduce_dgamma_qk_bf16_tm<128,64,3>`):** all 320
loads are on SB5, and the adds wait on SB5 once per 192 rows (`FADD ... wait=[5]`, lines 879 / 951 of the local SASS),
which also drains the two buffers just prefetched → almost no overlap of loads and adds. ncu: 10.7 µs at B1, 32.7 µs at
B4 (≈24% of our backward at 18×6 B4 under ncu; in a normal run the partial rows are in L2, so less, not measured).

**Waves (72 registers -> 28 one-warp blocks per SM):** 18×6 B1/B2/B4 = 0.21/0.42/0.83 waves; 18×6 B8 = 1.66 (last wave
66% full); 32×8 B1/B2 = 0.35/0.69; **32×8 B4 = 1.39 (39% full)**; 32×8 B8 = 2.77 (77% full).

**The five losing cases and their causes** (1 = positions kernel waits at every token, §45.83; 2 = reduce waits as above;
3 = too few warps at small batch, 768 at B1; 4 = part-empty last wave):

| Case (backward) | Ours vs best rival (µs) | Causes |
| --- | --- | --- |
| 18×6 SFT B1 | 90.2 vs 51.3 compile | 1, 3 |
| 18×6 SFT B2 | 101.3 vs 87.4 Quack | 1, 3 |
| 32×8 SFT B1 | 99.9 vs 72.3 compile | 1, 3 |
| 32×8 SFT B4 | 256.6 vs 236.1 Quack | 1, 2, 4 |
| 18×6 pretraining B1 | 54.3 vs 51.8 compile | 2, 3 |

Also: 2-byte loads = ~6x the memory instructions of torch.compile (13.4 M vs 2.2 M at B4); DRAM 50–60% vs compile's
72–79%: no loss today, but it caps the gain at B8.

**What changed since 10-23 and what it gave** (same rival harness; the rivals ran at the same speed on both boxes, e.g.
compile 18×6 SFT B4 149.6 -> 150.9 µs): backward 18×6 pretraining B1 92.3 -> 54.3, B4 141.8 -> 119.3; 18×6 SFT B4 152.1 ->
128.5, B8 348.5 -> 271.2; 32×8 pretraining B4 313.8 -> 215.3; 32×8 SFT B4 327.1 -> 256.6; 18×6 SFT B1 only 97.8 -> 90.2.
From the two ncu runs (18×6): the new reduce 64.3 -> 32.7 µs at B4 (the biggest gain at B4/B8); the new kernel without
positions 82.0 -> 56.2 µs at B1, 108.1 -> 103.2 at B4; the new kernel with positions gained almost nothing (cause 1).

**Next:** cp.async ring in shared memory with `cp.async.wait_group` for the backward AND the reduce (fixes causes 1 and 2,
and gives 16-byte loads), keeping the per-token math and the dgamma order (bit-identical); then the 44-case test and the
rival bench on the 1x H200 box. Causes 3 and 4 are separate decisions (3 changes the dgamma grouping).

### 45.86 2026-10-24 evening: the cp.async backward and reduce (written, compile-checked; GPU test next)

Library file (laptop SFT tree) md5 **5ef36fa4**; the noon version that passed 44/44 is kept as
`port_1024/FusedRMSRoPE_bf16_1024_tested_07953145.cu` and is still selectable: `-DBF16_BWD_ASYNC=0 -DBF16_RED_ASYNC=0`.
- `fused_qknorm_rope_bwd_ca_bf16_tm<IS_NEOX, HD, HAS_POS, DEPTH>` (default at hd 128): DEPTH token slots in SHARED
  memory (dz row, x row, cos|sin row, rstd, next position; 1,040 B per slot), filled with cp.async (16-byte pieces:
  dz by lanes 0-15, x by lanes 16-31, cos|sin by all 32 lanes; rstd and the position 4 bytes each); before token j:
  `cp.async.wait_group<DEPTH-1>` + `__syncwarp()`; one commit per refill (empty past the last token). The per-token math
  is the pinned code of the register kernel reading from shared memory.
- `reduce_dgamma_qk_async_bf16_tm<HD, U, STAGES>` (default): STAGES buffers of U rows × 32 columns in shared memory,
  cp.async 16-byte pieces (4 rows per instruction), `wait_group<STAGES-1>`, each lane adds its column in row order.

**Compile checks (nvcc 13.0, library flags, sm_90a):** backward 48–64 registers (was 72) -> 32 one-warp blocks per SM;
3,120 B shared per block; reduce 83 registers, 24,576 B shared; no spills. SASS of the loop, all 4 variants: **no plain
global loads** (per 3 tokens: 9 `LDGSTS.E.BYPASS.128` + 3–6 `LDGSTS.E`, 24 `LDS.U16` + 15–18 `LDS`), waits = `DEPBAR.LE
SB0, 0x2` (= at most 2 groups in flight), and the per-token FP instruction forms **identical** to the original's. Reduce:
16 `LDGSTS.128` per 64 rows, one `DEPBAR.LE SB0, 0x2` per buffer. Global memory instructions per token: 4–5 (was 13).

**GPU run prepared:** `~/Downloads/rope_ca_1024.tgz` (library file + `run_ca_tests.sh`): port test for ca_d3 (default),
pp_d3 (noon version, baseline in the same session), ca_d4, ca_d6; then the rival bench with the default build (+ Liger).

### 45.87 2026-10-24 18:44–18:51 IST: the cp.async version PASSES and is the fastest in all 16 cases

Box: new 1x H200 (86.38.238.224), GPU 0, `run_ca_tests.sh`. Results: `port_1024/run_ca_tests_part1_from_paste.txt`
(port tests), `port_1024/rival_ca_1024_tables_from_paste.txt` (rival bench); on the box `port_1024/port_test_{ca_d3,pp_d3,
ca_d4,ca_d6}.log`, `run_rivals_ca.log`, `rivals/rival_results_ca_1024.json`.

**Correctness: 44 of 44 cases bit-identical in all 8 outputs for all 4 builds** (cp.async at DEPTH 3, 4, 6; the noon
register version).

**Backward + reduce (µs per call) vs Kruthik's original, same session:**

| Batch | Pretraining: original / noon (pp) / **cp.async (D3)** | SFT packed: original / noon (pp) / **cp.async (D3)** |
| --- | --- | --- |
| B1 | 102.3 / 57.0 / **46.2 (2.21x)** | 188.5 / 92.4 / **47.5 (3.97x)** |
| B2 | 123.2 / 68.8 / **60.3 (2.04x)** | 211.0 / 104.1 / **61.7 (3.42x)** |
| B4 | 169.0 / 120.4 / **104.4 (1.62x)** | 253.3 / 132.0 / **109.2 (2.32x)** |
| B8 | 401.7 / 252.2 / **198.2 (2.03x)** | 550.7 / 275.1 / **202.6 (2.72x)** |

Totals (fwd + bwd + reduce) at D3: SFT 3.09x / 2.52x / 1.81x / 2.03x, pretraining 1.85x / 1.65x / 1.37x / 1.61x (B1/B2/B4/B8).
With positions now as fast as without (47.5 vs 46.2 µs at B1): the scoreboard waits of §45.83 are gone.
DEPTH: 4 is faster at B1 (42.3 / 43.7 µs) but slower at B4/B8 (113.4 / 116.9 at B4); 6 is slower everywhere -> **DEPTH 3
stays the default** (B4 is our training size).

**Rival bench (kernel time, µs), ours vs the best of torch.compile, Quack, Liger, FlashAttention, eager:**

| Heads | Case | B1 bwd | B2 bwd | B4 bwd | B8 bwd |
| --- | --- | --- | --- | --- | --- |
| 18×6 | pretraining | 43.4 vs 51.9 (1.20x) | 57.4 vs 87.1 (1.52x) | 102.2 vs 149.7 (1.46x) | 197.9 vs 278.7 (1.41x) |
| 18×6 | SFT | 45.1 vs 51.6 (1.14x) | 58.5 vs 87.1 (1.49x) | **107.2 vs 148.9 (1.39x)** | 198.3 vs 277.4 (1.40x) |
| 32×8 | pretraining | 52.7 vs 72.3 (1.37x) | 84.9 vs 127.7 (1.50x) | 183.9 vs 235.8 (1.28x) | 318.0 vs 445.3 (1.40x) |
| 32×8 | SFT | 54.9 vs 72.5 (1.32x) | 88.1 vs 127.0 (1.44x) | 175.8 vs 236.2 (1.34x) | 328.1 vs 451.2 (1.38x) |

**Ours fastest in 16/16 forward (1.06–1.49x), 16/16 backward (1.14–1.52x), 16/16 total (1.15–1.46x)**; most exact
(err 1.00 / 1.00, dgamma ≤ 1.2e-6; compile 1.5–1.7 / 2.55–2.66; Quack, FlashAttention ~2.0 / 2.1; Liger, eager 3.3–3.8).
History of the backward wins: 10-23 port 2/16 -> 10-24 noon 11/16 -> 10-24 evening 16/16.
Our SFT training size (18×6, B4): forward 68.8 vs 94.3, backward 107.2 vs 148.9, total 176.0 vs 243.2 µs (1.38x vs
torch.compile, the best rival).

**Next:** trainer integration (rebuild the SFT library with this file, step-0 loss identical, step time); commit/push
(my call).

### 45.88 2026-10-24 18:59–19:07 IST: repeat run (run 2) confirms §45.87

Same box, GPU 0 idle before the run (0 MiB, 0 %), same script and library (5ef36fa4); run 1's box logs kept in
`~/gautam_k/rope_1024/run1/`. Summary: `port_1024/run_ca_tests_run2_from_paste.txt`.
- **Correctness again 44 of 44 bit-identical for all 4 builds.**
- Port test, cp.async D3, backward + reduce, run 1 -> run 2: within **−0.2% .. +1.1%** for all 8 cases (e.g. SFT B4
  109.2 -> 109.9 µs, B1 47.5 -> 48.0 µs).
- Rival bench, our backward kernel time run 1 -> run 2: **−0.3% .. +2.4%** (largest: 18×6 pretraining B8 197.9 -> 202.6 µs).
- **Again fastest in 16/16 forward, 16/16 backward, 16/16 total**; smallest margins: forward 1.06x (32×8 SFT B8, 223.2 vs
  compile 235.6), backward 1.13x (18×6 SFT B1, 45.3 vs compile 51.4). The noise between runs (≤ 2.4%) is far below the
  smallest margin (6%).

### 45.89 2026-10-24 evening: RTX 3060 port test, and the plan for the trainer A/B

**RTX 3060 (laptop, sm_86), port test with the cp.async library 5ef36fa4** (`port_1024/port_test_rtx3060_sm86.log`):
**44 of 44 bit-identical** (correctness holds on a second architecture). Speed vs Kruthik's original is mixed there:
with positions 1.02–1.09x faster, without positions **0.82–0.96x (slower)**. The RTX 3060 has ~13x less memory bandwidth than
the H200, a 3 MB L2 and 28 SMs; the cp.async depth and the reduce buffers were tuned on H200. So the speed-up is H200
specific (to be re-tuned per architecture later); the correctness is not.

**Trainer A/B plan** (scripts `profiling_reports/rope_bf16_port_2026_10_22/trainer_ab_1024/`, package
`~/Downloads/trainer_ab_1024.tgz`, box: new 1x H200 86.38.238.224, `~/gautam_k/trainer_ab`):
- **What differs:** only `FusedRMSRoPE_bf16.cu`: old = Kruthik's stage-B file 5338bd7b (the library of the Tülu run), new =
  5ef36fa4. The rest of the laptop tree is the run's own (all other changed files are dated 10-22 before the run's library
  build at 13:54 and trainer build at 15:03; `bluscript_sft.cpp` f2024ab2 = the run's source).
- **How:** two library builds snapshotted to `libs_old/` and `libs_new/`, ONE trainer binary built without a runpath to the
  library, so `LD_LIBRARY_PATH` alone picks the arm; checked three ways: `nm` before (cp.async kernels present only in new),
  `/proc/<pid>/maps` during every run, kernel names in nsys after.
- **Settings:** the real run's (fullchess init, 28 layers, theta 50000, B 4 × 5 = 20 rows, Muon 5e-5, all bf16, packed rows
  with positions), on the 320-row Tülu val set used as training data (16 steps per epoch); no checkpoints; val only at step 0
  and the last step; free clocks.
- **Step 1, `ab_1_setup.sh`** (CPU + 5.19 GB download, ~20 min): download fullchess, build old/new libraries + the trainer.
- **Step 2, `ab_2_check.sh`** (~8 min): runs old, new, old (32 steps each): |new − old| per step must not exceed the null
  |old2 − old| (the trainer is not bit-reproducible run to run, 10-22).
- **Step 3, `ab_3_nsys.sh`** (~8 min): one nsys run per arm; GPU time per step of the RoPE backward + reduce (140 calls per
  step, training only) and of all kernels.
- **Step 4, `ab_4_crossover.sh`** (~3 min per run; R = 6 rounds × 4 runs ≈ 70 min): end-to-end step time, rounds alternating
  old,new,new,old / new,old,old,new; mean dt of steps 10..62 per run; per-round new − old, mean and SE.
- **Expected:** RoPE backward + reduce per step old ≈ 140 × 253 µs ≈ 35 ms, new ≈ 140 × 109 µs ≈ 15 ms -> about **20 ms less
  per step**, ≈ 1.1% of the 1.86 s step (44,000 -> ~44,500 tokens/s). The summary parser, tried on two windows of the real
  run's log (a natural null), gave −0.84 ms (−0.04%) with ~4 ms noise per round, so 20 ms is well above the noise.

### 45.90 2026-10-24 19:30–20:05 IST: trainer A/B of the new RoPE kernel, SFT and pretraining (1x H200, GPU 0)

Setup (`trainer_ab_1024/`, box `~/gautam_k/trainer_ab`): libraries built twice from the run's tree, differing only in
`FusedRMSRoPE_bf16.cu`: **old** = Kruthik's stage-B 5338bd7b (libtensor 618eb7b7; the Tülu run's library), **new** = 5ef36fa4
(libtensor a449097a); one trainer `ccb73996` (source f2024ab2, runpath only the system dir), the arm chosen by
`LD_LIBRARY_PATH`. Verified: `cuobjdump -symbols` (new: 4 cp.async backward kernels, old: 0; a first check with `nm -D` saw
none in either, because the GPU kernels are not exported host symbols), `/proc/<pid>/maps` in every run, kernel names in nsys.
Results copied to `trainer_ab_1024/results/` (SFT sqlite files, pretraining logs, kernel summaries, clock/power logs).

**SFT** (fullchess init, 28 layers, B 4 × 5, packed positions, 320-row Tülu val set as data):
- Correctness (32 steps, old / new / old): step-0 train 1.293106 and val 1.175723 in all three (= the real Tülu run's
  step-0 val); per-step max |new − old| train 0.000203 vs null |old2 − old| 0.000221; norm 0.0008 vs 0.0012. PASS.
- nsys (32 steps per arm): RoPE backward 29.01 -> 14.12 ms/step, reduce 9.02 -> 2.28, forward 20.98 = 21.01 ->
  **backward + reduce 38.03 -> 16.40 ms/step (−21.6 ms, 2.32x, the same ratio as the standalone test)**. Per call: old
  207.2 + 2 × 32.2 µs, new 100.9 + 16.3 µs.
- Timeline (steps 5..29): one CUDA stream, GPU idle 3.4% and unchanged -> GPU-bound. Wall −12.9 ms/step (−0.68%): the
  RoPE kernels saved 21.6 ms but the other kernels took +9.6 ms (+0.54%); the two largest forward GEMMs unchanged, the
  backward-pass GEMMs and the attention backward +1–2%. One pair only (the 70-minute crossover was skipped).

**Pretraining** (the trainer's own pretraining config: 26 layers, theta 500000, B 8 × 32 micro-batches = 1,048,576 tokens
per step, Muon, all bf16, random init, the 5 edufineweb shards as the 4 domains; RoPE on the no-positions path; one nsys run
per arm, BLU_MAX_STEPS=6, the same shortened schedule in both; `ab_5_pretrain.sh`, run by me with the user's OK):
- Both exit 0, the right library mapped, 6 steps, peak 106,299 MiB. Step-0 train 11.290755 and val 11.294598 identical;
  later steps drift as run-to-run noise (lr 1.2e-3 from step 1 on a random init; grad norms 19 -> 200).
- nsys: per step 832 backward calls (26 × 32). RoPE backward 195.29 -> 149.98 ms/step, reduce 159.99 (2 launches per call)
  -> 25.99, forward 197.56 = 197.60 -> **backward + reduce 355.3 -> 176.0 ms/step (−179 ms, 2.02x; standalone B 8: 2.03x)**.
- Trainer dt, steps 2..5: old 21,182 ms, new 21,078 ms -> **−104 ms/step (−0.49%)**; timeline window: wall −102.8 ms,
  GPU idle 0.58% -> GPU-bound; the other kernels +94 ms (+0.40%): again the backward GEMMs +0.85..+2.87% and the attention
  backward +0.91%, while the forward attention, RMSNorm and other GEMMs stayed within ±0.2%.
- Clock/power (1 s samples while busy): power 691.4 vs 691.6 W (at the 700 W cap in both), SM clock mean 1557.2 vs 1553.6 MHz,
  median 1560 vs 1545 MHz.

**Reading:** the kernel saving shows up inside the real trainer exactly as in the standalone tests (2.32x SFT, 2.02x
pretraining). On a power-capped H200 (≈ 692 W in every run), only part of it reaches the step time: the old kernel's slow,
low-power stretches inside the backward pass let the other backward kernels clock higher; with them gone, those kernels run
~1–3% slower. Net: **SFT −12.9 ms/step (−0.68%), pretraining −104 ms/step (−0.49%)**, from single pairs (both runs
GPU-bound, idle unchanged). The pattern (backward kernels slower, forward kernels unchanged, clock median −15 MHz) is the
same in SFT and pretraining, which argues for a real power effect rather than noise; a same-process interleaved test would
settle the exact size (skipped for time).

### 45.91 2026-10-24: the RoPE kernel's benefit in simple terms (saved as asked; goes into today's EOD)

**1. Standalone kernel (H200, kernel alone)**

Against Kruthik's original kernel, backward + dgamma reduce per call:

| Batch | Pretraining (no positions) | SFT (packed positions) |
| --- | --- | --- |
| B1 | 102.3 → 46.2 µs (**2.21x**) | 188.5 → 47.5 µs (**3.97x**) |
| B4 | 169.0 → 104.4 µs (**1.62x**) | 253.3 → 109.2 µs (**2.32x**) |
| B8 | 401.7 → 198.2 µs (**2.03x**) | 550.7 → 202.6 µs (**2.72x**) |

Against the rivals (torch.compile, Quack, Liger, FlashAttention, plain PyTorch): **fastest in 16 of 16 cases for forward,
backward and total** (margins 1.06–1.52x), and the **most exact** in all. At our SFT size (18×6 heads, B4): **176.0 vs
243.2 µs for torch.compile (1.38x faster)**.

**2. Inside the real trainer (nsys, RoPE kernels only)**

| | SFT step (B4 × 5, 28 layers) | Pretraining step (B8 × 32, 26 layers) |
| --- | --- | --- |
| RoPE backward + reduce, old → new | 38.0 → **16.4 ms (2.32x)** | 355.3 → **176.0 ms (2.02x)** |
| Saved per step | **21.6 ms** | **179 ms** |
| RoPE share of the step, old → new | 2.0% → 0.9% | 1.7% → 0.8% |

The speed-up inside the trainer is the **same as standalone** (2.32x SFT; 2.02x vs 2.03x pretraining), so the standalone
results are real in training.

**3. Whole training step (what we actually get)**

| | SFT | Pretraining |
| --- | --- | --- |
| Step time, old → new | 1881.0 → 1868.2 ms | 21,182 → 21,078 ms |
| **Gain** | **−12.9 ms (0.68% faster)** | **−104 ms (0.49% faster)** |
| Over a whole run | our 11,590-step Tülu run: about **2.5 min** saved of ~6 h | a 22,600-step run like fullchess: about **0.5% less time** |

**Why the step gains less than the kernel saves:** the H200 sits at its **700 W power cap** in every run. The old slow
RoPE kernel was a low-power "rest" inside the backward pass; without it, the backward matrix kernels around it run 1–3%
slower (clock median 1560 → 1545 MHz). So about **60% (SFT) / 58% (pretraining)** of the kernel saving reaches the step time.

**Precision: no change, bit-exact.** The new kernels give **bit-identical** results to Kruthik's original in all 8 outputs
(forward q/k outputs, q/k rstd, dq, dk, dq_gamma, dk_gamma): 44 of 44 test cases, for every build tried (cp.async at
depths 3, 4, 6 and the register version), in two separate runs, on H200 and on RTX 3060. Same per-token math (pinned to the
original's compiled form), same lane mapping, same dgamma summation order. In the trainer: the step-0 loss is identical
(SFT train 1.293106 / val 1.175723; pretraining train 11.290755 / val 11.294598), and the later steps differ only within
the trainer's own run-to-run noise (old vs old differs as much as new vs old), which comes from other non-deterministic
kernels in the trainer, known since 10-22, not from RoPE.

**In one line:** the new RoPE kernel is 2–4x faster than Kruthik's and the fastest of all rivals, bit-exact; in real
training it halves the RoPE time and makes each step about 0.5–0.7% faster (RoPE is only ~2% of a step).

### 45.92 2026-10-24 evening: attention for lower GPUs (Ada sm_89, RTX 3060 sm_86) — causal mask + SFT packed rows added to the generic kernels

**Context (the doubt).** RoPE and the loss kernel already run the SFT features on every GPU. Attention did not: the
user asked which attention kernels Ada and the RTX 3060 run, whether they have the SFT features, and whether they are
optimized. Colleagues said there is one older "generic" kernel, without TMA and not optimized, that works on lower GPUs.

**What the code showed (BluTrain_sft TI = BluTrain_lottn for the whole attention folder, `diff -rq` empty).**
- The only files with "generic" in the name are `GQA_fwd_generic.cu` (kernel `gqa_v2`) and `GQA_bwd_generic.cu` (kernels
  `gqa_backward_dQ`, `gqa_backward_dKdV`). Written by sanjay 2026-06-24 (`339c4b4`, for sm103), moved into the generic
  files by madhu 2026-07-22 (`465fd8b`, "can run the GQA model in any GPU"). WMMA 16x16x16, one warp per block, 16 query
  rows per block, plain loads (no TMA, no cp.async), compiled for the machine arch (Makefile `-arch=sm_$(SM_ARCH)`).
- They had NO causal mask (kernel bodies have no causal check; `GQA_bwd_generic.cu` header said so; all 13 copies on the
  laptop, 4 versions, the same).
- Dispatch off Hopper before the change: a chat plan -> throw; causal head size 128 (our model) -> throw; causal head
  size 64 -> `gqa_v2` without a mask = full attention, a silent wrong result.
- The Ada-tuned kernels (`fused_attn_forward_kernel_tc_sm89`, `mem_efficient_bwd_unified_kernel_exp12`) are fp32 only and
  have one head count (no GQA); the trainer's GQA call accepts only the FlashAttention backend, so they are not reachable.

**What changed (laptop BluTrain_sft TI, not committed; work folder `profiling_reports/generic_attn_sft_2026_10_24/` with
`orig/`, `new/`, `diffs/`, `logs/`, `h200_sass_proof/`).**
1. `GqaChatPlan` (AttentionKernels.h): one more list, `gen_segs` = {b, t0, L, 0} per segment (FlashAttention-2 varlen's
   cu_seqlens list) + `n_gen_segs` + `gen_max_len`, appended LAST in the same single upload (the four Hopper lists keep
   their place and bytes). The list does not depend on the tile size, so the optimization step can change tiles freely.
2. The three kernels get a `CAUSAL` template flag. A block = (segment, head, tile counted from the segment START). The key
   loop stops at the tile's last row; a key after the row gets score -inf / p = 0; rows and keys past the segment end are
   read as zero and never written; dK/dV loop only over query tiles that can see the block's keys. Because tiles are
   counted from the segment start, a chat in a packed row runs the same tiles in the same order as the chat alone.
   Plain causal rows = the same code with one segment per row (t0 = 0, L = T). Launches use `getCurrentStream()`.
3. Dispatchers (both qkv entries, both packed-buffer entries): `generic_ok = is_causal && !(v62/Hopper kernel takes it)
   && not B300`. Hopper v62 / Vj2 checks are unchanged and run FIRST. The plan is accepted on the generic path. B300 is
   untouched. Causal head size 64 off B300 now runs the causal kernel (was full attention).
4. `AttentionOps.cpp`: one comment.

**H200 safety (proof).** The six Hopper attention files (v62, Vj2, Vj1d, Vj1, v44, RoPE_sm90) are byte-identical to lottn
(same md5) and do not include `AttentionKernels.h`; compiling them for sm_90a gives identical SASS. Only 4 files differ
from lottn in the attention folder + header + ops: the two generic files, the header, AttentionOps.cpp (comment).

**Tests on the local RTX 3060 (test_varlen_attention.cu, quick mode, head size 128): 292 passed, 0 failed, 1 KNOWN.**
- fp64 truth + bf16 floor references, FlashAttention's pass rule, for O, LSE, dQ, dK, dV: dense causal rows (T 256 / 4096,
  18/6 and 8/1 heads), real SFT rows, large inputs, every chat alone, packed rows (D2-D11 + 16 real packed stage-0 rows).
- Packed == alone BIT-EXACT in every layout (0 mismatched elements, dQ diff 0); isolation (change one chat -> others
  bit-identical); repeat 50x at 8x4096 and 20x at 60x1024: all five outputs bit-identical (no atomics).
- KNOWN: dQ of chat `row 1 @664 L=303` at 1.049x the limit = the same chat and the same ratio recorded 10-10 for Vj1d
  (dS from the bf16 P, which the generic dQ kernel also uses); proven then not to be packing.
- Speed (unoptimized): B=8 T=4096 18/6: fwd 726 ms, bwd 1,669 ms = 0.9 TFLOP/s. Packing skips work: a backward with 14%
  of the dense work takes 12.5% of the dense time.
- Harness changes: `VARLEN_HD` (64/128), `g_generic`, refusals that no longer apply removed (head size 64 and T % 256 are
  accepted now), the reference computes one kv-head group at a time (S/dP for all 18 heads at L = 4096 needed 7.2 GB, too
  much for 12 GB; the values are unchanged, G1 re-checks them), mode label fixed, and `chat_alone` / G4 used a hard-coded
  head size 128 (found when the head-size-64 run read past 64-wide buffers: a harness bug, now `g_hd`).
- Not run (decided 10-24): the full non-quick run (~40 min of repeats at this speed). Head size 64 quick run: see §45.94.

### 45.93 2026-10-24 night: matmul audit of the production pretraining trainer — where the fp32 accumulator could feed the next op directly (FIRST TASK on Monday 2026-10-26)

**Context (the doubt).** During the BluBLAS work an idea was parked as a future task: when a bf16 matmul is followed by an
op that works in fp32, take the output straight from the GEMM's fp32 accumulator, instead of rounding it to bf16 (losing
precision) and converting it back to fp32 in a separate kernel. Colleagues said this is already implemented, including in
attention. The user asked to go through the whole training script inch by inch, find every matmul and what is around it,
and check whether colleagues did it or not. The text below is the answer given on 10-24, kept in full as the user asked.
The user decided: this is the FIRST task on Monday morning 2026-10-26 (immediate future task). Nothing below is measured
yet; the GB numbers are estimates from tensor sizes.

**Matmul audit: which code, which settings**
- Trainer: lottn `train/production_zero/bluscript_zero_ckpt.cpp`, built with `MODE=production_zero`.
- Production config: `bluscript_zero_ckpt_config.yaml` sets:
  - `BLU_ALL_BF16=1` and `BLU_BF16_LOGITS=1`: the residual stream, logits and embedding are all bf16, so every
    `to_dtype` next to a GEMM is a no-op;
  - `grad_view: true` with `fp32_reduce: false`: micro-step gradients add up in the bf16 communication buffer;
  - Muon, ZeRO v2, and the BluBLAS backend.
- Library rule:
  - `matmul` gives its output the input's dtype (`GenMatmul.cpp`: `Tensor output(output_shape, A.dtype(), ...)`).
  - BluBLAS's bf16 functions (`Bgemm`, `BgemmStridedBatched`, `BgemmAddmm`, `BgemmSwiglu`) all write a
    `__nv_bfloat16* C`; none writes fp32.
- Shapes per micro-step: M = 8 x 4096 = 32,768 tokens, d = 2304, ffn = 6144, vocab = 50,304, 26 layers.

**Every matmul and what is next to it**

| Matmul | Output | Next op | fp32 accumulator used directly? |
|---|---|---|---|
| wQ, wK (2304->2304 / 768) | bf16 | `fused_rms_rope_bf16` (computes in fp32) | No: rounded to bf16, re-read |
| wV (2304->768) | bf16 | transpose copy -> attention | No, but attention needs bf16 anyway |
| Attention QK^T (v62) | fp32 registers | softmax on the registers | Yes (GQA_fwd_sm90.cu:924-976) |
| Attention PV | fp32 registers | rounded to bf16 once at the store | Yes (:993-996 P stays in registers, :1243 store) |
| wO (2304->2304) | bf16 | residual `add` (bf16) | No: rounded, then the add rounds again |
| gate_up (2304->12288) | bf16 | `swiglu` kernel | No (a fused version exists, unused; see below) |
| down (6144->2304) | bf16 | residual `add` | No |
| lm_head (tied wte, 2304->50304) | bf16 logits, 3.3 GB | cross-entropy (fp32 inside) | No |
| dX of wQ, wK, wV | 3 x bf16 [32768x2304] | engine `sum +=` (Engine.cpp:343): 2 bf16 add kernels | No: 3 roundings plus 2 add kernels |
| dW of all 6 weights | bf16 | `accumulate_grad` -> `*grad_ += update` into the bf16 buffer | No: 2 roundings plus an add kernel every micro-step |
| Vj2 S, dP | fp32 registers | dS in registers | Yes |
| Vj2 dQ | fp32 buffer | converted to bf16 once | Yes |
| Vj2 dK/dV | bf16 partial per query head (line 709) | `gqa_dkdv_greduce_vj2` adds the 3 partials in fp32 | Partly: one extra bf16 rounding (FA3 keeps fp32) |
| Muon B = b*A + c*A*A, X = a*X + B*X | - | the addition happens inside the GEMM | Yes: `cuda_addmm` -> BluBLAS SM90 addmm epilogue with a full-matrix bias (Optim.cpp ~1567; Bgemm_addmm_sm90.cu) |

**What colleagues did, and what they did not**
- Done:
  - attention forward and backward, except the Vj2 dK/dV partials;
  - the Muon Newton-Schulz linear combinations.
- Exists but not connected: BluBLAS has a Hopper GEMM + SwiGLU epilogue (`Bgemm_sm90_swiglu_fused_template.cuh`), but
  nothing in any TI tree calls it. Its old caller `cuda_matmul_swiglu` is gone; it is linked to an 08-14 NaN incident.
- Not the same thing: the "F2" fused gradient accumulation (AutogradMeta.cpp:177-193) is a separate widen+add kernel, not
  a GEMM epilogue. With `grad_view=true` in production, the dtypes match (bf16 + bf16), so a plain bf16 add runs instead
  and F2 is never used.

**Where the idea still applies (estimated traffic saved per micro-step)**
- C1 - dW accumulates inside the GEMM (beta = 1, writing straight into the gradient buffer). This removes 156 add kernels
  and one bf16 rounding per micro-step, about 5.9 GB less traffic. BluBLAS's Hopper dW kernels need a beta = 1 path; not
  yet checked whether they have one.
- C2 - one fused QKV weight [2304x3840]. The backward dX becomes one GEMM, and the sum of the three gradients happens
  inside the fp32 accumulator. That removes 2 bf16 adds and 2 roundings per layer, about 23.6 GB; the forward also reads h
  once instead of three times. Caveat: Muon orthogonalizes wQ, wK and wV as separate matrices, so the optimizer must keep
  splitting them.
- C3 - residual add inside the wO/down epilogue. About 15.7 GB less, and one rounding fewer per add. It also allows an
  fp32 residual stream at bf16 cost, which is the precision ALL_BF16 gave up. The tool mostly exists already:
  `autograd::addmm(x, merged, Wo)` with the MxN "bias" supported by the SM90 addmm. It needs a 2-D flatten and a check of
  `AddmmBackward` for 3-D inputs.
- C4 - lm_head fused with cross-entropy (Liger style). The 3.3 GB logits and their gradient never get written to memory.
  The biggest saving in peak memory and the biggest piece of work.
- C5 - connect the existing SwiGLU epilogue. The backward needs gate and up, so they must be stored or recomputed.
- C6 - Vj2 dK/dV partials in fp32. Removes the rounding behind the recorded KNOWN dV deviations (1.01-1.17x the limit).
  Costs about +15.7 GB of traffic, so it trades speed for precision.
- C7 - Muon's last Newton-Schulz GEMM writes fp32, skipping the `as_type(Float32)` cast. Small.

**Next (Monday 10-26, first):** one nsys run on the H200 to time the real kernels behind C1-C3 (the gradient-accumulation
adds, the two dh adds, the residual adds) before writing any code; needs a GPU (ask first).

### 45.94 2026-10-24 21:50–22:03 IST: generic attention at head size 64 (RTX 3060, quick) + the SFT package

- First head-size-64 run stopped in G6 with "illegal memory access": the harness's `chat_alone` and G4 had head size 128
  written as a number (copies read 128 values per token from 64-wide buffers). Harness bug, fixed (`g_hd`, 17 + 2
  places); the failed log is kept as `run3_quick_hd64_rtx3060_HARNESS_BUG_hd128_hardcoded.log`.
- Rerun (`run4_quick_hd64_rtx3060.log`): **291 passed, 1 failed**, 741 s. Packed == alone bit-exact everywhere; repeats
  (50x 8x4096, 20x 60x1024, 50x D10) bit-identical. The one failure: G11 D5 dQ of chat `row 0 @2688 L=64` at 1.175x the
  2x-floor limit; the same layout's packed-vs-alone check is bit-exact (not packing). Cause: the generic dQ kernel
  computes dS from the bf16-rounded P (as Vj1d did; the head-size-128 KNOWN entry 1.049x is the same mechanism). Fix
  (not run, user short on time): compute dS from the fp32 P as Vj2 / FA3 do; in the dQ kernel P is used only for dS.
- cp.async check (user question): the generic kernels use NO cp.async (sm_86 SASS: 0 LDGSTS, 174 / 252 LDG, 0 LDSM,
  one warp per block). cp.async exists from sm_80 (Ampere, Ada, Hopper); the generic kernel already needs sm_80+ (bf16
  WMMA). Our RoPE kernel uses it (204 LDGSTS) and passed on the RTX 3060. Optimization plan for Ampere/Ada: FA2 style
  (cp.async double-buffered K/V, ldmatrix + mma.sync, 4 warps).
- SFT package: `SFT_package_2026_10_24/` (+ .zip) in the experiment folder, start file `00_README_START_HERE.md`.
- Box 86.38.238.224 (1x H200) copied to `~/Downloads/box_backup_2026_10_24_1xh200/` except rivalenv (venv), pt_data
  (on the laptop) and ckpt/fullchess (public HF navingv/fullchess).

