# Results: the 2 x 2 window study, the kernel race, speed, memory

All numbers come from the runs of 2026-11-20 and 2026-11-21 (350M, 1 x H200 per run). Full analysis with every table
and caveat: `docs/sliding_window_attention_sft_analysis.md` sections 12.2 ("C against A", "Arm B") and 12.3.

---

## 1. The 2 x 2 study

**Setup.** The same 350M model pretrained with full attention (`kv4`) or with the window (`kv4_swa3`: 1,024, every 4th
layer global), each fine-tuned on Tülu 3 (`m2_tulu_A`, 1 epoch = 5,652 steps, LR 1e-4, 20 rows of 4,096 per step, Muon +
AdamW, bf16) with full attention or with the window; the same binary and settings for all four. Arm B ran on another
box (P015) with every input checked equal to the others' by md5 and CUDA runtime 13.0 instead of 13.4 (the model's
math runs in our own kernels, in the identical libraries; cuBLAS is not on its path); its step-0 val loss matched the
reference to 0.00004 and its export the trainer to 0.00001.

**Val loss during training** (the trainer's val, the same 308 rows for every arm):

| Step | A (full / full) | B (full / window) | B - A | C (window / window) | D (window / full) | C - D |
|---|---|---|---|---|---|---|
| 0 | 1.672250 | 1.678060 | +0.0058 | 1.668373 | 1.681077 | -0.0127 |
| 500 | 1.395124 | 1.397860 | +0.0027 | 1.394243 | 1.395336 | -0.0011 |
| 1,000 | 1.338270 | 1.340520 | +0.0023 | 1.337176 | 1.337712 | -0.0005 |
| 1,500 | 1.304564 | 1.307104 | +0.0025 | 1.303532 | 1.303804 | -0.0003 |
| 2,000 | 1.279233 | 1.281483 | +0.0023 | 1.277365 | 1.277615 | -0.0003 |
| 2,500 | 1.257578 | 1.259550 | +0.0020 | 1.255676 | 1.255697 | 0.0000 |
| 3,000 | 1.237261 | 1.239287 | +0.0020 | 1.235225 | 1.235015 | +0.0002 |
| 3,500 | 1.219137 | 1.221239 | +0.0021 | 1.217331 | 1.217229 | +0.0001 |
| 4,000 | 1.204701 | 1.206776 | +0.0021 | 1.202782 | 1.202523 | +0.0003 |
| 4,500 | 1.193088 | 1.195056 | +0.0020 | 1.191410 | 1.191190 | +0.0002 |
| 5,000 | 1.185149 | 1.187097 | +0.0019 | 1.183511 | 1.183230 | +0.0003 |
| 5,500 | 1.179824 | 1.181739 | +0.0019 | 1.178129 | 1.177882 | +0.0002 |
| 5,651 (final) | 1.178506 | 1.180413 | +0.0019 | 1.176781 | 1.176567 | +0.0002 |

**Final val loss by group** (HF transformers on the exports, each with the attention it was trained with; total = all
311 rows):

| Model | trainer final val | total (all 311 rows) | chats <= 1,024 | chats > 1,024 | pos 0-1,023 | pos 1,024-2,047 | pos 2,048+ |
|---|---|---|---|---|---|---|---|
| A | 1.178506 | 1.182986 | 1.191021 | 1.174183 | 1.135404 | 1.277946 | 1.796139 |
| B | 1.180413 | 1.185001 | 1.191413 | 1.177976 | 1.135756 | 1.285109 | 1.813233 |
| C | 1.176781 | 1.181186 | 1.187510 | 1.174256 | 1.132616 | 1.279082 | 1.803708 |
| D | 1.176567 | 1.180943 | 1.187694 | 1.173547 | 1.132838 | 1.276007 | 1.804080 |
| B served with full attention | (HF trainer rows 1.179826) | 1.184336 | 1.191413 | 1.176582 | 1.135761 | 1.281470 | 1.809601 |

**The two paths through the grid** (differences of the rows above; minus = lower loss after the step):

| Step | What changes | trainer final val | total | chats <= 1,024 | chats > 1,024 | pos 0-1,023 | pos 1,024-2,047 | pos 2,048+ |
|---|---|---|---|---|---|---|---|---|
| A -> B | window in SFT, full-attention base | +0.0019 | +0.0020 | +0.0004 | +0.0038 | +0.0004 | +0.0072 | +0.0171 |
| D -> C | window in SFT, window base | +0.0002 | +0.0002 | -0.0002 | +0.0007 | -0.0002 | +0.0031 | -0.0004 |
| A -> D | window in pretraining, full SFT | -0.0019 | -0.0020 | -0.0033 | -0.0006 | -0.0026 | -0.0019 | +0.0079 |
| B -> C | window in pretraining, window SFT | -0.0036 | -0.0038 | -0.0039 | -0.0037 | -0.0031 | -0.0060 | -0.0095 |
| A -> C | both | -0.0017 | -0.0018 | -0.0035 | +0.0001 | -0.0028 | +0.0011 | +0.0076 |
| (A -> B) - (D -> C) | interaction: the window in SFT on the full base minus on the window base | +0.0017 | +0.0018 | +0.0006 | +0.0031 | +0.0006 | +0.0041 | +0.0175 |
| B full attention - B window | B served with its pretraining attention | - | -0.0007 | 0.0000 | -0.0014 | 0.0000 | -0.0036 | -0.0036 |

**Each model run with the other attention type** (serving with the wrong attention), loss past position 2,048:

| Model | With its own attention | With the other type | Change |
|---|---|---|---|
| C (window / window) | 1.8037 | 2.0315 (full attention) | **+0.228** |
| A (full / full) | 1.7961 | 1.8506 (window) | +0.054 |
| D (window / full) | 1.8041 | 1.8172 (window) | +0.013 |
| B (full / window) | 1.8132 | 1.8096 (full attention) | -0.004 |

**Benchmarks** (lm-eval 0.4.13, `--apply_chat_template --fewshot_as_multiturn`, batch 32, HF transformers; GSM8K on
its first 400 problems; `_fullattn` = the same weights run with full attention in every layer):

| Model | IFEval prompt strict | IFEval inst strict | IFEval prompt loose | IFEval inst loose | MMLU | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | GSM8K flex / strict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.2440 | 0.3573 | 0.2643 | 0.3825 | 0.2472 | 0.4474 | 0.2167 | 0.2969 | 0.6208 | 0.5312 | 0.0300 / 0.0100 |
| B | 0.2680 | 0.3657 | 0.2828 | 0.3861 | 0.2467 | 0.4457 | 0.2201 | 0.2970 | 0.6273 | 0.5138 | 0.0275 / 0.0000 |
| C | 0.2403 | 0.3741 | 0.2569 | 0.3909 | 0.2507 | 0.4508 | 0.2031 | 0.2965 | 0.6235 | 0.4972 | 0.0175 / 0.0025 |
| D | 0.2255 | 0.3537 | 0.2403 | 0.3813 | 0.2520 | 0.4470 | 0.2014 | 0.2960 | 0.6192 | 0.5012 | 0.0225 / 0.0050 |
| C_fullattn | 0.2366 | 0.3741 | 0.2551 | 0.3921 | 0.2505 | 0.4499 | 0.2022 | 0.2960 | 0.6257 | 0.4972 | 0.0200 / 0.0000 |
| B_fullattn | 0.2643 | 0.3681 | 0.2810 | 0.3909 | 0.2471 | 0.4440 | 0.2210 | 0.2970 | 0.6284 | 0.5217 | 0.0250 / 0.0025 |

One run's standard errors (from the result files): IFEval prompt-level 0.018-0.019, MMLU 0.004, ARC-e 0.010, ARC-c
0.012, HellaSwag 0.005, PIQA 0.011, WinoGrande 0.014, GSM8K flexible 0.007-0.009. At 350M most of these benchmarks sit
near chance.

**What the study shows**
- **The window in SFT is free on a window-pretrained base**: C and D are the same within 0.003 in every group and
  within noise on every benchmark (D -> C: +0.0002).
- **On a full-attention base the window in SFT costs**: +0.0019 (A -> B), steady at about +0.002 from step 500 to the
  end, almost all of it in long chats (positions 1,024-2,047 +0.0072, 2,048+ +0.0171; chats of at most 1,024 tokens
  +0.0004). B even works better run with full attention than with its own window: one epoch of SFT does not make a
  full-attention model a window model.
- **So the window has to be learned in pretraining**; SFT keeps it for free but cannot add it for free (interaction
  +0.0017 overall, +0.0175 past position 2,048). This matches the published practice (every SWA model keeps the window
  in post-training) and the SWAA study (SWA on full-attention models hurts long context; fine-tuning narrows it).
- **Serving must match training for a window model**: C with full attention loses 0.23 past position 2,048.
- **Benchmarks** differ within noise; ARC-c and WinoGrande are a little higher for the full-attention base (A, B) than
  for the window base (C, D), so those gaps come with the base, not with the SFT attention.
- **Caveat**: one run per arm; the run-to-run noise (second seeds) is not measured. The B - A gap has the same size at
  every val check from step 500 on.

## 2. Speed of the window

| | Without the window | With the window | Change |
|---|---|---|---|
| 350M pretraining step (rows of 4,096, one sequence each) | - | - | **-8.6%** |
| 350M SFT step on Tülu (81,920 tokens; most chats short) | 476 ms | 474 ms | -0.4% |
| SFT arms on one H200 (per step) | A 483 ms, D 481 ms | C 473-478 ms, B 468 ms (another box) | - |

In SFT the window removes 13.6% of the attention pairs but only 1.3% of the step's FLOPs (15.4% of Tülu's tokens sit past
position 1,024 of their chat). The reasons for the window in SFT are the attention the model was pretrained with and the
inference memory, not SFT speed.

## 3. The kernel race with the window (2026-11-20 22:34-22:47)

350M heads 8 / 4, head size 128, bf16, window 1,024, causal; 8 rows of 4,096 tokens per call; 4 sets of real Tülu rows
and 4 rows of uniform chats; two rounds (ABBA); every library first passed a spike test of its window convention and an
fp64 reference check. Ratio = our time / theirs, forward + backward (below 1 = ours faster):

| Layout | ours fwd / bwd (ms) | ours / FA3 | ours / FA4 | ours / FA2 (PyTorch varlen) | ours / FlexAttention |
|---|---|---|---|---|---|
| real rows 0-7 | 0.1432 / 0.4572 | 0.649 | 0.690 | 0.398 | 0.535 |
| real rows 184-191 (longest chat) | 0.1486 / 0.4726 | 0.694 | 0.718 | 0.417 | 0.541 |
| real rows 56,520-56,527 | 0.1256 / 0.4169 | 0.673 | 0.681 | 0.408 | 0.518 |
| real rows 113,040-113,047 | 0.1476 / 0.4737 | 0.725 | 0.725 | 0.422 | 0.530 |
| uniform 512 | 0.0915 / 0.3261 | 0.623 | 0.624 | 0.430 | 0.605 |
| uniform 1,024 | 0.1372 / 0.4496 | 0.736 | 0.733 | 0.434 | 0.601 |
| uniform 2,048 | 0.2023 / 0.6174 | 0.812 | 0.825 | 0.469 | 0.625 |
| uniform 4,096 (pretraining-like) | 0.2268 / 0.6845 | 0.818 | 0.800 | 0.468 | 0.612 |

- **Ours is faster than every library on every layout**: 0.62-0.82 of FA3's time, 0.62-0.83 of FA4's, 0.40-0.47 of
  FA2's, 0.52-0.63 of FlexAttention's; the most on rows of many chats (real SFT rows: 0.65-0.73 of FA3).
- cuDNN was not raced: its window needs NVIDIA's cudnn-frontend Python package (not installed yet).
- Without the window (2026-11-18, 12 cases): ours beat FA3, FA4 and cuDNN in all 12 (`docs/sft_kernel_benchmarks.md`
  section 8).

## 4. Our code against Navin's original window code (same settings)

| Run | step-0 loss | step-29 loss | step (ms, medians of steps 11-29) |
|---|---|---|---|
| Navin's, no window | 11.0249 | 9.5164 | 13,529.5 |
| ours, no window | 11.0249 | 9.5101 | 13,244.3 |
| Navin's, window 1,024 | 11.0245 | 9.4942 | 12,416.0 |
| ours, window 1,024 | 11.0243 | 9.4871 | 12,193.1 |

The step-0 losses are equal, and the window changes the step-29 loss by the same amount in both. On the same GPU ours
is 2.8% faster in both modes.

## 5. Memory: the longest SFT row one H200 holds (350M, full attention, 1 row per step)

| Row length T (tokens) | Peak GPU memory (MiB) | Median step (ms, steps 2 and up) | Result |
|---|---|---|---|
| 16,384 | 30,941 | 115.8 | fits |
| 32,768 | 48,963 | 202.4 | fits |
| 65,536 | 83,981 | 387.7 | fits |
| 81,920 | 101,297 | 468.7 | fits |
| 98,304 | 118,069 | 564.2 | fits |
| 106,496 | 127,383 | 616.8 | fits |
| 114,688 | 136,217 | 663.5 | fits (the longest that fits) |
| 122,880 | 142,907 at the failure | - | out of memory |

So the 350M needs context parallelism from about 120,000 tokens per row (the window does not change this: attention
stores no score matrix either way).
