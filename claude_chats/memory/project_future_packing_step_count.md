---
name: project_future_packing_step_count
description: "FUTURE TASK (user 10-14): packing cuts optimizer steps (stage-0: 78 -> ~24/epoch at 64 rows/step); SmolLM3 saw IFEval -10% at eff. batch 128; research + ablation before the first packed SFT run; work doc §44.1"
metadata:
  type: project
---

The user asked me (2026-10-14) to REMIND this and to plan research + ablation tests + precautions for it.

**The issue** (from the frontier-training blog a colleague shared, "sequence packing" section, citing HF's SmolLM3 playbook):
packing puts more chats into each optimizer step, so the same data gives fewer steps. Small datasets got worse: SmolLM3
effective batch 128 lowered IFEval by up to 10%; above 32 an average drop. Large datasets: packing almost always helps.

**For us:** stage-0 = 5,000 chats, 1,517 packed rows (3.3 chats/row). v1 = 64 rows = 64 chats per step, 78 steps/epoch.
v2 at 64 rows/step = ~211 chats, ~24 steps/epoch. v2 at ~64 chats/step = ~20 rows, 78 steps/epoch.
The user's view: real runs may use much bigger datasets where it matters less, but it matters for stage-0 now.

**Plan (work doc §44.1), before the first packed SFT run (v2 Step 7):**
1. Research how TRL/SmolLM3, Tülu 3, OLMo, Hermes-4 size the step when packing (chats/tokens/rows), and LR vs batch.
2. Ablation on stage-0: (a) v1 64 chats/step, (b) v2 ~64 chats/step, (c) v2 64 rows/step; compare val loss (answer
   tokens), likelihood evals, base-vs-SFT forgetting. IFEval needs generation (not built yet).
3. Loader precaution: step defined by chats or tokens, not rows; log chats + trained tokens per step. 8 rows per
   micro-batch means ~20 rows/step needs a token budget or variable micro-batches.

Related: [[project_future_v2_packing]], [[project_state_2026_10_13_beat_rivals]].

**Exact (trainer formula, 10-14):** v1 156 steps / 5 warmup; v2 at 64 rows = **47 steps / 1 warmup** (~211 chats/step);
B=8 x 24 rows = 126 steps (~79 chats); B=8 x 16 = 189 (~53); B=4 x 20 = 151 (~66); B=8 x 20 is REFUSED
(rows/step must be a multiple of B x GPUs, `bluscript_zero_ckpt.cpp:947-952`). Packing doc section 9 has it all.

**Blog re-checked word for word (10-14):** the 10% IFEval figure is HF SmolLM3's (not a paper), at effective batch 128;
"hurts small datasets" is a separate general sentence. The blog's ONLY packing advice: lower the learning rate
(`#learning-rate-and-epochs`) -> ablation run (d): packed, 64 rows/step, lower LR. Counting steps in chats/tokens is OUR
plan, not the blog's. Effective batch = rows per step (64), NOT chats per row (3.3).
**Final ablation (10-14, packing_step_options.py):** 1 padding 156 steps | 2a pack-per-step 64 chats (17-23 rows, 95.9%)
| 2b pack-once + 61,075-trained-token budget (51-82 chats, ~150 steps) | 2c B=4x20 rows (151, no trainer change) | 3 64
rows (47) | 4 = 3 + lower LR Two seeds each. (Kathir's sort+pad = v1.5 bucketing, removed from the plan by the user 10-14.)
