---
name: reference-sft-sources-colleague-1006
description: "What the 3 SFT links a colleague shared (2026-10-06) contain and how they fit our 1.59B / GPT-2 / T=4096 plan — smol-course unit1/4, OpenThoughts org, Nvidia Llama-Nemotron post-training dataset"
metadata:
  node_type: memory
  type: reference
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-06T09:47:00.717Z
---

Researched 2026-10-06: a workflow of 3 web researchers + 1 checker, with the key counts re-checked against the HF API. Full raw results are in the session task output `wd410bo7j` (gone after the session).

**1. huggingface.co/learn/smol-course/en/unit1/4** (HF "a smol course", SmolLM3-3B)
- It is a TOY tutorial: 1k rows of SmolTalk2, 500 steps, effective batch 4, LR 5e-5. It trains on ALL tokens (plain `text`, TRL default), so don't copy that.
- HF's real SmolLM3 SFT (alignment-handbook recipe + blog) uses `assistant_only_loss`, including the end-of-turn token. That is the same as our masking plan.
- The closest public small-model recipe is **SmolLM2-1.7B**: SmolTalk (~1.1M rows), 2 epochs, LR 3e-4, global batch 128 × 8192.
- **SmolTalk2:** SFT config 3,383,242 rows (1,481,833 think + 1,901,409 no_think).
  - It has no dataset-level licence, and some upstream parts are non-commercial (Explore-Instruct-Rewriting is CC-BY-NC).
  - About 592k of its usable rows overlap SmolTalk v1.
- Page bugs:
  - it imports `DataCollatorForCompletionOnlyLM`, which is gone in TRL ≥0.20;
  - the model ids `SmolLM3-1.7B` and `SmolLM3-3B-Instruct` do not exist.

**2. huggingface.co/open-thoughts** (Bespoke Labs / DataComp): long chain-of-thought REASONING data, single-turn only.
- **Datasets** (all Apache-2.0 on the card, but some upstream questions are CC-BY-SA or "all rights reserved"):
  - OT-114k: 113,957 rows, DeepSeek-R1 traces, verified;
  - OT2-1M: 1,143,205 rows; the HF viewer only converted 300,850;
  - OT3-1.2M: 1,200,000 rows, 28 GB, QwQ traces. Stored code, then math, then science, so ALWAYS shuffle before taking a subset.
- **Almost none of it fits our T=4096:**
  - OT3 averages ~11.6K tokens; 62% of rows are cut off with no final answer; about 1 in 34 rows fits 4096 GPT-2 tokens;
  - paper Table 22: keeping only traces ≤4096 tokens cost **−22.3%** relative.
- **Use it for the THINK model later** (needs ~16K context). OpenThinker3-1.5B (Qwen2.5-1.5B base): LR 1.6e-4, global batch 256, 7 epochs, cutoff 16384.
- **The paper is useful now for experiment design:**
  - a size ladder 316/1K/3.16K/10K/31.6K/100K, with LR/batch/epochs set per size (Table 9);
  - one or two good sources beat many;
  - decontamination: 13-gram + Indel similarity ≥75%.

**3. huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset** (v1.1, 2025-04-08)
- **Size:** 33,011,757 rows: math 22,066,397, code 10,108,883, science 708,920, chat 39,792, safety 31,426, RL instruction-following 56,339 (prompts only). The HF viewer shows only ~1.07M because it converted only part of it.
- **Columns:** `input` (messages), `output`, `reasoning` on/off, `system_prompt` "detailed thinking on/off", per-row `license`, `generator`, `used_in_training`, `version`.
- **Licence:** mostly CC-BY-4.0, plus ODC-BY (WildChat prompts) and CC-BY-SA (StackOverflow). Rows made by Llama-3.x models bring **Llama 3.1/3.3 licence conditions** if we release a model trained on them.
- **All responses are synthetic.** Generators include DeepSeek-R1 (3.9M) and Qwen-2.5-Math-7B (19.8M).
- **For instruct v1:**
  - only the reasoning-OFF rows: chat-off 31,218, of which 22,644 are single-turn ITS rows, ~99% ≤4096 tokens (raw count, before the language filter); safety-off 15,713 (Mixtral refusals). These are small.
  - Multi-turn rows are stored as 3 growing prefixes: remove the duplicates.
  - Math/code-off rows are formulaic single-model answers that NVIDIA used only for Ultra: subsample them only for skill ablations.
- **Successors:** `nvidia/Nemotron-Post-Training-Dataset-v1` (25.7M) and `-v2`.

**Checker's ranking for our plan:**
1. smol-course / SmolLM recipes as a guide;
2. Nemotron chat+safety (off) as a small add-on;
3. the OpenThoughts paper for experiment design;
4. Nemotron math/code-off, subsampled, for ablations;
5. reasoning data only for the think model later;
6. agent data is out of scope.

Related: [[project-sft-workflow-1006]] (which datasets are public), [[project-sft-research-assignment]].
