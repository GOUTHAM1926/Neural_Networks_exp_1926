---
name: project-sft-attention-verdict
description: SFT attention researched 2026-09-25 — v1 (1 example/row, right-pad) needs NO attention change; packing needs a block-diagonal mask (colleagues' point). Pad input id 50256 not 65535; B300 can't run hd=128 causal attention
metadata:
  type: project
---

The full write-up is in the SFT doc §32. Raw: `profiling_reports/sft_attention_research_2026_09_25.json`.

**Verdict:**
- **v1 needs no attention change.** Batch rows never attend to each other (`GQA_fwd_sm90.cu:389,397`). Causal attention + right padding means a real token never sees a pad.
- The colleagues (post-training environment) are right for **packing, flattened batches, and left-padded RL/generation**. Those need `doc_start`-based block-diagonal masking in forward v62 (with a −inf guard) and backward Vj1d.

**New v1 rules:**
- The input pad id is **50256**. 65535 is a LABEL sentinel only.
- Input ids 50304–65534 leave uninitialised embedding rows, so they must be a hard error.
- EOT (50256) must be a target at the end of each completion.
- Never left-pad.
- Drop examples with 0 valid targets.
- v62 needs T%256==0 and T≤4096.

**B300:** there is no causal hd=128 attention kernel (it throws), and the loss forward ignores ignore_index. So SFT is H200-only. Found by code reading.

**Why:** the user challenged "no attention change?" twice and wanted evidence. Don't re-litigate; cite §32.

**How to apply:**
- Plan: v1 (pad) → v1.5 (length bucketing, ~2x) → v2 (packing + mask, ~2-6x).
- No numeric tests run yet. Pad-invariance and row-leakage tests come first.
