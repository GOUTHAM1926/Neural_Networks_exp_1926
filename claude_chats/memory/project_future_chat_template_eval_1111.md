---
name: project_future_chat_template_eval_1111
description: "FUTURE (user stopped it 11-11 11:43, will say when): format-study eval on Ada (parity (c) + 54 evals, all set up), loss ablation (answers-only vs all-token vs weight 0.1, 3 seeds), questions for Sanjay"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-11T06:14:10.095Z
---

The user stopped the chat-template tests on 2026-11-11 at 11:43 IST: "no need now; document them as future / pending
tasks; I will say when". Work doc 45.188, plan Q&A 237-241.

**State on the Ada node** (10.101.0.203, /mnt/volgrp03/3rd_floor/gautam_fmt): everything is ready and kept.
- vllmenv, code/sft scripts, eval_data, nltk_data;
- code/parity_c.sh (copy in ~/Downloads/lc_box/fmt_study/);
- partial outputs evals_fmt/m1_smol_v11_s1 (no done marker) and parity_eval/c_*.
- Sukanya's dedup job uses GPUs 0-3; the user allows us GPUs 4-7 only.

**Parity (a), (b), (d), (e) are done** (m1_smol_v11_s1, 200 chats, fp32):
- loss HF 1.567215 vs vLLM 1.567182;
- greedy answers 92.5% identical HF vs vLLM;
- vLLM vs itself 96.5% identical;
- vLLM load 194 s; compute a few seconds.

**FUTURE tasks:**
1. Parity (c), then the 54 evals (ONLY='_v(1|7|10|11|12|17)_s[0-9]+$', EVAL_GPUS_FILE).
   - Decide how to treat (d)'s 3.5%.
2. Loss ablation: v11 x {answers only, all tokens, prompt weight 0.1} x 3 seeds + M2 / M7 / M9.
   - First add a prompt-loss weight to fmt_torch_sft.py.
   - About 3-4 h.
3. Questions for Sanjay:
   - are the special ids 1 / 2;
   - are the data's own system turns kept;
   - is outside text encoded with special tokens off;
   - does generation stop on ids 2 and 0.

**Recorded view:**
- Sanjay's ChatML choice = our v11.
- Keep the data's own system turns, with no default system prompt.
- His all-token loss result is one run each, so it is not proven.

Related: [[project_ada_node_format_study_1102]].
