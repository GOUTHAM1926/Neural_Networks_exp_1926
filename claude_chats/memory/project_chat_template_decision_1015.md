---
name: project_chat_template_decision_1015
description: "DECIDED 2026-10-15: keep our chat template = Tülu 3's minus the \\n after an earlier answer's eos (the Table-13 53.0 'replace \\n w/ eos' reading); same template in SFT, DPO, RL, eval, inference"
metadata:
  type: project
---

User decision 2026-10-15 ("from now we will stick to that tulu3 chat template only", then chose "Keep ours (53.0 variant)"):
our template stays exactly as `sft/prepare_sft_data.py` builds it ("tulu-plain-v1"):
`<|system|>\n{s}\n<|user|>\n{u1}\n<|assistant|>\n{a1}<eos><|user|>\n{u2}\n<|assistant|>\n{a2}<eos>` (eos = 50256, a target; markers plain text, encoded piece by piece; generation prompt `<|assistant|>\n`, stop at 50256).

Facts checked 10-15 (sources): Tülu 3 real template (HF `allenai/Llama-3.1-Tulu-3-8B` tokenizer_config + paper App. B.3 Fig. 27)
= `eos+'\n'` after earlier answers, `eos` alone after the last ([open-instruct PR 376](https://github.com/allenai/open-instruct/pull/376), merged 2024-10-10, also masks that \n via mask_labels, dataset_transformation.py:1307-1368 @95a84bb).
Tülu 2 = `eos+'\n'` after every answer. Table 13 (§4.3.1): 53.0 replace-\n-w/-eos (exact text never published; ours matches the literal reading, not confirmed),
52.9 Zephyr, 52.8 Tülu 3, 52.6 Tülu 2, 51.6 Llama 3; Tülu skipped 53.0 "to avoid generation inconsistency with later steps".
Training targets are identical to Tülu 3's (the between-turn \n is never trained there). Consequence: never reuse Tülu's HF chat_template file as is;
all later stages call our own function; a custom HF chat_template + token-by-token check only if an outside tool (HF eval, vLLM, TRL) is used.
Known token traps: message ending "\n" + template "\n" -> [198,198] piecewise vs [628] whole-string; literal "<|endoftext|>" in user text.
The user's Google Doc "SFT : what it is" §3.2 wrongly called Tülu 2's shape "Tülu 3 exact" and §3.4 wrongly said Tülu 3 has dedicated role tokens (fix text given 10-15).
Special tokens (option A plain text vs B spare ids 50257-50303) is a SEPARATE open question: [[project_remind_1015_special_tokens]].
