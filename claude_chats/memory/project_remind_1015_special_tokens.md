---
name: project_remind_1015_special_tokens
description: "REMIND 2026-10-15 MORNING, start the day with this: special tokens vs plain-text role markers + chat templates (work doc §44.6); B-1 test 2 still pending"
metadata:
  type: project
---

**At the start of 2026-10-15, before anything else, remind the user:** "we start with special tokens and chat templates
(work doc §44.6)". User's words (10-14 evening): "we will start with this only ok? ... regarding those we will start tmrw
morning itself". It runs in parallel with the attention-kernel work.

Task = answer the 5 questions in `sft_work_documentation.md` §44.6.1: add `<|system|>/<|user|>/<|assistant|>` + end-of-turn
as special tokens or keep plain text (`<|assistant|>\n` = 7 GPT-2 pieces incl. `ass`+`istant`); if not, how the model
learns chat/turn start and end; if yes, when and the full process; how others do it and how it's going; chat templates.

Already checked 10-14 (in §44.6.2): tiktoken gpt2 n_vocab 50257, only special = 50256; model has 50,304 rows, so 47 spare
ids 50257-50303; ours = "tulu-plain-v1" (prepare_sft_data.py, encode_ordinary per piece, 50256 appended as a target);
Tülu 3 (Llama-3.1-Tulu-3-8B tokenizer_config) also uses plain-text markers + real `<|end_of_text|>`, only added `<pad>` 128256.
Still to check (§44.6.3-44.6.5): Llama 3 / Qwen ChatML / Zephyr / TRL setup_chat_format or clone_chat_template (TRL commit 478624d);
our output layer tied? loss over 50,304 or 50,257 rows?; HF apply_chat_template, generation prompt, assistant-only loss.
Cite every claim with a link or file:line ([[feedback_verify_claims_and_benchmark]]); simple English ([[feedback_no_analogies]]).

**Also pending from 10-14:** B-1 test 2 (speed race g4 vs b1, `gpu_runs/b1_1014/t2_speed.sh`, ~4 min) did NOT run: all 8
GPUs were taken by Sanjay's production training at ~19:50. Ask before running; check the GPU is free first.
Related: [[project_state_2026_10_13_beat_rivals]].

**RESEARCHED 10-15 (work doc §45.6).** Recommendation CHANGED to B: special ids 50257/50258/50259 for SFT AND DPO, rows init = mean of
trained rows 0-50256 (transformers mean_resizing / Unsloth / imone fix), eos stays 50256, content always encode_ordinary; decide with ONE
A-vs-B packed SFT ablation when GPU allows. Facts: Llama 3 BASE chat-token rows are ZERO (imone card) -> NaN risk; SmolLM2 base has
<|im_start|>/<|im_end|> at ids 1/2; Qwen2.5 & OLMo-2 base tokenizers also have ChatML ids; TRL clone_chat_template adds tokens at SFT;
OLMo 3: keep special tokens out of midtraining. Special tokens need NO tokenizer retraining (matched before BPE).
