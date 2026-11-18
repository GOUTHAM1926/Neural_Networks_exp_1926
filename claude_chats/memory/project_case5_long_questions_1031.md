---
name: project_case5_long_questions_1031
description: "DECIDED 10-31: drop is the ONLY long-chat method; split removed from dist_side_gautam_lottn working tree (proof byte-identical), docs written; user to commit+push 5 sft paths"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-31T06:34:00.471Z
---

2026-10-31: after the 10-30 split push (BluTrain 334ecbb) and the 10-31 studies (complex case-5 variant; the user's simple split), the user DECIDED: every chat longer than the row is dropped, the ONLY method — no option, no default, no second algorithm (like torchtitan). Reasons: SFT must teach whole answers that stop at the end token; cut answers and multi-turn chats cut into single turns confuse the model; more tokens of broken chats don't help; longer T is the permanent fix.

Done in ~/Downloads/BluTrain_push (NOT committed): removed --long-chats, --ctx-frac, split_trl, split_turns, spans, seq_complete.bin (prepare_sft_data.py, SftDataLoader.h, ref_val_loss.py), deleted sft/test_long_chat_split.py; README §3.2 rewritten; packing doc §7.8 = full history + reasoning + numbers (mirrored to work_docs). Proof: pushed vs new script data files byte-identical (2 mixes, 2 templates, 1 == 8 workers); self-tests pass. Records: profiling_reports/long_prompt_variants_2026_10_31/drop_only_check/, work doc 45.109-45.112, plan Q&A 139-148.

**Why:** the user's call on SFT intentions; quality over token count.
**How to apply:** never propose splitting again unless asked. The ONLY future option the user allows (simple): exactly TRL bfd_split + ONE rule: if ANY prompt (system + question of ANY turn, first or later) is as long as a whole row (T+1) or longer, the WHOLE chat is dropped (single and multi-turn); every other chat is split by bfd_split as it is (334ecbb split_trl lacks this rule). Prep limit lifted 10-31 to multiples of 256 <= 131,072 (+ histogram to T, run_sft.sh reads T from meta.json); prep run at 4k-128k on the long-source mix, drops = 10-30 measurement exactly. Next: STANDALONE kernels at 16k-128k first (correctness/precision then benchmarks vs rivals), THEN the 135M model; needs GPU go-ahead. Facts: torchtitan drops; Megatron-LM truncates. PUSH WAITS (user 10-31): push only after the long-context tests pass at 16k/32k/64k/128k (task T1: kernels correctness + precision + benchmarks vs rivals, lift the prep T<=4096 limit, trainer smoke); then push drop-only + limit lift together (not dist/Context_Parallelism, not TI). The drop reasoning + prompt-only examples live as a COMMENT at the length rule in _examine() (user: document in the code itself, not only in docs). See [[project_loss_sum_fix_1030]].
