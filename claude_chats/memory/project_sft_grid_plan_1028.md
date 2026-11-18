---
name: project_sft_grid_plan_1028
description: "135M SFT grid plan in work_docs(... ) /plan_grid_search_sft.md; tokenizer + factor 1 DECIDED; factor 4 (component 9) plan v3 (4.28, 10-29 evening): 15 arms x 3 mixes (m1, m2, d8) + LR check, M5b/M8 added, PROPOSED; X1-X3 side experiments; appendix A = non-grid Q&A"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T16:48:25.999Z
---

The plan file is `work_docs(mark-down - (.md) files ) /plan_grid_search_sft.md`. The folder name ends with a SPACE.
It was first created by mistake in a twin folder without the space; on 10-28 the user had it moved and renamed.

**User rule (10-28, repeated 10-29 evening):** document EVERYTHING in that one file:
- each question the user asked, and the answer with proof;
- who did what and who did not; what nobody measured; what the grid tests and why;
- every team's rendered template, the timeline, side experiments;
- even off-topic answers from the same thread (appendix A: KernelBench, preproduction-135M, RoPE for DPO, push deferral).

Mark a factor DECIDED only when the user says so.

**Decided:** the tokenizer = ours (component 0, no grid tests); factor 1, domains: Aya is out of the mixes, with one +Aya arm.

**State at 10-29 evening: factor 4 (chat format) plan v3 = §4.28, PROPOSED, waiting for the user's OK.**
- Arms 1-14 of 4.21 (layout Tü/Ch x encoding p/s x end 0/S/X, + 2 "as is" team copies) + arm 15 = no markers (control,
  from Price of Format's Natural Instruction / OLMo 3 midtraining format; never production).
- F1 on 3 mixes: m1 (multi-turn 61.0% of chats, system 15.1%), m2 (3.8%, 0.1%, 2.58 end targets per 1k tokens, 35% math),
  d8 (closest to final). Swap rule after F0 measures all 12 mixes. 135 runs + LR check (arms 1, 5, 11, 13 at 5e-5 and
  2e-4 on m1, 24 runs). F2 start values on m1+m2; F3 top 3 + A on 2 mixes at full length (24 runs, ~6.4 h); F4 at final
  settings. Total ~13.6 h on 4 GPUs; early stop after F1 ~4.7 h.
- Measurements M1-M7 + M5b (benchmarks with no template) + M8 (diversity: MiniLM semantic, distinct-n, self-BLEU,
  first-50-token entropy).
- Same chats, not same steps (m1 arm A 90,954 rows vs arm B 88,641); one shared chat list per mix (assert kept_row_ids).
- Decision rule includes data-independence (no significant opposite sign on another mix).

**Key facts established 10-29 evening (with proofs in `profiling_reports/sft_chat_format_2026_10_29/`):**
- Price of Format (2505.18949) DOES train (Llama-3.2-3B, 5 formats); its code very likely has pad = eos with TRL's old
  default collator -> every eot_id label removed; no prompt mask. Corrections 12-16.
- Tülu 3 (open-instruct) masks labels BY POSITION; it added <pad> because Llama 3.1 has none and asserts pad != eos.
- OLMo 3 midtraining kept "messages joined by double newlines" (no markers, no mask).
- Injection demo with our ids: inj_demo.py (plain marker typed = identical ids; ChatML parsing ON = forged ids 1/2,
  OFF = 7 ordinary pieces).
- Marker-like text in answers: 14 / 34 / 14 chats in m1/m2/m3 (mix_lt.py; the id-pair version was vacuous).

**10-29 night:** the user ACCEPTED the 4 Price-of-Format items (M8, M5b, arm 15, end-label check; build details at the end
of 4.28); the rest of 4.28 is still PROPOSED. Injection re-explained for zero background in 4.29 (when/who/why, real
cases remoteli.io 2022 and Bing "Sydney" 2023, the switch ON/OFF with `[1]` counts); pad = eos in 4.30 (Tülu's SFT is
by position; the risk is by-number code: their RL/preference code, HF tools; HF's own error message suggests pad = eos;
OUR pad = end = id 0, so never pass our data through by-number code). Open check: TI DPO code's pad handling.
Proposed order of the remaining factors (top of the plan file): 5 → 2, 3 → 6-9 → 10, 11, 12, 13.

**10-29 late night, plan 3.1 (§4.32):**
- **M9 added:** XSTest refusal and over-refusal, with and without the template (ChatBug's format mismatch).
- **ChatBug's message overflow** is now an M7 variant.
- **M7, M7b and M9 are decision criteria.**
- **Every added structure id must be SPECIAL.** I tested it: the switch does NOT protect "added, not special" ids.
- **Think and tool token table:** Qwen3, DeepSeek-R1, SmolLM3, OLMo 3, Hermes 3, Llama 3.1, Mistral v3, gpt-oss.
- **DPO script finding:** decided after these tests (user).
- **For colleagues:** `work_docs/.../chat_template_special_tokens_tests.md` + `.docx`, written from zero in the user's voice. `sft_work_documentation.md` §45.107 points to it.
- **Status:** component 0 DECIDED; component 9 plan complete, waiting for the user's OK. Nothing has been run.

**10-29 last: plan v4 consolidated in §4.38** (15 arms with exact formats, 3 mixes, 8 fairness and safety rules,
stages, M1-M9 + P0, the decision rule, and a change log), the **F0 build plan in §4.39** (13 steps, each with a file and
a check), and the **recovered 10-28 answers in §4.36**.
- **Two rules added in implementation review:**
  - template marker TEXT in the plain arms is converted with the switch OFF, or `<|im_start|>` silently becomes id 1;
  - generation stops on the arm's end id AND on id 0.
- **Ten incident stories** are in §4.37 and the colleague file Part 4.15.
- **Hidden size is 512** (base_hf `config.json`). HF export is untied; BluTrain is tied.
- **Waiting:** the user's DECIDED on §4.38, then F0. Box and GPU steps need the user's OK each time.

**Version 4.1 (§4.40-4.41).**
- **Full product path:** the server sets the roles, so a typed "system" ends up as `[1] user`.
- **Source audit, with outputs in `profiling_reports/sft_chat_format_2026_10_29/source_audit/`:**
  - The data comes only from official, pinned sources: smol-smoltalk f73fe857, Tülu 0225 d91a0785, Dolci bd3c8f3a.
  - lm-eval loads the official benchmark datasets. PIQA comes from baber/piqa (official sizes).
  - M8 WritingPrompts comes from Facebook's archive. M9 uses Paul/XSTest + the authors' strmatch classifier.
  - The HF token gets 403 on Meta and Google gated repos (licence not accepted on that account).
- **F0 steps 14-16 added:** source manifest, evaluation data, hand-written prompt files reviewed before F1.
- **Next session:** the user wants to continue the tests "tomorrow". Ask for DECIDED on §4.38, then build F0 (code and
  CPU), with box and GPU steps only on the user's OK.

**Side experiments, noted:** X1 think-SFT warm start (factor 13); X2 750B vs 900B (factor 11); X3 pretraining text
inside SFT data (factors 2 and 9); X4 safety recovery examples (Qi et al. 2406.05946; safety part of the data
factors). Plan §4.42-4.44: the fake assistant turn, the official names, how a cheater knows the format, and jailbreaks
with the when-fooled table. Q&A up to 111.

**10-30:** Q&A 120 + table in §4.47 and colleague Part 6.3: in ChatML versions only ids 1/2 are special, role words
stay plain (SmolLM2/Qwen); whole-role special ids = v4-v6 only (new ids 49152-49154 in the spare rows); plain
`<|im_start|>` = 44 108 331 79 6501 108 46 but its first piece can merge ('.<' = 15133), so never assert 7 pieces.
Offered (no answer yet): an extra arm with ChatML ids + each role as its own special id. docx now builds with
`pip install --target <scratchpad>/pylib_docx python-docx` (the scratchpad copy gets cleaned).

**10-30 later:** Q&A 121 + "More combinations" in §4.47 + corrections 18-19. Cohere Command R = start id + role as one
special id + end id (all special=True, Cohere/Command-nightly tokenizer); Phi-3 ends EVERY message (not exact v5);
GLM-4 = role special ids, no end, stops on next role id; DeepSeek-R1 role ids special=False. A format = 5 choices
(start, role, which messages end, end id, plain/special); the 15 versions tie 3 together. PROPOSED v16 (Cohere) +
v17 (Phi-3 exact), ~1h40 more in F1; optional v18 GLM-4, v19 Zephyr, v20 Llama 3. Role ids first-17 vs 49152+: same
math (identical never-used rows, pc overwrites). Evidence: profiling_reports/sft_chat_format_2026_10_30/.

**10-30 plan VERSION 5 = §4.48** (colleague file 6.0). The user: format study SEPARATE and FIRST (with 4 built-in
checks: 3 mixes, LR check, full-length confirm, re-check at final settings); EVALUATION ON vLLM (own venv on the box)
after a parity test: lm-eval README says vLLM differs occasionally, HF = reference; vLLM FAQ says batching gives
logprob noise, fp32 reduces it; one backend for all versions. Choices A-N table; proposed v16 Cohere, v17 Phi-3,
v18 GLM-4, v19 Zephyr, v20 Llama 3 (new id 49155), v21 system merged into user (Mistral v0.1), v22 default system
(SmolLM2/Qwen2.5); suggestion = all 7 (~+3 h). New rules 9-12 (rule 9: eval ids == training ids; lm-eval would make
v7-v9 template text into ids 1/2). Steps renamed: 1 build+check, 2 backend check+smoke, 3 screen, 4 start values,
5 confirm, 6 re-check. Time: steps 1-5, 22 versions + vLLM ~1 day + ~24 h (old: ~1 day + ~29 h). Q&A 122.
WAITING: which of v16-v22; DECIDED on 4.48; box/GPUs + OK to install vLLM.

**10-30 v5.1 = §4.49 (user decisions + step-1 progress).** User: all 22 versions IN; ONE LR 1e-4 (LR check removed;
final re-check covers it); plan approved ('nice plan'), mixes m1/m2/m3; 'cross-check & re-verify multiple times'.
Screen settings = stage-1 launcher: Muon, cosine, WARMUP 3% (not 10%, correction 20), ~128 chats/step, 1 GPU/run,
100k chats x 1 epoch (stage-1 launcher hard-codes 2 epochs -> new launcher needs an epochs setting).
CPU code in ~/Downloads/BluTrain_push/sft (uncommitted): chat_formats.py (22-version spec; self-test vs plan lines;
v1/v4/v11 == old templates on 20k chats), build_format_study.py (shared chat list, all 22), verify_format_study.py
(independent re-render checker; 3 negative controls caught), format_init.py (start weights from ckpt_A; v4 == ckpt_B,
v13 == ckpt_A byte-identical). hf_to_blutrain.py restored byte-identical. Data: ~/Downloads/sft135/data_fmt/<mix>/<v>/
{train_screen,val} (build_fmt_all.sh, logs/fmt_*); inits: ~/Downloads/sft135/init_fmt/v1..v22. GPU estimate: ~6-7.5
min/run training, ~22 GPU-h for 198 runs; evals unmeasured (vLLM target 5-10 min/model). TODO step 1: P0 table,
export per version + eval runner on token ids (rule 9), measurement scripts + dry runs, prompt files (show user),
screen launcher (1 epoch), source manifest. Then box/GPU steps only with the user's OK.

**10-30 later (step-1 progress, §4.49):** data m1/m2/m3 x 22 built (19 GB) and VERIFY OK on every chat (independent
checker). P0 done (p0_protect.py; plain v1-3, v7-9, v15 can fake an answer START; every END protected; v19 protected
by its id 0). fmt_lm_eval.py = lm-eval 0.4.12 'vllm-fmt' class (chat -> ids via chat_formats; stop ids added);
self-test vs independent renderer OK + negative control. blutrain_to_hf.py --format-version (stop ids, full tokenizer,
raise-exception template); v13 fake export -> Navin file bit-identical. Remaining step 1: measurement scripts M1-M9
(dry runs need the box), prompt files M3/M7/M7b (SHOW USER first), screen launcher (1 epoch), source manifest.
No transformers/torch/vLLM on the laptop; box lm-eval version unknown (check in step 2).

**10-30 afternoon: step 1 (CPU) essentially done (§4.49).** FINDING: trainer has NO data-order seed (rows in file
order) -> seeds = row orders: fmt_reorder.py, data_fmt/<mix>/<v>/train_screen{,_s2,_s3} (55 GB), all verified by
verify_format_study.py --splits. Loader skips last N mod B rows (<0.1%). run_sft.sh got EPOCHS (default 2).
New: fmt_eval_data.py (eval_data/: XSTest+classifier, M8 200 prompts, MiniLM, manifest), fmt_probes.py
(eval_data/probes/probes_review.md -> USER MUST REVIEW), fmt_measure.py (runner; stub-tested all 22),
fmt_lm_eval.py --check-dump, init_export.py (step-0 HF; v13 bit-identical), fmt_screen.sh (gate/runs/grid; dry-run
OK; deletes ckpts+final after HF export), fmt_evalq.sh (step0/queue; dry-run found ${cmd} bash bug, fixed),
fmt_manifest.py, gate_fmt/<v> (numpy ref loss, gate pass <0.2%). Next: user reviews probes; then box steps each
with OK: copy, build, vLLM env, gates, vLLM-vs-HF parity+timing, smoke, screen.

**10-30 evening re-verification (§4.50, Q&A 125, correction 21):** fixed: M2 sample (first 300 -> spread), M2 cap,
M4 split (normal/attacks/sampled/other), missing parity script (fmt_parity.py + hf-fmt + HfEngine), careless
tokenizer v20 renumbering bug, smoke stage (smokeruns, train_smoke), analysis script fmt_analyze.py. Real-model CPU
tests (scratchpad evallib: torch cpu, transformers 5.22, lm_eval 0.4.12): HF step-0 vs numpy gate ref within
0.0014% (v1/v11/v18); lm-eval hf-fmt 4 tasks + dump check 48 encodings OK; fmt_measure HF engine full run OK.
OPEN USER DECISION: decision rule -- plan's rule flags 6.3% on noise vs pooled rule 0.4% (fmt_analyze --rule).
Box env needs lm_eval[vllm,ifeval]. No agents/workflows used.

**10-30 late: stop/resume built + tested (§4.51):** fmt_screen.sh stop | stop-gpu N, GPUS_FILE re-read, READY = done,
restart redoes stopped runs (2 passes), setsid process groups; fmt_evalq.sh same (done flag, EVAL_GPUS_FILE);
fmt_analyze reads only done evals. Fake-trainer tests passed. Recommended decision rule: pooled (awaiting user).

**How to apply:**
- Wait for the user's OK on 4.28, then flip it to DECIDED and build F0.
- Then continue factors 2, 3 and 5-13 one by one in that file.

Related: [[project_sft_135m_1027]], [[feedback_simple_english_no_blabber]], [[feedback_docs_keep_full_detail]], [[feedback_explain_setup_before_numbers]], [[feedback_vacuous_checks]].

(Moved from the MEMORY.md index line on 2026-11-05:) the work_docs folder name ends with a SPACE ("work_docs(mark-down - (.md) files ) /").
