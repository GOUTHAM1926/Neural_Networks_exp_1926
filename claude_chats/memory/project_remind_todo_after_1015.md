---
name: project_remind_todo_after_1015
description: "10-16 MORNING FIRST: learning session on how attention blocks compute masked work, segstart vs seqlens (work doc §45.10, explain from zero with a tiny drawn grid); then 2 GPU tests (§45.8.1) when GPUs free; full to-do §45.8"
metadata:
  type: project
---

**UPDATE 10-16 ~11:45 IST: user moved the learning session to LATER; the 2 GPU tests are DONE (PASS, §45.11). NOW: optimize the packed
(seqlens) attention kernels, v62 forward + Vj2 backward (baseline BENCH + where-time-goes first).**

**(was) 2026-10-16 MORNING, FIRST OF ALL (user's decision 10-15 evening):** the learning session of work doc §45.10 --
how attention blocks compute masked work, why seqlens beats segstart on middle/long chats (every query block of a chat
reaches back to the chat's first key block, which segstart shares with the previous chat) and why segstart wins on tiny
chats (whole block per tiny chat + fixed cost per work piece). The 10-15 chat explanations did NOT land: start from
zero with a hand-drawn 16-token row / 4-token blocks, count every pair, then the real sizes and tables (§45.9-45.10).
Real terms only, no analogies. The 10-15 EOD stays unchanged (user's decision).

**At the start of the next session, before anything else, remind the user** (their request 2026-10-15 evening:
"document and remind all the tasks and tests to do and note everything"). The user will say when GPUs are free.

1. **Two mandatory GPU tests of design A** (box already built 10-15 18:14, `tva_a` + `bluscript_sft_bin`; commands
   copy-ready in `sft_work_documentation.md` §45.8.1). Ask the GPU index first (10-15 used GPU 1).
   - Test 1: `VARLEN_QUICK=1 VARLEN_ONLY=G11 ./tva_a` under Vj2 (~5-10 min, unmeasured). Pass = `backward mode: Vj2
     (default)`, `failed 0`, `ALL RUN CHECKS PASS`.
   - Test 2: packed e2e, `BLU_SFT_DATA=~/gautam_k/sft_data/e2e_packed`, B=1, 147 rows/step (~3 min). Pass = loader line
     `PACKED rows (seq_lengths.bin: 500 chats ...)`, step-0 val ~1.4450 (v1 1.445038), grad norm 0.7935, exits by itself.
   - Then pull both logs to `profiling_reports/`, write results in §45.7, tick §45.8.
2. **Then the to-do list, work doc §45.8.2-45.8.6:** re-run stage-0 sample runs packed (AdamW 2e-5 + Muon sweep),
   packed16, step-count ablation (runs 1/2a/2b/2c/3/4 x 2 seeds), special-token decision with the DPO team (+ row norms,
   their `sft_tokenizer.py`, their `allowed_special` bug) and the tokens ablation, pretraining step 0 Vj2 vs Vj1d; later
   kernel tests 3.1-3.9 (incl. B-1 test 2, sanitizer on Vj2-A, BENCH, full 605); paused forward optimisation; future
   tasks (bfd_split, DFT/focal, Q4/5/6/11/12/13, Kruthik embedding question, B300 ignore_index, casting #1/#3/#4,
   matmul racecheck, re-explain ignore_index).
3. **Commit risk:** nothing committed since 2026-09-28 (TI d96b991, BluTrain_sft 909ab75); the exact uncommitted file
   list is in §45.8.6. Offer the push block (the user commits/pushes).

Related: [[project_packed_sft_pipeline_plan_1015]], [[feedback_minimal_tests_finish_pipeline]], [[feedback_ask_before_every_gpu_run]].
