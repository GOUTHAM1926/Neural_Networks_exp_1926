---
name: project_loss_sum_fix_1030
description: "Loss-sum >65,536-row bug FIXED (B2) + long-chat split + 135M files: PUSHED 10-30 to lottn (TI 6aac3bf, BluTrain 334ecbb); next preproduction-135M + tell Adhitya"
metadata:
  type: project
---

2026-10-30, P022 H200 GPU 6. Full record: sft_work_documentation.md §45.108 parts 12-14; files in
`profiling_reports/loss_sum_fix_2026_10_30/` (diff, commit_message_TI.txt, sources, logs, nsys).

**The fix is B2.** In `sum_reduction_kernel`: one load while gridDim*blockDim >= n (the old code exactly); otherwise a
64-bit strided loop.

**Why B2:** 10 reductions were measured on the GPU (time, nsys GPU time, bits, determinism, memory).
- B2 is the only one with today's bits up to 65,536 rows, run-to-run determinism, no new memory or state, and today's
  GPU time.
- The always-on loop (B) cost +0.16-0.20 us per launch.
- CUB has the lowest GPU time but changes the bits.

**Results:**
- Test 154/154 (16 failures before).
- Bits identical up to 65,536 rows.
- Memory and sanitizer identical to before.
- Only `sum_reduction_kernel`'s SASS changed.

**State:**
- B2 and the test (with group H) are in `~/Downloads/BluTrain_push/Tensor-Implementations`, uncommitted.
- The test is hidden by the local `info/exclude`, so it needs `git add -f`.
- Do NOT add `test_varlen_attention.cu`.
- `LossKernels.cu` on preproduction-135M is byte-identical to lottn, so the commit cherry-picks cleanly.
- **Trainer check DONE on P022 GPU 6 (21:12-21:28).** Same binary, LD_LIBRARY_PATH swap:
  - 81,920 tokens per micro-batch: before 1.300650 vs B2 1.606812 (numpy ref 1.606273);
  - repeats show the trainer's backward is not bit-deterministic (~1e-4), the same before and after;
  - peak memory identical.
- Pushed lottn alone cannot run 135M (its loader hard-codes 50256). User: push the loss fix + the 19 local 135M files
  + the long-chat split TOGETHER.

**How to apply:** after the user pushes, bump BluTrain lottn's TI pointer, then do the preproduction-135M route.
Tell Adhitya: his run trees have no .git, so he needs a fresh clone.

Related: [[project_kernel_plan_1030]], [[project_adhitya_versions_proof_1030]], [[project_p022_box_1030]].

**Long-chat split (same push), state 22:30:**
- `prepare_sft_data.py --long-chats drop|split_trl|split_turns`, `--ctx-frac`, `--workers N` (parallel, 19x faster,
  byte-identical);
- `seq_complete.bin`, read by `SftDataLoader.h`;
- `ref_val_loss.py` copies it; `test_long_chat_split.py` = all tests;
- `run_sft.sh` sets `BLU_EMB_BWD=segmented`.

All verified: tests, real-data decode, GPU gate 1.780719 vs numpy 1.779986, 3 negative loader checks refused.
Docs updated (README, packing, loss docs mirrored; plan X6/X7, Q&A 137-138; work doc parts 17-18).

User decisions: long-question chats -> option A only (bigger T), others = future research X6; T > checkpoint length
-> extension first (X7).

PUSHED 2026-10-30 by the user: TI lottn f7aa40f..6aac3bf; BluTrain lottn 5c0999c..334ecbb (24 files incl. TI pointer 6aac3bf); verified on GitHub. Kept local: fmt_* / component-9 scripts, init_export.py.
Never add `dist/Context_Parallelism` or `test_varlen_attention.cu`.

**preproduction-135M DEFERRED (user, 10-30 ~23:05):** push there only after the remaining optimizations; the user
tells Kiruthik.
- Dry run: TI cherry-pick of 6aac3bf is clean; BluTrain 334ecbb conflicts only on the TI pointer (resolve with
  `update-index --cacheinfo 160000,<new TI>,Tensor-Implementations`).
- Use worktrees /tmp/ti_pp135 and /tmp/bt_pp135.
