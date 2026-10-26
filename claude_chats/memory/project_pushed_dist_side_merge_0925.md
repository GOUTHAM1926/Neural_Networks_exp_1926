---
name: project-pushed-dist-side-merge-0925
description: 2026-09-25 direct pushes to dist_side_merge — library ad10e7d (7 loss files) and BluTrain a264586 (F1 + #2 + trainer catch + [MEM] + submodule bump); box harness edits removed
metadata:
  type: project
---

**Library push:** `BlubridgeAI/Tensor-Implementations` `dist_side_merge` `8808f89..ad10e7d`.
- It holds the 7 loss files.
- It supersedes colleague commit `a407093`'s AccT partial fix in LossKernels.cu.

**BluTrain push:** `BlubridgeAI/BluTrain` `dist_side_merge` `03b94cf..a264586`.
- `ShardedOptimizer.cpp/.h`, `ZeroErrors.h`.
- The trainer's F1 try/catch and the `[MEM]` print.
- Submodule bump to `ad10e7d`.

**Why:** the user chose direct pushes over PRs ("we can revert"). Author was the company identity, and each remote was fetched first to confirm it hadn't moved. `sft_gautam` stays as our research branch; it diverges from dist_side_merge (base a4c80c1).

**How to apply:**
- Undo by `git revert`, never force-push.
- Colleagues must run `git submodule update --init` and delete the `*Loss*.o` files before rebuilding.
- The box copy (`rough-word-cracks-fin-02:/root/gautam_k/BluTrain`) no longer has the test harness. A 1-GPU re-test needs B=4 again.
- W=2 TESTED 10:33-10:55 (doc §19.10): overlap reduce=ON so on_grad_ready ran; reserved -2,810 MiB/GPU (= 5,616/2), live unchanged, +0.13% tok/s, loss in noise; F1 at W=2 both ranks [FATAL] exit 4. Still untested: B=8. CORRECTION 2026-09-26: Muon and ZeRO-v2 had the SAME F1 bug (they did not 'not apply'); fixed and pushed in 23ae1e5, see [[project-f1-muon-v2-pushed]]. #2 does not apply to them (v2 already converts directly).
- Paste gotcha: long single-line `-m` messages got truncated by the user's terminal. Use `git commit -F file` with a heredoc of short lines, and check with `git --no-pager log -1 --format=%B`.
- Doc: `loss_kernel_dtype_bugs.md` §19.9.

**Box handed over (2026-09-25 ~11:05):** `rough-word-cracks-fin-02` (2x H200) is now in use by colleagues. Do NOT ssh or run anything there. Everything needed is on GitHub or local (`profiling_reports/box_final_2026_09_25/`).
