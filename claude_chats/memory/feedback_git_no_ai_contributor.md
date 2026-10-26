---
name: feedback_git_no_ai_contributor
description: User does their own git commits/pushes; never commit/push or add AI co-author trailer
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 60a6dc55-8dda-45fe-864f-235f3482492e
  modified: 2026-07-30T10:44:14.980Z
---

The user runs **all `git commit` / `git push` themselves** and does NOT want the AI showing up as a contributor/author on the repo.

**Why:** it's their repo (GOUTHAM1926/Neural_Networks_exp_1926); the work should be attributed to them (`Goutham_Reddy_1926`), not to Claude.

**How to apply:** only **edit/stage** files in the working tree (Edit, `git rm --cached`, etc.) — then hand the user the exact `git add/commit/push` commands to run. Never run the commit/push myself, and never add a `Co-Authored-By: Claude` trailer. Author identity = whoever runs `git commit`, which must be the user.

**This covers BOTH repos — code AND docs.** 2026-07-30: I assumed the docs repo (`Neural_Networks_exp_1926`) was mine to commit directly and ran `git add -A && git commit && git push` on it. Two failures at once: (1) it violated this rule and the commit carried the user's name for something they didn't do, and (2) `git add -A` from a subdirectory swept in **1706 files / 5.2M lines** — an embedded git repo, `BluTrain_bf16_run/`, `sweep_results_cublas/`, profiling output — plus committed 164 pending `comparision_bench` deletions the user had not decided to commit. The user fixed it themselves with a force-push.

**Also:** never run `git add -A` / `git add .` in these repos. Stage explicit paths only — the working directories contain other checkouts and large generated output. See [[project_active_repo.md]].
