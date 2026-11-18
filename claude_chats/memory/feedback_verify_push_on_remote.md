---
name: feedback-verify-push-on-remote
description: "After every push, verify on the REMOTE that each change (every file and hunk) is really there; every change must be explained in code comments AND in the commit message — REMIND the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-31T11:22:10.204Z
---

The user (2026-10-31, before pushing the long-context fixes): after pushing to the remote, check that ALL pushed changes and
everything we did are present in the remote branch, change by change; and every change must be described in code comments
and in the commit messages.

**Why:** pushes have silently missed files before (submodule pointers, uncommitted working-tree edits, wrong branch), and a
change with no comment / commit text cannot be reviewed or traced later.

**How to apply:**
- Before the push: list every changed file per repo (git status / diff --stat, both BluTrain and the Tensor-Implementations
  submodule), and check each change has a comment at the code saying what and why (date), and that the draft commit
  message names every change (file + one line why).
- After the push (the user pushes): fetch the remote and verify each change ON THE REMOTE, not locally: `git fetch` then
  `git diff <local-commit> origin/<branch>` must be empty, `git log origin/<branch>` shows the commits, the submodule pointer
  on the remote equals the pushed TI commit, and spot-check each changed file with `git show origin/<branch>:<path>` (grep
  for the new code / comment). Report the check file by file.
- Remind the user of this check every time a push is planned. See [[feedback-git-no-ai-contributor]] (commit identity) and
  [[project-long-context-fixes-1031]] (the push list).
