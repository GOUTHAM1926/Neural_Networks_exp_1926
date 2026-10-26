---
name: project-sft-gautam-branch-facts
description: "Where sft_gautam really stands (checked 2026-10-06) — remote branch deleted 09-29 and kept as tag archive/sft_gautam; sft_research/BluTrain's TI still has a LOCAL sft_gautam branch with a stale origin ref"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-06T05:55:00.674Z
---

These were verified on 2026-10-06 after I wrongly told the user that `sft_research` held "nothing of ours".

**Folders and their git state:**
- `~/Downloads/sft_research` is NOT a git repo. Git walks up to an empty repo at `~`, which has 0 commits and a remote of GOUTHAM992003/ML_implement.
- **`~/Downloads/sft_research/BluTrain` IS our team repo** (BlubridgeAI/BluTrain, on `master` at 0095fee).
  - Its `Tensor-Implementations` is checked out on the **local branch `sft_gautam` (1ed2ac3)**, which tracks `origin/sft_gautam`.
  - The parent has uncommitted changes: the TI pointer moved to 1ed2ac3, and `bluscript_zero_wsd.cpp` (+21/−5). That is the old 09-22 bf16-logits edit to the OLD trainer, and it is superseded by `BLU_BF16_LOGITS` in `bluscript_zero_ckpt.cpp` on dist_side_merge.

**The remote branch:**
- `sft_gautam` was DELETED on GitHub on 09-29 by the user, and kept as tag `archive/sft_gautam` → 1ed2ac3.
- Evidence is in `~/.bash_history` lines 1930-1932: `git tag -a archive/sft_gautam 1ed2ac3 …`, then a push of the tag, then `git push … --delete sft_gautam`.
- The local `origin/sft_gautam` is stale, because a plain fetch never prunes. The tag exists locally in `~/Downloads/BluTrain_sft/Tensor-Implementations`.
- Both org repos are private, so `git ls-remote` without the user's token says "not found". I cannot confirm GitHub's live state myself.

**Is the work in dist_side_merge?** Checked file by file, 1ed2ac3 against dist_side_merge d96b991:
- the **7 SFT loss files are IDENTICAL**;
- the 2 loss tests are only local, in the BluTrain_sft clone (hidden with `info/exclude`);
- the `ReductionImpl.h` / `ReductionUtils.cpp` changes were 100% comments (0 code lines).

**So the SFT work continues on dist_side_merge:** `~/Downloads/BluTrain_sft` (BluTrain 909ab75 + TI d96b991). It has the task-2 `BLU_INIT_FROM` code, uncommitted (md5 ad7c5f…).

Related: [[project-sft-flow-walkthrough-and-decisions-1005]], [[feedback-commit-identity-company]].
