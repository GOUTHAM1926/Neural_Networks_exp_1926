---
name: feedback_branch_rule_sft_vs_experimental
description: "Branch rule (user 2026-10-30): SFT work + optimizations that affect SFT go on dist_side_gautam_lottn; experimental big reworks (matmul fusion scopes etc.) live separately (other branch / local folder)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-30T09:12:42.772Z
---

User, 2026-10-30: everything about SFT, and every optimization that affects SFT (logits on trained tokens only,
long-context checks, bfd_split, attention masked-work cuts, metadata prep), is done on OUR branch
`dist_side_gautam_lottn` (BluTrain + TI; laptop `~/Downloads/BluTrain_push`). Experimental, big-rework work such as the
matmul fusion scopes (C1-C7, C2F, C5/C5b, L1, C4 in `~/Downloads/BluTrain_fuse`) is kept SEPARATE and independent:
another branch or a local folder, never mixed into lottn.

**Why:** lottn is the branch that gets pushed and merged for SFT/pretraining users; big experimental reworks must not
leak into it before they are proven.
**How to apply:** before editing, name which tree the change goes into; SFT-affecting -> BluTrain_push (lottn);
fusion/rework -> BluTrain_fuse or a new local copy. The user still does all commits/pushes
([[feedback_git_no_ai_contributor]]). Related: [[project_matmul_epilogue_audit_1024]], [[project_kernel_plan_1030]].
