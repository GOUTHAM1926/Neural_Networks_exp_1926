---
name: feedback_edit_local_push_pull
description: ALWAYS edit in the LOCAL repo, push from local, pull on the server. NEVER edit or push from the server box.
metadata:
  type: feedback
---

**Workflow, stated firmly 2026-08-20 after work was lost:** edit files in the LOCAL checkout ->
user commits and pushes from LOCAL -> server does `git pull`. **Never modify source on the server
box, and never push from it.**

**Why:** on 2026-08-19 the whole day's V5 work (1x2 cluster fix, KLOOP_UNROLL, tk/dw/ss arms,
pins.sh V5 update, §H27zdt-dz doc) was applied directly to the working tree on
`dark-leaf-wilts-fin-03`. Nothing was committed, the box was released overnight, and the new box
`deep-ball-purrs-fin-03` cloned `origin/main` at e949a2c -- everything from that day was absent
from the remote. Verda boxes are reassigned constantly, so a server working tree is scratch space
that can vanish without warning. See [[feedback_gpu_pin_device_1]].

**Root cause that made it possible: there was NO local checkout of the daily-work repo**
`GOUTHAM1926/blublas.git`. Locally there is only `Neural_Networks_exp_1926` (holds
`cublas_rev_eng_binary/analyzations_h200.md`, no `src/`) and two checkouts of the TEAM repo
`Blubridge-Tech/BluBridge-BLAS`. If a local blublas checkout is missing, CLONE IT FIRST -- do not
work around it by editing on the box.

**How to apply:** all Edit/Write go to the local checkout. Give the user commit+push commands
(never run them -- see [[feedback_git_no_ai_contributor]]), then give the server a `git pull`.
rsync to the box is only for pulling profiling artifacts DOWN, never for shipping source UP.

**IT HAPPENED AGAIN, 2026-09-09 — caught only at box-release time.** The §H163 split-K pin and the
13 §H158 `ncw1` configs existed ONLY on `95.133.254.191`, in
`/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations/BluBridge-BLAS`. Zero local copies had
them. Rescued minutes before the instance was killed, into
`box_rescue_2026_09_09/` (4 tarballs, 2,276 files, 25 MB, all 5 distinct source trees).

**Two lessons:**
1. **Verify the local copy with md5, not by assuming the rule was followed.** The authoritative
   template was `4b0655fd54893c46ac6bda2197162cae` with 13 `ncw1` hits and 2 `BLUBLAS_H163_OFF`
   hits — grep for a marker string, don't just check the file exists.
2. **`find -maxdepth 8` from `~/Downloads` does NOT reach these files.** The path is
   `.../BluBridge-BLAS/src/level3/batched_gemm/BF16/SM90/<file>` = depth 9+. A shallow find
   reported "not present locally" for trees that WERE present, and would equally report
   "present" while missing the modified one. Always use `find / -name <file>` with no depth cap,
   or search by the marker string.

**Standing check before any box is released:** `find /root -name "<key source file>" -exec md5sum
{} \;` on the box, and confirm every distinct md5 has a local counterpart before saying it is safe.

**THE PATH, so it is never searched for again: `~/Downloads/blublas`.**
I concluded "no local clone exists" on 2026-09-12 from a `-maxdepth 4` find and was wrong -- the
same shallow-search mistake this note already warns about. See
[[project-local-blublas-checkout-path]].

**USER OVERRIDE 2026-09-30 (for team-branch patches like F4 on `dist_side_merge_BLUCCL`):** the user wants clone -> analyze -> patch -> test -> commit -> push ALL on the H200 box ("are u mad, I told we will clone the remote bluccl branch and apply and test patch on that itself, not local"); locally only fetch/pull afterwards. Follow that. Mitigation that does not contradict it: pull the diff + logs down after every step ([[feedback-pull-run-artifacts-immediately]]), and make sure the push happens before the box can be released.
