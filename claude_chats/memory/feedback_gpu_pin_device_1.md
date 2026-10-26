---
name: feedback_gpu_pin_device_1
description: "Verda boxes change constantly and the user's allocated GPU index changes with them; pin to the index the user names (10-14 14:13: GPU 0)"
metadata:
  type: feedback
---

The user rents Verda H200 boxes and they get **cancelled and reassigned constantly** — sometimes
several times in one working day. Each new box is a fresh `--depth 1` clone of `blublas.git` into
`~/gautam_k/blublas`, so nothing is built on it.

**The allocated GPU index CHANGES with the box.** Do not hardcode it and do not assume it carries
over. Observed so far:

    dull-tree-roars-fin-02    2x H200   device 1 (device 0 was a colleague's trainer)
    fair-life-bounces-fin-03  2x H200   both idle
    wide-song-wilts-fin-02    4x H200   device 1
    tall-soul-runs-fin-03     8x H200   device 3      (2026-08-12)
    innmi1srh2-p032           8x H200   device 4      (2026-10-19 18:25 IST, user: "ours is gpu 4")

**How to apply:** make the device a parameter (`GPU=<n>` / `CUDA_VISIBLE_DEVICES=<n>`) and use the
index the user most recently named. If they haven't said, ask — do not guess. Keep `NP <= visible
devices`.

**Consequence that keeps biting:** when the box changes, EVERY previously recorded timing becomes
non-comparable. Re-measure the comparator on the new box in the same session; never project a delta
from a number taken on a different box or a different session. See
[[feedback_verify_claims_and_benchmark]] and [[feedback_separate_build_and_run]].


**2026-09-01 (`light-home-closes-fin-03`, 95.133.252.40): 4x H200, ours is GPU 3 (the last one).**
`CUDA_VISIBLE_DEVICES=3` on every GPU command; 0-2 belong to colleagues. NV18 full mesh, 176
cores, 668 GB, CUDA 13.0 only. Boxes changed THREE times in two days (253.127 -> 254.186 ->
252.40) - each change voids prior timings, so only same-box same-session comparisons count.

**Update 2026-10-14:** OURS = GPU 7 again (10-13 evening it was GPU 2). Colleagues may take it at any time: run ONE test at a time and ask permission before EACH test, saying what it is and roughly how long it takes. Check the GPU is empty before every test.

**2026-10-14 14:13: OURS = GPU 0 now** (the user, mid-launch of test 3 on GPU 7: "STOP BRO OURS IS GPU 0"). GPU 7 is no longer ours. Before EVERY launch, ask which GPU is ours right now if more than ~1 hour passed since the user last named it.

**2026-10-14 evening: OURS = GPU 3** (the user: "gpu 3 is ours"). Replaces GPU 0 from 14:13.

**2026-10-14 ~19:05: OURS = GPU 1** (the user: "ours is gpu 1 from now"). Replaces GPU 3.
