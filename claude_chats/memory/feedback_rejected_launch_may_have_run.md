---
name: feedback_rejected_launch_may_have_run
description: "A rejected/interrupted ssh launch DOES start on the box (twice: 10-13, 10-14); check ps at once after any rejection and tell the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-13T07:28:31.747Z
---

On 2026-10-13 a box launch (nohup setsid run.sh on GPU 7) was rejected by the user, but it had already started at 12:52. I relaunched at 12:57 without checking. Result: two of my jobs ran on GPU 7 at once, and my new launch's `rm -f status` plus the new `1a_smoke.log` destroyed the first run's 1a log. The user then said STOP; I killed both.

**Why:** a rejection or interrupt does not prove the remote command never ran. Overlapping jobs spoil timing and the shared box.

**How to apply:** after any rejected or interrupted box launch, first run `ps -eo pid,lstart,cmd | grep -E "[r]un.sh|..."` and check GPU memory before anything else. Use a fresh log dir per launch (timestamped), never `rm` an existing status file. Related: [[feedback_ask_before_every_gpu_run]], [[feedback_box_ops_traps]].

**Happened AGAIN 2026-10-14 ~12:00:** test 3 (ncu on b2a) was "rejected" by the user, yet it had already run 1.5 min (lineinfo build done, ncu running on GPU 7) while the user's colleague wanted the GPU. The user had to ask me to kill it. So: right after ANY rejection of a box launch, check `ps` at once and tell the user it is running, unprompted; kill only on the user's word (by exact PIDs).
