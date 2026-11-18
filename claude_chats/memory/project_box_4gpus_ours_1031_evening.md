---
name: project-box-4gpus-ours-1031-evening
description: "2026-10-31 21:12 IST: user said ALL 4 GPUs of box 95.133.253.118 are ours 'till I say' — use all; still check each GPU free before every launch (a colleague's 4-GPU _jit_runner_ job held all 4 at 21:00-21:10)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-31T15:46:03.754Z
---

2026-10-31 21:12 IST, the user: "4 GPUs are ours now till I say, so use accordingly, use all if you want".
Before that the same evening: GPU 3 ours; GPU 0 ours 17:07-18:05, then the colleague's, then ours again from 18:50.
At ~21:00 someone's 4-GPU job (`./_jit_runner_`, root) took all four GPUs for ~10 minutes.

**Why:** GPU ownership on this shared box changes within hours; the user grants and revokes it by message.
**How to apply:** use GPUs 0-3 until the user says otherwise; still run the free-GPU check before every launch
(never share a GPU with another job); one runner per GPU. Note the rejected-launch trap held again at 21:11: a launch
the user rejected had already started both runners ([[feedback-rejected-launch-may-have-run]]).

(Moved from the MEMORY.md index line on 2026-11-05:) even with all 4 GPUs ours, still free-check every launch.
