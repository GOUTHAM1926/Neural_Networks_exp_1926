---
name: feedback_packing_only
description: "From 2026-10-22: SFT uses PACKING only, never padding again (user: 'fuck padding and forget it, we won't use it anywhere'); box 86.38.182.101 (4x H200), our GPU = 0"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-22T06:37:25.580Z
---

User, 2026-10-22: "we will use packing methodology, not padding ... we won't use it anywhere from now on, now we will use
only packing". Never propose padded data / padded runs again (data prep always `--pack ffd`, trainer on packed rows).

Same day: new box `ssh 86.38.182.101` (hostname short-city-completes-fin-03), 4x H200, **our GPU = 0**. The user runs every
command himself ("i will do everything manually, guide me with commands, one by one, no big complex commands, don't do
anything by yourself"): give 1-3 short commands per step, say what to send back, never run box commands myself.

**Why:** packing proved equal to padding (test 2, 10-21) and 3.3x faster.
**How to apply:** Tülu 3 SFT run (tulu-3-sft-olmo-2-mixture-0225) on the 23B-token 1.5B checkpoint, packed, GPU 0.
Related: [[project_sft_benchmarks_kathirvel_1021]], [[feedback_user_runs_all_box_commands]].
