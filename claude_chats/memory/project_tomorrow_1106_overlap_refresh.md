---
name: project_tomorrow_1106_overlap_refresh
description: "IMMEDIATE TASK 11-06 MORNING (user 11-05 ~21:33): measure BLU_OVERLAP_REFRESH (staged parameter refresh) ON vs OFF for multi-GPU pretraining -- item 7's multi-GPU half"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T16:03:25.168Z
---

**Task (user, 2026-11-05 ~21:33 IST): "ok for this too, but not now, tomorrow ... document that as immediate future task,
tomorrow morning we will run".**

**What the setting is:** `zero1.overlap_refresh` in the YAML, environment variable `BLU_OVERLAP_REFRESH`.
- It is already in the code. Kiruthik added it to the pretraining trainer on 10-01 (commit 0d65161); the SFT trainer
  got it from the pretraining trainer.
- It is OFF in the production YAML (`overlap_refresh: false`). It only takes effect at world_size > 1:
  `(world_size > 1) && !no_overlap && BLU_OVERLAP_REFRESH`.
- With it ON, after the update the new weights are sent in stages (the embedding first). The next forward starts
  while the later stages are still being sent, so the transfer hides behind compute. The trainer also skips its
  end-of-step cudaDeviceSynchronize, so dt becomes host time.

**Why measure it:** it is work doc 45.140 item 7's multi-GPU half ("the existing overlap_param_refresh option: measure
on / off"). Nobody has measured it yet.

**Already done 11-05 (P032, runO58):** a 2-GPU SFT CORRECTNESS check with it ON, old vs new code, after items 5 + 8
removed the Adam waits inside adam_flat_range / adam_shard_range. It is ordered by param_sync_event_. It was not a
speed measurement.

**How to apply:**
- Run the production 1.6B pretraining on multi-GPU (8 GPUs if given) with overlap_refresh off vs on: same step window,
  swapped, losses compared.
- Lock the GPUs with holders ([[feedback_guard_our_gpus_during_runs]]).
- Ask the user before running ([[feedback-ask-before-every-gpu-run]]).
- Check that dt means the same in both modes: with it ON, dt is host time.

**DONE EARLY, 11-05 21:36-21:58 (user: "do it now itself"), 4 GPUs on P032 (work doc 45.172):** ON is 22 ms per step
faster than OFF (5408.7 vs 5431.0 ms, -0.4%). Both ON runs were faster than both OFF runs. Losses are equal within
noise, 0 errors.
**STILL OPEN for 11-06:** the 8-GPU production run, off vs on, swapped. It needs a GPU grant (all 16 GPUs went to
others at 21:56). After that, decide whether to set overlap_refresh: true in the production YAML.
