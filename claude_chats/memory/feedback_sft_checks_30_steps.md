---
name: feedback_sft_checks_30_steps
description: "User 11-05 18:45: SFT A/B checks run 30 steps, not the full 60-step epoch -- stop the run after step 29 (schedule unchanged, never BLU_MAX_STEPS)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T13:15:34.587Z
---

User, 2026-11-05 ~18:45 IST (item 13): "why are you running 60 steps each time, do 30, 30 steps are enough to check for
upcoming checks". The item-3 SFT A/B used the whole data_4k epoch (60 steps per run, 6 runs).

**Why:** time -- the user wants every remaining item finished the same day; 30 steps give the step-time median (steps
11-29) and the loss comparison well enough.

**How to apply:** SFT checks (profile + plain A/B runs) stop after step 29: the test script watches run.log and stops
mpirun by its exact PID once "step 29" is logged (as test_item3_pz.sh does with NSTOP). Do NOT shorten with
BLU_MAX_STEPS / a smaller data set: that changes the LR schedule ([[feedback-never-shorten-max-steps-to-benchmark]]),
so the first 30 steps would no longer match earlier 60-step runs. Report medians over steps 11-29.
