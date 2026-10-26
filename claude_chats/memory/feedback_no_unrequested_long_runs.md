---
name: feedback-no-unrequested-long-runs
description: Run only what the user asked for; never start a long run on my own initiative; state the expected duration first and keep it to 1-2 minutes unless told otherwise
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-10T12:46:48.427Z
---

**Run only what the user asked.**
- Never launch an extra long job on my own initiative, e.g. a "final full G-series gate" nobody requested.
- Before any run, say how long it takes.
- Default to runs of 1-2 minutes. Use quick or windowed modes such as `VARLEN_QUICK=1` or `VARLEN_ONLY=D10` with a row window.
- Longer runs only when the user explicitly asks.

**Why:** on 2026-10-10 the user asked for a 1-minute cuDNN check, which was already done. I then started a 21-minute full re-run unasked ("run on GPU 7" was meant for the check). The user was furious: "bro 21mins no ... just one or two minutes", "21 mins for that, are you mad or what?". It wasted GPU time on a shared box and made the user think the cuDNN check itself took 21 minutes.

**How to apply:**
- If "run" is ambiguous and the requested thing has already finished, say it's done and show the result. Do not substitute another job.
- If a long run is genuinely needed, propose it with its duration and wait.
- Complements [[feedback-ask-before-every-gpu-run]]: on our GPU, no ritual for what WAS asked, but nothing extra.
