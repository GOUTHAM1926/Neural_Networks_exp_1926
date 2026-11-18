---
name: feedback_length_limits_follow_every_test
description: "When a test, benchmark or measurement uses another context length, lift every hard-coded length limit in the pipeline and run the REAL pipeline at that length; never work around a limit silently"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-31T06:58:59.221Z
---

When the context length changes for anything (a run, a benchmark, a measurement), check every hard-coded length limit along the pipeline (prep script, its histograms and checks, launcher, trainer defaults, loaders) and change it in the same piece of work; then run the real pipeline (preprocessing included) at that length. Never work around a limit (e.g. building rows by repacking shorter data) without saying so up front.

**Why:** on 2026-10-28 I benchmarked the SFT kernels at 8,192 with rows repacked from 4,096 data because `prepare_sft_data.py` refused T > 4,096, and I neither lifted the limit nor told the user; on 10-31 I even said `--seq-len` was free. The user found out on 10-31 and was angry ("if context length changes even in measurements or benchmarks we should change that hardcoded limit").

**How to apply:** before any length change, grep the pipeline for the old length (4096, 8192, ...) and list each hit; fix them with the work, test the real prep at the new length (counts vs an independent measurement), and state any remaining limit in the first message. Also: standalone kernels are verified at a length before any model run at it. Related: [[project_case5_long_questions_1031]], [[feedback_no_assumptions_decide_by_tests]].
