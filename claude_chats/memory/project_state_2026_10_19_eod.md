---
name: project_state_2026_10_19_eod
description: STATE at 2026-10-19 EOD + resume plan for 10-20 — best forward QPU (dense fwd 1.004x FA3 on the 8x node), next = full suite on QPU, adoption, diagonal-first with the loop order kept; resume file has commands for the node and for a new Verda box
metadata:
  type: project
---

**Read first on 10-20:** `profiling_reports/session_2026_10_19/RESUME_2026_10_20.md` (exact commands for both boxes).

- Best forward = QPU (`session_2026_10_19/scripts/cands/fwd_QPU.cu` 7e4197ea); backward Vj2 unchanged. Node scoreboard
  (§45.60): ours / FA3 dense fwd 1.004, f+b 0.976 dense / 0.933 packed; 11 of 12 matchups won. Laptop master still F3c.
- Next, in order: (1) full suite on QPU (`scripts/run_full_suite.sh <gpu> vj2_QPU`, 1-2 h on the node); (2) adopt QPU into the
  master (the user decides); (3) diagonal tile first while keeping QPU's steady-loop order (gate `scripts/loop_sass.py`).
- Box tomorrow is unknown: (A) the 8x node = nothing to set up (~/gautam_k has tree, libs incl. vj2_QPU, tva_a, benv, scripts);
  (B) a new Verda box = the RESUME file's copy + build + benv_setup.sh steps (scripts now use $HOME/gautam_k everywhere).
- 10-19 EOD went out with 2 paragraphs (user asked "minimally in 2 paras"). Still uncommitted: BluTrain_sft since 09-28, this
  repo's docs since 10-16 (the user commits).
- Older state: [[project_state_2026_10_17]] (the loss-item decisions there are still open, after attention).
