---
name: feedback-guides-for-other-teams-scope
description: "A how-to guide for another team starts at cloning the repo: no server/node setup, no box paths, no our-progress content"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T07:22:29.994Z
---

A how-to guide for another team (e.g. the data team's `how_to_handle_sft.md`, 2026-10-29) covers only what they do with the repository:
- clone (contributor and non-contributor, with a GitHub token);
- build, data, gate, run, watch, stop, export, eval, ablations, troubleshooting.

Leave out:
- SSH, box or node setup, tmux;
- P032 or `/localstorage` paths, my folders, shared-account rules;
- our stage results and our grid plans.

Use generic paths (`$WORK`).

**SHORT, simple commands only.** The second draft (1,000 lines, with a credential-helper function, git-lfs flags, a gate, eval, an ablation queue and big tables) was rejected as "too much bluff and complex commands". What was accepted was ~150 lines:
- a token plus a one-line `insteadOf` setting, and plain `git clone`;
- build in 2 commands;
- prep in 3 commands;
- one run command, a short settings table, watch, stop and export;
- 6 notes.

Give only what was asked, plus must-know details.

**Run command = the framework's own make launcher, not our helper scripts.** Use `BLU_*=... make run-train-mpi MODE=production_zero TRAIN_SRC=train/production_zero/bluscript_sft.cpp NP=1 RUN_NAME=x`, never `run_sft.sh` or `build_sft_trainer.sh`. The user (10-29): "where is the simple make run-mpi command... you gave our script commands".

How it works:
- A command-line `TRAIN_SRC` overrides the MODE's trainer.
- `BLU_*` variables set before make win over the YAML, which is loaded with `--config`.
- The model shape is applied after the YAML.
- The make compile line is the same as `_ablation_compile`.

**Why:** the user stopped the first draft mid-write: "not server or node, just how to handle after cloning my repo, not the things we are doing or our node setup".

**How to apply:** write commands with `$WORK` and `$S` placeholders. Keep only general lessons, stated without our numbers. Related: [[feedback-reports-in-my-voice-gdoc]], [[project_data_team_sft_guide_1029]].
