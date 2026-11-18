---
name: project_data_team_sft_guide_1029
description: Data-team SFT guide how_to_handle_sft.md + .docx in work_docs (10-29); it needs the 135M files pushed to BluTrain dist_side_gautam_lottn first
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T07:22:33.639Z
---

On 2026-10-29 the user asked for a guide so the data team can run SFT ablations themselves.

**Where it is:**
- `work_docs(mark-down - (.md) files ) /how_to_handle_sft.md` (the folder name ends with a space);
- `how_to_handle_sft.docx` next to it.
- Builder: `profiling_reports/how_to_handle_sft_2026_10_29/build_docx.py`. It uses python-docx, because the laptop has no LibreOffice or pandoc. Run it as `PYTHONPATH=<scratchpad>/pylib_docx python3 build_docx.py <md> <docx>`. Its numbered lists keep their numbers as text.

**Facts checked while writing it:**
- All 3 repos are private: BluTrain, Tensor-Implementations, kathir-23s/BluBridge-BLAS.
- `git -c credential.helper=...` carries the token into both submodule levels and stores it nowhere. Tested by a clone on 10-29.
- BluTrain uses git-lfs for `*.log`, `*.csv` and `*.txt`, but only one file is in it (`dist/.../counters.csv`). `-c filter.lfs.*` overrides avoid "git-lfs: not found".
- `dist/Context_Parallelism` is not needed for the build, so the guide inits only Tensor-Implementations (recursive).
- Val sets: m1 and d0-d8 share the 2,000 smol-smoltalk test chats. m2 uses 2,000 Tülu chats. m3 uses the smol set plus 800 Tülu math chats.

**Blocker:** the remote branch (BluTrain 5c0999c) lacks the 135M work. The guide expects it, so it must be pushed first:
- `hf_to_blutrain.py`, `blutrain_to_hf.py`, `ref_val_loss.py`;
- `mixes/*.json`, which the root `.gitignore` hides (`*.json`), so they need `git add -f`;
- the modified `bluscript_sft.cpp`, `SftDataLoader.h`, `prepare_sft_data.py`, `run_sft.sh`.

The user also shares the base HF folder with the team.

**Final shape of the guide** (~170 lines):
- token plus one `insteadOf` line, then plain clone;
- `make -j$(nproc)`;
- prep;
- `make run-train-mpi MODE=production_zero TRAIN_SRC=...bluscript_sft.cpp` with `BLU_*` settings.

**NOT yet run end to end through make** (all our SFT runs used `run_sft.sh` with no YAML). Through make, the production YAML applies too: bf16 logits, sizeclass exact, segmented embedding backward, checkpoint freq 1600 / keep 4, monitoring. Suggest one short gate run via make before the data team relies on it.

Related: [[feedback-guides-for-other-teams-scope]], [[project_sft_135m_1027]], [[project_sft_push_lottn_1027]].
