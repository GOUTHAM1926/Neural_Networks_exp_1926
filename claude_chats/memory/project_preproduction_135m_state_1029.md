---
name: project_preproduction_135m_state_1029
description: "State of preproduction-135M (TI + BluTrain) checked 10-29 via the GitHub API; PR #111 is live; BluTrain lacks our SFT pipeline; user deferred pushing the 135M files"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T11:30:23.717Z
---

Checked read-only on 2026-10-29 through the GitHub REST API. `gh` is not installed. The helper is
`scratchpad/ghcheck/gh.py`; it reads `~/.config/blubridge/github_pat` and never prints it.

**TI:**
- PR **#111** (preproduction-135M -> master) was opened by the user 10-27 19:35 IST. It is open and unmerged, with no
  reviews.
- It is LIVE: it follows the branch tip, so it now includes the DPO merges (89 commits, 136 files).
- Branch tip `1229fea`. On 10-29 kiruthikkanna-m merged three branches into it: our TI lottn `f7aa40f` (`0315c19`),
  `madhu_post-training` and `DPO_jeni`.
- Of our 30 TI files, 28 are byte-identical. LossOps.h and LossBackward.cpp only gained DPO loss additions. Our RoPE/norm
  files won over madhu's.
- BLAS is `ea666ab` everywhere.
- The CI check `ready-check.yml` only runs `echo "Ready"` (vacuous).
- The DPO merge also put 4 profiler files in the TI root (`dpo_*.ncu-repz`, `ncu_*.txt`).

**DPO uses the fp32 `fused_rms_rope`, not our fast `fused_rms_rope_bf16`.** The calls are correct (seqpack passes our
`pack_position` and `pack_chats`) but they get no RoPE speedup. The switch is about 10 lines in each of 4 files; that is
the DPO team's choice.

**BluTrain:**
- `preproduction-135M` = `7a61192`. It LACKS our 5 lottn commits (21 files): the sft/ pipeline, bluscript_sft.cpp, the
  DCP fix and NVTX.
- Its TI pointer is the stale `37f2bdc`.
- There is no BluTrain PR.
- A merge of our lottn would be clean except for the TI pointer. Adhitya's MixedDataLoaderLite constructor gained only a
  defaulted last parameter (`row_mix`).

**Push deferred (user, 10-29):** "let's not push these new things now, we need to do some more optimizations".
- The 19 local 135M files in `~/Downloads/BluTrain_push` stay unpushed.
- The TI `test_varlen_attention.cu` change is not needed.
- I warned that the pushed pipeline is more than "configs and paths" away from the 135M base:
  - the loader hard-codes end token 50256;
  - prep is GPT-2 only;
  - there are no shape or `BLU_EMBED_SCALE` knobs (a wrong embed scale gives no error);
  - there is no HF load or export.

**How to apply:**
- Before telling anyone to run 135M SFT from the repo, or sharing the data-team guide, push those files.
- Then merge BluTrain lottn into preproduction-135M and set the TI pointer to the TI tip.

Related: [[project_data_team_sft_guide_1029]], [[project_sft_push_lottn_1027]].
