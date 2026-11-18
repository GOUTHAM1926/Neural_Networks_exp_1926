---
name: feedback_sft_same_settings_as_pretraining
description: "User rule 2026-10-30: SFT runs take the SAME settings pretraining enables (pass the production YAML; run_sft.sh must force BLU_EMB_BWD=segmented); check launch scripts too, not only the trainer; REMIND the user"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-30T09:51:22.148Z
---

User, 2026-10-30, after we found every SFT run used the bf16 atomicAdd embedding backward (run_sft.sh passes no YAML
and unsets all BLU_* -> built-in defaults; production YAML says embedding.backward: segmented):
- From now on SFT runs get whatever pretraining enables: pass the production YAML
  (`train/production_zero/bluscript_zero_ckpt_config.yaml` + its parallelism file) and put only SFT-specific settings
  on top (env beats YAML, ConfigBridge.h).
- Even when launching through scripts like `sft/run_sft.sh`, `BLU_EMB_BWD=segmented` must be set (never atomic /
  cooperative / built-in default).
- Care for what is OUTSIDE the trainer too: launchers, run scripts, env, YAML; not only the training script.
- Audit which production-YAML settings the SFT runs miss; the USER does that audit by hand (see
  [[feedback_teach_me_do_it_myself]]).
- Watch out when passing the YAML: it also holds pretraining-only values (data mix, steps, schedule, checkpoint dirs);
  decide key by key which SFT overrides.

**Why:** a setting that lives only in the YAML silently differed between pretraining and SFT; no error was printed.
**How to apply:** remind the user of this rule before any SFT launch/script change; check the run log header for
`[config]` (YAML path, not "no YAML") and `[embedding] backward: segmented`.
Related: [[project_kernel_plan_1030]].
