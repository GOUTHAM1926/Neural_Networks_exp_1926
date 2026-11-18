---
name: project_p032_cleanup_1106
description: "P032 /localstorage/gautam_k cleaned 11-06 (user: colleague said we use ~300 GiB): 364 -> 68 GiB; what was deleted, what stays, where the records went"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-06T14:12:17.903Z
---

**2026-11-06 evening.** The user approved the list. `/localstorage/gautam_k` on P032 went from 364 GiB to 68 GiB.

**Deleted:**
- muon_guard_p032, tilemap_1102, fw, rope (rivalenv), lossenv, item13_p032, sftb8k, pvo_1106, sftpush, old, libs, sft_data,
  sftb, tmp_g0-5, ckpt (blumodel_step_8000, a 6.1 GiB pretraining shard);
- sft135/{hfcache, partcache, tmp, BluTrain};
- every sft135/runs/*/checkpoints (14 dirs, 35 GiB).
- Two of our own stale GPU monitors were also stopped: nvidia-smi -l 30 from the stopped 10-28 runs, PIDs 624534 and 716665.

**Kept:**
- sft135/data (all prepared SFT data, 45 GiB), ckpt_A/ckpt_B, base_hf, tokenizers, sft135/hf + evals + evalenv;
- the 10 final stage-1 safetensors in sft135/runs (plus logs);
- benv (FA3/FA4 builds), flash-attention source, cp_1106 (the CP work tree), rope_1024, small logs.
- sft135/bluscript_sft_bin* no longer have their library tree (sft135/BluTrain was deleted), so they cannot run.

**Records saved first (laptop):** `profiling_reports/p032_cleanup_2026_11_06/box_records/`. It holds the logs, csv, configs and scripts
of every deleted folder (max 20 MB per file), the pip freezes of the deleted envs, today's pvo_1106 results and both nsys
profiles, and the 1.6 GB of 8192-benchmark gpu-trace CSVs (the only copy).

**How to apply:** do not look on P032 for the deleted folders; rebuild trees from git. Ask before re-creating big data.
Related: [[feedback_box_backup_only_needed]], [[project_preprod_vs_ours_135m_1106]].
