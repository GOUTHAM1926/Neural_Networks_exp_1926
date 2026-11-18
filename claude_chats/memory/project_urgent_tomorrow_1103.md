---
name: project-urgent-tomorrow-1103
description: "URGENT 11-03 morning (the user, 11-02 22:35): review in detail + push the BluBLAS fixes (B3 + sm_80/86/89 C), then trainer off/on test, resume the Ada format study, CP task; the chat-template study goes into the 11-03 EOD"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-02T17:03:25.952Z
---

The user, 2026-11-02 ~22:35 IST: "pushing all of these new fixes, we will do tmrw analyzing in detail ... note as future
urgent tasks tmrw mrng". Order for 2026-11-03 morning:

1. Review every change in detail with the user, then push (user runs commits/pushes; no AI attribution; verify each file
   on the remote afterwards, see [[feedback_verify_push_on_remote]]):
   - B3: BluBLAS `BF16/SM90/Bgemm_sm90_warpspecialized_template.cuh` (host-only tile-table cache; kernel byte-identical)
     -> also the same change for the FP16 SM90 and SwiGLU templates (own copies of the cache), test them too;
   - C: the last-group fix in `BF16/SM89/Bgemm_core_template.cuh`, `FP16/Hgemm_core_template.cuh`,
     `FP32/SM86/Sgemm_core_template.cuh`.
   Files + diffs + results: profiling_reports/long_context_kernels_2026_10_31/tilemap_fix_2026_11_02/; work doc
   45.133-45.134, plan Q&A 196-197. BluBLAS is a submodule (ea666ab under TI d067e89): push BluBLAS, then bump TI.
2. Real trainer with the trained-rows switch (saved, not pushed) off / on, with A and B3, several runs, more steps.
3. Resume the Ada chat-format study: write the GPU list back into logs/gpus.txt (paused at 120/208; GPU 0 was lent to a
   colleague); restart command if the loop died: `cd /mnt/volgrp03/3rd_floor/gautam_fmt && GPUS_FILE=logs/gpus.txt
   bash code/fmt_screen_torch.sh grid logs/runs.txt` (setsid/nohup). The study is NOT in the 11-02 EOD; it goes into
   the 11-03 EOD.
4. Task 2 of 11-02 (not done): CP wrappers around our attention kernels for SFT and DPO, validate precision /
   performance / memory -> [[project-future-cp-ulysses-sft]].
5. FUTURE: split-K dW GEMMs (16k qkv/o dW, 128k o dW) are not bit-deterministic run to run.
Related: [[project-future-blublas-tile-map-sync]].

**11-03 15:56 (the user):** P032 GPUs 0 and 1 are OURS; "kill others' processes and run ours ... don't allow anyone's
process till we get completed". A guard (trainer/guard_g01.sh) stops any non-ours process on GPU 0/1 until the nsys runs
of the trainer lag test finish; every stop is logged in trainer/nsys_status. Valid for this test, not a standing rule.

**11-03 OUTCOME:** items 1-3 DONE: BluBLAS fix + trained-rows switch PUSHED and verified on the remote (BluBLAS
dist_side_gautam_lottn 297f372, TI d18f820, BluTrain 7184fe1); trainer on/off tests + nsys done; Ada study finished
(208/208, ChatML chosen, report written). Item 4 (CP wrappers) still open. Org PAT leaked via `bash -x` on 11-03:
the user must revoke + replace it. **11-04 queue:** Ada eval CPU prep -> evals on the user's word; Muon sync fix
(ShardedMuon.cpp:1703 + 1731-1749) + other CUDA-API items (45.138); switch edge-case tests; CP wrappers; P032 cleanup.
