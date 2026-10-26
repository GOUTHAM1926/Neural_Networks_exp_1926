---
name: project-state-2026-10-06-eod-sft
description: "PUT IN THE 10-06 EOD + REMIND 10-07: what was done to prove SFT works (stage-0 smoke run passed), the tested / NOT-tested table, and tomorrow's ordered plan"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-07T15:43:52.468Z
---

The user asked on 2026-10-06 (about 21:50 IST) for two things:
- **remind them of this table tomorrow;**
- **put it in the EOD**, both what was completed today to prove SFT works and what is planned for tomorrow.

When writing the EOD, follow [[feedback_eod_report_format]] and [[feedback_verda_usage_sheet]] exactly (re-read both first).

**Done on 10-06 to prove SFT works** (details in [[project-sft-workflow-1006]] and [[project-sft-loss-wiring-task6]]):
- **Task 2, the weights-only load (`BLU_INIT_FROM`):** passed on the H200, GPU 5. 262/262 weights match the checkpoint CRC, and step-0 val on flux was 2.4538.
- **Data:** chose UltraChat-200k (English only). Wrote `sft/prepare_sft_data.py` and built the stage-0 data: 5,000 train / 500 val rows, T=4096, 4.77M trained tokens (23.3% of positions).
- **Loader:** wrote `sft/SftDataLoader.h`. The loader test `sft/test_sft_dataloader.cpp` passed 5/5 on the laptop GPU.
- **Trainer, task 6:** added SFT mode (`BLU_SFT_DATA`): label 65535 ignored, the step's global trained-token count as the denominator, seed W, the summed logged loss, val over the whole val set, and the SFT schedule knobs. Compile-checked, with a control.
- **Stage-0 smoke run** (H200 GPU 5, AdamW, 64 rows/step, LR 2e-5, stopped on purpose at step 20):
  - val 1.446 → 1.353 → 1.332;
  - train 1.45 → about 1.2–1.3;
  - norm 0.74 → 0.20, no NaN;
  - exact counts: 60,263 at step 0 and 441,382 for val;
  - 5.3 s/step, 55% MFU, about 78 GB.
- **Docs:** `loss_calculation_in_pretraining_and_sft.md` (master + mirror). The 10-05 docs (muon bug, SFT report, flow decisions) were pushed in the morning as `5a02ef9`.

**The table, verbatim, as the user wants it reminded:**

| Part | Tested? |
|---|---|
| Data prep script and stage-0 data | ✅ self-test and file checks |
| SFT loader | ✅ 5/5 checks on the laptop GPU |
| Loading only the pretrained weights (task 2) | ✅ 262/262 weights match, on the H200 |
| SFT training (task 6), 1 GPU, AdamW | ✅ 20 steps on the H200: exact token counts, val 1.446 → 1.332, no NaN, 5.3 s per step |
| A full run to the end (156 steps) and saving the checkpoint | ❌ not yet: we stopped early, and the leaked `BLU_CKPT=0` had switched checkpoints off |
| Resuming an SFT run from its own checkpoint | ❌ not yet |
| 2 GPUs (adding up the count across GPUs, seed W = 2) | ❌ only simulated in the loader test, never run |
| Muon in SFT mode | ❌ not yet |
| The model's actual answers and benchmarks | ❌ no generation or eval tool yet |
| Pretraining still working after the change | ❌ compile-checked only. The pretraining path barely changed, but it hasn't been run |

**10-07 PROGRESS — full run DONE.** It ran on GPU 4, 15:01–15:29 IST, in `~/gautam_k/sft_runs/stage0_adamw_full/`. Logs are in `profiling_reports/sft_stage0_full_2026_10_07/`.
- 156/156 steps, AdamW.
- val 1.4464 → 1.3528 (step 10) → 1.2982 (50) → 1.2889 (80) → 1.2869 (100) → 1.2839 (155). It kept falling, so no overfit.
- Train mean was 1.316 in epoch 1 and 1.239 in epoch 2. Max norm 0.918, no NaN, and the LR ended at exactly 2e-6.
- Checkpoints `ckpt/blumodel_step_78` and `blumodel_step_156` were saved, 22.2 GB each and silent on success. Disk is now 50 GB free.
- 53% MFU, 125,026 MiB peak, clean exit.
- Step 0 was bit-identical to 10-06; step 10 val differed by 2e-5.
- The shell still leaked `BLU_MAX_STEPS=4`, `BLU_INIT_VERIFY=1`, `BLU_MIX_ROOT` and `BLU_TOKEN_MAX` from 10-06 (none harmful). Next time, `unset` them first.

**10-07 QUEUE (17:09–18:09, run_queue_1007.sh, md5 74132c23): ALL 4 DONE.** Logs are in `profiling_reports/sft_stage0_full_2026_10_07/{resume78,pretrain4,twogpu,muon}/`.
- **resume78:** PASS. Step 78 was EXACT (train 1.264598, trained 58076), every later step was within about 2e-4, and the final val was 1.283921 vs 1.283946.
- **pretrain4:** PASS. Step 0 was EXACT against 10-06 task 2 (train 2.327407, val 2.453832).
- **twogpu (GPUs 4,5):** PASS.
  - Step 0 was identical, including grad norm 0.7442, which proves seed W.
  - Trained counts were exact through step 77, the final val was 1.283918, and it took 2.68 s/step (2× faster).
  - Epoch-2 step grouping differs from 1 GPU by design (`M = N/(B·W)`; 1 GPU wraps 1 micro-batch into step 78). Possible fix: make each epoch a whole number of steps.
- **muon (LR 2e-5):** runs fine (nsrej 0, nsratio about 1.01–1.03), but the final val was 1.3192 vs AdamW 1.2839, so it learns slower. The suspected cause is that match_rms_adamw sizes the update like a settled AdamW. Next is a Muon LR sweep (5e-5, 1e-4).
- **fourgpu (GPUs 3,4,5,6, 18:2x IST): PASS.**
  - Step 0 was identical (train 1.450003, norm 0.7442, trained 60263), so seed 4 is correct.
  - Train matched 1 GPU through step 77, and matched 2 GPUs (trained exact) from step 78.
  - Speed was 1.35 s/step (4× the 1-GPU speed), 102 GB peak per GPU.
  - Val uses 480 rows (15 batches per rank, 426,157 tokens), so its value isn't directly comparable. It dropped 1.4432 → 1.2809, the same 0.162 drop as on 1 GPU.
  - Logs are in `…/fourgpu/`.
- **MUON LR SWEEP (19:09–19:33, GPUs 2-5, run_muon_sweep_1007.sh md5 0ce514b3):** Muon and the AdamW part used the same LR, like Kathir's `sft.py`. Final val on 480 rows, against fourgpu AdamW 2e-5 = 1.2809:
  - Muon 2e-5 → 1.3162;
  - Muon 5e-5 (Kathir's pick) → 1.2843;
  - Muon 1.2e-4 → best 1.2747 at step 80 (end of epoch 1), then overfit in epoch 2 (train 1.08), final 1.2820.
  - Takeaway: Muon needs about 2.5–6× AdamW's LR. Use 2 epochs at about 5e-5–8e-5 for now. The 1.2e-4 best at step 80 is a MID-RUN point (LR still 6.6e-5), not a finished 1-epoch run, so a real 1-epoch run at 1.2e-4 is the next test. The sweep also gave the AdamW part the Muon LR, and each run decays to LR/10.
  - **21:15 state.**
    - The user's 6-item check: all done except #5, evals.
    - The docs repo is ready to commit. `profiling_reports/` is gitignored, so it needs `git add -f`. Leave out `kathir_sft/` (Kathir's code) and the shared docx.
    - BluTrain is still held, per the user's 10-07 rule. Its backup in fc8e20c matches the laptop exactly (3 sft files plus the trainer diff), so nothing is at risk.
    - Next: ask Kathir for his converter and files, then evals on step_156, then a 1-epoch Muon run at 1.2e-4, then the speed-ups, then stage 1.
  - The SFT report now lives as the "SFT - what it is" tab of the shared Google Doc "PostTraining - Research and Planning". Its old §7/§8 were dropped, and the local sft_report.md was deleted by the user. The new section 8 for that tab (results, sweep, papers, learnings, Kathir comparison, plan) is in `profiling_reports/sft_report_2026_10_07/sft_doc_section8_1007.md`. The stale spots the user fixes by hand are listed in work doc §37.10.
  - Logs are in `…/muon_sweep/`.
  - The first start went to GPUs 3-6 (default) and its parent script died. The user OKed the kill and it was restarted on 2-5.
- **Kathir's `sft.py`** (`work_docs/sft.py`, PyTorch):
  - one LR for ZeroMuonAdamW; batch 512 conversations per step; linear warmup 3%, then linear decay to 0;
  - loss = sum / n_global (same as ours);
  - logits computed ONLY for trained positions (`h[m]` before lm_head), which is Q11 already done in PyTorch;
  - `zero_muon.py` is not available locally, so his Muon update scaling is unverified.
- **User 10-07: do NOT push to remote yet.** Next is comparing Kathir's `sft.py` with ours, then evals. His eval was only manual inference on prompts (base vs epoch 1 vs epoch 2), no benchmarks.
- **Disk:** 23 GB free (95%). I asked the user whether to delete the duplicate `resume78/ckpt/blumodel_step_156` (22 GB).
- **The code passed every test except eval** and is ready to push to `dist_side_merge`.

**Plan for 10-07, in order.** The user runs every box command, from a FRESH shell (no leaked `BLU_*` vars).
1. The full 156-step AdamW run with a final checkpoint, on GPU 5 (about 15 minutes).
2. The same run on GPUs 5+6. Its losses should match the 1-GPU run closely, which proves the all-reduce and seed W.
3. A resume test: restart from the run's own checkpoint mid-way.
4. A Muon run. Pick `BLU_SFT_MUON_LR` first.
5. Evaluation: ask Kathir about his eval suite (`~/kathir/Model-Eval-Infer-Suite-PostTrain`) for the checkpoint→HF converter and generation. Check that answers end with the end token, and compare base vs SFT on HellaSwag.
6. A short pretraining regression run (without `BLU_SFT_DATA`).
7. Push the 4 SFT files to `dist_side_merge` (the user pushes), and write today's results into the SFT doc and the loss doc.
8. Then stage 1: more data plus the LR sweep (5e-6 / 1e-5 / 2e-5).

**Box usage on 10-06:**
- GPU 5 on node `innmi1srh2-p032` (`-p 2224`) ran the task-2 test (4 steps) in the morning and the stage-0 smoke run (20 steps) from 21:38 to 21:44.
- The two trainer builds were CPU only.
