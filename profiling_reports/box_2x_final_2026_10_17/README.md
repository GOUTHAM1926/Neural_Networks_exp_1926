# Final copy of the 2x H200 box (root@86.38.238.36), 2026-10-17 ~21:15 IST

The box (tall-door-completes-fin-02, 2x H200, driver 580.178.04, CUDA 13.0 / nvcc V13.0.88, ncu 2025.3.1) is being
closed. Everything of ours that is needed later was copied here and checked: 622 files, md5 identical to the box.
Work doc: section 45.40 (end of 10-17 + the Monday plan).

## What is here

| folder | from the box | contents |
|---|---|---|
| `scripts/` | `/root/gautam_k/scripts/` | ab_multi.sh (race + ncu), build_cands.sh, run_*.sh, trace_1017.sh, scoreboard_free.sh, `cands/` = the EXACT source of every library built on this box (fwd_<name>.cu). Note: `cands/fwd_K5q.cu` (f35982af) is what built vj2_K5q (8dc36782); the laptop's `session_2026_10_17/candidates/fwd_K5q.cu` (8b83c8d9) is a later regeneration with extra switches, same code paths |
| `loss_bench/` | `/root/gautam_k/loss_bench/` | loss rival benchmark + item-3 test scripts, old_trainer/ (old LossOps.h), test binaries t_inplace_new / t_inplace_old |
| `runs/` | `/root/gautam_k/sft_v2/runs/` | every run of the day: k4b / k5 / k7 / k9 / kall / ko / w (attention races), trace_1017_* (timelines), ncu_k5q_1017 (K5q ncu, 2nd run), loss_1017 / loss_v2_1017 (loss vs rivals), item3_1017 (item 3 standalone + trainer A/B, try 2) |
| `top_level/` | `/root/gautam_k/*.log, *.tgz` | build logs, lossenv_setup.log, every packed results archive; `tmp_build_logs/` = the box's /tmp/build_<name>.log (34 candidate compile logs) |
| `tree_extra/` | `sft_v2/BluTrain_sft/bluscript_sft_old.cpp` | the old trainer used for the item-3 A/B (same md5 bb5f81ed as `sft_loss_vs_rivals_2026_10_17/old_trainer/`) |
| `manifests/` | (generated) | `libs_md5.txt` (35 libraries: md5, size, path, time), `data_and_binaries_md5.txt`, `lossenv_pip_freeze.txt` (torch 2.14.1+cu130, liger_kernel 0.8.4, cut-cross-entropy 25.1.1, transformers 5.20.1, triton 3.8.0), `box_system.txt` |

## What was NOT copied (and why)

- `libs/` (1.9 GB, 35 x libtensor.so): rebuilt from `scripts/cands/fwd_<name>.cu` with build_cands.sh on the next box;
  races always build both sides on the same box. md5s are in `manifests/libs_md5.txt`.
- `lossenv/` (5.6 GB Python venv): rebuilt with `sft_loss_vs_rivals_2026_10_17/loss_env_setup.sh`; versions in
  `manifests/lossenv_pip_freeze.txt`.
- `sft_v2/BluTrain_sft/` (630 MB): the source is IDENTICAL to the laptop master `~/Downloads/BluTrain_sft` (all 1,496 source
  files same md5, checked 21:10 IST; the box tree's GQA_fwd_sm90.cu = F3c = laptop master); the rest is build output.
- `sft_data/` (41 MB): stage0 train/labels.bin (md5 eec5e38b = the laptop's) and tiny_item3 (= `~/Downloads/sft_data/tiny_item3`).
- Other people's folders in /root (Ananya, jeni, sanjay, data, envv, Documents): not ours, not touched.
