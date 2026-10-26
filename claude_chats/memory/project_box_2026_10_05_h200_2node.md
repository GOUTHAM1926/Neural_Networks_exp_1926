---
name: project_box_2026_10_05_h200_2node
description: "8x H200 node innmi1srh2-p032 (ssh -p 2224 operator@103.42.51.252), back in use 10-19: OUR GPU = 4; ONE folder /localstorage/gautam_k (~/gautam_k is a link to it), TMPDIR=~/gautam_k/tmp"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-07T09:27:34.809Z
---

**The box (from 2026-10-05).** 2 nodes, 16× H200.
- Node 1 is `ssh -p 2224 operator@103.42.51.252` (host `innmi1srh2-p032`). Key login works from the laptop, with no password.
- Node 2 is `-p 2225` on the same IP. It asks for a password and is used by a colleague; the user has no access.
- **Our GPU = 2** (the user said "ours is gpu-2"). GPUs 3-7 run colleagues' jobs (`…catter_internode_compare_test`, `_jit_runner_`). Never touch those.
- **Our dir is `~/gautam_k`.** `data/flux_val_000000.bin` was already placed there.

**UPDATE 2026-10-07 (user): our GPU today is 4** ("ours is gpu 4"). Use `CUDA_VISIBLE_DEVICES=4` for 1-GPU runs. The second GPU for 2-GPU runs is not known yet, so ask before using one. Always use the index the user names that day.

**UPDATE 2026-10-06 (user): our GPUs are now 5 and 6** on the same node (innmi1srh2-p032, 8x H200, still connected). 1-GPU runs use **GPU 5**; 2-GPU runs use **5 and 6**. GPU 2 (yesterday's) is no longer ours. On 10-05 GPU 2 was later taken by Aparna's `_jit_runner_`, and the fix validation ran on GPU 1 with the user's OK.

What is on the box:
- CUDA 12.8 and 13.4 are in `/usr/local`; `nvcc` is not on PATH. OpenMPI 4.1.6, system NCCL, and 175 GB free disk.
- **Navin's step-8000 is only an HF conversion**, in Kathir's dir `~/kathir/Model-Eval-Infer-Suite-PostTrain/hf/blumodel_step_8000` (`model.safetensors`, 3.0 GB). It is read-only for us. Its `config.json` has `rope_theta 500000.0`, which matches our trainer. `blu_source.json` gives val_loss 1.866113 at step 8000.
- Our `BLU_INIT_FROM` needs the DCP files: `metadata.bin` + `data_rank_0.bin` (6.5 GB) from HF `navingv/curriculum_8btokens`.

**How to apply:**
- Ask before every box action ([[feedback-a<REDACTED:api-key>]]).
- Pin `CUDA_VISIBLE_DEVICES=2`.
- Ship code via git, not rsync up ([[feedback-edit-local-push-pull]]).
- Pull the logs down after every run.

**2026-10-19 18:20 IST: back on this node** (the Verda box 31.56.109.89 closed when the credits ran out; everything was pulled first, work doc 45.56).
- **Our GPU today = 4** (user, 18:25). At 18:15 GPUs 0, 3, 4, 5, 6 were free; 1 = dpo_train, 2 = `_runner_`, 7 = a lighteval python (not ours).
- **The root disk `/` is 100% full** (439 GB, 912 MB free): nothing can build in ~/gautam_k. `/localstorage` (md0, 25 TB, 22 TB free, world-writable like /tmp, colleagues' folders there) holds the attention setup:
  `/localstorage/gautam_K` (= Verda's layout: scripts/, libs/, sft_v2/BluTrain_sft, sft_v2/runs, tmp/) with `~/gautam_K` -> it (our scripts use `$HOME/gautam_K`) and `sft_data` -> `~/gautam_k/sft_data` (md5s = the laptop's).
- **nvcc writes temp files to /tmp (the full root disk):** export `TMPDIR=$HOME/gautam_K/tmp` for every build.
- **CUDA 13.4 here is compiler build 38657139; Verda's was 38855100.** Compare PF's SASS from this node with Verda's saved dumps (`session_2026_10_19/box_runs/pft_1019/sass_*.txt`) before reusing Verda's numbers.
- `~/gautam_k/benv` = Verda's versions: torch 2.14.1+cu130, cuDNN 9.24.0.43, FA3 + FA4 built from `~/gautam_k/step6/fa_src` at 94e22c9.
- **Do not touch** `~/gautam_k/BluTrain_sft` (branch dist_side_merge_BLUCCL, uncommitted DPO / bluscript work) or `BluTrain_bluccl`. `~/gautam_k/sft_v2` is the old 10-17 setup.
- 224 CPU threads; build with -j96 to leave CPU for the colleagues.
- **Cleanup 10-19 19:40 (user ran it):** deleted `~/gautam_k/sft_runs/stage0_adamw_full/ckpt/blumodel_step_156` (22.2 GB SFT ckpt; the user says it is on Hugging Face, no upload record in our notes; DPO reference logprobs need it), `BluTrain_bluccl` (all BLUCCL commits verified on GitHub), `libs`, `sft_v2` (runs = laptop copies). Kept `BluTrain_sft` (4 unsaved files: bluscript_zero_ckpt.cpp mod, compute_dpo_reference_logprobs.cpp, sft/SftDataLoader.h, TI bench_sparse_ce_sft.cu), benv, step6, sft_data, data, ckpt/blumodel_step_8000.
- **10-19 19:25: ONE folder.** Everything of ours is now in `/localstorage/gautam_k` (lowercase k); `~/gautam_k` is a LINK to it (old absolute
  paths, e.g. benv's, still work: `benv OK 2.14.1+cu130`). The old root-disk folder was copied (rsync, checksum dry run 0 differences,
  10,998,905,483 bytes both), then deleted; `gautam_K` (capital) and its link are GONE. Today's tree = `~/gautam_k/sft_v2/BluTrain_sft`,
  new scripts = `~/gautam_k/scripts` (old 10-17 ones in `scripts_old_1017`), `libs/`, `tmp/` (TMPDIR). Our scripts use GK=$HOME/gautam_k
  (ab_multi.sh 94ab6fae, build_cands.sh 20772174, run_pf_full_1019.sh 708f0dca). TRAP: never `rm -rf ~/gautam_k/` WITH a trailing
  slash -- through the link it deletes the real folder's contents. Root disk after: 75%, 105 GB free.

