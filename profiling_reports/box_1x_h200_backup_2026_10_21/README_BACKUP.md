# Backup of root@86.38.238.55:~/gautam_k (1x H200), 2026-10-21 ~19:45 IST

Copied with rsync, then checked file by file with `rsync -c` (checksums): 0 differences, 485 files, 1.4 GB.

| folder | what |
| --- | --- |
| `sftb/` | SFT benchmark scripts, `run_status`, all step logs in `runs/` (S1, A1, A3, A3b, A4, A5, B1, B2, C_P_padded, C_K_packed, C_S_samesteps), bench binaries in `bin/`, `tva_a` |
| `sft_v2/runs/sb_sft_1021/` | attention scoreboard raw: nsys reports, kernel traces, clock logs, summaries |
| `lottn/` | pretraining hand-over A/B: `ab/` scripts, `runs/` (loss, attention quick + full, trainer A/B), build logs, trainer binaries `pz_old` / `pz_new`, the tested libraries in `old_libs/` (BLUCCL) and `new_libs/` (dist_side_gautam_lottn) |
| `libs/sft_new/` | the exact SFT library binaries that were benchmarked (libtensor b5c9844c, libblublas d15d3c4b) |
| `sft_data/` | stage-0 padded, packed, `stage0_5k_samesteps_as_padded` (test 2), `tiny_item3` |
| `pt_data/` | pretraining data shards (edu, mix) used by the trainer A/B |
| `scripts/`, `step6/`, `loss_bench/` | benchmark scripts as they ran on the box |
| `local/` | yaml-cpp 0.8.0 static install used to build the trainers |
| `*.log`, `benv_pip_freeze.txt`, `sftb_results.tgz` | environment setup logs, pip freeze of the attention rivals' venv, packed results |

Not copied (can be rebuilt or downloaded again): `lossenv/` and `benv/` Python venvs (`loss_bench/loss_env_setup.sh`,
`scripts/benv_setup.sh`), `ckpt/` (Hugging Face navingv/curriculum_8btokens, blumodel_step_8000), `flash-attention/` and
`yaml-cpp-src/` (git clones), `tmp/`, the build trees `lottn/old`, `lottn/new`, `sft_v2/BluTrain_sft` (source = git commits /
laptop master; their built libraries are in `lottn/*_libs` and `libs/sft_new`).
