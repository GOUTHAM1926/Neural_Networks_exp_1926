---
name: project_sft_push_lottn_1027
description: "2026-10-27 SFT push onto dist_side_gautam_lottn — prepared in ~/Downloads/BluTrain_push, build-checked on the 2x H100 box; user commits/pushes; then H100 benchmark re-run (GPU 1)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-27T10:31:41.379Z
---

**Prepared 2026-10-27 in `~/Downloads/BluTrain_push`** (branch dist_side_gautam_lottn; base TI b1443a5, BluTrain 16bce82 = 3b80f15 + the user's unpushed profiler NVTX commit; BluBLAS stays ea666ab = PR #3).
- TI: 19 files = 16 SFT files from BluTrain_sft (md5 = SFT_package_2026_10_24 manifest) + tests test_rope_positions.cu, test_varlen_attention.cu, sanitize_vj2.cu. Only edit of mine: a 29-line comment header in FusedRMSRoPE_bf16.cu. test_loss_grad_inplace.cu LEFT OUT (calls the removed 5-arg opt-in loss API; in-place is the only path since 10-21).
- BluTrain: `train/production_zero/bluscript_sft.cpp` (3-way merge base 909ab75 / ours BLUCCL trainer / theirs BluTrain_sft/bluscript_sft.cpp; script scratchpad merge_trainer/resolve.py, every edit asserted) + `sft/` (prepare_sft_data.py, SftDataLoader.h, test, build_sft_trainer.sh, run_sft.sh, export_to_hf.py, verify_export.py, README.md, README_SFT_checkpoint_for_DPO.md, docs/ 5 md + 2 png). Never add dist/Context_Parallelism.
- Merge adaptations: MPI_Allreduce/MPI_Abort -> dist::comm::allreduce_scalar/abort_job; SftDataLoader device = local_rank; BLU_INIT_FROM load group = nccl_pg, else bring_up "dcp" (BluCCL W>1) or single-rank (W=1) kept in init_pg and reused by the checkpointer (`if (init_pg) ckpt_pg = init_pg`, BluCCL bring-up guarded `&& !init_pg`); fingerprint + shard CSV columns skipped in SFT; CSV adds trained_tokens. This branch's ModelConfig/YAML theta default = 10000 (old SFT tree 500000) -> always set BLU_ROPE_THETA.
- Build check: laptop has no MPI headers and an old pip NCCL -> built on the **2x H100 box root@217.177.40.23** (`/root/gautam_k/sftpush/BluTrain`, CUDA 13.0, NCCL 2.28.9, 80 cores, no gpu board, our GPU = 1, GPU 0 a colleague's): lib+dist exit 0 (libtensor b1b3947f), control pz + SFT trainer + loader test exit 0, 3 tests compile.
- H100 benchmark prep (for the post-push confirmation): /root/gautam_k/{flash-attention 94e22c9, scripts, loss_bench, step6, sft_data, sftb_1021, rope/(setup_1024 + port_1024 + rivals, TI -> sftpush tree)}, venvs benv (FA3 build) / lossenv / rope/rivalenv, ckpt/blumodel_step_8000 (md5 5119b20d verified).
**Why:** user (10-27): push all SFT work so colleagues can use it, then re-run the standalone benchmarks on the H100 to confirm the pushed kernels reproduce and still beat the rivals, then they decide next.

**Pushed 10-27 ~17:00 IST** (user): TI b1443a5..f7aa40f (bda3a34 rope, f7aa40f generic attention + tests), BluTrain 16bce82..a755643
(cb16155 TI bump, a755643 sft). TRAP: root .gitignore ignores *.py *.md *.png -> `git add sft` silently skipped 12 files;
fixed with sft/.gitignore (!*.py !*.md !*.png). Follow-up commits given to the user (sft files + DCP fix) — check pushed.
**DCP fix** (dist/Distributed_Checkpointing/src/DistributedCheckpointing.cpp): rescue_missing_shard_files required ALL files
in metadata (optimizer shards too) -> BLU_INIT_FROM crashed (exit 134) with only metadata.bin + data_rank_0.bin; now a
weights-only load (optimizer == nullptr) rescues only the module tensors' files. Verified on H100.
**H100 re-run of every 10-21/10-24 benchmark on the pushed code (box tree = pushed blobs, 1322/1322):** S1 292/292, FULL
1245/1245 (4 refusal checks removed 10-24), A1 scoreboard dense fwd+bwd vs FA3/FA4/cuDNN 0.981/0.964/0.879, packed
0.926/0.930/0.900 (H200 10-21: 0.982/0.970/0.873, 0.935/0.923/0.881), loss fastest 6/6 (Liger 1.07-1.22x), A4 old/new
0.53-0.99 bit-identical 6/6, A3b in-place bit-identical 12/12, RoPE bf16 16/16 fastest + 44/44 bit-identical x4 builds,
same-chats SFT run step 0 EXACT (1.449952 / val 1.446384). Results: profiling_reports/sft_benchmarks_2026_10_21/box_runs_h100_1027/.

**P032 re-run 10-27 19:29-19:53 IST (8x H200, GPUs 0-5, 6 lanes, pushed tree):** all pass and match 10-21. quick
292/292, full 1245/1245 (1375 s: CPU shared by 6 lanes), scoreboard dense fwd+bwd 0.980/0.963/0.867, packed
0.922/0.890/0.857 vs FA3/FA4/cuDNN; loss fastest 6/6 (Liger 1.16-1.21x on this box), old->new bit-identical, in-place
12/12; bf16 RoPE 16/16 fwd/bwd/total (min 1.05/1.12/1.14x), 44/44 x4 builds; SFT runs: padded step 0 exact, val
drop 0.1625; packed 0.1629; same-chats step 0 exact; step time -1.7..-2.6% vs 10-21; peak mem padded 117,039 (10-21
123,233), packed 82,987 (85,757) -- cause not checked. Results: profiling_reports/sft_benchmarks_2026_10_21/box_runs_p032_1027/.

(Moved from the MEMORY.md index line on 2026-11-05:) push = BluTrain 5c0999c -> TI f7aa40f.
