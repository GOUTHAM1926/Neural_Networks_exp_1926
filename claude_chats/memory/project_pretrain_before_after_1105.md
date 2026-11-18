---
name: project_pretrain_before_after_1105
description: "11-05 pretraining before/after of the 3 changes (production 1.6B YAML, 2x box, flux shards): optimizer phase 200 -> 182-184 ms/step (-16 to -19 ms, nsys); total 21.3 s step only -0.07%; empty_cache stall gone; YAML monitoring.production bug"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T08:53:36.814Z
---

Work doc 45.159; kit + results in profiling_reports/pretrain_before_after_2026_11_05/ (setup_pz.sh, make_cfg.sh, run_pz.sh,
queue_ab.sh, sample_env.sh, parse_pz.py, neighbor_fit.py, nsys_pz.sh, nsys_pz_g1.sh, nsys_gap.py, nsys_table.py;
box_runs/, box_nsys/). Box 86.38.238.51: /root/gautam_k/pretrain_1105 (bt_b0 = 7184fe1/d18f820/297f372 built there; AFTER =
/root/gautam_k/gemm_fix_1105/bt_fix2; bins bz_b0 4bea9f34, bz_after 99a77866; mix/ = 4 domain folders linking
flux_train_000001 + flux_val_000000).
Data: local ~/Downloads/Neural_Networks_exp_1926/flux_bin_data (100 train + 1 val shard, 200 MB = 100M raw uint16 GPT-2
tokens each, no header = DataLoader.h format); 1 train + the val shard is enough for short tests (user's idea, correct).
Production YAML does NOT load in this trainer: "unknown key 'monitoring.production'" (ConfigBridge.h never got the key
from Grafana commit dba5cf4) -> test used a copy minus that line; fix = one table line {"monitoring","production",
"BLU_PRODUCTION",Kind::Bool} (monitor reads BLU_PRODUCTION as 1/true/yes) -- NOT applied, user decides.
Results (1 GPU, global batch 1M tokens = 32 micro-batches, 21.3 s/step):
- total dt: change ~ -14 ms (+-8) = 0.07%; a colleague's job on GPU 1 moved our step by +66 ms (clock coupling) ->
  total dt can't resolve it; ABBA alone was fooled by the neighbour switching on/off.
- nsys optimizer phase (gap between embedding-forward starts at step boundaries minus the in-step gap), 6 runs:
  GPU 0 B0 200.79/201.05 -> AFTER 184.22/184.57 ms = -16.5 ms (-8.2%; work -8.1, idle -7.9); GPU 1 B0 199.55 ->
  AFTER 180.35 = -19.2 ms (-9.6%); same build repeats < 0.4 ms. B0's step-5 empty_cache stall 0.3-3.1 s once per run
  (gone in AFTER). Total effect on a 21.3 s pretraining step ~0.08% (optimizer ~0.2 s of it).
- training unchanged: step-0 loss identical; backward non-deterministic run to run (step-0 grad norm differs) in both
  builds; loss spread B0 vs AFTER = same-build spread; 0 NS rejections.
**How to apply:** for optimizer-only changes measure the optimizer phase (nsys boundary gaps), not total dt; check the
neighbour GPU; times in IST.
Related: [[project_batched_ns_1104]], [[project_gemm_fixes_1105]], [[feedback_times_in_ist]].

**SFT profile 11-05 (45.161, pushed code, GPU 0):** step 112 ms = 2 micro-batches 96.7 ms (GPU idle 4.6) + optimizer
window 15.3 ms (GPU busy 10.0 + idle 5.4). In that window: 14 cudaStreamSynchronize (5.3 ms), 379 memsets (zero_grad),
525 D2D copies (copy loops), 1,185 launches. Item 9 (cudaGetDeviceProperties) = 0.012 ms/step -> no value (code kept
local). Task-4 order by time: items 5-7 (syncs) -> 4 -> 3 -> 8; ceiling ~5-10 ms per SFT step.
**Item 9 PUSHED 16:05 IST + verified:** TI 963df8b (get_arch hardened: atomic flags, 64 slots + bounds, cudaDeviceGetAttribute;
SM90 attention launcher check via get_arch; old code kept as comments), BluTrain dfc3c9b. 45 get_arch() calls in 14 TI files.
Next: items 5+6+7 (14 syncs per SFT step).
**Item 4 PUSHED 16:48 IST + verified:** BluTrain 084114b (zero_grads_batched in ZeroOptimizerBase.h -> TI multi_tensor_zero_cuda;
3 zero_grad() use it at 1 GPU; SFT memsets 395 -> 23/step, step -0.7 ms). Item 3 code saved in items_3_4/saved/ (TI
multi_tensor_copy + 3 copy loops), NOT in the trees; next = explain item 3, wait for OK, then restore + build + test.

**Item 3 (copy loops batched) DONE + TESTED 11-05 18:52 IST, push pending (user runs items_3_4/push_item3/commands.sh,
then I run verify_remote.sh):** TI multi_tensor_copy_cuda (new, any alignment, 4 units in flight per thread) + zero /
cast kernels made alignment-safe (they crashed "misaligned address" on unaligned slices of the packed flat buffers) +
fp16 fallback converts once + every launch checked; BluTrain ZeroOptimizerBase.h helpers stage_grads_fp32_batched /
scatter_flat_batched used by ShardedOptimizer AND SymmetricShardedOptimizer (the saved draft had missed Symmetric) +
ShardedMuon scatter. Checks: 6 archs compile, CPU emulation of the real code, H200 + RTX 3060 unit tests, SFT -1.2 ms/step
(copies 558 -> 36/step), pretraining new path runs (D2D 369 -> 1/step) + losses within noise. Work doc 45.167. Tests on
P015 (2x box closed) ~/gautam_k/item3_1105.
**Item 13 (Muon per-matrix kernels batched) IN PROGRESS 18:56:** code in profiling_reports/.../item13/work (NOT in the
local trees until item 3 is pushed: ShardedMuon.cpp is in both items); bt_13 on P015 ~/gautam_k/item13_1105; tests_13.sh
(unit bit-identity, test_muon_rsag W=1/2 on GPUs 6,7, SFT 30-step A/B on GPU 7). run_muon_rsag_test.sh's include list
is stale (needs dist/Distributed_Checkpointing/include) -- pre-existing, fixed only in our build_rsag.sh.
**11-05 19:24 item 3 PUSHED + VERIFIED (TI 05e1d6c, BluTrain 07600db). Item 13 TESTED 19:40, push package
item13/push_item13/ ready (user runs commands.sh, then verify_remote.sh):** SFT -2.76 ms/step (P032 GPUs 6+7 swapped),
launches/step 4,390 -> 3,303, optimizer window 12.08 -> 8.48 ms; unit bit-identical; test_muon_rsag W=1/2 PASS;
pretraining losses within noise. Work doc 45.167 (item 3), 45.168 (item 13). OPEN: copy/zero size split (>= 128 MB to
the driver) so batched copy/zero are never slower for huge tensors (single 256 MB copy: kernel 174 us vs memcpy 133 us);
item 11 re-check (stager pool already reuses pinned buffers -> probably first save only); items 5+8, 6, 7, 12.
**11-05 19:48 item 13 PUSHED + VERIFIED (TI e2af706, BluTrain 1bec9af).** Clean pretraining rerun: no measurable change,
losses within noise. NEXT (proposed, waiting for the user's OK): 128 MB size split in TI copy / zero launchers (tensors
>= 128 MB -> cudaMemcpyAsync / cudaMemsetAsync, rest batched) so items 3 / 4 are never slower at any size; first 1-min check
zero vs memset at large sizes. Then items 5+8 (by-value Adam / scale / norm + alignment fix, then drop the syncs), 6, 7, 11
(re-check: stager pool), 12.
