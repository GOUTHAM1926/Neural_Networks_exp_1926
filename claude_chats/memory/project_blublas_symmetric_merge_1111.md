---
name: project_blublas_symmetric_merge_1111
description: "11-11: BluBLAS merge a6d4292 (ours) + cb42c13 (Aparna symmetric): laptop + P015 H200 tests ALL PASS (switch off = identical to a6d4292; switch on correct; 30-38% faster at n 2304, slower at 256/512); push package push_1111 ready, user pushes"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-11T11:11:38.422Z
---

Aparna (chat 11-11): "just merge and push, I'll check" -- we apply her BluBLAS symmetric change on top of our fixes.
Her work: BluBLAS `symmetric_change` cb42c13 (= master e392dbb); TI dc7414f + BluTrain bad5ecc "muon_optimization"
on the COMPILER line (TI coconut/infer-comp/jit-sft-nova, BluTrain master_v2_compiler) -- only TI coconut calls the
symmetric mode; same-name `newton_schulz_orthogonalize_batched` as ours (TI merge = duplicate definition -> build
fails). Her P032 tree /localstorage/Aparna (read-only checked): 1.5B JIT before/after -0.35%/-0.36%/-0.4% (1/2/8 GPU).
Report for her: work_docs blublas_my_fixes_vs_symmetric_change_merge_plan.md (section 8 added 11-11).

**Merge (laptop scratch clone, branch merge_symmetric, `git merge --no-commit cb42c13`, NOT committed):**
- hunks 1,2,5,6,8 bias -> ours (FullBatched); hunk 3 -> our GPU tables + NEW `blublas_tile_order_table_sym_lower`
  (blublas_tile_order_tables.cuh, own store, 256 tables, 1-thread fill, count formula); hunks 4,7 -> her plan.map;
  sm90_tile_plan takes the stream; her old code kept as comments; her SM89 scale-kernel line commented (user OK,
  option 2: dead after 0b552f0's bias-copy fallback).
- Patch: profiling_reports/blublas_symmetric_merge_2026_11_11/merge_a6d4292_plus_cb42c13.diff (md5 29ad4b9d,
  applies on a6d4292).
- Laptop results (same folder): SASS 5,453/5,453 identical + 25 new (mirror x4, sym fill x21); sm_86 suite 7,220
  pass both, bitwise equal except 32 fp32 split-K (same 32 differ base-vs-base 11-05: atomics); tile list = her
  builder 1,160 cases / 718,335 entries; test_symmetric_merge S on 3060: 75 PASS (all same-as-OFF, as expected).
- H200 package: ~/Downloads/lc_box/symmerge_1111_pkg (298 MB; scripts in records pkg/): tests_h200.sh setup|G|A|S|P|T
  + lockrun_h200.sh (G0/G1 env, holders, never kills). H default $HOME/gautam_k/symmerge_1111.

**Next:** user gives a box + 2 H200 GPUs -> user uploads + runs setup and lockrun -> results -> then commit (user,
real merge commit with cb42c13 as parent, our resolved files) + push BluBLAS + point TI branches; phase 2 = her
symmetric flag + tall/fused ideas in OUR batched NS, A/B per piece (small 135M shapes may lose: mirror launch).
Third line: Kathir's BluBLAS MoE-Trials conflicts with ours in Bgemm/Hgemm warpspecialized templates.
Related: [[project_gemm_fixes_1105]], [[project_batched_ns_1104]], [[project_future_blublas_tile_map_sync]].

**P015 H200 results (11-11 17:44-18:47 IST, GPU 7 only per user; logs in records box_results_p015/):** GEMM 7,220 x 2 libs
0 failed, bitwise equal except 222 fp32 split-K = the same 222 differ base-vs-base (null N); addmm suite all pass, same
lines; symmetric S 75/75 (7 visibly mirrored = the per-batch-bias NS addmm); P: n 2304 ON/OFF 0.62-0.71, n 768 0.87/1.04,
n 512/256 up to 1.38 (slower) -> threshold needed; SFT 4+4 runs: same loss/dt, step-0 norm 2.5155-58 vs 2.5152-54
explained: the 84 SFT GEMM shapes bit-identical in 4 runs (so GEMMs are not the source; attention atomics).
P015 traps: ALL GPUs Exclusive_Process (holders block our own tests -> lockrun skips holders) + persistence OFF
(~10 s GPU init per process -> one process per type, not per group).
**Push:** records push_1111/commands.sh (real merge commit, parents a6d4292 + cb42c13, rehearsed: diff md5 29ad4b9d) ->
TI pointer -> BluTrain pointer. Aparna doc section 9 = results. After push: verify on the remote file by file.

**PUSHED 2026-11-11 ~19:10 IST by the user + VERIFIED on GitHub file by file:** BluBLAS dist_side_gautam_lottn
a6d4292..cfa0e92 (merge, parents a6d4292 + master e392dbb, diff md5 29ad4b9d, 8 files equal to records merged_files),
TI 4d5c738 (BluBLAS -> cfa0e92, only that path), BluTrain 597360b (TI -> 4d5c738, only that path); author Goutham
Kumar, no AI trailer. master (e392dbb) and preproduction-135M (BluBLAS 35e51a0, TI a9e37e6, BluTrain 60754ce)
untouched -- user's plan: Aparna checks on our branch first, then everything goes to preproduction-135M together.
Next: Aparna's check; phase 2 (symmetric flag + size threshold in OUR batched NS, trainer A/B); personal-repo docs push.

**11-11 evening:** the user sent the updated doc (Read-first section = the message; sections 8-9) to Aparna. Waiting for her check.

**Docs pushed 11-11 ~19:40 IST + verified:** BluTrain d9d3251 (sft/docs optimizer Qwen correction + README row for BluBLAS cfa0e92), personal repo 288f513 (Aparna doc, work log 45.190-45.193, Q&A 244-255, optimizer doc mirror). profiling_reports/ is gitignored -> records stay local. Next: item 2 = phase 2 (explain first, then GPUs).

**Phase 2 ON HOLD (user 11-11 ~19:45 IST): "first we will check, then only we will do, else we don't".** Plan explained (P1 symmetric with size threshold + sweep, P2 fused scale+cast, P3 tall without transposes, P4 apply from bf16 + publish); nothing written or run. Do not start without the user's word.
