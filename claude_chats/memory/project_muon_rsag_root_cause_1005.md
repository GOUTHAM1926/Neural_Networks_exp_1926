---
name: project-muon-rsag-root-cause-1005
description: "Navin's flaky test_muon_rsag (W=1) ROOT CAUSE proven 2026-10-05 — float atomicAdd order in TI muon_sumsq_kernel, not his Muon code"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-05T16:33:34.001Z
---

PROVEN 2026-10-05 on H200 GPU 2 (BLUCCL clone b9b567f, TI a60099b, cuBLAS build):
- Root cause = TI `src/Kernels/cuda/optimizer/MuonKernels.cu:77` `atomicAdd(acc, thread_sum)` in `muon_sumsq_kernel` (:22), called by `muon_frobenius_scale_cuda` (:123) from `newton_schulz_orthogonalize` (`Optim.cpp:1548`). 96x96 tensor = 9 blocks; 9! add orders give 3 float totals (423b7d7d/7e/7f); only ...7f (29,736 of 362,880 orders) changes the scale, and that is exactly the bad output.
- Navin's ShardedMuon math is correct: weights, grad, momentum, NS input bit-identical in RB and RS, all steps, all 3 commits (HEAD, a3aba3e, e6eaeba).
- Odd value only in REDUCE_BROADCAST (17/20 HEAD, 11/20 a3aba3e, 0/20 e6eaeba; RS 0/40). a3aba3e changes GPU state before NS so the bad order shows; it does not cause it. Our F4 cleared (a3aba3e alone fails 7/10).
- Reproduced OUTSIDE the test with the real library function: L2 flushed 16/200, varying busy SMs 38/200; idle 0/200.
- Retracted: 09-30 sync/pinned-buffer lead, initcheck "uninitialized read" (false alarm), atomicAdd "refuted" (probe4 idle was 200/200 only because an idle GPU keeps the same order).
- FIX = V1n (user chose; Navin's ZeroKernels two-pass pattern): block partials + 1-block fixed-order finalize (`*acc =`, memset gone), red_blocks = min(768, need4) from numel only (768 < H200's 792 resident; cap 1024 = 2nd wave +7.7us). Patch `profiling_reports/muon_bug_2026_10_05/fix_v1n/patch_muon_v1n.py` (md5 guard f6d821c2…, 4 edits, new md5 4e1d9393…).
- VALIDATED 2026-10-05 GPU 1 (scratch TI `~/gautam_k/muon/fixroot`, neighbour symlinks needed or make rebuilds 151 files): same binary orig lib 19/20 FAIL vs fixed 0/20; stress 800/800 one result; NS outputs == old normal values. Precision: worst 18.4 ulp -> <=1 ulp; speed +0.3..+2.8 us/matrix.
- PUSHED 2026-10-05 (user, from box clone, company identity): TI dist_side_merge_BLUCCL a60099b..a9bc08c (MuonKernels.cu only, md5 == tested) + BluTrain dist_side_merge_BLUCCL b9b567f..35df307 (TI pointer only). Box clone's libtensor.so is STALE (built pre-patch) until `make -C Tensor-Implementations tensor WITH_CUBLAS=1 SM_ARCH=90`. Still to do: tell Navin + Kathir; dist_side_merge (non-BLUCCL) not patched. Untested: W=2, single-GPU nn::Muon, full training.
- Files: `profiling_reports/muon_bug_2026_10_05/` (run_debug*.sh, ns_probe4/5.cu, box_logs/muon_logs_2026_10_05.tgz).

Related: [[project-f4-fix-plan]] (where the flaky test was first seen).
