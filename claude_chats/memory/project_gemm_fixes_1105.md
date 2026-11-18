---
name: project_gemm_fixes_1105
description: "11-05: 13 BluBLAS GEMM/addmm bugs fixed + proven (2x H200 + RTX 3060); PUSHED 14:45 IST + VERIFIED on remote: BluBLAS a6d4292, TI df615ce, BluTrain 96f135c (31 files equal, pointers ok)"
metadata:
  type: project
---

Work doc 45.154-45.160 (45.160 = push + remote check). Fixes in ~/Downloads/BluTrain_push/Tensor-Implementations/BluBridge-BLAS (uncommitted until the
user runs profiling_reports/blublas_gemm_fixes_2026_11_05/push_1105/commands.sh: BluBLAS 2 commits -> TI 2 commits ->
BluTrain 1 commit; then VERIFY every file on the remote against push_1105/local_md5.txt (31 files)).
Bugs (all pre-existing; 0 regressions):
1 SM90 C stored via 2-D [batch*M,N] TMA -> ragged-M spill into next batch (race); fix 3-D C descriptor+store (CUTLASS L mode)
2 misaligned shapes -> new blublas_gemm_generic.cuh + 16-byte guards in dispatchers (CUTLASS can_implement style)
3 fp32 SM89 TN K-tail stale smem -> zero-fill; 4 TT broken (bf16/fp16 kernel, fp32 routed to NN) -> generic
5 fp32 small-shape fallback always split-K (beta lost); 6 fp32 NT 256x128 prologue committed < STAGES-1 groups (race)
7 same kernel no unaligned path -> route; 8 TN 32x32 swizzle overflow -> mask by BM; 9 bf16/fp16 gemv early return
before barrier; 10 fp16 SM90 TN M%128; 11 fallback CfgL BK=32 vs kernel BK=16 -> static_assert; 12 fp32 addmm kernel
ignores IsAligned -> route unaligned to copy+GEMM.
13 (found on the laptop RTX 3060, 45.158): fp32 sm_80/86 kernels (64x64 NN/NT/TN in Sgemm_tile64_SM86.cu, 128x128 NT/TN
in Sgemm_backward_{nt,tn}_SM86.cu) committed the cp.async group only inside `if (tile to load)` -> wait_group 1 skipped the
last K tile -> wrong values under load (batched); fix = commit always (empty group), like the sibling kernels + CUTLASS.
5 paired rounds: unfixed 148-160 failures/round, fixed 0. SAME defect left in DEAD code (no caller/header): bf16/fp16 SM8x
backward NT/TN (K<=32), old wmma bgemm/hgemm_async_strided_batched_kernel, blublasSgemm_TensorCore_Explicit, unused SM89
fp32 `_2` launchers -- listed in 45.158, not changed (user may decide delete/fix).
SASS each step: only intended kernels changed. Trainer step time unchanged. New lib deterministic run-to-run; fp32
split-K is not (atomic order) by design.
2x box 86.38.238.51 (root; Ananya also uses it): /root/gautam_k/gemm_fix_1105 has bt_b3 (11-04) and bt_fix2 (all fixes),
data_4k, ckpt_A, libyaml-cpp installed (user OK). GPU lock that WORKED: nvidia-smi -c EXCLUSIVE_PROCESS (root), restore
DEFAULT via EXIT trap; it also refuses our own second process ("device busy") -> never run two of ours on one GPU.
3060 test kit: profiling_reports/blublas_gemm_fixes_2026_11_05/run_local_sm86.sh (+ stress_sm86.sh); sm_86 library =
scratch copy built with `make -j12 blas SM_ARCH=86` after building the profiler (link needs it).
**How to apply:** after the user pushes, verify on remote; then task 3 (pretraining before/after on the 2x box with the
local flux_bin_data shards as a fake mix -- user's instruction 11-05 13:00), task 4 (CUDA-call items 45.140), task 5 (CP
wrappers). Related: [[project_tomorrow_1105]].

**PUSHED 2026-11-05 14:45 IST (user, commands.sh) and VERIFIED 14:51 IST (verify_remote.sh):** BluBLAS 0b552f0 + a6d4292,
TI dbb4e5f + df615ce, BluTrain 96f135c on dist_side_gautam_lottn; all 31 files byte-equal on the remote, both submodule
pointers correct, author Goutham Kumar, no AI trailer. verify_remote.sh (push_1105/) = the reusable remote check.
