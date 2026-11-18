---
name: project_batched_ns_1104
description: "11-04 batched Newton-Schulz + per-batch bias + addmm bias FIX (BluBLAS+TI) all TESTED on P032; B0->B3 SFT step 137.4->110.3 ms (-19.6%); NOT pushed (push after the 4 old GEMM bugs are fixed 11-05)"
metadata:
  type: project
---

Work doc 45.146-45.149. Patches in profiling_reports/muon_batched_ns_2026_11_04/ (blublas_per_batch_bias, ti_batched_ns,
blutrain_batched_ns); local trees ~/Downloads/BluTrain_push (+ TI, + BluBLAS submodule) hold the changes uncommitted.
P032 /localstorage/gautam_k/muon_guard_p032/bt_bns = built new tree (old = bt_new).
Results: unit tests pass (batch 37 bit-identical NS + weights; batch 5 rounding-level diffs); trainer 1 GPU 129.05 ->
111.45 ms (-13.6%), 2 GPU BluCCL pipelined -12.4%, fused -11.3%; losses within noise; nsrej equal.
Existing bug found: BluBLAS fallback addmm (sm_8x, or H200 when K%64/N%64 != 0) = alpha*AB + beta*C_old + bias, not
alpha*AB + beta*bias (TI's contract) -> separate fix needed; check callers.
UPDATE 21:55: addmm fix done (work doc 45.150-45.152): BluBLAS bf16/fp16 fallback = copy bias + ONE GEMM(alpha, beta),
beta 0 = plain GEMM, unsupported size = stderr + no bias; fp32 SM89 bias-row bug fixed; TI cuda_addmm routes by bias SHAPE.
SASS 733->739: 723 identical, 10 changed (fp32 addmm), 6 added. Tests all pass except cases hit by 4 OLD GEMM bugs
(see [[project_tomorrow_1105]]). P032 trees: bt_old B0, bt_new B1, bt_ec B2, bt_all B3. Results pulled to
profiling_reports/muon_batched_ns_2026_11_04/p032_results_2155/.
**Why:** user wants all tests done before pushing; user's GPU window 11-04 ended 20:30 (colleague's turn).
**How to apply:** next = docs + push (user decides), then the fallback-addmm fix as its own item. Ask before any GPU use.
Related: [[project_muon_gpu_guard_1104]], [[project_empty_cache_test_1104]].
