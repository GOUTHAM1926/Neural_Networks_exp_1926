---
name: project_reduction_32bit_guard
description: GPU reduction is 32-bit-indexed; guard throws for >2^31 tensors; PyTorch-split is the deferred real fix
metadata: 
  node_type: memory
  type: project
  originSessionId: 60a6dc55-8dda-45fe-864f-235f3482492e
---

The GPU reduction kernels (`ReductionKernels.cuh`) index memory with **uint32_t** and keep per-launch counts (`num_inputs`/`num_outputs`) in **int**. So they only support tensors up to: numel ≤ 2^32-1, and num_outputs/reduced_count ≤ 2^31-1. Bigger tensors (e.g. attention scores `[B,H,T,T]` or logits `[B,T,V]` at 7B+/long-seq) would **silently overflow** → wrong results. Same latent bug TensorFlow's GPU reduce has; PyTorch/CUB avoid it.

**Done (June 2026):** added `ensure_32bit_reduction_indexing(input, reduced_count)` to `src/UnaryOps/cuda/ReductionImplGPU.cu` — a `static inline` helper called at the top of all 4 GPU dispatchers (value/index/mean/variance). Throws `std::runtime_error` (msg contains "32-bit"/"indexing") when numel > 2^32-1 OR num_outputs/reduced_count > 2^31-1. Does NOT reject currently-working cases like `[65536,50257]` logits (flat idx 3.29B < 2^32). `reduction_final_test.cpp` got 3 public-API guard tests (uint8, ~2-4GB, reduce_sum/mean) — all fire correctly; **90/90 on sm_86**. Guard is BBP-only so far; propagate to sm89/sm86/master_gau.

**Deferred real fix = PyTorch-style split** (user chose guard-now). Researched deeply (PyTorch/CUB/Eigen/TF). Design: keep the fast 32-bit kernel untouched; when a tensor fails the 32-bit check, **split the OUTPUT dim into <2^31 chunks** (disjoint → no accumulation, no precision change, 64-bit host base ptr + 32-bit local index — the clean case covering attention/logits). The rarer **reduced-axis split** (reduced_count>2^31, e.g. reduce-all of >2B) needs serial partial-accumulation (per-op: sum→add, max→max, mean→sum-then-div) — that's the bug-prone part, do carefully with tests. Full PyTorch mechanism notes (can_use_32bit_indexing, split/narrow/view_offsets, AccumulationBuffer, base_idx) are in [[project_reduction_bugs]] context / work_docs/data_accessing_techniques.md. Won't drop perf: common path unchanged. Performance verdict: split slightly beats CUB-uint64 (keeps 32-bit hot path); precision identical (index width ≠ accumulation).
