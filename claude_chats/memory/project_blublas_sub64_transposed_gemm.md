---
name: project-blublas-sub64-transposed-gemm
description: FUTURE — BluBLAS bf16 GEMM with a transposed operand returns WRONG numbers (no error) when M/N are not multiples of 64; seen via Muon Newton-Schulz rejecting every update at 48x64 / 32x64
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-26T15:39:08.981Z
---

**Seen 2026-09-26** in the F1 test ([[project-f1-muon-v2-pushed]]). ShardedMuon's Newton-Schulz (`Optim.cpp:1524`, bf16 path) runs `cuda_matmul(Xb, Xb.t(), A)`. That goes to `blublasBgemmStridedBatched` with `opB = T` (`GenMatmul.cu:39`, under `WITH_MYBLAS`).
- Matrices 64x48 and 32x64 (so M = N = 48 / 32): every NS update was rejected by the `>1000` / non-finite guard, healthy data included.
- 256x192 and 128x256: 0 rejected.

**Why (likely, not isolated):** our SM90 residue notes ([[project_sm90_residue]]) validate the MN-major (TN/NN) layouts only for M%64 and N%64. There is no guard or fallback, so an unsupported shape silently computes garbage instead of refusing.

**Impact today:** probably none in training, because real matrices are multiples of 64. Also, the trainer prints `nsrej` every step, so a real run would show it.

**Future task:**
- Reproduce with a direct `cuda_matmul` vs CPU-reference test at M/N in {32, 48, 96, 100}.
- Then add either a shape guard with a cuBLAS or contiguous fallback, or real residue support.
