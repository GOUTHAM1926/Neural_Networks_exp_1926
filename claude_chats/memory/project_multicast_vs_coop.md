---
name: project-multicast-vs-coop
description: coopA/coopB in cuBLAS nvjet kernel names is a TILE-SHAPE tag, not multicast and not the CUDA Cooperative Groups API; the real missing feature is CUTLASS's Pingpong schedule.
metadata:
  type: project
---

**coopA/coopB is NOT multicast.** An earlier version of this note said the opposite; §H17 refuted
it from 60,733 captured cuBLAS kernel names: `1x2_h_bz_coopB` is the MOST COMMON kernel (23,325
shapes), and a 1x2 cluster multicasts **A** — so coopB cannot mean "B-multicast".

**What it actually tags is tile shape** (§H17 + template.cuh:971):
- `coopA` -> 128x256, 256x128 — both dims are EVEN multiples of 64
- `coopB` -> 192x192, 320x128, 192x128, 128x192 — a dim is an ODD multiple of 64
- no tag -> 112x128, 128x128, 176x128
Our own template states it: *"coopA = blocks/WG even, coopB = odd. Same emitted code either way --
an exhaustive 18-kernel opcode census found no instruction that differs."*

**It is NOT the CUDA Cooperative Groups API** (checked 2026-09-08 against
docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html — that page is
thread-sync only: `thread_block`, `tiled_partition`, `sync`, `reduce`; zero GEMM/tile/operand/TMA/
wgmma content). Same word, unrelated thing.

**The real concept, and the actual gap:** CUTLASS's **Cooperative vs Pingpong SCHEDULES** (separate
files, not a flag). Cooperative = 2 consumer warpgroups split ONE tile's M
(`AtomLayoutMNK Shape<2,1,1>`), **requires TileShape_M >= 128, falls back to Pingpong at M=64**
(sm90_gmma_builder.inl:1020-1022). Pingpong = each warpgroup owns a SEPARATE tile + an
OrderedSequenceBarrier. Multicast is a SEPARATE axis, derived only from cluster shape
(A mcast iff clusterN>1, B iff clusterM>1).

**Why this matters now:** it explains §H157/§H158. We have no Pingpong, so at BM=64 we must pick
between NCW=2 (M-split illegal -> half-width wgmma) and NCW=1 (full width, but half the consumer
warpgroups) — measured -9% to -34% and -9.98% respectively. cuBLAS/CUTLASS switch SCHEDULE instead.
**Pingpong is the named, documented feature to build if BM=64 is ever wanted.**

See [[project_sm90_roadmap]], [[feedback_kernel_name_suffix_inference]], [[project_hybrid_feature_verdicts]].
