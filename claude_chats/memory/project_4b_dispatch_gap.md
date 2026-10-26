---
name: project-4b-dispatch-gap
description: 4B trainer is 11.2% behind cuBLAS end-to-end and 24% behind on GEMM, entirely because the tier-2 tree picks unclustered/unrastered configs; every config cuBLAS uses is already compiled.
metadata:
  type: project
---

Measured 2026-09-01, H200 x2, ZeRO-1 NP=2, 5 steps, nsys both arms, same script/box/day.

- cuBLAS 30,299 tok/s / 44.45% MFU vs BluBLAS 27,247 tok/s. We are at **89.9% of cuBLAS**.
- Total GPU: cuBLAS 349.6 s, ours 387.6 s. GEMM: cuBLAS 153.5 s, ours 190.4 s.
- **36.9 s of the 38.0 s end-to-end gap is the GEMM** (97%). Everything else is identical.
- We are at **80.6% of cuBLAS on GEMM** — our worst deficit on any model.

Cause is DISPATCH, not the kernels. Tier-2 (the fitted tree) handed 4B bare configs:
cluster 1x1 and rs0 on 6 of 12 call sites. cuBLAS clusters (1x2 or 2x1) and sets a
raster (h or v) on **every single GEMM**. Worst case is nt 8192,3072,1024 at **2.01x**
where we pick the SAME tile (128x192) and differ only by cluster+raster — a controlled
experiment showing clustering alone is worth 2x there.

Every tile/cluster combo cuBLAS uses for 4B is ALREADY COMPILED in our library
(256x128 cl1x2, 128x256 cl2x1, 192x192 cl1x2, 320x128 cl1x2, 128x192 cl2x1, 192x192 cl2x1).
Nothing needs writing — this is a level-1 table population job.

Shape mapping was established without any extra run: the two arms' GEMM launch sequences
align exactly index-for-index, and per-slot GFLOP/time lands at 700-820 TFLOP/s for every
slot, which pins each launch to a weight. See [[project_open_tasks_2026_08_28]] and
[[feedback_measure_against_dispatch_not_pins]].

**Outcome (same day):** 13 of the 15 shapes pinned in level-1 to cuBLAS's own geometry; mean
89.2% of cuBLAS standalone, ten shapes improved by up to 26 points. Two shapes deliberately
left out — see [[project-128x192-kernel-weakness]]. Trainer A/B DONE: **27,247 -> 28,703 tok/s (+5.34%), 89.9% -> 94.7% of cuBLAS**, GEMM 190.4 -> 172.2 s (80.6% -> 89.1%). Pins verified by nsys: kernel counts match cuBLAS 1:1 on all 8 slots. The standalone projection (+1,550) transferred (+1,456) — unlike H27bn. Note `lookup()` is a
binary search so rows must be inserted SORTED, and no new entry points are ever needed because
BLUBLAS_SM90_CLUSTER_GRID already emits every config for every layout.
