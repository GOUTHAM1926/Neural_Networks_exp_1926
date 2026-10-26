---
name: scope-8-shapes
description: "P1 campaign scope = THE 8 shapes ONLY (128x256_2x2, M=4096) vs cuBLAS — never gate on 42-shape sweep averages"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
---

The P1-#1 campaign goal is to beat/match cuBLAS on THE 8 shapes where cuBLAS issues
the SAME nvjet_128x256_2x2 variant: 0.6b-K(4096,1024,1024), 1.7b-K(,,2048),
4b-K(,,2560), 8b-K(,,4096), 0.6b-O(,,2048), 0.6b-Down(,,3072),
1.7b-Down(4096,2048,6144), 3b-Down(4096,2048,11008).

**Why:** apples-to-apples vs cuBLAS. Skinny shapes (N<=512) are a FUTURE task
(split-K, §63.1) — cuBLAS uses different kernels there.

**How to apply:** gate every kernel change with `examples/p8_check.sh` (per-shape
duration vs cuBLAS, §60.31-comparable). NEVER quote the template test's 42-shape
sweep average (85-86%) as the campaign metric — it mixes in skinny shapes and
misleads (caused the 2026-07-16 split-K detour). §60.31 baseline: 0.6b-K 79.6%,
gap shrinks with K (8b-K ~95%).

Related: [[p1-campaign-state]]
