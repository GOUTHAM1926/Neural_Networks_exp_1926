---
name: project_persistent_cluster_grid_sizing
description: "Persistent clustered kernels must size the grid from real cluster co-residency (cudaOccupancyMaxActiveClusters), never sm_count/cluster_size"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-27T14:55:02.130Z
---

For a PERSISTENT kernel launched with thread-block clusters, the grid must be sized from
the number of clusters that can actually be CO-RESIDENT — never `sm_count / cluster_size`.

**Why:** on Hopper every CTA of a cluster must be co-scheduled within one GPC, so a GPC
whose SM count isn't a multiple of the cluster size strands the remainder. Measured on
H200 (132 SMs, 1 CTA/SM): cluster 1 → 132 resident, cluster 2 → 132 (0 stranded),
cluster **4 → only 120** (30 clusters, 12 CTAs stranded). Over-sizing is not a rounding
error in a persistent kernel — a stranded cluster can't start until a resident CTA
finishes its *entire* grid-stride loop, so it runs as a whole extra wave. Under-sizing is
safe (the loop just takes more iterations). Cost measured: **~36 points of %cuB** on every
size-4 cluster variant (90% → 53%); fixing it recovered ~20-26 points (→ 73-74%).

**How to apply:** query `cudaOccupancyMaxActiveClusters(&n, kernel, &cfg)` with the real
smem + cluster attribute, accept only if `n * cluster_size <= sm_count` (CUTLASS's guard,
tile_scheduler_params.h:268), else fall back to CUTLASS's `get_max_cta_occupancy`
(tile_scheduler_params.h:54-78, `max_sm_per_gpc=18`). CUTLASS uses raw `sm_count` ONLY
when cluster_size==1. Note the driver is less conservative than the heuristic (30 vs 29
clusters on our H200) so prefer the query. Cache per kernel+cluster size.
Also: `cluster_launch.hpp` validates only `grid % cluster == 0`, with **no co-residency
check** — an over-sized clustered launch succeeds silently, it does not error.

Residual after the fix is physics, not a bug: size-4 caps at 120/132 = 90.9% occupancy, so
**prefer cluster size 2 on a 132-SM H200**. Check the same rule if [[project_sm103_cuh_locked]]
ever uses clusters with a persistent grid. See [[project_sm90_roadmap]].
