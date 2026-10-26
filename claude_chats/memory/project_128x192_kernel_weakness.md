---
name: project-128x192-kernel-weakness
description: FUTURE TASK - our 128x192 2x1 v runs 2.1x slower than cuBLAS's on the identical geometry for tn 3072x1024x8192; this is a kernel weakness, not dispatch, so no table entry can fix it.
metadata:
  type: project
---

Parked 2026-09-01 by the user, to be picked up after the hybrid BluBLAS work.

On 4B shape **`tn 3072x1024x8192`** we run the SAME tile, cluster and raster as cuBLAS
(128x192, cluster 2x1, raster v) and reach only **47.4% of cuBLAS** — our worst shape anywhere.
Because we are already asking for exactly cuBLAS's geometry, this cannot be fixed by any
level-1 entry; it is the kernel itself. The shape is therefore deliberately LEFT OUT of the
dispatch table so it keeps falling through to tier-2.

Starting facts:
- `128x192 2x1` has only **2 registered configs** and no knob stacks (no dh/ubd/rs/skew),
  unlike 256x128 (21) or 192x192 (31).
- Small shape (M=3072, N=1024) so wave quantisation and epilogue cost weigh heavily.
- cuBLAS's kernel is `128x192_64x5_2x1_v_bz_coopB` — pipe depth **64x5** against ours.

Candidate steps: compile knobbed 128x192 2x1_v variants and re-race; check wave count and
cluster waste (see [[project_sm90_odd_tile_cluster_waste]] — clusters round the clustered axis
up and burn whole tile rows with maxRelErr staying 0); diff our SASS against theirs.

A second, smaller one parked with it: `nt 8192x8192x3072` on 320x128 1x2 h gives 78.1%, twelve
points WORSE than the 256x128 1x2 tier-2 already picks — also left out of the table.

Full detail in analyzations_h200.md §H63. Related: [[project_4b_dispatch_gap]].
