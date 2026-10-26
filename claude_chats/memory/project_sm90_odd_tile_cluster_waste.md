---
name: project_sm90_odd_tile_cluster_waste
description: SM90 clusters round the clustered axis UP, silently burning whole tile rows; cluster the LONG axis. Fixed V5 82.2->92.8.
metadata:
  type: project
---

`mega_grid_m = ceil(grid_m / CLUSTER_M)` in Bgemm_sm90_warpspecialized_template.cuh
(~line 3799) rounds the clustered axis up, and there is NO bounds guard anywhere
(`block_row >=`, `>= grid_m`, `A_x >= M` all return nothing). Phantom CTAs run a
FULL tile of wgmma on OOB-zero-filled operands and their store is dropped out of
bounds -- so **maxRelErr stays 0.00000 and no correctness test can ever catch it**.
The code already flagged it at ~line 3796: "Partial last cluster row/col not handled yet".

Detect it with the tensor counter, which is exact and noise-free:
`sm__inst_executed_pipe_tensor.sum / (k_iters * 96)` = an integer CTA-tile count.
V5 (2048x11264x16384 tn): ours 17,399,808 -> 708 = 12x59; cuBLAS 16,220,160 -> 660 = 11x60;
real work 649 = 11x59. M=2048 gives ceil(2048/192)=11 M-tiles, ODD, rounded to 12.

FIX = cluster the LONG axis, not the short one. Padding the axis with 59 tiles wastes one
short column (1.7%); padding the axis with 11 wastes one long row (9.1%). Same cluster size.
Switching V5's pin from `2x1_v` to `1x2_v` (only CLUSTER_M/N swapped; multicast re-gates
itself via NEEDS_A = MC_A && CLUSTER_N>1) gave **82.20 -> 92.80 %cuB, ranges separated,
bit-exact**. cuBLAS does the same thing -- its kernel is 11x60, and its `coopB` name is the
odd-tile cooperative variant we don't have.

Beat the +7.3% forecast at +12.4% TFLOPS because of WAVE QUANTISATION on top:
708 CTAs/132 SMs = 5.36 -> 6 waves; 660/132 = 5.00 exactly. Always check both.

V5 was the ONLY one of the 15 shapes with an odd clustered-axis tile count (V3/V4/V6 86,
V7/V8 128, V1/V2 16/8/22 -- all even), which is why it was the only laggard.
See [[project_persistent_cluster_grid_sizing]], [[project_step3_plan]], [[feedback_scope_8_shapes]].
