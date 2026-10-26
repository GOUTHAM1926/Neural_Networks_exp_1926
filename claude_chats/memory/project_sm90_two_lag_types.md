---
name: project-sm90-two-lag-types
description: SM90 192x192 has TWO distinct lags - V6 idle-time lag (fixed) vs V5 extra-work lag (open); must be stated separately in the EOD
metadata:
  type: project
---

**Say these SEPARATELY in the EOD and in any status. They are different diseases and conflating
them has already caused one wrong fix to be attempted.**

**V6 = IDLE LAG (FIXED, 84.80 -> 98.53).** Tensor-busy cycles were IDENTICAL to cuBLAS (0.72%
apart) and wgmma counts bit-identical. The whole deficit was 34,797 idle cycles = 1,780/tile,
spent draining the epilogue TMA store at the END of each tile. The next tile's MAINLOOP never
touches c_buf, so the wait moved to the top of the next epilogue (`EPI_LATE_STORE_WAIT`). Same fix
took V3 91.30->99.44 and V4 91.08->99.79.

**V5 = EXTRA-WORK LAG (OPEN, 82.46).** Tensor-busy cycles are **+20.1%** and wgmma instructions
**+7.3%**. Both kernels are 98.4% tensor-bound, so V5 is purely tensor-throughput-limited and the
cycle ratio (1.2035) equals the tensor-busy ratio (1.201). `_lsw` gave **+0.00** here.
Split: +7.3% more wgmma (padding explains only 3.7%, ~3.5% unexplained) and **+11.9% more tensor
cycles per wgmma** despite an IDENTICAL `HGMMA.64x96x16 .tnspA.tnspB` on both.
Dead on V5: load imbalance (1.3% vs 1.1%), wave quantisation (1.7%), padding alone, `_lsw`,
different wgmma shape.

**V5 is apples-to-apples** (cuBLAS `nvjet_sm90_tst_192x192_64x3_2x1_v_bz_coopB_NTN`: same tile,
3 stages, 2x1, v raster) so there is NO structural excuse -- it just does 20% more work than the
config requires.

**V6 is the ONLY shape where cuBLAS uses 4 stages.** V3/V4/V5 all get cuBLAS's 3-stage kernel, so
depth 3 IS the match there. See [[project-sm90-depth4-parked]], [[feedback-samples-not-static-counts]].
§H27zdm/dn/do/dp/dq in analyzations_h200.md.
