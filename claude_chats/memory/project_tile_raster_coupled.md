---
name: project-tile-raster-coupled
description: Tile orientation and raster direction are COUPLED - V2c gained +7.2 points from matching cuBLAS's (128x256, v) pair; sweeping (tile,raster) across all pins is the live lever
metadata:
  type: project
---

**FOUND 2026-08-27 (§H27b2). Tile shape and raster direction must be chosen TOGETHER.**
Measured on V2c (`2048x50304x16384 tn`), locked 1400:

    256x128 / h  (our pin)   92.4%
    256x128 / v              93.0%    raster alone:  +0.6
    128x256 / h              82.9%    tile alone:    -9.5  (!)
    128x256 / v              94.0%    together:      +1.6
    128x256 / v + ep32s2     94.6%

The transposed tile is **catastrophic** with the wrong raster and **best** with the right one.
Neither is meaningful alone.

**V2c: 89.7% -> 96.9% at locked clock, +7.2 points, bit-exact** using
`128x256x64_2x1_v_ep32s2_dh_shfl` (+5.8 from the shape, +1.4 from dh+shfl on top — they
compose). ncu showed cuBLAS running `nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN` there and
moving **6.807 GB vs our 7.765 (1.14x)** at equal tensor utilisation.

**PER-SHAPE, not config-wide.** V2, V2b and V2c share a config today but V2 (100.1%) and V2b
(99.8%) are already at parity locked and lose 0.2-0.45 with the new shape. Only V2c needs it —
so this requires a dispatch-table entry change for that one shape.

**THIS CORRECTS §H27aw**, which concluded "tile orientation is not the answer" after testing
256x128 arms against a 128x256 pin **without pairing the raster** — mismatched combinations.

**THE LIVE LEVER:** the (tile, raster) pair has NOT been checked against cuBLAS's choice for
the other 14 pins. V2c was found only because it stood out after clock and instruction
variables were removed. A cheap sweep — read cuBLAS's nvjet kernel name per shape, compare to
our pin, test the matching pair — is the natural next campaign and is where remaining points
most likely are. Instructions are done (§H27b1: vector density 0.58-9.25 -> 0.58-2.00, worth
only +0.5-0.9 total).

Locked-clock set mean with V2c fixed: **99.12% -> ~99.4%**.
See [[project-deadwork-removal-win]], [[feedback-vector-not-total-instructions]].
