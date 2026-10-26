---
name: project-tile-order-verdicts
description: Kota's three-way tile-order policy — we already own 2 of 3 branches, his thresholds are tile-specific and one is a 7.6% loss, Hilbert is now ours as TILE_ORDER.
metadata:
  type: project
---

His SM103 policy is three-way (documented at `Bgemm_sm103_cluster_template.cuh:1975`):
square→Hilbert, tall→row-major, wide→col-major. **His SM90 port dropped the wide/col-major
third entirely** and tightened tall to `gm > 4*gn`.

**Two of three branches are existing knobs of ours**, verified by tracing the decode, not by
name: tall row-major = `RASTER=1, SW=0`; wide col-major = `RASTER=0, SW=0`. Our dispatch table
already picks his tall rule on **8/8** tall production shapes.

**His thresholds must NOT be adopted as rules — both are tile-specific:**
- tall→SW=0 wins on 128x256 (0.25–0.47%, t=14–49) and **reverses on 256x128**
- wide→col-major on lm_head-fwd: pure col-major (his exact rule) is **−7.62%, the WORST of six
  arms**; the WINNER is col-major **with grouping**, **+0.84% (t=34.6)** — a combination his
  code cannot express, because his tile map and `SUPER_M` are mutually exclusive

**Hilbert is now implemented as `TILE_ORDER`** (slot 49; 0=Raster, 1=Hilbert), on-device decode,
gated to power-of-two square MEGA-grids (`md.hil_n`); a host map would be needed for other
square grids because his builder compacts the curve. Worth only +0.16%/−0.20%; square grids are
9 of 165 shape×tile combos. It composes with clusters and grouping — his SM90 cannot (his tile
map and cluster code are in separate files).

**How to apply:** measure tile order per tile, never generalise a threshold. Related:
[[project-tile-raster-coupled]], [[feedback-vacuous-checks]].
