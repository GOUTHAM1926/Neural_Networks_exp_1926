---
name: project_sm90_roadmap
description: "SM90 GEMM module feature roadmap — DONE vs UNDONE (corrected: coopA/B is NOT multicast)"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-25T13:21:40.380Z
---

Corrected feature inventory for the complete/universal SM90 BF16 GEMM module (analyzations_h200.md
§H15-H17). Goal: robust, efficient, future-proof, ALL features as validated composable knobs/variants.

DONE + H200-validated:
- Layouts NT/NN/TN; native MN-major TMA (no transpose, 13-19x faster than old); raw path deleted.
- Residue/partial for all real workloads (NT arbitrary M,N%8,K%8; TN/NN M%64,N%64,K%8; partial clusters).
- Clusters: 1x1, 1x2 (A-mc), 2x1 (B-mc) all layouts. 2x2 (simultaneous A+B mc) implemented, awaiting H200.

UNDONE (corrected — the earlier "coopA/B done" claim was WRONG; see §H17):
Tier 1 completeness (cluster axis; mechanism generic, config+gate):
- 2x2 validate (in flight); 1x4 / 4x1 / 4x2 cluster shapes.
Tier 2 perf knobs / tile+schedule menu (match cuBLAS per-shape):
- Raster h/v: mega-tile order is FIXED -> make it a knob (M-fastest vs N-fastest). NEW.
- Swizzle knob: descriptor side done (sw_tma_enum, SwizzleBytes param); finish consumer (wgmma
  make_gmma_desc layout_type) + validate SW128/SW64/SW32. Half-done.
- coopB / ODD-64-multiple tile dims (192, 320): our kernel is cooperative-only + EVEN-tile-only
  (WG_M=BM/2, ATOMS_PER_WG=WG_M/64 must be int -> BM%128). coopB = odd tiles needing a different
  warpgroup partition = STRUCTURAL variant. NOT multicast (proven §H17). UNDONE.
- Pingpong schedule (M=64 / alternate perf): each warpgroup owns a SEPARATE tile + order-barrier;
  STRUCTURAL variant (separate mainloop), CUTLASS falls to it at M=64. UNDONE.
- b1 sub-64 MN residue (rare); more tile shapes as needed.
Tier 3 capstone:
- Clock-locked benchmark sweep -> dispatcher winner-table. Needs the menu.

KEY facts (proven): coopA/coopB tracks TILE SHAPE (coopA=even-64-mult dims 128/256; coopB=odd 192/320),
orthogonal to cluster+multicast. Cooperative vs Pingpong = structural SCHEDULES (not knobs). Multicast-
operand = separate cluster-derived axis = our MULTICAST_A/B knob. See [[feedback_sm90_composable_knobs]].

SEQUENCE (user-set 2026-07-25 — DISPATCHER IS LAST; build ALL variants first, a winner-table is
meaningless with only 3 variants): Phase1 finish FEATURES as efficient measured knobs (raster h/v
[also fixes tall-skinny M=151936 ~70%], swizzle consumer+SW64/32, clusters 1x4/4x1). Phase2 STRUCTURAL
variants for completeness (coopB odd-64-mult tiles 192/320 = warpgroup generalization; Pingpong
schedule) — built regardless (completeness), NOT gated on measurement. Phase3 full TILE MENU (192/320
via coopB, 128x128, +split-K for small-K ~74% if it helps). Phase4 ALL-COMBINATION perf tests (every
layout x tile x cluster x raster x swizzle x schedule vs cuBLAS, nsys-verified). Phase5 DISPATCHER last.
EFFICIENCY woven in: each variant made efficient + %cuB/nsys measured AS built (not a separate phase).
MEASURED baseline: ~90% cuB, NN/TN=NT parity; pipelined-wgmma REGRESSED+reverted (§H16.7). nsys practice
adopted (§H16.8): run tests under `nsys profile --stats=true` to verify which kernels ran (ours+cuBLAS).
Ties [[project_sm90_residue]], [[project_sm90_native_mn_major]].
