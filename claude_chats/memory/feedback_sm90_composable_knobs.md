---
name: feedback_sm90_composable_knobs
description: "SM90 module — make EVERY feature (incl. swizzle) a composable validated template knob, never hardcode"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-25T10:42:47.074Z
---

The SM90 GEMM module must expose EVERY feature as a composable template knob, universal across all
kernel/tile variants — Layout × Tile × Cluster × Multicast × Raster × **Swizzle (SW128/SW64/SW32/…)**
× coopA/B, etc. — the same way the cluster/multicast knobs already are. Do NOT hardcode a mode (e.g.
128B swizzle, the literal `64`/`SWIZZLE_128B` scattered in the descriptor + wgmma + consumer) inside
the variants.

**Why:** completeness + future-proofing is the goal (a full general BLAS, not just what bf16 needs
today). Hardcoding one mode is a temporary design; retrofitting/parameterizing it later across every
variant is a hectic, error-prone task. Parameterizing now — especially while the code is already being
touched (e.g. the residue descriptor rework) — is cheap.

**How to apply:** make each knob a SINGLE-source template param (like CUTLASS's swizzle-atom template
drives both `make_tma_copy` and `make_gmma_desc`) wired consistently through descriptor box[0] + TMA
swizzle enum + wgmma layout_type (B128/B64/B32) + consumer addressing + smem atom. BUILD and VALIDATE
all combinations (validation tests ≠ shipping them: bf16 production still selects SW128 because it
performs best — the module just also SUPPORTS + has validated the others). The user corrected an
over-conservative YAGNI stance here: for THIS module, generality-as-validated-knobs IS the deliverable.
Links [[feedback_completeness_goal]], [[feedback_scope_8_shapes]].
