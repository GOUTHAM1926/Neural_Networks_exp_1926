---
name: project-mn4d-multicast-incompatible
description: Rank-4 MN-major TMA cannot be used with a multicast operand — one TMA writes contiguously, a K-sliced load needs strided sub-blocks. §H82's +1.46% is void.
metadata:
  type: project
---

The rank-4 (MN4D) TMA path gave **50% of C wrong on every NN shape with a cluster**, every bad
element exactly HALF its true value. Clean at cluster 1x1, broken at 2x1.

**Mechanism:** for an MN-major operand, multicast partitions along K — each CTA loads
`BK/CLUSTER_M` of the k-range. The consumer expects the rank-3 layout, atom-major with slices
nested inside each atom, at `i*ATOM_BYTES + s*slice_bytes` — **strided**. A single rank-4 TMA
writes its box **contiguously**, so no `dst` value can scatter into those sub-blocks. Not
fixable by correcting the offset; the shapes are incompatible.

**Fix (2026-09-04):** `MN4D_OK = !(A_IS_MN && NEEDS_A_MULTICAST) && !(B_IS_MN && NEEDS_B_MULTICAST)`,
used in BOTH the host descriptor gate and the kernel pick, so `k_gen4` IS `k_gen` for such a
config. Dead branch keeps a `__trap()`. Re-validated 0 bad, deterministic.

**Why:** §H82 claimed "bit-identical 5/5, +1.46%". That was checked on NT, where MN4D is a
no-op — see [[feedback-vacuous-checks]]. **The +1.46% is void.** For our production pins
(cluster 2x1, NN) B is MN-major and multicast, so NN now takes rank-3.

**How to apply:** rank-4 only helps a non-multicast MN-major operand. Before reviving it,
re-measure — the original speed claim no longer exists.
