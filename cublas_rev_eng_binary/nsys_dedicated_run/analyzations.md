
### 31. 2×2f DECODED from nsys launch records — "f" is NOT fused epilogue (2026-07-04)

**Correction of §3.3/§4:** "f = fused epilogue" was an unverified naming guess. The nsys sqlite
launch records (CUPTI_ACTIVITY_KIND_KERNEL, nsys_dedicated_run, consistent TN+NN) prove otherwise:

| kernel (128x256_64x6, same family) | clusterX | maxActiveClusters | CTAs resident | SM fill (148 SMs) |
|---|---|---|---|---|
| `..._2x2_2cta_h_bz`  (no f)        | **4**    | 33                | 132           | 89%  |
| `..._2x2f_2cta_h_bz` (f)           | **2**    | 74                | 148           | **100%** |

Same block (256 thr), same regs (255), ~same smem (213296) — the ONLY difference is the hardware
cluster size at launch. So:
- non-f "2x2" = rigid 4-CTA cluster (like our validated 2×2) — loses 16 SMs to 4-CTA granularity.
- "2x2f" = same tile family launched with 2-CTA clusters → 100% SM fill. The trailing f most
  plausibly = "fallback" (CUDA 12.8 cudaLaunchAttributePreferredClusterDimension: prefer (4,1,1),
  fall back to (2,1,1); CUPTI records the fallback/regular dim). VERIFIED: f→cluster 2, non-f→
  cluster 4, and the 89%→100% occupancy delta. HYPOTHESIS (flagged): the preferred-dim attribute
  itself isn't visible in the CUPTI export.
- Explains cuBLAS's heuristic: f on the LARGEST grids (llm-style, square-*, 7 Qwen shapes) where
  wave-tail/fragmentation cost of rigid 4-CTA dominates; non-f on medium grids.
- Explains the old "5× slower per call" anomaly (§6): different shapes routed, not a slower kernel.

**Implementation plan for our 2×2f variant (next):**
1. Launch via cudaLaunchKernelEx: cluster (2,1,1) + cudaLaunchAttributePreferredClusterDimension
   (4,1,1) (+ fallback enable). Remove compile-time __cluster_dims__ for this variant.
2. Kernel becomes CLUSTER-SIZE-ADAPTIVE at runtime: read %cluster_nctarank (cluster CTA count).
   If 4 → exact validated 2×2 path (A-multicast, CLUSTER_N=2 decode); if 2 → exact validated 2×1
   path. All barrier counts (empty_mbar=CLUSTER_N, CLC_EMPTY_ARRIVE, masks) become runtime values —
   mbarrier_init takes runtime counts, masks already computed from rank.
3. OPEN QUESTION to resolve first (research): tile-index assignment under mixed cluster sizes —
   blockIdx.x/CLUSTER_CTAS decode assumes uniform cluster size; with fallback, clusters in the SAME
   grid have different sizes. Need PTX/driver semantics of %clusterid / ctaid mapping under
   preferred-cluster launch + how CLC try_cancel responses map. Likely answer: decode from
   %clusterid or ctaid/%cluster_nctarank per-cluster, but MUST be verified before coding.
4. Validation: same suites; plus verify occupancy gain on B300 (maxActiveClusters 74-equivalent).
