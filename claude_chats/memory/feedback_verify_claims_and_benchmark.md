---
name: feedback_verify_claims_and_benchmark
description: User wants claims verified against sources and optimizations benchmarked before/after — no tidy-but-wrong stories
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 56fc6f68-72b2-429c-a154-b32dea36db1c
---

When writing technical docs or justifying an optimization, the user demands the
"absolute truth," not a neat-sounding story. Concretely: verify hardware/algorithm
claims against real sources (papers, the actual PyTorch/library code) before
documenting them, and flag anything unverified as unverified.

Specific corrections he pushed on: the "FADD = 4 cycles" claim (it's
architecture-specific: ~24 on Fermi … ~4 on Volta/Turing/Ampere/Ada; confirmed for
Ampere sm_86 by Abdelkhalik et al. 2022 arXiv:2208.11174) and the Volkov [2016]
attribution (he established the ILP-vs-occupancy principle via Little's Law, did NOT
prescribe "4 accumulators"; PyTorch does not cite Volkov at all).

Before committing to a kernel optimization, capture a BEFORE benchmark (timings +
bandwidth), implement, then re-measure AFTER and report the delta honestly — even if
the win is small. Example: 16-bit input-vectorization on the reduction kernel was
already ~92% bandwidth-bound on the RTX 3060, so it only gained +2–6% (but +25% on
awkward sizes). See [[project_reduction_bugs]].

**Why:** he's documenting for real understanding and won't accept hand-wavy or
flattering explanations. **How to apply:** WebSearch/read-the-source to confirm
before asserting; build a micro-benchmark before/after; state measured numbers and
scope limits plainly.
