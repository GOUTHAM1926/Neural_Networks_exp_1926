---
name: project-addmm-bias-done
description: Bias/addmm built in our style — we now beat his addmm 103-109%; the per-element bounds guard cost 148k cycles and was the whole deficit.
metadata:
  type: project
---

We had ZERO bias support. Now: `Sm90BiasMode {Runtime,Scalar,PerCol,PerRow,Full}` compile-time
specialised, one `apply_bias_pair<MODE,N_ALIGNED>` called from all 3 value-forming epilogue
paths, split-K excluded (matches him). Existing kernels **byte-identical 16/16**.

Semantics: `C = alpha*(AxB) + beta*bias`. **Bias is scaled by beta** (PyTorch addmm). M==N ties
break to PerCol, matching his test order.

**Final (2026-09-04): we beat his addmm on all 4 shapes — 103.2 / 104.3 / 103.6 / 108.8%**,
81.6% tensor pipe vs his 76.4%. Fused vs separate-bias-kernel: −21% to −34%.

**Three of my own bugs, worth remembering:**
1. Reused `beta` for both "scale the bias" and "read C back" → fused path re-read all of C,
   +47% SLOWER. His tree separates HasBeta/HasBias. Fixed with `READ_C_IN`. The first
   correctness test passed only because it memset C to 0.
2. **A PER-ELEMENT bounds guard cost 148,382 cycles** (475k vs 327k) and 24 points of tensor
   pipe. In a fully-unrolled epilogue a per-element branch kills straight-line codegen —
   ~400 integer address ops for 128 FFMAs. Fix: whether any tile is partial is a LAUNCH fact
   (`N % BN == 0`), so it became a template parameter; guarded arm then isn't compiled at all.
3. A stale `.o` faked a byte-identity pass. See [[feedback-vacuous-checks]].

**How to apply:** never put a bounds test per-element inside an unrolled epilogue — hoist it to
tile or launch level and specialise. And our base GEMM was already AHEAD of his here (314k vs
349k cycles) — a deficit in a fused kernel is not evidence the base kernel is slow.

Open: register addmm in the dispatcher + library build; his `BIAS_SMEM_PREFETCH` (Full-mode
bulk-TMA bias, his +23.5pp) not ported; addmm can't use A1B0 because beta!=0 — now unnecessary
with READ_C_IN false. Related: [[project-tile-order-verdicts]], [[feedback-samples-not-static-counts]].
