---
name: project-c7517-skew-waits
description: ptxas injects wgmma waits on every 192x192 pin; WGMMA_SKEW is the sole cause and it still pays for itself. Two future tasks parked.
metadata:
  type: project
---

**WGMMA_SKEW is the SOLE trigger** for the `C7517 wgmma.wait_group is injected` messages -- proven
by knob bisect on a local nvcc 13.0 sm_90a build, no GPU needed. Every other knob flipped off still
warns; skew off is clean. Costs DEPBAR **7->3** (V3/V4 pins) and **15->11** (V5, V6).
128x128 and 128x256 show ZERO warnings only because they **cannot** use skew
(`static_assert`: needs >=2 m64 atoms per warpgroup; a SPLIT_M 128x256 has 1) -- the
"192x192 is the warned family" correlation carries no information beyond that.

**PRECISION IS NOT AT RISK, EVER.** The injected wait is what makes the code CORRECT -- without it
accumulators would be read before the wgmma retired. `maxRelErr 0.00000` on all 555 measurements
of 2026-08-24. Do not re-litigate this.

**PERFORMANCE: skew PAYS FOR ITSELF, measured 2026-08-24 (§H27zL).** Removing skew:
V3 -0.55%, V4 -0.24%, V5 -0.16%, V6 -1.16%, mean **-0.53%**. KEEP SKEW.
This CORRECTS §H27zbC-RESULT ("the implementation does not pay yet", V3 -0.78% skew WORSE) --
that predates the 400 ms window and one-config-per-process, and the sign flipped.
**NOT established: the cost of the injected waits ALONE.** A skew-without-injected-waits build does
not exist, so "C7517 costs us nothing" is NOT proven -- it costs something small and unquantified
that skew's gain outweighs. Never claim it is free.

**§H27zt's fix does NOT apply here.** There a wait sat inside a runtime `if (prev_stage >= 0)`;
hoisting it gave V1 +7.85%. Here `drain_prev`, `skew_body`, the `WROT==0` branch and the tail are
ALL already unconditional. Injection sites located exactly (PTX line -> `.loc`): template **:3900**
(the wgmma asm, accumulators are `"+f"` read-modify-write) and the **k-loop header**, where the
loop-not-taken path cannot be cleared. Persistent kernel, so accumulators also carry across tiles.

## FUTURE TASKS (parked with evidence, not forgotten)

1. **C7515 (6) and C7519 (3) -- "Potential Performance Loss ... wgmma serialized".** All nine are
   `192x192 cluster 1x2` and **all have PDL=false, so NONE is a shipping pin** -- dead registry
   arms. Harmless today. Worth cleaning up or fixing if those arms are ever revived. The three
   C7519 all carry `KLOOP_UNROLL=true`, which is a candidate explanation for §H27zdy's
   "SASS target met, wall clock null".
2. **Get skew WITHOUT the injected waits.** Structural change to the peel/loop/tail so ptxas can
   prove a wait dominates every path. Would land ON TOP of skew's +0.53%. Risky: restructuring
   that code produced `maxRelErr 1.00000` on ten pins in §H27zzt. Do the PC-sample census first --
   if the DEPBARs are not on the hot path there is nothing to win.

Toolchain note: local CUDA 13.0 and the box's older ptxas emit DIFFERENT C7519 text and fire it on
different configs. Judge what we ship by the BOX build log. See [[project-test2-measurement-traps]],
[[feedback-samples-not-static-counts]].
