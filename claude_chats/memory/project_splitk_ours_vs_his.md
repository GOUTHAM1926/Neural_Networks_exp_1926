---
name: project_splitk_ours_vs_his
description: Split-K decision — OUR back-end + HIS front-end; our split-K is STRANDED by the B=2 config change (K 16384->8192) and falls to our worst site; 3 corrections to the first pass
metadata:
  type: project
---

**Decision: OUR back-end + HIS front-end. His atomicAdd path is retired.**

**BIGGEST FINDING — our split-K does NOT fire on production shapes since commit `7ebfc65` (today) dropped B*T 16384->8192.** Our only pinned split-K row is keyed to K=16384, there is no 1536x512 row, so tier 2 sends both 1.7B and 600M to `TN_64x128x64` = **our WORST site, 455.5 TF = 70.3% of cuBLAS**. His `sm90_tn_splitk_plan` fires. **The reduce-algorithm choice is worth ~1.0-1.4% of GPU time; the DISPATCH-RULE choice is worth the 70.3% -> 93-98% swing on two GEMMs per layer.** Take his selection front-end.

**Three corrections to the first pass:** (1) `ldc` is NOT a discriminator — ours makes the same flat `C[idx]` assumption in `Splitk_reduce.cuh:135`; (2) our comment claiming alpha-per-slice "raises it to the S-th power" is mathematically WRONG (`sum a*p = a*sum p`) — never cite it; what we actually gain is **beta**, which his split-K structurally cannot do; (3) the one same-shape data point (2048x512x16384, 128x128, sk2) reads **his 98.5% vs our 93.6%** — not apples-to-apples, must not be quoted as a 5-pt gap, but it does not favour us and the paired ABBA has never been run.

His template has ONE kernel body with an `if constexpr (SPLITK_WORKSPACE)` fork: Path A = fp32 `atomicAdd` into one slice + a pure cast finisher (**non-deterministic**); Path B = private fp32 slice per split + a real reduce kernel (deterministic). **Only Path B fires** on real 600M/1.7B (`{128,2}` -> `Sm90Config_128x128x64_regsplit_ws`); Path A needs `K>=16384` and production K is 8192. So on the live path **precision is a tie** — both are fp32 partials, fp32 ascending sum, one bf16 rounding. Don't claim a precision win there.

Our BACK-END wins on: (1) exact ceil partition vs his floor-twice + silent `while (sk>1 && k_units%sk) --sk` downgrade; (2) beta supported vs structurally impossible; (3) one budgeted process-wide arena vs his per-instantiation unbudgeted `static` (not `thread_local`, no mutex) = the exact §H27z.27 bug; (4) dtype-generic, reusable by SM103. His atomicAdd path is reachable at bn64/K>=16384 and is the only non-deterministic accumulation in either tree. Our determinism IS measured: `dispatch_table.h:634-636` 60/60 bit-exact, 5/5 runs identical.

Cost in production: 1.449% (600M) / 1.081% (1.7B) of GPU kernel time; the reduce adds 9.1-9.4% on top of the split GEMM.

**Three latent bugs in OURS — fix before the port ships:**
- **(a) REAL, live config:** PDL `launch_dependents` at :6091 fires INSIDE the tile loop, BEFORE the epilogue's workspace STG at :6457, so the reduce's `griddepcontrol.wait` does not order those writes. The :6086 comment's reasoning holds for a dependent reading inputs, not one reading this kernel's own output. `..._sk2_dh_rot2_pdl_d6_shfl` sets both SPLIT_K=2 and PDL=true.
- (b) odd-N: `ws_off` row-start parity can misalign the `float2` store (last column handled, row start not). Unreachable today — all split-K dispatch entries have even N.
- (c) `WGMMA_SKEW` + empty slice (S > k_iters_total) would hang on the unconditional peel at :5885. Unreachable today — all `sk*` configs set SKEW=false.
- Housekeeping: `Splitk_gating.cuh` is **dead code**, zero callers, included by nothing.

See [[project_hybrid_knob_port_list]] and §H69.
