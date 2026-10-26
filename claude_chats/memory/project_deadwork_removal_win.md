---
name: project-deadwork-removal-win
description: Dead-work sweep across all 15 pins (2026-08-27) - all bit-exact; the right knob set is DIFFERENT for every tile; 256x128 +1.95 paired, 128x256 dh-only +0.37% 7/7, 192x192 nothing
metadata:
  type: project
---

**SWEEP COMPLETE across all 15 pins, 50+ runs, every rung maxRelErr 0.00000 (bit-exact).**
The headline: **no knob generalises — each tile needs its own measurement.**

| family | pins | verdict | evidence |
|---|---|---|---|
| `256x128` | V2, V2b, V2c | **adopt `dh_dw_uad_ubd_cs`** | **+1.95 paired, believable** (floor 1.37); 3/3 single-shot |
| `128x256` | V7,V7b,V7c,V8,V8b,V8c,V8d | **adopt `dh` ONLY** | **+0.37% mean, 7/7, sign-test p=0.016** |
| `192x192` | V3, V4, V5, V6 | **no change** | `uad` twin is a BYTE-IDENTICAL k-loop |
| `128x128` | V1 | **no change** | `dw_uad` = −7.33 (§H27aj) |

**Sub-floor effects need the sign test.** 128x256's +0.37% is far below the 1.37-point
paired floor, so no single-shape paired run could resolve it; 7/7 same-sign across
INDEPENDENT shapes gives p = 0.016. Use this whenever the per-shape effect is small.

**Instruction count is NOT monotone with performance.** On 128x256 the deeper rungs remove
MORE instructions (−23.5% vs −17.3% static) and go BACKWARDS (2/7 positive, mean −0.32%).

**Per-knob behaviour, all measured:**
- `dh` — the only knob that helps on 128x256 (7/7); part of the winning stack on 256x128
- `dw` — needs `uad` on 128x128 and 256x128; a REGRESSION even with it on 128x256
- `uad` — essential with dw; a byte-identical NO-OP on 192x192 (ubd already covers it)
- `ubd` — essential on 192x192, no-op on 256x128
- `cs` — worth NOTHING anywhere (−0.07 V2c, −0.10 V1); kept only as bit-exact dead-work removal

**ADOPTION POLICY (user directive 2026-08-27):** keep a removal if **bit-exact AND not a
measured regression**, even at zero gain — "you removed something unwanted anyway". `cs`
stays at zero gain; `dw_uad` on 128x128 and 128x256 stays out as a regression.

**NOT YET APPLIED to `Bgemm_sm90_dispatch_table.h`** — the pin change is a separate gated
step. Projected set mean 96.1% -> ~96.7%.

**REMAINING DEAD WORK:** the descriptor base is still REBUILT every k-iteration rather than
carried — `S2UR SR_CgaCtaId` (loop-invariant), `UIMAD stage*copy_size` (a multiply), ~18
instructions of mask/shift/OR/sign-extend. cuBLAS keeps the base in a register and rewinds
at the bottom of the group. Next knob: `DESC_ACCUM`. Also the phase flip runs on the vector
path (3 instr). See [[project-instruction-track-refuted]], [[project-kloop-unroll-future]].
