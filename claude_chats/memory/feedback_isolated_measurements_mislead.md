---
name: feedback-isolated-measurements-mislead
description: "A null result only refutes a hypothesis if nothing bigger is masking it, and a microbenchmark's Nx must be scaled by that component's share of the real kernel — both directions cost a day on 2026-08-14"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-08-14T09:45:57.380Z
---

Two failure modes, same root cause: a number measured in isolation does not mean what it looks
like. Both bit on 2026-08-14 and each cost real time.

## 1. A NULL RESULT IS ONLY A REFUTATION IF NOTHING BIGGER IS MASKING IT

`128x256x64_2x1_h_swz64` (epilogue XOR swizzle, one template arg) measured:

    -0.2 pts at 2026-08-13 09:27Z   -> recorded as "hypothesis 1, bank conflicts: DEAD"
    +4.8 pts at 2026-08-14          -> the largest single win of the campaign

Same config, same shape, opposite verdict. Between them, §H27za4 removed a per-tile cluster barrier
that held **33.9% of all PC-sampling stall samples**. While it was there, removing bank conflicts
underneath it changed nothing measurable. The hypothesis was never wrong -- it was BURIED, and
"duration unchanged" got written down as a refutation.

**How to apply:** when a dominant cost is removed, RE-TEST every null result measured under it.
§H27z.94/.97/.98/.99 were all measured in that window and are all suspect. Same reasoning voided the
dispatcher's 437 table entries -- see [[project-sm90-dispatcher-done]].

## 2. A PROBE'S "Nx" MUST BE MULTIPLIED BY THAT COMPONENT'S SHARE OF THE KERNEL

`probe_stsm_banks.cu` measured stmatrix at **1.93x** over plain STS on the swizzled buffer. Real:

    store path as a share of the kernel      1.40%   (STS = 872 of 62,242 PC samples at K=512)
    so the maximum possible win              0.68%
    stmatrix's own added overhead            ~4.5%   -> it LOST by 3 points

The probe was correct. It optimised 1.4% of the workload. Before spending implementation time on
"the probe says Nx", multiply by the measured share -- and if that share has never been measured,
measure it first.

## 3. A CENSUS MUST PROVE IT ISOLATED ITS ARMS -- this failed THREE times in one campaign

    §H27zb0 phase 2  keyed on tile dims -> all FOUR configs returned byte-identical rows
    §H27zb1 phase 3  matched 6 kernels for one arm against 3 for the other -> every delta an artefact
    §H27zb6 phase 4  matched 36 kernels (every 192x192_2x1 variant x 3 layouts, 113 MB) as "one" census

Each time the table LOOKED like evidence -- the first read as "no change", the second as a big
negative delta. A census that cannot distinguish its own arms is worse than no census, because it
produces a number instead of an error.

**How to apply:** any census MUST print how many kernels it matched. Two arms with different counts
are NOT comparable. If the filter cannot isolate exactly ONE kernel, do not ship it. Discriminating
on tile dims is never enough -- our configs share tiles and differ in the TAIL template args
(EPI_TILE_N / EPI_STAGES / EPI_SWIZZLE_BYTES / EPI_ROWS / EPI_USE_STSM / RASTER).

Related trap: STATIC opcode counts are code SIZE, not execution. ncu already gives the dynamic
ratio; "which instruction costs TIME" is a dynamic question that only PC sampling answers. Do not
spend time disassembling libcublasLt (thousands of kernels, 12 min at 2.8% CPU, empty output) for a
static number that could not have answered it.

**Why both matter here:** this campaign is a long chain of hypotheses eliminated by measurement, so
a wrongly-recorded elimination is not one lost experiment, it removes a whole branch from the
search. State the masking risk WITH any null result, in the doc, at the time.
Related: [[feedback-verify-claims-and-benchmark]], [[project-step3-plan]].
