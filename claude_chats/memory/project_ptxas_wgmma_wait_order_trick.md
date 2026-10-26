---
name: project_ptxas_wgmma_wait_order_trick
description: ptxas hoists wgmma.wait_group 0 to ~32 instr after wait 1 whatever the source; an mbarrier arrive that DEPENDS on the softmax result pins the wait after it (FA3 gets this by accident from its K release)
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-17T06:55:13.369Z
---

Found 2026-10-17 on the attention forward (GQA_fwd_sm90.cu, candidate K1, session_2026_10_17/candidates/fwd_K1.cu):
- ptxas (12.8 and 13.0 alike) put WARPGROUP.DEPBAR.LE gsb0,0x0 (wait for P@V(n-1)) right after the FIRST row's max,
  ~32-48 instructions after the QK wait, and ran the other row's max + all 64 exps + packs after it, although the PTX
  had them all before. Tried and NOT effective: other softmax forms (1/4/8 chains, row by row), peeling the mask, fewer
  registers (loop used 184 of 232), two P register sets, moving the empty arrive, an independent dummy arrive,
  a never-taken branch fence (that one SPILLED 400 B and serialised every HGMMA: never again).
- What works: an `mbarrier.arrive` placed before the wait whose ADDRESS depends on a value computed at the end of the
  softmax (two count-1 barriers nobody waits on, index = (bits(l0+l1)^pa[0]^pa[31]) == ~0). ptxas keeps the wait after
  the arrive -> whole softmax (68 FMNMX, 66 MUFU, 66 FFMA, 65 FADD, 32 F2FP) before the wait, only 64 FMUL rescale after.
- FA3's SASS shows the same mechanism by accident: its K release arrive (mainloop :1185) is scheduled late and its
  P@V wait sits right after it.
- Tool: profiling_reports/session_2026_10_17/deep/sass_shape.py (counts per opcode before / after the P@V wait).

**Why:** the forward's dense gap vs FA3 (+20% cycles at equal clock) was mostly this placement (+0.198M of +0.288M).
**How to apply:** check sass_shape.py after ANY change to the forward loop; the same trick may apply to the backward.
Related: [[project_state_2026_10_17]].
