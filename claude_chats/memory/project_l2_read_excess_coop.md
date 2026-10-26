---
name: project-l2-read-excess-coop
description: The 1.64x L2 read excess was V1-specific - on the 192x192 pins it is 1.20x/1.45x with misses at parity and DRAM traffic FAVOURING us. The descriptor/TMA lead is weak; do not spend a session on it.
metadata:
  type: project
---

**SUPERSEDED by §H50 (2026-08-29).** The 1.64x L2 read-request excess and the 41.3-vs-24.1
requests-per-TMA figure (§H27bi) were measured on **V1** and **do not transfer** to the 192x192
pins that carry 73% of the trainer deficit. Measured there, by name, ncu clock-locked:

    gate_up-fwd    L2 tex requests 1.197x   L2 read MISSES 1.033x   DRAM read 0.845x
    gate_up-wgrad  L2 tex requests 1.452x   L2 read MISSES 0.859x   DRAM read 0.746x

**We move 15-25% LESS DRAM traffic than cuBLAS and miss L2 at or below parity.** The memory-side
story is not where the gap is. Two earlier explanations (coopA-from-a-kernel-name, B-retention)
were already retracted; the tensor-map box-geometry lead is now also weak for these kernels.

**How to apply:** do not open the CUtensorMap descriptor investigation on the strength of the V1
number. If it is ever reopened, re-measure requests-per-TMA on the actual dispatched kernel
first. See [[project-instruction-excess-worklist]] for where the gap actually is.
