---
name: diff-against-proven-code-first
description: "Diff against the proven reference FIRST: a failing mechanism vs our proven code/CUTLASS/cuBLAS SASS (07-23); any gap vs a rival whose source is local (FA3) via source + SASS + PC sampling before any candidate (user angry 10-19)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-23T13:22:47.855Z
---

When a hardware/kernel mechanism we ALREADY have working elsewhere (another arch
module, CUTLASS, captured cuBLAS SASS) fails in fresh code, diff the new code
against the proven implementation top-to-bottom FIRST — before inventing new
hypotheses, building isolation repros, or patching one difference at a time.

**Why:** During SM90 cluster bring-up the user repeatedly (and increasingly
forcefully) flagged that I was patching single guessed differences (1D→2D TMA,
mbarrier `.release`→plain, cluster-sync `.relaxed`→plain) and re-running on H200
each time, instead of comparing against our own proven code. The actual crash
(illegal instruction/illegal memory access at the first TMA) was a missing
`__grid_constant__` on the by-value `CUtensorMap` kernel param — instantly
visible by diffing the repro against our real SM90 kernel, our proven-336/336
SM103 kernel, and CUTLASS's `device_kernel.h:118`, all of which declare it
`const __grid_constant__ CUtensorMap`. Every earlier guess was the wrong layer.

**How to apply:** Before any "novel cause" hypothesis for a failing kernel, run
a direct grep/diff of the failing code's primitives + kernel signature + param
passing against (1) our other arch modules [[project_sm103_cuh_locked]], (2)
CUTLASS source, (3) captured cuBLAS SASS. Only after the diff shows nothing
should novel hypotheses begin. Cite the primary source (file:line) for each
confirmed difference — don't rely on paraphrase or an agent's summary. Ties to
[[feedback_verify_claims_and_benchmark]] (absolute proof before implementing).

## 2026-10-19: the same rule for performance gaps (was feedback_proof_first_compare_fa3_source.md)

2026-10-19 ~16:00 the user (angry): "why are you testing assumptions when we have the FA3 source ourselves... from morning you
are wasting my time... with solid proofs only we will proceed". A day of candidates (L1b, R8-R32, SKV, LH) was built on
INFERRED differences; most failed.

**The method that worked the same afternoon (45.52):** (1) PC sampling (ncu source page, warp-stall samples per instruction)
of FA3's real kernel and ours on the same box; (2) group every instruction of the steady loop by job, compare counts and
stall time; (3) trace each costed difference to FA3's source line (hopper/softmax.h etc.) and ours; (4) make ours do what
FA3's code does; (5) prove on the LAPTOP that the compiled SASS now has FA3's shape (local SASS was verified to equal the
box's for the measure) BEFORE asking for GPU time; (6) only then build/race.

**Why:** the user's time is the scarce resource; GPU runs of unproven ideas waste it.
**How to apply:** for any "why are we slower than X" question where X's source is local, do the source + SASS + PC-sampling
diff first and present the proven cause with numbers; no exploratory candidates. Related: [[feedback_verify_claims_and_benchmark]],
[[feedback_simple_english_no_blabber]].

**Added 10-19 17:40 (PFT, work doc 45.56):** a candidate that does not target the steady loop must leave the loop's SASS IDENTICAL to the base's (0 differing lines), checked on the laptop and again on the box build before any GPU time. PFT's loop was re-ordered by ptxas (same 481 instructions), I called it 'neutral' from the exp spacing, and it cost +40 cycles per step (+1.6% with the other half). An estimate is not a proof.
