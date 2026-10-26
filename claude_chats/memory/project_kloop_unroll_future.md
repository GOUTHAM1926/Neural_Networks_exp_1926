---
name: project-kloop-unroll-future
description: CORRECTED 2026-08-28 - the 'descriptor rebuild is uniform-datapath' premise is FALSE, it is largely VECTOR; DESC_ACCUM is REOPENED. KLOOP_UNROLL is REFUTED (2026-08-27) - it cuts total instructions 6.8% but raises VECTOR 36% and performance is flat; do not extend it, and do not write DESC_ACCUM either
metadata:
  type: project
---

**REFUTED AND CLOSED 2026-08-27 (§H27ay).** Measured, not argued.

`192x192x64_1x2_v_rs8`, `2048x11264x16384 tn`, one config per process:

    config                  ourTF    %cuB      total       uniform       VECTOR
    dh_ubd_dw_skew_lsw     654.14   95.0%   161,497,515   78,653,388   82,844,127
    dh_ubd_dwunr_skew_lsw  654.21   94.6%   150,481,284   37,690,092  112,791,192
    dh_ubd_unr_skew_lsw    562.85   81.6%   166,941,675   53,657,868  113,283,807

    +unroll on dw:  total -6.8%, uniform -52%, **VECTOR +36%**, perf FLAT
    +unroll no dw:  **-13.4 points**

Making `stage` a compile-time literal does remove the multiply, modulo and barrier-offset
arithmetic — but those are **UNIFORM** ops, the cheap kind — while the D literal bodies
inflate VECTOR work by a third. See [[feedback-vector-not-total-instructions]].

**This was the out-of-sample validation of the vector metric:** total said "good" (−6.8%),
vector said "bad" (+36%), the hardware agreed with vector. The template comment beside these
configs says *"judge these by smsp__inst_executed.sum first"* — following that would have
adopted a worthless change and extended it to ten more pins.

**CONSEQUENCES:**
- Do NOT extend KLOOP_UNROLL to the non-SKEW mainloop (it is SKEW-only today; 256x128 and
  128x256 = ten of fifteen pins would have needed new implementation). Cancelled.
- Do NOT write `DESC_ACCUM`. Both forms of "make the stage term cheap" are now measured and
  both fail: carried state (STAGE_SPECIALISE, +30.7% instructions) and literal bodies
  (KLOOP_UNROLL, +36% vector).
- The per-iteration descriptor rebuild (~18 instr) STAYS. Those are UNIFORM ops and the
  correlation says uniform carries no performance signal. §H27au's remaining-dead-work list
  was ranked by instruction count; re-ranked by datapath, items A/B/C are all uniform and
  not worth chasing. Only item D (phase flip on the VECTOR path, 3 instr) targets the right
  datapath.

**Where the vector excess actually is:** slot 34 (§H27zm) already records it — on SPLIT_M
tiles `row0 = g*WG_MATOMS*64` with runtime `g` lands `LEA/LOP3/SHF/VIADD` on the vector path
then `R2UR` back. That is what `UNIFORM_A_DESC` fixes, it affects **ten of fifteen pins**,
and it is why §H27au measured +1.95 on V2c (R2UR 16 → 1).

**CORRECTED 2026-08-28 (§H36). THE PREMISE FOR CLOSING DESC_ACCUM WAS WRONG.**
This file said *"the descriptor rebuild is uniform-datapath so leave it."* The full SASS census
shows it is largely **VECTOR**: per wgmma we run `LEA R11,R3,UR16,0xe` (base from a PER-THREAD
register) then `LOP3.LUT` mask, `SHF.R.U32.HI`, `LOP3.LUT` flag, then `R2UR` to cross to uniform
— and we execute **4,189,184 R2UR against cuBLAS's 5,120**. cuBLAS never re-packs: the descriptor
sits in a uniform register all kernel and advances by `UIADD3 UR16,UR16,0x80,URZ`.

**DESC_ACCUM is REOPENED and is item A of [[project-instruction-excess-worklist]] — 44.6% of our
entire instruction excess.** KLOOP_UNROLL itself stays refuted; only the descriptor conclusion
is overturned.
