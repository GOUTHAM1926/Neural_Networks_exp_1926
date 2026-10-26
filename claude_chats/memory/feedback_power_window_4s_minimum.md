---
name: feedback-power-window-4s-minimum
description: H200 power takes ~4s to PLATEAU - never quote an absolute watt figure from a shorter window; 8s+ adds asymmetric thermal drift. Measured, not assumed.
metadata:
  type: feedback
---

**Measured on H200 (§H227, `tools/h227_rampcal.sh`), `tn 3072x16384x8192`, locked 1200 MHz.**
The tell is full-window median vs **last-half** median: if they disagree, power is still climbing.

| window | ours / last-half | cuBLAS / last-half |
|---|---|---|
| 1 s | 370.3 / **486.6** W | 351.8 / **433.8** W |
| 2 s | 536.2 / 549.6 W | 522.5 / 532.9 W |
| **4 s** | 544.6 / 545.5 W | 526.5 / 526.5 W |
| 8 s | 546.4 / 547.1 W | 528.8 / 529.2 W |
| 24 s | 549.7 / 550.3 W | 534.4 / 534.7 W |

**It is a HARDWARE ramp** (sensor averaging + silicon reaching steady draw), so a software warm-up
inside a shorter window does not fix it - the window itself must be >= 4 s.

**Long is not free either:** 8 s -> 24 s drifts +3.3 W ours but +5.6 W cuBLAS. That drift is
**asymmetric between arms** and would fake a ~0.4% power gap. Honest window = **4-8 s**.

**What it cost:** a 1 s window read cuBLAS at 324.9 W against our 406-422 W - a 1.28x ratio, when
the truth is +2.73%. §H225's whole watt column was taken at 2.01 s and reads ~4% low.

**How to apply:**
- Absolute watts: window must be 4-8 s. State the window.
- A **paired** ABBA design tolerates a short window (the ramp is common-mode and cancels - that is
  why §H226's null control still came out null), but its *magnitude* is not established until
  re-run at >= 4 s. Throughput is harness-timed and never affected by this.
- Derive iterations from `2*M*N*K` so every shape gets the same seconds, never a fixed count.

See [[feedback-verify-claims-and-benchmark]] and [[project-test2-measurement-traps]]. This is the
4th time the INSTRUMENT produced the surprising number, not the kernel - periodic operands, the
`-lgc` that silently did not take, the leaked sampler, and now the ramp. **When a result is
physically impossible, calibrate the instrument before explaining the result.**
