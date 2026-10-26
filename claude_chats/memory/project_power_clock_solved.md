---
name: project-power-clock-solved
description: Clock deficit SOLVED 2026-08-27 and CONFIRMED END-TO-END ON THE TRAINER 2026-08-28 - higher power at equal clock saturates the cap and costs clock; it is NOT instructions; L2 requests is the suspect and the L2->power arrow is still untested
metadata:
  type: project
---

**SOLVED (§H27b4).** The whole chain, measured end to end:

    +12% power at equal clock  ->  100% cap saturation (cuBLAS 47.7%)  ->  -108 MHz  ->  free-clock deficit

**The measurement (V1, kind-data-thinks-fin-03, settled, unclamped):**

    lock   cuBLAS W   ours W   ratio    cap% ours/cuB   achieved clock
    1000       427      472    1.11x      0% /  0%      1005 / 1005
    1200       540      612    1.13x      0% /  0%      1200 / 1200
    1400       696      698    1.00x    100% / 47.7%    1301 / 1409   <- 1.00x is the CLAMP

**PROTOCOL THAT FINALLY WORKED** (three earlier attempts failed):
- `power.draw` is a **2.4 s trailing average** — useless at short windows. Use
  `power.draw.instant`.
- 100 Hz sampling **aliases** the instant field. Use **2 Hz**.
- Persistence mode MUST be on or the sensor sticks (a 511-TFLOPS GEMM read a flat 90.00 W).
- **30 s windows, 8 s discarded**, so the lagging sensor settles INSIDE each arm's window.
- Lock LOW enough that BOTH arms stay under the cap. **At free clocks both are clamped at the
  cap so power CANNOT differ — that is why every free-clock power reading was 1.00x.**

**IT IS NOT THE INSTRUCTIONS — decisive control.** `_shfl` cuts V1's k-loop vector ops
8.25 -> 2.00 (76%). At lock 1200: ours 621 W -> 608 W, ratio 1.15x -> 1.13x. **13 W, 2.1% of
power; 87% of the excess survives.** Consistent with the other two measurements of the same
track: +0.58 clock points (§H27am) and +0.5-0.9 total points (§H27b1).

**LEADING SUSPECT — L2 request volume.** On V1 with DRAM bytes at parity (1.00x):

    lts__t_requests   ours 4,405,890   cuBLAS 2,384,607   **1.85x**
    lts__t_sectors    ours 14,862,825  cuBLAS 8,345,518   1.78x
    shared wavefronts                                     0.98x
    tensor cycles                                         1.00x

Sectors-per-request are comparable (3.37 vs 3.50), so it is not granularity — we simply go to
L2 nearly twice as often for the same bytes (hit rate 69.8% vs 49.5%). L2 traffic is
energy-expensive and is the only major counter left near 2x.

**NEXT: why 1.85x the L2 requests for identical DRAM bytes?** Best-defined open question in
the project, with ~12% measured power and ~8 points of clock behind it.
See [[feedback-vector-not-total-instructions]], [[project-deadwork-removal-win]].

**CONFIRMED END TO END ON THE 1.7B TRAINER, 2026-08-28 (§H28c).** The earlier §H27bj claim that
this mechanism "does not reach the application" was based on a corrupted A/B (both arms cuBLAS)
and is retracted. Measured correctly:

    LOCKED 1410  same clock both arms, we draw +14.1 W median (+2.69%), p95 +15.7 W
    FREE         both pin the 700 W cap, sw_power_cap Active on 100% of samples,
                 our clock median 1785 vs 1815 MHz, time above 1830 MHz 23.7% vs 47.5%
    trainer gap free 1.64%  |  locked 0.18%  |  clock deficit 1.93%   <- arithmetic closes

Independent read of the same effect: the **non-GEMM** half of the trainer, byte-identical code in
both arms, runs **+0.82%** slower purely from the lower clock — memory-bound kernels untouched
(+0.01%), compute-bound ones paying it (`gqa62` +3.11%).

**The L2 -> power arrow remains the one untested link.** See
[[project-trainer-parity-1p7b]], [[project-open-tasks-2026-08-28]].

**2026-08-28: PER-PIN ATTRIBUTION FAILED, THE TRAINER NUMBER IS THE ONLY VALID ONE.** An attempt
to measure power per pin standalone (§H33/H34) was void — `-lgc` is not enforced on this box, the
GEMM loop clamps at the 700 W cap, and 12 of 15 pins had power clamped. **The trainer's +14.1 W
(+2.69%) at 1410 MHz with cap 0% active is the project's only valid power measurement.** Do not
quote the 3 surviving pins' 1.006 ratio — they are the lowest-power pins by construction.
The cause of the +2.69% is still UNKNOWN; instructions (1.811x) and L2 (1.312x) are both
untested, not refuted. See [[project-open-tasks-2026-08-28]] item 0 for the replacement test.

History (superseded 08-27, kept for its LEADS and tables): [[project_power_clock_track]].
