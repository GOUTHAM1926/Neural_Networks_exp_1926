---
name: project-mio-throttle-deferred
description: mio_throttle gate is now OPEN - the 192x192 mainloop work it was parked behind closed on 2026-08-18
metadata:
  type: project
---

`mio_throttle` (MIO = Memory Input/Output) is a **saturation** signal, not a stall to eliminate:
cuBLAS runs 10-16%, we run far lower. It was DEFERRED behind the 192x192 mainloop work because it
failed the falsification test on 128x256 (ours 0.0 there).

**2026-08-18: THE GATE IS OPEN.** The mainloop drain work closed with §H27zdb (SKEW, +1.62 on V3,
95.83%). On V3 the current reading is **ours 7.14 vs cuBLAS 15.01** — they run their memory pipe
at twice our pressure — alongside tensor idle 10.9% vs their 7.9%. Arithmetic: closing the tensor
idle alone would put V3 at ~99.4%, so the remaining gap IS the tensor idle.

DO NOT call low mio_throttle the cause. It has ALREADY FAILED a falsification test: on the
128x256 family (V7/V8) we run mio_throttle ~0.0 -- lower than V3 -- and those shapes sit at
96-98% of cuBLAS, our BEST. If low mio_throttle caused slowness they would be our worst. It is a
correlated difference, non-causal at least once, and must be re-tested on V3 specifically rather
than built on.

Next measurement before acting: split the remaining barrier stall (23.51 vs cuBLAS 18.40) by ROLE
with the PC-sample ledger (`tools/h27zd0_ledger.py`) — producer-waits-EMPTY vs consumer-waits-FULL.
Pre-fix it was producer 26.02% EMPTY / consumer 6.37% FULL / DEPBAR 32.28%.

The old "6/6 unsupported metrics" was a validator bug (grepped full names; `--query-metrics` lists
base names) and is fixed — not blocked. See [[project-step3-plan]], [[feedback-isolated-measurements-mislead]].
