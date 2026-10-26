---
name: project-192-variant-search-closed
description: REOPENED and REFUTED 2026-09-11 - the 192x192 search was NOT closed; a 131-variant combination sweep found a config that beats the dispatched one, confirmed by a 12-round crossover (+0.538% throughput, -0.339% power).
metadata:
  type: project
---

**THIS MEMORY WAS WRONG AND IS NOW CORRECTED.**

The old claim (§H48, 2026-08-29): "470 measurements, both pinned 192x192 configs rank #1 of 112,
no better variant exists, stop searching, optimise the pinned kernel." That conclusion did not
hold.

**§H225/§H226 (2026-09-11):** a 131-variant sweep on `tn 3072x3072x8192` found
`blublasBgemmSM90_bf16_tn_192x192x64_2x1_h_rs8_dh_ubd_dw_skew_lsw_pdl_wl_template` beating the
dispatched `..._192x192x64_template`. Confirmed by a **12-round paired crossover with a null
control**: **throughput +0.538% (t=+32.24), power -0.339% (t=-4.80, negative 12/12 rounds),
= +0.88% efficiency**. The null control came out non-significant on both metrics
(t=-0.50, t=-0.30), so the instrument is not vacuous.

**Confirmed, NOT pinned.** Pinning waits on a trainer A/B - see
[[project-4b-standalone-wins-trainer-loses]] and [[project-v2c-pin-regated]].

**Why the old screen missed it:** it measured knobs largely in isolation, and the winner is a
**2x1_h CLUSTER in combination**. Measured alone (§H224) that cluster LOSES - 579.1 vs 597.8
TFLOPS - and §H224b confirmed the multicast really was active. See
[[feedback-sweep-combinations-not-single-knobs]].

**How to apply:** the 192x192 search is OPEN. Do not quote "rank #1 of 112" or "stop searching".
Run combination sweeps, not one-knob screens, and gate every candidate on 12 rounds + a null.
