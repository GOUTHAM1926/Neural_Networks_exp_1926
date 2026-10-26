---
name: feedback-sweep-combinations-not-single-knobs
description: Single-knob A/B is structurally unable to find wins in a coupled pipeline - sweep COMBINATIONS first, isolate only to explain a winner. Confirmed three times.
metadata:
  type: feedback
---

A knob measured alone and rejected is **not** a knob rejected in combination. Three independent
confirmations on H200 BF16 GEMM:

* **192x192**: `2x1_h` cluster measured ALONE is a clear loss (579.1 vs 597.8 TFLOPS, §H224,
  multicast verified active via DRAM -5.1% in §H224b). Stacked with
  `rs8_dh_ubd_dw_skew_lsw_pdl_wl` it is the **best config on the tile** - +0.538% throughput,
  -0.339% power under a 12-round crossover (§H226).
* **256x128**: every dead-work knob a null individually, **+1.95% as a full stack**.
* **128x256**: `dh` ALONE +0.37% (7/7, p=0.016) where the full stack did nothing - the
  asymmetry runs both directions.

Six work-removal experiments (§H216-§H222: STAGE_CSEL, k-loop vector ops, DESC_HOIST, EPI_A1B0,
EPI_USE_STSM, instruction excess) were each measured one knob at a time and all came back null.
That does not mean those mechanisms are dead - it means the instrument was wrong. One
131-variant combination sweep found in a single pass what those six experiments missed.

**Why:** the pipeline is coupled. A knob that removes work can free a resource that another knob
is waiting on; alone it just moves the bottleneck and reads as a null.

**How to apply:** sweep the combination space FIRST (dlsym over the registered variant list is
cheap - ~131 configs in well under an hour with a per-variant `timeout`). Isolate a knob only
afterwards, to *explain* a winner. Never conclude "mechanism X is dead" from a single-knob null.
Then gate the winner on 12 rounds + a null control - see [[feedback-paired-null-instrument]].
