---
name: feedback-test-the-consumer-build
description: "Library-level tests must use the SAME build the consumer loads (TI `make blas` libblublas.so), not BluBLAS's standalone libmyblas.so; an `inline` host function that launches a `static` kernel from a header breaks by link order"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-03T10:47:47.281Z
---

2026-11-03: the D2 tile-table fix passed every library-level test (abtest, stall test, tiletab_test) on the BluBLAS
standalone build (libmyblas.so), but in the trainer (TI `make blas` build of libblublas.so) every GEMM call did
cudaMalloc + failed fill-kernel launch + cudaFree (66,033 pairs per 99 micro-batches, 14 ms GPU idle per micro-batch).
Cause: `inline` helper in a header included by SM90 (sm_90a) and SM103 (sm_100a/103a only) files; the linker kept ONE
copy (first object in link order = SM103 in TI's build), whose `static` fill kernel had no H200 code. Results stayed
bit-identical (silent fallback order), so output checks could not catch it; only nsys API counts did.

**Why:** two builds of "the same" library differ in link order, arch flags per file and objects; a test on the wrong
build proves nothing about what the trainer runs. A silent fallback hides failures from correctness checks.

**How to apply:**
- Run library tests against the exact .so the consumer resolves (ldd of the trainer binary), or at least also on it.
- After any host-side BluBLAS change, check nsys API counts (cudaMalloc/cudaFree/cudaMemcpy) in the REAL trainer.
- Header helpers that launch kernels from files with different -gencode sets must be `static inline` (per file), and
  error paths must print once, never fall back silently.
Related: [[feedback_verify_reachability_not_presence]], [[feedback_isolated_measurements_mislead]],
[[project-future-blublas-tile-map-sync]].

(Moved from the MEMORY.md index line on 2026-11-05:) includes the inline + static-kernel LINK-ORDER trap (the consumer build can pick a different copy of a kernel than the test library).
