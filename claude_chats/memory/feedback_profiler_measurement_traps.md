---
name: feedback_profiler_measurement_traps
description: Six silent measurement traps -- SIGPIPE-killed profiles, idle power windows, CSV commas in template names, ncu output behind stdout, nsys stats keeping STALE CSVs, empty nsys capture-range reports
metadata:
  type: feedback
---

All four hit in one session (2026-09-11) and all four fail SILENTLY, producing plausible
numbers rather than an error.

1. **Never pipe a profiler into an early-exiting filter.** `nsys profile ... | grep -m1` closes
   the pipe on the first match, nsys takes SIGPIPE before finalising, and the report lands at
   **0 bytes** while a valid-looking string is still captured. Measured: piped 0 bytes,
   redirected-to-file 195,342 bytes. It looked like "this layout launches no kernel".
2. **A power window taken from a process's wall time is dragged toward idle.** Allocation, host
   fill, upload and context teardown sit near 90 W against a busy 435-670 W. A verified
   620.1 W read as 545.9 W. Fix two ways that fail differently: have the binary print its own
   timed window, AND filter telemetry on `utilization.gpu >= 95`.
3. **nsys kernel names contain `", "` inside template arguments** -- any sed/cut on commas
   destroys the Name field. Use a real CSV reader, and locate the header (`Time (%)`) rather
   than assuming line 1: nsys prints progress lines first.
4. **ncu `--csv` output is preceded by the profiled program's own stdout**, so `DictReader`
   silently takes that line as the header and returns zero metrics. Find the `"ID"` line.

5. **`nsys stats` never overwrites an existing CSV** ("SKIPPED: output file ... exists"), even with
   `--force-export=true` (that flag only re-makes the .sqlite). A re-run then reads the OLD capture's
   numbers. Always pass `--force-overwrite=true` (2026-10-12: the re-run looked like "0 cuDNN kernels").
6. **`nsys --capture-range=cudaProfilerApi` with a torch script recorded nothing**: nsys printed
   "Capture range started/ended" but the report held no kernel and no NVTX data. Use NVTX ranges
   around the timed calls + `nvtx_kern_sum` instead, without a capture range (2026-10-12).
Same day, not a profiler trap but the same failure: a `BENCH_ONLY` prefix guard (`want("cudnn")`)
was False for longer prefixes, so a capture silently timed no cuDNN. Print the COUNT of the thing
under test (BENCH lines of it, its kernel names in the CSV) after every capture.

**Why:** each produced a confident wrong table, not a crash.

**How to apply:** after any profiling step, assert the artefact is non-empty and that the
parsed row count is what you expect, before reading a single number off it. See
[[feedback_isolated_measurements_mislead]], [[feedback_vacuous_checks]].

**FIVE AND SIX (2026-09-11, found by auditing a result rather than by it failing):**

5. **A leaked `nvidia-smi -lms` sampler ran for 7,545 seconds** -- over two hours -- from an A/B
   whose cleanup trap did not fire, polling at 10 Hz through several later measurements. With
   two or three such pollers the power sensor returns STUCK values: 19 busy samples all exactly
   84.8 W while a GEMM delivering 609 TFLOPS was running. **Kill every sampler on every exit
   path, and assert no foreign sampler exists before starting a measurement.**
6. **Do not poll the GPU while a measurement is running.** My own `nvidia-smi` status checks
   contended with the harness's sampler and produced the same stuck readings. Poll the OUTPUT
   FILE, never the device.

**AND: verify a variant is ACTIVE, not merely named.** A cluster sweep identified variants by
symbol name; had the template resolved to 1x1 the sweep would have been vacuous and would have
looked exactly like a clean null. Verified instead by three independent signals: the launch
attribute in source, a cluster-consistent grid, and DRAM traffic actually falling (the
functional signature of multicast). See [[feedback_verify_reachability_not_presence]].

