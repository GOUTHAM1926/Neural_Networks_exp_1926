---
name: feedback_test_log_format
description: Always emit ONE log file with full detail + a summary block when running B300 tests
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
---

When giving the user a command to run the SM103 template test (or similar) on the B300 server, always produce **one log file containing BOTH the full detailed reports AND a summary block** — never a grep-only view that hides the per-shape tables / MaxRelErr / TFLOPS.

**Why:** the user reviews results from the log file; a grep-filtered paste hid the detail (per-shape TFLOPS, %-of-cuBLAS, MaxRelErr, DEBUG-FAIL first-10 dumps) and caused confusion about whether reporting was intact.

**How to apply:** use the pattern
```
CUDA_VISIBLE_DEVICES=<idle> ./build/<test> 2>&1 | tee output.md
{ echo; echo "===== SUMMARY ====="; grep -E "=== Testing|Result:|##########|ALL TESTS|SOME TESTS" output.md; } >> output.md
```
so `output.md` has the full detail first, then a SUMMARY block at the end. The detailed reporting (per-shape table + `[DEBUG FAIL]` first-10 dump on rel_err≥0.05) is already in the test code — don't remove it. Related: [[feedback_verify_claims_and_benchmark]].
