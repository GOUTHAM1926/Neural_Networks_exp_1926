---
name: project-future-matmul-bf16-racecheck
description: "FUTURE TASK (user asked 2026-10-05) — compute-sanitizer racecheck warnings inside TI's own bf16 matmul kernel matmul_bf16_optimized<128,128,32,32,32>; never changed a result so far"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-05T16:52:10.278Z
---

FUTURE TASK, noted at the user's request on 2026-10-05: look at the shared-memory race warnings in TI's own bf16 matmul kernel.

**What was seen:**
- `compute-sanitizer --tool racecheck` on `test_muon_rsag` (H200, TI `a60099b`, cuBLAS build) reported 20 printed **Warnings** (not Errors).
- They are write/read hazards inside `OwnTensor::matmul_bf16_optimized<128,128,32,32,32>`, at offsets such as `+0x6f60` write vs `+0x7120…+0x7620` read, 128-256 hazards each.
- This is the kernel `cuda_matmul` uses in the bf16 Newton-Schulz.

**Why it is only a future task:**
- In more than 12,000 matmul calls during the 10-05 Muon investigation, the same input always gave the same output (ns_probe4: 1 result per input in 200 calls each; the run_debug3 per-operation fingerprints).
- So it did not cause the flaky test. That was `MuonKernels.cu` atomicAdd ([[project-muon-rsag-root-cause-1005]]).

**What to do:**
- Read the kernel's shared-memory loop: is a `__syncthreads()` missing between the last read of a tile and the next write, or is it a double-buffer racecheck false positive?
- Then rerun racecheck with `--racecheck-report hazard` and `--print-limit 0`.
- If real, write a stress test (busy SMs / L2 flush, as in ns_probe5 part 2).

**Where the logs are:** `profiling_reports/muon_bug_2026_10_05/box_final_2026_10_05/extracted/muon/logs/sanitizer_racecheck.txt` (and `box_logs/extracted/logs/`). The doc `muon_flaky_test_bug.md` section 10 also lists it.
