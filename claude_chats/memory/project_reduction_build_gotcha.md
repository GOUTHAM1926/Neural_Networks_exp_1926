---
name: project_reduction_build_gotcha
description: Stale-build trap - these Makefiles have historically not tracked .cuh/.h deps; verify a rebuild happened before debugging any header-only change
metadata:
  node_type: memory
  type: project
  originSessionId: 60a6dc55-8dda-45fe-864f-235f3482492e
  modified: 2026-08-06T07:05:01.301Z
---

**Before debugging ANY failure that follows a header-only change, confirm the build actually
rebuilt.** This has cost real time three times: Reduction, the SM90 1x2 cluster work, and the
§H27z.46 symmetric-split fix (first H200 run failed on a stale `libblublas.so`).

The Tensor-Implementations Makefiles (BBP, BluTrain, sm86/sm89/master_gau trees) historically had
NO `.d` dependency files — `make` does not recompile when an included header changes.

**Why it is worse than ordinary staleness:** nearly all the SM90 GEMM lives in
`Bgemm_sm90_warpspecialized_template.cuh`. When only some translation units rebuild you get a MIXED
binary — host launchers compute TMA descriptors from the new header while device kernels still
carry the old geometry. Descriptors and kernels disagree and *everything* fails, which reads like a
broken kernel rather than a broken build.

**Status:** standalone `BluBridge-BLAS/Makefile` has `-MMD -MP` + `-include *.d`.
`BluTrain/Tensor-Implementations/Makefile` had neither; fixed 2026-08-06 (`DEPFLAGS := -MMD -MP`
on the four flag vars + two `-include` lines). Watch for the half-fix its own comment records:
shipping `-include *.d` while never generating the `.d` files is silent. Other trees (BBP,
sm86/sm89/master_gau) are probably still unfixed — check before trusting a rebuild there.

**Original Reduction case:** after editing `include/ops/helpers/ReductionImpl.h` (used only by
`src/UnaryOps/cpu/Reduction.cpp`), plain `make tensor` relinks the STALE `Reduction.o` → CPU
middle-axis (ax{1}) reductions stay wrong while GPU is correct (42/54 on BluTrain). Fix:
`rm lib/objects/src/UnaryOps/cpu/Reduction.o lib/objects/src/UnaryOps/cpu/ReductionUtils.o`.

**How to apply:** if a header-only change misbehaves, force a clean rebuild of the affected objects
before theorising about the kernel — `rm -rf` the object dir for that lib is the reliable move.
See [[project_reduction_bugs]], [[feedback_verify_claims_and_benchmark]].
