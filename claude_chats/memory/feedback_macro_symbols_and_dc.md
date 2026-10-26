---
name: feedback-macro-symbols-and-dc
description: Never audit macro-generated symbols with grep; never build these kernels with -dc; and don't use a per-element relative error on near-zero GEMM output
metadata:
  type: feedback
---

Three traps that each cost real time on 2026-09-04.

**1. Never audit a macro-generated symbol set with a text search.** Our SM90 entry points are
pasted together by X-macros at preprocessing time, so `blublasBgemmSM90_bf16_nt_128x32x64_template`
appears NOWHERE in the sources even though we emit it. Grepping said we had 4 of Kota's 44; we
actually had 21, and every count in HYBRID_44_TRACKER.md was wrong for weeks because of it.
**Why:** a grep only ever sees hand-written wrappers.
**How to apply:** preprocess and read the symbols out --
`nvcc -E -x cu -I include -I src/.../SM90 probe.cu | grep -oE "blublasBgemmSM90_bf16_(nt|nn|tn)_[A-Za-z0-9_]+_template" | sort -u`

**2. `-dc` (relocatable device code) BREAKS these kernels.** Building with -dc made every kernel
on BOTH trees die with "misaligned address"; plain `-c` is correct and still parallelises.
**Why:** it changes how `__grid_constant__` TMA descriptors and cudaLaunchKernelEx cluster
launches are handled.
**How to apply:** use `-c`, one nvcc invocation PER TU (nvcc compiles TUs on one command line
sequentially -- three variant files in one call serialises ~36 min on a 44-core box).

**3. A per-element relative error is the wrong metric for these GEMMs.** Small signed inputs over
K=4096 land many outputs near zero, so the metric divides by nothing and reports huge errors for
correct kernels. The tell was both trees returning an IDENTICAL 1.117e+00.
**How to apply:** normalise by the largest magnitude in the reference matrix. Better, make fill
values multiples of 0.25 -- products are multiples of 0.0625, fp32 accumulation is then EXACT, and
a correct kernel scores err = 0.000e+00 so any real defect is unmissable.

Also: never diagnose from a run that followed a context-killing fault -- 12 entries "failed"
on leftover GPU state and all passed on a clean re-run. See [[feedback-vacuous-checks]].
