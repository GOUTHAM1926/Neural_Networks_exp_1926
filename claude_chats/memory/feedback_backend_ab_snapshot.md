---
name: feedback-backend-ab-snapshot
description: A trainer cuBLAS-vs-BluBLAS A/B cannot be done by build order - WITH_BLUBLAS is baked into ONE libtensor.so, so the second build silently overwrites the first; snapshot each backend and verify three ways
metadata:
  type: feedback
---

**Never A/B two BLAS backends by rebuilding between runs.** `WITH_BLUBLAS` selects what
`BluTrain/Tensor-Implementations/lib/libtensor.so` links against, and there is exactly **one** of
that file. Build `=1`, run, build `=0`, run — and both runs used the `=0` library.

**Why:** this silently voided a full day of trainer measurement (§H27bj, §H27bn). Four runs
agreed to 5 tok/s and I read that as excellent repeatability; it was the same binary four times.
The retracted conclusions were "end-to-end parity" and "the tuned dispatch is 1.4% slower".

**How to apply — snapshot, then verify three independent ways, every run:**

    build once per backend into its own dir:  /root/libs_blublas, /root/libs_cublas
    (libtensor.so + libblublas.so + the runner binary, which is ALSO a single overwritten file)

    CHECK 1 before  nm -D --undefined-only libtensor.so | grep -c blublas
                    -> 12 = BluBLAS, 0 = cuBLAS.  ABORT the run on mismatch, emit no number.
    CHECK 2 during  grep -c libblublas /proc/<pid>/maps    -> 5 = ours mapped, 0 = not
                    (skip the `timeout` wrapper - match on /proc/<pid>/comm, not pgrep -f)
    CHECK 3 after   GEMM kernel names in nsys  -> bgemm_sm90_* vs nvjet_sm90_*

`make all` does **not** build `libtensor.so`; `make train-bin MODE=production` does, and
`WITH_BLUBLAS ?= 0` is the default, so a plain `make` is always the cuBLAS backend.

**Generalise it:** whenever two arms share one build artefact, the artefact is the experiment's
weakest link. Snapshot it per arm and verify identity at run time, not at build time.
See [[project-trainer-parity-1p7b]], [[feedback-verify-claims-and-benchmark]].
