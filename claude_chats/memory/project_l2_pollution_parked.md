---
name: project-l2-pollution-parked
description: L2 pollution between GEMMs is real and the kernel is named (vec4/vec8 converts), but it is SYMMETRIC between cuBLAS and us, so it is parked - and the "2.04 points" figure was never measured.
metadata:
  type: project
---

**PARKED by Gautam 2026-09-09.** The convert kernels run identically in both arms, so the pollution
is the same for cuBLAS and for us — it is an environment we do not control, not our inefficiency.

**The "2.04 points" I quoted was NOT measured.** It came from a synthetic 256 MB polluter I chose
myself in a standalone replay. Never quote it as a decomposition of the trainer gap again.

**What IS proven** (h150_stream.txt): our lead over cuBLAS goes +4.83 ms → +4.64 ms → **−0.39 ms**
as an injected polluter grows 0 → 64 → 256 MB. Our kernels depend on a warm L2 more than cuBLAS's.

**The pollutant is identified** (real 1.7B trainer nsys, offline analysis, no GPU):
`vec4_convert_to_fp32` + `vec8_convert_to_bf16` — **1,255,024 launches, 1.5x all GEMMs combined,
7.2% of GPU time**. One convert touches 100.7 MB (hidden) to 553.6 MB (ffn gate) against a **60 MB**
L2. **60% of GEMM→GEMM transitions contain a convert; 39.7% are back-to-back with a warm L2.**

**The one measurement that would settle it, still not done:** `ncu` on a GEMM *inside* the trainer
vs standalone, comparing L2 hit rate and `dram__bytes`. **That same capture also settles §H156's
open assumption** — whether the split-K reduce's 10.5 MB really reaches HBM or is served from L2.
If L2 serves it, "the reduce is at the memory wall" has headroom I dismissed.

Trainer nsys captures are already local: `prod_runs_2026_09_02/1p7b_{blublas,cublas}/*.nsys-rep`,
and local nsys is at `/usr/local/cuda-13.0/bin/nsys` — this analysis needs no GPU.
See [[project_convert_kernels_future]], [[project_dispatched_not_best]].
