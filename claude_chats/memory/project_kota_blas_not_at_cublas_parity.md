---
name: project-kota-blas-not-at-cublas-parity
description: Kota's production BLAS is ~99.0% of cuBLAS in the 600M trainer, NOT at parity; and ~1/3 of any BLAS's trainer deficit is non-GEMM collateral in one kernel.
metadata:
  type: project
---

Kota's production SM90 BLAS sits at **99.0% of cuBLAS** in the 600M trainer, not at parity. Proven
from his own two logs (`outpput.md` WITH_BLUBLAS=1 and `outpput(1).md` WITH_BLUBLAS=0, same GPU,
same script, both under nsys): steps 3-9, his BLAS 72,178.1 vs his cuBLAS 72,951.7 = **98.940%,
-773.6 tok/s**. On our own plain runs the same library is 72,918.0 vs 73,583.8 = **99.095%,
-665.8 tok/s**. Our hybrid (our kernels + his dispatcher) is 72,336.0 = **98.304%, -1,247.8**.

**Why:** the belief "his run matches cuBLAS" keeps resurfacing and drives us to hunt for a missing
setting of his. There is none. His full preamble differs from ours in 4 lines and all are
LR-schedule / dataset-shard / PATH cosmetics; his two arms have byte-identical Configuration blocks.
See §H142.

**How to apply:** our target is **his 99.0%**, and the gap to close is **0.80 points / 582 tok/s**,
not the ~1.7% to cuBLAS. Also: only about **two thirds** of a BLAS's trainer deficit is in its GEMMs
— the other third is non-GEMM collateral on *identical* code with identical launch counts, and
**72% of that collateral is one kernel, `gqa_bwd_vj1d`** (+3.5% under his BLAS, +5.2% under ours).
Memory-bound kernels (`vec4_convert`, `vec8_convert`, `add_inplace`) show ~0%. That is the
power->clock-cap mechanism, so a GEMM-only optimisation plan understates what is winnable.
Related: [[project-kota-blas-4b-headtohead]], [[project-power-clock-solved]], [[project-trainer-parity-1p7b]].
