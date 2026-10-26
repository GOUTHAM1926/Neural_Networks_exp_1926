---
name: project-tensor-ceiling
description: 130% needs the tensor pipe at 244% of peak = IMPOSSIBLE; the apples-to-apples ceiling is ~102.8% and only fp8/sparsity can pass it. Both BluBLAS and cuBLAS are tensor-bound at 92-99% of peak with a 0.6% time gap at locked clock - so 105-109% end-to-end is NOT reachable on these tiles at bf16, and the arithmetic says why
metadata:
  type: project
---

**MEASURED 2026-08-28 (§H30/§H31), 15 pins, both arms, locked 1400 MHz.**

    TIME               1.006x   we are only +0.6% slower  <- the real work gap
    ELAPSED CYCLES     1.007x
    INSTRUCTIONS       1.811x   we ISSUE +81% more instructions in the SAME time
    tensor pipe % peak 92.7 ours vs 93.5 cuBLAS  (parity, 0.8 pp)

**On six of fifteen shapes BOTH kernels run at 97-99.6% of peak sustained tensor throughput** —
V8d 99.6/99.6, V5 98.6/98.6, V2c 98.5/98.9, V2b 98.2/98.5, V8c 98.1/98.4, V2 97.4/97.7. Neither
library can go faster there without issuing less math.

**THE ARITHMETIC THAT BOUNDS EVERYTHING.** GEMM is 37.24% of trainer kernel time, so
`our_total = 0.6276 + 0.3724 x g`:

    trainer 100%  ->  g = 1.000   (GEMM at parity)
    trainer 105%  ->  g = 0.872   our GEMM must be 12.8% FASTER than cuBLAS
    trainer 109%  ->  g = 0.778   our GEMM must be 22.2% FASTER
    GEMM time 0   ->  ceiling 159%

With both tensor-bound at 92-99% of peak, **a 13-22% GEMM speedup does not exist to be found**
on these tiles at bf16. Realistic: GEMM few % (~1.9% end to end) + power/clock recovery (~1.5%)
= about **103%**.

**Consequence for strategy:** parity-class levers (L2, power, stalls, dead work) top out near
103%. Beating cuBLAS by 30% requires leaving the apples-to-apples frame — different tiles,
different features, lower precision, or removing work from the 62.76% that is NOT the GEMM.
Largest identified candidate there: `convert_type_kernel` bf16->float 11.0% + float->bf16 10.2%
= **21.2% of total GPU time in separate kernels**. Previously parked because speeding them up
helps cuBLAS equally — true only while they stay SEPARATE kernels. Fusing them into our GEMM
prologue/epilogue removes them from OUR arm only. Not attempted, not measured.

Also: instruction excess costs no TIME (81% more, same duration), so instruction cuts are a
POWER lever at best, never a speed lever. See [[project-instruction-track-refuted]],
[[project-power-clock-solved]], [[project-open-tasks-2026-08-28]].

**QUANTIFIED 2026-08-29 (§H41). THE CEILING IS ~102.8% AND IT IS A HARDWARE LIMIT.**
Tensor pipe as % of PEAK across all 15 pins: mean ours **92.7%**, cuBLAS 93.6%. Only **7.3% of
idle pipe exists in the entire kernel set.**

    if EVERY shape hit 100% of peak -> GEMM 1.078x -> +2.78% end to end -> ~102.8% of cuBLAS
    trainer 105% would need the tensor pipe at 106% of peak
    trainer 130% would need it at 244% of peak
    trainer 140% would need it at 398% of peak

The pipe cannot exceed 100%, so **no tile / raster / cluster / feature combination can reach
130%.** Sweeping all ~720 variants is a real search whose answer is bounded at ~103%.

**THE ONLY ROUTES PAST IT:** fp8 (Hopper ~1979 TFLOPS vs ~990 bf16, a clean 2x — this is where
DeepGEMM's advantage actually comes from, a PRECISION change not a tiling one), structured
sparsity (2x), or fusing so fewer GEMMs run. All are outside apples-to-apples, which is exactly
why they can exceed the ceiling.

**INSTRUCTION CLEANUP DOES NOT UNLOCK TENSOR HEADROOM — they are independent levers.** We execute
81% more instructions than cuBLAS *in the same wall time at parity tensor utilisation*, so the
overhead is not starving the pipe. Decisive: on V8, cuBLAS carries almost none of our redundant
work and is ALSO stuck at 75.1%. The idle there is a shape property, not our integer math.

    instruction hunt -> power/clock path, ceiling ~+0.95%
    tensor headroom  -> time path,        ceiling ~+2.78%
See [[project-instruction-excess-worklist]].
