---
name: blutrain-integration
description: BluTrain 1.7B trains on our blublas at ~93-94% of cuBLAS after four bugs fixed 2026-08-06; next = BluTrain-side bugs T1/T2
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-08-06T14:38:15.782Z
---

**State 2026-08-06 (H200, sm_90).** BluTrain's 1.7B production config
(`train/production/bluscript_config.yaml`, d_model 2048 / 36 layers / GQA 16-4 / ffn 5632 /
vocab 50304 / weight_tying ON / BxT 4x4096 / grad_accum 64 = **1,726,376,960 params**) trains
correctly on our BLAS at **~93-94% of cuBLAS** end-to-end (26.5K vs 28.3K tok/s). Losses match
cuBLAS to <=8.1e-4, grad norms to ~0.02. Writeup: **analyzations_h200.md §H27z.47-.51**.

**2026-08-07 nsys Phase 1 (§H27z.53).** Per-KERNEL we are at **80.8% of cuBLAS on GEMM**
(638 vs 790 TFLOPS; 64.5% vs 79.8% of H200 peak). GEMM is only **41.7%** of our GPU kernel time, so
end-to-end 94% and per-kernel 81% are both true. All 37 non-GEMM kernels have identical launch
counts across the two backends -- the ONLY difference is the GEMM. Only **1 of the 15 shapes hits
tier-1**; the other 14 are tier-2 tree picks, and the dispatcher predicts the nsys-observed kernel
15/15. Biggest single loss: `tree_nn` returns 128x128 for large-N NN, so gate_up-fwd
(16384x11264x2048) is 75.0% and alone is 34% of the deficit -- cuBLAS uses 192x192 there and we
already own NN_192x192x64_1x2_v. Reusable analysis: `tools/nsys_train_gemm_pair.py`.

**BENCHMARK IS NOT SYMMETRIC -- say this with every end-to-end number.** `MatmulBackward.cu` has two
separate implementations; `#ifdef WITH_MYBLAS` runs wgrad on a secondary stream so dgrad/wgrad
overlap, the `#else` cuBLAS branch has zero secondary-stream references and serialises both on one
stream. Measured: 508 ms hidden for us over 10 steps, 0 ms for cuBLAS. The per-kernel 80.8% is
unaffected and is the honest library-vs-library number.

**Four bugs found and fixed that day — each hid the next:**
1. `BLUBLAS_HAS_SM90` never defined in BluTrain's Makefile → the whole SM90 block was
   `#ifdef`'d out (no link error: the extern decls are inside the same ifdef). Hopper silently ran
   SM89 Ada kernels.
2. `strideC` gate missing `batch > 1` → every backward wgrad (passes strideC=0 for 2-D) declined
   by SM90, sent to SM89.
3. `blublasCreate` never initialised `is_sm89`/`sm_count` → UB; kernel choice from uninit heap.
4. Grouped threadblock raster dropped output tiles when GROUP_ROWS ∤ grid_y → unwritten tiles kept
   recycled allocator memory → NaN. Was copy-pasted in BF16/SM89 + FP16 + FP32/SM86; now one
   `src/level3/batched_gemm/common/Gemm_block_swizzle.cuh`.

**RETRACTED: the old "60% of cuBLAS" 48M number was never our Hopper work** — bug 1 meant SM89
kernels were running. Ignore it.

**"SM80" is a misleading variable name**, not a module. `sm_ver = is_sm89 ? 89 : 80`; both branches
call SM89-module kernels. We have SM89/SM90/SM100/SM103 + FP32/SM86 only.

**Next session (T1/T2, both BluTrain's code not ours, need reproducers first):**
`MatmulBackward.cu:585/594` beta shadowing (local `beta=0` shadows caller's `beta_b` in the
non-reduction branch); and `!grad_a_done`/`!grad_b_done` have no bf16 fallback (float+fp16, and
float only, respectively).

**Debug switches (default off, free in production):** `BLUBLAS_SM90_TRACE=1` (logs every shape with
ACCEPT or decline reason), `BLUBLAS_SM90_DISABLE=1` (force legacy path — splits "is it SM90" in one
run), `BLUBLAS_NAN_CHECK=1` (scans A/B before + C after each GEMM at the public funnel, reports
max|value| and the overflow bound, aborts on first genuine NaN creation). Plus
`examples/test_bgemm_sm90_beta.cu` (beta 0/0.5/1 incl. beta=0 with C pre-filled NaN, **via the
public API** so it exercises real routing) and `data/blutrain_1p7b_shapes.tsv`.

**Run:** `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` then
`make run-train-mpi MODE=production NP=1 WITH_BLUBLAS=1` from `BluTrain/`. Binary is now
`_gautam_runner` (`TRAIN_RUNNER` var) so nvidia-smi distinguishes it from a colleague's
`_train_runner`. Data: `/root/gautam_k/blublas/flux_train_val_bin_data` (3 train + 1 val shard).
BluTrain's Makefile now has `-MMD -MP` header dep tracking, so a `.cuh` edit rebuilds.

**Testing traps proven the hard way:** a harness that memsets C to zero cannot detect unwritten
output tiles; and calling an internal entry point instead of the public API skips the routing that
production actually uses. Both let bug 4 through every existing test.

Related: [[project_active_repo]] [[feedback_verify_claims_and_benchmark]] [[project_reduction_build_gotcha]]

## 2026-08-24 -- OPTIMIZED MODULE PORTED, GEMM 80.8% -> 96.6% (§H27zO)

**THE PORT IS 5 FILES.** BluTrain builds a VENDORED copy at
`BluTrain/Tensor-Implementations/BluBridge-BLAS/` (`BLAS_DIR := BluBridge-BLAS`). Our top-level
`src/` is NEVER compiled into the trainer. Copy: `Bgemm_sm90_warpspecialized_template.cuh`,
`Bgemm_sm90_bf16.cuh`, `Bgemm_sm90_dispatch_table.h`, `Splitk_reduce.cuh`,
`Bgemm_sm90_dispatcher.cu` + the Makefile `EXTRA` line. `include/` is already identical.
`tools/verify_blutrain_wiring.sh` = 20/20 pre-flight. **Full BluTrain build is 10m02s**, measured --
the "~1 hour" figure quoted for months was never measured and is wrong. A dispatch-TABLE-only
change needs just `Bgemm_sm90_dispatcher.o` rebuilt (seconds): no `*_variants.cu` includes it.

**21/21 call sites verified** issuing the pinned config, cuBLAS issuing the same kernels as our
standalone identity gate. 34,720 GEMM launches in both backends. `gate_up-fwd` went **75.0% ->
98.3%** -- the August deficit that was "alone 34% of the gap" is gone.

**TWO MEASUREMENT FAULTS FOUND, BOTH REAL:**
1. **Backward runs on TWO streams.** grad_B on a secondary stream (8,680 launches = exactly the
   wgrad sites) while cuBLAS uses one. Our kernels were timed sharing SMs; cuBLAS's never were.
   `BLUTRAIN_BWD_SINGLE_STREAM=1` added, **default OFF** (the overlap is a real wall-clock win --
   it is a measurement tool, not a config change). Worth 180 ms, 93.1% -> 94.5%. It penalised the
   **dgrad** sites, not wgrad as I predicted (wV-dgrad 65.2 -> 89.5).
2. **`mpirun` forwards ONLY what `-x` names.** `BLUBLAS_SM90_TRACE` and `BLUBLAS_SM90_DISABLE`
   were never listed, so they silently did nothing. Fixed via `MPI_FORWARD_ENV`. **Any new env
   switch must be added there as well as read in the code.**

**nn/nt ARE CLOCK-LIMITED; tn IS NOT.** Bench (locked 1410) -> training (free): cuBLAS gains
+21.7%, we gain +11.7%. By layout our gain is nn +16.1, nt +15.8, **tn +3.4** -- three tn sites go
BACKWARDS with more clock. Locking training to 1400 recovers nn/nt to their standalone numbers
(lm_head-dgrad 88.2 -> 99.8) and does NOT recover tn (gate_up-wgrad 88.1 -> 88.6 vs 100.3
standalone). **At a held 1400: 96.6% of cuBLAS, nn 98.2 / nt 97.1 / tn 89.2.**

**OPEN: tn is 6-21 points worse in training than on the bench AT THE SAME CLOCK.** 78% of the
remaining 474 ms. Ruled out by measurement: clock, beta (all launches are `_bz` beta-zero), stream
concurrency, wrong kernel. **The cluster/raster mismatch is NOT the explanation** -- it is identical
in the standalone run that measured 100.3%, and a constant cannot explain a difference; do not
record it as the cause. Still live: **L2 residency** (our harness loops ~2000 iterations on the SAME
buffers; training touches fresh activations -- and the one tn site that DOES match, `lm_head-wgrad`
99.8%, is the one too large for L2 reuse to help either side), and tensor strides/alignment.

**EVERY LOCKED-CLOCK tn NUMBER WE HAVE QUOTED IS OPTIMISTIC.** The step-3 campaign measured at
1200-1400 locked, which isolates kernel work but hides a ceiling production hits.

**Run:** `make nsys-train MODE=production NP=1 WITH_BLUBLAS=1 NSYS_OUT=<name>` (profile config =
grad_accum 4, 10 steps, shapes unchanged), then
`python3 tools/nsys_train_gemm_pair.py <ours.sqlite> <cublas.sqlite>` -- pairs on correlationId,
never start time. Captures kept: nsys_blublas_repin / _1stream / _1stream_lock, nsys_cublas_repin /
_lock. The August baseline `nsys_train_blublas/` was deliberately not overwritten.
