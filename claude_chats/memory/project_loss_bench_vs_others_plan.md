---
name: project_loss_bench_vs_others_plan
description: "item 2 (bwd 1.6x) + item 3 (-5,670 MiB trainer) VERIFIED; 10-19 root cause = loss skips CudaNativeType; ALL pending loss work (CudaNativeType fix, full SFT run, decisions) AFTER the attention work"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-17T12:53:30.818Z
---

Agreed 2026-10-17 (user: "we will do on loss kernel once this attention thing is done ... i will tell when to do").
- Ours = the TRAINING pair: `sparse_cross_entropy_forward_cuda_impl_vec_save_stats` (F, saves row max + sum) +
  `sparse_ce_backward_cuda_vec_with_stats` (B, reuses them). Validation uses the no-stats forward.
- Report forward, backward AND forward+backward separately (user's request), at our SFT shape (8 x 4096, V 50,304, bf16,
  real stage-0 labels, ~23% trained), plus the memory kept between forward and backward.
- The others split the work differently, so only F+B (and memory) is like for like:
  PyTorch F.cross_entropy saves the full B x V log-softmax; Liger computes the GRADIENT IN THE FORWARD (backward ~ a scale);
  Apple CCE fuses the lm_head GEMM (compare vs our GEMM + loss). Say this next to every per-direction number.
- Existing standalone numbers (10-09, profiling_reports/sft_loss_bench_2026_10_09/): F 0.276 + B 1.600 = 1.878 ms; backward on
  trained rows only ~33% of HBM peak (never profiled) -> likely the place to optimise if the comparison shows a gap.
- Needs a torch env with liger-kernel and cut-cross-entropy (the 8x box's benv has torch; pip the two there).

- 2026-10-17 ~18:20 IST the user said GO (2x box, OUR GPU = 0, not 1). Built in profiling_reports/sft_loss_vs_rivals_2026_10_17/:
  loss_env_setup.sh (own venv /root/gautam_k/lossenv: torch 2.14.1 cu130 + liger-kernel + cut-cross-entropy + transformers,
  no pip cache), run_loss.sh [gpu] (ours = 10-09 bench unchanged, --warmup 2000 --iters 500; rivals = loss_rivals_1017.py:
  Part 1 torch_fp32 / torch_bf16 / torch_compile / liger on the SAME hash logits + micro-batches as ours -> losses must match;
  Part 2 lm_head + loss: gemm_only, torch, liger_flce, cce, cce_exact), loss_compare.py (laptop tables). Laptop checks done:
  hash = CUDA bit for bit, micro-batches 0/287/458/415 match ours.
- RESULTS 18:41 (§45.33): losses = PyTorch fp32 to 6 decimals. F+B median batch: ours 1.885, LIGER 1.525 (-19%), compile 3.917,
  torch bf16 6.231, torch fp32 21.14 ms. Liger faster in every trained case (min -2% .. nomask -39%); compile faster at nomask.
  Cause: our BACKWARD on trained rows runs at 1.5 TB/s (one 16-B load in flight per thread; LossKernels.cu
  sparseCENormalize_from_stats); compile's backward 3.3 TB/s. Part 2: Liger FLCE -2..-7% vs cuBLAS+ours with 0.55 vs 6.5 GiB;
  CCE -13% at median ONLY because it skips masked tokens in the lm_head (its kernels ~half cuBLAS speed; +41% at max, 3.4x nomask).
  Fix list (user decides): (1) lm_head + loss on trained positions only (~36 -> ~9.5 ms per micro-batch, est.),
  (2) backward: unroll 4-8 loads in flight + 16-B zero stores, (3) gradient in place into the logits (-3.1 GiB).

**Why:** the user wants to know whether our loss kernel beats the others; it was planned (§38.10) but never done.
**How to apply:** do NOT start it on my own; wait for the user's go after the attention forward work.
Related: [[project_sft_loss_done]], [[project_bf16_logits_h200_validated]], [[project_state_2026_10_17]].
- ITEM 2 DONE ON LAPTOP 10-17 (§45.35): real cause = bfloat16_t's user copy ctor -> LossVec<bf16> copied as 8 x 2-byte
  LDG/STG (no 128-bit access at all); fix loss_ld16/loss_st16 (uint4) + unroll 4; 20 configs bit-identical on RTX 3060, 1.46x
  there. H200 test = run_loss_v2.sh (old/new/old/new + gradient checksums). Forward has the same issue in its LDS (8 x LDS.U16): later.
- ITEM 2 VERIFIED ON H200 19:27 (§45.36): 6/6 bit-identical x 4 runs; median B 1.607 -> 1.008 ms (1.6x), F+B 1.883 -> 1.281;
  nomask B 2.65x, F+B 1.98x; bwd now ~4.0 TB/s; OURS NOW BEATS LIGER in every case (Liger +19..27% time). Not committed
  yet (user commits). Next = item 3 (gradient into logits, opt-in switch, standalone + trainer A/B with allocator peaks).
- ITEM 3 WRITTEN 10-17 (§45.37, not yet run): opt-in grad_into_logits (LossOps -> node -> in-place kernel; trainer
  BLU_LOSS_GRAD_INPLACE=1). ALSO FOUND: CUDA loss backward allocated the gradient TWICE (Tensor::empty before the branch + again
  in it) -> logits + 2 grads at once; fixed for everyone. Test: run_item3.sh [gpu] standalone|trainer (test_loss_grad_inplace.cu,
  3 arms old_out/new_out/new_in; trainer off/on x2 on a 64/16-row tiny set).
  Trainer phase now has an OLD arm (bluscript_sft_old.cpp = today's file minus item-3 lines + old LossOps.h, linked loss_v2).
- ITEM 3 STANDALONE VERIFIED ON H200 20:06 IST (§45.38): loss + W.grad + X.grad checksums identical in all 3 arms and both
  reps. allocated peak old 10,536 -> new-off 7,754 (double-allocation fix) -> new-on 4,610 MiB (-5,926); reserved peak
  12,952 -> 10,174 -> 4,618 (-8,334). TRAINER PHASE TRY 1 FAILED before step 0 = MY SCRIPT: the 2x box has only
  stage0 train/labels.bin, so `head -c` made empty inputs.bin; also 64 rows x 2 epochs = 2 steps is refused by the
  decay-tail check (needs BLU_SFT_EPOCHS=4). Fixed on the laptop (run_item3.sh fdf019ec; tiny set made on the laptop at
  ~/Downloads/sft_data/tiny_item3, md5 of all .bin 90d39b16). Try-1 logs: box_runs/item3_1017_try1/. NEXT: user OKs
  copying the set + script to the box and `run_item3.sh 0 trainer` (~10 min).
- ITEM 3 DONE 20:52 IST (§45.38): trainer A/B try 2 PASSED, 6/6 runs exit 0, step-0 loss 11.236371 in all; allocated
  peak old 125,046 -> off 122,264 (-2,782) -> on 119,376 (-2,888) = -5,670 MiB; reserved -5,120. Grad-norm 5th-digit
  differences = run-to-run noise (same binary differs; attention bwd dQ atomics). OPEN user decisions: switch default on?
  commit the 8 files (listed in §45.38).
- 10-19: item 2 re-checked (§45.42): the loss module never used the library rule CudaNativeType (CudaTraits.h:17-22; reduction
  uses it) -> our bfloat16_t (not trivially copyable) split 16-B vectors into 8 x 2-B; __nv_bfloat16 gives 1 x 16-B. REMIND
  AFTER THE ATTENTION WORK (user, 10-19): (1) loss -> CudaNativeType + verify bit-identical + H200 bench; (2) full-length SFT
  run with the switch on vs the 10-07 run; (3) default-on + commit decisions; (4) fwd LDS.U16 / B300; (5) item 1 after v1.
