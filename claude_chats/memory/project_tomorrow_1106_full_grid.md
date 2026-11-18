---
name: project_tomorrow_1106_full_grid
description: "IMMEDIATE TASK 11-06 (user 11-05 ~22:10): before/after of the 11-04 + 11-05 optimizations on the FULL grid -- SFT AND pretraining, 135M AND 1.6B (11-05 only had SFT=135M, pretraining=1.6B)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T16:40:05.634Z
---

**Task (user, 2026-11-05 ~22:10 IST, angry):** "for SFT you measured 135M and for pretraining 1.6B? You should have
measured 135M for pretraining too, and SFT for 1.6B too. Tomorrow we will measure these; document as an immediate
future task." The SFT -23% and pretraining -0.07% tables were two DIFFERENT models: not comparable.

**Grid to measure:** 4 cells = {SFT trainer, pretraining trainer} x {135M, 1.6B}. Each cell gets 3 builds:
- **B0 = before 11-04:** the 11-03 pushes: BluTrain 7184fe1, TI d18f820, BluBLAS 297f372.
- **B1 = after the 11-04 changes:** BluTrain 96f135c, TI df615ce, BluBLAS a6d4292. This has the Muon guard,
  empty_cache at step 0, batched NS + addmm, and the GEMM fixes.
- **B2 = after 11-05 (current):** BluTrain accc6c8, TI 66a32a7, BluBLAS a6d4292. This adds items 9, 4, 3, 13, 5+8, 6+7.

**Report per cell:** step time, tok/s, MFU, optimizer-phase time (nsys), and losses (noise check).

**How to apply:**
- Same GPU, same step window, swapped order; GPUs locked ([[feedback_guard_our_gpus_during_runs]]). Ask for GPUs
  first: all 16 GPUs went to others 11-05 21:56 ([[feedback-ask-before-every-gpu-run]]).
- Add multi-GPU (8 GPUs) at least for 1.6B pretraining, together with the overlap_refresh on/off
  ([[project_tomorrow_1106_overlap_refresh]]).
- Needed:
  - a 135M pretraining YAML (the production YAML with the 135M dims: d 512, 37 layers, 4/2 heads, ffn 1408,
    vocab 49280);
  - 1.6B SFT (the SFT trainer with N_LAYERS=26 D_MODEL=2304 Q_HEADS=18 KV_HEADS=6 FFN_HIDDEN=6144 VOCAB=50304). There
    is no 1.6B SFT start checkpoint, so use random init. That is fine for speed: losses are then only a noise check.
    data_4k token ids are < 49280 < 50304.
- Explain why the cells differ: optimizer share of the step = 1 optimizer step per (micro-batches per step). SFT has
  2 micro-batches per step; 1.6B pretraining at 1 GPU has 32 (1M tokens).

**Add the micro-batch count (user, ~22:15: "SFT was tested with 2 micro-batches, not 8").** Every 11-04 / 11-05 SFT A/B
used B=4, ROWS=8 = 2 micro-batches per optimizer step. That is the BEST case for optimizer-side gains: one optimizer
step per 2 micro-batches.
- Measure SFT at 2 (ROWS=8), 5 (ROWS=20, the run_sft.sh default) and 8 (ROWS=32) micro-batches, both models.
- Pretraining at the production batch (1.6B: 32 micro-batches at 1 GPU, 4 at 8 GPUs).
- Estimates to check (from 11-05: 48.9 ms per micro-batch, optimizer ~39.4 ms before / ~8.5 ms after): 2 -> -23%
  (measured), 5 -> ~-11%, 8 -> ~-7%.

**FULL MATRIX (user, ~22:20): "tomorrow we measure B = 2, 4, 8 and different micro-batch counts, 1.6B for SFT, and
pretraining for the smaller model too, for all those B and micro-batch variations".**
- Dimensions:
  - trainer {SFT, pretraining};
  - model {135M, 1.6B};
  - B (rows per micro-batch) {2, 4, 8};
  - micro-batches per optimizer step {2, 5, 8} (SFT: ROWS = B x count; pretraining: set the global batch in the YAML
    to B x T x count x W);
  - builds {B0 before 11-04, B2 = 11-05 heads} (B1 = after 11-04 only if time allows).
- Size: 2 x 2 x 3 x 3 x 2 = 72 configurations, x 2 (swapped order) = 144 runs.
- Run times: SFT runs ~1-1.5 min; pretraining 1.6B at 1 GPU ~0.7 s per micro-batch (8 micro-batches ~6 s per step,
  ~2-3 min per run with startup).
- On 8 GPUs in parallel (pairs old / new at the same time on the same GPU type, locked): roughly 1-1.5 h.
- Check memory first: 1.6B SFT at B=8, T=4096 (pretraining already runs B=8, so it should fit).
- Then the 8-GPU overlap on/off.

**DONE 2026-11-06 11:40-13:02 (work doc 45.174, 2x H200 box 95.133.254.108).** 36 cells, old/new simultaneous, 30
exact steps, 0 errors, 0 disturbed; the swapped pass was skipped (user).
- New vs old, average over the 9 B x micro-batch cells: SFT 135M -13.7%, pretraining 135M -11.8% (up to -32% at
  B2 x 2 micro-batches); SFT 1.6B and pretraining 1.6B ~-1.5% (cells +-1.5%: GPU 1 ~1.5% faster than GPU 0).
- Items 11 and 12: DROPPED by the user the same morning.
