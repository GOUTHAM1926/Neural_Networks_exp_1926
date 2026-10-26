---
name: project_rope_bf16_opt_1022
description: "10-23: PORTED (lib md5 6ec34cc9, 36/36 bit-identical, SFT 1.40-1.73x); rivals: our fwd fastest, torch.compile AND Quack bwd faster (14/16 each) (low occupancy, ncu-proven); NEXT 10-24 = detailed analysis before any bwd edit. Kruthik bf16 fused RMSNorm+RoPE optimization (10-22): ncu-proven causes, opt v1 results on H200 (SFT 1.25x, bit-identical), the real limit = L2 (cos/sin 24x re-read + scalar loads), v2 plan; rival bench blocked on python3.12-dev"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-22T16:05:24.515Z
---

Files: laptop `profiling_reports/rope_bf16_port_2026_10_22/` (bench_rope_bf16.cu, rope_bf16_ncu_1022.ncu-rep,
opt/FusedRMSRoPE_bf16_opt.cu + opt/bench_rope_opt.cu, rivals/ harness); box `~/gautam_k/rope_opt` (never touches the
run's library). User rule 10-22: only Kruthik's bf16 kernel from now on (fp32 kernel = reference numbers only).

- Baseline H200 B=4 (kernel only): fwd 78 us (81 pos), bwd 121 (211 pos), dgamma reduce 58 (1 block on 1 SM).
- ncu + SASS proofs: fwd 4 tokens/warp serialized (bound check `break`), bwd 1 warp/block walks 128 tokens with
  2-byte loads, positions = 2nd chained wait (23% stalls on IMAD.WIDE pos*128), reduce 0.3% DRAM.
  **Real limit = L2 ~70% in fwd AND bwd** (cos/sin 512 B row re-read for each of 24 heads ~200 MB/call; bwd 5.8x
  more load requests from scalar loads). DRAM only ~51%.
- opt v1 (H200, ALL 10 outputs bit-identical at B 1/2/4/8, pos and none): B2 positions prefetch bwd 211->135 us
  (1.56x); B1 depth-1 prefetch no gain (L2-bound, math << latency); R1 reduce unroll32 58->44; F1 hoisting fwd
  78->109 (WORSE: 56 regs vs 40 -> 32 vs 48 warps/SM). SFT total 350->280 (1.25x); pretraining 258->277 (F1).
- Bit-identity trick: fwd rotation must be pinned as `__fmaf_rn(a, own, lo ? -__fmul_rn(b, partner) : +...)`
  (the original's compiled form); left to the compiler the hi half paired differently (85/9.4M outputs 1 ulp off).
- v2 WRITTEN + MEASURED H200 (opt/FusedRMSRoPE_bf16_v2.cu, bench_rope_v2.cu; all variants bit-identical):
  fwd2 (warp per token, cos/sin once, HPW 2..24) 0.53-0.71x SLOWER; bwd2 (block of heads, smem cos/sin stages)
  0.69-0.72x no-pos, 1.2x pos (worse than v1 B2); reduce3 UNROLL128 58->36 us (1.6x). => **"L2 is the limit" REFUTED**
  (L2 70% was a symptom). Both fwd attempts made each warp do more serial work and lost to the original's many small
  warps. Real limit of fwd / no-pos bwd still UNPROVEN -> measure (rivals + ncu side by side), no more theories.
- **GPU 2 ran everything ~10-14% slower than GPU 0** (originals too: fwd 78.2 vs 68.9 us) -> all GPU-2 absolute
  numbers above are inflated; ratios unchanged. Time only on GPU 0 (ours). GPU-0 clean B=4: original fwd 68.9 /
  bwd 110.1 / bwd-pos 194.9 / reduce 48.2; v1 bwd-pos 118.3; reduce3<128> 30.1.
- Best bit-identical combo (GPU 0, B=4): pretraining = original fwd+bwd + reduce3<128> (227.2->209.1 us, 1.09x);
  SFT = original fwd + v1 bwd (B2) + reduce3<128> (313.5->218.9 us, 1.43x). Same ranking at B 1/2/8.
- Rival bench (eager/compile/liger/flashattn/quack): box venv `~/gautam_k/rivalenv` (torch 2.14.1+cu132, triton
  3.8.0) FAILS: Triton needs Python.h -> `apt-get install python3.12-dev` (asked user). Use `python -u ... 2>&1`.
- **10-23 PORT DONE** (laptop SFT lib `Tensor-Implementations/src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu` md5 6ec34cc9):
  pipelined positions bwd `fused_qknorm_rope_bwd_pf_bf16_tm` for **hd 128 only** + one-launch q+k reduce
  `reduce_dgamma_qk_bf16_tm<128>`; fwd + no-pos bwd + hd64-with-pos = Kruthik's code unchanged. The 10-22 benches had NO
  `--use_fast_math` (the lib has it) -> re-proven WITH lib flags: SASS 17/17, GPU port test **36/36 bit-identical**
  (`port_1023/`). hd64+pos with the pf kernel was NOT identical (ptxas pairs FMAs differently) -> kept original there.
  Timing GPU 3 hd128 18/6 T4096 fwd+bwd+reduce: pretraining 1.06-1.14x, SFT 1.40-1.73x (B4 322.7->223.7 us).
  NOT yet: library rebuild on a box + trainer check (step-0 loss identical, step time).
- **Rivals (rivals/rival_bench.py, fixed: Liger bwd rotates dz IN PLACE -> each rival gets its own dz copy):** our fwd
  fastest at every size and most exact (err 1.00 = bf16 rounding); **bwd: torch.compile AND Quack(+FA RoPE) each beat ours
  in 14/16 cases** (kernel us, rival_results_port_4x.json: 18x6 B1 ours 92.3 / compile 51.7 / quack 54.9; 32x8 B4 313.8 /
  243.0 / 236.6); ours fastest only at 18x6 B4 pretraining; at our SFT size (18x6 B4 packed) compile 149.6 vs ours 152.1.
  I first told the user "only compile beats us" -- WRONG, corrected 10-23 EOD. Quack not yet ncu-profiled.
- **ncu cause (rivals/ncu_bwd_*.ncu-repz):** our bwd = 1 warp/block, 128 tokens serial per warp -> occupancy 9% (B1) /
  36% (B4), DRAM 18% / 57%. compile = RoPE-bwd kernel 36,864-65,536 blocks x 128 thr (occ 87-90%) + per-row norm-bwd,
  DRAM 72-79%.
- **OPEN, user 10-23: "next edits need in-detail analysis, start tomorrow morning"** -> 10-24 first: analysis of compile's
  design vs ours, then decide: fewer tokens per warp (dq/dk stay bit-identical; dgamma float sum regrouped = last-bit
  change, still deterministic) vs keep strict bit-identity. No edit before that analysis.
- Box copy of everything: laptop `~/Downloads/box_backup_2026_10_23/rope_opt` (80 files, md5 identical to the 4x box).
- **10-24 ANALYSIS (laptop only, work doc §45.80, rivals/analysis_1024/):** at B4 our MAIN bwd kernel already beats
  compile's kernels (ncu 108 vs 135 us, 298 vs 466 MB); we lose on the dgamma REDUCE (64 vs 27 us; 8 blocks x 32 thr).
  Main kernel = 1 token in flight per warp (56% of stalls on the first bf16 convert; 82 us B1 vs 108 us B4). The 10-22
  depth-1 prefetch AND the port's pf kernel never overlapped: the prefetched values are converted to float right after
  their loads inside the `if next token` branch (SASS). Rivals: compile = RoPE-bwd pointwise + per-row norm-bwd (cp.async
  16 B) + at::reduce tree; Quack = 2,112 persistent CTAs x 128 thr, 16 B loads, double buffer, dw regs + smem, sum(0).
  OPTIONS: (1) real prefetch, raw bf16 kept, unconditional clamped loads, depth 2-3 -> BIT-IDENTICAL; (2) pipelined
  sequential reduce -> BIT-IDENTICAL (floor ~4.7 us at B4); (3) tree dgamma sum = only option needing the rounding
  decision (+~3 us). Estimate 18x6 B4 SFT 152 -> ~85-90 us vs compile 149.6 (NOT measured). Next: code 1+2, GPU test.
- **10-24 CODED (laptop lib md5 07953145, §45.81), NOT GPU-TESTED:** new bwd fused_qknorm_rope_bwd_pp_bf16_tm (hd128, pos + no-pos,
  DEPTH 3 raw-bf16 register slots, per-slot pos_next, pinned intrinsics = original's SASS form; pf kernel deleted) + reduce
  <HD,U=64,STAGES=3> (__ldcg, padded partial buffers, memsetAsync on stream). Laptop SASS: load->use 196-305 instr (orig
  22-27), per-token FP forms IDENTICAL in all 4 variants, 72 regs 0 spills. GPU test ready: port_1024/run_port_test.sh
  (44 cases). User 10-24: GPU not available yet -> run when they give a box.
- **10-24 ~16:00 GPU TEST PASSED (P032 GPU 4, §45.82):** 44/44 bit-identical at DEPTH 2, 3, 4. DEPTH 3 kept: bwd+reduce
  vs original B1 none 99.8->56.0, B1 pos 182.6->90.2, B4 none 168.2->119.6, B4 pos 249.0->131.2 (1.90x), B8 pos 1.95x.
  DEPTH 4 slower at B4 (89/116 regs -> >1 wave). OPEN: why B1 pos (and D2/D4) stay ~89 us while D3 none = 56; rival
  bench rerun; trainer check. Logs: port_1024/port_test_p032_gpu4*.log.
- **10-24 eve (new 1x H200 root@86.38.238.224, §45.83-45.84):** rival bench: ours fastest bwd 11/16 (was 2/16), total
  13/16; 18x6 B4 SFT bwd 128.5 vs quack 150.4 / compile 150.9. Losses = positions B1/B2 + 32x8 B4 SFT + 18x6 pre B1.
  CAUSE PROVEN (ncu + decoded SASS control bits, rivals/analysis_1024/sass_ctrl_bits.py): ptxas puts ALL LDGs on
  scoreboard SB5; positions kernel waits SB5 at every slot (3 drains/round), D3-none only once (lucky, 56 us). FIX
  proposed (not coded): cp.async smem ring + wait_group<DEPTH-1>. Box setup scripts: setup_1024/.
- **10-24 18:51 IST DONE: cp.async version (lib md5 5ef36fa4, §45.86-45.87) 44/44 bit-identical (D3/D4/D6) and FASTEST
  IN 16/16 fwd, 16/16 bwd, 16/16 total vs compile/Quack/Liger/FA/eager.** SFT B4 bwd+reduce vs original 253.3->109.2
  (2.32x); B1 SFT 3.97x. DEPTH 3 default. Noon version kept: -DBF16_BWD_ASYNC=0 -DBF16_RED_ASYNC=0 (snapshot
  port_1024/FusedRMSRoPE_bf16_1024_tested_07953145.cu). NEXT: trainer integration check, then commit (user).
- **10-24 eve PLAN agreed (user):** (1) SFT trainer A/B on 1x H200 86.38.238.224 ~/gautam_k/trainer_ab (old 5338bd7b vs new
  5ef36fa4, one trainer, libs_old/libs_new via LD_LIBRARY_PATH, fullchess init, 320-row val as data; steps ab_1_setup,
  ab_2_check, ab_3_nsys, ab_4_crossover; expect ~20 ms/step = ~1.1%); (2) THEN decide storage together (delete fullchess /
  builds / nsys or keep); (3) THEN pretraining trainer A/B (real pretraining trainer + a data shard) to confirm the
  standalone no-positions gain. RTX 3060: 44/44 but no-pos 0.82-0.96x (tuning is H200-specific).
- **10-24 20:05 TRAINER A/B DONE (§45.90, results trainer_ab_1024/results/):** SFT correct (new-old 0.000203 <= null
  0.000221); RoPE bwd+reduce in trainer SFT 38.03->16.40 ms/step (2.32x), pretraining 355.3->176.0 ms/step (2.02x) = standalone.
  Net step time (single pairs, GPU-bound): SFT -12.9 ms (-0.68%), pretraining -104 ms (-0.49%); the rest is eaten by backward
  GEMMs/attention-bwd running 1-3% slower at the 700 W power cap (clock median 1560->1545). Pending: storage decision
  (box: fullchess 5.2 GB, pt shards 1 GB, builds, nsys ~0.5 GB, rivalenv), git staging, EOD.
Related: [[project_tulu3_sft_run_1022]], [[project_dense_fwd_gap_1017]].
