---
name: project-future-blublas-tile-map-sync
description: "BluBLAS tile-map stall FIXED AND TESTED 11-02 (work doc 45.133-45.134, plan Q&A 196-197): B3 (kernel byte-identical, host-only cache) accepted, B2 (in-kernel order) rejected as slower; sm_80/86/89 last-group fix C tested; NOT pushed — user decides; trainer off/on test still to do"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-02T16:58:38.986Z
---

Analysis 11-02 (45.133): new M -> lm_head dX tall grid misses the 16-slot cache -> cudaMalloc + BLOCKING cudaMemcpy
(waits for all queued work on stream 0) + cudaFree when full; 20 shapes > 16 slots -> every call misses.

Fix tested 11-02 21:30-22:30 (45.134; records profiling_reports/long_context_kernels_2026_10_31/tilemap_fix_2026_11_02/):
- B2 (row order computed in the kernel) REJECTED: down dX +3-4%, 128k tall +0.3-0.8% on 3 H200s. Answer to "won't the
  CUTLASS way add overhead": yes, measurably, for some shapes.
- B3 ACCEPTED: kernel unchanged (nvcc -cubin md5 identical, 186 kernels); tall table depends only on grid_n -> one per
  (GPU, grid_n), power-of-two rows, grown >= 2x rarely, pinned + cudaMemcpyAsync on the GEMM stream + event, never freed,
  64 slots, mutex. Speed = A (59 shapes), precision bit-identical, memory equal, stall 20,059 us -> 22-29 us.
- C (sm_80/86/89 last group = rows left) in BF16 SM89 core, FP16 core, FP32 SM86 core (group 16): normal cases
  bit-identical + same speed on the 3060; broken cases now correct (A left up to 25M elements unwritten).
- NEW FUTURE: split-K dW GEMMs (16k qkv/o dW, 128k o dW) are not bit-deterministic run to run (A vs A).
- Still to do: push (user decides; apply B3 also to FP16 SM90 + SwiGLU templates), then the real trainer with the
  trained-rows switch off/on, A vs B3, several runs.
Related: [[project-future-logits-trained-only]], [[project-next-cudafree-normal-path]].

**11-03 trainer test (work doc 45.137; P032 trainer/ folder, GPUs 0/1, guard on by the user's word):**
- Timing (24 runs, no nsys, steps 10-59): A on-off median -3.3 ms; D2 was ~10 ms/step slower than A even with switch
  off -> nsys showed why: D2 helper `inline` -> linker kept the SM103 copy (TI build link order), fill kernel had no
  sm_90 code -> cudaMalloc + failed launch + cudaFree on EVERY GEMM call (66,033 pairs / 99 micro-batches, 14-18 ms
  GPU idle per micro-batch). Fixed = D2f: helper `static inline` + one-time stderr print on the error path; lib 79d3aeeb.
- nsys old lib switch on: 21 blocking tile-table cudaMemcpy (800-1008 B) in steps 10-59, CPU held 192 ms total, max
  21.7 ms; D2 0. Switch gain = GPU kernel time -1.5 ms/micro-batch (-3 ms/step, ~2%, data 79.8% trained).
- Biggest switch-on lag = TRAINER's own cudaMallocAsync: every switch-on run has a slow step 15 (236-1773 ms vs ~145,
  no nsys; 2.5-3.4 s single call under nsys after micro-batch 30); never with switch off. Plus cudaFree bursts at
  micro-batches 1 and 11 in all runs (10-30 s under nsys). -> [[project-next-cudafree-normal-path]].
- D2f re-test started 16:17 (nsys D2f_sw0 GPU 0, D2f_sw1 GPU 1, then timing rounds 7-10 A vs D2f).
- 16:53 PINNED nsys (own cores, run_pinned.sh): switch off A == D2f (busy, idle, all call counts; only -18,300
  memsets). Switch on: D2f removes 21 blocking copies (280 ms CPU) and ~0.8 ms GPU idle per micro-batch.
- Switch's own compute saving (nsys_lmhead.py, per kernel): lm_head + loss 4.45 -> 3.94 ms per micro-batch (-0.51 ms,
  -11%) = 0.7% of GPU time (lm_head+loss is ~6% of the 135M step; 80.6% rows trained). The earlier "2%" was whole-
  micro-batch noise. dX GEMM shrinks least (93%): odd row count -> other config; idea: pad trained rows to a tile multiple.
  Loss kernels need no change; train loss off == on within noise.
- User's plan (16:51): design the switch like the attention vj2/vj1d flag (default off), test binaries + training script
  on/off case by case, comment everything, then push. Next task: TI's own memory calls (18 TI files call
  cudaMallocAsync directly; step-15 spike; 227 cudaStreamSynchronize per micro-batch).
- **PUSHED 2026-11-03 ~20:30 by the user, verified on the remote file by file:** BluBLAS new branch
  dist_side_gautam_lottn 297f372 (12 files; master untouched), TI dist_side_gautam_lottn d067e89..d18f820 (5 RowGather
  files + BluBLAS pointer 297f372), BluTrain dist_side_gautam_lottn 91b6b54..7184fe1 (3 switch files + TI pointer
  d18f820). Commit messages: profiling_reports/long_context_kernels_2026_10_31/tilemap_fix_2026_11_02/push_1103/.
  Not done before the push (user's decision): library suite re-run on the final build; switch edge-case tests.
