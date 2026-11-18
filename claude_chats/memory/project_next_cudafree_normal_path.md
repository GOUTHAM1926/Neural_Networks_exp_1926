---
name: project-next-cudafree-normal-path
description: "Trainer CUDA call accounting DONE 11-03 (work doc 45.138): every alloc/free/copy/sync has an exact call site (cudatrace.so); empty_cache at steps 0/5, checkpoint pinned allocs, Muon 444 syncs/step; NEXT = design fixes with the user"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-03T13:18:28.462Z
---

Started 10-31 (889 cudaFree / 1,786 cudaMallocAsync / 838 cudaHostAlloc per 10 steps at 128k). ACCOUNTED 11-03
(work doc 45.138, plan Q&A 207+; P032 /localstorage/gautam_k/tilemap_1102/trainer/{nsys,ct}/; scripts nsys_account.py,
nsys_bursts.py, cudatrace.cpp/.so (LD_PRELOAD tracer), cudatrace_report.py, cudatrace_compare.py):
- nsys cannot record call stacks on P032 for us (perf_event_paranoid = 4, not root) -> own LD_PRELOAD tracer
  (works: trainer links libcudart.so.13 dynamically). Kernels launch via `__cudaLaunchKernel` (CUDA 13), not
  cudaLaunchKernel -> hook that for phase markers next time.
- 60 steps, no nsys: cudaMallocAsync 2,007 (29 s CPU) = cold start + refills after empty_cache;
  cudaFree 865 (14 s) = bluscript_sft.cpp:3832 `if (step == 0 || step == 5) empty_cache()` (also
  bluscript_zero_ckpt.cpp:3269) -> steps 1 and 6 take ~6 s each; cudaHostAlloc 822 (9-13 s) = checkpoint staging
  (PinnedCPUAllocator, a new pinned buffer per tensor, every save); Muon: ShardedMuon.cpp:1703 (NaN guard) + 1741-1743
  (`nsratio` DIAGNOSTIC ONLY) = 2 x 222 blocking 1-value D2H copies + syncs per step, ~15 ms CPU wait per step;
  forward/backward make no syncs/allocations.
- Options discussed with the user: guard decided on the GPU, diagnostic on the GPU / once per step / flag; empty_cache
  configurable; reuse pinned staging buffers; logits buffer at max size (switch on). Design + case tests first.
Related: [[project-future-blublas-tile-map-sync]], [[feedback-mpirun-core0-pinning]].
