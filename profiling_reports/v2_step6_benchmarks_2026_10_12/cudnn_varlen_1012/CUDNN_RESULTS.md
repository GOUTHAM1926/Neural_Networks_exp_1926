# cuDNN packed (varlen) attention vs ours, FA3, FA4 — 2026-10-12 evening

H200 box, GPU 7 (UUID GPU-37329774…), free clocks. Shape: 8 rows × 4096 tokens, 18 query heads / 6 KV heads, head dim 128, bf16, causal. I used the same 7 segment layouts as Step 6 (`segments_1012.txt`), and our harness re-exported them identically (`segments_check.txt`). Software: torch 2.14.1+cu130 and cuDNN 9.24.0. Our library is the 10-10 21:18 build (md5 62bf5359); the harness is `tva_step6` (169fad10).

Scripts: `../bench_step6_libs.py` (md5 in `../md5.txt`) and `../run_cudnn_varlen.sh` (check / main / nsys / stats).

## How cuDNN was called

torch's public `varlen_attn` never reaches cuDNN for causal attention. It turns causal into window (-1, 0) (`varlen.py:336`), and any window, or any GQA, goes to FA2 (`varlen.py:62-65`). So I called PyTorch's own op directly, copying its tested call argument for argument (`varlen.py:118-134` forward, `:559-576` backward). I timed three variants:

| name | what it is |
|---|---|
| cudnn_gqa | the 6 KV heads passed straight in. cuDNN sums dK/dV over heads with its own kernel, `fmha_reduce_head_ragged`. **This is the real cuDNN number for a GQA model.** |
| cudnn_exp | K/V copied to 18 heads before timing, so it times cuDNN kernels only. Not usable alone: dK/dV come out per query head. |
| cudnn_expsum | the same with the copy and the dK/dV group sum inside the timed region |

No PyTorch route sends 18/6 heads to the packed op, so `cudnn_gqa` first ran in its own process (`probe.log`).

## Correctness (gate before timing, every layout)

O, dQ, dK and dV are compared with FA3 on the same inputs, using relative max error with a pass limit of 2e-2.

| check | O | dQ | dK | dV | result |
|---|---|---|---|---|---|
| cuDNN (all 3 variants, all 7 layouts) | 0.9–1.6e-3 | 6.3–7.7e-3 | 5.1–5.5e-3 | 6.0–6.5e-3 | PASS 21/21 |
| FA4 vs FA3 (control, normal rounding) | 0.4–0.5e-3 | 1.6–1.9e-3 | 2.7e-3 | 1.5–1.6e-3 | PASS 7/7 |
| wrong head mapping h→h%6 (must fail) | 1.2–1.5 | 1.1–1.3 | 1.3–1.4 | 1.2–1.6 | FAIL 7/7, as required |

Kernel names prove cuDNN ran: `cudnn_generated_fort_native_sdpa_sm90_flash_fprop/bprop…`, plus `…_ragged` helper kernels. The 6-KV-head variant gives the same errors as the 18-head copy.

## Times per API call (median of 50 interleaved rounds, ms) — `tables.txt`

Forward:

| layout | ours A | ours B | FA3 | FA4 | cuDNN (gqa) |
|---|---|---|---|---|---|
| real rows, longest chats | 1.126 | 1.179 | **1.009** | 1.131 | 1.165 |
| real rows, middle | 0.540 | 0.581 | 0.533 | 0.621 | **0.534** |
| real rows, shortest chats | **0.326** | 0.385 | 0.342 | 0.447 | 0.341 |
| all chats 512 | **0.301** | 0.320 | 0.342 | 0.435 | 0.305 |
| all chats 1024 | 0.427 | 0.445 | 0.441 | 0.528 | **0.420** |
| all chats 2048 | 0.670 | 0.687 | **0.668** | 0.769 | 0.698 |
| one 4096 chat per row | 1.169 | 1.187 | **1.087** | 1.211 | 1.227 |

Backward:

| layout | ours A | ours B | FA3 | FA4 | cuDNN (gqa) |
|---|---|---|---|---|---|
| real rows, longest chats | 3.977 | 4.195 | 3.074 | **2.999** | 3.446 |
| real rows, middle | 1.927 | 2.020 | 1.586 | **1.581** | 1.724 |
| real rows, shortest chats | 1.171 | **1.147** | 1.227 | 1.161 | 1.149 |
| all chats 512 | 1.142 | 1.105 | 1.167 | **1.097** | 1.116 |
| all chats 1024 | 1.551 | 1.593 | 1.366 | **1.330** | 1.429 |
| all chats 2048 | 2.448 | 2.558 | **1.959** | 1.961 | 2.207 |
| one 4096 chat per row | 4.312 | 4.541 | **3.182** | 3.199 | 3.711 |

Noise: on 10-12, the same kernel varied by about 4% with its position in the round. Gaps under about 5% are not meaningful. Ours and FA3 today match 10-12 within that noise (ours A forward, longest chats: 1.126 vs 1.119).

- **Forward, ours A vs cuDNN:** within ±5% on every layout. Ours is ahead by 3–5% on the long and shortest chats (real longest, all-2048, all-4096, real shortest); cuDNN is ahead by at most 1.7% (all-1024, real middle). All within noise except the 4–5% leads.
- **Backward, ours A vs cuDNN:** equal on the shortest chats (2%). cuDNN is 9–16% faster on middle and long chats (real middle 1.12×, real longest 1.15×, all-4096 1.16×). This matches the dense result from 10-12 (ours 1.05–1.24× slower than cuDNN).
- **Backward order, middle and long chats:** FA3 ≈ FA4 < cuDNN < FA3 deterministic < ours A < ours B.

## Kernel time per call (nsys, real middle rows, µs) — NVTX ranges, `nsys_libs_{exp,gqa}_nvtx_kern_sum.csv`; ours from `nsys_ours_packed_cuda_gpu_kern_sum.csv`

| | forward | backward (all kernels) | backward main kernel |
|---|---|---|---|
| FA3 | 451 | 1509 | 1122 |
| cuDNN gqa | 457 | 1547 (incl. head sum 115) | 1154 |
| cuDNN exp (18 heads, no sum) | 481 | 1491 | 1213 |
| cuDNN expsum | 801 (325 = K/V copies) | 1757 (255 = PyTorch sums) | 1224 |
| FA3 deterministic | 450 | 1810 | 1423 |
| **ours A** | **519** | **1825** (main 1525 + dQ convert 107 + dK/dV head sum 104 + D rowsum 89; the dQ zero-fill memset is not in this list, so ours is slightly flattered) | 1525 |
| ours B | 562 | 1901 | 1601 |

At kernel level, ours A is 1.14× cuDNN in the forward and 1.18× in the backward. At API level the forward looks equal, because cuDNN's Python call adds about 0.08 ms per call. FA3's kernels ran about 5% slower today than on 10-12 (448 vs 427 µs forward), so compare only within the same capture.

Memory for one forward + backward on the real middle rows (outputs and workspace above the inputs): FA3 905 MiB, cuDNN gqa 1037, cuDNN exp 905, cuDNN expsum 1194.

## Three mistakes in my scripts, caught and fixed (all three failed silently)

1. **Empty first nsys capture.** It used `--capture-range=cudaProfilerApi` and recorded no kernels and no NVTX data (`nsys_libs_packed.stats.out`). Fix: NVTX ranges without a capture range.
2. **cuDNN never timed in the second capture.** The cuDNN guard asked `want("cudnn")`, which is False for the longer prefixes I used there. Fix: check the real contender names; the self-test now covers those prefixes (29 checks), and the runner prints the number of cuDNN lines after each capture. The `main` timing was not affected: it used the prefix `cudnn`, and libs.log has 63 cuDNN lines.
3. **Stale CSVs on the third try.** `nsys stats` keeps an existing CSV, so it showed the second capture's numbers. Fix: `--force-overwrite=true`, then re-exported on CPU (no new GPU run).
