# SFT loss kernel benchmark on H200: forward + backward, v1 padding

This document describes, end to end, how I measured our SFT loss kernels (sparse cross-entropy with label masking) on one H200: what I wanted to learn, the program, the data, every setting, every number, and how to read each one. Section 16 lists what you need to run your kernels under exactly the same conditions, so our numbers can be compared directly.

---

## 0. Summary

- **What I measured:** only the loss kernels, run alone (no model, no training run), forward and backward, at our SFT micro-batch: **8 rows × 4096 positions = 32,768 positions, vocabulary 50,304, bf16 logits**.
- **Data:** the label file of our SFT stage-0 dataset (`labels.bin`, 5,000 chats padded to 4096). The logits are random numbers made by a fixed formula.
- **Main result:** a typical batch (23.1% of positions trained) costs **0.276 ms forward + 1.600 ms backward = 1.878 ms** for the pair. One optimizer step (8 such batches on 1 GPU) costs about **15 ms** of loss time.
- **Masking saves about 3×** compared with computing every position (1.676 ms vs 5.376 ms on the same 8 rows).
- **All correctness checks passed** in all 6 cases.

---

## 1. Why this test

The SFT trainer pads every chat to 4096 positions (v1 padding). Only the assistant's answer tokens are trained. Prompt tokens and pad positions get the label 65535, which means "skip this position". I changed the loss kernels so that:

- the forward skips skipped positions completely;
- the backward writes an exact zero gradient for them;
- the loss is divided by the number of trained positions (not by all positions).

I wanted:
1. the time of the modified forward and backward at the real SFT shape;
2. how the time depends on how many positions are trained;
3. how much the masking and the saved statistics save;
4. how close the kernels get to the H200 memory bandwidth limit;
5. proof that the results are correct, so the timings are not from a kernel that skipped work it should have done.

---

## 2. What is measured

### 2.1 Code version

| Item | Value |
|---|---|
| BluTrain branch | `dist_side_merge`, commit `909ab75` |
| Tensor-Implementations (TI) commit | `d96b991` (2026-09-28) |
| `src/Kernels/cuda/loss/LossKernels.cu` md5 | `a634d8c48d93…` (same on laptop and box) |
| `src/autograd/operations/LossOps.cpp` md5 | `8de11149b506…` (same on laptop and box) |
| `src/autograd/backward/LossBackward.cpp` md5 | `5dda0ef922f5…` (same on laptop and box) |
| `include/ops/helpers/LossKernels.h` md5 | `299f02289800…` (laptop) |
| `lib/libtensor.so` on the box | built 2026-10-06 12:08, after the last change to `LossKernels.cu` (11:51) |

### 2.2 The functions the benchmark calls

These are the same library functions the trainer reaches through `autograd::sparse_cross_entropy_loss` on H200. Training micro-batches (gradient on) use the "save_stats" forward and the "with stats" backward. Validation (gradient off) uses the forward without stats.

| Short name | Library function (namespace `OwnTensor::cuda`) | Used by |
|---|---|---|
| **F** | `sparse_cross_entropy_forward_cuda_impl_vec_save_stats<bfloat16_t, uint16_t>` | trainer, every training micro-batch |
| **B** | `sparse_ce_backward_cuda_vec_with_stats<bfloat16_t, uint16_t>` | trainer, every training micro-batch |
| **Fv** | `sparse_cross_entropy_forward_cuda_impl_vec<bfloat16_t, uint16_t>` | validation |
| **Bo** | `sparse_cross_entropy_backward_cuda<bfloat16_t, uint16_t>` | reference only (a backward that does not use saved stats) |

### 2.3 The GPU kernels each function launches (H200 path)

| Function | Kernel | Grid | Block | Shared memory |
|---|---|---|---|---|
| F | `sparse_ce_forward_kernel_vec_save_stats` | 32,768 (one block per position) | 512 | 12,288 B |
| F, Fv | `sum_reduction_kernel` pass 1 | 128 | 256 | 1,024 B |
| F, Fv | `sum_reduction_kernel` pass 2 (divides by the denominator) | 1 | 256 | 1,024 B |
| B | `sparseCENormalize_from_stats` | 32,768 | 256 | 0 |
| Fv | `sparse_ce_forward_kernel_vec` | 32,768 | 512 | 12,288 B |
| Bo | `sparseCEReduce_kernel_optimized` | (2, 32,768) | 256 | 6,144 B |
| Bo | `sparseCENormalize_kernel_optimized` | 32,768 | 256 | 4,096 B |

- F and Fv also take two small scratch buffers per call (per-position losses and reduction partials) from the library's caching allocator. This is part of the timed call.
- Bo launches its two kernels on the default stream, not on the stream it is given. The benchmark's stream is a normal blocking stream, so the order and the timing are still correct. Bo is not used by the trainer.

### 2.4 What the kernels compute

For each position r with target t_r and logits z_r,0 … z_r,V−1:

- **Trained position** (t_r ≠ 65535):
  - row max: m_r = max_c z_r,c
  - row sum: s_r = Σ_c exp(z_r,c − m_r)
  - row loss: ℓ_r = log(s_r) + m_r − z_r,t_r
  - gradient: g_r,c = (exp(z_r,c − m_r) / s_r − [c = t_r]) × grad_output / denom
- **Skipped position** (t_r = 65535):
  - the forward does no work; row loss = 0, saved max = 0, saved sum = 1 (the sum must be 1, not 0: the backward computes 1/s);
  - the backward writes g_r,c = 0 for all 50,304 columns.
- **Loss** = (Σ_r ℓ_r) / denom, where denom = the number of trained positions.
  - In the trainer, denom is the trained count of the whole optimizer step (all micro-batches, all GPUs).
  - In this benchmark it is the trained count of the batch being timed.
- F saves m_r and s_r for every position so that B does not have to re-read the logits to recompute them. Bo recomputes them.
- Math precision: logits are read as bf16, all arithmetic is fp32, the loss and the saved statistics are fp32, the gradient is written as bf16. The library is compiled with `--use_fast_math -O3`.

---

## 3. Test setup

| Item | Value |
|---|---|
| Date of the run | 2026-10-09 |
| GPU | NVIDIA H200 (SXM), physical index 5 of 8 on node `innmi1srh2-p032`; the program sees it as device 0 through `CUDA_VISIBLE_DEVICES=5` |
| SMs / memory | 132 SMs, 139.8 GiB total, 139.3 GiB free at start |
| Driver | 595.71.05 (nvidia-smi header shows "CUDA Version: 13.2") |
| CUDA toolkit used to build | `/usr/local/cuda` → `/usr/local/cuda-13.4` |
| Build target | `sm_90`, flags from the TI Makefile: `-std=c++17 --use_fast_math -O3 --expt-relaxed-constexpr` |
| Clocks | not locked (production default). Just before the run: SM 1980 MHz, memory 3201 MHz, 133.7 W idle |
| Other work on the GPU | none: 4 MiB used, 0% utilization and no processes, both before and after the run |
| Benchmark source | `bench_sparse_ce_sft.cu`, md5 `f8a5bf8c98b2fed9477fc7dc028c215c` |
| Benchmark binary on the box | `~/gautam_k/loss_bench/bench_sparse_ce_sft`, md5 `a405d7c1abe6…` |
| Command | `./bench_sparse_ce_sft <labels.bin> --iters 50 --warmup 5` |

---

## 4. Inputs

### 4.1 Shape and types

| Buffer | Shape | Type | Size |
|---|---|---|---|
| Logits | [32,768, 50,304], row-major, contiguous | bf16 | 3,296,722,944 B = 3.30 GB |
| Targets (labels) | [32,768] | uint16; 65535 = skip | 64 KiB |
| Gradient of the logits (output of B / Bo) | [32,768, 50,304] | bf16 | 3.30 GB |
| Saved max, saved sum (output of F) | [32,768] each | fp32 | 128 KiB each |
| Loss (output of F / Fv) | 1 | fp32 | 4 B |
| grad_output (input of B / Bo) | 1 | fp32, value 1.0 | 4 B |
| denom (input of all four) | 1 | fp32, the trained count of the batch (1 for the all-masked case) | 4 B |

- 32,768 = B × T = 8 rows × 4096 positions: one SFT micro-batch.
- 50,304 = vocabulary rows of our model (GPT-2's 50,257, padded).
- Each logits row is 100,608 bytes, so every row starts on a 16-byte boundary. The kernels need this for their 16-byte vector loads.

### 4.2 Labels: real data

- **File:** `labels.bin` from our SFT stage-0 training set.
  - Box: `~/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin`
  - Laptop: `~/Downloads/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin`
- **Layout:** uint16, [5,000 rows, 4,096], row-major, little-endian. Value 65535 = skip. Every other value is a token id below 50,304 (the benchmark checks this when it loads the file).
- **Rule used to build it:** for each chat, `inputs = ids[:-1]` and `labels = ids[1:]`, then the labels of prompt tokens and pad positions are set to 65535.
- **Only the labels are used by the benchmark.** The loss kernel never reads the input token ids.

The other files in the same folder:

| File | Layout | md5 |
|---|---|---|
| `inputs.bin` | uint16 [5,000, 4,096], token ids fed to the model | `fdb9ef90957088cc9beb86bf0b1a9b65` |
| `labels.bin` | uint16 [5,000, 4,096], 65535 = skip | `eec5e38b1fedeafe5894be6f1a70f96f` |
| `valid_counts.bin` | uint32 [5,000], the number of trained positions in each row (I checked: equal to the count of labels ≠ 65535, all 5,000 rows) | `d57b5ac50e02c026570401eba5209995` |
| `meta.json` | how the data was built | `8912494bae5e89405e460be04e790ca3` |

How the data was built (from `meta.json`):
- source: `HuggingFaceH4/ultrachat_200k`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`, split `train_sft`;
- 5,000 random chats, seed 1234; 10 chats longer than 4,096 tokens were dropped;
- tokenizer: tiktoken GPT-2; chat template `tulu-plain-v1`; pad id 50256.

### 4.3 What the 4096 positions of a row contain

Measured over all 5,000 rows:

| Kind of position | Label | Share of all positions |
|---|---|---|
| Assistant answer token (**trained**) | the next token id | **23.30%** |
| Prompt token (system / user) | 65535 | 6.92% |
| Pad after the chat ends (to fill 4096) | 65535 | 69.78% |

The mean chat length is 1,237.8 tokens out of 4,096. Most skipped positions are pads.

### 4.4 Batches

- The benchmark cuts the file into 625 batches of 8 consecutive rows, in file order: batch b = rows 8b … 8b+7.
- Share of trained positions per batch, over the 625 batches:

| Min | Median | Mean | Max |
|---|---|---|---|
| 8.91% | 23.12% | 23.30% | 38.85% |

### 4.5 Logits: random, with a fixed formula

The loss kernel does the same work for any logit values: one pass over each trained row, one exp per element. So the logits are generated on the GPU with a fixed formula. For flat index i = row × 50,304 + column (0 … 1,648,361,471):

```
h = (uint32)(i * 2654435761) XOR 1234          (32-bit unsigned arithmetic)
h ^= h >> 16;  h *= 0x7feb352d;  h ^= h >> 15;  h *= 0x846ca68b;  h ^= h >> 16
u = (h >> 8) / 16777216.0                      (uniform in [0, 1))
logit[i] = bf16_round_to_nearest_even((u - 0.5) * 8)   (uniform in [-4, 4))
```

With this formula the expected loss is ln(50,304) + ln((e⁴ − e⁻⁴) / 8) = 10.826 + 1.920 = **12.746**. The measured losses (12.740 – 12.750) match it, which is one of the correctness checks.

---

## 5. The 6 test cases

Each case is one batch of 32,768 positions. Only the labels change from case to case; the logits are the same.

| Case | How it is chosen | Batch / rows | Trained positions | Share | ignore_index |
|---|---|---|---|---|---|
| **real_first** | the first 8 rows of the file | batch 0, rows 0–7 | 6,112 | 18.65% | 65535 |
| **real_min** | the batch with the fewest trained positions | batch 287, rows 2296–2303 | 2,920 | 8.91% | 65535 |
| **real_median** | the typical batch: sort the 625 batches by trained count and take position 312 | batch 458, rows 3664–3671 | 7,577 | 23.12% | 65535 |
| **real_max** | the batch with the most trained positions | batch 415, rows 3320–3327 | 12,729 | 38.85% | 65535 |
| **nomask** | rows 0–7 with masking turned off: skipped positions get target (i × 7919) mod 50,304, where i = position index 0…32,767, and ignore_index = −1 | batch 0 | 32,768 | 100% | −1 |
| **all_masked** | every target = 65535; denom set to 1 to avoid 0/0 | — | 0 | 0% | 65535 |

- **real_min / real_median / real_max** cover the real range a training run sees.
- **nomask** is the pretraining situation (every position trained). Compared with real_first (same rows), it shows what masking saves.
- **all_masked** is the floor: the forward does nothing and the backward only writes zeros.

---

## 6. The 5 timed operations

| Name | What is timed | Meaning |
|---|---|---|
| **F** | one call of F | the training forward alone |
| **B** | one call of B (it uses the statistics saved by the previous F) | the training backward alone |
| **F+B** | F then B, back to back, timed as one unit | **what one training micro-batch costs for the loss** |
| **Fv** | one call of Fv | the validation forward |
| **Fv+Bo** | Fv then Bo, timed as one unit | the same pair without saved statistics: the backward re-reads the trained rows to recompute max and sum. Reference only |

---

## 7. How the timing is done

For each case, in this order: F, B, F+B, Fv, Fv+Bo. For each operation:

1. 5 warm-up calls, not timed. These remove first-call effects such as the allocator filling its cache.
2. 50 timed calls. Each call gets its own pair of CUDA events, recorded on the same stream immediately before and after the call.
3. One synchronize at the end, then the 50 elapsed times are read.
4. Reported: **median** (the main number), min and mean, in milliseconds.

Other settings:
- All buffers are allocated once before the cases and reused.
- No cache flush between calls: the logits (3.30 GB) are far larger than the L2 cache, so the next call cannot find them in cache.
- One process; nothing else on the GPU.
- The event times include the launch gaps between the kernels inside one call (F launches 3 kernels). This is why F has a fixed cost even when nothing is trained (all_masked: 0.036 ms).

---

## 8. Bytes and "% of peak"

The program computes how many bytes each operation **must** move between the GPU and its memory. It then divides by the median time.

Let L = 32,768 × 50,304 × 2 B = 3.30 GB (the whole logits tensor, or the whole gradient), and Ltr = trained positions × 50,304 × 2 B (only the trained rows).

| Operation | Bytes counted |
|---|---|
| F | Ltr (read trained rows) + 18 B per position (label, 3 per-row outputs, reduction read) |
| B | Ltr (read trained rows) + L (write the whole gradient, zeros included) + 10 B per position |
| Fv | Ltr + 10 B per position |
| Bo | 2 × Ltr (trained rows read twice) + L + 10 B per position |
| F+B, Fv+Bo | the sum of their two parts |

- **GB/s** = bytes ÷ median time (1 GB = 10⁹ B).
- **% of peak** = GB/s ÷ 4,800 GB/s. 4.8 TB/s is the H200 SXM HBM3e bandwidth from NVIDIA's specification.
- These are **effective** numbers, from the bytes the work requires. They are not measured DRAM traffic. A kernel that moves extra bytes, or loses time to latency or compute, shows up as a lower %.

---

## 9. Correctness checks

These run after the timing for each case, so they cannot change the timing. A case shows **PASS** only if all of them hold.

| Check | What it proves |
|---|---|
| **loss** is finite, > 0 (exactly 0 for all_masked) | the forward produced a real number |
| **rows computed = expected**: the number of positions whose saved sum ≠ 1 equals the trained count | the forward really computed every trained position and skipped every skipped one. A trained row always has sum > 1; a skipped row has sum exactly 1 |
| **masked row … grad bit-zero**: the first skipped position's 50,304 gradient values are all exactly zero | the backward really zeroes skipped positions |
| **trained row …: g[target] < 0, sum ≈ 0**: for the first trained position, the gradient at its target is negative and the 50,304 values sum to about 0 | the gradient has the softmax − one-hot form |
| **no-stats path: loss, max\|dg\|**: Fv + Bo give the same loss (relative difference ≤ 1e-4) and the same gradient row (max difference ≤ 1% of the target gradient) | two independent code paths agree. Measured difference: **0** (identical) in every case |

---

## 10. Results

H200, B=8, T=4096, V=50,304, bf16 logits, median of 50 calls. Raw output: `bench_run1.log`.

### 10.1 Summary table

| Case | Trained | F ms | F % peak | B ms | B % peak | **F+B ms** | Fv ms | Fv+Bo ms | Check |
|---|---|---|---|---|---|---|---|---|---|
| real_first | 18.65% | 0.2340 | 54.8% | 1.4427 | 56.5% | **1.6762** | 0.2312 | 1.9935 | PASS |
| real_min | 8.91% | 0.1321 | 46.4% | 1.0954 | 68.3% | **1.2286** | 0.1304 | 1.3788 | PASS |
| real_median | 23.12% | 0.2763 | 57.5% | 1.5999 | 52.9% | **1.8778** | 0.2750 | 2.2577 | PASS |
| real_max | 38.85% | 0.4266 | 62.6% | 2.1617 | 44.1% | **2.5892** | 0.4260 | 3.2052 | PASS |
| nomask | 100% | 1.0245 | 67.1% | 4.3526 | 31.6% | **5.3762** | 1.0226 | 6.8840 | PASS |
| all_masked | 0% | 0.0355 | 0.3% | 0.8233 | 83.4% | **0.8667** | 0.0335 | 0.9167 | PASS |

Per optimizer step on 1 GPU (64 rows = 8 micro-batches), median batch: 8 × 1.8778 = **15.02 ms**.

### 10.2 Full numbers per case

| Case | Operation | Median ms | Min ms | Mean ms | GB moved | GB/s | % of 4.8 TB/s |
|---|---|---|---|---|---|---|---|
| real_first | F | 0.2340 | 0.2302 | 0.2338 | 0.616 | 2631 | 54.8 |
| | B | 1.4427 | 1.4376 | 1.4429 | 3.912 | 2712 | 56.5 |
| | F+B | 1.6762 | 1.6696 | 1.6762 | 4.527 | 2701 | 56.3 |
| | Fv | 0.2312 | 0.2300 | 0.2313 | 0.615 | 2661 | 55.4 |
| | Fv+Bo | 1.9935 | 1.9853 | 1.9934 | 5.142 | 2579 | 53.7 |
| real_min | F | 0.1321 | 0.1300 | 0.1320 | 0.294 | 2229 | 46.4 |
| | B | 1.0954 | 1.0897 | 1.0949 | 3.591 | 3278 | 68.3 |
| | F+B | 1.2286 | 1.2226 | 1.2292 | 3.885 | 3162 | 65.9 |
| | Fv | 0.1304 | 0.1281 | 0.1302 | 0.294 | 2256 | 47.0 |
| | Fv+Bo | 1.3788 | 1.3750 | 1.3788 | 4.179 | 3031 | 63.1 |
| real_median | F | 0.2763 | 0.2719 | 0.2762 | 0.763 | 2761 | 57.5 |
| | B | 1.5999 | 1.5951 | 1.6001 | 4.059 | 2537 | 52.9 |
| | F+B | 1.8778 | 1.8718 | 1.8775 | 4.822 | 2568 | 53.5 |
| | Fv | 0.2750 | 0.2699 | 0.2750 | 0.763 | 2773 | 57.8 |
| | Fv+Bo | 2.2577 | 2.2520 | 2.2580 | 5.584 | 2473 | 51.5 |
| real_max | F | 0.4266 | 0.4249 | 0.4266 | 1.281 | 3003 | 62.6 |
| | B | 2.1617 | 2.1533 | 2.1614 | 4.578 | 2118 | 44.1 |
| | F+B | 2.5892 | 2.5836 | 2.5894 | 5.859 | 2263 | 47.1 |
| | Fv | 0.4260 | 0.4241 | 0.4260 | 1.281 | 3007 | 62.6 |
| | Fv+Bo | 3.2052 | 3.1994 | 3.2050 | 7.139 | 2227 | 46.4 |
| nomask | F | 1.0245 | 1.0227 | 1.0245 | 3.297 | 3219 | 67.1 |
| | B | 4.3526 | 4.3432 | 4.3537 | 6.594 | 1515 | 31.6 |
| | F+B | 5.3762 | 5.3696 | 5.3762 | 9.891 | 1840 | 38.3 |
| | Fv | 1.0226 | 1.0207 | 1.0225 | 3.297 | 3224 | 67.2 |
| | Fv+Bo | 6.8840 | 6.8718 | 6.8836 | 13.188 | 1916 | 39.9 |
| all_masked | F | 0.0355 | 0.0348 | 0.0354 | 0.001 | 17 | 0.3 |
| | B | 0.8233 | 0.8224 | 0.8233 | 3.297 | 4005 | 83.4 |
| | F+B | 0.8667 | 0.8644 | 0.8669 | 3.298 | 3805 | 79.3 |
| | Fv | 0.0335 | 0.0330 | 0.0335 | 0.000 | 10 | 0.2 |
| | Fv+Bo | 0.9167 | 0.9145 | 0.9168 | 3.297 | 3597 | 74.9 |

Min and mean are within 2% of the median everywhere, so the run-to-run noise is small.

### 10.3 Correctness results

| Case | Loss (F) | Rows computed / expected | Skipped row grad = 0 | Trained row: g[target], sum | Fv+Bo: loss, max\|dg\| | Result |
|---|---|---|---|---|---|---|
| real_first | 12.742288 | 6,112 / 6,112 | row 0: yes | row 95: −1.640e-04, −4.3e-07 | 12.742288, 0 | PASS |
| real_min | 12.742343 | 2,920 / 2,920 | row 0: yes | row 406: −3.433e-04, −7.8e-07 | 12.742343, 0 | PASS |
| real_median | 12.740235 | 7,577 / 7,577 | row 0: yes | row 69: −1.316e-04, 3.6e-07 | 12.740235, 0 | PASS |
| real_max | 12.745141 | 12,729 / 12,729 | row 0: yes | row 103: −7.868e-05, −1.1e-07 | 12.745141, 0 | PASS |
| nomask | 12.750284 | 32,768 / 32,768 | (no skipped rows) | row 0: −3.052e-05, 3.3e-09 | 12.750284, 0 | PASS |
| all_masked | 0.000000 | 0 / 0 | row 0: yes | (no trained rows) | 0.000000, 0 | PASS |

g[target] ≈ −1/denom, as expected: for example, real_median 1/7,577 = 1.32e-04.

---

## 11. What each word in the output means

**Header lines**
- `device`: GPU name, number of SMs, total and free memory when the program started.
- `shape`: B, T, N = B × T positions, V = vocabulary, the data types, and the skip label.
- `labels`: the label file, its number of rows, the number of 8-row batches, and the mean trained share over the whole file.
- `timing`: warm-up calls and timed calls per operation.

**Per case**
- `== <case> <description>`: the case name and which rows it uses.
- `trained rows: X of 32768 (P%)`: positions with a real target (not skipped). P% is the **trained share**.
- `ignore_index`: the label value the kernels treat as "skip" (65535), or −1 for masking off.
- `F / B / F+B / Fv / Fv+Bo`: the operation (section 6).
- `median / min / mean ... ms`: over the 50 timed calls.
- `X GB`: bytes the operation must move (section 8).
- `GB/s`: those bytes ÷ the median time.
- `% of 4.8 TB/s`: how close to the H200 memory bandwidth limit.

**Checks line** (section 9)
- `loss`: the loss value from F.
- `rows computed N (expect M)`: positions the forward really computed vs the trained count.
- `masked row R grad bit-zero`: the first skipped position, and whether its gradient is exactly zero.
- `trained row R: g[target], sum`: the first trained position's gradient at its target, and the sum of its whole gradient row.
- `no-stats path: loss, max|dg|`: the loss from Fv, and the largest difference between the Bo and B gradients on that trained row.
- `ok / BAD / n/a`: each check passed, failed, or does not apply (no such row in this case).
- `PASS / FAIL`: all checks of the case.

**Summary**
- `F ms`, `B ms`, `F+B ms`, `Fv ms`, `Fv+Bo ms`: median times.
- `%peak`: as above.
- `check`: PASS / FAIL.
- `per optimizer step`: 8 × the median F+B of the median batch. On 1 GPU an optimizer step is 64 rows = 8 micro-batches of 8 rows.
- `ALL CHECKS PASS (0 cases failed)` and exit code 0: every case passed. The program exits with code 3 if any case fails.

---

## 12. What the results show

1. **A typical SFT micro-batch costs 1.88 ms of loss forward + backward.** The real range is 1.23 ms (fewest trained positions) to 2.59 ms (most).
2. **One optimizer step costs about 15 ms of loss time on 1 GPU** (8 × 1.878 ms). The 2026-10-06 one-GPU SFT run took about 5.3 s per step, so the loss kernels are about 0.3% of a step.
3. **Masking saves about 3×.** On the same rows 0–7:
   - forward 1.025 → 0.234 ms (4.4× less);
   - backward 4.353 → 1.443 ms (3.0× less);
   - pair 5.376 → 1.676 ms (3.2× less).
4. **Saving the statistics in the forward saves 17–22% of the pair:**
   - real_median: 1.878 vs 2.258 ms;
   - nomask: 5.376 vs 6.884 ms.
   - The saving costs almost nothing in the forward: F vs Fv differs by 0.6–2.8 µs.
5. **The backward is 85% of the pair** (real_median: 1.600 of 1.878 ms). Part of it is writing zeros:
   - all_masked shows that writing the whole 3.30 GB gradient as zeros takes 0.823 ms, at 83% of peak.
   - *Estimate (by subtraction, not measured directly):* in real_median, 76.9% of the positions are skipped, so about 0.77 × 0.823 ≈ 0.63 ms is zero-writing. The remaining ≈ 0.97 ms covers the trained rows, which move ≈ 1.53 GB, about 33% of peak.
   - nomask's backward (every row trained) is 31.6% of peak. So **the backward on trained rows is the slowest part of the loss.** The reason is not measured yet (an Nsight Compute profile would show it).
6. **The forward runs at 46–67% of peak.** It has a fixed cost of about 0.035 ms per call (all_masked: launches and the two reductions). The share of that fixed cost is larger when few positions are trained, which is why real_min is at 46% and nomask at 67%.
7. **The time grows in a straight line with the trained share.** From the 0% and 100% cases: F+B ≈ 0.87 ms + 4.51 ms × trained share. This predicts the three real batches within 0.04 ms (real_min 1.27 vs 1.23 measured, real_median 1.91 vs 1.88, real_max 2.62 vs 2.59).

---

## 13. How to run it

On a machine with the BluTrain repo and a built `libtensor.so` (TI commit `d96b991`), with `nvcc` from `/usr/local/cuda/bin` on PATH.

**1. Put the source in the TI test folder**

```bash
cp bench_sparse_ce_sft.cu <BluTrain>/Tensor-Implementations/Tests/kernel_tests/
```

**2. Build** (CPU only). This takes the exact nvcc line the TI Makefile would use, without rebuilding the library.

```bash
export PATH=/usr/local/cuda/bin:$PATH
cd <BluTrain>/Tensor-Implementations
mkdir -p ~/loss_bench
make -s -n run-cuda FILE=Tests/kernel_tests/bench_sparse_ce_sft.cu \
  | sed -e ':a' -e '/\\$/N; s/\\\n//; ta' > ~/loss_bench/make_n.txt
grep -- "-o run_cuda_runner" ~/loss_bench/make_n.txt | head -1 \
  | sed "s#-o run_cuda_runner#-o $HOME/loss_bench/bench_sparse_ce_sft#" > ~/loss_bench/build.sh
bash ~/loss_bench/build.sh
```

**3. Run** (one GPU, about 10 s, about 7 GB of GPU memory)

```bash
TI=<BluTrain>/Tensor-Implementations
export LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=<gpu index>
~/loss_bench/bench_sparse_ce_sft <path>/labels.bin --iters 50 --warmup 5 2>&1 | tee bench_run1.log
```

Options:
- `--case <name>`: run one case only (real_first, real_min, real_median, real_max, nomask, all_masked).
- `--iters N`: timed calls per operation (default 50).
- `--warmup N`: warm-up calls (default 5).

**4. nsys** (per-kernel times, one report per case)

```bash
NSYS=/opt/nvidia/nsight-systems/2026.3.2/bin/nsys
for c in real_first real_min real_median real_max nomask all_masked; do
  $NSYS profile -t cuda -o nsys/$c --force-overwrite true \
      ~/loss_bench/bench_sparse_ce_sft <path>/labels.bin --case $c --iters 20 --warmup 3 > nsys/$c.stdout 2>&1
  $NSYS stats -r cuda_gpu_kern_sum -f csv -o nsys/$c nsys/$c.nsys-rep
done
```

Do not pipe nsys into a command that exits early (for example `| grep -m1`): nsys is killed before it writes the report. Kernel names contain commas inside template arguments, so read the CSV with a real CSV reader.

---

## 14. Status of the nsys measurement

The CUDA-event results above are complete. The nsys per-kernel run (section 13, step 4) is prepared but **not run yet**. When it runs, it adds the time of each kernel inside F, B, Fv and Bo (for example, how much of F is the main kernel vs the two reductions).

---

## 15. What this test does not cover

- The loss inside the full trainer: allocations, dtype casts and the other kernels around the loss.
- The logits matmul (lm_head) forward and backward, which produce the logits and consume their gradient.
- v2 packing, where most of the pad positions disappear (the trained share per batch would be about 77%).
- Other GPUs (B300 uses a different forward kernel and is not tested here).
- Nsight Compute (why the trained-row backward reaches only about a third of peak).

---

## 16. How to benchmark your kernels under the same conditions

To compare your kernels with these numbers, keep everything below the same.

1. **GPU:** H200 SXM, nothing else running on it (check `nvidia-smi` before and after), clocks not locked.
2. **Shape:** 32,768 positions × 50,304, logits bf16 row-major contiguous, targets uint16 with 65535 = skip.
3. **Labels:** the same `labels.bin` (md5 `eec5e38b1fedeafe5894be6f1a70f96f`), cut into batches of 8 consecutive rows in file order. Use the same 4 batches:
   - batch 0 (rows 0–7);
   - batch 287 (rows 2296–2303);
   - batch 458 (rows 3664–3671);
   - batch 415 (rows 3320–3327).
   - Plus nomask (rows 0–7, skipped positions replaced by (i × 7919) mod 50,304, masking off) and all_masked (all 65535, denom = 1).
4. **Logits:** the formula in section 4.5. Any values give the same timing, but this formula lets you compare loss values to ours exactly.
5. **Outputs your kernels must produce**, so that the work is the same:
   - the loss = (sum of trained-position losses) / denom, fp32;
   - the full bf16 gradient [32,768 × 50,304], **including the zeros at skipped positions**, because the next operation in the trainer (the logits matmul backward) reads the whole gradient;
   - if your forward saves statistics for the backward, report F and B like ours; if it does not, report it like Fv and Bo.
   - If your kernel does not write zeros for skipped positions, report that separately: it is less work than ours, so the times are not directly comparable.
6. **Timing:** 5 warm-up calls, then 50 calls, each with its own CUDA event pair on one stream; report the median (plus min and mean). Report at least F, B and F+B (the pair back to back).
7. **Correctness:** your losses should match ours to about 1e-4 relative (section 10.3: 12.742288, 12.742343, 12.740235, 12.745141, 12.750284, 0). Skipped positions must have exactly zero gradient, and every trained position must be computed.
8. **Bytes and % of peak:** use the formulas of section 8, with 4.8 TB/s as the peak.
9. **Report:** the same table as section 10.1.

---

## 17. Files

| File | What it is | md5 |
|---|---|---|
| `bench_sparse_ce_sft.cu` | the benchmark source | `f8a5bf8c98b2fed9477fc7dc028c215c` |
| `bench_run1.log` | the raw output of the run in section 10 | — |
| `labels.bin` | labels, uint16 [5,000 × 4,096], 65535 = skip | `eec5e38b1fedeafe5894be6f1a70f96f` |
| `inputs.bin` | input token ids, uint16 [5,000 × 4,096] (not needed by the loss benchmark) | `fdb9ef90957088cc9beb86bf0b1a9b65` |
| `valid_counts.bin` | trained positions per row, uint32 [5,000] | `d57b5ac50e02c026570401eba5209995` |
| `meta.json` | how the data was built | `8912494bae5e89405e460be04e790ca3` |
