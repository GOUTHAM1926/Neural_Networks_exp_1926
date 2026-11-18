# Packed-row attention on H200: my experiments and fixes, set next to the Nova forward study of 13/11/2026

This note is for the author of `varlen-forward-study-nov13.md`. It lists what I already measured and changed in the
hand-written BluTrain attention kernels on the same questions, so you can reuse the results and skip the dead ends.

- **My kernels:** `GQA_fwd_sm90.cu` (forward) and `GQA_bwd_vj2_sm90.cu` (backward), in `Tensor-Implementations`, branch
  `dist_side_gautam_lottn`, folder `src/Kernels/cuda/attention/arch/`.
- **Full record of every change, with timings:** `attention_kernel_optimizations_h200_2026_10_10_to_2026_10_21.md`
  (677 lines, same folder as this note). This note is the short version, ordered by your study's sections.
- **Source of every number here:** my SFT work log `sft_work_documentation.md`, sections 40 to 45.125.

One warning before the numbers: my runs and yours used different boxes, dates, head counts and kernels. Do not compare
a microsecond of mine with a microsecond of yours. Compare only ratios against FA3 measured in the same session.

**Update, 13/11/2026 evening.** After writing sections 1 to 9, I ran your short-chat case with my kernel on P032 the
same day. Section 10 has those measurements. They changed four sentences of this note; section 11 lists each one.

**Second update, 13/11/2026 night.** I then measured length by length (section 12). Two things you should read there:
for chats of about 40 tokens and less FA3's forward IS faster than mine, so your cause 1 is real at those lengths; and
the small change "MX" of section 10.4 is now in the library (pushed), with one rule added.

---

## 1. The two kernels side by side

Your column is copied from section 7 of your study. Mine is from my source and work log.

| Point | Nova forward (your study) | My forward (`GQA_fwd_sm90.cu`) |
|---|---|---|
| Head dim 128 tile | 128 query rows x 128 keys | the same |
| Threads | 384: 1 loader + 2 compute warpgroups | the same |
| Registers, compute / loader | 232 / 40 | 232 / 40 |
| K/V stages in shared memory | not stated | 3 |
| Query heads of one KV head | one tile per query head | one tile per query head (the same; I do not pack heads) |
| Scheduling | one block per tile, not persistent | persistent: 132 blocks take units from a counter, heaviest first, in L2 sections (FA3's scheduler, since 16/10/2026) |
| Chat list | built by a GPU kernel on every call (4.5 to 66 µs) | built on the CPU once per micro-batch, uploaded with one copy, reused by every layer and by the backward |
| O store | TMA | staged in the last tile's K buffer, then 16-byte global stores (a TMA store was 1.6% slower for me, section 5.6) |
| Head dim 64 on H200 | 128 x 128 tile, 2 compute warpgroups | no fast kernel: causal head dim 64 on Hopper goes to my generic WMMA kernel |

So on your cause 1 (tiles per query head) my kernel is built the same way as yours. On your causes 3 and 4 (chat list,
scheduling) it already works the way your proposal list asks for.

---

## 2. Your four layouts and my matching tests

| Your layout | My matching test | Date | Timed against FA3 / FA4 / cuDNN? |
|---|---|---|---|
| one1 (one chat fills the row) | "dense" rows; rows of one 4,096-token chat | from 12/10/2026 | yes |
| one1 at long T | dense rows at 4k, 16k, 32k, 64k, 128k | 31/10/2026 | yes |
| pk1 (chats of 300 to 1,500) | real packed rows from our SFT data: `real_rows_752` (rows 752 to 759, typical chats) | from 12/10/2026 | yes |
| sh1 (chats of 32 to 160) | real rows with our shortest chats (rows 1504 to 1511). These chats are much longer than yours, see section 3.1 | 12/10/2026 | yes, but it is not the same case. The same case (chats of 32 to 160) was measured on 13/11/2026: section 10 |
| edge1 (one-token chats, lengths at tile edges) | test loads D4, D6, D7, D8, D11 of my test suite (section 6) | from 10/10/2026 | correctness only, as in your plan |

What I never ran: head dim 64 on H200 against the other libraries. Your D64 results are new information for me.
Chats of 32 to 160 tokens against the other libraries I had not run when I first wrote this note; I ran them on
13/11/2026 (section 10).

---

## 3. My measurements

### 3.1 Work against time on real packed rows (12/10/2026)

- **Setup:** H200, 8 rows x 4,096 tokens, 18 query heads / 6 KV heads, head dim 128, forward only, median of 20 calls,
  free clocks. The kernel was the first packed version (before the optimizations of section 5).
- **Columns:** "work" = query-key pairs as a share of a dense row. "time" = packed time as a share of the dense time
  (dense = 1.083 to 1.111 ms).

| Rows | Work, share of dense | Time, share of dense |
|---|---|---|
| 0 to 7 (longest chats) | 92.8% | 97.6% |
| 752 to 759 (typical chats) | 37.1% | 46.5% |
| 1504 to 1511 (shortest chats) | 15.5% | 28.5% |

This is the same effect as your section 1.3: time falls more slowly than work, because every work unit pays a fixed cost
(Q load, pipeline fill, output write). My shortest real rows still hold 15.5% of the dense work. Your sh1 at 4,096 holds
2.8%. So my "short" case is far from yours.

### 3.2 Where the forward started and ended against FA3 (head dim 128, 18/6 heads, 8 x 4,096)

Ratio = my kernel time / FA3 kernel time, nsys, same box and session, free clocks. Below 1 means mine is faster.

| Date | State | Dense forward | Packed forward (`real_rows_752`) |
|---|---|---|---|
| 13/10/2026 and 16/10/2026 | baseline v62 | 1.375 | 1.28 |
| 19/10/2026 morning | after K5q | 1.084 | 0.936 |
| 19/10/2026 evening | final (QPU) | 1.004 (984.2 against 980.4 µs) | 0.865 |

Final scoreboard of 19/10/2026 (forward QPU + backward Vj2), all three libraries:

| Mine / rival | Dense fwd | Dense bwd | Dense fwd+bwd | Packed fwd | Packed bwd | Packed fwd+bwd |
|---|---|---|---|---|---|---|
| FA3 | 1.004 | 0.968 | 0.976 | 0.865 | 0.953 | 0.933 |
| FA4 | 0.919 | 0.989 | 0.972 | 0.781 | 0.971 | 0.923 |
| cuDNN | 0.863 | 0.891 | 0.884 | 0.827 | 0.899 | 0.883 |

### 3.3 Long rows, 4k to 128k (31/10/2026)

- **Setup:** heads 4 / 2 (our 135M model), head dim 128, causal. Dense = 32,768 / T full rows per call (8 at 4k, 1 at
  32k and above). Packed = one real packed row of the data prepared at that T. At 32k to 128k that row is almost one long
  chat, so packed is close to dense there. One library per process under nsys, rounds in the order
  mine, FA3, FA4, cuDNN, then the reverse. About 5 s of GPU work per run.
- **Columns:** my time / rival time, forward + backward per call. Last column: the single directions where I am behind.

| T | Layout | vs FA3 | vs FA4 | vs cuDNN | Single directions above 1.00 |
|---|---|---|---|---|---|
| 4,096 | dense | 0.952 | 0.909 | 0.827 | forward vs FA3 1.003 |
| 4,096 | packed | 0.875 | 0.885 | 0.812 | none |
| 16,384 | dense | 0.984 | 0.980 | 0.856 | none |
| 16,384 | packed | 0.936 | 0.941 | 0.804 | none |
| 32,768 | dense | 0.989 | 0.990 | 0.860 | backward vs FA4 1.000 to 1.003 |
| 32,768 | packed | 0.986 | 0.981 | 0.829 | forward vs FA3 1.006 |
| 65,536 | dense | 0.995 | 0.996 | 0.863 | backward vs FA4 1.007 to 1.009 |
| 65,536 | packed | 0.990 | 0.988 | 0.836 | forward vs FA3 1.010 |
| 131,072 | dense | 0.999 | 0.998 | 0.858 | backward vs FA3 1.001 to 1.002, vs FA4 1.013 to 1.014 |
| 131,072 | packed | 0.993 | 0.990 | 0.829 | forward vs FA3 1.020 |

The lead over FA3 and FA4 shrinks as rows get longer: at long rows every kernel is near the tensor-core limit.

---

## 4. Your four causes, and what I know about each

### 4.1 Cause 1: one tile per query head (short chats)

- My kernel has the same structure: one tile per query head. Measured on 13/11/2026 (section 10): on rows of 32 to
  160-token chats it does NOT lose to FA3. Its forward takes 12.5% less time than FA3's forward. (When I first wrote
  this note I expected the same kind of loss as in your study. The measurement shows that was wrong.)
- I did not try FA3's PackGQA. On 17/10/2026 I marked it "not needed" for dense 4,096 rows, where every tile is full.
  On 13/11/2026 I built and raced a smaller change, "MX" (two query heads share one pair of 64-row tiles): section 10.4.
- Why I did not go further: our real SFT chats are not short (section 7). Median 454 to 905 tokens, so most tiles are full.

### 4.2 Cause 2: the head dim 64 tile

- I have nothing to add. My fast Hopper kernel is head dim 128 only. Both our models (135M with heads 4/2, 1.6B with
  heads 18/6) use head dim 128.

### 4.3 Cause 3: the chat-list kernel

- I do not have this cost. `gqa_chat_plan_build` (declared in `include/ops/helpers/AttentionKernels.h`) builds the work
  lists on the CPU from the chat lengths of the micro-batch and uploads them with one asynchronous copy.
- The SFT trainer builds one plan per micro-batch. Every layer's forward and backward reads the same plan.
- This matches the second option of your item 3 ("build the list once per step").

### 4.4 Cause 4: 11% behind FA3 at one 131,072-token chat

- My kernel also reads K/V once per query head. At 131,072 dense it is level with FA3 (0.999 forward + backward, and the
  forward alone is not above 1.00, section 3.3).
- That run used heads 4 / 2, so 2 query heads per KV head, not your 18 / 6. It is not the same test. But it shows that
  reading K/V per query head does not have to cost 11% at this length.
- What my kernel has at this point that your table says Nova does not: the persistent scheduler with L2 sections
  (section 5.2). I would test that before PackGQA for this one gap.
- My L2 measurements on dense 8 x 4,096, 18/6 (19/10/2026): FA3 puts cache hints on its TMA loads (K/V evict-last, Q
  evict-first). Without them my kernel read 520 MB from DRAM against FA3's 365 MB. With them, 432 to 447 MB, and the
  dense forward got 1.2 to 1.5% faster, but the packed forward got 0.6 to 0.7% slower. I kept the hints in the dense
  kernel only.

### 4.5 Your accuracy detail: LSE error growing with chat length

- I did not check this in my kernel. My tests gate LSE with the same 2x rule as O, and all pass, but I never looked at
  how the LSE error grows with length.

---

## 5. My forward fixes that touch your proposal list

Each line: date, what changed, result against the build before it. "Dense" = 8 x 4,096, 18/6. "Packed" =
`real_rows_752`. Minus = faster. "12/12" = faster in 12 of 12 rounds. Every change passed the 296-case test suite before
its race.

### 5.1 The loop order (the biggest single difference from FA3 at the start)

- **Baseline (13/10/2026):** I waited for QK of tile n, ran the softmax with nothing on the tensor cores, then started
  P·V of tile n. Tensor cores busy 64% against FA3's 83%.
- **F1 (16/10/2026):** FA3's order: the softmax of tile n runs while P·V of tile n-1 is still on the tensor cores.
  Dense -0.50%, packed -1.82% (12/12).
- **G1 (16/10/2026):** P·V as 8 full-width `m64n128k16` GEMMs instead of 16 half-width ones. Dense -3.34%, packed -3.22%.

### 5.2 Persistent scheduling (your item 4)

- **G5 (16/10/2026):** FA3's persistent scheduler. A unit is one query pair of one head (128 rows). Units are taken
  through a work counter, heaviest first, in sections of (batch, head) pairs sized by FA3's L2 rule.
  Dense -2.63%, packed -3.55% (12/12). SMs busy 96.7% to 98.6% dense, 94.4% to 97.7% packed.
- **Cost I found:** dense L2 hit rate fell from 92.2% to 83.0% and DRAM reads rose from 254 to 526 MB, because a section
  of 48 heads is too big for the L2. Still a net win.
- **F3c (16 and 17/10/2026):** FA3's hand-over of the next unit: the producer publishes it during the current unit's
  loads, the consumers read it before their epilogue. Dense no change, packed -0.84% to -1.55%.
- **F3b, rejected:** the same with FA3's 24 / 240 register split. Dense +1.11% slower. I kept 40 / 232.
- **F3, a hang:** the first hand-over version deadlocked on packed rows. Since then I check every barrier change on a
  CPU model of the protocol before any GPU run.

### 5.3 Unit boundaries (this is where short chats pay)

A packed row has many short units, so the cost at the start and end of a unit counts more than in dense rows.

- **K4b (17/10/2026):** the output write at the unit end cost 4,230 cycles per pair (9% of dense time, 21% of packed).
  Now O is written with stmatrix into the last tile's K buffer, then 16-byte global stores. Dense -5.26%, packed -7.37%
  (8/8). The first version (K4) could deadlock; K4b uses a counter that only grows.
- **PF (19/10/2026):** the wait for the first K/V tile of a unit cost 1,498 cycles per unit (FA3: 50). My producer
  waited for Q to be free, loaded Q, then loaded K/V. FA3 loads the first K as soon as a slot is free, then waits for Q.
  With FA3's order: 1,498 to 48 cycles. Dense -2.60%, packed -3.01% (8/8).
- **QPU (19/10/2026):** Q prefetched into L2 when a pair starts, and the producer publishes the decoded unit (batch,
  head, pair), so the consumers do no integer divisions at the unit start. Dense -0.81%, packed -0.13%.

### 5.4 Softmax

- **F2 (16/10/2026):** always rescale O (no vote, no branch). Dense -3.84%, packed -1.54%.
- **SM3m (19/10/2026):** exps placed like FA3's `softmax.h` (row by row, the next row's max between them) and FA3's
  causal-mask order. In my code the exps were packed too close and waited on each other: 367 cycles of exp wait per step
  against FA3's 248. After: 245. Dense -3.51%, packed -4.48% (8/8).
- **K5q (17/10/2026):** 16 of the exps before the P·V wait, one P register set. Packed -4.33%.

### 5.5 Things I tried that did not help (do not repeat)

| Change | Result |
|---|---|
| 2 K/V stages as FA3 has (K1n2) | dense +39% slower, packed +22% slower |
| FA3's split K and V pipelines (SKV) | dense +0.20%, packed +0.87% slower |
| CUTLASS-style barrier wait, spin wait, separate K and V barriers (W1, W1s, W2) | no change |
| GEMM descriptors built once (L1, L1b) | no change, or slightly slower (the saved instructions sat in waiting time) |
| Every 8th or 4th exp by a polynomial (P8w, P4w) | +2.7% and +7.1% slower |
| Different softmax math for dense and packed (K8, K9) | failed "packed chat = the same chat alone, bit for bit" |
| Diagonal tile first, as FA3 walks the keys (QPD) | dense +0.32%, packed -0.43%; not kept |

### 5.6 O store by TMA (your table lists TMA for Nova)

- **PFT (19/10/2026):** I wrote O by one TMA store as FA3 does on its dense path. The epilogue got 484 cycles cheaper,
  but the next unit's start grew by 668 cycles (the storing warp's wait) and the loop by 40 cycles per step.
  Dense +1.60% slower (0/8). Rejected.

---

## 6. Correctness tests I run (your edge1)

Test file: `Tests/kernel_tests/test_varlen_attention.cu`. Quick suite 296 checks, full suite 605.

- **Pass rule:** the same FlashAttention rule you use: |mine - exact| <= 2 x |plain bf16 attention - exact| + atol, per
  chat, for O, dQ, dK, dV, and also LSE.
- **Strongest check:** every chat in a packed row gives the same bits as the same chat run alone through the dense
  kernel (O, LSE, dK, dV).
- **Data loads:**

| Load | Chats in the row |
|---|---|
| D1, D10 | real SFT rows |
| D2, D3 | 2,048 + 2,048; 1,000 + 3,096 |
| D4 | 1 + 4,095 and 4,095 + 1 |
| D5 | 64 chats of 64; 32 chats of 128 |
| D6 | 4,096 one-token chats |
| D7 | 63 / 65 / 127 / 129 / 255 / 257, repeated |
| D8 | random lengths with pad tails of 0 / 1 / 16 / 63 / 64 / 65 / 255 |
| D9 | a row of padding only |
| D11 | the same 300-token chat at offsets 0 / 1 / 63 / 64 / 65 / 127 / 128 / 3,796 |

- **Long rows:** group G14 checks dense and packed at 8k to 128k (31/10/2026).
- One-token chats: dQ and dK are exactly 0 for them, so they are checked against a rounding bound, not the 2x rule.

---

## 7. The chat lengths of our real SFT data

Your section 1.6 says the chat-length histogram of our dataset is not measured. I measured it on 30/10/2026 from
`seq_lengths.bin` of the data prepared at T = 4,096 (tokens per chat):

| Data mix | Chats kept | Median | p90 | p99 | Chats above 4,096 (dropped) |
|---|---|---|---|---|---|
| m1 smol | 452,974 | 905 | 1,364 | 1,531 | 4,091 (0.90%) |
| m2 Tülu | 744,023 | 454 | 1,264 | 3,116 | 19,162 (2.51%) |
| m3 smol + math | 549,217 | 671 | 1,346 | 1,509 | 4,091 (0.74%) |

- Our data looks like your pk1 (median about 870), not like sh1 (median about 95).
- I did not compute the share of chats under 160 tokens. That number would say how much the sh1 gap matters in training.

---

## 8. What is still open on my side

1. Chats of 32 to 160 tokens against FA3 / FA4 / cuDNN: measured on 13/11/2026 (section 10). Still to run from that
   session, which was paused: one long chat against the three libraries, one of the three real-row groups against FA3.
2. Packing the rows of the 3 query heads of a KV head into one tile (FA3's PackGQA): not tried. The smaller change MX:
   built and raced on 13/11/2026 (section 10.4), not adopted yet.
3. A fast Hopper kernel for head dim 64: does not exist in my code.
4. LSE error against chat length: not checked.
5. Packed forward alone at 32k to 128k: 0.6 to 2.0% behind FA3.
6. The two kernels (Nova's and mine) have never been run on the same layouts on the same GPU. That run would show
   directly how much the persistent scheduler and the CPU-built chat list are worth on your sh1 and one1 cases.
   Section 10.2 has an indirect comparison only: the same box, the same chat-length range, FA3 as the common reference.

---

## 9. Where things are

| What | Where |
|---|---|
| Forward kernel | `Tensor-Implementations`, branch `dist_side_gautam_lottn`: `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu` |
| Backward kernel | same folder: `GQA_bwd_vj2_sm90.cu` |
| Chat plan (the per-chat work lists) | `include/ops/helpers/AttentionKernels.h`: `GqaChatPlan`, `gqa_chat_plan_build` |
| Test suite | `Tests/kernel_tests/test_varlen_attention.cu` |
| Every optimization, kept or rejected, with timings | `attention_kernel_optimizations_h200_2026_10_10_to_2026_10_21.md` |
| Day-by-day log | `sft_work_documentation.md`, sections 40 to 45.125, and section 45.194 for 13/11/2026 |
| The 13/11/2026 study (layouts, scripts, candidate MX, all logs) | `profiling_reports/attn_short_chats_2026_11_13/` |
| FA3 source I compared against | FlashAttention commit `94e22c9`, folder `hopper/` (your study used `f63c472`) |

---

## 10. Measured on 13/11/2026: your short-chat case with my kernel

### 10.1 Setup

- **Box:** P032 (the box of your study), H200, free clocks. Every layout was measured with all its contenders on one
  GPU (GPU 4 or GPU 5).
- **Shape:** 8 rows x 4,096 tokens per call (32,768 tokens, the same total as one of your T = 32,768 rows), 18 query
  heads / 6 KV heads, head dim 128, bf16, causal.
- **Layouts:**
  - "sh" = chats of 32 to 160 tokens, your sh1 range (353 chats in the call);
  - "pk" = chats of 300 to 1,500 tokens, your pk1 range (42 chats in the call);
  - two groups of 8 real packed rows of our SFT data.
  The lengths come from my own random seeds, so they are not your exact chats. A chat that reaches the end of its
  4,096-token row is cut there.
- **Method:** one library per process under nsys, forward + backward back to back, kernel times only, two rounds (order
  mine, FA3, FA4, cuDNN, then the reverse), about 5 s of GPU work per run.
- **"Forward" below** = the attention forward kernel alone. For FA3 this leaves out its prepare kernel. My kernel has no
  extra kernel: the chat list is built on the CPU once per micro-batch.
- **FA3** = commit `94e22c9`, bf16, head dim 128. I read its template arguments from the kernel name in the trace:
  Varlen 1, PackGQA 1, tile 128 x 128, `VarlenDynamicPersistentTileScheduler`. So it is FA3 with the head packing your
  study describes.

### 10.2 My kernel against FA3 / FA4 / cuDNN

Kernel time in µs per call (mean of the two rounds). Ratio = my time / the library's time; below 1.00 means mine is
faster.

| Layout | Direction | Mine | FA3 | FA4 | cuDNN | vs FA3 | vs FA4 | vs cuDNN |
|---|---|---|---|---|---|---|---|---|
| sh, chats of 32 to 160 | forward | 191.8 | 219.2 | 380.9 | 315.6 | 0.875 | 0.503 | 0.607 |
| sh, chats of 32 to 160 | backward (main kernel + its helper kernels) | 995.0 | 1,336.8 | 1,279.9 | 1,256.5 | 0.744 | 0.777 | 0.792 |
| sh, chats of 32 to 160 | forward + backward | 1,186.7 | 1,555.9 | 1,660.8 | 1,572.2 | 0.763 | 0.715 | 0.755 |
| pk, chats of 300 to 1,500 | forward | 341.2 | 375.3 | 442.1 | 420.8 | 0.909 | 0.772 | 0.811 |
| pk, chats of 300 to 1,500 | forward + backward | 1,639.8 | 1,779.5 | 1,795.9 | 1,837.2 | 0.922 | 0.913 | 0.893 |
| real rows, group 2 | forward | 389.4 | 429.2 | not run | not run | 0.907 | | |
| real rows, group 2 | forward + backward | 1,825.4 | 2,010.1 | not run | not run | 0.908 | | |
| real rows, group 1 (one round only) | forward | 449.3 | 490.8 | not run | not run | 0.916 | | |

Notes:
- pk: FA3 and FA4 have one round only. Their second-round runs failed, because another job started on the GPU at the
  same moment (the box runs every GPU in Exclusive_Process mode, so the run fails and gives no number).
- On sh every library ran at the clock limit (1,980 MHz). On pk and the real rows every library is at the 700 W power
  cap, and my kernel runs at a lower clock (1,688 to 1,725 MHz) than FA3 (1,785 to 1,815 MHz).
- Not measured in this session: one long chat per row against the libraries (my script wrote that layout's chat list
  twice and the library runner refused it; fixed, the rerun was not done before the session was paused), the third
  real-row group, FA4 and cuDNN on the real rows.

**An indirect look at the two kernels, yours and mine.** Your table (section 4, T = 32,768, sh1, 342 chats) and my sh
layout (8 x 4,096, 353 chats) hold the same total tokens and the same chat-length range, on the same box:

| | Your study | My run |
|---|---|---|
| FA3 | 221.6 µs (prepare + main kernel) | 219.2 µs (main kernel only) |
| attention forward kernel | 293.2 µs (Nova, the attention kernel alone) | 191.8 µs (mine) |

FA3 agrees between the two runs within a few percent. So on rows of 32 to 160-token chats my kernel takes about 100 µs
less than yours for the same total work. This is only an indication: the row shapes differ (1 x 32,768 against
8 x 4,096), and the two kernels were not raced in one session.

### 10.3 What one work unit costs in my kernel

In my kernel a work unit is one chat x one query head x one pair of 64-row query tiles (longer chats: a couple of
pairs). I timed three layouts where every chat has the same length, so every unit is the same:

| Every chat is | Rows used in the 128-row pair | Units per call | Forward | Forward time per unit (kernel time / units) |
|---|---|---|---|---|
| 40 tokens | 31% | 14,832 | 282.9 µs | 19.1 ns |
| 64 tokens | 50% | 9,216 | 180.4 µs | 19.6 ns |
| 120 tokens | 94% | 5,040 | 129.8 µs | 25.7 ns |

- A unit with 40 rows and a unit with 64 rows cost the same. A unit with 120 rows has 3 times the rows and 9 times the
  query-key pairs of the 40-row one and costs only 35% more.
- So on rows of very short chats the forward time is almost all fixed cost per unit (Q load, the wait for the first
  K / V tile, the output write, the hand-over to the next unit), and that fixed cost is small: 19 ns of kernel time per
  unit with 132 blocks working, which is about 2.5 µs for the one block that runs the unit.
- This is why my kernel is ahead of FA3 on short chats although it has more tiles than FA3: sections 5.2 and 5.3 are
  the changes that cut this fixed cost (persistent scheduler, the output write, the K / V load order, Q prefetch).
- The same three layouts would give your kernel's fixed cost per tile directly. The files are small (chat lengths
  only): `profiling_reports/attn_short_chats_2026_11_13/pkg/gen_layouts.py` writes them.
- Bound for full head packing in my kernel, from this table: one packed tile of 3 x 40 rows costs at most one
  120-token unit, so on rows of 40-token chats packing would save at least 1 - 25.7 / (3 x 19.1) = 55% of the forward,
  before any packing overhead.

### 10.4 A smaller change than PackGQA: "MX" (two query heads share one pair)

**The idle group.** My two compute warpgroups each own one 64-row query tile of a pair. A chat with an odd number of
64-row tiles (every chat of 64 tokens or less, 129 to 192, 257 to 320, ...) leaves the second group of its last pair
with no rows. It only runs the handshake.

**MX.** That idle group takes the same last tile of another query head of the same KV head. Both heads read the same
K and V, so the K / V loads of the unit do not change, and both groups run the same number of key tiles.
- Changed: the work list (which units exist) and the unit decode in three places.
- Not changed: main loop, softmax, mask, output write, the dense kernel (its machine code is identical, 0 lines differ).
- With 3 query heads per KV head, two heads share and the third keeps its old unit. With 2 per KV head every such tile
  is shared.

128-row pairs per call (8 x 4,096, heads 18/6):

| Layout | Today | MX | FA3-style packing (rows of 3 heads in one tile) |
|---|---|---|---|
| sh, chats of 32 to 160 | 7,974 | 6,798 (-14.7%) | 5,658 (-29.0%) |
| pk, chats of 300 to 1,500 | 5,040 | 4,878 (-3.2%) | 4,728 (-6.2%) |
| one chat of 4,096 per row | 4,608 | 4,608 | 4,608 |

**Race of MX against today's kernel** (GPU 5, one session, 4 rounds, forward + backward per run, about 4 s of GPU work
per run). Forward time, change against today's kernel in the same round; minus means faster. "MX code, old work list"
shows what the longer unit decode costs by itself.

| Layout | Today's kernel | MX code, old work list | MX code, MX work list |
|---|---|---|---|
| sh, chats of 32 to 160 | 0.1956 ms | +1.06% (slower in 4 of 4 rounds) | -8.01% (faster in 4 of 4, t -32) |
| pk, chats of 300 to 1,500 | 0.3461 ms | -0.54% (not significant) | -2.68% (4 of 4, t -7.4) |
| one chat of 4,096 per row | 0.9949 ms | -0.14% (not significant) | -0.06% (not significant) |
| real rows, group 1 | 0.4541 ms | +0.48% | -0.71% (3 of 4, not certain) |
| real rows, group 2 | 0.3910 ms | +0.90% | -1.05% (4 of 4, t -5.1) |
| real rows, group 3 | 0.5187 ms | +0.66% | -0.39% (3 of 4, not certain) |

- **Correctness:** my test suite passes on both MX versions (245 checks passed, 0 failed, each; 4 more groups need a
  data file that was not on the box). It includes "a chat in a packed row gives the same bits as the same chat alone".
- **ncu on the sh layout, today's kernel -> MX:** kernel time 198.7 -> 181.5 µs, SM cycles 342,995 -> 309,049
  (-9.9%), instructions 60.55 M -> 60.01 M (-0.9%), DRAM read 410 -> 375 MB (-8.5%). The same instructions, 10% fewer
  cycles: the fixed cost of a pair is paid once for two heads.
- **State:** MX is not adopted. The session was paused after 4 of 8 planned rounds; the full 605-check suite and a
  trainer run are still to do.

### 10.5 What I take from this for your study (my reading)

1. One tile per query head does not by itself make a kernel lose to FA3 on short chats. My kernel has it and is 12.5%
   ahead of FA3 (forward) on chats of 32 to 160 tokens. (Measured later, section 12: this holds for the 32 to 160 mix
   and for 64 tokens and above. When every chat is 32 tokens, FA3's forward wins. So tile count does decide at the
   very short end.)
2. What decides the short-chat time is the fixed cost per tile. From your section 7 table and my source, the two
   kernels differ there in: scheduling (132 persistent blocks that take units from a counter, against one block per
   tile), the chat list (built on the CPU once per micro-batch, against a kernel of 4.5 to 66 µs on every call), and the
   output write. I would measure your kernel's cost per tile with the three layouts of section 10.3 before building
   PackGQA.
3. If your two compute warpgroups each own 64 query rows, as mine do, MX is a small first step: it needs only the work
   list and the unit decode.

---

## 11. What changed in this note on 13/11/2026 (after the measurements of section 10)

| Where | The first text | Now |
|---|---|---|
| Section 2, below the table | "What I never ran: chats of 32 to 160 tokens timed against the other libraries, and head dim 64 ..." | head dim 64 never run; chats of 32 to 160 run on 13/11/2026 (section 10) |
| Section 2, table row sh1 | "yes, but it is not the same case" | the same, plus a pointer to section 10 |
| Section 4.1, first point | "My kernel has the same structure, so I expect the same kind of loss on 32 to 160-token chats. I have not measured it." | measured: no loss; the forward takes 12.5% less time than FA3's. The expectation was wrong |
| Section 4.1, second point | "I did not try FA3's PackGQA ..." | the same, plus the smaller change MX of section 10.4 |
| Section 8, items 1, 2, 6 | "not measured" / "not tried" / "never been run on the same layouts" | item 1 measured, item 2: MX built and raced, item 6: an indirect comparison added in section 10.2 |
| Section 9 | log sections "40 to 45.125" | plus section 45.194 and the study folder |
| Section 10.5, point 1 (second update) | "One tile per query head does not by itself make a kernel lose to FA3 on short chats ..." | kept, with its limit added: at 32 tokens per chat FA3's forward wins (section 12) |

---

## 12. Second update, 13/11/2026 night: length by length, and MX in the library

### 12.1 Against FA3 at single chat lengths

Same setup as section 10.1 (P032, 8 rows x 4,096, heads 18/6, head dim 128, nsys kernel times, FA3 `94e22c9` with
PackGQA on). In the "every chat" rows every chat of the row has that length. "Current" is my kernel as it was;
"with MX" is my kernel with the change of section 10.4. Ratio = my time / FA3's time, below 1.00 means mine is faster.

| Chats in the row | MX against current, forward | Current / FA3, forward | With MX / FA3, forward | Current / FA3, forward + backward | With MX / FA3, forward + backward |
|---|---|---|---|---|---|
| every chat 32 tokens | -24.9% | 1.536 | 1.146 | 0.675 | 0.643 |
| every chat 64 tokens | -21.3% | 0.788 | 0.620 | 0.669 | 0.645 |
| every chat 128 tokens | no change | 0.656 | 0.656 | 0.783 | 0.783 |
| every chat 192 tokens | -8.3% | 0.774 | 0.702 | 0.781 | 0.771 |
| every chat 704 tokens | -3.9% | 0.859 | 0.825 | 0.888 | 0.882 |
| every chat 2,048 tokens | no change | 0.911 | 0.911 | 0.948 | 0.948 |
| chats of 32 to 160, mixed | -8.0% | 0.875 | 0.802 | 0.763 | 0.752 |
| chats of 300 to 1,500, mixed | -2.7% | 0.909 | 0.897 | 0.922 | 0.920 |
| one chat of 4,096 per row | no change | 0.950 | 0.950 | 0.959 | 0.959 |
| real rows, 3 groups | -0.5% to -1.3% | 0.907 to 0.918 | 0.897 to 0.919 | 0.900 to 0.924 | 0.899 to 0.929 |

Forward kernel times in µs per call:

| Chats in the row | Current | With MX | FA3 |
|---|---|---|---|
| every chat 32 tokens | 347.0 | 259.0 | 226.0 |
| every chat 64 tokens | 179.6 | 141.4 | 228.0 |
| every chat 128 tokens | 123.6 | 123.6 | 188.3 |
| every chat 192 tokens | 166.1 | 150.7 | 214.6 |
| every chat 704 tokens | 271.1 | 260.4 | 315.7 |
| every chat 2,048 tokens | 560.0 | 560.0 | 614.5 |

- The "every chat" rows are one round per contender; the mixed rows two rounds.
- **At 32 tokens per chat FA3's forward wins**, 226 µs against 347 µs for my kernel as it was and 259 µs with MX. FA3
  needs the same number of tiles at 32 tokens as at 64 (3 heads x 32 rows fit one tile), and its time is the same
  (226 and 228 µs). My kernel needs twice the pairs at 32 as at 64, and its time doubles. So at about 40 tokens and
  below, packing all 3 heads into one tile decides the forward, as your study says.
- Forward + backward together my kernel is ahead at every length, also at 32 tokens (0.675, and 0.643 with MX): the
  backward is far faster than FA3's on such rows (946.5 µs against 1,634.0 µs for the main kernel).
- From 64 tokens up my forward is ahead of FA3 at every length measured, by 9% (2,048 tokens) to 38% (64 tokens, with MX).

### 12.2 MX against my kernel as it was, at 20 chat lengths

Three rounds, forward, the two builds raced in one session:

| Chat length | Forward with MX |
|---|---|
| 16 / 32 / 48 / 64 | -25.6% / -24.9% / -24.1% / -21.3% |
| 160 / 192 | -10.1% / -8.3% |
| 320 / 448 / 704 | -7.2% / -4.5% / -3.9% |
| 1,088 / 2,112 | -3.0% / -1.1% |
| 128, 256, 384, 512, 1,024, 2,048, 4,096 (even number of 64-row tiles) | no change |
| rows of 96-token chats that end in one 64-token leftover chat | +0.9% (slower) |

The last row is the one loss: a single chat with an odd tile count sent the whole work list to the MX kernel, whose
longer unit decode costs 0.5 to 1.1% on rows with many chats, while almost nothing was shared.

### 12.3 The rule added to MX

The work-list builder counts, from the chat lengths alone, how many units would share a tile. It uses MX only when that
is at least 2% of the units of the plain list; otherwise it builds the list of before and the kernel of before runs.
Measured by that share: 0.78% gave +0.9%, 1.75% gave +0.2% (not significant), 2.1% gave -0.5%, 4 to 7% gave -1 to -3%,
33% gave -21 to -26%. With the rule the 96-token rows are back at 0 (-0.02%) and the gains stay (160 tokens -9.7%,
192 tokens -9.0%).

### 12.4 State

- MX with the 2% rule is pushed: Tensor-Implementations `f1a7d98`, BluTrain `d9606c2`, branch `dist_side_gautam_lottn`.
- Tests on that form: my suite at heads 18/6 (quick 280 and full 333 checks) and at heads 4/2 (quick 273 checks), long
  rows of 8,192 tokens (69 checks), all passed; a 30-step SFT trainer run gives the same losses as before.
- The dense kernel and the plain packed kernel are unchanged, instruction for instruction.
- In a whole SFT step on our real data MX changes nothing measurable; its gain is on rows with many short chats, and
  only in the forward (forward + backward together: about -5% at 16 tokens, -3.6% at 64 tokens).
- Not built: FA3's packing of 3 heads into one tile. For chats of about 40 tokens and less it is what the forward needs.
