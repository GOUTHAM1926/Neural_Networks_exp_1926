# Varlen attention forward on H200: Nova vs FA3 / FA4 / cuDNN, head dims 64 and 128, B=1 (Nov 13)

This study answers two questions for the forward pass:

1. Is our forward correct at every length?
2. How fast is it against the reference libraries, and why?

It covers the head dims where performance matters (64 and 128) at all seven sequence lengths, with one packed row (B=1). Head dims 32, 96 and 256 are not covered yet (section 9). Section 1 explains the chat layouts every result is measured on.

## 1. The chat layouts

Every case runs one batch row (B=1) of T tokens that holds several chats back to back, with no padding between them. Each token attends causally to the earlier tokens of its own chat and to nothing outside it. A **layout** is the list of chat lengths in that row.

The tensors have the same shape in every layout; only the chat boundaries move. Those boundaries decide how much work there is, how many tiles each library launches, and which library is fastest.

### 1.1 The four layouts

| Layout | Chat lengths | Chats at T = 2048 … 131072 | Stands for | What it stresses |
|---|---|---|---|---|
| **one1** | One chat of T tokens | 1 | One document filling the context (long-context training). The same as dense causal attention. | The kernel's main loop. Every tile is full, but tile cost is very uneven: a head's first 128-row tile reads 1 key block, its last at T=131072 reads 1024. The order in which tiles run matters. |
| **pk1** | 300–1500 (median ≈ 870) | 3, 6, 9, 21, 37, 74, 144 | SFT packing of medium-length conversations | Many separate causal triangles, each with its own diagonal mask. A chat covers 3–12 tiles per head, and only its last tile is partly empty. |
| **sh1** | 32–160 (median ≈ 95) | 22, 42, 87, 173, 342, 682, 1364 | Many short chats: the hardest packing case | Tile fill and the fixed cost of each tile. About 75% of the chats fit in one 128-row tile, so most tiles hold one small triangle. |
| **edge1** | 200 one-token chats (100 at T=2048); chats of 63/64/65, 127/128/129, 191/192/193 and 255/256/257 tokens; one long chat with the rest of the row; a final one-token chat | 114 at T=2048, 214 above | Correctness only. **Not run in this study.** | Off-by-one masking at every tile edge (64, 128, 192 and 256 rows, ±1); the one-token softmax (LSE equals the single score); a chat that ends exactly at T. |

The random layouts begin with these chats (a shorter T stops earlier, and its last chat may be cut):

- pk1: 391, 573, 1236, 683, 826, 1474, 1168, 847, 1474, 669, …
- sh1: 55, 40, 34, 117, 91, 57, 125, 151, 86, 45, …

### 1.2 How the chat lengths are made

- **Code:** `b1_layouts` in `claude_handoff/suite_1112/src/attn_suite.cpp`, with `b4_packed` (`p3_varlen.cpp`) and `rows_of` (`test_varlen_kernels.cpp`).
- **Fixed seeds.** pk1 uses seed 7 and sh1 uses seed 11, so a layout is the same on every run and every machine.
- **Filling the row.** Lengths are drawn uniformly from the layout's range until the space left is smaller than the minimum.
  - A draw longer than the space left is cut to fit. For example, pk1 at T=2048 is 391, 573, 1084: the third draw was 1236.
  - A leftover smaller than the minimum becomes a chat of its own, attended like any other. This is why sh1 ends with an 11-token chat at T=32768 and a 10-token chat at T=65536, and pk1 ends with a 213-token chat at T=16384.
- **A longer T only adds chats.** The same seed gives the same draws, so every T starts with the same chats; only the last chat can be cut differently.
- **Every library gets the same chats.** The lengths are written to `results_quick/lens/<layout>_B1_T<T>.txt`, one `row start length` line per chat. `ref_fwd.py` builds the `cu_seqlens` for FA3, FA4 and cuDNN from these files.
- **Names.** The "1" means B=1. The B=4 versions in the older harness use the same draws and are named by their chat count at T=4096: one4, pk23, sh175, edge4.

### 1.3 Work per layout

Causal attention over a chat of L tokens costs L(L+1)/2 query–key pairs. Splitting T into chats of about L tokens therefore cuts the work to roughly L/T of one1's.

| T | pk1 chats | pk1 lengths | pk1 work vs one1 | sh1 chats | sh1 lengths | sh1 work vs one1 |
|---|---|---|---|---|---|---|
| 2048 | 3 | 391–1084 | 39.5% | 22 | 33–153 | 5.5% |
| 4096 | 6 | 387–1236 | 19.7% | 42 | 33–159 | 2.8% |
| 8192 | 9 | 391–1474 | 12.5% | 87 | 33–159 | 1.4% |
| 16384 | 21 | 213–1474 | 5.7% | 173 | 32–160 | 0.68% |
| 32768 | 37 | 319–1474 | 3.1% | 342 | 11–160 | 0.34% |
| 65536 | 74 | 319–1474 | 1.5% | 682 | 10–160 | 0.17% |
| 131072 | 144 | 301–1484 | 0.78% | 1364 | 32–160 | 0.085% |

The time does not fall as fast as the work.

- At T=131072, D128, our forward takes 128.7 ms on one1, 1.57 ms on pk1 and 1.23 ms on sh1.
- sh1 has a ninth of pk1's work but takes 78% of its time (FA3: 50%).
- On sh1 the time is mostly the fixed cost of each tile and each chat, not math. We run sh1 at 33–58 TFLOP/s and FA3 at 38–80, against 317–742 and 306–746 on one1.

### 1.4 Tiles per layout

A tile is one block of query rows, processed against the keys of its chat. Our tile has 128 rows at both D64 and D128. The layout decides how many tiles there are and how full they are:

- **Ours:** each query head is tiled on its own, so a chat of L tokens costs 18 × ⌈L/128⌉ tiles.
- **FA3:** the 3 query heads of a KV head are packed into one tile. The same chat costs 6 × ⌈3L/128⌉ tiles at D128, or 6 × ⌈3L/192⌉ at D64, where its tile has 192 rows.

Example chats from sh1 (tiles, then the share of tile rows that hold real tokens):

| Chat length | Ours | FA3 D128 | FA3 D64 |
|---|---|---|---|
| 40 | 18 tiles, 31% | 6 tiles, 94% | 6 tiles, 63% |
| 55 | 18 tiles, 43% | 12 tiles, 64% | 6 tiles, 86% |
| 117 | 18 tiles, 91% | 18 tiles, 91% | 12 tiles, 91% |
| 129 | 36 tiles, 50% | 24 tiles, 76% | 18 tiles, 67% |
| 160 | 36 tiles, 63% | 24 tiles, 94% | 18 tiles, 83% |

Packing helps most for chats well under 128 tokens and for chats just past a tile edge. It does nothing for a 117-token chat.

Over a whole layout, the fill barely changes with T:

| Layout | Fill, ours | Fill, FA3 D128 | Fill, FA3 D64 | Tiles ours / FA3 at T=131072, D128 | Same, D64 |
|---|---|---|---|---|---|
| one1 | 100% | 100% | 100% | 1.00 | 1.50 |
| pk1 | 89–93% | 96–98% | 96–97% | 1.05 | 1.55 |
| sh1 | 57–60% | 80–83% | 75–77% | 1.37 | 1.88 |

At D64, part of the ratio comes from FA3's larger tile (192 rows against our 128), not from empty rows. On one1 our tiles are full and there are still 1.5× as many.

### 1.5 What each layout tells us

- **one1 isolates the speed of the main loop.**
  - At D128 our tiles match FA3's, and we are level with it up to 32K. The 11% gap at T=131072 is a separate effect (section 7, point 4).
  - At D64 we are 3–11% behind FA3 with tiles that are just as full. That points at the tile shape (section 7, point 2).
- **pk1 is mostly main-loop speed, plus a little partial-tile waste.** 7–11% of our tile rows are empty, against 2–4% of FA3's. We are near FA3 at D128 and behind it at D64, for the same tile reason.
- **sh1 tests packing and scheduling.** Our 1.37× and 1.88× tile counts track the 1.38× and 1.62× kernel-time gaps. The chat-list kernel (66 µs with 1364 chats) adds the rest (section 7, points 1 and 3).

### 1.6 Limits of these layouts

- **They are synthetic, with uniform lengths.** Real SFT data usually has a skewed length distribution, with many short turns and a few long ones. We have not measured the chat-length histogram of our dataset. The closer it is to sh1, the more GQA packing (section 8, item 1) is worth.
- **edge1 was not run here.** A correctness-only run takes under 2 minutes per head dim on one GPU: `attn_suite fwdquick` with `FQ_LAYOUTS=edge1 FQ_TIME=0`.

## 2. Summary

- **Correct everywhere.** All 42 cases pass against an fp64 reference: 2 head dims × 7 lengths × 3 chat layouts (section 1). Our O error is 0.49–0.91× that of plain bf16 attention, against a limit of 2×. FA3, FA4 and cuDNN pass the same check on the same layouts (126 cases), so the timing comparison is like for like.
- **D128 is at FA3's level, except on short chats.**

  | Layout | Ours vs FA3 |
  |---|---|
  | One long chat | 0.98–1.01× up to T=32768, 1.03× at 65536, 1.11× slower at 131072 |
  | Chats of 300–1500 tokens | 0.92–1.10× (we are faster at long T) |
  | Chats of 32–160 tokens | 1.28–1.44× slower |

- **D64 is behind on every layout.** FA4 is the fastest library at D64.

  | Layout | Ours vs FA3 | Ours vs the fastest library |
  |---|---|---|
  | One long chat | 1.03–1.11× | 1.11–1.22× |
  | Chats of 300–1500 tokens | 1.04–1.23× | |
  | Chats of 32–160 tokens | 1.40–1.71× | |

- **The four causes (sections 6–7):**
  1. **Short chats: we launch too many, half-empty tiles.** We run one tile per query head. FA3 packs the 3 query heads that share a KV head into one tile.
     - At T=131072 with 32–160-token chats, our tiles are 60% full against FA3's 82%.
     - We launch 1.37× as many tiles as FA3 at D128 and 1.88× at D64.
     - These ratios match the kernel-time ratios: 1.38× at D128 and 1.62× at D64.
  2. **The D64 tile is smaller than the references'.** We use 128×128 with 2 compute warpgroups; FA3 and FA4 use 192×128 with 3.
  3. **The chat-list kernel is slow at B=1.** It takes 4.5–66 µs per call and grows with T, because it runs as one block per batch row. FA3's equivalent preparation kernel takes 2.4–9 µs.
  4. **At D128 with one 131072-token chat we fall 11% behind.** Our rate drops from 742 to 615 TFLOP/s between 32K and 128K; FA3 drops from 746 to 680. The likely cause is K/V being re-read for every query head (no GQA packing). This is not confirmed yet; it needs ncu.
- **One accuracy detail.** Our LSE error grows with chat length: 1.9e-6 at 2048, 1.8e-4 at 131072. FA3's is 5e-6 at 131072. We still pass with about 35× margin, but our row-sum accumulation is less precise than FA3's.

## 3. What was run

| Item | Value |
|---|---|
| GPU | p032 GPU 7 (H200), unlocked clocks; Nov 13, 12:31–12:42 |
| Our build | p032 tree = Nov 12 phase 5 state (nova-opt `dff63defe56f`, same libtensor) |
| References | FA3 `f63c472` (hopper/, built bf16, hdim 64/96/128/256); FA4 = `flash_attn.cute` at the same commit (Hopper path, JIT); cuDNN 9.19.0 + frontend 1.32.3; torch 2.11.0+cu128 |
| Shapes | B=1 packed row, T ∈ {2048, 4096, 8192, 16384, 32768, 65536, 131072}, 18 query heads / 6 KV heads, bf16, causal, scale 1/√D |
| Layouts | **one1**: one chat of T. **pk1**: chats of 300–1500 tokens (3 to 144 chats). **sh1**: chats of 32–160 tokens (22 to 1364 chats). The same chat lengths are fed to every library. Section 1 describes them in detail. |
| Timing | One warm-up, then 5 runs, under nsys. A run's time is the sum of all kernels of one forward call (host gaps excluded); the table shows the median. |
| Kernels per call | Ours: chat-list kernel + attention kernel. FA3: prepare + main. FA4: 1 kernel (3 for more than 256 chats). cuDNN: 3. |
| Run-to-run spread | Under 1% for every case except one1 at T=65536: 2.7–3.7% for both us and FA3, so read that row as ±3%. |
| Correctness | Ours: O and LSE of KV head 0's three query heads against an fp64 forward reference computed in the same process (`attn_ref_fwd`). Pass rule (FlashAttention's): \|ours − exact\| ≤ 2 \|bf16 attention − exact\| + atol. Every head is also checked for NaN / Inf. References: the same group against a torch fp64 reference (all rows of chats up to 4096 tokens, first and last 256 rows of longer ones). |
| Wall time | 11 min for both head dims |

Our graph is the trainer's form: the model's [B, T, heads, D] layout through transposes. The transposes fold into the kernel's loads, so there are no transpose kernels.

## 4. Results: head dim 128

Times are in µs (median of 5). **attn** = our attention kernel alone. **list** = our chat-list kernel. Each ratio is ours / FA3; a ratio below 1 means we are faster. **best** = the fastest of FA3, FA4 and cuDNN. TFLOP/s counts causal FLOPs (4·18·D·Σ L(L+1)/2) for the attention kernels alone.

| T | layout | chats | ours | attn | list | FA3 | FA4 | cuDNN | total / FA3 | kernel / FA3 | ours / best | TFLOP/s ours / FA3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | one1 | 1 | 43.3 | 38.7 | 4.6 | 43.9 | 45.5 | 62.0 | 0.99 | 0.93 | 0.99 | 500 / 467 |
| 2048 | pk1 | 3 | 28.3 | 22.9 | 5.4 | 28.3 | 30.8 | 39.8 | 1.00 | 0.89 | 1.00 | 334 / 296 |
| 2048 | sh1 | 22 | 26.4 | 19.5 | 6.9 | 20.7 | 22.4 | 27.2 | **1.28** | 1.07 | 1.28 | 55 / 59 |
| 4096 | one1 | 1 | 128.8 | 123.6 | 5.2 | 131.2 | 139.6 | 179.9 | 0.98 | 0.96 | 0.98 | 626 / 601 |
| 4096 | pk1 | 6 | 51.8 | 44.9 | 6.9 | 50.2 | 57.4 | 57.3 | 1.03 | 0.94 | 1.03 | 340 / 319 |
| 4096 | sh1 | 42 | 48.7 | 38.8 | 9.9 | 36.4 | 47.1 | 46.1 | **1.34** | 1.16 | 1.34 | 57 / 66 |
| 8192 | one1 | 1 | 448.7 | 443.0 | 5.7 | 452.8 | 471.7 | 574.2 | 0.99 | 0.98 | 0.99 | 698 / 687 |
| 8192 | pk1 | 9 | 106.7 | 98.4 | 8.3 | 101.6 | 117.6 | 117.6 | 1.05 | 0.99 | 1.05 | 394 / 391 |
| 8192 | sh1 | 87 | 88.9 | 77.3 | 11.6 | 63.9 | 93.1 | 80.4 | **1.39** | 1.28 | 1.39 | 55 / 70 |
| 16384 | one1 | 1 | 1700.8 | 1694.3 | 6.5 | 1697.1 | 1753.9 | 2058.6 | 1.00 | 1.00 | 1.00 | 730 / 730 |
| 16384 | pk1 | 21 | 201.3 | 190.6 | 10.7 | 182.8 | 220.4 | 210.9 | 1.10 | 1.06 | 1.10 | 370 / 391 |
| 16384 | sh1 | 173 | 162.7 | 148.3 | 14.4 | 117.7 | 209.3 | 153.1 | **1.38** | 1.31 | 1.38 | 57 / 74 |
| 32768 | one1 | 1 | 6673.7 | 6665.8 | 7.9 | 6638.6 | 6798.1 | 7810.3 | 1.01 | 1.00 | 1.01 | 742 / 746 |
| 32768 | pk1 | 37 | 403.0 | 385.9 | 17.1 | 404.8 | 451.7 | 406.1 | 1.00 | 0.96 | 1.00 | 396 / 381 |
| 32768 | sh1 | 342 | 314.8 | 293.2 | 21.6 | 221.6 | 409.6 | 289.2 | **1.42** | 1.36 | 1.42 | 58 / 78 |
| 65536 | one1 | 1 | 29112 | 29098 | 14.8 | 28137 | 29499 | 33676 | 1.03 (±3%) | 1.03 | 1.03 | 680 / 703 |
| 65536 | pk1 | 74 | 780.7 | 751.5 | 29.2 | 813.6 | 918.8 | 773.9 | 0.96 | 0.93 | 1.01 | 406 / 377 |
| 65536 | sh1 | 682 | 614.0 | 579.1 | 34.9 | 433.9 | 837.2 | 566.3 | **1.42** | 1.36 | 1.42 | 58 / 79 |
| 131072 | one1 | 1 | 128688 | 128656 | 31.3 | 116425 | 117703 | 146330 | **1.11** | 1.11 | 1.11 | 615 / 680 |
| 131072 | pk1 | 144 | 1572.6 | 1517.2 | 55.4 | 1705.3 | 1990.1 | 1532.1 | 0.92 | 0.89 | 1.03 | 407 / 364 |
| 131072 | sh1 | 1364 | 1225.5 | 1159.3 | 66.2 | 848.3 | 1717.8 | 1122.4 | **1.44** | 1.38 | 1.44 | 58 / 80 |

How the libraries compare at D128:
- **One long chat:** we match FA3 and beat FA4 and cuDNN up to 32K.
- **Medium chats:** from 32K on we are ahead of FA3 and within 3% of the fastest (cuDNN).
- **Short chats:** FA3 is clearly fastest; we are close to cuDNN; FA4 is the slowest from 8K on.

## 5. Results: head dim 64

| T | layout | chats | ours | attn | list | FA3 | FA4 | cuDNN | total / FA3 | kernel / FA3 | ours / best | TFLOP/s ours / FA3 / FA4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | one1 | 1 | 35.0 | 30.5 | 4.5 | 34.0 | 28.7 | 48.6 | 1.03 | 0.97 | 1.22 | 317 / 306 / 337 |
| 2048 | pk1 | 3 | 23.9 | 18.5 | 5.4 | 22.9 | 20.7 | 32.3 | 1.04 | 0.91 | 1.15 | 207 / 187 / 185 |
| 2048 | sh1 | 22 | 22.9 | 16.1 | 6.8 | 16.4 | 15.6 | 23.2 | **1.40** | 1.16 | 1.47 | 33 / 38 / 34 |
| 4096 | one1 | 1 | 99.9 | 95.1 | 4.8 | 96.3 | 90.2 | 133.9 | 1.04 | 1.01 | 1.11 | 407 / 412 / 429 |
| 4096 | pk1 | 6 | 41.8 | 35.4 | 6.4 | 36.4 | 36.5 | 44.2 | 1.15 | 1.04 | 1.15 | 216 / 224 / 209 |
| 4096 | sh1 | 42 | 37.0 | 29.0 | 8.0 | 26.5 | 29.6 | 38.0 | **1.40** | 1.23 | 1.40 | 38 / 47 / 37 |
| 8192 | one1 | 1 | 346.0 | 340.4 | 5.6 | 317.2 | 298.7 | 425.9 | 1.09 | 1.08 | 1.16 | 454 / 492 / 518 |
| 8192 | pk1 | 9 | 84.8 | 76.8 | 8.0 | 72.5 | 78.7 | 90.1 | 1.17 | 1.10 | 1.17 | 252 / 277 / 246 |
| 8192 | sh1 | 87 | 71.8 | 61.0 | 10.8 | 47.9 | 66.5 | 68.0 | **1.50** | 1.37 | 1.50 | 35 / 47 / 32 |
| 16384 | one1 | 1 | 1288.5 | 1281.9 | 6.6 | 1179.1 | 1099.2 | 1486.9 | 1.09 | 1.09 | 1.17 | 482 / 526 / 563 |
| 16384 | pk1 | 21 | 157.1 | 146.4 | 10.7 | 132.3 | 146.3 | 160.0 | 1.19 | 1.13 | 1.19 | 241 / 272 / 241 |
| 16384 | sh1 | 173 | 135.7 | 122.7 | 13.0 | 86.6 | 155.0 | 127.8 | **1.57** | 1.49 | 1.57 | 34 / 51 / 27 |
| 32768 | one1 | 1 | 4997.9 | 4990.1 | 7.8 | 4555.7 | 4207.0 | 5568.6 | 1.10 | 1.10 | 1.19 | 496 / 543 / 588 |
| 32768 | pk1 | 37 | 320.9 | 304.4 | 16.5 | 261.7 | 297.8 | 311.3 | 1.23 | 1.18 | 1.23 | 251 / 296 / 257 |
| 32768 | sh1 | 342 | 263.5 | 241.9 | 21.6 | 160.7 | 307.2 | 242.1 | **1.64** | 1.56 | 1.64 | 35 / 55 / 28 |
| 65536 | one1 | 1 | 19971 | 19959 | 12.3 | 17963 | 16679 | 22176 | 1.11 (±3%) | 1.11 | 1.20 | 496 / 551 / 593 |
| 65536 | pk1 | 74 | 630.1 | 601.5 | 28.6 | 526.6 | 616.0 | 593.8 | 1.20 | 1.15 | 1.20 | 253 / 292 / 247 |
| 65536 | sh1 | 682 | 513.7 | 478.9 | 34.8 | 308.7 | 623.2 | 474.8 | **1.66** | 1.60 | 1.66 | 35 / 56 / 27 |
| 131072 | one1 | 1 | 83891 | 83867 | 24.0 | 76202 | 71617 | 90676 | 1.10 | 1.10 | 1.17 | 472 / 519 / 553 |
| 131072 | pk1 | 144 | 1258.4 | 1203.9 | 54.5 | 1057.9 | 1331.0 | 1178.2 | 1.19 | 1.14 | 1.19 | 257 / 293 / 232 |
| 131072 | sh1 | 1364 | 1019.8 | 954.4 | 65.4 | 596.5 | 1272.9 | 944.9 | **1.71** | 1.62 | 1.71 | 35 / 57 / 26 |

How the libraries compare at D64:
- **One long chat:** FA4 is the fastest, 6–8% faster than FA3. We are 8–11% behind FA3 from 8K on.
- **Medium chats:** FA3 is fastest; we are 1.04–1.23× behind it.
- **Short chats:** the gap is the largest of all, 1.40–1.71×.

## 6. Correctness

Each cell is our max |error| / bf16 attention's max |error| over the 7 lengths. A value of 1.00 would mean as accurate as plain bf16 attention; the limit is 2.00.

| head dim | one1 | pk1 | sh1 | O, worst case | LSE, ours | LSE, bf16 attention |
|---|---|---|---|---|---|---|
| 128 | 0.77 | 0.55–0.91 | 0.49–0.63 | 1.56e-2 (pk1, T 65536 / 131072; bf16 attention 1.71e-2) | 9.5e-7 … 1.8e-4 | 6.2e-3 … 1.2e-2 |
| 64 | 0.63 | 0.55–0.66 | 0.52–0.67 | 1.12e-2 (sh1, T ≥ 32768; bf16 attention 1.71e-2 … 2.14e-2) | 9.5e-7 … 1.8e-4 | 6.3e-3 … 1.2e-2 |

No head has NaN or Inf anywhere. FA3, FA4 and cuDNN give O max errors of 4.6e-3 … 1.1e-2 on their inputs (the same distribution, a different draw).

**LSE grows with chat length.** For one chat our LSE error is:

| T | 2048 | 4096 | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|---|---|
| D128 | 1.9e-6 | 3.8e-6 | 5.7e-6 | 2.2e-5 | 4.3e-5 | 8.8e-5 | 1.8e-4 |
| D64 | 1.9e-6 | 2.9e-6 | 9.5e-6 | 1.8e-5 | 4.3e-5 | 9.2e-5 | 1.8e-4 |

- FA3's error barely grows: 1.3e-6 at 2048, 4.9e-6 at 131072.
- Short chats keep ours at ≤ 2.4e-6, so the error grows with the number of key tiles a row accumulates over.
- It passes with about 35× margin, but it shows our row-sum (softmax denominator) accumulation loses precision over long rows where FA3's does not. FA3 keeps per-thread row sums and reduces them once at the end. Section 8 item 5.

## 7. Why: our kernel next to FA3 and FA4

Sources:
- **Ours:** Tier 0 PTX + ptxas of the forward graphs (`runs/fwd_1113/tier0`) and the lowered host code.
- **FA3 / FA4:** the source study (`claude_handoff/fwd_1113/fa_fwd_study.md`, with file:line citations). It is source reading, not measurement.

| | Ours D64 | FA3 D64 | FA4 D64 | Ours D128 | FA3 D128 | FA4 D128 |
|---|---|---|---|---|---|---|
| Tile (query rows × keys) | 128×128 | **192×128** | **192×128** | 128×128 | 128×128 | 128×128 |
| Threads | 384: 1 loader + 2 compute warpgroups | 512: 1 + 3 | 512: 1 + 3 | 384: 1 + 2 | 384: 1 + 2 | 384: 1 + 2 |
| Registers, compute / loader | 232 / 40 | 160 / 32 | 160 / 32 | 232 / 40 | 240 / 24 | 224 / 40 |
| Shared memory | ~113 KB | 88 KB | 88 KB | ~225 KB | 160 KB | 160 KB |
| GQA | one tile per **query head**: K/V loaded 3× per KV head | 3 query heads **packed** in one tile: K/V once | packed | per query head | packed | packed |
| Scheduling | one block per (query head, 128-row tile); not persistent; tiles sorted most work first | **persistent**: 132 blocks take tiles from a counter, longest first, grouped so K/V stays in L2 | one block per tile, longest first | same as D64 | same as D64 | same as D64 |
| Blocks for one 131072 chat | 18,432 | 132 (12,288 tiles) | 12,288 | 18,432 | 132 (18,432 tiles) | 18,432 |
| O store | TMA | smem + per-row stores | TMA | TMA | per-row stores | per-row stores |
| Varlen preparation | chat-list kernel, **one 256-thread block per batch row** (bitonic sort of the tiles): 4.5–66 µs | prepare kernel: 2.4–9 µs | none (≤256 chats) | same as D64 | same | same |

What each gap comes from:

1. **Short chats (both head dims, the largest gap).**
   - **Our tiles are more numerous and emptier.** We cut each chat into 128-row tiles for each query head separately, so a 40-token chat becomes 18 tiles that are 31% full. FA3 packs the 3 query heads of a KV head into one tile, so the same chat becomes 6 tiles that are 94% full. Counted on the actual layouts:

     | layout, T | our tiles | FA3 tiles, D128 (128 rows) | FA3 tiles, D64 (192 rows) | fill: ours / FA3-128 / FA3-192 |
     |---|---|---|---|---|
     | sh1, 4096 | 990 | 714 | 498 | 58% / 81% / 77% |
     | sh1, 131072 | 30,726 | 22,482 | 16,362 | 60% / 82% / 75% |
     | pk1, 131072 | 19,728 | 18,822 | 12,726 | 93% / 98% / 97% |
     | one1, 131072 | 18,432 | 18,432 | 12,288 | 100% everywhere |

   - **The counts match the timings.** At sh1, T=131072 we have 1.37× FA3's tiles at D128 against a kernel-time ratio of 1.38×, and 1.88× its tiles at D64 against 1.62×.
   - **Packing alone is not the whole story.** FA4 also packs, but from T=8192 on it is slower than us on short chats. FA3's persistent scheduling also matters here: the next tile's loads overlap the current tile's epilogue, which counts when tiles are tiny.
   - **The chat-list kernel adds 7–66 µs** (point 3) on top of this.
2. **D64, all layouts: the tile.**
   - FA3 and FA4 use 192 query rows and 3 compute warpgroups at D64. At D64 the softmax is a larger share of the work than at D128, and a third warpgroup gives more room to hide it behind the tensor-core work.
   - We use the D128 shape (128 rows, 2 warpgroups), and on one long chat our kernel runs at 496 TFLOP/s against FA3's 551 and FA4's 593.
   - At D128, where our tile matches FA3's, we are at parity. That points at the tile shape rather than the mainloop.
   - For the medium and short layouts, the D64 gap adds up from this and point 1.
3. **The chat-list kernel at B=1.**
   - It builds and sorts the work-unit list with one thread block per batch row. At B=1 the whole list is sorted by one block, so its cost grows with T: 4.5 µs at 2048, 24–66 µs at 131072.
   - FA3's prepare kernel takes 2.4–9 µs at the same sizes.
   - At sh1, T=131072 it is 66 µs of our 377 µs gap to FA3 (D128).
4. **D128, one 131072-token chat (11%).**
   - We are at FA3's level up to 32K, then fall to 615 TFLOP/s; FA3 holds 680.
   - Likely cause: without GQA packing, each KV head's K/V (67 MB at T=131072) is streamed once per query head, three times in total. At these lengths that no longer fits in the 60 MB L2, while FA3 reads it once and groups heads to keep it in L2.
   - This is a hypothesis; an ncu comparison of L2 hit rate and DRAM bytes would confirm or rule it out.
5. **Where we are ahead.** Medium chats at D128 from 32K on (0.89–0.96 of FA3's attention-kernel time). FA3 and FA4 slow down as the chat count grows, and we do not. These measurements don't show which part of our design makes the difference.

## 8. What to improve in the forward (proposed order)

| # | Change | Expected effect (from the data above) | Head dims |
|---|---|---|---|
| 1 | **Pack the query heads of a KV head into one tile** (FA3/FA4's PackGQA) | Short chats: tile count −27% (D128) / −47% (D64 with item 2), towards FA3's 1.0×. K/V read once per KV head: probably the D128 131072 gap too. | all |
| 2 | **D64: 192-row tile with 3 compute warpgroups** | One long chat at D64: the 8–11% kernel gap to FA3, up to 17–20% to FA4 | 64 (also worth testing at 32 / 96) |
| 3 | **Chat-list kernel: use more than one block at B=1**, or build the list once per step (it is the same for every layer; check whether Nova already shares it) | At T=131072, 24–66 µs → FA3's ~9 µs: −15 to −57 µs per call | all |
| 4 | **Persistent scheduling** (132 blocks taking tiles longest first, the next tile's loads overlapping the epilogue) | The rest of the short-chat gap after item 1 | all |
| 5 | **LSE / row-sum accumulation**: find where the error grows (per-tile reduction across threads? log conversion?) and keep per-thread sums with one final reduction | Precision only | all |

Before changing anything, two measurements would confirm the hypotheses (GPU, about 10 min):
- ncu of our kernel vs FA3 at D128 one1 T=131072 (L2 hit rate, DRAM bytes) for item 1's long-T effect;
- ncu at D64 one1 T=32768 (tensor-pipe utilization and stall reasons) for item 2.

## 9. Not covered yet

- **Head dims 32, 96 and 256.** Their compile plans are known from Tier 0 (all lengths lower):

  | D | Our kernel |
  |---|---|
  | 32 | 1 warpgroup, 64×64 tile |
  | 96 | 128×128, 2 compute warpgroups |
  | 256 | generic path (mma.sync, no TMA, 128 threads); varlen 32 registers / 33 KB spill, dense 255 registers / 7.6 KB spill |

  For comparison:
  - FA3 runs D32 as D64 (padded); FA4 runs it exactly.
  - D96: 192×144 (FA3) / 192×128 (FA4).
  - D256: 128×80 on the fast path.

  The quick suite runs them with `DIMS="32 96 256"`. D128 and D64 took 5.2 and 5.7 min each; D256 will take longer, since our kernel is on the generic path.
- **Clocks were not locked.** Ours and the references ran back to back in the same session, so the ratios are fair, but absolute µs can move a few %.
- **Inputs and graph sizes.**
  - Our inputs and the references' are different random draws of N(0, 1). They are the same shapes and chat lengths, which is enough for timing and for the error bound.
  - Each of our graphs uses a chat list sized to its own layout. The trainer sizes it for the dataset, which only adds blocks that exit immediately.

## 10. Files and how to rerun

| What | Where |
|---|---|
| This report | `BluTrain/nov-doc/varlen-forward-study-nov13.md` (copy: `claude_handoff/fwd_1113/REPORT.md`) |
| Raw results (nsys summaries, tables, logs, chat lengths) | `claude_handoff/fwd_1113/results_quick/` (= p032 `/localstorage/mahesh/fwd1113/quick`) |
| FA3 / FA4 source study | `claude_handoff/fwd_1113/fa_fwd_study.md` |
| Scripts | `claude_handoff/fwd_1113/scripts/`: `run_fwdquick.sh` (driver: GPU check, watcher that stops on another job), `ref_fwd.py` (FA3 / FA4 / cuDNN), `fwd_summary.py` (nsys → per-call kernel sums), `fwd_table.py` |
| Suite changes (uncommitted, local == p032) | `attn_suite.cpp`: modes `fwdquick`, `fwdcheck`, `fwdtime`, `fwdemit`, `fwdlens`; edge1 uses 100 one-token chats below T=4096. `attn_ref.cu`: `attn_ref_fwd` (forward only). |
| Reference env on p032 | `/localstorage/mahesh/refs` (venv: torch, FA3 build, FA4, cuDNN frontend) |

Rerun on p032 (one GPU, about 5–6 min per head dim):

```bash
cd /localstorage/mahesh/fwd1113 && mkdir -p quick/logs && \
  DIMS="128 64" GPU=7 setsid nohup bash scripts/run_fwdquick.sh > quick/logs/run.out 2>&1 < /dev/null &
# results: quick/table.md, quick/time.txt, quick/fq_lines.txt, quick/ref_lines.txt, quick/logs/STATUS
```
