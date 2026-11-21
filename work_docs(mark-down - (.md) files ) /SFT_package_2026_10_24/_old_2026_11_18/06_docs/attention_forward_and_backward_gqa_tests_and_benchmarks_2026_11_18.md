# Attention forward and backward with several query heads per KV head: what I changed, how it compares with FA3, FA4 and cuDNN, and how you can check it (18/11/2026)

> **Update, 18/11/2026 evening.** I pushed the backward change to `dist_side_gautam_lottn` the same evening. The pushed
> files are the files of the patch next to this note plus 14 comment lines with new wording ("EXPERIMENT" -> "idea 1 /
> idea 2", and 3 old comments about the block order now say "rule 1"). I compiled both versions: the machine code is the
> same, so your test with the patch on TI f1a7d98 tests the pushed code. Before the push I also ran the pretraining
> trainer (1.6B production config, dense rows, 30 steps, runs pushed / new / new / pushed on one H200): new code 0.54%
> faster per step, the same step-0 loss, the loss differences no bigger than between two runs of the same code.

This note is for Uma Mahesh. It covers both of my attention kernels for packed rows, the forward and the backward.
Both were rebuilt in October 2026 (forward: sections 4.1 to 4.3; backward: section 6.1). The work your study started
then went in this order:

1. **The forward first.** Your forward study of 13/11/2026 (`varlen-forward-study-nov13.md`) named the cost of "one
   tile per query head" on rows of short chats and pointed at FA3's head packing. I measured your case with my kernel
   the same day and added "MX" with a "2% rule" to the forward (sections 4.4 to 4.8). That is pushed.
2. **Then the backward.** From 14/11 to 18/11/2026 I went after the same fact there: each KV head serves several query
   heads (3 in my 1.6B model, 2 in my 135M model). It mattered much more in the backward, because my backward did the
   work of each query head separately and then added the results of the heads together in extra kernels (sections 6
   to 11).
3. **Then both against FA3, FA4 and cuDNN** on 18/11/2026: 12 cases, forward, backward and both together, with the
   checks that the comparison is fair (sections 12 to 14).

**State.** The forward work is pushed (Tensor-Implementations `f1a7d98`). The backward change of 14/11 to 18/11 is
finished and tested on my side, but **it is not pushed yet**. I would like you to test it independently first.
Section 15 has the patch, ready builds on P035 and the exact commands. If your numbers differ from mine, please tell
me.

- **My kernels:** `GQA_fwd_sm90.cu` (forward), `GQA_bwd_vj2_sm90.cu` (backward) and the plan builder in
  `GQA_fwd_generic.cu`, in `Tensor-Implementations`, folder `src/Kernels/cuda/attention/arch/`.
- **Base:** my pushed code, BluTrain `d9606c2` on branch `dist_side_gautam_lottn`, with Tensor-Implementations
  `f1a7d98`. It has every forward change of this note, and every backward change up to 13/11.
- **The backward change to test:** `gqa_backward_ideas_1_2_4_vs_TI_f1a7d98.patch`, in the same folder as this note.
  3 files, 372 lines added, 31 removed. Every line I replaced is kept next to the new code as a comment, with the date
  and the reason.
- **The full October record** (every variant tried, kept or not):
  `attention_kernel_optimizations_h200_2026_10_10_to_2026_10_21.md`, same folder. **My 13/11 note to you:**
  `attention_packed_rows_our_experiments_for_nova_forward_study_2026_11_13.md`.
- **Every raw result:** section 16.

---

## 1. Summary

**What I changed, forward first:**

| Kernel | When | What | Result | State |
|---|---|---|---|---|
| forward | 16 to 19/10/2026 | rebuilt from v62 to QPU: FA3's loop order, full-width P·V, a persistent scheduler, a cheaper unit start and end, softmax placement | forward time from 1.375 × FA3's to 1.004 × (dense rows) and from 1.28 × to 0.865 × (packed rows) | pushed |
| forward | 13/11/2026 | MX: in a chat with an odd number of 64-row tiles, the idle compute group takes the same tile of another query head of the same KV head; plus the 2% rule | forward 21% to 26% less time on rows of chats of 64 tokens or less, 8% on chats of 32 to 160, 0 on long chats | pushed |
| backward | 10 to 16/10/2026 | rebuilt from Vj1d to Vj2 in FA3's layout, then tuned | main kernel 3,626 → 2,406.7 µs (dense rows), 3.0% faster than FA3's | pushed |
| backward | 14 to 18/11/2026 | ideas 1, 2 and 4 plus rules 1 and 2b (idea 3 built and dropped) | backward call 2.5% to 51.7% less time; training step 1.7% (135M) and 0.7% (1.6B) less time | tested, **not pushed: please test** |

The four backward changes of 14 to 18/11:

| Change | In one sentence |
|---|---|
| Idea 1 | In the first step of every block, half of the block's keys are hidden by the causal mask; that half no longer computes zeros. |
| Idea 2 | The first two 64-row query tiles of every chat get dQ from one block only, so that block writes dQ directly in bf16 instead of going through an fp32 buffer that must be zeroed, added into and converted. |
| Idea 4 | One block does the backward for all query heads of a KV head (3 at heads 18/6). It loads that KV head's keys and values once, adds the heads' dK and dV in its registers and writes the final dK and dV once. No partial buffers, no head-sum kernels. This is the idea that comes from your observation. |
| Rules 1 and 2b | A block order that keeps whole rows together in L2-sized sections; idea 4 only when the call has at least 2 KV heads. |

**What I measured on 18/11** (head dim 128, bf16, causal attention inside each chat, 8 rows of 4,096 tokens per call,
one H200, my final code against FA3, FA4 and cuDNN; below 1.00 = mine takes less time):

| | vs FA3 | vs FA4 | vs cuDNN | Section |
|---|---|---|---|---|
| Forward | 0.659 to 0.966 | 0.427 to 0.938 | 0.315 to 0.839 | 5, 13 |
| Backward | 0.357 to 0.916 | 0.371 to 0.914 | 0.217 to 0.805 | 13 |
| Forward + backward | 0.403 to 0.928 | 0.383 to 0.907 | 0.235 to 0.812 | 13 |

- In all 12 cases (2 head settings × 6 layouts) mine takes less time than each of the three, in the forward alone, the
  backward alone and both together.
- **Fair comparison** (section 14): every contender gets the same inputs; counting every kernel, counting every idle
  gap, or using each program's own timer instead of nsys gives the same order; all numbers are within the GPU's limits;
  nothing else ran on the GPU.
- **Correct** (section 10): every quick suite passes at 8 head settings; the full suites pass 332 of 333 checks at
  heads 4/2 and 18/6. The one remaining check is a 2-token padding piece where FA3 and FA4 give a dK bit-identical to
  mine; I recorded it as a known case.
- The big ratios appear only on rows of short chats (93 tokens on average): there every library runs far below the
  GPU's peak, because fixed costs per chat dominate. On rows of long chats mine is 4% (forward) and 9% (backward)
  faster than FA3.

---

## 2. Terms used in this note

- **Query, key and value heads.** Attention splits the model width into heads. Each query head reads one key head and
  one value head. I call that key / value pair a **KV head**.
- **GQA (grouped-query attention).** There are fewer KV heads than query heads; a group of G query heads shares one KV
  head. Query head h uses KV head h / G (integer division).
- **18/6 and 4/2.** Head settings written as "query heads / KV heads". 18/6 means 18 query heads and 6 KV heads, so
  G = 3: my 1.6B model. 4/2 means 4 query heads and 2 KV heads, so G = 2: my 135M model. Head dim is 128 in both.
- **Packed row, chat.** A training row of 4,096 tokens holds several chats back to back. Attention is causal inside a
  chat (token i sees tokens 0 to i of its own chat) and never crosses into another chat.
- **Dense row.** A row that is one sequence (pretraining). My benchmark "dense" shape is 8 rows × 4,096, heads 18/6.
- **Forward, backward.** The forward computes the output O and the log-sum-exp LSE of every query row. The backward gets
  dO, the gradient of O, and computes dQ, dK and dV.
- **Who adds into what.** dQ of a query row gets a contribution from every key that row sees. dK and dV of a key get a
  contribution from every query that sees it, and with GQA from all G query heads of the group.
- **Tile.** A rectangle of the attention matrix handled as one unit of work: 64 or 128 query rows by 128 keys.
- **Block, warpgroup, SM.** A CUDA thread block runs on one SM (the H200 has 132 streaming multiprocessors). My kernels'
  blocks have 384 threads, which are 3 warpgroups of 128: one loader and two compute groups.
- **Forward unit.** One 128-row query pair (two 64-row tiles) of one chat and one query head. Each compute group owns
  one of the two 64-row tiles.
- **Backward step.** A backward block owns 128 keys of a chat and walks the queries that see them, 64 rows at a time.
  One step is one 64-row query tile against the block's 128 keys.
- **PackGQA.** FA3's and FA4's way to serve GQA in the forward: the query rows of all G heads of a KV head go into one
  128-row tile, so K and V are loaded once for the G heads.
- **MX.** My smaller version of the same idea, in the forward only (section 4.5).
- **fp32 buffer, reduce-add, conversion.** When several blocks add into the same output, each block adds fp32 values into
  a buffer with atomic or bulk "reduce-add" instructions. The buffer must be zeroed first, and a separate kernel converts
  it to bf16 at the end.
- **Partial, head sum.** A partial is one query head's dK (or dV) before the G heads of a group are added. A head sum
  adds the G partials into the final dK (or dV).
- **D.** For each query row, D = the sum over the head dim of dO × O. The backward needs it (FlashAttention's formula). A
  "prep" kernel computes it.
- **TMA.** The H200's tensor memory accelerator, a copy engine that moves tiles between global and shared memory.
- **Race.** Two or more builds run one after another in the same session, several rounds, the order swapped every
  round, free clocks. A % is the change against the previous build (minus = less time). "8/8" = faster in 8 of 8
  rounds.
- **Pushed code, final code.** Pushed means my code as it is on the remote today (TI `f1a7d98`). Final means the pushed
  code plus the backward change of 14 to 18/11.

---

# Part A. The forward (done first)

## 3. How my forward works, and how FA3, FA4 and cuDNN do it

**Sources.** Mine: `GQA_fwd_sm90.cu` and my 13/11 note (section 1). FA3: FlashAttention commit `94e22c9`, folder
`hopper/`; the `flash_attn_3` 3.0.0 I benchmarked was built on P035 from a copy whose files I quote below are
byte-identical to that commit (md5 checked). FA4: the installed `flash-attn-4` 4.0.0b34,
`flash_attn/cute/interface.py`. cuDNN 9.24: closed source, so only its kernel names in my traces. "Today's kernel" =
the kernel name in the nsys traces of my 18/11 campaign.

| | Mine | FA3 | FA4 | cuDNN 9.24 |
|---|---|---|---|---|
| Tile, head dim 128 | 128 query rows × 128 keys | 128 × 128 (today's kernel: `TileShape_MNK` = 128, 128, 128) | 128 × 128 (`_tile_size_fwd_sm90` returns `FwdConfig(128, 128, ...)` for head dim up to 128) | 64 × 128 (kernel name `..._flash_fprop_..._64x128x128_...`) |
| Query heads of one KV head | one tile per query head; MX lets two heads share a half-empty pair (section 4.5) | PackGQA: always on for packed rows: `heuristics.h` line 11 `if (varlen_q) return true;`; today's kernel has the PackGQA flag set (template order in `mainloop_fwd_sm90_tma_gmma_ws.hpp` lines 31 to 33: causal 1, Varlen 1, PackGQA 1) | PackGQA on by default: `pack_gqa = qhead_per_kvhead > 1` (`interface.py` lines 999 to 1000; the exception in lines 1001 to 1010 is only for SM100 / SM110 at head dim 256) | not visible |
| Shared-memory stages for K / V | 3 | 2 (today's kernel: `Stages` = 2) | not checked | not visible |
| Scheduling | persistent: 132 blocks take units from a counter, heaviest first, in L2-sized sections (FA3's scheduler idea, since 16/10/2026) | persistent: `VarlenDynamicPersistentTileScheduler` (today's kernel), plus one `prepare_varlen_num_blocks` kernel per forward + backward iteration | not checked; on the short-chat rows its trace also has 4 small `cu_blocks` kernels per forward + backward iteration | not visible |
| Chat list | built on the CPU once per micro-batch, uploaded with one copy, reused by every layer and by the backward | `cu_seqlens` offsets and the per-call prepare kernel | `cu_seqlens` offsets | `cu_seqlens` offsets |

So FA3 and FA4 both pack the query heads of a KV head into one forward tile on packed rows, and mine does not (except
for MX). Section 5 shows that mine still takes less time in all 12 cases of today's benchmark; section 4.7 shows the
one place where FA3's packing wins: rows where every chat has about 40 tokens or less.

## 4. The forward changes, in order

### 4.1 Where it started (v62)

Measured on 13/10 and 16/10/2026 (kernel time, free clocks): my forward v62 took 1.375 × FA3's time on dense rows and
1.28 × on packed rows. At the same clock (ncu): 1,084 µs against FA3's 877 µs; the tensor cores were busy 64% of the
time against FA3's 83%. The main reason: I waited for QK of tile n, ran the softmax with nothing on the tensor cores,
and only then started P·V of tile n.

### 4.2 October 2026: v62 to QPU

Every change passed the test suite (296 checks then) before its race. "Dense" = 8 rows × 4,096, heads 18/6.
"Packed" = 8 real packed SFT rows (`real_rows_752`). Each % is against the build before it; minus = less time.

| Change (date) | What it does | Dense | Packed |
|---|---|---|---|
| G1 (16/10) | P·V as 8 full-width `m64n128k16` matrix products instead of 16 half-width ones | −3.34% (12/12) | −3.22% (12/12) |
| F1 (16/10) | FA3's loop order: the softmax of tile n runs while P·V of tile n − 1 is still on the tensor cores | −0.50% (12/12) | −1.82% (12/12) |
| G5 (16/10) | persistent scheduler: units through a work counter, heaviest first, in sections sized by FA3's L2 rule | −2.63% (12/12) | −3.55% (12/12) |
| F2 (16/10) | always rescale O (no vote, no branch); the unit and stage broadcast once | −3.84% (12/12) | −1.54% (12/12) |
| F3c (16 and 17/10) | FA3's hand-over of the next unit (the producer publishes it during the current unit's loads), with my 40 / 232 register split | no change | −0.84% to −1.55% |
| K1 (17/10) | the P·V wait after the whole softmax, the diagonal tile peeled, two P register sets | −0.95% (8/8) | 0.00% |
| K4b (17/10) | the output written into the last tile's K buffer (stmatrix), then 16-byte global stores; the write at a unit end cost 4,230 cycles per pair before | −5.26% (8/8) | −7.37% (8/8) |
| K5q (17/10) | 16 of the exponentials before the P·V wait, one P register set | −0.36% (7/8) | −4.33% (8/8) |
| LHOd (19/10) | FA3's L2 cache hints, in the dense kernel only | −0.34% | 0.00% |
| SM3m (19/10) | the exponentials spaced like FA3's `softmax.h` and FA3's causal-mask order; the exp wait per step went from 367 to 245 cycles | −3.51% (8/8) | −4.48% (8/8) |
| PF (19/10) | the first K / V tile of a unit is loaded before waiting for Q to be free (FA3's order); the wait for it went from 1,498 to 48 cycles | −2.60% (8/8) | −3.01% (8/8) |
| QPU (19/10) | Q prefetched into L2 when a pair starts, and the producer publishes the decoded unit | −0.81% (7/8) | −0.13% |

Tried and NOT kept (each measured; details in the October record):

| Change | Result |
|---|---|
| 2 K / V stages as FA3 has (K1n2) | dense +39%, packed +22% more time |
| FA3's split K and V pipelines (SKV) | dense +0.20%, packed +0.87% |
| FA3's 24 / 240 register split with the hand-over (F3b) | dense +1.11% |
| O written by one TMA store, as FA3's epilogue (PFT) | dense +1.60% (0/8) |
| every 8th or 4th exponential by a polynomial (P8w, P4w) | +2.7% and +7.1% |
| different softmax math for dense and packed rows (K8, K9) | failed "a packed chat gives the same bits as the same chat alone" |
| barrier wait forms, descriptors built once, row-split softmax, diagonal tile first (W1, W2, L1, R8 to R32, QPD) | no gain |

### 4.3 Where October ended

Scoreboard of 19/10/2026 evening (my forward QPU, nsys kernel times, free clocks; ratio = my time / the library's time):

| | Dense forward | Packed forward |
|---|---|---|
| vs FA3 | 1.004 (984.2 against 980.4 µs: a tie) | 0.865 |
| vs FA4 | 0.919 | 0.781 |
| vs cuDNN | 0.863 | 0.827 |

In my production pretraining trainer (21/10/2026, one H200), the new attention against the old kernels: forward 18.9%
faster, backward 40.5% faster; the whole step, together with loss-kernel fixes pushed at the same time, 5.98% less time.

### 4.4 13/11/2026: your study, and your case measured with my kernel

Your study found your Nova forward kernel 1.28 to 1.44 times slower than FA3 on rows of 32 to 160-token chats, and named
the cause "one tile per query head". My forward tiles per query head too, and I had never timed that case. I timed it
the same day (P032, 8 rows × 4,096, heads 18/6, head dim 128, two rounds, nsys kernel times):

| Layout | Direction | Mine | FA3 | FA4 | cuDNN | vs FA3 | vs FA4 | vs cuDNN |
|---|---|---|---|---|---|---|---|---|
| chats of 32 to 160 | forward | 191.8 µs | 219.2 | 380.9 | 315.6 | 0.875 | 0.503 | 0.607 |
| chats of 32 to 160 | forward + backward | 1,186.7 µs | 1,555.9 | 1,660.8 | 1,572.2 | 0.763 | 0.715 | 0.755 |
| chats of 300 to 1,500 | forward | 341.2 µs | 375.3 | 442.1 | 420.8 | 0.909 | 0.772 | 0.811 |

So on short chats my forward was already ahead of FA3, although it does not pack heads. The reason is the fixed cost
of a unit. In layouts where every chat has the same length, every unit is the same:

| Every chat is | Rows used in the 128-row pair | Units per call | Forward time | Time per unit |
|---|---|---|---|---|
| 40 tokens | 31% | 14,832 | 282.9 µs | 19.1 ns |
| 64 tokens | 50% | 9,216 | 180.4 µs | 19.6 ns |
| 120 tokens | 94% | 5,040 | 129.8 µs | 25.7 ns |

A unit with 120 rows has 9 times the query-key pairs of a 40-row unit and costs only 35% more: on short chats the
forward time is almost all fixed cost per unit (Q load, the wait for the first K / V tile, the output write, the hand-over
to the next unit). Section 4.2's changes G5, K4b, PF and QPU are the ones that cut that fixed cost.

### 4.5 MX: two query heads share one pair

**Waste.** Each compute group owns one 64-row tile of a pair. A chat with an odd number of 64-row tiles (every chat of
64 tokens or less, 129 to 192, 257 to 320, ...) leaves the second group of its last pair with no rows: it only runs the
handshake.

**Change.** That idle group takes the same last tile of another query head of the same KV head. Both heads read the same
K and V, so the unit's K / V loads do not change, and both groups run the same number of key tiles.

- Changed: the work list (which units exist) and the unit decode in three places; the KV-head count became a real
  argument of `gqa_chat_plan_build`.
- Not changed: the main loop, the softmax, the mask, the output write, and the dense kernel (its machine code is
  identical, 0 lines differ).
- With 3 query heads per KV head, two heads share and the third keeps its old unit. With 2 per KV head every such tile
  is shared.

128-row pairs per call (8 × 4,096, heads 18/6):

| Layout | Before | With MX | FA3-style packing (rows of 3 heads in one tile) |
|---|---|---|---|
| chats of 32 to 160 | 7,974 | 6,798 (−14.7%) | 5,658 (−29.0%) |
| chats of 300 to 1,500 | 5,040 | 4,878 (−3.2%) | 4,728 (−6.2%) |
| one chat of 4,096 per row | 4,608 | 4,608 | 4,608 |

**Race of MX against my forward before it** (P032 GPU 5, one session, 4 rounds, forward time):

| Layout | Before | MX code, old work list | MX code, MX work list |
|---|---|---|---|
| chats of 32 to 160 | 0.1956 ms | +1.06% (slower in 4 of 4) | −8.01% (faster in 4 of 4) |
| chats of 300 to 1,500 | 0.3461 ms | −0.54% | −2.68% (4 of 4) |
| one chat of 4,096 per row | 0.9949 ms | −0.14% | −0.06% |
| real rows, group 1 | 0.4541 ms | +0.48% | −0.71% (3 of 4) |
| real rows, group 2 | 0.3910 ms | +0.90% | −1.05% (4 of 4) |
| real rows, group 3 | 0.5187 ms | +0.66% | −0.39% (3 of 4) |

**ncu** on the short-chat layout, before → MX: kernel time 198.7 → 181.5 µs; SM cycles 342,995 → 309,049 (−9.9%);
instructions 60.55 M → 60.01 M (−0.9%); DRAM read 410 → 375 MB (−8.5%). The same instructions in 10% fewer cycles: the
fixed cost of a pair is paid once for two heads.

**By chat length** (20 lengths, 3 rounds, forward):

| Chat length | Forward with MX |
|---|---|
| 16 / 32 / 48 / 64 tokens | −25.6% / −24.9% / −24.1% / −21.3% |
| 160 / 192 | −10.1% / −8.3% |
| 320 / 448 / 704 | −7.2% / −4.5% / −3.9% |
| 1,088 / 2,112 | −3.0% / −1.1% |
| 128, 256, 384, 512, 1,024, 2,048, 4,096 (an even number of 64-row tiles) | no change |
| rows of 96-token chats that end in one 64-token chat | +0.9% (slower) |

### 4.6 The 2% rule

The one loss above: a single chat with an odd tile count sent the whole work list to the MX kernel, whose longer unit
decode costs 0.5% to 1.1% on rows with many chats, while almost nothing was shared.

**Rule.** The work-list builder counts, from the chat lengths alone, how many units would share a pair. It uses MX only
when that is at least 2% of the units of the plain list (`BLU_GQA_MX_MIN_PCT`); otherwise it builds the list of before
and the kernel of before runs. Measured by that share: 0.78% gave +0.9%, 1.75% gave +0.2% (not significant), 2.1% gave
−0.5%, 4% to 7% gave −1% to −3%, 33% gave −21% to −26%. With the rule the 96-token rows are back at 0 (−0.02%) and the
gains stay (160 tokens −9.7%, 192 tokens −9.0%). `BLU_GQA_MX=0` turns MX off.

### 4.7 Against FA3 by chat length

P032, 13/11/2026, 8 rows × 4,096, heads 18/6, nsys kernel times, FA3 `94e22c9` with PackGQA on. "Every chat N" = every
chat of the row has N tokens. Ratio = my time / FA3's time.

| Chats in the row | My forward / FA3, before MX | With MX / FA3 | Forward + backward / FA3, with MX |
|---|---|---|---|
| every chat 32 tokens | 1.536 | 1.146 | 0.643 |
| every chat 64 tokens | 0.788 | 0.620 | 0.645 |
| every chat 128 tokens | 0.656 | 0.656 | 0.783 |
| every chat 192 tokens | 0.774 | 0.702 | 0.771 |
| every chat 704 tokens | 0.859 | 0.825 | 0.882 |
| every chat 2,048 tokens | 0.911 | 0.911 | 0.948 |
| chats of 32 to 160, mixed | 0.875 | 0.802 | 0.752 |
| one chat of 4,096 per row | 0.950 | 0.950 | 0.959 |

- **At 32 tokens per chat FA3's forward wins:** 226 µs against 259 µs for mine with MX. FA3 needs the same number of
  tiles at 32 tokens as at 64 (3 heads × 32 rows fit one tile), and its time stays the same (226 and 228 µs). Mine needs
  twice the pairs at 32 as at 64. So at about 40 tokens and below, packing all 3 heads into one tile decides the
  forward, as your study says.
- Forward + backward together mine is ahead at every length, also at 32 tokens, because my backward is far faster than
  FA3's on such rows.
- From 64 tokens up my forward is ahead of FA3 at every length measured.

### 4.8 Tests and the push

- My suite at heads 18/6 (quick 280 and full 333 checks) and at 4/2 (quick 273), long rows of 8,192 tokens (69 checks):
  all passed, including "a packed chat gives the same bits as the same chat alone".
- SFT trainer, 135M, 30 steps: the same step-0 losses as before, step time unchanged (106.1 to 106.5 ms); the trainer
  used the MX kernel for 5,432 of 5,432 forward launches.
- Pushed on 13/11/2026: Tensor-Implementations `f1a7d98`, BluTrain `d9606c2`.
- In a whole SFT step on my real data MX changes nothing measurable: its gain is on rows with many short chats.
- Not built: FA3's packing of all 3 heads into one tile. For chats of about 40 tokens and less it is what the forward
  would need.

## 5. The forward in today's benchmark (18/11/2026)

Forward main kernel, µs per call, the mean of 2 rounds (the setup is in section 12; the 6 layouts in section 13.1).
"Pushed" is the same forward kernel as "mine" (the forward did not change after 13/11); it differs only through the
clock (section 13.6).

| Heads | Layout | Mine | Pushed | FA3 | FA4 | cuDNN | Mine / FA3 | Mine / FA4 | Mine / cuDNN |
|---|---|---|---|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 178.3 | 175.0 | 218.5 | 379.6 | 314.5 | 0.816 | 0.470 | 0.567 |
| 18/6 | mix, rows 8-15 | 344.4 | 336.4 | 376.0 | 442.8 | 419.7 | 0.916 | 0.778 | 0.821 |
| 18/6 | mix, rows 16-23 | 1,006.1 | 993.3 | 1,041.2 | 1,135.0 | 1,199.2 | 0.966 | 0.886 | 0.839 |
| 18/6 | real, rows 0-7 | 452.1 | 441.1 | 489.4 | 563.2 | 571.4 | 0.924 | 0.803 | 0.791 |
| 18/6 | real, rows 8-15 | 405.9 | 383.5 | 428.3 | 489.9 | 484.9 | 0.948 | 0.829 | 0.837 |
| 18/6 | real, rows 16-23 | 526.1 | 514.7 | 563.1 | 632.0 | 638.4 | 0.934 | 0.832 | 0.824 |
| 4/2 | mix, rows 0-7 | 41.4 | 42.1 | 62.8 | 97.0 | 131.5 | 0.659 | 0.427 | 0.315 |
| 4/2 | mix, rows 8-15 | 81.3 | 80.0 | 94.9 | 99.8 | 109.5 | 0.857 | 0.815 | 0.743 |
| 4/2 | mix, rows 16-23 | 226.6 | 224.0 | 237.8 | 241.6 | 289.1 | 0.953 | 0.938 | 0.784 |
| 4/2 | real, rows 0-7 | 101.5 | 102.8 | 120.6 | 126.6 | 142.8 | 0.841 | 0.802 | 0.710 |
| 4/2 | real, rows 8-15 | 94.1 | 92.3 | 108.4 | 111.8 | 119.1 | 0.868 | 0.842 | 0.790 |
| 4/2 | real, rows 16-23 | 118.5 | 118.4 | 135.9 | 141.4 | 160.0 | 0.872 | 0.838 | 0.740 |

- Mine is faster than all three in all 12 cases: 3.4% to 34.1% less time than FA3, 6.2% to 57.3% less than FA4, 16.1%
  to 68.5% less than cuDNN.
- The biggest gap is on short chats, where every chat leaves room in a tile and the fixed cost per unit (section 4.4)
  decides; at 4/2 MX shares every odd tile (G = 2).
- On long chats (mix rows 16-23) the gap to FA3 is 3.4% (18/6) and 4.7% (4/2): there the tiles are full, and the
  October changes of section 4.2 (loop order, full-width P·V, unit start and end, softmax placement) are what remain.
- Useful work per second (section 14.6): on long chats at 18/6 mine 615 TFLOPS, FA3 594, FA4 545, cuDNN 516 (the H200's
  bf16 peak is 989); on short chats mine 92, FA3 75, FA4 43, cuDNN 52.

---

# Part B. The backward (done after the forward)

## 6. How my backward worked before

### 6.1 October 2026: Vj1d to Vj2

Main kernel on dense rows: 3,626 µs → 2,406.7 µs (−33.6%), from 1.46 × FA3's time to 0.97 ×. The kept changes (each %
against the build before; races of 13 shapes):

| Change (date) | What it does | Result |
|---|---|---|
| Vj2, first version (12/10) | a new kernel in FA3's layout: 128 keys per block, a loader warp, a dQ-writer warp and 2 compute warpgroups; P and dS in registers; dS from the fp32 P | 5.2% to 13.0% faster than Vj1d; 605 / 605 checks; dK and dV bit-identical every run |
| Vj2 fixed (13/10) | shared memory used through real shared loads and stores (they had compiled to generic loads); descriptors built once | 18% to 22% faster than Vj1d; instructions 1,096 M → 840 M |
| Fix 1, dQ path (13/10) | dQ in thread order with 128-bit stores and one bulk reduce-add per group (FA3's scheme) | −5.7% (13/13) |
| Fix 4, block order (13/10) | heaviest key blocks first, 8 (batch, head) pairs per L2 section | −3.3% (10/13) |
| Fix 5, epilogue swizzle (13/10) | dK / dV staging in the 128-byte swizzle, swizzled TMA store | −1.1% (12/13) |
| G1 (13/10) | dV and dK as one full-width matrix product | −0.3% |
| G3 (13/10) | the LSE loads before the first score matrix product | −0.6% (9/13) |
| B-2a (14/10) | FA3's wait placement: wait only for dV before starting the dK product | −1.9% (13/13); equal to FA3 |
| G4 (14/10) | dQ stored unscaled; the scale applied once in the conversion kernel | −2.7% (13/13); 2,406.7 µs against FA3's 2,481.4: 3.0% faster |
| B-1 (14/10) | the fp32 dQ buffer zeroed inside the prep kernel | in the code |
| Vj2 for packed rows, design A (15 and 16/10) | Vj2 takes the per-chat plan and becomes the default | 26% / 22% / 12% faster than Vj1d on long / middle / short-chat rows; packed = alone bit-exact |

### 6.2 The pushed backward, and where its time went

Per backward call, 5 kernels:

| Kernel | What it does |
|---|---|
| prep (`bwd_prep_vj2_chats`) | zeroes the fp32 dQ buffer; computes D; copies LSE into 64-row slots |
| main (`gqa_bwd_vj2_kernel`) | one block per (128-key tile of a chat, query head); every step adds its dQ into the fp32 buffer; at the end the block writes its own bf16 dK and dV partial for its query head |
| head sum, twice (`gqa_dkdv_greduce_vj2`) | for dK and then for dV: adds the G partials of each KV head into the final bf16 values |
| dQ conversion (`convert_dq_frag_vj2`) | fp32 dQ × softmax scale, written as bf16 dQ |

So at heads 18/6 the 3 query heads of a KV head each ran their own block over the same 128 keys. The same K and V were
loaded 3 times, and the 3 partials (3 times the size of the final dK and dV) were written and then read again by the
head-sum kernels.

Measured on 14/11/2026 (P032, nsys, one backward call, median of 50 calls, µs):

| Layout | Heads | Whole call | prep | main | head sums | dQ conversion | Outside the main kernel |
|---|---|---|---|---|---|---|---|
| chats of 32 to 160 tokens | 18/6 | 994.5 | 196.8 | 560.6 | 104.9 | 128.1 | 43% |
| real packed rows | 18/6 | 1,546 | 171 | 1,158 | 105 | 108 | 25% |
| chats of 32 to 160 tokens | 4/2 | 242.8 | 45.7 | 130.1 | 30.9 | 30.4 | 44% |
| real packed rows | 4/2 | 347.6 | 39.5 | 246.5 | 29.9 | 26.7 | 28% |

On rows of short chats almost half of the backward time was spent outside the main kernel. This table set the order of
the ideas below.

## 7. The backward changes of 14 to 18/11, one by one

For each change: what was wasted, what I changed, and its effect alone. "Effect" is the time of the backward call
against my pushed code (minus = less time), from races of 4 rounds: my test program in timing mode with the two
libraries alternating, one layout per run, about 4 s of GPU work per run, CUDA-event times. The six race layouts are 8
rows × 4,096 tokens each (section 13.1 describes them in full):

- **sh:** chats of 32 to 160 tokens (353 chats);
- **pk:** chats of 300 to 1,500 tokens (42 chats);
- **one:** one chat of 4,096 tokens per row;
- **real 1, real 2, real 3:** three groups of 8 real packed rows of my SFT data.

### 7.1 Idea 1: the masked half of the first step does no work

**Waste.** A block owns keys k0 to k0 + 127 of a chat. Its first step holds queries k0 to k0 + 63. Compute group 1's
keys (k0 + 64 to k0 + 127) all come after every one of these queries, so the causal mask sets that group's whole score
matrix to zero in this step. Group 1 still ran the step in full: the two matrix products for the scores and for dP,
2,048 exponentials, and the dV and dK matrix products, all producing exact zeros.

**Change.** In that step group 1 now only writes zeros into its 64 rows of dS (the dQ matrix products of both groups
read all 128 rows), runs its dQ product (head columns 64 to 127, fed by group 0's dS rows) and hands that dQ on. Its dK
and dV get nothing for this step; before, they had exact zeros added. The barriers between the groups are unchanged.

**Values.** The same bits as before: only additions of zero are removed.

**Effect alone** (P032, 14/11/2026, heads 4/2): −3.05 / −2.74 / −1.40 / −2.02 / −2.23 / −1.59% for sh / pk / one /
real 1 / real 2 / real 3.

### 7.2 Idea 2: direct dQ for the first two query tiles of every chat

**Waste.** dQ of a 64-row query tile gets contributions from every key tile at or before it. Key tile kt of a chat
starts at query tile 2 × kt (128 keys are 2 query tiles), so query tiles 0 and 1 of a chat get dQ from key tile 0 only,
which is one block. Still, those tiles went through the fp32 path: zeroed by the prep kernel, reduce-added by the main
kernel, converted by the conversion kernel.

**Change.** The first two steps of key tile 0 (one step if the chat has a single query tile) store their dQ straight to
the output in bf16, with the softmax scale applied: the same rounding the conversion kernel does. The plan builder marks
those tiles (the 4th number of each dQ tile entry is now 1 for a direct tile; it used to be always 0). The prep kernel
does not zero them and the conversion kernel skips them. Rows without chats (dense rows) are not affected.

For a chat of at most 128 tokens (at most 2 query tiles), every dQ tile is now direct.

**Effect alone** (P032, 14/11/2026): heads 4/2: −19.16 / −1.85 / −0.37 / −1.79 / −1.70 / −0.59%; heads 18/6: −22.09 /
−2.89 / −0.40 / −1.60 / −1.66 / −0.83%.

**Ideas 1 and 2 together** add up. Heads 18/6 (P032, 16/11/2026, 4 rounds): −26.05 / −5.02 / −1.90 / −3.67 / −3.66 /
−2.65%.

### 7.3 Idea 3 (dropped): two query heads share a one-step block

**Idea.** A chat with an odd number of 64-row query tiles has, for every query head, a last block of one step in which
group 1 has no keys. Idea 3 merged the last blocks of two query heads of one KV head into one block (group 0 = head a,
group 1 = head b): the backward version of MX.

**Result.** It was correct (the quick suites pass at every head setting), but measured against ideas 1 + 2 it helped
only on sh and lost elsewhere. Heads 4/2: −1.42% on sh, and 0.9% to 2.3% MORE time on pk and the real rows. Heads 18/6:
−1.78 / +0.67 / −0.10 / +0.39 / +0.71 / +0.38%. My reading of why: the separate kernel launch, and losing the small
one-step blocks that used to fill the main kernel's last wave. Dropped. Idea 4 leaves nothing to share anyway.

### 7.4 Idea 4: all query heads of a KV head in one block (your observation)

**Waste.** The G query heads of a group all use the same keys and values, but each had its own block for the same
128 keys:

- the keys and values were loaded G times, once per query head;
- each block wrote its own bf16 dK and dV partial: G times the size of the final result, written once and read once
  more;
- then 2 head-sum kernels added the partials.

**Change.** A block is now one (128-key tile of a chat, KV head):

1. it loads that KV head's keys and values once;
2. it runs the steps of all G query heads one after another (G × steps). The loader, the dQ writer and both compute
   groups count steps across the head boundary, so the double-buffered shared-memory stages and the barriers simply
   continue into the next head;
3. dK and dV stay in the compute groups' fp32 registers over all G heads;
4. the block writes the final bf16 dK and dV rows of the KV head once, by TMA store;
5. the partial buffers are not allocated and the 2 head-sum kernels are not launched.

Ideas 1 and 2 happen at the first steps of every head inside this loop.

**Rounding.** dK and dV are now rounded once (after the fp32 sum over the heads) instead of twice (each partial to bf16,
then the sum). That is also the arithmetic of FA3 and FA4: fp32 accumulation, one conversion (section 9). In my tests,
4 known dV checks that failed by a small margin on the pushed code now pass (section 10.2).

**Plan builder** (`gqa_chat_plan_build`, host code). It now also builds a per-KV-head block list. Each entry is {row,
chat start, chat length, key tile, first query head of the group, that head's dQ slot, slot step from one head to the
next, G}. The list is appended at the end of the plan's single GPU allocation. Its place and length are kept in a small
table keyed by the plan's block-list pointer, not in a new field of `GqaChatPlan`: programs that hold a `GqaChatPlan` by
value (my test program does) would break if the struct changed. `gqa_chat_plan_free` removes the entry.

**Dense rows** (no chat list). The grid becomes (S / 128) × B × Hkv blocks instead of (S / 128) × B × Hq.

**Machine code.** The new path is a new template flag, `HSUM`, so it compiles into new kernel variants. Adding them
changed nothing in the 10 kernel variants of the ideas 1 + 2 build: 0 lines of machine code differ (laptop check, nvcc
13.0, sm_90a). Ideas 1 and 2 themselves do change the main kernel against the pushed code. The new variant uses 168
registers per thread with no spills.

**Effect alone** (P015, 17/11/2026, heads 4/2): −24.0 / −7.6 / −6.1 / −1.1 / −7.3 / +0.4%. The weak last group
(real 3) led to rule 1.

### 7.5 Rule 1: block order

**Problem.** The blocks run in the order of the plan's list. My list order was: sections of a fixed number of (row,
head) pairs, and inside a section the longest work first. With idea 4 the pairs are (row, KV head) and each one is G
times bigger, so the old section size no longer kept a row's work together in L2.

**Measured** (P015, 17/11/2026, ideas 1 + 2 + 4 against my pushed code, heads 18/6, backward call):

| Order | sh | pk | one | real 1 | real 2 | real 3 |
|---|---|---|---|---|---|---|
| one list, no sections | −51.4% | +4.9% | +0.2% | −3.7% | +7.6% | +7.8% |
| sections of 6 pairs | −52.0% | −17.5% | −5.6% | −12.4% | −15.8% | −9.7% |

**Rule 1.** Sections of whole rows: as many rows as fit in about 80 MB, counting G × S × 1 KB per (row, KV head) pair,
rounded down to a power of 2. At S = 4,096 that gives 8 pairs at 4/2 and 6 pairs at 18/6. The switch
`BLU_VJ2_HSUM_SECT` overrides it (n pairs per section, or −1 for one list).

### 7.6 Rule 2b: idea 4 only when the call has at least 2 KV heads

**Problem.** With context parallelism (CP, Ulysses form) each GPU gets a share of the heads. My 135M model (4/2) with
CP = 2 leaves 1 KV head per GPU. Idea 4 makes G times fewer blocks; there that means 144 blocks on 132 SMs, and the
trainer step took 2.0% MORE time (P015, 17/11/2026).

**First rule, wrong.** I first chose the path by the number of blocks. That broke the suite's bit-identity checks: a
chat packed in a row and the same chat run alone took different paths (one rounding against two), so their bits
differed.

**Rule 2b.** The path is chosen by the shape only: idea 4 runs when the call has at least `BLU_VJ2_HSUM_MINKV` KV heads
(default 2). Otherwise the per-query-head path with ideas 1 and 2 runs. After this rule: the CP 135M step 0.0% (the
same as the pushed code), the CP 1.6B step −0.5%.

### 7.7 Switches and messages

| Switch | Default | Effect |
|---|---|---|
| `BLU_VJ2_HSUM=0` | 1 | idea 4 off: the per-query-head path with ideas 1 and 2 |
| `BLU_VJ2_HSUM_MINKV=n` | 2 | idea 4 only if the call has at least n KV heads |
| `BLU_VJ2_HSUM_SECT=n` | 0, which means rule 1 | block order: n pairs per section, or −1 for one list |
| `BLU_VJ2_DKDV_F32=1` (existed before) | 0 | fp32 partials; they exist only on the per-query-head path, so idea 4 is off |

Ideas 1 and 2 have no switch: they are always on.

Messages: nothing is printed while idea 4 runs. When it is off, one line per reason, once per process. For example:

```text
[vj2] idea 4 (head sums inside the block) OFF: BLU_VJ2_HSUM=0 -> per-query-head path (ideas 1 + 2)
```

The four reasons are: `BLU_VJ2_HSUM=0`; `BLU_VJ2_DKDV_F32=1`; a plan built without the KV-head count (it has no
per-KV-head list); fewer KV heads than `BLU_VJ2_HSUM_MINKV`.

## 8. The backward code change (the patch)

| File | Lines | What changed |
|---|---|---|
| `GQA_bwd_vj2_sm90.cu` | +288 / −28 | Main kernel: template flag `HSUM`; the step loop over G × steps; idea 1 inside the loop; idea 2 (a new kernel argument, the bf16 dQ output); the final dK / dV epilogue. Prep kernel: does not zero direct dQ tiles. Conversion kernel: skips direct tiles. Launcher: the idea-4 decision (switches, rule 2b, messages); the `HSUM` launch (packed rows use the per-KV-head list, dense rows a grid ordered by rule 1); with idea 4, no partial buffers and no head-sum launches. |
| `GQA_fwd_generic.cu` | +72 / −3 | Plan builder only, host code: the direct flag in the dQ tile list; the per-KV-head block list with rule 1; the side table and its lookup `gqa_vj2_hsum_list`. No forward kernel changed. |
| `test_varlen_attention.cu` | +12 / −0 | One KNOWN entry (section 10.3). |

## 9. How FA3, FA4, cuDNN and my code do the backward with GQA

**Sources:** as in section 3. FA3's files quoted here (`flash_api.cpp`, `flash_bwd_launch_template.h`,
`epilogue_bwd.hpp`, `mainloop_bwd_sm90_tma_gmma_ws.hpp`) are byte-identical between commit `94e22c9` and the copy the
benchmarked `flash_attn_3` was built from. Kernel counts per call come from the nsys traces of my campaign (section
14.5).

| | FA3 | FA4 | cuDNN 9.24 | Mine, pushed | Mine, final |
|---|---|---|---|---|---|
| Backward tile, head dim 128, causal (queries × keys) | 64 × 128 (`flash_bwd_launch_template.h` lines 349 to 351) | 64 × 128 (`interface.py`, function `_tile_size_bwd_sm90`) | 64 × 64 (kernel name `..._flash_bprop_..._64x64x128_...`) | 64 × 128 | 64 × 128 |
| One block per | (key tile, query head) | (key tile, query head); the source says "pack_gqa backward not yet supported in bwd" (`interface.py` lines 2532 to 2533) | not visible; the kernel list has per-head results reduced by a separate kernel | (key tile, query head) | (key tile, KV head), with all G query heads inside |
| dK and dV over the G heads of a group | fp32 buffers `dk_accum`, `dv_accum` made with `torch::zeros` (`flash_api.cpp` lines 1491 to 1497); each block adds its head's dK and dV with bulk reduce-add or `atomicAdd` (`epilogue_bwd.hpp` lines 471, 480, 502, 511); 2 conversion kernels to bf16 (`flash_bwd_launch_template.h` lines 250 to 252) | the same scheme; the source says "GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads accumulate into the same dK/dV" (`interface.py` lines 2600 to 2604); fp32 `torch.zeros` buffers (lines 2631 to 2640); post-process kernels | 2 `fmha_reduce_head` kernels per call | a bf16 partial per query head, then 2 head-sum kernels | summed in fp32 registers inside the block, written once in bf16; no buffer, no extra kernel |
| dQ | fp32 buffer `dq_accum` (`flash_api.cpp` lines 1485 to 1489), zeroed by the prep kernel (`flash_bwd_preprocess_kernel.h` lines 21 and 231), conversion kernel (`flash_bwd_launch_template.h` line 225) | fp32 buffer (`interface.py` lines 2569 to 2598) and a post-process kernel | fp32, then `convert_dq_to_16bits` | fp32 buffer zeroed by prep, conversion kernel | the first 2 query tiles of every chat written directly in bf16; the other tiles as before |
| Kernels per backward call (traces) | 2 zero-fills, prep, main, dQ conversion, 2 dK / dV conversions; plus 1 scheduler kernel per forward + backward iteration | 2 zero-fills, prep, main, 3 post-process; on the short-chat rows also 4 small `cu_blocks` kernels per forward + backward iteration | 2 setup, 2 memsets, prep, main, dQ conversion, 2 head reductions; plus 8 small PyTorch kernels per forward + backward iteration | prep, main, 2 head sums, dQ conversion | prep, main, dQ conversion |

**What this means.** FA3 and FA4 handle the 3 query heads of a group with 3 separate blocks that all add into shared
fp32 buffers. Those buffers must be zeroed before and converted after, every call. cuDNN computes per-head results and
reduces them in separate kernels. My final code keeps the 3 heads in one block, so dK and dV need no buffer and no
extra kernel. dQ needs the fp32 path only for tiles that really have several contributors.

Note the contrast with the forward (section 3): there FA3 and FA4 pack the heads and mine does not; in the backward
neither FA3 nor FA4 packs heads, and my final code does.

## 10. Correctness tests of the backward change

### 10.1 The test program

- **File:** `Tests/kernel_tests/test_varlen_attention.cu`.
- **Truth:** for each chat, the exact answer in fp64. **Floor:** the same attention computed in plain bf16 (PyTorch's
  math at bf16, no fp32 upcast).
- **Pass rule,** per chat and per output (O, LSE, dQ, dK, dV): |mine − truth| ≤ 2 × max|floor − truth| + 2 × a small
  fp32 term. This is FlashAttention's own rule. The printed "ratio" is error / bound; a check passes when it is ≤ 1.
- **Bit-identity checks:** a chat packed in a row gives exactly the same bits as the same chat run alone (group G11), and
  a row's filler does not change its chats (G6c).
- **Data loads** include: real SFT rows (D1, D10); fixed lengths; 1 + 4,095 tokens; 4,096 one-token chats; padding tails
  of 0 to 255 tokens; a row of padding only; the same chat at many offsets; long rows up to 32,768 tokens (G14).
- **Quick suite** (`VARLEN_QUICK=1`): 273 checks at most head settings, 280 at 18/6. **Full suite:** 333 checks.

### 10.2 Results

| Test | Result |
|---|---|
| Quick suite at 4/2, 8/4 (my 350M model), 12/4, 16/4 and 6/6 | 273 of 273 each (P035, 17/11/2026) |
| Quick suite at 18/6 | 280 of 280 |
| Quick suite at 2/1 and 5/1 | pass |
| Long rows, T = 32,768, heads 18/6 (G14) | 69 of 69 |
| Full suite at 4/2 and at 18/6 | 332 of 333 each; the one check is the case of section 10.3 |
| The push build (my pushed clone + the patch), 18/11/2026 | quick suite 4/2: 273 passed, 0 failed; quick suite 18/6: 280 passed, 0 failed; the case of 10.3 shown as KNOWN; each of the 3 off reasons I could trigger (`BLU_VJ2_HSUM=0`, `BLU_VJ2_DKDV_F32=1`, heads 2/1) prints its one line, and the default prints none |

Two checks got better, both because dK and dV are now rounded once:

- "G11 D5, heads 12/4, dV" had ratio 1.015 on the pushed code and now passes;
- 4 dV checks of load D10 had ratio 1.171 on the pushed code (kept in the suite as KNOWN entries); D10's dV is now
  0.711.

### 10.3 The one check over the bound, and why I recorded it as known

**Where.** Group G11, load D10 (all real SFT rows, every chat checked alone), heads 18/6, dK of a 2-token padding piece:
row 1444, tokens 4,094 and 4,095. Ratio 1.136 (error 1.765e-2, bound 1.553e-2).

**Why.** The piece has 2 tokens. The first token's dS is 0. The second token's dS comes from dP − D, two nearly equal
numbers, so rounding decides the result. D = rowsum(dO × O) with O in bf16 is FlashAttention's formula.

**FA3 check** (P035, 18/11/2026). I exported the exact input bytes of that 8-row batch and ran FA3 (5 times), FA3 in
deterministic mode and FA4 on them. Their dK is bit-identical to mine: 1,536 of 1,536 values, ratio 1.136.
FlashAttention-2 inside PyTorch and my old per-head path give 0.919; cuDNN gives 0.931. The worst element: truth
1.68548; mine, FA3 and FA4 1.703125 (2.26 bf16 units in the last place); my old path, cuDNN and FA2 1.695312 (1.26).

**Training.** Padding positions get dO = 0 in training, so dK there is exactly 0. The test puts random dO on padding to
test it anyway.

**Decision.** A KNOWN entry with bound 1.136 × 1.01, in the test file (the third file of the patch). The case is still
checked in every run, and a bigger error fails.

## 11. Speed of the backward change against my pushed code

### 11.1 Backward call

Two separate measurements of the same change:

- **17/11 races:** my test program, the pushed and final libraries alternating, 4 rounds, CUDA-event times (P035 GPU 1);
- **18/11 campaign:** the nsys-based campaign of section 12 (P035 GPU 2).

| Layout | 4/2, 17/11 race | 4/2, 18/11 campaign | 18/6, 17/11 race | 18/6, 18/11 campaign |
|---|---|---|---|---|
| sh: chats of 32 to 160 | −46.9% | −47.0% | −51.7% | −50.9% |
| pk: chats of 300 to 1,500 | −16.9% | −17.4% | −16.8% | −17.7% |
| one chat of 4,096 per row | −6.4% | −7.0% | −4.7% | −5.1% |
| real 1 | −2.5% | −2.5% | −12.1% | −12.1% |
| real 2 | −14.4% | −14.8% | −15.7% | −14.7% |
| real 3 | −2.6% | −2.6% | −9.5% | −9.4% |

The two methods agree within about 1 point. Dense rows (no chats), heads 18/6, 17/11 race: −5.25%.

### 11.2 Training step

**Setup.** My SFT trainer, real SFT data (rows of 4,096 tokens), the same production settings for both sides, about 30
steps. Step time = the median of steps 11 to 29. Two runs per side. P035, 17/11/2026.

| Model | GPUs | Pushed, 2 runs | Final, 2 runs | Change |
|---|---|---|---|---|
| 135M | 1 | 105.24 / 105.70 ms | 103.18 / 104.09 ms | −1.7% |
| 1.6B shape | 1 | 791.78 / 790.43 ms | 785.70 / 785.77 ms | −0.7% |
| 135M, CP = 2 | 2 | 52.51 / 52.02 ms | 52.51 / 52.03 ms | 0.0% (1 KV head per GPU, so rule 2b turns idea 4 off) |
| 1.6B shape, CP = 2 | 2 | 279.17 / 279.16 ms | 278.56 / 276.76 ms | −0.5% |

**Losses.** The step-0 losses are identical. Over the 30 steps, the largest train-loss difference to the reference run
is 0.0004 to 0.0013 for the final code, and 0.0004 to 0.0009 between two runs of the pushed code: the same size.

**Why the step gain is small.** The attention backward is about 5% of a 1.6B step (my estimate of 14/11/2026), so even
−10% of it is about −0.5% of a step.

---

# Part C. Both kernels against FA3, FA4 and cuDNN (18/11/2026)

## 12. How I measured

### 12.1 Hardware and software

| Item | Value |
|---|---|
| Box, GPU | P035, GPU 2: NVIDIA H200, 141 GB, power limit 700 W, driver 595.91.07 |
| Clocks | free, not locked (the GPU picks its clock under the power limit, as in training) |
| My code | final library md5 `437b8663` (the push build minus the message change; the message change only decides when a line is printed), pushed library `be273224`, built with CUDA 13.4; the forward kernel is the same in both |
| FA3 | `flash_attn_3` 3.0.0, built from FlashAttention `94e22c9` `hopper/` (section 3) |
| FA4 | `flash-attn-4` 4.0.0b34 (CuTe DSL 4.8.1) |
| cuDNN | 9.24.0, through PyTorch 2.14.1 (CUDA 13.0) |
| Date | 18/11/2026, 14:38 to 15:59 IST |

### 12.2 Shape

8 rows × 4,096 tokens per call (32,768 tokens), head dim 128, bf16, causal attention inside each chat. Heads 18/6 (my
1.6B model) and 4/2 (my 135M model). The 6 layouts are described in section 13.1.

### 12.3 The contenders and how each is called

| Contender | How it is called |
|---|---|
| Mine, final | my test program in timing mode with `VARLEN_BENCH_IMPL=ours_A_plan` (the packed path, as the trainer uses it): forward, then backward, with a CUDA-event wait after each call |
| Mine, pushed | the same test program with the pushed library (the library is loaded at run time) |
| FA3 | `flash_attn_interface._flash_attn_forward` and `_flash_attn_backward` with `cu_seqlens`, causal, deterministic off |
| FA4 | `flash_attn.cute.interface._flash_attn_fwd` and `_flash_attn_bwd` with `cu_seqlens`, causal |
| cuDNN | `torch.ops.aten._cudnn_attention_forward` and `_cudnn_attention_backward` with the packed tensors and the chat offsets (the same call PyTorch's own varlen path makes), the KV heads passed directly. My benchmark script checks a cuDNN variant against FA3 on the same inputs (O, dQ, dK, dV) before it times it; that check ran when I added this contender (10/2026), and the timing runner of 18/11 does not repeat it |

For the three rivals:

- K and V have the real KV-head count (6 or 2); they are not copied up to the query-head count;
- the tensors are (tokens, heads, 128);
- the chat list is THE SAME as mine: my test program exports the chats of each layout, and the rivals' loader checks
  that they cover every row with no gap;
- their loop: 20 warm-up iterations (FA4 compiles there), then forward + backward back to back, with no wait.

**Iterations.** Per layout, set from my final code's time so that one run has about 5 s of GPU work. All 5 contenders
run the same number of iterations.

### 12.4 The runner (`rivals20.sh`)

- One process per contender per layout.
- 2 rounds: mine final, mine pushed, FA3, FA4, cuDNN; then the same in reverse order.
- One nsys session per plan of 3 layouts (30 runs), so nsys starts only once per plan.
- A "holder" process keeps the GPU initialised between runs (P035 has persistence mode off), so no run pays the GPU
  start-up.
- A watcher logs every other process on the GPU every second.
- Before any GPU work, CPU checks: duplicate chat lines removed, every layout must cover its rows exactly.

### 12.5 What is timed

Kernel times from nsys (CUDA trace), per run:

- **forward** = the forward main kernel;
- **backward call** = from the first helper kernel before the backward main kernel to the last helper kernel after it,
  gaps included;
- the median over the calls after the first 20; then the mean of the 2 rounds.

120 runs, 0 failed, about 78 minutes in total.

## 13. Results in every case

### 13.1 The 12 cases

6 layouts, each at heads 18/6 and at heads 4/2. The lengths are from my layout list (`LAYOUTS.txt`, 13/11/2026); the
averages are from the exported chat lists.

| Layout | Chats | Chat lengths (tokens) | Average length | What it stands for |
|---|---|---|---|---|
| mix, rows 0-7 ("short chats", sh) | 353 | 3 to 160: chats of 32 to 160 tokens, a few shorter where a chat is cut at the end of a row | 93 | rows full of short chats: the hardest case for every library |
| mix, rows 8-15 ("medium chats", pk) | 42 | 161 to 1,499 | 780 | rows of medium chats |
| mix, rows 16-23 ("long chats", one) | 8 | 4,096, one chat per row | 4,096 | long documents: the attention matrix is fully used |
| real, rows 0-7 (real 1) | 41 | 27 to 3,004; rows 0 to 7 of my packed SFT data | 799 | real training rows, start of the data |
| real, rows 8-15 (real 2) | 36 | 40 to 2,660; rows 15,256 to 15,263 | 910 | real training rows, middle |
| real, rows 16-23 (real 3) | 23 | 202 to 3,109; rows 30,512 to 30,519 | 1,425 | real training rows, end |

### 13.2 Forward + backward, every case

Each number is my final code's time divided by the other contender's time, for one forward + backward call (the
campaign's kernel times). Below 1.00 means mine is faster. In brackets: how much less time mine takes than that
contender.

| Heads | Layout | vs FA3 | vs FA4 | vs cuDNN | vs my pushed code |
|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 0.428 (57.2% less) | 0.402 (59.8% less) | 0.424 (57.6% less) | 0.570 (43.0% less) |
| 18/6 | mix, rows 8-15 | 0.795 (20.5% less) | 0.787 (21.3% less) | 0.771 (22.9% less) | 0.865 (13.5% less) |
| 18/6 | mix, rows 16-23 | 0.928 (7.2% less) | 0.907 (9.3% less) | 0.812 (18.8% less) | 0.964 (3.6% less) |
| 18/6 | real, rows 0-7 | 0.818 (18.2% less) | 0.825 (17.5% less) | 0.770 (23.0% less) | 0.910 (9.0% less) |
| 18/6 | real, rows 8-15 | 0.815 (18.5% less) | 0.817 (18.3% less) | 0.782 (21.8% less) | 0.896 (10.4% less) |
| 18/6 | real, rows 16-23 | 0.861 (13.9% less) | 0.853 (14.7% less) | 0.809 (19.1% less) | 0.932 (6.8% less) |
| 4/2 | mix, rows 0-7 | 0.403 (59.7% less) | 0.383 (61.7% less) | 0.235 (76.5% less) | 0.597 (40.3% less) |
| 4/2 | mix, rows 8-15 | 0.733 (26.7% less) | 0.718 (28.2% less) | 0.668 (33.2% less) | 0.867 (13.3% less) |
| 4/2 | mix, rows 16-23 | 0.890 (11.0% less) | 0.852 (14.8% less) | 0.762 (23.8% less) | 0.949 (5.1% less) |
| 4/2 | real, rows 0-7 | 0.817 (18.3% less) | 0.824 (17.6% less) | 0.752 (24.8% less) | 0.977 (2.3% less) |
| 4/2 | real, rows 8-15 | 0.744 (25.6% less) | 0.755 (24.5% less) | 0.711 (28.9% less) | 0.889 (11.1% less) |
| 4/2 | real, rows 16-23 | 0.854 (14.6% less) | 0.845 (15.5% less) | 0.778 (22.2% less) | 0.980 (2.0% less) |

### 13.3 Forward and backward separately, every case

Same ratios, split into the forward main kernel (fwd), the whole backward call (bwd) and both together (f+b).

| Heads | Layout | vs FA3: fwd / bwd / f+b | vs FA4: fwd / bwd / f+b | vs cuDNN: fwd / bwd / f+b | vs pushed: fwd / bwd / f+b |
|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 0.816 / 0.365 / 0.428 | 0.470 / 0.381 / 0.402 | 0.567 / 0.388 / 0.424 | 1.019 / 0.491 / 0.570 |
| 18/6 | mix, rows 8-15 | 0.916 / 0.762 / 0.795 | 0.778 / 0.790 / 0.787 | 0.821 / 0.756 / 0.771 | 1.024 / 0.823 / 0.865 |
| 18/6 | mix, rows 16-23 | 0.966 / 0.916 / 0.928 | 0.886 / 0.914 / 0.907 | 0.839 / 0.804 / 0.812 | 1.013 / 0.949 / 0.964 |
| 18/6 | real, rows 0-7 | 0.924 / 0.790 / 0.818 | 0.803 / 0.833 / 0.825 | 0.791 / 0.763 / 0.770 | 1.025 / 0.879 / 0.910 |
| 18/6 | real, rows 8-15 | 0.948 / 0.778 / 0.815 | 0.829 / 0.813 / 0.817 | 0.837 / 0.765 / 0.782 | 1.058 / 0.853 / 0.896 |
| 18/6 | real, rows 16-23 | 0.934 / 0.840 / 0.861 | 0.832 / 0.860 / 0.853 | 0.824 / 0.805 / 0.809 | 1.022 / 0.906 / 0.932 |
| 4/2 | mix, rows 0-7 | 0.659 / 0.357 / 0.403 | 0.427 / 0.371 / 0.383 | 0.315 / 0.217 / 0.235 | 0.985 / 0.530 / 0.597 |
| 4/2 | mix, rows 8-15 | 0.857 / 0.699 / 0.733 | 0.815 / 0.691 / 0.718 | 0.743 / 0.647 / 0.668 | 1.017 / 0.826 / 0.867 |
| 4/2 | mix, rows 16-23 | 0.953 / 0.871 / 0.890 | 0.938 / 0.827 / 0.852 | 0.784 / 0.755 / 0.762 | 1.012 / 0.930 / 0.949 |
| 4/2 | real, rows 0-7 | 0.841 / 0.810 / 0.817 | 0.802 / 0.831 / 0.824 | 0.710 / 0.764 / 0.752 | 0.987 / 0.975 / 0.977 |
| 4/2 | real, rows 8-15 | 0.868 / 0.710 / 0.744 | 0.842 / 0.730 / 0.755 | 0.790 / 0.687 / 0.711 | 1.019 / 0.852 / 0.889 |
| 4/2 | real, rows 16-23 | 0.872 / 0.848 / 0.854 | 0.838 / 0.847 / 0.845 | 0.740 / 0.790 / 0.778 | 1.000 / 0.974 / 0.980 |

Against the three rivals, none of the 108 numbers reaches 1.00. The worst are 0.966 (forward), 0.916 (backward) and
0.928 (both).

### 13.4 Times per call

Microseconds per call, the mean of the 2 rounds. f+b = forward main kernel + whole backward call. The last column is the
median GPU clock while busy, for mine / my pushed code / FA3. The forward of every contender is in section 5.

| Heads | Layout | Mine fwd | Mine bwd | **Mine f+b** | Pushed f+b | FA3 f+b | FA4 f+b | cuDNN f+b | MHz |
|---|---|---|---|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 178 | 488 | **666** | 1,169 | 1,555 | 1,658 | 1,571 | 1920 / 1980 / 1980 |
| 18/6 | mix, rows 8-15 | 344 | 1,069 | **1,414** | 1,635 | 1,779 | 1,797 | 1,834 | 1658 / 1702 / 1808 |
| 18/6 | mix, rows 16-23 | 1,006 | 2,982 | **3,988** | 4,135 | 4,298 | 4,397 | 4,909 | 1620 / 1635 / 1662 |
| 18/6 | real, rows 0-7 | 452 | 1,414 | **1,866** | 2,050 | 2,280 | 2,261 | 2,424 | 1650 / 1702 / 1796 |
| 18/6 | real, rows 8-15 | 406 | 1,226 | **1,631** | 1,820 | 2,003 | 1,998 | 2,087 | 1612 / 1710 / 1792 |
| 18/6 | real, rows 16-23 | 526 | 1,630 | **2,157** | 2,315 | 2,505 | 2,529 | 2,664 | 1628 / 1680 / 1732 |
| 4/2 | mix, rows 0-7 | 41 | 128 | **169** | 283 | 420 | 441 | 721 | 1980 / 1980 / 1980 |
| 4/2 | mix, rows 8-15 | 81 | 244 | **326** | 376 | 445 | 454 | 488 | 1815 / 1860 / 1898 |
| 4/2 | mix, rows 16-23 | 227 | 660 | **886** | 934 | 996 | 1,040 | 1,164 | 1650 / 1680 / 1692 |
| 4/2 | real, rows 0-7 | 101 | 357 | **458** | 469 | 561 | 556 | 610 | 1868 / 1815 / 1890 |
| 4/2 | real, rows 8-15 | 94 | 278 | **372** | 419 | 500 | 493 | 523 | 1785 / 1838 / 1908 |
| 4/2 | real, rows 16-23 | 119 | 399 | **517** | 528 | 606 | 612 | 665 | 1792 / 1778 / 1826 |

### 13.5 Case by case: where the margin comes from

The numbers in this section are from the kernel-by-kernel count of section 14.5 (all kernels of one iteration, round
1, µs). They differ from 13.4 by about 1% because 14.5 adds kernel times and 13.4 uses the backward-call span.

**Short chats (mix, rows 0-7).** 353 chats of 93 tokens on average, so every chat has at most 3 query tiles and at most
2 key tiles. Every library runs at 3% to 9% of the GPU's peak here (section 14.6): the work per chat is tiny and the
fixed costs per block and per call dominate.

| Heads 18/6, µs per iteration | Forward | Backward main | Backward helpers | Total |
|---|---|---|---|---|
| Mine, final | 180 | 311 | 178 (prep 159, dQ conversion 19) | 669 |
| Mine, pushed | 175 | 560 | 429 (prep 196, dQ conversion 128, head sums 105) | 1,163 |
| FA3 | 219 | 801 | 532 (prep 182, dQ conversion 131, dK/dV conversions 108, zero-fills 105, scheduler 6) | 1,552 |
| FA4 | 380 | 823 | 562 (post-process 262, prep 189, zero-fills 104, small kernels 8) | 1,766 |
| cuDNN | 314 | 635 | 626 (head reductions 280, dQ conversion 166, prep 158, setup 13, other 10) | 1,575 |

Against FA3: helpers 354 µs less, main kernel 490 µs less, forward 39 µs less.

- **Helpers:** with chats of at most 160 tokens almost every dQ tile is direct (idea 2), so my dQ conversion has almost
  nothing left to do (19 µs against FA3's 131). Idea 4 needs no dK / dV buffer, so there is nothing to zero and nothing to
  convert (FA3: 105 + 108 µs).
- **Main kernel:** one block per KV head instead of one per query head (idea 4: K and V loaded once instead of 3 times,
  no fp32 adds for dK and dV); fewer dQ adds (idea 2); and the masked half of every block's first step does no work
  (idea 1), which for a chat of at most 128 tokens is a large part of the block.
- **Forward:** the fixed cost per unit (section 4.4) and MX (section 4.5).
- **At heads 4/2:** mine 167 µs, FA3 417, cuDNN 726. cuDNN's helpers alone take 397 µs, more than my whole call.

**Medium chats (mix, rows 8-15).** 42 chats of 780 tokens on average. Each chat has many query tiles, so idea 2's "first
two tiles" are a small share; idea 4 still removes the whole dK / dV buffer path.

- Heads 18/6: mine 1,432 µs (forward 350, main 818, helpers 264) against FA3 1,791 (381, 1,008, 402): main 190 µs less,
  helpers 138 µs less, forward 31 µs less.
- Heads 4/2: mine 324 against FA3 439.

**Long chats (mix, rows 16-23).** One chat of 4,096 tokens per row: the attention matrix is fully used and every library
is near its best (18/6: mine 615 TFLOPS forward and 519 backward, FA3 594 and 475; section 14.6).

| Heads 18/6, µs per iteration | Forward | Backward main | Backward helpers | Total |
|---|---|---|---|---|
| Mine, final | 1,019 | 2,738 | 278 (prep 173, dQ conversion 105) | 4,034 |
| FA3 | 1,057 | 2,907 | 386 (prep 154, dQ conversion 106, dK/dV conversions 76, zero-fills 47, scheduler 4) | 4,350 |

The main kernel takes 169 µs less (5.8% less than FA3's). The helpers take 108 µs less: FA3's zero-fills and dK / dV
conversions (123 µs) have no counterpart in mine. The forward takes 38 µs less. Heads 4/2: mine 889 µs against FA3 994.

**Real rows (three groups).** Chats of 27 to 3,109 tokens, averages 799 to 1,425.

- Heads 18/6: mine 1,632 to 2,188 µs against FA3 2,017 to 2,507 (13.9% to 18.5% less time in 13.2). Per call, my helpers
  are 264 to 274 µs against FA3's 395 to 404, my main kernel is 168 to 241 µs shorter, my forward 26 to 39 µs shorter.
- Heads 4/2: mine 373 to 514 µs against FA3 496 to 604 (14.6% to 25.6% less).
- **The small gain against my pushed code at 4/2, rows 0-7 and 16-23 (2.3% and 2.0%).** At G = 2 a KV head is shared by
  only 2 query heads. My main kernel gets longer (rows 0-7: 266 to 294 µs, because it now does the head sum), while the
  helpers get shorter (97 to 60 µs). Net, 465 to 455 µs. At 18/6 (G = 3) the same rows gain 9.0%.

### 13.6 The forward against my pushed code: 1% to 6% slower, from the clock

- The forward kernel is the same in both (only host code of the plan builder changed in its file).
- In every layout except mix 4/2 rows 0-7, all runs hit the 700 W power limit. My final backward does more work per
  clock cycle, so the GPU lowers its clock for the whole loop, the forward included. Examples: mix 18/6 rows 0-7, 1,920
  MHz against 1,980 MHz for the pushed code; real 18/6 rows 8-15, 1,612 against 1,710 MHz.
- Where no run hit the limit (mix 4/2 rows 0-7, every contender at 1,980 MHz), my final forward takes 0.985 of the pushed
  code's time. On 17/11 a forward-only race (no backward in the loop) gave −0.27% to +0.14%, which is noise.
- My code ran at a LOWER clock than the rivals in most layouts (real 18/6 rows 8-15: about 1,612 against 1,790 MHz) and
  still took less time.

### 13.7 The same numbers as on other days

Forward + backward against FA3:

| Run | 18/6 mix rows 0-7 | 18/6 mix rows 8-15 | 18/6 real rows 0-7 | 18/6 real rows 8-15 | 18/6 real rows 16-23 |
|---|---|---|---|---|---|
| 13/11, my pushed code (P015) | 0.752 | 0.920 | 0.899 | 0.908 | 0.929 |
| 18/11, my pushed code (this campaign) | 0.751 | 0.919 | 0.899 | 0.909 | 0.924 |
| 17/11, my final code (P035 GPU 1) | 0.425 | 0.792 | 0.817 | 0.799 | not measured |
| 18/11 morning, my final code (P035 GPU 6, the runner's validation) | not measured | not measured | 0.816 | not measured | 0.864 |
| 18/11, my final code (this campaign) | 0.428 | 0.795 | 0.818 | 0.815 | 0.861 |

The 2% on real rows 8-15 comes from the GPU: on 17/11 GPU 1 ran my code at 1,635 MHz, and on 18/11 GPU 2 ran it at
1,612 MHz, while FA3's clock was about the same on both (1,785 and 1,793 MHz).

## 14. Is the comparison fair? The checks

### 14.1 Nothing else ran on the GPU

- **The watcher** (every process on GPU 2, every second): only my holder and the current run, during the whole campaign.
  All 14 entries it logged were my own processes caught while exiting.
- **GPU telemetry** (power, clock and use, about 8 samples per second): between my runs only the idle holder is on the
  GPU. In all 116 gaps between runs (9.4 minutes in total), GPU use was 0% and power 126 to 133 W (idle) in every
  sample. A job of anyone else would have shown up there.
- Another team's job started on GPU 2 at 16:00:01, after my last benchmark run (15:54:57).

### 14.2 Same problem for every contender

Head dim 128, bf16, causal inside each chat, 8 × 4,096 tokens, the same heads, the same chat list (exported by my test
program), K and V at the real KV-head count, the same number of iterations, the same GPU and session, 2 rounds in
opposite orders. Not identical: the random input values (different seeds; speed does not depend on the values) and the
loop (mine waits after each call; see 14.3, count C).

### 14.3 Counting three ways

From each run's own nsys trace, in the steady part of its loop, per iteration (forward + backward):

- **A** = what I reported: forward main kernel + backward-call span. The span only includes helper kernels whose names
  are on a fixed list for each library.
- **B** = ALL kernels added up (and memsets and memcopies), whatever their names. Nothing can hide here.
- **C** = the GPU timeline per iteration: every kernel AND every idle gap. This is the worst case for mine, because my
  test program waits after each call and the rivals' loop does not.

| Mine against | A (reported) | B (all kernels) | C (with all gaps) |
|---|---|---|---|
| FA3 | 0.403 to 0.928 | 0.400 to 0.928 | 0.453 to 0.938 |
| FA4 | 0.383 to 0.907 | 0.347 to 0.910 | 0.397 to 0.920 |
| cuDNN | 0.235 to 0.812 | 0.230 to 0.824 | 0.269 to 0.830 |
| my pushed code | 0.570 to 0.980 | 0.575 to 0.980 | 0.587 to 0.974 |

- B matches A within 1.5% for every contender in every case, except FA4 on the two short-chat rows, where B is 6.4%
  (18/6) and 8.9% (4/2) higher than A. There A favoured FA4.
- My code launches no kernel outside A, in any of the 12 cases.
- FA4 (on the short-chat rows) and cuDNN (everywhere) have about 8 µs per iteration of small extra kernels that A does
  NOT count against them.
- Mine is faster in all 48 comparisons under all three counts.

### 14.4 Each program's own timer, no nsys

Mine: the test program's own median per call (CUDA events), forward + backward. Rivals: the script's wall clock over the
timed loop divided by the iterations (Python launch overhead included). Milliseconds per forward + backward:

| Heads | Layout | Mine | Pushed | FA3 | FA4 | cuDNN | Mine / FA3 | Mine / FA4 | Mine / cuDNN | Mine / pushed |
|---|---|---|---|---|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 0.681 | 1.183 | 1.560 | 1.780 | 1.603 | 0.436 | 0.382 | 0.425 | 0.575 |
| 18/6 | mix, rows 8-15 | 1.429 | 1.651 | 1.801 | 1.820 | 1.879 | 0.793 | 0.785 | 0.761 | 0.866 |
| 18/6 | mix, rows 16-23 | 4.003 | 4.154 | 4.358 | 4.443 | 4.926 | 0.919 | 0.901 | 0.813 | 0.964 |
| 18/6 | real, rows 0-7 | 1.880 | 2.066 | 2.314 | 2.289 | 2.455 | 0.813 | 0.822 | 0.766 | 0.910 |
| 18/6 | real, rows 8-15 | 1.648 | 1.835 | 2.026 | 2.025 | 2.125 | 0.813 | 0.814 | 0.775 | 0.898 |
| 18/6 | real, rows 16-23 | 2.171 | 2.326 | 2.514 | 2.533 | 2.721 | 0.864 | 0.857 | 0.798 | 0.934 |
| 4/2 | mix, rows 0-7 | 0.184 | 0.298 | 0.425 | 0.495 | 0.755 | 0.433 | 0.371 | 0.243 | 0.617 |
| 4/2 | mix, rows 8-15 | 0.340 | 0.391 | 0.450 | 0.458 | 0.514 | 0.756 | 0.743 | 0.662 | 0.871 |
| 4/2 | mix, rows 16-23 | 0.901 | 0.948 | 1.003 | 1.047 | 1.199 | 0.897 | 0.860 | 0.751 | 0.950 |
| 4/2 | real, rows 0-7 | 0.473 | 0.484 | 0.567 | 0.561 | 0.637 | 0.834 | 0.842 | 0.743 | 0.977 |
| 4/2 | real, rows 8-15 | 0.392 | 0.434 | 0.553 | 0.498 | 0.551 | 0.708 | 0.786 | 0.710 | 0.902 |
| 4/2 | real, rows 16-23 | 0.531 | 0.545 | 0.614 | 0.618 | 0.692 | 0.865 | 0.860 | 0.768 | 0.975 |

The same order in every case, with ratios close to 13.2. This is also the quickest check for you (section 15.1).

### 14.5 Kernel by kernel, every case

All kernels of one iteration (round 1, steady loop, µs): forward / backward main / backward helpers and anything else /
total.

| Heads | Layout | Mine, final | Mine, pushed | FA3 | FA4 | cuDNN |
|---|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 180 / 311 / 178 / **669** | 175 / 560 / 429 / 1,163 | 219 / 801 / 532 / 1,552 | 380 / 823 / 562 / 1,766 | 314 / 635 / 626 / 1,575 |
| 18/6 | mix, rows 8-15 | 350 / 818 / 264 / **1,432** | 337 / 907 / 388 / 1,632 | 381 / 1,008 / 402 / 1,791 | 449 / 958 / 406 / 1,812 | 424 / 1,007 / 419 / 1,851 |
| 18/6 | mix, rows 16-23 | 1,019 / 2,738 / 278 / **4,034** | 1,006 / 2,796 / 384 / 4,186 | 1,057 / 2,907 / 386 / 4,350 | 1,146 / 2,898 / 391 / 4,435 | 1,198 / 3,301 / 400 / 4,900 |
| 18/6 | real, rows 0-7 | 458 / 1,164 / 264 / **1,886** | 446 / 1,231 / 388 / 2,064 | 497 / 1,405 / 404 / 2,306 | 570 / 1,305 / 405 / 2,280 | 573 / 1,409 / 449 / 2,431 |
| 18/6 | real, rows 8-15 | 407 / 956 / 269 / **1,632** | 389 / 1,057 / 389 / 1,834 | 433 / 1,182 / 401 / 2,017 | 497 / 1,115 / 406 / 2,018 | 488 / 1,189 / 421 / 2,097 |
| 18/6 | real, rows 16-23 | 535 / 1,379 / 274 / **2,188** | 521 / 1,426 / 388 / 2,335 | 566 / 1,547 / 395 / 2,507 | 634 / 1,496 / 399 / 2,530 | 646 / 1,641 / 408 / 2,696 |
| 4/2 | mix, rows 0-7 | 42 / 83 / 43 / **167** | 42 / 130 / 106 / 279 | 63 / 196 / 158 / 417 | 97 / 209 / 175 / 481 | 132 / 197 / 397 / 726 |
| 4/2 | mix, rows 8-15 | 81 / 181 / 61 / **324** | 80 / 196 / 97 / 373 | 95 / 227 / 117 / 439 | 100 / 231 / 119 / 450 | 110 / 243 / 136 / 488 |
| 4/2 | mix, rows 16-23 | 228 / 596 / 65 / **889** | 225 / 610 / 96 / 931 | 239 / 642 / 113 / 994 | 243 / 685 / 114 / 1,042 | 291 / 765 / 117 / 1,174 |
| 4/2 | real, rows 0-7 | 101 / 294 / 60 / **455** | 103 / 266 / 97 / 465 | 121 / 317 / 118 / 556 | 127 / 308 / 119 / 554 | 143 / 325 / 143 / 610 |
| 4/2 | real, rows 8-15 | 95 / 216 / 62 / **373** | 92 / 226 / 97 / 415 | 109 / 270 / 117 / 496 | 112 / 260 / 118 / 490 | 120 / 272 / 133 / 525 |
| 4/2 | real, rows 16-23 | 118 / 333 / 63 / **514** | 119 / 310 / 97 / 526 | 137 / 353 / 114 / 604 | 142 / 353 / 116 / 611 | 160 / 381 / 125 / 666 |

In every case mine has the smallest total, the smallest helpers and the smallest forward against the rivals. My
backward main kernel is the shortest of all five everywhere except 4/2 real rows 0-7 and 16-23, where my pushed code's
is shorter (it leaves the head sum to separate kernels; section 13.5).

### 14.6 Within the GPU's limits

Useful work per second, computed the same way for everyone: forward FLOPs = 4 × query heads × 128 × the sum over chats
of length² / 2 (causal), backward = 2.5 × forward. The H200's bf16 peak is 989 TFLOPS.

| Heads | Layout | Chats | Average length | Forward TFLOPS: mine / FA3 / FA4 / cuDNN | Backward TFLOPS: mine / FA3 / FA4 / cuDNN |
|---|---|---|---|---|---|
| 18/6 | mix, rows 0-7 | 353 | 93 | 92 / 75 / 43 / 52 | 85 / 31 / 32 / 33 |
| 18/6 | mix, rows 8-15 | 42 | 780 | 426 / 390 / 331 / 350 | 343 / 261 / 271 / 259 |
| 18/6 | mix, rows 16-23 | 8 | 4,096 | 615 / 594 / 545 / 516 | 519 / 475 / 474 / 417 |
| 18/6 | real, rows 0-7 | 41 | 799 | 501 / 463 / 402 / 396 | 400 / 316 / 333 / 305 |
| 18/6 | real, rows 8-15 | 36 | 910 | 448 / 425 / 371 / 375 | 371 / 289 / 302 / 284 |
| 18/6 | real, rows 16-23 | 23 | 1,425 | 520 / 486 / 433 / 428 | 419 / 352 / 360 / 338 |
| 4/2 | mix, rows 0-7 | 353 | 93 | 89 / 58 / 38 / 28 | 72 / 26 / 27 / 16 |
| 4/2 | mix, rows 8-15 | 42 | 780 | 401 / 344 / 327 / 298 | 334 / 233 / 230 / 216 |
| 4/2 | mix, rows 16-23 | 8 | 4,096 | 607 / 578 / 569 / 475 | 521 / 453 / 430 / 393 |
| 4/2 | real, rows 0-7 | 41 | 799 | 496 / 417 / 398 / 352 | 353 / 286 / 293 / 269 |
| 4/2 | real, rows 8-15 | 36 | 910 | 430 / 373 / 362 / 340 | 364 / 258 / 265 / 250 |
| 4/2 | real, rows 16-23 | 23 | 1,425 | 513 / 447 / 430 / 380 | 381 / 323 / 323 / 301 |

- **Long chats:** FA3 reaches 594 TFLOPS in the forward at 18/6, its normal level. So FA3 was set up correctly and not
  slowed down by my setup, and mine is only about 4% ahead in the forward and 9% in the backward.
- **Short chats:** every library is at 3% to 9% of the peak. FA3's backward does only 31 TFLOPS of useful work there, and
  that is where the big ratios come from.
- No number is close to the peak, so no result is physically impossible.

### 14.7 What is not identical, and what this does not show

- My test program waits after each call; the rivals' loop does not. Count C (14.3) includes those gaps, and mine still
  wins every comparison.
- The random input values differ (different seeds). Speed does not depend on the values.
- My chat plan is built on the CPU once per micro-batch and reused by every layer; its CPU time is not in these kernel
  times. FA3, FA4 and cuDNN have no such step, but run small kernels every call; those are in count B.
- Free clocks: mine ran at a lower clock than the rivals in most cases, so the time ratios are, if anything, cautious.
- Only head dim 128, rows of 4,096 tokens, bf16, causal, no dropout, no sliding window, one H200. FA3's deterministic
  mode was not timed here (it is slower than its default).
- Rows where every chat has about 40 tokens or less were not in this campaign; there FA3's forward is faster than mine
  (section 4.7), while forward + backward together mine was still faster.
- No training run against the rivals; the training comparison is only against my pushed code (11.2).

---

# Part D. Checking it yourself

## 15. How to check it yourself

Please use a GPU that is free for you, and write outputs into your own folder: my folders on P035 are only for reading.

### 15.1 On P035 with my builds (quickest)

```bash
L=/localstorage/gautam_k/attn_1117
NEW=$L/bt_p18                                # my pushed clone + the patch, built 18/11 (libtensor md5 b961ae5f...)
OLD=$L/bt_push                               # my pushed code, TI f1a7d98 (libtensor md5 be273224...)
TVA=$L/bin/tva_p18                           # my test program, built from bt_p18's own test file
PY=/localstorage/gautam_k/benv/bin/python    # torch 2.14.1, FA3 3.0.0, FA4 4.0.0b34, cuDNN 9.24
PKG=/localstorage/gautam_k/attn_1113/pkg     # rivals_free_1016.py, bench_step6_libs.py, layouts/
LAB=$L/stage0_labels/labels.bin              # real SFT rows for the suite groups D10 and G11
libs() { echo $1/Tensor-Implementations/lib:$1/Profiler/lib:$1/dist/lib; }
export CUDA_VISIBLE_DEVICES=<your GPU>
OUT=<your folder>; mkdir -p $OUT
```

**(a) Correctness, quick suite (about 3 minutes at 4/2, 9 at 18/6):**

```bash
env LD_LIBRARY_PATH=$(libs $NEW) USE_VJ2=1 VARLEN_BENCH_T=4096 VARLEN_STAGE0_LABELS=$LAB \
    VARLEN_QUICK=1 VARLEN_HEADS=18/6 $TVA > $OUT/quick_18x6.log 2>&1
grep "checks passed" $OUT/quick_18x6.log     # expect: checks passed 280, failed 0 (at VARLEN_HEADS=4/2: 273)
```

Without `VARLEN_QUICK=1` it runs the full suite (about 30 minutes). I have not run the full suite on the push build
itself. On the final code without the KNOWN entry it gave 332 passed and 1 failed (the case of 10.3); with the entry,
that case is listed under KNOWN instead.

**(b) The known case of 10.3 (about 30 seconds):**

```bash
env LD_LIBRARY_PATH=$(libs $NEW) USE_VJ2=1 VARLEN_BENCH_T=4096 VARLEN_STAGE0_LABELS=$LAB \
    VARLEN_ONLY=G11 VARLEN_CASE=D10 VARLEN_HEADS=18/6 VARLEN_D10_FIRST=1440 VARLEN_D10_ROWS=8 $TVA | tail -4
# expect: checks passed 13, failed 0, and a line "KNOWN G11 D10 ... dK pad row 1444 @4094 L=2: ratio 1.136"
```

**(c) My timing on one case, without nsys** (the case "18/6, mix, rows 0-7"; CUDA events around each call):

```bash
env LD_LIBRARY_PATH=$(libs $NEW) USE_VJ2=1 VARLEN_REAL_DIR=$PKG/layouts/mix/train VARLEN_BENCH_T=4096 \
    VARLEN_HEADS=18/6 VARLEN_ONLY=BENCH VARLEN_CASE=real4096_rows_0-7 VARLEN_BENCH_IMPL=ours_A_plan \
    VARLEN_BENCH_ITERS=2000 VARLEN_BENCH_EXPORT=$OUT/segs_raw.txt $TVA | grep ^BENCH
env LD_LIBRARY_PATH=$(libs $OLD) USE_VJ2=1 VARLEN_REAL_DIR=$PKG/layouts/mix/train VARLEN_BENCH_T=4096 \
    VARLEN_HEADS=18/6 VARLEN_ONLY=BENCH VARLEN_CASE=real4096_rows_0-7 VARLEN_BENCH_IMPL=ours_A_plan \
    VARLEN_BENCH_ITERS=2000 $TVA | grep ^BENCH
awk '!seen[$0]++' $OUT/segs_raw.txt > $OUT/segs.txt    # the chat list of this case (repeated lines removed)
```

Each run prints two lines, `dir=fwd` and `dir=bwd`, with `med_ms`. Add them. Expected (14.4): about 0.68 ms with the new
library and 1.18 ms with the pushed one.

**(d) The rivals on the same chats, without nsys:**

```bash
for r in fa3 fa4 cudnn; do
  env BENCH_T=4096 BENCH_HEADS=18/6 $PY $PKG/rivals_free_1016.py packed $r $OUT/segs.txt real4096_rows_0-7 2000 | grep ^RIVAL
done
# per_iter_ms = forward + backward per iteration; expected (14.4): FA3 about 1.56, FA4 1.78, cuDNN 1.60
```

To time the rivals' forward alone, put `RIVAL_FWD_ONLY=1` in front of the command.

**(e) The forward with and without MX.** Add `BLU_GQA_MX=0` to the command of (c) to run my forward without MX, and
compare the `dir=fwd` lines (on this short-chat layout MX takes about 8% off the forward, section 4.5).

**(f) Other cases.** Change `VARLEN_CASE` and the last argument of the rival command to `real4096_rows_8-15` or
`real4096_rows_16-23`; use `$PKG/layouts/real/train` for the real rows; use `VARLEN_HEADS=4/2` and `BENCH_HEADS=4/2`
for my 135M heads (8/4 for my 350M model). For rows 16-23 my test program writes the chat list twice; the `awk` line
removes the repeat.

**(g) Kernel times like mine.** Run (c) or (d) under nsys:
`nsys profile --trace=cuda --sample=none --cpuctxsw=none -o $OUT/run <command>`, then
`nsys stats --report cuda_gpu_trace --format csv -o $OUT/run $OUT/run.nsys-rep`, and add the kernel times per
iteration.

### 15.2 From a fresh clone

The 4 repositories are private; ask me for access (I do not put a token in this note).

```bash
git clone --recurse-submodules -b dist_side_gautam_lottn https://github.com/BlubridgeAI/BluTrain.git $WORK/BluTrain
cd $WORK/BluTrain && git checkout d9606c2 && git submodule update --init --recursive
git -C Tensor-Implementations apply <path to>/gqa_backward_ideas_1_2_4_vs_TI_f1a7d98.patch
md5sum Tensor-Implementations/src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu \
       Tensor-Implementations/src/Kernels/cuda/attention/arch/GQA_fwd_generic.cu \
       Tensor-Implementations/Tests/kernel_tests/test_varlen_attention.cu
# expect 25a58ee129ae3c8d8ca2a7bfe24e1cac, a8b3b2148901f655240b642aebc0156e, 00b06b7999c9bca7a1dfe93bb0760e6e
make -j96 blublas profiler tensor dist SM_ARCH=90    # CUDA 13 and yaml-cpp needed (on P035: CPATH and
                                                      # LIBRARY_PATH to /localstorage/gautam_k/deps/yaml/usr)
T=$WORK/BluTrain; TI=$T/Tensor-Implementations; CUDA=/usr/local/cuda
nvcc -I$TI/include -I$T/Profiler/include -I$T/dist/Distributed_Checkpointing/include -I$CUDA/include -DWITH_CUDA \
     -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
     -o $OUT/tva $TI/Tests/kernel_tests/test_varlen_attention.cu \
     -L$CUDA/lib64 -L$TI/lib -L$T/Profiler/lib -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml \
     -lprofiler -lblublas -lcuda
```

Then use `$OUT/tva` and `libs $T` in the commands of 15.1. Without the patch, the same build is my pushed code (with
all the forward changes of Part A). The patch md5 is `8c3f39f7a5b6663a7eb29564ed4e4983`. I checked that both
`git apply` and `patch -p1` turn the files of TI `f1a7d98` into exactly the files above.

### 15.3 My whole campaign

```bash
mkdir -p $OUT/campaign
cp -r /localstorage/gautam_k/attn_1117/dk18 $OUT/dk18     # the scripts and the 4 plan files
# in $OUT/dk18/rivals20.sh:
#   line 21: K=$L/dk18  ->  K=$OUT/dk18   (keep L and HERE: they point to my builds, layouts and Python files)
#   line 28: remove "$L/pkg14/status" (that is my own status file)
LIBP=$(libs $NEW) TVA=$TVA bash $OUT/dk18/rivals20.sh <gpu> $OUT/campaign/g<gpu> \
     $OUT/dk18/c_mix186.txt $OUT/dk18/c_real186.txt $OUT/dk18/c_mix42.txt $OUT/dk18/c_real42.txt
python3 $OUT/dk18/collect20.py $OUT/campaign            # the table of 13.3, plus my 13/11 records
python3 $OUT/dk18/apples20.py $OUT/campaign/g<gpu>      # the counts A, B, C of 14.3 (add --kernels for 14.5)
```

- It refuses to start if the GPU has any process, and waits (it never touches another process) if one appears between
  runs.
- 4 plans on one GPU take about 78 minutes. You can also run one plan per GPU, with one `rivals20.sh` per GPU.
- Each layout folder gets a `summary.txt` with the medians, clocks and power of every contender.

### 15.4 Checks I suggest

1. Time mine and FA3 with your own timer or profiler on the same exported chat list (15.1, steps c, d, g).
2. Other chat lengths: my test program has built-in rows of equal chats (`VARLEN_CASE=uniform_512` up to
   `uniform_4096`), and `$PKG/layouts/sw1` to `sw5` hold rows where every chat has the same length, from 16 to 2,112
   tokens (`$PKG/layouts/SWEEP.txt` lists them). The forward at about 40 tokens and less is where FA3 should win
   (section 4.7).
3. Other heads: 8/4 (my 350M model), 12/4, 16/4.
4. FA3's configuration: read the template arguments in the names of FA3's kernels in your trace (tile, PackGQA,
   scheduler), as in section 3.
5. Correctness against FA3: `/localstorage/gautam_k/attn_1117/dk18/crosslib_dk18.py` compares O, dQ, dK and dV of mine
   with FA3, FA4 and cuDNN on the exported bytes of one batch.

### 15.5 What you should see

- **Correctness:** the counts of 10.2.
- **Times:** within a few percent of 13.4 and 14.4 on an H200 at 700 W. Different GPUs run at slightly different
  clocks (GPU 1 and GPU 2 of P035 differed by 1.4%), so compare ratios measured in the same session, not microseconds
  across days or boxes.
- **Signs of a problem:**
  - a rival much slower than in 13.4: check that nothing else runs on the GPU and that FA4's compile has finished
    within its warm-up;
  - mine much faster than in 13.4: check that the chat list is the same. Count the chats per layout: 353 / 42 / 8 for
    mix and 41 / 36 / 23 for real.

---

## 16. Where things are

| What | Where |
|---|---|
| The backward patch | `gqa_backward_ideas_1_2_4_vs_TI_f1a7d98.patch`, next to this note |
| October record of both kernels (every variant, kept or not) | `attention_kernel_optimizations_h200_2026_10_10_to_2026_10_21.md`, next to this note |
| My 13/11 note to you (forward, MX) | `attention_packed_rows_our_experiments_for_nova_forward_study_2026_11_13.md`, next to this note |
| My built push tree, test program | P035: `/localstorage/gautam_k/attn_1117/bt_p18`, `/localstorage/gautam_k/attn_1117/bin/tva_p18` |
| My built pushed code | P035: `/localstorage/gautam_k/attn_1117/bt_push` |
| The 3 changed files alone | P035: `/localstorage/gautam_k/attn_1117/dk18/p18_files/` (same paths as in Tensor-Implementations) |
| Python with FA3, FA4, cuDNN | P035: `/localstorage/gautam_k/benv` |
| FA3 source I quote | FlashAttention `94e22c9`, `hopper/`; on P035 `/localstorage/gautam_k/flash-attention/hopper` (the build source) |
| FA4 source I quote | P035: `/localstorage/gautam_k/benv/lib/python3.12/site-packages/flash_attn/cute/interface.py` |
| Layouts | P035: `/localstorage/gautam_k/attn_1113/pkg/layouts/` (`mix`, `real`, `sw1` to `sw5`, `LAYOUTS.txt`, `SWEEP.txt`) |
| Campaign scripts | P035: `/localstorage/gautam_k/attn_1117/dk18/` (`rivals20.sh`, `scoreboard19.py`, `split19.py`, `watch20.sh`, `foreign20.py`, `collect20.py`, `apples20.py`, plans `c_*.txt`) and `/localstorage/gautam_k/attn_1113/pkg/` (`rivals_free_1016.py`, `bench_step6_libs.py`, `seq_names.py`, `bwd_breakdown.py`) |
| Campaign results (per run: traces, outputs, summaries) | P035: `/localstorage/gautam_k/attn_1117/rivals20/g2/` |
| Copies on my laptop | repo `Neural_Networks_exp_1926`: `profiling_reports/attn_bwd_p015_2026_11_17/box_results_p035/rivals20/` (summaries, final table) and `.../dk18/box_results/` (`apples20_all.txt`, `apples20_kernels_all.txt`, `apples20_tflops.txt`, `apples20_own_timers.txt`, push-tree checks `p18_check/`, the dK check) |
| Backward test results of 17/11 (suites, races, trainer) | laptop: `profiling_reports/attn_bwd_p015_2026_11_17/box_results_p035/` (`runs35g/` for the trainer) |
| The 13/11 forward study (layouts, scripts, MX candidates, logs) | laptop: `profiling_reports/attn_short_chats_2026_11_13/` |
| Day-by-day log | `sft_work_documentation.md`: sections 40 to 45.62 (October), 45.194 and 45.195 (13/11, MX), 45.197 and 45.198 (14 to 17/11, backward) |
