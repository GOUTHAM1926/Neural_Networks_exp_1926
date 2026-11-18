# Context parallelism (CP) from zero: what it is, why, when, how, and what we use

Written 2026-11-18. Every statement about our code was checked in the code itself:

- our SFT trainer: `BluTrain/train/production_zero/bluscript_sft.cpp` (switch `BLU_SFT_CP=U`), `BluTrain/sft/README.md`
  section 5.2;
- Adhitya's CP library: the submodule `BluTrain/dist/Context_Parallelism` at commit `cf1372d`, folder
  `context_parallel/` (files named below);
- measured results: 2026-11-07 runs on 2x H200 (`profiling_reports/cp_2026_11_07_2x/box_results/results_2x.txt`, also
  written in the trainer's comments).

Numbers marked "calculated" are arithmetic, not measurements. Everything else is measured or read from the code.

---

## 0. Short answers first

| Your question | Answer |
|---|---|
| Is ring attention one way to do CP? | Yes. |
| Is Ulysses another way? | Yes. |
| Is there a hybrid of both in Adhitya's submodule? | Yes: `ContextParallel::enable_hybrid` = Ulysses inside a group of GPUs, ring between the groups (`HybridUlyssesBackward.h`, `ContextParallel.h`). |
| Which one does our SFT trainer use today? | **Ulysses only** (`BLU_SFT_CP=U`). Ring and hybrid are in Adhitya's library but are not wired into our trainer. |
| Can ring attention not do GQA? Is Ulysses the only way for GQA? | **No, ring can do GQA.** Adhitya's ring kernels are GQA kernels (`GQA_causal_cp_fwd_sm90.cu` / `GQA_causal_cp_bwd_sm90.cu`: Q has Hq heads, K and V have Hkv heads). GQA even makes ring cheaper, because ring sends only K and V, and GQA has fewer K/V heads. |
| Then why Ulysses for us? | With Ulysses every GPU gets **whole rows** back before attention, so our own packed-chat attention kernels (chat plan, per-chat masks, per-chat RoPE positions) run **unchanged**. Ring would need different kernels: attention between pieces of a row on different GPUs, with per-chat masks across the cut points, plus a merge step. |
| What limits Ulysses with GQA? | The number of CP GPUs (U) must divide the number of KV heads (our trainer copies no KV heads). 135M (4 query / 2 KV heads): U = 2. 350M (8 / 4): U = 2 or 4. 1.6B (18 / 6): U = 2, 3 or 6. |

---

## 1. The words used in this file

- **token**: one piece of text the model reads (a word or part of a word).
- **row**: what one training example looks like inside the trainer: T tokens in a line (for example T = 4,096 or
  131,072).
- **chat**: one conversation. In SFT we pack several chats into one row; the space left at the end is **padding**.
- **layer**: the model is a stack of identical layers (the 350M model has 28). Each layer has an attention part and an
  MLP part.
- **attention**: the part of a layer where each token looks at the tokens before it **in the same chat** and takes
  information from them. "Causal" = a token sees only itself and earlier tokens, never later ones.
- **query, key, value (Q, K, V)**: three vectors computed for every token. Token i compares its query with the keys of
  the tokens it can see; the comparison gives weights (softmax); the output of token i is the weighted sum of those
  tokens' values.
- **head**: attention is done several times in parallel on separate parts of the vectors; each part is a head. **D** =
  head size (128 for our models).
- **GQA (grouped-query attention)**: there are Hq query heads but only Hkv key/value heads; G = Hq / Hkv query heads
  share one K/V head. Query head h uses K/V head h / G (integer division). Example 350M: Hq = 8, Hkv = 4, G = 2: query
  heads 0 and 1 use K/V head 0, query heads 2 and 3 use K/V head 1, and so on.
- **activations**: the numbers each layer keeps in GPU memory during the forward pass, because the backward pass needs
  them. They grow with the number of tokens.
- **bf16**: the number format of these tensors: 2 bytes per number.
- **all-to-all**: a communication step where every GPU sends a different part of its data to every other GPU, and
  receives a part from each of them.
- **send / receive (P2P)**: one GPU sends a buffer to one other GPU.
- **NCCL**: NVIDIA's library that does these GPU-to-GPU communication steps.
- **LSE (log-sum-exp)**: for each query, LSE = log( sum over the keys it saw of exp(score) ). Attention kernels save it;
  it is what allows two partial attention results to be combined exactly (section 5.3).

---

## 2. Why: one long row can be too big for one GPU

During training, every layer keeps several tensors **for every token** until the backward pass uses them (the inputs of
the norms, Q, K, V, the attention output O, the MLP inputs, ...). So the memory grows in a straight line with the
number of tokens in the row.

Measured on one H200 (141 GB memory) with our SFT trainer:

| Model | Memory per token | Row of 64k tokens | Row of 131k tokens |
|---|---|---|---|
| 135M | 0.7545 MiB (measured on one GPU) | 61,503 MiB | 110,951 MiB (fits) |
| 1.6B shape | 2.11 MiB (derived from the measured CP=2 peaks, 2026-11-17) | does **not** fit on one GPU (out of memory, measured 2026-11-07) | does not fit |

For the 350M model (8 query / 4 KV heads, D 128, 28 layers) the same method gives ~1.05 MiB per token (calculated,
2026-11-17): the longest row that fits one H200 is about 121,000 tokens (calculated; the GPU check is prepared but was
not run yet).

Attention does not keep the T x T table of scores in memory (our kernels, like FlashAttention, recompute it in the
backward pass), so the problem is the per-token tensors of all layers, not the score table.

**CP is the answer when one row does not fit on one GPU** (and it also makes long rows faster, section 8).

---

## 3. The idea of CP

Split the tokens of ONE row over U GPUs. GPU r keeps tokens [r x T/U, (r+1) x T/U): its **slice**. Each GPU then keeps
the activations of only T/U tokens, so the memory per GPU drops by about U times.

Which parts of a layer need tokens that live on other GPUs?

| Part of the layer | Works on each token alone? | Needs other GPUs? |
|---|---|---|
| RMSNorm | yes | no |
| wQ, wK, wV projections (make Q, K, V) | yes | no |
| QK-norm + RoPE (position rotation) | yes (each token needs only its own position number) | no |
| **attention** | **no**: token i needs K and V of all earlier tokens of its chat | **yes** |
| wO projection, MLP (SwiGLU) | yes | no |
| lm_head + loss | yes | no (only the final sums are added across GPUs) |

So **CP is a method to do attention when the tokens of a row are spread over several GPUs.** Everything else stays as
it is. There are three methods: Ulysses, ring, hybrid.

---

## 4. Method 1: Ulysses (what our SFT trainer uses)

### 4.1 The idea in one sentence

Just before attention, the GPUs swap data so that each GPU holds **all T tokens but only some of the heads**; each GPU
then runs normal attention on whole rows for its heads; just after attention they swap back to "my tokens, all heads".

The two swaps are all-to-alls. In Adhitya's library: `cp::ulysses_combine` (slice of tokens, all heads -> all tokens,
some heads) and `cp::ulysses_partition` (the exact reverse) in `UlyssesAttention.h`.

### 4.2 Step by step on a tiny example

Setup: one row of T = 8 tokens (numbered 0..7), U = 2 GPUs, Hq = 4 query heads (q0..q3), Hkv = 2 K/V heads (kv0, kv1),
G = 2 (q0, q1 use kv0; q2, q3 use kv1).

**Before attention** (after wQ / wK / wV and RoPE, done on each GPU's own slice):

| | holds |
|---|---|
| GPU 0 | tokens 0-3 of q0, q1, q2, q3, kv0, kv1 |
| GPU 1 | tokens 4-7 of q0, q1, q2, q3, kv0, kv1 |

**All-to-all "combine"**: GPU 0 sends its tokens 0-3 of q2, q3, kv1 to GPU 1 and receives GPU 1's tokens 4-7 of q0, q1,
kv0. GPU 1 does the opposite. After it:

| | holds |
|---|---|
| GPU 0 | tokens 0-7 (the WHOLE row) of q0, q1 and kv0 |
| GPU 1 | tokens 0-7 (the WHOLE row) of q2, q3 and kv1 |

Note: each query head arrives on the same GPU as the K/V head it uses (q0, q1 with kv0). That is why U must divide Hkv:
the cut between GPUs must fall between K/V groups.

**Attention**: each GPU runs the normal causal attention over all 8 tokens for its heads. Because every GPU has whole
rows, every chat is complete on it: chat boundaries, causal masks and positions are exactly as without CP.

**All-to-all "partition"**: the attention outputs go back: GPU 0 gets tokens 0-3 of all 4 heads' outputs, GPU 1 tokens
4-7. Then wO and the MLP run on the slices again.

### 4.3 One layer with real shapes

B = rows per GPU per step, T = row length, U = CP GPUs. Shapes are [rows, heads, tokens, D].

| Step | Q on one GPU | K and V on one GPU |
|---|---|---|
| after wQ/wK/wV + RoPE (own slice) | [B, Hq, T/U, D] | [B, Hkv, T/U, D] |
| after combine | [B, Hq/U, T, D] | [B, Hkv/U, T, D] |
| attention output O | [B, Hq/U, T, D] | |
| after partition | [B, Hq, T/U, D] | |

Example 350M, one row of 131,072 tokens, U = 2: Q goes from [1, 8, 65,536, 128] to [1, 4, 131,072, 128]. Same number
of values (8,388,608 x 8 = 67,108,864), different arrangement.

### 4.4 Backward pass

An all-to-all only moves numbers, it does not change them. So its gradient is just the reverse move:

- gradient of "combine" = "partition" applied to the incoming gradient (`UlyssesCombineBackward`);
- gradient of "partition" = "combine" applied to the incoming gradient (`UlyssesPartitionBackward`).

In the backward pass: the gradient of O comes in as slices -> combine -> our attention backward (whole rows, Hq/U heads)
gives dQ, dK, dV -> partition -> back to slices.

### 4.5 Why Ulysses fits our kernels

After "combine" each GPU has whole rows, so:

- our packed attention with the **chat plan** (the list of chats in the row) runs exactly as on one GPU, only with
  Hq/U and Hkv/U heads;
- our QK-norm + RoPE ran BEFORE the swap, on the slice, with each token's true position inside its chat;
- no kernel was changed for CP. The trainer only adds the two swaps around attention (`cp_combine` / `cp_partition`
  in `bluscript_sft.cpp`).

### 4.6 The rules (our trainer stops with an error otherwise)

1. U divides the number of GPUs (GPUs = data-parallel groups x U).
2. U divides Hq **and** Hkv (section 4.2: the cut must fall between K/V groups).
3. U divides T.
4. Not together with `BLU_SFT_TRAINED_ROWS` (its row lists are for whole rows).

What that allows:

| Model | Hq / Hkv | Possible U |
|---|---|---|
| 135M | 4 / 2 | 2 |
| 350M | 8 / 4 | 2, 4 |
| 1.6B | 18 / 6 | 2, 3, 6 |

**More GPUs per row than K/V heads**: Adhitya's own Ulysses version (`forward_ulysses_gqa` in `ContextParallel.h`)
allows it by **copying K/V heads**: when Hkv < U and U is a multiple of Hkv, each K/V head is copied rep = U / Hkv times
before the all-to-all (the "DeepSpeed rule": Hkv % U == 0 or U % Hkv == 0). Our trainer does not do this copy, so for
us U <= Hkv. The other way past this limit is ring or the hybrid (sections 5 and 6).

### 4.7 How much data Ulysses moves

In an all-to-all over U GPUs, each GPU keeps 1/U of a tensor and sends the other (U-1)/U of it.

Forward, per layer, per GPU: Q, K, V go out in "combine", O goes out in "partition":

  sent = (U-1)/U x (Hq + Hkv + Hkv + Hq) x (T/U) x D x 2 bytes

The backward pass moves the same amount again (dO in, dQ/dK/dV out).

Calculated for one row of T = 131,072 tokens, D = 128, bf16:

| Model | U | Sent per GPU per layer, forward | Forward + backward, all layers |
|---|---|---|---|
| 135M (4 / 2) | 2 | 96 MiB | (x 2 x layers) |
| 350M (8 / 4) | 2 | 192 MiB | 192 x 2 x 28 = 10.5 GiB per step |
| 350M (8 / 4) | 4 | 144 MiB | 144 x 2 x 28 = 7.9 GiB per step |

The amount per GPU goes DOWN when U goes up (each GPU's slice gets smaller). In our trainer the all-to-alls run on a
blocking stream (they do not overlap with compute).

---

## 5. Method 2: ring attention

### 5.1 The idea in one sentence

Each GPU keeps **its own tokens and all heads**; the K/V pieces travel from GPU to GPU around a ring; each GPU computes
the attention of its own queries against each K/V piece that arrives, and combines the partial results exactly with
their LSE values.

In Adhitya's library: `RingRotator.h` (sends the K/V buffers around the ring, can overlap the send with compute),
`SDPAMerger.h` (combines partial results), `LoadBalancer.h` (HeadTail split, section 5.4), `ContextParallel.h`
(`forward_cp`), kernels `GQA_causal_cp_fwd_sm90.cu` / `GQA_causal_cp_bwd_sm90.cu`.

### 5.2 Step by step on a tiny example (simple split)

Setup: T = 8 tokens, N = 2 GPUs. GPU 0 holds tokens 0-3 (Q0, K0, V0), GPU 1 holds tokens 4-7 (Q1, K1, V1), all heads.

| Ring step | GPU 0 computes | GPU 1 computes | sent at the same time |
|---|---|---|---|
| 1 | Q0 against K0/V0 (causal inside the piece) -> partial result A0 + its LSE | Q1 against K1/V1 (causal inside) -> A1 + LSE | GPU 0 sends K0/V0 to GPU 1; GPU 1 sends K1/V1 to GPU 0 |
| 2 | Q0 against K1/V1: tokens 4-7 are in the FUTURE of tokens 0-3, so nothing to do (skipped) | Q1 against K0/V0: all earlier, full attention -> partial B1 + LSE; merge A1 and B1 | |

Final: GPU 0's output = A0; GPU 1's output = merge(A1, B1). With N GPUs there are N steps; each step a GPU computes with
one K/V piece and passes it on.

### 5.3 How two partial results are combined exactly (the LSE merge)

A query's attention over all keys can be computed in pieces: compute the output and the LSE over piece A, the same over
piece B, then combine:

  lse = log( exp(lse_A) + exp(lse_B) )
  out = out_A x exp(lse_A - lse) + out_B x exp(lse_B - lse)

Adhitya's merger writes the same thing in a numerically safer form (`SDPAMerger.h`):

  out = out - sigmoid(lse_B - lse) x (out - out_B),   lse = lse - log( sigmoid(lse - lse_B) )

Worked example (one query, values of size 1 to keep it readable): piece A has 2 keys with scores 1 and 2 and values 10
and 20; piece B has 1 key with score 3 and value 30.

| | calculation | result |
|---|---|---|
| all at once | weights exp(1), exp(2), exp(3) = 2.718, 7.389, 20.086, sum 30.193 -> 0.0900, 0.2447, 0.6652 | out = 0.0900 x 10 + 0.2447 x 20 + 0.6652 x 30 = **25.752** |
| piece A alone | lse_A = log(2.718 + 7.389) = 2.3133; out_A = (2.718 x 10 + 7.389 x 20) / 10.107 | out_A = 17.311 |
| piece B alone | lse_B = log(20.086) = 3.0000; out_B | out_B = 30 |
| merge | sigmoid(lse_B - lse_A) = sigmoid(0.6867) = 0.6652; out = 17.311 - 0.6652 x (17.311 - 30) | **25.752** (same), lse = 3.4076 = log(30.193) |

So the ring result is exactly the one-GPU result (up to normal rounding).

### 5.4 The causal problem: uneven work, and the HeadTail fix

With the simple split, the GPU holding the late tokens does much more work: late tokens see many earlier tokens.

Work counted in "units": one diagonal block (a piece against itself, causal = half a square) = 1 unit; one full block
(a piece against an earlier piece) = 2 units.

Simple split, N = 2 (from section 5.2): GPU 0 = 1 unit (Q0 vs K0); GPU 1 = 1 (Q1 vs K1) + 2 (Q1 vs K0) = 3 units. GPU 0
waits 2/3 of the time.

**HeadTail split** (Adhitya's `LoadBalancer.h`, the same as PyTorch's round-robin): cut the row into 2N chunks; GPU r
takes chunk r and chunk 2N-1-r. N = 2, T = 8: chunks c0 = tokens 0-1, c1 = 2-3, c2 = 4-5, c3 = 6-7; GPU 0 takes c0 and
c3, GPU 1 takes c1 and c2. Work (chunks of 2 tokens, same units):

| GPU | its query chunks | work |
|---|---|---|
| 0 | c0: diagonal (1); c3: c0, c1, c2 full (3 x 2 = 6) + diagonal (1) | 1 + 7 = **8** |
| 1 | c1: c0 full (2) + diagonal (1); c2: c0, c1 full (4) + diagonal (1) | 3 + 5 = **8** |

Equal work. The price: each GPU's tokens are no longer one contiguous piece, so positions (RoPE) and masks must follow
the permutation (Adhitya's library does this per ring step: `RopeDeltas.h`, and `DocMask.h` for the chat ids).

Ulysses does not have this problem: every GPU runs the whole causal row for its heads, so the work is equal by itself.

### 5.5 Backward pass of ring

Each GPU must get the gradients of ITS keys and values (dK, dV) from every query that looked at them, and those queries
live on all GPUs. So in the standard ring backward the K/V pieces travel around the ring again together with running
dK/dV buffers: each GPU adds its own queries' contribution to the visiting buffers and passes them on, while the dQ of
its own queries is summed on the GPU itself. Adhitya's version runs the per-step backward kernel in reverse ring order
(`ContextParallelBackward.h`, `GQACausalCP_sm90.h`); I did not trace its buffer details line by line.

### 5.6 Packed chats in a ring

Our rows hold several chats. In a ring a chat can be cut between two GPUs, and a K/V piece from another GPU can contain
tokens of a different chat. So every ring step needs to know, for each token, **which chat it belongs to** (to block
attention between chats) and **its position inside its chat** (RoPE). Adhitya's library has this: `DocMask.h` /
`DocIdKernel.cu` (document ids per token; every GPU of a CP group reads the same rows, so it can work out the ids of the
other GPUs' pieces without sending them). For our SFT it would also need our per-chat positions.

### 5.7 GQA in a ring

Ring sends only K and V (the queries stay home). With GQA, K and V have Hkv heads instead of Hq, so the data sent is G
times smaller than without GQA. Adhitya's ring kernels take Q with Hq heads and K/V with Hkv heads (query head h uses
K/V head h / G inside the kernel). **So ring works with GQA**; nothing about GQA forces Ulysses.

What ring needs from the attention kernel, and what our current kernels do not have:

- attention of a piece of queries against a piece of keys of a DIFFERENT length (T_q is not T_k), with or without the
  causal mask depending on the step (Adhitya forked our older v62 forward / v44 backward into
  `GQA_causal_cp_*_sm90.cu` for exactly this);
- the LSE out of each step for the merge (our kernels already return LSE);
- per-chat masks across pieces (doc ids) and per-chat positions.

Our current SFT kernels (v62 forward with the chat plan, Vj2 backward) work on rows where every chat is complete:
queries and keys are the same tokens. They have no "queries from one piece, keys from another piece" mode. That is the
reason we chose Ulysses first.

### 5.8 How much data ring moves

Forward, per layer, per GPU: (N-1) steps, each sends one K piece and one V piece:

  sent = (N-1) x 2 x Hkv x (T/N) x D x 2 bytes

(The backward pass sends K/V again plus the dK/dV buffers, so roughly 2-3 times the forward amount.)

Calculated for one row of T = 131,072 tokens, D = 128, bf16, 350M (8 / 4 heads):

| N | Ring, sent per GPU per layer, forward | Ulysses with U = N (section 4.7) |
|---|---|---|
| 2 | 128 MiB | 192 MiB |
| 4 | 192 MiB | 144 MiB |

Ring's amount per GPU does NOT go down when N goes up (each GPU must see every K/V piece once). Ring can hide its
sending behind the compute of the current step (Adhitya's `RingRotator` overlap); with few GPUs per row and GQA, ring
sends less than Ulysses.

---

## 6. Method 3: hybrid (Ulysses inside, ring outside)

Adhitya's `ContextParallel::enable_hybrid(ulysses_pg, ulysses_size, ...)`: the CP GPUs are arranged as
(Ulysses size) x (ring size). Inside each Ulysses group the GPUs swap heads with all-to-alls (fast, for example inside
one node); between the groups the K/V pieces travel in a ring (`HybridUlyssesBackward.h`: "Ulysses (all-to-all, inner)
nested inside Ring (P2P rotation, outer)").

Example: 8 GPUs per row for the 135M (only 2 K/V heads):

- Ulysses alone: U <= 2, so at most 2 GPUs per row;
- hybrid: Ulysses 2 (each pair swaps heads) x ring 4 (the 4 pairs pass K/V pieces around) = 8 GPUs per row; each GPU
  holds T/8 tokens.

So the hybrid keeps Ulysses inside the K/V-head limit and grows the GPU count with ring.

---

## 7. The three methods side by side

| | Ulysses | Ring | Hybrid |
|---|---|---|---|
| What each GPU holds during attention | all tokens, Hq/U heads | its tokens, all heads | its group's tokens, (heads / Ulysses size) |
| Communication | 2 all-to-alls per layer forward (Q, K, V in; O out), 2 backward | (N-1) sends of K/V per layer forward; K/V + dK/dV backward | both |
| Limit on GPUs per row | U divides Hq and Hkv (ours: U <= Hkv; Adhitya's: also U % Hkv == 0 with K/V copies) | N divides T (2N with HeadTail); no head limit | Ulysses part as Ulysses; ring part free |
| Causal work balance | equal by itself | needs HeadTail | needs HeadTail in the ring part |
| Partial results to merge | none | yes, LSE merge every step | yes |
| Packed chats | free (whole rows on each GPU) | needs per-token chat ids and positions per step | as ring |
| Kernel needed | our normal kernels, unchanged | piece-against-piece kernels (T_q not T_k), causal on/off per step | as ring |
| GQA | yes, within the head rule | yes (sends only K/V: G times less than without GQA) | yes |
| In Adhitya's library | yes (`UlyssesAttention.h`, `forward_ulysses_gqa`) | yes (`forward_cp`, `RingRotator.h`, `SDPAMerger.h`, `LoadBalancer.h`) | yes (`enable_hybrid`) |
| In our SFT trainer | **yes** (`BLU_SFT_CP=U`) | no | no |

---

## 8. What we use today, and why

- **Ulysses, with our own kernels**, behind the switch `BLU_SFT_CP=U` (default 1 = off). Decided 2026-10-31: no CP while
  a row fits one GPU; then Ulysses with our kernels (keeps the chat plan, per-chat RoPE, our faster backward); ring or
  hybrid only when more GPUs per row than K/V heads are needed.
- **How a step runs with CP** (README 5.2): the U GPUs of a CP group read the SAME rows (the data loader uses the
  data-parallel rank = rank / U); each GPU keeps its slice of every row with its labels and positions; per layer: norm,
  projections, QK-norm + RoPE on the slice -> combine -> our packed attention on whole rows -> partition -> wO, MLP on
  the slice; lm_head + loss on the slice, divided by the trained-token count of the whole step over all GPUs (so every
  trained token weighs the same as without CP); backward: the reverse all-to-alls; ZeRO reduces gradients over all
  GPUs as before.
- **The CP group**: split off the world NCCL group (`ncclCommSplit`), with a blocking CUDA stream (so the all-to-alls
  are ordered after our kernels on the default stream).
- **Run**: `SFT_CP=2 NP=2 GPUS=0,1 ... bash sft/run_sft.sh`.
- **Needs** Tensor-Implementations d59740b or later: before that fix, under CP the gradient of k and of the k-norm gamma
  was dropped in every layer (step-0 gradient norm 7.5% lower than without CP; found and fixed 2026-11-07).

**Measured (2026-11-07, 2x H200; CP=1 = one GPU, CP=2 = two GPUs share each row):**

| Case | CP=1 | CP=2 | |
|---|---|---|---|
| 135M, 4k rows, step-0 gradient norm | 3.0473 | 3.0473 | equal |
| 135M, 4k rows, 30 steps, largest loss difference | | 0.000997 vs CP=1 | two identical CP=1 runs differ by 0.000813 |
| 135M, 64k rows, time per step | 250.7 ms | 158.2 ms | 1.58x faster |
| 135M, 64k rows, peak memory per GPU | 61.5 GB | 36.3 GB | 59% of CP=1 |
| 135M, 128k rows, time per step | 492.9 ms | 297.9 ms | 1.65x faster |
| 135M, 128k rows, peak memory per GPU | 111.0 GB | 62.2 GB | 56% of CP=1 |
| 1.6B shape, 4k rows, step-0 norm | | +0.017% vs CP=1 | |
| 1.6B shape, 64k rows | out of memory | fits, 96.6 GB per GPU | CP makes it possible |

(2 GPUs would ideally be 2x faster and use 50% memory each; the gap is the communication, the smaller kernels on each
GPU, and the memory that does not shrink with tokens: weights, optimizer, buffers.)

**Not tested yet**: CP together with data parallelism (more GPUs than U, needs 4+ GPUs).

---

## 9. When to use CP

| Model | Longest row on one H200 | With CP=2 |
|---|---|---|
| 135M | 131,072 tokens measured (110,951 MiB); ~173k calculated | 128k measured: 62.2 GB per GPU |
| 350M | ~121,000 tokens (calculated) | ~234,000 tokens (calculated; ~85 GB per GPU at 131,072) |
| 1.6B | 64k does not fit (measured) | 64k fits (measured, 96.6 GB per GPU) |

Rule: use CP when the row does not fit one GPU; also use it when long rows must train faster (1.58-1.65x with 2 GPUs
for the 135M at 64k-128k). For short rows (for example 4k) CP is not needed: the row fits, and CP's communication and
smaller per-GPU kernels cost more than they give.

---

## 10. What depends on what

| Thing | Depends on | Why |
|---|---|---|
| Possible U (Ulysses) | Hq and Hkv of the model | the head swap must keep each K/V group whole (section 4.2) |
| Possible U | T | each GPU gets exactly T/U tokens |
| Number of GPUs | U x data-parallel groups | each CP group of U GPUs trains the same rows |
| Data loader | data-parallel rank = rank / U | all GPUs of a CP group must read the same rows |
| Loss | the trained-token count of the whole step over all GPUs | each GPU sees only a slice; dividing by its own count would weigh tokens wrongly |
| RoPE positions | the token's position inside its chat (not inside the slice) | the slice starts in the middle of a chat |
| Attention kernel (Ulysses) | nothing new | whole rows on each GPU |
| Attention kernel (ring) | piece-against-piece support, LSE, chat ids, positions | section 5.7 |
| Our backward's idea 4 (head sums inside the block) | the K/V heads **per GPU** | with CP=2 the 135M has 1 K/V head per GPU; the rule `BLU_VJ2_HSUM_MINKV` (2) turns idea 4 off there, because with 1 K/V head it was 2.0% slower per step (17/11 trainer A/B) |
| Communication speed | the GPU links (NVLink inside a node) | all-to-alls and ring sends use them |
| ZeRO (optimizer sharding) | all GPUs (data parallel x CP) | gradients are reduced over every GPU as before |

What does NOT depend on CP: the model weights, the optimizer, the MLP, the norms, the projections, the lm_head (they
work per token), and the results (CP=2 trains the same as one GPU within run-to-run variation, section 8).

---

## 11. Summary in five lines

1. CP = split one long row's tokens over several GPUs, because one GPU cannot hold all of its activations.
2. Only attention needs tokens from other GPUs; there are three ways to do it: Ulysses (swap tokens for heads), ring
   (pass K/V pieces around and merge with LSE), hybrid (both).
3. Ring works with GQA (GQA makes it send less); Ulysses works with GQA only while U divides the K/V heads.
4. We use Ulysses because each GPU gets whole rows, so our packed-chat kernels run unchanged; it is tested and trains
   the same as one GPU (2026-11-07).
5. For more GPUs per row than K/V heads: K/V-head copies, ring, or the hybrid; Adhitya's library has all three, our
   trainer has none of them wired yet.
