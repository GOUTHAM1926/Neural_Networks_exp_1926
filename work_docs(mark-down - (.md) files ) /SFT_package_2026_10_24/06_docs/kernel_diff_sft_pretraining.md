# Pretraining vs SFT: what changes, kernel side and non-kernel side

Written 2026-10-22. Code references are to the laptop SFT master `~/Downloads/BluTrain_sft` unless a library is named.
Library references are to the clones in `~/Downloads/sft_research/<library>` at the commit given in section 7.

Note (2026-11-19): the line numbers below point into the trainer of 2026-10-22. The trainer has grown since (the 135M
shape settings, the trained-rows switch, context parallelism, the export), so the current list of its SFT parts, found by
searching for quoted text, is in `sft/README.md` section 5. The rule (exactly three kernels change) and the reasoning
below are unchanged; section 3.2 describes the attention as of 2026-11-18; the dated sections at the end list what
changed after 2026-10-22.

---

## 0. Short answer

- On the kernel side, packed SFT changes exactly **three** kernels: **RoPE** (per-chat positions), **attention**
  (each chat attends only to itself) and the **loss** (ignore the untrained tokens, one denominator for the whole
  step). Every other kernel in the model runs the pretraining code unchanged.
- On the non-kernel side, the changes are: the data loader, the backward seed, the SFT settings (`BLU_SFT_*`),
  the start checkpoint (`BLU_INIT_FROM`), building the packed-row inputs every micro-batch, and freeing them at exit.
- The whole difference between the pretraining trainer `bluscript_zero_ckpt.cpp` and the SFT trainer
  `bluscript_sft.cpp` is **61 lines**. (On 2026-10-22. On 2026-11-18: +808 / -40 lines, because the SFT trainer
  also got the 135M shape settings, the trained-rows switch, context parallelism and the export; the kernel rule is the same.)
- Every library checked (HF transformers, TRL, torchtitan, Megatron-LM, open-instruct, LLaMA-Factory, axolotl, Liger,
  FlashAttention, TransformerEngine) changes the same three things for packed SFT. The only kernel-level extra is
  TRL computing `lm_head` only on trained tokens. It is a speed-up with the same math, and it is our pending item Q11.
  (2026-11-03: built as the switch `BLU_SFT_TRAINED_ROWS=1`, off by default; see the section of 2026-11-03 at the end.)

---

## 1. The rule: which ops must change

In a packed row, several chats sit one after the other. An op needs a change only if its result for one token
depends on something that packing changes. There are exactly three ways that can happen:

1. **The op mixes tokens with each other.** Then a token of chat 2 could read chat 1. That must be blocked.
2. **The op uses the token's position.** In a packed row, the position in the row is not the position in the chat.
3. **The op uses the labels.** In SFT, most labels are "do not train" (prompt and pad), and the denominator changes.

An op that works on one token at a time, with no position and no label, gives every token the same result whether
the chat sits alone in a row or packed next to other chats. Such an op needs no change.

### Worked example (one packed row, T = 8)

```
row position t   :  0    1    2    3    4    5    6    7
which segment    :  A    A    A    B    B    pad  pad  pad
position in chat :  0    1    2    0    1    0    1    2      <- RoPE uses this row (restarts at every segment)
label            :  x    a2   a3   x    b2   x    x    x      <- x = 65535 (not trained)
```

- Attention: token 4 (B's second token) may look at tokens 3 and 4 only. It must not see 0, 1, 2 (chat A).
  The pad tail (5, 6, 7) is a segment of its own and looks only at itself.
- RoPE: token 3 is rotated as position 0 (the start of chat B), not as position 3.
- Loss: only the tokens with a real label count (here a2, a3, b2), and the denominator counts trained tokens, not
  the 8 positions.
- RMSNorm, the linear layers, SwiGLU, the residual add: token 4 gets exactly the same result as if chat B were alone
  in its own row. Nothing to change.

---

## 2. Every op in our model, checked

From the forward code of `bluscript_sft.cpp` (attention block `forward` at line 604, MLP block at 716, full model at 779):

| Op | Mixes tokens? | Uses position? | Uses labels? | Change for SFT? |
|---|---|---|---|---|
| Embedding lookup (`wte`) | no | no (the model has no learned position table) | no | no |
| RMSNorm (before attention, before MLP, final) | no (one token's own vector) | no | no | no |
| Linear layers = GEMMs (qkv, attention out, gate_up, down, lm_head) | no: output row i is computed from input row i only | no | no | no |
| QK-RMSNorm (per token, per head) | no | no | no | no |
| SwiGLU, residual add, dtype casts, reshape / transpose | no | no | no | no |
| **RoPE** | no | **yes** | no | **yes: per-chat positions** |
| **Attention** | **yes** | no | no | **yes: chat plan** |
| **Cross-entropy loss** | no | no | **yes** | **yes: ignore index + global denominator** |

Other facts that make this complete:

- Dropout is 0 (`dropout_p=0.0f`, `bluscript_sft.cpp:677`), so there is no random op that could differ.
- There is no BatchNorm or any other op that takes statistics over the batch.
- The weight gradients (for example dW = Xᵀ·dY of a linear layer, or the dgamma of RMSNorm) are sums over all tokens.
  That is the normal sum over the batch. The pad tail adds exactly zero to these sums: its labels are all 65535, so
  its logit gradients are zero, and it attends only to itself, so nothing flows from it into a chat.

---

## 3. The three kernel changes in detail

### 3.1 RoPE: per-chat positions

**What changes.** RoPE rotates q and k of every token by an angle that depends on the position. In pretraining,
the position is the token's index in the row, `t`. In a packed row it must be the index inside the token's chat.

**How it is passed.**

```
SftDataLoader (reads seq_lengths.bin)  ->  positions [B, T] int32 on the GPU (last_positions(), SftDataLoader.h:163)
bluscript_sft.cpp:649                  ->  fused_rms_rope(..., g_packed.positions)
op layer (Normalizations.cpp)          ->  checks it (int32, CUDA, [B, T], contiguous) -> device pointer or nullptr
launcher (.cu)                         ->  pointer == nullptr ? kernel<HAS_POS=false> : kernel<HAS_POS=true>
kernel                                 ->  cache row = HAS_POS ? positions[b*T + t] : t + pos_offset
```

The trainer never passes `HAS_POS` itself. It passes the positions tensor, or nothing. The launcher picks the
compiled kernel from "is the pointer null".

**Pretraining is unchanged.** `HAS_POS` is a template (compile-time) parameter, so the `HAS_POS=false` kernel is a
separate compiled function with no positions code in it at all. Proofs:
- fp32 kernels (2026-10-10): 13 of 13 kernels SASS-identical to the old ones.
- Kruthik's bf16 kernels (2026-10-22, stage B): 9 of 9 kernels SASS-identical to his original (`FusedRMSRoPE_bf16.cu`,
  BLUCCL 02d0d18). The `HAS_POS=true` versions are different code (8 to 32 more instructions in the binary), so the
  comparison is not empty. Records: `profiling_reports/rope_bf16_port_2026_10_22/` (`orig.sass`, `new.sass`,
  `sass_compare.py`).

**Why restart at 0 when per-chat attention already separates the chats.** RoPE has the property that the
attention score between query i and key j depends only on the distance i − j. If chat B used positions 3, 4 instead
of 0, 1, every distance inside chat B stays the same, so in exact math the result is the same. In bf16/fp32 the
rounding differs a little, because the rotation angles are different numbers. We restart because:
1. Packed then equals alone bit for bit (stage-B check 3 below). That lets every packed test compare against a chat
   run alone.
2. Training sees the same positions as inference, where every chat starts at 0.
3. Every library does it (section 6).
4. The cost is small (section 8).

### 3.2 Attention: the chat plan

**What changes.** In a packed row, plain causal attention would let chat B read chat A. Each chat must attend only
to itself.

**How it is passed.** The attention kernel does not read the positions. It gets its own input, the "chat plan":

```
SftDataLoader                       -> chat list per micro-batch: (row, start, length) for every chat and the pad tail
                                       (last_segment_rows / starts / lengths, SftDataLoader.h:164-166)
bluscript_sft.cpp (packed_rows_of)  -> gqa_chat_plan_make(rows, starts, lengths, n, B, T, Hq, Hkv)
                                       (built on the host, one asynchronous upload, shared by all 26 layers,
                                        forward and backward; until 2026-11-13 the call had no Hkv)
bluscript_sft.cpp (attention block) -> scaled_dot_product_attention(..., g_packed.chats)
```

**Mixed units (added 2026-11-13).** The plan is built with the KV-head count too (`Hkv`, the last argument).

- A forward work unit is one chat x one query head x a pair of 64-row query tiles, one tile for each of the two
  compute groups of the kernel.
- A chat with an odd number of 64-row tiles (64 tokens or less, 129-192, 257-320, ...) left the second group of its
  last pair with no rows: it only waited.
- With mixed units that group runs the same last tile of ANOTHER query head of the same KV head. Both heads read the
  same K and V, so the loads do not change. With 3 query heads per KV head two heads share and the third stays alone;
  with 2 per KV head both share.
- The 2% rule: the builder mixes only when at least 2% of the plan's units would share a tile. Below that the plan and
  the kernel are exactly the ones of before (the mixed-unit kernel has a longer first step per unit, which costs
  0.5-1% of the forward on rows with many chats).
- Results do not change: each chat of a packed row still gives the same bits as the same chat alone.
- Measured on H200 (8 rows x 4,096, heads 18/6, forward, against the plan of before): -21 to -26% on rows of chats of
  64 tokens or less, -8 to -10% at 160-192 tokens, -0.5 to -1.3% on real SFT rows, no change on rows without such
  chats. The step time of a 135M SFT run on real data does not change measurably (the attention forward is a small
  part of the step); the gain is for data with many short chats.
- Switches (environment of the trainer binary): `BLU_GQA_MX=0` never mixes; `BLU_GQA_MX_MIN_PCT=<n>` changes the 2% limit.
- Code: `GqaChatPlan::mx_hkv` and the builder's comments in `include/ops/helpers/AttentionKernels.h` and
  `src/Kernels/cuda/attention/arch/GQA_fwd_generic.cu`; the kernel instantiation `MODE_CHATS_MX` in
  `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu`. The dense kernel and the plain packed kernel are unchanged.

**Backward: ideas 1, 2 and 4 (added 2026-11-18).** The Hopper backward (Vj2, `GQA_bwd_vj2_sm90.cu`) got three changes;
pretraining (no plan, dense rows) gets ideas 1 and 4 too:
- idea 1: in a 128-key block's first query step, all keys of the second compute group come after all queries of the
  step, so that group skips the S / dP GEMMs, the exps and the dV / dK GEMMs (all exact zeros) and runs only its dQ GEMM;
- idea 2 (packed rows only): the plan marks a chat's first two dQ tiles (`dq_tiles`, 4th number = 1); key block 0 is
  their only contributor and writes them in bf16 directly, so the fp32 dQ buffer, its zeroing and the convert skip them;
- idea 4: one block per (128-key tile, KV head) adds the G query heads' dK / dV in fp32 registers and writes the final
  dK / dV: no per-head partials, no head-sum kernels. For packed rows the plan builds a per-KV-head block list when it
  knows Hkv and G >= 2 (kept in a side table keyed by the plan, so `GqaChatPlan` keeps its layout). On when Hkv >= 2
  (`BLU_VJ2_HSUM_MINKV`), decided from the shape only, so packed == alone stays bit-exact.
- Switches: `BLU_VJ2_HSUM=0`, `BLU_VJ2_HSUM_MINKV=<n>`, `BLU_VJ2_HSUM_SECT=<n>`; `BLU_VJ2_DKDV_F32=1` also turns idea 4
  off. Tests, H200 timings (against the code of before and FA3 / FA4 / cuDNN) and the trainer A/Bs: `sft/README.md`
  section 6.3.

With no plan (`chats = nullptr`, the default in `include/autograd/operations/AttentionOps.h:133`), the attention
runs the plain causal kernel, unchanged. This is the "design A" (the `cu_seqlens` way of FlashAttention varlen).

**Example.** For the row in section 1, the plan holds three segments: (row 0, start 0, length 3), (row 0, start 3,
length 2), (row 0, start 5, length 3). Query 4 is in segment 2, so it attends to keys 3..4 only.

### 3.3 Loss: ignore index and global denominator

The loss function is `sparse_cross_entropy_loss(logits, target, ignore_index, denom)`
(`include/autograd/operations/LossOps.h:48-50`). Its defaults are `ignore_index = -1, denom = nullptr`.

| | Pretraining (default) | SFT |
|---|---|---|
| ignore index | `-1`. Token ids are 0..50303 and never −1, so nothing is ignored | `65535` (`SftDataLoader::kIgnore`, `SftDataLoader.h:48`). Labels are stored as uint16, which cannot hold −1, and 65535 is outside the 50,304-row vocabulary |
| denominator | `nullptr`: each micro-batch's loss is divided by its own B×T tokens | the whole step's trained-token count: this rank's next GRAD_ACCUM micro-batches (from `valid_counts.bin`), summed over all ranks with `MPI_Allreduce` (`bluscript_sft.cpp:2399-2406`) |
| backward seed | `1/GRAD_ACCUM`: turns the micro-batch losses into their mean | `W` (number of ranks): the micro-batch losses just add up, and `W` cancels the ÷W average that ZeRO does over the ranks (`bluscript_sft.cpp:1995-2001`) |
| call | `sparse_cross_entropy_loss(logits, batch.target, -1, nullptr)` | `sparse_cross_entropy_loss(logits, batch.target, SftDataLoader::kIgnore, &sft_denom)` (`bluscript_sft.cpp:2417-2419`) |

**Why the denominator changed, with numbers.** Take one step with two micro-batches. Micro-batch 1 has 1,000 trained
tokens, micro-batch 2 has 9,000.

- Mean of the two means (the pretraining way): loss = ½·(sum₁/1,000) + ½·(sum₂/9,000). Each token of micro-batch 1
  gets weight 1/2,000, each token of micro-batch 2 gets 1/18,000. A token counts 9× more just because it landed in a
  micro-batch with few trained tokens.
- Global count (the SFT way): loss = (sum₁ + sum₂)/10,000. Every trained token gets weight 1/10,000.

In pretraining every micro-batch has exactly B×T trained tokens, so the two ways give the same number. That is why
pretraining can keep the simple default.

**Pad and prompt tokens.** Their label is 65535, so they add nothing to the loss and get a zero gradient.

---

## 4. Non-kernel changes

The full diff `bluscript_zero_ckpt.cpp` → `bluscript_sft.cpp` is 61 lines. Line numbers are in `bluscript_sft.cpp`.

| What | Pretraining | SFT | Where |
|---|---|---|---|
| Mode switch | — | `BLU_SFT_DATA=<dir>` sets `sft = true` | 975-976 |
| Data | token shards (`DataLoader`) | `SftDataLoader`: inputs, labels, valid_counts; plus seq_lengths / seq_offsets for packed rows | 2409-2410 |
| Packed-row inputs | none (`g_packed` empty) | `g_packed = packed_rows_of(...)` every micro-batch: positions + chat plan | 2410-2412 (train), 2297-2298 (val) |
| Validation on pretraining shards | — | `g_packed = PackedRows{}` (plain causal rows) | 2339 |
| Loss denominator | none | global trained-token count, refilled every step | 2399-2406 |
| Backward seed | `1/GRAD_ACCUM` | `W` | 1995-2001 |
| Rows per micro-batch | `cfg.B` | `BLU_SFT_B` | 964-965, 979-983 |
| Settings | pretraining schedule | `BLU_SFT_LR`, `BLU_SFT_MIN_LR`, `BLU_SFT_EPOCHS`, `BLU_SFT_WARMUP`, `BLU_SFT_ROWS_PER_STEP`, `BLU_SFT_VAL_FREQ`, ... | 984-1018 |
| Start weights | random or resume | `BLU_INIT_FROM=<checkpoint>` (weights only); a warning if missing | 1031 |
| The old sequential RoPE path | — | throws for packed rows (it takes no positions) | 660-662 |
| Exit | — | `g_packed = PackedRows{}` before the allocator dies (a hang at exit was found on 2026-10-15) | 2873-2878 |

Note: the first rows of this table (the SFT loader, the loss with 65535 and the global denominator, the seed `W`)
are also in `bluscript_zero_ckpt.cpp` (lines 939, 1963, 2361, 2369). That file already trains SFT with one chat per
row (v1). `bluscript_sft.cpp` adds only the packed-row parts.

**Data preparation (outside the trainer).** `sft/prepare_sft_data.py` applies the chat template, sets prompt labels to
65535, and labels every chat on its own **before** packing (`inputs = ids[:-1]`, `labels = ids[1:]`,
`prepare_sft_data.py:136-138`). So the last target of a chat is its own end token, and no label points into the next
chat (checked at `:314`).

---

## 5. How it is switched on

| Switch | What it turns on |
|---|---|
| `BLU_SFT_DATA` set | SFT loader, loss with 65535 + global denominator, seed `W`, SFT settings |
| the data folder has `seq_lengths.bin` + `seq_offsets.bin` (`SftDataLoader.h:67-73`) | packed rows: RoPE positions + attention chat plan (`bluscript_sft.cpp:125`: `if (!l.packed()) return PackedRows{};`) |

There is no separate switch to turn the positions off. One could be added (about 5 lines: pass an empty tensor
instead of `g_packed.positions`), but it is not worth it (section 8).

With one chat per row (v1 data), there is no plan and no positions. That is correct: with one chat starting at
t = 0, the positions would be 0..T−1 anyway.

---

## 6. What the other libraries change (from their source)

### 6.1 Summary table

| Library | RoPE | Attention | Loss | Extra |
|---|---|---|---|---|
| HF transformers | `position_ids` | varlen from `position_ids`, or a same-chat mask | ignore −100, `sum / num_items_in_batch` (global count) | — |
| TRL | positions restart per chat | padding-free varlen, forced on for chat packing | uses HF's | first-token label fix; `lm_head` on trained tokens only (default) |
| torchtitan | `rope_cache[positions]` | document mask or varlen | sum ÷ `global_valid_tokens` | chat template + prompt masking |
| Megatron-LM | positions per chat; THD RoPE from `cu_seqlens` | `cu_seqlens` (THD) | `loss_mask`, per-token loss required with packing | — |
| open-instruct (AllenAI, the Tülu 3 code) | via HF collator | via HF collator | HF loss, no global count passed | init of new special-token embeddings |
| LLaMA-Factory | — | `packing` lets chats attend to each other; `neat_packing` does not | — | — |
| axolotl | per-chat `position_ids` | block-diagonal attention | — | — |
| Liger-Kernel | takes cos/sin picked by `position_ids` | — | fused lm_head + CE with ignore | runs lm_head on every token |
| FlashAttention / TransformerEngine | TE fused RoPE from `cu_seqlens` | varlen `cu_seqlens` / THD | — | — |

### 6.2 Details and proofs

**HF transformers** (4cc2aa8, 2026-10-08)
- RoPE: `LlamaRotaryEmbedding.forward(x, position_ids)` builds the angles from `position_ids`
  (`models/llama/modeling_llama.py:110-111`).
- Attention: FlashAttention path detects packing from `position_ids` (`_is_packed_sequence`,
  `modeling_flash_attention_utils.py:558-571`) and builds `cu_seqlens` from it. SDPA path builds a same-chat mask
  (`masking_utils.py:183-189`, `find_packed_sequence_indices` at `:734-763`).
- Data: `DataCollatorWithFlattening` returns `position_ids` that restart at 0 for every sequence, on by default
  (`data/data_collator.py:1377-1378, 1409, 1456`).
- Loss: ignore −100; with `num_items_in_batch` it uses `reduction="sum"` and divides by that count
  (`loss/loss_utils.py:35-45`). The Trainer counts the trained tokens over all accumulation micro-batches
  (`trainer.py:2264-2279`), sums the count over GPUs by default (`average_tokens_across_devices=True`,
  `training_args.py:856-857`), and multiplies the loss by the number of GPUs (`trainer.py:2182-2191`).
  **This is the same scheme as ours** (global count + seed `W`).

**TRL** (478624d, 2026-09-18)
- Packing with Best-Fit-Decreasing forces padding-free mode (`trainer/sft_trainer.py:1180`; config help
  `sft_config.py:239-247`).
- Positions restart per chat, built from `seq_lengths` (`sft_trainer.py:528-552`). Example from the docstring:
  `position_ids: [[0, 1, 2, 0, 1]]` (`:465`).
- The label at every chat's first token is set to −100 (`:516`). Reason: HF shifts the labels inside the loss, so
  without this the last token of chat 1 would be trained to predict the first token of chat 2. **We do not need it**:
  our labels are shifted per chat before packing (section 4).
- Default loss `chunked_nll`: the trained tokens are moved to the front (`argsort` on the mask), and `lm_head` +
  cross-entropy run only on the chunks that hold trained tokens (`sft_config.py:281-292`,
  `sft_trainer.py:189-199`). Same math, less work. **This is our Q11 item** ("logits only at trained positions").
- Other SFT defaults: learning rate 2e-5 (`sft_config.py:141-142`), `completion_only_loss` / `assistant_only_loss`
  for the label masks (`:259-280`).

**torchtitan** (1e4d33d, 2026-09-18)
- RoPE: `rope_cache[:num_tokens]` without positions, `rope_cache[positions]` with them
  (`models/common/rope.py:382-385`). The same design as ours.
- Packing builds `np.arange(...)` positions for every document segment (`components/data/packing.py:167`).
- Attention: a document mask from the positions (`get_document_mask_mod`, `models/common/attention.py:467`) or varlen
  (`create_varlen_metadata_for_document`, `:598`).
- Loss: `reduction="sum"` with `ignore_index` (`components/loss.py:33-60`), divided by `global_valid_tokens`, which is
  summed over the accumulation micro-batches and over the data-parallel ranks (`trainer.py:360-393`).
- SFT data: chat template, prompt labels set to ignore (`hf_datasets/text_datasets.py:159-196`).

**Megatron-LM** (8b2f660, 2026-09-18)
- `--sft` (`megatron/training/arguments.py:3907`) turns on `cu_seqlens`, the packed THD format
  (`pretrain_gpt.py:118-119`).
- Positions restart per chat, hard-coded, with an assert that the generic pretraining flag is off
  (`megatron/training/datasets/sft_dataset.py:149-150`). The THD RoPE path also finds each chat from `cu_seqlens`
  (`megatron/core/models/common/embeddings/rope_utils.py:208`, `_apply_rotary_pos_emb_thd`).
- Loss: a `loss_mask` that zeroes pad and prompt (`sft_dataset.py:193-195`); `sum(losses * loss_mask)` plus a token
  count (`pretrain_gpt.py:255-258`). Packing **requires** `--calculate-per-token-loss`, with the comment that
  per-microbatch loss averaging would weight tokens unevenly depending on how samples were packed
  (`arguments.py:1734-1738`). The same reason as our global denominator.
- Pretraining: `--reset-position-ids` exists but is off by default (`arguments.py:3488-3489`,
  `core/datasets/gpt_dataset.py:848-868`). Like our pretraining, documents in a row continue their positions.

**open-instruct** (AllenAI, the code behind Tülu 3; ccaae65, 2026-10-21)
- Packing through `TensorDataCollatorWithFlattening` (`open_instruct/finetune.py:59, 324-326, 664`), i.e. HF's
  position_ids + varlen attention.
- Loss: `loss = outputs.loss` (`:848`) with no global token count passed, so each micro-batch is a mean over its own
  trained tokens (`loss_utils.py:39` uses `"mean"` when no count is given). This differs from HF Trainer, TRL,
  torchtitan, Megatron and us.
- Extra: when new special tokens are put into reserved vocabulary rows, their embedding rows are initialised from
  the pieces the string used to split into (`:637-640`). **Does not apply to us**: our chat template is ordinary GPT-2
  tokens, no new ids.

**LLaMA-Factory** (c683e59, 2026-10-21)
- Two modes: `packing` ("Enable sequences packing", `src/llamafactory/hparams/data_args.py:108-110`) and
  `neat_packing` ("Enable sequence packing without cross-attention", `:112-114`; it also turns `packing` on,
  `:197-198`). So plain `packing` lets the chats in a row attend to each other. Only `neat_packing` changes the
  attention.

**axolotl** (2f98a74, 2026-10-22)
- `sample_packing`: "efficient multi-packing with block diagonal attention and per sequence position_ids"
  (`src/axolotl/utils/schemas/config.py:731-735`). The same two changes: attention + RoPE positions.

**Liger-Kernel** (19d92e7, 2026-09-17), the optional kernels used by TRL / axolotl / LLaMA-Factory
- Fused linear cross-entropy with `ignore_index` (`src/liger_kernel/ops/fused_linear_cross_entropy.py:131-132`).
  It runs `lm_head` on **every** token, in chunks (`:159-168`); ignored tokens get zero loss and gradient.
- RoPE takes cos/sin already picked by `position_ids` (`src/liger_kernel/transformers/rope.py:8`).

**FlashAttention** (94e22c9) and **TransformerEngine** (39c30c5), the kernel libraries
- FlashAttention varlen: `flash_attn_varlen_func(..., cu_seqlens_q, cu_seqlens_k, ...)`
  (FA2 `flash_attn/flash_attn_interface.py:1391`, FA3 `hopper/flash_attn_interface.py:890`).
- TransformerEngine: attention in "thd" format with `cu_seqlens`; fused RoPE finds each token's sequence from
  `cu_seqlens` inside the kernel (`transformer_engine/common/fused_rope/fused_rope.cu:22-28, 154`).

### 6.3 Verdict

- All of them change the same three things: RoPE positions, attention boundaries, loss masking + normalization.
  We change exactly these three.
- Restarting the positions per chat is automatic everywhere for chat packing. Megatron hard-codes it, TRL forces
  it, HF has it on by default. Pretraining does not restart by default (Megatron flag off; our pretraining passes no
  positions).
- Normalization: HF Trainer, TRL, torchtitan, Megatron and we use the global trained-token count. open-instruct's
  current code averages per micro-batch.
- Kernel-level extra: only TRL's `lm_head` on trained tokens (our Q11, a speed-up).

---

## 7. Library commits used

| Library | Commit | Date |
|---|---|---|
| transformers | 4cc2aa8 | 2026-10-08 |
| TRL | 478624d | 2026-09-18 |
| torchtitan | 1e4d33d | 2026-09-18 |
| Megatron-LM | 8b2f660 | 2026-09-18 |
| Liger-Kernel | 19d92e7 | 2026-09-17 |
| flash-attention | 94e22c9 | 2026-10-06 |
| TransformerEngine | 39c30c5 | 2026-10-09 |
| open-instruct | ccaae65 (shallow clone; scripts/docs folders did not check out, git-lfs missing; `open_instruct/finetune.py` is complete) | 2026-10-21 |
| LLaMA-Factory | c683e59 | 2026-10-21 |
| axolotl | 2f98a74 | 2026-10-22 |

---

## 8. Cost of the per-chat positions, and the decision

### 8.1 Stage-B test (H200, box 86.38.182.101 GPU 0, 2026-10-22)

`profiling_reports/rope_bf16_port_2026_10_22/test_rope_bf16_positions.cu`. Shape: T 4096, 18 query / 6 KV heads,
head dim 128. Every value compared bit for bit:

| Check | B=4 | B=8 |
|---|---|---|
| 1. positions 0..T−1 vs no positions | all 8 outputs bit-identical | same |
| 2. bf16 with packed positions vs the fp32 path with the same positions | all 8 bit-identical; packed q_out differs from no positions in 30.0M of 37.7M values (positions are used) | same; 57.0M of 75.5M differ |
| 3. one chat inside a packed row vs the same chat alone | bit-identical | bit-identical |

Time, forward + backward, one layer, median of 200 calls:

| Path | B=4 | B=8 |
|---|---|---|
| bf16 kernel, no positions (pretraining) | 0.2427 ms | 0.5409 ms |
| bf16 kernel, with positions (SFT after stage C) | 0.3295 ms (+35.8%) | 0.6907 ms (+27.7%) |
| fp32 path with positions, copies included (SFT today) | 0.7578 ms | 1.5294 ms |

### 8.2 Why yesterday showed +0.7% / +2.5% and today +28% / +36%

They are two different kernels.
- **Yesterday (2026-10-21, `sft_kernel_benchmarks.md` section 2)** measured **our fp32 kernel**. At B=8:
  forward 0.209 → 0.210 ms, backward 0.508 → 0.521 ms with positions.
- **Today** measured **Kruthik's bf16 kernel**, which got positions this morning (stage B). It had not been timed
  with positions before.

Why the bf16 kernel pays more (from its code; the forward/backward split has **not** been measured yet, the test
timed them together):
- With positions, the address of the cos/sin row needs the position value first. That is one extra memory read that
  must finish before the cos/sin read can start. Without positions the address is plain arithmetic, so both reads
  can be in flight at once.
- Kruthik's backward runs one warp per block (`block(32)`, grid `(BF16_BWD_T_CHUNKS = 32, heads, B)`), and each warp
  walks its tokens one after another (`for (t = t0; t < T; t += 32)`, 128 tokens per warp at T 4096). The extra wait
  repeats in every iteration.
- The bf16 kernel moves half the bytes of the fp32 kernel, so it is faster overall (0.54 ms vs 0.72 ms at B=8). The
  same extra wait is then a bigger share of a shorter time.

### 8.3 What it costs in the real run

Our model has 26 layers (`n_layers = 26`, `bluscript_sft.cpp:160`). One 20-row step at B=4 = 5 micro-batches:

| Path | RoPE time per step (26 layers × 5 micro-batches) |
|---|---|
| fp32 path (SFT today) | 0.7578 × 130 = 98.5 ms |
| bf16 + positions (after stage C) | 0.3295 × 130 = 42.8 ms |
| bf16 without positions | 0.2427 × 130 = 31.6 ms |

- One step is about 20 × 4096 / 50,300 tok/s ≈ 1.63 s.
- Switching to bf16 with positions saves 55.7 ms per step, about **3.4%** of the step.
- The positions themselves cost 11.3 ms per step, about **0.7%** (0.6% at B=8).
- Over the planned run (2 epochs ≈ 11,590 steps): the positions cost about **2.2 minutes** of about 5.2 hours.

### 8.4 The options

| Option | Time | What we lose / gain |
|---|---|---|
| A. Keep the restart (as built) | +0.7% vs no positions; still −3.4% vs today | keeps packed == alone bit-exact, training = inference positions, the same as every library |
| B. Drop the restart (pass no positions; attention still per chat) | saves ~2 min per run | same math in theory, but bf16 rounding differs, packed == alone is no longer bit-exact, and we would differ from every library |
| C. Keep the restart and make the bf16 positions read cheaper (for example read the position for the next token early, inside the `HAS_POS=true` code only) | could bring the cost toward the fp32 kernel's +2%, saving up to ~2 min per run | about an hour of work + one box test; pretraining untouched (the `HAS_POS=false` code stays SASS-identical) |

### 8.5 Decision

- **Train now with A.** Stage C (bf16 RoPE with positions) makes SFT 3.4% faster than today, and the positions cost
  about 2 minutes of a 5-hour run. That is too small to delay the run or to give up the bit-exact property.
- **Not B.** It saves about 2 minutes and loses the bit-exact packed == alone check that all our packed tests rely on.
- **C later, as a small follow-up.** Do it in a separate folder on the box, or after the run: rebuilding
  `libtensor.so` while a run is using it can crash the run.
- **2026-10-24: C done in effect.** The new bf16 backward keeps several tokens in flight with cp.async, so the extra
  read of the position no longer stalls each token. Backward + gamma reduce at B 4 (H200): 109.2 µs with positions
  against 104.4 µs without (+4.6%); before, 253.3 against 169.0 µs (+50%). Work log §45.91, README section 6.2.

## 2026-10-31: long-context fixes

The long-context fixes (RoPE tables, loss fallback backward, `contiguous()` past 2^32 elements, embedding and sm120
SwiGLU grid limits) are in kernels both trainers share: pretraining at long T gets them too. README section 6.4.

## 2026-11-03 to 11-07: the trained-rows switch, and fixes in code both trainers share

- **The trained-rows switch (2026-11-03, off by default).** With `BLU_SFT_TRAINED_ROWS=1` the SFT trainer runs two
  more kernels than pretraining: `gather_rows` before the lm_head (only the tokens that have a label) and, in the
  backward, `scatter_rows`; the lm_head GEMMs and the loss then work on those rows only (TI d18f820, BluTrain 7184fe1).
  With the switch off (the default) the difference stays exactly the three kernels above. README sections 5 and 9.
- **Shared code, so pretraining got the same changes:** the BluBLAS GEMM fixes (297f372, 0b552f0, a6d4292), the
  optimizer speed-ups and the multi-tensor fixes (README section 5.1), and the `autograd::contiguous` fix (TI d59740b,
  README section 6.5: no change for either trainer today).

## 2026-11-13 and 2026-11-18: the attention

- **2026-11-13:** the forward got "mixed units" for chats with an odd number of 64-row tiles (packed rows only, with the
  2% rule); pretraining (plain rows, no chat plan) runs the kernel of before. Section 3.2.
- **2026-11-18:** the backward got ideas 1, 2 and 4. Pretraining gets ideas 1 and 4 too (dense rows): its step was
  0.54% faster on the 1.6B production config. Section 3.2 and README section 6.3.
- Still exactly three kernels differ between SFT and pretraining (with the trained-rows switch off).
