# Sliding-window attention (SWA) and post-training: what is published, what is measured, what is open

Written 2026-11-17. Purpose: decide how SFT (and later DPO) should handle a model that is pretrained with sliding-window
attention, and plan the ablations that close the gaps. Every published statement below is quoted from the source it
cites (fetched on 2026-11-17); internal numbers come from the run folders named next to them.

## 1. What sliding-window attention is (and is not)

- **The rule.** In a windowed layer, the token at position i attends to the W positions right before it, itself
  included: keys j with i - W < j <= i. Nothing is chosen by score: it is a fixed band of positions. For W = 1,024 the
  token at position 32,000 sees positions 30,977 to 32,000. Choosing "important" tokens by score is a different family
  (block or top-k selection), not SWA.
- **Local and global layers.** Modern models mix windowed ("local") layers with normal ("global") layers. Our 350M
  candidate (`kv4_swa3`) has every 4th layer global: 7 of 28 layers (indices 3, 7, ..., 27) see all earlier tokens,
  21 see the last 1,024.
- **Why far tokens are not lost.** The global layers see the whole context directly. After the first global layer,
  the hidden state at every later position already carries what it read from far back, and the windowed layers above
  read those nearby hidden states. Mistral 7B: "SWA exploits the stacked layers of a transformer to attend information
  beyond the window size W ... hi can access tokens from the input layer at a distance of up to W × k tokens".
- **In SFT with packed chats** the mask is (same chat) AND (causal) AND, in windowed layers only, (within W). A token
  at chat position 1,500 sees chat positions 477-1,500 in the 21 windowed layers and 0-1,500 in the 7 global layers.
  For a chat of at most 1,024 tokens, SWA and full attention are identical.

## 2. Published evidence

### 2.1 Models pretrained with SWA keep it in post-training (no published model switches)

| Model | Attention in pretraining | Post-trained release |
|---|---|---|
| Mistral 7B (arXiv 2310.06825) | every layer windowed, W = 4,096 | "We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct"; HF `Mistral-7B-Instruct-v0.1` config: `sliding_window: 4096` |
| Gemma 2 (arXiv 2408.00118) | "local sliding window attention and global attention in every other layer", local 4,096, global 8,192 | Gemma 2 IT, same layers |
| Gemma 3 (arXiv 2503.19786) | "5 local layers for every global layer", local span 1,024; "only the global layers attend to long context" | Gemma 3 IT ("our novel post-training recipe"), same layers |
| gpt-oss (arXiv 2508.10925) | "alternate between banded window and fully dense patterns, where the bandwidth is 128 tokens" | HF `gpt-oss-20b` config: `layer_types` alternating `sliding_attention` / `full_attention`, `sliding_window: 128` |

Mistral 7B v0.2 set `sliding_window` to null (no official reason published; community threads only).

### 2.2 Pretraining ablations of SWA itself

- **Gemma 3, section 5.2:** "Impact of Local:Global ratio on the perplexity ... The impact is minimal, even with 7-to-1
  local to global." and "The sliding window can be reduced significantly without impacting perplexity." (figures 3 and
  4; the paper gives curves, not a table).
- **Longformer (arXiv 2004.05150):** local and global attention are both needed (local builds context, global builds
  the full-sequence representation).

### 2.3 Switching between full attention and SWA after pretraining

| Direction | Source | What it found |
|---|---|---|
| full-attention model, then SWA | SWAA (arXiv 2512.10411), Qwen3-4B | naive SWA "suffers from catastrophic long context performance collapse, which stems from ... the training inference mismatch when naively applying SWA to models pretrained with Full Attention (FA), and the inherent structural inability to access distant information"; "Combining structural methods (FA Decode or Interleaving Layers) with lightweight fine tuning virtually eliminates the performance gap"; LoRA used "because full parameter fine tuning often leads to overfitting and degradation of the model's original capabilities" |
| full-attention model, SWA at inference without training | "Sliding-window beats linear attention" (arXiv 2608.28444) | SWA with attention sinks "performs as well or better than most retrofitted Linear Attention models" |
| SWA-pretrained model, then full-attention fine-tuning | LASER (arXiv 2512.19701), Gemma 3 270M-12B, one regression task (ChipDV) | "We ablate this by fine-tuning both SWA and full-attention variants ... Full attention consistently matches or outperforms SWA" (one narrow task, not chat SFT) |
| sparse local attention in fine-tuning, full at inference | LongLoRA (arXiv 2309.12307) | shifted short attention in training, standard attention at inference, "comparable performance" (needs the shift so information crosses groups) |

## 3. Internal measurements (colleagues, P015, `/localstorage/test/BluTrain`, branch `swa-135m`, not on GitHub)

All pretraining, 5.0B tokens, rows of 4,096, the same init and the same batches (paired), val loss of the final step.

| Run | Change against the full-attention baseline | Val loss difference | Throughput |
|---|---|---|---|
| 135M, `swa_ablation` (28 of 37 layers windowed, W 1,024) | SWA vs global | -0.0025 (2.7958 vs 2.7983); code +0.0032, the other domains lower | 616.7k vs 549.9k tok/s (x1.12) |
| 350M, `hp350m/swa3` (W 1,024, every 4th global) | SWA vs `base` | -0.0011 (noise floor from a seed pair: 0.0002) | 177k vs 158k tok/s |
| 350M, `hp350m/swa5` (every 6th global) | SWA vs `base` | -0.0008 | 176k |
| 350M, `hp350m/kv4_swa3` (the chosen config) | 4 KV heads + SWA vs `base` | -0.0091 (kv4 alone -0.0053) | 172k |

- Kernels (Navin's local commit `b913f32`, Hopper only, plain rows only): forward 0.209 ms vs causal 0.394 ms, backward
  0.757 ms vs 1.266 ms at B 16, heads 4/2, T 4,096, W 1,024; 25 checks pass (fp64 reference, exact key counts).
- Serving (vLLM, 135M): KV cache per request at 4k: 67 MiB (SWA) vs 127 MiB (global).
- So at 4,096-token pretraining, SWA is not worse than full attention, and it is cheaper. No SFT run exists yet.

## 4. Our SFT data against the window (W = 1,024)

| Data set | Chats longer than 1,024 tokens | Share of tokens in those chats |
|---|---|---|
| m1 smol | 42.8% | 66.2% |
| m2 tulu | 17.8% | 44.8% |
| d8 all | 27.9% | 53.2% |

A large part of SFT is in chats where SWA and full attention differ, so the choice matters.

## 5. What is NOT published (the gaps our ablations close)

1. No controlled 2 x 2 study: the same model pretrained with full attention or SWA, each post-trained with full
   attention or SWA, compared on chat / instruction benchmarks.
2. No study of SWA with packed chat-masked SFT rows (the window counted inside each chat).
3. Nothing at small scale (135M / 350M) for post-training.
4. Nothing for DPO with SWA.
5. No split of the effect by chat length (chats shorter vs longer than W) or by position inside a chat.
6. LASER is the only published "SWA-pretrained, full-attention fine-tuned" case, on one regression task.

## 6. Ablation plan

**Starting models (already trained, paired):** `hp350m/base` (full attention) and `hp350m/swa3` (W 1,024, every 4th
global), same data, init and steps; also `kv4` / `kv4_swa3` (the chosen config), and the 135M `global` / `swa` pair.

**Arms (same SFT data, steps, LR, seed):**

| Arm | Pretraining | SFT | Question |
|---|---|---|---|
| A | full | full | baseline |
| B | full | SWA | does SWA in SFT hurt a full-attention model (SWAA direction)? |
| C | SWA | SWA | the published practice |
| D | SWA | full | can we skip SWA in SFT (LASER direction)? |
| A2 / C2 | as A / C | as A / C, second seed | noise floor |

**Evaluation of every arm:** SFT train loss; val loss overall, by chat length (up to 1,024 vs longer) and by position
inside a chat (0-1k, 1k-2k, 2k-4k); the eval suite of the 135M SFT grid; a long-chat recall test (the answer depends
on a fact more than 1,024 tokens back in the same chat); each SFT model also evaluated with the other attention
pattern (train / inference consistency); step time and memory. Decision rule: a difference smaller than twice the
seed-pair noise is a tie.

**Order:**
1. Packed-row SWA in our SFT path, correct but not optimized: bring `b913f32` onto our branch, count the window from
   each chat's start in the forward work list and the backward block list, run the suite against the fp64 reference.
   Arms B and C need this.
2. Run arms A-D (+ seeds) and decide.
3. Only if SWA in SFT is kept: optimize (sharing / direct dQ / skip-masked ideas inside windows), race against
   FA3 varlen `window_size`, FA4 and FlexAttention, benchmark for pretraining and SFT; then DPO reuses it.

### 6.1 Decisions and step plan (2026-11-17 evening)

**User decisions (17/11):**
- the order above is agreed;
- all SWA work and all ablations run on the 350M model only, with no 135M runs;
- DPO is not our work. The DPO team calls the same varlen attention function and passes the window per layer (the
  trailing `window` argument of Navin's API, 0 = full attention), so there is no kernel work for them. This replaces
  "then DPO reuses it" in step 3 above.

**Proposed settings (PROPOSED, waiting for the user's OK):**
- **Checkpoint pair:** `hp350m/kv4` (full attention) and `hp350m/kv4_swa3` (W 1,024; layer i is global iff
  (i + 1) % 4 == 0).
  - Same init, same data, same 5.0B tokens. They differ only in the window.
  - `kv4_swa3` is the chosen 350M config.
  - `base` / `swa3` is a repeat pair if the result is close. No new pretraining is needed.
- **SFT row length:** 4,096 tokens, the pretraining row length (RoPE theta 50,000, trained to 4,096; corrected
  2026-11-17 17:10 from the runs' blu_meta.json, it was written 10,000 from Ram's separate PyTorch script).
  - So no CP is needed.
  - Chats longer than 4,096 tokens are dropped, as in all our SFT.
  - The window changes the result only for chats of 1,025-4,096 tokens.
- **RoPE base (checked 2026-11-17 18:25):**
  - the hp350m screen checkpoints were trained by `swa_ablation/bin/bluscript_135m_swa` with base 50,000
    (blu_meta.json);
  - our SFT trainer's default is 10,000 (`bluscript_sft.cpp:327`), so every arm passes `ROPE_THETA=50000`; the
    step-0 gate (numpy reference reading the base from the checkpoint's meta) catches a mismatch. Rule: the base
    always comes from the checkpoint, never from a default;
  - Ram's PyTorch script and Navin's BluTrain 350M script both default to 10,000. Speed comparisons are
    unaffected (the base changes table values, not work). Open question for the colleagues: which base will the
    production 350M use (the screen tested 50,000 only).
- **Data and training settings:**
  - data m1_smol: smol-smoltalk f73fe857 minus explore-instruct-rewrite; 42.8% of its chats are longer than 1,024
    tokens;
  - template tulu-plain-v1;
  - 1 epoch (~3,570 steps), 128 chats per step (~25.5 rows; the memory model gives ~123 of 142 GB);
  - LR 1e-4 (the LR of the 135M SFT screens), cosine, warmup 3%;
  - Muon (to check: the same optimizer as the pretraining), production YAML, `BLU_EMB_BWD=segmented`;
  - the second seed is a second row order (fmt_reorder.py), because the trainer has no data-order seed.
- **Validation:**
  - the whole val split as one file, giving a curve every 500 steps;
  - plus group files built with label masks: chats of up to 1,024 tokens vs longer chats, and answer positions
    0-1k / 1k-2k / 2k-4k inside the chat;
  - checkpoints every 1,000 steps, each evaluated on every group file.

| Step | What | GPU | Estimate |
|---|---|---|---|
| 0 | read-only checks on P015 (configs, tokenizer, optimizer, steps); converter (hf_to_blutrain.py for the 350M shape) + gate (step-0 loss vs an independent numpy reference within 0.2%); real prep + val group files; 30-step smoke (step time, memory) | 1, ~30 min | ~3 h |
| 1 | arms A and D (full attention in SFT, no new code) | 2 | ~1-1.5 h per arm |
| 2 | packed-row SWA: merge b913f32 onto the pushed tree; forward and backward window counted inside each chat; per-layer window in bluscript_sft.cpp; HF export with the window. Tests: fp64 reference + exact key counts on packed rows (chats shorter than, equal to and longer than W; every chat-start offset; 8/4 heads and others); Navin's dense tests on the merged tree; window off bit-identical to the pushed build (kernels + 30-step trainer, which keeps A and D valid); kv4_swa3 gate with the window | 1 for tests | ~1 day |
| 3 | arms B and C, then the seed repeats A2 and C2 | 2 | ~3 h wall |
| 4 | evaluation: val loss by group; each model also with the other attention type; long-chat recall; IFEval + ARC-e; loss curves, step time, memory | 1 | ~1-2 h |
| 5 | decision; results written here as section 7 | - | - |
| 6 | only if SWA stays in SFT: optimize; race FA3 `window_size` / FA4 / FlexAttention on pretraining rows and SFT rows; push | 1 | later |

Note (2026-11-18): the pushed tree changed after this plan was written: the attention forward has "mixed units"
(2026-11-13: an idle compute group runs the same tile of another query head of the same KV head) and the backward has
ideas 1, 2 and 4 (2026-11-18: the fully masked half of a block's first step is skipped; a chat's first two dQ tiles are
written directly; one block adds the query heads of a KV head inside itself). TI 334b244, BluTrain c52ca81. So step 2
merges `b913f32` (written on the Vj2 and v62 of 2026-11-04) onto these new kernels, and the window has to be applied
and tested in each new path: the shared tiles of mixed units, the per-KV-head blocks of idea 4, the skipped half of
idea 1 (with a window, the masked part of a step is different) and the direct dQ tiles of idea 2.

- **Where the arm time comes from:** for the 135M, stage 1 measured 0.18 s fixed + 0.0129 s per row per step. The
  350M needs about 2.6x the compute per token, which gives ~1.1-1.2 s per step. The smoke run replaces this
  estimate.
- **What each result means:**
  - C as good as D: keep SWA in SFT and in inference, as in the published practice, and go to step 6;
  - D clearly better on long chats: SFT with full attention, and then inference also uses full attention, which loses
    SWA's smaller KV cache (to discuss);
  - B against A: whether a full-attention model can switch to SWA in SFT.

## 7. SWA in SFT with packed chats: how it works, who supports it, what it saves (2026-11-17, user questions)

The user's questions:
- In SFT our kernels work per chat. Does the window apply to every chat?
- Are the tokens of a chat past 1,024 "reduced"?
- Is a per-chat window weird? Does anyone really do it in post-training (TRL, kernel libraries)?
- How much does it save in SFT?
- If it matters, is it only because pretraining used SWA and SFT would use full attention?

### 7.1 The mechanism, per chat (nothing is dropped)

- **Inside a packed row, a token attends only to earlier tokens of its own chat.** That is the chat mask (same chat)
  AND causal. With SWA, a windowed layer adds one more condition: the key is among the last W = 1,024 positions of
  that chat. Positions are counted inside the chat, from the chat's first token.
- **No token is dropped, shortened or left out of the loss.** Every token of every chat still goes through every
  layer and still gets its loss. The only change: in the 21 windowed layers, a token at chat position p > 1,024
  reads the keys of positions p-1,023 .. p, instead of 0 .. p.
  - Tokens at chat positions up to 1,024 get exactly the same result as with full attention, in every layer.
  - In the 7 global layers, every token still reads its whole chat.
- **A packed chat gives the same result as the same chat alone, in a row of its own.** This is the same rule our
  suite checks today ("packed chat == the same chat alone, bit for bit"). A per-chat window is therefore not a new
  idea: it is exactly what the model computes on one chat without packing. Packing only puts several chats in one row.
- **Row length vs window.** The row (context) length is 4,096 and the window is 1,024. With rows of 4,096, SWA is
  NOT full attention: the windowed layers see 1,024 back. Only a window >= the row length equals full attention
  (Navin's code: "0 or W >= T is plain causal").

### 7.2 Who supports SWA together with packed sequences (sources fetched 2026-11-17)

| Stack | What the source shows |
|---|---|
| HF transformers `masking_utils.create_sliding_window_causal_mask` | builds `sliding_window_causal_mask_function` (`kv_idx > q_idx - sliding_window` AND causal). When position_ids show packed sequences (`find_packed_sequence_indices`: a new sequence starts where position ids restart), it wraps it with `packed_sequence_compatibility_function` = AND(same sequence, the window on positions remapped "to their actual values" inside each sequence). That is a per-chat window. |
| HF flash-attention path (what TRL padding-free / packing uses) | `flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)`; packed batches go to `flash_varlen_fn(q, k, v, cu_seqlens_q=..., cu_seqlens_k=..., **flash_kwargs)`, so each sequence of the varlen batch gets its own window |
| FlashAttention (varlen) | "If window_size != (-1, -1), implements sliding window local attention. Query at position i will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive", with i inside each sequence given by cu_seqlens |
| PyTorch FlexAttention (official blog) | names "Sliding Window Attention + Document Masking + Causal + Context Parallelism" as a combination people want; `generate_doc_mask_mod(mask_mod, document_id)` takes any mask (e.g. `and_masks(causal_mask, sliding_window)`) and applies it on per-document positions (`q_logical = q_idx - offsets[document_id[q_idx]]`) AND same document |
| TRL | packing ("BFD", default) and padding-free: "available only for SFT training and setups that use FlashAttention (or its variants)". TRL itself has no attention code: the model's HF attention (above) applies the window, so a Gemma / Mistral SFT with TRL packing gets per-chat windows by default. Padding-free is "temporarily unavailable in DPOTrainer" (falls back to padding). |
| NVIDIA TransformerEngine (Megatron / NeMo) | "Sliding window attention (SWA): flash-attention has SWA(left, right) support for all mask types except top-left causal masks"; cuDNN attention supports SWA(left, 0) from 9.2; `thd` layout = "variable sequence lengths in a batch" (packed); TE's JAX example runs packed THD + GQA + causal sliding window together |
| Mistral's own reference code | handles several sequences in one batch with a per-sequence window ("seqlens [5,7,2] ... sliding_window 3 --> only cache last 3 tokens in each sequence") |
| vLLM (inference) | hybrid KV-cache manager for "Sliding window attention (sw) + full attention (full): gpt-oss, Gemma 2/3, Ministral, cohere": "Sliding window layers: reserve slots only for the most recent sliding_window_size tokens" |

**Released instruct models keep the window:**
- Gemma 3 4B IT config (unsloth mirror of the official file): `sliding_window` 1024, `sliding_window_pattern` 6,
  34 layers.
- Mistral-7B-Instruct-v0.1: `sliding_window` 4096.
- gpt-oss (all released checkpoints are post-trained): `layer_types` sliding / full, `sliding_window` 128.

**Why they keep it (their own words):**
- Gemma 3: "We also change the architecture of the model to reduce the KV-cache memory that tends to explode with
  long context"; "A challenge with long context is the memory explosion of the KV cache during inference. To reduce
  this issue, we interleave multiple local layers between each global layer, and assign a smaller span of only 1024
  tokens to the local layers"; with "global only" the KV cache is "a memory overhead of 60%, ... reduced to less than
  15% with 1:3 and sliding windows of 1024".
- So the reason is inference memory and long-context cost, and the post-trained model must run the same attention
  it is served with.

**What no source says:**
- No source states a measured SFT quality effect of SWA vs full attention.
- No source says "we tried full attention in post-training and switched back".
- The published practice is "keep the architecture", and the frameworks simply apply the model's config. This is gap 1
  of section 5, which arms C vs D measure.

### 7.3 What SWA saves in OUR SFT (calculated from the packed data, 350M shape)

Method:
- pairs = (query, key) dot products inside each chat;
- full causal = L(L+1)/2 per chat of L tokens;
- window: W(W+1)/2 + (L - W) W when L > W;
- 28 layers = 7 global + 21 windowed;
- FLOPs: attention 4 x 128 x 8 per pair per layer forward (backward 2.5x); linear layers + lm_head 2 x params per
  token forward (backward 2x).

| Data (rows of 4,096) | Chats > 1,024 (tokens in them) | Tokens past chat position 1,024 | Pairs saved per windowed layer | Pairs saved, all 28 layers | Attention share of FLOPs | FLOPs saved |
|---|---|---|---|---|---|---|
| m1 smol (452,974 chats, mean 822, 0.3% > 2,048) | 42.8% (66.2%) | 12.9% | 4.8% | 3.6% | 9.2% | 0.33% |
| m2 tulu (744,023 chats, mean 622, 2.9% > 2,048) | 17.8% (44.8%) | 15.4% | 18.2% | 13.6% | 9.6% | 1.31% |

| One chat of N tokens | Pairs per windowed layer | All 28 layers |
|---|---|---|
| 4,096 | 43.7% | 57.8% |
| 8,192 | 23.4% | 42.6% |
| 32,768 | 6.2% | 29.6% |
| 131,072 | 1.6% | 26.2% |

**Reading:**
- In today's SFT data (rows of 4,096, most chats under 2,048 tokens), the window saves almost nothing: 0.3-1.3% of
  the training FLOPs.
  - Attention is ~9-10% of the FLOPs.
  - Most chats over 1,024 are only a little over.
  - The time share of attention is higher than its FLOP share (the attention kernels run at lower efficiency than
    the GEMMs), so the time saved is likely a few percent at most. Measure it in the smoke run of each arm.
- Long-context SFT is different: for chats of 8k-128k tokens the window cuts the attention work to 26-43%. That is
  where SWA matters for SFT speed and memory traffic.
- Since the speed gain in today's SFT is small, the reason to keep SWA in SFT is consistency, not speed:
  - the 21 windowed layers were pretrained never seeing a key more than 1,023 positions back;
  - the deployed model is meant to run with the window (smaller KV cache, as in Gemma 3 and vLLM).
- SFT with full attention would be a train / inference change in both directions (pretraining -> SFT, SFT ->
  serving). That is what arms C and D measure.
- **For our tests:** 12.9-15.4% of the SFT tokens are the only tokens whose windowed layers see less. The quality
  difference between arms C and D (if any) comes from those tokens, so report the loss on them separately (the
  position > 1,024 group of section 6.1).

## 8. Long context (e.g. 64k): does the window change? (2026-11-17, user question)

**Terms:**
- "Local" = the windowed layers, with a fixed reach of W = 1,024.
- "Global" is not a size: a global layer reads the whole chat / row, so its reach is the row length. That is 4,096 in
  pretraining, and it becomes 64k automatically with 64k rows.

**What every published model with SWA did when it went to long context: the window stays the same, and only the
global layers' positions are extended.**

| Model | Local window | Context | How the long context was reached (source) |
|---|---|---|---|
| Gemma 3 4B/12B/27B | 1,024 (5 local : 1 global) | 128k | "pre-train our models with 32K sequences and then scale ... up to 128K tokens at the end of pre-training while rescaling RoPE ... scaling factor of 8"; "We increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local layers at 10k". IT config: sliding_window 1024, rope_theta 1e6, rope_local_base_freq 1e4, rope_scaling linear 8, max_position_embeddings 131072 |
| gpt-oss | 128 (alternating) | 131,072 | "extend the context length of dense layers to 131,072 tokens using YaRN". Config: yarn factor 32 from original_max_position_embeddings 4096, sliding_window 128 |
| Command A (Cohere, 111B) | 4,096 (3 local : 1 global) | 256k | model card: "three layers with sliding window attention (window size 4096) and RoPE ... A fourth layer uses global attention without positional embeddings". Config: sliding_window 4096, pattern 4, max_position_embeddings 262144. Post-trained with SFT + preference training |
| Llama 4 Scout | chunked local attention 8,192 (3 RoPE : 1 NoPE global) | 10M | config: attention_chunk_size 8192; no_rope_layers: 12 of 48 layers without RoPE; attn_temperature_tuning; rope_scaling llama3 factor 16 from 8192 |
| Mistral 7B | 4,096 (all layers) | v0.1: 8k trained | the exception: v0.2 (32k) DROPPED the window (sliding_window null, rope_theta 1e6). No official reason |

Research: SWAN-GPT (arXiv 2504.08719) uses NoPE global layers + RoPE local layers with a FIXED window of 512. Trained
on 1,024-token sequences, it keeps its quality beyond the training length, with attention-score scaling at inference.

**Why the window does not need to change:**
1. A windowed layer only ever sees keys at most 1,023 positions back, in a 4k row and in a 64k row alike. Nothing
   is new for it: its RoPE distances stay inside what it was trained on, so Gemma 3 keeps its base at 10k.
2. Only the global layers see new, longer distances (4k -> 64k). They need the position extension (larger RoPE base,
   position interpolation, YaRN, or no positions at all = NoPE), plus a long-context training phase (Gemma 3: at
   the end of pretraining).
3. Memory: a windowed layer keeps the KV of 1,024 tokens at any context length. A window that grew with the context
   would give that saving back. Gemma 3's ablation: "The sliding window can be reduced significantly without
   impacting perplexity".
4. Changing W after pretraining would be a new train / inference change for the windowed layers, which have never
   seen keys beyond the old W. No published model grows W with the context. Gemma changed W only between
   generations (Gemma 2: 4,096 -> Gemma 3: 1,024), as an architecture choice made before pretraining.

**For our 350M (pretrained rows of 4,096, W 1,024, every 4th layer global, RoPE theta 50,000 in ALL layers: hp350m
blu_meta.json / HF config; corrected 17:10, first written as 10,000), a 64k
version would follow the same recipe:**
- keep W = 1,024 and the windowed layers' RoPE base 50,000;
- extend the 7 global layers: a larger RoPE base (Gemma 3: 1M) and / or interpolation 64k / 4k = 16x, or YaRN;
- then a long-context training phase on long documents (the colleagues' pretraining decision);
- then long-context SFT with exactly those settings.

**Code status:**
- Our SFT trainer builds ONE RoPE table for all layers (`bluscript_sft.cpp:954-955`, one `BLU_ROPE_THETA`, no
  scaling). Gemma-3-style RoPE needs a second table for the global layers (selected by the same layer rule as the
  window), and the HF export needs `rope_theta` / `rope_local_base_freq`.
- The window kernels do not change (W stays 1,024).
- At 64k rows a windowed layer computes ~3% of the full attention pairs, and memory per token is the same as
  without SWA. A 64k row of the 350M fits one H200: CP is needed only above ~121k tokens per row (section 4 of
  project notes, cp350).

### 8.1 Training context vs inference context: YaRN (2026-11-17, from the morning meeting)

The team's plan (Adhitya, CP): pretrain with 32k rows; the inference side reaches longer contexts (e.g. 131k) by
changing the RoPE scaling (YaRN), without training on 131k rows. **This is exactly what Qwen does:**
- **Qwen3 report:** the long-context pretraining stage trains on 32,768-token sequences and raises the RoPE base
  from 10,000 to 1,000,000 (ABF). For longer inputs, "to enable length extrapolation, we utilize YARN with a
  scaling_factor=4" (with Dual Chunk Attention), i.e. 4x the trained length at inference.
- **Qwen3 model cards:**
  - Qwen3-4B / 8B: "32,768 natively and 131,072 tokens with YaRN".
  - The YaRN block (`"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768`) is added to
    the config only when long inputs are needed. The card's note: static YaRN "remains constant regardless of input
    length, potentially impacting performance on shorter texts".
  - Qwen3-0.6B lists 32,768.
- **So the two numbers mean different things:**
  - the training context = the row length the model was trained on (32k): this is what sets memory and the CP need
    in training;
  - the inference context = what it can read when served (131k = 32k x YaRN factor 4).
- **YaRN in one line:** it keeps the RoPE dimensions that encode short distances as trained and stretches the ones
  that encode long distances, so positions up to factor x the trained length map into the trained range (plus a
  small attention temperature change).
- **With SWA:** YaRN goes on the global layers only (gpt-oss: "extend the context length of dense layers ... using
  YaRN"); the windowed layers keep W = 1,024 and their RoPE base (section 8). Our export needs per-layer RoPE
  settings for that.
- **For our 350M with 32k pretraining:** SFT rows up to 32k need no CP (one H200 holds ~121k tokens per row), and
  serving at 131k = YaRN x4 on the global layers.

## 9. How the kernel libraries implement SWA for packed (varlen) rows (source code read 2026-11-17)

| Library | Packed rows + window | How the kernel does it (from the source) |
|---|---|---|
| FlashAttention 2 (`flash_fwd_kernel.h`, `flash_bwd_kernel.h`, `mask.h`) | yes (varlen + `window_size`) | **Forward:** per query block, `n_block_min = max(0, (m_block * kBlockM + seqlen_k - seqlen_q - window_size_left) / kBlockN)`, so the key loop starts at the window. **Backward:** per key block, `m_block_max = min(m_block_max, ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM))`, so the query loop stops at the window. **Masks:** element masks (`apply_mask_local`: `col_idx < row_idx + ... - window_size_left` is masked) only in the edge blocks. `seqlen_q/k` are the sequence's own lengths (cu_seqlens), so everything is per sequence |
| FlashAttention 3, Hopper (`hopper/block.h`) | yes | `get_n_block_min_max`: `n_idx_left = n_idx - window_size_left; n_block_min = max(0, n_idx_left / kBlockN)`; `get_m_block_min_max` for the backward; per-sequence lengths from `SeqlenInfo` (varlen); also `attention_chunk` (Llama-4 chunked local attention) and PackGQA |
| FlashAttention 4, CuTe DSL (`flash_attn/cute/block_info.py`) | yes | the same two functions: `get_n_block_min_max` (`n_idx_left = n_idx - window_size_left`) and `get_m_block_min_max` (`m_idx_left = m_idx + window_size_left`) with `is_local`, per-sequence `SeqlenInfoQK` |
| cuDNN SDPA (frontend `Attention.md`) | yes | the Hopper prefill and backward rows list Ragged / Padded layouts and Masking "None, Causal, Sliding window, Additive Bias, Softcap, Arbitrary masking"; `set_diagonal_band_left_bound` / `set_sliding_window_length` ("Masks columns at or before row_idx - left_bound") |
| PyTorch FlexAttention (`flex_attention.py`, `BlockMask`) | yes | a block-sparse list per query block (`kv_num_blocks`, `kv_indices`): only the listed key blocks are visited. Optional `full_kv_*` blocks "don't require masking at all, then we can skip applying mask_mod to these". Document + window = `generate_doc_mask_mod(and_masks(causal, sliding_window), document_id)` |
| xformers | yes | `BlockDiagonalCausalLocalAttentionMask` and variants (packed sequences, causal and local), implemented in the mslk package |
| FlashInfer (inference prefill / decode) | yes, forward only | `window_left: int` "Left window size for sliding-window attention; -1 disables it" for ragged / paged batches |

**Same design everywhere, and it is the design planned for our packed SWA (step 2):**
- **Forward:** for each query tile of a sequence, the key loop starts at max(sequence start, query tile - W) instead
  of the sequence start, and ends at the diagonal.
- **Backward:** for each key tile, the query loop ends at min(sequence end, key tile + W) instead of the sequence end.
- **Element masks:** only in the tile that holds the window edge and in the diagonal tile. Every other tile runs
  unmasked.
- **Ours:**
  - Navin's dense kernel already works this way ("pair p reads key tiles max(0, p - W/128) .. p"; backward
    `nIter = min(nIter, win/BM + 2)`).
  - For packed rows we replace 0 with the chat's first tile in the forward work list, and cap the backward block's
    query steps at the chat's end and the window.
  - Our plan builder can also pre-cut the ranges, as FlexAttention's block lists do.

## Sources

- Mistral 7B: https://arxiv.org/abs/2310.06825 ; https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
- Gemma 2: https://arxiv.org/abs/2408.00118
- Gemma 3: https://arxiv.org/abs/2503.19786
- gpt-oss: https://arxiv.org/abs/2508.10925 ; https://huggingface.co/openai/gpt-oss-20b
- SWAA: https://arxiv.org/abs/2512.10411
- LASER: https://arxiv.org/abs/2512.19701
- Sliding-window beats linear attention: https://arxiv.org/abs/2608.28444
- NLL-guided full-attention layer selection: https://arxiv.org/abs/2606.27791
- LongLoRA: https://arxiv.org/abs/2309.12307
- Longformer: https://arxiv.org/abs/2004.05150
- HF transformers masking_utils (`kv_idx > q_idx - sliding_window`) and FlashAttention interface (window
  "[i - window_size[0], i + window_size[1]] inclusive"): github.com/huggingface/transformers, github.com/Dao-AILab/flash-attention
- Mistral v0.2 sliding window discussion (community, unofficial): https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/85
- Section 7 (fetched 2026-11-17):
  - HF transformers `masking_utils.py`, `modeling_flash_attention_utils.py`: github.com/huggingface/transformers;
  - FlashAttention `flash_attn_interface.py`: github.com/Dao-AILab/flash-attention;
  - FlexAttention blog: https://pytorch.org/blog/flexattention/ ;
  - TRL docs (raw markdown): https://github.com/huggingface/trl/blob/main/docs/source/reducing_memory_usage.md ;
  - TransformerEngine attention guide: https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/examples/attention/attention.html ;
  - vLLM hybrid KV-cache manager: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html ;
  - Gemma 3 4B IT config (mirror): https://huggingface.co/unsloth/gemma-3-4b-it/blob/main/config.json ;
  - Mistral reference code (cache.py): github.com/mistralai/mistral-inference.
- Section 7.3 numbers: scratchpad script swa_cost.py on ~/Downloads/sft135/data/{m1_smol_A,m2_tulu_A}/train (rows of 4,096).
