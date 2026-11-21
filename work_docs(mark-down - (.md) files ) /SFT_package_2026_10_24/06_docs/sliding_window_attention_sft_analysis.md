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
  - *(2026-11-19, reproduced by us, step 2 of the SWA plan:)*
    - **Source:** his exact committed code, copied with `git archive` from P015 into our folder on P035
      (`/localstorage/gautam_k/swa_1119`): BluTrain `ac6714d`, TI `b913f32`, BluBLAS `ea666ab`.
    - **Run:** built there, then `test_swa_attention` run once on P035 GPU 5 (H200, the user's permission).
    - **Result:** **ALL PASSED, 24 of 24 checks**:
      - exact key counts (|LSE - log(keys seen)| <= 1.07e-6);
      - spike rows / dV counts 0 off;
      - random inputs against the fp64 reference: the window error is about the causal error;
      - window = T bitwise equal to causal, and windowed runs repeat bit for bit.
    - **Speed** at B 16, heads 4/2, T 4,096:
      - forward 0.391 ms causal vs 0.214 ms W 1,024 (x1.82); backward 1.249 vs 0.771 ms (x1.62);
      - W 512: forward x2.67, backward x2.16.
      - These are within ~2% of his numbers above.
    - **Records:** `profiling_reports/swa_navin_test_2026_11_19/` (test_navin.log, status, scripts).
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

Note (2026-11-19, the user's decisions after the preproduction-135M comparison runs):
- **SWA comes first**, before the forward / backward cross-check items (written as future tasks in
  `attention_explanation.md` section 9.1).
- **Every applicable optimization is also tried on SWA.** Because SWA is built into our current kernels (not on
  Navin's 2026-11-04 base), everything already in them (October rebuild, mixed units, ideas 1, 2 and 4, rules 1 and 2b)
  carries over; the section 9.1 items are then tried on SWA too (the window's left edge has its own all-zero blocks,
  like the diagonal).
- **A complete rival survey before the kernel work:** who implements SWA (PyTorch, kernel libraries, pretraining
  frameworks, post-training frameworks, inference engines), for pretraining rows and for packed SFT rows, forward and
  backward; then the rivals to race are fixed, and the strongest one is the target. Section 9 below already covers
  the kernel libraries read on 2026-11-17.
- **One kernel for both:** the pretraining row is the packed math with one segment per row, so the SWA support and
  every SWA optimization serve pretraining and SFT together (both shapes are measured).
- **What `b913f32` wires (from its stat, 2026-11-19):** a Tensor-Implementations commit of 10 files: the forward
  (`GQA_fwd_sm90.cu` MODE_WIN) and backward (`GQA_bwd_vj2_sm90.cu` WIN) kernels, their dispatchers
  (`GQA_fwd_generic.cu`, `GQA_bwd_generic.cu`), the API (`AttentionKernels.h`: `gqa_flash_attn_{forward,backward}_qkv`
  with a trailing `window`, 0 = full), the autograd op (`AttentionOps.h/.cpp`) and its backward node
  (`AttentionBackward.h/.cpp`), and `Tests/kernel_tests/test_swa_attention.cu` (fp64 reference, exact key counts).
  Any path that cannot honour a window throws (no silent full attention). **Not in it:** packed rows (both kernels
  refuse them), the trainers (the `BLU_SWA_WINDOW` / `BLU_SWA_GLOBAL_EVERY` code of the screen binary is elsewhere on
  P015), a production YAML key, our SFT trainer, the HF export code. Context parallelism needs no change. Nothing of it
  has run on a GPU on my side yet (2026-11-17 was a reading review).

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

**How tiles are accessed and computed in causal_attention in pretraining and sft/post_training (with and without
sliding_window_attention (swa)) : (animation)** *(added 2026-11-20)*
- SFT packed row (3 chats): https://claude.ai/artifact/EBiPEihREutWGnkpQqi5Sy
- Pretraining row (one sequence): https://claude.ai/artifact/EBiPEihREutWGnkpQqi5Sy#pretraining
- It shows the three conditions of section 7.1 tile by tile:
  - same chat: tiles between chats never exist;
  - causal: a mask in the diagonal tile only;
  - window: a mask in the window-edge tile only, and the tiles further back are skipped.

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

*(2026-11-19: section 10 adds the backward check per kernel, every pretraining and post-training framework, the
published H100 speeds, and the race list. FA3 is the first rival.)*

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

## 10. Every SWA implementation for pretraining and post-training, and the race list (2026-11-19, step 1 of the SWA plan)

**What was checked:**
- The source code of 38 projects at their tip on 19/11 (commit list in 10.6), plus PyTorch main `3dc057e03e` (19/11)
  and JAX main. Read only: nothing was run.
- The question for each project: does it do SWA in training (forward AND backward), with packed rows, on H100 / H200
  (SM90), and which kernel does the work?
- Sections 7.2 and 9 (17/11) stay as they are; this section adds the missing pretraining frameworks, post-training
  frameworks and kernel libraries, and the backward check per kernel.

### 10.1 Kernels (the real rivals: every framework below ends in one of them)

| Kernel | Window forward | Window backward | Packed rows | H100 / H200 | Evidence (file:line at the 19/11 tip) |
|---|---|---|---|---|---|
| FlashAttention 2 (`flash_attn`, csrc) | yes | yes | yes (varlen) | runs, not written for Hopper | section 9 |
| FlashAttention 3 (`hopper/`) | yes | yes | yes (varlen), GQA + PackGQA | Hopper-only kernel | backward template has `Is_local` (`flash_bwd_launch_template.h:28`); `params.is_local` set from `window_size_left/right` (`flash_api.cpp:149`); refused only in builds made with `FLASHATTENTION_DISABLE_LOCAL` (`flash_api.cpp:165-166`) |
| FlashAttention 4 (`flash_attn/cute`, CuTe DSL) | yes | yes, also on SM90 | yes (`cu_seqlens`), GQA | SM80-SM120 | `flash_bwd_sm90.py`: `is_local`, `window_size_left/right` (lines 58, 371-372), varlen (388-397), `qhead_per_kvhead` (56); `interface.py` has no window restriction for SM90 (only "local not yet supported with qv", an MLA-only path) |
| cuDNN SDPA | yes | yes | Ragged (THD) | Hopper prefill and backward rows | support matrix (backend 9.18.1): Hopper Prefill and Bprop list "Ragged" layouts and masking footnote 4 "None, Causal, Sliding window, ..." (`docs/operations/Attention.md:50-52, 167`); `set_sliding_window_length(W)` = `set_diagonal_band_left_bound(W)`, "masks columns at or before row_idx - left_bound" (lines 292-307, 436) |
| PyTorch FlexAttention | yes (any `mask_mod`: window AND same document) | yes | yes (document mask) | Triton kernel; `BACKEND="FLASH"` = the FA4 CuTe kernels, "Experimental", with a block-sparse FLASH backward | `flex_attention.py:151, 315, 964` |
| PyTorch `varlen_attn` (`torch.nn.attention.varlen`) | yes | yes | yes | a window other than full / causal rejects cuDNN ("window_size must be (-1, -1) or causal (-1, 0)", `varlen.py:160-161`) and runs `aten._flash_attention_forward/backward` (lines 280, 771): FA2 built into PyTorch, or FA3 / FA4 when activated (torchtitan does this, 10.2) | `varlen.py` |
| xformers | yes | yes | yes | | section 9 (`BlockDiagonalCausalLocalAttentionMask`) |
| MagiAttention FFA (SandAI) | yes | yes | yes (query / key ranges) | built on FA3 (Hopper); `FFA_FA4` for Blackwell | `infer_attn_mask_from_sliding_window` turns a window into `AttnSlice` ranges (`magi_attention/api/functools.py:180`); extra mask types INV-CAUSAL / BI-CAUSAL exist because a window would otherwise need "each row individually" (`docs/source/blog/magi_attn.md:122`); "MFU comparable to FA3" (line 109) |
| FlashMask (Baidu) | yes | yes | yes | PaddlePaddle only | run through `paddle.nn.functional.flash_attention.flashmask_attention` (MagiAttention `exps/attn/baselines/attn_impl.py:69-75`); a Hopper version is in the FlashMask-3 paper (arXiv 2610.17901, Oct 2026) |
| FLA `parallel_attn` (flash-linear-attention) | yes | yes (separate dQ and dK/dV kernels) | yes (`cu_seqlens`) | Triton, any GPU, not written for Hopper | `fla/ops/attn/parallel.py`: forward start `max((i_t*BT - W + 1)//BS*BS, 0)` (line 101); dK/dV end `min(..., (i_t+1)*BT + W - 1)` (line 471) |
| JAX `jax.nn.dot_product_attention` | yes (`local_window_size`) | yes | sequence lengths | cuDNN on GPU | `jax/_src/nn/functions.py:1029` |
| FlashInfer | yes (`window_left`) | NO attention backward | ragged / paged | | inference only (no autograd attention at the tip) |
| NATTEN | a causal 1-D neighborhood is a window per sequence | yes (Hopper / Blackwell kernels) | packing only for plain attention, not for the window mode | | `src/natten/functional.py:74-181`: not a packed-SWA rival |
| TileLang | example code: sink + sliding window, forward + backward, varlen GQA | | | | `examples/attention_sink/`: examples, not a library |
| FlagAttention | only a gated window variant ("Wall Attention"); plain SWA goes to flash-attn | | | | `README.md:49-50, 297`: not a plain-SWA rival |
| ThunderKittens | no SWA attention kernel (window code only in linear-attention demos) | | | | no |
| Block-Sparse-Attention (MIT Han lab) | streaming masks (sink + local), built on FA2, aimed at inference | | | | not a training rival |

### 10.2 Pretraining frameworks: which kernel does their SWA

| Framework | How SWA is set | Kernel on H100 / H200 | Evidence |
|---|---|---|---|
| Megatron-Core (also NeMo, Megatron-Bridge) | `window_size` + `window_attn_skip_freq` (which layers are full) | TransformerEngine: **on Hopper+ TE picks FusedAttention = cuDNN whenever both cuDNN and FlashAttention qualify, "for performance reasons"**; between FlashAttention versions it prefers FA3 over FA4 on SM90 ("FA4's SM90 backward has limitations (MLA, non-standard head dims, SplitKV)") | Megatron `transformer_config.py:310-314`, `extensions/transformer_engine.py:2481-2488`; TE `dot_product_attention/utils.py:671-682, 1960-1966` |
| TransformerEngine (rules) | FlashAttention 2/3/4 and FusedAttention (cuDNN) all take a window; cuDNN SWA only without dropout and with s_q <= s_kv; FA4 off for SM100 head dim 256 with a window | | TE `utils.py:1463-1525` |
| torchtitan | `varlen` backend: PyTorch `varlen_attn` with a window, routed to FA3 on Hopper / FA4 on Blackwell; `flex` backend: window + document `mask_mod`; `flex_flash`: Flex on the FA4 CuTe kernels (Hopper / Blackwell only) | FA3 (varlen) or FlexAttention | `tools/utils.py:72-91`, `models/common/config_utils.py:100-112`, `models/common/attention/attention.py:325-336, 1047` |
| MaxText (JAX) | `sliding_window_size` | GPU: cuDNN through TE-JAX `DotProductAttention`, packed THD rows built from segment ids, `window_size` passed; TPU: Splash `LocalMask` | `src/maxtext/layers/attention_op.py:3140-3207, 2946` |
| OLMo-core (OLMo 3) | backends torch / flash_2 / flash_3 / flash_4 / te / flex, all take a window | **OLMo 3 configs: pattern [4096, 4096, 4096, -1], last layer full, default backend flash_2.** The same layout as our 350M (3 windowed : 1 global, last layer global), with W 4,096 instead of 1,024 | `src/olmo_core/nn/attention/backend.py:43-76`, `nn/transformer/config.py:820-831` |
| llm-foundry | `sliding_window_size` | FA2 `flash_attn_varlen_func(window_size=...)`; last commit 2026-03-25 | `llmfoundry/models/layers/attention.py:418-430` |
| GPT-NeoX | `sliding_window_width` | flash-attn `window_size` | `megatron/model/transformer.py:783-789` |
| nanotron | Qwen `sliding_window_size` | flex_attention or flash with the window | `src/nanotron/models/qwen.py:97-123` |
| modded-nanogpt (GPT-2 speedrun, 8x H100) | short / long windows per layer, growing during training | **varlen FA3** (a patched FA3 from the HF kernels hub); code note: "use flash_attn over flex_attn" | `track_1_short/model/attention.py:1-39, 217`, `track_1_short/config.py:74-87` |
| axlearn (Apple, JAX) | sliding-window bias | GPU: cuDNN (JAX cuDNN fused attention), except decoding / dropout / explicit bias; TPU: Splash | `axlearn/common/flash_attention/gpu_attention.py:792-857` |
| DeepSpeed | SWA only in its inference kernels | training uses the model's own attention code | `deepspeed/ops/transformer/inference/` |
| litgpt | `sliding_window_size` | builds a dense T x T mask in Python and calls SDPA: no fast SWA path | `litgpt/model.py:541-561` |
| ColossalAI | through the model files (Qwen2, Mixtral) | FA2 (`flash_attn`) in its shardformer attention; not checked deeper | `colossalai/shardformer/layer/attn.py:33, 621` |
| levanter (JAX) | Gemma 2 / 3 window in the model | repo frozen since 2025-11-07, not followed | `src/levanter/models/gemma.py:46, 471-533` |

### 10.3 Post-training frameworks

| Framework | Kernel for SWA with packed chats | Evidence |
|---|---|---|
| TRL | none of its own: the model's HF attention (flash varlen with the window, section 7.2) | no window code in `trl/` |
| Axolotl, LLaMA-Factory | HF attention (FA2 varlen with the model's window) | no window kernels of their own |
| Unsloth | FA2 varlen (`window_size_left/right`) if installed, else xformers, else PyTorch `varlen_attn` (torch >= 2.11 has `window_size`), else SDPA per segment | `unsloth/utils/attention_dispatch.py:48-50, 533-670` |
| torchtune | Gemma 2 window + packed documents through a FlexAttention `BlockMask` | `torchtune/models/gemma2/_attention.py:49-194` |
| OpenRLHF | HF `flash_attention_2` with `packing_samples` | `openrlhf/models/actor.py:62-152` |
| veRL | FSDP: HF `_flash_attention_forward`; Megatron: TE | `verl/models/transformers/monkey_patch.py:24-147` |
| NeMo-RL | Automodel: `flash_attention_2` with sequence packing; Megatron: TE | `nemo_rl/models/automodel/setup.py:380-395` |
| open-instruct (OLMo 3 post-training) | padding-free collator + HF attention named from the OLMo-core backends | `open_instruct/dpo_tune_cache.py:57-339`, `finetune.py:59` |
| ms-swift | HF trainers + `swift/megatron` (TE) | |
| xtuner | flash-attn `window_size` in its Mistral / Phi-3 dispatch | `xtuner/model/modules/dispatch/mistral.py:125`, `phi3.py:34-42` |
| slime | Megatron + its own `flash_dot_product_attention` plugin calling flash-attn with the layer's window | `slime_plugins/models/flash_dot_product_attention.py:57-180` |
| SkyRL | Megatron (TE) worker; an HF patch for the gpt-oss window | `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py:218-229` |

**Reading:**
- No framework, for pretraining or post-training, has its own SWA kernel.
- Every one ends in: FA2, FA3, FA4, cuDNN, FlexAttention (Triton or FA4 CuTe), or PyTorch's flash path (FA2 built in,
  or FA3 / FA4).
- So the race list is these kernels.
- Inference engines (vLLM, section 7.2; SGLang; TensorRT-LLM; llama.cpp) are not training rivals and were not
  re-read today.

### 10.4 Published speed of the kernels on SWA (H100)

**Source:** the MagiAttention team's kernel benchmark (blog `cp_benchmark.md`, figures
`assets/magi_attn/exp/kernel/h100/sw_causal_mask/{fwd,bwd}/flops_report.png` in their repo).

**Setup:**
- H100; self-attention with ONE sequence per batch (not packed).
- GQA 64:8:8 heads, head dim 128, bf16, window 1,024 (the same window and head dim as our 350M; our GQA is 8:4).
- Sequence lengths 1k-32k.
- TFLOPs/s are counted on the mask area (only the computed pairs), so higher = faster for the same work.
- cuDNN was called through TransformerEngine; FA4 was not in the H100 test; library versions are not printed on the
  figures.

**Columns:** forward and backward TFLOPs/s at 4k / 8k / 32k tokens, read off the bar charts (about +-5).

| Kernel | fwd 4k | fwd 8k | fwd 32k | bwd 4k | bwd 8k | bwd 32k |
|---|---|---|---|---|---|---|
| FA3 | ~536 | ~524 | ~523 | ~333 | ~329 | ~332 |
| MagiAttention FFA | ~459 | ~469 | ~453 | ~300 | ~312 | ~310 |
| cuDNN | ~372 | ~391 | ~393 | ~265 | ~282 | ~282 |
| FA2 | ~245 | ~256 | ~256 | ~210 | ~220 | ~224 |
| FlashMask | ~253 | ~256 | ~190 | ~192 | ~176 | ~181 |
| FlexAttention | ~3 | ~1 | failed | ~6 | failed | failed |

**Reading:**
- FA3 is the fastest at every length from 2k up, forward and backward.
- Against FA3, from 4k up:
  - FFA reaches 86-90% forward and 90-95% backward;
  - cuDNN reaches 69-75% forward and 80-86% backward.
- Their FlexAttention bars (1-11 TFLOPs/s, failures from 8k / 16k) are far below every other report and look like a
  setup problem. We measure Flex ourselves.
- At 1k the window covers the whole sequence (= plain causal), so the 1k bars say nothing about SWA.

### 10.5 The race list and the first rival

**Race on H200** (the 18/11 campaign venv on P035 already has FA3, FA4 and cuDNN; Flex comes with PyTorch):
1. **FA3, varlen + `window_size`: the FIRST rival.**
   - Fastest in the published H100 SWA test (10.4), forward and backward.
   - Our closest rival in the 18/11 GQA race: ours took 0.43-0.93 of its time, vs 0.40-0.91 of FA4's and 0.24-0.81
     of cuDNN's.
   - It is what modded-nanogpt and torchtitan (varlen on Hopper) run, and what TE prefers over FA4 on SM90.
2. **cuDNN** (ragged + sliding window): the default of Megatron / NeMo / MaxText on H100 / H200 (TE picks it on
   Hopper+).
3. **FA4** (CuTe, SM90 forward + backward with a window).
4. **FlexAttention**: Triton, and `BACKEND="FLASH"` (the FA4 CuTe kernels with block lists). It is the torchtune /
   torchtitan / nanotron path.
5. **FA2**: what HF / TRL / OLMo-core / OpenRLHF / NeMo-RL use by default.
6. Optional:
   - MagiAttention FFA: needs its own JIT build; published at 86-95% of FA3 on SWA;
   - FLA `parallel_attn` (Triton).
   - FlashMask needs PaddlePaddle installed: skipped unless asked.

**Shapes:** our 350M (8 query heads, 4 KV heads, head dim 128, W 1,024):
- plain rows of 4,096 (pretraining);
- packed chat rows from m1 / m2 (SFT).

**Window convention (each library counts the window differently, by one):**

| Library | What it takes |
|---|---|
| FA2 / FA3 / FA4 | `window_size_left = W - 1` (HF passes `(sliding_window - 1, ...)`, section 7.2) |
| cuDNN | `left_bound = W` ("masks columns at or before row_idx - left_bound") |
| torchtitan / Flex mask | W keys including the query itself |
| PyTorch `varlen_attn` | docs say "(W, 0)": to be checked |

The race harness must set each library so that all compute exactly W = 1,024 keys per query. Check it against the
fp64 reference before any timing.

**The best algorithm:**
- All serious kernels use the same method:
  - FlashAttention tiles;
  - the key loop (forward) starts at the window, and the query loop (backward) stops at the window;
  - element masks only on the window-edge tile and the diagonal tile.
- The differences are in how well each is written for the GPU. FA3 = Hopper warp specialization + TMA + pipelining.
- The other forms are generalizations for arbitrary masks: FFA's ranges with atomic adds, FlashMask's per-column
  bounds, Flex's block lists, cuDNN's diagonal band. For causal SWA inside packed chats, direct loop bounds are the
  simplest, and in the published test the fastest.
- Our plan (section 9) is direct loop bounds, i.e. the FA3 design, inside our GQA kernels. Those kernels already beat
  FA3 on GQA (18/11).

### 10.6 Sources for section 10 (all read 2026-11-19, shallow clones of the tip)

Repos (commit, commit date):
- **Megatron and NVIDIA:** NVIDIA/Megatron-LM ccbf6563e2 (11-19); NVIDIA/TransformerEngine c2b5a322fc (11-18);
  NVIDIA/cudnn-frontend 86b74686c8 (11-19); NVIDIA-NeMo/RL 49822fcb67 (11-19).
- **Pretraining frameworks:**
  - pytorch/torchtitan 8592e69c3e (11-19); AI-Hypercomputer/maxtext 1e4e716f8e (11-19);
  - allenai/OLMo-core 704625ad94 (11-14); allenai/open-instruct 2f7889da92 (11-18);
  - mosaicml/llm-foundry 0cdb2f4296 (03-25); EleutherAI/gpt-neox 97bc4be320 (11-07);
  - huggingface/nanotron a1d47c7ef0 (10-08); deepspeedai/DeepSpeed b5e0ba67f2 (11-19);
  - KellerJordan/modded-nanogpt 4ea6b93733 (09-28); Lightning-AI/litgpt 5341568a65 (11-10);
  - stanford-crfm/levanter 982cef7f1d (2025-11-07); apple/axlearn 34aed43c1b (07-08);
  - hpcaitech/ColossalAI 14d1c15147 (11-18).
- **Post-training frameworks:**
  - huggingface/trl 1b3216ac65 (11-19); axolotl-ai-cloud/axolotl 8dc5974745 (11-19);
  - hiyouga/LLaMA-Factory 57712ad195 (11-19); unslothai/unsloth db42f61a40 (11-19);
  - unslothai/unsloth-zoo b3b1af014c (11-19); pytorch/torchtune ef542e4628 (11-09);
  - OpenRLHF/OpenRLHF 26494490bc (11-03); volcengine/verl a4e0dd2188 (11-19);
  - THUDM/slime 79aa41a884 (11-06); NovaSky-AI/SkyRL f789052fa9 (11-18);
  - modelscope/ms-swift 78a1697535 (11-19); InternLM/xtuner 090ec628c6 (11-19).
- **Kernel libraries:**
  - Dao-AILab/flash-attention 68bd095cdd (11-17); SandAI-org/MagiAttention 02c765aad5 (11-18);
  - SHI-Labs/NATTEN 39375ea1d2 (07-26); HazyResearch/ThunderKittens 67845f5fa4 (09-12);
  - tile-ai/tilelang b1367a59ce (11-19); fla-org/flash-linear-attention 42c22207f6 (11-19);
  - mit-han-lab/Block-Sparse-Attention 49d6c39e4d (01-17); FlagOpen/FlagAttention f1a6c36baa (11-19);
  - flashinfer-ai/flashinfer f914c29847 (11-19).
- Dates are 2026 unless written.

Other sources:
- PyTorch main 3dc057e03e (2026-11-19): `torch/nn/attention/varlen.py`, `torch/nn/attention/flex_attention.py`.
- JAX main: `jax/_src/nn/functions.py`.
- PyTorch varlen docs (2.14 / 2.15):
  - https://docs.pytorch.org/docs/2.14/nn.attention.varlen.html
  - https://docs.pytorch.org/docs/2.15/nn.attention.varlen.html
- MagiAttention kernel benchmark:
  - https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html
  - figures in the repo, path in 10.4.
- FlashMask-3 paper: https://arxiv.org/abs/2610.17901
- FlexAttention + FA4 backend: https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- Laptop copies: scratchpad `rivals_src/` (manifest.tsv) and `web/`. These are temporary; this table is the record.


## 11. Packed rows + window (step 7): how every library implements it, the extra checks they make, the bugs they hit, and what we take from it (2026-11-20)

**Why this section.** The user (20/11, during step 7): "did you compare how others implement it, what extra checks or
things they added? Research fully, so we don't make mistakes or miss anything." Sections 9 and 10 (17/11, 19/11) say
WHICH kernel each framework uses and the loop bounds. This section reads the source for the details: the edge cases
each library handles, the checks in their APIs, their tests, their bug history (commit log), and the model / serving
side (how HF and vLLM read the window), and compares each point with our step-7 code (forward modes 4 / 5, the backward
WIN copies for packed rows; laptop scratch copy of TI 55d1058, not pushed).

**What was read** (shallow clones of the tip on 20/11, read only, nothing run):
- kernels: Dao-AILab/flash-attention `dbabcc7` (2026-11-19; full commit history read for every local / window fix),
  NVIDIA/cudnn-frontend `2a74b4a`, NVIDIA/TransformerEngine `e6f9e862` (full history), pytorch/pytorch `d473a25`
  (`varlen.py`, `flex_attention.py`, their tests), facebookresearch/xformers `eedaeaf`, flashinfer `db23734`;
- model / serving: huggingface/transformers `439bba0` (masks, flash path, Qwen3), vllm-project/vllm `0fda86a`;
- frameworks: pytorch/torchtitan `b1c259c`, NVIDIA/Megatron-LM `84c7f16`, allenai/OLMo-core `704625a`,
  KellerJordan/modded-nanogpt `4ea6b93`, SandAI-org/MagiAttention `02c765a`;
- Navin's exporter (P015 `/localstorage/test/BluTrain/swa_ablation/tools/blu_to_hf.py`) and our SFT package's
  `03_trainer/export_to_hf.py`;
- web search for public bug reports (one hit worth keeping: Megatron-Bridge PR #2656, below).

### 11.1 The algorithm: the same everywhere, and the same as our step 7

| Point | FlashAttention 3 (the reference design) | Ours (step 7) | Same? |
|---|---|---|---|
| Window meaning | `window_size_left = W - 1`: keys `[i - (W-1), i]`; the element mask hides `col < row - window_left` (`hopper/mask.h:104-115`) | `masked = key <= qry - win`: W keys, itself included | yes |
| Per sequence | every index is inside the sequence (`SeqlenInfo`, cu_seqlens) | every index is chat-relative (design A) | yes |
| Forward key range | `n_block_min = max(0, (m_block*kBlockM + seqlen_k - seqlen_q - window_left) / kBlockN)` (`hopper/block.h` `get_n_block_min_max`) | pair p reads key tiles `max(0, p - W/128) .. p` | yes (W multiple of 128: exactly one edge tile, see 11.4) |
| Backward query range | `m_block_max = min(.., ceil(((n+1)*kBlockN + window_left) / kBlockM))` (`get_m_block_min_max`) | `nIter = min(steps to the chat end, W/64 + 2)`: rows up to `128kt + W + 127` cover the last query `128kt + 127 + W - 1` | yes |
| Masked tiles | causal tiles first, unmasked middle, local-masked tiles at the end (`get_n_block_min_before_local_mask`) | causal mask in the diagonal tile, window mask in the edge tile, all others unmasked | yes |
| Fully hidden row in a tile | `Check_inf`: max -inf -> scale 0, no NaN (`softmax.h:64-78`); on in the first tile always and in the local-masked tiles (`mainloop_fwd...:1181, 1263`) | the same guard in the window-edge tile (MASK 2); every row has a visible key in the next tile (proof in 11.4) | yes |
| Rows with no key at all | LSE = -inf, O = 0 (`softmax.h:151`) | cannot happen: in self-attention every query sees itself | n/a |
| Key block with no query | backward exits early and writes dK = dV = 0 (`mainloop_bwd...:441-445, 609-612`; `flash_bwd_kernel_sm90.h:271` `store_zero`) | cannot happen: `nIter >= 1` for every key block of a chat (a chat of 1 token: 1 step) | n/a |
| seqlen_q != seqlen_k (KV cache, chunked prefill) | bottom-right alignment `+ seqlen_k - seqlen_q` everywhere | not supported (training self-attention only) | **note**: needed only if our kernels ever serve inference |

### 11.2 The extra checks each library makes, and ours

| Check | Who does it (source) | Ours |
|---|---|---|
| A window that covers every sequence runs the plain causal kernel | FA2 / FA3: `if (window_size_left >= max_seqlen_k - 1) window_size_left = -1` (`hopper/flash_api.cpp:576`; FA2 `flash_api.cpp:423, 639`); commit `0842ec0` (2023-12-23) "Don't dispatch to local if window size >= seqlen_k" | dense rows: `W >= T` is plain causal. Packed rows: decided on T, not on the longest chat, so a batch whose chats are ALL <= W still runs mode 4 / 5. **Item A (11.6): use the plan's longest segment (`gen_max_len`) as FA does** |
| Causal forces the right side to 0 | FA2 / FA3 (`if (is_causal) window_size_right = 0`); TE `check_set_window_size` (`utils.py:3079-3121`) | our window has no right side (causal only) | ok |
| Window must be a positive integer / sentinel values | torchtitan `sliding_window_size must be an integer >= 1` (`attention.py:566-570`); OLMo-core "must be positive or -1" (`nn/attention/__init__.py:287-290`); TE checks per mask type | `window < 0` throws; `0` = off; `W % 128 != 0` throws (forward and backward) | ok; ours is stricter (multiple of 128) and says so |
| A path that cannot honour the window must not run silently | TE turns a backend off with a logged reason for every unsupported combination (FP8 + SWA on cuDNN < 9.21, dropout, s_q > s_kv, flash-attn < 2.3, top-left cross-attention: `utils.py:1464-1525`); PyTorch varlen refuses windows on cuDNN (`varlen.py:146-161`); HF docs warn that without the right kernels padding-free "mixes tokens across samples" and "the failure is silent" | every path without window support throws (non-Hopper, Vj1 / v44 / Vj1d, non-128 multiples); a chat plan on a path without plan support throws | ok |
| CP + SWA | TE: no cuDNN SWA with CP `p2p` / `a2a+p2p` (`utils.py:1401-1411`); SWA + CP for THD added `a75da0ca`; a CP window bug fixed `67c26e25` | our CP is Ulysses (all-to-all over heads): every GPU sees whole chats, so the window works unchanged | ok (re-check if ring / p2p CP is ever added) |
| Layers that share precomputed metadata must share the window | torchtitan `decoder.py:114-134` (layers of one backend share block masks: a different window is refused) | the plan is shared, the window is passed per call per layer | ok (different design) |
| Window sides near INT_MAX | FA4 commit `328dd7b` (2026-11-15) "Clamp huge local window sides before they overflow int32" | `windowed` needs `win < S`, so the kernels never see a huge window | ok |
| A window of 0 / "empty" windows | cuDNN: `left_bound = 0` is NOT "no bound", it masks every key (`Attention.md:436`); FA `d8dbfad` (window bound 0 treated as unbounded, fixed), `17498da` (intentionally empty windows widened to full attention, fixed) | `0` = off by definition (documented in AttentionKernels.h) | ok, but **our 0 means "off" while FA's left 0 means "itself only"**: keep the meaning in every export / harness |

### 11.3 Their tests, and what our step-8 test must add

| Test idea | Who (source) | In our step-8 plan? |
|---|---|---|
| Random window sizes per case | FA3 `test_flash_attn.py:206` (`torch.randint(0, seqlen_k, (2,))`); cuDNN fuzzer: random left bound 1 .. s_kv/2 (`sdpa/random_config.py:504-520`) | windows are multiples of 128 only: test W = 128, 256, 512, 1024, 2048, T - 128, and W >= T (= no window) |
| A SMALL window so an off-by-one is visible | TE `c3b3cd21` (2024-10-07): fixed cuDNN's window off by one and "reduce the window to make any errors more pronounced" | **add**: W = 128 with chats of W-1, W, W+1, 2W, 2W+1 tokens |
| Varlen with zero-length and odd lengths | FA3 `zero_lengths=True` (`:452-455`); our suite G1 has zero-length entries and 1-token chats | **add**: a zero-length segment entry and 1-token chats in the packed layouts |
| Tokens that belong to no sequence (gaps) | FA3 `add_unused_qkv` (`:368`); cuDNN ragged token / head gaps (`random_config.py:162-168`) | n/a: our plan must cover every token (the builder refuses gaps) |
| NaN in unused K / V slots must not change O | FA3 `:1161` (KV cache) | n/a for training (no unused slots; neighbours' values are finite); our isolation is "packed == alone, bit for bit" |
| GQA / MQA / MHA | FA3 `mha_type` | **add**: heads 8/4 (350M, idea 4 on), 4/1 (idea 4 off: per-query-head path), 8/8 (G = 1: no mixed units) |
| Deterministic repeat | FA (`deterministic`), cuDNN deterministic engine pins | **add**: repeat run: O, LSE, dK, dV bit for bit equal; dQ within FA's atol (our suite's G4 rule) |
| Reference rule | FA / TE: kernel error <= 2 x (PyTorch bf16 error) + atol | yes: fp64 per chat, Navin's T4 rule (window error <= 2 x causal error + 2e-3) |
| Bottom-right alignment q != k | PyTorch `test_numeric_window_bottom_right` (`:1517-1545`, repro of pytorch#203417) | n/a (self-attention) |
| Long rows | (our suite G14) | **add**: one row of 32,768 with one chat and W = 1,024 vs fp64 |
| Through autograd | (our E3) | yes: `scaled_dot_product_attention(chats, window)` = the direct calls |

The step-8 core check stays: **packed + window == every chat alone through the DENSE window kernel (MODE_WIN), bit for bit**
for O, LSE, dK, dV (dQ within FA's atol), with the suite's "alone" convention (row of roundup(L, 256), chat at
offset 0, random filler after it with dO = 0). The dense window kernel is the one validated in step 4 (Navin's 24/24,
our E1 against fp64), so this proves the packed version without trusting a new reference. Plus: chats <= W must be
bit for bit equal to the same packed batch with no window (needed for item A).

### 11.4 Two proofs for our design (W a multiple of 128)

- **Exactly one edge tile per pair.** Rows of pair p are 128p .. 128p+127. Row r sees keys r - W + 1 .. r. With
  W = 128w: the lowest visible key of row 128p is 128(p-w) + 1 (tile p-w, its first key hidden) and of row 128p + 127
  it is 128(p-w+1) (the first key of tile p-w+1, so tile p-w is fully hidden for that row). So the hidden keys of every
  row of the pair are in tile p-w or below, and tile p-w+1 is fully visible to every row: one masked edge tile, as the
  kernel assumes. FA3 needs up to two local-masked tiles because it accepts any W.
- **No NaN after the edge tile.** FlexAttention documents the trap (`flex_attention.py:240-252`, `ROWS_GUARANTEED_SAFE`):
  "Each row must have an unmasked key in the *first* block scheduled for it ... Local or windowed masks often do not
  ... and exp2 overflows to NaN." Ours: the first tile is the edge tile and has the guard; the second tile (p-w+1, or
  the diagonal when W = 128) holds key 128(p-w+1) <= r for every row, so every running max is finite from there on;
  the next rescale is `ex2(-inf - finite) = 0` on O = l = 0. Mixed units (mode 5) use the same rows and pair, so the
  same holds.

### 11.5 Model / serving side: two real problems found

| Finding | Source | Effect for us | Status |
|---|---|---|---|
| HF Qwen3 drops the window unless `use_sliding_window: true` | `configuration_qwen3.py:86`: `self.sliding_window = self.sliding_window if self.use_sliding_window else None`; layers read `layer_types` (`modeling_qwen3.py:208, 231`) | Navin's exporter is right (`use_sliding_window: W > 0`, `sliding_window: W`, `max_window_layers: L`, `layer_types`, `blu_to_hf.py:77-83`). **Our SFT package's `export_to_hf.py:165` writes `use_sliding_window: False, sliding_window: None`**: a SWA model exported with it becomes a full-attention HF model, silently | **must fix in step 9** (take Navin's fields) |
| **vLLM serves our layout with NO window at all** | `engine/arg_utils.py:2298-2304`: the model-level window is set only when every layer is sliding ("Otherwise CacheConfig.sliding_window will override the global layers"); `models/qwen3.py:360-370` builds the layers without `per_layer_sliding_window`; `layers/attention/attention.py:294-301` then falls back to the (unset) model-level window. Llama (`llama.py:185-204`) and Gemma 3 (`gemma3.py:153-185`) do read `layer_types` | an exported `Qwen3ForCausalLM` with 3 windowed : 1 global runs FULL attention in all 28 layers under vLLM: same as our training only up to 1,024 tokens; longer prompts get a different model and no KV-cache saving; no error | **must handle before any vLLM eval / serving**: a 5-line vLLM patch (pass `config.sliding_window if layer_types[i] == "sliding_attention"`, as llama.py does), or evaluate through HF; add an HF-vs-vLLM logits check on a prompt longer than W |
| Window off by one between conventions | Megatron-Bridge PR #2656 (2026-03): "Fix Mistral bridge to subtract 1 from sliding_window (was passing raw HF value)" (also Gemma 2 / 3, gpt-oss); TE `c3b3cd21` (cuDNN, 2024) | our window = HF `sliding_window` = W keys; converters must never pass W where W-1 is expected | the race harness must use: FA2/3/4 `(W-1, 0)`; cuDNN `left_bound = W`; Flex `q - kv < W`; **PyTorch `varlen_attn` `(W-1, 0)`, NOT `(W, 0)`**: its doc says "(W, 0) for ... sliding window of size W" (`varlen.py:384-386`) but its own test reference attends `i - W .. i` = W + 1 keys (`test_varlen_attention.py:262-271`) |
| Global layer placement | Megatron `window_attn_skip_freq = N`: layer_number (1-based) `% N != 0` is SWA (`utils.py:588-604`) = ours (layers 4, 8, .., 28 global); OLMo-core can force the first / last layer full (`nn/attention/__init__.py:263-283`) | same rule as ours; our last layer is global already | ok |

**Added 2026-11-20 (after 11.5 was written; the user: "did we document this?"):**
- **SGLang** (sgl-project/sglang `4b6985c`, 2026-11-20): `python/sglang/srt/models/qwen3.py` (1,003 lines) and `qwen2.py`
  contain no sliding-window code at all, so SGLang also serves our exported `Qwen3ForCausalLM` with full attention in
  all 28 layers, silently.
- **llama.cpp** (ggml-org/llama.cpp `4783b11`): sliding windows are handled per architecture (`hparams.is_swa(il)`,
  `llama-kv-cache-iswa`); the Qwen3 architecture file was not read, so a GGUF of our model must be checked the same way
  before it is used.
- **Why this stays written down even after we fix our side:** we cannot change vLLM / SGLang for everyone (a local patch
  is lost at their next update), and anyone who loads our exported model in them gets full attention with no error.

**Serving / evaluation checklist for a model trained with SWA (ours: W 1,024, every 4th layer global):**
1. HF export: `use_sliding_window: true`, `sliding_window: 1024`, `layer_types` (layer i "full_attention" iff
   (i + 1) % 4 == 0, else "sliding_attention"), `max_window_layers` = the layer count (Navin's `blu_to_hf.py:77-83`;
   our SFT package exporter is fixed in step 9).
2. HF transformers: honours it (per-layer window from `layer_types`). The reference for every check below.
3. vLLM: does NOT honour it for Qwen3 (all layers full). Patch `models/qwen3.py` to pass
   `per_layer_sliding_window = config.sliding_window if layer_types[i] == "sliding_attention" else None` (as `llama.py`
   and `gemma3.py` do), or evaluate through HF.
4. SGLang: does NOT honour it for Qwen3; same choice (patch or HF).
5. Any engine, before its numbers are used: logits of our trainer vs the HF export vs the engine on a prompt LONGER than
   1,024 tokens (below 1,024 the window changes nothing, so a short prompt cannot show the mismatch).

### 11.6 What we change or add (proposals; each waits for the user's OK)

| # | Item | Why | Where |
|---|---|---|---|
| A | Packed rows: run the window kernels only if some chat is longer than W (`window < plan->gen_max_len`), in the forward AND the backward routing (same rule, same plan, so both agree) | FA does exactly this (`0842ec0`); SFT batches whose chats all fit in W then run the kernels of before (bit-identical results, proven by the step-8 check "chats <= W: window == no window") | `GQA_fwd_generic.cu`, `GQA_bwd_generic.cu` (2 lines each) |
| B | Step-8 test additions from 11.3 (W = 128 with W-1 / W / W+1 / 2W / 2W+1 chats, zero-length and 1-token entries, heads 8/4 + 4/1 + 8/8, repeat determinism, one 32k row, autograd, chats <= W == no window) | the checks others found necessary; TE's off-by-one was only visible with a small window | `test_swa_packed.cu` (new) + `test_swa_extra.cu` E2 (packed + window must now RUN) |
| C | Export: SWA fields as in Navin's exporter | HF drops the window otherwise (11.5) | step 9 (`export_to_hf.py`) |
| D | vLLM: patch Qwen3 per-layer windows, or evaluate with HF only; plus an HF-vs-vLLM logits check past W | vLLM silently serves full attention for our layout (11.5) | step 9 / 11 (eval) |
| E | Race harness window conventions (11.5 row 3) checked against fp64 before timing | each library counts the window differently | step 12 |

**Done 2026-11-20 (the user's OK: "after fixing these 2 we will proceed"):** items A and B are in the step-7 patch
(`profiling_reports/swa_port_2026_11_20/swa_step7_packed_window.patch`, 7 files, md5 d6820acb; applied on the laptop
tree, 7 / 7 md5 equal to the files tested). **Step 8 on P035 GPU 7, 17:08-17:28 IST, all passed:**
- Navin's `test_swa_attention`: 24 / 24 (dense window, unchanged code);
- `test_swa_extra`: 13 / 13 in each of the 3 backward variants (default idea 4, `BLU_VJ2_HSUM=0`,
  `BLU_VJ2_DKDV_F32=1`); E2 now: packed + window 1024 runs, packed + window 1000 throws;
- `test_swa_packed` (new): 72 / 72 in each of the 3 variants:
  - P1: every chat of every layout == the same chat alone through the dense kernels, bit for bit for O, LSE, dK, dV;
    dQ at most 0.12 of FA's atol;
  - P2: chats <= W unchanged by the window, bit for bit, in batches where the window kernels ran (the proof behind A);
  - P3: chats > W vs fp64: every windowed error <= 2 x its causal error + 5e-6 (the printed worst margin -2.00e-03
    against a rule that allows 2e-3; -1.59e-03 once with fp32 partials);
  - P4 (item A), P5 refusals (incl. the zero-length segment), P6 repeat, P7 the 8k row, P8 autograd: all pass;
  - mixed units really ran (plan `mx_hkv` 4 for heads 8/4, 1 for 4/1; 0 for 8/8 as expected);
- the 18/11 push checks (no window): quick suites 273 / 0 and 280 / 0 (the same counts as before), dK KNOWN entry,
  idea-4 messages: all pass.
Records: `profiling_reports/swa_port_2026_11_20/box_results/step8/`. Commit text ready: `commit_ti_step7.txt`.

**Step 9 done 2026-11-20 (SWA in the SFT trainer; P035 GPUs 7 + 5, 19:42-20:01 IST; records
`profiling_reports/swa_port_2026_11_20/step9/`):** the window travels with the model (checkpoint record
`model_config.json`, converter `meta.json`, `BLU_EXPORT` metadata, HF `config.json`) and the SFT trainer compares the
base's record with its own window at start-up (stops on a mismatch; `BLU_SWA_MISMATCH_OK=1` for a deliberate arm).
Results, 350M (Navin's `kv4_swa3` base converted with the step-9 converter; SFT rows of 4,096):
- kernel test incl. the 350M CP head shapes (4/2, 2/1): 94 / 94 x 3 variants;
- **trainer vs HF transformers, step-0 val loss on 2,000 val chats (844 longer than 1,024):** window ON 1.732441 vs
  1.732514, window OFF 1.733759 vs 1.733824: the same model to 0.00007 in both modes; the window's effect +0.00132
  (trainer) vs +0.00131 (HF): the SWA base is better WITH its window, as expected;
- guard: window 0 without the override stops at start-up; with it, a warning;
- a run that ends: export + checkpoint record carry 1024 / 4; the exported HF config equals Navin's;
- SFT_CP=2 vs 1 GPU: step-0 gradient norm 1.5179 vs 1.5181, loss at step 29 1.682096 vs 1.682022;
- speed (1 H200): 474 ms per step (81,920 tokens, ~173k tokens/s) with the window vs 476 ms without (SFT data: most
  chats are short); CP=2 313 ms. One Tulu epoch (463.1M tokens) is about 45 min on one GPU.
- found and fixed: the converter stopped on an F32 export at `layers.0.q_gamma` (the QK-norm scales are fp32 in the
  trainer, so not bf16-exact); the HF checker's window-OFF mode (layer_types must become all full).

**Data for the SFT arms (the user, 2026-11-20): Tulu 3 (`m2_tulu_A`, 744,023 chats, 463.1M tokens per epoch, 311.6M
trained; Aya + hardcoded removed). Arms: A (kv4 full / full), C (kv4_swa3 + window), D (kv4_swa3 + full, override);
arm B dropped (a full-attention base converted to SWA in SFT is not our case).** OLMo 3 Instruct SFT counted with our
prep for comparison: 1,928,901 of 2,152,112 chats kept (197,363 tool-call chats our template has no place for), 1.198B
tokens per epoch, the same length profile as Tulu (2.6x larger).

**Optimization notes for step 14 (from the same reading):**
- the mirror of idea 1 at the window edge: in a key block's LAST backward step (queries `k0 + W + 64 ..`) group 0's
  keys `k0 .. k0+63` are all at or before `query - W`, so group 0 is fully masked there and could skip like idea 1
  does at the first step (today it computes zeros);
- FA4 `7108d1c` "R2P dual bound masking for local attention" (register-to-predicate masks) and `eed1971` (skip masks
  on unmasked blocks: ours already does);
- the plan's longest-chat-first order assumes causal cost (L^2/2); with a window a chat costs about L x min(L, W):
  weight the order by that (FA3's varlen scheduler orders by real per-tile cost);
- FLOP count for MFU (SFT wiring, step 9): exact pairs per chat `W(W+1)/2 + (L - W) W` for L > W, as OLMo-core
  (`nn/attention/__init__.py:609-624`); FA fixed a typo in its local FLOP count (`e980f0f`).

**Design choices others made, for later architecture decisions (not bugs):** Gemma 3 uses a smaller RoPE base on the
local layers (10k) than on the global ones (1M); Llama 4 / OLMo-core `no_global_rope` drop RoPE on the global layers;
gpt-oss pairs W = 128 with learned sinks (FA4 has sink support in forward and backward); modded-nanogpt grows the window
during training (our kernels take W per call, any multiple of 128).

**Sources (all read 2026-11-20):** the repos and commits listed at the top of this section (file:line in the tables);
FA commits `0842ec0`, `26d7d92`, `e980f0f`, `d8dbfad`, `17498da`, `328dd7b`, `eed1971`, `7108d1c`, `7cd4fcb`; TE commits
`c3b3cd21`, `838345eb`, `a75da0ca`, `67c26e25`, `c6a92a4d`; Megatron-Bridge PR #2656
(https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/2656); HF padding-free docs
(https://huggingface.co/docs/transformers/en/padding_free); laptop copies in the session scratchpad `rivals2/`
(temporary; this section is the record).

## 12. The SWA plan, step by step: what was done, with every result (2026-11-19 / 2026-11-20)

The plan of 6.1 grew into the numbered steps below (agreed with the user 2026-11-19 / 20; the 17/11 rule "optimize only
if SWA stays in SFT" was dropped: our 350M is pretrained with SWA, so the speed work pays in pretraining anyway). This
section is the one list of what happened; sections 10 and 11.6 keep the details they already had. All runs on P035
H200s, each on the GPU the user named.

| Step | What | When (IST) | Result | Details |
|---|---|---|---|---|
| 1 | rival survey (pretraining + post-training, plain + packed rows) | 19/11 | no framework has its own SWA kernel; first rival = FA3 varlen + `window_size` | section 10 |
| 2 | Navin's `b913f32` built unchanged (BluTrain `ac6714d`, TI `b913f32`, BluBLAS `ea666ab`), Navin's test | 19/11 20:51, GPU 5 | 24 / 24 | section 3 note |
| 3 | port onto our kernels (TI `334b244`) | 20/11 morning | below (12.1) | 12.1 |
| 4 | tests of the port | 20/11 13:12-13:21, GPU 1 | Navin's test 24 / 24; `test_swa_extra` 12 / 12 x 3 backward variants; quick suites 273 / 0 and 280 / 0, dK KNOWN, idea-4 messages: pass | 12.1 |
| 5 | pretraining trainer + smoke | 20/11, GPU 1 | 350M step -8.6% with W 1,024 | 12.1 |
| 6 | push | 20/11 ~14:40 | TI `334b244` -> `55d1058` (11 files), BluTrain `e24470c` -> `534538a` (trainer + TI pointer); every file md5-checked on GitHub | |
| 10 (part) | Navin's code vs ours, 350M, Navin's settings | 20/11 15:40, GPU 1 | ours -2.8% with and without the window | 12.1 |
| 7 | packed rows + window, items A + B | 20/11 16:00-16:50 | forward modes 4 / 5, 3 backward copies; SASS of every old kernel unchanged | 11.6 |
| 8 | tests | 20/11 17:08-17:28, GPU 7 | all pass | 11.6 |
| (push) | TI `55d1058` -> `2356565` | 20/11 ~17:45 | 7 / 7 files md5-equal on GitHub | |
| 9 | SFT trainer, checkpoint record, converter, exports, guard | 20/11 19:42-20:01, GPUs 7 + 5 | all pass | 11.6, note below |
| (push) | TI `2356565` -> `dd5b61a`; BluTrain `534538a` -> `651f9c4` | 20/11 ~20:10 | md5-equal on GitHub; `run_sft.sh` mode 755 | |
| 13 | SFT arms on Tülu 3: A, C, D | 20/11 from 20:09, GPU 7 | running (12.2) | 12.2 |
| 11 | docs | 20/11 evening | this section; `all_sft_optimizations.md` 4.17; the SFT package; work log §45.200-§45.201; the BluTrain SFT docs (`sft/README.md` 7, 9, 10; `kernel_diff_sft_pretraining.md`; `sft_kernel_benchmarks.md` 9) | |
| (push) | BluTrain `651f9c4` -> `bfc712b` (docs only: 3 files, +59 lines) | 20/11 ~20:40 | md5-equal on GitHub; TI pointer still `dd5b61a` | |
| 12 | race vs FA3 / cuDNN / FA4 / Flex / FA2 on window rows | next | | 10.5, 11.5 (conventions) |
| 14 | optimize | after 12 | ideas in the step-14 notes of 11.6 | 11.6 |

**Note on step 9's data (added 2026-11-20 evening, to avoid a misreading of 11.6):** the step-9 runs (the trainer vs
HF check on 2,000 val chats, 844 longer than 1,024; the speed 474 / 476 ms; CP = 2) used the `d0_gen_A` data (the 135M
stage-0 general mix, rows of 4,096), not Tülu 3. The Tülu step time comes from the arms (12.2): about 481 ms per step.

### 12.1 Steps 3-6 and the comparison with Navin's code (2026-11-20 morning and afternoon)

**Step 3, the port** (laptop, TI branch `swa_port` on `334b244`; records `profiling_reports/swa_port_2026_11_20/`):
- 7 of Navin's 10 files applied unchanged (the backward dispatcher `GQA_bwd_generic.cu`, the API `AttentionKernels.h`,
  the autograd op `AttentionOps.h/.cpp` and its backward node `AttentionBackward.h/.cpp`, Navin's test
  `test_swa_attention.cu`); the 3 kernel files ported by hand (`swa_port_3_kernel_files.patch`, md5 23deb619): `GQA_fwd_sm90.cu` (MODE_WIN = 2, MODE_CHATS_MX
  moved to 3), `GQA_fwd_generic.cu` (Navin's checks inside our routing), `GQA_bwd_vj2_sm90.cu` (WIN = the 4th template flag,
  together with F32P and HSUM, i.e. with the fp32 partials, idea 4 and ideas 1 / 2);
- `kv_first` / `shared_nkv` / `kv_row_base` sit behind `if constexpr`, so every old kernel is byte-identical machine code
  (cuobjdump on the laptop).

**Step 4, tests** (P035 GPU 1, 13:12-13:21): Navin's `test_swa_attention` 24 / 24; our `test_swa_extra.cu` 12 / 12 in
each of the 3 backward variants (the 350M shape against fp64; every path that cannot honour a window throws; the autograd
node carries the window); the 18/11 push checks (quick 273 / 0 and 280 / 0, the dK KNOWN entry, the idea-4 messages).
Navin's T6 timing (16 rows of 4,096, heads 4/2, our kernels): forward 0.396 ms causal, 0.213 ms W 1,024 (1.85x);
backward 1.187 / 0.721 ms (1.65x). W 512: forward 0.141 ms (2.81x), backward 0.466 ms (2.51x, against 1.168 ms causal
in that run). Navin's own build on 19/11 (Navin's older backward, without ideas 1, 2, 4): forward 0.391 -> 0.214 ms, backward
1.249 -> 0.771 ms.

**Step 5, the pretraining trainer** (`train/production_zero/bluscript_zero_ckpt.cpp`: `BLU_SWA_WINDOW`,
`BLU_SWA_GLOBAL_EVERY`, the per-layer window, the windowed FLOP count; patch `swa_step5_trainer.patch`). Smoke, P035
GPU 1: the production trainer, the 350M kv4 shape (d 1,024, 28 layers, heads 8/4, ffn 2,560, T 4,096, vocab 50,304),
micro-batch 8, 32 micro-batches (1,048,576 tokens) per step, flux shards, 30 steps, medians of steps 11-29:

| Run | Step (ms) | tokens/s | Note |
|---|---|---|---|
| a, no window | 6,748.7 | 155,375 | |
| b, W 1,024 | 6,166.8 | 170,036 | step -8.6%, tokens/s +9.4% (Navin's screen on P015: +8.9%) |
| c, W 4,096 (>= T) | = a | = a | step-0 loss identical to a: a window >= T changes nothing |
| d, W 1,000 | - | - | refused at start-up with the message (not a multiple of 128) |

The first smoke try failed (the config's companion YAML was not copied); fixed and re-run.

**Step 6, the push** (the user, ~14:40): TI `334b244` -> `55d1058` (11 files, 11 / 11 md5-equal on GitHub); BluTrain
`e24470c` -> `534538a` (the trainer, md5 b538437a, + the TI pointer); author Goutham Kumar, no co-author line.

**Navin's code vs ours (cmp20, P035 GPU 1, 15:40; Navin's binary built from Navin's tree, Navin's settings: micro-batch 16, 2.1M
tokens per step, Muon; medians of steps 11-29):**

| | Navin's | Ours | Ours against Navin's |
|---|---|---|---|
| no window | 13,627.2 ms, 153,894 tokens/s, MFU 44.5% | 13,244.3 ms, 158,344 tokens/s, MFU 45.8% | step -2.8% |
| W 1,024 | 12,550.4 ms, 167,098 tokens/s, MFU 43.3% | 12,193.1 ms, 171,995 tokens/s, MFU 44.6% | step -2.8% |
| the window's saving | -7.9% | -7.9% | |

- Not identical in one setting: Navin's runs used `BLU_EMBED_SCALE=1` (x32; step-0 losses 19.7 vs 11.0). The step times
  compare; the losses do not. The user decided not to re-run ("doesn't matter; next time the same settings").
  `cmp20c.sh` (Navin's binary with scale 0) is ready, not run.
- Open: Navin's W 1,024 step here (12,550 ms) vs Navin's own P015 log (12,092 ms); Navin's no-window step matches Navin's log
  (13,627 vs 13,674 ms).
- Machine code (P035, CUDA 13.4, Navin's tree `swa_1119`): Navin's window forward (mode 2) is byte-identical to ours (4,016
  instructions, 0 lines differ), so Navin's SWA is not a naive kernel: it is our October forward plus the FA3-style window.
  Navin's plain / packed forward differs slightly (3,944 vs 3,952 instructions: older base). Navin's backward is the Vj2 of
  2026-11-04 + WIN, without ideas 1, 2, 4, without the fp32-partials switch, and with no packed + window. The nsys
  breakdown's "ours forward +6% with the window" was the capture (clock / data), not the code.

### 12.2 Step 13: the SFT arms on Tülu 3 (running since 2026-11-20 20:09 IST)

**Setup (identical for every arm, `sft/run_sft.sh` on the pushed code; binary `sft_swa` 3af3e48d = BluTrain 651f9c4 +
TI dd5b61a):**
- data `m2_tulu_A`: tulu-3-sft-olmo-2-mixture-0225 minus Aya and the hardcoded set; 744,023 chats in 113,052 packed rows
  of 4,096 tokens (463.1M tokens per epoch, 311.6M trained);
- 1 epoch = 5,652 steps; 20 rows (81,920 tokens, ~132 chats) per step = micro-batch 4 x 5 accumulation steps; LR 1e-4,
  warm-up 170 steps (3%), cosine to 1e-5; Muon (matrices) + AdamW (the rest), all bf16, segmented embedding backward;
- val every 500 steps (and at step 0 and the end) on the val split (2,000 chats, 311 rows); checkpoints every 1,000
  steps and at the end; `EXPORT=1` (final.safetensors, for the HF evaluation);
- the 350M shape: 28 layers, d 1,024, heads 8/4, ffn 2,560, vocab 49,280, RoPE theta 50,000, `EMBED_SCALE=1` (the
  embedding output multiplied by sqrt(1,024) = 32, as the base was trained).

| Arm | Base (converted 2026-11-20) | SFT attention | Start-up check |
|---|---|---|---|
| A | `kv4` (full attention) | full | base window 0 = run window 0: match |
| C | `kv4_swa3` (W 1,024, every 4th layer global) | W 1,024 / every 4 | match |
| D | `kv4_swa3` | full | mismatch, allowed by `BLU_SWA_MISMATCH_OK=1` (warning printed) |

Arm B (full-attention base switched to the window in SFT) was dropped by the user (not our case). One GPU (7), the arms
one after the other: C, then D, then A (about 47 minutes each).

**Results so far (filled in as they come):**

| Arm | val step 0 | val 500 | val 1,000 | final val | step (ms) |
|---|---|---|---|---|---|
| C | 1.668373 | 1.394243 | 1.337176 | (running) | ~481 |
| D | | | | | |
| A | | | | | |

After the arms: val loss by groups (chats of at most 1,024 tokens vs longer; answer positions 0-1k / 1k-2k / 2k-4k),
each model also with the other attention type, and HF evaluations of the exports (through HF transformers: vLLM and
SGLang ignore per-layer windows, 11.5).

**Final results (2026-11-20 22:30 IST; the table above was the state at 20:21, kept as it was):**

Arms C on GPU 7 (20:09-20:58, alone), D on GPU 4 and A on GPU 5 (20:51-21:41, side by side), all 5,652 steps, exports
and the last 2 checkpoints written, 0 error lines. Step times come from different GPUs, so they compare only roughly.

| Arm | Pretrained | SFT | val step 0 | val 500 | val 1,000 | final val (trainer) | step (ms) |
|---|---|---|---|---|---|---|---|
| C | window | window | 1.668373 | 1.394243 | 1.337176 | 1.176781 | ~473-478 |
| D | window | full attention | 1.681077 | 1.395336 | 1.337712 | 1.176567 | ~481 |
| A | full attention | full attention | 1.672250 | 1.395124 | 1.338270 | 1.178506 | ~483 |

**Val loss by groups** (HF transformers on the exports, `hf_val_groups.py`, all 311 val rows = 2,000 chats, 850,765
labeled tokens; "native" = the attention the model was trained with, "other" = the other type; the trainer's val reads
308 of the 311 rows, so "total" is the all-rows number and "trainer rows" the like-for-like one):

| Model | Mode | total | trainer rows | chats <= 1,024 | chats > 1,024 | pos 0-1,023 | pos 1,024-2,047 | pos 2,048+ |
|---|---|---|---|---|---|---|---|---|
| C | native (window) | 1.181186 | - | 1.187510 | 1.174256 | 1.132616 | 1.279082 | 1.803708 |
| C | full attention | 1.193851 | - | 1.187510 | 1.200799 | 1.132600 | 1.302142 | 2.031492 |
| D | native (full) | 1.180943 | 1.176537 | 1.187694 | 1.173547 | 1.132838 | 1.276007 | 1.804080 |
| D | window | 1.182758 | 1.178270 | 1.187694 | 1.177350 | 1.132834 | 1.284937 | 1.817248 |
| A | native (full) | 1.182986 | 1.178484 | 1.191021 | 1.174183 | 1.135404 | 1.277946 | 1.796139 |
| A | window | 1.189449 | 1.184683 | 1.191021 | 1.187727 | 1.135411 | 1.307434 | 1.850631 |
| base kv4_swa3 | native (window) | 1.673091 | - | 1.680500 | 1.664974 | 1.623543 | 1.764602 | 2.337123 |
| base kv4_swa3 | full attention | 1.686149 | - | 1.680500 | 1.692337 | 1.623541 | 1.796111 | 2.544847 |
| base kv4 | native (full) | 1.676887 | 1.672310 | 1.686870 | 1.665948 | 1.628539 | 1.764137 | 2.331914 |
| base kv4 | window | 1.682762 | 1.678098 | 1.686870 | 1.678262 | 1.628502 | 1.789375 | 2.387766 |

Tokens per group: chats <= 1,024: 444,797; > 1,024: 405,968; positions 0-1,023: 694,198; 1,024-2,047: 121,514; 2,048+:
35,053. ("-": computed before the trainer-rows column was added at 21:20.)

- **HF export = trainer, like for like:** D 1.176537 vs 1.176567, A 1.178484 vs 1.178506, base kv4 1.672310 vs A's step
  0 1.672250 (all within 0.00006, as in step 9). The 0.0044 gap of the all-rows "total" is only the 3 val rows the
  trainer's val loader does not use (77 micro-batches x 4 rows = 308 of 311).
- **Chats of at most 1,024 tokens give the same loss with and without the window** (C 1.187510 in both modes), as they
  must.
- **C against D (the decision pair):** total 1.176781 vs 1.176567 (trainer); chats > 1,024: 1.1743 vs 1.1735; positions
  1,024-2,047: 1.2791 vs 1.2760; positions 2,048+: 1.8037 vs 1.8041. So the same within about 0.003 everywhere.
- **Serving C with full attention (what vLLM / SGLang do for our Qwen3 export, 11.5):** total +0.0127, positions
  1,024-2,047 +0.023, positions 2,048+ +0.228 (1.804 -> 2.031). Short chats unchanged. A model fine-tuned with the window
  must be served with it.
- **D with the window** (D was fine-tuned with full attention on the window base): +0.0018 total, +0.013 at positions
  2,048+: small, because its base was pretrained with the window.
- **A** comes from the other base (kv4, full attention): short chats 1.1910 (C / D: 1.1875 / 1.1877), positions 2,048+
  1.7961 (C / D: 1.8037 / 1.8041).

**Benchmarks** (lm-eval, the 10-23 dev suite of the 135M work unchanged: `--apply_chat_template --fewshot_as_multiturn`,
batch 32, HF transformers; GSM8K on its first 400 problems; C_fullattn = C's weights with full attention in every layer):

| Model | IFEval prompt strict | IFEval inst strict | IFEval prompt loose | IFEval inst loose | MMLU | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | GSM8K flex / strict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C | 0.2403 | 0.3741 | 0.2569 | 0.3909 | 0.2507 | 0.4508 | 0.2031 | 0.2965 | 0.6235 | 0.4972 | 0.0175 / 0.0025 |
| C_fullattn | 0.2366 | 0.3741 | 0.2551 | 0.3921 | 0.2505 | 0.4499 | 0.2022 | 0.2960 | 0.6257 | 0.4972 | 0.0200 / 0.0000 |
| D | 0.2255 | 0.3537 | 0.2403 | 0.3813 | 0.2520 | 0.4470 | 0.2014 | 0.2960 | 0.6192 | 0.5012 | 0.0225 / 0.0050 |
| A | 0.2440 | 0.3573 | 0.2643 | 0.3825 | 0.2472 | 0.4474 | 0.2167 | 0.2969 | 0.6208 | 0.5312 | 0.0300 / 0.0100 |

(acc for MMLU / ARC / HellaSwag / PIQA / WinoGrande; the acc_norm values are in the result files.) At 350M these
benchmarks sit near chance (MMLU 0.25) and their prompts are short, so the window changes little in them: C and
C_fullattn differ by at most 0.004. The differences between C and D are within the benchmarks' own standard errors
(from the result files: IFEval prompt-level 0.018, WinoGrande 0.014, ARC-c 0.012). A differs more on two of them:
WinoGrande 0.531 against 0.497 (C) / 0.501 (D), about 2.4 standard errors, and ARC-c 0.217 against 0.203 / 0.201,
about 1.2; A comes from the other pretraining run (kv4), so this compares the two bases as much as the SFT.

**What this says for the C-or-D decision (facts; the decision is the user's):** fine-tuning with the window (C) gives
the same val loss as fine-tuning with full attention on the same base (D), in every group and within about 0.003,
and the same benchmarks within noise. Keeping the window keeps the attention the model was pretrained with and the
smaller KV cache at inference (11.5 / 7.3), as every published SWA model does. The condition is the serving engine: it
must honour the per-layer window (HF transformers does; vLLM / SGLang need the 11.5 patch), because C served with full
attention loses 0.23 of loss past position 2,048.

Incident during the evaluation (21:42, fixed): the first lm-eval runs (swa25_eval.sh) failed when IFEval was scored,
because NLTK refuses its punkt_tab data in the eval venv (copied from P032, owned by another account) and in
/home/operator/nltk_data (writable by other accounts). A private copy (`swa9_1120/nltk_data`, owned by us) with
NLTK_DATA set fixed it; swa25b_eval.sh re-ran the 4 lm-eval jobs (21:49-22:27, all exit 0). The failed logs are kept
in `evals/_failed_nltk/`. Records: `profiling_reports/swa_port_2026_11_20/step9/box_results/eval_1120/`.

**C against A: the window in both stages against full attention in both stages (added 2026-11-21, a user question).**

Which arms ran: C (window in pretraining and in SFT), D (window in pretraining, full attention in SFT) and A (full
attention in both, the setup of every SFT run before 2026-11-20). The plan's fourth arm (section 6), B (full attention
in pretraining, window in SFT), was dropped on 2026-11-20 at 19:13 IST by the user: it asks whether a full-attention
model can be switched to the window in SFT, and our 350M is pretrained with the window. The plan's second-seed repeats
(A2, C2) were not run, so the run-to-run noise of these SFT runs is not measured.
*(2026-11-21 ~11:40: the user asked for B after all, to check the conclusion below along the other path through the
2 x 2; it ran on P015 GPU 3, 12:23-13:52 IST: "Arm B" at the end of this section.)*

A and C differ in both stages (base `kv4` against `kv4_swa3`, and the SFT attention). So C against A compares the whole
window model with the whole full-attention model; the effect of the window in SFT alone is C against D (above). The
first column is the same comparison on the two bases before SFT (numbers from the tables above; minus = the window
model has the lower loss):

| Group | Before SFT: base `kv4_swa3` - base `kv4` | After SFT: C - A |
|---|---|---|
| val loss, trainer (308 rows; before SFT = the arms' step-0 val) | -0.0039 (1.668373 vs 1.672250) | -0.0017 (1.176781 vs 1.178506) |
| HF, all 311 rows | -0.0038 | -0.0018 |
| chats <= 1,024 tokens | -0.0064 | -0.0035 |
| chats > 1,024 tokens | -0.0010 | +0.0001 |
| positions 0-1,023 | -0.0050 | -0.0028 |
| positions 1,024-2,047 | +0.0005 | +0.0011 |
| positions 2,048+ (35,053 tokens, 4% of the labeled tokens) | +0.0052 | +0.0076 |

- C's val loss is 0.0017 lower than A's overall: lower on chats of at most 1,024 tokens and on the first 1,024
  positions, the same on chats longer than 1,024 tokens, and higher far back in long chats (+0.0076 past position
  2,048, where full attention sees more keys).
- This is the pattern the two bases already had before SFT. SFT halved the overall gap (0.0038 -> 0.0018); the gap past
  position 2,048 went from 0.0052 to 0.0076, on 4% of the tokens.
- Benchmarks (the table above): C is higher on IFEval instruction-level (strict 0.374 vs 0.357, loose 0.391 vs 0.383),
  MMLU, ARC-e and PIQA (by 0.003-0.004); A is higher on IFEval prompt-level (strict 0.244 vs 0.240, loose 0.264 vs
  0.257), ARC-c (0.217 vs 0.203), WinoGrande (0.531 vs 0.497) and GSM8K (flexible match: 12 vs 7 of 400 problems);
  HellaSwag is equal within 0.0004. Against one run's standard error (from the result files), only WinoGrande differs
  by more than 2 (0.034 against 0.014: 2.4); the others are within 1.5 (GSM8K flexible 1.5, ARC-c 1.1, MMLU 1.0, the
  rest 0.4 or less; IFEval instruction-level has no standard error in the files). At 350M, WinoGrande sits near its
  chance level (0.5).
- So in quality the window model and the full-attention model are about the same. What the window adds (measured
  2026-11-20): the 350M pretraining step 8.6% faster; the SFT step 0.4% faster (474 vs 476 ms, most Tülu chats are
  short); at inference a KV cache of 44% of full attention's at 4,096 tokens (27% at 32,768), provided the serving
  engine honours the per-layer window (11.5).

**Arm B (2026-11-21): the full-attention base fine-tuned WITH the window; the 2 x 2 is complete.**

Why (the user, 2026-11-21): with A, D and C, the split of the A-vs-C gap was measured along one path only (A -> D -> C:
first the pretraining changes, then the SFT attention). B gives the other path (A -> B -> C) and tests whether the
window in SFT costs the same on both bases.

**Setup.** Identical to A / C / D: the base350 / common settings of `swa24_arms_par.sh`, the same binary `sft_swa`
(md5 3af3e48d), data `m2_tulu_A`, 1 epoch = 5,652 steps, val every 500 steps, checkpoints every 1,000, `EXPORT=1`; base
`kv4` with `SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4 SWA_MISMATCH_OK=1`. Script `swa27_armB.sh` (the same checks and evaluation
as `swa25b_eval.sh`).
- Where: P015 GPU 3 (the user: "GPU 3 in P015 is ours"; P035's 8 GPUs were held by another team's 8-GPU run), 12:23 to
  13:52 IST, of which training 48 min; step ~468 ms (A / C / D on P035: ~473-483 ms).
- The inputs were made on P015 and checked against P035 by md5, because every transfer from the boxes to the laptop had
  dropped to ~0.3 MB/s since ~11:50 (10:56: 14.7 MB/s), so the planned 12 GB copy was not possible:
  - the binary, our built libraries and the BluTrain scripts: copied from P035 (the link recovered at 12:21: 25 s);
    5,560 files equal;
  - the data: the laptop's copy (`~/Downloads/sft135`, where it was prepared on 2026-10-29); all 12 files equal;
  - the base: Navin's original HF export on P015 (`/localstorage/test/BluTrain/hp350m/hf/kv4`, equal to P035's copy in
    all 7 files), converted on P015 with the same `hf_to_blutrain.py` (sha256 62b03e47...) and `--check`: `raw/kv4`
    equal to P035's in all 284 files (meta.json included; embedding check 0 of 50,462,720 differ);
  - prepenv and evalenv: rebuilt with pip at P035's exact versions (34 and 110 packages; torch 2.14.1+cu130,
    transformers 5.21.0, lm_eval 0.4.13); the lm-eval task data downloaded on P015.
- One difference: P015's CUDA 13.0 runtime instead of P035's 13.4 (copying P035's would have taken ~25 min). The GPU code
  is in our libraries (the same files); in this build only `FusedLinearGelu_sm89.o` references cuBLAS (the sm89 GELU
  path: the 350M is SwiGLU on H200, so B never calls it); cuRAND only in `TensorFactory.o` / `RNG.o` (random init / RNG
  state: B loads every weight from the base); every cudart / cuRAND function the binaries import exists in 13.0; the
  same driver (595.91.07) on both boxes.
- Checks passed: the start-up line "sliding window ON ... (7 global, 21 windowed): LLLG x 7" and the mismatch warning
  (base window 0, this run 1024); step-0 val 1.678060 against the kv4 base run with the window in HF (1.678098, from
  2026-11-20): 0.000038 apart; the HF export's native trainer-rows total 1.180420 against the trainer's final val
  1.180413: 0.000007 apart. Records: `profiling_reports/swa_port_2026_11_21/armB/` (scripts, logs, `box_results/`).

**Val loss during training** (the trainer's val, the same 308 rows for every arm):

| Step | A (full / full) | B (full / window) | B - A | C (window / window) | D (window / full) | C - D |
|---|---|---|---|---|---|---|
| 0 | 1.672250 | 1.678060 | +0.0058 | 1.668373 | 1.681077 | -0.0127 |
| 500 | 1.395124 | 1.397860 | +0.0027 | 1.394243 | 1.395336 | -0.0011 |
| 1,000 | 1.338270 | 1.340520 | +0.0023 | 1.337176 | 1.337712 | -0.0005 |
| 1,500 | 1.304564 | 1.307104 | +0.0025 | 1.303532 | 1.303804 | -0.0003 |
| 2,000 | 1.279233 | 1.281483 | +0.0023 | 1.277365 | 1.277615 | -0.0003 |
| 2,500 | 1.257578 | 1.259550 | +0.0020 | 1.255676 | 1.255697 | 0.0000 |
| 3,000 | 1.237261 | 1.239287 | +0.0020 | 1.235225 | 1.235015 | +0.0002 |
| 3,500 | 1.219137 | 1.221239 | +0.0021 | 1.217331 | 1.217229 | +0.0001 |
| 4,000 | 1.204701 | 1.206776 | +0.0021 | 1.202782 | 1.202523 | +0.0003 |
| 4,500 | 1.193088 | 1.195056 | +0.0020 | 1.191410 | 1.191190 | +0.0002 |
| 5,000 | 1.185149 | 1.187097 | +0.0019 | 1.183511 | 1.183230 | +0.0003 |
| 5,500 | 1.179824 | 1.181739 | +0.0019 | 1.178129 | 1.177882 | +0.0002 |
| 5,651 (final) | 1.178506 | 1.180413 | +0.0019 | 1.176781 | 1.176567 | +0.0002 |

**Val loss by group** (HF transformers on the exports, native attention; total = all 311 rows):

| Model | trainer final val | total | chats <= 1,024 | chats > 1,024 | pos 0-1,023 | pos 1,024-2,047 | pos 2,048+ |
|---|---|---|---|---|---|---|---|
| A | 1.178506 | 1.182986 | 1.191021 | 1.174183 | 1.135404 | 1.277946 | 1.796139 |
| B | 1.180413 | 1.185001 | 1.191413 | 1.177976 | 1.135756 | 1.285109 | 1.813233 |
| C | 1.176781 | 1.181186 | 1.187510 | 1.174256 | 1.132616 | 1.279082 | 1.803708 |
| D | 1.176567 | 1.180943 | 1.187694 | 1.173547 | 1.132838 | 1.276007 | 1.804080 |
| B served with full attention | (HF trainer rows 1.179826) | 1.184336 | 1.191413 | 1.176582 | 1.135761 | 1.281470 | 1.809601 |

**The two paths through the 2 x 2** (differences of the rows above; minus = lower loss after the step):

| Step | What changes | trainer final val | total | chats <= 1,024 | chats > 1,024 | pos 0-1,023 | pos 1,024-2,047 | pos 2,048+ |
|---|---|---|---|---|---|---|---|---|
| A -> B | window in SFT, full-attention base | +0.0019 | +0.0020 | +0.0004 | +0.0038 | +0.0004 | +0.0072 | +0.0171 |
| D -> C | window in SFT, window base | +0.0002 | +0.0002 | -0.0002 | +0.0007 | -0.0002 | +0.0031 | -0.0004 |
| A -> D | window in pretraining, full SFT | -0.0019 | -0.0020 | -0.0033 | -0.0006 | -0.0026 | -0.0019 | +0.0079 |
| B -> C | window in pretraining, window SFT | -0.0036 | -0.0038 | -0.0039 | -0.0037 | -0.0031 | -0.0060 | -0.0095 |
| A -> C | both | -0.0017 | -0.0018 | -0.0035 | +0.0001 | -0.0028 | +0.0011 | +0.0076 |
| (A -> B) - (D -> C) | interaction: the window in SFT on the full base minus on the window base | +0.0017 | +0.0018 | +0.0006 | +0.0031 | +0.0006 | +0.0041 | +0.0175 |
| B full attention - B window | B served with its pretraining attention | - | -0.0007 | 0.0000 | -0.0014 | 0.0000 | -0.0036 | -0.0036 |

**Benchmarks** (the same suite and settings as above; B_fullattn = B's weights with full attention in every layer):

| Model | IFEval prompt strict | IFEval inst strict | IFEval prompt loose | IFEval inst loose | MMLU | ARC-e | ARC-c | HellaSwag | PIQA | WinoGrande | GSM8K flex / strict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.2440 | 0.3573 | 0.2643 | 0.3825 | 0.2472 | 0.4474 | 0.2167 | 0.2969 | 0.6208 | 0.5312 | 0.0300 / 0.0100 |
| B | 0.2680 | 0.3657 | 0.2828 | 0.3861 | 0.2467 | 0.4457 | 0.2201 | 0.2970 | 0.6273 | 0.5138 | 0.0275 / 0.0000 |
| C | 0.2403 | 0.3741 | 0.2569 | 0.3909 | 0.2507 | 0.4508 | 0.2031 | 0.2965 | 0.6235 | 0.4972 | 0.0175 / 0.0025 |
| D | 0.2255 | 0.3537 | 0.2403 | 0.3813 | 0.2520 | 0.4470 | 0.2014 | 0.2960 | 0.6192 | 0.5012 | 0.0225 / 0.0050 |
| C_fullattn | 0.2366 | 0.3741 | 0.2551 | 0.3921 | 0.2505 | 0.4499 | 0.2022 | 0.2960 | 0.6257 | 0.4972 | 0.0200 / 0.0000 |
| B_fullattn | 0.2643 | 0.3681 | 0.2810 | 0.3909 | 0.2471 | 0.4440 | 0.2210 | 0.2970 | 0.6284 | 0.5217 | 0.0250 / 0.0025 |

**What B shows:**
- **The window in SFT is free only on a base pretrained with it.** On the window base it changes nothing (D -> C: +0.0002
  at the end, within 0.0003 from step 1,000 on, within 0.003 in every group). On the full-attention base it costs
  +0.0019 (A -> B), and that gap stays at about +0.002 at every val check from step 500 to the end: SFT recovers about
  two thirds of the step-0 cost (+0.0058, the base run with the window) and then no more within the epoch.
- **The cost sits where the window hides keys:** chats of at most 1,024 tokens +0.0004 (the window changes nothing
  there), positions 1,024-2,047 +0.0072, positions 2,048+ +0.0171 (on 4% of the labeled tokens).
- **One epoch of window SFT does not make a full-attention model a window model:** B served with full attention (its
  pretraining attention) is better than with the window it was fine-tuned with (-0.0007 overall, -0.0036 at positions
  1,024+). For comparison, C served with full attention is +0.228 worse past position 2,048, and A served with the
  window +0.054.
- **So the conclusion of the C-against-A note needs a correction:** the A-vs-C gap comes "from the pretraining, not
  the SFT" only along A -> D -> C. Along A -> B -> C the window in SFT costs on the full-attention base, and the window
  in pretraining gains more with window SFT (B -> C -0.0036) than with full SFT (A -> D -0.0019). Interaction: +0.0017
  overall, +0.0175 past position 2,048. The window has to be learned in pretraining; adding it only in SFT costs
  long-chat quality. This is the published practice (2.1: SWA models keep SWA in post-training) and matches SWAA (2.3:
  SWA on full-attention models hurts long context; fine-tuning narrows the gap, here by about two thirds).
- **Benchmarks:** B is within noise of A (the largest differences: IFEval prompt-level strict +0.024, 1.3 standard
  errors; WinoGrande -0.017, 1.2). The two benchmarks where A was ahead of C (the C-against-A note) are also ahead in B:
  ARC-c A 0.217 / B 0.220 against C 0.203 / D 0.201, WinoGrande A 0.531 / B 0.514 against C 0.497 / D 0.501. So those
  gaps come with the base (`kv4` against `kv4_swa3`), not with the SFT attention. B and B_fullattn differ by at most
  0.008 (short prompts, as C and C_fullattn).
- **For our 350M (pretrained with the window) nothing changes:** C and D are the same, so SFT keeps the window and
  serving must honour it.
- Caveat: one run per arm; the run-to-run noise (the plan's seed repeats) is still not measured. The B - A gap has the
  same size at every val check from step 500 on.

*(2026-11-21 ~15:00: the four checkpoints are on Hugging Face, `hf://buckets/gautam1926/checkpoints_sft/checkpoints_sft_swa_experimentations/swa350m_sft_{A_full_full, B_full_window, C_window_window, D_window_full}/`: each `hf/` + `blumodel_step_5652/` + a model card + `MD5SUMS.txt`, checked by public download. The handover for the DPO and RL teams: `work_docs/DPO_RL_handover_350M_SWA_2026_11_21/`.)*

### 12.3 Step 12: the window race against the other libraries (2026-11-20 22:34-22:47 IST)

**Setup.** P035 GPUs 7, 4, 5 (the user's), scheduled after the evaluation (`race/swa26_chain.sh`). The 350M shape: heads
8/4, head size 128, bf16, window 1,024 (W keys, itself included), causal; rows of 4,096 tokens, 8 rows (32,768 tokens)
per call. 8 layouts: 4 sets of 8 real Tulu rows (`m2_tulu_A/train` rows 0-7, 184-191 = the rows with the longest chat,
56,520-56,527, 113,040-113,047) and rows of uniform chats of 512 / 1,024 / 2,048 / 4,096 tokens (uniform_4096 = one
sequence per row, as in pretraining). Each layout on one GPU; two rounds, contenders in order then reversed (ABBA); one
process per run; both sides timed the same way: CUDA events around 20 back-to-back calls, 5 repeats, the median; the
table gives the mean of the two rounds.
- ours: `bin/tva_swa` (the BENCH copy of `test_varlen_attention.cu` with `VARLEN_BENCH_WINDOW` / `VARLEN_BENCH_LOOP`,
  `race/patch_tva_window.py`; the library is the pushed TI dd5b61a): the packed path (chat plan, window per chat); on
  uniform_4096 also the plain-row path of pretraining (`ours_dense_fullrow`).
- FA3 3.0.0 and FA4 (`window_size_left = W - 1`, `window_size_right = 0`); PyTorch 2.14.1 `varlen_attn` with
  `window_size = (W - 1, 0)` (PyTorch sends any window to its FlashAttention-2 kernels, so this is the FA2 entry);
  FlexAttention (torch.compile, Triton) with the block mask "same chat AND q >= kv AND q - kv < W".
- **The gate (before any timing):** every library passed the spike test of the window convention (query j + W - 1 sees
  key j: 0.0294; query j + W does not: 0.0002) and the fp64 reference (O 2.1e-3, dQ 5.1e-3, dK 3.3e-3, dV 2.5e-3; FA2:
  dK 4.4e-3, dV 3.6e-3; limit 2e-2). (A first gate at 22:29 failed fa4 / tvar / flex on a harness bug: the spike test
  read the gradients before the backward ran; fixed and re-run.)
- **cuDNN is not in the race:** its window needs NVIDIA's cudnn-frontend Python package, not installed in the bench
  venv (an install needs the user's OK); PyTorch's cuDNN varlen op has no window parameter.

**Results** (ms per call; ratio = ours / theirs for forward + backward, below 1 = ours faster):

| Layout | ours fwd / bwd (ms) | ours / FA3 | ours / FA4 | ours / FA2 (PyTorch varlen) | ours / FlexAttention |
|---|---|---|---|---|---|
| real rows 0-7 | 0.1432 / 0.4572 | 0.649 | 0.690 | 0.398 | 0.535 |
| real rows 184-191 (longest chat) | 0.1486 / 0.4726 | 0.694 | 0.718 | 0.417 | 0.541 |
| real rows 56,520-56,527 | 0.1256 / 0.4169 | 0.673 | 0.681 | 0.408 | 0.518 |
| real rows 113,040-113,047 | 0.1476 / 0.4737 | 0.725 | 0.725 | 0.422 | 0.530 |
| uniform 512 | 0.0915 / 0.3261 | 0.623 | 0.624 | 0.430 | 0.605 |
| uniform 1,024 | 0.1372 / 0.4496 | 0.736 | 0.733 | 0.434 | 0.601 |
| uniform 2,048 | 0.2023 / 0.6174 | 0.812 | 0.825 | 0.469 | 0.625 |
| uniform 4,096 (pretraining-like) | 0.2268 / 0.6845 | 0.818 | 0.800 | 0.468 | 0.612 |

- **Ours is faster than every library on every layout:** forward + backward 0.62-0.82 of FA3's time, 0.62-0.83 of
  FA4's, 0.40-0.47 of FA2's and 0.52-0.63 of FlexAttention's. Forward alone 0.64-0.88 of FA3 / FA4; backward alone
  0.60-0.81.
- The gain is largest on rows of many chats (real rows, uniform 512), as in the 18/11 race without the window: there the
  fixed costs of the GQA backward (head sums, the dQ round trip) are a large share, and ideas 1, 2 and 4 remove them.
- uniform_4096: ours' packed path vs our pretraining path (plain rows, `ours_dense_fullrow`): forward + backward 1.014,
  so the packed path with one chat per row costs 1.4% more than the plain path; pretraining uses the plain path.
- **Noise:** the median difference between the two rounds is 0.29% (82 pairs). The exception is uniform_4096 on GPU 5
  (5-9% between rounds for every contender, at the end of that GPU's share), but within each round ours / FA3 is the
  same (forward 0.839 in round 1, 0.848 in round 2), which is what the ABBA order is for.
- Records: `profiling_reports/swa_port_2026_11_20/race/box_results/` (race_table.txt, 82 run logs, the gate logs, the
  first try in `aborted_2230/`).

**Loss parity with Navin's code, every setting equal (cmp20d.sh, GPU 4, 22:48-23:11; the step-10 part waived at 15:40):**
Navin's trainer with `BLU_EMBED_SCALE=0` (ours has no embedding scale), compared with our runs of cmp20 (same settings,
GPU 1):

| Run | step-0 loss | step-29 loss | step (ms, medians of steps 11-29) |
|---|---|---|---|
| Navin's, no window | 11.0249 | 9.5164 | 13,529.5 |
| ours, no window | 11.0249 | 9.5101 | 13,244.3 |
| Navin's, window 1,024 | 11.0245 | 9.4942 | 12,416.0 |
| ours, window 1,024 | 11.0243 | 9.4871 | 12,193.1 |

The step-0 losses are equal (no window: identical to 4 decimals; window: 0.0002 apart), and the window lowers the
step-29 loss by the same amount in both (-0.022 / -0.023). The step-29 losses differ by 0.006-0.007 between the two
trainers. The step times here are from different GPUs (Navin's on GPU 4 tonight, ours on GPU 1 at 15:40), so they do
not replace the same-GPU comparison of 12.1 (ours 2.8% faster in both modes).

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
