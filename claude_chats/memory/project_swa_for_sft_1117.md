---
name: project-swa-for-sft-1117
description: 2026-11-17 sliding-window attention (SWA) for SFT of the 350M model: where the colleagues' kernel is (local only), formula check vs public sources (correct, HF convention), kernel review (correct for dense rows, no packed rows), what SFT support needs, and the open test (SWA checkpoint with full attention vs SWA)
metadata:
  type: project
---

**Where SWA lives:** only Navin's LOCAL commit b913f32 (2026-11-04, "Sliding-window causal GQA on the Hopper kernels:
v62 MODE_WIN forward, Vj2 WIN backward") in P015 /localstorage/test/BluTrain (branch swa-135m, no upstream; TI base =
a9e37e6 of 10-29, i.e. BEFORE our 11-13 forward MX change and the backward experiments). GitHub code search (org
token): 0 hits for BLU_SWA_WINDOW / swa_window; no swa branch in BluTrain or TI remotes. Diff copied to
profiling_reports/swa_review_2026_11_17/b913f32.diff. Trainer side: BLU_SWA_WINDOW / BLU_SWA_GLOBAL_EVERY (layer i
global iff (i+1) % every == 0), W multiple of 128, Hopper dense rows only; packed rows -> static_assert / exit.
**Formula verdict (fetched sources 2026-11-17):** query i sees keys i-W < j <= i (W keys incl. itself) = HF
masking_utils `kv_idx > q_idx - sliding_window`, HF flash path window_size=(W-1, W-1) with causal, FlashAttention
"[i - left, i + right] inclusive", Ram's torch script `q - kv < W`. PyTorch attention-gym example uses `q - kv <= W`
(W+1 keys): a different convention, off by one if copied.
**Kernel review (reading only, not run by me):** fwd pair p reads key tiles max(0, p - W/128) .. p, window edge masked
only in that first tile (middle tiles fully inside for all 128 rows), FA3 Check_inf guard for rows fully masked in it;
bwd key block sees W/64 + 2 query steps, edge steps masked key <= q - W. Correct for dense rows. Their test
Tests/kernel_tests/test_swa_attention.cu (fp64 ref + exact key counts) not yet run by me.
**Why it matters for SFT:** SWA has no parameters (weights load into full attention), but outputs differ for every chat
longer than W = 1,024: our data m1_smol 42.8% of chats / 66.2% of tokens, m2_tulu 17.8% / 44.8%, d8_gen_all 27.9% /
53.2% are in chats > 1,024 tokens. OPEN TEST (needs the SWA-pretrained 350M checkpoint path + 1 GPU): val loss of that
checkpoint on our SFT val data with SWA vs with full attention; decides whether SFT needs SWA. Inference must use the
same attention as SFT. Support = merge b913f32 onto our branch (conflicts in GQA_fwd_sm90.cu / GQA_bwd_vj2_sm90.cu),
add packed rows (window in chat-relative positions in the MX work list and the Vj2 block list), tests, races vs FA3
varlen window_size / FA4 / FlexAttention. Ulysses CP needs no change for SWA (each GPU holds the full row for its heads).
**Published practice (fetched 2026-11-17, quotes in scratchpad swa_refs/):** models pretrained with SWA keep the same
attention in post-training and inference: Mistral 7B (all layers SWA W=4096; "We also provide a model fine-tuned to
follow instructions, Mistral 7B -- Instruct"; HF Mistral-7B-Instruct-v0.1 config sliding_window 4096; "SWA exploits the
stacked layers ... access tokens ... up to W x k", ~131K span); Gemma 2 (local 4096 / global, every other layer; IT
models); Gemma 3 (5 local : 1 global, local span 1024, "only the global layers attend to long context"; IT models);
gpt-oss (alternating banded 128 / dense; HF gpt-oss-20b layer_types sliding/full, sliding_window 128). The user
(17/11) objected that SWA cannot work in SFT (a token past 1,024 in a chat could not see its chat) and asked how the
1,024 tokens are "selected": answered = positional (the previous W positions of the same chat, no score-based
selection), the global layers (7 of 28: indices 3, 7, ..., 27) see the whole chat, and stacked layers pass it on.
Explain this mechanism FIRST when SWA comes up again.
**Deep analysis written 2026-11-17:** `work_docs(...)/post_Training_research/sliding_window_attention_sft_analysis.md`
(published evidence, internal numbers, gaps, 2x2 ablation plan, order). Internal results (P015 /localstorage/test/
BluTrain): 135M swa_ablation SWA -0.0025 val vs global (5B tok, paired), 350M hp350m swa3 -0.0011 (noise 0.0002),
kv4_swa3 -0.0091 = the CHOSEN 350M config (so the 350M IS pretrained with SWA); +12% tok/s. Checkpoints for the
ablation exist: hp350m/runs/{base, swa3, swa5, kv4, kv4_swa3, ...}/final.safetensors + hp350m/hf/<tag>/. Ablation arms:
A full->full, B full->SWA, C SWA->SWA, D SWA->full (+ seed pairs). Order proposed: functional packed-row SWA first (B, C
need it) -> ablations -> optimize only if SWA-SFT kept -> DPO. Waiting for the user's OK (explain-first rule).
**17/11 evening, user decisions:**
- order agreed: support first (correct, not optimized), then the ablations, then optimize only if SWA stays in SFT;
- 350M ONLY, no 135M runs;
- DPO is NOT ours: the DPO team passes the window per layer to the same attention call (trailing `window`, 0 = full).
Step plan 0-6 is in the doc, section 6.1, marked PROPOSED:
- pair kv4 / kv4_swa3, rows of 4,096 (no CP), m1_smol, 1 epoch, LR 1e-4;
- arms A and D run now, with no new code; B and C after the packed-SWA support.
Proposed P015 with 2 GPUs (7, 6). Waiting for the OK and the GPUs.
The user thought every length limit was removed. Told them the cap is still in prepare_sft_data.py: :697-698 (T % 256,
T <= 131072) and the default 4096 at :668. Trainer and loader have no cap (T comes from meta.json). Removing the cap is
the future task [[project-future-seq-len-no-default-no-cap]]. The 350M ~121.6k is a GPU-memory limit, not code.
**17/11 ~15:40, deep research for the user (doc section 7):**
- **Per-chat window = standard.** The sources:
  - HF masking_utils: packed_sequence_compatibility_function = AND(same sequence, the window on positions remapped
    inside each sequence);
  - HF flash path: window_size (W-1, W-1) into flash_varlen_fn with cu_seqlens;
  - FlexAttention blog: generate_doc_mask_mod over and_masks(causal, sliding_window), "SWA + Document Masking +
    Causal + CP";
  - TE: SWA + thd;
  - TRL packing / padding-free = the model's HF attention, so Gemma / Mistral SFT gets per-chat windows by default;
  - instruct configs keep the window (Gemma 3 4B IT sliding_window 1024, pattern 6; Mistral-Instruct 4096; gpt-oss
    128);
  - Gemma 3's reason is KV-cache memory (global only = 60% overhead vs < 15%); vLLM has a hybrid manager.
- **No one published an SFT quality comparison.**
- **What SWA saves in OUR SFT** (scratchpad swa_cost.py, rows of 4,096, 350M = 21 windowed + 7 global):
  - m1: 42.8% of chats > 1024 but only 12.9% of tokens past chat position 1,024; pairs saved 3.6%; FLOPs saved
    0.33%;
  - m2: 15.4% of tokens past 1,024; 13.6% pairs; 1.31% FLOPs;
  - one long chat: 8k 42.6%, 32k 29.6%, 128k 26.2% of the full attention.
  - So it is NOT a speed reason at 4k: the reason is consistency (pretraining -> SFT -> serving).
- User 17/11: remove the --seq-len default and the 131,072 cap (lift every limit + real prep test past 131k), after
  the bwd work and the SWA support.
- The user confused row length 4,096 with the window. Told them: SWA == full only if W >= T.
**17/11 ~16:00, user question: "at 64k does the local window change?" (doc section 8).**
Answer: NO. Every SWA model keeps W and extends only the global layers:
- Gemma 3: 32k -> 128k at the end of pretraining; global RoPE 10k -> 1M + linear x8; local stays 10k / 1024;
- gpt-oss: YaRN x32 on the dense layers, from 4096; window 128 stays;
- Command A: W 4096 at 256k, global layers NoPE;
- Llama 4: chunk 8192 with RoPE, 12 of 48 layers NoPE global, 10M;
- Mistral v0.2 is the exception (dropped the window at 32k, no official reason);
- SWAN-GPT research: fixed W 512 + NoPE global.
Code gap for a long-context 350M: bluscript_sft.cpp:954-955 builds ONE RoPE table (one BLU_ROPE_THETA, no
scaling), so it needs a second table for the global layers. Kernels are unchanged.
**17/11 ~16:10, kernel libraries (doc section 9).** All of them do varlen + window the same way:
- FA2: n_block_min = max(0, (m_block*kBlockM + seqlen_k - seqlen_q - wl) / kBlockN); bwd m_block_max from wl;
  apply_mask_local only at the edges;
- FA3 block.h and FA4 block_info.py: get_n_block_min_max / get_m_block_min_max with window_size_left, per-sequence
  SeqlenInfo;
- cuDNN: Ragged + "Sliding window" (diagonal_band_left_bound);
- Flex: BlockMask lists + full blocks skip the mask_mod;
- xformers: BlockDiagonalCausalLocalAttentionMask;
- FlashInfer: window_left (forward only).
This is our planned packed design: the forward key loop starts at max(chat start, q - W), the backward query loop
stops at min(chat end, k + W), and masks are applied only on the edge and diagonal tiles.
**17/11 ~16:25, YaRN / Qwen (doc 8.1).**
- The user's meeting: Adhitya pretrains at 32k, and inference uses YaRN to reach 131k. That is CORRECT (the user
  asked me "don't make it negative"). Qwen3 does exactly this:
  - the 32,768 stage + RoPE base 1e6 (ABF);
  - YaRN factor 4 (+ DCA) at inference = 131,072;
  - cards: 4B / 8B "32,768 natively and 131,072 tokens with YaRN"; 0.6B lists 32,768;
  - static YaRN can hurt short texts, so enable it only for long inputs.
- With SWA, YaRN goes on the global layers only.
**17/11 17:00-17:30, SWA prep during the user's break (CPU only):**
- **P015 hp350m pair verified (read-only):**
  - runs/{kv4, kv4_swa3}/env.txt are identical except BLU_SWA_WINDOW=0 vs 1024 + BLU_SWA_GLOBAL_EVERY=4;
  - 2,384 steps, ~2.1M tokens per step = 5.0B tokens; Muon; BLU_ALL_BF16=1;
  - final.safetensors 1.44 GB (fp32); hf/<tag>/ = Qwen3ForCausalLM: layer_types sliding/full (full at 3, 7, ..., 27),
    sliding_window 1024, max_position 4096, tie false (export), blu_meta.json;
  - RoPE THETA 50,000 (NOT 10,000: the 10k was from Ram's torch script; doc corrected 17:10);
  - qk_eps 1e-6, rms 1e-5, embed_scale 1, weight_tying 1;
  - tokenizer md5 c56e5ce7 == our sft135 base_hf: our SFT data is usable as-is.
- **Converter:** profiling_reports/swa_sft_2026_11_17/hf_to_blutrain_350m.py (copy: F32, blu_meta.json, bf16(sqrt d)
  scale, --window / --global-every windowed numpy reference). NOT run: needs numpy + tokenizers, i.e. a venv in
  ~/gautam_k on P015 (ASK the user first).
- **Kernel plan from Navin's b913f32:**
  - fwd MODE_WIN: kv_first = max(0, pair - win_tiles), prologue edge mask, Check_inf;
  - bwd WIN: nIter = min(nIter, win/64 + 2), edge steps take the mask copy, masked |= key <= qry - win.
  - Both are per-chat ready because CHATS indices are segment-relative. So packed = combine the flags (fwd mode
    CHATS + WIN, incl. MX units), drop the bwd static_assert / launcher refusal, give the plan builder windowed
    work for LPT, add the API window per layer, the trainer BLU_SWA_*, and HF export layer_types.
**RoPE base facts (17/11 18:25):**
- the hp350m screen = binary swa_ablation/bin/bluscript_135m_swa, default theta 50,000;
- Ram's torch_compile_fa4_350M.py and Navin's bluscript_zero_ckpt_bluccl_multi_135m_eqab_fusedemb.cpp default to
  10,000;
- our bluscript_sft.cpp defaults to 10,000, so SFT of hp350m needs ROPE_THETA=50000 (the gate catches a mismatch);
- speed comparisons are unaffected;
- the user was told to ask the colleagues which base the production 350M uses.
**17/11 ~20:45, user: "we will start tomorrow that SWA work cleanly".** Nothing of SWA has run on a GPU yet.
TOMORROW (18/11) order:
1. Ask the OK for a small venv (numpy, tokenizers) in our folder on the box.
2. Run hf_to_blutrain_350m.py on hp350m/hf/{kv4, kv4_swa3} + the numpy gate (ROPE_THETA=50000!).
3. Explain the packed SWA kernel changes first, then code:
   - fwd CHATS + WIN incl. MX;
   - bwd CHATS + WIN, plan work under the window;
   - API window per layer, trainer BLU_SWA_*, HF export layer_types.
4. Tests, then arms A / D (no new kernel), B / C after.
P015 comes back from the data team at 10:00.
