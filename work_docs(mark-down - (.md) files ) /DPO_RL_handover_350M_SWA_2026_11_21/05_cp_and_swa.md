# Context parallelism and sliding-window attention: how they work, alone and together

Detailed sources in `docs/`: the BluTrain SFT guide sections 5.2 and 5.3, `context_parallelism_a_to_z.md`,
`cp_sft_design.md`, `context_parallelism_sft_experiments.md`, `sliding_window_attention_sft_analysis.md` (sections 7, 9,
11, 12).

---

## 1. Sliding-window attention (SWA)

**The rule.** The same causal grouped-query attention, with one more rule in the windowed layers: a token sees only the
last W tokens, itself included (Hugging Face `sliding_window = W`). Every N-th layer stays global: layer i is global iff
(i + 1) % N == 0. The 350M window model: W = 1,024, N = 4, so 21 windowed and 7 global layers (`LLLG` x 7). No
parameter changes: the window is part of the model like `rope_theta` (a wrong one loads fine and trains wrong).

**In packed SFT rows the window is per chat.** A row holds several chats; each chat attends only to itself (causal),
and in a windowed layer only to the last W tokens of itself. Tokens further back still reach a token through the global
layers. A chat of at most W tokens is not changed by the window at all.

**Window conventions differ between libraries (by one):** ours and HF: W keys including the token itself; FA2 / FA3 /
FA4: `window_size_left = W - 1, window_size_right = 0`; cuDNN: `left_bound = W`; FlexAttention mask:
`q - kv < W`. Our value 0 means "no window" (FA's left 0 means "the token itself only").

**The kernels** (Tensor-Implementations, Hopper / sm_90a):
- API: `gqa_flash_attn_forward_qkv(...)` / `gqa_flash_attn_backward_qkv(...)` in `include/ops/helpers/AttentionKernels.h`
  take a trailing `window` (0 = full attention); W must be a multiple of 128 (each 128-row tile then has exactly one
  edge tile). The autograd op (`AttentionOps.cpp`) and its backward node (`AttentionBackward.cpp`) carry it.
- Forward (`GQA_fwd_sm90.cu`): modes 0 DENSE, 1 CHATS (packed rows), 2 WIN, 3 CHATS_MX (mixed units), 4 CHATS_WIN,
  5 CHATS_MX_WIN; the key loop starts at the window, element masks only on the edge tile and the diagonal tile.
- Backward (`GQA_bwd_vj2_sm90.cu`): the Vj2 kernel with a WIN template copy; the query loop stops at the window; the
  backward ideas 1, 2, 4 work inside the window too.
- Rule "item A": in packed rows the window kernels run only if some chat is longer than W (the plan's longest chat);
  a batch whose chats all fit in W runs the kernels of before, bit for bit (checked).
- Any path that cannot honour a window (non-Hopper generic kernels with packed rows, the older Vj1 / v44 backwards, a
  window that is not a multiple of 128) **throws**: there is no silent full attention.
- Tests: `Tests/kernel_tests/test_swa_attention.cu` (Navin's: fp64 reference, exact key counts), `test_swa_extra.cu`,
  `test_swa_packed.cu` (W = 128 with chats of W-1, W, W+1, 2W, 2W+1 tokens, empty and 1-token chats, several head
  shapes, repeat determinism, a 32k row, autograd, chats <= W equal to no window).

**The trainers** (both `bluscript_sft.cpp` and the pretraining `bluscript_zero_ckpt.cpp`):
- `BLU_SWA_WINDOW=W BLU_SWA_GLOBAL_EVERY=N` (`run_sft.sh SWA_WINDOW=W SWA_GLOBAL_EVERY=N`); `ModelConfig::attn_window(i)`
  gives each layer's window; both attention calls (with and without CP) get it.
- The start-up block prints one line saying exactly which attention runs, e.g. `attention : sliding window ON: the last
  1024 tokens of each chat (< row length 4096), every 4th layer global (7 global, 21 windowed): LLLGLLLGLLLGLLLGLLLGLLLGLLLG`.
- **The window travels with the model and is checked:** every checkpoint folder gets `model_config.json` (shape,
  rope_theta, embed scale, window); `sft/hf_to_blutrain.py` writes the HF config's window into its `meta.json`; at
  start-up the trainer compares the base's record with its own window layer by layer and **stops** on a mismatch;
  `BLU_SWA_MISMATCH_OK=1` allows a deliberate one (printed as `[sft] WARNING (BLU_SWA_MISMATCH_OK=1, on purpose)`).
- The exports carry it: `BLU_EXPORT` metadata (`swa_window`, `swa_global_every`), and `sft/blutrain_to_hf.py` /
  `sft/export_to_hf.py` write `use_sliding_window: true`, `sliding_window`, `layer_types`, `max_window_layers`.
- The MFU line counts the windowed layers' attention as min(t, W) keys per query.

**Speed.** Pretraining the 350M (rows of 4,096, one sequence each): -8.6% step time. SFT on Tülu (most chats short):
474 vs 476 ms per step (-0.4%; the window removes 13.6% of the attention pairs, 1.3% of the step's FLOPs). Kernels:
faster than FA3 / FA4 / FA2 / FlexAttention with the same window on every layout (`07_results.md` section 3).

**Memory at inference** (the reason for the window): a windowed layer keeps only the last W tokens in the KV cache.
For the 350M (2 KiB per token per layer):

| Context | KV cache with the window, as a share of full attention's |
|---|---|
| 4,096 | 44% |
| 32,768 | 27% |
| 131,072 | 26% |

## 2. Context parallelism (CP, Ulysses): one row shared by U GPUs

When one row does not fit one GPU (or to train long rows faster), U GPUs share every row (`BLU_SFT_CP=U`,
`run_sft.sh SFT_CP=U NP=U GPUS=...`). The two all-to-alls are Adhitya's (`dist/Context_Parallelism`, header-only),
used as they are; no kernel changes.

```
the U GPUs of a CP group read the SAME rows; each keeps its slice of every row: tokens [r*T/U, (r+1)*T/U)
per layer:
  RMSNorm, wQ / wK / wV, fused QK-norm + RoPE on the slice (each token keeps its true position inside its chat)
  all-to-all "combine"    [B, H, T/U, hd] -> [B, H/U, T, hd]: every GPU now holds the WHOLE row for H/U heads
  packed attention with the whole row's chat plan (and the window), exactly as on one GPU
  all-to-all "partition"  back to the slice
  wO, SwiGLU MLP on the slice
loss on the slice / the step's trained-token count over all GPUs (each CP group's rows counted once)
backward: each all-to-all's gradient is the inverse all-to-all; ZeRO reduces the gradients as before
```

- **Rules** (the trainer stops otherwise): U divides the GPU count, the query heads, the KV heads and T; not together
  with `BLU_SFT_TRAINED_ROWS`. The 350M (8 query / 4 KV heads): U = 2 or 4. More GPUs per row than KV heads would need
  ring attention or a ring + Ulysses hybrid (not wired; Adhitya has studied the window version of ring attention).
- Rows per step = B x accumulation x (GPUs / U); `ROWS` must be a multiple of B x NP / U.
- Needs Tensor-Implementations `d59740b` or later (before it, k's gradient was dropped under CP).
- **Results** (2026-11-07, 2x H200): CP = 2 trains the same as without CP, within the normal run-to-run variation, at
  4k, 64k and 128k tokens per row; at 64k / 128k it is 1.58x / 1.65x faster than one GPU and needs 56-59% of one GPU's
  memory per GPU; the 1.6B shape at 64k tokens fits only with CP = 2.
- **When the 350M needs CP:** one H200 holds up to 114,688 tokens per row (136.2 GB at peak, 664 ms per step, full
  attention); 122,880 runs out of memory. So CP from about 120,000 tokens per row (`07_results.md` section 5).
- Not tested yet: CP together with data parallelism (more GPUs than U).

## 3. CP and the window together

With Ulysses nothing changes for the window: at the attention call every GPU holds whole rows and whole chats (only
fewer heads), and the window is per chat and per head. Checked with the 350M window model (2026-11-20): CP = 2 against
1 GPU, the same rows in the same order: step-0 gradient norm 1.5179 vs 1.5181, loss at step 29 1.682096 vs 1.682022,
step-0 val loss identical; 313 ms per step on 2 GPUs vs 474 on one. (Ring attention would need window-aware changes;
TransformerEngine turns its cuDNN path off for the window with ring-style CP.)

## 4. Serving and generation with the window (RL rollouts, sampling, evaluation)

- **Hugging Face transformers** honours the per-layer window of the config (5.21.0 used here).
- **vLLM and SGLang do not honour `layer_types` for Qwen3 configs**: they serve every layer with full attention, with
  no error. Measured on C: total val loss +0.013, positions 1,024-2,047 +0.023, positions 2,048+ +0.228
  (1.804 -> 2.031). Short prompts are unchanged, so benchmarks with short prompts barely move (C vs C with full
  attention: within 0.004).
- Before trusting an engine for a window model: compare its logits with transformers on a prompt longer than 1,024
  tokens; patch the engine's Qwen3 attention to take the window per layer if they differ.
- For A and D (full attention) any engine is fine. B is better served with full attention than with its window
  (`07_results.md`), but use C, not B, for a window model.

## 5. What this means for DPO and RL

- **DPO** (BluTrain): start from C's `blumodel_step_5652` with the shape, theta and window flags of
  `00_START_HERE.md`; the reference model is the same checkpoint with the same window. If the DPO trainer packs
  sequences into rows, each sequence (prompt + chosen, prompt + rejected) is one "chat" of the chat plan: its positions
  restart at 0 and the window is counted inside it, exactly as in SFT. A DPO trainer that calls the attention kernels
  directly passes `window = 1024` for layer i unless (i + 1) % 4 == 0, else 0.
- **RL**: generate rollouts with the window (transformers, or a patched engine, section 4); train the policy with the
  window (BluTrain flags, or an HF-based trainer that honours `layer_types`). The model was trained on rows of 4,096
  tokens with RoPE theta 50,000; longer contexts are untested.
- **Memory**: the KV cache saving (44% at 4k) is the reason to keep the window; it also makes long rollouts cheaper.
