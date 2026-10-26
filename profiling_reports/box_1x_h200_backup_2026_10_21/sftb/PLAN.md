# SFT kernel benchmarks for Kathirvel -- plan (2026-10-21)

Ask (Kathirvel, 15:10): "need the kernel stats for SFT kernels against the peers". Bar: at least Jenifa's DPO report
(loss vs PyTorch, memory saved, attention vs FA3/FA4/cuDNN, padding vs packing curves + step time), done more completely,
all numbers measured FRESH on one box / one GPU, the final code.

## What "the SFT kernels" are (no new kernels -- three existing ones got SFT features)

| kernel | SFT feature added | our optimisations |
|---|---|---|
| sparse cross-entropy loss (fwd + bwd) | ignore_index masking, global 1/N trained-token denominator | 16-byte loads fwd + bwd, in-place gradient, single allocation |
| GQA causal attention (fwd QPU, bwd Vj2) | packed rows: per-chat plan (design A, cu_seqlens style) | QPU forward, Vj2 backward |
| fused RMSNorm + RoPE on q, k | per-token positions restarting at each chat | -- |

Everything else in the SFT step is the pretraining kernels, unchanged.

## Phase A -- kernels vs rivals, SFT shapes (B 8, T 4096, 18 / 6 heads, hd 128, V 50,304, bf16, real stage-0 labels)

| id | what | ours vs | script | GPU time |
|---|---|---|---|---|
| A1 | attention dense + packed, fwd / bwd / fwd+bwd, ABBA x2 | FA3, FA4, cuDNN | `session_2026_10_19/scripts/scoreboard_free.sh` | ~21 min |
| A2 | attention correctness | fp64 truth, packed == alone bit-exact, 1000-repeat race | full suite: TODAY on the 1x box, 1249 / 1249 (same attention code) -> cite; quick suite on the node | ~5 min |
| A3 | loss F, B, F+B (first / min / median / max trained batches, no-mask, all-masked) + memory | PyTorch fp32 / bf16 eager, torch.compile, Liger CE; lm_head + loss: Liger FLCE, Apple CCE | `sft_loss_vs_rivals_2026_10_17/run_loss.sh` (needs liger + cut-cross-entropy in the venv) | ~10 min |
| A4 | loss old vs new (before / after today's loss work), checksums | ours old | `run_loss_v2.sh` pattern (`bench_sparse_ce_sft_v2.cu`, old / new / old / new) | ~5 min |
| A5 | fused RMSNorm + RoPE fwd / bwd: with positions (packed) vs without (dense) = the feature's overhead | PyTorch eager, torch.compile, Liger (RMSNorm + RoPE) | NEW bench to write (CPU, during A1) | ~5 min |

## Phase B -- padding vs packing at kernel level (same chats, real stage-0 segments)

| id | what | script | GPU time |
|---|---|---|---|
| B1 | attention fwd / bwd: padded rows (1 chat per row) vs packed rows (First-Fit Decreasing), time for the SAME chats | `test_varlen_attention` VARLEN_ONLY=BENCH + `v2_step6_benchmarks_2026_10_12/bench_step6_libs.py` (segments_1012.txt) | ~5 min |
| B2 | loss F+B on padded vs packed rows (same chats; padded = mostly masked rows) | bench with the two label layouts | ~3 min |
| B3 | correctness: packed == padded loss | e2e "test 2" of 10-16 (step-0 val 1.445038 equal) -> cite, or re-run (~3 min, needs the base checkpoint) | ~3 min |

## Phase C -- SFT pipeline (after the kernel window; needs the base checkpoint + data on the GPU box)

Padding vs packing trainer runs on the same chats: step time, useful tokens / s, steps per epoch, peak memory, train / val
loss curves (must overlap, as in Jenifa's chart), and the "SFT really learns" proof (loss falls, val falls). Each run
~20-40 min. Against other frameworks (TRL / HF) end to end: later, by the user's decision.

## Overhead vs our pretraining kernels (inside A / B)
- attention: dense and packed are separate template instances (<0> / <1>), so the dense (pretraining) path carries no
  packed code; the dense numbers in A1 are the pretraining numbers (and today's test B: +18.9% fwd / +40.5% bwd vs BLUCCL).
- RoPE: positions == nullptr launches the original instance; A5 measures with vs without.
- loss: ignore_index -1 = pretraining; A3's no-mask case vs masked cases.

## Order on one GPU (after the break)
T+0   user: pick the box / GPU (8x node p032 has benv FA3 / FA4 / cuDNN, our libs, sft_data, scripts)
T+0   CPU: sync today's master (QPU adopted, loss fwd16 + in-place only) to the node and build (~15-20 min)
T+0   GPU: A1 scoreboard on libs/vj2_QPU (= today's attention code) while the build runs
T+21  GPU: A3 loss vs rivals (new build)      T+31  GPU: A4 loss old vs new
T+36  GPU: B1 + B2 padding vs packing, quick suite     T+46  GPU: A5 RoPE (bench written during A1)
T+55  pull everything, tables     then: Phase C after confirming where the base checkpoint is
