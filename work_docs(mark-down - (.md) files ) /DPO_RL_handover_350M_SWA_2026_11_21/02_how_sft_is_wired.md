# How our SFT is wired: data, loader, trainer, kernels, libraries, checkpoints

The full guide, with every detail and its history, is `docs/README_sft_BluTrain.md` (BluTrain's `sft/README.md`); the
"why" of each kernel change is `docs/kernel_diff_sft_pretraining.md`. This page is the map that connects them.
Code: BluTrain `dist_side_gautam_lottn` `651f9c4`, Tensor-Implementations `dd5b61a`, BluBridge-BLAS `cfa0e92`.

---

## 1. The whole path, one picture

```
Hugging Face chat dataset (allenai/tulu-3-sft-olmo-2-mixture-0225 @ d91a0785)
   |  sft/prepare_sft_data.py  (--mix sft/mixes/m2_tulu.json --template tulu-plain-v1 --pack ffd --seq-len 4096)
   |    - chat template tulu-plain-v1, every piece (role marker, message, newline) tokenized on its own
   |    - labels: answer tokens + the end token after each answer; everything else 65535 (ignored)
   |    - each chat labelled on its own BEFORE packing: no label points into the next chat
   |    - chats longer than the row dropped; whole chats packed into rows of 4,096 (First-Fit-Decreasing)
   v
files per split (train/, val/): inputs.bin, labels.bin, valid_counts.bin, seq_lengths.bin, seq_offsets.bin, meta.json
   |  sft/SftDataLoader.h  (one micro-batch = B rows per GPU)
   |    - copies the B rows to the GPU; positions [B, T] restarting at 0 for every chat; the segment list
   |      (row, start, length); the micro-batch's trained-token count
   v
train/production_zero/bluscript_sft.cpp  (BLU_SFT_DATA=<dir> turns SFT on)
   |  per micro-batch: g_packed = { positions, chat plan }  (one plan for every layer, forward AND backward)
   |  per layer (x28):
   |    RMSNorm -> wQ / wK / wV (BluBLAS GEMMs)
   |    -> fused QK-norm + RoPE with the per-chat positions               (FusedRMSRoPE_bf16.cu)
   |    -> [CP: all-to-all combine]                                       (dist/Context_Parallelism, BLU_SFT_CP)
   |    -> attention with the chat plan and this layer's window           (GQA_fwd_sm90.cu / GQA_bwd_vj2_sm90.cu)
   |    -> [CP: all-to-all partition]
   |    -> wO -> RMSNorm -> SwiGLU MLP (BluBLAS GEMMs)
   |  lm_head -> cross-entropy with ignore index 65535, divided by the STEP's global trained-token count
   |  backward with the same plan and positions; backward seed W (the GPU count)
   v
optimizer step: Muon (matrices) + AdamW (embedding, norms), ZeRO-1, fp32 masters -- exactly as in pretraining
   |  every 1,000 steps: DCP checkpoint (metadata.bin + data_rank_0.bin) + model_config.json (shape, window)
   |  at the end (BLU_EXPORT): final.safetensors (fp32, BluTrain names + metadata incl. the window)
   v
sft/blutrain_to_hf.py -> Hugging Face Qwen3ForCausalLM (bf16, window fields, tokenizer, chat template)
```

## 2. The data files (written by `prepare_sft_data.py`, read by `SftDataLoader.h`)

| File | Content |
|---|---|
| `inputs.bin` | uint16 [N, T] row-major: the token ids of each packed row (`inputs = ids[:-1]` of each chat) |
| `labels.bin` | uint16 [N, T]: the next token where it is trained (`labels = ids[1:]` masked), 65535 elsewhere |
| `valid_counts.bin` | uint32 [N]: trained tokens per row (for the loss denominator) |
| `seq_lengths.bin` | int32: the length of every chat, row after row (its presence turns packed mode on) |
| `seq_offsets.bin` | int64 [N + 1]: where each row's chats start in `seq_lengths.bin` |
| `meta.json` | N, T, pad / end / bos ids (0), ignore id (65535), vocab rows (49,280), template, tokenizer path + sha256, the mix (dataset, revision, sources dropped), seed, drops by reason, length histogram, trained tokens, the script's sha256 |

The row length T comes from `meta.json`; the trainer's `BLU_SFT_T` must equal it (`run_sft.sh` sets it).

## 3. The trainer's SFT parts (`bluscript_sft.cpp`)

The SFT trainer is the pretraining trainer (`bluscript_zero_ckpt.cpp`) plus the SFT parts; without `BLU_SFT_DATA` it
trains exactly like pretraining. The SFT parts, by what to search for in the file:

| Search for | What it does |
|---|---|
| `"SFT mode: BLU_SFT_DATA"` | the SFT settings, the train and val loaders, the step count from the epochs |
| `"PackedRows g_packed"`, `packed_rows_of()` | per micro-batch: positions + the chat plan (`gqa_chat_plan_make`) from the loader's segments |
| `g_packed.positions` | passed to the fused QK-norm + RoPE: positions restart at 0 for every chat |
| `g_packed.chats` | passed to `scaled_dot_product_attention`: each chat attends only to itself (null = plain causal) |
| `valid_tokens_ahead`, `sft_denom` | the step's trained-token count over all GPUs, given to the loss with ignore index 65535 |
| `grad_scale` | the backward seed: W in SFT (each micro loss is already divided by the step's global count) |
| `"SFT: the whole val set"` | validation: all val losses summed, divided once by the val set's trained-token count |
| `"initialise the weights from another run"`, `"BLU_INIT_FROM_RAW"` | start weights: DCP checkpoint or converted HF model; weights only, fresh optimizer |
| `"135M base"` | the shape settings (`BLU_D_MODEL`, ..., `BLU_EMBED_SCALE`) for any model shape |
| `swa_window`, `attn_window`, `CausalGQA::window`, `write_model_record`, `read_window_record` | the window per layer, the model record next to every checkpoint, the start-up comparison with the base's record |
| `"struct SftCp"`, `cp_make_group`, `cp_combine` / `cp_partition`, `cp_slice` | context parallelism (skipped entirely at U = 1) |
| `trained_rows`, `gather_rows` | `BLU_SFT_TRAINED_ROWS=1`: lm_head + loss only on labelled tokens |
| `"BLU_EXPORT"` | the final safetensors export |
| `skip_micro_batches` | resume: skips exactly the micro-batches already trained |

The full list with the reasons: `docs/README_sft_BluTrain.md` section 5; the differences to the pretraining trainer:
`03_sft_vs_pretraining_trainer.md`.

## 4. The kernels and how the trainer reaches them

Packed SFT changes exactly three kernels; the window adds one more rule to the attention. Everything else (GEMMs,
norms, SwiGLU, embedding, optimizer kernels) is the pretraining code.

| Op | Trainer call | Library entry (Tensor-Implementations) | Kernel files | What SFT / the window changes |
|---|---|---|---|---|
| QK-norm + RoPE | `fused_rms_rope_bf16(..., g_packed.positions)` | `include/ops/helpers/FusedRMSRoPEKernels.h` | `src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu` (+ the sm90 fwd / bwd kernels) | positions restart at 0 for every chat (pretraining: 0..T-1) |
| attention | `scaled_dot_product_attention(q, k, v, ..., g_packed.chats)` with the layer's window (`CausalGQA::window`) | autograd op `src/autograd/operations/AttentionOps.cpp` -> `gqa_flash_attn_forward_qkv` / `gqa_flash_attn_backward_qkv` (`include/ops/helpers/AttentionKernels.h`, trailing `window`, 0 = full) | `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu` (modes 0-5), `GQA_bwd_vj2_sm90.cu` (Vj2, WIN copy), dispatchers `GQA_fwd_generic.cu` / `GQA_bwd_generic.cu` | each chat attends only to itself (the chat plan: work lists of (chat, tile) units, built once per micro-batch); windowed layers see the last W keys of their own chat |
| loss | sparse cross-entropy with ignore index 65535 and the global denominator | `include/autograd/operations/LossOps.h`, `src/autograd/backward/LossBackward.cpp` | `src/Kernels/cuda/loss/LossKernels.cu` | untrained tokens ignored; one denominator for the whole step (every trained token has the same weight) |
| embedding backward | `BLU_EMB_BWD=segmented` | (BluCCL fused embedding) | `src/Kernels/cuda/embedding/EmbeddingKernels.cu` | not an SFT change: the deterministic fp32 path that pretraining uses, set by `run_sft.sh` |
| GEMMs | every linear layer | BluBridge-BLAS (`libblublas.so`) | `BluBridge-BLAS/` | unchanged |

The attention's chat plan: for each micro-batch the trainer turns the loader's segment list into a plan (which query
tile of which chat and head runs on which compute unit, sorted longest first; the backward's block lists per KV head);
the forward and backward of every layer reuse it. With the window the plan's lists are cut at each chat's window
(forward modes 4 / 5, backward WIN); a batch whose chats all fit in the window runs the window-free kernels. Inside the
Hopper kernels: TMA loads, warp specialization, "mixed units" in the forward (an idle compute group runs another query
head of the same KV head), and the backward ideas 1, 2, 4 (skip the fully masked half of a block's first step; write a
chat's first two dQ tiles directly; add the query heads' dK / dV inside one block). Details:
`docs/README_sft_BluTrain.md` section 6.3, `docs/attention_explanation.md`, `docs/sft_kernel_benchmarks.md`.

Other GPUs (Ada, Ampere): generic kernels with the causal mask and packed rows (no window: a window throws there).

## 5. How the binary is built and linked

```
make -j$(nproc) profiler tensor dist SM_ARCH=90     # libprofiler.so, libtensor.so (+ libblublas.so), libdist.so
bash sft/build_sft_trainer.sh                       # bluscript_sft.cpp -> ./bluscript_sft_bin (mpicxx, production_zero
                                                    # compile line, USE_FUSED_RMS_ROPE=1, -I dist/Context_Parallelism)
```

| Library | From | Holds |
|---|---|---|
| `Tensor-Implementations/lib/libtensor.so` | Tensor-Implementations | tensors, autograd, every kernel above (sm_90a code compiled in) |
| `Tensor-Implementations/lib/libblublas.so` | BluBridge-BLAS | the GEMMs (incl. batched Newton-Schulz for Muon) |
| `dist/lib/libdist.so` | BluTrain `dist/` | communication (NCCL / BluCCL), ZeRO, the CP group |
| `Profiler/lib/libprofiler.so` | BluTrain `Profiler/` | timing / profiling hooks |
| system | CUDA runtime, cuRAND, NCCL, OpenMPI, yaml-cpp | cuBLAS is linked but only the sm89 GELU path uses it (not the 350M on H200) |

At run time the libraries are found through `make -s print_run_ld_path SM_ARCH=90` (`run_sft.sh` puts it first on
`LD_LIBRARY_PATH`). The run copies the binary into `RUN_DIR/bluscript_sft_bin.used` and logs its md5.

## 6. Checkpoints, exports, conversions

| Artifact | Written by | Read by | Notes |
|---|---|---|---|
| `checkpoints/blumodel_step_N/` (`metadata.bin`, `data_rank_0.bin`) | the trainer (DCP), every `BLU_CKPT_FREQ` steps and at the end | `BLU_INIT_FROM` (weights only), `RESUME=1` (everything) | weights + optimizer state; one file per rank |
| `checkpoints/blumodel_step_N/model_config.json` | the trainer (both trainers, since 2026-11-20) | the start-up window check | `blutrain_model_record` v1: shape, rope_theta, eps, embed scale, tying, `swa_window`, `swa_global_every` |
| `final.safetensors` | the trainer with `BLU_EXPORT` (`run_sft.sh EXPORT=1`) | `sft/blutrain_to_hf.py` | fp32, BluTrain names, rank 0; metadata = shape + window + steps |
| HF folder | `sft/blutrain_to_hf.py --export ... --base-hf ... --tokenizer ... --out ...` (or `sft/export_to_hf.py` from a DCP checkpoint) | transformers, lm-eval, RL frameworks | bf16; `config.json` with the window fields; `verify_export.py` / `hf_val_groups.py` prove it equals the trainer |
| raw folder (`meta.json` + tensors) | `sft/hf_to_blutrain.py --hf-dir ... --out ... --check` | `BLU_INIT_FROM_RAW` | the HF model in BluTrain layout; `meta.json` carries the window |
