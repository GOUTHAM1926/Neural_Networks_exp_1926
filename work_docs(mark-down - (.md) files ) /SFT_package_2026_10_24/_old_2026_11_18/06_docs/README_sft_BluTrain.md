# SFT in BluTrain — what is here, how it works, why

This folder and `train/production_zero/bluscript_sft.cpp` hold the SFT (supervised fine-tuning) pipeline:
data preparation, the data loader, the trainer, the scripts to build and run it, and the export to Hugging Face
format. The SFT kernel changes (loss, RoPE, attention) are in Tensor-Implementations. This file is the map: what
each part does, where the change is, why it was made, and how it was tested.

All of this was built and tested from 2026-09-18 to 2026-10-24, and brought onto `dist_side_gautam_lottn` on
2026-10-27. The trainer was then compile-checked against this branch. The full work log is Gautam's SFT work
documentation (sections are cited as "work log §N").

Added on this branch since then, each in its section: the loss-sum fix (2026-10-30, section 6.1); long context
16k-128k and drop-only long chats (2026-10-31, sections 6.4 and 3.2); the trained-rows lm_head switch, off by default
(2026-11-03, sections 5 and 9); the BluBLAS GEMM fixes (2026-11-03 to 11-05, section 7); the optimizer and CPU-wait
speed-ups both trainers got (2026-11-03 to 11-05, section 5.1); the `autograd::contiguous` fix (2026-11-07, section 6.5);
context parallelism, one row shared by several GPUs (2026-11-06/07, section 5.2); the attention forward's mixed units
(2026-11-13, section 6.3); the attention backward ideas 1, 2 and 4 (2026-11-18, section 6.3).
This file was last brought up to date on 2026-11-19 (before that: 2026-11-11).

---

## 0. What is where

| Path | What |
|---|---|
| `sft/prepare_sft_data.py` | turns a Hugging Face chat dataset into SFT files (template, labels, packing) |
| `sft/SftDataLoader.h` | reads those files during training (masked labels, per-chat positions, trained-token counts; with the trained-rows switch also the trained tokens of each micro-batch, 2026-11-03) |
| `sft/test_sft_dataloader.cpp` | checks the loader against the files (build line in its header); the end id and the vocabulary rows come from the data's `meta.json` (2026-10-31), so it runs on GPT-2 and 135M data |
| `sft/ref_val_loss.py` | the end-to-end gate: a numpy reference of the step-0 val loss, every chat alone; `--make-subset` writes the rows as a small train/val folder for the trainer |
| `sft/hf_to_blutrain.py`, `sft/blutrain_to_hf.py` | the 135M base: Hugging Face weights into the trainer's raw format (`INIT_FROM_RAW`), and a trained export back to a Hugging Face model (for the benchmarks) |
| `sft/mixes/*.json` | the data mixes for `prepare_sft_data.py --mix` (m1-m3: the 10-28 135M runs) |
| `train/production_zero/bluscript_sft.cpp` | the SFT trainer = this directory's pretraining trainer + the SFT parts (`BLU_SFT_DATA`) |
| `dist/Context_Parallelism` (submodule, Adhitya's) | the SFT trainer's context parallelism uses its Ulysses all-to-alls as they are (header-only; `Makefile` adds `-I dist/Context_Parallelism`; section 5.2) |
| `sft/build_sft_trainer.sh` | builds `./bluscript_sft_bin` with the production_zero compile line |
| `sft/run_sft.sh` | launches a run with the settings of the 2026-10-22 Tülu 3 run; its header lists every knob (the 135M shape, `SCHEDULE`, `SFT_TRAINED_ROWS`, `INIT_FROM_RAW`, `EXPORT`, ...); the row length T comes from the data's `meta.json` |
| `sft/export_to_hf.py`, `sft/verify_export.py` | export a checkpoint to Hugging Face format (`Qwen3ForCausalLM`), and prove the export gives the trainer's val loss |
| `sft/README_SFT_checkpoint_for_DPO.md` | the hand-over note for the DPO team: the Tülu 3 SFT checkpoint, settings, data, results |
| `sft/docs/` | the design documents (kernel differences, packing, loss averaging, how SFT learns, the benchmarks vs other libraries, the optimizer choice: Muon vs AdamW) |

Tensor-Implementations files of the SFT work (all on this branch):

| Part | Files |
|---|---|
| loss: ignore index, global denominator, fp32 loss scalar, gradient over the logits, 16-byte loads | `src/Kernels/cuda/loss/LossKernels.cu`, `include/ops/helpers/LossKernels.h`, `src/autograd/backward/LossBackward.cpp`, `include/autograd/operations/LossOps.h` (on the branch since 2026-09-25 / 10-21) |
| loss: the sum counts every row (it stopped at 65,536 rows; section 6.1) | `src/Kernels/cuda/loss/LossKernels.cu` `sum_reduction_kernel`; test `Tests/kernel_tests/test_sparse_ce_ignore_index.cu` group H (2026-10-30) |
| attention: packed rows (chat plan), QPU forward, Vj2 backward; mixed units in the forward (2026-11-13); backward ideas 1, 2, 4 (2026-11-18) | `include/ops/helpers/AttentionKernels.h`, `src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu`, `GQA_bwd_vj2_sm90.cu`, `GQA_bwd_vj1d_sm90.cu`, `GQA_bwd_vj1_sm90.cu`, `GQA_fwd_generic.cu`, `GQA_bwd_generic.cu`, `src/autograd/operations/AttentionOps.cpp`, `src/autograd/backward/AttentionBackward.cpp` and their headers (Hopper parts on the branch since 2026-10-21) |
| attention for every other GPU (Ada, Ampere): causal mask + packed rows in the generic kernels | `GQA_fwd_generic.cu`, `GQA_bwd_generic.cu`, `AttentionKernels.h`, `AttentionOps.cpp` (2026-10-24, added 2026-10-27) |
| RoPE: per-chat positions in every QK-norm + RoPE kernel | `src/Kernels/cuda/norm/arch/FusedRMSRoPE_{fwd,bwd}_sm90{,_kernel}.cu`, `src/UnaryOps/cpu/Normalizations.cpp`, `include/ops/UnaryOps/Normalizations.h`, `include/ops/helpers/FusedRMSRoPEKernels.h`, `src/autograd/operations/NormalizationOps.cpp`, `src/autograd/backward/NormalizationBackward.cpp` and their headers (added 2026-10-27) |
| RoPE: the bf16 kernel the trainers use, with positions and the 2-4x faster backward | `src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu` (added 2026-10-27; its header summarises the change, the speed and the bit-identity proof) |
| tests | `Tests/kernel_tests/test_varlen_attention.cu`, `test_rope_positions.cu`, `sanitize_vj2.cu` |
| long context (16k-128k): RoPE tables, loss fallback backward, `contiguous()` past 2^32 elements, embedding and sm120 SwiGLU grid limits; fast math only for the loss file; optional fp32 dK/dV partials (off) | `src/Kernels/cuda/attention/RoPEKernels.cu`, `arch/RoPE_sm90.cu`, `src/Kernels/cuda/loss/LossKernels.cu`, `src/Views/ContiguousKernel.cu`, `src/Views/arch/ContiguousKernel_sm90.cu`, `src/Kernels/cuda/embedding/EmbeddingKernels.cu`, `src/Kernels/cuda/activations/arch/SwiGLUKernels_sm120.cu`, `GQA_bwd_vj2_sm90.cu`, `Makefile`, test G14 in `test_varlen_attention.cu` (2026-10-31; section 6.4) |
| lm_head + loss on the trained tokens only (`BLU_SFT_TRAINED_ROWS=1`, off by default): row gather / scatter with autograd | `include/ops/helpers/RowGatherKernels.h`, `src/Kernels/cuda/rows/RowGatherKernels.cu`, `include/autograd/operations/RowGatherOps.h`, `src/autograd/operations/RowGatherOps.cpp`, `include/autograd/backward/RowGatherBackward.h` (d18f820, 2026-11-03; only the SFT trainer calls them, and only with the switch on) |
| `autograd::contiguous`: an input that is already contiguous is returned as is (before, the 2nd or later output of an op with several outputs lost its gradient through it) | `src/autograd/operations/ReshapeOps.cpp`; test `Tests/kernel_tests/test_contig_alias.cu` (d59740b, 2026-11-07; section 6.5) |
| optimizer kernels both trainers use: Muon Newton-Schulz guard on the GPU, batched Newton-Schulz, batched Muon prep, multi-tensor copy / zero / Adam / scale, with the bugs found in them on the way | d98d6ee, dbb4e5f, 963df8b, 05e1d6c, e2af706, 66a32a7 (2026-11-04/05; section 5.1) |
| BluBridge-BLAS (the GEMM library, a submodule of Tensor-Implementations): tile-order tables without CPU waits, deterministic BN=64 split-K, sm_80/86/89 last-group fix (297f372); addmm bias (0b552f0); 7 more groups of GEMM bugs (a6d4292); merged with Aparna's symmetric-output mode (cfa0e92, 2026-11-11: off by default and the SFT trainer never turns it on -- every existing kernel byte-identical, the GEMMs of the 135M SFT step bit-identical, same loss and step time) | `BluBridge-BLAS/` (2026-11-03 to 11-11; details in its commits; section 7) |

---

## 1. SFT in simple words

**Pretraining** teaches the model to continue any text. The data is one long stream of text cut into rows of
T = 4,096 tokens, and every token is a target.

**SFT** teaches the model to answer like an assistant. The data is chats: a user message, then an assistant answer
(sometimes several turns). Four things differ from pretraining:

1. **Only the answer is graded.** Every token that is not part of an answer gets the label 65535: no loss and no
   gradient for that token.
2. **The loss is one average over the whole step.** The losses of all graded tokens in the step (all micro-batches,
   all GPUs) are added and divided once by the number of graded tokens in that whole step. Every graded token then
   has the same weight, whether its chat was long or short.
3. **Several chats share one row (packing), and they must not see each other.** One chat per row plus padding wastes
   most of the GPU's work, so several whole chats go into one row. Attention is told where each chat starts and ends,
   and each chat attends only to its own earlier tokens.
4. **Each chat's positions start at 0.** RoPE gets a per-token position list that restarts at 0 for every chat, so a
   chat inside a row is rotated exactly as if it were alone.

One packed row, T = 4,096:

```
row tokens:  [ chat 1: 1,000 tokens ][ chat 2: 2,500 tokens ][ pad tail: 596 ]
positions:     0 1 2 ... 999           0 1 2 ... 2499          0 1 ... 595
attention:   chat 1 sees only chat 1;  chat 2 sees only chat 2;  pad tail sees only itself
labels:      65535 on prompts and pad; the real next token on answers (and on the end token after each answer)
```

**Only three kernels change for SFT: RoPE, attention and the loss.** Every other kernel (matmuls, RMSNorm, SwiGLU,
embedding, optimizer) runs the pretraining code. Proof, and the comparison with 10 other libraries:
`sft/docs/kernel_diff_sft_pretraining.md`. (With the experimental trained-rows switch on, two more run: a row gather
before the lm_head and its scatter in the backward; section 5. It is off by default.)

---

## 2. The journey of one chat

```
Hugging Face dataset (e.g. allenai/tulu-3-sft-olmo-2-mixture-0225)
   |  sft/prepare_sft_data.py
   |    - chat template "tulu-plain-v1", every piece tokenized on its own
   |    - labels: answer tokens + the end token after each answer; everything else 65535
   |    - whole chats packed into rows with First-Fit-Decreasing (--pack ffd)
   v
files: inputs.bin, labels.bin, valid_counts.bin, seq_lengths.bin, seq_offsets.bin, meta.json
   |  sft/SftDataLoader.h   (one micro-batch = B rows)
   |    - copies the B rows to the GPU (inputs and labels)
   |    - builds positions [B, T] (restart at 0 per chat) and the segment list (row, start, length)
   v
train/production_zero/bluscript_sft.cpp   (BLU_SFT_DATA=<dir> turns SFT mode on)
   |    - g_packed = { positions, chat plan }, built once per micro-batch
   |    - per layer: RMSNorm -> wQ/wK/wV -> fused QK-norm + RoPE (with positions)
   |                 -> attention (with the chat plan) -> wO -> SwiGLU MLP
   |    - lm_head -> loss (ignore 65535, divide by the step's global graded-token count)
   |    - backward: the same plan and positions are used again
   v
optimizer step (Muon / AdamW, ZeRO-1), exactly as in pretraining
```

---

## 3. Preparing the data (`sft/prepare_sft_data.py`)

1. Reads a chat dataset (pinned revision; arguments at lines 207-225).
2. Writes each chat with the template **"tulu-plain-v1"** (lines 15-22):
   `<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...`
   The role markers are plain text; after every answer comes the real end token `<|endoftext|>` (id 50256).
   It is Tülu 3's template minus one newline between turns (decision 2026-10-15, work log §45.1).
3. Tokenizes each piece separately (`tokenize_conversation`, line 47), so a marker never merges with its neighbour.
4. Marks what is trained (lines 62-77): answer tokens and the end token after each answer; every assistant turn,
   not only the last one.
5. Shifts by one (`build_row`, line 79): `inputs = ids[:-1]`, `labels = ids[1:]`; untrained positions get 65535
   (`IGNORE_ID`); padding uses 50256 (`PAD_ID`).
6. Packs (`--pack ffd`): each chat is labelled on its own first, so no target crosses into the next chat; then whole
   chats go into rows with **First-Fit-Decreasing** (`ffd_pack`); the rows are shuffled.
   Chats longer than T are dropped (section 3.2).
7. Checks everything (lines 309-326) and writes the files (formats at lines 3-14):

| File | Type | Meaning |
|---|---|---|
| `inputs.bin` | uint16 [N, T] | token ids the model reads, padded with 50256 |
| `labels.bin` | uint16 [N, T] | the target at each position; 65535 = not trained |
| `valid_counts.bin` | uint32 [N] | trained labels per row (for the global denominator) |
| `seq_lengths.bin` | int32 [all chats] | packed only: the chat lengths of every row, row after row |
| `seq_offsets.bin` | int64 [N + 1] | packed only: row r's chats are `seq_lengths[seq_offsets[r] : seq_offsets[r+1]]`; the rest of the row is its pad tail |
| `meta.json` | text | settings, dataset revision, tool versions, counts, why rows were dropped |

**First-Fit vs Best-Fit Decreasing:** both sort the chats longest first; First-Fit takes the first row with room,
Best-Fit the tightest. On our data both give the same number of rows (5,000 chats -> 1,517 rows each; the minimum
possible is 1,511), the same proven worst case, and First-Fit is 1.3-1.9x faster to compute (`sft/docs/sft_packing_algorithms_bfd_ffd.md`).
**Packing vs padding:** the same 5,000 chats need 1,517 packed rows instead of 5,000 padded ones (3.3x fewer); the full
SFT run reached the same val loss drop (0.1625 vs 0.1629) in 5.1 min instead of 17.0 (`sft/docs/sft_kernel_benchmarks.md`).
From 2026-10-22 SFT uses packed rows only.

```
python3 sft/prepare_sft_data.py --self-test
# the Tülu 3 data of the 2026-10-22 run (seed 1234; val = the first 2,000 kept chats, train = all the others):
python3 sft/prepare_sft_data.py --out <dir>/val   --dataset allenai/tulu-3-sft-olmo-2-mixture-0225 \
    --revision d91a0785ade02942520280fb484866fce41e448f --split train --drop-source ai2-adapt-dev/tulu_v3.9_aya_100k \
    --seq-len 4096 --pack ffd --num 2000
python3 sft/prepare_sft_data.py --out <dir>/train ...same options... --num 0 --skip-kept 2000
```

### 3.1 The 135M base: tokenizer and templates (2026-10-27/28)

- `--tokenizer <tokenizer.json>`: the team's BPE (49,152 ids; the model has 49,280 rows). Its special ids: 0
  `<|endoftext|>`, 1 `<|im_start|>`, 2 `<|im_end|>`. Every chat starts with id 0 (every pretraining document does), and
  padding is id 0. Without `--tokenizer` the script uses GPT-2 (end token and padding 50256).
- `--template`: `tulu-plain-v1` (role markers as plain text, the end token after every answer), `tulu-special-v1` (the
  markers as single new tokens; needs the extended tokenizer from `sft/hf_to_blutrain.py --new-tokens`) or `chatml-v1`
  (`<|im_start|>role\n...<|im_end|>`).
- `meta.json` records `end_id`; the loader reads it (GPT-2 data without it: 50256).
- `--mix <file.json>` (`sft/mixes/`): several datasets in one packed set; each part's first `val_num` usable chats go
  to `val`, the rest to `train`, so train and val never share a chat.

### 3.2 Chats longer than the row are dropped (decided 2026-10-31)

A chat longer than T (the row length) is **dropped** (`meta.json` `drops: {"too_long": n}`). It is never cut or split:
every kept chat is whole, with every answer and its end token. This is the only method; there is no option.

Why (the user's decision, after building and testing splits on 2026-10-30 and 2026-10-31):
- SFT must teach the model to answer fully and to stop at the end token. A cut answer has no end token in its first
  piece, and the next piece starts in the middle of the answer, either without the question (TRL's `bfd_split`) or with
  the same question repeated in front of an answer that starts halfway (our `split_turns`). Both confuse the model.
- A multi-turn chat cannot be cut into single turns: every turn depends on the turns before it ("What is 2+2?" ...
  "Times 3?").
- More tokens of broken chats do not help. The same choice as torchtitan ("avoiding training on truncated responses").
- The permanent fix is a longer row with a long-context checkpoint (section 11): of the 23,253 chats over 4,096 tokens
  in the 10-28 mixes, 10,915 fit whole at 8,192, 19,538 at 16,384, 22,043 at 32,768.

If a split is ever needed: exactly TRL's `bfd_split`, with one rule added: if any prompt of a chat (the question of any
turn, first or later, with the system message) is as long as a whole row, the whole chat is dropped, single-turn and
multi-turn alike; nothing more (examples: the
comment at the length rule in `prepare_sft_data.py`, and the packing doc section 7.8.2). Everything we built, measured and decided, with the numbers:
`sft/docs/sft_packing_algorithms_bfd_ffd.md` section 7.8.

### 3.3 Parallel preparation: `--workers N` (2026-10-30)

The rows are tokenized by N processes, in chunks, and the results are taken back in the dataset's shuffled order, so
the held-out val chats, `--num` and every count come out exactly as in the serial run. 20 workers: 41.8 s instead of
about 13 min for the long-source mix (2026-10-30); the data files are byte-identical to the serial run (checked again
2026-10-31: 1 and 8 workers, tulu-plain-v1 and chatml-v1). `--workers 1` (default) is the serial loop.

---

## 4. Reading the data in training (`sft/SftDataLoader.h`)

- Reads the files once and checks them (constructor from line 46; checks at lines 190-220). A bad file stops the
  run, because it would otherwise train silently on wrong targets.
- `next_batch()` (line 102) returns B rows on the GPU, the same `Batch` struct as the pretraining loader.
- Differences from the pretraining loader (lines 19-23): targets come from `labels.bin` (not the input shifted by
  one); whole rows (not a window over a text stream); it reports the trained-token count of each micro-batch.
- Packed rows (when `seq_lengths.bin` exists): `last_positions()` = int32 [B, T], each token's index inside its chat;
  `last_segment_rows/starts/lengths()` = every chat of the B rows and then each row's pad tail as a segment of its
  own (line 137). Together they cover every token exactly once; this list is the input of the attention plan.
- Ranks never share a row: micro-batch m of rank r is rows `[(m*W + r)*B, +B)` (line 24).
- The old per-token format (`seg_start.bin`) is refused (line 16).
- Its 6th constructor argument is the GPU (device ordinal): the trainer passes the **local** rank.
- Its 8th argument `vocab_rows` = the model's embedding rows (the trainer passes `cfg.vocab_size`). **Required** since
  2026-10-31: it defaulted to 50,304 (the 1.7B's rows), but the 135M base has 49,280 (49,152 tokenizer ids + 128 spare),
  so any fixed default is wrong for one of our models. Every input id must be below it, and the data's `meta.json`
  `vocab_rows` must equal it: data prepared for another model / tokenizer stops the run with a clear error. A 135M run
  therefore needs `VOCAB=49280` (section 8). Note: two different tokenizers of the same size are not caught (the
  checkpoint does not record its tokenizer); use the tokenizer the checkpoint was pretrained with.
- With the trained-rows switch (`set_trained_rows(true)`, set by `BLU_SFT_TRAINED_ROWS=1`, 2026-11-03): for each
  micro-batch it also lists every token that has a label, as a row index into the [B x T, d_model] activations, with its
  label (`last_trained_rows()`, `last_trained_labels()`), uploaded with the batch and padded to a multiple of 256
  (padding: index -1, label 65535). With the switch off (the default) nothing changes.

---

## 5. The trainer (`train/production_zero/bluscript_sft.cpp`)

It is `bluscript_zero_ckpt.cpp` plus the SFT parts. Without `BLU_SFT_DATA` it trains exactly like
`bluscript_zero_ckpt.cpp` (same model, ZeRO-1 AdamW / Muon, DCP checkpoints, NCCL or BluCCL, the YAML config).

| Setting | Default | Meaning |
|---|---|---|
| `BLU_SFT_DATA` | (empty = pretraining) | folder with `train/` and `val/` from `prepare_sft_data.py` |
| `BLU_SFT_B` | 8 | rows per micro-batch per GPU |
| `BLU_SFT_ROWS_PER_STEP` | 64 | rows per optimizer step (all micro-batches, all GPUs); a multiple of B x GPUs |
| `BLU_SFT_EPOCHS` | 2 | passes over the train set; the number of steps is computed from it |
| `BLU_SFT_LR`, `BLU_SFT_MIN_LR` | 2e-5, lr/10 | AdamW peak and final learning rate (cosine) |
| `BLU_SFT_MUON_LR`, `BLU_SFT_MUON_MIN_LR` | required with `BLU_OPTIMIZER=muon` | Muon learning rates (the pretraining Muon LR would wreck the weights) |
| `BLU_SFT_WARMUP` | 0.03 | warm-up as a fraction of all steps |
| `BLU_SFT_VAL_FREQ` | 10 | validation every N steps, over the whole val set |
| `BLU_INIT_FROM` | — | start from another run's checkpoint: weights only, fresh optimizer (`BLU_INIT_VERIFY=1` checks every weight's CRC32) |
| `BLU_N_LAYERS`, `BLU_ROPE_THETA` | 26, 10000 on this branch | the model the checkpoint was trained as. **Always set both**: a wrong theta gives no error, only a worse model |
| `BLU_D_MODEL`, `BLU_Q_HEADS`, `BLU_KV_HEADS`, `BLU_FFN_HIDDEN`, `BLU_VOCAB`, `BLU_EMBED_SCALE` | the built-in model / the YAML | the other shape settings (2026-10-27, with the names of preproduction-135M). They must be the values of the run that made the checkpoint: a wrong width, head count, ffn size or vocab stops the load (shape mismatch), a wrong `BLU_EMBED_SCALE` does **not** (no weight depends on it). The 135M base: 512 / 4 / 2 / 1408 / 49280 / 1 |
| `BLU_INIT_FROM_RAW` | — | start from `sft/hf_to_blutrain.py` output (a Hugging Face model such as the 135M base) instead of a DCP checkpoint; weights only, fresh optimizer |
| `BLU_SFT_T` | the YAML's context length | the row length (= the RoPE cache length); must equal the data's `meta.json` `"T"` (the loader checks); `run_sft.sh` sets it from the data (2026-10-28, 10-31) |
| `BLU_SFT_SCHEDULE` | cosine | the learning-rate shape after warm-up: `cosine`, `linear`, `constant` or `wsd` (warm-up, stable, decay); every decaying shape ends at the min LR (2026-10-28) |
| `BLU_SFT_DECAY_FRAC` | 0.2 | with `wsd`: the decay part, as a fraction of all steps (`run_sft.sh` does not pass it, so its `wsd` runs use 0.2) |
| `BLU_SFT_TRAINED_ROWS` | 0 (off) | 1 = the lm_head and the loss run only on the tokens that have a label (gathered before the lm_head, the gradient scattered back); same math, experimental (2026-11-03; sections 7 and 9) |
| `BLU_SFT_CP` | 1 (off) | U > 1: U GPUs share every row (context parallelism, section 5.2); U must divide the GPU count, the query heads, the KV heads and T; not with `BLU_SFT_TRAINED_ROWS` |
| `BLU_EXPORT` | — | `<path>.safetensors`: the final weights (fp32, BluTrain names, rank 0); `run_sft.sh EXPORT=1` writes `<RUN_DIR>/final.safetensors`, and `sft/blutrain_to_hf.py` makes a Hugging Face model of it |

Packed rows are used automatically when the data has `seq_lengths.bin`; they need `USE_FUSED_RMS_ROPE=1`
(`build_sft_trainer.sh` sets it). The sequential RoPE path has no positions, so the trainer stops with an error there.

**The SFT parts, place by place** (search for the quoted text):

| Where | What | Why |
|---|---|---|
| `"PackedRows g_packed"`, `packed_rows_of()` | per micro-batch: positions + the chat plan (`gqa_chat_plan_make`) from the loader's segment list | one plan for every layer's forward AND backward |
| `g_packed.positions` (attention block) | `fused_rms_rope_bf16(..., g_packed.positions)` (and the fp32 path) | RoPE positions restart at 0 for every chat |
| `g_packed.chats` | `scaled_dot_product_attention(..., g_packed.chats)` | each chat attends only to itself; null = plain causal (pretraining) |
| `"BLU_N_LAYERS / BLU_ROPE_THETA"` | model shape overrides | must match the starting checkpoint |
| `"SFT mode: BLU_SFT_DATA"` | settings, the two loaders, the step count from the epochs | SFT on; the pretraining mixture is never opened |
| `"initialise the weights from another run"` | `BLU_INIT_FROM`: weights only, before the optimizers copy them into their fp32 masters | SFT starts from a pretrained model with a fresh optimizer (torchtitan's `initial_load_model_only`) |
| `train_loader` | the pretraining loader is not created in SFT mode | it would need the pretraining shards |
| `grad_scale` | backward seed: **W** in SFT, 1/GRAD_ACCUM in pretraining | each micro loss is already divided by the step's global count, and ZeRO averages over the W GPUs |
| `skip_micro_batches` | resume: skip exactly the micro-batches already trained | a restarted run continues on the right rows |
| `"SFT: the whole val set"` | validation: all val losses added, divided once by the set's trained-token count | the same rule as training |
| `valid_tokens_ahead`, `sft_denom` | the step's global trained-token count, all-reduced, passed to the loss with ignore index 65535 | every trained token of the step has the same weight (torchtitan's `global_valid_tokens`) |
| `loss_accum` | in SFT the micro losses just add up | dividing again would make the logged loss GRAD_ACCUM x W too small |
| `trained_tokens` (CSV) | the step's trained tokens, last column | the loss denominator, for the record |
| `"135M base"` | the other shape settings and `BLU_EMBED_SCALE` | any model shape (the 135M base) from the same trainer |
| `"BLU_INIT_FROM_RAW"` | weights from `sft/hf_to_blutrain.py` output | start from a Hugging Face model |
| `"BLU_EXPORT"` | the final weights as a safetensors file | the export for Hugging Face and the benchmarks |
| `trained_rows`, `model.head_rows`, `gather_rows` (2026-11-03) | with `BLU_SFT_TRAINED_ROWS=1`: the final hidden rows of the tokens that have a label are gathered before the lm_head, the loss gets the matching labels (`last_trained_labels()`), and the gradient goes back through the gather's backward (a scatter) | the lm_head and the loss skip the untrained tokens (TRL and Unsloth do the same); off = every row, as before |
| `"struct SftCp"`, `cp_make_group`, `cp_combine` / `cp_partition`, `cp_slice`, the `g_cp.U` branch of the attention block (2026-11-06/07) | the CP group, the two all-to-alls around attention, each GPU's token slice of the ids, labels and positions | one row shared by U GPUs (section 5.2); at U = 1 every one of these lines is skipped |

**Brought onto this branch on 2026-10-27** by a 3-way merge (base = the pretraining trainer the SFT trainer started
from, 909ab75; 10 conflicts, each resolved by keeping both sides). Adapted to this branch:
raw `MPI_Allreduce` / `MPI_Abort` -> `dist::comm::allreduce_scalar` / `abort_job` (they work under mpirun and BluRun);
the SFT loaders take the **local** rank as their GPU (the branch fixed the same bug in the pretraining loader);
`BLU_INIT_FROM` loads over an NCCL group under BluCCL too (DCP has no BluCCL backend), and the checkpointer reuses that
group; the shard-fingerprint check and the per-domain CSV columns are skipped in SFT mode; the CSV keeps this branch's
columns and adds `trained_tokens`. A check that every line the SFT work added is in the merged file found only the
lines changed on purpose, and no SFT-removed line left.

Why each loss rule (mean per micro-step vs over the whole step vs sum): `sft/docs/loss_calculation_in_pretraining_and_sft.md`.

### 5.1 Speed changes both trainers got (2026-11-03 to 11-05)

These changes are in the code SFT shares with pretraining (the ZeRO optimizers, ShardedMuon, the multi-tensor kernels,
the GEMM library), so pretraining got them too. Each was measured alone, then all of them together.

| Change | Commits | Measured |
|---|---|---|
| GEMM tile-order tables built on the GPU (no CPU wait for a new GEMM shape) | BluBLAS 297f372 | new-shape wait 20 ms -> 3.7-5.2 us; 16k split-K dW 10-15% faster |
| Muon Newton-Schulz guard and nsratio decided on the GPU | TI d98d6ee, BluTrain 37514ff | 222 CPU waits per step removed (135M); step -5.4% |
| `empty_cache` at step 0 only (not at step 5) | BluTrain e9e310c | one slow refill step removed (-6 s per run at B 4) |
| batched Newton-Schulz (same-shape matrices in one batched GEMM) | TI dbb4e5f, BluTrain 96f135c, BluBLAS 0b552f0 | 222-matrix Newton-Schulz 21.6 -> 4.8 ms; step -13.6% (1 GPU); bit-identical weights |
| `get_arch()` once per GPU | TI 963df8b | same speed on CUDA 13 (it caches the call); removes about 1 ms per call where the CUDA stack does not |
| `zero_grad`: one multi-tensor launch | BluTrain 084114b | memsets per step 395 -> 23; step -0.70 ms |
| parameter scatter / gradient staging: a few multi-tensor copies instead of one call per tensor | TI 05e1d6c, BluTrain 07600db | SFT copy lists 4.5-8.7x faster (e.g. 770 -> 157 us) |
| Muon per-matrix prep (momentum, weight decay, gradient norm) batched | TI e2af706, BluTrain 1bec9af | about 1,000 fewer launches per SFT step; SFT step 108.0 -> 105.3 ms (-2.6%) |
| no CPU wait after each multi-tensor norm / scale / Adam | TI 66a32a7, BluTrain 476808b | stream syncs 14 -> 7 per step; step time the same within noise |
| CPU waits that ordered nothing removed (Muon norm, 1-GPU publish, parameter scatter) | BluTrain accc6c8 | waits per SFT step 10 -> 5; step time the same within noise |

All of them together (the code of 2026-11-03 against the code of 2026-11-05, 36 settings, 30 steps each, median of
steps 11-29; work log §45.174): SFT 135M -13.7% on average (best -31.8%), SFT 1.6B -1.7%; pretraining 135M -11.8%,
1.6B -1.3%; the same losses, 0 errors. preproduction-135M against this branch (135M, the config of the 900B base,
2026-11-06): SFT step 348.7 -> 309.1 ms (-11.4%).

One bug fixed on the way matters for correctness: the multi-tensor Adam / scale / norm kernels shared ONE pinned upload
buffer, so a second call could overwrite the first call's tensor list before the GPU read it (wrong results, no error;
shown 3 times out of 3). The waits after every call hid it. The list now travels inside the launch, as in PyTorch's
`multi_tensor_apply` (TI 66a32a7; results bit-identical on H100 / H200).

### 5.2 Context parallelism: one row shared by several GPUs (`BLU_SFT_CP=U`, 2026-11-06/07)

When one row does not fit one GPU (or to train long rows faster), U GPUs share every row, the Ulysses way. The two
all-to-alls are Adhitya's functions from the `dist/Context_Parallelism` submodule, used as they are (header-only; the
`Makefile` adds its include path). No kernel changes: our QK-norm + RoPE (per-chat positions) and our packed attention
(chat plan) run exactly as without CP.

One step with CP:

```
the U GPUs of one CP group read the SAME rows (the loader uses the data-parallel rank = rank / U of world / U)
each GPU keeps its own slice of every row: tokens [r*T/U, (r+1)*T/U) with their labels and RoPE positions
per layer:
  RMSNorm, wQ / wK / wV, fused QK-norm + RoPE   on the slice (each token keeps its true position inside its chat)
  all-to-all "combine"                          [B, H, T/U, hd] -> [B, H/U, T, hd]: every GPU now holds the WHOLE row
                                                for H/U heads
  packed attention with the whole row's plan    exactly as on one GPU (the chat plan is built for H/U heads)
  all-to-all "partition"                        [B, Hq/U, T, hd] -> [B, Hq, T/U, hd]: back to the slice
  wO, SwiGLU MLP                                on the slice
lm_head + loss on the slice: the slice's trained-token sum / the step's trained-token count over all GPUs (each CP
group's rows counted once), so every trained token has the same weight, as without CP
backward: each all-to-all's gradient is the inverse all-to-all (his UlyssesCombineBackward / UlyssesPartitionBackward);
ZeRO reduces the gradients over all GPUs as before
```

- **Rules** (the trainer stops with an error otherwise): U divides the GPU count, the query heads, the KV heads and T;
  not together with `BLU_SFT_TRAINED_ROWS` (its row lists are for whole rows). So the 135M (4 q / 2 KV heads) allows
  U = 2, and the 1.6B (18 q / 6 KV heads) U = 2, 3 or 6 (on 8 GPUs: U = 2). Rows per step = B x GRAD_ACCUM x (GPUs / U).
  More CP GPUs than KV heads would need copies of the KV heads or Adhitya's ring + Ulysses hybrid (not wired; for SFT
  its ring part would also need per-chat positions).
- **The CP group:** `ncclCommSplit` off the world NCCL group, with a blocking CUDA stream, as Adhitya's DeviceMesh does:
  his all-to-alls do not wait on the compute stream, and a blocking stream is ordered after the default stream our
  kernels use.
- **Needs Tensor-Implementations d59740b or later** (section 6.5): before it, k's gradient and the k-norm gamma gradient
  were dropped under CP (step-0 gradient norm 7.5% lower than without CP).
- **Run:** `SFT_CP=2 NP=2 GPUS=0,1 ... bash sft/run_sft.sh` (the rest as in section 8).
- **Results** (2026-11-07, 2x H200; section 7): CP=2 trains the same as without CP, within the normal run-to-run
  variation, at 4k, 64k and 128k tokens per row; at 64k / 128k it is 1.58x / 1.65x faster than one GPU and needs 56-59%
  of one GPU's memory per GPU; the 1.6B shape at 64k tokens fits only with CP=2.
- **Not tested yet:** CP together with data parallelism (more GPUs than U).

---

## 6. The kernel changes (Tensor-Implementations)

### 6.1 Loss: ignore index and the global denominator

- `sparse_cross_entropy_loss(logits, targets, ignore_index = -1, denom = nullptr)` (`include/autograd/operations/LossOps.h`):
  `ignore_index = -1` = no masking (pretraining unchanged); SFT passes 65535. `denom = nullptr` = divide by the batch
  as before; SFT passes a 1-element GPU tensor with the step's global trained-token count (no CPU wait).
- On this branch: ad10e7d (ignore index, denominator, fp32 loss scalar, dtype-aware vector loads), e857c77 (the
  gradient allocated once), 43a5c3f (backward 16-byte loads/stores: bf16 backward kernel 4.35 -> 1.64 ms), 37f2196
  (the gradient written over the logits: -5.8 GiB at B 8 x T 4096 bf16, bit-identical), fc4459b (forward 16-byte
  loads: 1.03 -> 0.93 ms).
- Tests: 136 masking / denominator cases, 0 failures (work log §26); the in-place gradient bit-identical to the separate
  buffer, -5,926 MiB (work log §45.38). Those harnesses are in the SFT work records, not in this repo: the in-place
  test compared the old opt-in switch with the separate buffer, and since 2026-10-21 in-place is the only path.
- **The loss sum counted only the first 65,536 rows (fixed 2026-10-30).** The forward adds the per-token losses in two
  steps: one partial sum per 256 rows, then ONE block of 256 threads adds the partials. Each thread read one partial, so
  only 256 x 256 = 65,536 rows reached the loss, while the denominator counted all of them. The printed loss read about
  true x 65,536 / N (0.80x at 81,920 tokens per GPU, 0.50x at 131,072). Gradients were never affected: the backward uses
  the saved per-row max / sum and the denominator, not the summed loss. This was in the library since at least 2026-04,
  so it applies to **pretraining too**: any run with more than 65,536 tokens per GPU in one loss call printed a wrong
  loss (long-context runs, large B x T). Fix (`sum_reduction_kernel`): one load per thread while the grid covers all
  rows (so every sum up to 65,536 rows is bit-identical to before), otherwise a loop over all partials. Chosen against
  nine other ways (an always-on loop, PyTorch's one-block loop, multi-pass, atomicAdd, CUB, a last-block finish, a
  torch.sum-like reduction) measured on H200: the only one with today's bits, the same result every run, no new memory
  and today's speed (work log §45.108 parts 12-16).
- Not done yet: launching the loss kernels with the library's `CudaNativeType<T>` mapping (the 16-byte helpers do the
  same job today; work log §45.42).

### 6.2 RoPE: per-chat positions

- API: `fused_rms_rope(..., positions)` and `fused_rms_rope_bf16(..., positions)`
  (`include/autograd/operations/NormalizationOps.h`): optional int32 [B, T]; empty = pretraining (position t + pos_offset).
  It is saved for the backward, which must rotate by the same positions.
- Every kernel has a `HAS_POS` template flag: false = the original kernel, unchanged (checked by SASS diff); true = cache
  row `positions[b*T + t]`. The fp32 Hopper kernels (`norm/arch/FusedRMSRoPE_*_sm90*.cu`), the portable kernels and the
  CPU reference (`Normalizations.cpp`) take it; so does the bf16 kernel the trainers use.
- The bf16 kernel (`src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu`, Kruthik's fused bf16 RMSNorm + RoPE) also got a
  2-4x faster backward (2026-10-23/24: tokens kept in flight with cp.async, one launch for the q + k dgamma reduce);
  **bit-identical** to the original in all 8 outputs, 44/44 cases on H200 and RTX 3060; fastest of torch eager,
  torch.compile, Liger, FlashAttention and Quack in 16 of 16 cases; the whole trainer step -0.68% (SFT) / -0.49%
  (pretraining). The numbers are in the file's header.
- Test: `Tests/kernel_tests/test_rope_positions.cu` (identity, shift, packed == each chat alone bit-exact, random vs fp64,
  API checks).

### 6.3 Attention: the chat plan

- `struct GqaChatPlan` (`include/ops/helpers/AttentionKernels.h`): one GPU allocation with the lists every kernel needs,
  built from the segment list (row, start, length) by `gqa_chat_plan_make` / `gqa_chat_plan_build`
  (`GQA_fwd_generic.cu`), which checks that the segments cover every token exactly once and uploads with ONE async copy.

| List | Used by | One entry |
|---|---|---|
| `fwd_units` | Hopper forward (v62 / QPU) | {row, start, length, couple + head}; since 2026-11-13 also "mixed" units (see below) |
| `bwd_tiles` | Hopper backward Vj1d | one 64-key tile of a chat |
| `vj2_blocks` + `dq_tiles` | Hopper backward Vj2 (default) | one 128-key block of a chat for one head; padded dQ slots |
| `gen_segs` (2026-10-24) | every other GPU (generic kernels) | {row, start, length, 0} per chat |

- **Mixed units (2026-11-13).** `gqa_chat_plan_make(..., Hq, Hkv)` now takes the KV-head count. In a chat with an odd
  number of 64-row tiles (64 tokens or less, 129-192, ...), the second compute group of the last pair had no rows; it now
  runs the same tile of another query head of the same KV head (same K / V). Used only when at least 2% of the plan's
  units would share a tile; otherwise the plan and the kernel are the ones of before. Same results; forward -21 to -26%
  on rows of chats of 64 tokens or less, -0.5 to -1.3% on real SFT rows (H200, heads 18/6); no measurable change of the
  SFT step time on real data. Switches: `BLU_GQA_MX=0`, `BLU_GQA_MX_MIN_PCT=<n>`. Details:
  `docs/kernel_diff_sft_pretraining.md` section 3.2.
- **Backward: ideas 1, 2 and 4 (2026-11-14 to 11-18; `GQA_bwd_vj2_sm90.cu`, plan builder in `GQA_fwd_generic.cu`).**
  Before: Vj2 ran one block per (128-key tile, query head); every query head wrote a bf16 dK / dV partial and two
  head-sum kernels added the G partials; dQ went through an fp32 buffer (zeroed by a prep kernel, reduce-added per
  step, converted to bf16 at the end). Three changes; packed == alone stays bit-exact:
  - idea 1: in a block's first query step every key of the second compute group (keys 64-127) comes after every query
    of the step, so the causal mask makes that group's whole P zero. The group now only writes zero dS rows and runs its
    dQ GEMM; the S / dP GEMMs, the exps and the dV / dK GEMMs (which added exact zeros) are gone. Dense rows too.
  - idea 2 (packed rows): a chat's first two 64-row query tiles get dQ from one block only (key block 0). That block
    writes them in bf16 directly (the convert kernel's rounding); the prep and convert kernels skip them (`dq_tiles`,
    4th number = 1).
  - idea 4: one block per (128-key tile, KV head) runs the G query heads of the group one after another, keeps dK / dV
    in fp32 registers across the heads and writes the final dK / dV (one rounding, the way FA3 adds GQA heads): no
    partials, no head-sum kernels. On when the call has at least `BLU_VJ2_HSUM_MINKV` KV heads (default 2), decided
    from the shape only, so a chat packed and the same chat alone take the same path. Packed rows use a per-KV-head
    block list that the plan builds when it knows Hkv and G >= 2 (kept in a side table keyed by the plan, so
    `GqaChatPlan` keeps its layout). Block order: sections of whole rows, as many as fit about 80 MB (G x T x 1 KB per
    (row, KV head) pair), rounded down to a power of 2. Dense rows (pretraining) too.
  - Switches: `BLU_VJ2_HSUM=0` turns idea 4 off; `BLU_VJ2_DKDV_F32=1` (section 6.4) also turns it off;
    `BLU_VJ2_HSUM_SECT=<n>` sets the block order (n pairs per section, -1 = one section). Ideas 1 and 2 have no switch.
    When idea 4 is off, one line per reason is printed once per process; nothing is printed while it is on.
  - Results (H200, 2026-11-17/18): the test suite on the push tree passes (quick, heads 4/2: 273 pass / 0 fail; 18/6:
    280 / 0; one dK entry is listed as known: a 2-token pad piece where FA3 and FA4 give a bit-identical dK, and pad
    tokens get no gradient in training). Forward + backward call, 12 cases (heads 18/6 and 4/2, T 4,096, packed rows of
    mixed and of real chats): 0.570-0.980 of the time of the code of before (the backward call faster in all 12), and
    0.235-0.928 of the time of FA3 / FA4 / cuDNN (all 12, same GPU and session). SFT trainer, step time (median of
    steps 11-29, runs base / new / new / base): 135M -1.7%, 1.6B shape -0.7%, losses within the run-to-run spread.
    Pretraining trainer, 1.6B production config (dense rows; 2 micro-batches of 8 x 4,096 per step instead of 32,
    everything else as in production): step -0.54% (1,508.8 -> 1,500.6 ms; the same build repeats within 0.04-0.06%),
    step-0 loss identical, losses within the run-to-run spread. Work log §45.197-45.199.
- Every kernel counts its tiles from the chat's start, so a chat in a packed row runs the same tiles in the same order
  as the same chat alone: **bit-for-bit the same result** (every packed test checks it).
- Which kernel runs where:

| GPU | Forward | Backward | Packed chats |
|---|---|---|---|
| H100 / H200, causal, head size 128 | `GQA_fwd_sm90.cu` (QPU) | Vj2 (`GQA_bwd_vj2_sm90.cu`) by default; Vj1d with `USE_VJ2=0` | yes |
| any other GPU except B300 (Ada, RTX 30xx, A100, ...) | generic `gqa_v2<CAUSAL>` (`GQA_fwd_generic.cu`) | generic `gqa_backward_dQ` / `gqa_backward_dKdV` (`GQA_bwd_generic.cu`) | yes (since 2026-10-24) |
| B300 | head size 64 only | head size 64 only | no (refused) |

- Why a per-chat list (design A, "cu_seqlens") and not one number per token (design B, "seg_start", removed 2026-10-15):
  design A is what FlashAttention 2/3/4, TransformerEngine, PyTorch varlen and TRL use; in our kernels it was 2-16%
  faster forward and 3-5% faster backward, and only design A gives packed == alone bit-exact for every chat (work log §45.7).
- Speed (H200, every library on the same box, 2026-10-21): dense forward ties FA3 (+0.6%); dense forward + backward
  takes 1.8% / 3.0% / 12.7% less time than FA3 / FA4 / cuDNN; packed rows 6.5% / 7.7% / 11.9% less
  (`sft/docs/sft_kernel_benchmarks.md`).
- Lower GPUs (2026-10-24): before this, any GPU other than H100/H200 had no correct causal kernel for our model (head size
  128 stopped, head size 64 silently ran without a mask). The generic kernels now have the causal mask and the chat plan;
  correct, NOT optimized yet (0.9 TFLOP/s on an RTX 3060). The Hopper files are not touched by this change.
- Tests: `Tests/kernel_tests/test_varlen_attention.cu` (G1-G13, against an fp64 reference and a bf16 floor: pass when the
  error is at most 2x the floor's, FlashAttention's own rule; packed == alone bit-exact) and `sanitize_vj2.cu`.

### 6.4 Long context, 16k-128k (2026-10-31)

Every kernel the 135M SFT trainer runs was checked forward and backward at 4k-131k (work log §45.113-45.126):
- RoPE (`RoPEKernels.cu`, `arch/RoPE_sm90.cu`): the cos / sin table builder put rows in grid y (max 65,535 blocks),
  so at T >= 65,536 the table was never built and training rotated with an unfilled table. Flat grid for the tables,
  token in grid x for the per-token kernels; launch errors throw.
- Loss (`LossKernels.cu`): the fallback (validation) backward put rows in grid y and failed at >= 65,536 rows; now
  (row, part) in grid x.
- `contiguous()` (`Views/ContiguousKernel.cu`, `arch/ContiguousKernel_sm90.cu`): past 2^31 / 2^32 elements (the logits
  at 128k = 6.46e9 values) it used 32-bit indexes; now split into < 2^31 pieces as PyTorch does, 64-bit as fallback.
- Embedding (`EmbeddingKernels.cu`): tokens in grid y failed above 524,280 tokens per call; tokens now in grid x.
- SwiGLU on sm120 (`SwiGLUKernels_sm120.cu`): rows in grid y (max 262,140) -> capped + a row loop; and an old bug:
  for hidden % 4 != 0 the fp32 float4 kernel used wrong row strides; float4 now only for hidden % 4 == 0 and 16-byte
  aligned pointers, else the scalar kernel.
- Attention backward (`GQA_bwd_vj2_sm90.cu`): optional fp32 dK / dV partials, `BLU_VJ2_DKDV_F32=1`, OFF by default
  (default = the same SASS as before). More precise, but the attention backward is +3-14% and the step +0.8%.
  2026-11-18: with idea 4 on (section 6.3) the G heads' dK / dV are already added in fp32 and rounded once, so the
  default no longer has the per-head bf16 rounding this switch was for; `BLU_VJ2_DKDV_F32=1` now also turns idea 4 off
  (the per-query-head path with fp32 partials runs; ideas 1 and 2 stay on).
- `Makefile`: `--use_fast_math` only for `LossKernels.cu`; every other kernel uses exact math (step +0.4%, noise).
- Test: G14 in `test_varlen_attention.cu`: real packed rows at any T against a blocked fp64 reference.

### 6.5 `autograd::contiguous` returns an already-contiguous input as is (2026-11-07)

- The bug: `Tensor::contiguous()` returns the same tensor (no copy) when it is already contiguous. The autograd wrapper
  `autograd::contiguous` then put a new backward node (`ContiguousBackward`) on that same tensor, that is on its INPUT,
  and kept the input's output number. For the 2nd or later output of an op with several outputs (the fused QK-norm +
  RoPE op returns q as output 0 and k as output 1), the next op's link pointed at slot 1 of a node that has only slot
  0, and that gradient was dropped silently: k's gradient and the k-norm gamma gradient were 0 in every layer.
- Found by the SFT context-parallel work (section 5.2), which passes q / k / v through
  `autograd::contiguous` before the all-to-all: the step-0 gradient norm was 2.591 instead of 2.800.
- Fix (`src/autograd/operations/ReshapeOps.cpp`, d59740b): already contiguous (offset 0) -> return the input and record
  nothing, the same rule as `autograd::to_dtype` and PyTorch's `contiguous()`. A non-contiguous input is unchanged (a
  copy plus `ContiguousBackward`). The old body is kept there as a comment.
- Effect on today's trainers: none. Every other call passes a non-contiguous view (a transpose), which takes the same
  path as before; normal SFT with and without the fix differs less than two identical runs do (section 7). The same
  holds for pretraining.
- Test: `Tests/kernel_tests/test_contig_alias.cu`: the real RoPE op, the loss straight on q / k against the loss
  through `autograd::contiguous`: equal gradients with the fix; k and k-gamma 0 without it.

---

## 7. Tests and results

| What | Where | Result |
|---|---|---|
| Data script | `prepare_sft_data.py --self-test` | First-Fit fast == plain loop; labels and packing checked |
| Long-chat split removed (2026-10-31) | the pushed script (drop mode) vs the new one, same data | inputs, labels, valid counts, seq_lengths, seq_offsets byte-identical (train and val) on the long-source mix and on a 6,000-chat mix in tulu-plain-v1 and chatml-v1; 1 worker == 8 workers; `--self-test` passes for GPT-2 and the three 135M templates |
| Loss sum past 65,536 rows | `Tests/kernel_tests/test_sparse_ce_ignore_index.cu` group H | 154/154 after the fix (16 failures before); bits unchanged up to 65,536 rows; trainer val loss at 81,920 tokens 1.3007 before -> 1.6068 after = numpy reference 1.6063 |
| Data loader | `sft/test_sft_dataloader.cpp` | data, counts, ranks, epochs, errors (5 checks); 2026-10-31: all 5 pass on the packed 135M data, including data of another model refused |
| Long context 4k-131k (2026-10-31) | work log §45.113-45.126; G14 in `test_varlen_attention.cu` | every trainer kernel forward + backward against references; the 135M trainer runs at 4k-128k |
| Loss | work log §26, §45.36-38 (harnesses in the work records) | 136 cases 0 failures; in-place gradient bit-identical |
| RoPE positions (fp32 kernels) | `test_rope_positions.cu` | 37/37; packed == each chat alone bit-exact |
| RoPE bf16 kernel vs the original | the port test of work log §45.82-45.90 (not in this repo) | 44/44 bit-identical on H200 and RTX 3060 |
| Attention, H200 | `test_varlen_attention.cu` G1-G13 | full suite 1249/1249 (2026-10-21); packed == alone bit-exact |
| Attention, RTX 3060, generic | `test_varlen_attention.cu` quick | hd 128: 292 pass, 1 known; hd 64: 291 pass, 1 fail (below) |
| Real SFT run (Tülu 3, 1.7B, 1x H200) | work log §45.67, §45.71-45.74; `README_SFT_checkpoint_for_DPO.md` | 11,590 steps, val 1.1757 -> 0.8501, 1.88 s/step, 52% MFU |
| Trained-rows switch (2026-11-03) | H200, 135M, 60 steps, 480 rows of 4,096 tokens, 79.8% of the tokens trained | on vs off: the same loss (step 59: 1.270407 / 1.270379, the size of run-to-run differences); lm_head + loss kernels 4.45 -> 3.94 ms per micro-batch (-11%, about 0.7% of the step); with BluBLAS 297f372: 21 blocking table copies -> 0, about 0.8 ms less GPU idle per micro-batch |
| BluBLAS GEMM / addmm (2026-11-03 to 11-05) | the BluBridge-BLAS test suites (described in its commits) | 6,644 GEMM + 13,302 addmm cases, 0 failed on 2x H200 (the old library: 654 of the 6,644 failed); 7,220 cases, 0 failed on an RTX 3060 |
| Speed-ups of section 5.1, all together (2026-11-06) | 36 settings, 30 steps, median of steps 11-29 (work log §45.174) | SFT 135M -13.7% on average, SFT 1.6B -1.7%; the same losses, 0 errors |
| `autograd::contiguous` fix (2026-11-07) | `test_contig_alias.cu`; the TI gradient checker (autocheck); 135M SFT, 30 steps, H200 | gradients through `autograd::contiguous` equal the run without it (k 156.991745, k-gamma 350.833604; before the fix 0 and 0) on an RTX 3060 and an H200; autocheck 43/43; normal SFT fix vs no fix: largest train-loss difference 0.000604, two identical runs 0.000630 |
| Long rows on H200 (2026-11-07) | 135M base (pretrained at 4,096 tokens), long-source mix, rows of 65,536 / 131,072 tokens, B 1, 30 steps | no errors; step-0 val loss 3.216672 at both lengths (the 2026-10-31 value); peak 60,913 / 110,361 MiB (nvidia-smi); the loss on tokens past position 4,096 is about 6.9 (section 11) |
| Context parallelism, 4k (2026-11-07) | 2x H200, 135M ckpt_A, a 480-row train set of 4,096-token rows, B 4, 4 rows per step, 30 steps | step-0 gradient norm: no CP 3.0473, CP=2 3.0473 (gammas 0.0458 = 0.0458); CP=2 without the TI fix of section 6.5, same box: 2.8367 (-6.9%); over 30 steps CP=2 differs from no CP by at most 0.000997 in loss (two identical no-CP runs: 0.000813) |
| Context parallelism, 64k / 128k (2026-11-07) | 2x H200, 135M, long-source mix, B 1, 30 steps; CP=1 twice (GPU 0, GPU 1) and CP=2 | CP=2 within CP=1's own run-to-run variation (step-0 norm -0.02% / +0.18% vs CP=1; CP=1 vs CP=1: +0.56% / +0.24%); step 250.7 -> 158.2 ms / 492.9 -> 297.9 ms; peak per GPU 61.5 -> 36.3 GB / 111.0 -> 62.2 GB (nvidia-smi, including a 0.6 GB lock process) |
| Context parallelism, 1.6B shape (2026-11-07) | 2x H200, random init, 26 layers, 18 q / 6 KV heads, vocab 49,280 | 4k: CP=2 step-0 norm +0.017% vs no CP, 30-step loss difference at most 0.000737, step 480.0 -> 274.5 ms; 64k: CP=2 runs at 96.6 GB per GPU (861 ms per step), one GPU runs out of memory |
| Attention backward ideas 1 + 2 + 4 (2026-11-17/18, section 6.3) | `test_varlen_attention.cu` quick suites on the push tree; one H200 session with the code of before, FA3, FA4 and cuDNN (12 cases); SFT and pretraining trainers, 30 steps, base / new / new / base | 273 / 0 (heads 4/2) and 280 / 0 (18/6), one dK entry known (FA3 / FA4 bit-identical there); forward + backward call 0.570-0.980 of the code of before, 0.235-0.928 of FA3 / FA4 / cuDNN; SFT step 135M -1.7%, 1.6B shape -0.7%; pretraining 1.6B (production config, 2 micro-batches per step) -0.54%; losses within the run-to-run spread |

---

## 8. Build and run on this branch

```
make -j$(nproc) profiler tensor dist SM_ARCH=90          # the libraries (CPU only)
bash sft/build_sft_trainer.sh                            # -> ./bluscript_sft_bin
SFT_DATA=<data dir> INIT_FROM=<checkpoint dir> N_LAYERS=28 ROPE_THETA=50000 RUN_DIR=<new folder> bash sft/run_sft.sh
```

`run_sft.sh` sets every variable of the 2026-10-22 run (20 rows per step, Muon + AdamW at 5e-5 -- why Muon:
`sft/docs/optimizer_choice_muon_vs_adamw_for_sft.md` --, 2 epochs, bf16,
validation every 200 steps, checkpoints every 1000) and removes any other `BLU_*` variable first. Since 2026-10-30 it
also sets `BLU_EMB_BWD=segmented`, the embedding backward pretraining uses (its YAML: `embedding.backward: segmented`;
fp32 sums, deterministic). Before that, every SFT run got the library's default, a bf16 atomicAdd (not deterministic),
because the script passes no YAML. Rule: an SFT run uses the same settings as pretraining. `NP=2 GPUS=0,1` runs on
two GPUs (ROWS must stay a multiple of B x NP). `RESUME=1` continues the run's own newest checkpoint.

The 135M base also needs its shape (`N_LAYERS=37 ROPE_THETA=50000 D_MODEL=512 Q_HEADS=4 KV_HEADS=2 FFN_HIDDEN=1408
VOCAB=49280 EMBED_SCALE=1`, header of `run_sft.sh`); without `VOCAB=49280` the loader stops (the data says 49,280 rows,
the built-in model 50,304; 2026-10-31).

The other knobs of `run_sft.sh` (all listed in its header): `SCHEDULE=cosine|linear|constant|wsd` (`wsd` with the
default decay fraction 0.2), `SFT_TRAINED_ROWS=1` (the experimental trained-rows switch; section 9),
`INIT_FROM_RAW=<dir>` (start from a Hugging Face model converted by `sft/hf_to_blutrain.py`), `EXPORT=1` (the final
weights as safetensors), `SFT_CP=U` (context parallelism with NP GPUs, section 5.2; with CP, ROWS must be a multiple of
B x NP / U). The row length is read from the data's `meta.json`; an `SFT_T` given by hand must match it.

---

## 9. Known limits and open items

- Chats longer than the row are dropped (section 3.2): the fix is a longer row with a long-context checkpoint
  (section 11).
- Row length T: `prepare_sft_data.py --seq-len` accepts multiples of 256 up to 131,072 (2026-10-31; it was 4,096).
  The kernels are tested at 4,096-131,072 (2026-10-31, after the long-context fixes of section 6.4): every kernel the
  135M trainer runs, forward and backward, standalone against references, then the 135M trainer at 4k / 8k / 16k /
  32k / 64k / 128k (20 to 100 steps). `run_sft.sh` takes T from the data's `meta.json`.
- `sft/test_sft_dataloader.cpp`: fixed 2026-10-31 (end id and vocabulary rows from `meta.json`; a new case: data of
  another model must be refused); all 5 checks pass on the packed 135M data at T 4,096.
- The SFT backward is not bit-identical run to run (step-0 grad norm varies by about 1e-4 between identical runs;
  2026-10-30); which kernel is not found yet. Still so on 2026-11-07, after the deterministic split-K fix below:
  2.5158 vs 2.5157 at 4,096 tokens on an H200 (two identical runs).
  2026-11-18: one known source is the attention backward's dQ: it is added in fp32 with reduce-add, as in FA3's
  default backward, so its last bits can change from run to run (O, LSE, dK and dV repeat bit for bit). Since
  2026-11-18 a chat's first two dQ tiles are written directly by one block and repeat exactly; the other tiles still
  vary.
- lm_head + loss on the trained tokens only (TRL / Unsloth / Apple CCE do this; same math, less work; work log §33,
  §45.34): built and measured 2026-10-31 (work log §45.129-45.131). **On this branch since 2026-11-03 as a switch, OFF
  by default:** `BLU_SFT_TRAINED_ROWS=1` (BluTrain 7184fe1; TI d18f820, which also brought BluBLAS 297f372). Two
  BluBLAS issues had blocked it; both were analyzed, fixed and tested on 2026-11-02 (work log §45.133-45.134) and
  pushed in BluBLAS 297f372 (2026-11-03). The two issues, as found (the 2026-11-02 analysis corrected the 2026-10-31
  text of this item: issue 1 was described as "16 slots per GEMM tile-grid shape", it is 16 tables in total for all
  GEMMs; issue 2 as "wrong rows in the last partial 2,048-row block when M is not a multiple of 2,048", it is the last
  group of 8 tile rows, as below):
  (1) On H100/H200 the results are right and the lm_head + loss alone is 21-43% faster, but in the trainer BluBLAS's
  tile-order table cache (`sm90_tile_map_cached`, host side: 16 tables in total for all GEMMs, one per tile-grid size)
  did a cudaMalloc + a `cudaMemcpy` that waits for ALL queued GPU work (+ a `cudaFree` when full) for every new grid
  size. Only the lm_head backward dX GEMM ([trained rows x 512], a tall grid) gets a new size every micro-batch (the
  forward [rows x 49280] is not tall, dW [49280 x 512] never changes size), so the CPU waited once per micro-batch.
  Fixed-shape training (pretraining, SFT without the switch) does not hit it. Fix (tested): the kernel is unchanged
  (byte-identical code); a tall table depends only on the number of tile columns, so one table per (GPU, column count)
  serves every row count, grows only a few times per run, is copied without the CPU waiting and is never freed; same
  speed, bit-identical results and same memory on 59 GEMM shapes, and the wait went from 20 ms to 22-29 us per call.
  (2) On sm_80/86/89 (A100, RTX 30xx/40xx, RTX 6000 Ada) BluBLAS's BF16 and FP16 GEMMs visit tile rows in groups of 8
  (FP32 on sm_80/86: 16) and skip part of the last group when the number of tile rows is above 8 and not a multiple of
  8: those output tiles are never written (wrong rows, e.g. M = 6656 with 256-row tiles: rows 6144 onward). Our normal
  path never hits it (B x T is a multiple of 4,096); H100/H200 do not use these kernels. Fix (tested on an RTX 3060,
  29 cases): the last group gets only the rows left; identical results and speed where the old order worked, correct
  output where it was broken. The plan of 2026-11-02 (push both fixes, measure the switch in the real trainer, add it
  back as a switch) was done on 2026-11-03. Measured in the trainer (section 7): the same loss, lm_head + loss -11%
  (about 0.7% of the step). It stays off by default while it is experimental (open points in the next item).
- Trained-rows switch, open points (2026-11-03): the library's caching allocator allocates a new logits block when a
  micro-batch has more trained tokens than any cached block holds (one slow step for each new maximum, e.g. step 15);
  a fix is planned. Not tested yet: a micro-batch with 0 trained tokens, and more than one GPU.
- BluBLAS's split-K weight-gradient GEMMs with a very large K (e.g. [1024 x 512] with K = 16,384) were not
  bit-identical run to run (1-11 values differ; found 2026-11-02); another reason the backward is not bit-identical.
  Fixed 2026-11-03 for even N (BluBLAS 297f372: an fp32 slice per split and an ordered reduce instead of atomicAdd;
  1 distinct output in 20 runs); odd N unchanged.
- The loss kernels do not use `CudaNativeType<T>` yet (work log §45.42).
- Generic attention (non-Hopper GPUs) is correct but slow; next: a FlashAttention-2-style rewrite (cp.async, `ldmatrix` +
  `mma.sync`, 4 warps) and the dK/dV register spills at head size 128.
- Generic dQ computes dS from the bf16-rounded P: one known case at head size 128 (1.049x the error limit) and one at head
  size 64 (1.175x, `row 0 @2688 L=64`); packed == alone is still bit-exact. Fix: dS from the fp32 P, as Vj2 and FA3 do.
- B300: packed SFT rows are not supported (causal attention there is head size 64 only).
- Attention backward idea 4 (head sums inside the block, 2026-11-18, section 6.3) needs at least 2 KV heads in the
  call (`BLU_VJ2_HSUM_MINKV`, default 2). With context parallelism CP=2 on the 135M model each GPU holds 1 KV head, so
  those calls use the per-query-head path (ideas 1 and 2 still on) and print one "idea 4 OFF" line once per process.
  The 1.6B shape with CP=2 (3 KV heads per GPU) keeps idea 4.
- Context parallelism (section 5.2, 2026-11-07): tested with CP=2 on 2 GPUs (135M at 4k-128k, the 1.6B shape at 4k
  and 64k). Not tested yet: CP together with data parallelism (more GPUs than U). U must divide the KV heads, so more
  CP GPUs than KV heads (e.g. U = 4 on the 135M) is not possible with plain Ulysses (copies of the KV heads or Adhitya's
  ring + Ulysses hybrid would be needed). (Until 2026-11-07 this item said the work was still being built and tested.)

---

## 10. Where to read more

| Question | Read |
|---|---|
| Which kernels change for SFT and why, with proof from 10 libraries | `sft/docs/kernel_diff_sft_pretraining.md` |
| Packing: First-Fit vs Best-Fit, cross-chat attention, cu_seqlens, step counts | `sft/docs/sft_packing_algorithms_bfd_ffd.md` |
| How the loss is averaged (micro-step mean vs step mean vs sum) | `sft/docs/loss_calculation_in_pretraining_and_sft.md` |
| How the model learns in SFT | `sft/docs/How_learning_happens_in_sft.md` |
| Our SFT kernels vs PyTorch, torch.compile, Liger, Apple CCE, FA3, FA4, cuDNN (H200) | `sft/docs/sft_kernel_benchmarks.md` |
| Muon or AdamW for SFT: my test (7 Oct) and the Moonlight reference (all four pretraining x SFT pairs) | `sft/docs/optimizer_choice_muon_vs_adamw_for_sft.md` |
| The Tülu 3 SFT checkpoint for DPO | `sft/README_SFT_checkpoint_for_DPO.md` |
| The attention changes of 2026-11-13 and 2026-11-18 in full: how FA3 / FA4 / cuDNN do it, every case, the fairness checks | `attention_forward_and_backward_gqa_tests_and_benchmarks_2026_11_18.md` in Gautam's work docs |
| Every SFT optimization since the start, each with its context and before / after numbers | `all_sft_optimizations.md` in Gautam's work docs (SFT package) |

---

## 11. The row length T and the checkpoint's context length

- **What T decides with packing.** Positions restart at 0 for every chat, so T does not change what the model sees for
  a chat that fits: a 32k row of short chats looks to the model exactly like several 4k rows. T only decides the
  longest chat that fits whole, the memory (tokens per GPU = B x T) and how many chats share a step.
- **T longer than the checkpoint was trained at** (for example SFT at 32k on a 4k checkpoint) is possible: T is only a
  setting and RoPE has no learned position table. Short chats are fine. The long chats are the problem: the
  checkpoint never used positions past its trained length, and attention over distances it never trained on is out
  of distribution (quality falls sharply past the trained length). SFT alone does not fix that well: SFT data is
  mostly short, so the new positions are rarely trained (labs spend large budgets on extension: Llama 3 about 800B
  tokens for 8k -> 128k); the cheap extension methods change RoPE first (Position Interpolation, about 1,000
  fine-tuning steps; YaRN, about 0.1% of the pretraining tokens); and the model would learn long attention and the
  chat behaviour at the same time.
- **So the order is:** context extension first (RoPE scaling such as YaRN plus long documents: the long-context
  pretraining stages), then SFT at up to that length, with some long chats in the SFT data so the long ability does
  not fade. "SFT at 32k from a 4k checkpoint with YaRN on" is a side experiment (work log §45.108 part 18), not the
  default.
- **How long our long chats are** (the 10-28 mixes, the 23,253 chats over 4,096 tokens): at T = 16,384, 84% fit whole;
  at 32,768, 95% (every LongAlign chat); at 65,536, 99%. So with a long-context checkpoint most of them are no longer dropped.
- **Memory:** it grows with the tokens per GPU, not with T alone. Measured on H200: the 135M at 81,920 tokens per GPU
  used 71,465 MiB; the 1.6B at 32,768 tokens peaked at 88,139 MiB. Context parallelism is needed only when ONE row of
  length T does not fit one GPU.
- **Measured (2026-11-07, H200): what a 4,096-token checkpoint does on long chats.** The 135M base was pretrained at
  4,096 tokens (its `meta.json`: `context_length` 4096). On rows of 65,536 and 131,072 tokens of the long-source mix
  (30 steps each, B 1), the steps whose trained tokens all sit before position 4,096 have loss 1.63-2.43 and gradient
  norm about 1.4-2.0; the steps with long chats reach loss 5-9 and gradient norms up to 1,117, and are up to 6x slower
  (attention work grows with the square of the chat length). Fitted over the 30 steps: the loss is about 1.70 on tokens
  before position 4,096 and about 6.9 on tokens after it (correlation 0.95 at 65,536, 0.92 at 131,072). This is the
  fall in quality described above, in numbers: long-chat SFT needs a long-context checkpoint first.
- **Memory at 64k / 128k (2026-11-07, H200, 135M, B 1):** 60,913 MiB at 65,536 tokens and 110,361 MiB at 131,072
  (nvidia-smi peak, which includes the CUDA context; the trainer's own peak at 131,072 on 2026-10-31: 107,626 MiB
  allocated, 108,986 MiB reserved). It grows by about 0.75 MiB per token.
- **With context parallelism (section 5.2, 2026-11-07, H200):** CP=2 needs 62.2 GB per GPU at 131,072 tokens (one GPU:
  111.0 GB). The 1.6B shape at 65,536 tokens does not fit one H200 (out of memory at 143 GB) but fits with CP=2 (96.6 GB
  per GPU).
