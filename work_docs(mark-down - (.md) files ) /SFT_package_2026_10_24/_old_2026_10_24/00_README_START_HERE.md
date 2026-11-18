# SFT in BluTrain — everything in one place (package of 2026-10-24)

This package holds every file we changed or added to make SFT (supervised fine-tuning) work in BluTrain:
the data preparation script, the data loader, the SFT trainer, every library (kernel) change, the tests,
and the documents that explain each decision. This file is the map. It explains each part in simple words,
says where each change is (file and line, in the copies inside this package), and why it was made.

All line numbers below point into the files **inside this package**. All code is from the laptop tree
`~/Downloads/BluTrain_sft` on 2026-10-24.

---

## 0. What is in this package

| Folder | What it holds |
|---|---|
| `01_preprocessing/` | `prepare_sft_data.py` — turns a Hugging Face chat dataset into SFT files |
| `02_dataloader/` | `SftDataLoader.h` (reads the SFT files during training) and its test |
| `03_trainer/` | `bluscript_sft.cpp` (the SFT trainer), the pretraining trainer it started from (`bluscript_zero_ckpt_PRETRAINING_base_HEAD_909ab75.cpp`), and the full diff between them (`pretraining_to_sft_trainer.diff`) |
| `04_library/current/` | the current version of every Tensor-Implementations (TI) file we changed, at its path in the TI tree |
| `04_library/diffs_vs_TI_HEAD_d96b991/` | one diff per changed TI file, against TI commit `d96b991` (the base of this tree) |
| `04_library/committed_ad10e7d_...patch` | the loss change we already pushed on 2026-09-25 (ignore index + global denominator), which is inside `d96b991`'s history |
| `05_tests/` | the kernel tests (`test_varlen_attention.cu`, `test_rope_positions.cu`, `test_loss_grad_inplace.cu`, `sanitize_vj2.cu`) and the 2026-10-24 test logs |
| `06_docs/` | the full documents: `kernel_diff_sft_pretraining.md`, `sft_packing_algorithms_bfd_ffd.md`, `loss_calculation_in_pretraining_and_sft.md`, `How_learning_happens_in_sft.md`, the full work log `sft_work_documentation.md`, and more |
| `07_generic_attention_2026_10_24/` | the 2026-10-24 work on attention for lower GPUs: original files, new files, diffs, logs, the H200 safety proof |

---

## 1. SFT in simple words

**Pretraining** teaches the model to continue any text. The training data is one long stream of text, cut into
rows of T = 4,096 tokens. Every token in a row is a target: the model is graded on predicting every next token.

**SFT** teaches the model to answer like an assistant. The data is chats: a user message, then an assistant answer
(sometimes several turns). Four things are different from pretraining:

1. **Only the answer is graded.** The model should learn to write answers, not to write user questions. So every
   token that is not part of an answer gets the label 65535, which means "no loss, no gradient for this token".
2. **The loss is one average over the whole step.** We add up the losses of all graded tokens in the step (all
   micro-batches, all GPUs) and divide once by the number of graded tokens in that whole step. Then every graded
   token has the same weight, whether its chat was long or short.
3. **Several chats share one row (packing), and they must not see each other.** A chat is usually much shorter than
   4,096 tokens. Putting one chat per row and filling the rest with padding wastes most of the GPU's work. So we put
   several whole chats into one row. But then plain causal attention would let chat 2 read chat 1's tokens. That is
   wrong: chat 2 must behave as if it were alone. So attention is told where each chat starts and ends, and each chat
   attends only to its own earlier tokens.
4. **Each chat's positions start at 0.** RoPE tells the model where each token sits. In a packed row, chat 2 starts
   at, for example, token 1,000 of the row. Inside its chat, its first token must still be position 0, exactly as if
   the chat were alone. So RoPE gets a per-token position list that restarts at 0 for every chat.

Worked example, one packed row, T = 4,096:

```
row tokens:  [ chat 1: 1,000 tokens ][ chat 2: 2,500 tokens ][ pad tail: 596 ]
positions:     0 1 2 ... 999           0 1 2 ... 2499          0 1 ... 595
attention:   chat 1 sees only chat 1;  chat 2 sees only chat 2;  pad tail sees only itself
labels:      65535 on prompts and pad, the real next token on answers (and on the end token after each answer)
```

**Only three kernels change for SFT: RoPE, attention, and the loss.** Every other kernel (matmuls, RMSNorm, SwiGLU,
embedding, optimizer) runs exactly the pretraining code. Proof and the comparison with 10 other libraries:
`06_docs/kernel_diff_sft_pretraining.md` sections 0-2 and 6.

---

## 2. The journey of one chat, start to end

```
Hugging Face dataset (e.g. HuggingFaceH4/ultrachat_200k)
   |
   |  01_preprocessing/prepare_sft_data.py
   |    - chat template "tulu-plain-v1", tokenize each piece on its own
   |    - labels: answer tokens + the end token after each answer; everything else 65535
   |    - pack whole chats into rows with First-Fit-Decreasing (--pack ffd)
   v
files: inputs.bin, labels.bin, valid_counts.bin, seq_lengths.bin, seq_offsets.bin, meta.json
   |
   |  02_dataloader/SftDataLoader.h   (one micro-batch = B rows)
   |    - copies the B rows to the GPU (inputs and labels)
   |    - builds positions [B, T] (restart at 0 per chat) and the segment list (row, start, length)
   v
03_trainer/bluscript_sft.cpp   (BLU_SFT_DATA=<dir> turns SFT mode on)
   |    - g_packed = { positions, chat plan }   built once per micro-batch
   |    - per layer: RMSNorm -> wQ/wK/wV -> fused QK-norm + RoPE (with positions)
   |                 -> attention (with the chat plan) -> wO -> SwiGLU MLP
   |    - lm_head -> loss (ignore 65535, divide by the step's global graded-token count)
   |    - backward: the same plan and positions are used again
   v
optimizer step (Muon / AdamW, ZeRO), exactly as in pretraining
```

---

## 3. Step 1 — preparing the data (`01_preprocessing/prepare_sft_data.py`, 397 lines)

**What it does, in order:**

1. **Reads a chat dataset** (default `HuggingFaceH4/ultrachat_200k`, pinned revision, arguments at lines 207-225).
2. **Writes each chat with the template "tulu-plain-v1"** (lines 15-22):
   `<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...`
   The role markers are plain text. After every answer comes the real end token `<|endoftext|>` (id 50256).
   Why this template: it is Tülu 3's template minus one newline between turns (decision 2026-10-15,
   `06_docs/sft_work_documentation.md` §45.6 and the decision record).
3. **Tokenizes each piece separately** (`tokenize_conversation`, line 47). A marker and the text next to it are never
   merged into one BPE token, so the model always sees the markers the same way.
4. **Marks what is trained** (lines 62-77): answer tokens and the end token after each answer are trained; markers,
   system and user text are not. Every assistant turn is trained, not only the last one.
5. **Shifts by one** (`build_row`, line 79): `inputs = ids[:-1]`, `labels = ids[1:]`, and every untrained position
   gets label 65535 (`IGNORE_ID`, line 41). Padding uses id 50256 (`PAD_ID`, line 40).
6. **Packs** (with `--pack ffd`, lines 279-317): every chat is labelled on its own first, so no target ever crosses into
   the next chat. Then whole chats are packed into rows with **First-Fit-Decreasing** (`ffd_pack`, line 91): sort the
   chats from longest to shortest, put each one into the first row that still has room. The rows are then shuffled.
   Chats longer than T are dropped (splitting them is a future task, work log §45.4).
7. **Checks everything** (lines 309-326): every row's chat lengths match, the pad tail has no trained label, every
   label is a real token or 65535.
8. **Writes the files** (lines 328-373, formats at lines 3-14):

| File | Type | Meaning |
|---|---|---|
| `inputs.bin` | uint16 [N, T] | token ids the model reads, padded with 50256 |
| `labels.bin` | uint16 [N, T] | the target at each position; 65535 = not trained |
| `valid_counts.bin` | uint32 [N] | number of trained labels in each row (for the global denominator) |
| `seq_lengths.bin` | int32 [all chats] | packed only: the lengths of the chats of every row, row after row (TRL's `seq_lengths`) |
| `seq_offsets.bin` | int64 [N + 1] | packed only: row r's chats are `seq_lengths[seq_offsets[r] : seq_offsets[r+1]]`; the rest of the row is its pad tail |
| `meta.json` | text | settings, dataset revision, tool versions, counts, why rows were dropped |

**Why First-Fit-Decreasing and not Best-Fit-Decreasing** (`06_docs/sft_packing_algorithms_bfd_ffd.md` section 13):
both sort the chats longest first. First-Fit takes the first row with room. Best-Fit searches all rows for the
tightest fit. On our data both give exactly the same number of rows (stage-0 data: 5,000 chats -> 1,517 rows each;
the smallest possible is 1,511, so 0.4% above). In thousands of random tests from our chat lengths there was no
difference. Both have the same proven worst case (11/9 of the minimum). With a search tree both pack 1,000,000 chats
in under 1 second; First-Fit is 1.3-1.9 times faster. So either is fine; we use First-Fit-Decreasing (simpler, a little
faster). TRL uses Best-Fit-Decreasing; Hermes 4 uses First-Fit-Decreasing.

**Why packing and not padding:** with one chat per row, the 5,000 stage-0 chats need 5,000 rows, mostly padding. Packed,
they need 1,517 rows: about 3.3 times fewer rows for the same chats, so far fewer GPU steps for the same data.
Measured gains and the step-count effect are in work log §38 (Q10) and the packing document sections 8-9.

**Run it:**
```
python3 sft/prepare_sft_data.py --out <dir>/train --split train_sft --num 5000 --pack ffd
python3 sft/prepare_sft_data.py --out <dir>/val   --split test_sft  --num 500  --pack ffd
python3 sft/prepare_sft_data.py --self-test
```

---

## 4. Step 2 — reading the data during training (`02_dataloader/SftDataLoader.h`, 264 lines)

The file's header (lines 1-30) lists everything. In simple words:

- **It reads the files once and checks them** (constructor from line 46; checks at lines 190-220). A bad file stops
  the run, because a bad file would otherwise train silently on wrong targets.
- **Each call to `next_batch()` (line 102)** returns B rows: inputs and labels copied to the GPU. It is the same
  `Batch` struct as the pretraining loader, so the trainer's forward and loss code does not change.
- **The three differences from the pretraining loader** (lines 19-23): the target comes from `labels.bin` (not the
  input shifted by one); whole rows (not a window over one text stream); it reports the trained-token count of each
  micro-batch (for the global denominator).
- **Packed rows** (when `seq_lengths.bin` exists, `packed()` at line 162): for every batch it also builds
  - `last_positions()` (line 163): int32 [B, T], each token's index inside its chat (restarts at 0 per chat);
  - `last_segment_rows/starts/lengths()` (lines 164-166): the segment list of the B rows — every chat, and then the
    row's pad tail as a segment of its own (line 137). Together they cover every token of every row exactly once.
    This list is the input of the attention plan.
- **Ranks never share a row:** micro-batch m of rank r is rows `[(m*W + r)*B, +B)` (line 24).
- The old per-token format (`seg_start.bin`) is refused (line 16), so old data cannot train with chats seeing each other.

---

## 5. Step 3 — the SFT trainer (`03_trainer/bluscript_sft.cpp`)

`bluscript_sft.cpp` is the pretraining trainer `bluscript_zero_ckpt.cpp` plus the SFT parts. The full difference is
in `03_trainer/pretraining_to_sft_trainer.diff` (+313 / -35 lines, 24 places).

**What turns SFT on:** one setting, `BLU_SFT_DATA=<dir>` (line 994: `const bool sft = !sft_dir.empty();`). Without it
the program trains exactly like pretraining. With it:

| Setting | Default | Meaning |
|---|---|---|
| `BLU_SFT_DATA` | (empty = pretraining) | folder with `train/` and `val/` from `prepare_sft_data.py` |
| `BLU_SFT_B` | 8 | rows per micro-batch per GPU |
| `BLU_SFT_ROWS_PER_STEP` | 64 | rows per optimizer step (all micro-batches, all GPUs) |
| `BLU_SFT_EPOCHS` | 2 | passes over the train set; the number of steps is computed from it |
| `BLU_SFT_LR`, `BLU_SFT_MIN_LR` | 2e-5, lr/10 | AdamW learning rate schedule |
| `BLU_SFT_MUON_LR`, `BLU_SFT_MUON_MIN_LR` | must be set for Muon | Muon learning rates |
| `BLU_SFT_WARMUP` | 0.03 | warm-up as a fraction of all steps |
| `BLU_SFT_VAL_FREQ` | 10 | validation every N steps |
| `BLU_INIT_FROM` | — | start from another run's checkpoint (weights only) |
| `BLU_N_LAYERS`, `BLU_ROPE_THETA` | 26, 500000 | build the model the checkpoint was trained as |

Packed rows are used automatically when the data folder has `seq_lengths.bin`. They need the trainer built with
`-DUSE_FUSED_RMS_ROPE=1` (the sequential RoPE path cannot take positions; it stops with a clear error, line 672).

**The changes, place by place** (line numbers in `bluscript_sft.cpp`):

| Where | What | Why |
|---|---|---|
| 79-80, 91 | includes: checkpoint metadata + CRC32; `SftDataLoader.h` | weight loading check; the SFT loader |
| 104-131 | `PackedRows g_packed` and `packed_rows_of()` | builds, once per micro-batch, the positions and the chat plan (`gqa_chat_plan_make`) from the loader's segment lists |
| 641-668 | the attention block calls `fused_rms_rope_bf16(..., g_packed.positions)` | RoPE positions restart at 0 for every chat |
| 672-674 | stop if packed rows meet the sequential RoPE path | that path has no positions; it would silently be wrong |
| 687-690 | `scaled_dot_product_attention(..., g_packed.chats)` | each chat attends only to itself; null = plain causal (pretraining) |
| 896-903 | `BLU_N_LAYERS`, `BLU_ROPE_THETA` | the model must match the checkpoint it starts from |
| 970-1057 | SFT mode block: settings, loaders, step count | turns on SFT; the pretraining data mixture is never opened |
| 1163-1240 | `BLU_INIT_FROM`: load weights only (+ CRC32 check) | SFT starts from a pretrained model with a fresh optimizer |
| ~1944-1980 | the pretraining loader is not created in SFT mode | it would need the pretraining shards |
| ~1994-2020 | log file name and the **backward seed** | pretraining seeds 1/GRAD_ACCUM; SFT seeds world_size, because each micro-loss is already divided by the step's global count and ZeRO averages over GPUs |
| ~2257-2270 | resume: skip exactly the used micro-batches | a restarted SFT run continues with the right rows |
| ~2289-2330 | SFT validation: all val losses added, divided once by the val set's trained-token count | the same rule as training |
| ~2415-2440 | **the global denominator**: `valid_tokens_ahead(GRAD_ACCUM)`, all-reduced over GPUs, into `sft_denom`; loss call with ignore index 65535 and `&sft_denom` | every trained token of the step has the same weight (torchtitan's `global_valid_tokens`) |
| ~2483-2500 | loss logging: micro-losses just add up in SFT | dividing again would be wrong |
| ~2890-2896 | free `g_packed` before exit | it is a global; freeing it after the GPU allocator is gone hung the process at exit (found 2026-10-15) |

Why each loss rule was chosen (mean per micro-step vs over the whole step vs sum loss):
`06_docs/loss_calculation_in_pretraining_and_sft.md` and the hyperparameter doc sections 13-15.

---

## 6. Step 4 — the library (kernel) changes

### 6.1 Loss — ignore index and the global denominator

Two parts:
- **Pushed on 2026-09-25** (`04_library/committed_ad10e7d_...patch`): `sparse_cross_entropy_loss` got
  `ignore_index` and `denom`, an fp32 loss scalar, and dtype-aware vector loads (bugs A/B/C).
- **Later (uncommitted, in `diffs_vs_TI_HEAD_d96b991/`)**: the gradient is written over the logits (no second
  rows x vocab buffer; bit-identical), 16-byte loads, and the library's CUDA type mapping.

The API (`04_library/current/include/autograd/operations/LossOps.h`, lines 29-50):
- `ignore_index = -1` (default) = no masking, pretraining unchanged; SFT passes 65535.
- `denom = nullptr` (default) = divide by the batch size as before; SFT passes a 1-element GPU tensor with the step's
  global trained-token count. It stays on the GPU, so there is no CPU wait.
Kernels: `src/Kernels/cuda/loss/LossKernels.cu` (ignore index from line 167; denominator 243-271; in-place backward
764, 903, 1154); backward wiring `src/autograd/backward/LossBackward.cpp` (192-216, 355-370).

Why: a token with label 65535 must give 0 loss and 0 gradient; the denominator must be the whole step's count (a
per-micro-step mean gives short chats more weight). Work log §22-23 and §25-30.

### 6.2 RoPE — per-chat positions

- **API**: `fused_rms_rope(..., positions)` and `fused_rms_rope_bf16(..., positions)`
  (`include/autograd/operations/NormalizationOps.h`, lines 36-70). `positions` is an optional int32 [B, T] GPU tensor.
  Empty = pretraining (position = t + pos_offset). It is saved for the backward, which must rotate by the same positions.
- **The bf16 kernel used by training**: `src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu` (new file; Kruthik's fused bf16
  RMSNorm + RoPE with our per-chat positions, then optimized 2026-10-22..24: cp.async backward, 2-4x faster than the
  original, bit-identical, fastest of all rivals in 16/16 cases; positions at lines 77-221).
- **The fp32 Hopper kernels** (`src/Kernels/cuda/norm/arch/FusedRMSRoPE_*_sm90*.cu`) and the CPU reference
  (`src/UnaryOps/cpu/Normalizations.cpp`, 996-1003) take the same positions.
- **Wiring**: `src/autograd/operations/NormalizationOps.cpp` (97-161) and `src/autograd/backward/NormalizationBackward.cpp` (70-97).

Why: RoPE encodes "where is this token". Inside a packed row a chat must see its own positions 0, 1, 2, ... exactly as
if it were alone, or the model learns wrong position patterns. TRL builds `position_ids` the same way.
Cost and decision: `06_docs/kernel_diff_sft_pretraining.md` section 8.

### 6.3 Attention — the chat plan

**The plan** (`include/ops/helpers/AttentionKernels.h`, `struct GqaChatPlan` at line 125, builder at 146-152):
one GPU memory block holds several lists, all built from the same segment list (row, start, length):

| List | Used by | One entry |
|---|---|---|
| `fwd_units` | H200 forward (v62) | {row, start, length, couple+head} |
| `bwd_tiles` | H200 backward Vj1d | one 64-key tile of a chat |
| `vj2_blocks` + `dq_tiles` | H200 backward Vj2 (default) | one 128-key block of a chat for one head; padded dQ slots |
| `gen_segs` (added 2026-10-24) | every other GPU (generic kernels) | {row, start, length, 0} per chat |

The builder (`src/Kernels/cuda/attention/arch/GQA_fwd_generic.cu`, `gqa_chat_plan_build` at line 494) checks that the
segments cover every token of every row exactly once (a gap or overlap stops the run), builds all lists, and uploads
them in ONE asynchronous copy. The trainer builds the plan once per micro-batch; every layer's forward AND backward use
the same plan (the backward node keeps a shared pointer to it: `src/autograd/backward/AttentionBackward.cpp` line 209;
used at 249).

**How a kernel uses it (all kernels, same rule):** a block of work is (chat, head, tile counted from the CHAT'S start).
The loop over keys stops at the tile's last row (causal), and keys before the chat's start are never visited. Because
tiles are counted from the chat's start, a chat inside a packed row runs exactly the same tiles in exactly the same
order as the same chat alone in its own row. So the result is **bit-for-bit the same** — every packed test checks this.

**Which kernel runs where** (dispatch in `GQA_fwd_generic.cu` line 372 and `GQA_bwd_generic.cu` line 785):

| GPU | Forward | Backward | Chats (SFT) |
|---|---|---|---|
| H100 / H200, causal, head size 128 | v62 (`GQA_fwd_sm90.cu`, chat units at lines 1302-1326) | Vj2 by default (`GQA_bwd_vj2_sm90.cu`, CHATS template at 438, prep at 380); Vj1d with `USE_VJ2=0` (`GQA_bwd_vj1d_sm90.cu`, MODE_CHATS at 419-617) | yes |
| any other GPU except B300 (Ada sm_89, RTX 3060 sm_86, A100...), and Hopper calls v62/Vj2 cannot take | generic `gqa_v2<CAUSAL>` (`GQA_fwd_generic.cu` line 74; mask at 168) | generic `gqa_backward_dQ` / `gqa_backward_dKdV` (`GQA_bwd_generic.cu` lines 210, 387) | yes (since 2026-10-24) |
| B300 | head size 64 only (unchanged) | head size 64 only | no (refused) |

`USE_VJ2=0` is a run-time environment variable (put it before the command, or `-x USE_VJ2` with mpirun); it is read
once per process (`gqa_bwd_mode`, `GQA_bwd_generic.cu` line 116).

**Why a per-chat list (design A, "cu_seqlens") and not one number per token (design B, "seg_start")**
(work log §45.7, measured 2026-10-10/12):

| | design B (removed 2026-10-15) | design A (kept) |
|---|---|---|
| what the kernel gets | one number per token: where its chat starts | a list of chats per row: (row, start, length) |
| how work is cut | fixed row tiles; a tile on a chat border mixes two chats and needs an extra mask | per chat; every chat starts on its own tiles |
| who uses it | FlexAttention document mask; HF SDPA/eager | FlashAttention 2/3/4, TransformerEngine/Megatron, PyTorch varlen_attn, HF flash path, TRL |
| speed (our kernels) | — | forward 2-16% faster; backward 3-5% faster on middle and long chats |
| packed == alone bit-exact | not for chats that start inside a tile | yes, for every chat |

**Lower GPUs (2026-10-24):** before this change, any GPU other than H100/H200 had NO correct causal kernel for our
model: head size 128 stopped with an error, head size 64 silently ran full attention (no mask). The generic kernels
(written 2026-06, WMMA, one warp per block, no TMA) now have the causal mask and the chat plan. They are correct but
NOT optimized yet (0.9 TFLOP/s on the RTX 3060). The H200 code is untouched: the six Hopper files are byte-identical to
lottn and do not include the changed header (`07_generic_attention_2026_10_24/h200_safety_proof/`). Full record:
work log §45.92.

---

## 7. Tests and results

| What | Where | Result |
|---|---|---|
| Data script self-test | `prepare_sft_data.py --self-test` | First-Fit fast == plain loop; labels and packing checked |
| Data loader | `02_dataloader/test_sft_dataloader.cpp` | positions, segments, checks |
| Loss (ignore index, denominator, in-place gradient) | `05_tests/test_loss_grad_inplace.cu`; work log §26, §45.36-38 | 136 cases 0 failures; in-place gradient bit-identical, -5,926 MiB |
| RoPE positions | `05_tests/test_rope_positions.cu`; RoPE port test (work log §45.82-45.90) | 44/44 bit-identical on H200 and RTX 3060 |
| Attention, H200 (v62 + Vj2/Vj1d) | `05_tests/test_varlen_attention.cu` G1-G13 (work log §40-45) | all pass; packed == alone bit-exact |
| Attention, RTX 3060, generic, head size 128 | `05_tests/logs/run2_quick_hd128_rtx3060.log` | 292 passed, 0 failed, 1 known |
| Attention, RTX 3060, generic, head size 64 | `05_tests/logs/run4_quick_hd64_rtx3060.log` | 291 passed, 1 failed (the dQ case in section 9); packed == alone bit-exact; repeats bit-identical |
| Real SFT run (Tülu 3 data, 4x H200) | work log §45.67, §45.71-45.73 | finished 2026-10-23 at step 11,590, val loss 0.8501; checkpoints evaluated and handed to the DPO team |

How the attention test decides "correct": it compares against an fp64 reference and a bf16 "floor" reference built
from the same inputs, and passes when the kernel's error is at most 2 x the floor's error (FlashAttention's own test
rule). It also checks that every packed chat equals the same chat alone, bit for bit.

---

## 8. How to build and run (H200 box)

```
# library + trainer
make -j$(nproc) profiler tensor dist SM_ARCH=90
mpic++ ... -DUSE_FUSED_RMS_ROPE=1 ... -o bluscript_sft_bin bluscript_sft.cpp ...   (full line: work log §45.90)

# SFT run (packed data, start from a pretrained checkpoint)
BLU_SFT_DATA=<dir> BLU_INIT_FROM=<checkpoint> BLU_N_LAYERS=<layers of that checkpoint> BLU_ROPE_THETA=<its theta> \
BLU_ALL_BF16=1 BLU_OPTIMIZER=muon BLU_SFT_LR=5e-5 BLU_SFT_MUON_LR=5e-5 BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=64 \
mpirun -np <GPUs> -x BLU_SFT_DATA -x BLU_INIT_FROM ... ./bluscript_sft_bin
```

---

## 9. Known limits and open items

- **Chats longer than T are dropped** (splitting them like TRL's `bfd_split` is a future task, work log §45.4).
- **lm_head on trained tokens only** (TRL does this; same math, faster): future task Q11, work log §33.
- **Generic attention is correct but slow** (0.9 TFLOP/s on the RTX 3060). Today it uses NO cp.async: its sm_86
  machine code has 0 `LDGSTS` (cp.async) instructions, 174 / 252 plain `LDG` loads (forward / backward), 0 `ldmatrix`,
  and one warp per block. cp.async exists on every GPU from sm_80 (A100, RTX 30xx / A6000 sm_86, Ada sm_89, Hopper);
  only TMA needs sm_90+. The generic kernel already needs sm_80+ (bf16 WMMA), so cp.async adds no new limit. Our RoPE
  kernel already uses cp.async (204 `LDGSTS`) and passed 44/44 on the RTX 3060. Next step: a FlashAttention-2-style
  rewrite for Ampere and Ada (cp.async double-buffered K/V loads, `ldmatrix` + `mma.sync` instead of WMMA, 4 warps),
  and fixing the dK/dV register spills at head size 128.
- **Generic dQ uses the bf16-rounded P to compute dS** (like Vj1d did). This gives one known case at head size 128
  (1.049x the error limit, same chat and ratio as recorded for Vj1d on 2026-10-10) and one at head size 64 (1.175x,
  `row 0 @2688 L=64`). Packed == alone is still bit-exact there, so it is not caused by packing. Fix: compute dS from
  the fp32 P (as Vj2 and FA3 do); P is used only for dS in the dQ kernel, so this is a small change.
- **B300**: SFT packed rows are not supported (causal attention there is head size 64 only).
- **Matmul fusions for pretraining** (separate topic, first task on 2026-10-26): work log §45.93.

---

## 10. Where to read more

| Question | Read |
|---|---|
| What is SFT, how does it differ from pretraining, what the libraries do | `06_docs/sft_work_documentation.md` §1-23; `How_learning_happens_in_sft.md` |
| Which kernels change and why, with proof from 10 libraries | `06_docs/kernel_diff_sft_pretraining.md` |
| Packing: First-Fit vs Best-Fit, cross-chat attention, cu_seqlens, step counts | `06_docs/sft_packing_algorithms_bfd_ffd.md` |
| How the loss is averaged (micro-step mean vs step mean vs sum) | `06_docs/loss_calculation_in_pretraining_and_sft.md`; hyperparameter doc §13-15 |
| The design A vs B decision, the packed pipeline | work log §41.3, §45.5, §45.7 |
| The attention kernels on H200 (v62, Vj2) and the benchmarks vs FA3/FA4/cuDNN | work log §42-45.61 |
| The Tülu 3 SFT run and its evaluation | work log §45.67, §45.71-45.74; `overfitting_memorization_...md` |
| Lower-GPU attention (2026-10-24) | work log §45.92; `07_generic_attention_2026_10_24/` |
