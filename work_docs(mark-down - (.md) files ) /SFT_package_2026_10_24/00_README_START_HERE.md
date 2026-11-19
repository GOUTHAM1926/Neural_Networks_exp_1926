# SFT in BluTrain — everything in one place (package updated 2026-11-18)

This package holds every file we changed or added to make SFT (supervised fine-tuning) work in BluTrain: the data
preparation script, the data loader, the SFT trainer, every library (kernel) change, the tests, and the documents that
explain each decision.

- **Code in this version:** the 2026-11-18 push of branch `dist_side_gautam_lottn`: BluTrain `c52ca81`, Tensor-Implementations `334b244` (BluBridge-BLAS `cfa0e92`).
- **The package of 2026-10-24** (its start page and every code copy its line numbers point into) is kept unchanged in
  `_old_2026_10_24/`. Nothing was deleted.
- **The full, current guide** is `06_docs/README_sft_BluTrain.md`, a copy of BluTrain's `sft/README.md`. This page is the
  map of the package and the list of what changed since 2026-10-24.

---

## 0. What is in this package

| Folder | What it holds |
| --- | --- |
| `01_preprocessing/` | `prepare_sft_data.py` (a chat dataset -> SFT files: chat template, labels, chats longer than the row dropped, packing) and `mixes/` (the 12 data mixes) |
| `02_dataloader/` | `SftDataLoader.h` (reads the SFT files during training) and its test `test_sft_dataloader.cpp` |
| `03_trainer/` | `bluscript_sft.cpp` (the SFT trainer); `bluscript_zero_ckpt_PRETRAINING_d9606c2.cpp` (the pretraining trainer of the same branch) and the full diff between them (`pretraining_to_sft_trainer.diff`); `run_sft.sh`, `build_sft_trainer.sh`; `ConfigBridge.h` and the production YAMLs; the checkpoint tools (`hf_to_blutrain.py`, `blutrain_to_hf.py`, `export_to_hf.py`, `verify_export.py`, `ref_val_loss.py`) |
| `04_library/current/` | 35 Tensor-Implementations (TI) files at their TI paths: the 27 files of the 2026-10-24 package, now in their current version, and 8 files changed for SFT since then (`ReshapeOps.cpp`, `RoPEKernels.cu`, `RoPE_sm90.cu`, `EmbeddingKernels.cu`, `ContiguousKernel.cu`, `ContiguousKernel_sm90.cu`, `SwiGLUKernels_sm120.cu`, `Makefile`) |
| `04_library/diffs_vs_TI_d96b991/` | one diff per file against TI `d96b991` (the base of the 2026-10-24 package). These are whole-file diffs: a few files (for example the `Makefile`) also hold changes that are not about SFT |
| `04_library/committed_ad10e7d_...patch` | the loss change pushed on 2026-09-25 (ignore index + global denominator), unchanged |
| `05_tests/` | `test_varlen_attention.cu`, `test_rope_positions.cu`, `test_sparse_ce_ignore_index.cu`, `test_contig_alias.cu`, `sanitize_vj2.cu` |
| `06_docs/` | `README_sft_BluTrain.md` (the guide), the BluTrain `sft/docs/` set (kernel diff, packing, loss, learning, optimizer choice, kernel benchmarks + its 2 figures), `README_SFT_checkpoint_for_DPO.md`, the work log (`sft_work_documentation.md`), the plan (`plan_grid_search_sft.md`), how to run SFT (`how_to_handle_sft.md`), the context-parallel docs, the chat-template and chat-format studies, the hyperparameter, overfitting and learning-rate docs, and the attention write-up of 2026-11-18 |
| `all_sft_optimizations.md` | every SFT optimization from the first day (2026-09-17) to 2026-11-18: context, before, change, after, how measured, numbers, sources; model and matrix shapes, a map and a timeline |
| `07_generic_attention_2026_10_24/` | the 2026-10-24 work on attention for lower GPUs (unchanged) |
| `_old_2026_10_24/` | the package of 2026-10-24, unchanged |
| `MANIFEST_md5.txt` | the md5 of every file in this package |

---

## 1. Where to start

- **Every optimization made for SFT, with before / after numbers:** `all_sft_optimizations.md` (this folder).
- **What SFT is and how it works here:** `06_docs/README_sft_BluTrain.md` sections 1-2 (SFT in simple words, the journey
  of one chat), then 3 (data), 4 (loader), 5 (trainer), 6 (kernel changes), 7 (tests and results), 8 (build and run),
  9 (known limits), 11 (row length and the checkpoint's context length).
- **To run an SFT training:** `06_docs/how_to_handle_sft.md` (access, clone, build, prepare, run).
- **Why each kernel changes and pretraining does not:** `06_docs/kernel_diff_sft_pretraining.md`.
- **Every test and decision, in order:** `06_docs/sft_work_documentation.md` (work log; section 45 is the SFT series).

---

## 2. What changed since the package of 2026-10-24

The dates and section numbers are those of `06_docs/README_sft_BluTrain.md`, which describes each change in full.

| Date | Change | Where |
| --- | --- | --- |
| 2026-10-27/28 | The 135M base model: its tokenizer and chat templates; SFT pushed to the branch `dist_side_gautam_lottn` | README 3.1 |
| 2026-10-30 | `prepare_sft_data.py --workers N` (parallel preparation, same output as one worker) | README 3.3 |
| 2026-10-30 | The loss sum counted only the first 65,536 rows of a loss call: fixed (pretraining was affected too) | README 6.1, 7 |
| 2026-10-31 | A chat longer than the row is dropped: the only method (the split built on 10-30 was removed) | README 3.2 |
| 2026-10-31 | Long context, 16k-128k: RoPE tables, loss fallback backward, `contiguous()` past 2^31 elements, embedding, SwiGLU on sm120; optional fp32 dK / dV partials (`BLU_VJ2_DKDV_F32=1`); exact math everywhere except the loss | README 6.4 |
| 2026-11-03 to 11-05 | Speed changes both trainers got: Muon's Newton-Schulz batched and decided on the GPU, fewer copies / zeroings / launches / CPU waits per step, GEMM tile tables built on the GPU (all together: SFT 135M -13.7%, 1.6B -1.7%); BluBLAS GEMM / addmm fixes | README 5.1, 7 |
| 2026-11-06/07 | Context parallelism: one row shared by several GPUs (`BLU_SFT_CP=U`) | README 5.2 |
| 2026-11-07 | `autograd::contiguous` returns an already-contiguous input as is (it had silently dropped k's gradient in the fused QK-norm + RoPE op) | README 6.5 |
| 2026-11-13 | Attention forward: "mixed units" for chats with an odd number of 64-row tiles, with the 2% rule | README 6.3 |
| 2026-11-18 | Attention backward: ideas 1, 2 and 4 (skip the fully masked half of a block's first step; write a chat's first two dQ tiles directly; add the query heads' dK / dV inside one block), with two rules; tested against FA3 / FA4 / cuDNN and in the SFT and pretraining trainers | README 6.3, 7; `sft_kernel_benchmarks.md` section 8 |

What the 2026-10-24 start page (`_old_2026_10_24/00_README_START_HERE.md`) says that is no longer true:

- it describes the code of the laptop tree `~/Downloads/BluTrain_sft` of 2026-10-24, before the branch above; its line
  numbers point into the copies in `_old_2026_10_24/`, not into the files of this version;
- it predates every change in the table above (for example: the tokenizer was GPT-2's, the Hopper attention had
  neither the mixed units nor the backward ideas, and there was no context parallelism).

---

## 3. Build, run, test

- Build and run on the branch: `06_docs/README_sft_BluTrain.md` section 8, and `06_docs/how_to_handle_sft.md`.
- Tests and their results: `06_docs/README_sft_BluTrain.md` section 7. The test programs are in `05_tests/`; in the TI
  tree they are in `Tests/kernel_tests/`.
- The files in `01`-`05` are copies. The source of truth is the branch: clone it (see `how_to_handle_sft.md`) to build.
