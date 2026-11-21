# SFT in BluTrain — everything in one place (package updated 2026-11-20)

This package holds every file we changed or added to make SFT (supervised fine-tuning) work in BluTrain: the data
preparation script, the data loader, the SFT trainer, every library (kernel) change, the tests, and the documents that
explain each decision.

- **Code in this version:** the 2026-11-20 push of branch `dist_side_gautam_lottn`: BluTrain `651f9c4`, Tensor-Implementations `dd5b61a` (BluBridge-BLAS `cfa0e92`, unchanged). The version of 2026-11-18 was BluTrain `c52ca81`, Tensor-Implementations `334b244`.
  Three docs in `06_docs/` are one step ahead of `651f9c4`: `README_sft_BluTrain.md` (sections 7, 9, 10),
  `kernel_diff_sft_pretraining.md` (section "2026-11-20") and `sft_kernel_benchmarks.md` (section 9) hold the
  sliding-window additions of 2026-11-20 evening, which go to BluTrain in the next docs commit.
  *Pushed 2026-11-20 ~20:40 IST: BluTrain `651f9c4` -> `bfc712b` (docs only: these 3 files, +59 lines); checked on
  GitHub file by file, md5-equal to the copies here. So the docs are at BluTrain `bfc712b`, the code at `651f9c4` / TI
  `dd5b61a` (unchanged).*
  *2026-11-20 ~23:20: `06_docs/sft_kernel_benchmarks.md` is one step ahead of `bfc712b` again (section 9: the window race
  against FA3 / FA4 / FA2 / FlexAttention, +11 lines); it goes to BluTrain with the next docs push.*
- **The package of 2026-10-24** (its start page and every code copy its line numbers point into) is kept unchanged in
  `_old_2026_10_24/`. Nothing was deleted.
- **The copies replaced on 2026-11-20** (the code of the 2026-11-18 push that changed since: 9 TI files and their diffs,
  5 trainer files, the pretraining trainer of `d9606c2` and the trainer diff, the guide, the attention write-up of
  2026-11-18, and the start page of 2026-11-18) are kept unchanged in `_old_2026_11_18/`. The work log and the plan only
  grow: their 2026-11-18 text is unchanged at the start of the new copies (checked byte for byte).
- **The full, current guide** is `06_docs/README_sft_BluTrain.md`, a copy of BluTrain's `sft/README.md`. This page is the
  map of the package and the list of what changed since 2026-10-24.

---

## 0. What is in this package

| Folder | What it holds |
| --- | --- |
| `01_preprocessing/` | `prepare_sft_data.py` (a chat dataset -> SFT files: chat template, labels, chats longer than the row dropped, packing) and `mixes/` (the 12 data mixes) |
| `02_dataloader/` | `SftDataLoader.h` (reads the SFT files during training) and its test `test_sft_dataloader.cpp` |
| `03_trainer/` | `bluscript_sft.cpp` (the SFT trainer); `bluscript_zero_ckpt_PRETRAINING_651f9c4.cpp` (the pretraining trainer of the same branch; until 2026-11-18 the `d9606c2` copy, now in `_old_2026_11_18/`) and the full diff between them (`pretraining_to_sft_trainer.diff`); `run_sft.sh`, `build_sft_trainer.sh`; `ConfigBridge.h` and the production YAMLs; the checkpoint tools (`hf_to_blutrain.py`, `blutrain_to_hf.py`, `export_to_hf.py`, `verify_export.py`, `ref_val_loss.py`) |
| `04_library/current/` | 35 Tensor-Implementations (TI) files at their TI paths: the 27 files of the 2026-10-24 package, now in their current version, and 8 files changed for SFT since then (`ReshapeOps.cpp`, `RoPEKernels.cu`, `RoPE_sm90.cu`, `EmbeddingKernels.cu`, `ContiguousKernel.cu`, `ContiguousKernel_sm90.cu`, `SwiGLUKernels_sm120.cu`, `Makefile`). On 2026-11-20, 9 of them changed for the sliding window (`AttentionKernels.h`, `AttentionOps.h` / `.cpp`, `AttentionBackward.h` / `.cpp`, `GQA_fwd_sm90.cu`, `GQA_bwd_vj2_sm90.cu`, `GQA_fwd_generic.cu`, `GQA_bwd_generic.cu`); all 35 are at TI `dd5b61a` |
| `04_library/diffs_vs_TI_d96b991/` | one diff per file against TI `d96b991` (the base of the 2026-10-24 package). These are whole-file diffs: a few files (for example the `Makefile`) also hold changes that are not about SFT |
| `04_library/committed_ad10e7d_...patch` | the loss change pushed on 2026-09-25 (ignore index + global denominator), unchanged |
| `05_tests/` | `test_varlen_attention.cu`, `test_rope_positions.cu`, `test_sparse_ce_ignore_index.cu`, `test_contig_alias.cu`, `sanitize_vj2.cu`; since 2026-11-20 the sliding-window tests `test_swa_attention.cu` (Navin's), `test_swa_extra.cu` and `test_swa_packed.cu` |
| `06_docs/` | `README_sft_BluTrain.md` (the guide), the BluTrain `sft/docs/` set (kernel diff, packing, loss, learning, optimizer choice, kernel benchmarks + its 2 figures), `README_SFT_checkpoint_for_DPO.md`, the work log (`sft_work_documentation.md`), the plan (`plan_grid_search_sft.md`), how to run SFT (`how_to_handle_sft.md`), the context-parallel docs, the chat-template and chat-format studies, the hyperparameter, overfitting and learning-rate docs, and the attention write-up of 2026-11-18; since 2026-11-20 also the sliding-window analysis (`sliding_window_attention_sft_analysis.md`) and the attention explainer (`attention_explanation.md`) |
| `all_sft_optimizations.md` | every SFT optimization from the first day (2026-09-17) to 2026-11-20 (section 4.17: the sliding window): context, before, change, after, how measured, numbers, sources; model and matrix shapes, a map and a timeline |
| `07_generic_attention_2026_10_24/` | the 2026-10-24 work on attention for lower GPUs (unchanged) |
| `_old_2026_10_24/` | the package of 2026-10-24, unchanged |
| `_old_2026_11_18/` | the copies of the 2026-11-18 package that were replaced on 2026-11-20, unchanged, at the same paths |
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
- **The sliding window of the 350M model (2026-11-20):** `all_sft_optimizations.md` section 4.17, then
  `06_docs/sliding_window_attention_sft_analysis.md` (sections 11-12: how other libraries do it, every result) and
  `06_docs/README_sft_BluTrain.md` section 5.3 (how to run SFT with the window).
- **The 350M SFT checkpoints for the DPO and RL teams (2026-11-21):** the handover package `../DPO_RL_handover_350M_SWA_2026_11_21/00_START_HERE.md` (the four checkpoints on Hugging Face, how our SFT is wired, every flag, the procedures, the results).

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
| 2026-11-19 | `all_sft_optimizations.md` (this folder) written; every SFT doc brought up to date; preproduction-135M compared with our branch (SFT step 135M -24.9%, 350M -12.0%, 1.6B -1.1%; pretraining -2.9% / -0.8% / -0.7%; the setup differences are listed there) | work log 45.200 |
| 2026-11-20 | Sliding-window attention for the 350M model: Navin's kernels ported onto ours, packed rows + window, the SFT trainer (`BLU_SWA_WINDOW`, `BLU_SWA_GLOBAL_EVERY`; the base's window checked at start-up, `BLU_SWA_MISMATCH_OK`), the window carried by checkpoints (`model_config.json`), the converter and the exports; 3 new tests | README 5.3; `all_sft_optimizations.md` 4.17; work log 45.201 |

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
