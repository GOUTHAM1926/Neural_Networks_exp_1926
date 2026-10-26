---
name: project_sft_flow_walkthrough_and_decisions_1005
description: "2026-10-05 report §3.1 walkthrough (steps 1-4 done, resume at step 5) + decisions on range check (in dataloader, not kernel) and denominator (after dataloader, seed W)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-05T10:31:37.656Z
---

**Walkthrough of SFT report §3.1 ("The whole flow", 10 steps)**
- Steps 1-4 were explained on 2026-10-05: load dataset, chat template, tokenizer, labels.
- Each explanation used real data and real code: the Tülu 3 SFT mix rows, tiktoken GPT-2, the template files of 5 models, and the kernel lines.
- **Resume at step 5 (pad + write).**
- The user owns the plan for steps 1-5 (report §8.1 tasks 3-4) unless their lead gives it to a data team. Even then, the file format, the labels and the template are the trainer's contract.

**Decisions 2026-10-05**, written into loss doc §26 SFT items 4-6, master + mirror:
- **Range check goes in the SFT dataloader**, once per file:
  - inputs < `vocab_size`;
  - labels < `vocab_size` or == the ignore_index;
  - use `cfg.vocab_size`, never 50304 written as a literal.
- **No kernel bounds check.** Report §8.1 task 7 became a comment only. Both loss calls take their labels from a dataloader (`bluscript_zero_ckpt.cpp:2240` train, `:2179` val). See [[no-redundant-boundary-fixes]].
- **Global denominator (report task 6): do it after the dataloader.** Pass `ignore_index=65535` and `&denom` (the step's count, all-reduced with SUM), and seed backward with **W instead of 1/GRAD_ACCUM**, because the ZeRO reduce is `op=avg`.

**Facts checked that day, so they don't need re-deriving:**
- `denom=nullptr` divides by this micro-batch's own non-ignored count. That equals B·T only when nothing is ignored.
- The kernel compares `static_cast<int64_t>(uint16 label) == int64 ignore_index`. So -1 (or any negative) never matches, and -100 written into uint16 becomes 65436.
- PyTorch requires int64 (Long) labels and indices, and nanoGPT stores uint16 on disk then converts to int64.

**How to apply:** the user wanted code comments at the kernel and the trainer loss call. I proposed them but they are not written yet; ask before editing team code.

**Correction (same day):** the trainer's loader is `MixedDataLoaderLite` (TI `include/MixedDataloader.h`, `bluscript_zero_ckpt.cpp:1829`). It is not `dist/dataloader_dist.h`, so report §3.9 names the wrong file. It already has an opt-in id check (`BLU_TOKEN_MAX`, `:580-607`), which runs per batch and only prints. The SFT check builds on it. The walkthrough is now at step 6.
