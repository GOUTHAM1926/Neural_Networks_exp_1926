# Verification request: the loss-sum bug, the embedding backward, and context-parallel facts

**To:** the Claude Code session on Adhitya's machine (and Adhitya)
**From:** Gautam (SFT and kernels; BluTrain / Tensor-Implementations branch `dist_side_gautam_lottn`)
**Date:** 2026-10-30
**What we ask:** check each item below against YOUR current code and run logs, then write back one reply document in
the format of section 5. We could not check it ourselves: our copy of `Context_Parallelism` is commit `661b6ef`
(2026-08-30), and we cannot see your run logs.

---

## Rules for the verifying Claude

1. Read only. Do not edit, commit or push anything.
2. Do not start any GPU job unless Adhitya explicitly says yes. The GPU test in Q6 is optional.
3. Every answer needs evidence: file path + line number + commit, a quoted code line, or a quoted log line.
4. If something cannot be checked, write `UNKNOWN` and the reason. Do not guess.
5. Do not paste tokens, passwords or keys into the reply.
6. Our line numbers refer to our versions (Tensor-Implementations `f7aa40f`, Context_Parallelism `661b6ef`). Yours may
   differ: find the same code by function name.

---

## 1. Main suspicion: the loss value ignores every row after row 65,536

### 1.1 The code (Tensor-Implementations `f7aa40f`, `src/Kernels/cuda/loss/LossKernels.cu`)

- The cross-entropy forward first writes ONE loss value per row (one row = one token position) into a float array.
  Rows whose target is the ignore index write 0.
- Then it adds the rows up in two kernel launches of `sum_reduction_kernel` (our lines 249-273):
  - Each thread loads exactly ONE value: `sdata[tid] = (i < n) ? static_cast<float>(input[i]) : 0.0f;`. There is no
    loop.
  - Stage 1: `<<<reduce_blocks, 256>>>` with `reduce_blocks = ceil(N / 256)`. This gives one partial sum per 256 rows.
  - Stage 2: `<<<1, 256>>>` over the `reduce_blocks` partials. Thread i reads partial i, so only partials 0-255 are
    read.
- Result: rows 65,536 and later (256 x 256) never reach the loss value of that call.
- The denominator is counted over ALL N rows. `count_valid_targets_kernel` (our lines 185-205) loops over every row. So
  the printed loss is about **true loss x 65,536 / N**. At N = 131,072 that is half the true value.
- The same two-stage sum is used:
  - in the training forward `sparse_cross_entropy_forward_cuda_impl_vec_save_stats` (our lines ~1110-1125);
  - in the forward without saved statistics `sparse_cross_entropy_forward_cuda_impl_vec` (our lines ~561-584);
  - on B300 too: the sm103 forward fills the same per-row array, and the same two-stage sum follows;
  - in the other losses of the same file (MSE, BCE and others, our lines ~1418-1714).
- This code has been in the library since at least 2026-04-10 (commit `b31e831`, which moved the file to this path).
  It is not specific to SFT.

### 1.2 What it should NOT affect (please confirm in your version)

- **Gradients.** The backward uses the per-row max / sum saved by the forward and the denominator. It never reads the
  summed loss. So the weights train correctly; only the printed loss number is wrong.
- **Your perplexity / needle evaluation.** If it uses `context_parallel/EvalNLLKernel.cu` (one block per position,
  fixed-tree reductions), it does not go through this two-stage sum.

### 1.3 When it fires in your trainer (from our copy `661b6ef`)

- `Scripts/Blutrain/bluscriptCP.cpp` calls the library loss `autograd::sparse_cross_entropy_loss(logits, shy.y_local)`:
  - for training, our line ~2860;
  - for validation, our line ~2811.
- The logits there are `[B, T_local, vocab]` (comment at our line ~2856), with T_local = T / CP size. The default
  B = 2 (our line 126, env `CP_B`); T comes from `CP_T`.
- So rows per loss call: **N = B x T_local**. The printed loss is wrong exactly when **N > 65,536**.

  | B | T | CP size | rows per call N | hit? |
  | --- | --- | --- | --- | --- |
  | 2 | 65,536 | 2 | 65,536 | no (the largest size summed in full) |
  | 2 | 131,072 | 2 | 131,072 | yes: printed loss about 1/2 of the truth |
  | 2 | 131,072 | 4 | 65,536 | no |
  | 1 | 262,144 | 2 | 131,072 | yes: about 1/2 |
  | 3 | 65,536 | 2 | 98,304 | yes: about 2/3 |

- Please also check:
  - the windowed validation near our lines ~2510-2540 (`win_loss`, `win_tok`): how many rows go into each call;
  - any other trainer or script of yours that calls this loss.

### 1.4 Questions

- **Q1.** Which Tensor-Implementations commit does your context-parallel build link against? The Makefile uses
  `$(BLUTRAIN_ROOT)/Tensor-Implementations`; give `git rev-parse HEAD` and the branch.
  - In that version, does `sum_reduction_kernel` load one value per thread with no loop?
  - Is the stage-2 launch a single block?
  - Quote the lines.
- **Q2.** List every call of the library losses in your trainers and scripts (search for `sparse_cross_entropy_loss` and
  the other loss entry points). For each call, give the formula for its row count N.
- **Q3.** In your library version, does the cross-entropy backward use the saved per-row statistics and the
  denominator, and never the summed loss? Quote the code.
- **Q4.** Does your evaluation path (EvalNLL or anything else) use its own kernel, not this sum? Quote the code.
- **Q5. Run inventory.** For every training and evaluation run you can find (logs, configs, scripts, notes), give:
  - run name, date and purpose;
  - B, T and CP size;
  - attention mode (ring / Ulysses / hybrid);
  - rows per loss call N, and hit? (N > 65,536);
  - whether that run's printed training or validation loss was used in a decision, a comparison or a report.
- **Q6 (optional, GPU, only with Adhitya's OK).** A test that needs no reference:
  - Make fp32 logits that are all zero, any vocab V, any valid targets, N rows. Then every row's loss is exactly ln V,
    so the printed loss must be ln V for every N.
  - Run N = 65,536, 65,537, 131,072 and 262,144.
  - Expected with the bug: ln V for N <= 65,536, and about ln V x 65,536 / N above it. Examples:
    - V = 50,304: ln V = 10.825840, half = 5.412920;
    - V = 49,280: ln V = 10.805274, half = 5.402637.
- **Q7.** For any run that was hit: if an evaluation of the same data through your own NLL kernel exists, compare the
  two. They should differ by about the factor 65,536 / N.

---

## 2. Second suspicion: which embedding-backward kernel your trainer actually gets

- In our trainers the embedding backward kernel is chosen at run time, by env `BLU_EMB_BWD` or YAML
  `embedding.backward`:
  - `segmented` = the BluCCL fused sorted-segment kernel: fp32 sums, deterministic;
  - `cooperative` = deterministic shared-memory sums;
  - `atomic` = one atomicAdd per token.
- If nothing is set, the library's built-in default runs: atomicAdd in bf16 when the embedding gradient is bf16, the
  cooperative kernel when it is fp32. See `src/Kernels/cuda/embedding/EmbeddingKernels.cu`,
  `embedding_bwd_use_cooperative`, env `BLUTRAIN_EMB_BWD` / `BLUTRAIN_EMB_BWD_COOPERATIVE`.
- Our production pretraining config sets `segmented`. Our own config comment says the bf16 atomicAdd path is
  non-deterministic, with large errors on frequent tokens.
- We found today that our SFT runs got the built-in default by mistake: the launcher passed no config.
- In our copy of `bluscriptCP.cpp` we found no `BLU_EMB_BWD` and no fused-embedding install code.

- **Q8.** In your trainer:
  - Is the token-embedding weight and its gradient fp32 or bf16?
  - Which embedding backward kernel actually runs? Look at the env vars above, any fused-embedding hook, and any log
    line that names it.
  - If it is bf16 + atomicAdd, say so plainly.

---

## 3. Facts we need for our long-context SFT plan (questions, not suspicions)

- **Q9.** Which commit and branch of `Context_Parallelism` are you using now? Our BluTrain branch records `cf1372d`; our
  local copy is `661b6ef`.
- **Q10.** Does any of your attention paths (ring, Ulysses, hybrid) support several documents in one row with masking
  between them (document masking, cu_seqlens, segment ids)? In `661b6ef` we found none.
- **Q11.** RoPE positions: one offset per sequence, or restarting per document?
- **Q12.** Which library kernels and which commit are your ring kernels (`GQA_causal_cp_fwd_sm90.cu` /
  `GQA_causal_cp_bwd_sm90.cu`) derived from? Which attention kernel does the Ulysses path call on H100 / H200 today?
  `ContextParallel.h` says `gqa_fused_flash_attn_forward` is Blackwell-only after the library upgrade.
- **Q13.** The largest B x T per GPU you have run, per GPU type (H100 / H200 / B300) and model size, with the peak
  memory if logged.

---

## 4. What we are doing on our side

- A fix of `sum_reduction_kernel` is being written: each thread adds a strided series instead of one value.
  - For N <= 65,536 every thread still gets at most one value, so the result stays bit-identical to today.
  - Above that, every row is counted.
- It will be tested at 65,536 / 65,537 / 131,072 / 262,144 / 524,288 rows against a CPU sum, and timed.
- Your build links this same library, so the fix will reach your trainer too once it is merged. We will share the
  commit.

---

## 5. Reply format

Please write one markdown file, `cp_verification_reply_<date>.md`, with:

1. **Summary table:** `Q | answer (YES / NO / UNKNOWN / value) | evidence in one line`, for Q1-Q13.
2. **One section per question,** with the quoted code, commits and log lines.
3. **The run inventory of Q5 as a table:** `run | date | purpose | B | T | CP | mode | rows per loss call | hit? | loss
   used for decisions?`.
4. **A closing paragraph:**
   - which runs' printed losses are affected, and by what factor;
   - whether the embedding backward of Q8 needs attention;
   - anything else you found that looks wrong, marked clearly as a NEW finding with its evidence.
