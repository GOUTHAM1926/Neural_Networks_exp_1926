---
name: project-future-v2-packing
description: "FUTURE TASK (2026-10-08, REVISED same day): implement v1.5 bucketing THEN v2 packing; use Kruthik's fused embedding backward (fix fixed-N) instead of atomicAdd; SFT Q&A question 10; measured x2.84 / x3.60 work cut"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-08T10:58:17.284Z
---

On 2026-10-08 the user first said "skip v1.5, do v2", then REVISED it the same day to: **do v1.5 (length bucketing) too, and then v2 (packing)**. Also: **use Kruthik's fused embedding backward (`EmbeddingBwdFused`, BLUCCL) instead of the bf16 atomicAdd kernel**, modifying it as needed. The required fix: pass the real token count N into `run()` instead of the install-time B×T (`embedding_fused.cu:1300` `(void)N`; `run()` sorts `N_` at `:1189`), and size its workspace for the largest N. v1.5 changes N every micro-batch, so this fix is mandatory for v1.5.
- v1.5 also brings new GEMM shapes (M = 8·T), so check the BluBLAS dispatch and the attention T%256 rule.

This is SFT Q&A **question 10** (padding vs variable-length attention) in the shared doc. Remind the user of it as the next implementation task.

**Why (measured 2026-10-08 on the real stage-0 data, 5,000 chats, mean 1,238 tokens; script + output in `profiling_reports/q10_packing_sim_2026_10_08/`, re-run 10-09 with the same numbers; v2 = 1,517 rows, 16.3 pad tokens per row, 1–12 chats per row (NOT 1–5 as first said)):**

| | real tokens | work vs v1 |
|---|---|---|
| v1 (pad to 4096) | 30.2% | 100% |
| v1.5 realistic | 79.8% | 35.2% (×2.84) |
| v2 packing with mask | 99.6% | 27.8% (×3.60) |
| v2 without the mask | 99.6% | 30.3% (×3.30), and WRONG (chats see each other) |

These are work ratios (upper bounds); per-step optimizer and all-reduce costs don't shrink.

**What v2 needs:**
- **Attention:** a block-diagonal mask in forward v62 AND backward Vj1d, plus a guard for fully-masked tiles (they give NaN today).
- **RoPE:** positions restart at 0 for each chat.
- **Loader:**
  - first-fit-decreasing packing (10-14: our sim + D10 test use first-fit; best-fit gives the identical 1,517 rows on stage-0, `q10_packing_sim_2026_10_08/ffd_vs_bfd_output.txt`);
  - NEVER split a chat across rows; drop chats longer than 4096 as today;
  - pad the small leftover at the end of each row (measured: 16 tokens per row on average, 1,517 rows);
  - pass chat boundaries to the attention.
- **Step size:** chats per row vary (1–12), so define the step by tokens or chats, not rows. The global-denominator loss already weights every token equally.
- **Validation:** compare with PyTorch SDPA using an explicit mask; benchmark against FlashAttention-2 varlen and FlexAttention document masking. A mask bug raises no error.

**Side effects of v2:**
- Tokens per micro-batch stay 8 × 4096, so Kruthik's fused embedding kernel (fixed N, `embedding_fused.cu:1300`) still works. It would NOT work with v1.5.
- Pads drop from ~70% to ~0.4% of positions, so the bf16 atomicAdd pad-row contention in the embedding backward mostly disappears.
- Q11 (logits only at trained positions) then saves only ~1.5% instead of ~4.9%.
- **Packing reference doc (10-14):** `work_docs/sft_packing_algorithms_bfd_ffd.md` — FFD vs BFD (16+ computed examples, neither always wins), over-long chats (we drop), TRL bfd / bfd_split / wrapped, varlen vs cross-document attention, 12 corrections to a pasted explanation.
- **10-14 review + study:** 1,511 rows (the minimum) IS reachable on stage-0 (subset-sum method, `subset_sum_packing_1511.py`); FFD/BFD stop at 1,517. FFD vs BFD (2 studies, ~27k tests + all 3.1M small inputs): same rows on our kind of data; where they differ BFD wins ~15:1 (by 1 row); FFD 1.3-1.9x faster with a tree, both <1 s per 1M chats -> either fine, MUST use a search tree; user picks (BFD tiny edge, FFD = our tests). In transformers, SDPA/eager ALSO get a same-chat mask from packed positions (masking_utils.py) — my "FlashAttention only" claim was wrong.
