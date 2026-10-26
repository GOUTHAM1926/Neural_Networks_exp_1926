---
name: project-state-2026-10-09
description: "STATE end of 2026-10-09: v2 step 1 (library survey) done, step 2 next; loss-kernel standalone H200 bench done + colleague doc; nsys Block D and log scp pending; today's research not yet in the work doc"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-09T14:59:24.342Z
---

**v2 (packing + variable-length attention):**
- **Step 1 DONE** (in chat, not documented yet). Every library uses one of two designs:
  - **A** = each chat is its own sequence via `cu_seqlens`: FA2, FA3, FA4, PyTorch mem-eff / varlen_attn, Transformer Engine.
  - **B** = one row + a block mask: FlexAttention, torchtitan.
  - The frameworks have no kernel of their own: HF transformers, TRL, Megatron (through TE), torchtitan.
- Our SFT path on H200 is **v62 fwd + Vj1d bwd** (bf16, GQA 18/6, hd 128).
- **Step 2 NEXT:** read Vj1d, RoPE and the loader, then choose A or B. After that, step 3 (design) and step 4 (build).

**Loss-kernel standalone bench (side task for a colleague, H200 GPU 5):**
- Files in `profiling_reports/sft_loss_bench_2026_10_09/`:
  - `bench_sparse_ce_sft.cu` (md5 f8a5bf8c…);
  - the colleague doc `SFT_loss_kernel_benchmark_H200.md`.
- Median batch (23.1% trained): F 0.276, B 1.600, F+B 1.878 ms; 15 ms per step.
- Masking is 3.2× faster than no masking; saving stats saves 17–22%; all 6 cases PASS.
- **Pending:**
  - the user scp's `bench_run1.log` from the box (`~/gautam_k/loss_bench/`);
  - nsys Block D was prepared but not run.

**Other open items:**
- The shared doc tab still says "1–5" chats per row in the kernel section Step 5; it should be 1–12.
- Today's research is not written into the work doc yet: the library internals, causal skip vs mask, the attention types, flash vs mem-eff precision, and the loss bench.
- Nothing is committed. The new `profiling_reports/` folders need `git add -f`.
- **Later:** benchmark the loss kernel vs PyTorch / Liger / Apple CCE; Q11; Q4, Q5, Q6, Q12, Q13.
- **Parked:** [[project-future-b300-loss-fwd-ignore]].
