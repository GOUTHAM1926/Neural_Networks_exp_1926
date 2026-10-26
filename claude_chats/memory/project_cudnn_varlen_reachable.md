---
name: project_cudnn_varlen_reachable
description: cuDNN varlen IS reachable from torch 2.14 (no TE); MEASURED 10-12: fwd ours≈cuDNN, bwd cuDNN 9-16% faster mid/long
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-12T09:35:32.559Z
---

On 2026-10-12 I told the user cuDNN varlen was reachable only through Transformer Engine. That was wrong. I checked the box's torch 2.14 source and corrected it:
- `torch.ops.aten._cudnn_attention_forward(q, k, v, bias, cum_seq_q, cum_seq_k, max_q, max_k, compute_lse, dropout, is_causal, ...)` takes per-chat offsets. That is design A.
- Its backward is `_cudnn_attention_backward` (`torchgen/packaged/autograd/derivatives.yaml:2951-2953`).
- Nested (jagged) SDPA also reaches it, but cuDNN is LAST in the default order: FLASH, EFFICIENT, MATH, CUDNN (`torch/nested/_internal/sdpa.py:331-334`). It runs only when forced.
- Limits on the nested route:
  - head dim must be ≤128 and a multiple of 8;
  - the longest chat must be longer than 128 tokens;
  - during training, different q/k head counts are refused, so GQA needs K/V expanded or a direct op call.
- `varlen_attn` = FA2, so our "torch varlen" column was never cuDNN.
- The `nvidia-cudnn-frontend` pip package (1.31.0) is a second route.

**Why:** cuDNN is the main bar, and packed cuDNN is the one library missing from the Step 6 tables.

**How to apply:** never say "needs TE" again.

**Status 2026-10-12 evening: MEASURED** (GPU 7). Record: `profiling_reports/v2_step6_benchmarks_2026_10_12/cudnn_varlen_1012/CUDNN_RESULTS.md`; work doc §41.7.
- The 6-KV-head direct op WORKS: cuDNN sums the heads with `fmha_reduce_head_ragged`. It passed the 4-tensor gate on all 7 layouts, and the must-fail control failed.
- Forward: ours A is within 5% of cuDNN.
- Backward: cuDNN is 9-16% faster on middle/long chats, equal on short.
- Kernel level, real middle rows: forward A 519 / cuDNN 457 / FA3 451 µs; backward A 1825 / cuDNN 1547 / FA3 1509 µs.
- To re-run: box `~/gautam_k/step6/run_cudnn_varlen.sh check|main|nsys [libs]|stats`.

Facts for reading the results:
- Public `varlen_attn` causal → window (-1,0) → always FA2 (`varlen.py:62-65,336`).
- The 6-KV-head direct op is untested by PyTorch, so it is probed first.
- In dense SDPA, cuDNN already ran GQA natively (`cudnn::fusion::fmha_reduce_head` in the 10-12 nsys).

Related: [[project_v2_build_progress]], [[feedback_verify_claims_and_benchmark]].
