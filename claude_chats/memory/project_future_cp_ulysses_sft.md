---
name: project-future-cp-ulysses-sft
description: "FUTURE (user will say when): context parallelism for long-context SFT = Adhitya's Ulysses all-to-all wrappers AROUND OUR SFT kernels (not our features into his kernels); agreed 2026-10-31"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-31T14:27:22.361Z
---

Agreed 2026-10-31 ("excellent approach ... document this as future task and be ready, I will tell later when needed").
Full analysis: SFT work doc 45.127 + follow-up, plan Q&A 178 and 180.

- Order: no CP while a row fits one GPU (135M: 131,072 on one H200, B 1, 107.6 GB); then Ulysses with OUR kernels; ring
  only past that (his rotator / LSE merger / load balancer / doc-id ring, ids + positions from our chats).
- Ulysses = his `ulysses_combine` / `ulysses_partition` (context_parallel/UlyssesAttention.h at CP cf1372d: NCCL
  all-to-all + reshapes, backward = reverse): our fused RoPE on the slice with per-chat positions -> combine -> our
  scaled_dot_product_attention(..., chats) for the whole row, Hq/U heads -> partition.
- Glue in OUR SFT trainer behind a CP switch: CP group (his DeviceMesh / ProcessGroupNCCL), per-rank slices of inputs /
  labels / positions, loss = slice trained-token SUM / GLOBAL trained count (all-reduce over DP x CP), gradients summed
  (his trainer averages per-rank local means: wrong weighting for SFT), ZeRO over DP x CP.
- Limit: U divides KV heads (135M: 2; 1.6B: 2, 3, 6 -> 128k at U 6); beyond: KV-head replication or ring.
- His CP builds its RoPE table on the CPU (YARNOps.cpp), so the TI table bug did not hit his runs. YaRN: we have none;
  needed only for chats longer than the base pretraining length; decide by a test.

**Why:** keeps our faster kernels (Vj2 backward, chat plan, per-chat RoPE) and avoids changing his kernels.
**How to apply:** when the user asks for CP, start from 45.127; agree with Adhitya's team first (his repo).
Related: [[project-long-context-fixes-1031]].
