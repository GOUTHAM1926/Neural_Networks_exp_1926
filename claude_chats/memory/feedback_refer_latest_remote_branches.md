---
name: feedback-refer-latest-remote-branches
description: Before citing BluTrain/TI code, refer to the LATEST remote dist_side_merge AND dist_side_merge_BLUCCL, never a stale local clone
metadata:
  type: feedback
---
Cite code only from the latest remote state of BOTH team branches: `dist_side_merge` and `dist_side_merge_BLUCCL` (BluTrain + Tensor-Implementations). The local clone `~/Downloads/BluTrain_sft` can be stale.

**Why:** 2026-09-30 I explained the embedding backward from local dist_side_merge only (2 kernels: atomicAdd / fp32 cooperative); a colleague's atomic-free kernel (`EmbeddingBwdFused`, BluCCL `embedding_fused.cu`, `BLUCCL_FUSED_EMB=1`) lives on dist_side_merge_BLUCCL. User: "latest only refer that itself".

**How to apply:** the user runs the fetch (private repo needs the token in the URL; plain `git fetch origin` fails "Repository not found"):
`git fetch https://<TOKEN>@github.com/BlubridgeAI/<repo>.git dist_side_merge:refs/remotes/origin/dist_side_merge dist_side_merge_BLUCCL:refs/remotes/origin/dist_side_merge_BLUCCL`
Then read with `git show origin/<branch>:<path>`, and say which branch + commit each fact is from. Related: [[feedback-verify-claims-and-benchmark]].
