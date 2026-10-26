---
name: project-bgrada-bf16-unsupported
description: cuBLASLt EPILOGUE_BGRADA backward bias-grad fusion never fired with bf16 bias output on sm_86 — heuristic returned 0 algos
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
---

Attempted to fuse `grad_bias = sum_i grad_output[i, :]` into the grad_B backward matmul via `CUBLASLT_EPILOGUE_BGRADA` for the bf16 GPT-2 path. Wired into `AddmmBackward::apply` since that (not `LinearBackward`) is the actual backward path — the bias-grad was being computed by `unified_reduce_kernel<__nv_bfloat16, SumOp>` via `reduce_to_shape`, NOT by `reduce_bias_kernel_bf16` (which is dead code in this codebase).

**Result:** cuBLASLt's heuristic returned 0 algos → fast path never fired. nsys14 still showed all 240 bf16 unified_reduce calls. My fallback logic was buggy and ran the backward matmul twice (the with-bias variant did matmul + failed BGRADA, then my `if (!fused) cuda_matmul_backward` ran it again), causing a ~17 ms/step regression. Fix and revert both completed; baseline restored.

**Why:** Likely cuBLASLt for sm_86 + COMPUTE_32F + bf16 GEMM + BGRADA requires the bias output to be fp32, not bf16. We set `CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = CUDA_R_16BF` matching the grad_bias dtype. PyTorch's analogous backward bias-grad paths use fp32 bias output.

**How to apply:** If revisiting BGRADA/BGRADB for bf16 GEMMs on sm_86, allocate the grad_bias as fp32 + add a tiny N-element fp32→bf16 convert kernel after, rather than passing bf16 directly. Also: in the call site, never duplicate the matmul backward — when the fused variant declines, only fall back for the BIAS computation (the matmul work is already done inside the fused helper).

Related: [[project-epilogue-bias-forward]] (the forward EPILOGUE_BIAS path DID work with bf16 bias because the bias is an input there, not an output).
