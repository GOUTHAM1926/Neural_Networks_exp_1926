---
name: project_matmul_epilogue_audit_1024
description: "FIRST TASK MONDAY 2026-10-26 — matmul audit of the production pretraining trainer (C1-C7, \"use the fp32 accumulator directly\"); full text in sft_work_documentation.md §45.93"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-24T16:24:24.738Z
---

**The user decided (10-24 night):** this is the FIRST task on Monday morning, 2026-10-26, the immediate future task. Start there before anything else.

Full answer text, kept as the user asked: `work_docs(mark-down - (.md) files )/sft_work_documentation.md` §45.93. Note that the folder name ends with a space.

**Trainer:** lottn `train/production_zero/bluscript_zero_ckpt.cpp` with the production config:
- `BLU_ALL_BF16=1` + `BLU_BF16_LOGITS=1` (residual, logits and embedding all bf16, so every `to_dtype` around the GEMMs is a no-op);
- `grad_view=true`, `fp32_reduce=false` (micro-step grads accumulate in the bf16 wire buffer);
- Muon, ZeRO v2, BluBLAS.

**Library rule:**
- `matmul` output dtype = the input dtype (GenMatmul.cpp).
- The BluBLAS bf16 API has NO fp32-output GEMM.

**Already fused (proof):**
- Attention v62 forward: S, P and O stay in registers (GQA_fwd_sm90.cu:924-996, 1243).
- Vj2: S, dP and dQ are fine.
- Muon Newton-Schulz: `cuda_addmm` → BluBLAS SM90 addmm epilogue with a full-matrix bias.
- F2 (AutogradMeta.cpp:177-193) is a separate widen+add kernel, not an epilogue, and it is unused under grad_view.

**Exists but unwired:** the BluBLAS SM90 GEMM+SwiGLU epilogue (no TI caller; `cuda_matmul_swiglu` is gone; 08-14 NaN history).

**Open (estimates per micro-step):**
- **C1** dW beta=1 epilogue: −5.9 GB.
- **C2** fused QKV: −23.6 GB (caveat: Muon is per matrix).
- **C3** residual add in the epilogue: −15.7 GB, and it allows an fp32 residual at bf16 cost (addmm M·N bias; needs a 2-D flatten and an AddmmBackward check).
- **C4** lm_head+CE fused: 3.3 GB of logits.
- **C5** wire the SwiGLU epilogue.
- **C6** Vj2 dK/dV fp32 partials: +15.7 GB (a precision trade).
- **C7** Muon: the last NS GEMM writes fp32.

**How to apply:** Monday step 1 = one nsys run on the H200 timing the kernels behind C1–C3, before any code. Ask before the GPU run.
