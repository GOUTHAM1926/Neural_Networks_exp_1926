---
name: project_blukernels_release_audit
description: Honesty audit of public Blu-Kernels repo vs GPT-2 124M production (may27_406k)
metadata: 
  node_type: memory
  type: project
  originSessionId: d9a70324-ba75-4894-89c3-21a2ac17e35d
---

Audited github.com/Blubridge-Tech/Blu-Kernels (public kernel release) against GPT-2 124M production source in BluTrain_BestPrecisionEnv_may27_406k.

**GPT-2 124M facts (verified in the real driver, not the class model):** trains in **fp32**; driver is `Tensor-Implementations/gpt2_fmha.cpp` (+ `gpt2_fmha_ddp.cpp` for 8-GPU), which build the model **inline** and do **NOT** include the `gpt2_scripts/model/` class model. GOTCHA: the class model (`MLP.h:53`) calls `fused_bias_gelu`, but the real drivers call **standalone `autograd::gelu`** (`gpt2_fmha.cpp:270`, `ddp:335`; fused line is commented out). An agent that traces the class model wrongly concludes "fused" — always check the driver. Loss = `sparse_cross_entropy_loss`; optimizer = AdamW (multi_tensor_adam).

**Verdict — almost all released kernels are faithful, algorithm-preserving professional rewrites (just comment/dead-code cleanup):** layernorm (×3, verbatim), CE forward (byte-identical), CE backward (math-identical float4 rewrite), AdamW multi_tensor (byte-identical), both attention backward kernels (byte-identical), standalone gelu fwd/bwd (faithful — and correct, since GPT-2 used standalone not fused).

**CRITICAL: may27_406k is NOT the latest source.** The user has newer code than may27_406k (e.g. output.md holds the latest attention forward). Two of my audit "misrepresentation" flags were WRONG because I/agents diffed against stale may27_406k instead of the latest code — same trap twice (gelu usage + attention forward). Always confirm the source repo is current before flagging a release file as "different from production."

**Only ONE genuine discrepancy (fixed):**
1. Reduction (`unified_reduce_standalone.cu`) — originally released SINGLE-CTA only; GPT-2 bias-grad (OuterContiguous, F=2304/768 × B*T=16384) actually used MULTI-CTA + vec4. FIXED by us: ported multi-CTA global-reduce + vec4 outer kernel into the standalone, verified (8/8 pass, 4 sanitizers clean). vec4 kernel = OuterContiguous only; Inner/Generic/small-Outer use scalar unified_reduce_kernel; multi-CTA in both. See [[project_reduction_bugs]].

**Attention FORWARD — CORRECT (earlier flag retracted):** released `fused_attn_forward_kernel_tc_sm89.cu` is the register-resident-O kernel (`O_acc`, `loadB_nt_x4`, `s_alpha` deferred rescale, `s_out=s_q` reuse). Normalized diff vs the user's latest (output.md active kernel, lines ~1136-1745) = IDENTICAL kernel body; only packaging differs (inlined struct/includes, stripped profiler/launcher, `(void)gather;`). may27_406k just has the older smem-O version. Released forward is faithful to the latest code.

**Cross-cutting:** README perf claims (2.15x CE, 1.70x attn, GELU/LN GB/s) have NO benchmark harness shipped except 2 bf16-GEMM-vs-cuBLAS probes; no PyTorch baseline anywhere — unverified, should ship harness or drop numbers. BLAS fp32 sgemm kernels are GPT-2's path only when built `WITH_BLUBLAS=1` (default build uses cuBLAS) — README should say opt-in. bf16 GEMMs honestly labeled non-GPT-2 library code. README has 4 stale `<!-- TODO: add image -->` comments (lines 153/159/182/205) above working <img> tags — just delete. CPU-reference self-test is professional (matches llm.c); ThunderKittens uses a shipped PyTorch-reference generator + timing harness — recommend same for compute-bound GEMM/attention.
