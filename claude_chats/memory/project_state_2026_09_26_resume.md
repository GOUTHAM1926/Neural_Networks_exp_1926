---
name: project-state-2026-09-26-resume
description: End of 2026-09-26: 4 fix sets pushed (Bug D, F1 Muon/v2, 7 reductions, GPU as_type); docs in loss_kernel_dtype_bugs.md §17.3/§22-§26 (pushed 30a04f8); NEXT = F2 fused grad accumulation; box released
metadata:
  type: project
---

**Pushed on 2026-09-26** (company identity, `dist_side_merge`):

| lib | BluTrain | what |
|---|---|---|
| `7810aea` | `e3633e8` | Bug D + conversions |
| — | `23ae1e5` | F1 for Muon + v2 |
| `97735ad` | `f1618d8` | 7 reduction bugs |
| `3a08ade` | `4e92e6c` | GPU as_type |

See [[project-f1-muon-v2-pushed]], [[project-reduction-fixes-pushed-0926]] and
[[project-conversion-kernels-pushed-0926]].

**Docs:**
- MASTER: `~/Downloads/sft_research/loss_kernel_dtype_bugs.md`, sections §17.3 and §22-§26. §26 is the full
  pending-task list. The SFT doc has a status line.
- MIRROR: `work_docs(...)/`. Pushed in the Neural_Networks repo as `30a04f8` (personal identity).

**Box:** `fair-soul-wanders-fin-02` was released. The final copy is in
`profiling_reports/box_fair_soul_2026_09_26_FINAL/`. The per-topic folders are `bugd_2026_09_26/final`,
`f1_muon_v2_2026_09_26`, `reduce_gpu_2026_09_26` and `conversion_kernels_2026_09_26`.

**NEXT (user, 2026-09-27): F2.**
- Location: `AutogradMeta.cpp:163-170`, which does `*grad_ += update.as_type(fp32)`.
- The fix: a fused `grad_fp32 += float(bf16)` kernel. Traffic goes from 18 B to 10 B per element, with no
  temporary, and the result is bit-identical.
- Measure: exactness, kernel time, and a trainer A/B at GRAD_ACCUM=4.
- On a new box: set up the identity in BOTH repos first, and always build with `make -j$(nproc)`.

**After F2:** the rest of §26, including the reduction speed A/B, BluBLAS sub-64 transposed GEMM, SFT tasks 1-7,
and rotating the GitHub tokens.
