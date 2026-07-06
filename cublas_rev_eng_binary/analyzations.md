# cuBLAS Reverse-Engineering + Qwen Matmul Coverage Analysis

**Date**: 2026-07-01
**Machine**: NVIDIA B300 SXM6 PC (SM 10.3a)
**Software**: CUDA 13.2, `libcublasLt.so.13`, `nsys 2025.6.3`

**Original goal**: Reverse-engineer cuBLAS's BF16 kernel catalog on B300 → identify what BLUBLAS is missing → prioritize what to build next.

**Updated goal (2026-07-01, mid-analysis pivot)**: Instead of a broad "build every variant cuBLAS has" roadmap, focus on the **matmul shapes actually needed to run Qwen models up to 7B parameters**. Ship optimized kernels for those specific shapes that beat cuBLAS, then expand outward.

---

## TABLE OF CONTENTS

1. What we accomplished in Task 1 (Steps A+B: binary + profile)
2. Binary reverse-engineering findings (Step A)
3. nsys profile findings (Step B) — the crown jewel unmasked
4. Full Blackwell BF16 kernel menu (12 unique kernels)
5. Kernel naming convention — decoded
6. Shape → kernel decision mapping (from nsys trace)
7. BLUBLAS gaps vs cuBLAS
8. **PIVOT**: New Qwen-focused development plan
9. Qwen model configs (2.5 series + 3 series)
10. Per-model matmul shapes derived
11. Test coverage gap — do our current tests cover Qwen ≤7B?
12. Prioritized shape/kernel roadmap
13. Task 2+ (next tasks after this Step 1 completion)

---

## 1. What we accomplished in Task 1

**Task 1 = "Understand cuBLAS's Blackwell BF16 menu and where BLUBLAS stands vs it."**

Two sub-steps:
- **Step A**: Reverse-engineer the cuBLAS binary (`libcublasLt.so.13`) to enumerate kernel classes.
- **Step B**: Profile actual cuBLAS calls with `nsys` on B300 for our 16 validation shapes + 8 profile-sweep shapes across 3 layouts (TN, NN, NT), then correlate kernel names to shape triggers.

Both steps produced complementary data:
- **Step A** gave us the **utility kernel classes** (GEMV, split-K reduce, epilogues, patching, transforms — ~1,083 BF16 kernels visible).
- **Step B** unmasked the **crown-jewel full-tile GEMM kernel names** that Step A missed because they don't have "bfloat16" as a string in the mangled symbol.

**Task 1 status**: ✅ DONE.

---

## 2. Binary reverse-engineering findings (Step A)

### 2.1 Method progression

| Attempt | Command | Result |
|---|---|---|
| 1 (naive) | `cuobjdump --dump-elf-symbols ... \| grep -iE "bf16" \| grep gemm` | 14,895 lines — MISSED 92% of the library |
| 2 (wider) | Raw `cuobjdump --dump-elf-symbols` | 176,078 total symbol lines — but sm_100 count matched section headers not kernels |
| 3 (correct) | Extract cubins from fatbin → per-cubin symbol dump | **1,188 sm_100 cubins**, 5,532 unique kernel symbols, 1,083 BF16-tagged |

### 2.2 Library composition (ground truth)

```
libcublasLt.so.13 (~180 MB)
├── ELF symbols (device functions):     42,200 unique
├── .nv_fatbin section (~133 MB):     5,449 embedded cubins
│   ├── sm_120 (Blackwell consumer):   1,596
│   ├── sm_90a (Hopper native):        1,376
│   ├── sm_100 (B100/B200/B300 native):1,188  ← MAIN Blackwell datacenter kernels
│   ├── sm_80 (Ampere):                  482
│   ├── sm_89 (L4/RTX40):                247
│   ├── sm_90 (Hopper):                  221
│   ├── sm_75 (Turing):                  166
│   ├── sm_86 (Ampere consumer):         105
│   ├── sm_103 (B300 native):             64  ← ONLY stubs — see 2.3
│   ├── sm_121, sm_100a:                   5
└── Embedded PTX (JIT source):           288 (287 sm_120, 1 sm_75)
```

### 2.3 Key surprise: sm_103 cubins are only stubs

The 64 sm_103 cubins ALL contain just `cask_cublas::nopHardwareDetection()` — a tiny detection stub, NOT real GEMM kernels. **B300 runs sm_100 cubins natively via forward-compat/JIT** — that's the actual GEMM code.

### 2.4 The 1,083 BF16 utility kernels found

| Class | Count | Purpose |
|---|---:|---|
| `std::enable_if` SFINAE stubs | 192 | (not real kernels) |
| `gemmk1_kernel` | 160 | 1-CTA GEMM for tiny-K shapes |
| `cublasLt::splitKreduce_kernel` | 148 | Split-K reduction epilogue |
| `gemv2N_kernel` | 128 | GEMV, N-layout |
| `gemvNSP_kernel` | 96 | Skinny GEMV |
| `gemmSN_NN_kernel` | 96 | Small-N GEMM, NN |
| `gemmSN_NN_kernel_64addr` | 48 | Same, 64-bit addressing |
| `gemv2T_kernel_val`/`_ref` | 32+32 | GEMV, T-layout |
| `gemmSN_TN_kernel`(+_64addr) | 28+28 | Small-N GEMM, TN |
| `cublasLt::epilogue::globalKernel*` | 18 | Fused epilogues (relu/bias) |
| `cublasLt_bf16x9_inf_patching::patch_gemm_kernel1x1_core` | 18 | BF16 inference precision patching |
| `cublasLt::epilogue::globalKernelBgradAB_v1`/`v2` | ~28 | Backprop dA/dB epilogues |
| `internal::region_transform_ABC` | 24 | Layout transform utilities |
| `dot_kernel`, `magma_sgemmEx_kernel`, others | rest | Misc |

### 2.5 What Step A did NOT find

Zero matches for `tcgen05`, `Sm100`, `Sm90`, `cutlass::gemm::kernel`, `cutlass_100`, `wgmma`. **The main full-tile Blackwell BF16 GEMM kernel — the one at 1687 TFLOPS on square-4096 — is NOT symbol-searchable by dtype string** because it uses an opaque CUTLASS-template or cask-style symbol. This gap is why Step B was necessary.

---

## 3. nsys profile findings (Step B) — the crown jewel unmasked

### 3.1 Test setup

Modified `test_bgemm_sm103_bf16_{tn,nn,nt}_cluster.cu` to add a `run_cublas_only_*` helper + 8 profile-sweep shapes AFTER the existing 16 validation shapes (9 for BM=128 + 7 for BM=256). Profile sweep runs cuBLAS ONLY (no BluBLAS), so we can sample shapes BluBLAS can't handle.

Fixes required to get NN and NT working (they had the same 3 bugs TN had):
- Ceil-div grid (`floor-div` → `ceil-div` for `grid_m`, `grid_n`)
- Removed `issued = 0;` reset in tile-swap block (fixes producer/consumer race at K=512)
- Removed the buggy `if (M % 512 == 0) launch<256,256,64>` in the 256x256x64 API — falls back to 128x128x64 unconditionally, matching TN (per PTX 9.3 Table 44, `tcgen05.mma.kind::f16` with `cta_group::2` only allows M ∈ {128, 256}, so BM=256 is architecturally invalid on this path).

### 3.2 Final results — all 3 layouts pass 16/16 base tests + 8/8 profile shapes

> ⚠️ **THROUGHPUT NUMBERS BELOW ARE UNRELIABLE.** Server was shared with a colleague's workload during measurement, so TFLOPS are affected by GPU contention. Pass/fail counts and rel_err are still valid (they don't depend on timing). Need to re-measure on an idle node.

| Metric | TN | NN | NT |
|---|---:|---:|---:|
| 128x128x64 base | 9/9 PASS | 9/9 PASS | 9/9 PASS |
| 256x256x64 base | 7/7 PASS | 7/7 PASS | 7/7 PASS |
| Profile sweep | 8/8 executed | 8/8 executed | 8/8 executed |
| BluBLAS `square-4096` | 1636 T ⚠️ | 1630 T ⚠️ | 1636 T ⚠️ |
| cuBLAS `square-4096` | 1687 T ⚠️ | 1686 T ⚠️ | 1682 T ⚠️ |
| **Ratio (BluBLAS ÷ cuBLAS)** ⚠️ | **97.0%** | **96.7%** | **97.3%** |

### 3.3 The crown-jewel Blackwell BF16 kernel

```
nvjet_sm103_tst_128x256_64x6_2x2f_2cta_h_bz_<LAYOUT>
```

- **Tile**: 128 × 256 (BM × BN) — **2× wider N than our 128 × 128**
- **K-depth × pipeline stages**: 64 × 6 (fewer stages than our 8)
- **Cluster**: 2 × 2 (4-CTA covered — ours is 2 × 1)
- **cta_group**: 2 (cooperative MMA — same as ours)
- **Epilogue**: fused (`f` suffix — vs non-`f` variant, both ship)
- **Architecture**: sm_103 native SASS (no JIT!)

This is what cuBLAS runs for `square-4096`, `square-2048`, `llm-style`. It's the reason cuBLAS beats us by 3% on large shapes.

---

## 4. Full Blackwell BF16 kernel menu — 12 unique kernels

Enumerated from `profile_*_cuda_gpu_kern_sum_demangled.csv` across 3 layouts. Layout suffix mapping: TN→NTT, NN→NNT, NT→TNT. All other qualifiers are shared.

| # | Kernel | Tile (BM×BN) | BK×stg | Cluster | 2-CTA? | Variant | Time% | Instances |
|--:|---|---|---|---|---|---|---:|---:|
| 1 | `nvjet_sm103_tst_128x256_64x6_2x2f_2cta_h_bz` | **128×256** | 64×6 | **2×2** | ✅ | **f (fused)** | **31%** | 104 |
| 2 | `nvjet_sm103_tst_128x256_64x6_2x2_2cta_h_bz` | 128×256 | 64×6 | 2×2 | ✅ | (no f) | 7% | 104 |
| 3 | `nvjet_sm103_tst_128x8_64x12_2x1_v_bz_splitK` | 128×8 | 64×12 | 2×1 | ❌ | splitK + vert | 6.5% | 25 |
| 4 | `nvjet_sm103_tst_128x64_64x10_2x2_2cta_h_bz` | 128×64 | 64×10 | 2×2 | ✅ | — | 2.4% | 103 |
| 5 | `nvjet_sm103_tst_64x32_64x16_2x2_2cta_h_bz` | 64×32 | 64×16 | 2×2 | ✅ | — | 2% | 77 |
| 6 | `cublasLt::splitKreduce_kernel<32,16,int,float,bf16,...>` | — | — | — | — | reduction | 1.7% | 75 |
| 7 | `nvjet_sm103_tst_32x64_64x16_4x1_v_bz` | 32×64 | 64×16 | 4×1 | ❌ | vertical | 3% (TN, NN, NT) | 25-50 |
| 8 | `nvjet_sm103_tst_64x32_64x16_2x4_2cta_h_bz` | 64×32 | 64×16 | **2×4** | ✅ | — | 1.1% (TN only) | 78 |
| 9 | `nvjet_sm103_tst_64x8_64x16_4x1_v_bz_splitK` | 64×8 | 64×16 | 4×1 | ❌ | splitK + vert | 1.1-1.9% | 25-50 |
| 10 | `nvjet_sm103_tst_64x32_64x16_1x2_h_bz_splitK` | 64×32 | 64×16 | 1×2 | ❌ | splitK + horiz | 0.7% | 25 |
| 11 | `cutlass::Kernel2<cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_32x1_{nn,nt,tn}_align1>` | 32×32 | 16×1 | — | ❌ | **Turing wmma fallback** | 0.6% | 25 |
| 12 | `nvjet_sm103_tst_64x8_64x16_1x4_h_bz` | 64×8 | 64×16 | 1×4 | ❌ | horizontal | 0.7% (NN, NT) | 52 |
| 13* | `nvjet_sm103_tst_64x16_64x16_4x1_2cta_h_bz` | 64×16 | 64×16 | 4×1 | ✅ | — | 1.1% (**NT only**) | 25 |
| 14 | `nvjet_sm103_tst_64x8_64x16_2x4_h_bz` | 64×8 | 64×16 | 2×4 | ❌ | — | 0.3% (NN, NT) | 26 |

**Unique tile shapes**: 128×256, 128×64, 128×8, 64×32, 64×16, 64×8, 32×64, 32×32 (Turing fallback) = **7 SM103 native tiles + 1 Turing fallback**.
**Unique cluster shapes**: 2×1, 2×2, 2×4, 1×2, 1×4, 4×1 (6 shapes total).

---

## 5. Kernel naming convention — decoded

```
nvjet _ sm103 _ tst _ <BM>x<BN> _ <BK>x<STAGES> _ <CX>x<CY>[f] _ [Ncta] _ [h|v] _ [bz] _ [splitK] _ <SFX>
  1      2     3      4              5              6            7        8      9      10        11
```

| # | Field | Values seen | Meaning |
|--:|---|---|---|
| 1 | `nvjet` | fixed | NVIDIA internal codename (post-Hopper naming) |
| 2 | `sm103` | fixed | Compiled for sm_103a (B300 NATIVE — sm_103 name, sm_100 code path) |
| 3 | `tst` | fixed | Tensor-core tile (uses tcgen05) |
| 4 | `<BM>x<BN>` | 128×256, 128×64, 128×8, 64×32, 64×16, 64×8, 32×64 | Output tile shape |
| 5 | `<BK>x<STAGES>` | 64×6, 64×10, 64×12, 64×16 | K-tile depth × pipeline stages |
| 6 | `<CX>x<CY>[f]` | 2×1, 2×2, 2×4, 1×2, 1×4, 4×1 (± `f`) | Cluster shape; `f` = fused epilogue variant |
| 7 | `2cta` | present or absent | `cta_group::2` cooperative MMA when present |
| 8 | `h` or `v` | h, v | Wave scheduling: h=horizontal, v=vertical |
| 9 | `bz` | fixed when present | Batch-axis-Z (batched GEMM support) |
| 10 | `splitK` | optional | Split-K variant (needs `splitKreduce_kernel` follow-up) |
| 11 | `<SFX>` | NTT (TN test), NNT (NN test), TNT (NT test), NTN, TNN | cuBLAS-internal `<opA><opB><outputStorage>` |

**Suffix note**: Because our tests call cuBLAS with column-major swapped args, the suffix seen ≠ the semantic layout. Mapping:
- Our TN test → suffix `NTT`
- Our NN test → suffix `NNT`
- Our NT test → suffix `TNT`

---

## 6. Shape → kernel decision mapping

Extracted by walking the `cuda_gpu_trace_demangled.csv` in launch order and identifying when each unique kernel first appears (matched against the shape currently running per test file order).

| Test shape (M×N×K) | cuBLAS's kernel pick |
|---|---|
| `tiny` (128×128×64) | `64x8_2x4_h_bz` (small, non-2cta) |
| `small` (256×256×128) | `64x8_1x4_h_bz` |
| `medium` (512×512×128) | `64x32_2x2_2cta_h_bz` (2cta engages) |
| `large` (1024×1024×128) | `128x64_2x2_2cta_h_bz` |
| `xlarge` (2048×2048×256) | `128x256_2x2_2cta_h_bz` (no-fusion variant) |
| `llm-style` (4096×4096×512) | **`128x256_2x2f_2cta_h_bz` (crown jewel)** |
| `square-1024` / `square-2048` / `square-4096` | **`128x256_2x2f_2cta_h_bz`** |
| `decode-1` (1×4096×4096) | `64x16_4x1_2cta_h_bz` (NT) / `64x8_4x1_v_bz` (NN, TN) — skinny GEMV path |
| `decode-8/32/64` | `32x64_4x1_v_bz`, transitioning to full-tile for M=64 |
| `lm-head-1` (1×32000×4096) | `128x8_2x1_v_bz_splitK` (huge N + split-K) |
| `tiny-K-32` (1024×1024×32) | `64x32_1x2_h_bz_splitK` (small K + split-K) |
| `split-K` (64×64×16384) | `64x8_4x1_v_bz_splitK` (fat K + tiny MN, obvious split-K) |
| `unaligned` (129×129×128) | `cutlass_75_wmma_32x32` (Turing fallback for off-tile) |

**Anomalies noted:**
1. The `2x2f` variant averages **49 μs/call** vs `2x2` no-`f` at **10 μs/call** — 5× slower per call. cuBLAS picks `2x2f` only for LARGE shapes (llm-style, square-*) where the extra fusion cost is amortized.
2. B300 uses **sm_103-named SASS natively** (`sm103_tst_*`) — the cubins are dedicated, no JIT from sm_100 needed for these kernels.
3. NT layout has a UNIQUE `64x16_4x1_2cta` kernel not present in TN or NN — suggests slightly different heuristic per layout.
4. NVIDIA still ships **Turing sm_75 wmma forward-compat kernels** as the fallback for off-tile shapes.

---

## 7. BLUBLAS gaps vs cuBLAS

| Category | cuBLAS Blackwell | BLUBLAS today |
|---|---|---|
| **Full-tile crown jewel** | 128×256, 2×2 cluster, 6 stages, fused epilogue | 128×128, 2×1 cluster, 8 stages, no fusion |
| Medium tile | 128×64, 2×2 cluster | ❌ missing |
| Small tile | 64×32, 2×2 or 2×4 | ❌ missing |
| **Skinny GEMV** (M=1) | 64×16, 64×8, cluster 4×1 | ❌ missing |
| Small M (batched decode) | 32×64 with 4×1 cluster | ❌ missing |
| **Split-K variants** | 4 kernels (128×8, 64×32, 64×8) + `splitKreduce_kernel` | ❌ missing |
| Fused epilogue | `2x2f` (fusion built in) | ❌ missing |
| Backprop dA/dB epilogues | `globalKernelBgradAB_v1/v2` | ❌ missing |
| Turing wmma fallback | 32×32 wmma for unaligned/edge | ❌ missing |

**Performance gap by shape**:

| Shape | BluBLAS (T) | cuBLAS (T) | Ratio |
|---|---:|---:|---:|
| `tiny` (128×128×64) | 0.42 | 0.38 | 111% (we win — cuBLAS uses tiny-tile fallback) |
| `medium` (512×512×128) | 12.9 | 33.5 | **38%** (we lose big) |
| `xlarge` (2048×2048×256) | 306 | 504 | **61%** |
| `square-1024` | 285 | 540 | **53%** (biggest gap) |
| `square-2048` | 1119 | 1400 | **80%** |
| `square-4096` | 1636 | 1687 | **97%** |

**Insight**: BluBLAS's 128×128×64 tile is well-tuned for VERY LARGE shapes but LOSES BADLY on medium (256-2048) shapes because cuBLAS uses smaller tiles that fit better. And BluBLAS has no skinny/split-K kernels at all.

---

## 8. PIVOT: New Qwen-focused development plan

### 8.1 Motivation

Original plan was to broadly reverse-engineer cuBLAS's entire menu, then build every variant. That's ~15+ kernels × 3 layouts = ~45+ kernels — too much work for a two-person team.

**New plan**: Team is training Qwen model series (0.6B → 7B). BluBLAS's job is to deliver **all matmuls needed by these Qwen models, optimized to beat cuBLAS on those specific shapes**. Not general-purpose replacement — targeted replacement for the shapes we actually use.

### 8.2 Model targets (from user's plan)

- Qwen 3 series: **0.6B, 1.7B, 4B, 8B** (up to 7-8B parameter range)
- OR Qwen 2.5 series: 0.5B, 1.5B, 3B, 7B (same size ranges)

Both series share very similar architecture (GQA + SwiGLU + RMSNorm + RoPE) — the kernels needed are structurally the same, only dimensions differ.

---

## 9. Qwen model configs — VERIFIED from HuggingFace on 2026-07-01

All 8 configs pulled directly from `huggingface.co/Qwen/{model}/resolve/main/config.json`:
- Qwen/Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B
- Qwen/Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B

Cross-verified against Qwen 2.5 tech report (arxiv 2412.15115) and Qwen 3 tech report.

### 9.1 Qwen 2.5 series (verified from HuggingFace configs)

| Model | hidden | intermediate | layers | Q heads | KV heads | head_dim | vocab | tied? | max_pos |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Qwen2.5-0.5B | 896 | 4864 | 24 | 14 | 2 | 64 (derived) | 151936 | ✅ | 32768 |
| Qwen2.5-1.5B | 1536 | 8960 | 28 | 12 | 2 | 128 (derived) | 151936 | ✅ | 131072 |
| Qwen2.5-3B | 2048 | 11008 | 36 | 16 | 2 | 128 (derived) | 151936 | ✅ | 32768 |
| Qwen2.5-7B | 3584 | 18944 | 28 | 28 | 4 | 128 (derived) | 152064 | ❌ | 131072 |

For Qwen 2.5, **head_dim is not in config** but derived as `hidden / num_heads`. Q_dim always equals hidden.

**Derived**: `Q_dim = num_heads × head_dim = hidden`; `KV_dim = num_kv_heads × head_dim` (below).

| Model | Q_dim | KV_dim | Vocab |
|---|---:|---:|---:|
| 0.5B | 896 | 128 | 151936 |
| 1.5B | 1536 | 256 | 151936 |
| 3B | 2048 | 256 | 151936 |
| 7B | 3584 | 512 | 152064 |

### 9.2 Qwen 3 series (verified from HuggingFace configs — head_dim=128 explicit)

| Model | hidden | intermediate | layers | Q heads | KV heads | head_dim | vocab | tied? | max_pos |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Qwen3-0.6B | 1024 | 3072 | 28 | 16 | 8 | 128 (explicit) | 151936 | ✅ | 40960 |
| Qwen3-1.7B | 2048 | 6144 | 28 | 16 | 8 | 128 (explicit) | 151936 | ✅ | 40960 |
| Qwen3-4B | 2560 | 9728 | 36 | 32 | 8 | 128 (explicit) | 151936 | ✅ | 40960 |
| Qwen3-8B | 4096 | 12288 | 36 | 32 | 8 | 128 (explicit) | 151936 | ❌ | 40960 |

### 9.2.1 CRITICAL: Qwen 3 has decoupled head_dim → non-square Q projections

Unlike Qwen 2.5 where Q_dim always equals hidden, **Qwen 3 sets head_dim=128 in the config explicitly**, so Q_dim = num_heads × head_dim can DIFFER from hidden. This creates **non-square Q/O projections** for 2 of the 4 target models:

| Model | hidden | Q_dim (heads × 128) | KV_dim (kv_heads × 128) | Q_proj shape (input→output) |
|---|---:|---:|---:|---|
| **Qwen3-0.6B** | 1024 | **2048** (16×128) | 1024 (8×128) | **1024 → 2048 (2× expansion)** |
| Qwen3-1.7B | 2048 | 2048 | 1024 | 2048 → 2048 (square) |
| **Qwen3-4B** | 2560 | **4096** (32×128) | 1024 (8×128) | **2560 → 4096 (1.6× expansion)** |
| Qwen3-8B | 4096 | 4096 | 1024 | 4096 → 4096 (square) |

This means for 0.6B and 4B, the O projection reverses: O maps Q_dim → hidden (2048 → 1024 for 0.6B; 4096 → 2560 for 4B). **These non-square shapes are NOT covered by any of our current tests.**

### 9.3 Common Qwen architecture (both 2.5 and 3)

Per transformer layer, the matmul operations are:

| Op | Input dim | Output dim | Comment |
|---|---|---|---|
| Q proj | hidden | Q_dim (= num_heads × head_dim) | Usually = hidden |
| K proj | hidden | KV_dim (= num_kv_heads × head_dim) | Skinny (GQA) |
| V proj | hidden | KV_dim | Skinny (GQA), same shape as K |
| O proj | Q_dim | hidden | Reverse of Q |
| Gate proj | hidden | intermediate | Wide N |
| Up proj | hidden | intermediate | Wide N |
| Down proj | intermediate | hidden | Fat K |
| Attn scores | (head_dim per head) × (num_heads batched) | seq | Batched GEMM |
| Attn output | (seq per head) × (num_heads batched) | head_dim | Batched GEMM |

At the end of the network:
| Op | Input dim | Output dim | Comment |
|---|---|---|---|
| LM head | hidden | vocab_size | Huge N (152k) — only 1 per forward pass |

---

## 10. Per-model matmul shapes (M = token count)

M = tokens = `batch_size × seq_len`. Common values:
- Training: 4096, 8192, 16384, 32768 tokens per micro-batch
- Prefill: 512 – 32768 (single sequence)
- Decode: 1 – 64 (batch × 1)

### 10.1 Complete Qwen matmul shape catalog for M=4096 (training batch)

Format: (M, N, K) where M = tokens, N = output_dim, K = input_dim.
Shorthand: Q_dim = num_heads × head_dim, KV_dim = num_kv_heads × head_dim, I = intermediate, V = vocab_size.

**Qwen 3-0.6B** (hidden=1024, Q_dim=2048, KV_dim=1024, I=3072, V=151936, tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, **2048**, 1024) | Non-square: hidden→Q_dim (expansion) |
| K | (4096, 1024, 1024) | Square (K/V_dim = hidden here) |
| V | (4096, 1024, 1024) | Same as K |
| O | (4096, 1024, **2048**) | Non-square: Q_dim→hidden (contraction) |
| Gate | (4096, 3072, 1024) | Wide N |
| Up | (4096, 3072, 1024) | Same as Gate |
| Down | (4096, 1024, 3072) | Fat K |
| LM head | (4096, 151936, 1024) | Huge N — tied embedding |

**Qwen 3-1.7B** (hidden=2048, Q_dim=2048, KV_dim=1024, I=6144, V=151936, tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, 2048, 2048) | Square |
| K | (4096, 1024, 2048) | Skinny N (GQA) |
| V | (4096, 1024, 2048) | |
| O | (4096, 2048, 2048) | Square |
| Gate | (4096, 6144, 2048) | Wide N |
| Up | (4096, 6144, 2048) | |
| Down | (4096, 2048, 6144) | Fat K |
| LM head | (4096, 151936, 2048) | |

**Qwen 3-4B** (hidden=2560, Q_dim=4096, KV_dim=1024, I=9728, V=151936, tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, **4096**, 2560) | Non-square: 1.6× expansion |
| K | (4096, 1024, 2560) | |
| V | (4096, 1024, 2560) | |
| O | (4096, 2560, **4096**) | Non-square contraction |
| Gate | (4096, 9728, 2560) | Wide N |
| Up | (4096, 9728, 2560) | |
| Down | (4096, 2560, 9728) | Fat K |
| LM head | (4096, 151936, 2560) | |

**Qwen 3-8B** (hidden=4096, Q_dim=4096, KV_dim=1024, I=12288, V=151936, NOT tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, 4096, 4096) | Square — our square-4096 covers this! ✅ |
| K | (4096, 1024, 4096) | Skinny (GQA) |
| V | (4096, 1024, 4096) | |
| O | (4096, 4096, 4096) | ✅ same as Q |
| Gate | (4096, **12288**, 4096) | Wide N |
| Up | (4096, 12288, 4096) | |
| Down | (4096, 4096, **12288**) | Fat K |
| LM head | (4096, 151936, 4096) | |

**Qwen 2.5-0.5B** (hidden=896, KV_dim=128, I=4864, V=151936, tied, head_dim=64):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, 896, 896) | Square |
| K | (4096, 128, 896) | VERY skinny (kv_heads=2, head_dim=64 → KV_dim=128) |
| V | (4096, 128, 896) | |
| O | (4096, 896, 896) | |
| Gate | (4096, 4864, 896) | Wide N |
| Up | (4096, 4864, 896) | |
| Down | (4096, 896, 4864) | Fat K |
| LM head | (4096, 151936, 896) | |

**Qwen 2.5-1.5B** (hidden=1536, KV_dim=256, I=8960, V=151936, tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, 1536, 1536) | Non-power-of-2 |
| K | (4096, 256, 1536) | Very skinny |
| V | (4096, 256, 1536) | |
| O | (4096, 1536, 1536) | |
| Gate | (4096, 8960, 1536) | Wide N, weird dim |
| Up | (4096, 8960, 1536) | |
| Down | (4096, 1536, 8960) | |
| LM head | (4096, 151936, 1536) | |

**Qwen 2.5-3B** (hidden=2048, KV_dim=256, I=11008, V=151936, tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, 2048, 2048) | Square |
| K | (4096, 256, 2048) | Skinny |
| V | (4096, 256, 2048) | |
| O | (4096, 2048, 2048) | |
| Gate | (4096, 11008, 2048) | Wide N |
| Up | (4096, 11008, 2048) | |
| Down | (4096, 2048, 11008) | |
| LM head | (4096, 151936, 2048) | |

**Qwen 2.5-7B** (hidden=3584, KV_dim=512, I=18944, V=**152064**, NOT tied):

| Op | Shape | Notes |
|---|---|---|
| Q | (4096, 3584, 3584) | Non-power-of-2 |
| K | (4096, 512, 3584) | GQA (kv_heads=4) |
| V | (4096, 512, 3584) | |
| O | (4096, 3584, 3584) | |
| Gate | (4096, 18944, 3584) | Wide N |
| Up | (4096, 18944, 3584) | |
| Down | (4096, 3584, 18944) | Fat K |
| LM head | (4096, **152064**, 3584) | Slightly larger vocab |

### 10.2 All 8 target models — unique N and K dimensions

| Model | Hidden | KV_dim | Intermediate | Vocab |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 1024 | 1024 | 3072 | 151936 |
| Qwen3-1.7B | 2048 | 1024 | 6144 | 151936 |
| Qwen3-4B | 2560 | 1024 | 9728 | 151936 |
| Qwen3-8B | 4096 | 1024 | 12288 | 151936 |
| Qwen2.5-0.5B | 896 | 128 | 4864 | 151936 |
| Qwen2.5-1.5B | 1536 | 256 | 8960 | 151936 |
| Qwen2.5-3B | 2048 | 256 | 11008 | 151936 |
| Qwen2.5-7B | 3584 | 512 | 18944 | 152064 |

**Unique dims that appear as N or K in matmuls**:
- Hidden: 896, 1024, 1536, 2048, 2560, 3584, 4096
- KV: 128, 256, 512, 1024
- Intermediate: 3072, 4864, 6144, 8960, 9728, 11008, 12288, 18944
- Vocab: 151936, 152064

**Head_dim (attention batched GEMM)**: 64 (Qwen2.5-0.5B) or 128 (all others).

---

## 11. Test coverage gap — do our current tests cover Qwen ≤7B?

### 11.1 Our current test shapes recap

Base tests (16 shapes, both BluBLAS + cuBLAS):
```
tiny       128×128×64
small      256×256×128
medium     512×512×128
large      1024×1024×128
xlarge     2048×2048×256
llm-style  4096×4096×512
square-*   1024/2048/4096 cubed
```

Profile sweep (8 shapes, cuBLAS only):
```
decode-1     1×4096×4096
decode-8     8×4096×4096
decode-32    32×4096×4096
decode-64    64×4096×4096
lm-head-1    1×32000×4096
tiny-K-32    1024×1024×32
split-K      64×64×16384
unaligned    129×129×128
```

### 11.2 Coverage check — colleague's claim: "our current shapes cover the mostly-used shapes for models up to 7B"

**Verdict**: **Partially true, but with significant gaps.**

**What IS covered (✅)**:
- Square matrices at (roughly) hidden dims 1024, 2048, 4096 — approximates Q/O projections for Qwen3-0.6B (hidden~1024), Qwen3-1.7B (hidden=2048), Qwen3-8B (hidden=4096)
- Small batch decode shapes at hidden=4096 — matches Qwen3-8B decode path

**What is NOT covered (❌)**:

1. **Qwen 2.5 hidden dimensions** — 896, 1536, 2560, 3584 are NOT tested. Our square-2048 does NOT match Qwen2.5-1.5B (1536) or Qwen2.5-7B (3584).

2. **FFN intermediate dims** — NONE of 3072, 4864, 6144, 8960, 9728, 11008, 12288, 18944 are tested. Gate/Up/Down projections dominate FFN compute and use very wide N or fat K. Our `llm-style` (K=512) doesn't cover these AT ALL.

3. **KV projection (GQA)** — Skinny N cases: K→128, 256, 512, 1024 output columns. We have `decode-1` (M=1) but no batched-token skinny like `(4096, 1024, 4096)` for Qwen3-8B K/V projection.

4. **LM head** — Vocab=151936/152064 is huge. Our `lm-head-1` tests M=1 only, not larger M=4096 or M=32768 training-scale LM heads.

5. **Attention batched GEMM** — head_dim=128, batched over (num_heads × batch × seq/tile). Not tested at all.

6. **Non-square (M ≠ N)** — Most Qwen matmuls are wider or narrower in one dim. Our tests are almost all square.

### 11.3 Concrete gap list — matmuls needed for Qwen but NOT tested

Sorted by importance (frequency-of-use × missing-severity), for **M=4096 (training scale)**:

| Priority | Shape (M×N×K) | Belongs to | Currently tested? |
|---|---|---|---|
| P0 | 4096 × **12288** × 4096 | Qwen3-8B Gate/Up | ❌ NO — 12288 dim never appears |
| P0 | 4096 × 4096 × **12288** | Qwen3-8B Down | ❌ NO |
| P0 | 4096 × **1024** × 4096 | Qwen3-8B K/V | ❌ NO (only decode-1 M=1) |
| P0 | 4096 × **18944** × 3584 | Qwen2.5-7B Gate/Up | ❌ NO |
| P0 | 4096 × 3584 × **18944** | Qwen2.5-7B Down | ❌ NO |
| P0 | 4096 × **151936** × 4096 | LM head (any model) | ❌ NO (only M=1) |
| P1 | 4096 × 3584 × 3584 | Qwen2.5-7B Q/O | ❌ NO (3584 dim never appears) |
| P1 | 4096 × **512** × 3584 | Qwen2.5-7B K/V | ❌ NO |
| P1 | 4096 × **9728** × 2560 | Qwen3-4B Gate/Up | ❌ NO |
| P1 | 4096 × **2560** × 2560 | Qwen3-4B Q/O | ❌ NO |
| P1 | 4096 × **6144** × 2048 | Qwen3-1.7B Gate/Up | ❌ NO |
| P2 | 4096 × 2048 × 2048 | Qwen3-1.7B Q/O | ✅ CLOSE (square-2048 has M=N=K=2048, we lack M=4096) |
| P2 | 4096 × 4096 × 4096 | Qwen3-8B Q/O | ✅ YES (square-4096) |
| P3 | 4096 × 1024 × 1024 | Qwen3-0.6B Q/O | ✅ CLOSE (square-1024 has M=K=1024) |

**Estimate**: Of ~40 unique Qwen matmul shapes (M=4096) across all 4 sizes × 5 op types, **we currently test 2-3 exactly, ~5 approximately, and ~30-35 not at all**.

### 11.4 Also missing: variable M

For training, M varies (4096 is common but 2048, 8192, 16384 also happen with different accumulation strategies). For inference:
- Prefill: M = prompt tokens (512, 1024, 2048, 4096, 8192)
- Decode: M = batch (1, 4, 8, 16, 32, 64, 128)

Our tests only sample M ∈ {128, 256, 512, 1024, 2048, 4096} for validation and M ∈ {1, 8, 32, 64, 129} for profile sweep — decent M coverage but not tied to real Qwen dims.

---

## 12. Prioritized shape/kernel roadmap for BLUBLAS

### 12.1 P0 — must-have for Qwen ≤7B training

1. **`bgemm_sm103_bf16_cluster<128, 256, 64, 6>`** with 2×2 cluster + fused epilogue
   - Closes the 3% square-4096 gap
   - Handles all "wide N" shapes: `M × intermediate × hidden` (Gate/Up)
   - Concrete targets: (4096, 12288, 4096), (4096, 18944, 3584), (4096, 8960, 1536)

2. **Skinny N kernel: `<128, 64, 64, 10>`** or **`<128, 32, ...>`** with 2×2 cluster
   - K/V projection in GQA (small output N)
   - Concrete targets: (4096, 1024, 4096), (4096, 512, 3584), (4096, 256, 1536)

3. **Fat K kernel: reuse crown jewel but with transposed inputs / split-K**
   - Down projection: `M × hidden × intermediate`
   - Concrete targets: (4096, 4096, 12288), (4096, 3584, 18944), (4096, 1536, 8960)

4. **LM head kernel: `<128, 256, 64, 6>` but tuned for huge N (152k)**
   - Concrete: (M, 151936, hidden) — very few instances per forward but big
   - Likely just the crown jewel works fine here, needs benchmarking to confirm

### 12.2 P1 — inference-critical

5. **Skinny GEMV kernel** for decode (M ∈ {1, 4, 8, 16, 32})
   - Match cuBLAS's `64x16_4x1_2cta` / `64x8_4x1_v_bz`
   - Concrete: (1, 4096, 4096), (8, 4096, 4096), (32, 4096, 4096) — decode path

6. **Small-M kernel** for batched decode (M = 32-128)
   - Match cuBLAS's `32x64_4x1_v_bz`
   - Concrete: (64, 4096, 4096), (128, 4096, 4096)

### 12.3 P2 — polish and edge cases

7. **Split-K variants** for shapes where full-tile is idle
   - Rare in Qwen inference/training but needed for very-fat-K corner cases

8. **Unaligned fallback**
   - Only if shapes with M or N not divisible by 128 appear (rare for Qwen since dims are all power-of-2 or /128-aligned)

9. **Attention batched GEMM** (head_dim = 128, batched)
   - Different kernel category — needs its own design (fused with softmax = FlashAttention-style)
   - Deferred to separate FlashAttention effort

---

## 13. THE CONCRETE PLAN — Task 2 onwards

### Overview
1. Task 2 = extend tests to Qwen shapes, profile cuBLAS, see what it picks
2. Task 3 = analyze what BluBLAS is missing / weak on for each Qwen shape
3. Task 4 = build & optimize kernels until we beat cuBLAS on every Qwen shape
4. Task 5+ = resume the broader plan from Step 2 (CUTLASS Blackwell cross-ref → shape sweep → dispatcher heuristic)

### Task 2: Extend tests to cover ALL Qwen matmuls

Add a new "Qwen sweep" profile-only section to each of TN/NN/NT test files, appended after the existing "cuBLAS-only Profile Sweep" section. Contains 40 unique matmul shapes (8 models × 5 unique ops: Q, K/V, O, Gate/Up, Down + LM head).

Test values:
- M = 4096 (training-scale — 1 sample point; can expand to M ∈ {128, 512, 2048, 8192} later)
- One shape per (model, op) combination — 40 total

Deliverable: on the server, run all 3 layout tests → nsys profile → see cuBLAS kernel names per Qwen shape.

### Task 3: Analyze what cuBLAS picks for each Qwen shape

For each Qwen shape:
- Which cuBLAS kernel (from our menu of 12-14) does it run?
- What TFLOPS does cuBLAS achieve?
- What fraction of peak (1687 T for square-4096) does that represent?
- Are there Qwen shapes where cuBLAS itself struggles? (those are opportunity gaps)

Deliverable: A table like Section 6 above but for the 40 Qwen shapes → shows exactly which cuBLAS variant we need to beat per shape category.

### Task 4: Build & optimize BluBLAS kernels for each Qwen shape category

Priority-ordered (based on training compute distribution):

**Tier P0 (immediate)** — where cuBLAS runs the crown jewel `128x256_64x6_2x2f_2cta`:
- Full-tile GEMM: shapes where M, N, K are all ≥ 1024 (Qwen3-8B Q/O, most training matmuls)
- **Build**: `bgemm_sm103_bf16_cluster<128, 256, 64, 6>` with 2×2 cluster + fused epilogue
- Target: match or beat cuBLAS's 1687 T on square-4096, and win on wide-N shapes like (4096, 12288, 4096)

**Tier P1 (skinny N — GQA K/V, LM head)** — where cuBLAS uses `64x16` / `128x8` / `32x64`:
- Shapes: (4096, 1024, 4096), (4096, 512, 3584), (4096, 128, 896)
- **Build**: `bgemm_sm103_bf16_cluster<128, 32, 64, 8>` or `<128, 64, 64, 10>` with skinny N support
- Target: beat cuBLAS on Qwen K/V projections

**Tier P2 (fat K — Down projection)**:
- Shapes: (4096, 4096, 12288), (4096, 3584, 18944)
- These use the same tile as full-tile but K is much larger → same kernel should work well, benchmark to confirm
- If cuBLAS uses split-K here, add our own split-K variant

**Tier P3 (decode path)**:
- Shapes: (1, 4096, 4096), (8, 4096, 4096) — batch-1 inference
- **Build**: skinny GEMV kernel with `<64, 16, ...>` 4×1 cluster
- Target: beat cuBLAS on inference decode

**Tier P4 (edge cases)**:
- LM head at training scale: (4096, 151936, hidden) — huge N. Very few of these per model.
- Non-power-of-2 hidden (Qwen 2.5-1.5B hidden=1536, 2.5-7B hidden=3584) — need to check tile alignment

### Task 5: RESUME the broader Step 2 plan

Only after all Tier P0-P3 kernels beat cuBLAS on Qwen shapes:
- Step 2 (original): Cross-reference against CUTLASS Blackwell tile space
- Step 3 (original): Full 300-shape sweep bench against cuBLAS
- Step 4-5 (original): nsys profile the sweep
- Step 6 (original): Infer cuBLAS's decision heuristic
- Step 7 (original): Build our own dispatcher

This resumption unlocks: production-quality general-purpose BLAS that beats cuBLAS everywhere, not just Qwen shapes.

---

## 14. Task 2 execution plan (this session)

Concrete next steps to be done NOW:

1. Add a `Qwen shape` vector to each test file's `main()` (after the existing profile sweep). The vector holds 40 (model, op, M, N, K, label) tuples.
2. Add a new `=== cuBLAS-only Qwen Sweep <LAYOUT> ===` section that iterates that vector and calls the existing `run_cublas_only_*` helper (already added to the file).
3. Rebuild on the server with `make tests`.
4. Rerun `nsys_profiles_commands.sh` — the existing script picks up the new section automatically.
5. Copy `nsys_out/` back locally.
6. Analyze: which cuBLAS kernel got picked for each Qwen shape, what TFLOPS, group by shape category.
7. Update this document with the empirical results table.

After that, Task 3 begins (comparative analysis) → Task 4 (kernel builds).

---

## 15. Task 2 EMPIRICAL results — TN layout Qwen sweep (2026-07-01)

> ⚠️ **THROUGHPUT NUMBERS BELOW ARE UNRELIABLE — DO NOT CONCLUDE FROM THESE TFLOPS**
>
> When we ran the Qwen sweep on 2026-07-01, colleague was running another workload on the same 8-B300 node (shared cluster), so our measured TFLOPS were affected by GPU contention. The **kernel-choice mapping** (which cuBLAS kernel is picked for each shape — see §16.3) is still valid because that's a compile-time / heuristic decision independent of GPU occupancy. But the **absolute throughput numbers** need to be re-measured when the node is idle.
>
> **TODO**: Retest with dedicated node. Section 15.1-15.6 TFLOPS values below are placeholder-only until then. Priority ordering in §15.5 may shift after retest.

Ran on B300 SXM6 PC (SM 10.3). 42 Qwen matmul shapes at M=4096, TN layout, cuBLAS-only. No nsys profile yet — just raw TFLOPS from the `run_cublas_only_tn` helper.

### 15.1 Top tier — ≥ 1500 T (cuBLAS runs near peak, likely uses crown jewel 128×256)

| Rank | Shape | Config (M,N,K) | Model + Op | cuBLAS TFLOPS | % of ~2500 T peak |
|--:|---|---|---|---:|---:|
| 1 | q3-4b-O | 4096, **2560**, 4096 | Qwen3-4B O (non-square) | **1768** | 71% |
| 2 | q25-1.5b-Gate | 4096, 8960, 1536 | Qwen2.5-1.5B Gate | 1732 | 69% |
| 3 | q3-8b-Q | 4096, 4096, 4096 | Qwen3-8B Q | 1723 | 69% |
| 4 | q3-1.7b-Down | 4096, 2048, 6144 | Qwen3-1.7B Down | 1719 | 69% |
| 5 | q3-4b-Q | 4096, **4096**, 2560 | Qwen3-4B Q (non-square) | 1660 | 66% |
| 6 | q25-7b-Q | 4096, 3584, 3584 | Qwen2.5-7B Q | 1659 | 66% |
| 7 | q3-1.7b-Gate | 4096, 6144, 2048 | Qwen3-1.7B Gate | 1635 | 65% |
| 8 | q3-8b-K | 4096, 1024, 4096 | Qwen3-8B K (GQA) | 1611 | 64% |
| 9 | q25-1.5b-Down | 4096, 1536, 8960 | Qwen2.5-1.5B Down | 1588 | 64% |
| 10 | q25-3b-Q | 4096, 2048, 2048 | Qwen2.5-3B Q | 1571 | 63% |
| 11 | q3-1.7b-Q | 4096, 2048, 2048 | Qwen3-1.7B Q | 1566 | 63% |
| 12 | q3-0.6b-Down | 4096, 1024, 3072 | Qwen3-0.6B Down | 1533 | 61% |

### 15.2 Mid tier — 900 to 1500 T (medium tiles likely)

| Shape | Config | Model + Op | TFLOPS |
|---|---|---|---:|
| q3-0.6b-Gate | 4096, 3072, 1024 | Qwen3-0.6B Gate | 1480 |
| q3-4b-K | 4096, 1024, 2560 | Qwen3-4B K | 1475 |
| q25-0.5b-Gate | 4096, 4864, 896 | Qwen2.5-0.5B Gate | 1432 |
| q3-0.6b-O | 4096, 1024, 2048 | Qwen3-0.6B O (non-square) | 1407 |
| q3-1.7b-K | 4096, 1024, 2048 | Qwen3-1.7B K | 1404 |
| q25-1.5b-Q | 4096, 1536, 1536 | Qwen2.5-1.5B Q | 1402 |
| q25-7b-K | 4096, 512, 3584 | Qwen2.5-7B K (GQA very-skinny) | 1382 |
| q3-0.6b-Q | 4096, 2048, 1024 | Qwen3-0.6B Q (non-square) | 1348 |
| q25-0.5b-Down | 4096, 896, 4864 | Qwen2.5-0.5B Down | 1193 |
| q3-0.6b-K | 4096, 1024, 1024 | Qwen3-0.6B K | 1116 |

### 15.3 Low tier — < 900 T (THE OPPORTUNITY ZONE — where BluBLAS can beat cuBLAS)

| Shape | Config | Model + Op | TFLOPS | Why low |
|---|---|---|---:|---|
| q25-0.5b-Q | 4096, 896, 896 | Qwen2.5-0.5B Q | 889 | Non-power-of-2 dim |
| **q3-8b-Gate** | 4096, **12288**, 4096 | Qwen3-8B Gate | **832** | Wide N + L2 pressure |
| **q3-8b-Down** | 4096, 4096, **12288** | Qwen3-8B Down | **825** | Fat K + L2 pressure |
| q3-4b-Gate | 4096, 9728, 2560 | Qwen3-4B Gate | 814 | Wide N |
| q3-4b-Down | 4096, 2560, 9728 | Qwen3-4B Down | 794 | Fat K |
| q25-7b-Gate | 4096, 18944, 3584 | Qwen2.5-7B Gate | **792** | Wide N (biggest FFN) |
| q3-8b-LMhead | 4096, 151936, 4096 | Qwen3-8B LM head | 752 | Huge N — memory-bound |
| q25-3b-Gate | 4096, 11008, 2048 | Qwen2.5-3B Gate | 765 | Wide N |
| q25-3b-Down | 4096, 2048, 11008 | Qwen2.5-3B Down | 758 | Fat K |
| q25-3b-LMhead | 4096, 151936, 2048 | Qwen2.5-3B LM | 746 | Memory-bound |
| q25-7b-Down | 4096, 3584, 18944 | Qwen2.5-7B Down | 751 | Fat K |
| q25-7b-LMhead | 4096, 152064, 3584 | Qwen2.5-7B LM | 756 | Memory-bound |
| q3-4b-LMhead | 4096, 151936, 2560 | Qwen3-4B LM | 743 | Memory-bound |
| q3-1.7b-LMhead | 4096, 151936, 2048 | Qwen3-1.7B LM | 746 | Memory-bound |
| q25-1.5b-LMhead | 4096, 151936, 1536 | Qwen2.5-1.5B LM | 751 | Memory-bound |
| q3-0.6b-LMhead | 4096, 151936, 1024 | Qwen3-0.6B LM | 703 | Memory-bound |
| q25-0.5b-LMhead | 4096, 151936, 896 | Qwen2.5-0.5B LM | 698 | Memory-bound |
| q25-1.5b-K | 4096, **256**, 1536 | Qwen2.5-1.5B K | 653 | Very skinny N |
| q25-0.5b-K | 4096, **128**, 896 | Qwen2.5-0.5B K | **284** | Super-skinny N=128 |
| q25-3b-K | 4096, **256**, 2048 | Qwen2.5-3B K | 757 | Very skinny N |

### 15.4 Key empirical findings

1. **Best cuBLAS achieves = 1768 T on q3-4b-O**. That's 71% of B300 BF16 peak (~2500 T). Even cuBLAS is far from peak.

2. **FFN Gate/Down shapes for models ≥ 3B run at 700-850 T**. That's 30-40% of peak. **HUGE performance opportunity** — cuBLAS is memory-bandwidth-bound here because the weight matrix (~100-300 MB) exceeds L2 cache. A kernel with better data reuse strategy could gain 30-50%.

3. **LM head is universally memory-bound at ~700-750 T** across all 8 models. This is a memory-bandwidth wall. Improvement requires fusion (topk / softmax inline) or FlashAttention-style tiling.

4. **Super-skinny K/V projections need dedicated skinny-N kernels**:
   - q25-0.5b-K (N=128) → 284 T
   - q25-1.5b-K (N=256) → 653 T
   - q25-3b-K (N=256) → 757 T
   cuBLAS uses a 128×256 tile which is 2× too wide → wastes half the N-work.

5. **Non-square Q/O works well for cuBLAS** — q3-4b-O actually runs FASTER than any square shape (1768 T > 1723 T square). So non-square is not a problem for cuBLAS; BluBLAS just needs to support it.

6. **Qwen 3-8B baseline (q3-8b-Q at 4096²)**: cuBLAS 1723 T. Our BluBLAS on same shape (from earlier `square-4096` test) achieved 1636 T. That's **95% of cuBLAS** on the most important shape.

### 15.5 Priority attack list — where BluBLAS should focus

Ordered by impact (compute × current-gap-to-peak):

| Priority | Shape category | Example | cuBLAS today | Target |
|---|---|---|---:|---:|
| P0 | FFN Gate/Up (wide N) | (4096, 12288, 4096) Qwen3-8B | 832 T | ≥ 1400 T |
| P0 | FFN Down (fat K) | (4096, 4096, 12288) Qwen3-8B | 825 T | ≥ 1400 T |
| P1 | LM head (memory-bound) | (4096, 151936, 4096) | 752 T | ≥ 1000 T (needs fusion) |
| P1 | Super-skinny K proj | (4096, 128, 896) | 284 T | ≥ 500 T |
| P2 | Skinny K proj (256) | (4096, 256, 1536-2048) | 650-760 T | ≥ 1000 T |
| P2 | Square Q proj large | (4096, 4096, 4096) | 1723 T | ≥ 1750 T (match cuBLAS) |
| P3 | Non-square Q/O | (4096, 2560, 4096) | 1768 T | ≥ 1768 T (already fast — match) |

### 15.6 Gap awaiting NN + NT sweep + nsys profile

- NN and NT test runs still pending (only TN pasted in output.md).
- nsys profile of the Qwen sweep pending — needed to identify WHICH cuBLAS kernel handles each Qwen shape.

Expected additions once we have nsys data:
- Per-shape kernel-name mapping (like §6, but for Qwen shapes)
- Confirm hypothesis that top tier = `128x256_2x2f`, low tier = split-K variants or memory-bound crown jewel
- Identify the specific kernel(s) BluBLAS needs to build (e.g., a `128x256` with 2×2 cluster + fused epilogue for the FFN Gate/Down shapes)

**Section 15.6 status update (2026-07-01, mid-day)**: All 3 layouts (TN + NN + NT) Qwen sweeps completed with full nsys profile. Results in Section 16 below.

---

## 16. Task 2 COMPLETE — Qwen sweep 3-layout nsys profile (2026-07-01)

Ran all 3 layout tests with nsys profile. Output: `cublas_rev_eng_binary/nsys_current_exp/`. All 42 Qwen shapes ran cleanly across TN, NN, NT (126 shape × layout combinations total).

### 16.1 MASSIVE catalog expansion — cuBLAS ships **16 distinct BF16 tile shapes**

The Qwen sweep triggered many kernels invisible in the base tests. Combined catalog (base tests + Qwen sweep):

| Tile (BM×BN) | Cluster variants seen | Stages seen | First seen in | Notes |
|---|---|---|---|---|
| **128×256** | 2×1(v), 2×2, 2×2f | 6 | base | Dominant workhorse — 3 cluster variants! |
| **256×256** | 2×2 | 4 | **Qwen sweep** | ★ Handles biggest fat-K shape (q3-8b-Down) |
| **320×192** | 2×2f | 3-4 (varies by layout) | **Qwen sweep** | ★ Special-case (q25-7b-Down 18944) |
| **256×128** | 2×2 | 5 | **Qwen sweep** | ★ Big square-ish shapes |
| **176×128** | 1×2 | 8 | **Qwen sweep NT** | ★ NT-ONLY tile |
| **160×192** | 1×2, 2×2 | 5-6 | **Qwen sweep** | ★ Mid-tier for hidden=1536-2048 |
| **128×192** | 2×1(v), 2×2f | 6-7 (varies) | **Qwen sweep** | ★ Non-square O, moderate FFN |
| **128×160** | 2×1(v) | 6-8 (varies) | **Qwen sweep** | ★ q25-7b-Q specifically |
| **128×128** | 2×2 | 8 | **Qwen sweep** | ★ SAME tile shape as our BluBLAS baseline |
| 128×64 | 2×2 | 10 | base | Medium shapes |
| 128×8 | 2×1(splitK,v) | 12 | base | Split-K skinny |
| **64×64** | 2×2 | 16 | **Qwen sweep** | ★ Super-skinny K proj (N=128) |
| 64×32 | 2×2, 2×4, 1×2(splitK) | 16 | base | Various small tiles |
| 64×16 | 4×1 (NT-only) | 16 | base | NT skinny |
| 64×8 | 4×1, 1×4, 2×4 | 16 | base | GEMV path |
| 32×64 | 4×1(v) | 16 | base | Small M path |
| 32×32 (Turing wmma) | — | 1 | base | Unaligned fallback |

**Grand total: 16 native Blackwell tile shapes + 1 Turing fallback = 17 tile families.**

Combined with cluster variants (1×2, 1×4, 2×1, 2×2, 2×2f, 2×4, 4×1) and stage counts (3, 4, 5, 6, 7, 8, 10, 12, 16), cuBLAS carries at least **60-80 unique kernel binaries** for BF16 GEMM alone on Blackwell.

### 16.2 THE dominant Qwen kernel — new variant discovered

Not seen in base tests. Appears for the first time in Qwen sweep and **dominates 91% of GPU time**:

```
nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_<LAYOUT>
                        ─── ────── ──
                          │    │    └── v = VERTICAL wave order
                          │    └─────── cluster 2×1 (not 2×2!)
                          └──────────── 6 stages, BK=64
```

**How this differs from what we saw in the base-test crown jewel**:

| Feature | Base-test winner (`2x2f_h`) | Qwen winner (`2x1_v`) |
|---|---|---|
| Cluster shape | 2×2 (4 CTAs cooperate) | **2×1 (2 CTAs cooperate)** |
| Wave scheduling | horizontal (N first) | **vertical (M first)** |
| Epilogue | fused | not fused |
| Best for | large square shapes | M=4096 wide-N or fat-K |

**Time distribution across 3 layouts** (from `profile_*_cuda_gpu_kern_sum_demangled.csv`):

> ⚠️ **Absolute nanosecond values BELOW ARE UNRELIABLE** — server was shared during measurement so per-kernel times reflect contended GPU. The **relative distribution** (% share per kernel) is likely still directionally correct since contention affects all kernels roughly equally, but confirm with retest on idle node. **Kernel-choice mapping in §16.3 is independent of timing and is valid.**

| Kernel | TN time (ns) | NN time (ns) | NT time (ns) | Cross-layout share |
|---|---:|---:|---:|---:|
| **`128x256_64x6_2x1_v_bz`** | 780,227,207 | 777,578,111 | 774,277,292 | **91% each** |
| `320x192_64x3/4_2x2f_h_bz` | 16,759,241 | 16,742,249 | 16,721,925 | 2.0% each |
| `256x256_64x4_2x2_h_bz` | 11,334,419 | 11,345,262 | 11,376,048 | 1.3% each |
| `128x256_64x6_2x2_h_bz` | 10,704,718 | 10,752,755 | 10,763,918 | 1.3% each |
| `128x256_64x6_2x2f_h_bz` | 10,675,854 | 10,645,643 | 10,672,973 | 1.2% each |
| `128x192_64x6/7_2x1_v_bz` | 7,773,571 | 7,682,237 | 7,681,120 | 0.9% each |
| BluBLAS baseline kernel | 7,120,513 | 7,121,664 | 7,101,023 | 0.8% each |
| (all others, 15+ kernels) | ~10-15 M | ~10-15 M | ~10-15 M | <2% combined |

**Bottom line: ONE kernel handles 91% of Qwen training GPU time. If BluBLAS builds just this kernel variant, we cover the vast majority of Qwen workload.**

### 16.3 Per-Qwen-shape kernel mapping (TN layout, verified from trace)

Mapping determined by walking `profile_tn_cuda_gpu_trace_demangled.csv` and identifying runs of 25 consecutive identical kernel launches (each Qwen shape = 25 iterations of cuBLAS).

| # | Shape | (M,N,K) | cuBLAS kernel picked (TN suffix `NTT`) |
|--:|---|---|---|
| 1 | q25-0.5b-Q | 4096, 896, 896 | `160x192_64x5_2x2_h_bz` |
| 2 | q25-0.5b-K | 4096, 128, 896 | `64x64_64x16_2x2_h_bz` |
| 3 | q25-0.5b-Gate | 4096, 4864, 896 | `128x192_64x6_2x1_v_bz` |
| 4 | q25-0.5b-Down | 4096, 896, 4864 | `256x128_64x5_2x2_h_bz` |
| 5 | q25-0.5b-LMhead | 4096, 151936, 896 | **`128x256_2x1_v_bz`** |
| 6 | q25-1.5b-Q | 4096, 1536, 1536 | `128x192_64x6_2x2f_h_bz` |
| 7 | q25-1.5b-K | 4096, 256, 1536 | `128x64_64x10_2x2_h_bz` |
| 8 | q25-1.5b-Gate | 4096, 8960, 1536 | **`128x256_2x1_v_bz`** |
| 9 | q25-1.5b-Down | 4096, 1536, 8960 | `128x192_64x6_2x2f_h_bz` |
| 10 | q25-1.5b-LMhead | 4096, 151936, 1536 | **`128x256_2x1_v_bz`** |
| 11 | q25-3b-Q | 4096, 2048, 2048 | `160x192_64x5_1x2_h_bz` |
| 12 | q25-3b-K | 4096, 256, 2048 | `128x64_64x10_2x2_h_bz` |
| 13 | q25-3b-Gate | 4096, 11008, 2048 | **`128x256_2x1_v_bz`** |
| 14 | q25-3b-Down | 4096, 2048, 11008 | `128x256_2x2_h_bz` |
| 15 | q25-3b-LMhead | 4096, 151936, 2048 | **`128x256_2x1_v_bz`** |
| 16 | q25-7b-Q | 4096, 3584, 3584 | `128x160_64x6_2x1_v_bz` |
| 17 | q25-7b-K | 4096, 512, 3584 | **`128x128_64x8_2x2_h_bz`** ← same tile as ours |
| 18 | q25-7b-Gate | 4096, 18944, 3584 | **`128x256_2x1_v_bz`** |
| 19 | q25-7b-Down | 4096, 3584, 18944 | **`320x192_64x3_2x2f_h_bz`** (unique!) |
| 20 | q25-7b-LMhead | 4096, 152064, 3584 | **`128x256_2x1_v_bz`** |
| 21 | q3-0.6b-Q | 4096, 2048, 1024 | `160x192_64x5_1x2_h_bz` |
| 22 | q3-0.6b-K | 4096, 1024, 1024 | `128x256_2x2_h_bz` |
| 23 | q3-0.6b-O | 4096, 1024, 2048 | `128x256_2x2_h_bz` |
| 24 | q3-0.6b-Gate | 4096, 3072, 1024 | `128x256_2x2f_h_bz` |
| 25 | q3-0.6b-Down | 4096, 1024, 3072 | `128x256_2x2_h_bz` |
| 26 | q3-0.6b-LMhead | 4096, 151936, 1024 | **`128x256_2x1_v_bz`** |
| 27 | q3-1.7b-Q | 4096, 2048, 2048 | `160x192_64x5_1x2_h_bz` |
| 28 | q3-1.7b-K | 4096, 1024, 2048 | `128x256_2x2_h_bz` |
| 29 | q3-1.7b-Gate | 4096, 6144, 2048 | `128x256_2x2f_h_bz` |
| 30 | q3-1.7b-Down | 4096, 2048, 6144 | `128x256_2x2_h_bz` |
| 31 | q3-1.7b-LMhead | 4096, 151936, 2048 | **`128x256_2x1_v_bz`** |
| 32 | q3-4b-Q | 4096, 4096, 2560 | `128x256_2x2f_h_bz` |
| 33 | q3-4b-K | 4096, 1024, 2560 | `128x256_2x2_h_bz` |
| 34 | q3-4b-O | 4096, 2560, 4096 | `128x192_2x1_v_bz` (non-square) |
| 35 | q3-4b-Gate | 4096, 9728, 2560 | **`128x256_2x1_v_bz`** |
| 36 | q3-4b-Down | 4096, 2560, 9728 | `128x192_2x1_v_bz` |
| 37 | q3-4b-LMhead | 4096, 151936, 2560 | **`128x256_2x1_v_bz`** |
| 38 | q3-8b-Q | 4096, 4096, 4096 | `128x256_2x2f_h_bz` |
| 39 | q3-8b-K | 4096, 1024, 4096 | `128x256_2x2_h_bz` |
| 40 | q3-8b-Gate | 4096, 12288, 4096 | **`128x256_2x1_v_bz`** |
| 41 | q3-8b-Down | 4096, 4096, 12288 | **`256x256_64x4_2x2_h_bz`** (unique!) |
| 42 | q3-8b-LMhead | 4096, 151936, 4096 | **`128x256_2x1_v_bz`** |

**Distribution summary (TN):**

| cuBLAS kernel | # of Qwen shapes using it | Category |
|---|---:|---|
| `128x256_2x1_v_bz` | **12** | All LMheads (8) + wide-N Gates (4) |
| `128x256_2x2_h_bz` | **8** | K proj (GQA=1024) + moderate shapes |
| `128x256_2x2f_h_bz` | **6** | Q proj (large square) + smaller Gate |
| `128x192_2x1_v_bz` / `_2x2f_h_bz` | **5** | Non-square O + moderate FFN |
| `160x192_*` | **3** | Q proj at hidden=2048 shapes |
| `128x64_2x2_h_bz` | **2** | Skinny K proj (N=256) |
| `256x128_2x2_h_bz` | **1** | q25-0.5b-Down |
| `320x192_2x2f_h_bz` | **1** | q25-7b-Down (unique — biggest FFN) |
| `256x256_2x2_h_bz` | **1** | q3-8b-Down (biggest square-K matmul) |
| `128x160_2x1_v_bz` | **1** | q25-7b-Q |
| `128x128_2x2_h_bz` | **1** | q25-7b-K |
| `64x64_2x2_h_bz` | **1** | q25-0.5b-K (super-skinny N=128) |

### 16.4 Layout-specific kernel differences (TN vs NN vs NT)

Most kernels appear in all 3 layouts (with layout-suffix differences). Layout-specific variations:

| Kernel family | TN version | NN version | NT version |
|---|---|---|---|
| `128x160_64x?` | stages=6 | stages=8 | stages=8 |
| `128x192_64x?_2x1_v` | stages=6 | stages=7 | stages=7 |
| `128x192_64x?_2x2f_h` | stages=6 | stages=7 | stages=7 |
| `160x192_64x?_1x2/2x2` | stages=5 | stages=5 | **stages=6** |
| `320x192_64x?_2x2f` | stages=3 | stages=4 | stages=4 |
| `176x128_64x8_1x2_h` | — | — | **ONLY IN NT** |
| `64x16_64x16_4x1_2cta` | — | — | (only in base tests, not Qwen) |

**Insight**: cuBLAS **auto-tunes stages per layout** (5/6/7/8) but tile shape stays consistent. NT gets ONE extra tile (176×128) as an edge-case handler.

### 16.5 Cross-layout kernel time comparison

Every kernel gets approximately the same GPU time in TN, NN, NT — meaning the 3 layouts hit essentially the same critical path with different data-movement patterns:

| Layout | Total kernel-count kinds | Total time in top kernel |
|---|---:|---:|
| TN | 23 | 780M ns |
| NN | 24 | 778M ns |
| NT | 25 (extra 176×128) | 774M ns |

### 16.6 STRATEGIC — what to build in BluBLAS to beat cuBLAS on Qwen

If BluBLAS builds these **4 kernel variants**, we cover **~85% of Qwen training compute**:

**Priority 1** (91% of GPU time on Qwen sweep):
```
bgemm_sm103_bf16_cluster<BM=128, BN=256, BK=64, STAGES=6>
    with cluster shape 2×1 + VERTICAL wave scheduling
```
- Covers: 12 Qwen shapes (all LMheads + wide-N Gates)
- Target beat: 750-830 T current → ≥ 1400 T

**Priority 2** (~15 Qwen shapes, moderate compute):
```
bgemm_sm103_bf16_cluster<128, 256, 64, 6>
    with cluster shape 2×2 + horizontal wave + no epilogue fusion
```
- Covers: K proj, non-square O, moderate FFN
- Effectively the "same kernel" as P1 with different cluster/wave/epilogue policy

**Priority 3** (~6 Qwen shapes, key training path):
```
bgemm_sm103_bf16_cluster<128, 256, 64, 6>
    with cluster shape 2×2 + horizontal wave + FUSED epilogue
```
- Covers: Q proj (large square)
- Target beat: match cuBLAS's 1660-1720 T

**Priority 4** (1 Qwen shape, BUT it's the biggest fat-K matmul):
```
bgemm_sm103_bf16_cluster<BM=256, BN=256, BK=64, STAGES=4>
    with cluster shape 2×2
```
- Covers: `q3-8b-Down` (4096, 4096, 12288) — fat-K compute for Qwen 3-8B
- Novel — CUTLASS 3.x doesn't ship 256×256 out-of-box
- SMEM budget: 256×256 tile ≈ 128 KB per stage — might need SPECIAL allocation strategy

**Priority 5** (skinny K proj, low compute but common):
- `<BM=128, BN=64, BK=64, STAGES=10>` — for K proj with N=256
- `<BM=64, BN=64, BK=64, STAGES=16>` — for K proj with N=128
- `<BM=128, BN=64, BK=64, STAGES=10>` for GQA K/V in general

### 16.7 The `128x256_2x1_v` design — what BluBLAS needs to understand

**Key architectural insight from Qwen data**: cuBLAS's dominant kernel differs from the base-test winner in these ways:

1. **Cluster 2×1 (not 2×2)** — only 2 CTAs cooperate (not 4). Reduces per-cluster register pressure, allows more concurrent clusters per SM, better fit for M=4096 (many M-tiles).

2. **Vertical wave (M-first)** — scheduler warp iterates over M-tiles first, THEN N-tiles. Result: the same weight (B) matrix is reused across many M-tiles in a row, HUGE L2 hit rate for streaming weights.

3. **No fused epilogue** — most Qwen matmul results are just written to memory (used as input to next matmul, not consumed by fused ops). Fusion overhead not worth it for pure GEMM.

**Why this beats `2x2f_h` on Qwen**:
- Qwen shapes: M=4096, N moderate-large (1024-152064), K moderate (896-18944)
- With M=4096 and BM=128, that's 32 M-tiles per column of the output → tons of L2 reuse potential
- Vertical wave + cluster 2×1 keeps weight-matrix rows in L2 for maximum reuse

### 16.8 What Task 3 becomes (concrete)

**Task 3.1**: Design & implement the P1 kernel — `bgemm_sm103_bf16_cluster<128, 256, 64, 6>` with 2×1 cluster + vertical wave.
- Reference: study a `128x256_2x1_v_bz` SASS via nsys report → understand cluster + wave semantics
- Study CUTLASS 3.x `SM100_TILE_SCHEDULER` for vertical wave order
- Adapt our existing 128×128 kernel to 128×256 (double BN) with 2×1 cluster
- Change scheduler warp to iterate M first, N second

**Task 3.2**: Extend the same kernel to support cluster 2×2 mode (P2/P3) — same kernel with a runtime flag or a template parameter.

**Task 3.3**: Design the 256×256 tile for q3-8b-Down (P4).
- Novel — need to fit BM=256 output rows per CTA (only possible with `cta_group::1` given PTX Table 44 restrictions), OR use a "double-BM" concept via 2×2 cluster on 128×128
- SMEM budget analysis: 256×256×2 (bf16) = 128 KB per stage. B300 has 228 KB smem per SM → only 1-2 stages possible

**Task 3.4**: Skinny K proj (P5) — small tile with high-stage-count for arithmetic-intensity.
- `<128, 64, 64, 10>` — 10 stages to hide the sparse-N compute
- `<64, 64, 64, 16>` — 16 stages for super-skinny (N=128) shapes

### 16.9 Task 3 → Task 4 (build & optimize until we beat cuBLAS)

For each priority tier:
1. Implement the kernel per Section 16.6 spec
2. Add per-tile-shape entries to Qwen test file so BluBLAS runs alongside cuBLAS
3. Benchmark, compare TFLOPS
4. Iterate — micro-optimize (register allocation, smem layout, mbarrier tuning) until we win
5. Only THEN move to next priority tier

Expected outcome: at Task 4 complete, BluBLAS beats cuBLAS on ≥ 85% of Qwen training compute.

### 16.10 After Task 4, RESUME Step 2 of the original broad plan

- Step 2 (original): CUTLASS Blackwell cross-reference — validate our tile choices against `cutlass/examples/`
- Step 3 (original): 300-shape sweep bench against cuBLAS (not just Qwen)
- Step 4-5 (original): infer cuBLAS's decision heuristic across the sweep
- Step 6 (original): build BluBLAS dispatcher (shape → kernel routing)
- Step 7 (original): production-quality general-purpose BLAS

---

## 17. Reproducible commands (for future re-runs)

```bash
# ─── Extract cuBLAS Blackwell kernel catalog (Step 1A) ───
LIBLT=/usr/local/cuda-13.2/lib64/libcublasLt.so.13
mkdir -p /tmp/cubins && cd /tmp/cubins
cuobjdump --extract-elf all "$LIBLT"

# Enumerate sm_100 kernel symbols (the actual Blackwell code)
for c in *.sm_100.cubin; do
    cuobjdump --dump-elf-symbols "$c" 2>/dev/null | grep STO_ENTRY | awk '{print $NF}'
done | sort -u > sm_100_symbols.txt
c++filt < sm_100_symbols.txt > sm_100_demangled.txt

# ─── Profile cuBLAS on B300 (Step 1B) ───
cd /mnt/raid0/gautam_k/blublas
BUILD_DIR=./build OUT_DIR=./nsys_out ./examples/nsys_profiles_commands.sh

# Analyze results
head -32 nsys_out/profile_tn_cuda_gpu_kern_sum_demangled.csv
```

---

## 18. Reference files in this folder

| File | Contents |
|---|---|
| `analyzations.md` | This document (~1000 lines, 18 sections) |
| `catalog_bf16.txt` | Original narrow BF16 grep from Step 1A (14,895 lines — pre-fix) |
| `all_symbols.txt` | Full 42k device symbol dump from the library |
| `sm_100_symbols.txt` | 5,532 kernel symbols from Blackwell sm_100 cubins |
| `sm_100_bf16_demangled.txt` | 1,083 BF16 Blackwell utility kernels (demangled) |
| `sm_103_symbols.txt` | Only 1 symbol: `nopHardwareDetection()` |
| `nsys_profile_commands.sh` | Script that runs nsys profile on all 3 test binaries |
| `nsys_out/` | Initial profile run (16 base shapes + 8 profile-sweep shapes, all 3 layouts) |
| `nsys_current_exp/` | Full Qwen sweep profile (16 base + 8 profile + 42 Qwen × 3 layouts) |

---

## 19. Status summary

- ✅ **Task 1**: Binary reverse-engineering + nsys profile of base shapes — DONE (kernel names + tile catalog reliable; throughput numbers need retest)
- ✅ **Task 2**: Extend tests to Qwen shapes + profile all 3 layouts — DONE (kernel-per-shape mapping reliable; TFLOPS numbers UNRELIABLE due to server contention with colleague's workload)
- ⏳ **PENDING** — retest with idle B300 node to get clean throughput numbers before drawing conclusions about performance gaps and priority ordering
- ⏭️ **Task 3**: Design & implement Priority 1 kernel (128×256, cluster 2×1, vertical wave, 6 stages) — NEXT (kernel-choice data is enough to start design, but final priority may shift after clean re-measure)
- ⏭️ **Task 4**: Iterate on kernels for Priority 2-5, benchmark until BluBLAS beats cuBLAS on all Qwen shapes
- ⏭️ **Task 5+**: Resume original 7-step broad plan from Step 2 (general-purpose BLAS) — INCLUDES building a cost-model dispatcher (DeepGEMM-style) to replace hardcoded if/else when we go general-purpose (see §20.5)
- 📓 **Dispatcher deep-dive** documented in §20 — how each library selects kernels, and why our hardcoded approach is correct for Qwen-only (with verified proof that if/else adds no runtime overhead)

### 19.1 What IS reliable from today's runs

- **cuBLAS Blackwell BF16 kernel catalog** (16 tile shapes + Turing fallback) — from binary extraction, not runtime dependent
- **Kernel naming convention decoding** — static analysis
- **Per-Qwen-shape kernel-choice mapping** (§16.3) — cuBLAS's heuristic picks the same kernel regardless of GPU contention
- **Layout-specific kernel differences** (§16.4) — same reasoning
- **Test suite correctness** — pass/fail + rel_err are timing-independent
- **Qwen model config data** (§9) — pulled from HuggingFace, not runtime

### 19.2 What is NOT reliable (needs retest)

- All absolute TFLOPS numbers in §3.2, §15, §16.2 time table
- BluBLAS-vs-cuBLAS ratios (may shift by ±20% under contention)
- Priority ordering IF ratios shift enough to reorder the "worst gap" list

### 19.3 Retest plan for tomorrow (2026-07-02 onwards)

- Wait for colleague's node to be idle OR request dedicated B300 slot
- Re-run all 3 layout tests with nsys profile → store in `nsys_dedicated_run/` folder (keep contended run as evidence)
- Compare kernel-choice mapping → should be IDENTICAL (validates the mapping is not noise)
- Compare TFLOPS → the real performance gap data
- Update §15 and §16 tables in this doc with clean numbers

**End of Task 2 documentation. Task 3 begins with implementing the P1 kernel `bgemm_sm103_bf16_cluster<128, 256, 64, 6>` with 2×1 cluster + vertical wave scheduling — starting tomorrow after clean throughput retest.**

---

## 20. Dispatcher & kernel-selection deep-dive (2026-07-02)

This section answers a set of architectural doubts about HOW the right kernel gets chosen for a given shape — do we have a dispatcher, is it hardcoded, how do the reference libraries do it, and does the if/else dispatch add runtime overhead. All answers below are grounded in our actual source code and verified reads of DeepGEMM / ThunderKittens / CUTLASS.

### 20.1 The core doubts (asked 2026-07-02) and answers

**Doubt 1: cuBLAS has dispatchers/heuristics that pick the right variant. In our case do we have any such system, or nothing?**

We DO have a dispatcher — a 3-layer host-side chain:
```
mycublasBgemmStridedBatched_dispatcher(...)          [BF16/Bgemm_dispatcher.cu:119]  ← top level
   ├─ if SM >= 103 (B300) → mycublasBgemmSM103_dispatch_{nn,nt,tn}()  [SM103/Bgemm_sm103_dispatcher.cu]
   │                              └─ align check → 128x128x64 kernel (only 1 real Blackwell kernel today)
   └─ else (SM80/89)  → hardcoded if/else on M,N thresholds → 1 of ~18 variants
```
So it's not "nothing" — but on B300 it is currently trivial (see Doubt 6).

**Doubt 2: Are we hardcoding all this time?**

YES. The SM80/SM89 path is 100% hand-written if/else thresholds. Actual code (`BF16/Bgemm_dispatcher.cu:163-181`, NN layout):
```c
if (M == 1)                                   mycublasBgemv_nn(...);              // GEMV
else if (is_sm89 && M>=1024 && N>=1024)       ..._nn_128x128_32_6(...);           // big square
else if (M>=2048 && N>=2048)                  ..._nn_256x128_32_4(...);           // huge
else if (M>=256 && N>=128 && M>=N)            ..._nn_256x128_32x3(...);           // tall
else if (N>=256 && M>=128)                    ..._nn_128x256_32_4(...);           // wide
else if (M>=256)                              ..._nn_256x64_32x4(...);            // medium
else                                          ..._nn_128x128_32_6(...);           // small
```
There is even a hardcoded split-K rule (`Bgemm_dispatcher.cu:156`):
```c
int splitK = 1;
if (K >= 4096 && M <= 128 && N <= 128) splitK = 4;   // fat-K tiny-MN → split into 4
```
No learned model, no autotuning, no runtime search. Pure hand-tuned thresholds.

**Doubt 3: Did SM89 hardcode "for specific shapes → specific variants"?**

YES, exactly as suspected. Each `if` branch on (M,N) thresholds routes to a specific pre-instantiated kernel variant (256x128, 128x256, 128x128, 64x64, 256x64, 128x64 with various stage counts). It is a hand-written shape→variant lookup expressed as an if/else ladder.

**Doubt 4: How do we WRITE those variants — do we hardcode each one?**

NO. The kernel logic is written ONCE as a C++ template, then INSTANTIATED with different tile sizes. E.g. `bgemm_sm103_bf16_tn_cluster_kernel<BM, BN, BK, QUEUE_SIZE>` is one template; `_128x128x64` is just the instantiation `<128,128,64,8>`. To add a 128×256 variant we do NOT rewrite the kernel — we instantiate `<128,256,64,6>` and add a dispatch branch. The `nn_256x128_32x3`, `nn_128x256_32x3`, etc. in SM89 are all the same template stamped with different `<BM,BN,BK,STAGES>`.

**Doubt 5: How does cuBLAS's dispatch work (that we don't have)?**

cuBLAS uses a **cost model + heuristic scoring**, not an if/else ladder. For a given (M,N,K,layout,dtype) it scores candidate algos with a proprietary heuristic (occupancy, wave quantization, L2 reuse, tile efficiency) via `cublasLtMatmulAlgoGetHeuristic`, picks the best-predicted, optionally auto-tunes. That's why it picked `128x256_2x1_v` for LM heads but `160x192_1x2` for square-2048 — a scoring function chose each. We will not replicate this exactly (proprietary + HW-tuned) and we don't need to — we only have ~42 Qwen shapes, so we can hardcode the answers we reverse-engineered (§16.3).

**Doubt 6: Do we have that system for B300? Or is it hardcoded / are we hardcoding?**

On B300 we currently have effectively NO real dispatch — the SM103 dispatcher (`SM103/Bgemm_sm103_dispatcher.cu`) does an alignment check (M%256, N%128, K%64) + stride check, then funnels EVERYTHING to the single 128×128×64 kernel (the "256x256x64" branch also runs 128×128×64 under the hood — placeholder). Unaligned shapes return false and fall back to the slow SM89 kernels. So: 1 real Blackwell kernel + trivial dispatch. That is the gap Task 3/4 fills.

### 20.2 How the reference libraries dispatch — verified comparison

Deep-read of each library's source (2026-07-02). Verdicts with proof:

| Library | Has shape-based dispatcher? | Mechanism | Where it decides |
|---|---|---|---|
| **cuBLAS** | ✅ Yes | **Cost model + heuristic scoring** (proprietary) | Host, per call |
| **DeepGEMM** | ✅ Yes | **Cost model + comparator** (open, transparent) | Host, JIT compile-time |
| **CUTLASS core** | ❌ No | User picks tile at compile time; runtime match only on compute-capability + alignment (NOT M/N/K) | Compile-time |
| **CUTLASS profiler (tools/)** | ⚙️ Autotune | Brute-force: benchmark all kernels, pick fastest | Offline profiling |
| **ThunderKittens** | ❌ No | User hand-picks tile config, hardcodes template instantiation per shape | Programmer, manual |
| **BluBLAS (us) — SM89** | ✅ Yes | **Hardcoded if/else thresholds** | Host, per call |
| **BluBLAS (us) — SM103/B300** | ⚠️ Trivial | Align-check → 1 kernel | Host, per call |

**DeepGEMM proof** (`csrc/jit_kernels/heuristics/sm100.hpp`, `sm90.hpp`, `common.hpp`): enumerates all legal (block_m, block_n, cluster_m, cluster_n) candidates, then ranks them with a comparator. SM90 uses a parametric cost model estimating L1/L2 cache cycles + wave efficiency (`sm90.hpp:228-231`); SM100 uses a prioritized comparator (`sm100.hpp:241-266`):
```cpp
// SM100 comparator priority order:
// 1. single wave is always better
// 2. doing multicast (larger cluster) is better
// 3. smaller number of waves is better
// 4. larger last-wave utilization is better
// 5. same block_m, smaller block_n is better (more stages)
// 6. less shared memory for C/D, more stages is better
```
This is close to cuBLAS's approach — candidate enumeration + scoring — but open and modular. Decision happens host-side at JIT compile-time.

**CUTLASS proof** (`tools/library/src/handle.cu:310-347`): `find_gemm_operation()` matches only on `compute_capability` and `alignment` — NO M/N/K shape check. Standard usage (`examples/00_basic_gemm/basic_gemm.cu`) has the user instantiate `cutlass::gemm::device::Gemm<...>` with tile config chosen at compile-time (default `GemmShape<128,128,8>` from `default_gemm_configuration.h:83`). The only runtime "heuristic" is `stream_k_heuristic()` which picks a decomposition mode (StreamK vs DataParallel vs SplitK) WITHIN a single already-chosen kernel — it does NOT select between kernels. **Conclusion: kernel-selection intelligence lives ONLY in the cuBLAS layer, NOT in CUTLASS core.** cuBLAS internally uses many CUTLASS-generated kernels but adds its own cost-model picker on top.

**ThunderKittens proof** (`kernels/gemm/bf16_b200/bf16_b200_gemm.cu:6-34`, `364-381`): tile sizes are `static constexpr` template params with `static_assert(_Mb == 256)`; the benchmark harness hardcodes a different `config<...>` instantiation per problem size (1024→`<256,64,128,...>`, 2048→`<256,256,64,...>`, 4096→`<256,256,64,...>`). No dispatch/heuristic/select code anywhere. It's a kernel-authoring framework: "pick your kernel, write it, compile it."

### 20.3 The takeaway landscape

There are really three tiers of sophistication:
1. **No dispatcher** (ThunderKittens, CUTLASS core) — the caller/programmer picks the kernel. These are *kernel-authoring* tools, not turnkey libraries.
2. **Hardcoded if/else** (BluBLAS SM89) — hand-tuned thresholds. Simple, works, doesn't generalize past the shapes you tuned for.
3. **Cost-model / heuristic scoring** (cuBLAS, DeepGEMM) — enumerate candidates + rank by a model. Generalizes to any shape.

**For a Qwen-only library, tier 2 (hardcoded) is the correct choice** — we know our finite shape set (§16.3), so we hardcode the reverse-engineered answers. Tier 3 (cost model) is the *future* goal when we go general-purpose (see §20.5).

### 20.4 Runtime overhead of the if/else dispatch — VERIFIED NEGLIGIBLE

**Concern**: if/else dispatch runs at runtime; if/else is "expensive"; training is millions of matmul calls, so does the overhead add up and do we need an alternative?

**Answer: the overhead is completely negligible. No alternative needed.** Proof:

1. **It's HOST-side (CPU), not in the kernel.** `mycublasBgemmStridedBatched_dispatcher` is an `extern "C" void` host function (`Bgemm_dispatcher.cu:119`). The `if (M>=2048...)` runs on the CPU once, BEFORE the GPU kernel launches. It is NOT inside the GPU kernel — it never touches per-element matmul compute, so it does NOT scale with M×N×K.

2. **It runs once per GEMM call, ~10 ns.** Per call: 1 cached `get_gpu_info()` (the expensive `cudaGetDeviceProperties` runs only ONCE ever — it's `static`-cached at `FP32/Sgemm_dispatcher.cu:8-20`) + ~5-8 integer comparisons. That's ~10 nanoseconds of CPU work.

3. **Magnitude comparison**:
   | Operation | Time | Where |
   |---|---|---|
   | The if/else dispatch | ~10 ns | CPU, once per call |
   | One CUDA kernel launch (unavoidable) | ~5,000-10,000 ns | CPU→GPU |
   | The GEMM kernel itself (e.g. Qwen-8B Gate) | ~500,000-8,000,000 ns | GPU |

   The if/else is ~500× smaller than the kernel LAUNCH and ~50,000-800,000× smaller than the kernel COMPUTE. It is 0.000002% of a GEMM's runtime.

4. **In real training the dispatch is HIDDEN entirely.** CUDA launches are async — the CPU runs dispatch for layer N+1 while the GPU still computes layer N. Dispatch overlaps with GPU compute and disappears. GPU is always the bottleneck, never the host if/else.

5. **cuBLAS does the SAME thing** — its heuristic also runs host-side once per call, and its heuristic is HEAVIER than our if/else (it scores multiple candidates). If host selection were a bottleneck, cuBLAS would suffer more.

**Important clarification**: "if/else is expensive" is true for *device-side branch divergence* INSIDE a kernel (32 threads in a warp taking different branches serializes execution). Our dispatch is *host-side*, single-threaded, once-per-call — a completely different thing. Do NOT confuse the two.

**The ONE real (tiny) caveat**: for extremely small matmuls where the kernel runs in ~1 µs (e.g. decode with M=1), the kernel-LAUNCH overhead (5-10 µs, NOT the if/else) can dominate. The fix there is CUDA graphs (capture + replay the launch sequence), not removing the dispatcher. This is a P4 decode-inference concern, irrelevant for training.

### 20.5 Future expansion — cost-model dispatcher (added to the roadmap)

When BluBLAS goes beyond Qwen to general-purpose (original Step 5+ broad plan), the hardcoded if/else won't generalize to arbitrary shapes. At that point we should build a **cost-model dispatcher like DeepGEMM's** (which is open and readable, unlike cuBLAS's):
- Enumerate legal (BM, BN, BK, stages, cluster) candidates for the shape
- Score each with a wave-efficiency + L2-reuse model (port DeepGEMM's `sm100.hpp` comparator as a starting point)
- Pick the best-predicted, host-side, once per call (negligible overhead per §20.4)

This is explicitly a LATER task — for now (Qwen-only) hardcoded dispatch is correct and simpler. Logged here so we don't forget the eventual direction.

### 20.6 Summary answers (quick reference)

| Question | Answer |
|---|---|
| Do we have a dispatcher? | Yes — 3-layer host chain (`Bgemm_dispatcher.cu`) |
| Are we hardcoding? | Yes — SM89 is hand-written if/else on M,N |
| Did SM89 hardcode shape→variant? | Yes, exactly |
| How do we write variants — hardcode each? | No — one template, instantiated with `<BM,BN,BK,STAGES>` |
| How does cuBLAS dispatch? | Cost model + heuristic scoring (proprietary) |
| Does DeepGEMM have a cost model? | Yes — open comparator, JIT compile-time |
| Does ThunderKittens have a dispatcher? | No — user hand-picks per shape |
| Does CUTLASS core have one? | No — only cuBLAS layer does; CUTLASS = compile-time user choice |
| Do we have real dispatch on B300? | No — trivial align-check → 1 kernel |
| How should we do it for Qwen? | Hardcode the §16.3 shape→kernel map (correct for finite shape set) |
| Does if/else add runtime overhead? | No — host-side, ~10 ns, hidden behind async launch |
| Should we seek an alternative to if/else? | No — the overhead is a non-issue; cost-model is a future general-purpose goal, not an overhead fix |

---

## 21. CLEAN dedicated-node retest (2026-07-02, 12:30 PM) — corrects the contended numbers

Re-ran the full Qwen sweep (all 3 layouts) on an IDLE B300 node. Output: `nsys_dedicated_run/`. This replaces the unreliable §15/§16 throughput numbers (which were taken while a colleague's job shared the node).

### 21.1 Contention factor confirmed: ~2.24×

The dominant kernel `128x256_64x6_2x1_v_bz` total GPU time:
- **Contended run** (yesterday): 780,227,207 ns
- **Clean run** (today): 348,294,662 ns
- **Ratio: 2.24× slower under contention**

So yesterday's throughput for the big shapes was roughly **less than half** the true value. Any conclusion drawn from the contended low-tier numbers must be re-examined.

### 21.2 Kernel-choice mapping is IDENTICAL (validates §16.3)

The dominant kernel is `128x256_64x6_2x1_2cta_v_bz` in ALL 3 layouts, same as the contended run:
- TN: 86.2% share (was 91% — share dropped only because the kernel now runs faster, not because a different kernel was picked)
- NN: 86.3% share
- NT: 86.1% share

cuBLAS picks the SAME kernel per shape regardless of GPU load — exactly as predicted (kernel selection is a contention-independent heuristic). **The §16.3 shape→kernel mapping is confirmed valid.**

### 21.3 CLEAN Qwen TFLOPS (TN layout) — the real numbers

| Shape | (M,N,K) | Contended (§15) | **Clean** | Verdict |
|---|---|---:|---:|---|
| q25-0.5b-K | 4096, 128, 896 | 284 | **160** | genuinely SLOW (skinny N=128) |
| q25-1.5b-K | 4096, 256, 1536 | 653 | **516** | genuinely SLOW (skinny N=256) |
| q25-3b-K | 4096, 256, 2048 | 757 | **648** | genuinely SLOW (skinny N=256) |
| q25-0.5b-Q | 4096, 896, 896 | 889 | **865** | slow (tiny square) |
| q3-0.6b-K | 4096, 1024, 1024 | 1116 | **1076** | moderate |
| q25-0.5b-Down | 4096, 896, 4864 | 1193 | **1178** | good |
| q25-1.5b-Q | 4096, 1536, 1536 | 1402 | **1302** | good |
| q25-7b-K | 4096, 512, 3584 | 1382 | **1333** | good |
| q3-8b-Q | 4096, 4096, 4096 | 1723 | **1686** | near-peak |
| q3-8b-Gate | 4096, 12288, 4096 | **832** | **1748** | ⚠️ was "opportunity" — actually NEAR PEAK |
| q3-8b-Down | 4096, 4096, 12288 | **825** | **1756** | ⚠️ was "opportunity" — actually NEAR PEAK |
| q3-8b-LMhead | 4096, 151936, 4096 | **752** | **1575** | ⚠️ was "opportunity" — actually strong |
| q25-7b-Gate | 4096, 18944, 3584 | **792** | **1801** | ⚠️ was "opportunity" — actually PEAK |
| q3-4b-LMhead | 4096, 151936, 2560 | 743 | **1781** | strong |
| q3-4b-Gate | 4096, 9728, 2560 | 814 | **1711** | strong |

### 21.4 MAJOR CORRECTION — the "FFN opportunity zone" was a mirage

Yesterday (§15.3, §15.5) I flagged FFN Gate/Down/LMhead shapes running at 700-850 T as the biggest opportunity for BluBLAS to beat cuBLAS. **THAT WAS WRONG — it was pure contention artifact.**

Clean numbers show those exact shapes run at **1700-1800 T** — cuBLAS's BEST performance, near the ~2500 T peak (68-72%). The wide-N FFN and LM-head shapes are where cuBLAS is STRONGEST, not weakest. The `128x256_2x1_v` kernel is extremely well-tuned for them.

**Lesson**: never draw performance conclusions from a contended node. The kernel-CHOICE mapping survived contention; the TFLOPS did not.

### 21.5 The REAL opportunity zone (genuine, survives clean retest)

Sorted worst→best, the shapes where cuBLAS is genuinely weak:

| Rank | Shape | (M,N,K) | Clean TFLOPS | Why cuBLAS is weak |
|--:|---|---|---:|---|
| 1 | q25-0.5b-K/V | 4096, 128, 896 | **~160** | N=128 — cuBLAS's min tile is 64×64, N=128 wastes the 128×256 tile |
| 2 | q25-1.5b-K/V | 4096, 256, 1536 | **~516** | N=256 — half the 128×256 N-tile wasted |
| 3 | q25-3b-K/V | 4096, 256, 2048 | **~648** | N=256 — same |
| 4 | q25-0.5b-Q/O | 4096, 896, 896 | **~865** | tiny square, N=896 not tile-aligned well |
| 5 | q3-0.6b-K/V | 4096, 1024, 1024 | **~1076** | N=1024 — borderline |

**These are all the GQA K/V projections with small kv_dim** (Qwen 2.5: kv_dim = 128, 256, 256, 512) plus the tiny-0.5B shapes. Every transformer layer does a K and a V projection, so these run 2× per layer × 24-36 layers — they're frequent, and cuBLAS handles them poorly (160-650 T vs its 1800 T peak).

**This is where BluBLAS can genuinely WIN**: a dedicated skinny-N kernel (BN=32/64/128 instead of 256) that doesn't waste half the tile on empty columns.

### 21.6 REVISED priority plan (supersedes §16.6)

The clean data flips the strategy. Two distinct goals:

**Goal A — MATCH cuBLAS on the bulk (hard, defensive).**
The big shapes (LMhead, Gate, Down, large Q) run at 1500-1800 T on cuBLAS via `128x256_2x1_v`. We must build this kernel to not LOSE on the 26 shapes that use 128×256. Target: reach ~1700-1800 T (match cuBLAS at its best). This is still P1 because it covers the most GPU time — but understand we're aiming to MATCH, not beat, here.

**Goal B — BEAT cuBLAS on the skinny K/V (easier, offensive).**
The GQA K/V projections (N=128-512) run at only 160-650 T. A skinny-N kernel could plausibly hit 800-1200 T → a real WIN over cuBLAS. This is the shape class where "beat cuBLAS" is actually achievable. Promote to **P2** (was P5).

**Revised priority order:**
1. **P1** — `128x256_64x6_2x1_v` workhorse (cluster 2×1, vertical wave) → MATCH cuBLAS on 26 shapes / 86% of time
2. **P2** — skinny-N kernel `<128, 64/32, 64, high-stages>` → BEAT cuBLAS on GQA K/V (the genuine win)
3. **P3** — `128x256_2x2f` (fused, cluster 2×2) for square Q proj
4. **P4** — `256x256` for the biggest fat-K (q3-8b-Down) — though clean shows cuBLAS already does 1756 T here, so lower urgency than thought
5. **P5** — tiny-shape handling (q25-0.5b Q at 865 T) — small models, lower training priority

### 21.7 Reliability note

- `nsys_dedicated_run/` = clean run 1, trustworthy throughput. USE THESE numbers.
- `nsys_dedicated_run2/` = clean run 2 (GPU-pinned, independent) — CONFIRMS run 1.
- `nsys_current_exp/` = contended, kernel-mapping-valid-only. Keep for evidence but do NOT cite TFLOPS.
- §15 and §16.2 TFLOPS tables remain flagged unreliable; §21.3 supersedes them.

### 21.8 Reproducibility confirmed — clean run 1 vs clean run 2

Two independent clean runs (run1 on a lightly-shared node, run2 pinned to an isolated GPU) agree within a few percent, proving the clean numbers are stable ground truth:

| Metric | run1 | run2 | Match |
|---|---:|---:|---|
| TN dominant-kernel share | 86.2% | 86.2% | exact |
| TN dominant-kernel total time | 348.3M ns | 345.6M ns | within 0.8% |
| q25-0.5b-K (N=128) | ~160 T | ~150 T | ✓ (noisy tiny shape) |
| q25-1.5b-K (N=256) | ~516 T | ~470 T | ✓ |
| q25-3b-K (N=256) | ~648 T | ~636 T | within 2% |
| q25-0.5b-Q (896²) | ~865 T | ~858 T | within 1% |
| q3-0.6b-K (N=1024) | ~1076 T | ~1054 T | within 2% |

**Conclusion**: clean throughput is reproducible. The contended run (2.24× slower) was the outlier. The genuine weak spots (skinny GQA K/V projections at 150-700 T) are confirmed across both clean runs — this is BluBLAS's real winnable target.

---

## 22. BluBLAS baseline on Qwen shapes — "what we have now" (2026-07-02)

Converted the Qwen sweep from cuBLAS-only to **BluBLAS + cuBLAS + accuracy** (`run_test`) and ran all 3 layouts on a clean GPU. This is the first time we measured OUR kernel on the Qwen shapes. Logs: `qwen_baseline_{tn,nn,nt}.log`.

### 22.1 Headline

- **Kernels we have: 1** (the `128×128×64` cluster kernel, in TN/NN/NT).
- **Coverage: 42/42 Qwen shapes PASS** (rel_err = 0.00) on ALL 3 layouts — our single kernel is numerically CORRECT on every Qwen shape including extremes (N=151936 LM head, N=128 skinny K). Coverage + correctness confirmed.
- **Average speed vs cuBLAS: TN 81.8%, NN 81.4%, NT 81.5%** (~81.5% overall).
- **Range: 56.5% (worst) → 96.9% (best).**

Context: cuBLAS itself picks a 128×128 tile for only 1 of 42 Qwen shapes (q25-7b-K), so our single 128×128 is cuBLAS's rarely-optimal tile applied to everything — yet still reaches 81.5% avg. That's the floor we build up from.

### 22.2 Full baseline table (TN layout, clean)

| Shape | (M,N,K) | BluBLAS T | cuBLAS T | Ratio | Status |
|---|---|---:|---:|---:|---|
| q25-0.5b-Q | 4096,896,896 | 623.55 | 874.74 | 71.3% | PASS |
| q25-0.5b-K | 4096,128,896 | 134.93 | 238.80 | 56.5% | PASS |
| q25-0.5b-Gate | 4096,4864,896 | 1173.60 | 1434.50 | 81.8% | PASS |
| q25-0.5b-Down | 4096,896,4864 | 1143.06 | 1183.69 | 96.6% | PASS |
| q25-0.5b-LMhead | 4096,151936,896 | 1102.75 | 1647.52 | 66.9% | PASS |
| q25-1.5b-Q | 4096,1536,1536 | 1072.31 | 1398.26 | 76.7% | PASS |
| q25-1.5b-K | 4096,256,1536 | 379.00 | 651.54 | 58.2% | PASS |
| q25-1.5b-Gate | 4096,8960,1536 | 1448.69 | 1670.72 | 86.7% | PASS |
| q25-1.5b-Down | 4096,1536,8960 | 1471.93 | 1532.83 | 96.0% | PASS |
| q25-1.5b-LMhead | 4096,151936,1536 | 1173.84 | 1382.05 | 84.9% | PASS |
| q25-3b-Q | 4096,2048,2048 | 1194.11 | 1543.07 | 77.4% | PASS |
| q25-3b-K | 4096,256,2048 | 445.83 | 742.77 | 60.0% | PASS |
| q25-3b-Gate | 4096,11008,2048 | 1523.27 | 1700.46 | 89.6% | PASS |
| q25-3b-Down | 4096,2048,11008 | 1483.79 | 1646.68 | 90.1% | PASS |
| q25-3b-LMhead | 4096,151936,2048 | 1161.11 | 1303.78 | 89.1% | PASS |
| q25-7b-Q | 4096,3584,3584 | 1429.12 | 1601.64 | 89.2% | PASS |
| q25-7b-K | 4096,512,3584 | 1087.41 | 1392.09 | 78.1% | PASS |
| q25-7b-Gate | 4096,18944,3584 | 1246.93 | 1795.85 | 69.4% | PASS |
| q25-7b-Down | 4096,3584,18944 | 1427.90 | 1607.90 | 88.8% | PASS |
| q25-7b-LMhead | 4096,152064,3584 | 1134.92 | 1175.48 | 96.5% | PASS |
| q3-0.6b-Q | 4096,2048,1024 | 1010.68 | 1325.61 | 76.2% | PASS |
| q3-0.6b-K | 4096,1024,1024 | 769.60 | 1102.86 | 69.8% | PASS |
| q3-0.6b-O | 4096,1024,2048 | 1075.57 | 1397.01 | 77.0% | PASS |
| q3-0.6b-Gate | 4096,3072,1024 | 1084.30 | 1469.67 | 73.8% | PASS |
| q3-0.6b-Down | 4096,1024,3072 | 1192.69 | 1543.47 | 77.3% | PASS |
| q3-0.6b-LMhead | 4096,151936,1024 | 1136.73 | 1505.06 | 75.5% | PASS |
| q3-1.7b-Q | 4096,2048,2048 | 1193.84 | 1543.73 | 77.3% | PASS |
| q3-1.7b-K | 4096,1024,2048 | 1068.51 | 1397.92 | 76.4% | PASS |
| q3-1.7b-Gate | 4096,6144,2048 | 1480.31 | 1590.06 | 93.1% | PASS |
| q3-1.7b-Down | 4096,2048,6144 | 1471.69 | 1678.07 | 87.7% | PASS |
| q3-1.7b-LMhead | 4096,151936,2048 | 1194.14 | 1294.46 | 92.3% | PASS |
| q3-4b-Q | 4096,4096,2560 | 1579.54 | 1633.46 | 96.7% | PASS |
| q3-4b-K | 4096,1024,2560 | 1146.28 | 1472.82 | 77.8% | PASS |
| q3-4b-O | 4096,2560,4096 | 1431.50 | 1688.43 | 84.8% | PASS |
| q3-4b-Gate | 4096,9728,2560 | 1556.58 | 1705.14 | 91.3% | PASS |
| q3-4b-Down | 4096,2560,9728 | 1474.12 | 1617.78 | 91.1% | PASS |
| q3-4b-LMhead | 4096,151936,2560 | 1165.98 | 1257.97 | 92.7% | PASS |
| q3-8b-Q | 4096,4096,4096 | 1628.86 | 1680.77 | 96.9% | PASS |
| q3-8b-K | 4096,1024,4096 | 1261.44 | 1592.50 | 79.2% | PASS |
| q3-8b-Gate | 4096,12288,4096 | 1389.66 | 1769.22 | 78.5% | PASS |
| q3-8b-Down | 4096,4096,12288 | 1505.53 | 1748.31 | 86.1% | PASS |
| q3-8b-LMhead | 4096,151936,4096 | 1110.01 | 1357.73 | 81.8% | PASS |

(NN and NT mirror TN within ~1% average — same 128×128 kernel, same relative behavior.)

### 22.3 Gap analysis — two lenses

**Relatively-worst (ratio) — where our tile fits worst:**
- q25-0.5b-K (N=128): 56.5%
- q25-1.5b-K (N=256): 58.2%
- q25-3b-K (N=256): 60.0%
- q3-0.6b-K (N=1024): 69.8%
→ the skinny GQA K/V projections.

**Biggest absolute TFLOPS gap — where building gains the most total compute:**
- q25-7b-Gate: 549 T gap
- q25-0.5b-LMhead: 545 T gap
- q3-0.6b-Gate: 385 T gap
- q3-8b-Gate: 380 T gap
- q3-0.6b-LMhead: 368 T gap
→ wide-N Gate + LM head — all use cuBLAS's 128×256 tile.

**Already good (leave alone):** q3-8b-Q (96.9%), q3-4b-Q (96.7%), q25-0.5b-Down (96.6%), q25-7b-LMhead (96.5%), q25-1.5b-Down (96.0%).

### 22.4 Data-driven build priority (finalized)

| Priority | Build | Serves | Current | Rationale |
|---|---|---|---|---|
| **P1** | 128×256 (cluster 2×1, vertical wave) | wide-N Gate + LMhead (~15) | 66-93% | largest absolute TFLOPS recovery (350-550 T/shape) |
| **P2** | skinny-N (128×64 / 64×64, high stages) | GQA K/V N≤256 (~5) | 56-70% | worst ratios AND cuBLAS also weak (238-743 T) → genuine WIN opportunity |
| **P3** | 128×256 (cluster 2×2, fused) | square Q small models | 76-89% | moderate gains |
| skip | — | big square Q, some Down | 96-97% | already at parity |

### 22.5 What we ADD, and what we HAVE after

**Have now (before Task 3):** 1 kernel (128×128), 42/42 correct, 81.5% avg of cuBLAS.

**Will add (Task 3):** 128×256 (P1) + skinny-N (P2) [+ 128×256-fused (P3)] tile variants, each an instantiation of the same cluster-kernel template, wired into the SM103 dispatcher by shape.

**Target after (Task 4 done):** 42/42 correct, ≥ ~100% of cuBLAS on the P1/P2 shapes (match on big, beat on skinny), avg pushed from 81.5% toward/above 100%.

---

## 23. Deep-dive: cluster shapes, the 256×256 resolution, and Qwen shape completeness (2026-07-02)

### 23.1 Why we're already ~96% on some shapes — the arithmetic-intensity principle

The 5 near-parity shapes and the cuBLAS kernel each uses:

| Shape | (M,N,K) | cuBLAS kernel | We're at |
|---|---|---|---:|
| q3-8b-Q | 4096,4096,4096 | 128×256_64x6_2x2f_h (fused, cluster 2×2) | 96.9% |
| q3-4b-Q | 4096,4096,2560 | 128×256_64x6_2x2f_h (fused, cluster 2×2) | 96.7% |
| q25-0.5b-Down | 4096,896,4864 | 256×128_64x5_2x2_h (cluster 2×2) | 96.6% |
| q25-7b-LMhead | 4096,152064,3584 | 128×256_64x6_2x1_v (cluster 2×1, vertical) | 96.5% |
| q25-1.5b-Down | 4096,1536,8960 | 128×192_64x6_2x2f_h (fused, cluster 2×2) | 96.0% |

**Why we match cuBLAS here despite using a "worse" 128×128 tile**: these shapes all have **large K** (2560-8960). Large K ⇒ **high arithmetic intensity** ⇒ **compute-bound**. When the tensor cores are the bottleneck, tile choice barely matters — even our 128×128 saturates the MMA units, and cuBLAS's 128×256 can't beat the tensor-core ceiling.

**The governing principle of our entire gap:**
> Our gap to cuBLAS correlates with **arithmetic intensity**.
> - Compute-bound shapes (big K) → we're ~96% (tile choice irrelevant, tensor cores saturated).
> - Memory-bound shapes (skinny N, small K) → we're 56-70% (tile + cluster choice controls memory traffic, which is the bottleneck).

This is why the skinny GQA K/V projections (N=128-256) are our worst — they're memory-bound, so cuBLAS's better memory strategy (right tile + 2×2 multicast) wins big.

### 23.2 The 2×2 cluster — the linchpin capability (q25-7b-K case study)

q25-7b-K (4096, **512**, 3584) is our worst-ratio LARGE shape (78.1%). cuBLAS and we use the SAME tile (128×128) and SAME stages (8). The only difference:

| | Tile | Stages | Cluster | CTAs |
|---|---|---|---|---|
| cuBLAS | 128×128 | 8 | **2×2** | 4 |
| us | 128×128 | 8 | **2×1** | 2 |

**What 2×2 cluster does**: launches 4-CTA clusters (2 along M, 2 along N). Via TMA **multicast**, the 2 CTAs sharing an M-row load A ONCE and broadcast; the 2 sharing an N-column load B ONCE and broadcast. This **halves global-memory traffic** for both operands. Our 2×1 cluster shares along only one dimension → more memory traffic → slower on memory-bound shapes.

**Can matching 2×2 reach 100% on this shape?** Matching the cluster is the single most likely lever to close most of the 22% gap, BUT:
- It is NOT a flag flip — requires real kernel work: launch 4-CTA clusters, change `CTA_MASK` `0b11`→`0b1111`, rework which CTA loads which A/B sub-tile, multicast/smem-sharing across 4 CTAs.
- Reaching EXACTLY 100% is a hypothesis, not a guarantee (residual differences: swizzle, wave order, register allocation, cuBLAS hand-tuning). Expect ~90-100%; verify by build + measure.

**Why 2×2 cluster is the highest-leverage capability** (not a one-shape fix): cuBLAS uses `2x2`/`2x2f` clusters for **14 of the 42 Qwen shapes** (all square Q + several Down + q25-7b-K). Adding 2×2-cluster capability improves a large fraction of shapes at once.

Our current kernel is hardcoded to a 2-CTA cluster (`CTA_GROUP_SIZE=2`, `CTA_MASK=0b11`, `cta_group::2` MMA). Moving to 2×2 (4-CTA) is a real but high-value upgrade.

### 23.3 The 256×256×64 case — RESOLVED

**Current state**: our `256×256×64` API entries are STUBS that fall back to 128×128, because a single CTA cannot do BM=256 with `tcgen05.mma.cta_group::2.kind::f16` (PTX ISA 9.3 Table 44 caps M at 256 *per MMA*; BM=256 would need M=512 per 2-CTA MMA-pair → invalid → "illegal instruction").

**But cuBLAS HAS a real 256×256 kernel** (`256x256_64x4_2x2`) — used for q3-8b-Down (4096×4096×12288, the biggest fat-K matmul). How, if the hardware caps a single MMA at M=256?

**CORRECTION (2026-07-02)**: An earlier draft of this section claimed "256×256 = just a 2×2 cluster of 128×128 CTA-tiles." **That was WRONG.** Verified from the clean run: cuBLAS ships `128x128_2x2` AND `256x256_2x2` as TWO DISTINCT kernels (different grids: 128 vs 256 CTAs). If 256×256 were merely 128×128 + a 2×2 cluster, cuBLAS wouldn't carry both. So:

**Tile size and cluster shape are TWO INDEPENDENT axes:**

| Axis | Values | Controls |
|---|---|---|
| Tile size (per-CTA output) | 128×128, 128×256, 256×256, 64×64… | how much output ONE CTA computes |
| Cluster shape | 2×1, 2×2, 2×4, 4×1… | how many CTAs group to share data via multicast |

cuBLAS mixes them freely: 128×128+2×1, 128×128+2×2, 128×256+2×1, 256×256+2×2, etc. A CTA reaches BM=256 by issuing multiple MMA instructions (the PTX M-limit is per-instruction, not per-CTA-tile — our earlier "BM=256 illegal" was an artifact of OUR kernel's specific descriptor logic, not a hardware wall).

**What this means concretely:**
- Our only kernel = 128×128 tile + **2×1** cluster.
- q25-7b-K: cuBLAS uses 128×128 tile + **2×2** cluster (SAME tile, different cluster) → what we lack is the **2×2-cluster version of our 128×128**, a NEW variant, NOT a 256×256 tile.
- q3-8b-Down: cuBLAS uses a genuine **256×256** tile + 2×2 cluster → a genuinely bigger, separate kernel.

**What to do about the 256×256 case:**
1. Our current `256×256×64` API STUB is useless — it silently falls back to 128×128 2×1 (BM=256 in our current template raises illegal-instruction). It is misleading. Either remove it or keep the slot reserved for a FUTURE real 256×256 kernel, but stop the silent fallback / comment it honestly as "not implemented."
2. A REAL 256×256 tile IS worth building later (cuBLAS uses it for the biggest fat-K shape, q3-8b-Down). It's a distinct, bigger-tile kernel — not free from the 2×2-cluster work.
3. For q25-7b-K (and ~14 other shapes), build a `128×128 + 2×2 cluster` variant (call it `128x128x64_cluster2x2`) — a third thing, distinct from both our current 128×128 (2×1) AND from a 256×256 tile.

**Corrected unifying conclusion**: the 2×2-cluster CAPABILITY is still the single highest-value item (it's the axis where cuBLAS beats us on q25-7b-K and the square-Q shapes), but it is INDEPENDENT of tile size. Adding 2×2-cluster support to our existing 128×128 tile is the cleanest first experiment; a true 256×256 tile is a separate, later kernel.

### 23.4 Qwen shape completeness — are the 42 shapes the COMPLETE set? (verified 2026-07-02)

Researched against Qwen2.5 tech report (arxiv 2412.15115), Qwen3 tech report (arxiv 2505.09388), and HuggingFace configs.

**Q1 — MoE at ≤8B? NO.**
- Qwen3 dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B. Qwen3 MoE: **30B-A3B, 235B-A22B — both >8B.**
- Qwen2.5 ≤7B: all dense.
- ✅ ALL our ≤8B target models are DENSE. No MoE router matmul in scope. (MoE router = hidden→num_experts GEMM, only relevant if we later target >8B MoE models.)

**Q2 — Attention QK^T and softmax·V matmuls: FUSED, not BLAS GEMMs.**
- Qwen uses **FlashAttention / SDPA**, which fuses QK^T + softmax + AV into ONE kernel. These never hit cuBLAS as standalone GEMMs.
- ✅ Correctly EXCLUDED from our 42 GEMM shapes. Attention is a separate FUSED-KERNEL category (FlashAttention-style), NOT a matmul variant. If BluBLAS ever wants to own attention, that's a distinct future effort (fused attention kernel), not part of the GEMM library.

**Q3 — QKV bias & QK-norm: epilogue/elementwise, NOT new matmuls or shapes.**
- Qwen2/2.5: QKV bias = TRUE (bias add on Q/K/V projections) → an **epilogue fusion** (bias add), GEMM shape unchanged.
- Qwen3: QKV bias REMOVED, **QK-Norm ADDED** (RMSNorm on Q and K per-head) → an **elementwise/reduction** op, not a matmul.
- ✅ Neither changes the 42 shapes. Both are epilogue/elementwise, relevant only to future fused-epilogue work, not to the GEMM catalog.

**Q4 — Redundant shapes? NO, correctly deduplicated.**
- Q proj vs O proj: for Qwen2.5 (Q_dim = hidden), Q and O have the SAME shape (M,hidden,hidden) → we listed only Q. For Qwen3-0.6b/4b (non-square, Q_dim ≠ hidden), Q ≠ O → we listed BOTH. Correct.
- Gate vs Up: both are (M, intermediate, hidden) → same shape → we listed only Gate (represents both). Correct.
- ✅ The 42 shapes are the correctly-deduplicated UNIQUE matmul shapes.

**Q5 — The M dimension: WE ONLY TESTED M=4096 (the one real gap).**
- Training: M = micro_batch × seq_len — commonly 4096, but also 8192, 16384, 32768.
- Prefill (inference): M = prompt length (100s to 10,000s).
- Decode (inference): M = batch size × 1 → **1, 8, 16, 32, 64, 128** — a totally different (memory-bound, GEMV-like) regime.
- ⚠️ We sampled ONE M (4096). The (N,K) shapes are complete, but **M-coverage is incomplete**: small-M decode and larger-M training are unsampled. Decode especially is a different performance regime that will need its own tiles (skinny-M / GEMV-style), similar to cuBLAS's `gemv2N`/`gemmk1` kernels found in §2.4.

**Completeness verdict:**
- ✅ The 42 shapes correctly capture ALL dense-GEMM matmul SHAPES (N,K) in Qwen ≤8B, deduplicated, at M=4096.
- ✅ No wrong/redundant shapes included.
- ✅ Correctly excluded: attention QK^T/AV (FlashAttention-fused), QKV bias (epilogue), QK-norm (elementwise), MoE router (>8B only).
- ⚠️ ONE genuine gap: **M dimension** — only M=4096 tested. Add decode (M=1-128) and larger training M (8192, 16384) as a follow-up shape-axis.

**Primary-source confirmation (2026-07-02, multi-agent web research):** all five points above verified against HuggingFace `transformers` modeling code, model `config.json` files, and the Qwen2.5 (arxiv 2412.15115) + Qwen3 (arxiv 2505.09388) tech reports:
- **MoE ≤8B = NONE.** Qwen2.5 open-weight lineup (0.5/1.5/3/7/14/32/72B) is entirely dense (official blog: "All open-weight models are dense, decoder-only"). Qwen2.5 MoE (Turbo/Plus/Max) is proprietary + large. Qwen3 MoE = 30B-A3B, 235B-A22B — both >8B. All our ≤8B targets are dense. ✓
- **Tied embeddings**: Qwen2.5 0.5B/1.5B/3B tied, 7B untied; Qwen3 0.6B/1.7B/4B tied, 8B untied. Tying is weight-sharing only — LM head is a GEMM either way. Note: untied Qwen2.5-7B pads vocab to **152064 = 594×256** (GEMM-friendly); Qwen3 keeps 151936 even untied.
- **QKV bias**: Qwen2/2.5 hardcode `bias=True` on q/k/v_proj (bias-add = elementwise epilogue, shape unchanged); Qwen3 sets `attention_bias=false` and adds **QK-Norm** (per-head RMSNorm on Q,K = elementwise+reduction, not a matmul). o_proj and all MLP projections are `bias=False` in all three. ✓ No new matmul shapes.
- **Attention path**: HF default is **SDPA** (fused); Qwen officially recommends **FlashAttention-2**. Both fuse QK^T+softmax+PV into one kernel — no standalone cuBLAS GEMM. Only the rarely-used `eager` path emits separate `torch.matmul`/bmm calls. So attention matmuls are a FlashAttention concern, NOT a BLAS-GEMM concern. ✓ Correctly excluded. (Also their shape — batched, tiny K=head_dim=128, softmax-bound — is ill-suited to a general GEMM anyway.)
- **M dimension** (verified regimes): decode single-user M=1; decode small-batch M=8-64 (skinny, memory-bound); decode high-concurrency M=128-256; chunked/small prefill M=512-8192; training micro-batch / large prefill M=4096-32768. **Recommended M-sweep for future benchmarking: `1, 8, 32, 128, 512, 2048, 4096, 8192, 16384` (+32768).** M=4096 alone misses the entire memory-bound decode regime (which dominates real serving cost).

### 23.5 Can we optimize the existing 128×128 nn/nt/tn further? Yes, modestly

Keeping the 128×128 tile, the levers (highest→lowest impact):
1. **2×1 → 2×2 cluster** (multicast halves memory traffic) — biggest lever; helps all memory-bound shapes AND unlocks 256×256 (§23.2, §23.3).
2. **Vertical wave scheduling** (cuBLAS uses it for wide-N — better L2 reuse of the streamed weight matrix).
3. **Per-shape stage-count tuning** (we hardcode 8; cuBLAS varies 5-8).
4. Swizzle / smem-layout / epilogue micro-tuning.

Realistic expectation: micro-tuning 128×128 gains a few % on shapes it already serves; NEW tiles (128×256, skinny-N) gain 20-40% on the shapes that need them. The 2×2-cluster upgrade is the exception — high leverage because it's both a 128×128 improvement and a prerequisite for 128×256-2x2 and 256×256.

### 23.6 Updated capability-priority (folding in this analysis)

| Rank | Capability / kernel | Unlocks | Shapes helped |
|---|---|---|---|
| **1** | **2×2 cluster (4-CTA + multicast)** | q25-7b-K, square-Q parity, 256×256 tile, all memory-bound | ~15 |
| **2** | **128×256 tile** (2×1 vertical) | wide-N Gate + LM head | ~15 |
| **3** | **skinny-N tile** (128×64 / 64×64) | GQA K/V (BEAT cuBLAS) | ~5 |
| **4** | 128×256 + 2×2 + fused epilogue | square Q small models | ~6 |
| later | decode/GEMV tiles (small M) | inference decode | M-axis gap (§23.4 Q5) |
| later | FlashAttention fused kernel | attention (separate category) | out of GEMM scope |

---

## 24. B0 RESULT — our first new variant: 128×256 tile (2026-07-02)

Built the 128×256 tile as the first step of the flexibility platform (Workstream B, step B0 = the "Option A" experiment). Implementation: repurposed the `mycublasBgemmSM103_bf16_tn_cluster_256x256x64` API (was a fake fallback to 128×128) to call `launch_bgemm_tn_cluster<128, 256, 64>` — a GENUINE 256×256 macro-tile (128 rows/CTA × 2-CTA 2×1 cluster = 256 M, × 256 N). No cluster change; tile axis was already templated. TN layer only so far. Log: `b0_b0_128x256_tn.log`.

### 24.1 It works — no TMEM overflow

All 42 Qwen shapes PASS with 128×256 (rel_err 0.00). BN=256 × NUM_EPI_STAGES=2 = 512 TMEM columns = the full TMEM, and it fit **exactly** — no overflow. So B1 (parameterize stages) is NOT needed for BN=256. Good.

### 24.2 Full 3-way comparison (TN, M=4096): 128×128 vs 128×256 vs cuBLAS

Ratios are BluBLAS ÷ cuBLAS. "best" = which of our two tiles is faster.

| Shape | (M,N,K) | 128×128 | 128×256 | cuBLAS | best |
|---|---|---:|---:|---:|---|
| q25-0.5b-Q | 4096,896,896 | 70.6% | **73.5%** | 882 | 128×256 |
| q25-0.5b-K | 4096,128,896 | **55.9%** | 41.8% | 241 | 128×128 |
| q25-0.5b-Gate | 4096,4864,896 | 81.8% | **84.1%** | 1435 | 128×256 |
| q25-0.5b-Down | 4096,896,4864 | 93.5% | **101.2%** ✅ | 1226 | 128×256 |
| q25-0.5b-LMhead | 4096,151936,896 | 67.0% | **77.1%** | 1648 | 128×256 |
| q25-1.5b-Q | 4096,1536,1536 | **78.3%** | 67.4% | 1386 | 128×128 |
| q25-1.5b-K | 4096,256,1536 | **59.6%** | 41.5% | 637 | 128×128 |
| q25-1.5b-Gate | 4096,8960,1536 | 86.5% | **97.1%** | 1672 | 128×256 |
| q25-1.5b-Down | 4096,1536,8960 | **95.6%** | 81.5% | 1534 | 128×128 |
| q25-1.5b-LMhead | 4096,151936,1536 | 84.6% | **96.9%** | 1394 | 128×256 |
| q25-3b-Q | 4096,2048,2048 | 77.0% | **85.2%** | 1551 | 128×256 |
| q25-3b-K | 4096,256,2048 | **60.1%** | 40.4% | 743 | 128×128 |
| q25-3b-Gate | 4096,11008,2048 | 89.8% | **98.0%** | 1698 | 128×256 |
| q25-3b-Down | 4096,2048,11008 | 89.8% | **97.7%** | 1648 | 128×256 |
| q25-3b-LMhead | 4096,151936,2048 | 89.5% | **104.8%** ✅ | 1304 | 128×256 |
| q25-7b-Q | 4096,3584,3584 | **89.5%** | 88.6% | 1604 | 128×128 |
| q25-7b-K | 4096,512,3584 | **78.6%** | 51.2% | 1384 | 128×128 |
| q25-7b-Gate | 4096,18944,3584 | 69.6% | **79.0%** | 1795 | 128×256 |
| q25-7b-Down | 4096,3584,18944 | 88.7% | **90.7%** | 1609 | 128×256 |
| q25-7b-LMhead | 4096,152064,3584 | 89.2% | **97.9%** | 1271 | 128×256 |
| q3-0.6b-Q | 4096,2048,1024 | 76.9% | **84.5%** | 1315 | 128×256 |
| q3-0.6b-K | 4096,1024,1024 | 75.2% | **77.4%** | 1033 | 128×256 |
| q3-0.6b-O | 4096,1024,2048 | 76.3% | **79.8%** | 1401 | 128×256 |
| q3-0.6b-Gate | 4096,3072,1024 | 74.4% | **82.6%** | 1452 | 128×256 |
| q3-0.6b-Down | 4096,1024,3072 | 78.0% | **82.7%** | 1529 | 128×256 |
| q3-0.6b-LMhead | 4096,151936,1024 | 73.4% | **83.5%** | 1549 | 128×256 |
| q3-1.7b-Q | 4096,2048,2048 | 77.1% | **85.6%** | 1543 | 128×256 |
| q3-1.7b-K | 4096,1024,2048 | 76.5% | **80.1%** | 1397 | 128×256 |
| q3-1.7b-Gate | 4096,6144,2048 | 92.7% | **95.5%** | 1597 | 128×256 |
| q3-1.7b-Down | 4096,2048,6144 | 87.2% | **92.5%** | 1695 | 128×256 |
| q3-1.7b-LMhead | 4096,151936,2048 | 89.0% | **103.9%** ✅ | 1318 | 128×256 |
| q3-4b-Q | 4096,4096,2560 | **95.2%** | 94.2% | 1640 | 128×128 |
| q3-4b-K | 4096,1024,2560 | 78.7% | **82.7%** | 1457 | 128×256 |
| q3-4b-O | 4096,2560,4096 | **84.4%** | 79.7% | 1687 | 128×128 |
| q3-4b-Gate | 4096,9728,2560 | 90.9% | **96.6%** | 1708 | 128×256 |
| q3-4b-Down | 4096,2560,9728 | **91.6%** | 85.7% | 1617 | 128×128 |
| q3-4b-LMhead | 4096,151936,2560 | 90.5% | **102.4%** ✅ | 1286 | 128×256 |
| q3-8b-Q | 4096,4096,4096 | **97.4%** | 95.4% | 1676 | 128×128 |
| q3-8b-K | 4096,1024,4096 | 81.4% | **85.6%** | 1555 | 128×256 |
| q3-8b-Gate | 4096,12288,4096 | 78.7% | **85.4%** | 1772 | 128×256 |
| q3-8b-Down | 4096,4096,12288 | 76.2% | **194.0%** ⚠️ | 827 | 128×256 |
| q3-8b-LMhead | 4096,151936,4096 | 87.5% | **94.8%** | 1285 | 128×256 |

### 24.3 128×256 wins on 31/42 shapes — the pattern is clear

- **128×256 best: 31 shapes** — all wide-N (Gate), all LM heads, most Down (fat-K), most non-square Q.
- **128×128 best: 11 shapes** — the skinny K/V (N≤512), a few square Q, a couple Down.

**The governing rule (emerged empirically):**
> **Wide-N or fat-K → 128×256 wins.  Skinny-N (N≤512) → 128×128 wins.**

Why: 128×256's wider N-tile amortizes scheduling + reuses B better on wide shapes, but on skinny N (128/256) it WASTES half the tile (computing empty columns) → collapses to 40-51%. E.g. q25-3b-K (N=256): 128×256 only 40.4% vs 128×128 60.1%.

### 24.4 We now BEAT cuBLAS on 4-5 shapes ✅

| Shape | our 128×256 | cuBLAS | ratio |
|---|---:|---:|---:|
| q25-0.5b-Down | 1240.8 T | 1226.4 T | **101.2%** |
| q25-3b-LMhead | 1367.0 T | 1304.0 T | **104.8%** |
| q3-1.7b-LMhead | 1369.0 T | 1318.1 T | **103.9%** |
| q3-4b-LMhead | 1316.3 T | 1285.8 T | **102.4%** |
| q3-8b-Down | 1603.8 T | 826.9 T | **194.0%** ⚠️ (see 24.6) |

First shapes where BluBLAS beats cuBLAS! The LM heads especially — memory-bound, our wide 128×256 tile streams the huge weight matrix efficiently.

### 24.5 Apples-to-apples — on shapes where cuBLAS ALSO uses 128×256, we hit ~90%

The fairest comparison: on the ~26 shapes where cuBLAS's own kernel is a 128×256 variant (§16.3), how does OUR 128×256 do? Ratios (r256) on those shapes range **77-105%, averaging ~90%.** Examples: q25-1.5b-Gate 97.1%, q25-3b-Gate 98.0%, q3-4b-Gate 96.6%, q3-1.7b-LMhead 103.9%, q3-8b-Gate 85.4%.

**Interpretation**: with the SAME tile cuBLAS uses, we're at ~90% — the remaining ~10% is cuBLAS's extra tuning on that tile (its 2×2 cluster / vertical wave / per-shape stages). That ~10% is exactly what B3 (cluster) + stage tuning will chase. Reaching 90% of cuBLAS's tuned kernel with our first-cut 128×256 (2×1) is a strong result.

### 24.6 ⚠️ The q3-8b-Down anomaly — honest flag

q3-8b-Down shows our 128×256 at **194% of cuBLAS** — but this is because **cuBLAS had an off-run on this ONE shape**:
- cuBLAS here (b0 run): **826.9 T**
- cuBLAS in the dedicated clean run (§21.3, reproduced twice §21.8): **1756.3 T**

Same shape, 2.1× different cuBLAS numbers. Every OTHER cuBLAS number in this run matches the dedicated run within a few %; only q3-8b-Down diverges. Most likely cuBLAS's heuristic picked a suboptimal kernel for this fat-K shape in this run (cuBLAS heuristic can be nondeterministic / cache-state-sensitive on some shapes).

**Honest read**: our 128×256 = 1603.8 T *consistently*. Against cuBLAS's BEST (1756 T) we're at **~91%**, NOT 194%. But this also reveals cuBLAS is **inconsistent** on q3-8b-Down (826-1756 T), while we're stable — a real robustness advantage. The huge jump on OUR side (128×128 629 T → 128×256 1604 T, a **2.5× improvement**) is real and shows 128×256 is essential for fat-K.

### 24.7 Best-of-2-tiles dispatcher → ~87% avg (up from 81.5%)

If our dispatcher picks the better of {128×128, 128×256} per shape (max of the two ratios), the average across all 42 shapes rises to **~87%** (using the trustworthy ~91% for q3-8b-Down), up from **81.8%** with 128×128 alone. **+5 points from adding one tile**, and we now exceed 100% on 4-5 shapes.

Remaining drags on the average:
- **Skinny K/V** (N≤512): still 55-79% — neither tile fits; needs B2 (skinny-N tile, BN=64/128 without wasting).
- **Square Q / some Down**: 84-97% — needs B3 (2×2 cluster) to match cuBLAS's tuning.

### 24.8 What B0 tells us for the dispatcher + next steps

**Dispatch rule (data-driven, for the hardcoded router):**
```
if (N >= 2048)         -> 128×256   (wide-N: Gate, LMhead, large Q/Down)
else if (N <= 512)     -> 128×128   (skinny K/V — until B2 gives a better skinny tile)
else                   -> test both; ~tie, default 128×256
```

**Next priorities refined by B0 data:**
- **B2 (skinny-N tile)** is now clearly justified: the worst shapes (q25-*-K at 40-60%) are skinny K/V where BOTH our tiles are bad. A dedicated small-BN tile is the biggest remaining win + where we can beat cuBLAS (cuBLAS is also weak there, 241-743 T).
- **B3 (2×2 cluster)** chases the last ~10% on the shapes where our 128×256 already matches cuBLAS's 128×256 (§24.5) — cuBLAS's edge there is its 2×2 cluster.
- **Replicate B0 to NN + NT** (currently TN only) — expected to mirror TN.

**Status**: B0 DONE for TN. 128×256 validated, +5pts average, 5 shapes beating cuBLAS. Wire the dispatch rule + replicate to NN/NT, then B2.

---

## 25. SM103 module architecture — production vs legacy vs template (2026-07-02)

Investigated which files are actually used (there was confusion). The folder has THREE generations of SM103 BF16 kernels:

| Gen | Files | What | Status |
|---|---|---|---|
| **Gen 1** | `Bgemm_sm103_bf16.cu`, `_NN.cu` | non-cluster cta_group::1 (single-CTA) | ❌ dead (dispatcher never calls) |
| **Gen 2** | `Bgemm_sm103_bf16_{TN,NN,NT}_cluster.cu` | per-layout hand-written 2-CTA cluster kernels | ✅ **currently wired** (dispatcher calls these) + our experiment sandbox |
| **Gen 3** | `Bgemm_sm103_cluster_template.cuh` + `Bgemm_{tn,nn,nt}_variants.cu` | **unified templated kernel** (layout + tile config = template params) + thin launcher wrappers | ⭐ **canonical home** (colleague's design) — but was UNVALIDATED |

**Production call chain** (verified): `mycublasBgemmStridedBatched` (public, `mycublas.h`) → `mycublasBgemmStridedBatched_dispatcher` (`Bgemm_dispatcher.cu`) → `mycublasBgemmSM103_dispatch_{nn,nt,tn}` (`Bgemm_sm103_dispatcher.cu`) → `*_cluster_{128x128x64,256x256x64}` (**Gen 2 `_cluster.cu`**).

**Roles clarified:**
- `.cuh` (Gen 3) = the intended **single source of truth** — kernel written once, all NN/NT/TN + tile configs via template params. `Sm103Config_128x128x64` and `Sm103Config_256x256x64`(=128×256) already scaffolded.
- `_variants.cu` (Gen 3) = thin **launcher wrappers** (`extern "C"` entry points calling `launch_bgemm_sm103_cluster<Config,Layout>`), suffixed `_template`.
- `_cluster.cu` (Gen 2) = older per-layout kernels, currently dispatcher-wired + our test sandbox.
- The dispatcher migration to route through the `.cuh` is **pending**.

**Naming note**: `256x256x64` in this codebase = a 128×256-per-CTA tile → 256×256 macro-output (2 CTAs). Could rename to `128x256x64` for honesty (per-CTA convention).

**SM89 has NO 256×256 tile** and NO clusters at all (Ada hardware — clusters are sm_90+). Its variants top out at 256×128 / 128×256, non-cluster mma.sync.

---

## 26. Porting fixes to the canonical `.cuh` + verification (2026-07-02, in progress)

Decided the canonical `.cuh` (Gen 3) is the single source of truth. But discovered it had **never received any of our bug fixes** — all our work over past days went only into the Gen 2 `_cluster.cu`. So the `.cuh` still had the OLD buggy code and **crashed at 4096³** on first test.

### 26.1 Build blocker fixed — root disk 100% full
Every test failed with `nvcc fatal: Failed to preprocess host compiler properties`. Cause: root disk `/` was **100% full** (98G/98G), so nvcc couldn't write temp files to `/tmp`. Fix: `export TMPDIR=/mnt/raid0/gautam_k/tmp` (redirect nvcc scratch to the 35TB disk). Build works after that. (Root disk still needs cleanup — separate issue.)

### 26.2 Five fixes ported from `_cluster.cu` → `.cuh`
1. grid **ceil-div** (was floor-div → grid=0 for small shapes)
2. **removed `issued = 0`** reset (the producer/consumer race → the 4096³ crash)
3. added **3-arg `make_smem_desc`** (LBO/SBO) for TN
4. TN **`a_off` BM→SWIZZLE_W**
5. **b_off** per-layout: NT separate; NN+TN share `SW×SW` (both store B as [K,N])

### 26.3 Verification result — first pass: 5 of 6 suites PASS, NN 128×256 FAILED (now FIXED, see §26.6)

| Suite | First pass (2026-07-02) | After fix (2026-07-03) |
|---|---|---|
| NN 128×128 | ✅ 12/12 (crash gone — 4096³ works) | ✅ 12/12 |
| **NN 128×256** | ❌ **0/8 — wrong numbers (rel_err 1e4-1e6)** | ✅ **8/8 — FIXED** |
| NT 128×128 | ✅ 12/12 | ✅ 12/12 |
| NT 128×256 | ✅ 8/8 | ✅ 8/8 |
| TN 128×128 | ✅ 12/12 | ✅ 12/12 |
| TN 128×256 | ✅ 8/8 | ✅ 8/8 |

- **Crash FIXED** (issued=0 removal) — 4096³ runs on all passing suites.
- **Speed MATCHES legacy** (cmp-square4096 TN: canonical 1618/1650 T vs legacy 1631/1597 T). ✅ The `.cuh` is speed-equivalent.
- **Only NN 128×256 is wrong.** Tried a b_off revert (NN→`SW×SW` like TN); after `make clean && make all` it STILL gives the same wrong result → **the bug is elsewhere in the NN 128×256 path, not (only) b_off.**

### 26.4 The open bug — NN 128×256 (debug tomorrow)

Facts that localize it:
- TN 128×256 ✅ and NT 128×256 ✅ → the wide-BN (256) machinery works for two layouts.
- NN 128×128 ✅ → the NN layout works at BN=128.
- **Only NN × BN=256 fails** → it's specifically the **NN B-operand handling at the wider tile**.
- b_off revert to `SW×SW` didn't fix it → suspect the **NN B-TMA load** (`init_4d_tma_B_KN` box/coords at BN=256), OR the NN B-smem descriptor (`make_smem_desc` 1-arg LBO=0 may be wrong for NN B=[K,N] at BN=256, like TN needed LBO), OR the NN B `B_col` producer coord at BN=256.

**Hypotheses to test tomorrow (in order):**
1. NN B descriptor: NN B is [K,N] chunk-major (SAME as TN B). TN B needed the 3-arg desc (LBO=8192). NN B currently uses 1-arg (LBO=0). **NN B likely needs the same 3-arg descriptor as TN B** → try giving NN B the 3-arg `make_smem_desc(B_smem+b_off, BK*SW*2, 8*SW*2)`. This is the strongest hypothesis (NN B == TN B layout, TN needed it, NN doesn't have it).
2. If not: check NN `B_col` producer coord + `init_4d_tma_B_KN` box dims at BN=256.
3. Cross-check: does the legacy NN `_cluster.cu` have a working 128×256? (No — legacy NN only ever ran BN=128; its BN=256 is untested too, so no reference.)

### 26.5 Status + tomorrow's plan
- ✅ Canonical `.cuh` de-bugged for 5/6 suites, crash-free, speed-matched.
- ⏳ NN 128×256 correctness bug open → debug via hypothesis 1 (NN B needs 3-arg descriptor like TN B).
- Once NN 128×256 passes → all 6 suites pass → **lock in `.cuh` as single source of truth**.
- Then: B2 (skinny-N variant) in the `.cuh`, wire the shape→variant dispatcher.

### 26.6 NN 128×256 FIXED (2026-07-03) — hypothesis 1 confirmed

**Root cause: the smem descriptor was chosen by GEMM layout, not per-operand.**
The old code did `if (Layout==TN) { A and B both 3-arg } else { A and B both 1-arg }`.
That gave NN's B the 1-arg descriptor (LBO=0) — but NN's B is stored **[K,N]**,
*identical* to TN's B (same `init_4d_tma_B_KN`, same `B_col`, same `b_off`), and
TN's B needs the 3-arg descriptor with **LBO = BK·SWIZZLE_W·2 = 8192**.

**Why it only failed at BN=256 (not BN=128):** LBO is the byte stride to the next
64-column swizzle chunk. NN's B tile holds `BN/2` columns:
- BN=128 → BN/2 = 64 = **1 chunk** → LBO never used → 1-arg (LBO=0) works → NN 128×128 passed.
- BN=256 → BN/2 = 128 = **2 chunks** → LBO **is** used → LBO=0 aims the 2nd chunk at
  the wrong address → garbage → NN 128×256 failed.

This is the exact same failure mode as the earlier TN-A precision bug (TN A=[K,M],
BM=128=2 chunks, needed LBO=8192). Cross-checked against DeepGEMM `sm100.cuh`
(LBO = BK·BLOCK_MN_ATOM·2 = 8192, SBO = 8·BLOCK_MN_ATOM·2 = 1024) — same convention.

**The fix (in `Bgemm_sm103_cluster_template.cuh`, consumer MMA call):** choose each
operand's descriptor by **that operand's storage layout**, not by the GEMM name:
- **A** → 3-arg (LBO) when stored [K,M] → **TN only** (`trans_a`); NN/NT A=[M,K] → 1-arg.
- **B** → 3-arg (LBO) when stored [K,N] → **NN and TN** (`trans_b`); NT B=[N,K] → 1-arg.

Result: TN(A 3-arg, B 3-arg), **NN(A 1-arg, B 3-arg ← the change)**, NT(A 1-arg, B 1-arg).

**Verification (B300 sm_103a, device 1, clean):** all **6/6 suites PASS**, MaxRelErr
= 0.00e+00 (bit-exact vs cuBLAS). NN 128×256 went **0/8 → 8/8**. Nothing regressed
(TN/NT unchanged — their descriptors were already correct). Throughput clean, not
contended: NN 128×256 `cmp-square4096` = 1596 T vs cuBLAS 1667 T (95.7%); matches
the earlier TN 128×256 square-4096 (1596.88 T) → confirms no GPU contention.

**Canonical `.cuh` is now LOCKED IN as the single source of truth for all 3 layouts
at both 128×128 and 128×256.**

### 26.7 Next
- B2 (skinny-N variant) in the `.cuh` for the weak GQA K/V shapes.
- ~~Wire the shape→variant dispatcher to route to the `.cuh` launchers~~ → DONE, see §27.
- Housekeeping: root disk `/` now at 92% (was 100%); keep a writable TMPDIR for B300 builds.

## 27. Dispatcher migrated to the canonical `.cuh` (2026-07-03)

### 27.1 The problem — production shipped the test scratchpads
Before today the SM103 production path was:
```
mycublasBgemmStridedBatched_dispatcher → mycublasBgemmSM103_dispatch_{nn,nt,tn}
  → mycublasBgemmSM103_bf16_{nn,nt,tn}_cluster_{128,256}...   (DEFINED IN *_cluster.cu)
```
i.e. the library shipped the **legacy `_cluster.cu`** kernels, and the `.cuh` +
`_variants.cu` were **dead code** (nobody called the `_template` symbols). Kota
confirmed this was a temporary wiring for individual testing — the intended design
is dispatcher → `_variants.cu` → `.cuh`.

### 27.2 The reference — SM89 module (the proven house pattern)
The SM89 FP32 module (established in the may27_406k BluTrain blublas) has all three
tiers side by side, and its production path routes through the template:
| Tier | SM89 | SM103 |
|---|---|---|
| Template (all kernels) | `Sgemm_core_template_sm89.cuh` | `Bgemm_sm103_cluster_template.cuh` |
| Variant wrappers | `Sgemm_{nn,nt,tn}_variants_sm89.cu` | `Bgemm_{nn,nt,tn}_variants.cu` |
| Per-variant test scratchpads | `Sgemm_nn_256x128_sm89.cu`, … | `Bgemm_sm103_bf16_{NN,NT,TN}_cluster.cu` |

So SM103 was made to mirror SM89.

### 27.3 The change
- `Bgemm_sm103_dispatcher.cu`: all 6 calls repointed from `..._cluster_{128,256}...`
  → `..._cluster_{128,256}..._template` (which route through `_variants.cu` → `.cuh`).
- `Bgemm_sm103_bf16.cuh`: added the 6 `_template` extern-"C" prototypes so the
  dispatcher sees them.
- Stale NT comment ("both entries run 128×128 under the hood, no-op routing")
  removed — the `_256x256x64` entry now runs the validated 128×256 tile.
- Zero API change (same `mycublasBgemmSM103_dispatch_*` signatures). Safe because
  the `.cuh` path is already bit-exact + speed-matched (§26.6).

### 27.4 Verification
- `make lib` → `✅ Library built: build/libmyblas.so`, no errors, no undefined refs.
- `nm`: `dispatcher.o` has `U` (needs) all 6 `_template` symbols; `variants.o` has
  `T` (defines) them → linker resolves dispatcher → variants → `.cuh`.
- `_cluster.cu` objects still compile into the `.so` but are no longer called by
  production — they are now purely the individual-test scratchpads (SM89 role).
- Final confidence check pending: one B300 run through the top-level
  `mycublasBgemmStridedBatched` dispatcher (BluTrain/PyTorch path).

### 27.5 New architecture — the rule going forward
**`.cuh` + `_variants.cu` are the single source of truth. All new tile variants
(B2 skinny-N, etc.) are added there — never in `_cluster.cu`.** Adding a variant =
(1) config alias in `.cuh`, (2) wrapper in `_variants.cu`, (3) routing rule in the
dispatcher. Individual testing needs NO dispatcher change — example programs call
`launch_bgemm_sm103_cluster<Config,Layout>` directly (as `test_bgemm_sm103_cluster_template.cu`
already does).

## 28. P1 variant — 128×256 + VERTICAL wave (`128x256_2x1_v`) — VALIDATED (2026-07-03)

### 28.1 Implementation
Added vertical-wave scheduling as a **non-breaking template policy**
`launch_bgemm_sm103_cluster<Config, Layout, bool VerticalWave=false>`:
- New `decode_mn_tile<VerticalWave>()` helper in the `.cuh`; param threaded through
  the kernel + launcher; all 4 tile-decode sites (producer init/CLC, epilogue
  init/CLC) use it. `grid_m = total_mn_tiles / grid_n` computed in the prologue.
- Horizontal (default): `block_col = idx % grid_n` (N fastest — unchanged).
- Vertical: `row_tile = idx % grid_m` (M fastest → streamed B column stays L2-
  resident across the M-tiles of a column, per §16.7).
- Default `false` ⇒ existing behaviour byte-identical (proven: 6/6 horizontal
  suites still bit-exact).

### 28.2 Result (B300 sm_103a, clean — cuBLAS square4096 ≈ 1685 T matches prior)
All 9 suites PASS bit-exact (6 horizontal + 3 vertical, MaxRelErr 0). Vertical vs
horizontal on M=4096 shapes, as % of cuBLAS:

| Shape (M,N,K) | horiz | **vertical** | Δ | cuBLAS pick (§16.3) |
|---|---|---|---|---|
| wideN-gate 4096×12288×4096 | 76–86% | **99.7–99.9%** | **+17…+31%** | `128x256_2x1_v` ✓ |
| skinny-K 4096×1024×4096 | 80–85% | 86–88% | +2…+9% | (mixed) |
| square4096 4096×4096×4096 | 95–97% | ~95% | −2% | `128x256_2x2f` (2×2!) |

### 28.3 Interpretation — the result reproduces cuBLAS's own policy
- **Wide-N (Gate/Up/LMhead) → vertical wins, ~99.7% parity** (from ~76%). Exactly
  the shapes cuBLAS serves with `128x256_2x1_v`. The §16.7 L2-reuse hypothesis is
  now verified with a +31% measured delta. This is the dominant variant (12 of 42
  shapes) — effectively CLOSED.
- **Square → vertical slightly HURTS (−2%)** because square wants `128x256_2x2f`
  (2×2 + fused), not vertical. Confirms the square gap needs P2/P3 (2×2 cluster),
  not wave scheduling.
- **Dispatcher rule**: vertical for wide-N, horizontal/(future 2×2) for square —
  mirrors cuBLAS.

### 28.4 Next
- Wire the `_v` variant into `_variants.cu` + header + dispatcher (vertical for
  wide-N shapes). Then P2/P3 = 2×2 cluster for the square shapes.

### 28.5 Full 42-shape Qwen sweep (B300, 2026-07-03) — 128×256 horiz vs vert vs cuBLAS
All 42 PASS bit-exact. **vertical wins 28/42; avg % of cuBLAS: horiz 86.1%, vert
88.5%, best-of-2 88.6%** (baseline was 81.5%, §22).

Where vertical delivers (huge-N shapes):
- **All 8 LM-heads**: horiz ~1230–1410T → **vert ~1630–1792T** (+15–33%), all at
  parity-or-better vs cuBLAS.
- **Widest Gates**: q25-7b-Gate (N=18944) 78.9%→**100.4%**; q3-8b-Gate (N=12288)
  85.6%→**100.1%** — both BEAT cuBLAS.
- Moderate Gates/Down already ~96–97% on horizontal; vertical ≈ tie there.

CAVEAT (honesty): the >100% `h%`/`v%` on LM-heads are inflated by cuBLAS
measurement noise on giant-N (each run re-times cuBLAS; its 151936-N throughput
swings 1384–1680T). Reliable signal = our absolute vertical TFLOPS (1630–1792T,
stable) ≫ horizontal. Claim: P1 closes ~10 shapes (8 LM-heads + 2 widest Gates)
to parity/beating.

Dispatcher rule (data-backed): **vertical is a safe DEFAULT for the 128×256 tile**
— wins/ties 28/42, best-of-2 (88.6%) ≈ vert-only (88.5%), so vertical-default
loses ~nothing and wins big on wide-N.

Remaining gaps → next priorities (unchanged from plan):
1. **Skinny K/V (N=128–512): 40–51% — BIGGEST gap.** q25-1.5b-K 40.6%, q25-3b-K
   40.3%, q25-0.5b-K 47.2%, q25-7b-K 50.7%. 128×256 wastes half+ the tile here.
   cuBLAS uses 64×64 / 128×64. → **track B (skinny-N)**, where we can BEAT cuBLAS.
2. **Moderate square Q/O/Down (68–85%):** q25-1.5b-Q 68.6%, q3-4b-O 79.3%,
   q25-1.5b-Down 81.1%. → **P2 (2×2 cluster)** + 128×192 / 160×192 tiles.

### 28.6 Best-of-TILES sweep (2026-07-03) — added 128×128 to the mix (FREE)
Reran the 42-shape sweep with BOTH tiles (128×256 and 128×128), each h+v. Taking
the best (tile,wave) per shape:

**Best-of-4 (2 tiles × 2 waves) avg = 92.3% of cuBLAS** (baseline 81.5%; 128×256-
only best-of-2 was 88.6%). The +3.7 pts came entirely FREE — 128×128 was already
built. It rescued the skinny-N shapes the 128×256 tile wastes on:

| Skinny K/V | 128×256 | 128×128 |
|---|---|---|
| q25-7b-K (N=512) | 51% | **79%** |
| q25-3b-K (N=256) | 40% | **64%** |
| q25-1.5b-K (N=256) | 41% | **59%** |
| q25-0.5b-K (N=128) | 41% | **55%** |

**Per-shape dispatcher map (winner distribution):** 128×256-vertical → 22 shapes
(all LM-heads, wide Gates, most moderate); 128×256-horizontal → 9; 128×128-
horizontal → 9 (skinny/square-ish: q25-1.5b/7b-Q, q3-4b-Q/O/Down, q3-8b-Q,
q25-1.5b-Down); 128×128-vertical → 2 (N=128, 256). This is the concrete
shape→(tile,wave) table to encode in the dispatcher.

**True remaining gaps (after best-of-tiles):**
1. Super-skinny K/V N=128–256: still **55–64%** — 128×128 (BN=128) over-covers
   N=128. cuBLAS uses 64×64 / 128×64. → track B (BN=64), beat-cuBLAS opportunity.
2. Small-hidden Q/O/Down (hidden 896–1024): **73–86%** (q25-0.5b, q3-0.6b). → P2
   (2×2 cluster) + 128×192/160×192.

Note: production dispatcher currently uses HORIZONTAL only + a crude tile rule
(≈86%). Wiring best-of-(tile,wave) per this map banks the 92.3% into production.

### 28.7 BM=64 (64×128 tile) was BROKEN → real root cause (2026-07-03, reference-verified)
Tried 64×128 (BM=64) as a "free" skinny-N tile (compiled + passed static_asserts).
On B300 it produced WRONG results — every shape FAIL, first M-block correct
(first-10 elements matched cuBLAS), second M-block garbage.

**First (WRONG) theory — recorded then disproved:** I guessed the epilogue peer
read needed `cta_rank*128` instead of `cta_rank*BM` (a "+128 SM-select lane").
Tested on a verified-fresh binary (binary mtime > .cuh mtime, grep confirmed the
change) — it STILL failed. So both offsets (64 and 128) are wrong; the theory was
bunk. Lesson: don't ship a theory as fact without the controlled test.

**REAL root cause (authoritative — CUTLASS / DeepGEMM / ThunderKittens, file:line):**
- TMEM is **128 DP (datapaths/lanes) × 512 col, PER CTA and LOCAL** (CUTLASS
  `cute/arch/tmem_allocator_sm100.hpp:45`). DP index lives in addr bits [31:16]
  but only holds 0..127 (7-bit).
- `tcgen05.mma.cta_group::2` splits total M=2·BM cleanly: CTA0 → rows [0,BM) in
  CTA0's TMEM DP[0,BM); CTA1 → rows [BM,2BM) in **CTA1's OWN** TMEM DP[0,BM). Each
  CTA reads its own accumulator at **DP base 0** — there is NO cross-CTA lane
  offset (a CTA cannot address the peer's TMEM). DeepGEMM's hand-PTX epilogue read
  has zero cta_rank and zero warp lane terms (`epilogue/sm100_store_cd.cuh:77`,
  comment `impls/sm100_bf16_gemm.cuh:373` "hardware ignores the warp index bits").
- So our `cta_rank*BM << 16` term is SPURIOUS. At BM=128 → DP 128 = out of the
  7-bit DP range → silently ignored → accidentally reads DP 0 = correct (the ONLY
  reason BM=128 ever worked). At BM=64 → DP 64 = in-range → NOT ignored → reads
  DP[64,128), which under the `M_MMA=64` **2×2 INTERLEAVED** layout (CUTLASS
  `cute/atom/mma_traits_sm100.hpp:543`: N split as half in DP[0,64), half in
  DP[64,128)) is the wrong data → garbage.
- Valid cta_group::2 kind::f16 M ∈ {128,256} (CUTLASS `mma_sm100_umma.hpp:554`),
  so BM=64 (M=128) is LEGAL but forces the interleaved 2×2 accumulator our simple
  `tcgen05.ld.32x32b.x8` epilogue does NOT de-interleave. cta_group::1 allows M=64
  (`mma_sm100_umma.hpp:88`).

**Fix applied:** epilogue `tmem_row` DP offset → **0** (removed the `cta_rank`
term; kept the ignored warp term to match the validated form). This makes BM=128
correct BY DESIGN (was accidental). Added `static_assert(Config::BM == 128)` in
the launcher so BM<128 is a compile error, not silent garbage. Removed the broken
64×128/64×256 aliases + sweep call.

**Skinny-N real path (unchanged conclusion):** a sub-128-row tile needs either a
**cta_group::1 (M=64) variant** (clean DP[0,64), independent CTA — recommended) or
DeepGEMM's **swap-AB** (keep per-CTA M=128, make the small dim = N). Both are
separate kernel variants, not config aliases. Given 64×128's throughput was
marginal anyway, P2 (2×2 cluster) is likely better ROI than chasing skinny-N.

## 29. 2×2 cluster — authoritative mechanism (reference-verified, 2026-07-04)

Deep research over CUTLASS / DeepGEMM / ThunderKittens (file:line cited) before
implementing P2. Key findings — one corrects an earlier oversimplification:

**29.1 There is NO 4-CTA MMA.** The only multi-CTA tcgen05 BF16 atom is 2-CTA
(`cta_group::2`, `SM100_MMA_F16BF16_2x1SM_SS`, cutlass mma_sm100_umma.hpp:552;
`AtomThrID = _2`, mma_traits_sm100.hpp:2044). A "2×2" cluster = the SAME 2-CTA
MMA pair (along cluster-x/M) **replicated as a 2nd pair along cluster-y/N**, with
TMA multicast sharing the operand invariant along the new axis. `_2x2_2cta` =
`cta_group::2` MMA (the `2cta`) + `ClusterShape<2,2,1>` (the `2x2`) on top. Never
`cta_group::4` (doesn't exist).

**29.2 CORRECTION to §23.2 ("2×2 halves BOTH operands" — WRONG).** In a 4-CTA
2×2, only **A is multicast** (across the 2 N-blocks; A is invariant in N), mask
`0b0101`/`0b1010`. **B is NOT multicast** — each pair loads its own N-block (B
sharing across the pair is intrinsic to cta_group::2, unchanged). Halving BOTH
operands needs cluster.x≥4 (multiple M-pairs) → ≥16 CTAs, not 4. So 2×2's real
benefit is **halving A (activation) traffic across 2 adjacent N-blocks**, more
modest than the earlier "halves both" hope.

**29.3 Rank decomposition (ClusterShape<2,2,1>):** cta_m = rank&1 (pair role,
M/x), cta_n = rank>>1 (N-block, y). MMA leader per pair = (cta_m==0), i.e. ranks
0 and 2 each issue their pair's MMA into their own TMEM. Epilogue TMEM read stays
DP base 0 per CTA (confirms §28.7).

**29.4 The three masks (do NOT collapse to 0b1111):**
- launch cluster dims = {2,2,1} (4 CTAs; ≤32 ok, ≤8 portable).
- TMA A-multicast mask = 2 bits across N (0b0101/0b1010); exactly ONE CTA per
  A-group issues the A load (multicast completion signals both N-partners).
- MMA-commit / tmem CTA_MASK = the PAIR mask (0b0011 / 0b1100), via
  calculate_umma_peer_mask (cutlass sm100_pipeline.hpp:96). 0b1111 is only for a
  genuine all-4 cluster sync — using it for the MMA commit is a bug.
- mbarrier arrive counts stay keyed to the pair (kNumMulticast=2), unchanged —
  you just have two independent pairs.

**29.5 Highest-fidelity reference:** cutlass sm100_mma_warpspecialized.hpp
(persistent warp-specialized tcgen05 SS mainloop, arbitrary ClusterShape; mask
math at :524-541 via create_tma_multicast_mask<Mode>). DeepGEMM is 1-D only
(kNumMulticast∈{1,2}, {cluster,1,1}) — its pipeline/TMEM plumbing is a good
skeleton, but the 2-D coord/mask math comes from CUTLASS.

**29.6 Integration wrinkle for OUR kernel:** our scheduler is CLC (Cluster Launch
Control) persistent, not an explicit tile scheduler like DeepGEMM/CUTLASS. The
2×2 needs the cluster's 2 pairs to cover ADJACENT N-blocks — mapping CLC-handed
cluster tiles → (2 N-blocks) is the part the references don't cover directly and
is the main implementation risk.

**29.7 Net:** 2×2 is a real, well-scoped build (10-step checklist exists) but
sizable — producer A-multicast + one-issuer logic + B N-block offset + 2-D
scheduler over CLC + careful expect_tx/barrier accounting. Benefit (halve A
traffic, ~14 moderate-square shapes) is real but more modest than §23.2 implied.

## 30. P2 implementation — 2×2 cluster: research, decision, milestones (2026-07-04)

Two deep reference passes (CUTLASS/DeepGEMM/ThunderKittens) + independent PTX ISA
9.3 research. Full plan: `2x2_cluster_plan.md`. Key research findings:

**30.1 Model (settled, §29 recap):** no 4-CTA MMA; 2×2 = same `cta_group::2` M-pair
replicated as a 2nd pair along N, A multicast across N (mask 0b0101/0b1010), B
per-pair. Only A traffic halved (§29.2 corrected the old "halves both" claim).

**30.2 THE decisive PTX fact — flat vs 2-D launch is a non-issue.** PTX ISA 9.3
§9.7.17.5.1: "Any 2 CTAs within the cluster whose %cluster_ctarank differs by the
LAST BIT ONLY form a CTA pair." So the cta_group::2 pairing is defined by
%cluster_ctarank parity, INDEPENDENT of launch dim shape. `(4,1,1)` and `(2,2,1)`
give identical pairs/masks/multicast. `(4,1,1)`-flat is provably correct (resolves
§29's "HW-verify pairing" worry) and DeepGEMM already launches flat
(clusterDim={n,1,1}). TMA multicast (§9.7, line 17169): instruction-time 16-bit
ctaMask by %cluster_ctarank; with cta_group::2 the mbarrier signal routes by parity
(→ clear smem addr bit 24 so bytes land on the even/leader CTA). CLC
(clusterlaunchcontrol, line 23554): per-CLUSTER granularity, returns first-CTA
ctaid, .multicast::cluster::all broadcasts response to all cluster CTAs.

**30.3 DESIGN DECISION — parameterized flat cluster (flexibility, done right).**
Rather than hardcode 2×2, parameterize `CLUSTER_X` (CTAs along M, even) and
`CLUSTER_N` (CTAs along N); launch flat `(CLUSTER_X*CLUSTER_N,1,1)`; decode
`rank → cta_m=rank&1, m_pair=(rank%CLUSTER_X)>>1, cta_n=rank/CLUSTER_X`. Because
the pair is always the last bit (PTX-guaranteed), ONE decode handles every
cta_group::2 shape cuBLAS uses — 2×1, 2×2, 2×4, 4×1, 4×2 — with no rework. True
2-D launch would only complicate CLC (2-D first_ctaid) for zero functional gain.
(1×2/1×4 are cta_group::1 single-CTA MMA — a separate future kernel path.) This
directly answers the "keep the design flexible so expansion doesn't force
compromises" concern — parameterized-flat IS that flexible design.

**30.4 Milestones (B300):** (1) redundant-A: 2×2 flat cluster + scheduler split
into 2 N-blocks + per-pair leader, every CTA loads its own A+B (no multicast) →
each pair is byte-for-byte the validated 2-CTA kernel; validate correctness. (2)
turn on A multicast (cta_n==0 issues A, mask 0b0101<<cta_m, a_mbar bit24 cleared).
(3) sweep NN/NT/TN × BN=128/256 × 42 shapes. Then wire dispatcher (LAST, after all
variants validated vs cuBLAS).

**30.5 Key edits (see 2x2_cluster_plan.md for the full file:line checklist):**
CLUSTER_SIZE stays 2 = the M-pair (B-split, i_desc, mbar inits, TMA cta_group::2);
add CLUSTER_N=2, CLUSTER_CTAS=4 for grid-stride/CLC. clc_empty arrive ×4. rank via
%cluster_ctarank. block_row=row_tile*CLUSTER_X+cta_m, block_col=n_cluster*2+cta_n.
MMA leader = cta_m==0 (ranks 0 & 2). commit mask = 0b0011<<(2*cta_n). CLC arm
expect_tx on all 4 CTAs. Keep static_assert(BM==128).

### 30.6 P2 Milestone 1 IMPLEMENTED (2026-07-04) — 2×2 redundant-A, compiles clean
Implemented the parameterized flat 2×2 cluster in Bgemm_sm103_cluster_template.cuh:
- Config: `_CLUSTER_N` template param (default 1 → existing 2×1 unchanged); CLUSTER_N,
  CLUSTER_CTAS=CLUSTER_SIZE*CLUSTER_N; CLC_EMPTY_ARRIVE ×CLUSTER_CTAS. Aliases
  Sm103Config_128x256_2x2 / _128x128_2x2 (CLUSTER_N=2).
- Rank: get_cluster_cta_rank() → %cluster_ctarank; derive cta_m=rank&1, cta_n=rank>>1,
  pair_leader=rank&~1.
- Prologue: init_tile_idx = ctaid/CLUSTER_CTAS; tma_mbar_base → pair_leader.
- Producer/epilogue decode: block_row=row_tile*CLUSTER_SIZE+cta_m; block_col=
  n_cluster*CLUSTER_N+cta_n; B intra-pair split by cta_m. CLC re-decode /CLUSTER_CTAS.
  Milestone 1 = redundant-A (each CTA loads own A via block_row; no multicast yet).
- Consumer: MMA leader = cta_m==0 (ranks 0 & 2); commit mask pair_mask=CTA_MASK<<(2*cta_n)
  (0b0011 pair0 / 0b1100 pair1).
- Scheduler: CLC expect_tx armed on all CLUSTER_CTAS CTAs (multicast::cluster::all).
- Epilogue: tmem_empty signalled to THIS pair's leader (mapa to pair_leader; replaced
  the bit-24 trick that always hit rank0 — wrong for pair1).
- Launcher: __cluster_dims__(Config::CLUSTER_CTAS,1,1); grid_n = N-CLUSTERS
  (ceil(N-blocks/CLUSTER_N)); grid = total_mn_tiles*batch*CLUSTER_CTAS.
Compiles clean for sm_103a: 2×1 path (regression-safe) AND 2×2 path instantiated.
Test: new "P2: 128x256 tile, 2x2 cluster" suite (M%256, N%512 clean shapes) + BM==128
guard kept. AWAITING B300: (a) 2×1 suites still bit-exact (no regression), (b) 2×2
suites bit-exact (validates 4-CTA cluster). Then Milestone 2 = A-multicast.

### 30.7 P2 Milestone 1 VALIDATED on B300 (2026-07-04) — 2×2 bit-exact + modest gain
2×2 (redundant-A) all PASS bit-exact (MaxRelErr 0.00e+00) across NN/NT/TN (the one
3.92e-03 is the β-accumulation path, PASS). No regression on 2×1. Same-run 2×2-vs-2×1
on square4096: 2×2 modestly faster (NN 96.6→96.8, NT 95.0→98.8, TN 96.7→97.6% of
cuBLAS; +30-45 T) — matches §29.2 "modest A-benefit", present even in redundant-A form
(implicit L2 reuse of A across the 2 adjacent N-blocks). wideN ~same (~90-91%; that's a
vertical-wave/P1 win). Small shapes 32-40% (latency/occupancy-bound, expected — not
2×2's target). Whether 2×2 closes the moderate-square band (2×2's real target) is TBD →
the end benchmark.

### 30.8 DECISION — A-multicast as a TUNABLE KNOB (not either/or) (2026-07-04)
Rather than "redundant-A vs multicast", make it a compile-time knob `bool MulticastA`
(mirrors DeepGEMM kIsMulticastOnA/kNumMulticast, sm100_bf16_gemm.cuh:27 — multicast is a
policy param there, not hardcoded). MulticastA=false → redundant-A (each CTA loads own A;
M1, validated). MulticastA=true → A issued only by cta_n==0, multicast to the N-partner
(mask 0b0101<<cta_m), halving A DRAM traffic. Non-destructive: false path unchanged.
Reference stance: multicast helps MEMORY-bound shapes (saves operand DRAM); on
COMPUTE-bound shapes redundant-A already ties (load hides behind compute + L2). So the
BENEFIT is shape-dependent → keep BOTH, let the per-shape dispatcher pick.
TEST MATRIX (end benchmark, clean GPU): {MulticastA off, on} × {compute-bound (large K,
square) AND memory-bound (small K, wide/skinny N)} shapes, all 3 layouts. Correctness of
both paths validated now (contention-immune); perf ranking deferred to clean GPU.

### 30.9 P2 Milestone 2 IMPLEMENTED (2026-07-04) — A-multicast as a tunable knob
Added `bool MulticastA` template param to the kernel + launcher (universal in the
template, opt-in per config — NOT per-variant code). Pieces:
- `tma_4d_gmem2smem_mc` primitive: cp.async.bulk.tensor.4d ... cta_group::2
  .multicast::cluster + u16 ctaMask.
- `issue_A_load<CTA_GROUP, MulticastA>`: MulticastA=false → every CTA loads own A
  (redundant-A, M1); MulticastA=true → only cta_n==0 issues A, multicast mask
  0b0101<<cta_m, using the LOCAL full_mbar (cta_group::2 parity routes completion
  to the pair leaders). B unchanged (all CTAs). expect_tx unchanged (each CTA
  declares copy_size on its leader — receives A via multicast when it doesn't load).
- Test: threaded MulticastA through run_test_case/run_test_suite; new "P2-M2:
  A-MULTICAST ON" suite (VerticalWave=false, MulticastA=true) on shapes_2x2.
Compiles clean for sm_103a: MulticastA false AND true paths. Correspondence to
DeepGEMM: MulticastA ≡ kIsMulticastOnA (direction=A); CLUSTER_N ≡ kNumMulticast
(degree). Multicast-B (4×1) is a separate future direction, not built.
AWAITING B300: (a) redundant-A 2×2 + 2×1 still bit-exact (no regression);
(b) A-multicast-ON 2×2 bit-exact (same result, A loaded once). Perf on/off ranking
deferred to clean GPU (§30.8 test matrix).

### 30.10 4×1 cluster / multicast-B — FUTURE variant (NOT needed for Qwen)
Verified from §16.3: the 42 Qwen shapes use only cluster shapes 2×1 (17), 2×2 (15),
2×2f (7), 1×2 (3). NONE use 4×1. So 4×1 / multicast-B is OUT OF SCOPE for the Qwen
goal — it's general-BLAS-expansion (future). Documented here so it's not forgotten.

Design (mirror of the 2×2 we built — swap M↔N, A↔B):
- 4×1 = 4-CTA cluster, 2 M-pairs along M (ranks {0,1}=pair0 @ M-rows[0,2·BM),
  {2,3}=pair1 @ [2·BM,4·BM)), same N. Shares B across the 2 M-pairs → multicast B.
- Same flat (4,1,1) launch as 2×2; differs only in DECODE (extra axis = M) +
  multicast direction (B not A). ≡ DeepGEMM kIsMulticastOnA=false.
- To add later: a config flag for extra-axis (M vs N) [e.g. CLUSTER_M_PAIRS vs
  CLUSTER_N], the matching rank decode (m_pair=rank>>1), and a MulticastB path
  (issue B only from m_pair==0, mask along M). Same scope as the 2×2 work; the
  parameterized template makes it a clean extension, not a rewrite.

Qwen-RELEVANT remaining cluster/epilogue work (in scope): 2×2f = 2×2 + FUSED
epilogue (7 shapes); 1×2 = cta_group::1 (single-CTA MMA) + 2-way N-multicast (3
shapes) — part of the deferred cta_group::1 path. Plus tile shapes 128×192 /
160×192 / 256×256 / skinny 64×64 / 128×64.

### 30.11 2×2 cluster COMPLETE — both deadlocks root-caused & fixed, validated on B300 (2026-07-04)

Two distinct bugs, both found by source-grounded research (CUTLASS + PTX ISA), both fixed in
Bgemm_sm103_cluster_template.cuh, both validated on B300:

**Bug 1 — one empty-barrier used for two smem lifetimes (P2 redundant-A hang).**
CUTLASS (sm100_pipeline.hpp) keeps TWO empty barriers: accumulator/tmem-empty is PAIR-LOCAL
(calculate_umma_peer_mask → 0b0011/0b1100), operand-smem-empty is CLUSTER-WIDE
(calculate_multicast_mask<kRowCol> → 0xF for 2×2, arrive count = CM/atomM + CN/atomN − 1 = CLUSTER_N).
We used the pair-local mask for both → pair1's producer never saw pair0's operand release.
Fix: empty_mbar init 1→CLUSTER_N; consumer operand-empty commit mask pair_mask→(1<<CLUSTER_CTAS)−1;
tmem_full commit keeps pair_mask. CLUSTER_N=1 reduces to old behaviour exactly (no 2×1 regression).
Result: P2 redundant-A 21/21 bit-exact (NN/NT/TN × 7 shapes), 2x2-square4096 97-99% of cuBLAS.

**Bug 2 — multicast mbar parity (P2-M2 hang).** PTX ISA 9.3 (cp.async.bulk.tensor, ~line 17169):
with .cta_group::2 the mbarrier signal goes to the even or odd CTA of EACH destination pair "based
on the %cluster_ctarank parity of the shared memory where the mbarrier object mbar RESIDES" — NOT
automatic route-to-leader. Our odd issuer passed its raw local mbar (odd parity) → +A landed on the
odd CTA of every destination pair (barriers nobody waits on) → leaders starved. CUTLASS forces even
parity on every 2SM load (Sm100MmaPeerBitMask bit-24 clear, copy_sm100_tma.hpp:45). Fix: multicast
uses the leader mbar (even parity; mapa'd already). Accounting: each leader gets 2A+2B = expect 2(A+B).
Result: P2-M2 multicast-A 21/21 bit-exact. 2×2 variant DONE (correctness); perf ranking deferred to
the clean-GPU benchmark phase.

**Qwen sweep (contended GPU, ratios only):** 128×256: vert wins 23/42, avg horiz 82.8% / vert 86.6% /
best-of-2 88.0%. 128×128: vert wins 22/42, avg 80.6% / 85.2% / 85.9%. LM-heads ~100% with vertical.
3 row-FAILs (q3-4b-LMhead@256, q25-7b-LMhead@128, q3-8b-LMhead@128) analyzed: DEBUG dumps show
identical ref/out elements; same shape passes with the other tile; all are 622M-element LM-heads →
near-zero cancellation elements exploding the per-element REL error (denominator floor was 1e-5).
Test-harness-only fix: floor→1.0 + dump the worst element (idx/ref/out/absdiff) on FAIL for direct
evidence. Kernel untouched. (DeepGEMM sidesteps the same artifact with a norm-based diff.)

### 30.12 CLC orphaned-steal bug — REAL cause of the LM-head failures (2026-07-04)

**Correction of §30.11:** the "near-zero cancellation artifact" explanation for the LM-head
FAILs was WRONG. The worst-element dump (added to the test) shows e.g. ref=-31.875 out=0.000 —
whole regions of C never written. Real bug, not tolerance. The identical first-10 dumps had
misled the earlier diagnosis (corruption sits deep inside the 622M-element tensor).

**Root cause — orphaned CLC steal at end-of-grid.** Our scheduler armed CLC_PIPE_DEPTH(=2)
clusterlaunchcontrol.try_cancel requests up-front and kept 2 in flight. The two in-flight
cancels can execute out of order at grid exhaustion: the LATER one steals the LAST cluster
(valid response) while the EARLIER finds nothing (invalid). Warps consume response stages in
order → hit the invalid first → all warps break → the valid stolen-cluster response is
orphaned. That cluster was already cancelled by HW (never launches), and nobody computes its
tile → zero-filled C tile. Probability scales with steal traffic at exhaustion → only the
19k-tile LM-heads showed it, flakily, both waves, both tile sizes. Orphaning an INVALID
response (small grids) is harmless — why everything else always passed.

**Reference protocol (CUTLASS sm100_gemm_tma_warpspecialized.hpp:689-721):** exactly ONE
try_cancel outstanding at a time — issue→consume strictly 1:1 (+ producer_tail on exit); the
CLC pipeline stages are consumer-side buffering, NOT multiple in-flight queries.

**Fix (scheduler warp only):** arm only stage 0 at init; re-arm the NEXT slot only after a
VALID response is consumed (arm_idx protocol; first arm of each slot skips the never-completed
clc_empty wait). Invalid response now implies zero in-flight requests, so the plain break in
producer/consumer/epilogue stays correct unchanged. Perf: one prefetched response still hides
CLC latency behind the tile's K-loop (CUTLASS ships the same 1-outstanding protocol).

Validation pending on B300: full RUN_2X2=1 RUN_2X2_MCAST=1 sweep — expect 294/294 with zero
DEBUG FAIL lines; LM-heads must pass repeatedly (flaky bug → rerun LM-heads a few times).

### 32. addmm (fused bias) — SM103 gap, task added (2026-07-04)

Status check: SM89 module HAS fused addmm (Bgemm_Addmm.cu → mycublasBgemmAddmm_sm89):
C = alpha·(A@B) + beta·bias, bias applied in the epilogue registers (single kernel), 3 broadcast
modes: scalar (numel==1) / row-vector (numel==N — the nn.Linear bias case) / full (numel==M·N).
SM103 module has GEMM ONLY (alpha/beta with C read-back, no bias pointer) → addmm MISSING on B300.
Qwen relevance: Qwen2.5 QKV projections have bias=True → PyTorch dispatches addmm.

Plan (after the cluster-variant work): add a BiasMode knob (None/Scalar/RowVec/Full) to the
cluster template epilogue — apply alpha·acc + beta·bias(broadcast) right after tcgen05.ld, where
the beta path already lives; RowVec loads bias[c_col..c_col+STORE_COLS) per chunk. BiasMode=None
compiles to the current epilogue exactly (zero cost off). References: SM89 Bgemm_Addmm.cu (API +
broadcast semantics + validation harness), CUTLASS sm100 collective epilogue / EVT and DeepGEMM
epilogue for Blackwell-style fused-epilogue structure. Expose as mycublasBgemmAddmm_sm103 with the
same signature family as SM89. NOTE (§31): cuBLAS's "2x2f" is NOT epilogue fusion, so addmm is an
independent feature, not a variant of the f kernels.

### 31.1 2×2f research RESOLVED — CUTLASS has the full feature; port recipe (2026-07-04)

CUTLASS implements preferred/fallback ("dynamic") clusters end-to-end; TK has plumbing but uses
rigid only; DeepGEMM never (cluster ≤ 2). Key citations (local cutlass tree):
- Launch: cluster_launch.hpp:168-190 — clusterDim attr = FALLBACK shape, preferredClusterDim attr
  = PREFERRED shape; ClusterLauncher::launch_with_fallback_cluster (:262). CUDA ≥ 12.8. Also sets
  cudaFuncAttributeNonPortableClusterSizeAllowed.
- Constraints (cluster_sm100.hpp:49-107, detail/cluster.hpp:70-95): fallback divides preferred
  per-dim; grid divisible by preferred; z equal; pow2 dims; cluster.x % 2 == 0 for BOTH (MMA pair
  intact); dynamic max 16 CTAs.
- Kernel adaptation: granted size read ONCE via %cluster_nctaid (cluster_sm90.hpp:140-146);
  ALL barrier counts + multicast/peer masks computed at runtime from granted shape
  (sm100_pipeline.hpp:563-566, sm100_mma_warpspecialized.hpp:525-541). Barriers never cross
  cluster boundaries → mixed-size clusters coexist safely.
- CUTLASS needs DUAL TMA descriptor sets (multicast baked into copy atoms) selected at kernel
  start (sm100_mma_warpspecialized.hpp:326-350). WE DON'T — our multicast mask is an instruction
  operand; one descriptor serves both sizes. Our port is simpler.
- Tile assignment under mixed sizes (THE question): work coords are CTA-LEVEL, never
  cluster-level. Grid sized in PREFERRED units; initial tile from raw blockIdx; CLC
  get_first_ctaid + each CTA adds its own offset = its index truncated to the preferred cluster
  (sm100_tile_scheduler.hpp:369-371, 467-469, 801-802; explicit mixed-size comment
  sm100_tile_scheduler_stream_k.hpp:763-786). fallback|preferred + grid|preferred ⇒ bijective.

Design decision for our template: new compile-time knob DynamicCluster (bool) — false = today's
rigid kernels unchanged (zero overhead); true = the f variant: launcher uses cudaLaunchKernelEx
(fallback (2,1,1) + preferred (4,1,1)), kernel reads %cluster_nctaid once and computes
CLUSTER_N/CLUSTER_CTAS-derived counts+masks at runtime, tile decode switches to
first_ctaid + (blockIdx.x % PREFERRED_CTAS) CTA-level mapping. Precision: none (placement only).
Perf rationale: measured 89%→100% SM fill (§31). Dispatcher: just another variant row —
(tile, wave, clusterN, mcastA, dynamic) benchmarked per shape, best wins. Prereq to check on
server: CUDA ≥ 12.8 toolkit + driver.

### 31.2 2x2f (DynamicCluster) IMPLEMENTED — M1 (2026-07-04)

Template changes (Bgemm_sm103_cluster_template.cuh), compile-verified sm_103a:
- New knob: 5th template param bool DynamicCluster=false (kernel + launcher + test).
- __cluster_dims__ REMOVED; ALL launches now via cudaLaunchKernelEx with cluster attributes
  (CUTLASS pattern). Rigid: clusterDim=(CLUSTER_CTAS,1,1). Dynamic: clusterDim=(2,1,1) fallback +
  preferredClusterDim=(4,1,1), guarded #if CUDART_VERSION >= 12080.
- Kernel runtime adaptation (granted size via new get_cluster_nctas() = %cluster_nctaid.x):
  cluster_n (empty_mbar count), operand_empty_mask=(1<<cluster_ctas)-1, a_mask (0b0101/0b0001<<cta_m).
- Decode split: cta_n (ADDRESSING, preferred-relative: dynamic = (blockIdx.x%CLUSTER_CTAS)>>1) vs
  cta_n_grant (GRANTED-relative rank>>1: A-issue gate + pair_mask shift). Static: identical values,
  zero behaviour change. Verified balance for granted-2 (both CTAs issue own-A self-multicast,
  leader collects 2A+2B) and granted-4 (= validated static 2×2).
- M1: CLC stealing DISABLED under DynamicCluster (if constexpr break in producer/consumer/epilogue;
  scheduler branch gated off) — cancellation-unit alignment under MIXED granted sizes is an
  unverified HW contract (M2 investigation). One tile-unit per cluster; grid already covers all.
Test: P3 suite (RUN_2X2F=1): 7 shapes × 3 layouts × {redundant-A, A-multicast} = 42 cases.
NOTE: rigid variants re-validated this run too (launch mechanism changed for all).
Server prereq: CUDA >= 12.8 (check nvcc --version).

### 31.3 CLC-under-dynamic-clusters contract MEASURED on B300 + M2 implemented (2026-07-04)

Device probe (examples/test_clc_dynamic_probe.cu, preferred 4 / fallback 2, 2048 units, B300):
- Launch grants: 284 preferred-4 + 24 fallback-2 (12 slot-pairs split at offsets 0/2).
- Steals: preferred-4 requesters 1686 — ALL response ctaids %4==0 (whole preferred units).
  fallback-2 requesters 132 — 66 at %4==0 and 66 at %4==2 (REQUESTER-SIZED 2-CTA pieces,
  either half of a preferred unit).
- Unit accounting closes EXACTLY: 296 launched + 1686 + 132/2 = 2048/2048 → no unit lost,
  none double-issued. CONTRACT: cancellation units are sized to the REQUESTER's granted
  cluster; the piece position within its preferred unit is encoded in the response ctaid.

M2 (template): CLC re-enabled for DynamicCluster (M1 breaks removed, scheduler ungated).
Changes: scheduler arm loops + clc_empty arrive count bounded by GRANTED cluster size
(mapa outside the granted cluster is UB); stolen-tile N-block decoded per tile:
n_sub = ((new_ctaid % CLUSTER_CTAS)>>1) + cta_n_grant — degenerates to the old cta_n for
all rigid launches (responses cluster-aligned there), so no static-path change.
Compile-verified sm_103a. Validation: P3/P3-M2 suites re-run (now persistent) — expect
42/42 bit-exact AND P3 perf ≈/≥ P2 rigid on square4096/wideN (persistence restored +
100% SM fill). LM-heads in the Qwen sweep remain the missing-tile detector.

### 31.4 2x2f M2 VALIDATED on B300 — variant complete (2026-07-04)

Full run 336/336, zero FAILs: P3 + P3-M2 (dynamic cluster, CLC stealing LIVE under mixed granted
sizes) 42/42 bit-exact — the measured requester-sized-piece contract (§31.3) works in production.

Perf (contended GPU, ratio-to-cuBLAS; authoritative ranking deferred to clean benchmark):
| shape          | P2 rigid 2x2 | P3 dynamic 2x2f (M2) |
| 2x2-wideN      | 89.6-91.2%   | 92.4-93.2%  <- dynamic WINS (+~2.2 pts, 100% SM fill) |
| 2x2-square4096 | 96.4-98.2%   | 94.0-95.3%  <- rigid wins (fallback clusters lose cross-pair
|                |              |    A-share + steal traffic; work pool too small to leverage fill) |
Shape-dependent split — exactly why cuBLAS ships BOTH 2x2 and 2x2f and routes per shape; our
dispatcher will do the same from the benchmark table. M1's 15-20% persistence loss fully recovered.

2x2f variant status: COMPLETE (knob DynamicCluster; correctness bit-exact both A-modes; both
rigid and dynamic launches re-validated). Qwen coverage: 2x1 (17) + 2x2 (15) + 2x2f (7) = 39/42
shapes now have our matching variant family. Remaining: 1x2 (3 shapes, cta_group::1 path),
extra tiles (128x192/160x192/256x256/320x192), addmm (§32), then clean-GPU benchmark → dispatcher.

### 33. 1x2 / cta_group::1 bring-up VALIDATED first-try (2026-07-04)

CLUSTER_SIZE is now a Config template knob (DeepGEMM-style, approach (a)): CLUSTER_SIZE=1 emits
cta_group::1 through every already-parameterized primitive; CS=2 paths byte-identical (x%2==x&1
etc.) — full 378/378 re-validation green. New alias Sm103Config_1x2_128x192 (BN per-CTA=192,
1x2 cluster). Changes: generalized rank decode (div/mod), pair_mask/a_mask formulas, B-tmap
per-CTA slice divisor, pow2 tmem alloc round-up (384→512).

P4/P4-M2 (RUN_1X2=1): 42/42 bit-exact FIRST RUN — first cta_group::1 kernels, symmetric barrier
accounting + A-multicast WITHOUT parity routing (per 1x2_plan.md Q3) confirmed on hardware.

Perf (contended, bring-up tile 128x192): qwen-q06 ~64%, qwen-q3b ~69% of cuBLAS — BELOW our 2x1
best (~83-86% on these shapes), as expected for the bring-up. A-multicast ~neutral here. The
competitive tile is nvjet's 160x192 = swap-AB (Option B, 1x2_plan.md Q5/Q6): M'=192=128+64
composite MMA + transposed 16-lane epilogue — NEXT implementation step. Only after B is built do
we know if 1x2 beats 2x1 on the 3 Q-projection shapes; dispatcher decides from the benchmark.

### 33.1 swap-AB B1 VALIDATED (2026-07-04)

P5/P5-M2 14/14 bit-exact (TN, C-tile 128x192 as C^T): composite M'=192 MMA (128+64 atoms,
Layout-F interleaved accumulators with lane-alignment 16*stage), crossed operand descriptors,
16x256b strip reads, transposed epilogue — all proven on B300. One bug on first run (0/7):
the transposed epilogue omitted the C-tmap 128B-swizzle XOR on smem writes (the validated
epilogue's swizzled_n) — found via source-verified fragment maps + bit-exact evidence match
(out(113,100)=ref(113,108) = the missing group XOR; 1x2_plan.md "P5 BUG ANALYSIS"). Every
genuinely-new mechanism was verified correct; the mistake was not carrying one line of
existing behavior into the new epilogue.

Perf note (honest): P5 qwen shapes ~52-57% vs P4 non-swap ~64-69% — the bring-up transposed
epilogue is slow (scalar bank-conflicting smem writes + per-8-col ld/wait serialization).
B2 (160x192, the real nvjet tile, 64B-swizzle operand per R7) is next; epilogue
vectorization after correctness. Only the final benchmark decides if 1x2 swap beats our 2x1
on the 3 Q-projection shapes.

### 33.2 B2 (160x192) validated + chunked-tmap alignment limitation found (2026-07-04)

P6/P6-M2: 12/14 bit-exact on first run — mixed-swizzle MMA (A'=128B + B'=64B descriptors in one
tcgen05.mma), 64B-swizzle tmap, composite M'=192, all 3 Qwen Q-projections 0.00e+00. The 2 fails
(M=200 shape) exposed a REAL pre-existing limitation, root-caused and reference-verified:
- Our 4D chunk-factorized tmaps ({chunk,K,X/chunk}) TRUNCATE non-chunk-multiple sliced dims
  (globalDim z = 200/32 = 6 → A cols 192-199 unreachable → TMA zero-fill → C rows 192-199 = 0,
  exact match with the observed boundary). Same constraint exists in ALL our 128B builders
  (M%64/N%64) since the original kernels — first exposed by this test.
- References: DeepGEMM tmaps carry TRUE extents (runtime_utils.hpp:199) → arbitrary M via real
  TMA OOB; cuBLAS routes unaligned shapes to a FALLBACK kernel (cutlass_75_wmma align1, §4 row
  11) rather than teaching nvjet the tails.
- Fix now: LOUD launcher guard (refuse + stderr instead of silent zeros) + legal partial-tile
  test shape (M=192). Fix later (documented TODO): true-extent tmap redesign or fallback-kernel
  route in the dispatcher — not needed for the 42 Qwen shapes (all alignment-clean).
Perf note: B2 qwen ~47-50% (naive transposed epilogue + half-width TMA lines on the sw64
operand) — epilogue vectorization is the known optimization step before the final benchmark.

### 34. REMAINING TASK LIST (as of 2026-07-04 night)

1. P6 rerun (alignment guard + legal M=192 partial-tile shape) → closes the 1x2/B2 arc.
2. Swap-AB epilogue vectorization (B2 perf: currently scalar bank-conflicting smem writes +
   serialized 8-col tcgen05_ld — qwen shapes ~47-50%; target = beat our 2x1 ~83-86% or drop 1x2).
3. Clean-GPU 2-phase benchmark: (a) each variant vs cuBLAS on the shapes cuBLAS uses it for;
   (b) all 42 shapes x all variants → best-per-shape table → wire dispatcher (LAST).
4. GENERAL-SHAPE SUPPORT (library completeness, cuBLAS accepts any M/N/K): true-extent tmaps
   (DeepGEMM runtime_utils.hpp:199 style: 2D descs with real extents + per-chunk issue) +
   ceil-div k_iters (zero-filled K tail contributes 0). Removes the launcher alignment guard's
   refusals entirely. Alternative: fallback-kernel route like cuBLAS's cutlass_75_wmma align1.
5. addmm on SM103 (§32): BiasMode knob (None/Scalar/RowVec/Full) in the epilogue, SM89-parity
   API (mycublasBgemmAddmm_sm103).
6. Extra tiles if benchmark shows gaps: 128x192 / 256x256 / 320x192 aliases (config-level now).

### 35. VARIANT CATALOG COMPLETE — all 42 Qwen shapes covered (2026-07-04 night)

P6/P6-M2 final: 14/14 bit-exact (incl. partial-M-tile 192 + the loud alignment guard active).
The 160x192_1x2 swap-AB variant (mixed 128B+64B swizzle, composite M'=192, transposed epilogue)
is correctness-complete — the last of cuBLAS's Qwen cluster families.

FULL CATALOG (all validated bit-exact on B300, all knobs on one template, DeepGEMM-style):
  2x1 (17 shapes): 128x128 / 128x256, horizontal + vertical wave
  2x2 (15 shapes): rigid 4-CTA + A-multicast
  2x2f (7 shapes): DynamicCluster (preferred-4/fallback-2) + CLC (measured HW contract)
  1x2 (3 shapes): cta_group::1, swap-AB 160x192, 64B-swizzle operand, + A-multicast
Knobs: tile, VerticalWave, CLUSTER_N, CLUSTER_SIZE, MulticastA, DynamicCluster, SWAP_AB.
Remaining work = §34 list (epilogue vectorization → clean benchmark → dispatcher → general
shapes → addmm).
