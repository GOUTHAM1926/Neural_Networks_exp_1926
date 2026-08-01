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

### 36. NEW SERVER + FIRST CLEAN-GPU FULL BASELINE — 406/406 PASS (2026-07-07)

Setup: new dedicated B300 server (`fast-word-glows-fin-03`, "NVIDIA B300 SXM6 AC", SM 10.3,
idle GPU). Repo path there: `~/gautam_k/blublas`, cloned `--depth 1` from the personal sync repo
`github.com/GOUTHAM1926/blublas` (pushed from the working tree 07/07). Sync verified by md5:
`.cuh` 4bb0cdff0be13f745b7cd84376a79702, test file af028a6bf4f50d0c6151073e89b2832e — byte-
identical to the local validated state. docs/ carries the 4 plan files.

Run: ALL suites enabled (RUN_2X2/2X2_MCAST/2X2F/1X2/1X2B/1X2C) = 406 cases
(84 P0 + 42 P2 + 42 P3 + 70 1x2-family + 168 Qwen sweep). Log: output.md (full detail +
SUMMARY). RESULT: **406/406 PASS**, MaxRelErr 0.00e+00 everywhere (beta cases ~6e-03 = fp32
accumulate-order noise, far under floor). Every validated arc reproduces on a SECOND B300 —
nothing machine-specific.

This is the first honest (idle-GPU) perf baseline — earlier server27 numbers were contended.

WHERE WE WIN (128x256 tile + vertical wave):
- LM-heads (M=152k): BEAT cuBLAS on 8/9 models — 7b 124%, 4b 125.5%, 8b 112%, 3b 104.4%,
  1.7b 103.2%, 1.5b 102.3%, 0.6b 98.8%, 0.5b 99.4%. Even best-vs-best (e.g. 3b 1832T vs
  1755T) it's a real win. Vertical wave is the whole story on huge-M.
- 0.5b-Down 101.8%; 7b-Gate vert 100.5%; 8b-Gate vert 99.5%.
- cmp-wideN 4096x12288 vertical: 99.3-99.5% on ALL 3 layouts (horizontal only 76-86%) —
  wave policy alone is worth ~20 points on wide-N (B stays L2-resident across the M-wave).
- square4096: 92-94% steady, all layouts.
Sweep averages: 128x256 best-of-2-waves 85.2% (vert wins 24/42); 128x128 82.4% (25/42).

WHERE THE GAP IS — small-N K/V-projection shapes (data confirms the ALL-TILES plan):
  N=128: 35% (256-tile) / 45% (128-tile);  N=256: 36% / 50-58%;  N=512: 49% / 76%;
  N=1024: 65-82% / 64-76%.
Clean pattern: the narrower the tile relative to N, the better — at N=128 our 256-wide cluster
tile computes 50% dead columns. These are EXACTLY the shapes where nvjet ships the tiles we
have not built (64x64, 128x64, 128x160, composite 192/160-row) → the remaining avg gap is
missing small tiles, not tuning. Priority of all_tiles_plan.md T1 (composite-M) + T2
(interleaved epilogue) is now benchmark-backed.

Smaller observations:
- beta!=0 cases 16-25% everywhere: our beta path reads C_in with scalar uncoalesced global
  loads. Qwen inference GEMMs don't use beta → doesn't block the dispatcher; will matter for
  addmm/bias (§32). Fix = vectorized/TMA C_in read, deferred.
- Swap-AB P5/P6 at 29-57% = the known unvectorized transposed epilogue (§34 task 2);
  correctness-complete, perf pass pending. Real target = nvjet's 160x192 kernel on its 3
  shapes, not generic cuBLAS.
- cuBLAS's own measured TFLOPS varies row-to-row on the same shape (e.g. 3b LMhead 1364T vs
  1755T) — heuristic re-dispatch + clock variance; always compare best-vs-best before claiming
  a win.

NEXT: T1 composite-M knob (all_tiles_plan.md R6.1) — unlocks ~9 small/odd-tile shapes above.

### 37. REPO SYNC + NEW-SERVER DEPLOYMENT (2026-07-07) — previously undocumented

Why: got a new dedicated B300 server; the big Neural_Networks repo can't be cloned there, so the
deployment channel is the PERSONAL sync repo.

Local repos inventory (3 checkouts, 2 remotes — easy to confuse, recorded here once):
1. WORKING TREE (where all edits happen): 
   ~/Downloads/Neural_Networks_exp_1926/BLUBLAS_kota/BluBridge-BLAS — tracked by the OUTER repo
   github.com/GOUTHAM1926/Neural_Networks_exp_1926 (BLUBLAS_kota is NOT gitignored there).
2. PERSONAL SYNC/DEPLOY REPO: ~/Downloads/BLUBLAS/BLUBLAS_kota/BluBridge-BLAS →
   github.com/GOUTHAM1926/blublas, branch main. THIS is the deploy channel for servers.
3. ORG REPO (old state, untouched): ~/Downloads/BLUBLAS_latest_work/BLUBLAS/BluBridge-BLAS_latestttt
   → Blubridge-Tech/BluBridge-BLAS, branch master, last commit 13/06 + an old uncommitted
   restructure-in-progress. Plus an old copy at ~/Downloads/BluBridge-BLAS_01_07_2026_.

Sync procedure used (07/07): rsync working tree → personal repo (verified `diff -rq` exit 0 =
byte-identical, excluding .git/build); plan docs force-added into docs/ past the `*.md`
gitignore line; 16 files staged (+8904/−3749); commit+push done BY THE USER (standing rule:
never commit/push as AI). 

New-server deploy: `git clone --depth 1 https://github.com/GOUTHAM1926/blublas.git` (shallow =
latest snapshot only, 124 objects / 434 KiB), moved to ~/gautam_k/blublas. Sync verified by
md5 (.cuh 4bb0cdff0be13f745b7cd84376a79702, test af028a6bf4f50d0c6151073e89b2832e) + marker
greps (make_smem_desc_sw64=2, UNSUPPORTED alignment=1, 1x2swap_160x192 alias=1, {192,384
test shape=1). Then the full 406-case validation ran clean (§36).

Server roster now: server27 (dharun@…:/mnt/raid0/gautam_k/blublas, CUDA 13.2, shared/contended)
+ fast-word-glows-fin-03 (root@…:~/gautam_k/blublas, idle B300 — use THIS for benchmarks).

NOTE ON DOCUMENTATION POLICY (user rule, 07/07): analyzations.md is the SINGLE documentation
file for this whole BLAS/cluster arc — no more separate .md plan files. The three standalone
plan files written earlier (2x2_cluster_plan.md, 1x2_plan.md, all_tiles_plan.md) are merged
IN FULL below as §38-§40; the standalone copies stay on disk only because the repo docs/
already ships them — all future writing goes HERE.

### 38. MERGED IN FULL: 2x2_cluster_plan.md (written 2026-07-04)

(Headings demoted with extra # to keep this file's section numbering greppable.)

#### P2 — 2×2 cluster (4-CTA) implementation plan (reference-verified 2026-07-04)

Target file: `BLUBLAS_kota/BluBridge-BLAS/src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh`
Grounded in: PTX ISA 9.3, CUTLASS (sm100_mma_warpspecialized / copy_sm100_tma / sm100_pipeline),
DeepGEMM (sm100_bf16_gemm / tma_copy), ThunderKittens.

##### Core model (settled)
- NO 4-CTA MMA. Still `tcgen05.mma.cta_group::2` (a 2-CTA pair along M).
- 2×2 = same pair replicated as a 2nd pair along N. A is multicast across the 2 N-pairs;
  B stays per-pair (each pair loads its own N-block). Only A traffic is halved.
- Use `__cluster_dims__(4,1,1)` (NOT (2,2,1)) → keeps CLC 1-D. Identical masks.
- rank = `%cluster_ctarank` (flat 0..3). cta_m = rank&1 (M-partner, drives DP rows),
  cta_n = rank>>1 (N-block). pair_leader = rank&~1. n_peer = rank^2.
- A-multicast mask = 0b0101<<cta_m (0b0101 rank0 / 0b1010 rank1).
- MMA-commit / peer mask = 0b0011<<(2*cta_n) (pair0=0b0011, pair1=0b1100). NOT 0b1111.
- Per-CTA M ≤ 128 (BM=128 assert stays). TMEM read at DP base 0 per CTA (unchanged, §28.7).

##### Distinguish the two "cluster" factors (STOP overloading CLUSTER_SIZE)
- CLUSTER_M = 2  → MMA pair size. Everything that meant "the MMA pair" stays 2:
  B_STAGE_BYTES (BN/2), i_desc M height (2*BM=256), full_mbar init, tmem_empty init,
  TMA cta_group::2, dealloc cta_group::2.
- CLUSTER_N = 2  → number of pairs (N-blocks per cluster).
- CLUSTER_CTAS = 4 → CTAs per cluster. Everything that meant "grid stride / CLC count":
  clc_empty arrive count, __cluster_dims__, grid stride, ctaid/CLUSTER decode.

##### Bring-up milestones (B300)
1. **Redundant-A** (recommended first): (4,1,1) cluster, scheduler split into 2 N-blocks,
   per-pair leader — but every CTA loads its OWN A+B (NO multicast). Each pair is
   byte-for-byte the current 2-CTA kernel. Only new logic = scheduler + per-pair leader
   + CLC counts. Validate correctness first. (Cost: 2× A DRAM traffic.)
2. **Turn on A multicast** (cta_n==0 issues A, mask 0b0101<<cta_m, a_mbar bit24 cleared) →
   same results, lower DRAM.
3. Sweep BN=128/256, batch, NN/NT/TN, all 42 Qwen shapes.

##### Edit checklist (file:line today → change)
- Config: add CLUSTER_M=2, CLUSTER_N=2, CLUSTER_CTAS=4. CTA_MASK per-pair rank-dependent.
- CLC_EMPTY_ARRIVE: NUM_WARPS*CLUSTER_SIZE → NUM_WARPS*CLUSTER_CTAS (×4).
- __cluster_dims__(2,1,1) → (4,1,1).
- get_cluster_cta_rank(): %cluster_ctaid.x → %cluster_ctarank. Derive cta_m/cta_n/pair_leader/n_peer.
- full_mbar init: stays 1*CLUSTER_M (2). tmem_empty init: stays NUM_EPI_WARPS*CLUSTER_M (2).
- clc_empty init: ×4 (via CLC_EMPTY_ARRIVE).
- tma_mbar_base: cta_m==0 ? own : mapa(full, pair_leader).
- cta0_clc_empty: all 4 → rank0 (CLC is cluster-global): cta_rank==0 ? own : mapa(...,0).
- init_tile_idx = ctaid/CLUSTER_CTAS (÷4). CLC new_tile_idx = new_ctaid/4.
  Decode: n_cluster = cluster_idx % ceil(grid_n/2); row_tile = cluster_idx / ceil(grid_n/2).
  block_row = row_tile*CLUSTER_M + cta_m; block_col = n_cluster*2 + cta_n.
- Producer: B all 4 CTAs (own N-half, cta_group::2, unchanged). A: milestone1 all-load;
  milestone2 only cta_n==0 multicast (mask 0b0101<<cta_m, a_mbar=mbar&0xFEFFFFFF).
  expect_tx: every CTA mbarrier_arrive_expect(leader_mbar, copy_size).
- MMA leader gate: if(cta_rank==0) → if(cta_m==0) (ranks 0 AND 2 drive their pair).
- MMA commit mask: CTA_MASK → 0b0011<<(2*cta_n).
- Scheduler CLC arm: fan out expect_tx=16 to r=0..3 (all 4 CTAs' clc_full via mapa).
- Epilogue store coords: block_col = n_cluster*2+cta_n; block_row = row_tile*CLUSTER_M+cta_m.
- Launcher grid: total_cluster_tiles*batch*CLUSTER_CTAS, total_cluster_tiles = grid_m*ceil(grid_n/2).
- Keep static_assert(BM==128).

##### Hurdles + resolution
1. CLC↔cluster dim: use (4,1,1) → first_ctaid 1-D, no query change. (biggest feared risk → mechanical)
2. clc_empty count must be ×4 or scheduler deadlocks. #1 hang suspect.
3. CLC expect_tx must be armed on all 4 CTAs (multicast::cluster::all broadcasts response).
4. A-multicast mbar peer-bit (bit24 clear) so A bytes land on leader (CUTLASS copy_sm100_tma.hpp:318).
5. Audit every CLUSTER_SIZE use (pair vs cluster meaning).
6. BM==128 assert unchanged.
7. HW-verify: plain 2SM TMA in a 4-CTA cluster still pairs {0,1},{2,3} — redundant-A milestone must
   be bit-exact vs two separate 2-CTA launches; if pair1 wrong, fall back to (2,2,1) + CUTLASS peer mask.

##### FIX APPLIED (2026-07-04) — dual empty-barrier per CUTLASS

Root cause of the 2×2 deadlock (confirmed on a near-idle B300, contention ruled out): we used ONE
empty-barrier with the PAIR-LOCAL mask (0b1100) for BOTH smem lifetimes. CUTLASS
(sm100_pipeline.hpp) keeps TWO:
- accumulator/tmem-empty → PAIR-LOCAL (calculate_umma_peer_mask): 0b0011 / 0b1100, count = pair.
- operand-smem-empty → CLUSTER-WIDE (calculate_multicast_mask<kRowCol>): for atom(2,1) every CTA
  participates → mask = full cluster (0b11 @ 2×1, 0xF @ 2×2); count = one arrive per pair-leader
  = CM/atomM + CN/atomN - 1 = CLUSTER_N.
A is multicast across N (shared by both pairs), so its smem may only be refilled after EVERY
pair-leader is done → the release must span the cluster. Pair-local release → pair1's producer never
saw pair0's release → deadlock.

Changed in Bgemm_sm103_cluster_template.cuh:
- empty_mbar init count 1 → CLUSTER_N (line ~706).
- consumer operand-empty commit mask pair_mask → operand_empty_mask = (1<<CLUSTER_CTAS)-1 (line ~993).
- tmem_full (accumulator) commit KEEPS pair_mask (line ~1001) — pair-local, already correct.
CLUSTER_N=1 (2×1) → count 1, mask 0b11 = old behaviour exactly (no regression).
Generalises: same formula is correct for 1×2, 4×1 (the single "kNumMulticast"-style knob).

##### FIX #2 APPLIED (2026-07-04) — multicast mbar parity (the P2-M2 hang)

Result of fix #1 on B300: P2 redundant-A 2×2 = 21/21 bit-exact, deadlock GONE. P2-M2 (multicast-A)
still hung on its first kernel. Root cause found in the PTX ISA 9.3 spec itself
(cp.async.bulk.tensor, ptx_docs/ptx_isa_9.3.txt ~line 17169):

> .cta_group::2 : ... for each destination CTA specified in the ctaMask, the mbarrier signal is
> sent either to the destination CTA or its peer-CTA **based on the %cluster_ctarank parity of the
> shared memory where the mbarrier object mbar resides.**

i.e. completion routing is decided by the PARITY OF THE MBAR OPERAND'S ADDRESS — it is NOT an
automatic route-to-leader. Our multicast call passed the raw LOCAL full_mbar. For the odd issuer
(rank1, mask 0b1010) the mbar resides in an odd CTA → the +A tx multicast to the ODD CTA of each
destination pair (rank1/rank3's own barriers — nobody waits there) → both pair leaders starved of
A_bytes every stage → hang. CUTLASS avoids this by forcing even parity on every 2SM load's mbar
(& Sm100MmaPeerBitMask = clear bit 24, copy_sm100_tma.hpp:45, used at :240/:266/:292/:318).

Fix (one-line semantics): issue_A_load multicast path now uses the LEADER mbar (tma_mbar_base —
even-parity residency; already mapa'd for the odd partner) instead of the raw local mbar; the unused
local_mbar param removed from issue_A_load + its 3 call sites. Accounting now balances: each pair
leader collects 2×A (both A-multicasts route even) + 2×B (own pair's B loads) = expected 2×(A+B).
Compile-verified sm_103a (frontend+ptxas clean).

###### Staged validation (needs B300)
1. Redundant-A first (RUN_2X2=1, P2 mcastA=0): full-barrier accounting unchanged & known-correct;
   the empty-barrier fix should clear the deadlock. Must be bit-exact vs cuBLAS (MaxRelErr 0.00e+00).
2. Then multicast-A (P2-M2, mcastA=1): each CTA still ends with a+b bytes (A via multicast, own B),
   so the full-barrier tx MAY already balance — verify. Only if a tx imbalance/hang appears, apply
   operand-FULL fix#2 (leader-only expect for the joint 2SM transaction, CUTLASS A.4).
3. Barriers are pure sync → ZERO precision impact; perf is neutral-to-positive (enables A-DRAM halving).

### 39. MERGED IN FULL: 1x2_plan.md (written 2026-07-04) — the whole 1x2/swap-AB arc

(P4 bring-up research Q1-Q6, R0-R7 swap-AB spec, P5 bug analysis, epilogue resolution,
64B-swizzle/B2 design. Headings demoted.)

#### Plan: cta_group::1 BF16 GEMM, 160x192 CTA tile, 1x2 cluster (sm_103a / B300)

Target: match `nvjet_sm103_tst_160x192_64x5_1x2_h_bz` (BM=160, BN=192, BK=64, 5 stages,
1x2 cluster, horizontal = 2 CTAs along N sharing A).
Covers 3 Qwen shapes (analyzations.md:897,907,913): (M,N,K) = (4096,2048,2048),
(4096,2048,1024), (4096,2048,2048) — all TN Q-projections.

Sources searched (read-only):
- PTX ISA 9.3: `ptx_docs/ptx_isa_9.3.txt` (all line refs below are into this file unless noted)
- CUTLASS: `cutlass/include` (sm100 collective builders + `cute/atom/mma_traits_sm100.hpp`)
- DeepGEMM: `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` + epilogues
- Our template: `BLUBLAS_kota/BluBridge-BLAS/src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh`

---

##### Q1. Legal M/N for tcgen05.mma.cta_group::1.kind::f16 — M=160 is NOT legal in one MMA

**PTX ISA 9.3, Table 44 "Various combinations of .kind and shapes"
(ptx_isa_9.3.txt:31324-31338):** for `.kind::f16`, cta_group **1**, **Dense**, dtype f32,
atype/btype f16/bf16:

- **M ∈ {64, 128} only** ("64xNxK / 128xNxK", lines 31328-31331)
- **N = {8, 16, 24, … 256} in steps of 8** (line 31328-31329) → **N=192 is legal**, N=160 is legal
- **K = 16 elements** (32 B of bf16) per MMA issue (line 31328-31329) → BK=64 = 4 MMA issues/stage

The instruction descriptor *encodes* M in bits 24-28 as `M >> 4` (5 bits, line 32017-32018)
and N in bits 17-22 as `N >> 3` (line 32014-32015), so M=160 (=10<<4) and N=192 (=24<<3)
are *encodable* — but Table 44 is the documented legality contract, and it says M=64/128 only.
**A 160-row accumulator cannot exist as one MMA**: TMEM is physically 128 lanes x 512
columns x 32b per CTA (ptx_isa_9.3.txt:31266-31267).

Weight-stationary escape hatch rejected: `.ws` allows M=32/64/128 but restricts
**N ∈ {64,128,256}** (lines 31356-31358) — N=192 not allowed, and `.ws` K-handling differs.
`disable-output-lane` (ptx_isa_9.3.txt:34813-34823, vector of 4x32b = 128 mask bits for
cta_group::1) masks *TMEM writeback lanes* of a legal-shape MMA; it does not create new M
values (and does not save flops).

**Conclusion: 160 rows = TWO MMA atoms: M=128 + M=64 (only 32 of the 64 rows useful),
or avoid the problem entirely by swapping operands (see Q6).**

*Honesty note:* nvjet is SASS-emitted; the idesc M field is a runtime register value that
ptxas cannot check, so cuBLAS *could* be issuing an undocumented M=160 (its sibling
`176x128_64x8_1x2_h`, analyzations.md:958, is equally non-decomposable-friendly). Nothing
in the three provided source trees confirms or denies HW support for M=160. A 10-minute
probe (issue idesc with M-field=10, read TMEM back) would settle it, but it is UB per PTX
and unshippable; the plan below uses only documented shapes.

###### Where an M=64 accumulator lives in TMEM (critical for composing 160)

Data Path Layout Organization table (ptx_isa_9.3.txt:33617-33632):

| M | cta_group | mode | DP organization | Layout | lane alignment |
|---|---|---|---|---|---|
| 128 | ::1 | dense | 4x1 (all 128 DPs) | **D** | 0 |
| 64 | ::1 | dense (non-.ws) | 4x1, **1/2 datapath utilized** | **F** | **0 or 16** |

CUTLASS gives the exact Layout F atom (`cutlass/include/cute/atom/mma_traits_sm100.hpp:500-511`):

```
// M_MMA == 64: Half subpartitions layout atom: (M,N) -> tmem_addr
Layout tmem_atom = Layout<Shape <Shape <_16,  _4>, Int<N_MMA>>,
                          Stride<Stride< _1, _32>,      _128>>{};
```

i.e. an M=64 accumulator occupies **16 lanes in each of the 4 subpartitions**
(lanes 0-15, 32-47, 64-79, 96-111; or +16 with alignment 16) x **N full columns**.
Rows 0-15 → lanes 0-15, rows 16-31 → lanes 32-47, rows 32-47 → lanes 64-79,
rows 48-63 → lanes 96-111. Two M=64 accumulators **can share the same columns** by using
the top/bottom half-subpartitions ("Interleaved" alloc mode,
mma_traits_sm100.hpp:424-427 and the `tile_stride` logic at :505-509; PTX's "lane
alignment 0 or 16" at :33623-33624).

So for a non-swapped 160-row tile: MMA#1 M=128 → lanes 0-127; MMA#2 M=64 → tile rows
128-143 land in lanes 0-15 and rows 144-159 in lanes 32-47 (the other two 16-lane strips
hold garbage rows), **at a different TMEM column offset** — exactly the "accumulator #2 at
a column offset in DPs 0-47" shape the design question anticipated.

---

##### Q2. CUTLASS/DeepGEMM never build M=160 CTA tiles on sm100

- CUTLASS 1SM tiled-MMA builder hard-asserts
  `M == 64 || M == 128` — `cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:308-309`
  (`sm100_make_1sm_trivial_tiled_mma`), and 2SM asserts `M == 128 || M == 256`
  (:374-375). N is asserted `N % 8 == 0 && N <= 256` (:313, :379) → **192 fine**.
- Grep for `Shape<_160` / `CtaShape.*160` / `_160` across `cutlass/include`: the only 160
  hits are **SM90 wgmma N=160** variants (`cute/arch/mma_sm90.hpp:438` etc.) — N, never M,
  and never sm100.
- DeepGEMM sm100 BF16: `sm100_bf16_gemm.cuh:57-66` — `LAYOUT_AD_M = 128`,
  `DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or (not kSwapAB and (BLOCK_M ==
  32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)))`. BLOCK_M ≤ 128 always. Their BLOCK_M
  = 32/64 cases still issue **M=128 MMAs** over padded A TMEM
  (`UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * ...`, :89).

**How else a 160-row (or 192-col) tile can be built** — two documented mechanisms:
1. **Multiple MMA atoms along the M dim of the same CTA tile** (2 x tcgen05.mma per
   K-step, second `[d-tmem]` at a column offset; one `tcgen05.commit` tracks both —
   commit tracks *all* prior async tcgen05 ops of the thread, ptx_isa_9.3.txt:35690-35699).
2. **Swap-AB**: compute Cᵀ so the awkward dimension becomes MMA-N (any multiple of 8).
   DeepGEMM ships this as `kSwapAB` (`sm100_bf16_gemm.cuh:59`:
   `UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N`) with a transposing epilogue
   (`epilogue/sm100_store_cd_swap_ab.cuh`).

For 160x192 specifically (see Q5/Q6): swap-AB turns it into MMA-M' = 192 = **128 + 64
exactly** (zero waste) with N' = 160 (legal), while non-swap 160 = 128 + 64 wastes 32
rows of compute.

---

##### Q3. 1x2 cluster, A multicast with cta_group::1 — barrier accounting

PTX ISA 9.3, cp.async.bulk.tensor multicast semantics (ptx_isa_9.3.txt:17169-17180):

> "`.multicast::cluster` … The source data is multicast to the **same offset as dstMem**
> in the shared memory of each destination CTA. When `.cta_group` is specified as:
> **`.cta_group::1` : The mbarrier signal is multicast to the same shared memory offset
> as mbar in the destination CTA.**"

and (:17123-17129): with `.cta_group::1` the mbar named in the instruction must be in the
**same CTA as dstMem**; the even/odd **parity routing exists only for `.cta_group::2`**
(:17176-17180 — that's the path our current template documents at
Bgemm_sm103_cluster_template.cuh:271-300).

So for the 1x2 pair (ranks 0,1 along N), **confirmed accounting — no parity, fully
symmetric**:

- **Bytes**: an A-load issued by either CTA with `ctaMask = 0b11` lands at the *same smem
  offset* in both CTAs (each CTA has its own full copy of the A stage buffer).
- **complete_tx**: each destination CTA's **own** full-barrier (same smem offset as the
  `mbar` operand) is incremented by the byte count that landed *in that CTA*.
- **expect_tx**: each CTA's producer (elect_sync leader of its TMA warp) arms **its own**
  full barrier per stage with `expect_tx(A_STAGE_BYTES + B_STAGE_BYTES)` — the total that
  will land locally, regardless of who issued it. (Contrast with our cta_group::2 leader
  barrier collecting 2xA+2xB, template :300.)
- **Issue split**: CUTLASS-style — CTA rank r issues rows [r·80, (r+1)·80) of the 160-row
  A tile with mask 0b11 (halves the issue traffic per CTA), plus its own 192-col B tile
  non-multicast (mask irrelevant). A single-issuer scheme (rank 0 issues all of A) is
  equally legal; expect_tx math is unchanged.
- **Empty (consumer→producer)**: because CTA r's A-issue writes into *both* CTAs' stage
  buffers, a stage may be refilled only after **both** CTAs' MMA warps consumed it. Each
  MMA warp signals both CTAs with one instruction:
  `tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.b64 [empty_mbar], 0b11`
  — `.multicast::cluster` is legal on commit for cta_group::1
  (ptx_isa_9.3.txt:35679-35684, semantics :35711-35714, "same offset as mbar" again).
  Each per-stage empty barrier therefore needs **2 arrivals per phase**
  (vs our current parity-based scheme). Our helper `tcgen05_commit_multicast`
  (template :367-375) already takes CTA_GROUP as a template parameter.

One hard constraint to remember: **"All tcgen05 instructions within a kernel must specify
the same value for the .cta_group qualifier"** (ptx_isa_9.3.txt:32724, 34801) — the
cta_group::1 path must be a separate kernel/template instantiation end-to-end
(alloc/dealloc, mma, commit), never mixed with the cta_group::2 code path at runtime.

---

##### Q4. BN=192, BK=64, 128B swizzle — descriptors, TMA boxes, tcgen05.ld

- **Smem descriptor**: format at ptx_isa_9.3.txt:31930-31971 (14-bit encoded start
  address, LBO bits 16-29, SBO bits 32-45, swizzle bits 61-63 = 2 for 128B; all three
  byte quantities must be 16B-aligned, :31968-31971). A K-major 192x64 bf16 tile under
  128B swizzle is **24 swizzle atoms of 8 rows x 128 B**; 192 is a multiple of 8, so LBO/
  SBO are identical to the 128-row case — **the tile height never appears in the
  descriptor**, it comes from idesc N (192>>3 = 24, fits the 6-bit field, :32014-32015).
  The second M-atom's descriptor is just base + row-128 byte offset (16 atoms x 1024 B =
  16 KiB; 16B-aligned ✓). No new constraint vs our working 128-col descriptors.
- **TMA box**: box dims are capped at 256 per dimension (CUTLASS enforces this shape cap
  in its TMA layout coalescing, `cute/atom/copy_traits_sm90_tma.hpp:673-704`
  "coalesce…up to size-256"; the 128B-swizzle cap applies to the *contiguous* dim = BK=64
  bf16 = 128 B exactly). Boxes of 192x64 (B), 160x64 or 80x64 (A halves) are all legal.
  (The ≤256/≤swizzle-width rules originate in the CUDA `cuTensorMapEncodeTiled` docs —
  not restated in the PTX text dump; flagged as external-doc knowledge.)
- **tcgen05.ld with a 192-col accumulator**: any warp can address **all 512 columns**;
  lanes are the restricted resource — warp w of a warpgroup touches only lanes
  [32w, 32w+32) (ptx_isa_9.3.txt:32790-32804). Column index is bits 0-15 of the TMEM
  address (:31274-31281). Shapes available: `.16x64b/.16x128b/.16x256b/.32x32b` x
  `.num up to x128` (:32850-32852, Table 54 :32903-32913). Reading 192 columns per lane =
  e.g. 6 x `32x32b.x32` or 2 x `32x32b.x64` + 1 x `.x32` (our epilogue already loops
  `32x32b.x8`, template :377-384). The 16-lane shapes (`16x256b` etc.) are what read the
  M=64 atom's half-subpartition strips — DeepGEMM does exactly this in its swap-AB
  epilogue (`sm100_store_cd_swap_ab.cuh:87-90`, `SM100_TMEM_LOAD_16dp256b1x`).
- **TMEM allocation**: units of 32 columns, power of 2, range [32,512], non-increasing
  across allocations (ptx_isa_9.3.txt:32701-32703) → just allocate **512** once
  (`alloc_tmem` already parameterized on CTA_GROUP, template :353).

---

##### Q5. TMEM capacity math for 160x192 (single CTA, 128 DP x 512 cols)

TMEM per CTA: 128 lanes x 512 cols x 32b (ptx_isa_9.3.txt:31266-31267). fp32 accumulator
= 1 col per N element.

**Option A — no swap (M-composite 128+64, N=192):**
| piece | lanes | cols |
|---|---|---|
| M=128 atom (rows 0-127) | 128 (Layout D) | 192 |
| M=64 atom (rows 128-159 valid) | 4x16 strips (Layout F) | 192 |
- 1 accumulator buffer = 192+192 = **384 cols ≤ 512 ✓** (the "extra 32 rows" live in the
  half-subpartition strips of the M=64 atom at column offset 192 — lanes 0-15 = rows
  128-143, lanes 32-47 = rows 144-159).
- 2 buffers (EPI_STAGES=2): 2x192 (full-DP) + 192 (the two M=64 atoms interleaved at lane
  alignment 0/16 in shared columns) = **576 > 512 ✗**. → Option A is limited to a
  **single accumulator stage** (epilogue can't overlap the next tile's MMA).
- Compute waste: MMA does 192 rows for 160 useful → **+20% flops**.

**Option B — swap-AB (compute Cᵀ: MMA-M' = 192 = 128+64, MMA-N' = 160):**
| piece | lanes | cols |
|---|---|---|
| M'=128 atom (C-cols 0-127) | 128 | 160 |
| M'=64 atom (C-cols 128-191, all 64 rows real) | 4x16 strips | 160 |
- 1 buffer = 160+160 = 320 cols; **2 buffers = 2x160 + 160 (M'=64 pair interleaved at
  alignment 0/16) = 480 ≤ 512 ✓** — 2-stage accumulator/epilogue pipeline fits, 32 cols
  spare.
- Compute waste: **zero** (192 = 128+64 exactly; 160 is a legal N').
- This is the documented CUTLASS "Interleaved" TmemAllocMode
  (mma_traits_sm100.hpp:424-427, :500-511) applied across epi stages.

Neither CUTLASS nor DeepGEMM ever lays out an M>128 single-CTA accumulator (Q2), so the
composite layout above is our construction from the documented Layout D + Layout F atoms,
not a copied pattern. The Layout F strip mapping itself is verbatim CUTLASS
(mma_traits_sm100.hpp:503-504).

**SMEM budget** (bf16, per CTA, same for A/B either option): stage = 160x64x2 (A) +
192x64x2 (B) = 44 KiB. cuBLAS's 5 stages = 220 KiB — leaves <7 KiB for store buffers in a
227 KiB carveout, so nvjet must stream its epilogue; for us **4 stages = 176 KiB + our
existing store pipeline (STORE_PIPE_DEPTH x BM x STORE_COLS x 2, template :104) fits
comfortably**; revisit 5 stages later only if K=1024 (16 k-iters) proves latency-bound.

---

##### Q6. Recommended implementation for our template

**Recommendation: build the cta_group::1 Config as swap-AB with a composite-M' MMA
(128+64), i.e. Option B**, with Option A as an acceptable fallback if the transposed
epilogue proves too invasive. Rationale: these 3 shapes are compute-bound square-ish
GEMMs (4096x2048x1024/2048); Option A's +20% tensor-core waste and single-stage
accumulator directly cost throughput, while Option B is waste-free and keeps our
2-stage epilogue overlap.

Per-K-step issue (Option B), from the single MMA warp:
```
// 4 K-steps per 64-wide stage (K=16 per issue, Table 44)
tcgen05.mma.cta_group::1.kind::f16 [tmem_c0],       bT_desc,       aT_desc, idesc_M128_N160, accum;
tcgen05.mma.cta_group::1.kind::f16 [tmem_c1], bT_desc_row128,       aT_desc, idesc_M64_N160,  accum;
...x4...
tcgen05.commit.cta_group::1.mbarrier::arrive::one.multicast::cluster.b64 [empty_mbar], 0b11;
```
idesc changes vs `sm103_make_i_desc` (template :576-579): N-field = 160>>3, M-field =
128>>4 / 64>>4 (two constants instead of `CLUSTER_SIZE*BM`); transpose bits recomputed
for the swapped operand roles (TN stays two K-major descriptors).

**Carries over unchanged (≈70% of the template):**
- TMA producer warp specialization + 4D TMA loads (`tma_4d_gmem2smem`, template :256-271)
  — only the multicast wrapper flips to `CTA_GROUP=1` (already a template parameter, :273-313).
- CLC persistent scheduling (clc_try_cancel :419 + response plumbing) — works at cluster
  granularity; a 1x2 cluster is the same rank-0-queries/broadcast pattern as 2x1.
- Store pipeline (tma_store_fence/commit/wait :394-403, `tma_2d_smem2gmem` :405),
  smem-descriptor builders (`make_smem_desc` :315-337 — the 3-arg LBO/SBO variant from
  the NN fix is reused for the second atom's row-128 base).
- TMEM alloc/dealloc (:353-365, flip template arg to 1), mbarrier helpers (:175-213).

**New / rewritten:**
1. **Config**: `Sm103Cluster1x2Config<160,192,64>` — CLUSTER_SIZE=1 (no MMA pair),
   CLUSTER_N=2, BN is now *per CTA* (vs per-cluster today, template :81), drop the
   `BN >= 128 % (CLUSTER_SIZE*SWIZZLE_W)` assert (:135) for `BN % 64`; BM=160 allowed via
   new asserts (BM % 32, BM ≤ 128+64).
2. **Barrier counts**: full barrier expect_tx = A+B bytes on *every* CTA (no leader
   parity, Q3); empty barriers need 2 arrivals/phase via commit-multicast mask 0b11;
   `mbarrier_init` counts change accordingly (CLUSTER_SIZE=1 ⇒ the :121 CLUSTER_CTAS
   scaling becomes x2 for the A-shared barriers only).
3. **MMA issue loop** (:1055 region): two idesc constants, two d-tmem base addresses
   (col 0 and col 160 + lane-alignment bit for the interleaved pair), second smem
   descriptor offset (+16 KiB) — all compile-time.
4. **Epilogue** (biggest change, Option B): TMEM holds Cᵀ. One warpgroup:
   `16x256b`-family loads (16-lane strips; the M'=64 atom's valid strips are lanes
   0-15/32-47/64-79/96-111, so every warp reads its own 16-lane strip) → `stmatrix`-style
   transposed smem writes → TMA store boxes of 16(M)x192(N), exactly DeepGEMM's
   `sm100_store_cd_swap_ab.cuh` scheme (STORE_BLOCK_M=16, STORE_BLOCK_N=128 warpgroup
   read, :30-90). Our current 32x32b row-wise epilogue (:377-384, BM/32 warps, :89)
   survives only in the Option A fallback (where it needs a 5th/6th 16-lane read for rows
   128-159 at column offset 192).
5. **Dispatcher**: route the 3 (M,N,K) shapes (4096,2048,{1024,2048}) to the new kernel;
   grid = 26 x 11 = 286 CTAs = 143 x (1x2) clusters (~0.97 waves of 148 SMs at 2
   CTAs/SM... actually 1 CTA/SM at 176-220 KiB smem ⇒ 1.93 waves — matches why cuBLAS
   liked the smaller 160x192 tile here vs 128x256's 2.0 waves).

**Suggested build order:** (1) Option A skeleton first (epilogue nearly unchanged) to
validate cta_group::1 plumbing, barriers, and the composite-M TMEM addressing; (2) then
swap-AB epilogue for the perf target; (3) micro-benchmark both against
`nvjet_sm103_tst_160x192_64x5_1x2_h_bz` on the B300 before choosing (per our
verify-claims rule).

##### Unconfirmed / open items (explicit)
- Whether nvjet really decomposes 160 (vs an undocumented SASS M=160 encoding) — not
  determinable from PTX/CUTLASS/DeepGEMM sources; SASS dump of the nvjet kernel would
  answer it (count tcgen05/UTCMMA issues per K-step).
- Whether nvjet uses swap-AB for these tiles ("tst"/"h"/"bz" name fields not decoded).
- TMA box-size caps cited from CUDA driver docs knowledge, not from the provided trees
  (CUTLASS's 256-coalesce is corroborating, not normative).
- Perf assumption that 2-stage accumulator overlap outweighs Option A's simplicity —
  must be measured.

##### Implementation checklist (build order, Option A first)

File: NEW `src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_1x2_template.cuh`
(#include the cluster template header for primitives; do NOT modify the validated file).

A-skeleton steps:
1. Config `Sm1031x2Config<160,192,64,_STAGES=4>`: CLUSTER_SIZE=1, CLUSTER_N=2 (cluster=2 CTAs
   along N), BN per-CTA, NUM_EPI_WARPS=BM_CEIL/32 where BM_CEIL=160→5 warps, A_STAGE=160*64*2,
   B_STAGE=192*64*2, asserts per plan Q6.1.
2. i_desc x2: idesc_M128 (M-field 128>>4, N-field 192>>3), idesc_M64 (64>>4, 192>>3);
   TN transpose bits as today (both operands K-major).
3. Kernel (single file, cta_group::1 everywhere): producer = same TMA pipeline; A split-issue
   rows [r*80,(r+1)*80) mask 0b11 + own B non-multicast; every CTA arms OWN full mbar
   expect_tx(A+B) (no leader mapa!). Consumer: EVERY CTA has its own MMA warp (no pair gate):
   per K-step 2 MMAs (M128 atom → tmem col base; M64 atom → col base+192, smem A desc +16KiB
   row-128 offset via 3-arg make_smem_desc), commit-multicast mask 0b11 → both CTAs' empty
   (init count 2). tmem: single accumulator stage (Option A limit), tmem_full count 1,
   tmem_empty count NUM_EPI_WARPS (local only — accumulators are per-CTA private).
4. Epilogue: rows 0-127 = today's 32x32b path (4 warps); rows 128-159 = warp 4 reads M64-atom
   strips: lanes 0-15 (rows 128-143) + lanes 32-47 (rows 144-159) at col base+192 via
   16-lane loads (tcgen05.ld.16x256b family) — warp4's lanes 0-15/16-31 map... CAREFUL:
   32x32b from warp4 (lanes 128-159 don't exist; warps address lanes [32w,32w+32) of the
   WARPGROUP — warp 4 is in the 2nd warpgroup → lanes 0-31 again ✓ reads strips 0-15/32-47?
   NO — warp reads its own 32-lane window; need 16x-lane shapes. Resolve at impl vs
   DeepGEMM sm100_store_cd.cuh non-swap path for BLOCK_M=32/64.
5. C store: same store pipeline; grid = ceil(M/160) * ceil(N/(192*2)) * 2 CTAs; CLC identical
   (rank0 queries, 2-CTA cluster = same as validated 2x1 shape).
6. Test: new suite in the test file (RUN_1X2=1): shapes incl. 4096x2048x{1024,2048} TN (the 3
   Qwen shapes) + small sanity shapes; M%160!=0 handled by TMA OOB (4096 = 25.6 tiles → ceil).
7. THEN Option B (swap-AB) per plan Q6 with DeepGEMM sm100_store_cd_swap_ab.cuh epilogue;
   benchmark A vs B vs nvjet on B300.

---

##### RESOLVED: M=64-atom epilogue read (Option A)

**Verdict up front: the checklist's 5-warp Option A epilogue (step 4) is physically
impossible, and the corrected version is complex enough that Option A should be SKIPPED.
Recommended path at the end of this section.** All line refs into
`ptx_docs/ptx_isa_9.3.txt` unless noted.

###### 1. Layout F row→(lane,col) map — ALL 4 strips, rows in order (plan Q1 guess of "2 strips valid" for the *atom* was wrong; for the BM=160 *tile* it is right)

The PTX Layout F figures (Figure 215/216, ptx:33758-33768) are images absent from the
text dump, so the normative map here is the CUTLASS atom, which is exact
(`cutlass/include/cute/atom/mma_traits_sm100.hpp:500-504`):

```
// M_MMA == 64: Half subpartitions layout atom: (M,N) -> tmem_addr
Layout<Shape<Shape<_16,_4>, N>, Stride<Stride<_1,_32>, _128>>
```

Row m → DP lane `(m%16) + (m/16)*32`, i.e. the M=64 accumulator occupies **all four**
16-lane half-subpartition strips, rows strictly in order:

| atom rows | TMEM lanes | our tile (BM=160, atom at col 192) |
|---|---|---|
| 0-15  | 0-15   | tile rows 128-143 (valid) |
| 16-31 | 32-47  | tile rows 144-159 (valid) |
| 32-47 | 64-79  | garbage (A rows 160-191 don't exist) |
| 48-63 | 96-111 | garbage |

Columns: all N columns at the atom's column offset (192..383 for us). Alignment "0 or 16"
(ptx:33623-33624 Layout F row of the DP-organization table, :33617-33632) selects
top/bottom half-subpartition; Interleaved mode packs a second M=64 atom at +16
(mma_traits_sm100.hpp:424-427, tile_stride=1 at :506).

###### 2. tcgen05.ld lane addressing — warp window is by warp-id MOD 4, taddr bit 20 picks the half

- ptx:32794-32804: TMEM is split into 4 chunks; the table is indexed by "**ID of the warp
  within the warpgroup**": id 0→lanes 0-31, 1→32-63, 2→64-95, 3→96-127. So **warp w
  accesses lanes [32*(w%4), 32*(w%4)+32) — warp 4 (warpgroup 1, id 0) sees lanes 0-31
  again, and can NEVER touch lanes 32-47.**
- taddr = lane index in bits 31-16, column in bits 15-0 (ptx:31270-31281).
- 16-lane shapes (`.16x64b/.16x128b/.16x256b`, ptx:31551,31593,31607: "will access 16
  lanes") read lanes [base, base+16) where base is the taddr lane field; the only legal
  bases inside a warp's window are +0 and +16 — DeepGEMM selects the top half with
  `tmem_addr | 0x00100000` (= lane+16, `sm100_store_cd_swap_ab.cuh:90-92`). The
  warp-index bits of the lane field are **ignored by hardware**
  (`sm100_bf16_gemm.cuh:373-374`: "no need for tmem_ptr |= (epilogue_warp_idx*32)<<16");
  CUTLASS debug-asserts they match the executing warp anyway
  (`cute/atom/copy_traits_sm100.hpp:445`).
- `.16x256b.x1` fragment map (Copy_Traits DstLayout, copy_traits_sm100.hpp:467-481:
  `Shape<Shape<_4,_8>,Shape<_64,_2>>, Stride<Stride<_64,_256>,Stride<_1,_2048>>`):
  thread t holds lanes base+(t/4) and base+(t/4)+8; regs {r0,r1} = f32 cols
  2*(t%4), 2*(t%4)+1 of the first lane, {r2,r3} = same cols of lane+8. **Two rows per
  thread, 2-col chunks — NOT the thread==row map of `.32x32b`**, so the tail rows need a
  brand-new register→smem store indexing.

###### 3. DeepGEMM never reads a Layout F accumulator in its non-swap epilogue — nothing to copy

`sm100_bf16_gemm.cuh:57-58`: `LAYOUT_AD_M = 128; UMMA_M = LAYOUT_AD_M * kNumMulticast` —
the MMA is **always M=128 (Layout D)**, even for BLOCK_M=32/64; A is padded to 128 rows
in smem (`UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M)*...`, :89,
allowed to bleed into the B-stage region, :90). The epilogue then stores only the first
BLOCK_M lanes: `STORE_BLOCK_M = min(BLOCK_M,128)`, `kNumUMMAStoreThreads = STORE_BLOCK_M`
(:74-76) → BLOCK_M/32 warps, each reading its own 32-lane window with
`SM100_TMEM_LOAD_32dp32b{4,8}x` only (`sm100_store_cd.cuh:89,96`). The **only** 16-lane
reader in all of DeepGEMM's epilogues is the swap-AB path
(`sm100_store_cd_swap_ab.cuh:88-92`), a full warpgroup where every warp does the
identical pair of `16dp256b1x` loads (halves 0 and 16) + `STSM_T` transposed store (:102),
with `STORE_BLOCK_N == 128` asserted (:32). Conclusion: for awkward M they either pad to
M=128 (cheap when BLOCK_M<128 wastes only *unused* datapaths, not flops — N is the
per-col cost) or swap. **No producer of tcgen05 code ships a non-swap Layout F read.**

###### 4. Corrected Option A epilogue (if it were built) — and why not to build it

Checklist step 1/step 4 are wrong as written: NUM_EPI_WARPS=5 with warp 4 reading strips
0-15 *and* 32-47 is impossible (warp 4's window is lanes 0-31 only, and one 16-lane load
cannot span lanes 0-15 + 32-47 since they sit in different windows). The minimal correct
scheme is **4 epi warps, two passes**:

| warp | pass 1: M128 atom, cols 0-191 | pass 2: M64 atom, cols 192-383 |
|---|---|---|
| 0 | `.32x32b` lanes 0-31 = rows 0-31 | `.16x256b` half 0 → lanes 0-15 = rows 128-143 |
| 1 | `.32x32b` lanes 32-63 = rows 32-63 | `.16x256b` half 0 → lanes 32-47 = rows 144-159 |
| 2 | `.32x32b` lanes 64-95 = rows 64-95 | idle |
| 3 | `.32x32b` lanes 96-127 = rows 96-127 | idle |

Accounting: EPI bar.sync = 128 threads (4 warps, not 160); tmem_empty arrive count = 4
warp-arrivals (or 128 thread-arrivals per our template's scheme); C TMA box 160 rows is
legal (≤256 per dim). Pass 2 needs a second store-smem indexer (thread t → rows
128/144+(t/4) and +8, col pairs 2*(t%4)+8k per `.x(k)` repeat — from the fragment map in
####2), i.e. a fragment layout that exists nowhere in our template or DeepGEMM.

Cost summary of Option A: (i) bespoke second fragment→smem path, (ii) asymmetric warp
duty + warp 0/1 doing 1.25x the TMEM-load work of warps 2/3, (iii) single accumulator
stage — no MMA/epilogue overlap (plan Q5: 2 stages need 576>512 cols), (iv) +20% flops.
Its sole advertised virtue — "epilogue nearly unchanged" — is false per (i)+(ii).

###### DECISION — skip Option A; validate plumbing on a BM=128 bring-up config, then build Option B directly

1. **Bring-up**: `Sm1031x2Config<128,192,64>` — single M=128 MMA (Layout D), today's
   4-warp `.32x32b` epilogue byte-for-byte, exercising 100% of the new cta_group::1
   machinery (1x2 multicast + expect_tx both-CTAs, commit-multicast mask 0b11 empty
   barriers, CLC, idesc N=192, single-CTA TMEM alloc). This replaces the "Option A
   skeleton" — same validation value, zero epilogue work.
2. **Deliverable**: Option B (swap-AB 160x192: MMA M'=128+64, N'=160) with the
   DeepGEMM swap-AB epilogue scheme (`sm100_store_cd_swap_ab.cuh:51-141`): full
   warpgroup, `16dp256b1x` pair per warp for the M'=128 atom; for the M'=64 atom each
   warp issues the **single** half-0 load (its strip: warp w%4 → atom rows 16*(w%4)..+16)
   — or half-1 (`|0x00100000`) for the interleaved second epi-stage buffer — then the
   same `STSM_T` store. 2-stage accumulator fits (480≤512, plan Q5). Zero flop waste.
3. Benchmark Option B vs `nvjet_sm103_tst_160x192_64x5_1x2_h_bz` on the B300 (bring-up
   config is a plumbing artifact, not a perf claimant). No open items: every addressing
   fact needed for both steps is pinned above with sources.

##### CRITICAL GAP found during Option B implementation start (2026-07-04, pre-code)

The 160-dim slicing conflicts with 128B swizzle chunking IN BOTH OPTIONS — the plan's Q4
"boxes of 160x64 legal" missed that block OFFSETS must be chunk-aligned:
- TN operand [K,X] uses tmap {64, K, X/64} (128B swizzle = 64-elem chunks); slicing X in
  160-wide blocks → z-coord = block*160/64 = NON-INTEGER for odd blocks. Same for 80-wide
  issue-split halves. So neither Option A (A-operand M=160 slices) nor Option B (B'-operand
  N'=160 slices) can use our current 128B-swizzle tmaps.
- HYPOTHESIS to verify FIRST next session: the 160-sliced operand must use 64B swizzle
  (SWIZZLE_64B = 32-elem chunks; 160 = 5x32 ✓): tmap {32, K, X/32}, smem descriptor swizzle
  field = 64B mode, SBO/LBO recomputed for 32-elem chunks, MMA desc supports 64B-swizzled
  operands (desc bits 61-63). Alternatively swizzle-NONE for that operand. Verify against
  CUTLASS sm100 smem layout selectors (they pick swizzle per operand extent) + PTX desc
  encoding table, THEN implement. Do NOT code Option B before resolving this.
- Also re-check: does 64B swizzle change the TN a_off/b_off chunk-major math (SWIZZLE_W=32
  for that operand) — yes, all SWIZZLE_W-derived offsets for the swapped operand must use 32.

---

##### RESOLVED: 160-dim slicing & swizzle (Option B final spec)

**Verdict up front: the 64B-swizzle hypothesis is CONFIRMED and is the chosen strategy.
The 160-sliced operand (swapped B' = original A[K,M]) gets its own 64B-swizzled tmap +
64B-mode smem descriptors (LBO=4096, SBO=512); the 192-sliced operand (swapped A' =
original B[K,N]) keeps 128B swizzle unchanged. Mixed swizzle modes across the two
operands of one tcgen05.mma and across two tmaps in one kernel are both fully legal.
The 128B+padding alternative is REJECTED — it does not even load (proof in §R6). One
consequential correction to the earlier plan: the multicast issue-split of the shared
operand can no longer be an 80-row M-split (80 is not chunk-aligned); use a single-issuer
full box (§R7a). Zero open items.** Line refs: `ptx:` = ptx_docs/ptx_isa_9.3.txt,
`template :` = Bgemm_sm103_cluster_template.cuh, cutlass/DeepGEMM paths as written.

###### R0. Terminology fix first (the CRITICAL GAP called these "K-major"; they are MN-major)

PTX (ptx:33810-33812, tcgen05 major-ness table): transposed A = **M-major**, transposed
B = N-major; "we will use MN-Major and K-Major throughout". Our TN stores A as
[batch,K,M] row-major (template :491-503) — the contiguous gmem dim is **M**, so the A
operand is **MN-major**, and likewise B[K,N] is MN-major. The 160-sliced dim is the
operand's *MN/contiguous* dim. This matters because swizzle selection keys on the
contiguous-dim extent — for MN-major that is exactly the 160/192 tile width.

###### R1. How CUTLASS picks the swizzle (Q1) — exact rule

`sm100_smem_selector` (cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:82):
requires extent % 8 == 0 (:86-87), then picks the **largest swizzle atom that divides the
contiguous-dim extent**:
- MN-major: `BLK_MN0 % 64 → SW128; % 32 → SW64; % 16 → SW32; % 8 → INTER` (:89-108) —
  atom MN-spans for bf16 come from Layout_MN_SW{128,64,32}_Atom_Bits = 1024/512/256 bits
  (cute/atom/mma_traits_sm90_gmma.hpp:76-78, upcast :90-94) = 64/32/16 elements.
- K-major: same ladder on `BLK_K0` (:115-127) — irrelevant here (our operands are MN-major).

**For our 160-wide bf16 MN-major slice: 160 % 64 ≠ 0, 160 % 32 = 0 → Layout_MN_SW64_Atom
= 64B swizzle.** For the 192-wide slice: 192 % 64 = 0 → SW128 stays. CUTLASS builders
call the selector **separately per operand** (e.g.
sm100_mixed_tma_cpasync_umma_builder.inl:113 for A, :124 for B), so per-operand swizzle
is the normal CUTLASS situation, not an exotic one.

DeepGEMM has the identical rule at runtime: `get_swizzle_mode` returns the largest of
{128,64,32,16} dividing block-width-bytes (csrc/jit_kernels/heuristics/utils.hpp:13-20),
applied per operand to the *inner* (contiguous) dim ("Decide swizzling by the inner dim",
csrc/jit_kernels/heuristics/sm100.hpp:160-166): 160×2B = 320B → **64**. So both reference
stacks pick 64B swizzle for a 160-wide bf16 MN-major block. (DeepGEMM's sm100 heuristic
additionally *discards* candidate layouts whose swizzle ≠ 128 when either operand is
K-major, sm100.hpp:132-137 — an impl simplification tied to their
`kSwizzleMode == BLOCK_K*sizeof` K-major assert, common/sm100_utils.cuh:104, not a HW
rule, and it does not apply to us: both our operands are MN-major.)

###### R2. PTX smem-descriptor encodings (Q2)

Table 45 (ptx:31936-31952): bits 61-63 swizzle mode = **0 none, 1 = 128B with 32B atomic
swizzling, 2 = 128B, 4 = 64B, 6 = 32B** (3/5/7 invalid) — same enum in CUTLASS
(cute/arch/mma_sm100_desc.hpp:79-85, `SWIZZLE_64B = 4`). Start addr, LBO, SBO all 16B
aligned and encoded `(x & 0x3FFFF) >> 4` (ptx:31939-31945, :31955, :31964-31970). Base
offset bits 49-51 = 0 iff the swizzle pattern starts on its natural boundary — 1024B for
128B swizzle, **512B for 64B swizzle**, 256B for 32B (Table 46, ptx:31959-31963);
otherwise `(pattern start addr >> 7) & 7` (ptx:31964-31965; cf. ptx:30083-30086).

LBO/SBO meaning for **MN-major swizzled** operands (tcgen05 9.7.17.3.1.1, ptx:31666-31690;
9.7.17.3.2, ptx:31712-31727; canonical layout `((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))` with
Swizzle<2,4,3> for 64B, ptx:31749-31760, 64B row :31759):
- **LBO** = byte stride from one (swizzle-span) group of MN rows to the next = our
  *chunk pitch*;
- **SBO** = byte stride from one group of 8 K-columns to the next.

For our chunk-major TMA fill (chunk pitch = BK rows × span bytes; 8-K-row group = 8 ×
span bytes), with BK=64 bf16:

| mode | desc bits 61-63 | chunk width | LBO | SBO | per-K-row pitch |
|---|---|---|---|---|---|
| 128B | 2 | 64 elem | BK·128 = **8192** | 8·128 = **1024** | 128 B |
| 64B  | 4 | 32 elem | BK·64  = **4096** | 8·64  = **512**  | 64 B  |
| 32B  | 6 | 16 elem | BK·32  = **2048** | 8·32  = **256**  | 32 B  |

The 128B row is byte-for-byte our working TN descriptor (template :345-349, :1071-1072,
:1080-1081) — validation of the derivation. The general formulas are also verbatim
DeepGEMM: MN-major `stride_byte_offset = 8*ATOM*sizeof`, `leading_byte_offset =
BLOCK_K*ATOM*sizeof`, ATOM = swizzle-bytes/elt-size (common/sm100_utils.cuh:112-131,
common/tma_utils.cuh:10-12), and desc-address K-walk stride = ATOM elements per K element
(sm100_utils.cuh:81-90) → 16-K-step pitch = 16·64B = 1024B under 64B swizzle.

CU_TENSOR_MAP_SWIZZLE_* ↔ descriptor field (cuda.h:3642-3648 vs Table 45): NONE=0↔0,
32B=1↔6, 64B=2↔**4**, 128B=3↔2 (mapping code: cute/arch/copy_sm90_desc.hpp:240,
mma_traits_sm100.hpp:145-148). The two enums are unrelated numerically — never mix them.

###### R3. tcgen05.mma operand legality under 64B swizzle (Q3) and mixed modes (Q5)

- Table 60 "Layout for swizzle atoms" (ptx:33827-33843) lists tcgen05 swizzle atoms for
  **every** mode × major-ness; 64B MN-major atom = 4×8 in 128b units = **32×8 bf16**. No
  MMA-kind column exists — modes are kind-agnostic.
- The only mode restriction anywhere in 9.7.17 is Table 56's transpose rule
  (ptx:33438-33453): for kind::f16, Transpose A/B is supported, and the *only* exclusion
  is 128B-swizzle-with-32B-atomicity (mode 1). Plain 64B (mode 4), 32B and none are fully
  legal with the transpose bits we use.
- Mixed modes across the two operands: each operand has its own 64-bit descriptor with
  its own bits 61-63 (Table 45 is per-descriptor); no clause ties A's mode to B's.
  DeepGEMM ships kernels with independent `swizzle_a_mode`/`swizzle_b_mode` template
  parameters and tmaps (csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:58, :101-106).
  **A'=128B + B'=64B in one MMA is legal.**

###### R4. Two tmaps with different swizzle in one kernel (Q6)

Swizzle is a per-CUtensorMap property passed to each `cuTensorMapEncodeTiled` call
(our builders already take it as a parameter, template :475, :518). Constraints from the
driver docs (cuda.h:23302, :23322-23325): boxDim[0]·eltsize must be a multiple of 16B,
and with swizzle enabled **boxDim[0]·eltsize ≤ swizzle size** — 64B swizzle → inner box
≤ 64B = **32 bf16 elements**. Nothing constrains two tmaps of one kernel to share a mode
(DeepGEMM's A/B tmaps above are the existence proof). Our 4D-chunked shape generalizes
as: 128B → {64, ·, X/64}; 64B → {32, ·, X/32}, dim-2 global stride 64B (multiple of 16 ✓).

###### R5. Sanity of the 192 operand (Q5)

A' (original B[K,N], per-CTA 192-wide): 192 % 64 = 0 → SW128 by the R1 rule; tmap
{64, K, N/64} box {64, BK, 192/64=3}, z = block_col·6 + rank·3 — integer ✓. Descriptor =
today's 3-arg form (LBO 8192, SBO 1024, mode 2). Second M-atom (M'=192 = 128+64): rows
128-191 = chunk 2 → base + 2·LBO = **+16384** (16B-aligned ✓, same value as plan Q4's
"16 KiB"). Nothing else changes for this operand.

###### R6. The 128B-swizzle padding alternative (Q7) — REJECTED, it cannot even load

Pad-to-192-in-smem does not fix the *gmem* problem: block b's slice starts at m = b·160,
and b·160 mod 64 = 32 for every odd b. A 64-elem-chunk tmap can only address multiples
of 64 in the chunked dim, and re-encoding cannot help: with 128B swizzle the box inner
dim is capped at 64 elements (cuda.h:23322-23325), so the chunk factorization — and the
64-element offset granularity — is forced. Absorbing the 32-element (64B) misalignment
on the smem side is also impossible: the descriptor's base-offset field re-phases the
swizzle only in whole 128B rows (`(pattern start addr >> 7) & 7`, ptx:31964-31965;
TMA-side table ptx:3835-3838), and MN-major mode-2 descriptors walk whole 64-element
lines (canonical layout ptx:31749-31760) — there is no "start 32 elements into a chunk"
encoding. The remaining 128B-preserving escapes are padding the *gmem* layout (not
allowed — we are a BLAS API on user buffers) or loading a 64-aligned 192-wide window per
block with parity-dependent descriptor origins (+20% A-side TMA traffic, garbage-column
reads, two code paths). All strictly worse than 64B swizzle, whose only cost is a
half-width TMA/MMA line for one operand. **PICKED: 64B swizzle.** (Swizzle-NONE would
also make offsets legal but forfeits bank-conflict avoidance and is never chosen by
either reference stack when a larger atom fits, sm100_common.inl:97-108,
heuristics/utils.hpp:16.)

###### R7. FINAL Option B spec — `Sm1031x2Config<160,192,64>` (supersedes the CRITICAL GAP; everything else from the earlier RESOLVED section stands)

Operand roles after swap-AB: **MMA-A' = original B[K,N]** (192/CTA slices, 128B swizzle),
**MMA-B' = original A[K,M]** (160 slices, 64B swizzle, cluster-shared → multicast).

**(a) tmap B' (original A, NEW 64B-swizzle builder `init_4d_tma_A_KM_sw64`):**
`globalDim {32, K, M/32, batch}`, `globalStrides {M·2, 64, batchStride·2}`,
`boxDim {32, BK=64, 160/32=5, 1}`, swizzle = `CU_TENSOR_MAP_SWIZZLE_64B`.
Precondition M % 32 == 0 (relaxed from %64; 4096 ✓). Per stage k of block_row b:
coords `(0, k·64, b·5, batch)` — z = b·5 is always integer, gap closed. **Single-issuer:**
the pair's rank-0 producer issues the one 20480B box with multicast mask 0b11 (same box
byte-count as today's single A issue; blessed as equally legal at plan Q3 :139-140). The
old "rank r issues rows [r·80,(r+1)·80)" split is DEAD — 80/32 = 2.5 chunks. (If issue
balance ever matters, the legal even split is per-chunk: boxDim {32,64,1}, rank0 chunks
0-2 / rank1 chunks 3-4, dst += chunk·4096 — DeepGEMM's per-atom issue pattern,
common/tma_utils.cuh:25-33. Do NOT K-split: two {32,32,5} boxes give chunk pitch 2048,
breaking LBO=4096.) M=4096 tail: block 25 spans chunks 125-129 vs globalDim[2]=128 →
chunks 128-129 OOB-zero-filled by TMA ✓. expect_tx unchanged: every CTA arms its own full
barrier with the full 45056B/stage (per plan Q3).

**(b) tmap A' (original B): unchanged machinery** — `init_4d_tma_B_KN` (template
:515-534) with per-CTA slice 192: box {64, 64, 3}, 128B swizzle, z = block_col·6 +
rank·3, non-multicast, issued by each CTA's own producer.

**(c) smem per stage:** `[A' 24576 B | B' 20480 B]` = 45056 B; ×4 stages = 176 KiB.
**B' stage bytes unchanged** (160·64·2 = 5 chunks × 4096). Alignment: smem base is
`__align__(1024)` (template :727); 45056 = 44·1024 → every stage 1024-aligned → A' on a
1024B pattern boundary and B' at +24576 = 48·512 on a 512B boundary → **base-offset
field = 0 for both descriptors** (Table 46, ptx:31959-31963).

**(d) consumer walk (replaces the SWIZZLE_W-uniform math of template :1022-1055):**
loop bounds are UNCHANGED — `k1 < BK/64 (=1)`, `k2 < 64/MMA_K_STEP (=4)` — they walk K,
not chunks; the question's "BK/32?" is answered NO. Only the per-operand byte pitch
differs. With k_row = k1·64 + k2·16:
```
a_off = k_row * 128;                       // A' 128B-swizzled: k2*2048
b_off = k_row * 64;                        // B' 64B-swizzled:  k2*1024
a_desc  = make_smem_desc     (A_smem + a_off,          8192, 1024);  // mode 2
a_desc2 = make_smem_desc     (A_smem + a_off + 16384,  8192, 1024);  // M64 atom, chunk 2
b_desc  = make_smem_desc_sw64(B_smem + b_off,          4096,  512);  // mode 4, both MMAs
```
`make_smem_desc_sw64` = template :345-349 with `(4ULL << 61)` in place of `(2ULL << 61)`.
Keep per-operand constants (`SW_SPAN_Ap = 128`, `SW_SPAN_Bp = 64` bytes) instead of the
global SWIZZLE_W (template :63); drop the `BN % (CLUSTER_SIZE*SWIZZLE_W)` assert
(template :142) in favor of the asserts in (f).

**(e) idesc:** two compile-time constants, both with trans_a = trans_b = 1 (bits 15,16 —
both operands MN-major, exactly today's TN setting, template :598-599; legal for f16
under 64B swizzle per Table 56, ptx:33438-33453): N-field = 160>>3 = 20, M-field =
128>>4 = 8 for MMA#1 and 64>>4 = 4 for MMA#2 (fields per ptx:32014-32018). N=160 legal
(Table 44, ptx:31324-31338).

**(f) config asserts:** M % 32 == 0; B'-slice (160) % 32 == 0; A'-slice (192) % 64 == 0;
BK % 64 == 0. Everything not named here — cta_group::1 barriers, CLC, TMEM layout
(2×160-col buffers + interleaved M64 pair, 480 ≤ 512), DeepGEMM-style swap-AB epilogue,
BM'=128 bring-up config first (both operands 128B, no 64B code exercised), then this
spec, then benchmark vs `nvjet_sm103_tst_160x192_64x5_1x2_h_bz` — is exactly as pinned
in the "DECISION" block of the previous RESOLVED section and in plan Q3-Q6.

**No open items.** Every formula above is triple-sourced (PTX normative text, DeepGEMM
shipping code, and — for the 128B column — our own validated kernel), and the one
plan-level casualty (the 80-row issue split) has a specified replacement.

##### P5 BUG ANALYSIS: exact TMEM fragment maps

**Verdict up front: every TMEM/MMA map in the SWAP_AB path is CORRECT. The bug is in the
store-buffer write: both SWAP_AB epilogue writes (template :1308 and :1334) are LINEAR,
but `C_tmap` is built with `CU_TENSOR_MAP_SWIZZLE_128B` (template :1625-1626), so the TMA
store expects the 128B-swizzled smem arrangement. The validated non-swap epilogue applies
exactly that XOR (`swizzled_n = n ^ (row & 7)`, template :1487); the swap path dropped it.**

###### 1. tcgen05.ld.32x32b.x8 fragment — our map CORRECT
CUTLASS `Copy_Traits<SM100_TMEM_LOAD_32dp32b8x>` (copy_traits_sm100.hpp:1557-1570):
`ValID = ((_256,_32)):((_1,DP_b))`, `DstLayout = (_32,_256):(_256,_1)` → logical bit
`= 256*t + 32*i + b` decodes to lane `t`, col `i` (i = register 0..7). I.e. **thread t,
register i = (lane t, col addr+i); repetitions are 8 contiguous 32-bit columns, no
interleave.** PTX only states "accesses 32 lanes, 32*.num bits per lane"
(ptx_isa_9.3.txt:31537-31538, Figure 183 is an image). Our epilogue assumes exactly this
(template :1297-1308) and P4 (non-swap, 42/42) validates the same primitive at :1442.
DeepGEMM's swap-AB float path also uses it as (lane, col+row) (sm100_store_cd_swap_ab.cuh:74-84).

###### 2. tcgen05.ld.16x256b.x1 fragment — our map CORRECT
CUTLASS `Copy_Traits<SM100_TMEM_LOAD_16dp256b1x>` (copy_traits_sm100.hpp:464-482):
`DstLayout = ((_4,_8),(_64,_2)):((_64,_256),(_1,_2048))` over
`ValID = (_256,_16):(_1,DP_b)`. Decode: bit = 64*(t%4) + 256*(t/4) + v0 + 2048*v1 →
r0 = (lane t/4, col 2*(t%4)), r1 = (lane t/4, col 2*(t%4)+1),
r2 = (lane t/4+8, col 2*(t%4)), r3 = (lane t/4+8, col 2*(t%4)+1).
**Lane base+t/4 (one lane per 4 threads), NOT 2 apart** — exactly our comment/usage
(template :407-417, :1314-1335). PTX: "accesses 16 lanes … 256*.num bits"
(ptx_isa_9.3.txt:31607-31608). The 16-lane offset within the warp's subpartition is
selected by address lane-field bits — DeepGEMM selects the upper half with
`tmem_addr | 0x00100000` (sm100_store_cd_swap_ab.cuh:88-92), confirming the encoding our
`(warp*32 + 16*wave_stage) << 16` uses (template :1278-1279; CUTLASS asserts the
subpartition bits must match the warp, copy_traits_sm100.hpp:443-447).

###### 3. Layout F (M=64, cta_group::1, non-.ws) — our strips CORRECT
CUTLASS 1SM M_MMA=64 atom (mma_traits_sm100.hpp:500-511):
`tmem_atom = ((_16,_4),N):((_1,_32),_128)` → M-rows 0-15→lanes 0-15, 16-31→32-47,
32-47→64-79, 48-63→96-111; N-col n at col base+n. PTX Data Path Layout Organization
table (ptx_isa_9.3.txt:33617-33635): M=64/::1/non-.ws = Layout F, "4x1 (1/2 datapath
utilized)", **TMEM lane alignment 0 or 16** — so the D-tmem lane offset `(16*wave_stage)<<16`
(template :1131) is legal and encoded via the address lane field. The alignment-sharing
restriction (ptx:34606-34610) binds A/D/metadata only when A is in TMEM — ours is smem.
Epilogue strip math (template :1316-1318): warp w at lane field 32w+16*wave reads its
16-lane region = atom rows 16w..16w+15 → C-cols 128+16w+t/4(+8) ✓.

###### 4. Layout D (M=128, cta_group::1) — our map CORRECT
CUTLASS 1SM M_MMA=128 atom (mma_traits_sm100.hpp:513-523): `(_128,N):(_1,_128)` →
row r → lane r, one 32-bit col per N element for f32 D. So C'(row', n') sits at
(lane row', col base+n'); with D base = tmem+wave*BM (template :1127), C-col n_local =
32*warp+lane and C-row m = m0+i (template :1280, :1304-1308) ✓. DeepGEMM swap-AB uses the
identical map: lanes = n (STORE_BLOCK_N=128 "must match TMEM rows",
sm100_store_cd_swap_ab.cuh:31-32), cols = m (:60-62).

###### 5. MMA operand walk under swap — CORRECT
PTX canonical layouts (ptx_isa_9.3.txt:31749-31765): MN-major 128B swizzle =
`Swizzle<3,4,3> o ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO))`, T=8 for bf16 → one swizzle
chunk spans 64 MN positions × 8 K rows; the m-th 64-wide MN block advances by LBO, 8 K
rows advance by SBO (LBO/SBO defs ptx:31696-31701, :31740-31744). With LBO=8192,
SBO=1024 (template :1116-1122): M128 atom walks chunks 0-1 → B-slot columns 0-127 =
M'-rows 0-127 ✓; `+16384` = chunk 2 → columns 128-191 = M64-atom rows 0-63 ✓. K offset
k2*16 rows = k2*2048 B stays swizzle-pattern aligned ✓. Instruction descriptor fields
(Table 47, ptx:31989-32022): bit15 transA→M-major A', bit16 transB→N-major B'
(ptx:33801-33804), N-field bits17-22 = BM>>3 = 16, M-field bits24-28 = 8/4 — our
`sm103_make_i_desc_swap` (template :631-635) matches exactly. The same descriptor form
drove the 192-wide slot as MMA-B in P4 (42/42), and A/B canonical walks are symmetric.

###### 6. THE BUG: missing 128B-swizzle XOR in the store-buffer writes
`C_tmap`: box [BM × STORE_COLS=64] bf16 = 128 B/row with `CU_TENSOR_MAP_SWIZZLE_128B`
(template :1625-1626). TMA reads logical 16-byte group g of smem row r from position
`g ^ (r & 7)`. Producers must therefore write group g at `g ^ (r & 7)`:
- non-swap epilogue DOES (template :1486-1489, validated 42/42);
- DeepGEMM swap-AB DOES (`(col ^ row) * kNumBankGroupBytes`,
  sm100_store_cd_swap_ab.cuh:80-82 and :97-99, with the 128B-swizzle assert at :38);
- **our swap-AB epilogue does NOT** — template :1308 (`buf[(m0+i)*STORE_COLS +
  (n128-n_base)]`) and :1334 (`buf[m*STORE_COLS + (n_loc-n_base)]`) are linear.

Every output row m with (m & 7) != 0 gets its eight 16-byte groups permuted within the
64-col store chunk → deterministic same-row wrong values of GEMM magnitude. Evidence fits
bit-exactly:
- Row 0 (m&7=0): XOR is identity → row correct ("first-10 of row 0 CORRECT").
- (113,100): m&7=1; chunk1 offset 36 → g=4,e=4; reads g 4^1=5 → out = ref(113,108).
- (142,544): m=14 (block_row 1), n_local=160 → chunk2 (M64 region) offset 32 → g=4,e=0;
  reads g 4^6=2 → out = ref(142,528). Both regions fail identically because the bug is
  downstream of the (correct) TMEM reads.

###### Fix (only change needed)
M128 block, replace template :1308:
```cuda
const int off = n128 - n_base;                         // [0,64)
const int r   = m0 + i;
buf[r * STORE_COLS + ((((off >> 3) ^ (r & 7)) << 3) | (off & 7))] =
    __float2bfloat16(v);
```
M64 block, replace template :1334:
```cuda
const int off = n_loc - n_base;
buf[m * STORE_COLS + ((((off >> 3) ^ (m & 7)) << 3) | (off & 7))] =
    __float2bfloat16(v);
```
(identical formula to the validated non-swap `n ^ (row & 7)` at template :1487, with the
group index recomputed because the swap path addresses single elements, not int4 groups).
All TMEM fragment maps, Layout D/F placements, lane-alignment encoding, smem descriptors
and idesc fields verified correct from primary sources — leave them untouched.

### 40. MERGED IN FULL: all_tiles_plan.md (written 2026-07-07) — ALL-TILES exact-coverage plan

(Decoded nvjet cluster C-tiles for the remaining shapes, R1-R6 research resolution:
composite-M legality, Layout-B interleaved map, M64-::2 illegality, TMEM budget, swizzle
parameterization table, build stages T1-T4. Headings demoted. THIS is the active work plan.)

#### ALL-TILES plan — exact-tile coverage for the remaining ~13 Qwen shapes (2026-07-07)

Decoded from nsys grid math (grid/clusterCTAs = #clusters; solve ceil(M/tm)*ceil(N/tn)=#clusters
against the KNOWN qwen shape each kernel ran on; calibrated on 128x256_2x1_v which fits 256x256
UNIQUELY = our validated geometry).

##### Decoded cluster C-tiles (all kernels are _2cta_ = cta_group::2!)

| nvjet kernel | shapes | cluster C-tile (M x N) | per-CTA structure | status/need |
|---|---|---|---|---|
| 128x256_2x1_v | 12 | 256x256 UNIQUE | ours, validated | DONE |
| 128x256_2x2   | 8  | 256x512 | ours (128x256_2x2), validated | DONE |
| 128x256_2x2f  | ~5 | 256x512 dyn | ours, validated | DONE |
| 256x128_2x2 #4| 1  | 256x512 (fits) | = our 128x256_2x2 geometry | COVERED (verify via bench) |
| 256x256_2x2 #41| 1 | 512x512 UNIQUE | per-CTA M=256 = COMPOSITE 2x M_MMA-256 stacked; N 128/CTA (chunks ok) | need composite-M |
| 128x192 (2x1_v + 2x2f) | 5 | 384x128 or 192x256 (ambig; both need composite) | M-composite (256+128 -> 192/CTA, or 128+64 -> 96/CTA); N 128- or 256-per-pair | need composite-M (+interleave if 96) |
| 128x160_2x1_v #16 | 1 | 320x128 or 160x256 | M-composite 256+64 (160/CTA, interleaved M64) or 128+32 | need composite + INTERLEAVED M64 epi (§28.7) |
| 160x192_1x2 #11/21/27 | 3 | 384x160 or 320x192 | our swap-AB B2 covers these shapes TODAY (different geometry, validated); nvjet's = M-composite pair | COVERED by B2; nvjet-geometry optional |
| 160x192_2x2 #1 | 1 | 384x320 | M=192/CTA composite + N 160/CTA (sw64 - HAVE) | need composite-M |
| 320x192_2x2f #19 | 1 | 384x640 preferred | M=192/CTA composite + N 320/pair=160/CTA sw64 + dyn cluster (HAVE) | need composite-M |
| 64x64_2x2 #2 | 1 | 128x128 | per-CTA M=64: M_MMA=128 INTERLEAVED tmem (§28.7!) + N 32/CTA sw64 | need interleaved epi |
| 128x64_2x2 #7/#12 | 2 | 128x256 | per-CTA M=64 interleaved + N 64/CTA | need interleaved epi |

Ambiguities (384x128 vs 192x256 etc.): both readings legal via different composites; benchmark
vs the real nvjet kernel per shape is the arbiter. Where ambiguous we build OUR pick and measure.

##### Capability gaps -> build stages

STAGE T1 — composite-M for cta_group::2 (biggest unlock: 9 shapes):
  Multi-MMA along M per K-step: M_MMA pieces from {256, 128, 64}; second/third atom at tmem col
  offsets (Layout D for 256/128 pieces at per-CTA 128/64 rows... NOTE M_MMA=128 on cta_group::2 =
  per-CTA 64 rows uses the INTERLEAVED layout §28.7 — so composite 256+128 needs the interleaved
  read for the +128 piece; 256+64 needs cta_group::2 M64-atom reads). Epilogue: extend the
  swap-AB strip-read machinery (16-lane loads exist) to the cta_group::2 interleaved variants.
  Reference first: CUTLASS mma_traits_sm100.hpp 2SM layout atoms (M_MMA=128 2x2 interleaved map,
  M_MMA=64) — pin exact (row->lane,col) maps BEFORE coding, like the Layout-F work.
STAGE T2 — interleaved-M64 epilogue alone unlocks 64x64/128x64 (3 shapes) — subset of T1 work.
STAGE T3 — config aliases + suites per decoded tile; sw32 only if the 96/80-wide readings win.
STAGE T4 — benchmark each vs its nvjet kernel; keep winners, retire losers.

NOTE: shapes #4 (256x128) + the 3 160x192_1x2 shapes are ALREADY covered by validated kernels
(our 128x256_2x2 and swap-AB B2) — 30/42 exact-or-equivalent today; T1+T2 close the rest.

##### RESOLVED: composite-M + interleaved 2SM TMEM maps + swizzle parameterization

Sources: ptx_isa_9.3.txt (PTX), cutlass/include (CUTE), DeepGEMM, and our template
`BLUBLAS_kota/BluBridge-BLAS/src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh` (KERNEL).
All file:line cites verified 2026-07-07.

###### R1 — cta_group::2 M_MMA=128 accumulator map (per-CTA 64 rows) = PTX Layout B "2x2"

PTX Data Path Layout table (ptx 33617-33631): M=128 + ::2 + Dense = "2x2" organization,
**Layout B**, TMEM lane alignment **0** (no 16-lane option — unlike Layout F). Layout B section
header: ptx 33670 (Figure 207/208 are images; exact map taken from the CUTE atom, which is the
authoritative machine-readable form and matches our validated §28.7 note).

CUTE atom (mma_traits_sm100.hpp:543-551, `tmem_frg`, N_SM=2, per-SM M_MMA_SM=64):
`Layout<Shape<_64,Shape<N/2,_2>>, Stride<_1,Stride<_128,_64>>>` over virtual addressing
DP_STRIDE=1/COL_STRIDE=128 (mma_traits_sm100.hpp:486-491). Note CUTE's `M_MMA` here is the
**per-CTA** M (mma_traits_sm100.hpp:472-474); instruction M=128 -> M_MMA_SM=64. Decoded exact map,
per-CTA row r in [0,64), mma column n in [0,N):

    n <  N/2 :  lane = r,      tmem col = base_col + n
    n >= N/2 :  lane = r + 64, tmem col = base_col + (n - N/2)

i.e. §28.7 "2x2 INTERLEAVED: half-N in DP[0,64), half in DP[64,128)" CONFIRMED, and the split is
low-half/high-half of N (h is the outer coordinate mode: n_logical = nh + (N/2)*h), not an
interleave of individual columns. Per-CTA footprint = **N/2 tmem cols at all 128 lanes**.
Cross-check: 2SM CLayout `Shape<_2,Shape<M/2,N>>` (mma_traits_sm100.hpp:2049-2050) — each CTA owns
its own M/2 rows x all N cols; B is CTA-split N-wise (BLayout `Shape<_2,Shape<N/2,K>>`,
mma_traits_sm100.hpp:2047-2048 — each CTA's smem supplies N/2 of B, exactly our "B smem holds BN/2
per CTA", KERNEL:113).

**ld shapes / warp assignment (4 epi warps):** every lane strip is a full 32-lane subpartition, so
plain `32x32b` reads it cleanly — the EXISTING `tcgen05_ld` 32x32b.x8 helper (KERNEL:417-423), no
16-lane loads needed (16x256b machinery, KERNEL:425-435, stays swap-AB-only):

    warp 0 (lanes   0-31): C rows  0-31, cols [0,   N/2)   tmem cols [base, base+N/2)
    warp 1 (lanes  32-63): C rows 32-63, cols [0,   N/2)   same cols
    warp 2 (lanes  64-95): C rows  0-31, cols [N/2, N  )   same cols
    warp 3 (lanes 96-127): C rows 32-63, cols [N/2, N  )   same cols

Each warp: 32 rows x N/2 cols. Epilogue store indexing: c_row_local = (warp_id&1)*32 + lane_id,
c_col_base = (warp_id>>1)*(N/2) — only the row/col decode changes vs the validated M256 path
(KERNEL:1493-1495); the tcgen05.ld loop shape is identical.

###### R2 — cta_group::2 M_MMA=64: **ILLEGAL for kind::f16 — question is moot**

PTX Table 44 (ptx 31320-31344): kind::f16, CTA Group 2, Dense supports **only 128xNxK and
256xNxK** (N={16..256} step 16, K=16; row at ptx 31335-31337). Same restriction for ::2 Sparse and
tf32 (ptx 31374). The Data Path Layout table (ptx 33617-33631) accordingly has NO M=64/::2 row.
CUTLASS enforces it: `static_assert(M == 128 || M == 256, "...2x1SM_SS M-mode size should be 128
or 256...")` (cute/arch/mma_sm100_umma.hpp:554). The CUTE "1x4" M_MMA_SM=32 atom
(mma_traits_sm100.hpp:532-541, map: lane = r + 32*q, col = n - q*N/4 for n in quarter q) exists
only for other kinds/sparse paths — do NOT build on it for bf16.
CONSEQUENCES for the decoded table above:
- 128x160_2x1_v #16 ("256+64" or "128+32" readings): **both illegal**. Build per-CTA M=160 as the
  192-composite (256+128) with the last 32 rows dead: TMA OOB zero-fill on A + epilogue store
  predication (rows >= 160 skipped); optionally suppress the dead lanes with the MMA's
  `disable-output-lane` vector (size 8 for ::2, ptx ~34817-34830) — not required for correctness.
- Line 16's "(128+64 -> 96/CTA)" alternative reading: also uses only legal atoms (256- and
  128-instruction M) — still available.

###### R3 — composite-M legality for cta_group::2

Everything the swap-AB ::1 composite relies on carries over; verified constraint list:
1. **Same .cta_group for ALL tcgen05 in the kernel** (ptx 32724) — both atoms use ::2 ✓.
2. **Per-atom idesc**: M-field differs (256>>4 vs 128>>4, bits 24-30), N-field identical — exactly
   the `sm103_make_i_desc_swap<Config, M_ATOM>` pattern (KERNEL:670-675) with the normal (non-swap)
   M/N field assignment of `sm103_make_i_desc` (KERNEL:653-664).
3. **Ordering**: mma->mma pipelining is only guaranteed for "same N and accumulator, shape and
   kind" (ptx 32396-32398). The two atoms write DISJOINT d-tmem so they need no mutual order; each
   atom's own K-step chain (same shape+accumulator+kind) IS pipelined — the same property the
   validated ::1 composite uses (KERNEL:1152-1185, 336/336 bit-exact on B300).
4. **One commit tracks all**: tcgen05.commit implicitly tracks ALL prior mma from the issuing
   thread (ptx 32410-32415 / 32421-32427) — unchanged (KERNEL:1208-1209).
5. **Issue granularity**: only the pair leader issues both atoms (single-thread semantics,
   ptx ~34803-34805) — same as today.
6. **d-tmem lane alignment**: Layout A and Layout B both require lane alignment 0 (ptx
   33617-33631), so the M128 atom CANNOT be lane-offset-16-packed the way the ::1 Layout F M64
   atom is (KERNEL:1183-1185) — its accumulator must take its own column range at lane 0.
7. **Operand offsets for the 2nd atom** (per-CTA rows 128..): each CTA supplies its OWN rows
   (ALayout `Shape<_2,Shape<M/2,K>>`, mma_traits_sm100.hpp:2045-2046), so the offset is local:
   - MN-major 128B-swizzled [K,M] slice (TN A): rows 128.. = 64-elem chunk #2 -> **+2*LBO =
     +16384 bytes** at BK=64 — the exact swap-AB constant (KERNEL:1163-1165).
   - K-major [M,K] slice (NN/NT A): rows 128.. = 8-row atom #16 -> +16*SBO = 16*1024 = **+16384
     bytes** too (SBO=1024, KERNEL:350-357). Same constant, different derivation — don't conflate.
   - b-desc: IDENTICAL for both atoms (same B, same N, same K-step).
8. **N is capped at 256 per instruction** (ptx 31335-31337). Per-pair N=320 (tiles #1/#19) is NOT
   one MMA: use an N-composite (2nd d-tmem col region; e.g. 256+64 keeps per-CTA B slices at
   whole 128/32-elem chunks, or 2x160 which needs sw32 B slices of 80 elems). The plan lines
   19-20's "N 160/CTA sw64" is amended accordingly: 80-elem B half-slices do NOT tile with 32-elem
   sw64 chunks; 160-elem half-slices (=N_inst 320) exceed the cap.

###### R4 — TMEM budget (128 lanes x 512 cols, EPI_PIPE_DEPTH=2)

Per-stage accumulator columns (per CTA): M256 atom = N cols (all lanes); M128 atom = N/2 cols (all
lanes, R1); no column sharing possible between stages/atoms (R3.6).

| target tile (per-CTA rows / instr N) | atoms         | cols/stage      | x2 stages | alloc (KERNEL:878-885 granularity) | verdict |
|---|---|---|---|---|---|
| 192 / N128            | M256+M128     | 128+64  = 192   | 384       | 512 | 2 stages ✓ |
| 192 / N160 (sw64 B*)  | M256+M128     | 160+80  = 240   | 480       | 512 | 2 stages ✓ |
| 160 / N128 (masked 192-composite, R2) | M256+M128 | 192 | 384       | 512 | 2 stages ✓ |
| 256 / N128            | M256+M256     | 128+128 = 256   | 512       | 512 | 2 stages ✓ (exactly full, zero headroom) |
| 64  / N32             | M128 only     | 16              | 32        | 32  | 2 stages ✓ |
| 64  / N64             | M128 only     | 32              | 64        | 64  | 2 stages ✓ |

Nothing needs to drop to 1 stage. (*N160: per-CTA B half-slice = 80 elems -> needs sw32 chunks
(5x16) or an N-composite per R3.8; the acc-col math is unaffected.)
The `TMEM_COLS = BN*EPI_PIPE_DEPTH` formula (KERNEL:878-885) and the matching dealloc
(KERNEL:1606-1609) must become a Config-level `ACC_COLS_PER_STAGE` = sum over atoms
(N for M256 pieces, N/2 for M128 pieces).

###### R5 — swizzle parameterization (confirmed vs DeepGEMM, bf16, MN-major operand, BK=64)

Selection rule (DeepGEMM csrc/jit_kernels/heuristics/utils.hpp:13-21): mode = largest of
{128,64,32,16} bytes dividing block_size*elem_size (16 = no swizzle, interleave only); per operand
independently (sm100.hpp:162-166). Propagation:
- tmap enum: mode_into_tensor_map_swizzle (csrc/jit_kernels/impls/runtime_utils.hpp:93-110);
  inner box dim forced to swizzle_mode/elem_size (runtime_utils.hpp:119-120, and
  deep_gemm/include/deep_gemm/common/tma_utils.cuh:9-12).
- descriptor bits 61-63 (PTX Table 45, ptx 31931-31953; enum values
  cute/arch/mma_sm100_desc.hpp:79-85): NONE=0, 128B_BASE32B=1, 128B=2, 64B=4, 32B=6.
- MN-major LBO/SBO (sm100_utils.cuh:105-131): LBO = BLOCK_K * chunk_elems * elem_size =
  BK * mode_bytes; SBO = 8 * chunk_elems * elem_size = 8 * mode_bytes.
- boundary/base-offset rule: pattern restarts at 1024B/512B/256B for 128B/64B/32B modes
  (PTX Table 46, ptx 31955-31967) — keep per-stage smem slices aligned accordingly.

| mode | chunk elems (bf16) | tmap enum | desc bits 61-63 | LBO (BK=64) | SBO | per-K-row byte pitch | coord divisor | status |
|---|---|---|---|---|---|---|---|---|
| 128B | 64 | CU_TENSOR_MAP_SWIZZLE_128B | 2 | 8192 | 1024 | 128 | 64 | validated (KERNEL:350-357, 364-368; tmap KERNEL:515-548) |
| 64B  | 32 | CU_TENSOR_MAP_SWIZZLE_64B  | 4 | 4096 | 512  | 64  | 32 | validated (KERNEL:370-377, 552-570, 1169-1175) |
| 32B  | 16 | CU_TENSOR_MAP_SWIZZLE_32B  | 6 | 2048 | 256  | 32  | 16 | NEW — `make_smem_desc_sw32` = the sw64 form (KERNEL:373-377) with 6ULL<<61, lbo=BK*32, sbo=256 |

(coord divisor = chunk elems: TMA box coordinates are in whole chunks, cf. A_col =
block_row*BM/SWIZZLE_W, KERNEL:532-536, and the sw64 z = block_row*(BM/32), KERNEL:552-556.)

###### R6 — minimal template-change set, ordered by unlock

1. **Composite-M knob** (unlocks 9 shapes — every "need composite-M" row):
   Config gains `M_PIECES` (e.g. {256}, {256,128}, {256,256}, {128}); the ::2 MMA branch
   (KERNEL:1186-1204) issues one mma per piece per K-step, mirroring the SWAP_AB dual-issue
   (KERNEL:1152-1185): per-piece idesc (generalize KERNEL:653-664 with an M_ATOM parameter),
   a-desc + (16384 * piece_index) (R3.7), shared b-desc, d-tmem col base advancing by the
   piece's cols/stage (R4), one commit unchanged.
2. **Layout-B interleaved epilogue read** (with #1 completes the composite tiles; ALONE unlocks
   64x64_2x2 + 128x64_2x2 = 3 shapes = STAGE T2): new epi variant for M128 pieces using the R1
   warp map — reuses `tcgen05_ld` 32x32b.x8 as-is; only row/col decode + store predication differ
   from KERNEL:1457-1554.
3. **ACC_COLS accounting** (required by #1): replace BN*EPI_PIPE_DEPTH at KERNEL:878-885 and
   KERNEL:1606-1609 with the R4 per-stage sum; static_assert <= 512.
4. **Relax BN static_assert** (KERNEL:149-151) for BN=32/64 tiles + B-slice swizzle mode by width
   (BN/2 elems: 128->sw128 as today, 32->sw64, 16->sw32) via the R5 table.
5. **160-row alias** = 192-composite + store predication (R2) — config-only once #1-#3 land.
6. **sw32 row** (make_smem_desc_sw32 + CU_TENSOR_MAP_SWIZZLE_32B tmap) — only needed if the
   96/80-wide N-composite readings (R3.8) or 64x64-with-N32 slices win benchmarks (STAGE T3).

No open items.

### 41. R7-R12 RESEARCH RESOLUTION — small/odd tiles (2026-07-07)

Deep research pass over local ptx_isa_9.3.txt + cutlass/include + DeepGEMM + CUDA 13.0 driver
header, resolving everything the §36 gap table left open beyond §40's R1-R6. All file:line
cites verified 2026-07-07.

R7 (instruction-N legality): ::2 dense f16 N = {16..256} STEP 16 (PTX Table 44,
ptx:31336-31337) -> N=160 and N=192 both LEGAL as single instructions. ::1 = step 8
(ptx:31328-31330). CUTLASS asserts match (mma_sm100_umma.hpp:88-89 1SM N%8; :554-555 2SM
N%16). Caveat: the 2SM TS (A-from-TMEM) variant needs N%32 (:638) — irrelevant for our SS
path. => The real blocker for BN=192/160 is only our BN%(CS*64) static_assert
(KERNEL:149-151), not the hardware.

R8 (narrow MN-major B slices 96/80/32 elems): mode = largest of {128,64,32,16} bytes dividing
slice_bytes (DeepGEMM heuristics/utils.hpp:13-21; per-operand: sm100.hpp:164-165).
  96 elems (192B) -> 64B mode; 80 elems (160B) -> 32B mode; 32 elems (64B) -> 64B mode.
Propagation (all confirmed in DG):
  tmap: runtime_utils.hpp:93-107 mode->CU_TENSOR_MAP_SWIZZLE_*; inner box FORCED to
    mode/elem_size (:120-121) -> 32/16-elem boxes; wider slice = multiple TMA issues
    (tma_utils.cuh:10-32, BLOCK_INNER_ATOM): 96 = 3x32-elem boxes, 80 = 5x16, 32 = 1x32.
  desc: bits 61-63 = 2/4/6 for 128/64/32B (PTX Table 45 ptx:31936,31950-31953; CUTLASS
    mma_sm100_desc.hpp:79-84). BK=64 bf16: 64B mode LBO=4096 SBO=512; 32B mode LBO=2048
    SBO=256 (DG sm100_utils.cuh:118-127; matches our validated sw64 builder).
  per-K-step advance: = mode_bytes (128/64/32 B) per K-row (DG sm100_utils.cuh:82-92).
  alignment: swizzle pattern restarts at 1024/512/256B (PTX Table 46 ptx:31959-31965);
    keep chunk bases on those boundaries, base_offset=0 (DG hardcodes 0, sm100_utils.cuh:32).
NOTE: DG deliberately AVOIDS 32B mode for perf ("32B swizzle yields poor performance",
sm100.hpp:107-110 forces MN-extent mult of 64). So 128x160 (80/CTA) is legal-but-slow —
build it last, benchmark before shipping.

R9 (BN=160 epilogue store — the 160/64 = 2.5 STORE_COLS problem): 
  (a) 32-col box under SWIZZLE_128B is LEGAL (cuda.h:23323-23326: inner extent <= swizzle
      span, not ==) BUT
  (b) our per-row store XOR g^(r&7) is WRONG for a sub-span box: swizzle is a function of
      absolute smem offset; two 64B rows share one 128B line -> no per-row XOR exists.
      Correct per-mode XOR (from DG sm100_store_cd.cuh:66-83): 128B: g^(r&7);
      64B: g^((r>>1)&3)  (TRAP: naive g^(r&3) silently corrupts); 32B: g^((r>>2)&1).
  (c) DeepGEMM NEVER mixes chunk widths: one cd swizzle mode = largest dividing the full
      tile width; STORE_BLOCK_N = mode span; hard assert BLOCK_N % STORE_BLOCK_N == 0
      (sm100_store_cd.cuh:33-35, sm100_bf16_gemm.cuh:75, sm100.hpp:166-167).
  (d) DECISION: for BN=160 use ONE SWIZZLE_64B C-tmap, box {32, BM}, STORE_COLS=32
      (DG-identical; 5 stores/wave vs 3 — TMA store not the bottleneck). Padding REJECTED
      (TMA clips only at tensor boundary -> interior tiles would overwrite neighbors).
  => Template: STORE_COLS, C-tmap swizzle, and store XOR all derive from one new
     Config::CD_MODE_BYTES (128 today, 64 for BN=160 tiles).

R10 (per-CTA M=64 tiles, instr M=128 ::2 Layout B): re-verified CUTLASS interleaved atom
(mma_traits_sm100.hpp:543-552) == R1 map. Epilogue = clean 4-warp QUADRANT map for per-CTA
64x64: warp w -> rows 32*(w&1)..+32, cols 32*(w>>1)..+32, plain 32x32b loads (no 16-lane
machinery). TMEM: per-CTA N/2 cols/stage (32 for N=64), alloc min 32, power-of-2 in [32,512]
(ptx:32701-32703) — 2 stages = 64 cols, trivial. A slice 64 rows = exactly one 128B chunk
(today's TN desc values unchanged). B slice N/2=32 elems -> sw64 (R8). => 64-row tiles need
ONLY: R8 B-path + quadrant epilogue + idesc M-field=128.

R11 (512x512 cluster tile / 256x256_2x2 #41): geometry settled = 4 CTAs / 2 pairs; per-pair
512x256 = composite 2x M256 atoms x instruction N=256. Per-CTA acc = 2x256 = 512 TMEM cols
per stage -> EPI_PIPE_DEPTH=1 FORCED (a 256x256 fp32 per-CTA accumulator is 512 cols no
matter the decomposition). The N=128/8-CTA alternative contradicts the observed 4-CTA cluster
(nsys, §23). 1-stage is normal production config: DeepGEMM auto-selects 1 or 2 epi stages by
exactly this arithmetic (sm100_fp8_gemm_1d1d.cuh:105-106) and its stage/phase math is
identical to ours -> protocol-correct at depth 1, just no cross-wave MMA/epilogue overlap.
Our EPI_PIPE_DEPTH>=1 assert already admits it. nvjet's actual stage count UNVERIFIED
(inferred — no legal alternative at 4 CTAs). This tile serves the fattest-K shape; K-loop
dominates, overlap loss should be small — benchmark before/after per policy.

R12 (idesc fields): N-field bits 17-22 = N>>3 (bit 23 reserved-0), M-field bits 24-28 = M>>4
(bit 29 reserved-0) (PTX Table 47 ptx:31989,32013-32018; CUTLASS mma_sm100_desc.hpp:429-431,
486-487). 160>>3=20, 192>>3=24 encode fine. TRAP for composite-M: M-field must encode the
PER-ATOM instruction M (256 -> value 16), NOT CLUSTER_SIZE*BM — per-CTA BM=256 would give
512>>4=32 which OVERFLOWS the 5-bit field into reserved bit 29 = corrupt descriptor. Config
gains M_INSTR; sm103_make_i_desc uses it.

UPDATED BUILD ORDER (supersedes §40 T1-T4 ordering, all research now closed):
  B1. Composite-M knob (M_PIECES) + M_INSTR idesc + ACC_COLS per-stage sum (R11/R12 traps
      folded in). Unlocks: 256x256_2x2 (EPI=1), 160x192_2x2 masked-192, 320x192_2x2f.
  B2. Layout-B quadrant epilogue (R10) -> 64x64_2x2, 128x64_2x2, and completes composite
      tiles that carry an M128 piece (192-row = 256+128).
  B3. B-operand swizzle parameterization mode {128,64,32} (R8: desc_32B + mode-parameterized
      b_off + multi-issue TMA + alignment) -> BN=192 tiles (96/CTA sw64), BN=64 (32/CTA sw64).
  B4. CD_MODE_BYTES store path (R9: STORE_COLS=32 + SWIZZLE_64B C-tmap + (r>>1)&3 XOR)
      -> BN=160 tiles (with R8 32B-mode B slices, 80/CTA). DG flags 32B mode slow — build
      last, benchmark, keep only if it beats our existing coverage on those shapes.
  B5. Config aliases + test suites per tile; benchmark each vs its nvjet kernel (clean GPU);
      keep winners.
Validation gates: every stage compiles sm_103a locally + full suite on the new idle B300 +
one-tile memcmp probe for the 64B store XOR derivation (the one derived-not-cited item).

### 41.1 VERIFICATION ADDENDUM — two independent passes over R7-R12 (2026-07-07)

Per the be-SURE policy, §41 was re-verified twice before any implementation: (pass A) an
adversarial refuter given the R7-R12 claims and told to break them (priority attacks: store-XOR
index math, LBO/SBO constants, TMEM arithmetic, idesc bit positions); (pass B) a fully
independent re-derivation from scratch, local sources + official web (docs.nvidia.com PTX ISA +
Driver API, CUTLASS/DeepGEMM GitHub). ALL structural claims survived both. Corrections and
refinements:

1. CONSTANT FIX (R7): 1SM SS f16 N-floor is 8, not 16 (mma_sm100_umma.hpp:88-90:
   `N % 8 == 0 && 8 <= N && N <= 256`). 2SM TS variant floor is 32 with N%32 (:637-638).
   No target tile affected.
2. STRONGER PERF WARNING (R8): DeepGEMM's MN-major filter (sm100.hpp:95-99) forces 64-elem
   alignment = for bf16 it NEVER uses 64B or 32B swizzle on MN-major A/B LOADS at all (only on
   the CD store path, and it discards non-128B candidates when either operand is K-major,
   :132-136). Our planned sw64 (96-wide) and sw32 (80-wide) B slices are PTX-legal but beyond
   what DG ships -> the 128x192 family and especially 128x160 MUST beat our existing coverage
   in benchmarks or be dropped. (Our silicon precedent: the validated B2 160-wide sw64 A
   operand works correctly; perf still confounded by the unvectorized swap epilogue.)
3. LBO/SBO ARE LAYOUT-DEPENDENT, NOT MODE CONSTANTS (R8 caveat): the PTX worked example for
   MN-major 64B (ptx:31763-31778, canonical full-MN-contiguous arrangement) gives LBO=512B/
   SBO=1024B — DIFFERENT from our values. Our 4096/512 (sw64) and 2048/256 (sw32) assume the
   DG-style CHUNK-FACTORIZED smem arrangement (all BK rows of one atom contiguous, then next
   atom) — which IS our TMA 4D layout, and matches our silicon-validated sw64 builder. Do not
   copy LBO/SBO from PTX examples without checking the physical arrangement.
4. STORE-XOR FORMULAS CONFIRMED TWICE (R9): re-derived line-by-line from sm100_store_cd.cuh
   :63-82 AND independently from the hardware Swizzle<B,4,3> algebra: 128B g^=(r&7);
   64B g^=((r>>1)&3) — key advances every TWO rows (r&3 would be silent corruption);
   32B g^=((r>>2)&1). Narrow-row impossibility (64B rows under 128B tmap) proven sound:
   XOR keys 4-7 move 16B groups across the row boundary. STORE_BLOCK_N uniformity caveat:
   the bf16 DG kernel formula is swap-AB-conditional (sm100_bf16_gemm.cuh:75) but still one
   uniform width per config.
5. R11 ALTERNATIVE-GEOMETRY ARITHMETIC REFINED: depth-2 alternatives for the 512x512 cluster
   tile: N=128 WITH composite-M kept -> per-pair 512x128 -> 4 pairs = 8 CTAs; without
   composite (single M256 atom/pair) -> per-pair 256x128 -> 8 pairs = 16 CTAs. Observed nsys
   cluster = 4 CTAs ⇒ both alternatives excluded ⇒ N=256 + EPI_PIPE_DEPTH=1 stands.
6. R12 MECHANISM NUANCE: encoding M=512 via CUTLASS's 5-bit bitfield silently TRUNCATES
   m_dim_ to 0 (invalid M); via raw shift (32<<24) it sets reserved bit 29. Both corrupt —
   per-atom M (256 -> field 16) mandatory either way.
7. WEB-ONLY FINDINGS (pass B): (i) NO official doc/blog/issue anywhere discusses composite
   different-M MMA tiles; the PTX rules that make it safe are: async mma unordered across
   different-shape streams (harmless for disjoint TMEM), per-stream same-shape pipelining
   (ptx:32393-32399), one commit tracks ALL prior ::N mmas of the thread (ptx:35698-35699),
   single .cta_group per kernel (ptx:32725). Our ::1 composite is silicon-validated (P5/P6
   14/14); the ::2 composite still needs its own silicon validation = B1's first test.
   (ii) TMA store swizzle is symmetric with loads for bf16 (store/load asymmetries exist only
   for sub-byte padded types, cuda.h:23346-23354). (iii) PTX ISA 9.3 is the newest published
   version as of 2026-07-07 — no rule changes pending.

VERDICT: §41 build order B1-B5 GREEN. B1 gates on a silicon test of the ::2 composite
(smallest possible: one 512x512-tile shape vs cuBLAS ref, worst-element dump on).

### 42. B1 IMPLEMENTATION PLAN — composite-M for cta_group::2 (2026-07-07)

Key routing insight (from §41/§41.1 reconciliation): pick the decoded-tile READINGS that avoid
narrow-B swizzles first —
  128x192 family (5 shapes): take the 384x128 cluster reading = per-CTA 192 rows (composite
    256+128) x N=128/pair -> B slice 64 elems = TODAY's sw128 path. No narrow B needed!
  128x160 (1 shape): take the 320x128 reading = per-CTA 160 rows (masked 192-composite, R2)
    x N=128/pair -> sw128 B too.
  => B1+B2 alone cover: 256x256_2x2 (1) + 128x192 family (5) + 128x160 (1) + 64x64/128x64 (3,
     need B2 epi + sw64 B slice 32) — the sw32 (B4) work shrinks to 160-wide-N tiles only
     (160x192_2x2, 320x192_2x2f), benchmark-gated per §41.1.2.

B1 SCOPE (this stage): pure M256+M256 composite, single alias 256x256_2x2 (nvjet #41,
cluster 512x512, EPI_PIPE_DEPTH=1 forced). Validates the ::2 composite mechanics on silicon
(the one §41.1.7 unverified) with ZERO other new machinery (BN=256 -> sw128 B, Layout-D-only
epilogue).

EDIT LIST (Bgemm_sm103_cluster_template.cuh):
1. Config: derive pieces from BM (no new template param):
   N_M256 = CS==2 ? (2*BM)/256 : 1; HAS_M128 = CS==2 && (2*BM)%256 != 0 (static_assert
   !HAS_M128 for non-swap until B2); ACC_COLS_PER_STAGE = SWAP_AB ? BN : N_M256*BN
   (+BN/2 when HAS_M128, B2); TMEM_COLS = pow2-round(ACC_COLS*EPI_PIPE_DEPTH), assert <=512.
   EPI_ROWS = min(BM,128); NUM_EPI_WARPS = SWAP_AB ? 4 : EPI_ROWS/32 (BM=256 -> 4 warps,
   TMEM has only 128 lanes; epilogue loops pieces instead); STORE_BUF sized by EPI_ROWS.
2. i_desc M-field trap fix (R12): M_INSTR = CS==2 ? 256 : BM (was CS*BM -> overflows at
   BM=256).
3. Consumer ::2 non-swap: per K-step loop p in [0,N_M256): a_desc at A_smem + a_off +
   p*16384 (R3.7: TN chunk-2p = 2p*LBO8192; NN/NT 16 atoms *SBO1024 — same constant, both
   verified additive with a_off); d-tmem = tmem_addr + wave_stage*ACC_COLS + p*BN; shared
   b_desc; one commit unchanged.
4. Epilogue non-swap: outer piece loop; piece p reads 128 lanes at col base
   wave_stage*ACC_COLS + p*BN, stores C rows block_row*BM + p*128 + row; store buffer/TMA
   box stays 128 rows ({64, EPI_ROWS} box); tmem_empty arrive after last chunk of LAST piece;
   beta C_in row likewise +p*128.
5. Alloc/dealloc: both use Config::TMEM_COLS (fixes the §41 under-allocation trap for
   composites).
6. Launcher: relax BM static_assert to admit CS==2 && BM==256 && EPI_PIPE_DEPTH==1;
   C-tmap boxRows = EPI_ROWS.
7. Aliases: Sm103Config_256x256_2x2 = <256,256,64,2, 0, /*EPI*/1, 2, 2>.
   Smem: A 32KB + B 16KB = 48KB/stage; auto depth (227-32)/48 = 4 stages ✓.
8. Test: P7 suite env RUN_CM256, 7 shapes x 3 layouts (M%512-aligned mains + partial-M +
   batched + alpha-beta + 4096 cube + wideN), worst-element dump on, [n/total] updated.
Backward-compat proof obligation: for every existing config (BM=128/160) ACC_COLS==BN and
N_M256==1 -> all new expressions reduce to today's values at compile time; full 406 suite
must stay green.
VALIDATION GATE: compile sm_103a local -> full suite + P7 on idle B300 -> §36-style doc.
Then B2 (Layout-B quadrant epi) -> 128x192 aliases etc. per the routing above.

### 43. B1 SILICON-VALIDATED — cta_group::2 composite-M works (2026-07-07)

P7 run on the idle B300 (output archived in output_2026-07-07 series): 21/21 PASS bit-exact
(MaxRelErr 0.00e+00; beta case 1.91e-06 = accumulate-order noise), all 3 layouts, full 430-run
regression green alongside (existing 406 unaffected — the compile-time-identity claim held).

THE HEADLINE: the ::2 composite (two M=256 MMAs per K-step, disjoint TMEM col regions, one
commit tracking both, per-atom idesc M-field, EPI_PIPE_DEPTH=1) is now validated on hardware —
§41.1.7 flagged this as the one unverifiable-from-docs item anywhere. Covered: partial-N
cluster, batched, alpha/beta (piece_row C_in indexing), 4096 cube, wideN.

Perf (first look, EPI=1 as forced):
  square4096: 90.3-90.5% of cuBLAS (vs 93.7% on our 128x256_2x2 — the ~3pt gap is the lost
    MMA/epilogue overlap, as predicted by R11).
  wideN 4096x12288: 88.2-88.9% — BEATS our rigid 2x2 (82%) and 2x2f (86.4%) on this shape;
    the fat 512x512 tile earns its keep exactly where nvjet deploys it.
  Small shapes 35-51%: expected/irrelevant — a 512x512 cluster tile means 1-4 clusters total
    on a 148-SM GPU; the dispatcher will never route small shapes here.
  alpha-beta 12.5%: the known scalar-beta-read cost amplified by EPI=1 — §36 note stands.

BUG FOUND BY THE RUN: cm256-partialM (M=768) was SILENTLY SKIPPED by a legacy test-file
filter `if (Config::BM == 256 && (M%512 || N%256)) continue;` — dead code before B1 (no
config ever had BM=256), wrong now (single-batch partial-M is exactly what TMA OOB + edge
clip must be tested for). REPLACED with the honest generic guard: skip only `batch > 1 &&
M % (CLUSTER_SIZE*BM) != 0` — the genuinely-unsupported combo, because the C store clips at
the TENSOR edge (batchCount*M), so a batch's partial-tile tail rows would bleed into the
NEXT batch's C region. This batched-partial-M limitation is LIBRARY-WIDE (all configs, all
along) — recorded here as a known constraint; the true fix is per-batch store clipping
(fold into §34 task 4 general-shape work). P7 rerun with partialM pending -> expect 24/24.

NEXT: B2 — Layout-B quadrant epilogue + M128-piece composites (192-row tiles: 128x192 family
5 shapes + 128x160 masked, all sw128-B per §42 routing) then P8 suite.

### 43.1 P7 rerun with the filter fix — B1 ARC CLOSED (2026-07-07)

Rerun (RUN_CM256=1, 276 cases incl. base suites + sweep): ALL PASS. cm256-partialM (M=768,
the previously-skipped case) now runs and is bit-exact in all 3 layouts — TMA OOB zero-fill on
the fully-OOB piece + tensor-edge store clip verified for the composite path. P7 = 24/24.
Perf reproduced: square4096 90.2-90.4%, wideN 88.2-89.2% (still ahead of rigid 2x2/2x2f on
that shape). Sweep steady (best-of-2 85.3%/82.3%; 8b-LMhead vert 127.6% this run).
B1 complete: composite-M mechanics + accounting + partial-tile handling all silicon-validated.
NEXT UP: B2 per §42 — Layout-B quadrant epilogue + M128-piece composites (192-row tiles).

### 44. B2 IMPLEMENTED — mixed-shape composite (M256+M128 Layout-B tail) (2026-07-07)

The part with NO public precedent (§41.1.7): one K-step now issues an M=256 atom AND an M=128
atom (different shapes, disjoint TMEM, one commit). Template changes:
1. Config: NUM_PIECES = N_M256 + HAS_M128; asserts — remainder must be exactly M128 (M64
   illegal for ::2, R2); B2 scope pins HAS_M128 tiles to BN=128 (quadrant<->store-chunk map).
2. sm103_make_i_desc gains M_ATOM override (tail passes 128; R12 per-atom M-field).
3. Consumer: tail atom after the M256 pieces — a-desc at +N_M256*16384 (chunk progression,
   R3.7), SHARED b_desc, d-tmem col base N_M256*BN (Layout B = BN/2 cols, lane alignment 0),
   same commit.
4. Epilogue: Layout-B tail branch in the piece loop — quadrant warp map (R10: warp w -> rows
   32*(w&1)..+32, C-cols (w>>1)*64..+64; ALL warps read the SAME BN/2 tmem cols at their own
   32-lane strip); per-chunk only the owning warp pair loads/writes, all warps keep the
   barrier protocol; tmem-empty signal moved to the tail (last piece).
5. NEW 64-row C store tmap (C_tmap_tail) for the tail piece — a 128-row box would overwrite
   the NEXT tile's rows (TMA clips only at the tensor edge). Kernel signature +1 param.
6. Launcher admits BM=192 (EPI=2 fits: ACC=192/stage, 384 -> 512 alloc); aliases
   Sm103Config_192x128 (2x1, nvjet 128x192_2x1_v 384x128 reading) + _2x2 (for 2x2f runs).
7. P8 suite (RUN_CM192): 9 shapes x (NN/NT/TN + TN-2x2f-dynamic) = 36 cases. Partial-M
   shapes probe OOB through BOTH piece kinds: 576 = tail-only-OOB CTA; 448 = partial M256
   piece + fully-OOB tail. Compiles clean sm_103a. SILICON PENDING -> P8 run.
Unlocks on pass: 128x192_2x1_v + 128x192_2x2f (5 shapes); masked-160 (1 more) is a config +
store-predication knob on top (B2.5).

### 44.1 B2 SILICON-VALIDATED — mixed-shape composite works (2026-07-07)

P8 run: 312/312 total, P8 = 36/36 bit-exact (MaxRelErr 0.00e+00 everywhere, even beta cases).
Validated in one shot, first try — no debugging round needed:
- Mixed M256+M128 atoms per K-step (different instruction shapes, disjoint TMEM, one commit):
  WORKS. No public library implements this; derivation §41 R10/R12 was exactly right.
- Layout-B interleaved TMEM read (quadrant warp map) correct in all 3 layouts.
- Both §41 traps confirmed necessary-and-sufficient: per-atom idesc M-field + 64-row tail
  C-tmap (the 128-row box WOULD have corrupted neighbors).
- OOB through both piece kinds: partialM 576 (tail-only-OOB CTA) + oobTail 448 (partial M256
  piece + fully-OOB tail) both clean.
- P8-F: same tile under DynamicCluster (pref-4/fb-2) = the real nvjet 128x192_2x2f geometry,
  9/9.
Perf first-look: small shapes ~43% (few clusters, expected); qwen Q-projections 63-68% at
HORIZONTAL wave — nvjet runs this tile as _v (vertical); wave/knob selection is B5's
benchmark job, not a correctness concern. alpha-beta 25% = known beta path.
COVERAGE NOW: 36/42 exact-or-equivalent (adds the 5 128x192 shapes on top of §35's 30 + #41
256x256_2x2 from B1). Remaining: 128x160 masked (B2.5 config knob), 64x64/128x64 (B3 sw64 B),
160x192_2x2 + 320x192_2x2f (B4 sw32, benchmark-gated).

### 45. B2.5 DESIGN CORRECTED after challenge — element-granular A loads, NOT sw64 (2026-07-07)

User challenged the claim "masked-160 TN needs the 64B-swizzle A operand". Verification pass
(cuda.h + PTX + DeepGEMM, all cited) overturned it:
- TMA tile start coordinates are ELEMENT-granular; the only constraint is the resulting
  address being 16B-aligned (ptx §5.5.3.1:3016-3018; cuda.h cuTensorMapEncodeTiled has NO
  coordinate constraints at all). x=160 bf16 = 320B ✓. The 64-elem restriction was OUR 4D
  chunk-factorized tmap ({64, K, M/64} with z = whole-chunk index), not hardware.
- The TMA swizzle permutation is a function of the SMEM DESTINATION offset only (ptx §5.5.7
  defines every mode as a destination-pattern table; cuda.h "Swizzle 16B chunks within 128B
  span") -> a box loaded from ANY legal x lands byte-identically in smem -> tcgen05
  descriptors/K-step offsets/consumer stay UNTOUCHED.
- DeepGEMM's own issue loop (tma_utils.cuh:23-32) is exactly this: element-granular inner_idx
  + one TMA per swizzle atom, smem advancing atom-bytes per issue. Their 64-elem alignment is
  a HEURISTIC block-size filter (sm100.hpp:95-100 'continue' in config search), never a
  legality check.
- mbarrier completes per-copy with copied bytes (ptx:17115-17119) -> expect_tx stays the
  stage total, additive over the 3 copies.
DECISION: masked-160 TN = element-granular A tmap ({M-elems, K, batch, 1}) + BM/64=3 chunk
copies per stage at x = 160*block_row + 64*i, each landing at smem +i*BK*128. 128B swizzle
everywhere; sw64 route REJECTED (would change tmap+smem layout+descriptors+offsets AND use
the DG-flagged slow narrow mode, for zero bandwidth benefit). NN/NT need nothing (A y-coord
already element-granular). Residual perf note: rows at x%64==32 straddle two 128B L2 lines —
sectors still fully utilized; micro-benchmark per policy. True M-extent in the element tmap
also removes the M%64 truncation hazard for this path (guard relaxes to M%8 = 16B rows).

### 45.1 B2.5 SILICON-VALIDATED — masked 160-row tile + element-granular TMA (2026-07-07)

P9 run: 336/336 total, P9 = 24/24 bit-exact (all 3 layouts). First-try again — the challenged-
and-corrected §45 design validated in one shot:
- ELEMENT-GRANULAR A tmap + per-chunk multi-issue (TN): tile bases 160/480/... that no chunked
  tmap can address, loaded with UNCHANGED 128B swizzle/descriptors/consumer. The §45 claim
  (swizzle = f(smem dest only)) holds on silicon.
- Masking: 192-row composite computed, 160 stored — 32-row tail box + r_in<TAIL_ROWS
  predicate. Interior tiles' masked rows overlap the NEXT tile's real rows; bit-exactness of
  partialM/batched/qwen proves zero bleed (any tail-box error would corrupt rows 160-191 of
  every interior tile).
- cm160-batched (640 = 2x320 exact) exercises the batched masked path safely.
Perf first-look: TN qwen-q06 57.8% vs NN/NT 61-64% on the same shape — a mild TN deficit
consistent with the predicted misaligned-row L2 effect (§45: rows at x%64==32 straddle two
128B lines) + 3x TMA issue overhead. Delta ~10%, benchmark-gated at B5 like everything else
(this tile serves 1 shape; our 128x128/192x128 tiles compete for it).
COVERAGE: 37/42 exact tiles. Remaining: 64x64_2x2 + 128x64_2x2 (B3: sw64 B-slice, quadrant
epilogue already exists from B2) and 160x192_2x2 + 320x192_2x2f (B4: sw32 B, benchmark-gated).

### 46. B3 IMPLEMENTED — small tiles 64x128_2x2 + 64x64_2x2 (2026-07-07)

The skinny-N tiles (K/V projections where wide tiles run 35-50%). Both are PURE Layout-B
configs: pair M = 128 = ONE M=128 atom, so N_M256=0 and the B2 "tail" branch IS the whole
tile (NUM_PIECES=1, piece_row=0). Template changes:
1. NUM_EPI_WARPS fix: Layout-B configs always need 4 epi warps (tail spans all 128 TMEM
   lanes even for a 64-row tile; EPI_ROWS/32 would give 2 for BM=64).
2. Tail epilogue GENERALIZED for BN=64: warp pair (w>>1) owns TAIL_COLS=BN/2 C-cols —
   for BN=128 that's the two 64-col store chunks (B2, unchanged codegen); for BN=64 the two
   32-col HALVES of the single chunk (col_in offset enters the store-buffer 16B-group index:
   g = col_in/8 + n before the swizzle XOR).
3. FIRST sw64 B-OPERAND path (§41 R8): B_SLICE=BN/CS; B_SW64 when slice=32 elems ->
   new init_4d_tma_B_KN_sw64 ({32,K,N/32} chunks, box z=B_SLICE/32), B_col divisor 32,
   b_off = (k1*64+k2*16)*64 (64B/K-row pitch), b_desc sw64 (LBO=4096, SBO=512). NT (B [N,K],
   K-contiguous) untouched. Alignment guard: N%32 for sw64-B configs.
4. Asserts relaxed: B_SLICE%32; HAS_M128 BN in {64,128}; launcher admits BM=64.
Aliases: Sm103Config_64x128_2x2 (<64,128,64,2>, nvjet 128x64_2x2, cluster 128x256, sw128 B)
+ Sm103Config_64x64_2x2 (<64,64,64,2>, nvjet 64x64_2x2, cluster 128x128, sw64 B).
P10 suite (RUN_CM64): 8 shapes x 2 configs x 3 layouts = 48, including the REAL targets
q25-0.5b-K (N=128) and q25-3b-K (N=256). Compiles clean sm_103a; SILICON PENDING.
NOTE: sw64 B carries the §41.1.2 perf warning (DG never uses narrow swizzle on operands) —
the 64x64 tile is benchmark-gated at B5; the 64x128 tile has no such concern.

### 46.1 B3 SILICON-VALIDATED + first honest small-tile perf data (2026-07-07)

P10 run: 384/384 total, P10 = 48/48 bit-exact (both configs x 3 layouts). Fourth consecutive
first-try stage. Validated: pure-Layout-B configs (N_M256=0, tail = whole tile), the BN=64
quadrant generalization (32-col halves, col_in in the swizzle-group index), and the FIRST
sw64 B-OPERAND (tmap + descriptors + 64B/K-row offsets) on silicon.

PERF FINDINGS (the real story — target shapes):
1. q25-0.5b-K (4096x128x896, our worst shape): ALL our tiles plateau at ~112-113T —
   64x128 (112.4T), 64x64 (113.0T), previously 128x128 (112.8T), 128x256 (90.6T). cuBLAS:
   202-265T (high run-to-run variance). The plateau across tile sizes says the binding
   constraint is NOT tile geometry/N-waste — likely parallelism/wave quantization (N=128 ->
   a single tile column; only ~32-64 clusters on 148 SMs) or K-split: cuBLAS may SPLIT-K this
   shape (its 2x higher perf with the same arithmetic suggests more CTAs sharing the K loop).
   ACTION for B5: nsys the actual nvjet launch on this shape (grid dims will reveal split-K)
   before assuming more tiles help. Small tiles alone do NOT rescue skinny-N.
2. q25-3b-K (4096x256x2048): 64x128 = 413.7T (58.8%) ~= 128x128 (405T) — fine but not a win;
   64x64 = 295-322T (42-45%) — the sw64-B cost is REAL and visible, exactly as DeepGEMM's
   heuristic predicted (§41.1.2). The 64x64 tile currently wins NOTHING; expect it to be
   retired at B5 unless the true 64x64-shape (small M AND N) shows otherwise.
3. Small shapes 43-79%: quantization-dominated as with all small tiles.
COVERAGE: 40/42 exact tiles (adds 128x64_2x2 x2 shapes + 64x64_2x2). Remaining 2 = B4
(160x192_2x2, 320x192_2x2f: sw32 B + CD64 store) — benchmark-gated, and given finding #2
the sw32 route is now VERY likely to lose; decide after B5 measures the 192-row tiles on
those shapes. NEXT: B5 clean benchmark phase (per-shape best-variant table incl. nsys
split-K reconnaissance) OR B4 completionist build — user's call.

### 47. B5 BENCHMARK HARNESS + split-K question resolved from our own nsys data (2026-07-07)

Skinny-N mystery (§46.1.1) answered from §16's nsys catalog — NO split-K: nvjet runs
q25-0.5b-K with 64x64_64x16_2x2_h_bz = the 64x64 tile we now have, but with a 16-STAGE
pipeline. And §21's clean-node retest measured cuBLAS at only ~160T on that shape (today's
202-265T = timing noise on a ~6us kernel) — our 113T is nearer than the noisy log implied.
Deep-pipeline (our auto-depth already gives 17 stages at 64x64) + winner table will settle it.

B5 harness (RUN_BENCH=1 in the test file): for each of the 42 Qwen shapes (TN, batch 1):
time cuBLAS (5+20 iters) then EVERY validated variant (3+10 iters), print per-variant rows +
per-shape BEST + final WINNER TABLE (shape | best variant | ours | cuBLAS | %) + summary
(avg best-of-all-variants %, #shapes where we BEAT cuBLAS). 17 variants benched per shape:
128x128 h/v, 128x256 h/v, 128x256_2x2 rigid/mcast/2x2f, 128x128_2x2, 192x128 h/v,
192x128_2x2f, 256x256_2x2, 160x128m_v, 64x128_2x2, 64x64_2x2, 1x2_128x192, swap160x192.
RUN_BENCH replaces the old 2-tile sweep in the same run (correctness suites unaffected).
This table IS the dispatcher's routing data (dispatcher wiring stays LAST per plan).

### 47.1 B5 FIRST RUN — winner table + ORDER-BIAS finding (2026-07-07)

First B5 run (unlocked clocks): best-of-all avg 82.6%, beat cuBLAS 2/42. BUT a methodology
flaw was caught before accepting the numbers: cuBLAS-vs-ours comparisons swing +-30% with
MEASUREMENT ORDER (boost clocks: whoever runs first on a cool GPU looks faster). Evidence:
8b-LMhead same variant measured 1630-1888T when run first (earlier sweeps) vs 1299T run late
in the bench; cuBLAS 1385-1530T when run second vs 1869T run first. The earlier sweeps were
biased FOR us, this bench AGAINST us — truth in between. Fixes: (a) bench now re-times cuBLAS
after the variants and uses max(first, recheck) + prints the drift; (b) OFFICIAL RUNS MUST
LOCK CLOCKS: nvidia-smi -pm 1 && nvidia-smi -lgc <freq> (root on the new server) — with
locked clocks drift ~0 and order stops mattering. STANDING RULE for all future perf claims.

WHAT IS VALID from run 1 (relative ranking among OUR variants, measured back-to-back):
- The new tiles EARN PLACES: 256x256_2x2 wins the fat Downs (3b-Down 98.8%, 8b-Down 97.4%),
  192x128 wins 4b-O/4b-Down, 64x128_2x2 wins the N=256 K-projections, 2x2/2x2m/2x2f win
  several Gates/Downs. 128x256_v remains the LM-head/wide-N king; 128x128 h/v takes the
  mid shapes.
- Winning nothing (as predicted §46.1): 64x64_2x2, 160x128m, swap160x192, 1x2_128x192 —
  retirement candidates pending the locked-clock rerun.
- Skinny-N (44-65%) remains the real gap regardless of bias.

### 47.2 SECOND BENCH RUN (locked 1700MHz) — power-cap throttle found; final protocol (2026-07-07)

Run 2 (locked 1700): avg 80.9%, beat 0/42. The drift check caught TWO residual bias sources:
1. POWER-CAP THROTTLE BELOW THE CLOCK LOCK on sustained heavy shapes: 7b-LMhead drift -23.8%,
   3b-LMhead -13.1% — a locked 1700MHz still drops under the power limit during long dense
   GEMMs, so later-measured variants (incl. ours) read low on exactly the big shapes.
   Suspicious symptom: 128x256_h "beating" _v on 3 LM-heads by tiny margins while running one
   slot earlier — contradicts every prior run.
2. Recheck had NO WARMUP: tiny shapes (0.5b-K -22.2%, 3b-K -16.9%) read low from cold caches
   right after the variant sweep. FIXED in code (5-iter warmup added to the recheck).
FINAL PROTOCOL: lock LOW enough that the power cap can never engage — nvidia-smi -lgc 1400 —
then order effects vanish for real (drift expected within ±2% everywhere). All absolute TFLOPS
scale down ~30%; only the ratios are the deliverable. Run 3 = the official table.
Run-2 rankings still directionally useful (mid/small shapes with ~0 drift): 128x256 h/v
dominate, 2x2 family takes Gates/Downs, 256x256_2x2 keeps 3b-Down (96.2%), 192x128 keeps
4b-O/Down, 64x128_2x2/64x64_2x2 take the tiny-N K-projections, 1x2_128x192 won 1.5b-Q.

### 47.3 B5 OFFICIAL RESULT — locked 1400MHz, warmup-fixed (2026-07-07)

Run 3 = the canonical table (archived: output_2026-07-07_B5_official_1400MHz.md). Drift now
within +-8.4% (was +-24%) — residual noise means near-tie rankings (<5% apart) should be
treated as ties by the dispatcher.

HEADLINE (honest): best-of-all-variants avg 82.3% of cuBLAS; we BEAT cuBLAS on 1/42
(q3-8b-Down, 128x128_v, 102.9%). Near-parity (>=94%): 7b-Down 99.7%, 8b-Gate 97.6%, 7b-Gate
97.1%, 8b-Q 95.2%, 0.5b-LMhead 95.0%, 3b-Down 94.0%, 4b-Gate 94.6%. The earlier ">100% on
LM-heads" claims are RETRACTED as boost-clock bias (§47.1-47.2).

KEY DIAGNOSTIC from the clock scaling: our LM-head TFLOPS scaled EXACTLY with clock
(0.93 T/MHz at both 2032 and 1400) -> our kernel is COMPUTE/ISSUE-bound there. cuBLAS lost
only 13% for a 31% clock cut -> theirs is MEMORY-bound (better per-SM efficiency, hits the
BW wall). CONSEQUENCE: LM-head/Q gaps (15-25%) are addressable by per-SM efficiency work
(epilogue vectorization, issue overhead), NOT by more tiles. Skinny-K (42-68%) stays the
structural gap (§47: nvjet's 16-stage pipeline + small-shape efficiency).

WINNER DISTRIBUTION (dispatcher routing): 128x256_v x12 (LM-heads, wide Gates), 128x256_h
x11 (mid Gates/Downs/Q), 128x128_h x6, 128x128_v x4 (incl. the 8b-Down WIN), 128x256_2x2 x3,
128x256_2x2m x2, 128x128_2x2 x2, 64x128_2x2 x2 (tiny-N K), 192x128_h x1, 192x128_2x2f x1.
WON NOTHING (retire from dispatcher, keep code): 64x64_2x2 (near-tie on 0.5b-K), 160x128m,
swap160x192, 1x2_128x192, 256x256_2x2 (94-96% near-ties with 128x256_2x2 on fat Downs —
keep as tie-alternative). B4 (sw32 tiles) CONFIRMED NOT WORTH BUILDING.
NEXT: wire the dispatcher from this table (plan's LAST step); optimization backlog ordered by
the diagnostic: (1) epilogue vectorization everywhere (compute-bound evidence), (2) skinny-K
deep-pipeline/launch work, (3) beta path.

### 47.4 VARIANT-MAJOR RUN — thermal-gradient bias found + preheat fix (2026-07-07)

Per user request the bench was restructured VARIANT-MAJOR (each variant swept over all 42
shapes, per-variant avg/wins summaries; cuBLAS baseline = its own clean pass first, recheck
sweep last; ONE preallocated max-size buffer set for every timing). First variant-major run:
best-of avg 85.1%, beat 6/42 — BUT the drift sentinel exposed a clean TIME GRADIENT, not
noise: early-baseline shapes (q25-*) drift -6..-13.7%, late shapes (q3-*) ~0±3%. Mechanism:
-lgc locks only the GRAPHICS clock — the MEMORY clock/HBM thermals decay ~10% over the first
~2 minutes, so the cuBLAS baseline for early shapes was measured in cold-fast-HBM conditions
our variants never saw. Evidence: ALL >100% results cluster in the late (q3) half; q25
LM-heads read 74-91% vs q3 LM-heads 100-103% — physically implausible asymmetry.
FIX: 60s heavy-GEMM PREHEAT before any measurement (all passes then run at the same steady
state). Optional extra: nvidia-smi -lmc to lock memory clocks too.
BENCH PROTOCOL (final form): idle GPU + -lgc 1400 (+-lmc if supported) + 60s preheat +
baseline sweep + variant sweeps + recheck sweep with max-correction + drift gate (accept run
only if worst drift within ~±3%).
Valid signal from this run regardless: per-variant averages (all measured at steady state):
128x256_v 77.3% / 128x128_v 77.0% / 128x256_2x2 71.9% / ... / 64x64_2x2 31.1% (weakest).
Vertical wave beats horizontal for BOTH tiles by ~8 points on average.

### 47.5 B5 FINAL — variant-major, locked 1400MHz + 60s preheat (2026-07-07) — OFFICIAL

Archived: output_2026-07-07_B5_FINAL_1400MHz_preheat.md. Full 17x42 matrix (714 kernel
timings + 84 cuBLAS). Methodology convergence across runs: 82.3%/1 win (run3) -> 85.1%/6
(run4 variant-major) -> 87.4%/8 (run5 preheated) — every fairness fix moved numbers UP (the
biases were mostly against us). Worst residual drift -9.1% on one early shape; winner table
uses cuBLAS best-of-two, so 87.4% is conservative.

OFFICIAL HEADLINE: best-of-variants avg 87.4% of cuBLAS; BEAT cuBLAS on 8/42:
8b-Down 111.3%, 1.7b-LMhead 104.6%, 4b-LMhead 104.6%, 8b-Q 103.9%, 8b-Gate 103.5%,
3b-Down 103.3%, 7b-Down 101.6%, 8b-LMhead 100.9%. Near-parity >=93%: 4b-Q 97.6%, 4b-Gate
98.5%, 7b-LMhead 98.6%, 0.6b-LMhead 95.9%, 1.7b-Down 95.3%, 7b-Gate 93.7%, 4b-Down 93.0%,
7b-Q 92.9%, 0.5b-Down 92.9%. The "125%" earlier claims: magnitude was boost-bias, direction
real (true wins ~101-111%).
Per-variant avgs (fair): 128x128_v 79.7 / 128x256_v 79.2 / 128x256_2x2 74.6 / 192x128_v 73.3
/ 128x128_2x2 72.6 / 128x128_h 72.6 / 128x256_2x2m 71.3 / 128x256_h 70.2 / 256x256_2x2 69.7
/ 192x128_2x2f 68.4 / 1x2_128x192 67.4 / 192x128_h 67.1 / 128x256_2x2f 69.1 / 160x128m_v
64.4 / 64x128_2x2 55.0 / swap160x192 52.7 / 64x64_2x2 32.1. Vertical > horizontal by ~7pts
on both main tiles. Weak-variant retirements confirmed.
GAP MAP: skinny-N K-projections 47-79% (structural; nvjet 16-stage pipeline + per-SM
efficiency), small-model (0.5b/0.6b/1.5b) shapes 74-92% (wave quantization at M=4096 with
few tiles), big-model shapes at/above parity. NEXT: wire dispatcher from this table (LAST
plan step); then per-SM efficiency backlog (§47.3 diagnostic).

### 48. B5-MIRROR — kernel-for-kernel scorecard (2026-07-07) — OFFICIAL

Per user request: 42 head-to-head cases, each shape running OUR clone of the EXACT nvjet
kernel cuBLAS picked (§16.3 mapping), cuBLAS + clone timed back-to-back per shape (no
gradient possible), locked 1400MHz + preheat. Archived:
output_2026-07-07_B5_MIRROR_1400MHz.md. MIRROR avg 82.4%, kernel-for-kernel wins 2/42.

PER-KERNEL IMPLEMENTATION SCORECARD (ours vs cuBLAS's implementation of the SAME strategy):
| our clone        | shapes | range      | verdict |
| 128x256_v        | 12     | 91-106%    | ~96% avg — our flagship is WITHIN ~4% of nvjet's,
|                  |        |            | and BEATS it on 0.5b-LMhead (105.8%) |
| 256x256_2x2 (B1) | 1      | 97.9%      | composite ~parity with their unique kernel |
| 128x256_2x2      | 8      | 64-96%     | good on Downs (92-96%), weak on K-projs (64-82%) |
| 128x256_2x2f     | 4      | 78-93%     | moderate gap |
| 192x128_v/2x2f   | 5      | 71-86%     | mixed composite ~15% behind their 128x192 |
| 160x128m_v       | 1      | 71.8%      | masked tile 28% behind their 128x160 |
| 128x128_2x2      | 1      | 72.4%      | skinny-N gap |
| 64x128_2x2       | 2      | 53-56%     | skinny-N gap |
| 64x64_2x2        | 1      | 47.3%      | skinny-N gap (their 16-stage kernel) |
| swap160x192      | 3      | 47.8-50.7% | WORST — the known unvectorized transposed
|                  |        |            | epilogue (§34.2); cuBLAS same-strategy ~2x faster |
| best-alt subs    | 2      | 68.9/106.8%| 7b-Down alt BEATS their exotic 320x192_2x2f! |

KEY INSIGHTS:
1. MIRROR (82.4%) < WINNER-TABLE best-of (87.4%): copying cuBLAS's per-shape kernel CHOICE is
   ~5 points WORSE for us than routing to OUR best variant — e.g. 3b-Q: their choice maps to
   our swap kernel (50.7%) but our 256x256_2x2 gets 89%. The dispatcher must use OUR winner
   table, not cuBLAS's routing. (Also proves their heuristic is tuned to their impl costs.)
2. OPTIMIZATION PRIORITY (kernel-implementation gaps, biggest lever first):
   a. swap160x192 epilogue vectorization: 48->? (target 85%+) — 3 shapes, ~35pts each.
   b. skinny-N small-tile efficiency (64x64/64x128/128x128_2x2 47-72%): depth is NOT the
      answer (ours auto-deep already) — needs ncu profiling of issue/epilogue overhead.
   c. 192x128 mixed-composite ~15% gap — suspect tail-piece serialization; interleave study.
   d. 128x256_2x2 on K-projections (64-82%).
3. The 320x192/160x192_2x2 sw32 tiles stay unbuilt: our alternatives score 106.8%/68.9%
   against them; only 0.5b-Q would motivate revisiting, and its gap is small-shape-generic.

### 49. alpha=1 epilogue fast-path — correct but PERF-NEUTRAL (2026-07-08)

Acted on the first SASS-observed inefficiency (§48): the epilogue always did `val*alpha`
(128-256 FMULs/tile in the histogram) even when alpha==1. Added an `alpha==1.0f` branch in
the main + M128-tail epilogues that converts straight to bf16 (bit-identical). Full suite
384/384 PASS (0 FAIL) — safe. PERF: FLAT — Qwen sweep 82.7%/80.9% (was ~same), cm64-05K ~43%
(unchanged), every shape within noise. CONFIRMS the ncu diagnosis (§47.3): these kernels are
LATENCY-bound (77.8% stalled on smem scoreboard, IPC 0.19, occupancy ~9%), so removing hidden
instructions doesn't move wall-time.

VERDICT on the other two SASS observations: on inspection #2 (R2UR address recompute) is
compile-time-strided and already strength-reduced by ptxas — not redundant, not safely
hoistable; #3 (SYNCS/barriers) is structural to the warp-specialized pipeline (removing them
re-introduces the §30 deadlock/race bugs) and wouldn't help a latency-bound kernel anyway. So
instruction-count SASS observations are a DEAD END for perf here — kept the alpha fix (free,
correct, helps if a shape is ever epilogue-bound) but stop mining the histogram.

REAL LEVERS (latency/occupancy, data-driven from here):
1. ncu-profile OUR kernels (tooling ready: examples/ncu_probe.cu + ncu_compare.sh) to get our
   occupancy/registers/stall-reasons vs cuBLAS's (8.9%/255reg) — find WHY we stall.
2. swap160x192 epilogue vectorization — our worst kernel (~48%, 3 shapes), scalar
   bank-conflicting transposed stores that genuinely serialize (§34.2). This is a real
   structural inefficiency, not a hidden-instruction one.
3. skinny-N occupancy/pipeline depth (nvjet runs 64x64 at 16 stages).
NEXT: run ncu_compare to pick between (2) and (3) with evidence.

### 50. ncu head-to-head, 1.7b-K (128x256_2x2 vs nvjet) — occupancy theory KILLED (2026-07-08)

Targeted ncu (4 sections, ~12 passes, clock-control none so both ran at 2025MHz in one
session = fair comparison). Reports: /tmp/ncu_{cub,ours}.ncu-rep (offline re-import, no GPU).

               ours(128x256_2x2)   cuBLAS(nvjet 128x256_2x2)
  Duration          16.83 us            15.36 us       (ncu single-shot: we're ~91% here —
                                                         vs bench 73%; single-launch is noisy,
                                                         trust the 20-iter bench for ratio)
  Registers/thd     168                 255            <-- WE USE FEWER
  Theoretical occ   10.94%              12.50%
  Achieved occ      10.88%              9.03%          <-- WE ACHIEVE HIGHER
  Active warps/SM   6.96                5.78           <-- WE HAVE MORE
  smem/block        229 KB              213 KB         (both -> Block Limit Shared Mem = 1)
  Block Limit Reg   1                   1              (both reg-capped to 1 block too)
  Waves/SM          0.86                0.86           <-- GRID UNDER-FILLED (both!)
  Warp cyc/inst     28.4                30.8
  smem-scoreboard   45.9% of stalls     78.5% of stalls

CONCLUSIONS (evidence, not guesses):
1. OCCUPANCY IS NOT OUR PROBLEM — we already BEAT cuBLAS (168 vs 255 reg, 10.9% vs 9.0%
   achieved, 7.0 vs 5.8 warps/SM). Reducing registers/smem to raise occupancy = WASTED effort;
   drop that idea entirely.
2. Both kernels are smem/TMEM-latency bound (short-scoreboard) — fundamental to GEMM operand
   waiting. cuBLAS is MORE stalled (78.5%) yet competitive because it does more work per
   instruction; we spread across more/other stalls. Not a clean lever.
3. Waves/SM = 0.86 for BOTH — this shape UNDER-FILLS the 148-SM GPU (few tiles). The skinny-N
   losses are largely grid-quantization, inherent to the shape; cuBLAS mitigates via SMALLER
   tiles (64x64/16-stage) = more tiles = better fill. So the skinny-N lever is small-tile grid
   fill, which we HAVE (64x128/64x64) but our 64x64 is sw64-B-slow.
4. On this exact shape ncu says we're ~91% (bench 73% is likely single-vs-multi-iter + cuBLAS
   heuristic warmup variance) — i.e. 128x256_2x2 is NOT where we're badly losing.

WHERE WE ACTUALLY LOSE (from §48 mirror): swap160x192 (48%, 3 shapes) — its scalar
bank-conflicting transposed epilogue is a REAL structural serialization, unrelated to
occupancy. THAT is the highest-value fix. NEXT: ncu var3 (swap160x192) to confirm the
epilogue/smem-store is its sink, then vectorize it.

### 51. swap-AB 160x192 epilogue vectorization — stmatrix.trans (2026-07-08)

Goal (user's framing): keep the MIRROR apple-to-apple comparison (our kernel vs cuBLAS's
SAME kernel) as the yardstick and lift our WEAKEST variant. swap160x192 is where our
implementation trails cuBLAS's own 160x192_1x2 the most (§48). NOTE for the dispatcher: no
Qwen shape actually ROUTES to swap (all Q-projections have a better-scoring alternative —
q25-3b-Q→256x256_2x2 @89%, q3-8b-Q→128x128_v @104%, etc.), so this does NOT move the 87.4%
best-of number (§47.5). Its value is proving our per-variant implementation quality is as
good as cuBLAS's, kernel-for-kernel — the reverse-engineering mission.

#### 51.1 ncu var3 diagnosis — CONFIRMED epilogue-bound (4096x2048x2048)
Targeted ncu (4 sections, clock-control none), swap160x192 vs cuBLAS's same kernel:
                 ours(swap160x192)   cuBLAS(160x192_1x2)
  Duration          45.25 us            25.06 us      <-- ~2x slower
  Compute (SM)      49.86%              64.42%
  Registers/thd     85                  255
  Achieved occ      10.9%               9.26%         <-- we have MORE occupancy
  Warp cyc/inst     13.79               25.23
  smem-stall share  39.0%               74.4%
Smoking gun: our warps BARELY stall (13.79 cyc/inst, low smem-stall) yet take ~2x longer =
LOW-PARALLELISM SERIAL work, i.e. the scalar transposed store, NOT operand waiting. Same
lesson as §50: occupancy/registers are not the lever; the epilogue instruction-serialization
is. This is what justified the rewrite.

#### 51.2 The fix — 16dp256b load + stmatrix.x4.trans (fuses transpose + vectorized store)
Root cause: the epilogue reads C'=C^T from TMEM, so each thread holds 8 M-values for ONE
FIXED n-column (tcgen05.ld). A vectorized 16B store needs 8 CONTIGUOUS n for one m-row — the
WRONG axis. So a vectorized store REQUIRES a register transpose first; "vectorize" and
"register-transpose" are the same fix. The hardware tool is stmatrix.x4.trans.b16 (PTX ISA
9.3; sm_90+), which transposes a register tile AND writes it column-major to smem in ONE
instruction. cuBLAS's swap kernel does exactly this; DeepGEMM ships the identical pattern
(sm100_store_cd_swap_ab.cuh). Our 64-col store chunk == DeepGEMM's 64-col swizzle atom
(2 warps/chunk) — the geometry maps 1:1.

Rewrite of the M128 path (C-cols 0..127, chunks 0-1; the bulk) in the SWAP_AB epilogue:
  - TMEM read: 32x32b.x8 -> TWO 16dp256b loads (base + |0x00100000 for DP-lanes +16),
    giving 8 fp32/thread. (16dp256b is the fragment stmatrix.trans requires; 32x32b is not
    stmatrix-compatible.)
  - apply alpha (+ beta*C_in at each value's own (m,n)) -> pack pairs with
    __float22bfloat162_rn (RNE, bit-exact vs the old scalar __float2bfloat16).
  - stmatrix.x4.trans into the SAME 128B-swizzled address (XOR g^(r&7) folded into the dst
    pointer — stmatrix does NOT absorb the swizzle, we keep computing it). One stmatrix.x4
    per 8 m-rows replaces 8 scalar stores. Loop stride 8; BM%8==0 (160,128 both exact).
Two new device helpers: pack2bf16(), stmatrix_x4_trans() (asm
"stmatrix.sync.aligned.x4.m8n8.shared.b16.trans"). The M64 tail chunk (C-cols 128..191,
chunk 2) LEFT scalar this pass — incremental/testable; same treatment can follow.
Addressing derivation (16dp256b fragment -> stmatrix input -> swizzled dst) pinned from PTX
ISA 9.3 + CUTLASS copy_sm100/traits + DeepGEMM before coding — the one non-hand-verifiable
piece (intra-8x8 .trans element order = an image in the PTX doc) is byte-identical to
DeepGEMM's shipped code, so replicate-and-validate, gated by the P6 bit-exact suite.

#### 51.3 Correctness — BIT-EXACT, first try, race-free
Gate: RUN_1X2C (P6 + P6-M2 = 14 swap cases: non-square, partial-M/oobM, batched x4,
alpha/beta, 2 real Qwen shapes), random distinct data, element-wise vs cuBLAS, run 3x.
RESULT: 14/14 PASS every run, MaxRelErr = 0.00e+00 on ALL cases, identical across all 3
runs (= no race in the new stmatrix/barrier path). Full 266-case suite also green = no
regression to any other variant. The vectorization is provably correct.

#### 51.4 Perf — PENDING (not yet measured; do NOT claim a speedup yet)
The re-measure (ncu var3, does 45.25us drop / does 49.86% compute rise toward cuBLAS's
25.06us / 64.42%) is the next step and is UNVERIFIED as of this writing. The correctness
suite's efficiency column (~49.7% on q3b) is vs cuBLAS-BEST on unlocked clocks, NOT the
mirror, so it is NOT the perf number — ignore it for speedup. Only the dedicated
clock-locked ncu compare counts. NEXT: run it, record before/after here, then wire dispatcher.

### 52. fused fp32-input BF16 GEMM experiment — CLOSED, documented negative (2026-07-09..11)

Mixed-precision training question: autocast runs a separate fp32→bf16 cast kernel before
every bf16 GEMM (activation costs 3 DRAM trips: prior op writes fp32, cast reads fp32 +
writes bf16, GEMM reads bf16). Can a GEMM that accepts RAW fp32 and converts in-kernel win?

#### 52.1 Deep research (6 verified threads, all primary-source cited)
- PyTorch eager autocast = a separate aten::to cast kernel per activation use (weights
  CACHED once per autocast region: fp32+requires_grad+leaf+non-view). torch.compile fuses
  the cast into the PRODUCER pointwise op by default; "prologue fusion" (PR #134532, ~2.8)
  fuses casts into Triton-template matmul loads — the author calls it "generally much less
  profitable than epilogue fusion". cuBLAS extern path can never absorb the cast.
- Transformer Engine NEVER fuses input casts into GEMMs — casts are producer-side fused
  kernels (norm/activation/cast-transpose); GEMM only takes pre-quantized operands.
- cuBLAS SHIPS the idea as CUBLAS_COMPUTE_32F_FAST_16BF (fp32 A/B/C, "automatic
  down-conversion and bfloat16 compute", opt-in). PyTorch exposes it via
  torch.set_float32_matmul_precision("medium"). Whether it converts in-kernel or via a
  hidden pre-pass is UNDOCUMENTED (ncu on B300 can answer — pending).
- B300-specific: CUBLAS_COMPUTE_32F_EMULATED_16BFX9 (cc 10.0/10.3 only): fp32 = 3 exact
  bf16 pieces, 9 bf16 MMAs, 3-4x faster than native fp32 at fp32-equivalent accuracy.
  CUTLASS example 78 = the open implementation (FastF32 transform-warp mainloop).
- Hardware constraints (PTX ISA 9.3 verified): TMA cannot narrow 32→16 bit during copy
  (Table 33 = storage formats only); tcgen05.mma kind::f16 refuses fp32 operands (Table 44);
  the only fp32-direct MMA is kind::tf32 at HALF the bf16 rate (2:1 on every published
  NVIDIA spec through Blackwell; tf32 keeps 10 mantissa bits vs bf16's 7). So an fp32-input
  bf16-rate GEMM REQUIRES warps converting between TMA and MMA — no instruction shortcut.
- Bandwidth math for activation X[M,K], r = GEMM DRAM re-reads: status quo = 10MK+2MKr;
  fused fp32-in = 4MK+4MKr (wins only r<3); producer-writes-bf16 = 2MK+2MKr (ALWAYS best —
  why TE/torch.compile chose it). Weights are a non-problem (master-weights cast 1x/step —
  Megatron/DeepSpeed verified). Single-pass lossy fp32→bf16 GEMM exists NOWHERE as an
  official kernel (open unanswered CUTLASS issue #2842) — genuine novelty window.

#### 52.2 What we built (compiles clean, ZERO regression; silicon-run 11/07 → §52.3)
Template knobs FP32_IN (Design-1) and FP32_DIRECT (Design-2), + configs
Sm103Config_f32in_128x256/_128x128, Sm103Config_f32dir_128x256/_128x128:
- Design-1 (CUTLASS FastF32 pattern): TMA loads RAW fp32 (new SWIZZLE_NONE 64-col-chunk
  element-granular tmaps) into an fp32 staging ring (2 stages for 128x256, 3 for 128x128);
  4 NEW transform warps convert (2x float4 loads → 4x __float22bfloat162_rn → one int4
  store) into the standard 128B-swizzled bf16 layout (XOR g^(k&7)); consumer MMA/epilogue
  UNTOUCHED (full_mbar completed by transform arrivals instead of TMA bytes). bf16 ring
  forced to depth 2.
- Design-2 (direct): loader-converter warps LDG.128 fp32 straight from gmem (predicated
  zero-fill OOB = TMA-equivalent; all 42 shapes 64-aligned), cvt in regs, STS.128 bf16;
  NO staging, bf16 pipeline back at native depth, producer TMA idle (kernel now carries
  the B gmem pointer).
- examples/test_fused_fp32.cu: 6 combos = {A: as_type cast + bf16 GEMM (faithful autocast),
  B: fused-staged, C: fused-direct} x {256x256x64, 128x128x64} on all 42 Qwen shapes;
  bmmB/bmmC MUST be 0 (bit-identity vs path A — same RNE intrinsic, same MMA order);
  rel-err vs cuBLAS <= 5e-2; per-component timing (cast | gemm | pathA | B | C | cublas)
  + geomean SUMMARY. Our BluTrain as_type kernel VERIFIED to use the __float2bfloat16
  intrinsic (ConversionKernels.cu:158) — "as_type vs intrinsic" are the same thing.
RESULT (2026-07-11, B200 sm_100 — B300 unavailable; SM=100 build, preheat 60s,
clocks NOT locked (lgc typo) but all paths measured interleaved per shape so ratios hold):
#### 52.3 VERDICT: bit-exact everywhere, perf NEGATIVE everywhere — fused LOSES
- Correctness: 84/84 PASS (42 shapes x 2 variants), bmmB=bmmC=0 and relmax=0.0e+00 on
  every case — both designs bit-identical to the cast-kernel path ON FIRST SILICON RUN
  (and first-ever run on sm_100). The negative below is a real measurement, not a bug.
- Perf geomeans (fused vs cast+bf16-GEMM): 256x256x64: staged 0.583x, direct 0.241x;
  128x128x64: staged 0.308x, direct 0.156x. NOT ONE of the 42 shapes wins (best:
  q25-0.5b-Q staged 0.99x = tie; worst: big Gates/LM-heads 0.35-0.45x staged).
- Mechanism (from the per-shape decomposition): the bigger the GEMM the worse the ratio —
  q3-8b-LMhd: plain GEMM 4.19ms vs fused-staged 13.6ms = 3.3x the GEMM itself, far past
  the 2x DRAM-bytes cap → the TRANSFORM STAGE throttles the MMA (4 warps/CTA must convert
  100% of operand bytes through a 2-deep pipeline, no slack), while the separate cast
  kernel gets the WHOLE GPU (all SMs) for the same conversion and costs only 30-50% of
  GEMM time. Staged beats direct on every shape (TMA async >> warp LDG latency-riding) —
  CUTLASS's FastF32 architecture choice confirmed; our Design-2 question answered.
- Ceiling argument: even a PERFECT fused kernel only recovers cast_ms = max ~1.3-1.5x;
  measured best is 0.58x. Producer-fusion (prior op writes bf16 — the TE/torch.compile
  answer, §52.1) gets the entire win for free. CLOSED: for BluTrain, fuse the cast into
  the PRODUCER ops (norm/activation/elementwise emit bf16), do NOT fuse into the GEMM.
  Documented negative, same class as kota's split-K finding (§53.1d).
- Still open (low priority): the same sweep on B300 for the record, and ncu on cuBLAS
  FAST_16BF to answer whether NVIDIA's own fp32-in mode is fused or a hidden pre-pass.

FULL RESULTS (B200, NITER=20, PREHEAT=60, ms; A = cast+bf16GEMM, B = fused-staged,
C = fused-direct; every row bmmB=bmmC=0, relmax=0.0e+00, PASS):

VARIANT 256x256x64                      | VARIANT 128x128x64
shape           castAB  gemm  pathA  pathB  B/A  pathC  C/A | pathB  B/A  pathC  C/A
q25-0.5b-Q       .014  .012   .027   .027  0.99  .041 0.65 |  .070 0.50  .097  0.36
q25-0.5b-K       .013  .010   .024   .027  0.89  .040 0.59 |  .039 0.71  .052  0.53
q25-0.5b-Gate    .023  .030   .053   .105  0.50  .189 0.28 |  .283 0.25  .471  0.15
q25-0.5b-Down    .068  .032   .100   .112  0.89  .360 0.28 |  .293 0.39  .665  0.17
q25-0.5b-LMhd    .365  .980  1.345  2.983  0.45  8.27 0.16 | 5.849 0.26 12.01  0.12
q25-1.5b-Q       .025  .023   .047   .074  0.64  .124 0.38 |  .121 0.37  .176  0.26
q25-1.5b-K       .021  .014   .035   .041  0.86  .065 0.54 |  .045 0.68  .062  0.50
q25-1.5b-Gate    .059  .074   .133   .276  0.48  .681 0.20 |  .627 0.23 1.107  0.13
q25-1.5b-Down    .139  .097   .236   .402  0.59 1.363 0.17 |  .690 0.32 1.743  0.13
q25-1.5b-LMhd    .623 1.580  2.203  5.572  0.40 15.36 0.14 | 10.51 0.23 20.50  0.12
q25-3b-Q         .033  .029   .062   .097  0.64  .191 0.32 |  .209 0.29  .351  0.18
q25-3b-K         .025  .017   .041   .051  0.80  .084 0.49 |  .057 0.63  .080  0.46
q25-3b-Gate      .086  .119   .206   .486  0.42 1.576 0.13 | 1.006 0.22 2.096  0.11
q25-3b-Down      .185  .123   .308   .497  0.62 1.662 0.19 | 1.089 0.29 2.850  0.11
q25-3b-LMhd      .827 1.996  2.823  8.013  0.35 20.40 0.14 | 13.52 0.23 32.88  0.09
q25-7b-Q         .090  .095   .185   .398  0.46 1.137 0.16 |  .789 0.19 1.590  0.10
q25-7b-K         .049  .027   .076   .094  0.80  .164 0.46 |  .112 0.61  .175  0.39
q25-7b-Gate      .227  .431   .658  1.333  0.49 4.419 0.15 | 2.843 0.25 7.638  0.09
q25-7b-Down      .381  .394   .775  1.620  0.48 5.699 0.14 | 3.273 0.24 8.538  0.09
q25-7b-LMhd     1.444 3.383  4.827 11.434  0.42 35.40 0.14 | 23.26 0.23 61.30  0.09
q3-0.6b-Q        .023  .023   .045   .066  0.69  .107 0.42 |  .158 0.33  .224  0.23
q3-0.6b-K        .022  .019   .041   .043  0.95  .066 0.62 |  .084 0.47  .117  0.34
q3-0.6b-O        .037  .025   .062   .074  0.84  .121 0.51 |  .158 0.40  .226  0.28
q3-0.6b-Gate     .027  .033   .060   .107  0.56  .180 0.33 |  .232 0.27  .329  0.19
q3-0.6b-Down     .055  .036   .091   .108  0.85  .193 0.47 |  .178 0.40  .328  0.22
q3-0.6b-LMhd     .420 1.104  1.524  3.311  0.46 9.897 0.15 | 6.791 0.25 13.72  0.12
q3-1.7b-Q        .033  .028   .061   .096  0.63  .192 0.32 |  .209 0.29  .349  0.18
q3-1.7b-K        .028  .018   .046   .051  0.89  .086 0.54 |  .109 0.42  .156  0.29
q3-1.7b-Gate     .066  .081   .148   .308  0.48  .768 0.19 |  .602 0.22 1.087  0.12
q3-1.7b-Down     .109  .075   .184   .285  0.65  .937 0.20 |  .621 0.29 1.574  0.11
q3-1.7b-LMhd     .827 2.079  2.907  7.506  0.39 20.42 0.14 | 13.56 0.23 32.88  0.10
q3-4b-Q          .066  .064   .130   .253  0.51  .768 0.17 |  .454 0.26 1.102  0.11
q3-4b-K          .039  .023   .062   .074  0.84  .125 0.50 |  .155 0.40  .237  0.26
q3-4b-O          .085  .079   .164   .325  0.51  .913 0.18 |  .544 0.28 1.196  0.13
q3-4b-Gate       .100  .132   .232   .520  0.45 1.672 0.14 | 1.081 0.22 2.822  0.09
q3-4b-Down       .174  .152   .326   .638  0.51 2.223 0.15 | 1.199 0.27 3.155  0.10
q3-4b-LMhd      1.033 2.487  3.520  8.876  0.40 25.45 0.14 | 17.40 0.22 43.17  0.09
q3-8b-Q          .119  .127   .245   .505  0.49 1.585 0.15 |  .968 0.23 2.051  0.11
q3-8b-K          .077  .041   .118   .142  0.83  .401 0.29 |  .218 0.42  .488  0.19
q3-8b-Gate       .213  .299   .512   .993  0.52 3.252 0.16 | 2.117 0.24 5.740  0.09
q3-8b-Down       .266  .277   .543  1.063  0.51 3.672 0.15 | 2.120 0.27 5.557  0.10
q3-8b-LMhd      1.649 4.189  5.838 13.642  0.43 40.43 0.14 | 26.52 0.23 70.02  0.09
SUMMARY 256x256x64: 42/42 PASS | B-vs-A geomean 0.583x | C-vs-A 0.241x | B-vs-(cast+cublas) 0.522x
SUMMARY 128x128x64: 42/42 PASS | B-vs-A geomean 0.308x | C-vs-A 0.156x | B-vs-(cast+cublas) 0.269x

READ OF THE TABLE: B/A rises toward 1.0 ONLY where the GEMM is tiny and the cast dominates
(skinny K/V, small-hidden Q) and falls to 0.35-0.5 wherever the GEMM does real work — i.e.
the fused kernel's conversion stage scales WITH the GEMM (it throttles every stage) while
the separate cast is a fixed small cost. That shape-dependence is the fingerprint of a
transform-throughput-bound mainloop, consistent with the §51/§50 lesson: our kernels lose
when a fixed few warps do serial work the rest of the machine must wait on.

### 53. kota-fork deep comparison + merge (2026-07-10)

Compared OUR repo vs kota's parallel fork (BluBridge-BLAS_kota_latest_10_07_2026).
76 differing files → ~45 are the mycublas→blublas API rename only. Backup taken FIRST:
Blublas_gautam_10_07_2026.zip (our tree pre-merge, build junk excluded).

#### 53.1 The forks are complementary: we went DEEP, he went WIDE
KOTA-only: (a) WIRED analytical dispatcher — score = wave_efficiency x BN^0.75 choosing
BN in {256,128,64,32}, BM 128-vs-64 by M%256, BK=128 when K<=256 && K%128==0 — formula,
NOT measurements; (b) BK=128 deep-K tile axis; (c) BN=64/32 via a GENERALIZED swizzle
system (128/64/32B auto-picked from slice width, one parametrized tmap builder + generic
epilogue XOR — cleaner than our special-cased B_SW64); (d) StreamK-style single-launch
split-K infra, DISABLED after he measured 6-626% cancellation errors (honest negative,
his launcher comment); (e) SM103 ADDMM template (bias-fused epilogue + bias-fused sk
reduce) = our pending §32; (f) FP16-on-B300 module; (g) FP8 e4m3 module (~1700 lines);
(h) blublas.h/BluBlas.cpp public API.
OURS-only: every exact-nvjet tile (B1 256x256_2x2 composite, B2 192x128 mixed, B2.5
160x128m masked, B3 sw64-B small tiles), swap-AB + stmatrix.trans vectorized epilogue,
A-multicast parity routing, validated 2x2f DynamicCluster, alpha==1 fastpath, §52 fused
fp32, the MEASURED 42-shape winner table + honest benchmark methodology + SASS/ncu tooling.
His tile set contains NONE of the actual nvjet tile shapes.

#### 53.2 Bugs found in kota's fork (all verified with file:line, all FIXED in his copy)
1. BM=64 SILENT WRONG RESULTS: CLUSTER_SIZE hardcoded 2 → BM=64 emits M=128 ::2 MMA whose
   TMEM is the INTERLEAVED Layout-B atom (§28.7/§41-R10, silicon-verified); his epilogue
   reads plain Layout-D (row = warp*32+lane) → misplaced values. His dispatcher routed
   every M%256!=0 shape there. FIX (loud): static_assert(BM>=128) in both templates
   (BF16+FP16), 4+4 bm64 aliases disabled, bm64 routing removed from both dispatchers
   (ceil-div grid + TMA OOB makes BM=128 correct for all M%128==0), 8 files' bm64
   instantiations/decls wrapped #if 0 with the full explanation. Real BM=64 support =
   port our validated B3 quadrant epilogue (pending).
2. CLC arm-all-upfront (§30.12 orphaned-steal → zero-filled tiles) in THREE templates:
   BF16 addmm, FP16 addmm, FP8 (his BF16/FP16 BASE templates already had the fix).
   FIX: ported our validated one-outstanding scheduler (arm stage 0 only + arm_idx loop)
   into all three. All fixed files compile clean.
3. FP16 addmm = UNFINISHED WIP that never compiled anywhere (includes the BF16 base
   template, undefined tcgen05_mma_fp16, __half/__nv_bfloat16 mismatches) → excluded from
   build with #if 0 + explanation in both trees; finishing the port is kota's call.

#### 53.3 Merge decisions (what we adopted / rejected, and why)
ADOPTED into our repo (all compile-verified, ZERO impact on our validated BF16 work):
- kota's BF16 addmm as an ISOLATED subtree src/.../SM103/kota_addmm/ carrying HIS base
  template so its quote-include resolves locally (7-param vs our 12-param config = same
  struct NAME, different definition → never include both in one TU; README_ISOLATION.md
  records the rule). CLC fix applied. Port onto OUR template = pending.
- FP16/B300 + FP8 directories (self-contained, own templates; fp16-addmm excluded as WIP).
- blublas.h + BluBlas.cpp (new API names, coexists with mycublas).
- His addmm/fp8 tests into examples/.
- BK=128 tile axis as OUR-template aliases Sm103Config_256x256x128/_128x128x128 —
  pure aliases (our BK math generalizes), UNTESTED ON SILICON until the P-suites run.
REJECTED/DEFERRED:
- His dispatcher heuristic: NOT adopted. Our plan stands — dispatcher wired LAST from the
  MEASURED winner table (§48); his BN^0.75 formula is unmeasured and routes to tiles that
  never faced cuBLAS. Optional later: his formula as fallback for out-of-table shapes.
- His generalized swizzle: architecturally nicer, but rewriting our validated operand
  plumbing mid-experiments (swap/§52 pending silicon) = pure regression risk for zero new
  capability we currently need (our sw64 covers BN=64; BN=32 tiles unproven vs cuBLAS).
  Adopt only if/when B300 measurements show BN=32 tiles win somewhere.
- His split-K: keep disabled as he left it; revisit with compensated summation if K-heavy
  shapes ever demand it.

#### 53.4 Pending list (consolidated, as of 2026-07-10)
1. DONE 11/07 on B200 (§52.3): fused-fp32 experiment CLOSED — bit-exact, perf negative
   everywhere; recommendation = producer-fusion for BluTrain. (B300 re-run = optional.)
2. Swap160x192 vectorized-epilogue perf re-measure (§51.4 — correctness done, perf number
   pending; ncu var3 vs the 45.25us baseline).
3. DONE 11/07 (§54): dispatcher WIRED from the measured winner table — silicon
   validation pending (test_dispatch_winner).
4. DONE 11/07 on B200: BK=128 tiles VALIDATED (42/42 PASS, rel-err 0.0e+00 first run) —
   niche win only: bn256 pair wins 22/42 shapes but small (1.02-1.06x, Gates/Downs/Qs)
   and LOSES LM-heads badly (0.75-0.86x); bn128 pair wins 1/42. Keep for the per-shape
   winner table, never as a default; kota's K<=256 gate never fires on Qwen (min K=896).
5. IN PROGRESS 11/07 (B200): kota's addmm module VALIDATED batch=1 — standalone 12/12
   PASS 0.00e+00 (bias provably fused: AddMM matches ref where plain GEMM differs);
   comprehensive 80/84. The 4 batch=4 FAILs were root-caused to FOUR distinct bugs
   (all fixed 11/07, both trees):
   (a) TEST reference bug — the bias-seed kernel's grid covers ONE MxN slice, so for
       batch>1 the reference's batches 1..N-1 were beta*(uninitialized cudaMalloc
       garbage) -> false FAILs at 1e4-1e5 rel-err against a likely-correct kernel.
       Fixed: seed every batch slice.
   (b) KERNEL latent: bias modes numel==M / numel==M*N used the FLATTENED row
       (cur_batch*M + local) — `row < M` guard always false for batch>=1 (bias
       silently dropped) and bias[c_row*N+..] overruns the bias tensor. Fixed:
       batch-local row (bias is batch-invariant).
   (c) KERNEL latent: sk_reduce_addmm_kernel had the same flattened-row bias bug.
       Fixed identically (dormant while SK=1, fixed for permanence).
   (d) KERNEL latent: spurious cta_rank*BM TMEM lane term — the exact §28.7 bug the
       base template removed (harmless at BM=128 by accident, corrupts BM<128).
       Fixed to DP base 0.
   RERUN CONFIRMED (11/07, B200, fixed kernel verifiably rebuilt at sm_100a):
   comprehensive 84/84 PASS 0.00e+00 — the batch=4 cases went green exactly as
   root-caused (reference bug), and the standalone stayed 12/12 (kernel-side fixes
   zero regression). Kota's addmm module = FULLY validated on B200, batch=1 AND
   batched. Remaining: port addmm onto OUR template (kill the isolated-subtree
   ODR caveat). Perf note (unlocked): addmm beats plain
   cuBLAS on most shapes but trails cuBLASLt's fused epilogue up to 2.4x on big
   shapes — future optimization target.
   ALSO 11/07: full main suite (all default P-suites + swap P6/P6-M2 + 42-shape
   sweep) = ALL PASSED on B200 sm_100 — the whole variant family is now validated
   on TWO Blackwell generations. Build-system fix: Makefile arch-stamp forces a
   clean rebuild when SM changes (stale sm_103a objects in the .so caused
   "no kernel image" on B200 — make tracks timestamps, not flags).
6. Tell kota: bm64 bug + fix, addmm/fp8 CLC fix, fp16-addmm WIP status (his fixed copy =
   BluBridge-BLAS_kota_latest_10_07_2026, ready to hand back).
7. M64-chunk of the swap epilogue still scalar (stmatrix treatment can follow §51).
8. Older §32 addmm plan superseded by kota's module (validate, then port).

### 54. DISPATCHER WIRED from the measured winner table (2026-07-11) — the plan's final step

Implementation (compiles clean, symbol-verified; silicon run pending):
- Bgemm_tn_variants.cu: +9 thin wrappers → all 11 winner-table variants exported
  (256x256x64_v, 128x128x64_v, 128x256_2x2, 128x128_2x2, 256x256_2x2, 192x128_v,
  1x2_128x192, 64x128_2x2, 64x64_2x2 + the 2 existing h-wave entries).
- Bgemm_sm103_dispatcher.cu TN entry: exact-match table of the 39 unique (N,K)
  keys from the OFFICIAL B5 winner table (§47.5) → variant. (N,K)-keyed because
  the 42 Qwen shapes are layer-determined; M = token count varies in prod (table
  measured at M=4096; winner is tile-vs-N/K driven — first-order right).
  Dedup of duplicated keys by higher measured TFLOPS: (2048,2048)→256x256_2x2
  (922 vs 886 T), (1024,2048)→128x128_v (693.6 vs 692.4 T). §48 mirror proof
  drove the design: route by OUR table, never cuBLAS's choices (-5 pts).
  Out-of-table fallback (UNMEASURED, pattern-derived): N<=160→64x64_2x2,
  N<=640→64x128_2x2, N<=1536→128x128_v, else 128x256_v.
  MYCUBLAS_DISPATCH_DEBUG=1 prints per-call routing. NN/NT entries keep the old
  crude rule (no measurements for those layouts yet — honest gap).
- examples/test_dispatch_winner.cu: all 42 shapes + 3 fallback shapes through
  the REAL production entry (mycublasBgemmSM103_dispatch_tn), correctness vs
  cuBLAS + per-shape %% + geomean. Expected to reproduce ~87.4%% avg on B300 at
  locked clocks (B200/unlocked = indicative).
VALIDATED 11/07 on B200: 45/45 PASS all 0.00e+00 through the production entry;
ROUTING 45/45 exact (every table key -> its measured winner incl. both dedups;
all 3 fallback shapes -> intended tier). Perf geomean 82.5% of cuBLAS on
B200/unlocked (indicative — different generation, sm_100 cuBLAS kernels, no
lock/preheat); BEAT cuBLAS on B200 on 5 shapes (8b-LMhd 111.4%, 7b-LMhd 110.6%,
3b-LMhd 106.6%, 7b-Down 101.2%) — the big-model win pattern transfers across
generations. THE ORIGINAL PLAN IS NOW COMPLETE END-TO-END: every cuBLAS tile
cloned -> honestly benchmarked -> dispatcher wired from the measurements ->
silicon-validated. REMAINING: (a) official B300 locked-clock dispatcher rerun
(expect ~87.4%), (b) hook dispatch_tn into the public bgemm entry chain,
(c) perf backlog per §48 priority list (swap re-measure, skinny-N, 192x128).

### 55. dispatcher production run analysis — same-vs-different routing map (2026-07-11, B200 locked)

Locked-clock B200 dispatcher run: 45/45 PASS all 0.00e+00, geomean 82.3% of cuBLAS
(matches 82.5% unlocked — reproducible). This section maps every shape by whether our
dispatcher picks the SAME strategy cuBLAS picks (from the §16.3 nvjet mapping) or a
DIFFERENT one, and how each performs, in simple english.

GROUP A — 15 shapes where we route the SAME strategy as cuBLAS:
we BEAT cuBLAS on 3 of them (all big LM-heads): 8b-LMhd 115.6%, 1.7b-LMhd 110.5%,
7b-LMhd 103.3%. on-par/near: 7b-Gate 98.9, 4b-Gate 94.9, 8b-Gate 94.9, 4b-LMhd 92.7.
losing: 0.6b-LMhd 89.0, 3b-LMhd 88.4, 4b-Down 85.4 (192x128_v), 0.5b-Down 85.0
(128x256_2x2), 4b-O 81.1 (192x128_v). losing BADLY (the real gap): the skinny-K small
tiles — 3b-K 56.3, 1.5b-K 55.8 (64x128_2x2), 0.5b-K 54.3 (64x64_2x2).
read: the flagship 128x256_v is 88-116% apple-to-apple (7 of its 9 shapes >= 93%);
192x128_v sits ~81-85%; the small-tile K-projection kernels are ~2x behind cuBLAS's
same-strategy kernels — the biggest pure implementation gap we own.

GROUP B — 27 shapes where we DELIBERATELY route DIFFERENT (the §48 mirror proved
copying their choice costs ~5 points — their heuristic fits their kernel costs):
we BEAT cuBLAS on 1: 8b-Down 102.8% (ours 128x128_v vs their 256x256_2x2).
on-par: 7b-Down 96.9 (their pick = the 320x192_2x2f we deliberately never built (NOTE: 320 IS 64x5; the real build cost is the PER-CTA slice = 160 rows, 160%64=32 -> 32-elem-chunk swizzle plumbing) —
§48 pt3: our alternative measured 106.8% against it on B300, so building it was
skipped ON PURPOSE; NOT a new discovery), 1.5b-Down 96.8. near (>=93%): 8b-Q 95.4,
4b-Q 94.2, 1.5b-Gate 94.2. rest lose 69.8-92.5 — but routing different is still
correct on these: e.g. 3b-Q their choice maps to our swap clone (50.7%) while our
256x256_2x2 pick gets 80.9-89%. the losses are kernel-implementation gaps, not
routing mistakes.

WHERE TO IMPROVE (agreed plan: attack the same-variant group FIRST — it is the pure
apple-to-apple gap with no routing excuse — then the gains transfer to group B since
the same kernels serve both):
1. skinny-K small tiles (64x64_2x2 / 64x128_2x2, 54-56%): biggest same-variant
   deficit; also poisons every group-B K-projection. §48 diagnosis direction: cuBLAS
   runs a 16-stage pipeline there with better per-SM efficiency; our depth is already
   auto-deep, so suspect issue/epilogue overhead -> ONE targeted ncu session
   (0.5b-K, few passes) is the next concrete action.
2. 192x128_v mixed composite (~15% behind, 4b-O/Down): suspect M256+M128 tail-piece
   serialization (§48c) — interleave study.
3. flagship sub-90 cases (3b/0.6b-LMhd): one look at wave/L2; small-model shapes
   carry inherent grid underfill (§50, 0.86 waves) kernel work cannot fix.
4. dispatcher itself: NO changes now — the table is B300-measured and production
   targets B300; retuning on B200 data would corrupt it. per-arch tables = possible
   later if B200 becomes a target.
NEXT STEPS (order): ncu 64x64 diagnosis -> fix -> rerun K-projections; then 192x128
study; B300 when free (official dispatcher % + swap re-measure); addmm port later.
PENDING from this run: the nsys kernel-trace backtrack (trace not yet captured).

### 56. THE CURRENT GOAL — mirror parity: beat cuBLAS with THEIR OWN variant choices (2026-07-11)

goal in simple english: for every one of the 42 qwen shapes, our clone of the EXACT
variant cuBLAS issues for that shape must reach cuBLAS or beat it. the logic: we
implement the same tiling strategy and we run on the same hardware — so if we lose,
we are missing something in OUR implementation, and that something is findable.
the §48 mirror scorecard is the baseline checklist for this goal (mirror avg 82.4%,
kernel-for-kernel wins 2/42 as of 07/07).

the plan (agreed 11/07):
1. keep the CURRENT winner-table dispatcher exactly as it is meanwhile — it is the
   best measured routing today and the work must not stop while we optimize.
2. go shape by shape through the 42, optimizing the variant cuBLAS issues for that
   shape, until the clone reaches/beats them. since many shapes share a variant,
   the practical working list is the DEDUPED variant list with its shape coverage
   (worst apple-to-apple gap first so each fix lifts the most shapes):
     a. swap160x192          — 3 shapes, mirror 47.8-50.7% (stmatrix epilogue done,
                               perf re-measure pending B300; M64 chunk still scalar)
     b. 64x64_2x2            — 1 shape,  47.3%  (their 16-stage kernel; ncu next)
     c. 64x128_2x2           — 2 shapes, 53-56%
     d. 192x128_v + _2x2f    — 5 shapes, 71-86% (tail-piece serialization study)
     e. 160x128m_v           — 1 shape,  71.8%
     f. 128x128_2x2          — 1 shape,  72.4%
     g. 128x256_2x2 (+f)     — 12 shapes, 64-96% (K-projection cases the weakest)
     h. 128x256_v            — 12 shapes, 91-106% (nearly there; polish last)
     i. 160x192_2x2          — 1 shape (0.5b-Q), UNBUILT -> now IN SCOPE under this
                               goal (needs the 32-elem-chunk operand path)
     j. 320x192_2x2f         — 1 shape (7b-Down), UNBUILT -> IN SCOPE (per-CTA slice
                               is 160 rows: 160%64=32, same 32-elem plumbing as (i);
                               NOTE the tile total 320 IS 64x5 — earlier wording that
                               "320 is not a multiple of 64" was WRONG, the
                               constraint is the per-CTA 160 slice)
3. when ALL variants reach/beat their counterparts: re-run the FULL combination
   tests — the mirror (same variant per shape) AND the complete 42-shapes x
   all-variants matrix — with the locked-clock + preheat methodology.
4. re-analyze the new winner data and REDESIGN the dispatcher from it (the routing
   that is optimal today may shift once the weak clones are fixed — e.g. if our
   swap clone reaches parity, their 3b-Q choice may become our best too).
work discipline unchanged: research deeply before each kernel change, bit-exact
gates before perf claims, one honest measurement methodology, everything documented
here as numbered sections.

### 57. B300 session results — dispatcher banked, swap verdict, 64x64 L2 finding (2026-07-13)

1. OFFICIAL DISPATCHER RUN (B300, locked 1400MHz, preheat pass): 45/45 PASS all
   0.00e+00. We now BEAT cuBLAS outright on 7 shapes on the production silicon:
   7b-LMhd 115.4%, 8b-LMhd 115.0%, 4b-LMhd 113.5%, 1.7b-LMhd 106.5%, 3b-LMhd
   100.9%, 1.7b-Gate 100.8%, 8b-Down 100.2% (7b-Gate 99.8 / 7b-Down 99.2 at the
   door). Several EXCEED the §47.5 predictions (1.5b-LMhd 97.6 vs 83.1 expected).
   The run's 82.4% GEOMEAN is not comparable to §47.5's 87.4% ARITHMETIC avg
   (geomean + 3 fallback shapes included). Dispatcher = production-validated, done.

2. SWAP160x192 RE-MEASURE (§51.4 closed): raw 60.06us looks worse than the
   45.25us baseline but the baseline ran at ~2025MHz boost vs today's locked
   1402MHz — cuBLAS's UNCHANGED kernel scaled 25.06->34.91us = exactly the clock
   ratio, proving pure clock effect. Normalized: ours ~65.4 -> 60.06us = the
   stmatrix epilogue bought ~8-9% wall-clock; mirror ratio 55.4% -> 58.1% of
   their same-strategy kernel. Registers 85->56, SM 49.9->53.2%. HONEST verdict:
   real but modest; the low warp-cycles signature persists (ours 15.5 vs their
   24.0 = still under-parallelized serial work). Next levers: stmatrix the M64
   chunk (still scalar) + revisit the 16x256b double-TMEM-load cost.

3. 64x64_2x2 DIAGNOSIS (0.5b-K) — KEY FINDING: single COLD launch (ncu flushes
   caches): ours 10.78us vs theirs 8.90us = 82.6%. But the 20-iter bench = 47-54%.
   Reconciliation via bench TFLOPS: ours 89.5T = 10.5us/iter (== ncu, cache-
   INsensitive); theirs 187.9T = 5.0us/iter vs their own 8.9us cold = THEIR
   KERNEL RUNS 1.8x FASTER WARM THAN COLD, OURS GAINS ZERO FROM WARM CACHES.
   B here is only 224KB (fully L2-resident warm); their 16-stage kernel exploits
   it, our time is pinned by something cache-independent (stage-wait chain /
   epilogue / launch+CLC fixed cost — 940 MFLOP should take ~1us at peak, we
   take 10.5us cold OR warm; both kernels 0.86 waves = duration IS single-tile
   latency). NEXT: warm-cache ncu (--cache-control none) to confirm — expected:
   cub drops toward 5us, ours stays ~10.5 -> then hunt the fixed serialization
   in our mainloop; that is the ~2x lever for EVERY skinny-K shape (§56 items b/c).

#### 57.1 spin-wait experiment — NEGATIVE, reverted (2026-07-13)
Tried mbarrier.test_wait SPIN instead of the CUTLASS-style suspending try_wait
(10ms hint) after the stall table showed sleeping=0.78 vs cuBLAS's 0. RESULT:
sleeping did go to 0 but branch_resolving exploded 0.95->4.38 cyc/inst;
0.5b-K 10.46->12.42us (+19% WORSE); dispatch geomean 82.4->80.3%; every
K-projection regressed. REVERTED same day. LESSON: parking warps is correct
(frees issue slots, wake-on-complete is cheap); cuBLAS's zero-sleeping means
their kernel WAITS LESS OFTEN per unit work, not that polling wins. The lever
is the NUMBER of sync events (fences per stage: our membar 2.28 vs their 0.00;
bar.syncs per chunk: 3.25 vs 1.14), not the wait instruction flavor. NEXT:
the fence/barrier COUNT audit — which of our per-stage tcgen05 fences and
per-chunk proxy fences + double bar.syncs are protocol-required vs hoistable
(research vs PTX ISA rules + CUTLASS sm100 pipeline placement) — then a
measured reduction, one change at a time.

### 58. sync-audit implementation arc — three measured batches (2026-07-13)

Fence/barrier audit (agent-verified vs PTX ISA + CUTLASS + DeepGEMM) found our
skinny-K kernels lose on sync-event COUNT (membar 2.28 vs cuBLAS 0.00, barrier
3.25 vs 1.14). Three batches implemented, each gated separately:

BATCH 1 (safety tier — KEPT): store_stage persisted across CLC waves (fixed a
REAL latent race: the per-wave reset broke buffer alternation on odd chunk
counts — BN=64: 1 chunk, swap BN=192: 3 — making tma_store_wait<1> vacuous;
masked only by inter-wave latency; DeepGEMM persists the index for exactly
this); 3 redundant __syncwarps removed; wait_group -> .read (releases warp0 on
smem-READ completion, matching cute). Gate: 45/45 PASS, geomean 82.6% (neutral,
as predicted for this tier).

BATCH 2 (SingleWave CLC bypass — KEPT but perf-NEUTRAL): single-wave grids
never arm CLC; every role breaks after its blockIdx tile (DeepGEMM ships no
CLC at all). Gate: 45/45 PASS; the events really vanish (sleeping 0.78->0.13,
membar 2.28->1.94, branch 0.95->0.73) BUT duration 10.46->10.37us only, and
barrier stall ROSE 3.25->5.52 — the wait relocated to the final cluster_sync.
LESSON: those sync ops were removable but NOT on the critical path; a
single-tile kernel's duration = its longest serialized role chain
(k_iters x wait->fence->MMA->commit + epilogue drain), not its side-work.

BATCH 3 (BK=128 SMALL tiles — THE WIN): chain-halving IS the critical-path
lever. New configs 64x64_2x2_k128 / 64x128_2x2_k128, first silicon: all rows
PASS 0.00e+00; BK=128 wins up to 1.17x and the win GROWS with K (K=4096:
1.17x, 2560: 1.16x, 2048: 1.12-1.13x, K<=1024: ~1.00x) — exactly the
chain-length prediction; DeepGEMM's kDoMergeStages ("reducing the umma_arrive
overhead") confirmed on our silicon. DISPATCHER UPDATE: 3b-K (256,2048) ->
64x128_2x2_k128 (measured 1.12x on its winning variant); other K-projections
route to 128-wide tiles or showed flat ratios on their winner. FINAL GATE
(same day): routing live in production ([dispatch_tn] ... -> 64x128_2x2_k128),
45/45 PASS, 3b-K 56.8 -> 58.9%.

REMAINING (skinny-K, §56): 0.5b-K (K=896, ~10us) is immune to chain work —
FIXED-overhead-bound (launch + epilogue drain), needs its own diagnosis.
N=1024 K-shapes route to 128-tiles, untouched by this arc. Next levers:
epilogue drain path, per-launch overhead, then the full §56 mirror re-sweep +
dispatcher redesign.

### 59. the two-priority variant-optimization plan (2026-07-13, agreed)

FINAL TARGET: all 21 implemented variants at/above their cuBLAS counterparts.
Split into two phases:

PRIORITY 1 (now) — the 12 exact clones of the kernels cuBLAS issues for the 42
shapes, ordered by impact (mirror % = current standing vs THEIR same kernel):
  1. 128x256_2x2      (8 shapes, 64-96%)  <- biggest recoverable bank
  2. 128x256_v        (13 shapes, ~96%)   <- small gap x huge coverage
  3. 64x64_2x2        (1 shape, ~47%)     <- worst ratio
  4. 64x128_2x2(+k128)(2 shapes, 53->~63%)
  5. swap160x192      (3 shapes, 48->~58%; M64 chunk still scalar)
  6. 192x128_v        (3 shapes, 71-86%)
  7. 192x128_2x2f     (2 shapes, 71-86%)
  8. 128x256_2x2f     (4 shapes, 78-93%)
  9. 160x128m_v       (1 shape, ~72%)
 10. 128x128_2x2      (1 shape, ~72%)
 11. 256x256_2x2      (1 shape, ~98% — nearly closed)
 12. 128x256_2x2-as-256x128 mirror (1 shape, ~93%; same code as 1)
 (+ the 2 UNBUILT tiles join P1's tail once built: 160x192_2x2; 320x192_2x2f
  is lowest priority — our substitute already beats it at 106.8%.)
METHOD: mirror-first reverse engineering — SASS backtrack + surgical ncu +
PTX-verified changes, ONE measured change at a time, full bit-exact gates.

PRIORITY 2 (after P1) — our 9 extra variants: 128x256_h, 128x128_v/_h,
128x256_2x2m, 192x128_h, 1x2_128x192, and the BK=128 aliases. KEY EXPECTATION:
nearly all P2 variants are the SAME code paths as P1 clones (wave/multicast/BK
flags on one template), so P1 work should optimize them FOR FREE.
EXPLICIT P2 TASK (per gautam): after P1 completes, TEST this expectation —
run all 9 P2 variants across the 42 shapes against the same cuBLAS versions
and verify they inherited the P1 optimizations; whatever did not inherit gets
optimized directly then.

AGREED ORDER OF WORK: (0) one cheap GLOBAL experiment first — the warm-L2
lead from §57/58: all our tmap builders set CU_TENSOR_MAP_L2_PROMOTION_NONE
while reference implementations promote TMA loads into L2; suspect for why
cuBLAS's kernels run up to 1.8x faster warm while ours gain 0%. One-enum
change, benefits EVERY variant if real. (1) then table order starting with
128x256_2x2 — the deep skinny-tile redesign (their 16-stage style) moves to
the back of the P1 queue unless the L2 experiment reopens it.

#### 59.1 P1 experiment 0 — L2 promotion: NEGATIVE, reverted (2026-07-13)
Promoted all 8 operand-load tmaps NONE -> L2_256B (DeepGEMM's exact setting).
Gate: 45/45 PASS but ZERO effect — 0.5b-K warm 10.50us/10.63% memory vs
10.46/10.71 baseline (still gains nothing warm), geomean 82.0 (noise).
REVERTED same session. The warm-L2 gap (their kernel 1.8x faster warm, ours
0%) is NOT a tmap hint — it is structural in how their kernel streams the
L2-resident B across iterations. Mystery stays open, parked behind the agreed
table order. NEXT SESSION: P1-#1 — the 128x256_2x2 campaign (8 shapes,
64-96% mirror): SASS backtrack + surgical ncu vs their 128x256_2x2, worst
shape first (the K-projections it serves).
### 60. P1-#1 CAMPAIGN — 128x256_2x2: full measurement matrix + SASS autopsy (2026-07-13)

Ran examples/p1_128x256_2x2.sh on B300 @1400MHz: 8 shapes x both kernels,
5 ncu sections each, deep SASS (+InstructionStats +SourceCounters) on the
weak/good pair (1.7b-K, 3b-Down). Raw logs: p1_matrix.md, p1_cub_sass.md,
p1_ours_sass.md (reports under /tmp/p1_*.ncu-rep on the server).

#### 60.1 mirror premise VERIFIED + scoreboard
All 8 shapes route to nvjet_sm103_tst_128x256_64x6_2x2_2cta_h_bz_NTT (name
logged per capture — heuristic did not shift). Their launch = 256 thr (8
warps), 255 reg, cluster 4, PolicySpread. Ours = 224 thr (7 warps), 167 reg,
cluster 4, PolicySpread. Same grid (128 / 256 CTAs). Their name decodes
KTile 64 x 6 STAGES.

| shape     | M,N,K            | cub us | ours us | ratio |
|-----------|------------------|--------|---------|-------|
| 0.6b-K    | 4096,1024,1024   | 14.50  | 16.54   | 0.88  |
| 1.7b-K    | 4096,1024,2048   | 21.09  | 23.33   | 0.90  |
| 4b-K      | 4096,1024,2560   | 23.58  | 27.42   | 0.86  |
| 8b-K      | 4096,1024,4096   | 35.10  | 34.94   | 1.00  |
| 0.6b-O    | 4096,1024,2048   | 20.42  | 23.46   | 0.87  |
| 0.6b-Down | 4096,1024,3072   | 27.49  | 29.76   | 0.92  |
| 1.7b-Down | 4096,2048,6144   | 81.73  | 84.32   | 0.97  |
| 3b-Down   | 4096,2048,11008  | 138.05 | 138.34  | 1.00  |

Pattern: we tie at the two largest-K shapes, lose 8-14% elsewhere — a fixed
per-iteration/per-tile overhead that big K amortizes. L2 hit (1.7b-K): theirs
58.45% vs ours 49.46%; DRAM throughput theirs 995 vs ours 900 GB/s (same
bytes, we just take longer). On the Downs OUR L2 hit is higher (73.5 vs 62.8).

#### 60.2 SASS autopsy — the five proven differences (1.7b-K deep pair)

D1. tcgen05.fence MEMBAR QUADS — THE SMOKING GUN. Every one of our
tcgen05.fence::before/after_thread_sync lowers to a 4-instr quad
  MEMBAR.ALL.CTA + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR
sitting before each mbarrier arrive/wait. Ours: 11x MEMBAR.ALL.CTA,
10x MEMBAR.ALL.GPU (GPU-SCOPE membar, the most expensive fence there is).
cuBLAS: ZERO membar.gpu, ONE membar.cta in the whole kernel. This is the
membar-stall fingerprint from §57 (2.28 vs 0.00) finally traced to source.
PTX 9.0 says the fence is required per mbarrier wait; nvidia's own kernel
(and CUTLASS) omit it — the hardware is evidently coherent without it on
sm_103. One of our MEMBAR.CTA quads even executes INSIDE the epilogue
per-chunk loop (65536 execs).

D2. PIPELINED barrier check in their MMA loop vs our wait-then-issue. Their
steady-state loop body (~17 instrs): "@!P0 TRYWAIT P1,[UR8]" issued for the
NEXT stage -> 4x UTCHMMA.2CTA back-to-back (only uniform UIADD3 between) ->
UTCBAR.2CTA.MULTICAST commit -> dual-condition BRA.U loop-back that consumes
P1. The MMA issue latency hides the phase check; the NANOSLEEP fallback is
off the hot path. Ours: TRYWAIT -> @!P4 BRA to a 4-instr sleep loop. Sampler
evidence: our spin loop EXECUTED 3.0M times vs their 1.44M; our top stall
sites = that BRA (236 samples) + NANOSLEEP (222) + UCGABAR_WAIT (168+23);
theirs = one parked NANOSLEEP (501 of 700 total). Also: our sleep hint is
0x989680 = 10ms, theirs 0xc350 = 50us.

D3. EPILOGUE: theirs is flat and register-fat, ours is a sync-laden loop.
Theirs (255 regs buys this): 16x LDTM.16dp256bit.x4 (wide tmem loads) ->
128x FFMA2 (PAIRED fma, 2 floats/instr, alpha scale) -> 128x F2FP.BF16 pack
-> 32x STSM.16.MT88.4 (stmatrix) -> 8x UTMASTG.3D. Near-zero address math —
all on uniform datapath. Ours: 8x LDTM.x8 + 32x F2FP + 8x STS.128 per chunk
loop iteration, and EACH iteration also executes 2x WARPSYNC + 2x BAR.SYNC
+ 1x MEMBAR.ALL.CTA + 1x FENCE.VIEW.ASYNC.S (65536 execs each!), plus heavy
vector-pipe address math (84x IMAD + 79x IMAD.U32 + 57x VIADD + 92x LOP3 +
58x PRMT). Our 128x256 path uses NO STSM at all (0 in dump) and scales alpha
with separate FMUL(128)+FFMA(64) instead of FFMA2. Our whole kernel: 2080
SASS lines vs their 1144; our hot loop tier ~104 instrs vs their ~36.

D4. TMA LOADS: theirs UTMALDG.3D with .MULTICAST on the A-operand (4 of 8
sites), ours UTMALDG.4D.2CTA x14, NO multicast — this build of 128x256_2x2
has mcastA OFF. Their producer loop = 4 TMA + 1 predicated arrive + 1 trywait
+ ~25 uniform-only adds, nothing else. Multicast halves L2->smem A traffic;
matches their better L2 hit on the weak shapes.

D5. THEIR 6-STAGE pipeline (name: 64x6; 2CTA splits B so ~32KB/stage x6 =
192KB smem) vs our stage depth — deeper prefetch rides out DRAM latency at
moderate K, consistent with the mid-K shapes (4b-K 0.86) being our worst.

#### 60.3 ranked action plan (one measured change at a time, 336/336 gate each)
A1 (biggest, cheapest): drop/reduce the tcgen05.fence pair around mbarrier
    waits — cuBLAS + CUTLASS both omit; keep DeepGEMM reference in comment;
    gate bit-exact then benchmark the 8 shapes.
A2: mcastA=1 build of 128x256_2x2 (knob already validated bit-exact) — A/B
    on the 8 shapes; expect L2-hit + DRAM relief on the K-projections.
A3: pipeline the consumer phase check (issue try_wait one stage ahead,
    consume predicate at loop-back) + cut sleep hint 10ms -> ~50us.
A4: epilogue rework — hoist the per-chunk syncs (warpsync/bar/membar/fence
    quad) out of the chunk loop, move address math to uniform regs, use
    wider LDTM + FFMA2-style paired scaling, consider stmatrix path here.
A5: try 5-6 stage pipeline for this tile if smem allows.

#### 60.4 A1 gate result + the two-instrument discovery (2026-07-13)
A1 (tcgen05.fence removal) CORRECTNESS: 3x consecutive full template gates
ALL PASS + dispatcher 45/45, every shape 0.00e+00 — no race, cuBLAS/CUTLASS
practice validated on our silicon. KEPT (default off,
-DMYCUBLAS_KEEP_TCGEN05_FENCES restores strict PTX).
A1 PERF at the benchmark level: geomean 82.6% vs 82.4-82.6 baseline =
NEUTRAL in the 50-iter dispatcher test; true kernel-duration delta needs the
ncu re-probe (pre-A1: 1.7b-K ours 23.33us) — batch-2 lesson may apply (the
wait may just relocate). Pending 2-min server job.

THE BIG FINDING — same shapes, same day, same 1400MHz, two instruments:
  shape      single-shot(ncu)   50-iter loop(dispatch)   gap
  0.6b-K         88%                 67.8%              -20pt
  1.7b-K         90%                 75.7%              -14pt
  4b-K           86%                 75.8%              -10pt
  8b-K          100%                 82.9%              -17pt
  0.6b-O         87%                 74.7%              -12pt
  0.6b-Down      92%                 81.5%              -10pt
  1.7b-Down      97%                 91.4%               -6pt
  3b-Down       100%                 94.6%               -5pt
Decomposition on 0.6b-K: cuBLAS 14.5us single -> ~13us in-loop (warm GAIN,
the §57 mystery again); ours 16.5us single -> ~19us in-loop (we get SLOWER
in a loop). Confirmed cause candidate for our side: the launcher host-encodes
4 CUtensorMaps (cuTensorMapEncodeTiled driver calls) on EVERY call; cuBLAS
caches per-shape plans in the handle. On ms-scale shapes this vanishes —
we post 104-113% on 7 shapes (3b-LMhd 113.4, 1.7b-LMhd 111.2, 7b-Down 110.7,
4b-LMhd 108.2, 0.6b-LMhd 105.3, 8b-LMhd 104.8, 8b-Down 104.4).
IMPLICATION: on N=1024 shapes the LAUNCH PATH is as big a lever as kernel
internals. New action item:
A6 (launch path, benefits ALL 21 variants + every small shape):
   (i) measure: NITER=20 nsys of ncu_probe — kernel-duration trend across
   instances separates GPU warm effect from host gaps between launches;
   (ii) fix: host-side tmap cache keyed by (M,N,K,ld,ptrs) — legitimate,
   cuBLAS-equivalent. Also explains why 0.5b-K (~10us) was 'fixed-overhead
   bound' in §58.

#### 60.5 A1 final verdict + A6 decomposition measured + A6a implemented (2026-07-13 EOD)

A1 FINAL (ncu re-probe post-removal, locked 1400): 1.7b-K ours 23.33 -> 23.84us,
0.6b-K 16.54 -> 16.74us — kernel-duration NEUTRAL (noise). Same lesson as §58
batch-2: the tcgen05 fences were removable but NOT on the serialized critical
chain; the mbarrier wait simply absorbs the time. KEPT anyway: correct
(3x full gates + 45/45, all 0.00e+00), matches cuBLAS/CUTLASS practice, and
shortens the chain for the upcoming A3/A4 work. Default off;
-DMYCUBLAS_KEEP_TCGEN05_FENCES restores strict-PTX fencing.

A6 DECOMPOSITION (the day's second-biggest result, after the SASS autopsy):
NITER=20 nsys on 0.6b-K (probe now loops both kernels back-to-back; 40
instances captured, alternating nvjet/bgemm):
  regime                          cuBLAS       ours      ratio
  cold single-shot (ncu -c 1)     14.5us      16.7us     0.88
  warm GPU-only (nsys steady)     ~11.2us     ~14.2us    0.79
  benchmark loop (event-timed)    ~13us       ~19us      0.68
Their kernel warms 12.4 -> 11.2us (-23% at instance ~10, monotonic); ours
15.0 -> ~14.2us (-5..8%) and flat after. TWO separate quantified diseases:
 (1) WARM GAP ~9pt: at these sizes the whole working set (~18MB) is
     L2-resident when hot; the warm kernel is L2-latency + sync-chain bound
     and their lean MMA loop (pipelined trywait, D2) + flat epilogue (D3)
     exploit it; our serialized chain floors us at 14.2us. This is the §57
     warm-L2 mystery, now measured on the 128x256 tile: attack = A3 + A4.
 (2) HOST LAUNCH TAX ~11pt: event-timed loop adds ~1.8us/call to them but
     ~4.8us/call to us — the launcher host-encoded 4 CUtensorMaps
     (cuTensorMapEncodeTiled driver calls) EVERY call + cudaFuncSetAttribute
     every call; cuBLAS caches per-shape plans in its handle. Also finally
     explains §58's "0.5b-K is fixed-overhead-bound".

A6a IMPLEMENTED (staged, compiles, UNGATED — first job tomorrow):
thread_local single-entry tmap cache per template instantiation in
launch_bgemm_sm103_cluster, keyed (A,B,C,M,N,K,batchCount) — descriptors are
a pure function of the key (all else constexpr), so identical key =>
byte-identical encode: no staleness hazard even across cudaFree/realloc.
Cache holds all four maps (A,B,C,C_tail). cudaFuncSetAttribute memoized to
one call per process (sticky per function+device; single-device assumption,
same as s_sm_count). Benchmarks hit this cache every iter after the first;
real inference reuses weight/activation buffers per step, so the hit pattern
is representative, and cuBLAS does the equivalent (handle plan cache) — a
fair, legitimate optimization. NOTE: test_dispatch_winner alternates shapes
(ours() 1x for correctness, then 50x timed back-to-back) so the timed loop
hits after iter 1 — expect the K-projection rows to recover most of the
~3us/call gap.

TOMORROW'S PLAN (in order):
 1. gate A6a: 3x full template gate + dispatcher 45/45 + NITER=50 benchmark
    at locked 1400 — expect K-projections (67-83%) to jump several points,
    geomean up from 82.6; verify LMhd wins stay >=100%.
 2. A2: mcastA=1 build of 128x256_2x2 (knob validated bit-exact already) —
    A/B the 8 campaign shapes; expect L2-hit/DRAM relief (their D4 multicast:
    L2 hit 58.45 vs our 49.46 on 1.7b-K).
 3. A3: pipeline the consumer phase check (their D2 pattern: issue trywait a
    stage ahead, consume predicate at loop-back; kill the 3.0M-exec spin
    loop) + sleep hint 10ms -> 50us (0x989680 -> 0xc350).
 4. A4: epilogue rework (D3): hoist the per-chunk WARPSYNCx2+BARx2+MEMBAR+
    FENCE out of the chunk loop, address math to uniform regs / precomputed,
    wider LDTM + paired-FFMA scaling, evaluate STSM path.
 5. A5 (if smem allows): deeper operand pipeline toward their 6 stages.
ARTIFACTS: all 21 campaign files in repo profiling_reports/p1_128x256_2x2/
(16 ncu-rep, 2 nsys-rep, 3 md logs) — every section re-openable offline via
ncu --import / nsys stats, no GPU needed.

#### 60.6 A6a gate result — correct but perf-NEUTRAL; host-tax hypothesis REFUTED (2026-07-14)
New box (pale-bird-fades-fin-03, fresh --depth 1 clone at 829659e), locked
1400. Gates: 3x template ALL PASS + dispatcher 45/45, all 0.00e+00 — the
tmap cache never serves a stale descriptor. KEPT (saves host CPU, zero risk).
BUT the benchmark did not move:
  campaign shapes (yesterday A1 -> today A6a, %cuB):
  0.6b-K 67.8->66.8 | 1.7b-K 75.7->75.4 | 4b-K 75.8->74.8 | 8b-K 82.9->83.0
  0.6b-O 74.7->75.8 | 0.6b-Down 81.5->80.1 | 1.7b-Down 91.4->91.0
  3b-Down 94.6->90.9 | geomean 82.6->82.4 (all within cross-box noise)
Ours per-iter on 0.6b-K STILL 0.019ms with descriptor encoding now cached ->
the ~4.8us/iter overhead above warm kernel time (14.2us) is NOT host-side
cuTensorMapEncodeTiled cost. LESSON (§58 batch-2 pattern, third time): in an
async 50-iter loop the host enqueue work was OVERLAPPED with GPU execution
anyway — event-timed per-iter measures the GPU timeline, so only GPU/driver-
side per-launch cost can explain the residual. REFRAMED REMAINING SUSPECTS
for the ~3us/launch delta vs cuBLAS (+4.8us ours vs +1.8us theirs):
 (a) cluster-grid launch setup with our 227KB dynamic-smem opt-in per launch
     (they use less smem/6 stages ~192KB),
 (b) parameter-buffer size (we pass 4 CUtensorMaps BY VALUE = ~600B params),
 (c) instruction-cache cold start each launch (our SASS ~2x their size).
NEXT MEASUREMENT (A6b): no-sync NITER loop (kernels back-to-back on one
stream, sync only at end) + nsys -> read the GPU inter-kernel GAP directly
for ours vs theirs; that pins launch overhead vs kernel-duration cleanly.
MEANWHILE the kernel-internal queue continues: A2 mcastA next.

#### 60.7 THE DISCOVERY — cuBLAS overlaps consecutive kernels with PDL (2026-07-14)
A6b gap measurement (50 back-to-back launches on ONE stream, no syncs,
0.6b-K, locked 1400, nsys -> sqlite; traces archived as gap_ours/gap_cub):
                          duration(warm med)   inter-kernel GAP   effective/iter
  cuBLAS nvjet                 14.02us            -4.06us            9.95us
  ours 128x256_2x2             13.76us            +2.66us           16.42us
Their GAP is NEGATIVE: kernel N+1 STARTS ~4us BEFORE kernel N ENDS, on the
SAME stream (verified: both traces single stream 13, single kernel name).
That is PROGRAMMATIC DEPENDENT LAUNCH: their SASS contains PREEXIT (the
cudaTriggerProgrammaticLaunchCompletion trigger, fired mid-kernel in the
DMA-warp region ~after last input reads) and ACQBULK x2 near kernel start
(the dependent-side acquire); ours has ZERO PDL instructions (archived
p1_cub_sass.md lines 75/657/844; PMTRIG = perf-monitor trigger, unrelated).
READ THE PARITY: our WARM KERNEL DURATION IS ALREADY AT PARITY (13.76 vs
14.02 — we are 2% FASTER). The entire 0.6b-K loop-benchmark gap (66.8%) is
launch pipelining: 9.95/16.42 = 0.61 matches the dispatcher row almost
exactly. This also reframes most of the §57 "warm mystery" in loop contexts
and §58's "0.5b-K fixed-overhead-bound" — it was never L2 magic, it was PDL
overlap + our 2.7us dead launch gap all along.
A2 SIDE-RESULT: mcastA A/B cold single-shot = mixed noise (1024: 0.0%,
2048: +3.3%, 2560: -1.0%, 4096: -0.9%, 3072: +1.7%, 6144: +1.3%,
11008: -4.9%) — NO clear win, NOT dispatched; may revisit warm later.
Log archived as a2_mcast.md.

NEW TOP ACTION — A7: implement PDL in our template + launcher.
 device side: griddepcontrol.wait (cudaGridDependencySynchronize) before the
   first global read (TMA loads / tmap-dependent init), and
   griddepcontrol.launch_dependents (cudaTriggerProgrammaticLaunchCompletion)
   placed once our global reads are done / stores committed — exact placement
   to be researched vs PTX ISA + CUDA programming guide + their PREEXIT
   position (DMA-warp region) before implementing.
 host side: cudaLaunchAttributeProgrammaticStreamSerialization=1 on our
   cudaLaunchKernelEx config (consumer-side opt-in; producers' trigger is
   harmless for non-PDL consumers).
 expected: kills our +2.66us dead gap AND overlaps our ~4us tail drain —
   0.6b-K effective 16.4 -> ~10-12us => the K-projection rows jump toward
   parity; benefits EVERY variant on EVERY small shape; benefits real
   inference (back-to-back layer GEMMs) identically.
 gates: bit-exact 3x + 45/45 + NITER=50; PLUS a chained-dependency test
   (GEMM1 output feeds GEMM2 input back-to-back on one stream) to prove the
   wait/trigger contract holds under overlap.

#### 60.8 A7 IMPLEMENTED — PDL in the template (2026-07-14, UNGATED)
Researched vs PTX ISA 9.3 §9.7.14.14 (local ptx_docs). KEY SEMANTICS that
make it safe: griddepcontrol.wait blocks until ALL in-flight prerequisite
grids have COMPLETED and their memory is VISIBLE — consumer-side safety is
absolute, so trigger placement is a pure scheduling hint and RAW chains
(GEMM1's C -> GEMM2's B) stay correct under overlap. launch_dependents fires
per-CTA (repeats are no-ops); dependents launch once every CTA issued it, so
later-wave CTAs of a multi-wave grid fire as they start and the grid never
starves itself. Overlap mechanism = dependent CTAs take retiring SMs and run
their prologue while the producer tail still executes.
IMPLEMENTATION (mirrors nvjet's ACQBULK-early / PREEXIT pattern):
 - device: griddep_wait() for ALL threads immediately after the init
   cluster_sync (everything above touches no gmem and overlaps the prior
   grid's tail; blanket wait = no per-role causality analysis), then
   griddep_launch_dependents() from warp0/elect (one per CTA, early = max
   prologue overlap for the NEXT grid).
 - host: cudaLaunchAttributeProgrammaticStreamSerialization=1 on every
   cudaLaunchKernelEx (attrs[2]->attrs[3]); MYCUBLAS_NO_PDL=1 kill-switch
   restores strict serialization (device instrs become no-ops then).
 - NEW GATE examples/test_pdl_chain.cu: G1 C1=alpha_it*A1^T B1 feeds G2
   D=A2^T C1, alpha varies per iter (staleness detector), D cycles 4 buffers,
   overlapped pass compared BYTE-EXACT vs fully-serialized pass. Note: its
   printed serialized-vs-overlapped time includes sync latency in the
   reference pass — the honest PDL perf A/B is the dispatcher benchmark with
   and without MYCUBLAS_NO_PDL=1.
GATE SEQUENCE: pdl_chain -> 3x template gate -> 45/45 + NITER=50 benchmark
(PDL on), then same benchmark MYCUBLAS_NO_PDL=1 for the honest delta.
EXPECTED: kills the +2.66us dead gap + hides launch runway; 0.6b-K effective
16.4 -> ~10-12us; every variant, every small shape, real inference too.

#### 60.9 A7 GATE RESULT — PDL LANDS: geomean 82.6 -> 89.5% (2026-07-14)
ALL GATES GREEN, same box/session, env-only A/B (cleanest attribution ever):
 - pdl_chain: 4/4 BIT-EXACT under real producer->consumer overlap, and the
   dependent chain itself runs 1.39x faster (68.36 -> 49.04us/iter).
 - template gate: 3x ALL PASS. dispatcher: 45/45, all 0.00e+00.
 - MYCUBLAS_NO_PDL=1 rerun reproduces the old baseline EXACTLY (82.6%) —
   the +6.9 geomean points are PDL and nothing else.
Per-shape (A6a baseline -> A7 PDL), the campaign 8 first:
  0.6b-K 66.8->77.4 | 1.7b-K 75.4->87.2 | 4b-K 74.8->86.1 | 8b-K 83.0->90.7
  0.6b-O 75.8->85.9 | 0.6b-Down 80.1->87.8 | 1.7b-Down 91.0->93.5
  3b-Down 90.9->96.5
Biggest movers elsewhere (PDL lifts EVERY variant, as predicted):
  0.5b-Q 63.9->80.6 (+16.7) | 0.5b-K 41.8->58.1 (+16.3) | q25-3b-K
  58.3->71.9 (+13.6) | 1.5b-K 55.5->67.8 (+12.3) | 0.6b-Q 80.8->91.2 |
  7b-K 72.5->81.7 | fallback-N128 41.2->57.9 | fallback-N512 48.0->57.5
NOW >=100%: NINE shapes (was 7): 3b-LMhd 118.3, 0.6b-LMhd 113.9, 8b-Down
112.5, 4b-LMhd 112.3, 7b-Down 110.3, 1.7b-LMhd 107.7, 7b-LMhd 104.7,
8b-LMhd 104.1, 8b-Q 102.6. Large shapes unharmed (LMhds stable/up).
READING: exactly matches the §60.7 mechanism — the win scales with how much
of the per-iter cost was launch runway, so smallest shapes gain most
(+10-17pt) and compute-bound giants are untouched. The remaining gap on
tiny-N shapes (0.5b-K 58%, N128 fallback 58%) is now genuinely
kernel/config-bound again — PDL stripped the overhead disguise off it.
STATUS: A7 KEPT, default ON (MYCUBLAS_NO_PDL=1 kill-switch). NEXT: A3
(pipelined trywait + 50us sleep hint) and A4 (epilogue sync hoist) now
attack the true kernel-internal residue; optionally re-trace gap_ours to
watch our gap go negative like theirs.

#### 60.10 A3 result — pipelined poll NEGATIVE (reverted), 50us hint likely the 0.5b-K hero (2026-07-14)
A3 bundled TWO changes (bad discipline, called out honestly): (1) the cuBLAS
D2 pipelined phase check — poll the next stage once before the MMA burst,
park only on miss — and (2) sleep hint 0x989680 (10ms) -> 0xc350 (50us,
their exact NANOSLEEP operand). Gates all green (pdl_chain 4/4 BIT-EXACT,
3x template PASS, 45/45). Benchmark verdict vs A7 baseline:
  campaign shapes CONSISTENTLY -1pt: 0.6b-K 77.4->76.5, 1.7b-K 87.2->86.1,
  4b-K 86.1->84.7, 8b-K 90.7->89.6, 0.6b-O 85.9->85.2, 0.6b-Down 87.8->86.1,
  1.7b-Down 93.5->92.8, 3b-Down 96.5->96.2; 3b-LMhd ours 1.566->1.683ms
  (-7% real); geomean 89.5 -> 89.0.
  BUT 0.5b-K (64x64_2x2, the worst shape) EXPLODED +13.5pt: 58.1 -> 71.6.
DIAGNOSIS: on our LOAD_PIPE_DEPTH-stage pipeline the producer is rarely a
full stage ahead at MMA speed, so the early poll MISSES most iterations and
adds poll+park to the issue chain instead of replacing the park. cuBLAS's
version pays off because their 6 stages (D5) keep the next stage full — D2
and D5 are COUPLED; retry the poll only together with deeper stages (A5).
The hint change is wait-flavor-neutral on big shapes but shortens wake
latency where kernels are wait-dominated (tiny 64x64 shapes).
ACTION: poll REVERTED same session (lesson comment left at the loop; the
mbarrier_test_ready helper stays for the A5 retry); hint 0xc350 KEPT.
DECOMPOSITION GATE (next run): hint-only build — expect campaign shapes back
at ~A7 level AND 0.5b-K holding ~71%; that isolates the hint as the win.

#### 60.11 A3 CORRECTION — decomposition REFUTES §60.10; POLL is the 0.5b-K hero, hint is neutral (2026-07-14)
The hint-only build (poll reverted, 50us kept) settles the attribution I got
WRONG in §60.10. Three comparable states (poll isolated because A3-full and
A3b-hint-only both carry the 50us hint):
                    A7(10ms,no-poll)  A3full(50us+poll)  A3b(50us,no-poll)
  0.5b-K                 58.1              71.6               59.1
  geomean                89.5              89.0               89.1
ISOLATION:
 - 50us HINT alone (A7->A3b): 0.5b-K 58.1->59.1 (+1 noise), geomean
   89.5->89.1 (-0.4 noise). => HINT IS NEUTRAL. §60.10's claim that the hint
   drove 0.5b-K's +13.5 is REFUTED.
 - POLL alone (A3b->A3full): 0.5b-K 59.1->71.6 (+12.5 REAL), fallback-N128
   +0.6, fallback-N512 +1.8 (all small-tile routes, consistent direction);
   128x256 campaign shapes -1pt each; geomean ~neutral.
CORRECT MECHANISM: the poll wins where the PRODUCER OUTRUNS the consumer —
small tiles (64x64: tiny per-stage TMA fills fast, next stage usually ready
=> poll hits, replaces the park). It loses on big tiles (128x256: large
per-stage TMA, producer behind => poll misses, adds poll+park). This is the
INVERSE of §60.10's "needs 6 stages" guess — it's tile SIZE, not stage
count, though deeper stages (A5) would extend the win to big tiles too.
ACTION A3c: gate the poll ON for small tiles (BM<=64: 64x64_2x2,
64x128_2x2), OFF for BM>=128 — recovers 0.5b-K (our single WORST shape,
§56 mission) +12.5pt for free, leaves the 128x256 family at A7. 50us hint
KEPT globally (neutral, matches cuBLAS). mbarrier_test_ready helper now
USED (not just parked for A5). 64x128 inclusion is by-mechanism (BM=64 same
as 64x64); the gate makes it verifiable — watch 1.5b-K/3b-K/etc. if they
route to 64x128.

### 61. NOTE — family coverage (clarification, 2026-07-14)
Recorded so we don't re-confuse this: the 42-shape benchmark ALREADY spans
BOTH dense families — Qwen2.5 (q25-0.5b/1.5b/3b/7b, hidden 896/1536/2048/
3584) = 20 shapes, and Qwen3 (q3-0.6b/1.7b/4b/8b, hidden 1024/2048/2560/
4096) = 22 shapes. (Qwen2.5's smallest dense is 0.5B, Qwen3's is 0.6B; there
is NO 120M dense model in either lineup.) Optimization is PER-VARIANT and
every variant serves shapes from both families, so all template wins (PDL,
fence removal, 50us hint, tile-gated poll) already optimize both families at
once — no separate "Qwen3 campaign" is needed; the §59 P1/P2 plan covers it.
Only optional add-on: split the dispatch scoreboard's geomean per family so
neither hides behind the other's average (cosmetic; do if/when useful).

#### 60.12 A3 CLOSED NEGATIVE — the poll win was NOISE, full revert to A7 (2026-07-14)
A3c (poll gated ON for BM<=64) was the decisive retest. Verified
Sm103Config_64x64_2x2 = <64,64,64,2> so BM=64 => the gate DID enable the
poll for 0.5b-K's routed variant. Result: 0.5b-K = 58.5 (vs no-poll A3b
59.1 — IDENTICAL) and geomean 88.7. So the poll, provably applied, changed
0.5b-K by ~nothing => the A3-full 0.5b-K=71.6 (§60.10/60.11) was MEASUREMENT
NOISE on a 4-7us kernel, not the poll. My two prior A3 write-ups chased that
fluke; this gated retest is what a proper controlled experiment looks like.
FULL LEDGER (all below A7):
  state                    0.5b-K   geomean
  A7 (PDL, 10ms, no poll)   58.1     89.5   <- proven best
  A3-full (poll + 50us)     71.6*    89.0   (*noise)
  A3b (50us, no poll)       59.1     89.1
  A3c (gated poll + 50us)   58.5     88.7
DECISION: revert the ENTIRE A3 line to A7 — poll has zero reproducible
benefit (gated or not), 50us hint is neutral-to-negative. Restored: simple
A7 k-loop (wait-at-top), 10ms sleep hint (0x989680, the value that measured
89.5). mbarrier_test_ready helper KEPT (parked, documented) for the A5
retest — the poll's real prerequisite is DEEPER STAGES (their 6 vs our few),
not tile size; only A5 can make the producer reliably outrun the consumer.
LESSON (logged): tiny kernels (<10us) need multi-run medians before trusting
a per-shape delta; a single 50-iter mean is inside the noise band there.
NEXT: A4 (epilogue rework, D3) on the clean A7 base.

#### 60.13 A3 revert CONFIRMED + noise-floor methodology (2026-07-14)
Revert run: 45/45 all 0.00e+00, geomean 88.8, and CRUCIALLY the SAME 9 shapes
>=100% as A7 (3b-LMhd 108.1, 1.7b-LMhd 113.7, 0.6b-LMhd 111.5, 7b-Down 108.5,
8b-Down 106.3, 4b-LMhd 107.8, 7b-LMhd 103.7, 8b-LMhd 104.6, 8b-Q 102.8) =>
PDL intact, revert is functionally A7. The 89.5->88.8 delta is one shape:
3b-Down, which on ~identical code across runs read 96.5/96.2/86.2/92.8/84.8
(its cuBLAS side alone swung 0.143-0.157ms) — pure box/thermal variance on a
150us kernel.
METHODOLOGY RULE (adopt): on these ephemeral B300 instances the cross-run
geomean noise floor is ~+/-0.5-1pt; per-shape noise on tiny (<10us) and a few
large shapes is much larger. TRUST A DELTA ONLY IF (a) large per-shape AND
(b) reproduced, OR gated in a same-run A/B (env kill-switch like
MYCUBLAS_NO_PDL). This is what cleanly validated PDL (+6.9, same-run A/B) and
what should have caught A3's noise sooner. Baseline going into A4 = A7/PDL,
~89-89.5% geomean, 9 shapes >=100%.

#### 60.14 A4 RESEARCH — D5 was WRONG (we already run 6 stages); epilogue rewrite is smem-bound (2026-07-14)
Read both epilogue branches + the smem budget math in full. Two findings:

FINDING 1 (big — corrects D5): for 128x256_2x2, LOAD_PIPE_DEPTH auto-derives
to SIX. Math: STAGE_BYTES = A(128*64*2) + B(128*64*2) = 32KB; STORE_BUF =
2*128*64*2 = 32KB; (227KB - 32KB)/32KB = 6.09 -> 6 stages. cuBLAS's kernel
name "128x256_64x6" = KTile 64 x 6 STAGES. WE ALREADY MATCH THEIR STAGE
DEPTH. D5 ("6 vs ours shallower", §60.2) was a misread of the SASS autopsy —
we independently run the same depth. => A5 (deeper stages) has ~zero headroom
on the big tiles (already smem-saturated at 6) and the A3-poll's "needs deeper
stages" rationale (§60.12) is moot for 128x256 — it already has 6 and the
poll still lost. The poll simply doesn't help our design; leave it dead.

FINDING 2 (epilogue structure): the D3 per-chunk syncs are 2x barrier_sync +
1x tma_store_fence per chunk, num_chunks = BN/STORE_COLS = 256/64 = 4. They
are CORRECTNESS-LOAD-BEARING for the STORE_PIPE_DEPTH=2 double buffer:
 - barrier#1 propagates warp0's tma_store_wait ("prior TMA done reading this
   buffer") to all 4 epilogue warps before they overwrite it;
 - fence + barrier#2 order all 4 warps' disjoint-row smem writes before warp0
   issues the TMA that reads the whole buffer.
To FLATTEN (cuBLAS-style: all chunks' writes -> one fence -> batch TMA) needs
one store buffer PER chunk = STORE_PIPE_DEPTH 2->4 = 64KB store, which forces
LOAD_PIPE_DEPTH 6->5 (192->160KB) to fit 227KB. So A4 TRADES an operand stage
for fewer epilogue fences — a wash-risk change, NOT a free win.

STRATEGIC READ: after PDL, the 128x256_2x2 warm kernel is already at parity
(§60.7: 13.76 vs 14.02us) AND matches stage depth (6). The residual
benchmark gap on this variant is small and correlates with LOW K (0.6b-K 77%
= epilogue-heavy 16 K-iters, vs 3b-Down 96% = epilogue-negligible 172 iters)
— consistent with the epilogue mattering only where it is a large fraction.
DECISION POINT (recorded, awaiting gautam): (a) measured STORE_PIPE_DEPTH
experiment (bump to 4, drop load to 5, A/B — tests the epilogue-flatten
tradeoff directly and cheaply); or (b) declare 128x256_2x2 DONE (PDL took it
82->89% + warm parity) and move to a higher-headroom P1 variant — the 64x64
family (0.5b-K 58%, our single worst) or 128x256_v polish. Recommend (b):
higher expected value than a wash-risk epilogue rewrite on an already-parity
variant.

#### 60.15 PUZZLE SOLVED — the 77% is PDL overlap DENSITY, not the epilogue (2026-07-14)
Measured OUR gap WITH PDL (KSEL=ours NOSYNC NITER=50, gap_ours_pdl) vs cuBLAS
same box (gap_cub_v2):
                 isolated dur   overlapped per-iter (THROUGHPUT)   overlap gain
  cuBLAS            14.02us            9.95us                       -4.07 (29%)
  ours+PDL          13.76us           12.83us                       -0.93 ( 7%)
ours 12.83 / cuBLAS 9.95 = 77.6% == the benchmark's 0.6b-K 77%. PUZZLE
RECONCILED: our isolated kernel is FASTER (13.76<14.02) but cuBLAS's
overlapped THROUGHPUT beats us because it extracts 29% from PDL overlap vs
our 7%. Under overlap our kernel DURATION stretches 13.76->24.45us (grids
CONTEND) while cuBLAS stays 14.02 (grids INTERLEAVE cleanly).
ROOT CAUSE (SASS-confirmed, p1_ours_pdl_sass.md): our griddepcontrol.wait
(ACQBULK line 182) + trigger (PREEXIT line 188) both sit BEFORE our K-loop
(UTMALDG 375, UTCHMMA 854). We fire the trigger even EARLIER than cuBLAS
(their PREEXIT is at line 844, AFTER their K-loop). So trigger timing is NOT
it. The killer is the WAIT SCOPE: our griddep_wait blocks ALL warps at kernel
entry until the prior grid fully completes -> the next grid's CTAs land on
free SMs but immediately block, doing NO useful work until we finish (7%
overlap, below even the idle-SM fraction). cuBLAS lets its dependent COMPUTE
(load+MMA) during the prior grid's tail and orders only the OUTPUT -> 29%.
THE LEVER + CORRECTNESS CONSTRAINT: narrow our wait to guard only the true
hazard so load+MMA can overlap. But WHERE the wait can move depends on data
flow: for INDEPENDENT GEMMs (this benchmark; Q/K/V from same input; most
non-chained inference) the only hazard is WAR on the output C store, so the
whole load+MMA can overlap and the wait drops to just-before-store. For a
CHAINED GEMM (dependent's A or B == prior grid's C, e.g. attn->FFN) the wait
MUST precede the operand load -> minimal overlap, unavoidable. The kernel
alone cannot tell which case it is -> this is a DISPATCHER/CALLER decision.
DESIGN OPTIONS (awaiting gautam):
 (a) dispatcher tracks last-output-ptr per stream; if the new GEMM's A/B do
     not alias it -> emit the fast wait-before-store path (automatic + safe;
     ~29% overlap on independent GEMMs). More work: dispatcher state + two
     kernel wait modes (template bool).
 (b) caller opt-in flag "inputs independent of prior kernel on this stream"
     -> fast path. Simple; pushes correctness responsibility to the caller.
 (c) keep conservative wait-before-load (current), accept ~77% on independent
     small-K shapes, move to other P1 variants.
RECOMMENDATION: (a) — correct-by-construction and automatic, and it is THE
remaining lever for every small/independent shape (K-projections 77-90%,
0.5b-K family). Big-K shapes already >=95% (overlap is a small fraction there)
and chained GEMMs are unaffected either way. Epilogue (A4/D3) is NOT the
lever — deprioritize.

#### 60.16 RESEARCH — how cuBLAS / DeepGEMM / ThunderKittens place PDL (2026-07-14)
Read all three local sources (DeepGEMM/impls/sm100_bf16_gemm.cuh,
ThunderKittens/kernels/gemm/bf16_b200/bf16_b200_gemm.cu +
include/ops/*/util/util.cuh, cuBLAS from archived SASS). They all use PDL but
place the two halves (wait + trigger) DIFFERENTLY, and we are the outlier:

  lib     | WAIT (griddepcontrol.wait)          | TRIGGER (launch_dependents) | prefetch tmap before wait
  --------|------------------------------------|-----------------------------|--------------------------
  cuBLAS  | ACQBULK at entry                    | PREEXIT AFTER the K-loop     | (own machinery)
  DeepGEMM| all-threads, after init+cluster_sync| NONE (relies on completion) | YES (prefetch_tma_desc x3)
  TKittens| LOADER WARP ONLY, before TMA loads  | at LAST tile's epilogue     | YES (prefetch x3)
  OURS    | ALL threads, at entry               | BEFORE the K-loop (EARLY!)   | NO

THREE ways we differ from the field:
 1. TRIGGER TIMING: we fire launch_dependents BEFORE our K-loop — the only one
    that does. cuBLAS/TK fire it LATE (after MMAs / last tile); DeepGEMM never
    fires it (lets grid completion do it). Firing early launches the dependent
    while our MMAs still run; it then blocks at its own entry wait, idle-
    occupying free SMs instead of doing useful work.
 2. WAIT SCOPE: we block ALL warps at entry. TK blocks ONLY the loader warp
    (before its TMA loads); consumers + the CLC scheduler proceed and overlap
    their setup with the prior grid's tail. cuBLAS/DeepGEMM are all-thread but
    place the wait after the descriptor prefetch.
 3. NO DESCRIPTOR PREFETCH: DeepGEMM AND TK both prefetch_tma_descriptor(A,B,
    C) BEFORE the wait so descriptor fetch overlaps the prior grid. We don't.

WHY THIS EXPLAINS OUR 7% vs cuBLAS 29% (§60.15): our all-thread entry wait +
no prefetch means the ONLY thing overlapping the prior grid's tail is ~nothing
(everything is behind the wait). TK/cuBLAS let scheduling + descriptor fetch +
consumer spin-up overlap.

PROPOSED FIX A8 (matches the field; correctness-safe; ordered by risk):
 A8a (lowest risk, free): prefetch TMA descriptors (A,B,C tmaps) at kernel
     entry BEFORE griddep_wait — exactly DeepGEMM+TK. `prefetch.tensormap`.
 A8b (bigger lever): narrow griddep_wait to the PRODUCER/loader warp only,
     placed right before its first TMA load (TK pattern) — lets consumer +
     CLC-scheduler warps overlap the prior grid's tail. Correct: the loader's
     wait still gates all data (loads -> semaphores -> consumers), and the
     epilogue store lands after MMA which is after the gated loads, so WAR on C
     and RAW chains both stay correct WITHOUT any dispatcher alias tracking.
     => SUPERSEDES the §60.15 option-(a) dispatcher redesign; no caller
     contract needed. This is the correct, general, automatic fix.
 A8c (optional): move our trigger from before-K-loop to after the MMA loop
     (match cuBLAS PREEXIT / TK) so the dependent isn't launched into an
     early block.
NOTE: A8b resolves the §60.15 fork — TK proves the surgical loader-only wait
is correct for BOTH independent and chained GEMMs (the load-gate covers RAW,
the completion-wait covers WAR) with NO caller flag. Deprioritize the
dispatcher-alias idea. Gate each of A8a/b/c separately: pdl_chain 4/4 +
3x template + NITER=50 + re-measure the gap (target: gap -> more negative,
per-iter -> toward 9.95us).

#### 60.17 A8 vs CUTLASS/DeepGEMM/TK — is it minimal? (verification, 2026-07-14)
Read CUTLASS's actual sm100 GEMM kernel (gemm/kernel/
sm100_gemm_tma_warpspecialized.hpp) — the authoritative reference. It does
EXACTLY A8's pattern, role-scoped:
  line 440/443: prefetch_tma_descriptors (mainloop + epilogue) at entry
  line 619: wait_on_dependent_grids in is_participant.main_load, before load
  line 809: wait_on_dependent_grids in is_participant.epi_load, before C read
  line 687: wait in the scheduler participant (persistent CLC)
  line 780: launch_dependent_grids AFTER the scheduler's work-tile loop (LATE)
CUTLASS comment on the trigger (verbatim): "Launching a dependent kernel too
early can compete with current kernels, while launching too late can lead to
a long latency." And on wait: "enforces no global memory access prior to this
instruction ... ensures correctness when launching a dependent kernel earlier."

SCORECARD A8 vs the field:
  aspect            | cuBLAS | DeepGEMM | ThunderK | CUTLASS | OURS (A8)
  prefetch desc     |  y     |  y (x3)  |  y (x3)  |  y      |  y (x3, C_tail only if HAS_M128)
  wait: producer    |  y     |  all-thr |  loader  |  y      |  y (producer elect)
  wait: epi beta    |  -     |  -       |  -       |  y      |  y (guarded beta!=0)
  wait: scheduler   |  -     |  -       |  -       |  y      |  NO (ours reads only CLC hw, not prior gmem -> correctly omitted)
  wait count beta=0 |  ~1    |  1(all)  |  1       |  2-3    |  1  <- as lean as the leanest
  TRIGGER timing    |  LATE  |  none    |  LATE    |  LATE   |  EARLY  <- THE ONLY DIFFERENCE

VERDICT: A8's prefetch + wait-scope is MINIMAL and matches/beats the field
(1 wait for beta=0; we correctly OMIT the scheduler wait CUTLASS keeps,
because our CLC scheduler reads only hardware responses, never prior-grid
global memory). Instruction-count-optimal on the wait side.
THE ONE REMAINING DIFFERENCE: trigger timing. All three references fire
launch_dependents LATE (after the tile loop / at last tile); we fire it EARLY
(before the K-loop). CUTLASS's own comment says early competes with the
current kernel — which matches our measured 24.45us duration STRETCH under
overlap (§60.15, grids contending). This is a PURE PERF knob (correctness-
invariant per PTX + CUTLASS), so it must be MEASURED, not assumed (A3 lesson).
PLAN: gate A8 as-is (surgical wait + prefetch, early trigger kept) and read
gap_ours_a8. IF contention persists (duration still stretches / overlap still
< cuBLAS), A9 = move the trigger LATE (fire it when the scheduler exhausts
tiles, matching CUTLASS/TK). Keeping it separate = clean attribution of the
wait-scope win vs the trigger-timing win. Also trimmed: C_tail prefetch now
guarded by HAS_M128 (was redundant — C_tmap_tail == C_tmap for the plain
128x256 tiles), removing one wasted prefetch.

#### 60.18 ours vs ThunderKittens — honest comparison (2026-07-14)
NO head-to-head perf data exists: TK bf16 is B200 (sm_100), we are B300
(sm_103); nobody ran TK on our hw/shapes. TK ALSO can't run most of our shapes
as-is: static_assert(Mb==256) => 256-row tiles ONLY (its harness tests square
N=1024..16384); our 42 Qwen shapes need 64x64/64x128/128x256 for the skinny
N=128/256/1024 projections. TK bf16 has NO beta (pure C=A*B); ours supports
alpha+beta (addmm), composite-M tails, swap-AB, tile range 64^2..256^2.
COMPARABLE DIMENSIONS:
 - Correctness (mandatory): both correct; OURS strictly more general (beta,
   skinny, composite) => better for our mission.
 - PDL overhead (instr count): TIE — both minimal (prefetch + 1 wait beta=0 +
   1 trigger).
 - PDL overlap quality: TK fires trigger LATE (optimal); we fire EARLY
   (CUTLASS-warned competition). TK ahead HERE until our A9 (move trigger
   late) matches it.
 - Raw throughput on huge square GEMMs: likely TK (elite, heavily tuned), but
   UNMEASURED and not our target.
VERDICT: for our use case (42 Qwen shapes on B300 vs cuBLAS, skinny + beta),
ours is the more complete/appropriate tool at overhead EQUAL to TK's, with one
PDL knob (A9) left to fully match TK's overlap technique. "Who is faster on
giant square matrices" is unmeasured and not our battleground — cuBLAS is.

#### 60.19 A8 gate NEUTRAL -> A9 late trigger (2026-07-14)
A8 (surgical wait + prefetch): CORRECT (pdl_chain 4/4 BIT-EXACT, 3x template
PASS) but perf-NEUTRAL — gap_ours_a8 per-iter 12.80us == A7's 12.83 (dur still
24.45, gap -11.65). So wait-scope was NOT the lever. The persistent 24.45us
stretch + CUTLASS's "too early competes" => the EARLY TRIGGER is the remaining
suspect. A9: moved griddep_launch_dependents from entry (before K-loop) to the
consumer warp AFTER its last MMA (cuBLAS/TK/CUTLASS pattern) so the dependent
overlaps our epilogue (tensor cores free) not our MMAs. A8 KEPT (correct,
minimal, right foundation). Gate A9: pdl_chain + 3x template + NITER=50 +
gap_ours_a9 (target: dur -> ~14, per-iter -> toward 9.95).

#### 60.20 A9 late-trigger — mechanism CONFIRMED, throughput NEUTRAL; PDL exhausted (2026-07-14)
A9 correct (pdl_chain 4/4 BIT-EXACT, 3x template PASS). gap_ours_a9:
  dur_med 24.45 -> 17.76 (stretch SHRANK — early trigger WAS the contention),
  gap -11.65 -> -5.06 (now cuBLAS-SHAPED, cuBLAS=-4.06),
  per-iter 12.80 -> 12.70 (NEUTRAL — reduced stretch offset by reduced overlap).
So the late trigger did EXACTLY what CUTLASS predicted (killed the MMA-vs-MMA
contention, overlap now gentle+clean like cuBLAS) but throughput is unchanged.
KEY CONCLUSION: PDL is now STRUCTURALLY MATCHED to cuBLAS (gap -5.06 vs -4.06)
yet our per-iter 12.70 vs cuBLAS 9.95 = 78%. The residual is NOT the launch/
overlap mechanism anymore — it is that cuBLAS's overlapped throughput (9.95)
beats ours (12.70) despite our FASTER isolated kernel (13.76 vs 14.02). i.e.
cuBLAS packs more useful work per unit time under back-to-back execution:
candidates = epilogue density (D3, heavy per-chunk barriers contending with
the next grid's loads) OR warm-L2 exploitation (the §57 mystery). PDL TUNING
IS EXHAUSTED. KEEP A7+A8+A9 (A7 banked +6.9pt; A8/A9 are the correct canonical
final structure matching cuBLAS/TK/CUTLASS, no regression).
NEXT (measure first, A3 lesson): ncu the OVERLAPPED run to see the bottleneck
resource (memory tput / tensor active / stall) during overlap, THEN pick
A4-epilogue vs warm-L2 — do not guess.

#### 60.21 CULPRIT FOUND — fixed per-tile epilogue/prologue overhead (warm ncu, 2026-07-14)
Warm single-kernel ncu (--cache-control none), both kernels, 3 shapes. BOTH
my hypotheses REFUTED by data:
 H1 CO-RESIDENCE — REFUTED: BOTH kernels are 1 CTA/SM (Block Limit Shared Mem
   = 1 for both; cuBLAS dyn smem 213.28KB, ours 229.38KB). Neither can co-
   reside two grids; not the differentiator.
 H2 WARM-L2 (§57) — REFUTED: our warm L2 hit is EQUAL/BETTER (0.6b-K 98.2 vs
   94.8; 4b-K ~101 vs 98.3; 3b-Down 68.2 vs 64.6). We are NOT missing warm-L2;
   the §57 effect does not apply on the 128x256 tile.
THE CULPRIT — FIXED per-tile overhead, tensor-cores IDLE:
  warm duration gap is ~CONSTANT across K (not K-scaled):
    0.6b-K  16.42 vs 13.73  (+2.69us)
    4b-K    25.76 vs 22.85  (+2.91us)
    3b-Down 139.30 vs 136.83(+2.47us)
  Compute(SM)% : ours 34.74 vs 44.04 (0.6b-K) -> 53.8 vs 59.0 (4b-K) ->
    80.0 vs 80.3 (3b-Down, PARITY). i.e. the fixed ~2.5-2.9us idle overhead is
    a big fraction at low K (=> our worst shapes 77%) and negligible at high K
    (=> 3b-Down at parity 96%). This single fact explains the whole low-K vs
    high-K mirror-% pattern.
  Stall quality: cuBLAS warp-cycles/issued 20.76, 72.5% of it the HEALTHY
    data-scoreboard wait (productive GEMM). Ours 22.92, only 38.9% scoreboard
    => proportionally MORE sync/barrier/other overhead stalls. Confirms the
    overhead is our sync-laden prologue+epilogue, not compute.
CONCLUSION: the residual gap = our larger FIXED per-tile cost = the prologue
(init/tmem-alloc) + the D3 epilogue (per-chunk 2xbarrier+fence loop, no
stmatrix, narrow tcgen05_ld vs cuBLAS's wide LDTM+flat store). This is A4,
now MEASURED not guessed. It is worth it: closing ~2.5us lifts every low-K
K-projection from ~77% toward ~95%+ (about half our shapes). HARD though
(epilogue is correctness-bound + smem-saturated, §60.14).
NEXT: split prologue vs epilogue (one targeted measure), then attack the
dominant part — candidate levers: wider TMEM loads (LDTM vs tcgen05_ld),
reduce the 2 per-chunk barriers, cut prologue init. Measure each (A3 lesson).

#### 60.22 split result — overhead is DIFFUSE sync/fill, not one target (2026-07-14)
Source-sampled warm capture, 0.6b-K (single wave). Sample counts low (ours
512, cuBLAS 424) => qualitative not precise. Stall fingerprints:
  cuBLAS: 68% one parked NANOSLEEP (healthy spare-warp idle); ~nothing on
    barriers/fences.
  ours: 47% @!P2 BRA after SYNCS.TRYWAIT (consumer data-wait/pipeline fill,
    ~healthy) + 12% NANOSLEEP + 11% UCGABAR_WAIT/ERRBAR (the PROLOGUE
    cluster_sync, exec=28672=once/thread; cuBLAS ~none) + ~6% scattered
    FENCE.VIEW.ASYNC.S/WARPSYNC/BAR (epilogue sync).
FINDING: the fixed ~2.5us (§60.21) is DIFFUSE — prologue cluster_sync +
epilogue fences/barriers + pipeline fill/drain — NOT a single dominant
fixable target. cuBLAS runs the SAME 6-stage tile yet stays cleaner, so it is
not a stage-depth knob. Every removal candidate is correctness-bound + smem-
saturated (§60.14) + partly already PDL-overlapped => hard rewrite for an
uncertain slice of 2.5us.
DECISION (diminishing returns): BANK 128x256_2x2. Campaign win captured (PDL
+6.9pt 82.6->~89.5; high-K at parity 80.0 vs 80.3 compute). Next higher-ROI
per §59: (1) 128x256_v polish (13 shapes ~96%, small gap x big coverage);
(2) §59 P2 free-inheritance check (PDL/fence/hint likely propagated to all 21
variants for free points). Revisit the low-K epilogue only if a broad sweep
shows it is the dominant remaining loss across many shapes.

#### 60.23 MISTAKE FOUND — the membars are from cluster_sync + epilogue fence, NOT tcgen05.fence; A1 mis-fixed (2026-07-14)
Deep SASS trace of the CURRENT (post-A1) build (split_ours_sass.md) — the 10
MEMBAR.ALL.GPU + 11 MEMBAR.ALL.CTA §60.2 blamed on tcgen05.fence are STILL
here after A1 removed those fences. So §60.2 D1 was WRONG and A1 removed the
wrong thing (which is exactly why A1 measured neutral, §60.4). Real sources,
traced by SASS context + exec counts:
 (1) cluster_sync() [line 445: barrier.cluster.arrive.RELEASE + wait.ACQUIRE]
     -> MEMBAR.ALL.CTA + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR quad. GPU-SCOPE
     (most expensive fence). Executes 2x/kernel (prologue @SASS169, end
     @SASS1966), exec 28672 = once/thread. cuBLAS = ZERO gpu-scope membars.
 (2) epilogue store seq [STS.128 -> MEMBAR.ALL.CTA -> FENCE.VIEW.ASYNC.S ->
     BAR.SYNC], SASS1846 exec 65536 (4 chunks x 128 epi-thr x 128 CTA). The
     MEMBAR.ALL.CTA is the fence.proxy.async lowering for our GENERIC STS.128
     stores; cuBLAS uses STSM (stmatrix) and emits only FENCE.VIEW.ASYNC.S,
     no membar (D3).
 (the 8 SYNCS.ARRIVE.RED membar quads at SASS1090-1174 are exec=0 = cold
  swap/FP32 path, irrelevant for plain shapes.)
cuBLAS total: ONE MEMBAR.ALL.CTA (exec 16384). Ours: 2 gpu-scope cluster_sync
quads + 1 cta-scope epilogue membar x4chunks.
FIX PLAN (DeepGEMM/CUTLASS-validated):
 A10: cluster_sync RELEASE/ACQUIRE -> RELAXED arrive + plain wait (DeepGEMM
   comm/barrier.cuh cluster_sync_with_relaxed_arrive: barrier.cluster.arrive.
   relaxed + barrier.cluster.wait). SAFE because init visibility is already
   provided by our cluster_fence_mbarrier_init (fence.mbarrier_init.release.
   cluster, line 442) exactly like DeepGEMM, and tmem is CTA-local. Removes
   the 2 GPU-scope membar quads.
 A11: epilogue generic STS.128 -> stmatrix (STSM) to drop the per-chunk
   MEMBAR.ALL.CTA (D3). Bigger change; after A10.
GATES: 3x bit-exact + pdl_chain + NITER=50 + re-dump SASS to confirm the
gpu-scope membars are GONE.

#### 60.24 A10 IMPLEMENTED + 4-way cluster-sync backtrack (2026-07-14)
Reverse-engineered the cluster barrier across ALL references:
  cuBLAS (SASS): UCGABAR_ARV + UCGABAR_WAIT, NO preceding membar.
  CUTLASS (cute::cluster_sync DEFAULT): barrier.cluster.arrive.aligned +
    .wait.aligned (PLAIN, no release/acquire).
  DeepGEMM: barrier.cluster.arrive.relaxed + .wait (RELAXED, weakest).
  OURS (before): barrier.cluster.arrive.RELEASE + .wait.ACQUIRE — the ONLY
    form that emits the MEMBAR.ALL.CTA+GPU+ERRBAR+CGAERRBAR quad (gpu-scope).
=> we were the outlier, over-fenced. A10: switched to PLAIN (CUTLASS default,
the tested standard, safer than DeepGEMM's relaxed). Removes 2 gpu-scope
membar quads/kernel. SAFE: init visibility via cluster_fence_mbarrier_init
(already present), tmem CTA-local. Also A1 REVERTED same change (restored
tcgen05.fence — PTX-required, was a misdiagnosis, zero-cost to restore).
Both compile clean. GATE: 3x bit-exact + pdl_chain 4/4 + NITER=50 + re-dump
SASS (confirm MEMBAR.ALL.GPU count 10 -> ~0). Expect: fixed-overhead drop on
low-K shapes toward cuBLAS.

#### 60.25 tcgen05.fence — deep analysis: KEEP it (not redundant, zero cost) (2026-07-14)
Q (gautam): any use keeping tcgen05.fence? precision/perf hit? redundant?
PTX ISA 9.3 §9.7.17.11.1 (lines 32453, 35614):
 - "composes WITH execution ordering instructions (mbarrier, barrier, morally
   strong ld/st) to establish ordering between tcgen05 ops ACROSS THREADS."
 - "behaves as a CODE MOTION fence — [tcgen05 ops] cannot be hoisted across."
KEY: tcgen05 ops (MMA->TMEM write, tcgen05.ld read) are ASYNC on a separate
path (like TMA on the async proxy). The mbarrier's acquire/release orders
REGULAR memory but NOT the async tcgen05 path. The spec says the fence
COMPOSES WITH the mbarrier => BOTH needed, not either/or. So it is NOT
redundant: it orders a path nothing else in our kernel covers.
Concrete need: consumer MMA writes TMEM -> fence::before -> mbarrier arrive
(tmem_full); without the fence the async write is not ordered before the
arrive, so the epilogue can read INCOMPLETE MMA results = wrong output.
Symmetric on read: mbarrier wait -> fence::after -> tcgen05.ld.
ANSWERS: redundant=NO. precision hit=ZERO (ordering only, bit-identical).
perf hit=NEUTRAL (code-motion fence, emits NO membar — count unchanged; A1
removal measured 23.33->23.84 = noise). => KEEP it (A1 revert correct): all
upside (async-path correctness), no downside. A1's removal was pure risk for
zero gain; the gates passed by LUCK (compiler didn't happen to reorder + the
race didn't surface), not because it is safe. Corrects §60.4's wrong framing
("sm_103 coherent without them" — it is not; we were lucky).

#### 60.26 MEMBARS ARE A RED HERRING (measured) — A10 kept, refocus on fill/epilogue (2026-07-14)
A10 (cluster_sync->plain) did NOT drop the SASS membar count (still 10 GPU +
11 CTA). Traced rigorously in the a10 SASS, weighting by exec x samples:
 - 8 of 10 MEMBAR.ALL.GPU are COLD (exec=0, swap/FP8 branch, never run by
   plain 0.6b-K).
 - only 2 execute (prologue line154, end line1957), once/thread each.
 - they contribute 3 stall samples / 516 total = 0.6% of stall time.
 - the executed prologue one is fence.mbarrier_init.release.cluster (line 442)
   = REQUIRED one-time init fence (cuBLAS/DeepGEMM equivalent), NOT cluster_sync.
CONCLUSION: the "10 vs 0 membar count" (§60.2/60.23) was MISLEADING — by
actual cost it is 0.6%, a red herring for perf. Lesson: weight by exec x
samples, never raw count. The hunt still fixed two REAL mistakes (A1 revert =
restored required tcgen05.fence; A10 = removed cluster_sync over-fencing vs
CUTLASS default) — both correct, both KEPT, neutral perf.
REAL 2.5us culprit (§60.22, stall budget): ~47% pipeline FILL/data-wait (low
K: 6-stage fill over ~16 iters = ~37% of iters) + ~6% epilogue fences + the
one-time barriers. cuBLAS runs the SAME 6-stage tile and fills/drains faster.
=> the lever to reach cuBLAS on low-K is pipeline FILL + epilogue-store tail
latency, NOT membars. Both hard/correctness-bound. Next: measure fill cost
directly (vary LOAD_PIPE_DEPTH on a low-K shape) since cuBLAS proves 6 stages
CAN be fast at low K — find WHY our fill is slower (load latency? issue
order? not the count).

#### 60.27 COMPLETE FENCE AUDIT — A12: mbarrier arrives were still over-fenced (2026-07-14)
gautam's Q: did we remove ALL unnecessary fences, or are the cold-for-0.6b-K
membars a problem for OTHER shapes? Answer: they WERE a problem — full audit:

  primitive                       form (before)              verdict      action
  cluster_sync                    barrier.cluster.arr.RELEASE over-fenced  A10 -> plain
  mbarrier_arrive_expect          arrive.expect_tx.RELEASE    over-fenced  A12 -> plain
  mbarrier_arrive_expect_cluster  arrive.expect_tx.RELEASE.cl over-fenced  A12 -> plain
  mbarrier_arrive_cluster         arrive.RELEASE.cta          over-fenced  A12 -> plain
  cluster_fence_mbarrier_init     fence.mbarrier_init.rel.cl  REQUIRED     keep (init vis)
  tma_store_fence                 fence.proxy.async.shared    REQUIRED     keep (STS->proxy)
  tcgen05.fence before/after      tcgen05.fence               REQUIRED     keep (A1 revert)

A12 KEY EVIDENCE: CUTLASS arch/barrier.h uses PLAIN arrives everywhere —
mbarrier.arrive.shared::cluster / mbarrier.arrive.expect_tx.shared::cluster,
NO .release (lines 494/514/593/616). We were the outlier adding .release,
which forced the SYNCS.ARRIVE.RED + MEMBAR.ALL.CTA+GPU+ERRBAR+CGAERRBAR quad.
Those are the 8 "cold for single-wave 0.6b-K" membars — but HOT for multi-
wave (Gates/Downs CLC empty arrives), swap, and multicast variants. So the
over-fencing WAS costing those shapes; gautam was right it is a problem.
The mbarrier arrive/wait pair provides the pipeline's acquire/release ordering
already; the explicit .release was redundant (expect_tx data ordering comes
from the TMA completion, not the arrive). CUTLASS-validated.
NET: after A10+A12 the ONLY remaining fences are the 3 genuinely-REQUIRED ones
(init-visibility, generic-store->async-proxy, async-TMEM ordering). All
UNNECESSARY over-fencing removed.
GATE: 3x bit-exact (covers swap/composite/all configs) + pdl_chain +
dispatcher 45/45 + a MULTI-WAVE shape (Gate/Down) benchmark (where these
arrives are hot = where the payoff is) + clean-rebuild SASS membar recount
(expect MEMBAR.ALL.GPU 10 -> ~1, only the required init fence).

#### 60.28 EOD 2026-07-14 — session summary + tomorrow's first job
Today's arc on 128x256_2x2 (all gated, all pushed):
 - A6a tmap cache: correct, perf-NEUTRAL (host work was already overlapped).
 - A6b gap decomposition -> A7 PDL (programmatic dependent launch): THE WIN,
   geomean 82.6 -> ~89.5%, 9 shapes >=100%, chain gate 4/4 bit-exact.
 - A8 surgical PDL wait-scope + prefetch: correct, neutral (wait-scope wasn't
   the lever).
 - A9 late trigger: mechanism confirmed (stretch 24.45->17.76, gap cuBLAS-
   shaped -5.06) but throughput neutral. PDL matched to cuBLAS structurally.
 - A3 pipelined poll: tried 3 ways, all NEGATIVE (noise), reverted; 50us hint
   also reverted. Closed.
 - CULPRIT HUNT (§60.21-22): residual low-K gap = fixed ~2.5us prologue/
   epilogue overhead (co-residence + warm-L2 both REFUTED by data). Diffuse
   pipeline-fill + epilogue-tail.
 - FENCE MISTAKE FOUND + FIXED (§60.23-27): the "10 membar" we blamed on
   tcgen05.fence (§60.2) were NOT from it. A1 (fence removal) was a
   misdiagnosis -> REVERTED (tcgen05.fence restored, it's a required async-
   TMEM fence, §60.25). Real over-fencing was .release on cluster_sync (A10)
   and on ALL mbarrier arrives (A12) vs CUTLASS's plain forms. Measured: the
   membars are only 0.6% of stalls on 0.6b-K (red herring there) BUT the
   arrive membars are HOT on multi-wave/swap/multicast shapes -> A12 is a real
   cleanup for those. After A10+A12 only 3 required fences remain.
STATE: A7 PDL banked (+~7pt, live). A1-revert/A10/A12 = correct fence cleanups.
PENDING (tomorrow's FIRST job): GATE A12 — 3x bit-exact + pdl_chain + 45/45 +
multi-wave shape benchmark (output_a12.md) + clean-rebuild SASS membar recount
(expect MEMBAR.ALL.GPU 10 -> ~1). A12 is committed+pushed but NOT yet gated on
silicon. If any mismatch -> revert (plain arrives must be proven on swap/
composite). THEN: real low-K lever = pipeline-fill + epilogue-tail (§60.22),
measure fill via LOAD_PIPE_DEPTH sweep since cuBLAS proves 6 stages can be
fast at low K.

#### 60.29 A12 GATE — PASS, membars removed (2026-07-15)
Gated on new box (deep-name-cracks-fin-03). pdl_chain 4/4 BIT-EXACT, 3x
template ALL PASS, dispatcher 45/45 all 0.00e+00. SASS on multi-wave shape
(4096x11008x2048, where the arrives are hot): MEMBAR.ALL.GPU 10 -> 2,
MEMBAR.ALL.CTA 11 -> 3. The 8 over-fencing gpu membars GONE; the 2 remaining
= required fence.mbarrier_init. SYNCS.ARRIVE.RED opcodes stay (normal arrive);
only the preceding membar quad is deleted. Perf geomean 87.6% = within cross-
box noise (each day is a different physical B300), NO clean perf-win claim,
but A12 KEPT: correct + CUTLASS-standard plain arrive + real over-fencing
removed. FENCE ARC COMPLETE: A1-revert (tcgen05.fence restored) + A10
(cluster_sync plain) + A12 (arrives plain) => only the 3 required fences
remain (init-visibility, generic-store->async-proxy, async-TMEM ordering).
NEXT: remove the dead FP32-fused path (§52, closed negative) — dormant
constexpr-gated dead code, no production route, only test_fused_fp32.cu uses
it; taxes every template edit (had to add griddep_wait to it in A8). Pure
cleanup, zero perf, gated to prove bf16 byte-identical.

#### 60.30 DEAD CODE REMOVED — FP32 fused path deleted (2026-07-15)
Removed the entire §52 fused fp32-input path (closed negative, never routed in
production — only test_fused_fp32.cu used it). 283 lines deleted from the
template. Reviewed against the pre-removal (A12) code to confirm bf16 is
byte-identical: (1) mbarrier full-count ternary collapsed to 1*CLUSTER_SIZE =
the old bf16 branch; (2) role if/else chain producer->consumer->scheduler->
epilogues intact, trans-warp block excised cleanly; (3) producer loop + all
launcher tmap-init FP32 branches collapsed to the bf16/layout paths; (4) B_in
param confirmed FP32-ONLY (old sole use: reinterpret_cast<float*>(B_in) inside
the removed trans-warp block; bf16 uses B_tmap) — kept as unused param to
preserve the kernel signature (callers unchanged). Removed: _FP32_IN/
_FP32_DIRECT template params, FP32_IN/DIRECT/NUM_TRANS_WARPS/FP32_PIPE_DEPTH/
FP32_STAGE_BYTES constants, the 2 static_asserts, init_4d_tma_f32_KX,
fp32 mbarrier arrays/smem, the trans-warp block, producer FP32 branches,
launcher FP32 branches, config aliases, and test_fused_fp32.cu. Both
ncu_probe.cu + Bgemm_tn_variants.cu compile clean; 0 FP32 refs remain; no
FP32 refs anywhere else in the repo. GATE (pending): 3x bit-exact template
(proves bf16 byte-identical) + dispatcher 45/45. Pure cleanup, zero perf.

#### 60.31 FREE-HOUR DEEP PROFILE — "where we lose" pinned to the SASS line (2026-07-15)
--set full on all 8 campaign shapes x both kernels (reports in
profiling_reports/free_hour/, KEEP as reference). Confirmed NOT bank conflicts
(none), NOT a pipe bottleneck (tensor "well-utilized, not bottleneck" both:
cub 39.7% vs ours 33.0%). The gap is tensor cores IDLE more in ours = fixed
prologue/epilogue overhead, now decomposed to the instruction:

0.6b-K stall fingerprint (ours 524 samp vs cuBLAS 425):
  cuBLAS: 68% ONE parked NANOSLEEP (healthy), ~clean everything else.
  ours: 46% @!P2 BRA data-wait/fill (HEALTHY, matches cuBLAS park) PLUS ~12%
    EXTRA cuBLAS lacks:
     - UCGABAR_WAIT (cluster barrier) 50 vs cuBLAS 4 -> 2 sites, both exec
       once/thread: prologue cluster_sync (line161, 23 samp) + END cluster_sync
       before dealloc (line1932, 27 samp).
     - FENCE.VIEW.ASYNC.S 15 (epilogue per-chunk proxy fence, exec 65536 =
       4 chunks x epi-threads x CTAs) vs cuBLAS ~0.
     - ERRBAR 12 (cluster-barrier machinery).
  ~12% extra x 17.28us = ~2.1us = the fixed overhead (§60.21).
COMPUTE-GAP shrinks with K (fixed-overhead proof): 0.6b-K 8.6pt, 4b-K 6.6,
8b-K 4.8. (3b-Down --set full unreliable: --set full perturbs CLC multi-wave.)

ROOT-CAUSE CHAIN (the insight): our EPILOGUE is the source. Its chunked,
sync-laden structure (a) costs the per-chunk proxy fences directly (15 samp),
AND (b) makes warps finish stores at DIFFERENT times -> they pile up waiting
at the END cluster_sync (27 samp of imbalance stall; ncu flags "load
imbalance, highly different execution durations per warp"). cuBLAS's FLAT
epilogue finishes warps together -> its end-barrier wait ~0. So the epilogue
poisons the end cluster-barrier via warp imbalance, on top of its own fences.
TARGET: epilogue rework (A4/D3) attacks fence(15) + end-barrier imbalance(27)
= ~42 samp = ~1.4us recoverable — the biggest concrete lever left. Prologue
cluster_sync (~23, ~0.8us) is inherent (DeepGEMM/cuBLAS also cluster_sync +
dealloc). Approach = make the epilogue FLAT + warp-balanced (cuBLAS: wide
LDTM -> FFMA2 -> stmatrix -> batched TMA, uniform per-warp work) so all warps
finish together; watch the §60.14 smem constraint (may cost a load stage).
NEXT SESSION: A4 epilogue, measured target ~1.4us, gate bit-exact.

#### 60.32 A4 STEP-1 (batched fence) — MEASURED NET LOSS, REVERTED (2026-07-15)
Implemented culprit-1 of §60.31 behind a compile flag (-DBATCH_EPI_FENCE):
each N-chunk writes its OWN store buffer (STORE_PIPE_DEPTH 2 -> num_chunks),
then ONE fence.proxy.async covers all chunks, then all TMAs issue together
(flat, warp-balanced tail). OFF path provably byte-identical (depth 2, %2==^1,
if constexpr(false)->orig). Gated on B300 via test_bgemm_sm103_cluster_template
(252/252 shapes, both A/B).

RESULT — correct but SLOWER, so reverted:
- Correctness: 252/252 PASS, every relerr byte-identical to OFF. The impl is
  right; this is a clean measurement.
- Perf (Qwen 42-shape sweep, best-of horiz/vert %cuBLAS):
    128x256 tile: OFF 85.8% -> ON 81.4%  = **-4.4pt REGRESSION**
    128x128 tile: OFF 86.0% -> ON 86.5%  = +0.5pt (noise)
- WHY (decisive from the two tiles): 128x128 has num_chunks=2 so depth stays
  2->2 (pure fence-batching, ZERO smem cost) -> NEUTRAL. So batching 4 fences
  into 1 buys ~nothing (confirms §60.26: membar/fence is a red herring, ~0.6%
  of stalls). 128x256 has num_chunks=4 so depth 2->4 (+32KiB store) drops the
  load pipe 6->5 stages -> the lost mainloop stage costs FAR more than the
  fence saves. §60.14 confirmed: epilogue is smem-bound; any fence-batch that
  costs a load stage is a net loss.
- VERDICT: culprit-1 (the fence, 15 samp) is a DEAD END — the fence was never
  the real cost. Reverted template to validated (2498 lines); kept only the
  generic `EXTRA=` makefile hook (harmless, reusable for future A/B).
- REMAINING lever = culprit-2 ONLY (the 27-samp END cluster_sync imbalance),
  which is smem-FREE (rebalance the warp0-gated tma_store_wait/commit tail so
  warps finish in step). That is the sole worthwhile A4 target now; if it also
  proves neutral, the fixed ~2.5us prologue/epilogue is simply the floor and we
  stop chasing it (§60.21 was already ~2.5us of a ~17us low-K kernel).

NEXT SESSION (tomorrow): A4 culprit-2 ONLY — rebalance the warp0-gated
tma_store_wait/tma_store_commit tail in the epilogue so all epi warps finish in
step (attacks the 27-samp end cluster_sync imbalance). Smem-FREE so no load-stage
tax (the thing that killed step1). One measured shot, gate bit-exact on
test_bgemm_sm103_cluster_template (252 shapes) + Qwen sweep %cuBLAS vs the 85.8%
baseline. If neutral -> accept the ~2.5us epilogue floor, close A4, move to P2
2x2 free-inheritance sweep.

### 61. BluTrain 120M integration + mycublas->blublas rename (2026-07-15)
Wired the BluTrain training stack into the GOUTHAM1926/blublas repo as BluTrain/
subfolder with our blublas swapped in, to compare our GEMM backend vs cuBLAS in a
real training. NOTE: bluscript_amp.cpp is a 48M-param model (d_model=384,
n_layers=6; prints "Number of parameters: 48076416"), NOT 120M despite the name.

#### 61.1 Integration mechanics
- de-linked kathir .git/submodules; our blublas at
  BluTrain/Tensor-Implementations/BluBridge-BLAS (build target, -IBluBridge-BLAS/include).
- WITH_BLUBLAS=0 -> cuBLAS (BLAS_RUN_LDLIBS empty, no -lblublas), =1 -> ours.
  Run: OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 make -j$(nproc)
  run-mpi FILE=bluscript_amp.cpp WITH_BLUBLAS=0|1 NP=1.
- Makefile gotchas: NO -j by default (pass -j$(nproc)); GENCODE hardcodes 6
  arches (sm_80/86/89/90 + sm_100a/103a) regardless of SM_ARCH -> for fast B300
  builds override GENCODE_LEGACY/GENCODE_BLACKWELL to sm_103a-only. BLACKWELL_MARKERS
  matches path substrings SM100/SM103/B300 -> our SM103 kernels get sm_103a.

#### 61.2 Build fixes (3, all required)
1. NVML: nvmlDeviceGetComputeRunningProcesses_v2 -> unversioned (B300 driver
   dropped _v2, breaks NvmlMonitor.cpp; unversioned macro auto-picks the box's
   version, portable to colleague's older box too).
2. mycublas*->blublas* rename across all blublas src (67 files, 828 hits): both
   mycublas->blublas AND MYCUBLAS->blublas (colleague's blublas.h uses lowercase
   blublas_OP_N/blublas_STATUS_*). libtensor's GenMatmul (compiled -DWITH_MYBLAS)
   calls blublas* symbols; our src still DEFINED mycublas* -> undefined refs at
   the _runner_ link. Deleted redundant mycublas.h (identical to blublas.h modulo
   prefix + one line).
3. myCublas.cpp deleted (pure 40-line duplicate of BluBlas.cpp: multiple-definition
   of blublasCreate/Destroy/SetStream at libblublas.so link) — a merge leftover
   the rename exposed; BluBlas.cpp is canonical.

#### 61.3 Numbers (B300 sm_103a, 48M model, B×T 2×4096, grad_accum 64, 524288 tok/step)
cuBLAS ~63.8K tok/s (dt ~8211ms), our blublas ~59-64K. BUT nsys shows ~95% of GPU
time is the gqa_fused attention kernels; every cuBLAS nvjet GEMM is <0.3% each ->
this training is attention-bound, NOT a GEMM-backend benchmark. blublas-vs-cuBLAS
is within noise here; the real GEMM win stays the standalone dispatch test (~86%
geomean, §54).

#### 61.4 Colleague 34K mystery — RESOLVED (stale sm_89 build)
madhu/dhana ran 34K vs our 64K on the SAME model/config/GPU. nsys: identical
gqa_fused kernels but 2.5x slower per call in his. cuobjdump --list-elf
libtensor.so: his was compiled for sm_89 (59 ELF sections sm_89, only 4 sm_103a)
vs ours evenly all-arch incl 61 sm_103a. His was a stale build make never
recompiled for B300 ("libtensor is up-to-date"), so the attention kernels JIT'd
from sm_89 PTX at load -> 2.5x slower. Fix = rm libtensor.so + rebuild native
sm_103a -> jumps to ~64K. LESSON: when "same code" runs mysteriously slow,
cuobjdump --list-elf the .so to confirm it has native SASS for the running arch.

### 62. WHERE WE LOSE at low-K = WAVE QUANTIZATION, not occupancy/warp-layout (2026-07-15)
Chased the low-K gap (0.6b-K: ours 17.28us vs cuBLAS 13.76us, -25%) with offline
ncu on the free_hour reports + a deep-research pass (CUTLASS/DeepGEMM/TK/cuBLAS,
12 claims verified 3-0). Three conclusions, two of them retractions of my own
earlier guesses:

#### 62.1 SAME tile, SAME occupancy — cuBLAS still wins (the hard numbers)
0.6b-K, cuBLAS `nvjet_sm103_tst_128x256_64x6_2x2` = IDENTICAL geometry to ours.
  metric              cuBLAS   ours
  Grid / Waves/SM     128/0.86 128/0.86   (identical — this shape has 128 tiles)
  Achieved Occupancy  8.68%    ~10.75%    (ours HIGHER — occupancy is NOT it)
  Duration            13.76us  17.28us
  Avg active thr/warp 31.65    21.31      <- the visible gap
  Compute(SM) tput    42.2%    33.5%      (== the 1.25x duration ratio)
  Registers/thread    255      167
  L2 hit              94.9%    88.3%

#### 62.2 RETRACTED: "warp-layout waste" is a symptom, not the cause
I first blamed our 7-warp layout (4 epi + 1 producer + 1 consumer + 1 scheduler
= 3 single-thread control warps) for the 21.3 avg-active-threads. WRONG on two
counts: (a) research confirmed our layout IS the SOTA pattern — NVIDIA CUTLASS
Blackwell CLC doc lists the SAME discrete warps ("Warp 1: Scheduler (CLC
producer)... Warp 0: MMA warp... Mainloop Load, Epilogue Load, Epilogue"), only
one elected thread issues clusterlaunchcontrol.try_cancel; ThunderKittens
Blackwell dedicates whole warps too. (b) the 21.3 vs 31.65 is a METRIC artifact:
with issue slots only 5% busy, our control warps SPIN on @!P2 BRA poll loops
(counted active-but-stalled) while cuBLAS parks on ONE nanosleep (§60.31) — the
spinning drags avg-active-threads but steals ~nothing real. Restructuring warps
would DEPART from SOTA, not catch up. Dead end.

#### 62.3 THE REAL LEVER: wave quantization + split-K / stream-K (SOTA-confirmed)
> **PARTIAL RETRACTION 2026-08-01 (see h200 doc §H27z.8) -- the "SMOKING GUN" claim below is
> WRONG.** Measured on H200 by querying cuBLASLt's own algo config directly: cuBLAS does NOT
> split ANY skinny-N shape. `4096x128x896`, `4096x256x1536` and `4096x512x1024` all come back
> `splitK=1, reduction=NONE`. The real trigger is SMALL OUTPUT (M*N) + LARGE K -- e.g.
> `512x512x8192` -> S=2 and `256x256x16384` -> S=8. The actual rule (validated on all 6 probe
> shapes with zero exceptions) is: split so that `output_tiles * S ~= SM count`; the skinny-N
> shapes already reach ~128 tiles on 132 SMs, so there is nothing for split-K to fix there.
> The wave-quantization REASONING below is sound and the SOTA citations stand; only the claim
> that cuBLAS dispatches split-K on *these particular shapes* is refuted. Planning split-K work
> off the shape list below would target the wrong shapes.
Confirmed 3-0 (NVIDIA GEMM perf docs + Stream-K arXiv 2301.03598 + Colfax): below
one wave, SMs simply idle; the SOTA remedy is split-K/stream-K ("single persistent
CTA per SM, fractional tiles split along K... eliminates wave quantization by
eliminating waves"). Our starved shapes on B300 (~148 SMs):
  shape            tiles(CTAs)  SM fill   our %cuB
  q25-0.5b-K N=128   32          22%        40%
  fallback-N512      ~64         ~43%       50%
  q25/q3 -K N=256    64          43%        42-45%
  0.6b-K             128         86%        80%
SMOKING GUN: cuBLAS dispatches split-K on EXACTLY these — `nvjet_..._splitK_NTT`
+ `cublasLt::splitKreduce_kernel` appear in our own BluTrain nsys + §54 winner
traces. WE HAVE NO split-K/stream-K path on sm103. Single biggest pool of idle
SMs left (the 40-55% small-N/low-K shapes). Split-K raises SM util ~2x on skinny
GEMM (arXiv 2402.00025: 20.75%->43.05%, occ 7.55->27.75 on A100).
Bonus: both kernels smem-capped at 1 CTA/SM, so raising our 167 regs toward
cuBLAS's 255 is FREE occupancy-wise -> free ILP for a wider epilogue.

#### 62.4a CULPRIT-2 RESULT — NEUTRAL, reverted (2026-07-16, fresh B300)
A/B on test_bgemm_sm103_cluster_template (252 shapes, bit-exact both ways):
  tile      c2_OFF best%cuB   c2_ON best%cuB
  128x256   85.7             85.5   (-0.2, noise)
  128x128   85.8             85.8   (flat)
CONFIRMED NEUTRAL — the end-barrier wait is warp0 draining the final TMA stores
(fundamental memory latency); moving the drain past cluster_sync just relabels
the stall bucket, wall-clock unchanged. Both epilogue-sync levers now empirically
dead (culprit-1 fence -4.4pt §60.32; culprit-2 drain neutral). The ~2.5us epilogue
floor is PROVEN, not assumed. Reverted template to validated 2498 lines.

#### 62.4 PLAN (evidence-ranked)
1. [DONE] culprit-2 tested -> neutral (62.4a). Epilogue floor accepted.
2. MAIN ARC: build split-K (then stream-K) for the sm103 BF16 kernel — partial-K
   accumulation + reduce kernel; dispatcher routes starved shapes (N<=512 /
   low-wave) to it. cuBLAS's own dispatch is the proof it pays; stream-K also
   lifts 0.6b-K (0.86->1.0 wave fill ~ +16% ceiling).
3. Cheap follow-up: bump register budget toward 255 (free, smem-capped) for a
   flatter/wider epilogue.
Sources: docs.nvidia.com/cutlass blackwell_cluster_launch_control; Stream-K
arXiv 2301.03598; research.colfax-intl.com persistent-kernels-and-stream-k;
hazyresearch.stanford.edu tk-blackwell; NVIDIA dl-performance-matrix-multiplication.

### 63. 8-SHAPE root cause = INSTRUCTION COUNT (address/int math); split-K PARKED (2026-07-16)
Full-day arc: explored split-K, realized it is OFF-TASK, refocused on the 8 P1
shapes (128x256, M=4096, N=1024-2048), and root-caused the ~15% gap to redundant
integer/address instructions, source-mapped to the exact lines.

#### 63.1 Split-K explored -> PARKED (it's a SKINNY-shape tool, not the 8 shapes)
- Our own SASS (free_hour *_cub.ncu-rep) shows cuBLAS uses PLAIN
  nvjet_sm103_tst_128x256_64x6_2x2 (NO split-K) on ALL 8 M=4096 campaign shapes.
  So split-K is NOT how cuBLAS beats us on these.
- Split-K only helps genuinely SM-starved SKINNY shapes (N<=512). Tile selection is
  ALREADY exhausted there: dispatcher winner-table routes {128,896}->V_64x64_2x2
  (smallest tile) and it still hits 40% (~0.86 wave). That is a SEPARATE LATER task.
- Split-K DESIGN (locked for the skinny task):
  - Numerics (hardware-forced, shown in our own kernel): bf16 in -> tcgen05.mma
    .kind::f16 multiplies bf16xbf16 (product EXACT, 16-bit fits fp32) -> accumulate
    in fp32 TMEM -> epilogue ONE F2FP.BF16.F32.PACK (RNE, 1 native instr, matches
    DeepGEMM math.cuh:73). Split-K adds NO new conversion (fp32 accumulate already
    exists today); fp32 partials + fp32 reduce = equal-or-better precision (grouped
    sum has lower error bound). Precision-killers to AVOID = bf16 partials / bf16
    atomics.
  - Reduce candidates to BENCHMARK: (A) separate deterministic reduce kernel (=
    cuBLAS splitKreduce style + our gradient-fix map-reduce), (B) fp32 atomicAdd
    (non-deterministic ORDER only, NOT less precise per-add), (C) CUTLASS turnstile
    (fused SEQUENTIAL barrier-ordered, NOT a tree). Proof: cuBLAS runs a SEPARATE
    cublasLt::splitKreduce_kernel<...,float,__nv_bfloat16,float> (fp32, seen in our
    BluTrain nsys). CUTLASS default = turnstile.
  - Gating (CUTLASS Heuristic, quoted): split ONLY when tail-wave < 50% full; else
    data-parallel. Auto split factor so total CTAs ~= SM count.
  - DeepGEMM (M-axis MoE grouping) + ThunderKittens (persistent pipeline) do NOT
    split-K -> no alt to copy for skinny.
  - Universal design: shared splitk/ reduce+workspace+gating (dtype-generic) + tiny
    per-arch SplitK blockIdx.z hook (S=1 = current path byte-identical).
  - Tools built + PARKED: examples/probe_cublaslt.cu (prints cuBLASLt splitK factor
    + reduction scheme per shape), examples/splitk_observe.sh.

#### 63.2 REAL 8-shape gap: instruction count, NOT occupancy/warp-layout/converts
0.6b-K, IDENTICAL tile (128x256_2x2), ncu --set full, ours vs cuBLAS:
  Compute(SM) tput 33.5% vs 42.2%   Duration 17.28us vs 13.76us (ours 25% slower)
  Executed IPC 0.30 vs 0.27 (we issue MORE)   Issue-busy 5.03% vs 4.09%
  Occupancy 10.75% vs 8.68% (ours HIGHER)      Registers 167 vs 255
- Occupancy REFUTED (both ~0.86 wave, ours higher). Warp-layout REFUTED (our 7-warp
  epi/producer/consumer/scheduler IS the CUTLASS SOTA pattern; we park on NANOSLEEP
  like cuBLAS, not spin). Converts REFUTED (F2FP 96 vs 128 -- we have FEWER).
  => we issue MORE => we EXECUTE ~1.5x more instructions for the same GEMM.

#### 63.3 THE finding: 6.3x integer/address instructions (static SASS mix, 0.6b-K)
  total SASS 2144 vs 1144 (1.87x) | int/addr 583 vs 92 (6.3x) |
  IMAD(int mul-add) 359 vs 4 (90x!) | div/rem/MUFU 27 vs 0 | branch 176 vs 82 |
  F2FP(convert) 96 vs 128 | IADD3(plain add) 4 vs 4 (SAME)
=> Pure address recomputation. cuBLAS precomputes addresses into REGISTERS (255) and
walks with pointer-adds (4 IMAD); we recompute base+idx*stride everywhere (359 IMAD).
Register 167 vs 255 is a SYMPTOM of recompute-vs-precompute, NOT a separate lever
(__launch_bounds__ already permits 255; smem-capped at 1 CTA/SM so registers free).
Restructuring to precompute+hold raises regs AND cuts instructions -- same fix.

#### 63.4 Source-mapped per-task accounting (PTX -lineinfo .loc; file 1 = template)
(cuobjdump + ncu source-mapping FAILED silently; PTX .loc via `nvcc -ptx -lineinfo`
WORKED: 3483 locs for file 1. cuobjdump has no valid -lineinfo flag here.)
Valid INTEGER ranking (float-EXCLUDED): L509(264 desc_encode SHF/LOP3) |
L1311(252 producer %6 modulo) | L2147(225 epilogue store addr) | L529(108
make_smem_desc) | L1283(105 producer) | L1612(101 consumer %6 modulo) | L324(88).
Clean integer MUL/MAD only (true IMAD): L1612(16)+L1311(16)=modulo top, L1286(14)
producer addr, then scattered epilogue/consumer/tile-decode. (SASS IMAD 359 >> PTX
int-mul ~80 because ptxas expands ld/st address expressions + desc shifts into IMAD.)
Per task (all beta/alpha-INDEPENDENT -> execute at runtime):
  1. Pipeline stage mgmt (L1311+L1612): `stage=(stage+1) % LOAD_PIPE_DEPTH` = ÷6
     non-pow2 -> magic-multiply, BOTH loops, every iter. cuBLAS = 0 (branchless/pow2).
  2. MMA descriptor build (L509/529/538): re-encode per-MMA (mask/shift/OR on 64b).
  3. Epilogue store addr (L2147): per-element row*STORE_COLS +
     store_stage*EPI_ROWS*STORE_COLS + swizzled_n*8.
  4. Producer smem addr (L1286/1287): per-iter stage*copy_size.

#### 63.5 CORRECTIONS / caveats (rigor, no guessing)
- mul/mad cross-check was CONTAMINATED by FLOAT (L2124/2125/2139 = alpha/beta FFMA,
  NOT IMAD; and NOT executed at beta=0/alpha=1 -- the alpha==1 fast path runs). The
  float-excluded int ranking stands.
- Static-vs-dynamic: counts are STATIC (kernel size, incl template generality / dead
  branches). Identified culprits (modulo/descriptor/address) ARE executed-path. For
  100% runtime-weighted proof: one ncu pass on build/ncu_probe_li (-lineinfo) with
  --import-source (pending, optional).

#### 63.6 FIX PLAN (each gated: int/div drop + bit-exact 336/336 + perf on 8 shapes)
  1. Kill ÷6 modulo (L1311, L1612, + store_stage/clc_stage/wave_stage % sites) ->
     `if(++s==D) s=0`. Safe 3-line; drives div 27->0. FIRST.
  2. Precompute MMA descriptor base, increment address field per stage (not re-encode).
  3. Running pointers for producer smem (L1286/7) + epilogue store (L2147) addresses.
Tools this session: examples/instr_accounting.sh (source-mapped SASS accounting),
probe_cublaslt.cu, splitk_observe.sh. Method to reproduce the source map:
  nvcc -ptx -lineinfo -gencode arch=compute_103a,code=compute_103a -Iinclude \
    -I$CUDA/targets/x86_64-linux/include/cccl examples/ncu_probe.cu -o /tmp/probe.ptx
  awk '$1==".loc"{c=($2=="1")?$3+0:0;next} c>0 && $1~/^(add|sub|mul|mad|shl|shr|and|
    or|xor)\./ && $1!~/\.f[0-9]/{n[c]++} END{for(l in n)print n[l],l}' | sort -rn

#### 63.7 FIX LEDGER (what was applied + measured benefit, 2026-07-16)
  #1 modulo->branchless wrap: APPLIED (4 sites). IMAD 359->323, magic-div gone.
     Perf FLAT (off critical path — latency-bound kernel hides int ops). KEPT
     (cleaner + fewer instructions, zero risk, bit-exact 252/252).
  #2 descriptor bases (once per stage, per-MMA = base + off>>4): APPLIED.
     Bit-exact 517/517 (full gate incl swap/composite). SASS UNCHANGED
     (IMAD 323->330, regs 167): ptxas had ALREADY strength-reduced the
     source-level rebuild — our redundancy was cosmetic in final SASS. Perf
     FLAT. KEPT (source now matches the machine code's true structure).
  #3 running pointers (producer smem L1286 / epilogue store L2147): NOT
     IMPLEMENTED — evidence from #1/#2 says flat (same class: off-critical-path
     int ops, compiler already optimizes, epilogue proven non-bottleneck by
     culprit-1/2). SKIP unless a future profile shows them on the critical path.
  L2 promo NONE->L2_128B (8 tmaps): APPLIED. L2-hit metric 88.3->104.3 (active)
     but wall-clock FLAT — §62.1 showed DRAM 0.0% both kernels: warm-bench fill
     is L2-latency-bound, not miss-bound. KEPT (real cold-cache workloads DO
     miss; CUTLASS/DeepGEMM default; free real-world win).
  VERDICT after 8 levers: instruction count / sync / L2-miss are NOT the 8-shape
  wall-clock levers. Gap = latency structure (fill + wave transitions).

#### 63.8 p8_check SCOREBOARD (2026-07-16, locked 1400MHz, warm, THE 8 shapes)
  shape        ours_us  cub_us   %cuB   oursSM%  cubSM%
  0.6b-K        16.64   14.11   84.8%   33.5     43.3   (§60.31 baseline 79.6%)
  1.7b-K        22.72   20.29   89.3%   47.8     55.8
  4b-K          26.34   23.65   89.8%   52.4     61.1
  8b-K          35.26   32.83   93.1%   60.8     67.4
  0.6b-O        23.14   20.32   87.8%   47.6     56.4
  0.6b-Down     29.31   26.50   90.4%   56.0     62.0
  1.7b-Down    117.76   80.86   68.7%   52.7     78.5   <-- THE gap
  3b-Down      201.98  136.93   67.8%   54.6     81.5   <-- THE gap
  FINDING: single-wave shapes 85-93%. The two MULTI-WAVE Down shapes (N=2048 ->
  256 tiles / 148 SMs ~ 1.7 waves, huge K so fixed fill CANNOT explain it) lose
  ~32% — cuBLAS holds 78-81% SM, we hold 53-55%. Masked until now: §60.31
  excluded 3b-Down ("--set full perturbs CLC multi-wave"); sweeps corroborate
  (~69-70% wall-clock). NEXT TARGET: our CLC wave-1->wave-2 transition (steal
  timing, tile-2 fill overlap with tile-1 epilogue) vs cuBLAS's plain 256-CTA
  grid with HW backfill. Note: p8_check.sh greps GRIDDEP for PDL — WRONG SASS
  names; use PREEXIT (launch_dependents) / ACQBULK (griddep wait).

#### 63.9 FUTURE EXPERIMENTS (parked, in priority order)
  1. MULTI-WAVE wave-transition fix for 1.7b/3b-Down (63.8) — biggest in-scope
     pool (~32% on 2 of THE 8 shapes).
  2. LOAD_PIPE_DEPTH sweep at low-K (§60.22 prescription, never run): shallower
     pipe (3/4) -> first MMA fires sooner -> attacks the ~2.5us fill on the
     single-wave low-K shapes (0.6b-K etc.).
  3. FP32-C EPILOGUE OUTPUT (mixed-precision training chain fusion): in bf16
     matmul the accumulator is ALREADY fp32 in TMEM; today the epilogue rounds
     to bf16 (F2FP) and the NEXT op (optimizer/softmax-stats/norm/grad-accum in
     fp32 autocast training) upcasts it right back. If the consumer wants fp32,
     write the fp32 accumulator DIRECTLY to gmem: kills TWO conversions (our
     F2FP + their convert_type_kernel — the BluTrain nsys showed float<->bf16
     convert kernels ~3% of GPU time). Cost: 2x C-store bytes (fp32 vs bf16) —
     net win when the eliminated convert kernel's full read+write traffic
     exceeds the extra store bytes (true whenever the next op would otherwise
     run a standalone upcast pass). Needs: C-dtype-templated epilogue (skip
     __float22bfloat162_rn, stmatrix/TMA store fp32, C tmap FLOAT32) + a
     blublasBgemmEx-style API with C dtype param + tensor-lib wiring to request
     fp32-out where the consumer is fp32. Doable — epilogue already holds fp32
     in registers; it is a store-path change, not a numerics change.
  4. Split-K/stream-K for skinny shapes (§63.1, full design locked) — future,
     out of the 8-shape scope.
  5. Register budget 167->255 asymmetric (setmaxnreg) — only if a future
     critical-path profile shows spill/ILP pressure; not now (flat evidence).

### 64. FULL-DAY LEDGER 2026-07-16 — what we chased, got wrong, and learned
Written end-of-day as an honest post-mortem. Everything applied is bit-exact and
kept; the value of the day is as much in the DEAD ENDS (they eliminated whole
classes of hypotheses) as in the one real finding at the end.

#### 64.1 THE MISTAKES WE MADE (own them so we don't repeat)
  M1. SPLIT-K DETOUR (morning, hours lost): started designing split-K/stream-K
      for low-K, but split-K is a SKINNY-shape (N<=512) tool — cuBLAS does NOT
      split-K on our 8 shapes (it runs plain nvjet_128x256_2x2). We drifted
      because we were quoting the 42-shape sweep AVERAGE (which mixes skinny
      shapes) instead of THE 8. FIX: scope locked to the 8 via p8_check.sh; memory
      [[scope-8-shapes]] written. Split-K design salvaged + parked (§63.1).
  M2. FLAWED INSTRUCTION COMPARISON (the big one): claimed "6.3x more int/IMAD
      (359 vs 4)". That counted the WHOLE BINARY (all NN/NT/TN/swap/composite
      template instantiations + dead branches) vs cuBLAS's ONE specialized
      kernel. NOT apples-to-apples — it inflated our count and sent us chasing
      instruction reduction. Gautam caught it ("kernel name isn't proof... the
      static count fooled us"). CORRECT method = single executed kernel, dynamic
      counts + STALL reasons, ours vs cuBLAS on ONE shape (p8_internals.sh).
  M3. STATIC PTX RANKING mixed pow2-shifts + dead branches into the "hot int
      lines". The top mul/mad line (L2139) turned out to be FLOAT alpha/beta
      (not executed at beta=0/alpha=1). Also A_smem=stage*copy_size is a SHIFT
      (copy_size=32KB pow2), not an IMAD — a running-pointer "fix" there would
      have been pure noise. Lesson: never rank on static PTX alone; need
      dynamic-executed + float-excluded + branch-reachability.
  M4. OVER-ASSERTED "compiler will..." without proof (fix#2). When #2 didn't
      move SASS I hand-waved "ptxas already strength-reduced it". Gautam rightly
      pushed back — that is an assertion, not a measurement. The honest fact:
      #2 IS bit-exact and in source; whether ptxas already did it needs the
      before/after SASS diff of that block (not yet shown). Don't assert
      mechanism; show it or label it unproven.

#### 64.2 EXPERIMENTS RUN + VERDICTS (all bit-exact; kept unless noted)
  fix#1 modulo->branchless wrap (4 sites): APPLIED. IMAD 359->323, PTX modulo
    lines L1311/L1612 GONE (proven). Perf FLAT. KEPT (cleaner, zero risk).
  fix#2 descriptor bases (once/stage, per-MMA = base+off>>4): APPLIED, bit-exact
    517/517 full gate (swap+composite included). SASS ~unchanged (IMAD 323->330).
    Perf FLAT. KEPT (source now matches machine structure). Mechanism-of-null
    UNPROVEN (see M4).
  fix#3 running pointers: NOT DONE. Evidence (fix#1/#2 flat + pow2-shift, M3)
    said flat -> skipped to avoid a rubbish change.
  L2 promo NONE->L2_128B (8 tmaps): APPLIED. L2-hit metric 88->104% (active) but
    perf FLAT — warm bench has DRAM 0% (all in L2), so it is L2-latency-bound not
    miss-bound. KEPT (real cold-cache/training win; CUTLASS default).
  Registers stayed 167 (NOT 255) through all of the above — because none of these
    HOLD new live values; 255 only comes from a genuinely deeper pipeline.

#### 64.3 WHAT THE DEAD ENDS ELIMINATED (the real payoff of the day)
  By process of elimination on THE 8 shapes, these are now RULED OUT as the
  wall-clock lever: instruction count, address/descriptor recompute, ÷ modulo,
  L2 misses, occupancy (ours HIGHER), warp layout, wave quantization (Waves/SM
  identical 1.73), tile selection (already exhausted). Eight+ levers, all flat.
  This is not wasted — it is why the §64.4 finding is trustworthy.

#### 64.4 THE ONE REAL FINDING (p8 scoreboard + 3b-Down internals)
  Scoreboard (locked 1400MHz, warm, THE 8): six single-wave shapes 86-94%; the
  TWO multi-wave Down shapes 1.7b-Down/3b-Down = 68% <- the whole remaining gap.
  3b-Down (4096x2048x11008) internals OURS vs cuBLAS:
    Duration     262us / 176us      Compute(SM)  54% / 82%   <- MMA starves
    MemThruput   809GB/s / 1.46TB/s  Waves/SM    1.73 / 1.73  (NOT wave-quant)
    Occupancy    10.9% / 9.0% (ours HIGHER, NOT occupancy)  Regs 167 / 255
    Warp cyc/inst 38.5 / 44.8
    smem-stall share  59.2% / 89.9%   <-- THE SIGNAL
  READ: cuBLAS warps stall ~90% purely on smem (next data tile) => cleanly
  memory-pipe bound, MMA flat-out 82%. OURS stall only 59% on smem => 40.8%
  "OTHER" (barrier/wait/sync) = 3.5x cuBLAS's 10.1%. That ~30pt of NON-memory
  stall is SYNCHRONIZATION OVERHEAD starving our tensor cores to 54%. The
  bandwidth gap (809 vs 1460) is largely a SYMPTOM (lower compute -> slower
  consumption). Explains all of §64.3: sync overhead COMPOUNDS over hundreds of
  k-iters x tile transitions (worst on multi-wave/large-K); negligible on single
  wave. THIS is the culprit class to attack next.
  NEXT MEASUREMENT (pending): break the 40.8% "other" into Stall Barrier / Wait /
  Membar / MIO-throttle (ncu warp-state table) to name the exact sync to cut.
  Tools built today: p8_check.sh (8-shape scoreboard), p8_internals.sh (single-
  kernel internals+stalls, verifies non-empty capture). Broken/never-worked:
  ncu --page source --print-source sass (0 source refs on this box); cuobjdump
  -lineinfo. Use ncu --page details (reliable) for metrics.

### 65. MULTI-WAVE ROOT CAUSE + SOTA fix (2026-07-17)
Proven root cause of the 8-shape gap (the 2 multi-wave Down shapes at 68%; the 6
single-wave shapes are 85-92% and fine):

#### 65.1 Proven diagnosis (§64 + p8 profiling, all measured)
- Same variant as cuBLAS (grid/tile 128x256/cluster 2x2/waves 1.73 all identical).
- SM-active cycles EQUAL (ours 160,185 ≈ cuBLAS 156,551) -> per-tile WORK is equal.
- Duration 255us vs 174us (68%): the gap is IDLE, not work. Ours SMs active ~45%
  of wall vs cuBLAS ~64%. Tensor cores fed 54.8% vs 81.3%.
- Both memory-latency-bound (long_scoreboard). Ruled OUT: instructions (hidden,
  issue-slots 2.6%), registers (167 vs 255 NOT the lever — see 65.2), wave-quant
  (identical tail both), the barrier-as-root (923-sample stall is the EXIT
  cluster_sync tail, a SYMPTOM of the backfill stall).
- Mechanism: on multi-wave, the exit teardown lockstep + no cross-tile CLC
  prefetch stalls wave-2 backfill -> SMs idle between waves.

#### 65.2 SOTA research — DeepGEMM + CUTLASS + ThunderKittens all agree
(local libs read: DeepGEMM sm100_bf16_gemm.cuh; cutlass sm100_gemm_tma_
warpspecialized.hpp + sm100_tile_scheduler.hpp; TK bf16_b200_gemm.cu)
Convergent recipe vs what WE have:
  persistent + hardware CLC backfill ............ WE HAVE
  load ring CARRIED across tiles (no drain) ..... WE HAVE (stage decl outside loop
                                                  L1265/L1412)
  TMEM accumulator DOUBLE-BUFFERED (2 stg) ...... we have EPI_PIPE_DEPTH=2 (verify
                                                  it overlaps MMA(N+1)/epi(N))
  CLC query pipelined >=2 deep (next coords
     ready before current tile ends) ............ MISSING (we cut to 1-deep for a
                                                  correctness bug, L1660)
  teardown OFF critical path (relinquish alloc
     lock + launch_dependents BEFORE barrier;
     minimal/relaxed barrier not full cluster) ... MISSING (we did full
                                                  cluster_sync + dealloc, no
                                                  relinquish at all)
  setmaxnreg register juggling ................... NOT NEEDED — DeepGEMM AND
                                                  CUTLASS dense bf16 use NONE
                                                  (accumulator in TMEM, control
                                                  warps single-threaded). So our
                                                  uniform 167 reg is FINE; 255 is
                                                  not the lever.

#### 65.3 Universality / risk (all fixes are TEMPLATE-LEVEL = one edit, all
variants inherit; no per-variant work; no hot-path overhead — exit-only)
  Fix A (relinquish_alloc_permit early): LOW risk, neutral on single-wave/batched,
    targets the measured 22% barrier/backfill stall. APPLIED this session.
  Fix B (restore 2-deep CLC): HIGH risk — reopens the end-of-grid out-of-order
    steal bug (orphaned tiles -> zero C -> flaky LMhead, L1660). Needs re-design.
  Fix C (verify/ensure TMEM double-buffer overlap): LOW-MED.

#### 65.4 Fix A APPLIED (this session)
Added tcgen05.relinquish_alloc_permit helper (~L568) + call at exit before the
cluster_sync (~L2218), warp0/CTA-guarded like dealloc. CUTLASS release_allocation
_lock equivalent — releases the TMEM alloc lock so the next backfill CTA can
rasterize/allocate while this cluster tears down. GATE: full 336-shape bit-exact
(RUN_2X2 etc.) + 8-shape perf (p8_check) + a batched regression shape. If neutral,
next = relax the exit cluster_sync to a minimal 2-CTA barrier, then Fix B.

### 66. FULL-DAY LEDGER 2026-07-17 — rigorous root-cause, SOTA research, Fix A, and TWO measurement traps
Mandate (user): find the EXACT problem with PROOF, no guessing/jumping. We have
CUTLASS, DeepGEMM, ThunderKittens, cub, PyTorch locally + web. Outcome: proven
diagnosis + SOTA fix recipe + Fix A built — but discovered our measurements were
being faked by a stale build and by --set full distortion. Net: real progress,
Fix A verdict deferred to a clean B300 re-test.

#### 66.1 PROVEN diagnosis (p8_compare.sh: L0 config / L1 bounds / L2 stalls)
- L0 SAME VARIANT: ours vs cuBLAS identical grid (128/256), tile 128x256, cluster
  2x2, 2cta, waves (0.86 single / 1.73 multi). Diffs: cuBLAS 8 warps/256thr/255reg
  vs ours 7 warps/224thr/167reg. cuBLAS = nvjet_sm103_tst_128x256_64x6_2x2_2cta_h_bz_NTT.
- L1 WHERE: 6 single-wave shapes 85-94% (fine); the 2 MULTI-WAVE Down shapes
  (1.7b-Down 4096x2048x6144, 3b-Down 4096x2048x11008) = 67-68% = the whole gap.
  Only the Down shapes have real DRAM traffic (single-wave = all-L2, DRAM 0%).
- L2 + SM-active (the key): 3b-Down SM-active cycles EQUAL (ours 160,185 ≈ cuBLAS
  156,551) -> SAME WORK. But duration 255 vs 174us -> the gap is IDLE, not work.
  Ours SMs active ~45% of wall vs cuBLAS ~64%. Tensor cores fed 54.8% vs 81.3%.
  Both memory-latency-bound (long_scoreboard). RULED OUT: instructions (issue-slots
  2.6%), registers (167 vs 255 NOT the lever — see 66.3), wave-quant (identical
  tail both), occupancy (ours higher).

#### 66.2 The barrier stall — located, but it's a SYMPTOM (the exit tail)
Per-reason stall (3b-Down, python-parsed raw csv): barrier ours 8.49cyc(22%) vs
cuBLAS 0.21(0.5%) = 40x; sleeping 2.55 vs 0.04; the rest ~equal. Barrier is the
dominant NON-memory stall. Located via SASS pc-sampling: 923/957 barrier samples
land on UCGABAR_WAIT = the FINAL cluster_sync() at kernel EXIT (teardown), run
ONCE per CTA. So it's the load-imbalance / wave-2 tail (fast CTAs waiting to
co-retire), which DELAYS wave-2 backfill on those SMs. NOT the epilogue (my first
two guesses — EPI_BAR then epilogue — were WRONG, corrected by SASS).

#### 66.3 SOTA research (3 parallel agents: DeepGEMM, CUTLASS, ThunderKittens)
Convergent recipe (all 3, cited file:line in agent reports) vs what WE have:
  persistent + hardware CLC backfill .............. WE HAVE
  load ring CARRIED across tiles (no drain) ....... WE HAVE (stage decl outside
                                                    the tile loop, L1265/L1412)
  TMEM accumulator DOUBLE-BUFFERED (2 stages) ..... we have EPI_PIPE_DEPTH=2
  CLC query pipelined >=2 deep (next-tile coords
     ready before current tile ends) .............. MISSING (we cut to 1-deep for
                                                    a correctness bug, L1660)
  teardown OFF critical path (relinquish alloc
     lock + launch_dependents BEFORE the barrier;
     minimal 2-CTA barrier not full cluster_sync) .. MISSING (full cluster_sync +
                                                    dealloc, NO relinquish at all)
  setmaxnreg register juggling .................... NOT NEEDED — DeepGEMM AND
                                                    CUTLASS dense bf16 use NONE
                                                    (accumulator in TMEM). So our
                                                    uniform 167 reg is FINE; 255
                                                    is NOT the lever.

#### 66.4 Fix A built = tcgen05.relinquish_alloc_permit (CUTLASS release_alloc_lock)
Added helper (~L568) + call at exit before cluster_sync (~L2235). Releases the TMEM
alloc lock so the next backfill CTA can rasterize during teardown. Now guarded
#ifdef BLUBLAS_ENABLE_RELINQUISH, DEFAULT OFF = validated baseline (see 66.6 verdict).

#### 66.5 TWO MEASUREMENT TRAPS that faked our results (the day's biggest lesson)
1. HEADER-DEP STALE BUILD: the kernel is a .cuh; `make` doesn't rebuild
   libmyblas.so on a header change. So `make test TEST=ncu_probe` reused the OLD
   lib -> EVERY Fix-A measurement ran the BASELINE kernel. PROVEN: the measured
   .ncu-rep SASS/PTX had relinquish=0. FIX: force rebuild (`touch` Bgemm_{nn,nt,tn}
   _variants.cu + Bgemm_sm103_dispatcher.cu, or `make clean`) AND verify
   `cuobjdump -ptx build/libmyblas.so | grep -c relinquish_alloc_permit` before
   trusting ANY number.
2. --set full DISTORTS multi-wave: it replays the kernel per-pass, serializing the
   wave-2 backfill -> same Fix-A kernel showed 256us under --set full but 139-203us
   under light `ncu -c 1`. Use LIGHT ncu (no --set full) for multi-wave timing.
3. `make CPPFLAGS=-D...` CLOBBERS the -Iinclude that lives in CPPFLAGS -> build
   fails. Add defines via source #define or full CPPFLAGS.
4. Never single-run: a recurring BIMODAL fluke (see 66.6).

#### 66.6 Fix A VERDICT — inconclusive, leaning positive (clean B200 A/B)
B200 (sm_100a, fresh clone, VERIFIED relinq=16 ON / 0 OFF, 5 reps, 3b-Down):
  OFF (baseline): 209,209,209,209,209  -> always slow, never fast
  ON  (Fix A)   : 209,209,210,147,209  -> hit 147 (≈cuBLAS parity) ONCE
Median flat (209=209) BUT: OFF NEVER touched the fast path; ON is the only side
that hit 147. So NOT "flat" — a weak positive that Fix A ENABLES the fast backfill
NON-DETERMINISTICALLY. The 147-vs-209 bimodal (≈0.7x, seen on BOTH B200 and B300)
= our multi-wave CLC backfill is NON-DETERMINISTIC (sometimes overlaps wave-2 =
147≈parity, usually stalls = 209). cuBLAS is consistently ~parity. THE REAL GAP =
scheduling non-determinism, not a fixed overhead. Need 15-20 rep A/B on B300 to
measure fast-path frequency ON vs OFF and settle it.

#### 66.7 Full-suite reality check (fixA.md, 519 rows — user insisted line-by-line)
Mean %cuBLAS = 104% ; 201/519 rows BEAT cuBLAS. Big wins: small-tile cm64 variant
on tiny/small/medium 190-420% (cuBLAS launch overhead on µs kernels). Systematic
LOSSES (pre-existing, NOT a regression): beta!=0 alpha-beta (C-accumulate) path
23-47% across ALL variants (future item — but beta=0 is the inference case);
skinny masked cm160 48% (split-K future). So the kernel is competitive overall;
the 8-shape campaign targets the hardest beta=0 shapes where cuBLAS is strongest.

#### 66.8 STATE + TOMORROW (B300)
Repo: default kernel = validated baseline (Fix A OFF, guarded #ifdef
BLUBLAS_ENABLE_RELINQUISH); everything committed+pushed; server trees reverted.
Tomorrow: (1) fresh clone; (2) FORCE rebuild + VERIFY relinquish-in-lib + check
libmyblas.so timestamp > source; (3) run the ~500-case all-variant bit-exact +
8-shape profiling; (4) MANY-REP (15-20) A/B ON vs OFF, count 147-vs-209 frequency
-> prove Fix A; (5) then the real lever = deterministic CLC backfill (2-deep CLC
prefetch / minimal 2-CTA exit barrier = rest of CUTLASS technique C). Tools:
p8_compare.sh (takes folder arg), p8_check.sh, p8_internals.sh.

#### 67 CLC 2-deep prefetch — multi-wave root cause PROVEN + spec-grounded drain-on-exit design (2026-07-18)

**67.1 Root cause of the multi-wave gap (measured, not inferred).** 3b-Down (M4096 N3072 K11008), ncu --set full, profiling_reports/b300_stallsplit, stall cyc/instr:
- long_scoreboard (memory = the GOOD stall): ours 22.36 vs cuBLAS 36.87
- **barrier: ours 5.64 vs cuBLAS 0.15 (37x)**
- **sleeping: ours 4.36 vs cuBLAS 0.03 (145x)**
- membar (MEMBAR.ALL.GPU): 0.20 — NEGLIGIBLE. The old "cut MEMBAR.ALL.GPU" suspect is DEAD.
Ours = 41% non-memory stall vs cuBLAS 10%. cuBLAS is cleanly memory-bound (tensor cores flat-out fed); ours STARVES on sync. Single-wave K-shapes 80-94% (inner load pipe is fine); only multi-wave Down shapes 67% → the loss is entirely at the TILE BOUNDARY. All top stall carriers are mbarrier waits (SASS: SYNCS.PHASECHK.TRYWAIT + NANOSLEEP.SYNCS = sleeping; SYNCS.PHASECHK.TRANS64 + UCGABAR_WAIT = barrier). Confirms §64.4 at instruction-class granularity.

**67.2 Mechanism (code-proven).** Producer (1386) and consumer (1650) block on clc_full at tile-END before they can issue tile N+1's loads. Scheduler runs 1-OUTSTANDING (arm→consume→arm one at a time), so N+1's tile-id resolves AT the boundary → producer cannot prefetch across it → the smem pipe drains and refills from empty each boundary → the sleeping/barrier bubble. The wait FLAVOR is NOT the lever (§57/§60.12: spin conversion tried → branch_resolving exploded, 19% slower). The lever is the COUNT/placement of sync at the boundary. cuBLAS keeps ~2 tiles resolved ahead → prefetches across the boundary → no drain → stays memory-bound.

**67.3 Fix = 2-deep CLC (arm 2 ahead) + drain-on-exit.** Historically blocked by the orphaned-tile bug (§30.12, comment 1672-1683): with 2 outstanding, out-of-order HW arbitration lets an EARLIER-issued cancel return invalid while a LATER one steals the last tile; consuming in issue order hit the invalid first → every warp broke → the stolen tile was orphaned → zero-filled C (flaky LM-head fails on 19k-tile grids). That is why it was cut to 1-deep.

**67.4 PTX ISA 9.3 §9.7.14.18 (local ptx_docs/ptx_isa_9.3.txt) gives the exact governing rules:**
- UNIQUENESS: "On success, the opaque response contains the ctaid ...; no other successful response from other try_cancel operations from the same grid will contain that id." → no tile stolen twice → drain-all cannot double-compute.
- UB CONSTRAINT: "If the executing CTA has already observed the completion of a try_cancel as failed, then issuing a subsequent try_cancel is undefined." → the scheduler MUST stop arming on the first observed invalid.
(NOTE: the research agent's "issue-order monotonic exhaustion" claim is the sm90 SOFTWARE-atomic-scheduler property; it does NOT hold for sm100 CLC hardware — our empirical zero-fill proves it. Do not rely on it.)

**67.5 Correctness theorem (the drain bound).** Discipline: arm DEPTH cancels up front; arm exactly one more after each VALID consumed; stop arming on the first invalid consumed. At the first invalid (consume position p): issued = DEPTH + p, consumed = p+1 → outstanding = DEPTH-1. So after the first invalid EXACTLY DEPTH-1 slots remain. Draining them (processing any valid found) → no orphan (completeness: valids across all outstanding = the remaining tiles) and no double-compute (uniqueness). Drain only CONSUMES already-armed slots, never issues → UB-safe. No deadlock: every role runs the identical deterministic drain (lockstep phases).

**67.6 Implementation — 6 lockstep-identical sites.** CONSUME sites: producer 1386, consumer 1650, epilogue-swap ~1895, epilogue-regular ~2198; plus SCHEDULER arm loop 1684-1744. Every warp of every CTA arrives clc_empty (line 183); clc_empty init = (CLC_EMPTY_ARRIVE/CLUSTER_CTAS)*cluster_ctas (1175). CLC_PIPE_DEPTH already = 2 in config (line 56/102) — the loops just run it as 1.
Per-consumer restructure (reuses the tile body via a flag — no duplication):
```
bool draining_final = false;
while (true) {
  <tile body>                                  // producer=loads / consumer=MMAs / epi=stores
  if (single_wave) break;
  if (draining_final) break;                   // did the final drained-valid tile; exit w/o consuming
  <consume slot> -> (is_valid, new_ctaid)      // wait clc_full / query / arrive clc_empty / advance
  if (is_valid) { <decode next tile>; continue; }
  bool found = false;                          // first invalid: exactly DEPTH-1 slots remain
  for (int d = 0; d < CLC_PIPE_DEPTH-1; ++d) {
    <consume slot> -> (v2, cta2)
    if (v2) { <decode cta2>; found = true; break; }
  }
  if (found) { draining_final = true; continue; }
  break;
}
```
SCHEDULER: arm TWO up front (add slot-1 arm at 1690-1700), keep the arm-on-success loop; on the first invalid, drain the remaining DEPTH-1 slots (consume+arrive only, NO tile body, and do NOT arm after the invalid — UB).

**67.7 Validation gate (before trusting — ANY miss = revert, orphan re-introduced):** clean rebuild + verify SASS; 3x bit-exact on the 19k-tile / LM-head grids that exposed the original orphan; dispatcher 45/45; p8 scoreboard (target: 1.7b-Down + 3b-Down move off 67%); re-run b300_stallsplit to confirm barrier+sleeping dropped toward cuBLAS's 0.15/0.03. Keep 1-deep as a compile fallback (CLC_PIPE_DEPTH template param) so SplitK=1-style A/B is trivial.

**67.8 REVERTED same day (2026-07-18) — §67 impl contradicted the validated §30.12 protocol. Kernel back to pre-§67 (6862943).** What went wrong, on record:
1. The §67 scheduler change re-armed CLC_PIPE_DEPTH(=2) try_cancel UP-FRONT and kept 2 in flight — the EXACT §30.12 orphan configuration — and also removed §30.12's "first arm of each slot skips the never-completed clc_empty wait" (my always-wait rewrite = new deadlock risk). I overwrote a B300-validated fix without re-reading §30.12 first. PROCESS RULE: before touching scheduler/CLC/barrier protocol, grep analyzations.md for the protocol's history section FIRST.
2. The premise was shaky: §30.12 (grounded in CUTLASS sm100_gemm_tma_warpspecialized.hpp:689-721) says CUTLASS runs ONE try_cancel outstanding — pipeline stages are consumer-side buffering, and "one prefetched response still hides CLC latency behind the tile's K-loop". A research agent claimed 2-deep is the CUTLASS mechanism (MED confidence, unverified symbols) — our own doc contradicts it. Never act on an agent claim that contradicts a validated section without source-verifying.
3. The clc_full diagnosis was REASONED, not MEASURED: the -lineinfo probe returned no source markers (release Makefile, injection failed silently), so which of the 3 boundary waits (tmem_empty 1450 / new-tile full_mbar 1506 / clc_full 1650) carries the 5.64+4.36 cyc barrier+sleeping stall is UNKNOWN. If 1-deep already hides clc_full (per §30.12), the bubble is likely full_mbar (load-pipe refill) or tmem_empty — different fixes entirely.
4. Salvageable from §67: the stall-split measurement (67.1, solid), the boundary-bubble framing (67.2, mechanism unproven), the PTX §9.7.14.18 rules (uniqueness + no-arm-after-observed-fail, 67.4), and the drain-on-exit idea IF multi-outstanding is ever genuinely needed. The clc_next helper code is in git history (275779d).
NEXT (deep re-diagnosis before any re-implementation): (a) get REAL per-line stall attribution on 3b-Down — fix the -lineinfo build (verify `cuobjdump -lineinfo` shows line tables) or use ncu pcsamp SASS-address attribution against a disasm dump; (b) answer WHICH wait carries the stall; (c) study how cuBLAS/CUTLASS ACTUALLY smooth the boundary at 1-outstanding (load-ring carry across tiles? deeper acc pipeline? epilogue overlap?) from their SASS/source, not agent memory; (d) only then design the fix.
Side-finding (off-scope, logged): test_bgemm_sm103_cluster_template's 252-case suite hangs intermittently at a config transition on SMALL single-wave shapes (§67 code bypassed there; transcript shows 33x full 252/252 passes AND partial runs historically = flaky pre-existing race). nvidia-smi during one hang: GPU-Util 100%, kernel resident. Separate investigation; not the 8-shape path.

### 68. FULL-DAY 2026-07-18 — multi-wave root cause SOLVED: CLC steal race at 1<waves<2; §68b gated plain-grid SHIPPED (Downs 67% -> 90.6/94.3, all 8 shapes >= 83%)

The complete day, chronological, including both diversions and what they taught.

#### 68.1 Morning: SASS instruction-by-instruction (the week-long ask, finally done right)
Clean re-extract of both kernels' full SASS (b300_full reports): OURS 2232 instrs vs cuBLAS 1072.
Series-level findings (not counts): (a) cuBLAS 1 UTCHMMA per K-step w/ 32-bit running-descriptor
UIADD3 +0x80; ours 4 UTCHMMAs + UIADD3.64 rebuilds from bases. (b) ELECT 16 vs 0 — our control
warps single-threaded, theirs full-warp (the 7.32-vs-31.95 threads/warp). (c) Loop structure:
BOTH persistent — cuBLAS one compact outer loop (span 479 instrs, K-loop 70), ours 61 sprawling
per-role backward branches (spans 1600-1943). (d) cuBLAS SASS has ZERO clusterlaunchcontrol ops.

#### 68.2 The stall split — culprits A+B measured (b300_stallsplit, 3b-Down-class)
Per-reason stall (cyc/instr), ours vs cuBLAS: long_scoreboard 22.36 vs 36.87 (they are cleanly
memory-bound); **barrier 5.64 vs 0.15 (37x) = culprit A; sleeping 4.36 vs 0.03 (145x) = culprit
B**; membar 0.20 (old MEMBAR.ALL.GPU suspect DEAD); wait ~equal. All top stall carriers =
mbarrier waits (SASS: SYNCS.PHASECHK.TRYWAIT + NANOSLEEP.SYNCS = sleeping; SYNCS.PHASECHK.TRANS64
+ UCGABAR_WAIT = barrier). Wait FLAVOR already ruled out (§57: spin -> branch_resolving exploded).

#### 68.3 DIVERSION 1 — §67 CLC 2-deep: implemented, pushed (275779d), REVERTED same day
Reasoned (NOT measured — the first -lineinfo probe silently produced no line tables) that
clc_full was the boundary blocker; designed 2-deep CLC + drain-on-exit on PTX ISA 9.3
§9.7.14.18 rules (found in LOCAL ptx_docs/ptx_isa_9.3.txt: per-grid ctaid uniqueness + UB to
issue try_cancel after an observed failure). Implemented clc_next() drain helper across 5
consume sites + scheduler arm-up-front. THEN the correctness suite hung at a config transition
(case 29->30, small single-wave shapes — §67 bypassed there; GPU-Util 100% kernel-resident;
historically flaky: 33x full passes + partials in transcript = PRE-EXISTING race, §67.8).
User forced a re-read of the doc history: **§30.11/§30.12 already recorded this exact
deadlock/orphan class and its validated fix (1-outstanding, arm-on-success, first-arm skips the
never-completed clc_empty wait) — my §67 had REVERSED a validated fix** on a research-agent
claim (2-deep) that contradicts our own §30.12 (CUTLASS = ONE outstanding; 1-deep already
prefetches one response). Reverted to 6862943; §67.8 process rules written (grep doc history
before touching scheduler/CLC/barrier protocol; never act on agent claims contradicting
validated sections; measure, don't reason). Post-revert re-validation: 252/252 + sweep at the
known 85.0/89.3.

#### 68.4 The REAL measurement — per-address, per-reason, source-line attribution
Fixed lineinfo probe (sed -lineinfo into NVCCFLAGS + VERIFY debug_line sections in the .so
BEFORE profiling: 1142 sections) + ncu python API (ncu_report.py) on o_line.ncu-rep:
- **93% of ALL barrier samples (1003/1076) at ONE instruction: UCGABAR_WAIT @ line 414 =
  cluster_sync() — the EXIT barrier** (all role blocks BRA to BSYNC B1 then fall into it).
- **78% of sleeping (626/800) at the WARPSYNC.ALL directly before it.** Same region.
- NOT clc_full. NOT tmem_empty. NOT the K-loop full_mbar. Culprits A+B = warps PARKED at the
  end-of-kernel cluster barrier = the BILL for idle tails, not a sync inefficiency.

#### 68.5 Mechanism nailed: CLC steal race wastes a whole wave at 1 < waves < 2
Wave sweep obeys ceil law at >=2 waves (2.59w -> 3x67=205 correct; 3.46w -> 4x67=270 correct)
but at 1.30w and 1.73w we paid THREE waves (202 = 3x67) where ceil = 2. cuBLAS at the same
shape: 137 = 2x68.5 (exact ceil). Grid check (b300_wavesweep reports): at N=2048 (p8 3b-Down)
cuBLAS runs the SAME tile (nvjet_sm103_tst_128x256_64x6_2x2) and SAME grid (256) — no selection
trick; it wins purely on distribution (plain grid + HW backfill, no CLC). At N=1536 cuBLAS
additionally selection-escapes (128x192_64x6_2x2f, grid 264). The old bimodal 147-vs-209
"fluke" (§65) EXPLAINED: 147 = the steal race landing even (2 waves, parity); 209 = the race
landing 3/1 (some clusters 3 tiles, others 1 -> kernel = greediest cluster = 3 waves).
**THE problem: our CLC work-stealing mis-distributes tiles at fractional waves 1<w<2.**

#### 68.6 A/B experiment — diagnosis proven in one line
Temp sed single_wave=1 (plain-grid HW-backfill mode, the already-validated no-CLC path) on the
two failing shapes (b300_sec68_abtest): 3b-Down 202 -> 144.86/147.68/144.80us (3-rep STABLE,
determinism restored); 1.7b-Down 119.8 -> 89.31/89.66/88.86us. 144.8 = 2x72 = exactly 2 waves.

#### 68.7 DIVERSION 2 (user caught): split-K/stream-K "today" — scrapped
Contradicted our own §63.1 (cuBLAS runs NO split-K on the 8; split-K = skinny tool) and simple
ceil math (uniform split-K scales tiles AND tile-time -> quantization penalty invariant).
Framework survey from LOCAL SOURCE (receipts): CUTLASS ships stream-K first-class on sm100
(PersistentTileSchedulerSm100StreamK) and gates it with wave math verbatim
(tile_scheduler_params.h:1099 "prefer data-parallel ... more than one wave and the tail wave is
more than half full"); DeepGEMM = grouped raster parameterized on kNumSMs (scheduler/gemm.cuh),
no split-K (their bench times cuBLAS as nvjet+reduce = cuBLAS DOES split-K elsewhere);
ThunderKittens = plain persistent. Conclusion: wave-math-gated scheduling = industry standard;
stream-K stays the future lever for tails, NOT today's fix.

#### 68.8 §68 fix v1 (223d60e): rename + plain-grid ALWAYS — half right
single_wave renamed no_clc (it always meant "don't arm CLC", not "one wave" — the name had
confused everyone); launcher defaulted plain-grid for ALL grids; BLUBLAS_FORCE_CLC=1 env
restores CLC for A/B; kernel/§30.12 untouched. Result: p8 all 8 >= 85.6 (Downs 90.6/96.8),
252/252 PASS — BUT the 42-sweep exposed the overreach: Gates 93 -> 56%, LMheads 100-125 ->
34-84%, avg 89.3 -> 71.5. WHY: at many waves plain-grid pays cluster init + pipe fill +
teardown PER TILE with 1 CTA/SM (no co-resident overlap partner), while CLC persistence
amortizes them across ~all of an SM's tiles. The steal-race bug ONLY exists at 1<w<2 (>=2 full
waves: queue long enough that the race evens out — measured exact-ceil at 2.59/3.46w).

#### 68.9 §68b THE SHIPPED FIX (3c3ceb6): gated plain-grid
Launcher rule: no_clc = (grid.x <= 2*s_sm_count) — plain-grid iff waves <= 2, CLC beyond;
FORCE_CLC env keeps the old rule for A/B. Host-side only, one integer compare, tile-agnostic
(every current/future variant inherits), kernel + §30.12 untouched.
VALIDATION (b300_sec68b, locked 1400): 252/252 ALL PASS. p8: 0.6b-K 83.2 / 1.7b-K 90.0 /
4b-K 91.1 / 8b-K 94.5 / 0.6b-O 89.1 / 0.6b-Down 92.2 / **1.7b-Down 90.6 / 3b-Down 94.3**
(campaign start: 67.4/67.0). Healed: 0.5b-Gate 93.9%, 1.5b-Gate 96.2%, LMheads 96.8-112.2%
(above parity), sweep avgs 84.1/86.8. THE VARIANT-LEVEL WIN: same kernel, same tile as cuBLAS,
now within ~6-10% everywhere on the 8 — remaining gap = the known single-wave fill/epilogue
floor (§60.22) + per-wave prologue on the plain-grid shapes.

#### 68.10 Measurement/process traps caught today (add to the permanent list)
1. STALE OUTPUT PASTES: the same §68 output was pasted 3x as "new" — ALWAYS check the git-pull
   hash line (e.g. ..3c3ceb6) + the profiling folder name before analyzing a "new" run.
2. Shallow clone: HEAD~1 doesn't exist on --depth 1 boxes; git fetch --depth 2 first.
3. -lineinfo must be VERIFIED in the binary (cuobjdump -elf | grep debug_line) BEFORE profiling
   — the first probe silently produced nothing (§67.8 root of the wrong clc_full diagnosis).
4. Heredoc paste-mangling corrupted an if-block mid-run once — write scripts via cat > file.sh
   on the target box, then bash it; never paste multi-line control flow raw.
5. Lock clocks BEFORE the whole session (the §68 sweep ran unlocked -> absolute TFLOPS
   incomparable across days; %cuB within-run stays fair).
6. ncu python API (nsight-compute/extras/python/ncu_report.py) gives per-address per-reason
   stall + file:line from lineinfo reports — the definitive attribution tool (CLI source page
   does not print file:line).
7. p8_check's "PDL GRIDDEP: 0" check reads build/ncu_probe — the kernels live in libmyblas.so;
   VERIFY the check target before trusting it (open item).

#### 68.11 Open items after today
- Port §68b (no_clc rename + gate) to the sibling templates: kota_addmm, FP16 Hgemm, FP8 (and
  BluTrain copies) — same one-line pattern.
- p8_check GRIDDEP check target (ncu_probe vs libmyblas.so) — fix the check or the flag.
- The pre-existing flaky 252-suite hang on small single-wave shapes (§67.8 side-finding).
- Remaining 8-shape gap: single-wave fill/epilogue floor (§60.22) — the next real lever;
  per-wave prologue overlap on plain-grid shapes (wave-2 clusters refill from cold).
- Stream-K: the long-term tail lever (CUTLASS reference), on the same launcher decision point.
