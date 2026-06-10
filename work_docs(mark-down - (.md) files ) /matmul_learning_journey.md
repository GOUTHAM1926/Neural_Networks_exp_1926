# Matmul Learning Journey — from BLAS basics to GEMM optimization (with hands-on benchmarks)

**Author:** Goutham Reddy (Gautam_1926)

## Part 1 — What is BLAS (and what it is NOT)

**BLAS = Basic Linear Algebra Sub-programs.**

Two things BLAS is **not**:

- BLAS ≠ all of linear algebra. Linear algebra is vast (solving systems, decompositions, eigenvalues, SVD…).
- BLAS ≠ only matmul.

The insight: all those vast linear-algebra operations are built on **three most
fundamental operations**, and BLAS covers exactly these three "levels":

1. **Level-1** — vector–vector operations (e.g. `axpy: y = αx + y`, dot product)
2. **Level-2** — matrix–vector multiplications (e.g. `gemv: y = αAx + βy`)
3. **Level-3** — matrix–matrix multiplications (e.g. `gemm: C = αAB + βC`)

Everything above these (solving linear systems, **LU** decomposition, **Cholesky**
factorization, **eigenvalues**, **SVD**…) lives in a higher library called
**LAPACK = Linear Algebra PACKage**, which is **built on top of BLAS** — it calls
BLAS kernels internally to do the heavy lifting. BLAS/LAPACK are classically C/Fortran
(C++) packages.

In deep learning we **mostly use Level-3 (matrix–matrix multiply at large scale)** —
that's the workhorse.

### The library map (CPU package ↔ NVIDIA ↔ AMD)

| Role | CPU package | NVIDIA (CUDA) | AMD (ROCm) |
|---|---|---|---|
| 3 BLAS levels (L1/L2/L3) | **BLAS** | **cuBLAS** | **rocBLAS** |
| building-block / template GEMM kernels | — | **CUTLASS** | **Tensile** |
| higher linear algebra (solve, LU, Cholesky, eig, SVD) | **LAPACK** | **cuSolver** | **rocSolver** |

So: `BLAS ↔ cuBLAS ↔ rocBLAS` and `LAPACK ↔ cuSolver ↔ rocSolver`. LAPACK / cuSolver /
rocSolver sit **on top of** BLAS / cuBLAS / rocBLAS.

### Open-source status (important)

- **CPU packages** BLAS and LAPACK → **open-source.**
- **NVIDIA:** cuBLAS → **closed** (the real matmul kernels + tile-variants are hidden);
  cuSolver → **closed**; **CUTLASS → open-source** (the only open NVIDIA one).
- **AMD:** **all three open** — rocBLAS, Tensile, rocSolver.

*(CUTLASS = NVIDIA's open building blocks for GEMM; ships the **CuTe** layout library
(`cutlass/include/cute/`) and the newer Python **CuTe DSL** — more notes for B300 work.)*

---

## Part 2 — Why L1 = O(n), L2 = O(n²), L3 = O(n³), and what "memory-bound vs compute-bound" means

The deciding ratio is **Arithmetic Intensity = (math operations) ÷ (data fetched from RAM)** =
how much math you do per number you drag out of slow memory.

### Level-1 — vector ⊕ vector → **O(n)**

```
x:  [ ][ ][ ][ ][ ][ ][ ][ ]   ← n elements
        ↓ one multiply/add each
y:  [ ][ ][ ][ ][ ][ ][ ][ ]   ← n elements
```

A vector is a 1-D row of n elements; you touch each once with constant work →
`n × const = O(n)` operations. Data touched ≈ `2n`–`3n`.

- **Math:** ~n · **Data:** ~3n (read X, read Y, write Z) → **ratio ≈ n / 3n ≈ 0.33**
- For every ~3 numbers pulled from RAM you do ~1 op → **math cores starve → memory-bound.**

### Level-2 — matrix × vector → **O(n²)**

```
     A (n×n)            x        y
  [ row → n ]         [ ]      [y0] = row0 · x   (n mul-adds)
  [ row → n ]   ×     [ ]   =  [y1] = row1 · x
  [  ...     ]        [ ]      [..]
   n rows              n        n outputs
```

Each output = one row (n) dotted with the vector (n) = n work; n outputs → **n² ops.**

- **Math:** n² · **Data:** matrix dominates ≈ n² → **ratio ≈ n²/n² = 1**
- Every matrix element `A[i][j]` used **exactly once** → still **memory-bound.**

### Level-3 — matrix × matrix → **O(n³)** (the special one)

```
        B (n×n)
        [ | | | ]   each column has n elements
   A    [ c c c ]        C (n×n)
 [row]× [ o o o ]  =  [ C[i][j] = rowA · colB ]   ← n mul-adds per cell
 n rows [ l l l ]     n×n cells
```

Each `C[i][j]` = n mul-adds; there are n² outputs → **n³ ops.** But data = 3 matrices ≈ **n².**

- **Math:** n³ · **Data:** 3n² → **ratio ≈ n³/n² = n**
- If n = 1000, that's **1000 math ops per number fetched** → **math cores finally run flat-out → compute-bound.**

**The key difference:** in GEMM each data element gets **reused n times** — load a tile
into fast shared memory once, do n math ops on it. That reuse is why **GEMM is the only
BLAS op that can saturate the GPU's tensor cores / peak FLOPS.** L1/L2 just stream memory.

```
          arithmetic intensity (math per byte)
 L1:  ~0.33   → memory-bound  (starving)
 L2:  ~1      → memory-bound  (starving)
 L3:  ~n      → COMPUTE-bound (saturated)   ← grows with n, the whole point
```

---

## Part 3 — The optimization ladder: naive → peak (the 8 versions)

Every step fights the **same enemy: don't let the math units wait on memory.**

**V0 — naive triple loop (`ijk`).** Correct but slow: `B[k][j]` walks **down a column**
(stride N) → cache miss every step. All n³ work, but the reuse is wasted because data
falls out of cache before reuse.

**V1 — loop reorder (`ikj`).** Make the inner loop walk memory **contiguously** (see Part 4).
Same n³ math, ~contiguous access → several× faster. *Lesson: memory layout beats op-count.*

**V2 — blocking / tiling.** Chop matrices into small tiles that **fit in cache**; finish all
the math on a tile while it's hot, then move on → reuse jumps → moves toward compute-bound.

**V3 — GPU shared-memory tiling.** The "cache you control" on a GPU is **shared memory**.
Cooperatively load A-tile + B-tile global→shared (coalesced), `__syncthreads()`, compute
from shared. Watch for **coalescing** (warp reads consecutive addresses) and **bank conflicts**
(fix with padding/swizzle).

**V4 — register blocking.** Each thread computes a small **register tile** (e.g. 8×8),
reusing each loaded value across the tile. Memory pyramid: `global → shared → registers`.

**V5 — vectorization + software pipelining.** `float4` (16-byte) loads; **double-buffer** —
while computing tile k, prefetch tile k+1 (`cp.async` + multi-stage) → hides memory latency.

**V6 — Tensor Cores.** `mma.sync` does a whole small matrix-multiply per instruction (TF32/
FP16/BF16 inputs, FP32 accumulate). Where the huge FLOPS come from. `ldmatrix` loads operands.

**V7 — finishing touches.** **split-K** (slice K across blocks to fill the GPU; combine via
atomicAdd → non-deterministic), **fused epilogue** (`α·C + β·C_old` + bias + activation in the
same kernel), and **autotuning / variant selection** (pick the tile for the shape — exactly
what a dispatcher does).

One line: **naive → reorder → tile → shared-mem → register tiles → vectorize+pipeline →
tensor cores → split-K + fused epilogue + autotune.**

---

## Part 4 — The six loop orderings (and why `ikj` is best on CPU)

Three nested loops over `i, j, k` → **3! = 6** orderings: `ijk, jik, ikj, kij, jki, kji`.
**All 6 are numerically correct** (it's the same sum of products, just reordered — provided
C starts at 0). They differ only in **memory access speed**, decided by the **innermost** loop
(which sets the two *streamed* arrays) and the **middle** loop (which sets the *fixed* array's
reload pattern).

Row-major strides: `A[i][k]` → +1 over k ; `B[k][j]` → +1 over j, +N over k ; `C[i][j]` → +1 over j, +N over i.
Miss rate: contiguous `+1` ≈ **1/L** (L = elements per cache line, ~16 floats/64B) · column `+N` ≈ **1** (miss every access) · fixed-in-inner = 0 inside, paid at the middle loop.

| order (out→mid→in) | inner | A[i][k] | B[k][j] | C[i][j] | fixed array's reload (middle loop) | ≈ total misses | rank |
|---|---|---|---|---|---|---|---|
| **ikj** | j | fixed | stream `+1`→1/L | stream `+1`→1/L | A fixed, mid=k → **row, contiguous** ✓ | **≈ 2n³/L** | 🥇 best |
| **kij** | j | fixed | `+1`→1/L | `+1`→1/L | A fixed, mid=i → **column** ✗ | 2n³/L **+ n²** | 🥈 |
| **ijk** | k | `+1`→1/L | **`+N`→1** | fixed | C fixed, mid=j → **row** ✓ | ≈ **n³** | 🟡 |
| **jik** | k | `+1`→1/L | **`+N`→1** | fixed | C fixed, mid=i → **column** ✗ | ≈ n³ **+ n²** | 🟡 |
| **kji** | i | **`+N`→1** | fixed | **`+N`→1** | B fixed, mid=j → **row** ✓ | ≈ **2n³** | 🔴 |
| **jki** | i | **`+N`→1** | fixed | **`+N`→1** | B fixed, mid=k → **column** ✗ | ≈ 2n³ **+ n²** | 🔴 worst |

**Read it by the innermost var:**

- inner **j** (`ikj`,`kij`): B and C stream contiguously, A is a fixed scalar → **fastest.**
- inner **k** (`ijk`,`jik`): A contiguous, C in a register, but **B walks a column** → medium.
- inner **i** (`jki`,`kji`): A *and* C both walk columns → **worst.**

**The middle-loop tiebreak (a subtle but real point):** within a pair sharing an inner var,
the one whose **middle** loop reloads the *fixed* array contiguously (row) wins. So
`ikj > kij` (in `kij`, A is reloaded **down a column** → extra misses), `ijk > jik`, `kji > jki`.

**Ranking (fewest→most misses):** `ikj < kij < ijk < jik < kji < jki`.
**`ikj` is the single best** for row-major — the only ordering where *all three* arrays are
accessed contiguously.

---

## Part 5 — The "C must start at 0" rule (and the two ways to satisfy it)

Every ordering uses `C[i][j] += A[i][k]*B[k][j]`, and `+=` **reads C first**. So **every
`C[i][j]` must hold 0 before its first `+=`**, or you accumulate onto garbage. This is the
universal rule (and it's *why* GEMM has the `β·C` term — β=0 overwrite, β=1 accumulate).

Two ways to satisfy it, and the orderings split on it by **whether `k` is innermost**:

| ordering | each `C[i][j]` is touched… | how to make it start at 0 |
|---|---|---|
| **ijk, jik** (k innermost) | all K times **back-to-back** (one burst) | per-cell init / a register `sum` (or pre-zero) |
| **ikj, kij, jki, kji** (k not innermost) | K times **scattered** across the loop | **must pre-zero the whole C matrix first** |

Timeline intuition (2×2):

```
ijk:  C00 C00 | C01 C01 | C10 C10 | C11 C11     ← each cell FINISHED in one burst
ikj:  C00 C01 | C00 C01 | C10 C11 | C10 C11     ← each cell's pieces SCATTERED
            ↑ C00's 2nd piece comes after C01 → C00 must already be 0 and persist
```

- **k innermost (`ijk`/`jik`):** cell finishes in one burst → you can accumulate in a
  **register** (`sum=0; for k sum+=…; C=sum;`) and write C once → C never needs whole-matrix zeroing.
- **k not innermost (the other 4):** a cell's pieces are scattered → you **cannot** use a single
  register → C lives in memory and is **read-modify-written**, so the **whole C must be pre-zeroed.**

---

## Part 6 — Hands-on benchmarks (real numbers on this machine: RTX 3060 / sm_86, CUDA 13.0)

Files: `Desktop/Cpp_experiments_1926/cpp_experiments/cinit_vs_regsum_bench.cu` and
`ikj_vs_ijk_bench.cu`. All with warmup + averaging over many reps.

### Experiment A — for `ijk`: register `sum` vs "zero whole C then accumulate into C"
>
> Question: is it better to (1) accumulate in a register and write C once, or (2) zero all of C
> and read-modify-write C every k?

```
CPU (N=256):  REG 1.816 ms   vs  ZACC 1.829 ms   →  REG 1.01× (≈ tie)
GPU (N=2048): REG 21.4 ms    vs  ZACC 60.8 ms    →  REG 2.85× faster
```

- **CPU ~tie:** with `-O3`, in ZACC the compiler **register-promotes** `C[i][j]` across the
  inner k-loop (its address is loop-invariant) → both compile to nearly the same code; only the
  small O(n²) zeroing pass differs (negligible vs O(n³) compute).
- **GPU REG 2.85×:** here ZACC was a genuine global read-modify-write each k (this config didn't
  promote it) + a separate `cudaMemset`. **Caveat:** this used a 16×16 block shape that happens
  to favor the register kernel (better B-reuse) — see Experiment B's correction.
- **Takeaway:** register accumulation is **never worse**, sometimes much better → it's the right
  default. (This is *why* real GEMM accumulates the tile in registers and writes C once.)

## Part 7 — Glossary

- **GEMM** = General Matrix Multiply (`C=αAB+βC`). **GEMV** = matrix×vector. **AXPY** = `y=αx+y`.
  **addmm** = GEMM+bias. **TRMM** = triangular MM. **TRSM** = triangular solve. **SYRK** = `A·Aᵀ`.
- **NN/NT/TN/TT** = transpose flags for A,B (N=no, T=transpose).
- **Arithmetic intensity** = math-ops ÷ bytes-moved. **Coalescing** = a warp reads consecutive
  addresses. **Bank conflict** = shared-memory serialization. **Occupancy** = warps resident per SM.
- **MMA** = tensor-core multiply. **cp.async** = async global→shared copy. **split-K** = slice K
  across blocks. **epilogue** = post-multiply α/β/bias/activation. **register blocking** = each
  thread computes a small tile held in registers.

---

*Status: living doc — extend it as the journey continues (tiled GPU GEMM next: watch 60 ms →
a few ms). Benchmarks reproducible via the two `.cu` files in the experiments folder.*
