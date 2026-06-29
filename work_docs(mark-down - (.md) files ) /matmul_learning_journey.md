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

## Part 4 — The six loop orderings (and why `ikj` is best on CPU(with no optimization flags and no openmp parallelism across cpu threads))

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

## Part 8 — My Hands-on Journey: Coding the Naive `ijk` Loop (V1)

Yesterday, I embarked on coding the raw, naive `ijk` matrix multiplication kernel completely from scratch in C++. I documented every single step and decision I made directly inside my [`matmul_experiments.cpp`](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/matmul_experiments.cpp) file. Here is the step-by-step breakdown of my manual coding experimentation:

### Step 1: Solving the Matrix Allocation Problem

I started by allocating the $A$, $B$, and $C$ matrices, which led to a massive realization about memory layout. I had three choices:

1. **Stack 2D Arrays (`float A[M][K]`):** Simple, but causes a stack overflow for large dimensions (like $1000 \times 1000$).
2. **Heap 2D Pointer Arrays (`float** A`):** This is terrible for performance! Each row is a separate memory allocation, scattering the rows across RAM. Worse, it creates a **"pointer chasing"** nightmare. If I tried to copy this to the GPU, it would crash because the GPU cannot follow CPU pointers.
3. **Flat 1D Arrays on the Heap (`float* A = new float[M*K]`):** **This is the winner.** It scales to any size, ensures memory is contiguous (cache-friendly), and is exactly how the GPU operates. I access elements using math: `A[i * K + k]` instead of `A[i][k]`.

### Step 2: Perfecting Matrix Printing

I wrote loops to print the matrices to the console to verify my math. However, I ran into an issue where an extra space was printed after the last element of every row (e.g., `[1 1 1 ]`). I couldn't just insert `if` statements inside the `std::cout <<` chain.

- **The Fix:** I split the print statements and added a conditional check: `if (j < K - 1) std::cout << " ";`. This resulted in perfectly formatted rows like `[1 1 1]`.

### Step 3: Pausing Execution and Hardware Profiling

I wanted to actually *see* my CPU allocating the matrices in real-time. I added a `std::cin.get()` at the end of the program to freeze it before the memory was freed with `delete[]`.

- **The Discovery:** Opening Linux `htop`, I couldn't find my `./a.out` process initially. I learned that for a $1000 \times 1000$ matrix, it only took 12 MB of RAM, which is invisible on my 32GB system! I had to scale up to $10000 \times 10000$ matrices to see a massive **1.2 GB memory spike** in `htop`. I also learned to use `Shift + M` in the `top` command to sort by memory.

### Step 4: Building a Professional Benchmark Harness

I needed to measure how fast my `ijk` loop was running. I didn't just wrap a simple timer; I built a robust benchmarking setup:

- **`std::chrono`:** I used `std::chrono::high_resolution_clock` to measure execution down to the nanoseconds, safely converting it to a readable double in seconds.
- **The Accumulation Bug:** I initially ran my timer in a loop 10 times to get an average. But `C[i*N+j] += ...` accumulates! By run 10, the matrix held garbage astronomical numbers.
- **The Fix:** I learned to use `std::memset(C, 0, size_C)` before *every single run* to properly zero the output matrix.
- **Warmup Runs:** I implemented 3 "warmup" runs before the 10 timed runs to prime the CPU cache.

### Step 5: Testing with Realistic Deep Learning Shapes

Rather than benchmarking arbitrary dimensions like $2000 \times 2000$, I updated my parameters to represent a real Transformer workload. I used the dimensions of the **MLP (Feed-Forward) layer in GPT-2 Small**:

- $M = 1024$ (Sequence Length / Context Window)
- $K = 768$ (Embedding Dimension)
- $N = 3072$ (MLP Expansion Size, which is $4 \times K$)

### Step 6: Unleashing Multi-Threading with OpenMP

My raw C++ triple loop was only using 1 core of my CPU while the rest slept. I fixed this by including `#include <omp.h>` and adding a single compiler pragma:

```cpp
#pragma omp parallel for
for (int i = 0; i < M; ++i) { ... }
```

- **Why the Outer Loop?** I learned that putting the pragma on the outermost `i` loop is critical. If I put it on the innermost `j` loop, the CPU would waste all its time creating and destroying threads millions of times (thread overhead), making it agonizingly slow.

*(This completely sets the stage for testing GCC optimization flags like `-O3` and `-march=native`, which I documented deeply in my `compiler_flags_gcc.md` notes!)*

### Step 7: The Benchmark Results (Scaling the Optimizations)

After setting up the professional benchmarking harness and verifying the code, I ran a massive series of 14 experiments to see exactly how much performance I could squeeze out of my CPU using only compiler flags and OpenMP. I tested three specific matrix shapes:

1. **Small Square:** $M, N, K = 200$
2. **GPT-2 MLP Expansion:** $M=1024, N=768, K=3072$
3. **Large Square:** $M, N, K = 2000$

Here is the exact journey of how the execution time plummeted as I added optimizations:

#### 1. Baseline: No Optimizations (`g++ matmul_experiments.cpp`)

Running the raw C++ code with zero compiler help. Only one CPU thread is doing all the work.

- **$200 \times 200 \times 200$:** ~0.1348 secs
- **GPT-2 Shape:** ~6.32 secs
- **$2000 \times 2000 \times 2000$:** ~17.0 secs

*(Observation: Looking at the Linux system monitor, only 1 thread was at 100% usage while the rest were idle. The OS scheduler occasionally hopped the workload to different threads to spread the heat, but at any given moment, only one thread was executing the math).*

#### 2. Adding `-O2` (`g++ -O2 matmul_experiments.cpp`)

*Reminder: It is a capital letter "O", not a zero "0"!*
This flag enables automated generic software optimizations like Instruction Scheduling, Common Subexpression Elimination, Loop Optimizations, Dead Code Elimination, and Strict Function Inlining.

- **$200 \times 200 \times 200$:** ~0.00235 secs *(Massive drop!)*
- **GPT-2 Shape:** ~3.8 secs
- **$2000 \times 2000 \times 2000$:** ~7.0 secs

#### 3. Adding `-O3` (`g++ -O3 matmul_experiments.cpp`)

This flag enables even more aggressive software optimizations, particularly loop unrolling and aggressive vectorization logic.

- **$200 \times 200 \times 200$:** ~0.00065 secs
- **GPT-2 Shape:** ~1.04 secs
- **$2000 \times 2000 \times 2000$:** ~1.8 secs

#### 4. Adding `-march=native` (`g++ -O3 -march=native matmul_experiments.cpp`)

While `-O3` optimizes the software logic, `-march=native` allows the compiler to use my specific CPU's advanced physical hardware. My Intel i7 14700K supports **AVX2** (Advanced Vector Extensions), allowing it to load 256 bits (8 `float32` numbers) at once and compute them in a single clock cycle (SIMD Vectorization). *(Note: Our AMD EPYC 9754 server CPU supports AVX-512, which can do 16 `float32` numbers at once!)*

**The Symbiotic Relationship (The "AHA!" Moment):**
I ran experiments to see if I could use these flags independently, and proved they must be used together for maximum speed:

- **`-march=native` alone (~17.6 seconds):** Exactly the same as `-O0`! `-march=native` just tells the compiler what hardware I have. But without `-O3` giving permission to optimize, the compiler doesn't actually use the hardware.
- **`-O2` + `-march=native` (~7.4 seconds):** The compiler knows I have AVX2 hardware, but the `-O2` flag physically refuses to run the loop vectorization algorithms (running `g++ -Q -O2 --help=optimizers` proves `-ftree-loop-vectorize` is disabled in GCC 11 at `-O2`). So it just does the standard slow loop.
- **`-O3` alone (~1.8 seconds):** `-O3` gives permission to vectorize, but without `-march=native`, it assumes I have an ancient, generic CPU to ensure the code works anywhere. It uses old, basic 128-bit SSE vectorization instead of my modern AVX2.
- **`-O3` + `-march=native` (~1.10 seconds):** Together, they unlock the true potential. `-O3` gives the order to vectorize, and `-march=native` gives it the AVX2 weapons to do it!

- **$200 \times 200 \times 200$:** ~0.00048 secs
- **GPT-2 Shape:** ~0.555 secs
- **$2000 \times 2000 \times 2000$:** ~1.10 secs

#### 5. Unleashing OpenMP Multi-Threading (`-fopenmp`) (Experiments 14-18)

I activated `#pragma omp parallel for` in the code and compiled with the `-fopenmp` flag to enable multi-threading. Instead of 1 thread, all available threads (28) grab chunks of the outer `i` loop and work concurrently.

**The OpenMP Mental Model:**
If we have 28 threads and $M=2000$, OpenMP splits the rows evenly ($2000 / 28 \approx 71.4$). The first 12 threads take 72 rows, the rest take 71 rows.
For a given row in Matrix A, each thread multiplies that row by EVERY column in Matrix B to fill out one entire row of the output Matrix C. Because every thread is writing to a different row in C, there are no memory race conditions! Also, since all threads are reading Matrix B at the exact same time, Matrix B gets locked into the ultra-fast L3 CPU cache, completely eliminating the RAM bottleneck.

**The OpenMP Implementation Discoveries:**

OpenMP gives you two completely different categories of tools to control parallelism, and they work in fundamentally different ways:

**Category 1: Compiler Directives (Pragmas) — handled at compile-time**

- `#pragma omp parallel for` is called a **"pragma directive"** (or just "compiler directive"). The word `pragma` comes from the Greek word "pragmatikos" meaning "relating to action."
- A pragma is **not C++ code**. It is a secret instruction written directly to the compiler. When GCC sees `#pragma omp`, it physically rewrites your loop into multi-threaded code during compilation.
- **Does NOT need `#include <omp.h>`!** Because it's a compiler instruction, not a C++ function call. The compiler already knows what to do with it.
- **Requires `-fopenmp` flag** in the compilation command to activate. Without this flag, GCC treats the entire `#pragma` line as a harmless comment and ignores it completely — your code still compiles and runs fine, just single-threaded.

**Category 2: Runtime Functions (`omp_*()`) — executed while the program is running**

- Functions like `omp_set_num_threads()`, `omp_get_max_threads()`, and `omp_get_num_procs()` are **real C++ functions** that execute while your program is actually running (at runtime).
- **ALL of them REQUIRE `#include <omp.h>`!** — whether they are setters or getters. Without this header, the compiler has no idea what these functions are and will throw `"not declared in this scope"` errors.
- The available runtime functions:
  - `omp_set_num_threads(N)` — **SET**. Tells OpenMP to use exactly N threads for subsequent parallel regions.
  - `omp_get_max_threads()` — **GET** (read-only). Returns the maximum number of threads (= number of logical cores).
  - `omp_get_num_procs()` — **GET** (read-only). Returns the number of processors/cores available on the hardware.
  - Note: **`omp_set_max_threads()` does NOT exist!** There is no such function in OpenMP.

**Key equivalence:** Writing `omp_set_num_threads(omp_get_max_threads());` before a `#pragma omp parallel for` is **functionally identical** to just writing `#pragma omp parallel for` alone with no function call. Both will use all 28 threads. The only difference is that the explicit version makes your intent visually clear in the code.

**How OpenMP Allocates Threads (Important!):**

- **Default behavior:** If you just use `#pragma omp parallel for` without calling any function, OpenMP **always creates the maximum number of threads** equal to the number of logical CPU cores (28 on my i7-14700K). It does this **every single time**, regardless of what else is running on the system.
- **OpenMP has ZERO awareness of other programs.** It does not check if cores are "available" or "occupied." Even if all 28 cores are already running Chrome, VS Code, and other heavy programs at 100% load, OpenMP will still blindly create 28 threads. The **Linux kernel scheduler** is the only boss that decides who runs on which core — it juggles all programs using rapid time-slicing (switching between processes every few milliseconds).
- **`omp_set_num_threads(N)` does NOT stop other processes.** It only controls how many threads OpenMP itself creates. It cannot kick other programs off cores.

**The Final Benchmark Results (Scaling OpenMP):**

- **$2000 \times 2000 \times 2000$ (Experiment 14):**
  - OpenMP only: ~1.35 secs
  - OpenMP + `-O3` + `-march=native`: ~0.92 secs

- **GPT-2 Shape ($1024 \times 768 \times 3072$) (Experiments 15 & 16):**
  - OpenMP only (Exp 16): ~0.448 secs
  - OpenMP + `-O3` + `-march=native` (Exp 15): ~0.283 secs *(Down from 6.32 secs with no flags!)*

- **Small Square ($200 \times 200 \times 200$) (Experiments 17 & 18):**
  - OpenMP only (Exp 17): ~0.001243 secs
  - OpenMP + `-O3` + `-march=native` (Exp 18): ~0.00045 secs

**The Small Shape Saturation Limit:**
For small shapes like $200 \times 200$, applying extreme optimizations (OpenMP + `-O3` + `-march=native`) only reduced the time to `0.00045 secs` (0.45 milliseconds). At this microscopic timescale, the computational limit is saturated. The overhead of waking up and distributing work to 28 threads takes a massive percentage of the total execution time, and OS background jitter causes the time to fluctuate slightly. Parallelism benefits massive deep learning shapes incredibly, but for tiny shapes, it doesn't give much extra benefit over standard `-O3` logic.

**The Final Result:** Through sheer compiler magic and simple multi-threading, I took a naive `ijk` matrix multiplication for $2000 \times 2000$ from **17.0 seconds down to 0.92 seconds**!

*(All of these manual coding implementations and benchmarking logic are documented directly in my source file: [`matmul_experiments.cpp`](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/matmul_experiments.cpp))*

---

### The Great OpenMP Mystery: "Does OpenMP Disable `-O3` Optimizations?" (Deep Investigation)

While experimenting with `omp_set_num_threads(1)`, I noticed something very confusing:

- **Without OpenMP**, running with `-O3 -march=native` gave **~1.10 seconds**.
- **With OpenMP set to 1 thread**, running with `-O3 -march=native -fopenmp` gave **~7.16 seconds**!

Same code, same flags, same single thread doing all the work — but a **6.5× slowdown**! My first guess was that OpenMP was somehow blocking or disabling some of `-O3`'s optimizations (like loop interchange or vectorization). I decided to investigate this properly with real compiler evidence instead of guessing.

#### Proof 1: Checking if `-fopenmp` disables any `-O3` optimizer flags

I ran GCC's internal optimizer flag dump to compare what optimizations are enabled with and without `-fopenmp`:

```bash
$ g++ -Q -O3 -march=native --help=optimizers | grep -E "loop-interchange|loop-vectorize"
  -floop-interchange            [enabled]
  -ftree-loop-vectorize         [enabled]

$ g++ -Q -O3 -march=native -fopenmp --help=optimizers | grep -E "loop-interchange|loop-vectorize"
  -floop-interchange            [enabled]
  -ftree-loop-vectorize         [enabled]
```

**Result:** Every single optimization flag is **identical** in both cases. `-fopenmp` does NOT turn off any `-O3` flags at the compiler-options level.

#### Proof 2: Checking GCC's vectorization report

I asked GCC to tell me exactly how many loops it vectorized in each version:

```bash
$ g++ -O3 -march=native -fopt-info-vec-all test_noomp.cpp -S -o /dev/null 2>&1
  note: vectorized 0 loops in function.
  missed: not suitable for strided load   ← B[k*N+j] walks columns!

$ g++ -O3 -march=native -fopenmp -fopt-info-vec-all test_omp.cpp -S -o /dev/null 2>&1
  note: vectorized 0 loops in function.
  missed: not suitable for strided load   ← same problem!
```

**Result:** **NEITHER version vectorizes the matmul loop!** The `ijk` loop order has `B[k*N+j]` in the inner loop, which walks down a column of B (stride = N). GCC reports this as `"not suitable for strided load"` and refuses to vectorize it. So vectorization was never the difference between the two runs!

#### Proof 3: Comparing the actual x86 assembly (the ultimate proof)

I dumped the final machine code that GCC produces for both versions and compared the innermost loop. Here is what the CPU actually executes:

**Without OpenMP (inner loop):**

```asm
.L5:
    vmovss   (%rax), %xmm1            ; load A[i*K+k]
    vfmadd231ss (%rdx), %xmm1, %xmm0  ; xmm0 += A[i*K+k] * B[k*N+j]
    addq     $4, %rax                  ; move to next A element
    addq     %rdi, %rdx               ; move to next B row (stride N)
    vmovss   %xmm0, (%rcx)            ; store result to C[i*N+j]
    cmpq     %rax, %rsi
    jne      .L5                       ; loop back
```

**With OpenMP (inner loop):**

```asm
.L5:
    vmovss   (%rax), %xmm1            ; load A[i*K+k]
    vfmadd231ss (%rdx), %xmm1, %xmm0  ; xmm0 += A[i*K+k] * B[k*N+j]
    addq     $4, %rax                  ; move to next A element
    addq     %rdi, %rdx               ; move to next B row (stride N)
    vmovss   %xmm0, (%rcx)            ; store result to C[i*N+j]
    cmpq     %rax, %rsi
    jne      .L5                       ; loop back
```

**They are BYTE-FOR-BYTE IDENTICAL!** The actual math that the CPU executes is exactly the same code in both versions. GCC applies the exact same optimizations to both.

#### So what actually caused the 7-second slowdown?

The answer is **OpenMP runtime overhead**. Even when you set `omp_set_num_threads(1)`, the OpenMP runtime still does real work on every parallel region entry:

1. **`__builtin_GOMP_parallel()` call:** The compiler wraps your loop body into a separate function and calls the GOMP (GNU OpenMP) runtime library to launch it. Even with 1 thread, this function call happens every time.
2. **Thread chunk calculation:** OpenMP still computes `M / num_threads` and `M % num_threads` to figure out which rows to assign — visible in the assembly as `idivl` (integer divide) instructions.
3. **Data packing:** All your variables (`A`, `B`, `C`, `M`, `N`, `K`) get packed into a special `struct .omp_data_s` and passed through a pointer to the worker function (visible in the GCC IR dump as `.omp_data_o.2.M = M_6(D)` etc.).
4. **Runtime synchronization:** The GOMP library still sets up synchronization barriers, even for a single thread.

This overhead is tiny per-call, but it runs inside the benchmark loop (10 timed runs × the full matmul), and the overhead accumulates. For the non-OpenMP version, GCC compiles a clean, direct triple-nested loop with zero runtime overhead — just raw math instructions.

#### The Final Verdict

**OpenMP does NOT disable any `-O3` optimizations.** The generated machine code is identical. The slowdown with `omp_set_num_threads(1)` is 100% caused by the OpenMP runtime library overhead (`GOMP_parallel` calls, data struct packing, chunk calculation). When you use many threads (like 28), the parallelism benefit massively outweighs this overhead. But with only 1 thread, you pay all the overhead cost with zero parallelism benefit.

---

I have officially pushed the CPU to its absolute limits with the naive `ijk` ordering. Tomorrow, the GPU experiments begin, alongside testing the other 5 loop orderings (`ikj`, `kij`, etc.)!

---

## CPU Matrix Multiplication (ijk) — Ultimate Benchmark Summary

This section serves as the master record of all 28 experiments performed to optimize the naive `ijk` matrix multiplication loop using GCC compiler flags (`-O2`, `-O3`, `-march=native`) and OpenMP multi-threading.

### 1. Master Comparison Table

| Optimization Level / Flags | M,N,K = 200 (Small) | GPT-2 Shape (Medium) | M,N,K = 2000 (Large) |
| :--- | :--- | :--- | :--- |
| **Naive (No Flags)** | `~0.1348s` | `~6.32s` | `~17.00s` |
| **`-O2`** (Basic optimization) | `~0.00235s` | `~3.80s` | `~7.00s` |
| **`-O3`** (Loop optimizations) | `~0.00065s` | `~1.04s` | `~1.80s` |
| **`-O3 -march=native`** (Vectorization) | `~0.00048s` | `~0.555s` | `~1.10s` |
| **OpenMP Only** (All Threads) | `~0.00124s` | `~0.448s` | `~1.35s` |
| **OpenMP + `-O3 -march=native`** | `~0.00045s` | `~0.283s` | `~0.92s` |

### 2. OpenMP Thread Scaling Analysis (TRUE RESULTS)

*(Configuration: M,N,K = 2000 | Flags: `-O3 -march=native -fopenmp`)*

| Thread Count | Execution Time |
| :---: | :--- |
| **1 Thread** | `~6.63s` |
| **2 Threads** | `~3.43s` |
| **3 Threads** | `~2.64s` |
| **4 Threads** | `~1.92s` |
| **5 Threads** | `~1.59s` |
| **7 Threads** | `~1.04s` |
| **9 Threads** | `~1.19s` |
| **15 Threads** | `~0.84s` |
| **20 Threads** | `~0.74s` (Sweet Spot!) |
| **All Threads (28)** | `~0.91s` |

### 3. Four Core Insights & Engineering Takeaways

1. **The Vectorization Rule (`-O3` + `-march=native`)**
   For SIMD vectorization to happen, you *must* use both flags. `-march=native` only detects the hardware (AVX2), but without `-O3` giving the compiler permission to optimize, vectorization won't happen. If one is missing, vectorization fails.

2. **The "Small Workload Saturation" Principle**
   For tiny matrices ($200 \times 200$), compiler flags alone (`-O3 -march=native`) drop the time to sub-milliseconds (`0.00048s`). At this microscopic scale, the actual computation is saturated. Adding OpenMP multi-threading to small workloads provides almost zero benefit, because the OS overhead of waking up and distributing work to 28 threads takes more time than the actual math!

3. **The Thread Scaling "Sweet Spot" (Physical Cores vs Logical Cores)**
   When scaling OpenMP threads from 1 to 28 on a large workload ($2000 \times 2000$), performance improves drastically up to **20 threads (`0.74s`)**. This is the physical sweet spot for the i7-14700K (which has exactly 20 physical cores: 8 P-cores + 12 E-cores). Pushing it to all 28 threads (using the 8 Hyper-Threaded logical cores) actually *slows it down* to `0.91s`. Hyper-Threading doesn't add physical math units, it just shares existing ones, which adds thread-switching overhead!

4. **Strategy Dictates Optimization**
   You must choose your optimization strategy based on the workload size:
   - **Small Workloads:** Use Compiler Flags ONLY (`-O3 -march=native`). Avoid OpenMP overhead.
   - **Heavy Workloads:** Use Compiler Flags AND OpenMP. This is where parallelization shines and provides massive reductions in execution time.

---

## Part 9 — Loop Parallelization Depth Analysis (i vs j vs k loops)

A massive deep-dive was conducted to understand *exactly* what happens at the hardware and software levels when moving `#pragma omp parallel for` between the `i` (outer), `j` (middle), and `k` (inner) loops of the naive `ijk` matrix multiplication.

### Case 1: OpenMP on the `i-loop` (The Outer Loop) — The Best Approach

```cpp
#pragma omp parallel for
for (int i = 0; i < M; ++i) { 
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) { C[i * N + j] += A[i * K + k] * B[k * N + j]; }
    }
}
```

- **Thread Assignment:** OpenMP divides the `M` rows among the threads (e.g., if M=2000 and threads=28, each thread gets ~71 complete rows).

- **Software Overhead (MINIMAL):** Threads are woken up, work, and hit a synchronization barrier exactly **1 time**.
- **Hardware Memory Traffic (NONE):** Thread 0 writes to the top 71 rows of Matrix C, while Thread 1 writes to rows 72-142. Because they write to completely different memory locations, there is **no memory write contention**. Also, reading from Matrix B simultaneously causes no traffic jams because the hardware MESI cache protocol flags simultaneous reads as "Shared" (S-state), allowing perfect parallel reads.

### Case 2: OpenMP on the `j-loop` (The Middle Loop) — The Double Halt

```cpp
for (int i = 0; i < M; ++i) { 
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) { C[i * N + j] += A[i * K + k] * B[k * N + j]; }
    }
}
```

- **Thread Assignment:** The Master Thread owns the `i-loop`. For *every single row*, it wakes up all 28 threads and splits the columns (`j`) among them.

- **The Software Halt (Implicit Barrier):** At the end of every parallel block, OpenMP enforces an implicit barrier. If Thread 1 finishes early, it **cannot** go to the next row. It must wait for all threads to finish, hit the barrier, and go to sleep. The Master Thread then increments `i` and wakes them all up again. This wakes/halts 28 threads **M times (2000 times)**.
- **The Hardware Halt (False Sharing):** When threads write to different columns in the *same row* (e.g., `C[0][0]`, `C[0][1]`), they are writing to the **same 64-byte Cache Line**. The MESI protocol requires a core to obtain the "Modified" (M) state via a "Read-For-Ownership" (RFO) hardware signal before writing. This forces the cache line to violently bounce between cores on the CPU bus, halting execution.
- **The "Default Schedule" Discovery:** In default OpenMP (`schedule(static)`), columns are chunked (Thread 0 gets cols 0-70). Because they are 71 floats apart, False Sharing is mostly avoided. Through an ablation test (`ablation_test.cpp`), it was proven that the penalty of moving to the `j-loop` was **~0.55s purely from the 2000 Software Barriers**, and forcing hardware False Sharing (`schedule(static, 1)`) added another **~0.18s** of pure hardware stall.

### Case 3: OpenMP on the `k-loop` (The Inner Loop) — Apocalyptic Performance

```cpp
for (int i = 0; i < M; ++i) { 
    for (int j = 0; j < N; ++j) {
        #pragma omp parallel for
        for (int k = 0; k < K; ++k) { C[i * N + j] += A[i * K + k] * B[k * N + j]; }
    }
}
```

- **Thread Assignment:** The Master Thread controls rows and columns. It wakes up 28 threads to calculate the dot product for a **single output cell**.

- **Software Halt (CATASTROPHIC):** The thread pool is created, halted, and destroyed **M × N times (4,000,000 times)**!
- **Fatal Thread Race Condition (Data Race):** All threads are trying to write to the exact same cell (`C[i * N + j]`) at the exact same microsecond. Since `+=` is actually three instructions (LOAD, ADD, STORE), multiple threads will read `0`, add their partial sum, and overwrite each other's results. The final matrix will contain completely wrong mathematical answers.

---

## Part 10 — The `ikj` Loop: Hands-on Experiments and Deep Memory Architecture

After mastering the naive `ijk` loop and all its OpenMP parallelization behaviors, the next step was to implement the `ikj` loop ordering — the one our theory (Part 4) predicted would be the **fastest** on CPU due to superior cache spatial locality. This part documents the complete hands-on journey of coding, benchmarking, and deeply understanding **why** `ikj` is faster at the hardware level.

### Step 1: Writing the `ikj` Loop from Scratch

I manually rewrote the matmul kernel in [`matmul_experiments.cpp`](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/matmul_experiments.cpp) to use the `ikj` loop ordering:

```cpp
// ikj loop ordering
for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}
```

**Important:** I initially had a typo bug in the warmup loop where I wrote `+` instead of `*` in the expression (`A[i * K + k] + B[k * N + j]` instead of `A[i * K + k] * B[k * N + j]`). This was caught during code review and fixed.

### Step 2: Proving `ikj` Produces the Same Mathematical Output as `ijk`

My first reaction was: *"How can rearranging the loops produce the same result?"* So I worked through a visual proof using a tiny 2×2 matrix:

```text
      Matrix A             Matrix B             Matrix C (Starts at 0)
    [ A00   A01 ]        [ B00   B01 ]        [ C00   C01 ]
    [ A10   A11 ]        [ B10   B11 ]        [ C10   C11 ]
```

**The `ijk` loop (The "Dot Product" way):**

Computes one cell of C completely, then moves to the next:

- `C00 += A00 * B00`, then `C00 += A01 * B10` → C00 is 100% finished.
- `C01 += A00 * B01`, then `C01 += A01 * B11` → C01 is 100% finished.
- (Repeat for C10, C11...)

**The `ikj` loop (The "Broadcast" way):**

Takes one scalar from A, multiplies it across an entire row of B, and sprays partial results across a row of C:

- k=0: `C00 += A00 * B00`, `C01 += A00 * B01` → (both cells get partial results)
- k=1: `C00 += A01 * B10`, `C01 += A01 * B11` → (both cells now complete)

**The math is identical** because addition is commutative ($a + b = b + a$). Both orderings compute the same sum of products for every cell — they just add the terms in a different order.

**Critical requirement:** Because `ikj` builds up C's values in scattered pieces (a cell gets touched in multiple passes), **C must be pre-zeroed** before the loop starts (using `std::memset(C, 0, size_C)`). This is different from `ijk` where you can use a register accumulator.

### Step 3: The First `ikj` Benchmark — Raw Speed Without Any Flags

```bash
$ g++ matmul_experiments.cpp && ./a.out
Time taken (naive-ikj loop, no optimizations, M=N=K=2000): 12.2381 secs
```

**Recall:** The naive `ijk` loop with zero flags took **~17.0 seconds**. The `ikj` loop, with **zero compiler help**, is already **~1.4× faster** (17.0 → 12.2 seconds) — just from rearranging the loop order!

### Step 4: Why `ikj` is Faster — The Deep Hardware Explanation

#### The Core Hardware Rule: The "Cache Line"

As we discovered in our earlier tests (when we built `false_sharing_test.cpp`), **RAM never sends just one number to the CPU.** According to the **Intel® 64 and IA-32 Architectures Software Developer's Manual (Volume 3A, Chapter 11: Memory Cache Control)**:

> *"The L1 data cache, L2 cache, and L3 cache use a 64-byte line size."*
> *"Data is transferred between memory and the caches in blocks called cache lines... A cache line is the smallest unit of memory that can be transferred to or from a cache."*

When the CPU asks RAM for a single `float` (4 bytes), the hardware sends an entire **64-byte Cache Line** = **16 adjacent floats**.

**Official Intel Documentation Links:**

- Intel® 64 and IA-32 Architectures Software Developer's Manuals: [https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- Intel® 64 and IA-32 Architectures Optimization Reference Manual (Chapter 8: Multicore, Chapter 9: Memory Optimization): Same download page.

#### What Happens in the `ijk` Inner Loop (Bad Access Pattern)

```cpp
// Inner loop: k increments
C[i * N + j] += A[i * K + k] * B[k * N + j];
//                             ^^^^^^^^^
//                             k changes → B walks DOWN a column (stride = N = 2000)
```

When `k` goes from 0 to 1, `B[k * N + j]` jumps from `B[0*2000 + j]` to `B[1*2000 + j]` — that's a jump of **2000 floats = 8000 bytes = 125 cache lines**. Every single `k` increment triggers a **cache miss** because the next element is nowhere near the previous one. The 15 other floats loaded in each cache line are completely wasted.

#### What Happens in the `ikj` Inner Loop (Perfect Access Pattern)

```cpp
// Inner loop: j increments
C[i * N + j] += A[i * K + k] * B[k * N + j];
//  ^^^^^^^^^                    ^^^^^^^^^
//  j changes → C walks RIGHT    j changes → B walks RIGHT
//  (+1 stride, contiguous!)     (+1 stride, contiguous!)
```

When `j` goes from 0 to 1, both `C[i*N + j]` and `B[k*N + j]` move to the **very next float in memory** (stride = 1). When the CPU loads `B[k*N + 0]`, it gets `B[k*N + 0]` through `B[k*N + 15]` for free! The next 15 iterations are all **cache hits** — the data is already sitting in L1 Cache.

And `A[i * K + k]` doesn't change at all inside the inner `j` loop — it's a fixed scalar, which the compiler will keep in a register.

**Result:** `ikj` has a cache miss rate of approximately **1/16** (one miss per 16 floats = one miss per cache line), while `ijk` has a cache miss rate of approximately **1** (one miss per access). This is why `ikj` is faster with zero compiler flags.

### Step 5: Cache Lines vs. Vectorization — Two Completely Different Things

This was a massive "AHA!" moment. I initially confused cache line loading with SIMD vectorization. They are **completely different hardware subsystems**:

| Property | Cache Line Loading | SIMD Vectorization |
|---|---|---|
| **What moves data** | Memory Bus (hardware wires) | CPU Execution Units |
| **From → To** | **RAM → L1 Cache** | **L1 Cache → CPU Registers** |
| **Unit size** | Always 64 bytes (16 floats) | Depends: SSE=16B (4 floats), AVX2=32B (8 floats) |
| **Controlled by** | **Hardware** (always happens, no flags needed) | **Compiler** (needs `-O3 -march=native`) |
| **Can you disable it?** | **NO** — hardwired into the motherboard | YES — use `-O0` or `-fno-tree-vectorize` |

**The Memory Pyramid:**

```
                    ┌─────────────┐
                    │  Registers  │  ← Vectorization loads data HERE (L1→Regs)
                    │  (~0.3 ns)  │
                    ├─────────────┤
                    │  L1 Cache   │  ← Cache Lines load data HERE (RAM→L1)
                    │  (~1 ns)    │
                    ├─────────────┤
                    │  L2 Cache   │
                    │  (~4 ns)    │
                    ├─────────────┤
                    │  L3 Cache   │
                    │  (~10 ns)   │
                    ├─────────────┤
                    │  Global RAM │
                    │  (~100 ns)  │
                    └─────────────┘
```

**Key insight:** Cache line loading happens at the **bottom** of the pyramid (RAM→L1). Vectorization happens at the **top** (L1→Registers). They operate on completely different levels! Even with `-O0` (no vectorization), the hardware still loads 64 bytes per cache miss. That's why `ikj` is faster than `ijk` even with zero compiler flags.

### Step 6: Write-Back vs. Write-Through Cache Policies

When the CPU writes data, it does NOT write directly to Global RAM (that would be catastrophically slow at ~100ns per write). Modern Intel and AMD CPUs use a **Write-Back** cache policy.

**Write-Back (What Intel/AMD Actually Use):**

1. CPU wants to write to `C[0]`.
2. CPU first does a **READ** — loads the entire 64-byte cache line containing `C[0]` into L1 Cache (if not already there). This is called a **Read-For-Ownership (RFO)**.
3. CPU modifies `C[0]` **inside the L1 Cache only**. Global RAM has no idea `C[0]` changed.
4. The cache line is marked as **"Modified"** (the **M** in MESI protocol).
5. **Sometime later** (when the cache line is evicted or another core needs it), the CPU flushes the entire 64-byte block back to RAM.

**Write-Through (Older/Simpler Systems):**

- Every write immediately goes to both the cache AND Global RAM simultaneously.
- Much simpler but much slower — every `+=` would cost ~100ns.
- Modern CPUs do NOT use this for general data.

**This is exactly why False Sharing causes hardware stalls:** When Thread 0 writes to `C[0]` and Thread 1 writes to `C[1]`, they are both modifying the **same 64-byte cache line**. The MESI protocol forces ownership to bounce between cores — each core must issue an RFO to claim exclusive access before writing.

### Step 7: When Does False Sharing Actually Apply?

**False Sharing ONLY applies when there are multiple threads.** With a single thread, there is only one core accessing the cache line, so there is no contention and no RFO bouncing. The single core just writes to L1 Cache in ~1ns with zero hardware stalls.

So when benchmarking `ikj` vs `ijk` without OpenMP (single-threaded), the performance difference is **purely** from cache miss rates (spatial locality), NOT from False Sharing.

### Step 8: Temporary Variable (Register Accumulator) Optimization — Which Loop Orders Support It?

To use a temporary register variable (`float temp = 0; for(...) temp += ...; C[i*N+j] = temp;`), the innermost loop **must be `k`**. This is because `k` being innermost means `C[i][j]` coordinates are frozen inside the inner loop, so you can accumulate into a single register and write to C only once.

| Loop Order | Inner Loop | Can Use Register Accumulator? | Why? |
|---|---|---|---|
| **`ijk`** | k | ✅ YES | `C[i][j]` is fixed while `k` runs |
| **`jik`** | k | ✅ YES | `C[i][j]` is fixed while `k` runs |
| **`ikj`** | j | ❌ NO | `C[i][j]` changes every iteration (`j` moves) |
| **`kij`** | j | ❌ NO | `C[i][j]` changes every iteration |
| **`jki`** | i | ❌ NO | `C[i][j]` changes every iteration |
| **`kji`** | i | ❌ NO | `C[i][j]` changes every iteration |

For `ikj` (and the other 4 non-k-innermost orderings), multiple cells of C are being built up simultaneously in scattered pieces. There is no single register that can hold the partial sum — C **must** live in memory and be read-modify-written, which is why **C must be pre-zeroed**.

However, in `ikj`, the compiler can still be clever: `A[i * K + k]` is a **fixed scalar** in the inner `j` loop. The compiler will load it once into a register and broadcast it across all the `B[k*N + j]` multiplications — this is called **scalar broadcast** and it's very efficient.

### Step 9: Alternative Parallelization Techniques (Beyond OpenMP)

While OpenMP is the simplest way to add CPU parallelism, there are several other techniques at different levels:

| Technique | Level | Description |
|---|---|---|
| **OpenMP** | Thread-level (easiest) | Compiler directives (`#pragma`) for shared-memory parallelism |
| **std::thread / pthreads** | Thread-level (manual) | Raw C++/POSIX threading — full control but much more code |
| **MPI (Message Passing Interface)** | Process-level | For distributed computing across multiple machines |
| **SIMD Intrinsics** | Instruction-level | Direct AVX2/AVX-512 intrinsics (`_mm256_fmadd_ps`) — maximum control |
| **CUDA / HIP** | GPU-level | Massively parallel execution on GPU hardware |
| **TBB (Threading Building Blocks)** | Task-level | Intel's task-based parallelism library |

For our matmul experiments, we are systematically climbing this ladder: first mastering single-threaded optimization (loop reordering, compiler flags), then thread-level parallelism (OpenMP), and eventually GPU-level (CUDA).

---

## Part 10 Summary — Benchmark Table (ikj vs ijk)

| Configuration | `ijk` Time | `ikj` Time | Speedup |
|---|---|---|---|
| **No flags** (M,N,K=2000) | ~17.0s | ~12.24s | **1.39×** |

*(More `ikj` benchmarks with `-O3`, `-march=native`, and OpenMP are to be continued in future experiments.)*

---

### Step 10: The `volatile` Keyword — and How It Connects to the Register Accumulator

When I implemented the `float sum` temporary variable (register accumulator) in the `ijk` loop, I also studied the `volatile` keyword that was used in our earlier [`false_sharing_test.cpp`](file:///home/blu-bridge016/Downloads/Neural_Networks_exp_1926/false_sharing_test.cpp). These two concepts are **opposite sides of the same coin** — one forces memory access, the other avoids it.

#### What is `volatile`?

`volatile` is a **type qualifier** (a keyword — same family as `const`). It is NOT something that runs at runtime — it is an instruction to the **compiler** that changes how machine code is generated.

The three type qualifiers in C++ are:

- `const` — value cannot be changed after initialization
- `volatile` — value must be read/written from actual memory every time (compiler cannot cache it in a register)
- `mutable` — allows modification of a member even inside a `const` object

`volatile` is a **keyword**, not an **identifier**. An identifier is a name you create (like `sum`, `data`, `matmul`). Keywords are reserved words built into the language that you cannot use as variable names.

#### Why We Used `volatile` in `false_sharing_test.cpp`

In our False Sharing hardware test, we wrote:

```cpp
volatile float data[NUM_THREADS * 64] = {0};

// Each thread does 100 million writes:
for (int i = 0; i < 100000000; ++i) {
    data[t] += 1.0f;
}
```

**Without `volatile`, the compiler destroys the test:**

When you compile with `-O3`, the compiler looks at that loop and thinks: *"This loop just adds 1.0 to `data[t]` a hundred million times. That's the same as `data[t] += 100000000.0f`. I'll just do one addition!"*

The compiler-optimized assembly would look like:

```asm
; Compiler-optimized (NO memory writes in the loop!)
vmovss  xmm0, [data + t*4]     ; Load data[t] ONCE
vaddss  xmm0, xmm0, 100000000  ; Add 100 million in ONE instruction
vmovss  [data + t*4], xmm0     ; Store ONCE
```

The entire loop executes in **nanoseconds**. There are ZERO memory writes inside the loop. False Sharing can't happen because there's nothing to share! The test would be completely useless.

**With `volatile`, the compiler is FORCED to do real memory access every iteration:**

```asm
; Volatile-forced (100 million REAL memory writes!)
.L5:
    vmovss  xmm0, [data + t*4]     ; LOAD from memory (forced)
    vaddss  xmm0, xmm0, 1.0        ; ADD 1.0
    vmovss  [data + t*4], xmm0     ; STORE to memory (forced)
    inc     ecx                      ; i++
    cmp     ecx, 100000000          ; i < 100000000?
    jl      .L5                      ; loop back
```

Now there are 100 million real LOAD+STORE pairs. Every STORE hits the cache line. When 4 threads do this to adjacent slots (same 64-byte cache line), the MESI protocol fires 100 million RFO (Read-For-Ownership) signals → **massive measurable slowdown from False Sharing**. This is how we proved False Sharing exists at the hardware level.

#### The `float sum` Register Accumulator — The Exact Opposite

In `matmul_experiments.cpp`, I did the opposite — I used a temporary variable to **AVOID** memory access:

```cpp
for (int j = 0; j < N; ++j) {
    float sum = 0.0f;                            // Lives in a CPU register (~0.3ns access)
    for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];      // Register-only operation, no memory write
    }
    C[i * N + j] = sum;                           // Write to memory only ONCE
}
```

**Without the temp variable** (the old way): `C[i*N+j] += A[i*K+k] * B[k*N+j]` does a memory LOAD of `C[i*N+j]`, an ADD, and a memory STORE — **3 memory operations per iteration**. Over 2000 `k` iterations, that's 2000 LOADs + 2000 STOREs = **4000 memory operations** per cell.

**With the temp variable:** The inner loop only touches the `sum` register (~0.3ns per operation). The memory STORE to `C[i*N+j]` happens **exactly once** at the end. That's 2000 LOADs + 2000 STOREs → **1 STORE**. Massive savings.

#### The Complete Picture — They Are Opposites

| Feature | `volatile float data[t]` (False Sharing Test) | `float sum` (Matmul Optimization) |
|---|---|---|
| **Goal** | **FORCE** every access to go to memory | **AVOID** memory access entirely |
| **Where it lives** | In RAM / L1 Cache (compiler forced) | In a CPU Register (compiler encouraged) |
| **Compiler optimization** | **BLOCKED** — must read/write every iteration | **ENCOURAGED** — compiler keeps value in register |
| **Why we use it** | To **measure** hardware penalties (False Sharing) | To **eliminate** performance penalties |
| **Use case** | Hardware tests, device drivers, memory-mapped I/O | Performance-critical inner loops |

**The beautiful irony:** To prove that memory access is slow (False Sharing test), we had to force the compiler to actually do memory access using `volatile`. And to make our matmul fast, we did the exact opposite — we used a temp variable to tell the compiler "keep this in a register, don't touch memory."

---

## Part 11 — Compiler Flags Deep Dive: What `-O1` Actually Does

After optimizing our `ijk` loop with a manual `float sum` register accumulator (dropping the time from 17.0s to 12.7s), we ran our first test with a basic compiler optimization flag: `-O1`.

**The result was a massive speedup: 12.7s → 7.0s.**

But how did a "basic" flag nearly halve the execution time? To understand exactly what the compiler did, we queried GCC directly using:
`diff <(g++ -Q -O0 --help=optimizers) <(g++ -Q -O1 --help=optimizers) | grep "\[enabled\]"`

This revealed exactly 39 optimization algorithms that `-O1` turns on which `-O0` leaves off.

<details>
<summary><b>Click here to see the full list of 39 optimizations enabled by -O1</b></summary>

```text
  -fbranch-count-reg            [enabled]
  -fcombine-stack-adjustments   [enabled]
  -fcompare-elim                [enabled]
  -fcprop-registers             [enabled]
  -fdefer-pop                   [enabled]
  -fdse                         [enabled]
  -fforward-propagate           [enabled]
  -fguess-branch-probability    [enabled]
  -fif-conversion               [enabled]
  -fif-conversion2              [enabled]
  -finline                      [enabled]
  -finline-functions-called-once  [enabled]
  -fipa-modref                  [enabled]
  -fipa-profile                 [enabled]
  -fipa-pure-const              [enabled]
  -fipa-reference               [enabled]
  -fipa-reference-addressable   [enabled]
  -fmove-loop-invariants        [enabled]
  -fomit-frame-pointer          [enabled]
  -freorder-blocks              [enabled]
  -fshrink-wrap                 [enabled]
  -fsplit-wide-types            [enabled]
  -fssa-phiopt                  [enabled]
  -ftoplevel-reorder            [enabled]
  -ftree-bit-ccp                [enabled]
  -ftree-builtin-call-dce       [enabled]
  -ftree-ccp                    [enabled]
  -ftree-ch                     [enabled]
  -ftree-coalesce-vars          [enabled]
  -ftree-copy-prop              [enabled]
  -ftree-dce                    [enabled]
  -ftree-dominator-opts         [enabled]
  -ftree-dse                    [enabled]
  -ftree-fre                    [enabled]
  -ftree-pta                    [enabled]
  -ftree-sink                   [enabled]
  -ftree-slsr                   [enabled]
  -ftree-sra                    [enabled]
  -ftree-ter                    [enabled]
```

</details>

Here are the **Big 3** that directly annihilated the execution time of our matrix multiplication code:

### 1. `-fmove-loop-invariants` (The Multiplier Killer)

In our inner loop, we calculate the 1D flat array index for Matrix A: `A[i * K + k]`.
To the CPU, that means: *Calculate `i * K`, then add `k`.*

- **With `-O0`:** The CPU recalculates `i * K` on every single iteration of the `k` loop. Since `K = 2000` and there are 4,000,000 cells to compute, the CPU performs **8 billion useless integer multiplications**.
- **With `-O1`:** The `-fmove-loop-invariants` algorithm analyzes the `k` loop, realizes `i` and `K` never change inside it (they are "invariant"), and **pulls the `i * K` calculation outside the loop**. It calculates it exactly *once* per row, stores it in a register, and just adds `k` in the inner loop. This single flag deleted 8 billion math operations.

### 2. `-fomit-frame-pointer` (The Extra Register)

- **With `-O0`:** The compiler reserves a specific CPU register (`rbp`) exclusively to keep track of the "call stack" (the frame pointer). This makes debugging easier (e.g., in GDB), but steals a register from your math operations.

- **With `-O1`:** The `-fomit-frame-pointer` algorithm disables this debugging safety net, freeing up that register for raw math. In an intense matrix multiplication loop, having **one extra general-purpose CPU register** means the CPU doesn't have to spill variables back to the L1 Cache.

### 3. `-ftree-slsr` (Straight-Line Strength Reduction)

In our inner loop, we calculate the index for Matrix B: `B[k * N + j]`. As `k` increments (0, 1, 2, 3), the sequence of indices is: `0*N+j`, `1*N+j`, `2*N+j`...

- **With `-O0`:** The CPU performs a full integer multiplication (`k * N`) on every loop iteration.
- **With `-O1`:** The Strength Reduction algorithm recognizes this is an arithmetic sequence. Instead of doing expensive multiplication every loop, it transforms the code into a pointer that just **adds** `N` to the previous value. Addition is significantly cheaper and faster in hardware than multiplication.

### The Flag Ladder Summary

While `-O1` gave us a huge 5-second speedup, it still processes math one float at a time. The real magic happens when we climb the optimization ladder:

- **`-O0` (Debug):** Exactly what you wrote. Zero optimization. Slow.
- **`-O1` (Basic):** Logic optimization. Moves invariants, uses better registers, reduces math strength.
- **`-O2` (Normal Release):** Turns on almost everything that doesn't bloat the executable size. (Still no vectorization).
- **`-O3` (High Performance):** Turns on heavy algorithms like Loop Unrolling and **Loop Vectorization (`-ftree-loop-vectorize`)**. This looks at the loop and packs 8 floats into a single AVX2 instruction.
- **`-Ofast` (Reckless Speed):** `-O3` plus "unsafe" floating-point math optimizations (violating strict IEEE standards for extra speed).

**Next Step:** Pairing `-O3` with `-march=native` to unlock AVX2 loop vectorization and drop the execution time into the ~1 second range.

---

## Part 12 — Pointer Aliasing, Assembly Proofs, and Compiler Architecture

While testing the `-O1` flag, we made a shocking discovery: our `ijk` loop executed in **7.0s** both *with* the manual `float sum` temporary variable and *without* it. The compiler seemingly invented a temporary variable for us! To prove exactly how and why this happened, we generated the raw assembly code.

### 1. Pointer Aliasing and `-ftree-pta`
Normally, if you write `C[i * N + j] += A[...] * B[...]` in a generic function, the compiler panics. It fears **Pointer Aliasing** — the possibility that the memory pointers for `C` and `A` secretly overlap. If they overlap, keeping `C` in a register (and not writing it to RAM on every loop) would result in `A` reading corrupt, stale data. Because of this paranoia, the compiler is legally forced to write to RAM on every iteration, slowing the code down.

So why did it successfully optimize our code without the `float sum` variable?
Because our entire matrix multiplication was inside `main()`.
*   `-O1` enables **`-ftree-pta` (Pointer-To-Alias Analysis)**.
*   This "detective" algorithm traced `A`, `B`, and `C` back to their `new float[...]` allocations within the same scope.
*   Because `new` guarantees strictly separated memory blocks, `-ftree-pta` mathematically proved the pointers do not overlap. 
*   Once proven safe, **`-fmove-loop-invariants`** safely hoisted the memory load/store completely out of the inner loop, perfectly replicating our manual `float sum` logic.

### 2. The Assembly Proof (`g++ -S`)
We proved this by asking GCC to output human-readable Assembly instead of an executable binary:
`g++ -O1 -S matmul_experiments.cpp -o matmul_experiments.s -masm=intel`

In the generated `.s` file, we searched for the float multiplication instruction (`mulss`) to locate our innermost `k` loop (labeled `.L5` by the compiler). Here is the empirical proof of the register promotion:

```nasm
.L5:
    movss   xmm0, DWORD PTR [rdx]      ; Read from Matrix A in RAM
    mulss   xmm0, DWORD PTR [rax]      ; Multiply with Matrix B
    addss   xmm1, xmm0                 ; ADD DIRECTLY TO CPU REGISTER (xmm1)!
    add     rdx, 4                     ; k++
    add     rax, 8000
    cmp     rax, rcx
    jne     .L5                        ; Loop Boundary: Jump back up to .L5
    
    movss   DWORD PTR [r9+rsi*4], xmm1 ; MEMORY WRITE TO MATRIX C IS OUTSIDE THE LOOP!
```
When compiled *without* `-O1` (using `-O0`), the final `movss` (memory write) instruction is wedged inside the `.L5` loop, resulting in 4 billion slow memory writes and a 17.0s execution time.

### 3. CPU Architecture: `rax` vs `xmm`
The assembly reveals the architectural divide between General Purpose Registers (GPRs) and Vector Registers (SIMD):
*   **`rax`, `rcx`, `rdx` (GPRs):** These are 64-bit integer registers. They do not do floating-point math. Instead, they act as **Pointers**. Because memory addresses on a 64-bit OS are 64 bits long (e.g., `0x00007FFE4B3A1234`), the 64-bit `rax` register is the exact perfect size to hold a memory location. 
*   **`xmm`, `ymm`, `zmm` (SIMD):** These are massive floating-point math registers (128-bit, 256-bit, and 512-bit). `rax` tells the CPU *where* to look in memory, and the CPU scoops up the floats from that address and dumps them into an `xmm` register to do the actual `mulss` math.

### 4. Compilation Flags Clarification
During this deep dive, we clarified several critical compiler flags to avoid "naming illusions":
*   **`-masm=intel` vs AT&T:** By default, GCC outputs AT&T syntax (cluttered with `%` and `$`, and reading backwards: `addq $4, %rax`). We use `-masm=intel` to force standard, highly readable Intel syntax (`add rax, 4`).
*   **`-c` (Lowercase) vs `-C` (Uppercase):**
    *   **`-c` (Object Code):** Stops the compilation pipeline right before the Linker. Outputs raw, unlinked binary Machine Code (`.o` file).
    *   **`-C` (Keep Comments):** Instructs the Preprocessor not to strip out `//` and `/* */` comments. It does *not* stop the pipeline. 
    *   *Warning:* The `-o` flag only changes the output filename, not the file type. Running `g++ matmul.cpp -o obj.o` (without `-c`) creates a fully executable program disguised with a `.o` extension!

---

## Part 13 — Master Benchmark Timings Summary (ikj, -O1, and Register Accumulators)

Here is the master list of all benchmark timings recorded on the i7-14700K for the `2000 x 2000 x 2000` matrix multiplication (8 billion operations) leading up to the `-O1` discoveries:

### 1. The Zero-Optimization Baselines (`-O0`)
*   **`ijk` Loop (The Absolute Worst):** **~17.0 seconds**
    *   *Why so slow?* Massive L1 Cache Misses (fetching a new cache line for every single read of Matrix B) + writing to RAM 4 billion times.
*   **`ijk` Loop + `float sum` (The Register Accumulator Win):** **~12.7 seconds**
    *   *Why the speedup?* Even though Cache Misses were still terrible, we successfully deleted 3.996 billion RAM writes by forcing the CPU to accumulate the result in a hardware register before finally writing it to memory once.
*   **`ikj` Loop (Spatial Locality Win):** **~12.2 seconds**
    *   *Why the speedup?* Swapping the loops to traverse Matrix B sequentially reduced the cache miss rate from 100% to ~6% (1 miss per 16 floats). 

### 2. The First Optimization Gear (`-O1`)
*   **`ijk` Loop + `float sum` (Manual Register Accumulator):** **~7.0 seconds**
    *   *Why the speedup?* `-O1` turns on 39 optimizations, including `-fmove-loop-invariants` (stops recalculating `i*K` 8 billion times) and `-ftree-slsr` (turns expensive multiplication into cheap addition for `B`'s index).
*   **`ijk` Loop WITHOUT `float sum` (Automatic Register Accumulator):** **~7.0 seconds**
    *   *Why is it identical?* This was the ultimate realization. `-O1` turns on `-ftree-pta` (Pointer-To-Alias Analysis), which proved the `new float[]` matrices didn't overlap. This allowed the compiler to automatically pull the memory write out of the inner loop, perfectly replicating the manual `float sum` logic without us having to write it!
