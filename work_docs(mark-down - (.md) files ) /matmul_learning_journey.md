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
* **`-march=native` alone (~17.6 seconds):** Exactly the same as `-O0`! `-march=native` just tells the compiler what hardware I have. But without `-O3` giving permission to optimize, the compiler doesn't actually use the hardware.
* **`-O2` + `-march=native` (~7.4 seconds):** The compiler knows I have AVX2 hardware, but the `-O2` flag physically refuses to run the loop vectorization algorithms (running `g++ -Q -O2 --help=optimizers` proves `-ftree-loop-vectorize` is disabled in GCC 11 at `-O2`). So it just does the standard slow loop.
* **`-O3` alone (~1.8 seconds):** `-O3` gives permission to vectorize, but without `-march=native`, it assumes I have an ancient, generic CPU to ensure the code works anywhere. It uses old, basic 128-bit SSE vectorization instead of my modern AVX2.
* **`-O3` + `-march=native` (~1.10 seconds):** Together, they unlock the true potential. `-O3` gives the order to vectorize, and `-march=native` gives it the AVX2 weapons to do it!

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
* `#pragma omp parallel for` is called a **"pragma directive"** (or just "compiler directive"). The word `pragma` comes from the Greek word "pragmatikos" meaning "relating to action."
* A pragma is **not C++ code**. It is a secret instruction written directly to the compiler. When GCC sees `#pragma omp`, it physically rewrites your loop into multi-threaded code during compilation.
* **Does NOT need `#include <omp.h>`!** Because it's a compiler instruction, not a C++ function call. The compiler already knows what to do with it.
* **Requires `-fopenmp` flag** in the compilation command to activate. Without this flag, GCC treats the entire `#pragma` line as a harmless comment and ignores it completely — your code still compiles and runs fine, just single-threaded.

**Category 2: Runtime Functions (`omp_*()`) — executed while the program is running**
* Functions like `omp_set_num_threads()`, `omp_get_max_threads()`, and `omp_get_num_procs()` are **real C++ functions** that execute while your program is actually running (at runtime).
* **ALL of them REQUIRE `#include <omp.h>`!** — whether they are setters or getters. Without this header, the compiler has no idea what these functions are and will throw `"not declared in this scope"` errors.
* The available runtime functions:
  * `omp_set_num_threads(N)` — **SET**. Tells OpenMP to use exactly N threads for subsequent parallel regions.
  * `omp_get_max_threads()` — **GET** (read-only). Returns the maximum number of threads (= number of logical cores).
  * `omp_get_num_procs()` — **GET** (read-only). Returns the number of processors/cores available on the hardware.
  * Note: **`omp_set_max_threads()` does NOT exist!** There is no such function in OpenMP.

**Key equivalence:** Writing `omp_set_num_threads(omp_get_max_threads());` before a `#pragma omp parallel for` is **functionally identical** to just writing `#pragma omp parallel for` alone with no function call. Both will use all 28 threads. The only difference is that the explicit version makes your intent visually clear in the code.

**How OpenMP Allocates Threads (Important!):**
* **Default behavior:** If you just use `#pragma omp parallel for` without calling any function, OpenMP **always creates the maximum number of threads** equal to the number of logical CPU cores (28 on my i7-14700K). It does this **every single time**, regardless of what else is running on the system.
* **OpenMP has ZERO awareness of other programs.** It does not check if cores are "available" or "occupied." Even if all 28 cores are already running Chrome, VS Code, and other heavy programs at 100% load, OpenMP will still blindly create 28 threads. The **Linux kernel scheduler** is the only boss that decides who runs on which core — it juggles all programs using rapid time-slicing (switching between processes every few milliseconds).
* **`omp_set_num_threads(N)` does NOT stop other processes.** It only controls how many threads OpenMP itself creates. It cannot kick other programs off cores.

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
   * **Small Workloads:** Use Compiler Flags ONLY (`-O3 -march=native`). Avoid OpenMP overhead.
   * **Heavy Workloads:** Use Compiler Flags AND OpenMP. This is where parallelization shines and provides massive reductions in execution time.

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
* **Thread Assignment:** OpenMP divides the `M` rows among the threads (e.g., if M=2000 and threads=28, each thread gets ~71 complete rows).
* **Software Overhead (MINIMAL):** Threads are woken up, work, and hit a synchronization barrier exactly **1 time**. 
* **Hardware Memory Traffic (NONE):** Thread 0 writes to the top 71 rows of Matrix C, while Thread 1 writes to rows 72-142. Because they write to completely different memory locations, there is **no memory write contention**. Also, reading from Matrix B simultaneously causes no traffic jams because the hardware MESI cache protocol flags simultaneous reads as "Shared" (S-state), allowing perfect parallel reads.

### Case 2: OpenMP on the `j-loop` (The Middle Loop) — The Double Halt
```cpp
for (int i = 0; i < M; ++i) { 
    #pragma omp parallel for
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) { C[i * N + j] += A[i * K + k] * B[k * N + j]; }
    }
}
```
* **Thread Assignment:** The Master Thread owns the `i-loop`. For *every single row*, it wakes up all 28 threads and splits the columns (`j`) among them. 
* **The Software Halt (Implicit Barrier):** At the end of every parallel block, OpenMP enforces an implicit barrier. If Thread 1 finishes early, it **cannot** go to the next row. It must wait for all threads to finish, hit the barrier, and go to sleep. The Master Thread then increments `i` and wakes them all up again. This wakes/halts 28 threads **M times (2000 times)**.
* **The Hardware Halt (False Sharing):** When threads write to different columns in the *same row* (e.g., `C[0][0]`, `C[0][1]`), they are writing to the **same 64-byte Cache Line**. The MESI protocol requires a core to obtain the "Modified" (M) state via a "Read-For-Ownership" (RFO) hardware signal before writing. This forces the cache line to violently bounce between cores on the CPU bus, halting execution. 
* **The "Default Schedule" Discovery:** In default OpenMP (`schedule(static)`), columns are chunked (Thread 0 gets cols 0-70). Because they are 71 floats apart, False Sharing is mostly avoided. Through an ablation test (`ablation_test.cpp`), it was proven that the penalty of moving to the `j-loop` was **~0.55s purely from the 2000 Software Barriers**, and forcing hardware False Sharing (`schedule(static, 1)`) added another **~0.18s** of pure hardware stall.

### Case 3: OpenMP on the `k-loop` (The Inner Loop) — Apocalyptic Performance
```cpp
for (int i = 0; i < M; ++i) { 
    for (int j = 0; j < N; ++j) {
        #pragma omp parallel for
        for (int k = 0; k < K; ++k) { C[i * N + j] += A[i * K + k] * B[k * N + j]; }
    }
}
```
* **Thread Assignment:** The Master Thread controls rows and columns. It wakes up 28 threads to calculate the dot product for a **single output cell**.
* **Software Halt (CATASTROPHIC):** The thread pool is created, halted, and destroyed **M × N times (4,000,000 times)**!
* **Fatal Thread Race Condition (Data Race):** All threads are trying to write to the exact same cell (`C[i * N + j]`) at the exact same microsecond. Since `+=` is actually three instructions (LOAD, ADD, STORE), multiple threads will read `0`, add their partial sum, and overwrite each other's results. The final matrix will contain completely wrong mathematical answers.

