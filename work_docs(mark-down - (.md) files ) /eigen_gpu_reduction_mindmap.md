# Eigen GPU Reduction Architecture: `TensorReductionGpu.h`

This document provides a comprehensive breakdown of `eigen/unsupported/Eigen/src/Tensor/TensorReductionGpu.h`. 

To answer your fundamental question: **"If no kernel is in this file, and only launchers are here, then where are the exact files/kernels located?"**
**The kernels ARE inside this file!** Eigen strictly writes its CUDA/HIP Kernels inside header files using standard C++ templates overlaid with NVIDIA `__global__` pragmas. When a downstream compiler (like `nvcc` or `hipcc`) compiles a `.cu` file that `#include`s this header, the device-code compiler extracts these functions and compiles them into PTX/SASS GPU machine code. 

Here is the complete bifurcation and mapping of the file content.

---

## 1. Top-Level Dispatchers (The Structs)

Unlike the CPU where Eigen uses `InnerMostDimReducer`, `InnerMostDimPreserver`, `GenericDimReducer`, and `FullReducer`, the GPU architecture narrows the operations down into **Three Core Dimensions** depending on memory layouts.

### `FullReducer`
- **Purpose:** Reduces the ENTIRE tensor into a single scalar value.
- **Triggers:** Used when `normalized_axes` contains every dimension in the tensor.

### `InnerReducer`
- **Purpose:** Reduces the inner-most dimensions (fastest moving memory). 
- **Triggers:** Used when preserving the outer dimensions. Equivalent to flattening the matrix and summing up the internal rows. Highly optimized for memory coalescing.

### `OuterReducer`
- **Purpose:** Reduces the outer-most dimensions.
- **Triggers:** Used when reducing the sluggishly-moving memory dimensions while preserving the fast-moving inner dimensions (like reducing column-wise across rows).

---

## 2. The Launchers

The Reducer structs don't directly call kernels. Instead, they pass their logic to **Launchers**. Launchers calculate grid and block sizes based on the hardware constraints of the user's physical GPU device.

*   `FullReductionLauncher`: Calculates optimal CUDA block sizing (usually 256 threads) and calls `FullReductionKernel`.
*   `InnerReductionLauncher`: Calculates matrix-flattened blocks and calls `InnerReductionKernel`.

---

## 3. The GPU Kernels (`__global__` functions)

These are the functions that physically execute on the streaming multiprocessors (SMs) of the NVIDIA/AMD GPU.

### A. Initialization Kernels
Because GPU kernels are massively parallel, it can be computationally expensive (and cause race conditions) to have threads negotiate memory initialization. Eigen uses separate initialization kernels launched *before* the main job.
- **`ReductionInitKernel`**: Fires up to fill the initial output tensor with the reduction's identity element (e.g., `0` for Sum, `1` for Prod, `-inf` for Max).
- **`ReductionInitFullReduxKernelHalfFloat` & `ReductionInitKernelHalfFloat`**: Specialized variants to handle NVIDIA TensorCore fp16 16-bit half precision layouts.

### B. Core Compute Kernels
- **`FullReductionKernel`**: 
  - Each thread block reduces a sub-chunk of the tensor entirely into a single variable using high-speed Warp-Shuffling (`__shfl_down_sync`).
  - To prevent global conflicts, an atomic semaphore negotiates which block runs first. The winning block writes to global memory, and all subsequent blocks `atomicReduce` their blocks onto that address.
- **`InnerReductionKernel`**: 
  - Unrolls loops 16 times (`#pragma unroll`) for massive throughput.
  - Fetches elements, adds locally, warp-shuffles, and uses `atomicReduce` back to the specific row.
- **`OuterReductionKernel`**:
  - Handles strided operations (the slowest form of GPU memory access). Skips warp-shuffling because adjacent threads are processing unrelated accumulation buckets. Directly loops and calls `atomicReduce` at the end of its workload.

### C. Clean-up Kernels
- **`ReductionCleanupKernelHalfFloat`**: Used in FP16 operations to safely condense sub-blocks when the size isn't divisible by the packet width.

---

## 4. Helper Functions (The Secret Sauce)

Eigen implements custom atomic fallbacks because historically, NVIDIA GPUs lacked native atomic functions for all data types (like floats or FP16). 

- **`atomicReduce(T* output, T accum, R& reducer)`**: 
  - *What it does*: Safely accumulates a result into an output bucket in global memory, ensuring no two GPU threads overwrite each other.
  - *How it works*: If the hardware supports `atomicAdd`, it calls it natively. If the type is unsupported or the reducer is complex (like `ArgMax`), it implements an infinite `while` loop utilizing a Compare-And-Swap (`atomicCAS`) locking mechanism. It grabs the memory value, applies the user's reducer, and attempts to swap it back in if no other thread touched it in the meantime.
- **`atomicExchCustom`**: 
  - Standardizes pointer-swap instructions across different floating point bit-representations.

---

## Why this Architecture?

1. **Warp-Shuffle Synchronization (`__shfl_down_sync`):** Rather than writing to temporary memory, individual threads inside the SM directly share registers to sum elements 32-at-a-time. This saves the kernel from bottlenecking on the L2 Cache.
2. **Atomic Funneling:** Operations are intensely optimized to NEVER bottleneck global GPU locks. They stay localized inside registers and only `atomicReduce` at the absolute last possible millisecond.
3. **Explicit FP16 Handling:** A massive chunk of the file is dedicated to `HalfFloat` extensions. This is because modern AI operations rely on 16-bit floats for speed, but the underlying CUDA math protocols historically treated 16-bit pointers aggressively differently than 32-bit floats.
