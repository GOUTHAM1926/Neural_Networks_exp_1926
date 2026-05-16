# Statistical Reduction Operations: Algorithm Analysis & Framework Comparison

> Comprehensive analysis of mean, variance, and standard deviation reduction implementations
> across PyTorch, TensorFlow/Eigen, NumPy, and our library (master_gau).

---

## 1. Statistical Operations List

| Operation | Description | NaN Behavior |
|-----------|-------------|--------------|
| `mean` | Arithmetic mean along axis | Propagates NaN |
| `nanmean` | Mean ignoring NaN values | Ignores NaN, counts only valid |
| `var` | Variance along axis | Propagates NaN |
| `nanvar` | Variance ignoring NaN values | Ignores NaN, counts only valid |
| `std` | Standard deviation (sqrt of variance) | Propagates NaN |
| `nanstd` | Std ignoring NaN values | Ignores NaN, counts only valid |
| `var_mean` | Returns (variance, mean) in a single call | Propagates NaN |
| `std_mean` | Returns (std, mean) in a single call | Propagates NaN |

**Key parameters:**
- **`correction`** (aka `ddof`): Bessel's correction. `correction=1` gives sample variance (divide by N-1), `correction=0` gives population variance (divide by N).
- **`take_sqrt`**: Distinguishes `var` from `std`. When true, the final result is `sqrt(variance)`.
- **`keepdim`**: Whether reduced dimensions are retained as size-1 dims.

---

## 2. Two-Pass Algorithm

### How It Works

The two-pass algorithm computes variance in two sequential passes over the data:

**Pass 1 -- Compute the mean:**
```
mean = (1/N) * sum(x_i)
```

**Pass 2 -- Compute sum of squared differences from the mean:**
```
variance = (1/(N - correction)) * sum((x_i - mean)^2)
```

### Pseudocode

```cpp
// Pass 1: Compute mean
double sum = 0.0;
for (int i = 0; i < N; i++) {
    sum += data[i];
}
double mean = sum / N;

// Pass 2: Compute variance
double sq_diff_sum = 0.0;
for (int i = 0; i < N; i++) {
    double diff = data[i] - mean;
    sq_diff_sum += diff * diff;
}
double variance = sq_diff_sum / (N - correction);
```

### Numerical Stability Analysis

**Strengths:**
- Very good numerical stability because the second pass computes `(x_i - mean)^2` where the differences are centered around zero. This avoids catastrophic cancellation.
- The subtraction `x_i - mean` keeps values small, so the squared terms don't overflow for typical data.

**Weaknesses:**
- If the mean itself is computed with poor precision (e.g., accumulating float16 sums), the second pass inherits that error.
- Two full passes over the data means 2x the memory traffic -- critical for large tensors that don't fit in cache.

### When It's Good/Bad

| Good for | Bad for |
|----------|---------|
| Data fits in L1/L2 cache (second pass is cheap) | Large tensors that exceed cache (2x memory bandwidth) |
| When mean is already available from a previous op | Streaming/online data (can't revisit) |
| Simple to parallelize each pass independently | GPU reductions where kernel launch overhead matters |
| Easy to reason about correctness | Single-kernel GPU implementations |

---

## 3. Welford's Online Algorithm

### How It Works

Welford's algorithm computes mean, variance, and count in a **single pass** by maintaining a running state:

```
State: (mean, M2, count)
```

For each new value `x`:
```
count += 1
delta = x - mean
mean += delta / count
delta2 = x - mean        // Note: uses UPDATED mean
M2 += delta * delta2
```

Final variance:
```
variance = M2 / (count - correction)
std = sqrt(variance)
```

### The Update Formula (Derivation)

The key insight is the identity:
```
delta  = x_n - mean_{n-1}
delta2 = x_n - mean_n = x_n - (mean_{n-1} + delta/n) = delta * (1 - 1/n) = delta * (n-1)/n

M2_n = M2_{n-1} + delta * delta2
     = sum_{i=1}^{n} (x_i - mean_n)^2
```

This is mathematically equivalent to the two-pass algorithm but computed incrementally.

### Pseudocode

```cpp
double mean = 0.0;
double M2 = 0.0;
int64_t count = 0;

for (int i = 0; i < N; i++) {
    count += 1;
    double delta = data[i] - mean;
    mean += delta / count;
    double delta2 = data[i] - mean;  // uses updated mean
    M2 += delta * delta2;
}

double variance = M2 / (count - correction);
double std_dev = sqrt(variance);
```

### Numerical Stability Analysis

**Strengths:**
- Numerically stable because it avoids computing `sum(x^2) - N*mean^2` (the naive one-pass formula that suffers catastrophic cancellation).
- The running mean stays close to the data values, so `delta` terms remain small.
- Equivalent stability to the two-pass algorithm for most practical cases.

**Weaknesses:**
- The division `delta / count` introduces a small rounding error at each step. For very large N (billions), this accumulates slightly more error than two-pass.
- For pathological distributions (all values identical except one outlier), precision can differ from two-pass.

### Parallel Welford (Combining Two States)

When using multiple threads/warps, each computes a local Welford state. These are combined with:

```
Given states A = (mean_a, M2_a, count_a) and B = (mean_b, M2_b, count_b):

delta = mean_b - mean_a
count = count_a + count_b
mean = mean_a + delta * (count_b / count)
M2 = M2_a + M2_b + delta^2 * count_a * (count_b / count)
```

This is the **Chan-Golub-LeVeque** parallel combination formula. It allows combining partial results from different threads with no loss of numerical stability.

**This is the critical advantage**: parallel Welford gives the same result regardless of how work is split across threads.

---

## 4. Which Framework Uses Which?

### PyTorch: Welford's Algorithm (Single Pass)

PyTorch uses Welford's online algorithm for both CPU and GPU variance/std computation. The implementation lives in:

**File:** `aten/src/ATen/native/SharedReduceOps.h` (lines 68-149)

```cpp
template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;
};
```

The kernel dispatch in `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp` (lines 138-149):

```cpp
void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
        iter,
        WelfordOps<scalar_t, double, int64_t,
                   std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
        WelfordData<double, int64_t>());
  });
}
```

Key design decisions:
- **`binary_kernel_reduce`** is used because `var_mean` / `std_mean` return two outputs (variance/std + mean). The same kernel handles `var`, `std`, `var_mean`, and `std_mean`.
- Accumulation always in **double** even for float/half/bfloat16 inputs.
- `n` tracked as `int64_t` to avoid cumulative rounding, `nf` as float for the combine step.

### TensorFlow / Eigen

TensorFlow's variance reduction uses a **two-pass approach** in most CPU paths:
1. `tf.math.reduce_mean()` computes the mean.
2. `tf.math.reduce_variance()` = `reduce_mean(square(x - mean))`.

In the Python-level implementation (`tensorflow/python/ops/nn_impl.py`):
```python
def moments(x, axes, ...):
    mean = math_ops.reduce_mean(x, axes, keepdims=True)
    variance = math_ops.reduce_mean(
        math_ops.squared_difference(x, mean), axes, keepdims=keepdims)
    return mean, variance
```

Eigen (the underlying C++ library) provides a **two-pass** implementation for variance via its `Tensor::variance()` method. However, TensorFlow's XLA compiler can fuse these two passes into a single kernel on GPU.

### NumPy

NumPy uses a **two-pass algorithm** by default:

In `numpy/core/src/npymath/npy_math_internal.h.src` and the Python layer:
```python
# numpy/core/fromnumeric.py (simplified)
def var(a, axis=None, ddof=0):
    mean = a.mean(axis, keepdims=True)
    x = a - mean
    return (x * x.mean(axis)) / (N - ddof)
```

NumPy explicitly chose two-pass for simplicity and numerical stability. Their documentation notes this is "safe from overflow issues."

For `nanvar` and `nanstd`, NumPy masks NaN values first, then applies the same two-pass approach on the masked array.

### Our Library (master_gau): Two-Pass Algorithm

Our current implementation uses a **two-pass algorithm**:

**File:** `master_gau/include/ops/helpers/ReductionImpl.h` (line 1875)

```
// VARIANCE REDUCTION DISPATCHER (Two-pass algorithm)
```

**Pass 1:** Call `dispatch_mean_kernel<T, SumOp>()` to compute the mean (line 1926-1928):
```cpp
Tensor mean_tensor = is_nan_aware
    ? dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, true, stream)
    : dispatch_mean_kernel<T, SumOp>(input, normalized_axes, true, stream);
```

**Pass 2:** Iterate over all elements computing `(x_i - mean)^2` and accumulate (lines 2059-2117), with 3-path layout dispatch (InnerContiguous, OuterContiguous, Generic).

The `VarianceOp` struct in `ReductionOps.h` (line 516) stores a pre-computed `mean_value`:
```cpp
template <typename T>
struct VarianceOp {
    using AccT = AccumulatorType<T>;
    int64_t correction;
    AccT mean_value;     // Pre-computed mean from Pass 1

    DEVICE_HOST AccT reduce(const AccT& acc, const AccT& val) const {
        AccT diff = val - mean_value;
        return acc + diff * diff;
    }
};
```

### Summary Table

| Framework | Algorithm | Passes | var_mean native? |
|-----------|-----------|--------|------------------|
| **PyTorch** | Welford's online | 1 | Yes (single kernel) |
| **TensorFlow/Eigen** | Two-pass | 2 | Yes (`tf.nn.moments`) |
| **NumPy** | Two-pass | 2 | No (separate calls) |
| **Our library** | Two-pass | 2 | No (mean then variance) |

---

## 5. Comparison Table

| Property | Two-Pass | Welford's Online | Naive One-Pass (`sum(x^2) - N*mean^2`) |
|----------|----------|------------------|----------------------------------------|
| **Numerical precision** | Excellent | Excellent (slightly worse for N > 10^9) | Poor (catastrophic cancellation) |
| **Passes over data** | 2 | 1 | 1 |
| **Memory reads** | 2N | N | N |
| **Cache efficiency** | Poor for large tensors (data evicted between passes) | Optimal (single pass, data read once) | Optimal |
| **Memory overhead** | O(output_size) for mean tensor | O(1) per thread (mean, M2, count) | O(1) |
| **Parallelizability** | Easy (each pass is a standard reduction) | Good (parallel combine formula) | Easy |
| **SIMD compatibility** | Excellent (each pass is a simple vectorizable loop) | Moderate (division `delta/count` per element hurts throughput) | Excellent |
| **Thread combining** | Trivial (sum partial sums) | Requires Chan-Golub-LeVeque formula | Trivial |
| **Kernel launches (GPU)** | 2 (one per pass) | 1 | 1 |
| **var_mean efficiency** | Mean is "free" (already computed in pass 1) | Mean is "free" (part of Welford state) | Requires separate mean |
| **Code complexity** | Simple | Moderate | Simple |
| **Streaming data** | Impossible (needs full data twice) | Natural fit | Natural fit |

### The SIMD Problem with Welford

Welford's per-element update involves:
```
delta = x - mean
mean += delta / count     // <-- division is expensive in SIMD
delta2 = x - mean
M2 += delta * delta2
```

The `delta / count` division is the bottleneck. SIMD `_mm256_div_ps` has ~13-14 cycle latency vs ~4-5 for multiply. Two-pass avoids any per-element division in the inner loop:
```
// Pass 2 inner loop (perfectly SIMD-friendly):
diff = x - mean           // subtract (1 cycle)
acc += diff * diff         // fused multiply-add (4-5 cycles)
```

**However**, this can be mitigated in Welford by batching: accumulate a local sum over a small block, then update the Welford state once per block. PyTorch does NOT do this -- they accept the division cost because GPU warp-level reductions dominate their use case.

---

## 6. Which Should We Use?

### Recommendation: Hybrid Approach

**Keep two-pass for CPU.** Switch to Welford for GPU and for `var_mean`/`std_mean` combined operations.

#### Rationale

**For CPU (keep two-pass):**
1. Our existing dispatcher infrastructure (Strategy 1: InnerContiguous, Strategy 2: OuterContiguous, Generic) already optimizes memory access patterns for each pass independently.
2. Our SIMD engine (`Vectorized.h`) vectorizes `diff * diff` accumulation perfectly -- no per-element division.
3. OpenMP parallelizes each pass trivially with `#pragma omp parallel for`.
4. For typical tensor sizes that fit in L2/L3 cache, the second pass hits warm cache lines. The cost of 2x memory reads is minimal.
5. Our `dispatch_mean_kernel` is already battle-tested and handles all edge cases (NaN, integer upcast to double, half/bfloat16 accumulation).

**For GPU (switch to Welford):**
1. Two separate kernel launches for mean + variance have significant overhead (kernel launch ~5-20 microseconds each).
2. GPU memory bandwidth is the bottleneck -- reading the full tensor twice is wasteful.
3. PyTorch's `WelfordOps` is proven at scale on GPU.
4. The division cost in Welford is hidden by GPU's massive parallelism and memory latency.

**For `var_mean` / `std_mean` (add Welford option):**
1. Even on CPU, if the caller wants both mean and variance, Welford gives them in a single pass.
2. The two-pass approach computes mean anyway, so `var_mean` is essentially free if we return the intermediate mean.
3. For our current two-pass design, `var_mean` can simply return `(variance, mean_tensor)` without recomputation -- but we need to add the API.

#### Implementation Priority

| Priority | Task | Algorithm |
|----------|------|-----------|
| 1 (immediate) | Add `var_mean` / `std_mean` API using existing two-pass (return mean from pass 1) | Two-pass |
| 2 (high) | Add `std` / `nanstd` as `sqrt(variance)` wrappers | Two-pass |
| 3 (medium) | Implement Welford for GPU variance | Welford |
| 4 (low) | Optional Welford CPU path for streaming/very large tensors | Welford |

---

## 7. PyTorch's WelfordOps Deep Dive

### Source: `aten/src/ATen/native/SharedReduceOps.h`

#### WelfordData -- The State

```cpp
template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t mean;   // Running mean
  scalar_t m2;     // Sum of squared differences from running mean
  index_t n;       // Count as integer (avoids rounding error in count)
  scalar_t nf;     // Count as float (needed for combine where int32 may overflow)

  C10_HOST_DEVICE WelfordData() : mean(0), m2(0), n(0), nf(0) {}
};
```

**Design note:** Tracking count as both `int64_t` (`n`) and `double` (`nf`) is deliberate. The integer count avoids cumulative rounding in the reduce step (`new_n = acc.n + 1` is exact), while the float count is needed in combine where two large counts might overflow int32.

#### WelfordOps -- The Operations

**Template parameters:**
```cpp
template <typename scalar_t,      // Input type (float, half, bfloat16)
          typename acc_scalar_t,   // Accumulation type (always double)
          typename index_t,        // Count type (int64_t)
          typename res_t>          // Result type (tuple<scalar_t, scalar_t> for var_mean)
struct WelfordOps {
  acc_scalar_t correction;  // Bessel's correction (0 or 1)
  bool take_sqrt;           // true for std, false for var
```

#### The `reduce` Function (Process One Element)

```cpp
inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {
      new_mean,
      acc.m2 + delta * new_delta,
      new_n,
      new_nf,
    };
}
```

This is the standard Welford update. Note:
- `delta` uses the **old** mean
- `new_delta` uses the **new** mean (after update)
- `M2 += delta * new_delta` -- this is the key formula that maintains numerical stability

#### The `combine` Function (Merge Two Partial Results)

```cpp
inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) return b;
    if (b.nf == 0) return a;
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    return {
      a.mean + delta * nb_over_n,
      a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
      -1,          // n set to -1 (may not fit in index_t after combine)
      new_count
    };
}
```

This is the **Chan-Golub-LeVeque** parallel combination formula. Key details:
- `n` is set to `-1` after combine because the combined count may overflow `index_t` (especially if `index_t` is `int32`). Only `nf` (the float count) is reliable after combining.
- The formula `M2_combined = M2_a + M2_b + delta^2 * n_a * n_b / (n_a + n_b)` is rearranged to avoid computing `n_a * n_b` directly (which could overflow).

#### The `project` Function (Extract Final Result)

```cpp
inline C10_DEVICE res_t project(acc_t acc) const {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    res_t results(take_sqrt ? device_sqrt(var) : var, mean);
    return results;
}
```

This is where Bessel's correction and the sqrt decision happen:
- `divisor = max(count - correction, 0)` -- protects against division by zero when count <= correction.
- When `divisor == 0`, the result is `M2 / 0 = inf` (or NaN if M2 is also 0). This matches NumPy/PyTorch behavior.
- `res_t` is `tuple<scalar_t, scalar_t>` containing `(var_or_std, mean)` -- this is how `var_mean` and `std_mean` return both values from a single reduction.
- `take_sqrt` controls whether the result is variance or standard deviation.

#### GPU Warp Shuffle Support

```cpp
#if defined(__CUDACC__) || defined(__HIPCC__)
inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {
      WARP_SHFL_DOWN(acc.mean, offset),
      WARP_SHFL_DOWN(acc.m2, offset),
      WARP_SHFL_DOWN(acc.n, offset),
      WARP_SHFL_DOWN(acc.nf, offset)
    };
}
#endif
```

This enables warp-level reduction on GPU: each lane shuffles its Welford state to a neighbor, then they `combine()`. This avoids shared memory for the intra-warp reduction.

#### How var, std, var_mean, std_mean All Use the Same Kernel

In `ReduceOps.cpp`, all four operations go through `std_var_stub`:

```cpp
// var: correction=1, take_sqrt=false
// std: correction=1, take_sqrt=true
// var_mean: correction=1, take_sqrt=false, binary_kernel_reduce outputs (var, mean)
// std_mean: correction=1, take_sqrt=true, binary_kernel_reduce outputs (std, mean)
```

The `binary_kernel_reduce` handles two output tensors. For plain `var` or `std`, the second output (mean) is simply discarded. This means **PyTorch always computes the mean even when only variance is requested** -- there is no separate variance-only path.

---

## 8. Our Current Implementation

### What We Have Now

**Architecture:** Two-pass with explicit dispatcher.

**Pass 1 -- Mean:**
- `dispatch_mean_kernel<T, SumOp>()` (or `NanSumOp` for NaN-aware)
- Supports all dtypes including integers (upcast to double), half, bfloat16
- Uses 3-path layout dispatch: InnerContiguous, OuterContiguous, Generic
- OpenMP parallelized with `#pragma omp parallel for`
- Returns a keepdim=true tensor for broadcasting in pass 2

**Pass 2 -- Variance:**
- `dispatch_variance_kernel<T, VarianceOpType>()`
- Takes the mean tensor from pass 1
- Each thread reads `mean_data[output_index]` and accumulates `(x_i - mean)^2`
- Uses the same 3-path layout dispatch as pass 1
- NaN handling: `VarianceOp` propagates NaN, `NanVarianceOp` skips NaN

**Operations available:**
- `mean` / `nanmean` -- via `dispatch_mean_kernel`
- `var` / `nanvar` -- via `dispatch_variance_kernel`
- `std` / `nanstd` -- **NOT YET IMPLEMENTED** (need sqrt wrapper)
- `var_mean` / `std_mean` -- **NOT YET IMPLEMENTED** (need combined return)

### What Needs to Change

#### 1. Add `std` / `nanstd` (Trivial)

Simply call `dispatch_variance_kernel` and apply `sqrt()` to the output. Can be done as a thin wrapper:

```cpp
template <typename T, template <typename> class VarianceOpType>
Tensor dispatch_std_kernel(const Tensor& input,
                           const std::vector<int64_t>& normalized_axes,
                           bool keepdim, int64_t correction, cudaStream_t stream) {
    Tensor var_result = dispatch_variance_kernel<T, VarianceOpType>(
        input, normalized_axes, keepdim, correction, stream);
    // Element-wise sqrt
    apply_sqrt_inplace(var_result);
    return var_result;
}
```

#### 2. Add `var_mean` / `std_mean` (Low Effort)

Our two-pass design already computes the mean internally. We just need to expose it:

```cpp
template <typename T, template <typename> class VarianceOpType>
std::pair<Tensor, Tensor> dispatch_var_mean_kernel(
    const Tensor& input, const std::vector<int64_t>& normalized_axes,
    bool keepdim, int64_t correction, bool take_sqrt, cudaStream_t stream) {

    // Pass 1: compute mean (already done inside dispatch_variance_kernel)
    constexpr bool is_nan_aware = std::is_same_v<VarianceOpType<T>, NanVarianceOp<T>>;
    Tensor mean_tensor = is_nan_aware
        ? dispatch_mean_kernel<T, NanSumOp>(input, normalized_axes, keepdim, stream)
        : dispatch_mean_kernel<T, SumOp>(input, normalized_axes, keepdim, stream);

    // Pass 2: compute variance using the mean
    Tensor var_result = dispatch_variance_kernel_with_mean<T, VarianceOpType>(
        input, normalized_axes, keepdim, correction, mean_tensor, stream);

    if (take_sqrt) apply_sqrt_inplace(var_result);

    return {var_result, mean_tensor};
}
```

This avoids recomputing the mean -- the current `dispatch_variance_kernel` already calls `dispatch_mean_kernel` internally, so we just need to refactor to accept an externally-provided mean or return the internally-computed one.

#### 3. SIMD Vectorization of Variance Inner Loop (Medium Effort)

The pass-2 inner loop is currently scalar:

```cpp
AccT diff = val_acc - mean_val;
accumulator += diff * diff;
```

This is a textbook case for SIMD with our `Vectorized<T>` infrastructure:

```cpp
// Vectorized variance accumulation (InnerContiguous path)
using Vec = Vectorized<AccT>;
Vec v_mean = Vec(mean_val);
Vec v_acc = Vec(AccT(0));
int64_t j = 0;
for (; j + Vec::size() <= reduced_count; j += Vec::size()) {
    Vec v_data = Vec::loadu(in_ptr + j);
    Vec v_diff = v_data - v_mean;
    v_acc = v_acc + v_diff * v_diff;  // or fmadd(v_diff, v_diff, v_acc)
}
AccT accumulator = v_acc.reduce_add();
// Scalar tail
for (; j < reduced_count; j++) {
    AccT diff = static_cast<AccT>(in_ptr[j]) - mean_val;
    accumulator += diff * diff;
}
```

This would give ~4-8x speedup on the inner loop for float (AVX2: 8 floats) or ~2-4x for double (AVX2: 4 doubles).

#### 4. GPU Welford (Higher Effort)

For GPU, implement `WelfordData` and `WelfordOps` similar to PyTorch's design. This would:
- Eliminate the second kernel launch for variance
- Halve GPU memory bandwidth usage
- Enable native `var_mean` / `std_mean` in a single kernel

This requires integrating with our existing GPU reduction infrastructure (`dispatch_variance_gpu`).

### Implementation Plan

| Phase | Task | Files to Modify | Effort |
|-------|------|-----------------|--------|
| **Phase 1** | Add `dispatch_std_kernel` wrapper | `ReductionImpl.h` | 1 day |
| **Phase 1** | Add `dispatch_var_mean_kernel` and `dispatch_std_mean_kernel` | `ReductionImpl.h` | 1-2 days |
| **Phase 1** | Refactor `dispatch_variance_kernel` to optionally accept pre-computed mean | `ReductionImpl.h` | 1 day |
| **Phase 2** | SIMD vectorize variance inner loop (InnerContiguous path) | `ReductionImpl.h`, `Vectorized.h` | 2-3 days |
| **Phase 2** | SIMD vectorize mean inner loop if not already done | `ReductionImpl.h`, `Vectorized.h` | 1-2 days |
| **Phase 3** | Implement `WelfordData` / `WelfordOps` structs | New: `WelfordOps.h` or in `ReductionOps.h` | 2 days |
| **Phase 3** | GPU Welford variance kernel | GPU reduction files | 3-5 days |
| **Phase 4** | Benchmark two-pass vs Welford on CPU for large tensors | Tests | 2 days |
| **Phase 4** | Optional: CPU Welford path for tensors > L3 cache | `ReductionImpl.h` | 3 days |

### Architecture Diagram

```
Current (Two-Pass):
  var(input, axes)
    --> dispatch_mean_kernel(input, axes)     [Pass 1: mean]
    --> dispatch_variance_kernel(input, axes)  [Pass 2: sum((x-mean)^2) / (N-corr)]
    --> output

Proposed (Unified API):
  var_mean(input, axes)
    --> dispatch_mean_kernel(input, axes)     [Pass 1: mean]  -- returned
    --> dispatch_variance_kernel(input, axes, mean)  [Pass 2]  -- returned
    --> (variance, mean)

  std_mean(input, axes)
    --> var_mean(input, axes)
    --> sqrt(variance)
    --> (std, mean)

  GPU var_mean(input, axes)  [Future - Welford]:
    --> welford_kernel(input, axes)  [Single pass: mean + M2]
    --> project(M2, count, correction, take_sqrt)
    --> (var_or_std, mean)
```

---

## Appendix A: The Naive One-Pass Formula (Do NOT Use)

For completeness, the naive formula is:
```
variance = (sum(x^2) - N * mean^2) / (N - 1)
```

This suffers from **catastrophic cancellation** when `sum(x^2)` and `N * mean^2` are both large but nearly equal. Example:

```
Data: [10000000.0, 10000001.0, 10000002.0]
sum(x^2) = 300000060000005.0  (loses low-order bits in float32)
N * mean^2 = 300000060000003.0
Difference = 2.0 (should be ~1.0 -- already wrong!)
```

Neither PyTorch, TensorFlow, NumPy, nor our library uses this formula. It exists only as a cautionary example.

## Appendix B: Kahan Compensated Summation

An alternative to improve two-pass precision is Kahan compensated summation in pass 1:

```cpp
double sum = 0.0, c = 0.0;
for (int i = 0; i < N; i++) {
    double y = data[i] - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
}
```

This gives nearly double-precision accuracy even when accumulating float32 values. However, it adds per-element overhead and is harder to SIMD-vectorize. Our library already mitigates this by using double accumulation for float16/bfloat16 via `should_use_double_accumulation<T>()`.
