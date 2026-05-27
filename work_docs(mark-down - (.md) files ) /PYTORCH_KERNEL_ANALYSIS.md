# PyTorch CPU Reduction Kernel Organization - Detailed Analysis

## Overview
After analyzing PyTorch's actual source code (aten/src/ATen/native/cpu/), here's the REAL organization:

---

## **BIFURCATION 1: Vectorized Sum-like Operations**
**Uses: `binary_kernel_reduce_vec`**

### Operations:
- `sum()` - integers via `binary_kernel_reduce_vec`, floats via `cascade_sum`
- `nansum()` - via `cascade_sum` (ignores NaN)
- `prod()` - via `binary_kernel_reduce_vec`
- `all()` - via `and()` → `binary_kernel_reduce_vec`
- `any()` - via `or()` → `binary_kernel_reduce_vec`

### Key Features:
- Vectorized scalar reduction + vectorized loop
- Two lambda functions: scalar op and SIMD op
- Identity value provided
- Fast path for contiguous reductions

### Code Pattern:
```cpp
binary_kernel_reduce_vec(
    iter,
    [](scalar_t a, scalar_t b) -> scalar_t { return a + b; },  // scalar
    [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a + b; }  // SIMD
);
```

---

## **BIFURCATION 2: Min/Max Value Operations**
**Uses: `binary_kernel_reduce_vec` (most types) + `binary_kernel_reduce` (int64 special case)**

### Operations:
- `min()` - via `min_values_kernel_impl`
  - int64: `binary_kernel_reduce`
  - Others: `binary_kernel_reduce_vec`
- `max()` - via `max_values_kernel_impl` → `binary_kernel_reduce_vec`
- `nanmin()` - similar to min
- `nanmax()` - similar to max

### Key Features:
- Same infrastructure as Bifurcation 1!
- Just different reduction operator (min/max instead of +)
- Vectorized versions available

### Code Pattern (Line 362-368):
```cpp
binary_kernel_reduce_vec(
    iter,
    [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },  // scalar min
    [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return minimum(a, b); }  // SIMD min
);
```

---

## **BIFURCATION 3: Index-Returning Operations (Arg Operations)**
**Uses: `binary_kernel_reduce_lastdim` (if last dim) + `binary_kernel_reduce` (general case)**

### Operations:
- `argmax()` - Line 380-387
  - If last dim only: `binary_kernel_reduce_lastdim`
  - Else: implied general path
- `argmin()` - similar to argmax
- `nanargmax()` - similar to argmax
- `nanargmin()` - similar to argmin

### Key Features:
- Tracks both VALUE and INDEX in accumulator
- `binary_kernel_reduce_lastdim` is **faster** for last-dim reductions
- General `binary_kernel_reduce` handles arbitrary reduction axes

### Code Pattern (Line 382-385):
```cpp
if (is_reduce_lastdim(iter)) {
    binary_kernel_reduce_lastdim(iter, [&](...) { /* process index + value */ });
} else {
    binary_kernel_reduce(iter, ArgMaxOps<T>{}, init_value);
}
```

---

## **BIFURCATION 4: Statistical Operations (Variance/Std)**
**Uses: `binary_kernel_reduce` with `WelfordOps`**

### Operations:
- `std()` - Line 138-149
- `var()` - same kernel
- `nanstd()` - same kernel
- `nanvar()` - same kernel
- `std_mean()` - same kernel
- `var_mean()` - same kernel

### Key Features:
- Uses **WELFORD ALGORITHM** (one-pass, numerically stable)
- NOT two-pass algorithm!
- `WelfordOps` struct encapsulates Welford state
- Single accumulator tracks (count, mean, M2)

### Code Pattern (Line 138-149):
```cpp
binary_kernel_reduce(
    iter,
    WelfordOps<scalar_t, double, int64_t, std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
    WelfordData<double, int64_t>()  // Initial Welford state
);
```

---

## **BIFURCATION 5: Logical Operations (Xor Sum)**
**Uses: `binary_kernel_reduce`**

### Operations:
- `xor_sum()` - via `xor_sum_kernel_impl`

### Key Features:
- General binary_kernel_reduce (not vec)
- Handles XOR accumulation

---

## **SUMMARY TABLE**

| Operation | Bifurcation | Kernel | Path(s) |
|-----------|------------|--------|---------|
| sum | 1 | `binary_kernel_reduce_vec` or `cascade_sum` | Vectorized |
| prod | 1 | `binary_kernel_reduce_vec` | Vectorized |
| all | 1 | `and_stub` → `binary_kernel_reduce_vec` | Vectorized |
| any | 1 | `or_stub` → `binary_kernel_reduce_vec` | Vectorized |
| min | 2 | `binary_kernel_reduce_vec` (mostly) | Vectorized |
| max | 2 | `binary_kernel_reduce_vec` | Vectorized |
| argmax | 3 | `binary_kernel_reduce` + `binary_kernel_reduce_lastdim` | General + Optimized |
| argmin | 3 | `binary_kernel_reduce` + `binary_kernel_reduce_lastdim` | General + Optimized |
| std | 4 | `binary_kernel_reduce` with `WelfordOps` | One-pass Welford |
| var | 4 | `binary_kernel_reduce` with `WelfordOps` | One-pass Welford |
| xor_sum | 5 | `binary_kernel_reduce` | General |

---

## **KEY INSIGHTS**

1. **Bifurcations 1 & 2 are ESSENTIALLY THE SAME** - both use vectorized paths, just different operators
2. **Bifurcation 3** needs special handling for (value, index) state + lastdim optimization
3. **Bifurcation 4** uses Welford algorithm (NOT two-pass) - critical for numerical stability!
4. **All use `binary_kernel_reduce` or `binary_kernel_reduce_vec`** as the foundation
5. **`cascade_sum`** is a SPECIALIZED path for float summation (not a general kernel)

---

## **REFACTORING RECOMMENDATION**

Our implementation should match:

```
reduce_kernel (for Bif 1 & 2):
  - Vectorized paths if applicable
  - Single accumulator
  - Identity value provided

reduce_kernel_index (for Bif 3 general):
  - (value, index) pairs
  - Single output slice per thread with combining

reduce_kernel_index_lastdim (for Bif 3 optimized):
  - (value, index) pairs
  - Each thread processes complete output element
  - No combining needed

reduce_kernel_welford (for Bif 4):
  - Welford state accumulation
  - One-pass algorithm
  - NOT two-pass!
```

