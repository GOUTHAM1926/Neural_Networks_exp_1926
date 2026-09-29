# Loss-Kernel Dtype Bugs — Bug A, Bug B, Bug C, and the Redundant Cast

**Repo:** `Tensor-Implementations` · **Branch:** `sft_gautam` · **Date:** 2026-09-24
**Baseline compared against:** `8808f89` (submodule pinned by `BluTrain/dist_side_merge`)
**⚠️ OPEN, HIGH PRIORITY — Bug D (§17):** `float_to_bfloat16` turns every value ≥ 65,536 into Infinity, on the CPU
**and inside GPU kernels**. It is not fixed. It will be looked into on **2026-09-25**, together with #2 (§19.4).
**Status:** every claim below was **measured on the H200 production trainer on 2026-09-24 — see §16.**
§0–§15 were written before those results; where they disagree, §16 wins, and each superseded claim is marked where it stands.
**Pushed:** `sft_gautam` → `BlubridgeAI/Tensor-Implementations` at **`1ed2ac3`** (2026-09-24, author `goutham.kumar@blubridge.com`). That commit holds exactly the 7 files validated on the H200, checked file by file.

---

## 0. TL;DR

Three bugs in the sparse cross-entropy loss path, plus one redundant memory cost in the
training script. All four are invisible today because a single line in the trainer widens
the logits to fp32 before the loss. That line costs **~9.2 GiB of live peak memory per rank at the
production B=8** (measured 4,716 MiB at B=4, §16.3) and is the only thing preventing three separate failures.

| # | name | failure mode | fires when |
|---|------|--------------|------------|
| A | loss scalar squashed | reported loss quantized to ~0.06 | logits are bf16 |
| B | vector width hardcoded to 4 | crash **and** silent gradient corruption | logits are bf16 |
| C | grad scale read at wrong type | **all gradients become 0.0** | loss dtype != logits dtype |
| 4 | redundant fp32 upcast | ~9.2 GiB live at B=8 + 75% of loss-path DRAM traffic (both **measured**) | always (today) |

**Measured, standalone (RTX 3060):** bf16 loss error `1.565e-03 -> 1.127e-05` (**139x**), fp16 `2.007e-04 -> 8.629e-06` (**23x**).

**Measured on the H200 production trainer, B=4, two repetitions per arm (§16):**
- their code + `BLU_BF16_LOGITS=1` crashes at step 0 (the sanitizer names the kernel); ours runs 20/20
- live peak memory **−4,716 MiB** (exactly 3 × 1,572, identical in both reps) → **≈9.2 GiB at B=8**
- loss-path DRAM traffic **−74.97%** (ncu) · throughput **+0.60%**
- training loss **no detectable change** — fp32 and bf16 differ no more than fp32 differs from itself

**Found along the way:** Bug D in `Types.h` (§17), and a trainer that is not reproducible run to run (§18).

**2026-09-26, all pushed:** Bug D and the bf16/fp16 conversions (§17.3), F1 for Muon and ZeRO-v2 (§22), 7 reduction
bugs (§23), GPU `as_type` (§24); pushes and colleague steps in §25; every pending task in §26.

---

## 1. Background — what you need to know first

### 1.1 Logits

The last layer of the model produces, for every token position, **one raw score per
vocabulary word**. These are *logits*. They are **not probabilities** — they are arbitrary
real numbers like `-12.4` or `8.9`. Bigger means the model prefers that word.

```
x --[ lm_head matmul ]--> logits --[ softmax ]--> probabilities
                          ^                        ^
                   any real number           0..1, sums to 1
```

The loss kernel performs the softmax itself (that is what the `log(sum(exp(...)))` is), so
it consumes **logits**, never probabilities.

Shape at the production config (`bluscript_zero_ckpt.cpp`, B=8, T=4096, V=50304):

```
rows      = 8 x 4096          = 32,768
elements  = 32,768 x 50,304   = 1,648,361,472
fp32 size = 6.14 GiB          bf16 size = 3.07 GiB
```

**The logits tensor is the single largest object in the training step** — larger than the
model weights themselves.

### 1.2 Storage dtype vs compute dtype — the distinction that explains everything

```
STORAGE (in HBM)     = T       <- can be bf16
MATH    (in registers) = float <- ALWAYS fp32
```

Every kernel in this codebase computes in fp32. Tensor cores accumulate in fp32. Our loss
kernel does `expf` in fp32. **"It was computed in fp32" is true of everything and tells you
nothing about how it should be stored.**

Every one of these bugs lives at a **boundary where the kernel touches memory** — never in
the math.

### 1.3 The design rule

> **A tensor's storage dtype should match what its CONSUMER needs — not what produced it,
> and not how precisely it was computed.**

| tensor | consumer | correct storage |
|---|---|---|
| loss scalar | a human reading logs; the cross-rank allreduce | **fp32** (costs 4 bytes) |
| grad scale | the loss kernel's multiply | **fp32** (costs 4 bytes) |
| logits | the loss kernel (converts to fp32 on read, free) | **bf16** |
| grad_logits | a **bf16 matmul** (`LinearBackward.cpp:44`) | **bf16** |

This rule is why Bug A is a bug but the gradient's `static_cast<T>` on store is *not*.

---

## 2. Bug A — the loss scalar was squashed to bf16

### 2.1 The concept

The kernel sums the error of all 32,768 rows carefully in fp32, then writes the answer out.
The problem was **which container it wrote into**: the output type was copied from the
logits. bf16 logits meant a bf16 loss.

### 2.2 The mistake

bf16 keeps roughly 3 decimal digits. Near a loss of 369, the representable bf16 values are
`368` and `370` — nothing between. A carefully computed `369.1512` snaps to `370.0000`.

And it happened **three separate times** on the way out.

### 2.3 The code

`src/Kernels/cuda/loss/LossKernels.cu`

```cpp
// BEFORE  (8808f89)
:192, :288, :459   losses[row] = static_cast<T>(loss);                        // squash 1
:309, :485, :894   T* d_losses = allocate(batch_size * sizeof(T));            // squash 2
:314, :320, :323   sum_reduction_kernel<T><<<...>>>(d_partial, loss_output);  // squash 3

// AFTER
:323, :439         losses[row] = loss;                       // no cast at all
:456, :930, :1214  float *d_losses = allocate(batch_size * sizeof(float));
:476, :483, :486   sum_reduction_kernel<float, float><<<...>>>(...)
```

`include/ops/helpers/LossKernels.h`

```cpp
// BEFORE :26, :36, :46, :61, :94        T* loss_output;
// AFTER  :107, :137                     float *loss_output;
```

### 2.4 The danger

- The loss is the one number everybody reads to judge whether training is healthy. Moving
  in steps of 0.06 makes a real improvement of 0.02 **invisible**, and a slow upward drift
  look flat.
- In an A/B, both arms report the same rounded number and the conclusion is "no difference"
  when there was one.
- It does **not** damage the model — weights still learn correctly. It damages our ability
  to *see* what the model is doing, which is how wrong decisions get made.

### 2.5 The fix, and the measured result

Every intermediate container on the loss path is fp32 **by declaration**, never copied from
the logits type. One number per step, so fp32 costs nothing.

Total cost of keeping all three fp32:

```
d_losses    : 32,768 floats = 128 KB
d_partial   : ~few hundred  = ~1 KB
loss scalar : 1 float       = 4 bytes
              --------------------------
              ~129 KB   (0.002% of the 6.14 GB logits tensor)
```

**Measured** (`Tests/kernel_tests/test_sparse_ce_ignore_index.cu`, RTX 3060 sm_86, 2026-09-23):

```
bf16 loss error  1.565e-03 -> 1.127e-05    (139x smaller)
fp16 loss error  2.007e-04 -> 8.629e-06    ( 23x smaller)
```

The `369.1512` that came back as `370.0000` now reads `369.1480`.

---

## 3. Bug B — the vector width was hardcoded to 4

### 3.1 The concept

Reading memory one element at a time is slow, so the kernel reads a **16-byte block** at
once (the widest single read the hardware supports). How many elements fit in 16 bytes
**depends on the type**:

```
fp32 : 4 bytes each -> 4 elements per 16 bytes
bf16 : 2 bytes each -> 8 elements per 16 bytes
fp64 : 8 bytes each -> 2 elements per 16 bytes
```

### 3.2 The mistake

The number `4` was typed in by hand. The code never asked what type it was reading.

### 3.3 The code — read half

```cpp
// BEFORE  :112, :225, :644, :800
const int64_t vec_size = 4;
extern __shared__ float s_data[];                       // staging buffer typed FLOAT
cp_async_16(&s_data[tid * 4], &row_logits[j]);          // 16 raw BYTES copied in
...
float val = s_data[tid * 4 + k];                        // NO CAST - it is already float

// AFTER  :375-380
constexpr int VEC = LossVec<T>::WIDTH;                  // 4 float / 8 bf16 / 2 double
T *s_stage = reinterpret_cast<T *>(s_data);             // staging buffer typed T
cp_async_16(&s_stage[tid * VEC], &row_logits[v * VEC]);
...
float val = static_cast<float>(s_stage[tid * VEC + k]); // a REAL conversion
```

**Important detail:** the old vector path had **no `static_cast<float>` at all**, because
`s_data` was declared `float`. 8 bf16 values landed in a float array and were read as 4
garbage floats. That is not a conversion — **it is a type pun.**

The old **scalar/tail** path (`:132, :144, :241, :251, :409, :455`) *did* have
`static_cast<float>(row_logits[j])` and was always correct. Only the fast path was broken.
That is why a misaligned row (which fell back to the scalar loop) would have worked fine.

### 3.4 The code — write half

```cpp
// BEFORE  :727, :836
float4* grad4 = reinterpret_cast<float4*>(row_grad);          // row_grad is T* = bf16
reinterpret_cast<float4*>(row_grad)[j / 4] = out_grad;        // writes 16 B per 8 B span

// AFTER  :715-740
constexpr int VEC = LossVec<T>::WIDTH;
using vec_t = LossVec<T>;
const bool aligned16 =
    ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) &&
    ((reinterpret_cast<uintptr_t>(row_grad)   & 0xF) == 0);   // BOTH buffers checked
const int64_t vec_count = aligned16 ? (vocab_size / VEC) : 0;
vec_t *grad_v = reinterpret_cast<vec_t *>(row_grad);
...
out_grad.v[k] = static_cast<T>((prob - indicator) * f_scale);
grad_v[v] = out_grad;
```

`grad_logits` is allocated at the logits dtype — verified, `LossBackward.cpp:246-250`:

```cpp
TensorOptions opts = TensorOptions()
    .with_dtype(saved_logits_.dtype())          // gradient dtype = logits dtype
    .with_device(saved_logits_.device());
```

### 3.5 The dangers — four of them

1. **Crash (read).** A 16-byte read must start at an address divisible by 16. Stepping 4
   elements on a 2-byte type lands on multiples of 8, so half are illegal:
   `misaligned address`. Loud, and at least honest.
2. **Type pun (read).** Where it did not crash, 8 bf16 values were reinterpreted as 4
   floats. Pure garbage into the softmax.
3. **Buffer overrun (write).** 16 bytes written into an 8-byte span, spilling into the
   neighbour's gradients every time. **No crash. Silent corruption.** The model trains on
   wrong gradients and slowly degrades with nothing reported.
4. **No tail loop (write).** For `vocab_size % VEC != 0`, the trailing columns of **every
   gradient row** were never written — left as uninitialised garbage from `Tensor::empty`.
   Our vocab (50304) divides evenly by both 4 and 8, so this never fired in production, but
   it would for any other vocab.

A crash costs an afternoon. A silent wrong gradient costs a whole training run, and we
would be looking at the data or the learning rate for the cause.

### 3.6 The fix

```cpp
template <typename T> struct alignas(16) LossVec {
  static constexpr int WIDTH = 16 / sizeof(T);
  T v[WIDTH];
};
static_assert(sizeof(LossVec<float>)      == 16 && LossVec<float>::WIDTH      == 4, "");
static_assert(sizeof(LossVec<double>)     == 16 && LossVec<double>::WIDTH     == 2, "");
static_assert(sizeof(LossVec<float16_t>)  == 16 && LossVec<float16_t>::WIDTH  == 8, "");
static_assert(sizeof(LossVec<bfloat16_t>) == 16 && LossVec<bfloat16_t>::WIDTH == 8, "");
static_assert(alignof(LossVec<bfloat16_t>) == 16, "16-B vector loads need 16-B alignment");
```

The width is derived from the type, so it can never be out of step again. Applied to all
five live kernels (forward x2, backward x3). The `static_assert`s mean a future type that
does not fit fails the **build**, not the run.

Instantiated dtypes on the production path (`:1096-1098`, `:1110-1112`):
`float`, `float16_t`, `bfloat16_t` — forward and backward.

---

## 4. Bug C — the gradient scale was read at the wrong type

### 4.1 What the grad scale is

We cannot fit a full batch in memory, so it is split into 16 micro-batches whose gradients
are summed. That sum is 16x too large, so everything is multiplied by `1/16`.

**The grad scale is one number: `0.0625`.** Created by the trainer, before training starts.

`bluscript_zero_ckpt.cpp:1657`:

```cpp
Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                 1.0f / static_cast<float>(GRAD_ACCUM));
//                               ^^^^^^^^^^^^^^^^ no dtype given
```

`include/core/Tensor.h:24` — `Dtype dtype = Dtype::Float32;` — so **grad_scale is fp32,
always, unconditionally.** It has no relationship to the logits whatsoever.

### 4.2 The mistake

`src/autograd/backward/LossBackward.cpp`

```cpp
// BEFORE :302-303
auto dispatch_targets = [&](auto* logits_ptr, auto* grad_ptr, auto* grad_out_ptr) {
    using LogitT = std::remove_const_t<std::remove_pointer_t<decltype(logits_ptr)>>;
//        ^^^^^^ the type is taken from the LOGITS

// BEFORE :339-341
dispatch_targets(logits.data<float16_t>(),  grad.data<float16_t>(),  grad_output_cuda.data<float16_t>());
dispatch_targets(logits.data<bfloat16_t>(), grad.data<bfloat16_t>(), grad_output_cuda.data<bfloat16_t>());
//                                                                                    ^^^^^^^^^^^^^^^^
//                                                        the tensor is fp32. we asked for bf16.
```

`Tensor::data<T>()` is a **raw `reinterpret_cast`**. It does not inspect the tensor's dtype.
It hands you the bytes and believes whatever you say they are.

### 4.3 What actually gets read

`0.0625` occupies four bytes. Asking for bf16 takes only the **first two**:

```
        byte0    byte1    byte2    byte3
      +--------+--------+--------+--------+
      |  0x00  |  0x00  |  0x80  |  0x3D  |   = 0.0625 as fp32
      +--------+--------+--------+--------+
       \_______________/ \_________________/
         bf16 reads these   IGNORES these  <- all the information was here
              |
              v
          0x0000  =  0.0
```

Verified for every common scale:

| GRAD_ACCUM | scale | bytes (LE) | read as fp16 | read as bf16 |
|---|---|---|---|---|
| 1  | 1.0     | `00 00 80 3F` | **0.0** | **0.0** |
| 16 | 0.0625  | `00 00 80 3D` | **0.0** | **0.0** |
| 32 | 0.03125 | `00 00 00 3D` | **0.0** | **0.0** |
| —  | 0.1     | `CD CC CC 3D` | **-19.20** | **-107,479,040.0** |

Every clean power of two — which is exactly what loss scaling uses — reads as **exactly
zero**.

### 4.4 The danger

```
grad scale = 0.0
      |
      v
every gradient x 0.0 = 0.0
      |
      v
optimizer: "move each weight by 0.0"
      |
      v
THE MODEL NEVER CHANGES
```

Training runs. GPUs are busy. Loss prints every step. The model is frozen. No crash, no
warning, no log line — just hours of nothing.

For a non-power-of-two scale the other outcome is `-107 million`, i.e. instant NaN. Of the
two, the silent one is far more expensive.

### 4.5 Why it is COUPLED to Bug A — the critical part

Bug C is harmless *today* and becomes catastrophic *the moment Bug A is fixed alone.*

**Before fixing Bug A** (their branch, bf16 logits):

```
FORWARD
  loss_raw = sparse_ce(logits)        -> bf16          (Bug A makes it bf16)
  loss32   = to_dtype(loss_raw, fp32) -> a REAL node is built (bf16 != fp32)
BACKWARD
  loss32.backward(&grad_scale)        -> fp32 seed enters
  ToDtypeBackward converts            -> fp32 to bf16     <-- THE ACCIDENTAL SAVIOUR
  LossBackward receives               -> bf16
  reads it as data<bfloat16_t>()      -> CORRECT. types match by coincidence.
```

The backward seed is also self-consistent — `Engine.cpp:704-707` seeds with
`.with_dtype(root.dtype())`.

**After fixing Bug A only:**

```
FORWARD
  loss_raw = sparse_ce(logits)        -> fp32 ALWAYS   (Bug A fixed)
  loss32   = to_dtype(loss_raw, fp32) -> ReshapeOps.cpp:67 "if (dtype == target) return input"
                                         NO NODE IS BUILT. it vanishes.
BACKWARD
  loss32.backward(&grad_scale)        -> fp32 seed enters
  (no ToDtypeBackward - it does not exist)            <-- SAVIOUR GONE
  LossBackward receives               -> fp32
  reads it as data<bfloat16_t>()      -> WRONG. 0x0000 = 0.0. all gradients zero.
```

| state | outcome |
|---|---|
| Bug C alone | harmless, hidden |
| **Bug A fixed, Bug C not** | **catastrophic — worse than either alone** |
| both fixed | correct |

**Fixing Bug A without Bug C would make things worse, not better.** They must land together.

### 4.6 The fix

```cpp
// AFTER :327-329
if (grad_output_cuda.dtype() != Dtype::Float32) {
    grad_output_cuda = grad_output_cuda.as_type(Dtype::Float32);   // convert once, at the boundary
}

// AFTER :342-345
// grad_out_ptr is `const float*` for every logits dtype -- it is NOT
// deduced from the logits.
auto dispatch_targets = [&](auto* logits_ptr, auto* grad_ptr, const float* grad_out_ptr) {

// AFTER :383-387
dispatch_targets(logits.data<float>(),      grad.data<float>(),      grad_output_cuda.data<float>());
dispatch_targets(logits.data<float16_t>(),  grad.data<float16_t>(),  grad_output_cuda.data<float>());
dispatch_targets(logits.data<bfloat16_t>(), grad.data<bfloat16_t>(), grad_output_cuda.data<float>());
```

Because `const float*` is written into the signature, passing anything else is a
**compile error**, not a runtime throw. The mistake becomes unrepresentable.

---

## 5. Item 4 — the redundant fp32 upcast in the training script

### 5.1 What it is

`bluscript_zero_ckpt.cpp:789`

```cpp
return autograd::to_dtype(lm_head->forward(x_bf), cfg.logit_dtype());
//                        ^ already produces bf16   ^ resolves to Float32 today
```

The matmul already produces bf16. This line widens it to fp32, allocating a whole new
6.14 GB array via a separate convert kernel.

### 5.2 It is redundant — the kernel converts anyway

```
WITH the cast:
  convert kernel   bf16 -> fp32, writes 6.14 GB       <- a whole kernel + 6.14 GB
  inside kernel    static_cast<float>(float)          <- ZERO instructions (identity)

WITHOUT the cast:
  inside kernel    static_cast<float>(bf16)           <- a register-level convert, free
```

`static_cast<float>` on something already `float` is **identity** — the compiler emits
nothing. No runtime check is needed or wanted; an `if` would cost more than the thing it
avoids.

**The explicit cast pays 6.14 GB to do in advance what the loss kernel does for free.**

### 5.3 It is also the wall protecting all three bugs

```
WITH the cast (today):
  matmul -> bf16 [3.07 GB] -> [convert kernel] -> fp32 [6.14 GB] -> loss kernel, T = float   OK

WITHOUT the cast (BLU_BF16_LOGITS=1):
  matmul -> bf16 [3.07 GB] ------------------------------------> loss kernel, T = bfloat16_t  BOOM
```

The loss kernel **never sees the original bf16 tensor** — `to_dtype` creates a brand-new
fp32 tensor and that is what gets passed. So with the cast, `T = float` and **all three
bugs sleep**:

| | with cast (`T = float`) | without cast (`T = bfloat16_t`) |
|---|---|---|
| Bug A | asleep — `static_cast<float>(float)` | **fires** |
| Bug B | asleep — 4 floats = 16 B | **fires** |
| Bug C | asleep — loss fp32, seed fp32 | **fires** |

In practice **Bug B crashes first**, in the forward read, before Bug A's store is reached.
Bug B masks Bug A. That is why the 139x only became measurable after Bug B was fixed.

### 5.4 The costs

```
MEMORY held at once (per rank), B=8 T=4096 V=50304:
  saved_logits_ (kept fwd->bwd) :  6.14 GB fp32  ->  3.07 GB bf16
  grad_logits                   :  6.14 GB fp32  ->  3.07 GB bf16
  -- certain saving             :  6.14 GB
  bf16 matmul out + fp32 copy both live at peak  ->  up to 6.14 GB more
  -- possible peak saving       : 12.28 GB

BANDWIDTH on the loss path per micro-step:
  fp32 path (2 converts + fwd + bwd) :  36.84 GB
  bf16 path (no converts)            :   9.21 GB
  reduction                          :  75.0 %

Also: 2 fewer kernel launches per micro-step.
```

**These are CALCULATED from tensor sizes. NOT MEASURED.**

> **MEASURED 2026-09-24 (§16).** Memory: the live-peak saving at B=4 is **4,716 MiB = 3 × 1,572**,
> identical in both repetitions, i.e. **9,432 MiB ≈ 9.21 GiB at B=8**. The "certain" line above (6.14)
> under-counts; the "possible peak" line (12.28) over-counts, because it priced the extra live buffer as
> fp32 when it is bf16. The correct count: fp32 path = saved fp32 + grad fp32 + one bf16 buffer;
> bf16 path = saved bf16 + grad bf16.
> Traffic: ncu measured **19.714 → 4.935 GB (decimal) per micro-step at B=4 = −74.97%**, which is
> 36.72 → 9.19 GiB at B=8 against the 36.84 → 9.21 predicted here.

### 5.5 No code edit is needed on `dist_side_merge`

`ReshapeOps.cpp:67` — `if (input.dtype() == target) return input;` — verified.

Setting `bf16_logits = true` makes `logit_dtype()` return `Bfloat16`, the `to_dtype` call
becomes bf16 -> bf16, and **no cast node is built at all**. Item 4 is therefore
`BLU_BF16_LOGITS=1` — one env var, zero code edits.

**What the cast line costs when the flag is on** (`ReshapeOps.cpp:63-67`): three CPU bookkeeping macros
(`PROFILER_SCOPE`, `GraphRecordMode::record_forward`, `TRACK_ALLOC_SCOPE_DEEP`), one dtype comparison,
then `return input`. No GPU kernel, no allocation, no autograd node — ncu found no logits-sized convert in
the bf16 run (§16.5). The CPU cost is *estimated* at about 1 µs per micro-step (~0.0003% of a 392 ms
micro-step), not measured. Note it is a **convert**, not a reshape: `to_dtype` just lives in `ReshapeOps.cpp`.

### 5.6 The `loss32` line is now dead — but the comment is the real hazard

`LossOps.cpp:369-376` (CUDA) and `:520-528` (CPU) both build the loss as `Float32`. No
caller can produce anything else, so `to_dtype(loss_raw, Float32)` at `:1994` is provably a
no-op.

The line costs one host-side branch and is harmless. **The 15-line comment above it is
actively wrong** and should be replaced:

| the comment says | truth after the fix |
|---|---|
| "LossOps keys the loss scalar's dtype off the logits" | NO — fp32 by declaration |
| "reading a bf16 scalar as float yields garbage SILENTLY" | cannot happen, never bf16 |
| "backward() must be seeded at the dtype LossBackward reads grad_output at, **which is the LOGITS dtype**" | NO — fp32, nailed in the signature |

That last line **teaches the next reader exactly the misconception that created Bug C.**
Recommended: keep the line (avoids renaming `loss32` at 4 use sites, minimal merge
surface), replace the comment with a statement of the fp32 contract. Do **not** delete the
line and leave the old comment — that is the worst of both.

---

## 6. Why the gradient store is NOT a fourth bug

A reasonable question: the backward computes gradients in fp32 and stores them as bf16 —
is that not the same squash as Bug A?

**No**, and the design rule (§1.3) says why. Look at who consumes it —
`LinearBackward.cpp:44` and `:57`:

```cpp
cuda_matmul_backward(g_flat, a_flat, saved_weight_, grad_input_flat, grad_weight, stream);
//                   ^ grad_logits   ^ saved_weight_ is bf16, a_flat (x_bf) is bf16
```

`grad_logits` feeds straight into a **bf16 matmul**. Storing it as fp32 would mean:

```
compute fp32 -> store fp32 [6.14 GB] -> convert BACK to bf16 -> feed the GEMM
                     ^                          ^
              costs 3.07 GB extra      throws the precision away anyway
```

3.07 GB and an extra convert kernel, to hand the matmul the exact same bf16 numbers.

| | consumer | does precision survive? | verdict |
|---|---|---|---|
| loss scalar | human / allreduce | yes, and costs 4 bytes | **fp32 — Bug A was real** |
| grad_logits | bf16 GEMM | no, discarded immediately | **bf16 — correct as-is** |

Same-looking cast, opposite verdict. **The consumer decides.**

### Why bf16 and not fp16 for this

```
fp32 : 8 exponent bits, 23 mantissa bits
bf16 : 8 exponent bits,  7 mantissa bits   <- SAME range as fp32, coarser steps
fp16 : 5 exponent bits, 10 mantissa bits   <- smaller range, can underflow
```

bf16 has fp32's full range: a gradient representable in fp32 is representable in bf16, just
rounded. It cannot silently underflow the way fp16 can. This is why bf16 is the standard
choice for activation gradients (PyTorch AMP, Megatron, Liger all do this).

What must stay fp32 is the **optimizer masters and moments** — already locked down in this
codebase. AdamW's decoupled decay is a `(1 - lr*wd) = 1 - 6e-5` multiply and bf16's
relative grid is `2^-8 = 3.9e-3`, so a bf16 master would round the entire weight-decay term
away. That one is genuinely unsafe. Activation gradients are not.

---

## 7. Why nobody ever hit these

Every trainer in the repo widens the logits to fp32 before the loss. The bf16 branch of the
dispatch (`LossBackward.cpp:339/341` in the old code) was **never entered**.

"Mixed precision" means the **model** runs bf16 — not that the **loss** receives bf16.

The bugs sat armed since at least `8808f89` (2026-09-08), behind a branch nobody walked
into.

---

## 8. Option 1 vs Option 2 — the two valid designs, and which libraries use which

The root enabler of Bug C is that `Tensor::data<T>()` is a raw `reinterpret_cast` with no
dtype check. There are exactly two defensible ways to handle that, and **we were doing
neither.**

### Option 1 — ASSUME the type, but CHECK it. (PyTorch)

`aten/src/ATen/native/cuda/Loss.cu:567-580` binds `scalar_t` from `input.scalar_type()` and
then reads `grad_output.const_data_ptr<scalar_t>()` — **the same assumption we made.** The
difference is that `const_data_ptr` calls `check_type()`
(`aten/src/ATen/templates/TensorMethods.cpp:22`), which **throws** on mismatch.

- Detects the bug **at runtime**, framework-wide, at all ~827 call sites.
- Kept out-of-line on purpose: *"Purposely not defined here to avoid inlining"*,
  `ATen/core/TensorBase.h:650`.
- Needs an **alias allowance** — PyTorch permits a quantized tensor to be read as its
  underlying type.

### Option 2 — DON'T ASSUME. (Liger, Megatron) — **what we shipped**

- **Liger** multiplies tensors and lets Triton read the pointer's own dtype:
  `ops/cross_entropy.py:483`, `ops/utils.py:135` (`tl.load(grad_output_ptr)`).
- **Megatron** does `grad_input.mul_(grad_output.unsqueeze(dim=-1))`:
  `core/tensor_parallel/cross_entropy.py:19`.

Neither asserts a dtype anywhere. We reach the same category through **C++ types instead of
a dynamic framework**: the type is in the signature (`const float*`), so a mismatch is a
**compile error**.

### Why option 1 was rejected FOR NOW (not on merit)

Counts re-verified 2026-09-24 (the header note's 840/59/38 predate the 469-line dead-code
deletion):

```
827  data<T>() call sites total
 30  reinterpret ON PURPOSE  (data<__nv_bfloat16>, data<__half>, data<half>)
 16  indexed per-element inside a loop  (e.g. LossBackward.cpp:273
     `saved_targets_.data<int64_t>()[i]` per row)
```

1. **827 call sites** — an 840-site framework change landing on a branch whose test
   coverage is one cross-entropy kernel.
2. **Per-iteration cost.** The in-loop sites would pay the check per element on paths that
   have none today.
3. **30 sites reinterpret deliberately**, so the tensor's bytes can be fed to CUDA
   intrinsics. A strict check **fails to compile** there. PyTorch hit the same wall and
   solved it with an alias allowance — we would need an equivalent alias table first.
4. **dtype is a RUNTIME property**, so a check only surfaces mismatches on paths that
   actually execute. It cannot be validated by a test suite covering one kernel.

### The decisive distinction

> **Option 1 DETECTS the bug at runtime. Option 2 makes it UNREPRESENTABLE at compile time.**

Compile-time wins where it applies. But option 2 only applies where the type can be nailed
in a signature — option 1 covers **everything**. They are **complementary, not competing**,
and we should eventually have both.

### RESEARCH / FUTURE TASK — implementing option 1

Logged in `include/ops/helpers/LossKernels.h:71-82`:

- **a.** Build the alias table first (`bfloat16_t <-> __nv_bfloat16`,
  `float16_t <-> __half/half`), mirroring PyTorch's `isQIntType` allowance.
- **b.** Hoist the in-loop `data<T>()` calls out of their loops, the way PyTorch uses
  accessors, so the check is paid once per kernel and not per element.
- **c.** Keep it out-of-line, as PyTorch does (`ATen/core/TensorBase.h:650`).
- **d.** Gate it on a **full-repo build plus the widest test run available** — not the loss
  suite. The failure mode is a runtime throw in code we do not cover.

---

## 9. Audit — status on `dist_side_merge`

`BluTrain/dist_side_merge` (tip `74ec046`, 2026-09-18) pins submodule **`8808f89`**
(pinned by BluTrain tip 03b94cf, 2026-09-23 "DCP fix"), *not* the `Tensor-Implementations/dist_side_merge` tip
`c6f0dbb`. The pinned commit is what compiles, so that is what was audited.

| item | status | evidence |
|---|---|---|
| Bug A | **PRESENT** | `:192, :288, :459` `static_cast<T>`; `:309, :485, :894` `T* d_losses` |
| Bug B | **PRESENT** | `:112, :225, :644, :800` `tid * 4`; `:727, :836` `float4*` on `row_grad` |
| Bug C | present, **does not fire** | self-consistent there — see §4.5 |
| `ignore_index` (SFT) | **ABSENT** | 0 occurrences in `LossOps.cpp` |
| the fp32 upcast | **PRESENT and active** | `bf16_logits = false` at `:188` |

### A false claim in the script's comment

`bluscript_zero_ckpt.cpp` (~`:168-175`) states:

> *"bf16_logits is wired end to end and the sparse-CE kernels now accept it: LossKernels.cu's
> vectorized paths key every chunk off 16/sizeof(T) elements instead of a hard-coded 4, in
> the forward and in both backward variants (see VEC_ELEMS / Chunk16 there)."*

**`VEC_ELEMS` and `Chunk16` do not exist.**

```
git log --all -S "VEC_ELEMS" -- src/Kernels/cuda/loss/LossKernels.cu   -> (nothing)
git log --all -S "Chunk16"   -- src/Kernels/cuda/loss/LossKernels.cu   -> (nothing)
```

The pinned kernel is `tid * 4` throughout. **`BLU_BF16_LOGITS=1` will die with
`misaligned address`** — after printing a message telling the user it is safe:

```
[precision] bf16 logits ON - frees ~3 GiB ([B,T,vocab] fp32 activation + gradient). ...
```

The knob reads as tested and shipped. It is neither.

> ~~**Caveat:** the local copy of that branch is from 2026-09-18 …~~ **Resolved 2026-09-24**, on a fresh
> clone with all **84** pushed `Tensor-Implementations` branches:
> - `git log --all -S VEC_ELEMS -- src/Kernels/cuda/loss/` is **empty**, so `VEC_ELEMS` was never added to
>   the loss kernels in any pushed commit. (The only `VEC_ELEMS` in the repo is in `TensorDataManip.h`,
>   unrelated.)
> - `Chunk16` appears **nowhere** in the repo's history.
> - No branch tip has `VEC_ELEMS` in `LossKernels.cu`.
> - The comment was added in BluTrain **`beda19d` (2026-09-08, bruce-yeager, "memory fix")**. That commit
>   changed `bluscript_zero_ckpt.cpp`, `PlannerLimits.h` and `docs/memprobe*.cpp`. It never touched the loss
>   kernels or the submodule pointer.
>
> **If a colleague reports `BLU_BF16_LOGITS=1` running**, it cannot have been the pushed code, which
> crashes at step 0 (§15.3). Check, in this order:
> 1. `grep -c VEC_ELEMS Tensor-Implementations/src/Kernels/cuda/loss/LossKernels.cu` in *their* tree.
>    Non-zero means a local, unpushed Bug B fix. That fits their comment exactly: "kernels now accept it"
>    plus "the reported loss is bf16 and quantizes to ~0.06" is the state "Bug B fixed, Bug A not". It must
>    be reconciled with this port before merging, since both change `LossKernels.cu`.
> 2. The run log's config line. `logits float32` means the flag never reached the process — for example
>    `mpirun` without `-x BLU_BF16_LOGITS`.
> 3. `git -C Tensor-Implementations rev-parse --short HEAD` and the trainer file actually run.

### Where their comment is RIGHT

The same block says *"GRADIENTS ARE UNAFFECTED"*. **On their branch that is correct**, for
the reason in §4.5: the loss dtype and the seed dtype agree, so the wrong-type read lines up
by coincidence. Their design is self-consistent. The zero-gradient failure is **created by
fixing Bug A**, not by their code.

---

## 10. Benefits — expected vs measured

| benefit | expected (before the H200 run) | **measured 2026-09-24** |
|---|---|---|
| Reported-loss precision (bf16) | 139x | **139x** (RTX 3060, standalone) |
| Reported-loss precision (fp16) | 23x | **23x** (RTX 3060, standalone) |
| Memory, live peak | 6.14–12.28 GB at B=8 | **−4,716 MiB at B=4**, identical in both reps → **≈9.2 GiB at B=8** |
| Memory, reserved (what nvidia-smi shows) | — | **−5,876 MiB at B=4** |
| Loss-path DRAM traffic | −75% | **−74.97%** (19.714 → 4.935 GB per micro-step, ncu) |
| Kernel launches | 2 fewer per micro-step | **2 fewer** (ncu kernel sequence) |
| tok/s | unknown | **+0.60%** (n=2 per arm; every bf16 run faster than every fp32 run) |
| Training loss | "must track fp32" | **no detectable change** (§16.4) |

Memory saved != speed gained. 14.8 GB less traffic per micro-step would be 0.79–0.94% if every
byte became time at 4.0–4.8 TB/s; +0.60% was measured (§16.6).

### The benefit their own comment called impossible

> *"What is NOT fixed, and is **not fixable in those kernels**: ... Decoupling the loss dtype
> from the logits dtype needs a LossT template parameter through LossKernels/LossOps plus a
> grad_output dtype check in LossBackward."*

That is precisely what was built — fp32 `loss_output`, fp32 `grad_output`, `const float*` in
the dispatch. Their note says choose between the memory saving and a trustworthy loss. The
choice is removed.

---

## 11. Test plan

The fixes alone give **zero** benefit. At fp32 logits our entire change set is a bit-for-bit
no-op (§5.3). The flag is what delivers the memory.

| arm | config | expected |
|---|---|---|
| **0** | their branch, `BLU_BF16_LOGITS=1` | **should CRASH** — `misaligned address`. Proves Bug B on the real trainer. |
| **1** | their branch, `BLU_BF16_LOGITS=0` | baseline |
| **2** | our fixes, `BLU_BF16_LOGITS=0` | **must be BIT-IDENTICAL to arm 1** |
| **3** | our fixes, `BLU_BF16_LOGITS=1` | the 6.14 GB |

Arm 2 is the one people skip. Without it, a win in arm 3 could be the port changing
something unrelated. Three of the four arms are yes/no answers rather than noisy
measurements.

**Outcome:** executed 2026-09-24 — arms 0/1 in §15, clean arms 2/3 with two repetitions each in §16.

### ~~Arm 3 is a NUMERICAL change~~ — CORRECTED 2026-09-24: it is not

This section originally claimed `BLU_BF16_LOGITS=1` gives the logits "7 mantissa bits instead
of 23". **That was wrong.** The lm_head matmul already writes its output as bf16
(`GenMatmul.cpp:89`: `Tensor output(output_shape, A.dtype(), ...)`, with `A = x_bf`), so the logits
never held more than 7 mantissa bits of information. Widening to fp32 appends 16 zero bits
(`0x4049` → `0x40490000`, both exactly 3.140625). In backward, `ToDtypeBackward` rounds the fp32
gradient straight back to bf16 before the matmul backward reads it. Both paths hand the matmul
the same bf16 gradient, apart from fp32 summation order inside the loss kernel (4 vs 8 elements
per 16-byte load).

Measured: `test_cast_equivalence.cu` gives 99.999% of gradient elements bit-identical and the rest
1 ulp, identical on the RTX 3060 and the H200 (§16.2). In training, fp32 and bf16 differ by no more
than each differs from itself (§16.4). **The cast costs memory and time and buys nothing.**

The arm table's "arm 2 must be BIT-IDENTICAL to arm 1" was also the wrong expectation: the trainer
is not reproducible run to run (§18). What holds is that step 0 is identical (loss, val, gradient
norm) and later steps stay within the same-arm noise.

### Measurement discipline (from prior campaigns)

- Do **not** shorten `max_steps` to benchmark — changes warmup/LR, costs ~3.4% switching
  power = -2.46% throughput at the 700 W cap.
- Compare the **same step window** across arms.
- One config per process.
- Pull run artefacts off the box immediately.
- **Never compare single runs' early losses on this trainer.** The same binary on a clean GPU gave a
  step-3 loss of 14.98 in one run and 11.17 in the next (§18). Use ≥2 repetitions per arm.
- Measure memory with the allocator (`CachingCUDAAllocator::get_stats()`), not a 0.5 s nvidia-smi
  poller — the poller missed the true peak by ~660 MiB (§16.3).

---

## 12. Port list for `dist_side_merge`

`ignore_index` is **SFT-only** — pretraining has no masked labels. Leave it out of this
port: extra merge risk for zero pretraining benefit.

| # | file | change |
|---|---|---|
| 1 | `src/Kernels/cuda/loss/LossKernels.cu` | Bug A + Bug B (`LossVec<T>`, fp32 buffers) |
| 2 | `include/ops/helpers/LossKernels.h` | fp32 signatures + design note |
| 3 | `src/autograd/backward/LossBackward.cpp` | Bug C (`const float*` + boundary convert) |
| 4 | `src/autograd/operations/LossOps.cpp` | fp32 loss tensor, both paths |
| 5 | `bluscript_zero_ckpt.cpp` | **comment only** — correct the stale block (§5.6) |
| — | `BLU_BF16_LOGITS=1` | env var, no edit |

**This is a real merge, not a copy.** Their branch added `AccT`
(`std::conditional_t<(sizeof(T) < 4), float, T>` — a partial fix for the same squash) where
we added `LossVec<T>` + fp32 scalars. Those touch the same lines and conflict.

---

## 13. Open items (updated 2026-09-24 evening; the full current list is §26)

Done on 2026-09-24:
- [x] Bugs verified present at today's tip `8808f89`, on the box (§15.1)
- [x] Port onto `dist_side_merge` — 7 files, not 5 (§12, §20.3)
- [x] Run the arms on the H200 (§15, §16)
- [x] Measure memory, traffic and precision (§16)

Still open:
- [x] **Commit + push `sft_gautam`**: `1ed2ac3` on BlubridgeAI (the 7 loss files + both tests)
- [x] **DONE 2026-09-26, pushed `7810aea` (§17.3). Was: ⚠️ Bug D — HIGH PRIORITY, 2026-09-25 (with #2):** `float_to_bfloat16` (`Types.h:85`) turns every value ≥ 65,536
      into Infinity, on the CPU and inside GPU kernels. Find where it can hit, fix it, run the exhaustive test, then do a
      clean rebuild and a full-repo test run (plan in §17.1)
- [ ] Hand the port to the `dist_side_merge` owners as a proper branch/PR, recommending `BLU_BF16_LOGITS=1`
- [ ] Rewrite the stale comment block above `loss32` in `bluscript_zero_ckpt.cpp` (§5.6)
- [ ] Remove the false `VEC_ELEMS / Chunk16` claim in the same file (§9)
- [ ] More throughput repetitions: +0.60% is n=2 per arm, and the gap to the 0.79–0.94% bytes-only prediction is unexplained (§16.6)
- [ ] Which third buffer is live at the peak (bf16 matmul output vs `ToDtypeBackward` output): memory is measured, attribution is not (§16.3)
- [ ] The 580 MiB fp32 gap between their code and ours: measured by the poller only, probably a poller artefact (§16.3)
- [ ] Bug C's consequence (zero gradients): predicted, never observed; needs a Bug-A-only build
- [ ] The source of the trainer's run-to-run non-determinism (§18)
- [x] **DONE 2026-09-25, pushed `a264586` (§19.9). Was: #2:** convert the gradient shard straight into `shard_grad_fp32_` — the same
      fix already exists in `cast_master_range`, in the same file (§19.4)
- **Immediate future tasks — right after the SFT work, as training-script optimisations (decided 2026-09-24):**
  - [ ] **#1:** read how the embedding backward sums gradients for a bf16 weight (code, no GPU);
        then `BLU_BF16_EMBED` 0 vs 1 A-B-A-B; then a long run (§19.3)
  - [ ] **#3:** research option 1, `check_type()` in `data<T>()` (§19.5, §8)
  - [ ] **#4:** measure why the speed gain (+0.60%) is below the traffic-only prediction (0.79–0.94%):
        per-kernel durations of the loss path, from an unprofiled nsys run (§19.6)
- [ ] Apply the fixes on a colleague's multi-GPU system; run flag 0/1 before and after (§21)
- [ ] Refresh the `LossKernels.h` note counts (840/59/38 -> 827/16/30)
- [ ] **RESEARCH:** option 1 (`check_type()` in `data<T>()`) — see §8
- [ ] `batch_size > 65536` reduction cap
- [ ] Target bounds check
- [ ] sm103/B300 masking (`TODO(SFT)`)

---

## 14. Summary — the three mistakes in one table

| bug | the mistake |
|---|---|
| **Bug A** | copied the dtype from the logits **where it should not have** (loss scalar) |
| **Bug B** | did **not** copy the dtype from the logits **where it should have** (vector width) |
| **Bug C** | copied the dtype from the logits **where it should not have** (grad scale) |

A and C are the same mistake in two places. B is its mirror image. All three are invisible
until the logits stop being fp32 — and the line that keeps them fp32 is the 6.14 GB we are
trying to delete.

---

## 15. H200 VALIDATION — 2026-09-24 (measured, not calculated)

Box: Verda `cold-bear-flows-fin-02`, 1x NVIDIA H200 143,771 MiB, driver 580.178.04,
CUDA toolkit 13.0.88, gcc 13.3, 44 cores. Build: `sm_90a`, BluBridge-BLAS, 0 errors.

Clone: `BluTrain@dist_side_merge` tip `03b94cf` (2026-09-23), submodule `8808f89`.
**NOTE:** this is NOT the `a4070931` analysed earlier — the branch moved. Every bug
below was re-verified against `8808f89` on the box.

### 15.1 Smoke gate — all bugs present at today's tip

```
tid * 4                            : 11    (Bug B present)
losses[row] = static_cast<T>(loss) :  3    (Bug A, all three squashes)
VEC_ELEMS                          :  0    (the claimed fix DOES NOT EXIST)
auto* grad_out_ptr                 :  1    (Bug C present)
const float* grad_out_ptr          :  0    (not fixed upstream)
ignore_index in LossOps            :  0    (SFT work absent, as expected)
AccT                               : 41    (their partial squash fix)
7th file touching loss kernels     : NONE  (6-file swap is safe)
bf16_logits default                : false @ :188
```

### 15.2 Real model config (corrects §1.1 and §5.4)

The on-box config differs from the source defaults read from a stale ref:

```
  d_model 2304 | n_layers 26 | q/kv heads 18/6 | head_dim 128 | ffn 6144
  vocab 50304  | weight_tying ON | B x T = 8 x 4096 | world_size 1
  parameters   : 1,588,229,888  (1.588 B)
  param dtypes : 1472 M bf16 (AdamW G0) + 116 M fp32 (AdamW G1)
  optimizer/GPU: 16,848 + 1,328 = 18,176 MiB  UNSHARDED at world_size=1
  max_lr 0.0012 | warmup 200 | max_steps 8000 | grad_accum 32
  peak BF16    : 989.4 TFLOP/s | no activation checkpointing (HFU == MFU)
```

The logits figures in §5.4 are UNCHANGED: they depend only on `B*T*V`, not on model
size. fp32 6,288 MiB each, bf16 3,144 MiB each, saving 6,288 MiB at B=8.

### 15.3 ARM 0 — Bug B PROVEN, kernel named

`BLU_BF16_LOGITS=1` on their unmodified code:

```
[precision] bf16 logits ON - frees ~3 GiB ... Gradients are unaffected ...
[mix] step 0 entering phase 'c0-static'
terminate called after throwing an instance of 'std::runtime_error'
  what():  GPU->CPU transfer failed: misaligned address
```

The knob printed its own "this is safe" banner and died on step 0.

The first trace pointed at `Tensor::to_cpu()`, which is only where the STICKY CUDA
error surfaced -- not the fault site. `compute-sanitizer --tool memcheck` named it:

```
========= Invalid __global__ read of size 16 bytes
=========   at OwnTensor::cuda::sparse_ce_forward_kernel_vec<OwnTensor::bfloat16_t, unsigned short>+0x590
=========   by thread (65,0,0) in block (26,0,0)
=========   Access to 0x554607ec08 is misaligned
=========   inside the nearest allocation at 0x5545e00000 of size 3296722944 bytes
=========   Host Frame: sparse_cross_entropy_forward_cuda_impl_vec<bfloat16_t, unsigned short>
=========   Host Frame: autograd::sparse_cross_entropy_loss
=========   Host Frame: main in bluscript_zero_ckpt.cpp:2102
========= ERROR SUMMARY: 50253 errors
```

Six independent confirmations of the Bug B prediction:

| predicted | observed |
|---|---|
| the VECTORIZED loss kernel faults | `sparse_ce_forward_kernel_**vec**` |
| only for bf16 | templated `<bfloat16_t, unsigned short>` |
| a 16-byte read | `Invalid __global__ read of size 16 bytes` |
| address 8-aligned, not 16 | `0x...ec08` -> **mod 16 = 8** |
| it reads the logits tensor | allocation `3,296,722,944` == `B*T*V*2` EXACTLY |
| one block per row | fault offset -> row 26; sanitizer says `block (26,0,0)` |

50,253 errors, not one -- pervasive, not a corner case.

METHOD NOTE: the first run's stack trace was NOT sufficient evidence. `misaligned
address` is sticky in CUDA, so it surfaces at the next synchronising call and names
the wrong function. Only `compute-sanitizer` (or `CUDA_LAUNCH_BLOCKING=1`) identifies
the faulting kernel. Do not claim a kernel from a sticky-error stack trace.

### 15.4 ARM 1 at B=8 — OOM at exactly the tensor this work halves

`BLU_BF16_LOGITS=0` on their unmodified code:

```
  [val] step 0 | val_loss 11.282295 | commoncrawl 11.282296 | code 11.282296 | math 11.282296
[ALLOCATOR] cudaMallocAsync(size=6593445888) failed: out of memory
[ALLOCATOR] Hard OOM. Trimming cache...
[ALLOCATOR] FATAL: Still OOM after trim.
what():  CUDA OOM
```

Stack: `Tensor::backward -> backward_sequential -> SparseCrossEntropyLossBackward::apply
-> Tensor::empty -> Storage` = the `grad_logits_2d` allocation at LossBackward.cpp:250.

```
failed allocation          : 6,593,445,888 bytes
B*T*V*4 (fp32 grad_logits) : 6,593,445,888 bytes    EXACT MATCH
```

It trimmed the allocator cache and still could not find 6,288 MiB, so that memory was
live, not merely cached.

```
card               : 143,771 MiB
observed peak      : 141,240 MiB   (98.2% of card)
+ the failed alloc :   6,288 MiB
= would need       : 147,528 MiB   -> OVER BY 3,757 MiB
```

**Their shipped config does not fit on a single H200.** With bf16 logits it would need
141,240 MiB and fit with 2,531 MiB to spare -- but that margin is too thin to rely on
while the box is shared.

CONFOUND, stated plainly: `snippet_runner` (another user) held between 908 and 25,466
MiB throughout. The absolute peak is therefore contaminated and contention may have
triggered the failure earlier than it otherwise would. What survives the confound is
the arithmetic: 141,240 + 6,288 > 143,771 regardless of who else was on the card.

### 15.5 Why B=8 fails here but works in production

ZeRO-1 shards optimizer state, nothing else. At world_size=1 there is nothing to shard:

```
  8 GPUs : optimizer state per GPU = 18,176 / 8 =  2,272 MiB
  1 GPU  : optimizer state per GPU =             18,176 MiB
                                                 -----------
                                     we pay      +15,904 MiB
```

Activations are per-GPU regardless of world size, so B=8 costs the same per GPU either
way. The unsharded optimizer is the entire difference.

### 15.6 B=4 chosen for the A/B

Fixed costs (do not scale with B), from the run's own log:

```
  params  (1472M bf16 + 116M fp32) :  3,251 MiB
  grads                            :  3,251 MiB
  optimizer state (unsharded)      : 18,176 MiB
  FIXED                            : 24,677 MiB
```

Everything else -- activations, logits, grad_logits -- is exactly linear in B. From the
one measured point (B=8 needed 147,528 MiB), per unit of B = 15,356 MiB.

| B | grad_accum | fp32 | bf16 | saving |
|---|---|---|---|---|
| 8 | 32 | 147,528 OOM | 141,240 fits (2.5 GB spare) | 6,288 MiB |
| **4** | **64** | **86,103 fits** | **82,959 fits** | **3,144 MiB** |
| 2 | 128 | 55,390 fits | 53,818 fits | 1,572 MiB |

ROBUSTNESS ARGUMENT (independent of the fixed-cost breakdown): if F is the true fixed
cost, B=4 needs `(147,528 + F)/2`. For that to exceed 143,771 you would need
`F > 140,014 MiB` -- impossible, since the two logits tensors alone are 12,576 MiB of
the 147,528 total. B=4 fits no matter how wrong the breakdown is, because no term grows
faster than linearly in B.

B=4 is chosen over B=2: both fit, but B=4 keeps the saving at 3,144 MiB (unmissable)
with ~57 GB headroom so the co-tenant process cannot corrupt the measurement.

**MEASURED:** at B=4 the fp32 run's live peak is 95,038 MiB and its reserved peak 104,904 MiB (allocator,
§16.3). The 86,103 prediction above was 18% low: the fixed-cost parts list missed ~37 GB (CUDA context,
BLAS workspaces, NCCL, ZeRO staging, allocator cache). The robustness argument still held — B=4 ran.

### 15.7 Revised arm table

| arm | code | B | flag | outcome |
|---|---|---|---|---|
| 0 | theirs | 8 | bf16 | **CRASH** - misaligned, kernel named by sanitizer. DONE |
| 1 | theirs | 8 | fp32 | **OOM** at exactly the fp32 grad_logits. DONE (contended) |
| 1b | theirs | 4 | fp32 | baseline: loss curve + tok/s + peak memory |
| 2 | ours | 4 | fp32 | must match arm 1b |
| 3 | ours | 4 | bf16 | expect -3,144 MiB |

Arm 0's result is B-independent (alignment does not depend on batch size), so it does
not need re-running at B=4.

### 15.8 Operational notes

- `mpirun` needs `--allow-run-as-root` on this box.
- `BLU_MAX_STEPS` must be >= 4. At 2 the schedule collapses
  (`warmup + stable >= max_steps - 1`) and the run aborts before any GPU work,
  which makes `compute-sanitizer` report a misleading "0 errors".
- The trainer reports no memory statistics. Peak must be measured by polling
  `nvidia-smi --query-compute-apps` and filtering on the process name, so a co-tenant
  process is excluded.
- `B` is hardcoded at `bluscript_zero_ckpt.cpp:198` with no env override. Editing it
  survives the 6-file swap, since the tarball only contains `Tensor-Implementations/`.

---

## 16. H200 A/B — final measured results (2026-09-24)

### 16.1 What was run

| | |
|---|---|
| box | Verda `cold-bear-flows-fin-02`, 1× H200 143,771 MiB, driver 580.178.04, CUDA 13.0.88, gcc 13.3, 44 cores |
| code | `BluTrain@dist_side_merge` tip `03b94cf` (2026-09-23), submodule `8808f89`, plus our 7 loss files (§20.3) |
| trainer | `bluscript_zero_ckpt.cpp`, 1.588 B params, **B=4** × T=4096, grad_accum 64, world_size 1, BluBridge-BLAS |
| session | one exclusive GPU hour, 12:01–12:54 box time, run by `gpu_hour.sh` (Appendix A, md5 `e5d5c998…`) |
| binary | `_runner_OURS_MEM`, md5 `1967eba692a68363faec2df14ed1bb6b`; preflight checked the 4-parameter loss symbol, `count_valid_targets_cuda`, and the `[MEM]` / `BLU_VAL_STEPS` strings **in the binary** |
| conditions | GPU empty before and after every run. Three `watch nvidia-smi` loops (1 s, 0.1 s, 0.1 s) ran for the whole session — constant for every run |

Layout:

1. **Block 1** — the precision test on the H200's own sm90 convert path.
2. **Block 2** — four training runs in A-B-A-B order (fp32, bf16, fp32, bf16), 20 steps each, allocator memory stats, no GPU poller of our own.
3. **Block 3** — ncu DRAM bytes, arm 2 vs arm 3 (`BLU_MAX_STEPS=5`, `BLU_VAL_STEPS=1`).

### 16.2 Precision — the cast adds no information

`Tests/kernel_tests/test_cast_equivalence.cu` feeds **identical bf16 logits** through both paths. Both use
the library's own code: `convert_type_cuda_generic`, which is what `Tensor::as_type` calls, and the
production stats kernels. The predictions are written in the file header before the first run.
B=2048, V=50304.

| check | init-like (σ=0.5) | trained-like (σ=3, target +8) | prediction |
|---|---|---|---|
| C0 widen bf16→fp32 exact | 0 / 103,022,592 | 0 / 103,022,592 | 0 |
| C5 control: path B run twice | 0 differences | 0 differences | 0 |
| C1 row max | 0 / 2,048 | 0 / 2,048 | 0 |
| C2 row sum | worst rel 1.8e-7 | worst rel 4.3e-7 | ≤ ~1e-6 |
| C3 loss | bit-identical | bit-identical | ≤ ~1e-6 |
| **C4 gradient into the matmul** | **99.9993%** identical, 771 at 1 ulp, **0 > 1 ulp** | **99.9989%** identical, 1,124 at 1 ulp, **0 > 1 ulp** | none > 1 ulp |

**The output was identical on the RTX 3060 and on the H200, every line of both cases.** Those are two
architectures and two different convert kernels (`__float2bfloat16` generic vs `__float22bfloat162_rn` on
sm90), so they round identically and the loss kernels are deterministic across GPUs.

The mechanism is in §11 (corrected).

### 16.3 Memory — allocator stats, both repetitions

| | fp32, cast ON (rep1 = rep2) | bf16, cast OFF (rep1 = rep2) | saving |
|---|---|---|---|
| **live peak** (`active_peak`) | 95,038 MiB | 90,322 MiB | **−4,716 MiB** |
| reserved peak (`reserved_peak`) | 104,904 MiB | 99,028 MiB | −5,876 MiB |
| cached free at end | 57,701 MiB | 53,005 MiB | |

- **Both repetitions of each arm reported identical numbers**, to the MiB. So the saving is not noise.
- **4,716 = 3 × 1,572 exactly.** One bf16 logits tensor at B=4 is 1,648,361,472 bytes = 1,572 MiB. At the
  peak, the fp32 path holds three such units more than the bf16 path: the saved logits (fp32 vs bf16),
  the gradient (fp32 vs bf16), and one extra bf16 buffer that only the fp32 path has.
- **Which buffer is the third is not measured.** It is either the bf16 matmul output, or the
  `ToDtypeBackward` output that coexists with both fp32 tensors.
- **At B=8 this is 9,432 MiB ≈ 9.2 GiB of live peak memory.** Their banner says "frees ~3 GiB".

**History of this number, kept for honesty.** Earlier in the day the saving was quoted from a 0.5 s
nvidia-smi poller:
- first **4,704** (their fp32 vs our bf16 — two variables at once), "fully accounted to 0.26%";
- then **4,124** (our fp32 vs our bf16), "only 87% of the model".

The allocator shows why both were off. The poller under-read the reserved peak by ~660 MiB
(104,240 poller vs 104,904 allocator for the same config), because a half-second sample misses short
peaks. The first model (three tensors) was right.

The **580 MiB** gap between their fp32 and our fp32 was measured by the poller only. It is probably an
artefact of the poller, and is unresolved.

### 16.4 Loss and gradients — fp32 vs bf16 is inside the noise

| step | fp32 rep1 | fp32 rep2 | bf16 rep1 | bf16 rep2 |
|---|---|---|---|---|
| 0 | 11.298004 | 11.298004 | 11.298004 | 11.298004 |
| 1 | 11.960425 | 11.929283 | 11.952442 | 11.932809 |
| 2 | 10.046557 | 10.039776 | 10.042213 | 10.041757 |
| **3** | **14.983250** | **11.171641** | **15.361207** | **10.660113** |
| 4 | 12.608994 | 12.981804 | 12.588583 | 12.816082 |
| 10 | 9.117052 | 9.068036 | 8.480481 | 9.152593 |
| 19 | 7.969253 | 8.019695 | 7.865052 | 8.018976 |
| **val @19** | 8.137194 | 8.202946 | 8.046893 | 8.187601 |

| mean \|loss diff\| over 20 steps | |
|---|---|
| fp32 vs fp32 (same thing, twice) | 0.435 |
| bf16 vs bf16 (same thing, twice) | 0.611 |
| fp32 vs bf16 (rep1 / rep2) | 0.355 / 0.156 |

- The gradient norm at step 0 is 18.9713 in both fp32 runs and 18.9715 in both bf16 runs. The 0.0002 is
  the summation-order effect from test B.
- **Conclusion:** no detectable effect of bf16 logits on training. The bf16 val loss at step 19 is lower
  in both reps (−0.090, −0.015), but that is well inside the fp32 rep-to-rep spread (0.066). Not a result.

**Correction:** the step-3 split seen earlier in the day was put down to contention with a co-tenant
job. **That was wrong.** These runs had an empty GPU and split just as much. The split is the trainer's
own run-to-run non-determinism (§18).

### 16.5 DRAM traffic (ncu)

Both runs profiled exactly **2,500 kernels**: 3 validation forwards plus 3 complete training
micro-steps. The CSV's real header is on line 3, after ncu's `==PROF==` lines, and it was located, not
assumed. Units were forced to bytes.

| per training micro-step (GB, decimal) | arm 2, cast ON | arm 3, cast OFF |
|---|---|---|
| forward cast `vec4_convert_to_fp32` | 1.648 read + 3.268 write | **not launched** |
| loss forward `sparse_ce_forward_…_save_stats` | 3.298 + 0.001 | 1.649 + 0.000 |
| loss backward `sparseCENormalize_from_stats` | 3.297 + 3.274 | 1.655 + 1.631 |
| convert back `vec8_convert_to_bf16` | 3.297 + 1.631 | **not launched** |
| **total** | **19.714** | **4.935** |

- **−74.97%** against the 75% predicted in §5.4.
- **14.78 GB** less per micro-step, **946 GB** less per optimizer step.
- **2 fewer kernel launches** per micro-step.
- In arm 3 the census contains **no logits-sized convert at all**.
- The writes are a few MB short of the tensor size (3.268 vs 3.297) because dirty lines were still in the
  H200's 50 MB L2 cache when each kernel ended.

### 16.6 Throughput

| | fp32 rep1 | fp32 rep2 | bf16 rep1 | bf16 rep2 |
|---|---|---|---|---|
| steady tok/s (trainer's summary, steps 5–19) | 41,792 | 41,760 | 42,095 | 41,955 |
| min tok/s, steps 1–19 | 41,429 | 41,641 | 41,825 | 41,865 |

- bf16 − fp32 = **+248.7 tok/s = +0.60%**. Every bf16 run was faster than every fp32 run. n=2 per arm,
  so this is a direction plus a point estimate, not a settled number.
- **Cross-check against the traffic.** The micro-step is 392.2 ms. If every one of the 14.78 GB saved
  became saved time at 4.0–4.8 TB/s, the speedup would be **0.79–0.94%**. Measured: **0.60%**
  (2.35 ms). Roughly 1 ms of the traffic saving does not appear as time.
- *Hypothesis, not measured:* the bf16 loss kernels do the same math on half the bytes, so they may no
  longer be fully bandwidth-bound. ncu did not collect durations, and it locks clocks while profiling.

### 16.6b Re-verification on a fresh box (2026-09-25)

**Setup:**
- Box: `rough-word-cracks-fin-02` (2× H200). **GPU 1 only**; GPU 0 stayed at 0 MiB.
- Fresh clone of `dist_side_merge` `03b94cf` (the same tip), submodule `8808f89`.
- Our 7 files, md5-checked against `sft_gautam` `1ed2ac3`.
- The same three harness edits and the same script logic (`profiling_reports/ab_2026_09_25/ab_0925.sh`).
- Same run shape: A-B-A-B, 20 steps, B=4.
- Logs are local in `profiling_reports/ab_2026_09_25/box_logs/`.

| | fp32 rep1 | bf16 rep1 | fp32 rep2 | bf16 rep2 | vs 2026-09-24 |
|---|---|---|---|---|---|
| live peak (MiB) | 95,038 | 90,322 | 95,038 | 90,322 | **identical, to the MiB** |
| reserved peak (MiB) | 104,904 | 99,028 | 104,904 | 99,028 | **identical** |
| steady tok/s (steps 5–19) | 42,391 | 42,538 | 42,297 | 42,508 | all ~1.3% higher (different box) |
| step-0 grad norm | 18.9714 | 18.9712 | 18.9715 | 18.9714 | see the correction below |
| val @19 | 8.198 | 8.104 | 8.100 | 8.293 | bf16 is **not** lower this time |

- **Memory reproduces exactly:** −4,716 MiB live, −5,876 MiB reserved. Same as 09-24.
- **Throughput:** fp32 mean 42,344, bf16 mean 42,523, so **+179 tok/s = +0.42%** (09-24: +249 = +0.60%).
  - Every bf16 run beat every fp32 run again.
  - **Poller confound:** a `watch -n 0.1 nvidia-smi` started at 06:24:26, during run 1's last ~3 min, and ran
    through runs 2–4.
  - So **pair 2** (runs 3+4, both fully under the poller) is the clean pair: **+211 tok/s (+0.50%)**.
    Pair 1 is mixed: +147.
  - Conclusion: +0.4–0.6% across both days, n=4 pairs. The direction is solid; the size is only known roughly.
- **Loss:** mean |Δ| over 20 steps: fp32 vs fp32 0.401, bf16 vs bf16 0.333, fp32 vs bf16 0.013–0.635.
  - That is inside the same-arm noise, as on 09-24.
  - One pair (fp32 rep2 vs bf16 rep1) tracked to **0.013** mean |Δ**|**: two different arms nearly identical,
    which is more evidence the arms are numerically equivalent.
  - The 09-24 "bf16 val lower in both reps" did not repeat (8.104 / 8.293 vs 8.198 / 8.100). It was noise.
- **Correction to §18:** "step 0 is identical within an arm" is **false for the grad norm**. The two fp32 runs
  gave 18.9714 and 18.9715. The loss at step 0 is still identical in all four runs.
  - **Consequence for #2 (§19.4):** the planned check "step-0 grad norm identical before vs after" **cannot
    prove bit-identity**. Use a direct test instead: compare the buffer the old and new code produce, on the
    same input.

### 16.7 Earlier runs today (context; superseded where noted)

| run | result | status |
|---|---|---|
| arm 0: their code + bf16, B=8 | `misaligned address` at step 0; sanitizer: `sparse_ce_forward_kernel_vec<bfloat16_t>`, 50,253 errors (§15.3) | **stands** |
| arm 1: their code + fp32, B=8 | OOM at exactly `B*T*V*4` bytes (§15.4) | **stands** (co-tenant confound noted) |
| arm 1b: their code + fp32, B=4 | 5 steps, 41,685 tok/s, poller peak 104,820 | memory number superseded by §16.3 |
| arms 2/3, first attempt | Ananya's `snippet_runner` took up to 43,218 MiB; arm 2 OOMed with 149 MiB free | **void** (contention) |
| arm 2, second attempt | auto-resumed from arm 3's step-20 checkpoint → 0 steps, "Training complete" | **void** (the checkpoint trap, §20.9) |
| `arm2_final` / `clean_arm3` | 20 steps each; poller peaks 104,240 / 100,116 | superseded by §16 |

### 16.8 Corrections made during the day, in one place

1. "B=8 fits one H200" → it OOMs; ZeRO-1 shards nothing at world_size=1 (§15.5).
2. The B=4 peak prediction of 86,103 MiB was 18% low (§15.6).
3. "Arm 0 confirmed" was first claimed from a sticky-error stack trace → only the sanitizer proves the kernel (§15.3).
4. The port was first shipped as 6 files → 7 were needed; `git status` hides committed files (§20.3).
5. Arm 3 was said to be "a numerical change, 7 vs 23 bits" → wrong, the cast carries no information (§11).
6. The memory saving went 4,704 → 4,124 → **4,716** (allocator; the three-tensor model was right) (§16.3).
7. Throughput went +0.87% (mismatched windows) → +0.50% (n=1) → **+0.60%** (n=2, matched) (§16.6).
8. The step-3 split was "contention" → it is the **trainer's own non-determinism** (§18).
9. "Arm 2 must be bit-identical to arm 1" → wrong expectation; step 0 identical, the rest within noise (§11).

### 16.9 Checks that were broken, and caught

These all printed a plausible "pass" and would have gone unnoticed.

| check | why it could not fail | fixed by |
|---|---|---|
| the first `scp` of the tarball was pasted into the box shell | nothing was extracted; `make` said "up to date"; build `[SUCCESS]` | the five greps in the swap gate all failed loudly |
| `strings libtensor.so \| grep "fp32 accumulator"` | source comments never reach a binary | checking the mangled 4-parameter signature with `nm` |
| `nm \| grep LossVec` | a fully inlined template leaves no symbol | same |
| `grep -c "^[^ ]*error:"` | the anchor missed `collect2: error:` — the build had failed | unanchored `error` / `undefined reference` / `Error 1` |
| arm 2 "completed" | it resumed from a checkpoint and ran 0 steps | a gate that asserts 20 `^step` lines and 0 resume lines |
| peak memory parsed as 2,758 MiB | a CSV format change shifted the columns | parse the last field; then the allocator's own stats |
| the summary table on the box | mawk aborts on a missing input file; `2>/dev/null` hid it | pass only files that exist, never hide the error |
| the ncu `--launch-count` | validation runs first and uses the whole budget | the `BLU_VAL_STEPS` override |

---

## 17. Bug D — `float_to_bfloat16` turns every value ≥ 65,536 into Infinity

> **⚠️ OPEN — HIGH PRIORITY.** A silent wrong-answer bug in a core header that every kernel includes. It is not
> fixed. It will be looked into on **2026-09-25**, together with #2. The plan is in §17.1.

**Where:** `include/dtype/Types.h:85`, inside `detail::float_to_bfloat16`. This is the function behind the
`bfloat16_t(float)` constructor (`Types.h:199`), and therefore behind every `static_cast<T>(float)` with
`T = bfloat16_t`.

**It runs on the GPU too, not only the CPU.** It is declared `__device__ __host__` (`Types.h:66`), so any
CUDA kernel that builds a `bfloat16_t` from a float goes through it. The bulk convert kernels are **not**
affected: `convert_type_cuda_generic`, which `as_type` / `to_dtype` use, calls the CUDA intrinsics
`__float2bfloat16` / `__float22bfloat162_rn` (for example `ConversionKernelsSM103.cu:47`, `:72`).

```cpp
// Overflow, clamp to Infinity
if (exponent > 0x8E) {  // exponent > 142 decimal (127+15)      <-- the fp16 limit
    return static_cast<uint16_t>((sign >> 16) | 0x7F80u);       <-- +/-Inf
}
```

`127+15` is fp16's exponent limit; the fp16 converter at `Types.h:164` uses the same arithmetic. bf16 has
fp32's 8-bit exponent, so it is finite up to ~3.4e38. **Found and proven 2026-09-24** by a host test
(`profiling_reports/bugtest_2026_09_24/scripts/bf16_round_test.cu`). The test calls the real header
function against a correct round-to-nearest-even reference.

| input | library | correct |
|---|---|---|
| 65,504 / 65,535 | 65,536 | 65,536 ✓ |
| **65,536** | **inf** | 65,536 |
| **70,000** | **inf** | 70,144 |
| **1e10** | **inf** | 9.999e9 |
| **3e38** | **inf** | 3.004e38 |
| **−1e6** | **−inf** | −999,424 |
| 1.0, 3.14159, 1e-6, halfway 1+2⁻⁸, subnormal 1e-40 | correct | ✓ |

- **6 of 16 values mismatch.** Everything below 65,536 rounds correctly (RNE), including halfway cases
  and subnormals.
- **Impact on this work: none.** Loss gradients are at most ~1e-6, and no run produced Inf or NaN.
- **Impact elsewhere: unmeasured.** Any templated kernel that stores a float ≥ 65,536 into a `bfloat16_t`
  output silently produces Infinity, which then becomes NaN. A grep cannot count the call sites, because
  they are written as `static_cast<T>`.
- **Fix sketch:** drop the clamp. The RNE add already overflows to Inf only above bf16's real maximum.
  Keep the NaN branch.
- **Not fixed.** It is a core dtype header that every kernel uses, so it needs its own full-repo test run.

### 17.1 Plan (2026-09-25)

1. **Where it can hit — read the code first.** Candidate sites in `Tensor-Implementations` (`1ed2ac3`):
   - **33** `static_cast<T>(` in **8** `.cu`/`.cuh` files;
   - **17** explicit `bfloat16_t(...)` constructions;
   - the CPU ops that store bf16.

   These counts are **upper bounds, not proof**. A site matters only when it is built with `T = bfloat16_t`
   (check with `nm`), it is actually called, **and** its value can reach 65,536. The most likely are
   reductions with a bf16 output (sums, norms, loss totals), because a sum over many elements grows past
   65,536 easily.
2. **Fix:** delete the clamp (`Types.h:84-87`). The rounding add already overflows to Inf only above bf16's
   real maximum (~3.39e38). Keep the NaN/Inf branch at the top.
3. **Exhaustive test:** check **every one of the 2³² fp32 bit patterns** against a correct round-to-nearest-even
   reference, for example `__float2bfloat16_rn` on the GPU. That is ~4.3 billion values, seconds on a GPU,
   and much stronger than the 16-value test. NaN bit patterns may differ between the two (the library returns
   `0x7FC1`), so compare "is NaN" for NaNs, and bits for everything else.
4. **Clean rebuild, then a full test run.** `make` does not track header changes, so a partial build would
   keep the old function in stale objects. Then run the whole test suite, because every kernel includes this
   header.
5. **Leave the fp16 converter (`Types.h:164`) alone.** It uses the same `127+15` arithmetic, and for fp16 that
   limit is correct (fp16's maximum is 65,504). Confirm it has its own correct test.

---

### 17.2 Update 2026-09-26: fix proven exhaustively, blast radius

**The fix proven over all 2³² fp32 inputs**, CPU only, 6.5 s (`profiling_reports/bugd_2026_09_26/bugd_exhaustive.cpp`):

| check | result |
|---|---|
| fixed function (clamp at `Types.h:84-87` deleted) vs an independent exact RNE reference, all non-NaN inputs | **0 mismatches** |
| NaN inputs (16,777,214) | all stay NaN |
| old function vs the reference | **1,878,982,656 wrong (44% of all fp32 bit patterns)**, every one a finite \|x\| ≥ 65,536 |

- The prediction said 1,879,048,192. The 65,536 difference is the inputs within half a ulp of bf16's max, which
  *should* round to Inf. The old clamp gets those right by accident.
- For \|x\| < 65,536 the old and new code are the same code path, so they are bit-identical.
- A real overflow still becomes Inf after the fix, through the rounding carry.

**Where it runs.** The function is `__device__ __host__`, behind `bfloat16_t(float)` and `operator=(float)`.
- **Safe paths:** the CUDA intrinsics (`__float2bfloat16`, `__float22bfloat162_rn`): `convert_type_*`, the attention
  kernels (`gqa_*`), `multi_tensor_cast_f32_to_bf16`, BLAS epilogues. From an Aug-2026 nsys kernel list of the
  trainer.
- **Affected:** templated kernels on the library's own `bfloat16_t`, for example:
  - `scalar_mul_inplace<bfloat16_t>`: `ScalarOps.cu:89` does `static_cast<T>(s)`, so a scalar ≥ 65,536 becomes Inf;
  - `add_inplace_kernel<bfloat16_t>` (bfloat16_t operator arithmetic);
  - the CPU ops, and fills.
- **Concrete trigger: loss scaling.** Its standard initial scale is 65,536 = 2¹⁶ (`test_zero.cpp` uses
  `LossScaler(65536.0f)`). Multiplying a bf16 tensor by it gives Inf immediately.
- **In normal default training it is latent.** Activations and gradients are far below 65,536, and the loss is fp32
  now. It fires on outliers or divergence (then F1 stops the run, but the root cause is hidden), and on loss scaling.
- **Harmless side effect:** `get_lowest_value<bfloat16_t>()` / `get_max_value<bfloat16_t>()` (`ReductionOps.h`)
  return ±Inf instead of ±3.39e38. That is still a valid identity for max/min.

**Related, smaller: the fp16 converter** (`Types.h`, `float_to_float16`).
- It rounds half-up (`mant + 0x1000`), not RNE.
- It truncates subnormals.
- Its overflow limit is correct for fp16.
- It is not used by the trainer (bf16).

**Plan:**
1. Delete the 4-line clamp (comment why).
2. **Clean rebuild**: `Types.h` is included everywhere and `make` does not track headers.
3. Re-run the CUDA suites, plus a device test through `bfloat16_t(float)` with values ≥ 65,536.
4. A short trainer run: step-0 loss must be bit-identical (no training value reaches 65,536).

Target: library `dist_side_merge`.


### 17.3 Applied, tested and pushed (2026-09-26)

**Pushed:** library `dist_side_merge` `ad10e7d..7810aea`; BluTrain `e3633e8` (submodule bump). The first bump,
`7bd97a7`, was committed as `root@…` because that box's BluTrain had no identity set. It was amended to the
company identity and replaced with `--force-with-lease`.

**Files (4):** `include/dtype/Types.h`, `include/ops/helpers/Vectorized.h`, `src/UnaryOps/cpu/Activations.cpp`,
`include/ops/helpers/ReductionImpl.h`. Every change carries a `Modified:` comment with the reason and the
measurement.

| conversion | GPU (device) now | CPU (host) now | was |
|---|---|---|---|
| fp32 → bf16 | `__float2bfloat16_rn` (F2FP.BF16 on sm_80+) | bias trick `u += 0x7FFF + ((u>>16)&1)`, NaN → 0x7FFF | **Bug D**: every \|x\| ≥ 65,536 → Inf (1,878,982,656 of 2³²) |
| fp32 → fp16 | `__float2half_rn` | F16C `_cvtss_sh` when `__F16C__` (the library is built with `-mf16c`), else manual RNE | round-half-up + truncated subnormals: 100,815,872 of 2³² wrong |
| bf16 → fp32 | one shift (SHF.L.U32) on every arch | one shift | 3 branches that all returned the shift (same result) |
| double → bf16 / fp16 | `__double2bfloat16` / `__double2half` (one rounding) | round-to-odd into float, then one RNE (Boldo & Melquiond 2008) | double → float → bf16: two roundings |
| int → bf16 / fp16 | `__int2bfloat16_rn`, `__ll2bfloat16_rn`, … | int32 via double + round-to-odd; int64 fast path + `int64_to_float_rto` | through float: two roundings; int32 → bf16 also hit Bug D |
| CPU vector bf16 store (`Vectorized.h`) | — | rounds (RNE) | truncated (2.14 B of 2³² wrong) |
| CPU GELU bf16 tails (`Activations.cpp`, 3 sites) | — | `detail::float_to_bfloat16` | truncated |
| CPU mean (`ReductionImpl.h`, 7 sites) | — | one rounding (`static_cast<T>(double mean)`) | double → float → bf16 |

The GPU intrinsics are chosen per arch at compile time (`NV_IF_ELSE_TARGET` in `cuda_bf16.hpp` /
`cuda_fp16.hpp`). They cannot fail at runtime, so the manual bit code is CPU-only (`#if defined(__CUDA_ARCH__)`).
There is no C++ intrinsic for fp16 ↔ bf16. Going through fp32 is exact, and on the H200 it matched the direct
`cvt.rn.bf16.f16` (0 differences).

**Verification:**
- CPU, exhaustive against 7 references: 0 wrong over more than 13.4 B inputs, in both the F16C and the manual build.
  The references: PyTorch, TF/Eigen, NumPy, IEEE RNE, CUDA's host code, and 2 exact oracles.
- GPU test on the H200: 16/16 rows, 0 wrong. End-to-end test: 5/5.
- SASS on 7 archs: 0 `0x8e` clamps. `ScalarOps` instructions −48%.

**Speed** (the old and new `.so` in one process, ABBA order, 12 rounds, medians, 3% controls):
- GPU micro-benchmark: 1.03–1.76× faster on memory-bound kernels, 2–33× on conversion-heavy ones, 0 slower.
- Library benchmark: GPU ops 1.21–1.93× faster; CPU `as_type` and ops 1.46–3.8× faster. No op was slower in
  every round. The controls were within 3%.
- Precision: old 68,980,229 wrong, new 0.
- Two caveats: CPU double → bf16 `as_type` is 17% slower on the median (not in every round), and the
  multi-thread CPU reduce/GELU timings were too noisy to call.

**Artefacts:** `profiling_reports/bugd_2026_09_26/final/`: `patch_conversions.py`, `test_types_conversions_gpu.cu`,
`test_conversions_e2e.cu`, `bench/` (11 files), `review/` (the 4 patched files + `conversions.diff`), and
`box_results/`.

**Found while doing this, fixed later the same day:**
- The GPU `as_type` float box → §24.
- The GPU `reduce_mean` crash → §23.

**Left alone** (user decision): `fp4.h` double → fp4 (rounds twice) and `long double` → bf16 (CPU only, unused).

## 18. The trainer is not reproducible run to run

Four fp32 runs of this config today gave these **step-3 losses**:

| run | step-3 loss |
|---|---|
| their code, morning | 13.3687 |
| ours, `arm2_final` | 13.3632 |
| ours, rep1 | **14.9833** |
| ours, rep2 | **11.1716** |

All four used the same data and seed; the last two were the same binary on a clean GPU.

- **Step 0 is identical within an arm**: loss, val and gradient norm, to the printed precision.
- **The runs drift from step 1** (fp32 rep1 vs rep2: 0.031 at step 1).
- The 2-step warmup that `BLU_MAX_STEPS=20` produces makes early training chaotic (gradient norm ~1,030
  at step 1), and that amplifies the drift to O(1) by step 3.
- The close 13.3687 / 13.3632 match earlier in the day was luck.

**Consequence for everyone A/B-ing this trainer:** a single run per arm cannot compare early losses.
Use ≥2 repetitions and compare against the same-arm spread.

**Cause: not investigated.** Unverified candidates: atomics in backward reductions (for example the
embedding scatter-add), or GEMM algorithms chosen by timing-based autotuning.

**Update 2026-09-29** (the F2 ABAB, §27.4: B = 4, 20 steps, 2 runs per arm, one binary):
- Step 0 is identical in all 4 runs (11.298004). From step 1 they drift.
- By steps 3–5, two OLD runs of the same binary differ by up to **7.9** (step 5: 20.1567 vs 12.2290).
- Mean |Δloss| over the 20 steps: OLD vs OLD **1.14**, NEW vs NEW 0.44. Every OLD-vs-NEW pair falls inside that
  (0.06–1.09), and one OLD-NEW pair (0.06) is closer than either same-arm pair.
- Last step: OLD 8.30 / 7.97, NEW 7.99 / 8.01.

With noise this large, comparing losses between runs needs many repetitions or a deterministic mode. The cause is
still not investigated (§26).

---

## 19. Immediate future tasks: three casting follow-ups + one pending measurement (2026-09-24 — nothing changed yet)

**When:** **#2 on 2026-09-25**, because it is a small fix with a precedent in the same file. #1, #3 and #4
right after the SFT work, as training-script optimisations.
#1–#3 were found while analysing the casts (§19.3–§19.5). #4 is a measurement left open by §16.6 (§19.6).

> **Correction** to the first version of this section, which said the weight-gradient round trip and the
> fp32-only optimizer "pull in opposite directions". **That was wrong.** They apply to *different* weights
> (§19.1), so they don't conflict.

### 19.1 Which weights are bf16 and which are fp32 — proven from the parameter counts

| group | stored as | contents | count | matches log |
|---|---|---|---|---|
| **G0** | **bf16** | the 6 Linear matrices × 26 layers (`wQ wK wV wO gate_up down`, created `Dtype::Bfloat16`; the attention four at `bluscript_zero_ckpt.cpp:540-543`) | 26 × 56,623,104 = **1,472,200,704** | ✅ exact |
| **G1** | **fp32** | `wte` 50304×2304, plus every norm scale: 26 × (2×2304 block RMSNorm + 2×128 QK-norm) + 2304 final | 115,900,416 + 128,768 = **116,029,184** | ✅ exact |

The log's "non-decayed params: 128,768" is exactly the norm scales.

Activation dtypes:

| activation | dtype |
|---|---|
| residual stream | fp32 (`bf16_resid=false`) |
| input to each Linear | a bf16 copy of the residual (`:580`) |
| QK-norm + RoPE | fp32 (`qk_dtype()`) |
| attention | bf16 |
| logits | set by `BLU_BF16_LOGITS` |

### 19.2 The optimizer's master copy (the "notebook")

> `ShardedOptimizer.cpp` line numbers in this section are for `dist_side_merge` (`03b94cf`). The local
> BluTrain `master` has the same code 25 lines earlier (`:281`, `:285-288`, `:755`, `:948`).

Every group has an **fp32 master copy inside the optimizer**, whatever dtype the model's weight is.
`dist/zero/src/ShardedOptimizer.cpp:306` builds its options with `Dtype::Float32`, and `:310-313`
allocates `shard_grad_fp32_`, `shard_master_`, `shard_m_` and `shard_v_` with them.

Each step, the optimizer updates the master and then writes it back into the model's weight
(`cast_master_range`, `:780`).

**This master is what protects the tiny updates.** A weight of 0.02 with an Adam update of 1.05e-6: bf16's step
at 0.02 is 2^-13 = 1.22e-4, so any update below half of that (6.1e-5) rounds away. After 1,000 steps an fp32 master reaches 0.018950, while a bf16 master
stays frozen at 0.020020. The model weight's own dtype is only the compute copy.

The rule: **round freely where a number is used once; never where numbers are summed over time.**

### 19.3 #1 — `wte` is fp32, but the lm_head uses a bf16 copy

`wte` is used twice under weight tying:

```cpp
:746  x = to_dtype(wte.forward(idx), cfg.resid_dtype());   // INPUT:  the gather reads wte as stored (fp32)
:784  w_bf   = to_dtype(wte.weight, Dtype::Bfloat16);     // OUTPUT: a bf16 copy
:785  w_t_bf = contiguous(transpose(w_bf, 0, 1));         //         [c,v]
:786  logits = matmul(x_bf, w_t_bf);                      //         the lm_head uses the bf16 copy
```

Weight tying doesn't change whether the logits `[B,T,V]` exist; it only changes where the `[C,V]` weight
comes from.

**The round trip, in backward.** The lm_head half of `wte`'s gradient goes:
1. the GEMM sums it in fp32;
2. it is stored as bf16 (`MatrixBackward.cpp:73` takes the dtype of the bf16 copy);
3. `ToDtypeBackward` widens it back to fp32.

That last convert buys nothing: the value is bf16-rounded either way.

What an fp32 `wte` does and does not buy:

| | does fp32 `wte` help? |
|---|---|
| weight updates | **no.** The fp32 master (§19.2) protects them, so `wte` exists twice in fp32 |
| lm_head (output) | **no.** It is cast to bf16 before use |
| input embedding | **yes, real bits.** Its values come from the fp32 master; unlike the logits' fp32 copy, these are not zero-padding |
| gradient sum into `wte` | **yes — checked 2026-09-25, see §19.3.1.** With a bf16 `wte`, the embedding backward sums every occurrence of a token with bf16 `atomicAdd`. |

**The switch already exists.** The exact lines, in `bluscript_zero_ckpt.cpp` (dist_side_merge `03b94cf`):

```cpp
:189  bool bf16_embed  = false;   // wte (under weight tying, also the lm_head weight)
:193  Dtype embed_dtype() const { return bf16_embed  ? Dtype::Bfloat16 : Dtype::Float32; }
:699  nn::Embedding wte;            // token embedding [vocab, d_model]
:709  wte(static_cast<int>(c.vocab_size), static_cast<int>(c.d_model),
:710      /*padding_idx=*/-1, c.embed_dtype()),                                 // embed dtype
:715  init_embedding(wte, 0.02f, init_rng_);   // :522-524: copy_into_param(emb.weight, rng.normal(shape, std))
:874  cfg.bf16_embed  = getenv_int("BLU_BF16_EMBED",  all) != 0;
```

**Two ways to remove #1:**

| | `BLU_BF16_EMBED=1` | GEMM writes the `wte` gradient in fp32 (bf16 in, fp32 out) |
|---|---|---|
| code change | none | a BLAS epilogue + `MatrixBackward.cpp:73` |
| round trip | gone | gone |
| input-embedding precision | lower (bf16) | kept |
| gradient precision | depends on how the embedding backward sums | better than today |
| memory | `wte` 464 → 232 MB | unchanged |

- **Not checked:** whether BluBLAS has an fp32-output epilogue. cuBLAS supports bf16 inputs with fp32 output.
- **A bf16-in / fp32-out GEMM does not remove every cast.** It removes output-side widens (the `wte`
  gradient; Linear outputs into the fp32 residual at `:645` and `:684`). It does not remove input-side
  narrowing (fp32 residual → bf16 before each Linear, `:580`), nor the QK-norm / RoPE casts
  (`:618-633`), which are not GEMMs.

**Why this needs a longer test than the logits did.** For the logits, the fp32 copy was zero-padding and the
two paths were proven identical (§16.2), so 20 steps were enough. For the embedding it is a **real precision
change**. A 20-step run with warmup 2 has a noise floor of ~0.4 loss (§18) and cannot show equivalence.

Plan:
1. **Code check, no GPU:** how the embedding backward sums for a bf16 weight. If it sums in bf16, stop.
2. **Short A-B-A-B:** `BLU_BF16_EMBED` 0 vs 1, with `BLU_BF16_LOGITS=1` on both, 2 reps. Checks memory,
   tok/s, and that it runs cleanly.
3. **Long run:** a few hundred steps with proper warmup, ≥2 reps per arm, val loss compared at matched steps.

### 19.3.1 Code check (2026-09-25): what the fp32 `wte` really protects

Question: `wte` is fp32, but the lm_head only ever sees a bf16 copy of it. So is the fp32 `wte` needed at all?
Read on `sft_gautam` `1ed2ac3`. `wte`'s gradient is summed in three places:

| sum | code | fp32 `wte` (today) | bf16 `wte` (`BLU_BF16_EMBED=1`) |
|---|---|---|---|
| **(a) every occurrence of a token in one micro-step** (embedding backward) | `EmbeddingBackward.cpp:33-36`: `grad_weight` takes the dtype of `grad_output`, which is the embedding output's dtype = `wte`'s dtype (`EmbeddingOps.cpp:39-42`). Then `EmbeddingKernels.cu:415-426`: fp32 goes to `embedding_backward_kernel_cooperative` (fp32 shared-memory sums, no atomics); anything else goes to `embedding_backward_kernel_optimized<T>`, which does `atomicAdd` **in T** (`:147-148`) | fp32 sums, deterministic | **bf16 `atomicAdd`: every add rounds, in random order, so it is also non-deterministic** |
| **(b) input half + lm_head half** | `Engine.cpp:321`, `:342-343`: `sum = grads[0]; sum += grads[i]`, in the first grad's dtype | fp32 | bf16, **one** rounding |
| **(c) over the 64 micro-steps** | `AutogradMeta.cpp:112-116`, `:158`, `:172-173`: a bf16/fp16 update is upcast and accumulated in an **fp32** buffer ("Megatron main_grad style") | fp32 | fp32 — safe |

**How big is (a)?** Bit-exact CPU emulation of one `wte` row (C=2304) receiving K bf16 contributions, summed
the way each kernel does. Source: `profiling_reports/wte_analysis_2026_09_25/emb_bf16_sum.cpp` (gitignored).

| K (times the token appears in the micro-step) | bf16 atomic sum, rel. error | fp32 sum, rel. error | one bf16 rounding of the exact sum |
|---|---|---|---|
| 10 | 4.1e-3 | 0 | 1.7e-3 |
| 100 | 1.2e-2 | 4e-9 | 1.6e-3 |
| 1,000 | **3.7e-2** (3.2e-2 with a 0.2σ mean) | 2.5e-8 | 1.6e-3 |
| 4,000 | **7.5e-2** (**1.96e-1** with a 0.2σ mean) | 5.9e-8 | 1.7e-3 |

- The error grows with K: the running sum gets big, and each new small contribution is rounded away.
  At K = 1,000 the gradient of that word's row is **~4% wrong**, **23×** worse than rounding once.
- **K is large for exactly the most frequent tokens.** At B=8, one micro-step is 32,768 positions. A token at 3%
  frequency (",", ".", " the") gets ~1,000 contributions. *That frequency is an estimate: no token histogram of the
  training data was taken.*
- The input side also changes. The lookup (`:746`) reads `wte` as stored, straight into the fp32 residual, so a
  bf16 `wte` rounds the starting values (~0.2% relative). Smaller than (a), but it is real.

**Verdict:**
- **The user is right about the lm_head side.** fp32 master → fp32 `wte` → bf16 copy gives the matmul exactly the
  same bits as fp32 master → bf16 `wte`: one rounding either way.
- **But the fp32 `wte` is not useless.** It is the reason the embedding backward runs the fp32 cooperative kernel.
  **`BLU_BF16_EMBED=1` as it stands would put a several-% error, and non-determinism, on the gradients of the most
  frequent tokens.** Do not flip it.

**What #1 would even save (estimate, not measured):**
- The forward `to_dtype(wte)` reads 464 MB and writes 232 MB. The backward widen reads 232 MB and writes 464 MB.
- That is ≈1.39 GB per micro-step ≈ 0.32 ms at 4.4 TB/s ≈ **0.08%** of a 392 ms micro-step, plus 232 MB of weight
  memory.
- **#1 is low value. Keep `wte` fp32.**
- If it is ever done, the safe form is: bf16 `wte`, **plus** an embedding backward that sums in fp32 (fp32
  `grad_weight` and a cooperative kernel templated on the input type). That still leaves the ~0.2% input rounding,
  so it needs the long-run test.

### 19.3.2 ⚠️ Parked for a deeper re-analysis — what §19.3.1 assumed, and what looks suspicious

The user's call (2026-09-25): the §19.3.1 verdict is provisional. Something about it looks suspicious. We may be
looking at the wrong cause, or covering an earlier mistake of our own, so this is to be analysed again, deeper,
later. Until then, §19.3.1 is **"code-read + emulation"**, not a measured result.

**Assumptions §19.3.1 rests on (none measured):**
1. **The gradient values.** The emulation draws each position's gradient from a normal distribution (σ = 1e-4,
   mean 0 or 0.2σ). The real layer-0 gradient distribution was never looked at, so the % errors are only as
   good as that model.
2. **Token frequency.** "~1,000 contributions for a 3% token" is an estimate. No token histogram of the
   training data was taken.
3. **Hardware rounding.** The emulation assumes every bf16 `atomicAdd` rounds to nearest-even. Not checked on
   sm90. The real kernel was not run: only an emulation.
4. **Which kernel runs.** The claim that fp32 → cooperative kernel depends on `BLUTRAIN_EMB_BWD_COOPERATIVE`
   not being `0` in the environment, and on `grad_weight` being contiguous. Not checked in a real run
   (nsys kernel names would show it).
5. **Has bf16 `wte` ever run in training?** The trainer's default is `getenv_int("BLU_BF16_EMBED", all)`
   (`:874`). What `all` is, and whether any past or current run set it, has not been checked. If one did,
   its embedding gradients went through bf16 atomics.
6. **The lm_head half's path** (`to_dtype` → `transpose` → `contiguous` → matmul, and back) was taken from
   the code, not traced in a real backward.

**Suspicious things found while reading (not analysed yet):**
- **The comment lies.** `EmbeddingKernels.cu:123` says the atomicAdd kernel is "kept for strided (non-contiguous)
  weight tensors only". But `:415-426` sends **every non-fp32 dtype** there too (bf16, fp16).
- **Silent garbage for an out-of-range token on the GPU.** The embedding output is allocated uninitialised
  (`EmbeddingOps.cpp:42`, `Tensor output(output_shape, opts)`). The GPU kernels just `return` for
  `tok >= V` (`EmbeddingKernels.cu:48-50`, `:107`), leaving that row as whatever was in memory. The CPU path
  throws instead (`EmbeddingOps.cpp:58-61`). **SFT relevance:** 65,535 is our `ignore_index` marker for
  *targets*. If an SFT pipeline ever put 65,535 into the *input* ids, the embedding would silently emit
  garbage, not fail.
- **The cooperative kernel may be slow.** Its grid is `((C + 127) / 128, 1)` (`:420`) = **18 blocks** for
  C = 2304, and each block loops over all N positions serially. 18 blocks on a 132-SM H200 would leave most
  of the GPU idle. Performance not measured.
- **Non-determinism link (§18).** The bf16 path uses atomics and is order-dependent. With fp32 `wte` the
  cooperative kernel should be deterministic, so it should *not* be §18's cause. That is unverified too.

**What the re-analysis should do:**
- run the real kernel (both dtypes) on real layer-0 gradients;
- take a token histogram of the training data;
- check which embedding-backward kernel runs in nsys;
- check `all` / `BLU_BF16_EMBED` in the trainer and past run logs;
- time the cooperative kernel;
- check the out-of-range behaviour against the SFT data path.

### 19.4 #2 — the optimizer converts the gradient shard through a temporary

`dist/zero/src/ShardedOptimizer.cpp:973-980`:

```cpp
void ShardedOptimizer::cast_shard_grad_to_fp32() {
    if (compute_dtype_ == Dtype::Float32) {
        d2d(shard_grad_fp32_.data(), shard_grad_.data(), S_ * sizeof(float), device_);
    } else {
        Tensor f = shard_grad_.as_type(Dtype::Float32);                       // (1) a new fp32 temporary
        d2d(shard_grad_fp32_.data(), f.data(), S_ * sizeof(float), device_);  // (2) copied a second time
    }
}
```

- For the bf16 group (G0), `as_type` allocates a full fp32 temporary. `d2d` then copies it into
  `shard_grad_fp32_`, which is **already allocated** (`:310`).
- At world_size=1 that temporary is G0's 1,472,200,704 params × 4 bytes ≈ **5.9 GB** every optimizer step.
  At W=8 it is 1/8 of that per rank.
- The contended arm-2 run (§16.7) OOMed on a **5,890,899,968-byte** allocation in
  `convert_cuda_bf16_to_fp32_sm90`, right after step 0 (`arm2.log:63`). That is 5,888,802,816 + 2 MiB,
  which is consistent with this temporary. The backtrace was not decoded, so this is **not proven**.

**The fix has a precedent in the same file.** `cast_master_range` (`:780`, the master → weight direction)
carries this comment:

> *"Replaces the old `shard_master_.as_type()` + full-shard d2d, which allocated a whole extra S_-element
> temporary and copied it every step (114.2 MB / 0.36 ms in report5)"*

It converts straight into the destination with `convert_type_cuda_generic`. The same change applies here:
convert `shard_grad_` directly into `shard_grad_fp32_`. **Not done, not measured.** Scheduled for 2026-09-25.

The planned change (`else` branch only; the fp32 branch stays as it is):

```cpp
} else {
    OwnTensor::convert_type_cuda_generic(
        shard_grad_.data(), shard_grad_.dtype(),          // bf16 source (created with `oc`, :284 on master)
        shard_grad_fp32_.data(), Dtype::Float32,          // the fp32 buffer that already exists
        static_cast<int64_t>(S_), ocuda::getCurrentStream());
}
```

Before calling it done, check:

| check | why |
|---|---|
| **stream order** | The old `d2d` goes through `dev::copy_memory`. The new convert is async on `getCurrentStream()`. The loss-scaler unscale and `local_adam_step` that follow (`:1058+` on master) must run on the same stream, or be synchronised, otherwise Adam could read the buffer before it is filled. |
| **correctness** | bf16 → fp32 is exact, so `shard_grad_fp32_` must be **bit-identical** before and after. Compare it, not the loss. |
| **memory** | The allocator's `active_peak` / `reserved_peak` before and after. The 5.9 GB temporary lowers the peak **only if** the optimizer step is where the peak is. Not known. |
| **speed** | Expected to be tiny. Saved traffic ≈ 11.8 GB per optimizer step ≈ 2.7 ms at ~4.4 TB/s, once per ~25 s optimizer step (64 × 392 ms) ≈ **0.01%**. Estimated, not measured. |

**Code-verified 2026-09-25** (read on the local `origin/dist_side_merge` = `74ec046`, 2026-09-18; the box tip
`03b94cf` has the function at `:973`, so the patch is written against the function's text, not line numbers):

| fact | proof |
|---|---|
| the old path = one convert kernel + one copy, **both on `getCurrentStream()`** | `as_type` → `convert_type_cuda_generic(..., getCurrentStream())` (`Tensor-Implementations/src/core/AsTypeTensor.cpp:38-39`); `d2d` → `dev::copy_memory` → `cudaMemcpyAsync(..., getCurrentStream())` (`src/device/DeviceTransfer.cpp:52-53`) |
| everything after it is on the same stream | `multi_tensor_scale_cuda` (`MultiTensorKernels.cu:243`), `multi_tensor_adam_cuda` (`:326`), `multi_tensor_grad_norm_cuda` (`:187`) all use `getCurrentStream()` |
| source and destination dtype/size | `shard_grad_ = zeros(flat_shape(S_), oc)` with `oc` = `compute_dtype_`; `shard_grad_fp32_ = zeros(flat_shape(S_), of)` with `of` = fp32 (`ShardedOptimizer.cpp:288-293` @`74ec046`) |
| the include already exists | `#include "ops/helpers/ConversionKernels.cuh"` (`:20` @`74ec046`), which `cast_master_range` already uses |

- **So the new code launches exactly the kernel `as_type` launched, with the same arguments and stream.**
  Only the output pointer changes, from a fresh temporary to the persistent buffer.
- **Result:** bit-identical by construction, stream order unchanged, one 5.9 GB allocation and one 5.9 GB copy
  fewer per optimizer step (W=1).
- **One difference:** `as_type` also calls `set_cuda_device()`. `cast_master_range` (the precedent, same file)
  does not, and it relies on the rank's device already being current. So does the new code.
- **Patch:** `profiling_reports/fix2_2026_09_25/cast_shard_grad_to_fp32.new.cpp` (gitignored). It replaces the
  whole function.

**Validation plan (box):**
1. **Step-0 gradient norm identical** before vs after, to every printed digit. It is computed from
   `shard_grad_fp32_` right after this cast. Step 0 is identical within an arm (§18); step 1+ drifts
   anyway, so it cannot be used.
2. **`[MEM]` allocator peaks** before vs after, to see whether the temporary was at the peak.
3. **nsys:** the 5.9 GB device-to-device memcpy in the optimizer step is gone, and the convert kernel is
   still there, once.

**2026-09-25 — measured on the box, then independently cross-verified:**
- **Bit-identity test** (`Tests/kernel_tests/test_grad_cast_direct.cu`, box GPU 1): old vs new **0 mismatches** at
  n = 1,472,200,704 and at n = 1,000,003, including ±0, ±inf, NaN, subnormals and max. Both equal the exact answer
  (`bits << 16`). Allocator active peak: new path **+0 B**, old path **+5,888,802,816 B**. 0 predictions violated.
- **Machine code:**
  - the old runner's `cast_shard_grad_to_fp32` calls `Tensor::as_type` + `copy_memory`;
  - the new one tail-calls only `convert_type_cuda_generic`;
  - each runner carries its own copy of the function (`T` symbol, linked from `libdist.a`), so an old-vs-new binary
    A/B is real.
- **Two independent read-only verifiers** (on the exact box sources: BluTrain `03b94cf`, library `8808f89`) refuted
  nothing:
  - bit-identity holds: every bf16→fp32 path is an exact bit shift, and all buffers are 512-byte-aligned allocator
    blocks, so the same kernel runs;
  - stream order holds: `d2d` is `cudaMemcpyAsync` on the current stream, and all 7 readers of `shard_grad_fp32_`
    (Adam, unscale, norm, clip) run on the current stream; no host reads;
  - `SymmetricShardedOptimizer` already does the direct convert, and `ShardedMuon` has no such function.
- **Corrected claim: the memory saving is NOT proven.**
  - The temporary was a cached block, reused every step.
  - It is probably not live at the global peak: activations are freed by the optimizer step.
  - Reserved memory can move either way, through fragmentation.
  - Measure `[MEM]` before and after before quoting any saving. The certain gain is one S_×4 D2D copy (8·S_ bytes of
    traffic) and one allocator call per step.
- **The same anti-pattern at 3 more sites that run every step** (`as_type` into a temporary, then `d2d`):

  | site | runs | direction | direct fix |
  |---|---|---|---|
  | `:357` `init_master_from_params` | once, at construction | compute → fp32 | not worth it |
  | `:539` `on_grad_ready` | per param per step, W>1 overlap path (the production default) | fp32 grad → bf16 | convert into `flat_stage_ + offset` |
  | `:945` `gather_grads_into_flat` | per param per step, W=1 / reference path | fp32 grad → bf16 | same |
  | `:977` `cast_shard_grad_to_fp32` | per step | bf16 → fp32 | **done** (above) |
  | `:1022` `all_gather_params` | per step, non-staged (the production default) | fp32 master → bf16 | `cast_master_range(shard_begin_, shard_begin_ + S_)`: the same call, bit-identical |

  - `:945` really runs every step: `grad_view()` returns the fp32 main-grad (`AutogradMeta.cpp:158`, `:173`).
  - For `:539`, the hook's input is built in code not in the copies, and only a comment says it is fp32.
  - **Caveat for `:539` / `:945`:** the destination `flat_stage_ + offset` can be un-aligned, which picks the scalar
    kernel instead of the vec8 one. Both round to nearest-even (`__float2bfloat16` vs `__float22bfloat162_rn`).
    **NaN payloads are unverified**, so the test must cover misaligned destinations plus NaNs.
  - `SymmetricShardedOptimizer.cpp:603` has the same pattern, but only once at init.
- **Side findings (not #2, pre-existing, not fixed):**
  1. **Adam ignores `is_adamw` on H200/B300.** Non-Ada `multi_tensor_adam_cuda` never passes it
     (`MultiTensorKernels.cu:338-392`), and the kernel always does decoupled AdamW decay. It does not affect this
     trainer (AdamW); plain Adam with wd>0 silently becomes AdamW. Confirmed by reading `8808f89`.
  2. `d2d` ignores `device_.index`: the Device-only overload uses `cudaGetDevice()` (`DeviceTransfer.cpp:22-29`).
  3. The B300 fp32→bf16 grid-stride kernel uses `uint` counts (`ConversionKernelsSM103.cu:94-124`), which wraps at
     n ≥ 2³² elements.
  4. The `d_clip_coef_` host→device writes use blocking `cudaMemcpy` on the legacy stream (`:1069`, `:1172`,
     `:1215`). Ordered only if the compute stream is a blocking stream; not proven.

- **Where the change lives:** `dist/zero` is in the **BluTrain** repo (the dist team's code, branch
  `dist_side_merge`), not in `Tensor-Implementations` / `sft_gautam`. Decide where it is committed before
  pushing.

### 19.5 #3 — `data<T>()` never checks the dtype (Bug C's family)

**Bug C is fixed:** the grad scale is `const float*`. #3 is the reason Bug C was possible.

- `Tensor::data<T>()` is a raw `reinterpret_cast`, used at **827 call sites**. Any of them can read a tensor as
  the wrong type, and nothing reports it.
- Option 1, a PyTorch-style `check_type()`, would catch all of them at runtime (§8).
- **A known exposed site:** `fused_rms_rope_forward` (`src/UnaryOps/cpu/Normalizations.cpp:998`). It checks
  device and shape (`:1010-1026`), but not dtype, then reads `q`, `k`, `q_gamma` and `k_gamma` through
  `data<float>()` (`:1042-1046`). Only the trainer's `qk_dtype()` returning fp32 keeps a bf16 tensor out.

This is a research item, not an inefficiency.

### 19.6 #4 — the speed gain is smaller than the traffic saving predicts (measurement pending)

| | value | source |
|---|---|---|
| measured | **+0.60%** tok/s = +248.7 tok/s = **2.35 ms** per 392.2 ms micro-step, n=2 per arm | §16.6 |
| predicted from bytes alone | 14.78 GB less per micro-step at 4.0–4.8 TB/s = 3.1–3.7 ms = **0.79–0.94%** | §16.5 |
| unexplained | about **0.7–1.3 ms** per micro-step of the traffic saving does not show up as time | |

- **Hypothesis, not measured:** the bf16 loss kernels do the same math on half the bytes. So they may no
  longer be purely bandwidth-bound, and they save less time than their bytes suggest.
- **Why it is not known yet:** the ncu run (§16.5) collected DRAM bytes only, not durations. ncu also
  locks clocks while profiling, so its times would not be production times anyway.
- **Test plan:**
  1. One **unprofiled nsys** run per arm (`BLU_BF16_LOGITS` 0 and 1), the same B=4 config, ≥2 reps.
     Read the per-kernel duration of the four loss-path kernels: `vec4_convert_to_fp32`,
     `sparse_ce_forward_…_save_stats`, `sparseCENormalize_from_stats`, `vec8_convert_to_bf16`.
  2. Compare each kernel's duration with its bytes ÷ peak bandwidth. A kernel well above that line is not
     bandwidth-bound.
  3. If the bf16 loss kernels are not bandwidth-bound, run a small ncu section set (achieved bandwidth,
     SM throughput) on only those kernels to see what limits them.
  4. More tok/s repetitions, to tighten the +0.60% itself (n=2 today).

---

## 19.7 Optimizer + trainer audit (2026-09-25)

**Method:**
- 3 read-only auditors covered the ZeRO optimizers, the optimizer/convert kernels, and the trainer's wiring.
- Then 5 independent skeptics were told to *refute* each finding.
- All of it was read on the exact box sources (BluTrain `03b94cf`, library `8808f89`, the trainer at `03b94cf`).
- Raw verdicts: `profiling_reports/fix2_2026_09_25/audit_verdicts_F1-F9.json`.
- No GPU was used.

**Correction to this doc:** `GRAD_ACCUM = global_batch / (B·T·W)` (trainer `:926`). It is **64** in our 1-GPU test (B=4, W=1) and **4** in production (B=8, W=8). Earlier "per ~25 s optimizer step" figures are for the test config only.

| # | finding | verdict | production default? | severity | decision |
|---|---|---|---|---|---|
| F1 | a **NaN/Inf gradient is written into the fp32 master weights**. The skip is gated on a LossScaler (`ShardedOptimizer.cpp:1160`, `:1201`), and none is installed. A NaN norm gives clip coef 1.0 (`GradNormKernels.cu:190-193`); an Inf norm gives coef 0 and Inf·0 = NaN. The trainer checks only the loss, before `step()` (`:2270`). On a checkpoint step the NaN state is **saved**, and auto-resume reloads it. `BLU_GLOBAL_CLIP=1` does not prevent it | **CONFIRMED** | yes (needs one non-finite gradient) | **bug** | fix; see below |
| F2 | `accumulate_grad` does `*grad_ += update.as_type(fp32)` every micro-step for all 156 bf16 weights (`AutogradMeta.cpp:172-173`). That is an fp32 temp + add, where one mixed add would do. There is no fp32 += bf16 kernel at all (`TensorOps.cpp:203`) | CONFIRMED | yes, 4 micro-steps/step | waste, **~0.7% est.** at W=1 (11.8 GB/micro-step); production % unknown | **DONE (§27):** fused kernel, bit-identical, measured +0.52% / +0.60% tok/s at W=1 (64 micro-steps/step) |
| F3 | at W=1: 5 grad passes + 4 param passes per step where ~1 each would do; `shard_grad_` / `shard_master_compute_` duplicate `flat_stage_` (2.94 GB each) | CONFIRMED | only `:539`, `:977`, `:1022` run at W=8 (~1.8 ms/step est.) | waste | #2 family fix covers the production part |
| F4 | v1 rounds the fp32-accumulated grad to bf16 once for the reduce, then widens it back (`:186-194` refuses an fp32 wire). At W=1 there is no comm at all | PARTLY (one rounding; the sum itself stays fp32) | yes (bf16 NCCL avg at W=8) | design-choice | leave; the "bit-identical" header comment (`ShardedOptimizer.h:22-24`) is wrong for bf16 groups |
| F5 | clipping is **per dtype group** by default (`BLU_GLOBAL_CLIP=0`), not torch-style global. The logged norm is the combined pre-clip norm. The trainer's justification (`:2298-2329`) is written for Muon and does not hold for the default AdamW | CONFIRMED | yes | design-choice | leave; fix the comment |
| F6 | checkpoint + auto-resume ON by default, cwd-relative, no config check; a forced save at the last step means an identical re-run resumes at `max_steps` and runs 0 steps; the comment `:241-249` contradicts the code | PARTLY (it prints a resume line; not silent) | yes | risk | our scripts guard it (`BLU_CKPT=0` + `rm -rf`); tell colleagues |
| F7 | the DCP resume skips the layout-hash check; `set_config` silently keeps `step_count_=0` if the key is missing | PARTLY (sizes are checked; the key is always written) | effectively no | risk | leave |
| F8 | `getenv_int("X")` with X set but empty reads 0, not the default (`DataLoader.h:27-30`) | CONFIRMED | no | design-choice | leave (harmless) |
| F9 | library `nn::Adam` / `nn::AdamW` / `clip_grad_norm_`: a pinned-metadata race (>512 tensors) and a dangling `as_type` temp pointer | PARTLY | **no**: the trainer never uses them; ZeRO syncs after each launch | risk | leave |
| — | multi-tensor Adam ignores `is_adamw` (generic + sm89; sm90 variant is dead code) | CONFIRMED | **no**: the trainer hard-codes `AdamVariant::ADAMW` (`:1108`); nothing selects ADAM | — | **leave** (user's call; the colleague is right that it is a placeholder) |

**Reported by one auditor, not re-verified** (all off the production path, or cosmetic):
- Ada-only (sm_89):
  - the scale kernel skips factors >1 (`_sm89.cu:83`);
  - float4 accesses assume 16-B-aligned group offsets;
  - `fmaf` placement differs from the generic Adam, so the last bit is GPU-dependent.
- The generic convert kernel uses a 32-bit `blockIdx.x*blockDim.x` (wraps at n ≥ 2³²). It routes int64/double through float.
- The sm103 fp32→bf16 grid-stride kernel uses `uint` counts.
- Stale trainer comments:
  - `:880` "3.07 GiB" is the bf16 size;
  - `:2060` "GRAD_ACCUM=32";
  - `:1102` "103M embedding";
  - "1.62B bf16 group" (it is 1.472B);
  - `probe_layers {0,8,17,26,35}` with 26 layers.
- The CSV log is truncated on resume (`:1779-1780`).
- Weight decay applies to the tied wte (a design choice, matches nanoGPT).

**F1 fix options (decision pending):**
- (a) **skip** the step on a non-finite norm: drop `sc &&` at `:1160` / `:1201`, then set `last_step_skipped_` and return before `++step_count_`;
- (b) **stop before the update**: throw/abort on a non-finite norm, so the weights and the next checkpoint stay clean and the failure stays loud.

Either way the trainer should also not save a checkpoint on that step.

## 19.8 Fixes applied on the box (2026-09-25): F1 + the #2 family — and the tests

All edits are on the box (`rough-word-cracks-fin-02:/root/gautam_k/BluTrain`, tip `03b94cf`). Each carries a comment giving
what changed, what it was, and why. **Not committed anywhere yet.** Full diffs and logs, local copies:
`profiling_reports/fix2_2026_09_25/box_state/t12/`.

| file | change | kind |
|---|---|---|
| `dist/zero/src/ShardedOptimizer.cpp` | sites **B′** `on_grad_ready`, **B** `gather_grads_into_flat`, **C** `cast_shard_grad_to_fp32`, **D** `all_gather_params`: `as_type` temp + `d2d` → one `convert_type_cuda_generic` into the destination | fix (#2) |
| same | **F1**: both non-finite guards (`step()` per-group path + cached-norm path). With a LossScaler: unchanged. Without one: `throw ZeroNonFiniteGradError` **before** any update. The norm log shows `nan`, not 0. `nonfinite_grad_msg()` builds the error text | fix (F1) |
| `dist/zero/include/ZeroErrors.h` | new `ZeroNonFiniteGradError` | fix (F1) |
| `dist/zero/include/ShardedOptimizer.h` | declares `nonfinite_grad_msg` | fix (F1) |
| `bluscript_zero_ckpt.cpp` | `try { opt->step(); } catch (ZeroNonFiniteGradError) { [FATAL] line; MPI_Abort(4) }`, with the why + kernel backtrace in the comment | fix (F1, trainer side) |
| `bluscript_zero_ckpt.cpp` | B=4, `BLU_VAL_STEPS`, `[MEM]`, `BLU_TEST_NAN/INF_GRAD_STEP` + `BLU_TEST_BAD_GRAD_BF16` injector | **test harness only, never commit** (labelled in the code) |

**Machine-code proof:** disassembly of the new binary. Every one of the 4 functions calls `convert_type_cuda_generic` and none calls `Tensor::as_type`. In the old binary each one called `as_type` + `copy_memory`.

**Tests (H200, GPU 1):**

| test | result |
|---|---|
| **T1** `test_fix_family.cu` (sites B, C, D; the old path is an exact copy of the old code) | **0 predictions violated** (v2). The memory check first used the allocator's lifetime `active_peak`, which a later case cannot raise: that was a test flaw, and v2 uses live bytes. Results: **0 bit mismatches** at C (n = 1,472,200,704), D (same n), and B (8 params, 5 destinations misaligned, so the NEW path takes the scalar kernel), including NaN/Inf/halfway/subnormal/near-max. NaN payloads were identical too. Live memory held: C old path **5,888,802,816 B**, new **0**; D old **2,944,401,408 B**, new **0** |
| **T2a** before-F1 binary + NaN in a grad at step 2 | **bug reproduced**: step 2 printed the NaN group's norm as **0.0000**, then `ERROR: NaN/Inf loss at step 3`, MPI_Abort 2 |
| **T2b** new binary + NaN at step 2 (fp32 group) | `[FATAL] ... grad norm^2 = nan in the fp32 group (116029184 params) ... raised BEFORE any update`, exit **4**, no checkpoint |
| **T2c** new + Inf at step 2 | same, `grad norm^2 = inf`, exit 4 |
| **T2d** new + NaN in a **bf16** weight (`mlp[0]`, param dtype bf16) | `[FATAL] ... nan in the bf16 group (1472200704 params)`, exit 4. The NaN went through the new site-B convert and was still caught |

**Expected effect (estimates):**
- **Memory:** ≈ 0 at the peak. The temporaries only exist in the optimizer step, when activations are already freed (~48 GiB live vs the 88 GiB backward peak).
- **Speed:** ~0.05–0.06%.
- **Precision:** bit-identical on finite steps (F1 only acts on a non-finite norm).
- The #2 A/B (T3) will measure memory and speed.

## 19.9 Pushed to `dist_side_merge` (2026-09-25)

Direct pushes, the user's decision (no PR). Author: `Goutham Kumar <goutham.kumar@blubridge.com>`. Each remote was
fetched first to confirm nobody had pushed since 2026-09-23.

| repo | branch | range | contents |
|---|---|---|---|
| `BlubridgeAI/Tensor-Implementations` | `dist_side_merge` | `8808f89..ad10e7d` | the 7 loss files (Bugs A/B/C + ignore_index/denominator). It **supersedes `a407093`'s AccT partial fix** in `LossKernels.cu`; that commit's other 2 files (allocator) are untouched |
| `BlubridgeAI/BluTrain` | `dist_side_merge` | `03b94cf..a264586` | `dist/zero/src/ShardedOptimizer.cpp` (F1 + #2 sites B′/B/C/D), `ShardedOptimizer.h`, `ZeroErrors.h`, `bluscript_zero_ckpt.cpp` (F1 try/catch + `[MEM]` print), submodule `8808f89 → ad10e7d` |

**Not pushed:**
- the test-harness edits (B=4, `BLU_VAL_STEPS`, NaN/Inf injector): removed from the box copy before the commit, so a
  1-GPU re-test needs B=4 again;
- the 3 untracked test files in `Tensor-Implementations/Tests/kernel_tests/`: local copies are in
  `profiling_reports/fix2_2026_09_25/`.

**Pre-push checks:**
- Every caller of the changed loss API in both repos uses the 2-argument `sparse_cross_entropy_loss(logits, target)`,
  which still works through the defaults:
  - `bluscript_zero.cpp`, `bluscript_single.cpp`;
  - `train/production/bluscript.cpp`, `train/gpt2/train.cpp`;
  - 4 dist tests, `gpt2_fmha.cpp`.
- Nobody else calls the removed or changed kernel functions.
- The only NaN/Inf ZeRO test (`dist/zero/tests/test_zero.cpp:251-278`) uses a LossScaler, a path that is unchanged.
- The 3-agent deep check was attempted but hit the session limit; none ran.

**Behaviour change for other trainers:** `bluscript_zero.cpp` and `train/production/bluscript.cpp` also use
ShardedOptimizer v1 without the catch. On a NaN/Inf gradient they now terminate with the `ZeroNonFiniteGradError`
text, instead of writing NaN and dying one step later.

**Colleagues pulling this must:**
1. `git pull`, then `git submodule update --init` (the library pointer moved);
2. delete the stale loss objects before building (`make` does not track header changes, §20.3):
   `find Tensor-Implementations \( -name "*Loss*.o" -o -name "*CrossEntropy*.o" \) -delete`;
3. rebuild.

`BLU_BF16_LOGITS=1` now works: it used to crash at step 0.

**Still untested:** world_size ≥ 2 (site B′ / overlap path), B=8, Muon, ZeRO-v2, `BLU_GLOBAL_CLIP=1` in a real run.

## 19.10 2-GPU test (2026-09-25): site B′ and F1 at world_size 2

**Setup:**
- Both H200s (`CUDA_VISIBLE_DEVICES=0,1`, `mpirun -np 2`), B=4 per rank, `grad_accum=32`, `BLU_BF16_LOGITS=1`, 20 steps.
- A-B-A-B: `_runner_OURS_MEM` (old optimizer) vs `_runner_FIX_ALL` (F1 + #2; same code as the push plus the test harness).
- Every T4 run printed `[ZeRO-1] overlap: reduce=ON (hook-driven, during backward)`, so **`on_grad_ready` (site B′) ran**
  in the new arm.
- Script `profiling_reports/fix2_2026_09_25/t4_w2.sh`; logs in `box_state/t4/`.

| | old rep1 | new rep1 | old rep2 | new rep2 |
|---|---|---|---|---|
| step-0 loss | 11.297731 | 11.297731 | 11.297731 | 11.297731 |
| step-0 grad norm (computed after B′) | 18.8893 | 18.8892 | 18.8892 | 18.8893 |
| live peak (MiB) | 75,050 | 75,050 | 75,050 | 75,050 |
| **reserved peak (MiB)** | 79,376 | **76,566** | 79,376 | **76,566** |
| steady tok/s | 84,216 | 84,343 | 84,246 | 84,346 |

**Results:**
- **Reserved −2,810 MiB per GPU in both pairs**, which is exactly the ÷2 prediction (5,616 / 2 = 2,808). The ÷W rule for
  #2 holds.
- **The live peak is unchanged.**
- **Speed:** +0.13% (84,231 → 84,345), in the same direction in both pairs. That is small, and consistent with the
  expected ~0.05–0.1%.
- **Loss:**
  - mean |Δ| over 20 steps: old vs new 0.300 / 0.199;
  - same code twice: old 0.466, new 0.181. Inside the noise.
  - val@19: old 8.221 / 8.040, new 7.996 / 7.971. Overlapping.
- **Site B′ verdict:** correct. A wrong slice or offset would give a very different norm and loss. Bit-identity of the
  conversion itself is proven by `test_fix_family` (misaligned slices included).
- **F1 at W=2** (reference path `BLU_NO_OVERLAP_BWD=1`, NaN at step 2):
  - **both ranks** printed `[FATAL] ... grad norm^2 = nan ... raised BEFORE any update`, exit 4;
  - no hang, no checkpoint.
  - The injector cannot reach the optimizer on the overlap path: grads are copied inside backward. So F1 on the overlap
    path is covered by code reading only, not by a run. It is the same `step()` guard after the same all-reduce.

## 20. Reproduction recipe — every change made on the colleague's branch

The box copy will not survive a box release, so this section is the only complete record of what was
done to `dist_side_merge` today. None of it is committed to their branch.

### 20.1 Box preparation

```bash
mkdir -p /root/bugtest/mix
for d in commoncrawl code math; do ln -sfn /root/gautam_k /root/bugtest/mix/$d; done
cd /root/bugtest && git clone -b dist_side_merge https://github.com/BlubridgeAI/BluTrain.git
cd BluTrain && git submodule update --init --recursive
git log -1 --format="%h %ad %s" --date=short; git -C Tensor-Implementations rev-parse --short HEAD
```

The trainer needs three data domains. The loader takes any `*.bin` whose **file name** contains `train` or
`val` (`dist/dataloader_dist.h:38-39`), so one dataset symlinked three times works. For an A/B, all arms
see identical data.

Smoke gate, before building:

```bash
L=Tensor-Implementations/src/Kernels/cuda/loss/LossKernels.cu
grep -c "tid \* 4" $L                              # Bug B present:  > 0
grep -c "losses\[row\] = static_cast<T>" $L        # Bug A present:  3
grep -c "VEC_ELEMS" $L                             # their claimed fix: 0
grep -c "auto\* grad_out_ptr" Tensor-Implementations/src/autograd/backward/LossBackward.cpp   # Bug C: 1
```

### 20.2 Build their baseline

```bash
cd /root/bugtest/BluTrain
make build-mpi FILE=bluscript_zero_ckpt.cpp WITH_BLUBLAS=1 -j44 2>&1 | tee /root/bugtest/build_THEIRS.log
grep -oE "compute_90a|sm_90a|sm_86" /root/bugtest/build_THEIRS.log | sort -u     # expect compute_90a sm_90a
```

**Trap:** a saved copy of their `_runner_` stops working once `libtensor.so` is replaced. The behaviour
lives in the shared library (`symbol lookup error … sparse_cross_entropy_loss(Tensor const&, Tensor const&)`).
To re-run their baseline, restore their `src/` and `include/` and rebuild.

### 20.3 The port — 7 files

| file | carries |
|---|---|
| `src/Kernels/cuda/loss/LossKernels.cu` | Bug A + Bug B (`LossVec<T>`, fp32 buffers), `ignore_index`, denominator |
| `src/Kernels/cuda/loss/arch/CrossEntropyKernels_sm103.cu` | the Bug A signature (`float* losses`) — without it, a link error |
| `include/ops/helpers/LossKernels.h` | fp32 `loss_output` / `grad_output` signatures |
| `include/autograd/backward/LossBackward.h` | `ignore_index_`, `denom_`, new constructor parameters |
| `include/autograd/operations/LossOps.h` | the 4-parameter `sparse_cross_entropy_loss` — without it, a link error |
| `src/autograd/backward/LossBackward.cpp` | Bug C (`const float*` + boundary convert) |
| `src/autograd/operations/LossOps.cpp` | fp32 loss tensor on both paths, `ignore_index`, denominator |

Why 7:
- `LossOps.h` is in commit `a7da79d`, so `git status` does not show it. That commit's `ReductionImpl.h`
  and `ReductionUtils.cpp` changes are comment-only and stay out.
- Source of the files: `sft_gautam` once committed (`git show sft_gautam:<path>`), or the tarball
  `loss_fix_7files.tgz` (md5 `6df31c4623bf01a1b3c27d1dfe053b07`, local copy in
  `profiling_reports/bugtest_2026_09_24/scripts/`).

```bash
cd /root/bugtest/BluTrain/Tensor-Implementations
mkdir -p /root/bugtest/backup_theirs && cp -r src include /root/bugtest/backup_theirs/
tar xzvf /root/bugtest/loss_fix_7files.tgz
# gate 1 -- all must flip
grep -c "LossVec" src/Kernels/cuda/loss/LossKernels.cu                          # > 0
grep -c "tid \* 4" src/Kernels/cuda/loss/LossKernels.cu                         # 0
grep -c "AccT" src/Kernels/cuda/loss/LossKernels.cu                             # 0 (their partial fix, superseded)
grep -c "const float\* grad_out_ptr" src/autograd/backward/LossBackward.cpp     # 1
grep -c "ignore_index" src/autograd/operations/LossOps.cpp                      # > 0
# FORCE the rebuild: tar keeps the files' old mtimes, so make would otherwise relink the OLD objects
find . \( -name "*Loss*.o" -o -name "*CrossEntropy*.o" \) -print -delete
cd .. && rm -f _runner_
make build-mpi FILE=bluscript_zero_ckpt.cpp WITH_BLUBLAS=1 -j44 2>&1 | tee /root/bugtest/build_OURS.log
# gate 2 -- unanchored, so collect2 / undefined reference cannot hide
grep -ciE "error|undefined reference|Error 1" /root/bugtest/build_OURS.log      # 0
# gate 3 -- the BINARY, not the source
nm -C Tensor-Implementations/lib/libtensor.so | grep "sparse_cross_entropy_loss("   # must show (…, long, Tensor const*)
```

### 20.4 Harness-only edits to `bluscript_zero_ckpt.cpp` — box copy only, NEVER commit

```bash
cd /root/bugtest/BluTrain
# a) B=8 -> 4 (B=8 does not fit one H200). Check the line number first; it differs by branch.
sed -n '198p' bluscript_zero_ckpt.cpp
sed -i '198s/int64_t B            = 8;/int64_t B            = 4;/' bluscript_zero_ckpt.cpp
# b) print the allocator's own peak stats at the end of training
sed -i '/=== Training complete ===/a\        { auto st = CachingCUDAAllocator::instance().get_stats(); std::cout << "[MEM] active_peak " << st.active_peak/1048576 << " MiB | allocated_peak " << st.allocated_peak/1048576 << " MiB | reserved_peak " << st.reserved_peak/1048576 << " MiB | cached_free " << st.cached_free_bytes/1048576 << " MiB" << std::endl; }' bluscript_zero_ckpt.cpp
# c) make validation length overridable (default unchanged at 20)
sed -i '/getenv_int("BLU_MAX_STEPS", 0); ms > 0) {/i\    cfg.val_loss_steps = getenv_int("BLU_VAL_STEPS", cfg.val_loss_steps);' bluscript_zero_ckpt.cpp
grep -c "\[MEM\]" bluscript_zero_ckpt.cpp; grep -c "BLU_VAL_STEPS" bluscript_zero_ckpt.cpp   # 1 and 1
make build-mpi FILE=bluscript_zero_ckpt.cpp WITH_BLUBLAS=1 -j44 2>&1 | tail -3       # trainer only, ~45 s
cp _runner_ /root/bugtest/_runner_OURS_MEM
```

### 20.5 Running

```bash
export LDP=/usr/lib/x86_64-linux-gnu:/root/bugtest/BluTrain/Tensor-Implementations/lib:/root/bugtest/BluTrain/Profiler/lib:/root/bugtest/BluTrain/dist/lib
cd /root/bugtest/BluTrain && rm -rf checkpoints_bluscript
LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_MIX_ROOT=/root/bugtest/mix \
BLU_BF16_LOGITS=1 BLU_MAX_STEPS=20 BLU_CKPT=0 \
mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x BLU_MIX_ROOT -x BLU_BF16_LOGITS -x BLU_MAX_STEPS -x BLU_CKPT \
  /root/bugtest/_runner_OURS_MEM 2>&1 | tee /root/bugtest/one_arm.log
grep -c "^step" /root/bugtest/one_arm.log; grep -ci "resume\|loaded DCP" /root/bugtest/one_arm.log   # 20 and 0
```

- **Naming the faulting kernel:** prefix the binary with `compute-sanitizer --tool memcheck --print-limit 20`
  (with `BLU_MAX_STEPS=5`).
- **The full unattended session:**
  `ALLOW_POLLERS=1 setsid nohup bash /root/bugtest/gpu_hour.sh > /root/bugtest/gpu_hour.nohup 2>&1 < /dev/null &`
  - watch: `tail -f /root/bugtest/hour/session.log`
  - stop: `kill -- -$(cat /root/bugtest/hour/script.pgid)`

### 20.6 The local precision test

```bash
cd ~/Downloads/sft_research/BluTrain/Tensor-Implementations
make run-cuda FILE=Tests/kernel_tests/test_cast_equivalence.cu        # on the box: add WITH_BLUBLAS=1
# expect: "predictions violated: 0"; C4 99.999% identical, >1-ulp 0
```

### 20.7 Getting results off the box

```bash
ssh root@86.38.238.202 'cd /root/bugtest/hour && tar cf - .' | tar xf - -C <local dir>
```

`scp` with a quoted list of remote files fails on modern OpenSSH: it looks for one file with that whole
name.

### 20.8 Parsing ncu

Appendix B. It locates the real `"ID"` header (ncu prints `==PROF==` lines first), asserts the units are
bytes, and classifies the logits-path kernels by name **and** by size (2 or 4 bytes × 824,180,736
elements). It prints how many kernels it parsed before printing any number.

### 20.9 Operational gotchas hit today

| trap | symptom | fix |
|---|---|---|
| running as root | `mpirun has detected an attempt to run as root` | `--allow-run-as-root` |
| `BLU_MAX_STEPS` too small | `leaves no decay tail`, aborts before any GPU work (the sanitizer then says "0 errors") | ≥4; **5 verified** |
| checkpoints | the next run auto-resumes from the previous run's checkpoint and "completes" with 0 steps | `BLU_CKPT=0` **and** `rm -rf checkpoints_bluscript` |
| sticky CUDA errors | the trace blames `Tensor::to_cpu()`, not the faulting kernel | `compute-sanitizer` or `CUDA_LAUNCH_BLOCKING=1` |
| header edits and `make` | "up to date" after copying files in | delete the dependent `.o` files |
| mawk | aborts on a missing input file | pass only existing files; never `2>/dev/null` it |
| shared GPU | a co-tenant took 43 GB mid-run | check `nvidia-smi --query-compute-apps` before and after, never during |
| pollers | `watch nvidia-smi` loops left open (1 s, 0.1 s, 0.1 s) | the preflight lists them; they're constant across runs, so the A/B stays fair |
| B=8 | OOM on one GPU (ZeRO shards nothing at world_size=1) | B=4 |

---

## 21. Applying the fixes on a colleague's system

The 7 files are on `BlubridgeAI/Tensor-Implementations`, branch `sft_gautam`, at `1ed2ac3`. They are
identical, file by file, to what was validated on the H200. Take them from git, or from the tarball (§20.3).

| # | file (the same relative path on both sides, under `Tensor-Implementations/`) |
|---|---|
| 1 | `src/Kernels/cuda/loss/LossKernels.cu` |
| 2 | `src/Kernels/cuda/loss/arch/CrossEntropyKernels_sm103.cu` |
| 3 | `include/ops/helpers/LossKernels.h` |
| 4 | `include/autograd/backward/LossBackward.h` |
| 5 | `include/autograd/operations/LossOps.h` |
| 6 | `src/autograd/backward/LossBackward.cpp` |
| 7 | `src/autograd/operations/LossOps.cpp` |

**Before replacing:**
- Check his tree for a local, unpushed fix: `grep -c VEC_ELEMS Tensor-Implementations/src/Kernels/cuda/loss/LossKernels.cu`.
  More than 0 means a local fix exists: back it up and reconcile it first (§9).
- Back up his 7 files.

**After replacing:** delete `*Loss*.o` and `*CrossEntropy*.o` before building, because make does not track
header changes. Then build as usual.

**Runs — flag 0 and 1, before and after:**

| run | expected |
|---|---|
| before, `BLU_BF16_LOGITS=0` | runs; baseline memory |
| before, `BLU_BF16_LOGITS=1` | **crashes at step 0**, `misaligned address`. If it runs instead: a local fix exists, or the flag did not reach the ranks |
| after, `0` | runs; memory ≈ before-0; step-0 loss identical |
| after, `1` | **runs**; nvidia-smi ≈ 9 GiB or more lower per GPU at B=8 (9,432 MiB live, plus allocator cache) |

- **Pass the flag to every rank:** `mpirun … -x BLU_BF16_LOGITS …`. Check that the log shows `logits bfloat16`.
- **Before each run:** set `BLU_CKPT=0` and `rm -rf checkpoints_bluscript`, otherwise a run resumes the
  previous run's checkpoint.
- **Reading nvidia-smi:** look after step 1 is printed. The allocator doesn't return memory until the process
  exits, so one look mid-run gives the peak.
- **B=8** fits on 8 GPUs, because ZeRO shards the optimizer. On one GPU it OOMs — use B=4 there.

**Training-script edits: none required.** These text fixes are optional, but the current text is now wrong:

| where | what |
|---|---|
| comment near `:168` | claims a `VEC_ELEMS / Chunk16` fix that never existed (§9) |
| the 15-line comment above `:1994` (`loss32`) | says the loss follows the logits' dtype; it is fp32 (§5.6) |
| the `[precision] bf16 logits ON` banner | says "frees ~3 GiB" and "the loss quantizes to ~0.06"; with the fixes it frees ~9 GiB and the loss is exact |
| `:188` | `bool bf16_logits = true;` makes bf16 the default. Change the default; don't delete the cast line |

Our box edits (B=4, `[MEM]`, `BLU_VAL_STEPS`) are test-only and must not be carried over.

---


## 22. F1 for ShardedMuon and SymmetricShardedOptimizer (2026-09-26)

**Pushed:** BluTrain `dist_side_merge` `e3633e8..23ae1e5`. Files: `dist/zero/src/ShardedMuon.cpp`,
`dist/zero/src/SymmetricShardedOptimizer.cpp`, and a comment-only change in `bluscript_zero_ckpt.cpp`.

**The bug.** It is the same as v1's (§19.7 F1, fixed in `a264586`). Every guard was `if (sc && !isfinite(nsq))`,
and no trainer installs a LossScaler, so a NaN/Inf gradient was never caught. The 09-25 note "F1 doesn't apply to
Muon/v2" was wrong. Measured before the fix (standalone test, 1 × H200, bf16 and fp32):

| optimizer / tensor | what a NaN or Inf gradient did |
|---|---|
| v2 (`BLU_ZERO_V2=1`) | no error; NaN written into the weight and into m/v; step counted |
| Muon, 1-D tensors (AdamW path) | no error; NaN written into the weight |
| Muon, 2-D matrices (the body) | **silent**. `muon_momentum_cuda` writes the NaN into m **in place** before `muon_has_nonfinite_cuda` rejects the Newton-Schulz output. The weights stay finite, but m keeps the NaN, so that matrix's update is rejected on every later step: it stops learning with no error |
| `BLU_GLOBAL_CLIP=1` | no help. `reduce_and_measure` returns −1, the trainer only drops the global coefficient and calls `step()`, which measures the NaN again and goes on |

**The fix** is the v1 pattern. Without a scaler, `step()` throws `ZeroNonFiniteGradError` **before any update**.
With a scaler nothing changes (skip and back off). A NaN norm is logged as nan, not 0. `nonfinite_grad_msg` is
file-local, so no header changes and no trainer rebuild are forced. The trainer's existing catch wraps `step()` for
every group, so all three optimizers stop with one `[FATAL]` line and exit 4, and no checkpoint is written.

**Test** (`profiling_reports/f1_muon_v2_2026_09_26/test_f1_nonfinite.cpp`): 36 cases, covering v1/v2/Muon ×
bf16/fp32 × {finite, NaN matrix, NaN vector, Inf, global-clip path, with scaler}. Before: v1 12/12, v2 and Muon
fail all 16 no-scaler cases. After: **36/36**; weights, m/v and step count unchanged on every stopped step. The
trainer builds.

**Test traps (4 re-runs):**
- lr 3e-4 cannot move a bf16 weight in one step.
- Muon test matrices under 64 (64×48, 32×64) made every Newton-Schulz update be rejected, even healthy ones.
  BluBLAS's transposed bf16 GEMM is wrong there (§26).
- A healthy control belongs in the **first** run.

**Not covered:**
- `max_grad_norm = 0` with no scaler: v1/v2 measure no norm, so there is nothing to check.
- `bluscript_zero.cpp` and `train/production/bluscript.cpp` have no catch (they stop via `std::terminate`).
- Muon/v2 were not run at W > 1.

---

## 23. Reductions: 7 GPU/CPU bugs fixed (2026-09-26)

**Pushed:** library `7810aea..97735ad`; BluTrain `23ae1e5..f1618d8`. Files: `src/UnaryOps/cuda/ReductionImplGPU.cu`,
`include/ops/helpers/ReductionKernels.cuh`, `include/ops/helpers/ReductionOps.h`, `include/ops/helpers/ReductionImpl.h`.

**How it was found:** the §17.3 library benchmark crashed in `reduce_mean` (bf16, axis 0). compute-sanitizer:
"Invalid `__shared__` write of size 4" in `unified_reduce_kernel<__nv_bfloat16,…,MeanOps,…,Path 1,512>`.

**Test** (`profiling_reports/reduce_gpu_2026_09_26/test_reduce_mean_gpu.cu`):
- 43 cases, one process each, GPU **and CPU** against an exact double reference (1 output ulp).
- Covered: mean/sum/max; bf16/fp16/fp32/fp64/int32; leading, last and middle axes; non-consecutive axes;
  transposed views; block widths 8/16/32/>32; multi-CTA.

| | before | after |
|---|---|---|
| GPU | 9 ok / 13 WRONG / 11 CRASH (of the first 33) | **43/43** |
| CPU (checked for the first time) | wrong on all 6 middle-axis cases | **43/43** |
| compute-sanitizer | memcheck: invalid shared write; racecheck: 11,521 errors | memcheck 0 (4 cases); racecheck 0 hazards (2 cases) |

| # | bug | fix |
|---|---|---|
| 1 | **mean crash** (bf16/fp16/int): `dispatch_mean_gpu` sized the config and shared memory with the INPUT type, but the kernel stores `arg_t` (float = 4 B for a 2 B input) | `build_reduce_config<arg_t>`, as the other 3 dispatchers and PyTorch (`ReduceConfig(sizeof(arg_t))`, Reduce.cuh:1039). Block shape unchanged (MaxThreads = 512 for all types) |
| 2 | **wrong last-axis results** (every dtype/op): `block_x_reduce` treated the 2-D block as 1-D (`smem[threadIdx.x/32]` ignored the row; the shuffle crossed rows) | PyTorch's `block_x_reduce` line for line (Reduce.cuh:634, source 5f76830b728) |
| 3 | **integer mean = 0**: `factor = static_cast<int64_t>(1.0/n)` = 0 | `MeanOps::FactorT` = double for integer accumulators; exact int64 sum; `project` returns double |
| 3b | integer-mean dtype | **Float64**, as the CPU mean and NumPy (was Float32). GPU and CPU are bit-identical (tested) |
| 4 | **GPU middle axis** (`outer > 1`): config reset `num_outputs` to `inner`, so only the first slab was computed (30/35 wrong) | `num_outputs = outer*inner`, new `ReduceOp::outer_stride`, slab remap in the kernel |
| 5 | **GPU Generic path** (non-consecutive axes, transposed input): EMPTY offset calculators | new `setup_generic_path` in all 4 dispatchers; refuses more than 10 dims per side instead of computing wrongly |
| 6 | **CPU middle axis**: the OuterContiguous consumers assumed `outer == 1` | `cpu_demote_middle_to_generic` at all 6 CPU sites (the leading axis keeps SIMD) |
| 7 | **multi-CTA race**: `block_y_reduce` → `block_x_reduce` on the same shared slots, no barrier. The values were right (same value, same slot), but it is a data race | one `__syncthreads()` there (only the last CTA of each tile runs it) |

**Integer-mean output dtype, checked in each library's source:**
- PyTorch refuses integer mean (`ReduceOps.cpp:310`). With `dtype=` it sums in that float type.
- NumPy returns float64 (`_methods.py`: "Cast bool, unsigned int, and int to float64").
- TensorFlow sums in int64, then does an integer division and returns the int type (truncated; `reduction_ops.h`
  `CASTING_SPECIALIZATION`).
- JAX returns float32 by default (`to_inexact_dtype`).

Our choice combines them: PyTorch's int64 accumulator (`AccumulateType.h` int → int64, the same table as ours),
NumPy's fp64 output, and our CPU's formula. The accumulation cost does not depend on the output dtype; only the
final store grows by 4 B.

**History:** fixes 2, 4, 5 and 6 are the user's **June 2026** fixes (`master_gau_latest_sm_86_RTX_3060` / BBP,
validated 87/87 on sm_86 and sm_89). They were **never committed** to BlubridgeAI/Tensor-Implementations: no
commit or branch ever had the 2-D `block_x_reduce`. The 06-11 "merge the v2 compiler" commit (`3160082`) brought
in the unified kernel with the 1-D version. June Bug 4 (the trailing barrier) was needed for June's vec4 kernel,
which this tree does not have. Here, only the call-site barrier (fix 7) is needed.

**Build note:** the Tensor-Implementations Makefile sets no `-j`. Always `make -j$(nproc)`. `ReductionImplGPU.cu`
alone takes more than 4 minutes (one translation unit, every op × dtype).

**Not measured / not covered:**
- Reduction speed (no before/after A/B). The changes add one integer division per thread and one barrier per
  multi-CTA tile.
- argmax/argmin and variance over a middle axis (same remap/demote, not tested).

---

## 24. GPU `as_type`: no float box, complex converted (2026-09-26)

**Pushed:** library `97735ad..3a08ade`; BluTrain `f1618d8..4e92e6c`. File: `src/Kernels/cuda/misc/ConversionKernels.cu`.

**The bugs** (measured on the old code: 110 of 169 real dtype pairs ok, **59 mismatched**):

| bug | example from the test |
|---|---|
| every value went through a `float` temporary (24 bits) | int32 16,777,217 → int64 gave 16,777,216; uint32 0xFFFFFFFF → fp64 gave 4294967296 |
| negative int → unsigned through an out-of-range float | int8 −1 → uint8 gave **0** (CPU: 255) |
| uint → int through an out-of-range float | uint32 0xFFFFFFFF → int32 gave 0x7FFFFFFF (CPU: −1) |
| double → int rounded first | fp64 15939.99999999 → int16 gave 15940 (CPU: 15939) |
| int / double → bf16 / fp16 rounded twice | int32 147,324,935 → bf16 gave 0x4d0c (CPU: 0x4d0d) |
| a tiny double → bool | 1e-40 → false (CPU: true) |
| **complex → complex silently SKIPPED** | output left uninitialised, no error |
| 32-bit index product | wraps above 2³² elements |

**The fix** mirrors the CPU `as_type` (`AsTypeTensor.cpp`):
- bf16/fp16 sources keep the float path (exact for them).
- Wide sources → bf16/fp16 use the Types.h one-rounding constructors (the GPU intrinsics).
- Wide → wide is `static_cast<Dst>(x)`, direct.
- complex → complex is component by component; complex ↔ real throws in `convert_type_cuda_generic` too.
- The index is 64-bit.

fp32 → bf16/fp16 still use the same intrinsic, so they are bit-identical. The SM90/SM103 vector fast paths are
untouched.

**Test** (`profiling_reports/conversion_kernels_2026_09_26/test_as_type_gpu.cu`):
- GPU against CPU for all 169 real pairs and 6 complex pairs, 131,072 values each.
- Inputs: every bf16/fp16 bit pattern; the 2²⁴, 2⁵³, INT64 and uint boundaries; ±Inf, NaN, subnormals.
- Out-of-range float → int is skipped as undefined behaviour.
- Result after: **169/169**, complex 6/6, the direct kernel call refuses complex ↔ real.
- Speed: 9 hot pairs within 1%, e.g. fp32 → bf16 96.7 vs 96.9 µs and bf16 → fp32 102.3 vs 102.2 µs at 67 M elements.

**Unchanged:** fp32 subnormals (< 1.18e-38) still flush to zero on the GPU. The whole library is built with
`--use_fast_math`, so the FTZ counts are the same before and after.

**BluTrain impact today: none.** Its integer `as_type` calls convert only token ids and labels
(`DistributedNN.h:788, 801, 1613`, the CP script). Those are below 16.7 M, so they were exact through float.

**Test traps:**
- My fp32 → complex case reused the complex buffer, so `memcpy` overflowed the heap and the run died with "double
  free".
- The FTZ category briefly hid the real fp64 → bool bug (28,629 elements in the before-run, 0 after).

---

## 25. Everything pushed on 2026-09-26, and what colleagues must do

All commits are by `Goutham Kumar <goutham.kumar@blubridge.com>`, on branch `dist_side_merge`.

| work | Tensor-Implementations | BluTrain | section |
|---|---|---|---|
| Bug D + bf16/fp16 conversions | `ad10e7d..7810aea` | `e3633e8` | §17.3 |
| F1 for Muon + ZeRO-v2 | — | `e3633e8..23ae1e5` | §22 |
| 7 reduction bugs | `7810aea..97735ad` | `23ae1e5..f1618d8` | §23 |
| GPU `as_type` conversions | `97735ad..3a08ade` | `f1618d8..4e92e6c` | §24 |

**Colleagues must:**
1. `git pull` BluTrain, then `git submodule update --init`.
2. Rebuild with `make -j$(nproc) tensor WITH_BLUBLAS=1 SM_ARCH=<arch>`, then rebuild the trainer (`make build-mpi …`).

**Behaviour changes to announce:**
- The GPU integer mean now returns **Float64**. It was Float32, and its values were 0.
- GPU `as_type` complex → complex now converts, and complex ↔ real raises an error.
- Muon/v2 runs now stop with `[FATAL] … exit 4` on a NaN/Inf gradient, instead of corrupting the weights or
  silently freezing matrices.

**Box:** `fair-soul-wanders-fin-02` (1 × H200) was released after a final copy. Everything is in
`profiling_reports/box_fair_soul_2026_09_26_FINAL/` (125 files) and the per-topic folders named above.

---

## 26. Future tasks: every pending item (updated 2026-09-29)

**Done 2026-09-28/29:**
- [x] **F2 — fused gradient accumulation** (§27): bit-identical, +0.52% / +0.60% tok/s at W=1, −156 kernels and
      −156 allocations per micro-step, peak memory unchanged.
- [x] **LinearBackward** 3-D input gave the weight no gradient (§28).

**From F2 (2026-09-28/29):**
- [ ] The fused kernel runs at 3.4 TB/s in the trainer, against 4.2 TB/s for the old fp32 add (§27.7). About 50 ms
      per step (~0.2%) is left. Cause not investigated.
- [ ] Measure F2 at production (W=8, B=8, GRAD_ACCUM=4). Estimate ~0.25–0.3%, not measured.
- [ ] **F2b:** `zero_grad` as set-to-none instead of a zero-fill, so the first micro-step of each step assigns
      instead of adding into zeros. That saves one zero-fill and one accumulation per weight per step: 1/64 of the
      accumulation work at W=1, but 1/4 at production GRAD_ACCUM=4. First check that every reader of the grad
      handles an undefined grad.
- [ ] Let the weight-gradient GEMM accumulate straight into the fp32 main grad (beta = 1, as Megatron does). That
      removes the bf16 gradient tensor and the accumulation kernel altogether.
- [ ] A tensor-parallel run through `DTensor::Linear`, to confirm its weights now train (§28).

**From the BLUCCL port (2026-09-29, §30):**
- [ ] Multi-GPU and the BluCCL comm paths were not tested with our changes; F1 did not fire in the test run.

**From today's work:**
- [ ] Reduction **speed** A/B, before vs after §23 (not measured).
- [ ] Tests for argmax/argmin and variance over a middle axis (§23; same fixes, not measured).
- [ ] Review: `unified_reduce_kernel` returns early (`output_idx >= num_outputs`) before later `__syncthreads()`.
      PyTorch does not; it computes and masks. Sanitizers were clean on every tested case.
- [ ] CPU middle-axis reductions now use the Generic path (no SIMD). A per-slab SIMD loop could come if CPU speed there ever matters.
- [ ] `OffsetCalculator` holds 10 dims per side. The GPU Generic path now refuses more; extend it if a model ever needs that.
- [ ] Reduction scratch allocation: the stream is `getCurrentStream()` vs the kernel's stream (a secondary risk; see
      the reduction 32-bit index guard notes).
- [ ] **BluBLAS transposed bf16 GEMM** returns wrong numbers, with no error, when M/N are not multiples of 64 (seen
      via Muon Newton-Schulz at 48/32). Reproduce with a direct `cuda_matmul` vs CPU test at M/N ∈ {32, 48, 96, 100},
      then add a shape guard with a fallback, or real residue support.
- [ ] `ConditionalOps.cu`: complex32 from a double scalar rounds twice (4 spots).
- [ ] Optional: build `ConversionKernels.cu` without FTZ, if exact fp32 subnormals in GPU conversions are ever wanted.
- [ ] F1 gaps (§22):
      - `max_grad_norm = 0` with no scaler has no check;
      - `bluscript_zero.cpp` and `train/production/bluscript.cpp` have no catch;
      - Muon/v2 were not run at W > 1.
- [ ] §19.7 audit items still open: F3 (5 grad and 4 param passes per step at W=1), F4 (v1's bf16 wire at W=1), per
      the decisions in that table. F2 is done (§27).

**Carried from §13 (still open):**
- [ ] Hand the port to the `dist_side_merge` owners as a branch/PR, recommending `BLU_BF16_LOGITS=1`.
- [ ] Rewrite the stale comment above `loss32` in `bluscript_zero_ckpt.cpp` (§5.6), and remove the false
      `VEC_ELEMS / Chunk16` claim (§9).
- [ ] More throughput repetitions for the bf16-logits A/B (+0.60% is n = 2); explain the gap to 0.79–0.94% (§16.6, #4).
- [ ] Attribute the third live buffer at the peak, and the 580 MiB fp32 poller gap (§16.3).
- [ ] Observe Bug C's zero gradients with a Bug-A-only build.
- [ ] Trainer run-to-run non-determinism (§18). Measured larger on 2026-09-29: two identical runs differ by up to 7.9
      in loss at step 5, which makes any loss A/B between runs hard.
- [ ] #1 embedding backward for a bf16 weight (checked: keep fp32); #3 `check_type()` in `data<T>()` (§19.5, §8).
- [ ] Apply the fixes on a colleague's multi-GPU system; run flag 0/1 before and after (§21).
- [ ] Refresh the `LossKernels.h` note counts; the `batch_size > 65536` reduction cap; the target bounds check;
      sm103/B300 masking (`TODO(SFT)`).

**SFT (the assignment):**
1. [ ] Weights-only checkpoint loader for `navingv/curriculum_8btokens` `blumodel_step_8000` (Muon, 4-GPU DCP).
2. [ ] Pick the dataset.
3. [ ] Preprocessing script.
4. [ ] SFT dataloader.
5. [ ] Trainer changes: `ignore_index = 65535`, the denominator, SFT hyperparameters, weights-only load.
6. [ ] Loss-kernel bounds check.
7. [ ] A sample SFT run on an H200.

**Housekeeping:**
- [ ] **Rotate the GitHub tokens** used in today's push commands (the org PAT), and the token in the blublas remote URL.
      Also the org token pasted in chat on 2026-09-29.
- [ ] Commit the local `tf_issue_3dxz/` changes (unrelated to this doc) when ready.

---

## 27. F2: fused bf16/fp16 → fp32 gradient accumulation (2026-09-28, A/B repeated 2026-09-29)

**Pushed:** library `3a08ade..d96b991`; BluTrain `4e92e6c..909ab75` (§29).
**Off switch:** `BLUTRAIN_NO_FUSED_GRAD_ACC=1` restores the old path. Both paths give the same bits, so the switch
exists only for A/B timing. `BLUTRAIN_REPORT_FUSED_GRAD_ACC=1` prints the fused-call count at exit.

### 27.1 The waste (§19.7 F2)

`AutogradMeta::accumulate_grad` keeps an fp32 `grad_` (the "main grad") for every bf16 weight. After the first
micro-step of the run, every micro-step did `*grad_ += update.as_type(Float32)` (`AutogradMeta.cpp:173`):

| part | traffic per element | cost |
|---|---|---|
| `as_type` writes the widened update into a new full-size fp32 temporary | read 2 B, write 4 B | 1 kernel, 1 allocation, 1 free |
| the fp32 add reads the grad and the temporary, writes the grad | read 8 B, write 4 B | 1 kernel |
| **total** | **18 B** | **2 kernels, 1 allocation, 1 free** per weight per micro-step |

`zero_grad` fills with zeros rather than deleting (`ShardedOptimizer.cpp:1339-1348`). So the `+=` path runs on every
micro-step of every step; only the very first micro-step of the run assigns.

In the 1.7B trainer that is 26 layers × 6 bf16 weights (wQ, wK, wV, wO, gate_up, down) = **156 tensors,
1,472,200,704 elements**, 64 times per optimizer step at W=1, B=4 (GRAD_ACCUM = 1,048,576 / (4 × 4096)).

### 27.2 The fix (5 files, +221 / −1)

| file | lines | what |
|---|---|---|
| `src/TensorOps/cuda/TensorOpsAdd.cu` | +168 | `namespace grad_accum`: vector kernel, scalar kernel, launcher, call counter; `cuda_accumulate_lowp_into_fp32()` |
| `src/core/AutogradMeta.cpp` | +24 | calls the fused function before the old line 173; falls back to the old line when it returns false |
| `include/ops/helpers/KernelPolicy.h` | +9 | `disable_fused_grad_accum()` (`BLUTRAIN_NO_FUSED_GRAD_ACC`) |
| `include/ops/TensorOps.cuh` | +8 | the two declarations |
| `src/autograd/backward/LinearBackward.cpp` | +12 / −1 | the §28 fix (same commit) |

**The kernel** computes `acc_fp32[i] += float(update[i])` in one pass:
- Each thread reads 16 B of the update (8 bf16/fp16 values) and 2 × 16 B of the fp32 grad, widens in registers,
  adds, and writes the fp32 back: **10 B per element, 1 kernel, no temporary**.
- Grid-stride loop, 2× unrolled, 256 threads per block: the same shape as `add_inplace_kernel_sm90<float>`.
- A tail loop handles `n % 8`. A scalar kernel handles views that are not 16-B aligned.
- The function returns false and touches nothing unless: the grad is fp32; the update is bf16 or fp16; both are
  CUDA tensors on the same device; both are contiguous; both have the same shape. The caller then uses the old line.

**Why it is bit-identical:**
- bf16 → fp32 is a 16-bit shift (bf16 is the top half of an fp32). fp16 → fp32 is exact for every fp16 value.
- The add is the same single fp32 add as the old path: same file, same flags (`--use_fast_math`), `FADD.FTZ` in
  the SASS of both kernels.

### 27.3 Standalone tests (H200, `Tests/kernel_tests/test_fused_grad_accum.cu`)

| test | what it checks | result |
|---|---|---|
| P1 | new vs old, bit for bit, after 4 accumulation steps; 17 sizes from 1 to 33,554,437; bf16 and fp16; ±0, subnormals, ±Inf, NaN and overflow planted | 0 mismatches |
| P2 | views at an element offset (not 16-B aligned) take the scalar kernel; element 0 left untouched | 0 mismatches |
| P3 | unsupported inputs return false and leave the grad untouched: fp32 update, bf16 grad, shape mismatch, non-contiguous update, CPU tensors | 5 of 5 |
| P4 | control: skipping one update must change the result, so the test can fail (a gate only for n > 32: at n = 1 a tiny bf16 update is absorbed by fp32 rounding) | fires |
| P5 | wiring: `autograd::matmul` backward twice; the second pass goes through `accumulate_grad` (counter +1 with the switch on, +0 with it off); grad after 2 passes = 2 × grad after 1 | pass |
| P6 | speed, 12 rounds ABBA, median | table below |

**All checks: 136 PASS, 0 FAIL.** That is the F2 test run twice (switch on and off) plus the 16 checks of §28.

| elements | old (convert + add) | new (fused) | faster |
|---|---|---|---|
| 5,308,416 | 0.0251 ms (3.80 TB/s at 18 B) | 0.0127 ms (4.18 TB/s at 10 B) | 1.98× |
| 11,534,336 | 0.0545 ms (3.81 TB/s) | 0.0344 ms (3.35 TB/s) | 1.58× |
| 14,155,776 | 0.0658 ms (3.87 TB/s) | 0.0426 ms (3.32 TB/s) | 1.54× |
| 33,554,437 | 0.1471 ms (4.11 TB/s) | 0.1042 ms (3.22 TB/s) | 1.41× |

### 27.4 Trainer A/B

1.7B trainer (`bluscript_zero_ckpt.cpp`), 1 × H200, B = 4, T = 4096, GRAD_ACCUM = 64, BluBLAS, `BLU_BF16_LOGITS=1`,
20 steps per run, checkpoints off. One binary for both arms; only the env switch differs. tok/s is the mean over
steps 2–19.

| date, box | run | tok/s | time per step |
|---|---|---|---|
| 09-28, `fair-face-lies-fin-02` (CUDA 13.2) | OLD | 42,055.0 | 24,933.6 ms |
| | NEW | 42,305.7 | 24,786.0 ms |
| 09-29, `dry-door-shines-fin-02` (CUDA 13.0) | run1 OLD | 42,199.9 | 24,847.9 ms |
| | run2 NEW | 42,396.7 | 24,732.5 ms |
| | run3 OLD | 42,257.8 | 24,813.8 ms |
| | run4 NEW | 42,501.2 | 24,671.7 ms |

- **09-29: NEW +0.52%, −128.8 ms per step = −2.0 ms per micro-step.** The two OLD runs differ by 0.14%.
- The slower NEW run beats the faster OLD run. Both NEW runs beat both OLD runs in 18 of 18 steps. Each NEW run
  against the OLD run before it: faster in 36 of 36 steps (min +0.19%, median +0.51%, max +1.08%).
- 09-28 (one run per arm): +0.60%, −147.6 ms per step, NEW faster in 18 of 18 steps.
- Compare within a day only: the two boxes had different CUDA toolkits.

### 27.5 Direct evidence: exact counts

**nsys** (09-28; `--trace=cuda`; `f2_exact_window.py`). The window is cut on micro-step boundaries: from the start
of loss-forward kernel *k* to the start of kernel *k*+64. That is exactly 64 backward passes and exactly one
optimizer step. All 13 overlapping windows in each capture give the same counts.

| per optimizer step (64 micro-steps) | OLD | NEW | change |
|---|---|---|---|
| kernels launched (= `cudaLaunchKernel` calls) | 138,090 | 128,106 | **−9,984 = 156 × 64** |
| `vec4_convert_to_fp32` (bf16 → fp32) | 23,425 | 13,441 | −9,984 |
| `add_inplace_kernel_sm90<float>` | 16,768 | 6,784 | −9,984 |
| `accum_lowp_fp32_vec_kernel` (new) | 0 | 9,984 | +9,984 |
| GPU time in these three kernels | 1,070.4 ms | 931.7 ms | **−138.7 ms (−2.17 ms per micro-step)** |
| memcpy / memset / cudaMalloc / cudaFree | 528 / 2,058 / 0 / 0 | 528 / 2,058 / 0 / 0 | same |
| wall time per window (median of 13, under nsys) | 24,780.5 ms | 24,660.7 ms | −119.8 ms |

**Caching-allocator counters** (a box-only `[ALLOC]` print, §29; 09-29 runs, 20 steps = 1,279 micro-steps after the
first):

| | OLD | NEW | change |
|---|---|---|---|
| allocations | 2,557,236 | 2,357,712 | **−199,524 = 156 × 1,279** |
| frees | 2,556,659 | 2,357,135 | −199,524 |
| real `cudaMalloc` calls | 1,240 | 1,240 | same |
| fused calls | 0 | 199,524 | 156.000 per micro-step |

The two OLD runs have identical counters, and so do the two NEW runs.

### 27.6 Memory and precision

- **Peak memory: unchanged** in all 6 runs (active 90,322 MiB, reserved 93,410 MiB). The temporaries were made and
  freed one at a time, and the cache reused the same blocks. The largest was gate_up, 2304 × 12288 × 4 B = 108 MiB,
  so 108 MiB was the upper bound of any peak saving. The peak is set by activations. **The gain is time and GPU
  work, not memory.**
- **Precision:** step-0 loss is 11.298004 in all 6 runs. From step 1 the losses drift, but OLD-vs-OLD drifts as
  much as OLD-vs-NEW (§18 update), so precision rests on the bit-exact tests of §27.3.

### 27.7 The fused kernel is not yet at bandwidth

Effective bandwidth in the trainer (nsys, total bytes ÷ kernel time per micro-step):

| kernel | bytes per element | bandwidth |
|---|---|---|
| old `vec4_convert_to_fp32` | 6 | 3.89 TB/s |
| old `add_inplace_kernel_sm90<float>` | 12 | 4.20 TB/s |
| new fused kernel | 10 | **3.42 TB/s** |

- The fused kernel takes about as long as the old add alone (27.6 vs 26.9 µs per call), though it moves 17% fewer
  bytes. So the saving is almost the whole convert kernel.
- At 4.2 TB/s it would save about another 51 ms per step (~0.2%, estimate). Standalone P6 shows the same pattern
  (4.18 TB/s at 5.3 M elements, 3.2–3.4 TB/s from 11.5 M up). **Cause not investigated** (§26).

**Production (W=8, B=8, GRAD_ACCUM=4): not measured.** The accumulation work per micro-step is the same, but each
micro-step does about twice the compute of B=4, so expect roughly half the percentage (~0.25–0.3%, estimate).

### 27.8 Test traps

- **nsys inside mpirun cannot finish the run.** When `--duration` ends, nsys (the rank process) exits and mpirun
  tears down the job, even with `--kill=none`. The kernel data was complete, but the end-of-run prints never came.
  The allocator counts came from separate runs without nsys.
- `BLU_MAX_STEPS=3` aborts at start ("no decay tail"). 4 is the minimum.
- The patch anchor for the include lines matched twice (a commented-out copy of the file further down), so the
  patch uses a 2-line anchor.

**Files:** `profiling_reports/f2_2026_09_28/`:
- `patch_f2.py`, `run_f2_tests.sh`, `f2_trainer_ab.sh`, `f2_evidence.sh`, `f2_exact_window.py`, `harness_alloc_print.py`;
- `tests/`, `box_final/` (the 5 pushed sources, the 2 tests, the diff, the harness diff);
- `logs/` (standalone before/after, trainer A/B 09-28, `evidence/` nsys, `alloc/`, `abab_2026_09_29/`).

---

## 28. LinearBackward: a 3-D input gave the weight no gradient (2026-09-28)

**Pushed:** in the same commits as §27. File: `src/autograd/backward/LinearBackward.cpp:57`.

**The bug:**
- For a 3-D input, the backward flattens to 2-D ("Case 2"). That block declared a new local
  `Tensor grad_weight = ...`, which shadowed the `grad_weight` declared at the top of `apply()`.
- The matmul wrote into the inner one, and "2. Gradient for Weight" returned the outer one, which was still empty.
- The engine skips empty gradients, so nothing reported it. **The weight got no gradient and never learned.**
- In since `4a62424` (2026-03-10).

**Who it hits:** `DTensor::Linear` (`dist/Tensor-Parallelism`) calls `autograd::linear`, so tensor-parallel linear
layers with a 3-D input would not train. The 1.7B trainer never reached this code, because `nn::Linear` uses
`autograd::matmul` / `addmm`.

**The fix:** assign the outer variable (one line), as Case 1 already did.

**Test** (`Tests/kernel_tests/test_linear_backward_3d.cu`): `autograd::linear` against `autograd::matmul`, in fp32
and bf16, for 2-D and 3-D input, with and without bias.

| check | before | after |
|---|---|---|
| L1: the weight has a gradient (3-D) | FAIL (none) | PASS |
| L2: weight grad vs the matmul path, bit for bit (3-D) | FAIL (no grad) | PASS, 0 mismatches |
| L3: input grad vs matmul (3-D, with and without bias) | PASS | PASS, 0 mismatches |
| L4: weight and input grad vs matmul (2-D) | PASS | PASS, 0 mismatches |
| L5: with bias: weight grad vs matmul, bias grad vs column sums | weight FAIL | PASS (bias rel err 1.42e-6 fp32, 3.62e-3 bf16) |
| **total, fp32 + bf16** | **10 pass, 6 fail** | **16 / 16** |

**Not tested:** a real tensor-parallel training run through `DTensor::Linear` (§26).

---

## 29. Everything pushed on 2026-09-28, and what colleagues must do

Commits by `Goutham Kumar <goutham.kumar@blubridge.com>` on branch `dist_side_merge`:

| repo | commits | what |
|---|---|---|
| Tensor-Implementations | `3a08ade..d96b991` | F2 (§27) + LinearBackward (§28) + 2 tests: 7 files, +773 / −1 |
| BluTrain | `4e92e6c..909ab75` | submodule pointer to `d96b991` (1 file) |
| Neural_Networks_exp_1926 | `4ef618a..a98c05c`, `a98c05c..e6109ce` | all scripts, logs, nsys evidence, box copy; the 09-29 ABAB logs |

**Colleagues must:**
1. `git pull` BluTrain, then `git submodule update --init`.
2. Rebuild with `make -j$(nproc) tensor WITH_BLUBLAS=1 SM_ARCH=<arch>`, then rebuild the trainer (`make build-mpi …`).

**Behaviour changes:** none in the numbers. F2 is on by default and bit-identical to the old path.
`DTensor::Linear` weights with a 3-D input now get gradients.

**Not committed, on purpose:** the box-only test edits in `bluscript_zero_ckpt.cpp`: `B = 4` at line 198, and the
`[ALLOC]` allocator print after the `[MEM]` line. They are saved as
`profiling_reports/f2_2026_09_28/box_final/bluscript_harness_B4_ALLOC.diff` (`git apply` it on a fresh box).

**Fresh-box notes (`dry-door-shines-fin-02`, 2026-09-29):**
- The image had **no OpenMPI**. The `dist` stage failed with `/usr/bin/mpic++: No such file or directory`, and its
  nvcc calls with `nvcc fatal : Failed to preprocess host compiler properties` (they use `-ccbin=/usr/bin/mpic++`).
  Fix: `apt-get install -y openmpi-bin libopenmpi-dev`. The libraries built before `dist` were fine.
- CUDA 13.0 here; the 09-28 box had 13.2.
- A non-interactive `ssh` has no `nvcc` on PATH: `export PATH=/usr/local/cuda/bin:$PATH`.
- Data: `flux_train_000001–3.bin` + `flux_val_000000.bin` (4 × 200 MB) in `~/gautam_k/data`, and
  `~/gautam_k/mix/{commoncrawl,code,math}` linked to it (`BLU_MIX_ROOT=~/gautam_k/mix`).

**Boxes:** `fair-face-lies-fin-02` (09-28) was closed after a final copy (all 28 run files checked by md5).
`dry-door-shines-fin-02` (09-29) ran the ABAB; its logs are in `logs/abab_2026_09_29/`.

---

## 30. Ported to `dist_side_merge_BLUCCL` (2026-09-29)

**Why:** a colleague works on `dist_side_merge_BLUCCL` (BluCCL, the in-house collective library). Before this port it
had only one of our changes: the loss fixes (library `ad10e7d`), which the colleague had merged on 09-25.

**Pushed:** author `Goutham Kumar <goutham.kumar@blubridge.com>`, direct push, no PR (the user's decision).

| repo | branch | push | contents |
|---|---|---|---|
| Tensor-Implementations | `dist_side_merge_BLUCCL` | `38d0332..b975be5` | the library changes of `dist_side_merge` after `ad10e7d`: `7810aea`, `97735ad`, `3a08ade`, `d96b991` (15 files, +1536 / −216) |
| BluTrain | `dist_side_merge_BLUCCL` | `786098a..e6eaeba` | our BluTrain changes `a264586` + `23ae1e5` (6 files), the pointer to `b975be5`, and the new `[ALLOC]` print (+250 / −20) |

### 30.1 What the branch had before

| our change | in BLUCCL before? |
|---|---|
| loss fixes A/B/C + ignore_index / denominator (library `ad10e7d`) | **yes**, merged by the colleague on 09-25 (library `38d0332`) |
| F1 + #2 + trainer catch + `[MEM]` (BluTrain `a264586`) | no |
| Bug D + conversions (`7810aea`), reductions (`97735ad`), GPU `as_type` (`3a08ade`), F2 + LinearBackward (`d96b991`) | no |
| F1 for Muon + Symmetric (BluTrain `23ae1e5`) | no |

None of our BluTrain lines had been copied by hand (checked line by line). After the fork point `03b94cf`, BluTrain
`dist_side_merge` held only our 6 commits; BLUCCL had 15 commits of its own (3 authors).

### 30.2 Method: one patch per repo

- **Library:** `git diff ad10e7d origin/dist_side_merge`, applied onto their `38d0332`. Their 5 library files
  (`DataLoader.h`, `EmbeddingKernels.h`, `EmbeddingBackward.cpp`, `EmbeddingKernels.cu`, the BluBridge-BLAS pointer)
  and our 15 do not overlap.
- **BluTrain:** `git diff 03b94cf origin/dist_side_merge -- <our 6 files>`, applied with `git apply --3way` onto
  `786098a`. 5 of the 6 files were also changed by the colleague; all applied cleanly.
- The library went first, so that BluTrain's pointer never names a commit missing on GitHub.
- **Consequence:** git does not know these are the same changes as on `dist_side_merge`. A later merge of
  `dist_side_merge` into BLUCCL sees identical code; a conflict is possible only if one of these files changes
  again before that merge.

### 30.3 Checks before committing

**`dist_side_merge` itself** (so the source of the port is right):
- all 36 files of our 6 commits match the GitHub commit diffs line for line;
- every change is still in the tip (one comment line was reworded on purpose by `23ae1e5`);
- every real change has its explanation comment. The 26 spots without a nearby comment were each reviewed: `#include`
  lines, braces, formatting-only rewraps (mse / mae / bce), or explained further up in the same function. The
  "templating spaghetti" comment in `mae_loss_forward_cuda` is older than our commits and was left as it was.

**Library commit:** the 15 staged files equal `dist_side_merge`; their files are unchanged; the total against their
branch is +1536 / −216, exactly the patch.

**BluTrain commit:**
- in the 6 files, 0 of our added lines and 0 of theirs are missing, with no duplicates;
- `[MEM]` = 1, `[ALLOC]` = 1, their `B = 2` kept; `ZeroNonFiniteGradError` = 16, the same as `dist_side_merge`;
- the pointer is `b975be5`;
- their API rename (`ProcessGroupNCCL` → `dist::comm::ProcessGroupBase`, `min` / `max` → `op_min` / `op_max`) touches
  none of our lines;
- their newer BluBLAS pointer (`ea666ab`, PR #3, ahead of our `35e51a0`) is kept.

### 30.4 H200 test (before any push)

On `dry-door-shines-fin-02` (CUDA 13.0, BluBLAS), from an rsync of the local clone:
- **Build:** clean. Libraries, `dist` (including their `processGroupBluCCL.o`) and the trainer all built; the F2 symbol
  and both prints are in the binaries.
- **5-step run of their trainer as-is:** B = 2, global_batch 524,288, so grad_accum = 64 at W = 1.
  - `exit=0`, 5 steps, finite losses (11.280 at step 0), ~41.2–41.6K tok/s;
  - `[MEM]` active 70,598 / reserved 73,328 MiB; `[ALLOC]` printed;
  - fused calls **50,083 = 157 × 319** (5 × 64 − 1 micro-steps).
- Their branch has **157** bf16 weights, not 156: its output head is bf16. G0 is 1,588,101,120 against our
  1,472,200,704, and the difference 115,900,416 = 50,304 × 2,304.
- **Not tested:** multi-GPU, the BluCCL comm paths, F1 actually firing (no NaN occurred).

**Colleague must:** `git pull`, `git submodule update --init`, then rebuild. Behaviour changes: F1 stops on a NaN/Inf
gradient with exit 4; the GPU integer mean returns Float64; complex `as_type` converts, and complex ↔ real throws.

**Files:** `profiling_reports/bluccl_port_2026_09_29/` (`ti_ours.patch`, `bt_ours.patch`, `build_bluccl.log`,
`bluccl_run/`).

---

## Appendix A — `gpu_hour.sh` (the one-hour session, md5 `e5d5c998409fd84026f4fe8c4044581c`)

```bash
#!/usr/bin/env bash
# =============================================================================
# gpu_hour.sh -- one exclusive GPU hour on cold-bear-flows-fin-02, 2026-09-24
#
#   BLOCK 1  precision test B on H200 (sm90 convert path)          ~1 min
#   BLOCK 2  clean A-B-A-B training reruns, 20 steps each           ~30 min
#            fp32 noise floor, bf16 noise floor, fp32-vs-bf16 on a clean card,
#            allocator live/reserved memory from [MEM] (no GPU poller)
#   BLOCK 3  ncu DRAM bytes on the loss path, arm 2 vs arm 3        ~10 min
#   SUMMARY  one block at the end of session.log
#
# Profiler traps honoured: no nvidia-smi polling DURING a run (only between runs),
# leaked pollers refused at preflight, profilers write to files (never piped),
# --print-units base, every artefact's line/step count printed before any number.
#
# Run:   setsid nohup bash /root/bugtest/gpu_hour.sh > /root/bugtest/gpu_hour.nohup 2>&1 < /dev/null &
# Watch: tail -f /root/bugtest/hour/session.log
# Stop:  kill -- -$(cat /root/bugtest/hour/script.pgid)     (whole process group: bash, mpirun, ncu, _runner_)
# =============================================================================
set -u
BT=/root/bugtest
cd $BT/BluTrain || exit 1
LDP=/usr/lib/x86_64-linux-gnu:$BT/BluTrain/Tensor-Implementations/lib:$BT/BluTrain/Profiler/lib:$BT/BluTrain/dist/lib
OUT=$BT/hour
mkdir -p $OUT
exec > >(tee -a $OUT/session.log) 2>&1
ps -o pgid= -p $$ | tr -d " " > $OUT/script.pgid
BIN=$BT/_runner_OURS_MEM

run_train () {   # $1=tag  $2=BLU_BF16_LOGITS  $3=BLU_MAX_STEPS  $4=BLU_VAL_STEPS  (rest = optional profiler prefix)
  local tag=$1 flag=$2 steps=$3 vsteps=$4; shift 4
  rm -rf checkpoints_bluscript
  LD_LIBRARY_PATH=$LDP:${LD_LIBRARY_PATH:-} BLU_MIX_ROOT=$BT/mix \
  BLU_BF16_LOGITS=$flag BLU_MAX_STEPS=$steps BLU_VAL_STEPS=$vsteps BLU_CKPT=0 \
  mpirun --allow-run-as-root -np 1 \
    -x LD_LIBRARY_PATH -x BLU_MIX_ROOT -x BLU_BF16_LOGITS -x BLU_MAX_STEPS -x BLU_VAL_STEPS -x BLU_CKPT \
    "$@" $BIN > $OUT/$tag.log 2>&1
  return $?
}
cotenants () { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | tr '\n' ' '; }

echo "===================== PREFLIGHT $(date '+%F %T') ====================="
FAILS=0
need () {  # $1=label $2=actual $3=test-op $4=expected
  if [ "$2" "$3" "$4" ]; then echo "  ok    $1 = $2"; else echo "  FAIL  $1 = $2   (need $3 $4)"; FAILS=$((FAILS+1)); fi
}
POLL=$(pgrep -af "nvidia-smi|nvtop|gpustat" | grep -v pgrep)
if [ -n "$POLL" ]; then
  if [ "${ALLOW_POLLERS:-0}" = 1 ]; then
    echo "  WARN  GPU pollers running -- ALLOW_POLLERS=1, continuing. They run for the whole session, so every run sees the same load:"
    echo "$POLL" | sed 's/^/          /'
  else
    echo "!! GPU pollers running (close them, or relaunch with ALLOW_POLLERS=1):"; echo "$POLL"; exit 1
  fi
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "!! GPU not empty:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1; fi
[ -x $BIN ] || { echo "!! missing $BIN"; exit 1; }
md5sum $BIN
LIB=Tensor-Implementations/lib/libtensor.so
need "our 4-param loss symbol in libtensor.so"      "$(nm -DC $LIB | grep -c 'sparse_cross_entropy_loss(OwnTensor::Tensor const&, OwnTensor::Tensor const&, long, OwnTensor::Tensor const\*)')" -ge 1
need "count_valid_targets_cuda (ours only) in lib"  "$(nm -DC $LIB | grep -c 'count_valid_targets_cuda')" -ge 1
need "[MEM] print compiled into the BINARY"         "$(strings $BIN | grep -c '\[MEM\] active_peak')" -ge 1
need "BLU_VAL_STEPS compiled into the BINARY"       "$(strings $BIN | grep -c 'BLU_VAL_STEPS')" -ge 1
need "B=4 in trainer source"                        "$(grep -cE 'int64_t B +=[ ]*4;' bluscript_zero_ckpt.cpp)" -eq 1
need "data shards visible (train)"                  "$(ls $BT/mix/commoncrawl/*train*.bin 2>/dev/null | wc -l)" -ge 1
[ $FAILS -eq 0 ] || { echo "!! $FAILS preflight check(s) failed -- NOT running anything"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader

echo; echo "===================== BLOCK 1: test B on H200  $(date +%T) ====================="
cd $BT/BluTrain/Tensor-Implementations
mkdir -p Tests/kernel_tests && cp $BT/test_cast_equivalence.cu Tests/kernel_tests/ 2>/dev/null
PLANNED=$(make -n run-cuda FILE=Tests/kernel_tests/test_cast_equivalence.cu WITH_BLUBLAS=1 2>/dev/null | grep -c "nvcc")
if [ ! -f Tests/kernel_tests/test_cast_equivalence.cu ]; then
  echo "SKIP: test file not on the box"
elif ! make -n run-cuda FILE=Tests/kernel_tests/test_cast_equivalence.cu WITH_BLUBLAS=1 >/dev/null 2>&1; then
  echo "SKIP: this tree has no run-cuda make target"
elif [ "$PLANNED" -gt 1 ]; then
  echo "SKIP: make plans $PLANNED nvcc calls -> it would rebuild the library; not spending the hour on that"
else
  make run-cuda FILE=Tests/kernel_tests/test_cast_equivalence.cu WITH_BLUBLAS=1 > $OUT/testB_h200.log 2>&1
  echo "exit $?"
  grep -E "^C[0-9]|anti-vacuity|predictions violated|=====" $OUT/testB_h200.log
fi
cd $BT/BluTrain

echo; echo "===================== BLOCK 2: clean A-B-A-B reruns ====================="
for REP in 1 2; do for ARM in 2 3; do
  [ $ARM -eq 2 ] && FLAG=0 || FLAG=1
  TAG=rep${REP}_arm${ARM}
  echo "--- $TAG  BLU_BF16_LOGITS=$FLAG  start $(date +%T)  cotenants-before: [$(cotenants)]"
  run_train $TAG $FLAG 20 20; RC=$?
  echo "    end $(date +%T)  exit $RC  cotenants-after: [$(cotenants)]"
  echo "    steps (must be 20): $(grep -c '^step' $OUT/$TAG.log)   resume lines (must be 0): $(grep -ci 'resume\|loaded DCP' $OUT/$TAG.log)"
  echo "    min tok/s steps 1-19 (>40000 = uncontended): $(grep '^step' $OUT/$TAG.log | awk 'NR>1{if(m==""||$11+0<m)m=$11+0} END{printf "%.2f", m}')"
  grep "\[MEM\]" $OUT/$TAG.log | sed 's/^/    /'
done; done

echo; echo "===================== BLOCK 3: ncu DRAM traffic ====================="
NCU=$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
echo "ncu = ${NCU:-NOT FOUND}"
if [ -n "${NCU:-}" ]; then
  for ARM in 2 3; do
    [ $ARM -eq 2 ] && FLAG=0 || FLAG=1
    echo "--- ncu_arm$ARM  start $(date +%T)"
    run_train ncu_arm${ARM}_stdout $FLAG 5 1 \
      $NCU --kernel-name regex:"sparse_ce|sparseCE|vec8_convert_to_bf16|vec4_convert_to_fp32|convert_type_kernel" \
           --launch-count 2500 \
           --metrics dram__bytes_read.sum,dram__bytes_write.sum \
           --print-units base --csv --log-file $OUT/ncu_arm${ARM}.csv
    RC=$?
    echo "    end $(date +%T)  exit $RC  csv lines $(wc -l < $OUT/ncu_arm${ARM}.csv 2>/dev/null || echo MISSING)  steps $(grep -c '^step' $OUT/ncu_arm${ARM}_stdout.log)"
  done
fi

echo; echo "===================== SUMMARY $(date '+%F %T') ====================="
echo "--- test B on H200 ---"; grep -E "predictions violated" $OUT/testB_h200.log 2>/dev/null || echo "not run"
tbl () {  # $1 = field of the step line (5 = train loss, 14 = grad norm, 11 = tok/s)
  awk -v F="$1" '
    BEGIN { n = split("rep1_arm2 rep1_arm3 rep2_arm2 rep2_arm3", names, " "); mx = -1 }
    /^step/ { k = split(FILENAME, a, "/"); b = a[k]; sub(/[.]log$/, "", b); v[$2, b] = $F; if ($2+0 > mx) mx = $2+0 }
    END { for (s = 0; s <= mx; s++) { printf "%-5s", s
            for (i = 1; i <= n; i++) printf " %12s", (((s SUBSEP names[i]) in v) ? v[s, names[i]] : "-")
            print "" } }' "${RUNLOGS[@]}"
}
RUNLOGS=(); for t in rep1_arm2 rep1_arm3 rep2_arm2 rep2_arm3; do
  if [ -s $OUT/$t.log ]; then RUNLOGS+=($OUT/$t.log); else echo "  !! $t.log MISSING or empty (that run failed) -- shown as '-'"; fi
done
[ ${#RUNLOGS[@]} -gt 0 ] || RUNLOGS=(/dev/null)   # never let awk fall back to reading stdin
echo "--- step | train loss   rep1_arm2    rep1_arm3    rep2_arm2    rep2_arm3 ---"; tbl 5
echo "--- step | grad norm    rep1_arm2    rep1_arm3    rep2_arm2    rep2_arm3 ---"; tbl 14
echo "--- step | tok/s        rep1_arm2    rep1_arm3    rep2_arm2    rep2_arm3 ---"; tbl 11
echo "--- val at step 19 ---"
for t in rep1_arm2 rep1_arm3 rep2_arm2 rep2_arm3; do echo "  $t  $( [ -f $OUT/$t.log ] && grep '\[val\] step 19' $OUT/$t.log | awk '{print $6}' || echo MISSING)"; done
echo "--- steady tok/s (trainer's own summary) ---"
for t in rep1_arm2 rep1_arm3 rep2_arm2 rep2_arm3; do echo "  $t  $( [ -f $OUT/$t.log ] && grep 'steady tok/s' $OUT/$t.log | awk '{print $NF}' || echo MISSING)"; done
echo "--- [MEM] ---"; grep -H "\[MEM\]" $OUT/rep*.log | sed "s#$OUT/##"
echo "--- GPU pollers at end (compare with preflight) ---"; pgrep -af "nvidia-smi|nvtop|gpustat" | grep -v pgrep | sed "s/^/  /"
echo "--- artefacts ---"; ls -la $OUT
echo "===================== DONE ====================="
```

## Appendix B — `parse_ncu.py`

```python
import csv, re, sys, collections
f=sys.argv[1]
lines=open(f,errors='replace').read().splitlines()
h=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
K=collections.OrderedDict()
for r in csv.DictReader(lines[h:]):
    i=int(r['ID']); k=K.setdefault(i,{'name':r['Kernel Name'],'rd':0,'wr':0})
    v=float(r['Metric Value'].replace(',',''))
    assert r['Metric Unit']=='byte', r['Metric Unit']
    if r['Metric Name']=='dram__bytes_read.sum': k['rd']=v
    elif r['Metric Name']=='dram__bytes_write.sum': k['wr']=v
def short(n):
    m=re.search(r'(\w+)(<[^()]*>)?\(',n); b=m.group(1) if m else n[:40]
    t=m.group(2) or '' if m else ''
    t=t.replace('OwnTensor::','').replace('unsigned short','u16').replace(' ','')
    return b+t
N=4*4096*50304; E2,E4=2*N,4*N
def role(k):
    s=short(k['name']); rd,wr=k['rd'],k['wr']; near=lambda x,y: abs(x-y)<0.03*y
    if 'sparse' in s or 'sparseCE' in s: return 'LOSS'
    if near(rd,E2) and near(wr,E4): return 'LOGITS bf16->fp32 convert'
    if near(rd,E4) and near(wr,E2): return 'LOGITS fp32->bf16 convert'
    return None
print(f'{f}: {len(K)} kernels parsed (expect 2500)')
c=collections.Counter(short(k['name']) for k in K.values())
print('census:', ', '.join(f'{n} x{v}' for n,v in c.most_common()))
print(f'\nlogits-path kernels, in launch order (B*T*V = {N:,} elements; 2 B/elem = {E2/1e9:.3f} GB, 4 B/elem = {E4/1e9:.3f} GB):')
print(f'{"ID":>5}  {"role":27} {"kernel":58} {"read GB":>8} {"write GB":>8}')
rows=[(i,k,role(k)) for i,k in K.items() if role(k)]
for i,k,r in rows: print(f'{i:>5}  {r:27} {short(k["name"]):58} {k["rd"]/1e9:8.3f} {k["wr"]/1e9:8.3f}')
```

## Appendix C — `watch_hour.sh` (read-only remote watcher; reads the log file, never the GPU)

```bash
#!/usr/bin/env bash
# Read-only watcher for /root/bugtest/hour/session.log on the H200 box.
# Reads the LOG FILE over ssh every 180 s. Never runs nvidia-smi, never kills, never launches.
# Exits (which wakes Claude) on: a run finishing, an anomaly, DONE, an 18-min stall, or 3 ssh failures.
# Usage: watch_hour.sh <lines-already-seen>
SEEN=${1:?usage: watch_hour.sh <lines already seen>}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new root@86.38.238.202"
fails=0; prev=-1; still=0
while true; do
  sleep 180
  full=$($SSH 'cat /root/bugtest/hour/session.log' 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then
    fails=$((fails+1)); [ $fails -ge 3 ] && { echo "EVENT=SSH_FAILED x3 at $(date +%T)"; exit 2; }
    continue
  fi
  fails=0
  n=$(printf '%s\n' "$full" | wc -l)
  if [ "$n" -eq "$prev" ]; then still=$((still+1)); else still=0; fi; prev=$n
  new=$(printf '%s\n' "$full" | tail -n +$((SEEN+1)))
  if printf '%s\n' "$new" | grep -q "===================== DONE"; then echo "EVENT=DONE SEEN=$n"; printf '%s\n' "$new"; exit 0; fi
  if printf '%s\n' "$new" | grep -qE "FAIL |CUDA OOM|misaligned|Aborted|exit [1-9]"; then echo "EVENT=ANOMALY SEEN=$n"; printf '%s\n' "$new"; exit 1; fi
  if printf '%s\n' "$new" | grep -qE "min tok/s|csv lines"; then echo "EVENT=RUN_DONE SEEN=$n"; printf '%s\n' "$new"; exit 0; fi
  if [ $still -ge 6 ]; then echo "EVENT=STALL no new log lines for 18 min, SEEN=$n"; printf '%s\n' "$full" | tail -8; exit 3; fi
done
```
