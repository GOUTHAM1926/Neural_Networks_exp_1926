# Loss-Kernel Dtype Bugs — Bug A, Bug B, Bug C, and the Redundant Cast

**Repo:** `Tensor-Implementations` · **Branch:** `sft_gautam` · **Date:** 2026-09-24
**Baseline compared against:** `a4070931` (submodule pinned by `BluTrain/dist_side_merge`)

---

## 0. TL;DR

Three bugs in the sparse cross-entropy loss path, plus one redundant memory cost in the
training script. All four are invisible today because a single line in the trainer widens
the logits to fp32 before the loss. That line costs **6.14 GB per rank** and is the only
thing preventing three separate failures.

| # | name | failure mode | fires when |
|---|------|--------------|------------|
| A | loss scalar squashed | reported loss quantized to ~0.06 | logits are bf16 |
| B | vector width hardcoded to 4 | crash **and** silent gradient corruption | logits are bf16 |
| C | grad scale read at wrong type | **all gradients become 0.0** | loss dtype != logits dtype |
| 4 | redundant fp32 upcast | 6.14 GB + 75% wasted loss-path traffic | always (today) |

**Measured:** bf16 loss error `1.565e-03 -> 1.127e-05` (**139x**), fp16 `2.007e-04 -> 8.629e-06` (**23x**).
**Calculated, NOT measured:** 6.14 GB memory, 75% bandwidth.

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
// BEFORE  (a4070931)
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

### 5.5 No code edit is needed on `dist_side_merge`

`ReshapeOps.cpp:67` — `if (input.dtype() == target) return input;` — verified.

Setting `bf16_logits = true` makes `logit_dtype()` return `Bfloat16`, the `to_dtype` call
becomes bf16 -> bf16, and **no cast node is built at all**. Item 4 is therefore
`BLU_BF16_LOGITS=1` — one env var, zero code edits.

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

The bugs sat armed since at least `a4070931` (2026-09-08), behind a branch nobody walked
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

`BluTrain/dist_side_merge` (tip `74ec046`, 2026-09-18) pins submodule **`a4070931`**
(2026-09-08, "memory fix"), *not* the `Tensor-Implementations/dist_side_merge` tip
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

> **Caveat:** the local copy of that branch is from **2026-09-18**. If the fix landed after
> that on an unfetched branch, this conclusion changes. A `git fetch` is required before
> treating it as final.

### Where their comment is RIGHT

The same block says *"GRADIENTS ARE UNAFFECTED"*. **On their branch that is correct**, for
the reason in §4.5: the loss dtype and the seed dtype agree, so the wrong-type read lines up
by coincidence. Their design is self-consistent. The zero-gradient failure is **created by
fixing Bug A**, not by their code.

---

## 10. Expected benefits

| benefit | number | status |
|---|---|---|
| Reported-loss precision (bf16) | **139x** | **MEASURED** |
| Reported-loss precision (fp16) | **23x** | **MEASURED** |
| Memory, certain | 6.14 GB / rank | calculated |
| Memory, peak | up to 12.28 GB / rank | calculated, less certain |
| Loss-path bandwidth | 36.84 -> 9.21 GB/micro-step (75%) | calculated |
| Kernel launches | 2 fewer per micro-step | certain |
| tok/s | **unknown** | no estimate offered |

Memory saved != speed gained. The loss path is one slice of the step and there is no
measurement for this script. **That is what the H200 run is for.**

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

### Arm 3 is a NUMERICAL change, not only a memory one

`BLU_BF16_LOGITS=1` genuinely gives the logits and their gradients 7 mantissa bits instead
of 23. Standard practice says that is fine; standard practice is not evidence for *our*
model. Arm 3 must show:

1. **loss curve tracks arm 1** over the same step window — not merely a similar endpoint
2. **grad-norm tracks** — where a precision problem shows first
3. *then* tok/s and memory

If the loss curve diverges, the correct outcome is to keep the cast and take the
correctness fixes only.

### Measurement discipline (from prior campaigns)

- Do **not** shorten `max_steps` to benchmark — changes warmup/LR, costs ~3.4% switching
  power = -2.46% throughput at the 700 W cap.
- Compare the **same step window** across arms.
- One config per process.
- Pull run artefacts off the box immediately.

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

## 13. Open items

- [ ] `git fetch` both repos — confirm `VEC_ELEMS` is still absent at today's tip
- [ ] Commit the work on `sft_gautam` (nothing is committed yet)
- [ ] Port items 1-5 onto `dist_side_merge`
- [ ] Run the 4-arm test on H200
- [ ] Measure the 6.14 GB / 75% claims — currently calculated only
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
