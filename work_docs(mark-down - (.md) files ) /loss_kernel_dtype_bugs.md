# Loss-Kernel Dtype Bugs — Bug A, Bug B, Bug C, and the Redundant Cast

**Repo:** `Tensor-Implementations` · **Branch:** `sft_gautam` · **Date:** 2026-09-24
**Baseline compared against:** `8808f89` (submodule pinned by `BluTrain/dist_side_merge`)
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

## 13. Open items (updated 2026-09-24 evening)

Done on 2026-09-24:
- [x] Bugs verified present at today's tip `8808f89`, on the box (§15.1)
- [x] Port onto `dist_side_merge` — 7 files, not 5 (§12, §20.3)
- [x] Run the arms on the H200 (§15, §16)
- [x] Measure memory, traffic and precision (§16)

Still open:
- [x] **Commit + push `sft_gautam`**: `1ed2ac3` on BlubridgeAI (the 7 loss files + both tests)
- [ ] **Bug D**: fix the `float_to_bfloat16` overflow at 65,536 (`Types.h:85`); it needs its own full-repo test run (§17)
- [ ] Hand the port to the `dist_side_merge` owners as a proper branch/PR, recommending `BLU_BF16_LOGITS=1`
- [ ] Rewrite the stale comment block above `loss32` in `bluscript_zero_ckpt.cpp` (§5.6)
- [ ] Remove the false `VEC_ELEMS / Chunk16` claim in the same file (§9)
- [ ] More throughput repetitions: +0.60% is n=2 per arm, and the gap to the 0.79–0.94% bytes-only prediction is unexplained (§16.6)
- [ ] Which third buffer is live at the peak (bf16 matmul output vs `ToDtypeBackward` output): memory is measured, attribution is not (§16.3)
- [ ] The 580 MiB fp32 gap between their code and ours: measured by the poller only, probably a poller artefact (§16.3)
- [ ] Bug C's consequence (zero gradients): predicted, never observed; needs a Bug-A-only build
- [ ] The source of the trainer's run-to-run non-determinism (§18)
- **Immediate future tasks — right after the SFT work, as training-script optimisations (decided 2026-09-24):**
  - [ ] **#1:** read how the embedding backward sums gradients for a bf16 weight (code, no GPU);
        then `BLU_BF16_EMBED` 0 vs 1 A-B-A-B; then a long run (§19.3)
  - [ ] **#2:** convert the gradient shard straight into `shard_grad_fp32_` — the same fix already exists in `cast_master_range` (§19.4)
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

**Where:** `include/dtype/Types.h:85`, inside `detail::float_to_bfloat16`. This is the function behind the
`bfloat16_t(float)` constructor, and therefore behind every `static_cast<T>(float)` with `T = bfloat16_t`.

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

---

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

---

## 19. Immediate future tasks: three casting follow-ups + one pending measurement (2026-09-24 — nothing changed yet)

**When:** all four are done **right after the SFT work**, as training-script optimisations.
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
| gradient sum into `wte` | **unknown.** The embedding backward adds up contributions from every occurrence of each token across B·T positions, and across micro-steps. If that sum is done in bf16, precision is lost exactly where it matters. **Not checked yet.** |

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
convert `shard_grad_` directly into `shard_grad_fp32_`. **Not done, not measured.**

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
