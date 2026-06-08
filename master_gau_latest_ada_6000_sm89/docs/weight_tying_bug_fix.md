# Weight Tying Bug — Discovery, Fix, and Verification

**Author**: Goutham Reddy (Gautam_1926)
**Status**: Closed. All gradients now match PyTorch bit-equivalently within fp32 rounding.
**Scope**: Full timeline of the `wte.weight` gradient scramble bug in GPT-2 with
weight tying enabled — from the colleague's original HANDOFF.md flagging it as
"separate bug, not yet investigated" through the per-site temporary fix, the
permanent autograd-layer fix mirroring PyTorch's Gradient Layout Contract,
and the final verification against PyTorch's gradient dumps.

---

## Table of Contents

1. [What is weight tying — full understanding from scratch](#1-what-is-weight-tying--full-understanding-from-scratch)
   - [1.1 The two tensors at the heart of GPT-2](#11-the-two-tensors-at-the-heart-of-gpt-2)
   - [1.2 The Forward Pass — how the entrance works](#12-the-forward-pass--how-the-entrance-works-embedding-lookup)
   - [1.3 The Forward Pass — how the exit works](#13-the-forward-pass--how-the-exit-works-the-weight-tying-multiplication)
   - [1.4 Why weight tying saves parameters](#14-why-weight-tying-saves-parameters)
   - [1.5 How gradient accumulation works with weight tying](#15-how-gradient-accumulation-works-with-weight-tying)
   - [1.6 Memory savings — permanent vs temporary gradients](#16-memory-savings--permanent-vs-temporary-gradients)
   - [1.7 The re-initialization quirk](#17-the-re-initialization-quirk)
2. [How weight tying was implemented across our scripts](#2-how-weight-tying-was-implemented-across-our-scripts)
3. [Why static binding fails under DDP](#3-why-static-binding-fails-under-ddp)
4. [The bug — where it originated in HANDOFF.md](#4-the-bug--where-it-originated-in-handoffmd)
5. [Root cause analysis — the three-piece chain](#5-root-cause-analysis--the-three-piece-chain)
6. [Why loss kept descending despite scrambled gradients](#6-why-loss-kept-descending-despite-scrambled-gradients)
7. [Temporary fix — `.contiguous()` at TransposeBackward](#7-temporary-fix--contiguous-at-transposebackward)
8. [Verification with gradient dumps — the proof](#8-verification-with-gradient-dumps--the-proof)
9. [Permanent fix — porting PyTorch's Gradient Layout Contract](#9-permanent-fix--porting-pytorchs-gradient-layout-contract)
10. [The fishy moment — same code, different gradients](#10-the-fishy-moment--same-code-different-gradients)
11. [The real divergence cause — missing INIT_FROM_BIN](#11-the-real-divergence-cause--missing-init_from_bin)
12. [Final verification — cos = 1.0 against PyTorch](#12-final-verification--cos--10-against-pytorch)
13. [Files changed — complete list](#13-files-changed--complete-list)
14. [Metric reference — L2 norm vs cosine vs element ratio](#14-metric-reference--l2-norm-vs-cosine-vs-element-ratio)

---

## 1. What is weight tying — full understanding from scratch

*(This section documents every doubt I had about weight tying, from the very basics to the deep internals, as I explored it on May 14, 2026.)*

### 1.1 The two tensors at the heart of GPT-2

Before I can understand what weight tying is, I need to understand the two tensors that participate in it,

**The Entrance Tensor: `wte.weight` (Word Token Embedding)**

this is a giant grid of numbers, a tensor of shape `[50304, 768]`, it has 50304 rows because there are 50304 possible words/tokens in the GPT-2 vocabulary, and each row has 768 columns which are the numbers that represent the "meaning" of that word, when I first create this tensor before training starts, all these numbers are completely random, they are random decimals like 0.014 or -0.003 drawn from a bell curve (Normal Distribution with mean=0 and std=0.02), at this point the AI knows absolutely nothing

I verified this by looking at the exact code in both our C++ script and PyTorch,

in `fullgpt.py` (PyTorch), the initialization is:

```python
def _init_weights(self, module):
    if isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

in `gpt2_fmha_ddp.cpp` (our C++ script), the initialization is:

```cpp
wte.weight = init_rng_.normal(Shape{{vocab_size, n_embd}}, 0.02f);
```

both use the same bell curve with std=0.02, but there is a subtle catch which I will explain later in Section 11 about INIT_FROM_BIN

**The Exit Tensor: `lm_head` (Language Model Head)**

this would be a tensor of shape `[768, 50304]` that maps the final hidden state back to scores over the vocabulary, without weight tying this would be a completely separate tensor with its own 38.6 million random parameters

### 1.2 The Forward Pass — how the entrance works (embedding lookup)

when training begins, I feed a batch of text into the model, let's say my batch size is 16 and my sequence length is 1024, so my input is a tensor of shape `[16, 1024]` containing integer token IDs (values from 0 to 50303),

the embedding lookup is not a matrix multiplication, it is a table lookup, each integer token ID is used as a row index into the `wte.weight` table,

```
Input token IDs [16, 1024]:
Batch 0:  [  412,  8765,    13,  ...,   290 ]   ← 1024 token IDs
Batch 1:  [ 1026,   287,  5765,  ...,    11 ]
...

Embedding Table wte.weight [50304, 768]:
Row 0:    [ 0.013, -0.007,  0.021, ... ]   ← 768 numbers for token 0
Row 1:    [-0.015,  0.003,  0.018, ... ]   ← 768 numbers for token 1
...
Row 412:  [ 0.009,  0.014, -0.002, ... ]   ← 768 numbers for token 412
...

Output [16, 1024, 768]:
For Batch 0, Position 0: copies Row 412 → [ 0.009, 0.014, -0.002, ... ]
```

so I go from `[16, 1024]` integers to `[16, 1024, 768]` floating point numbers, each word has now been converted from a single integer into a rich 768-dimensional vector

this `[16, 1024, 768]` tensor then passes through all 12 transformer blocks (attention + MLP layers), the shape stays `[16, 1024, 768]` throughout, each position's 768-number vector gets refined with context from surrounding words

### 1.3 The Forward Pass — how the exit works (the weight tying multiplication)

now I am at the end of the transformer, I have a tensor of shape `[16, 1024, 768]` where each position contains 768 numbers representing the AI's "thought" about what the next word should be, but I need the AI to give me a score for all 50304 possible next words so I can pick the highest score

this is where weight tying comes in, instead of creating a brand new separate matrix of 50304 times 768 random numbers which would waste 154MB of GPU memory, I just reuse the same `wte.weight` tensor I already have,

the math is: I take `x` (shape `[16, 1024, 768]`) and multiply it by `wte.weight` transposed,

but there is a problem, I cannot multiply `[16, 1024, 768]` by `[50304, 768]` because the shapes do not line up for matrix multiplication, the inner dimensions must match, so I flip the embedding matrix sideways using transpose, creating `w_T` which has shape `[768, 50304]`,

```cpp
// In gpt2_fmha_ddp.cpp forward():
Tensor w_T = autograd::transpose(wte.weight, 0, 1);   // [50304, 768] → [768, 50304]
logits = autograd::matmul(x, w_T);                     // [16, 1024, 768] × [768, 50304] = [16, 1024, 50304]
```

the result is `[16, 1024, 50304]`, these are the raw scores (logits) for every possible next word at every position, physically what this multiplication does is a dot product, it compares the AI's 768-number "thought" against all 50304 word definitions in the dictionary, the higher the dot product score for a particular word, the more the AI thinks that word should come next

you might ask, that means those 16384 processed words are getting compared with 50304 dictionary entries right, what does this mean physically, yes it means for every single position in every single sequence, the model instantly measures how similar its internal thought vector is to every word in its vocabulary, and the word with the highest similarity score is its best guess for the next word

### 1.4 Why weight tying saves parameters

by using weight tying, I completely eliminate the need to learn separate parameters for the exit layer,

the dictionary size is 50304 words, each word has 768 numbers, so `50304 × 768 = 38,633,472` parameters, by tying the weights I instantly remove over 38.6 million parameters from the model,

for GPT-2 124M which normally has around 162 million parameters, removing 38.6 million brings it down to 124 million, that is nearly a 25% reduction, the GPU has 38 million fewer numbers to update every step, the optimizer runs faster, I use about 154MB less GPU RAM for the weights, and the AI actually learns better because it is focusing all its learning into one master dictionary instead of splitting its attention between two

### 1.5 How gradient accumulation works with weight tying

if both the entrance (embedding lookup) and the exit (matmul with transpose) use the same `wte.weight` tensor, then during the backward pass, the gradient `dW` must accumulate contributions from both paths,

```
dW = (embedding backward path contribution) + (LM head matmul backward path contribution)
```

the embedding path produces a contiguous gradient of shape `[50304, 768]`, the LM head path computes the gradient of the transposed view so it produces a gradient of shape `[768, 50304]` which must be transposed back to `[50304, 768]` before adding to `dW`, and this transpose operation during the backward pass is exactly where the bug was !

### 1.6 Memory savings — permanent vs temporary gradients

I had another doubt here: I said the weight tying saves gradient memory too, but is that really true since both paths will create gradient tensors anyway,

here is the answer I figured out,

**without weight tying (two separate parameters):** the optimizer needs `wte.weight.grad` (154MB) and `lm_head.weight.grad` (154MB) permanently throughout training, total = 308MB of permanent gradient storage

**with weight tying (single parameter):** only `wte.weight.grad` (154MB) exists as a permanent gradient buffer, the backward pass from the exit multiplication creates a temporary gradient tensor, adds it into `wte.weight.grad`, and then the temporary tensor is freed, total = 154MB of permanent gradient storage plus a brief 154MB temporary allocation during backward

so the saving is real, I go from 308MB permanent to 154MB permanent + 154MB temporary, the optimizer state savings are even bigger because Adam keeps `m` (first moment) and `v` (second moment) per parameter, removing 38.6M parameters removes `38.6M × 4 bytes × 2` = another 296MB of optimizer state

### 1.7 The re-initialization quirk

I had yet another doubt: in `gpt2_fmha_ddp.cpp`, if weight tying is true, why do we generate new random numbers and overwrite `wte.weight` again after it was already initialized,

```cpp
if (config.weight_tying) {
    Tensor lm_head_init = init_rng_.normal(wte.weight.shape(), 0.02f);
    wte.weight.copy_(lm_head_init);
}
```

the reason is purely to match PyTorch's random number generator sequence, PyTorch has a single RNG stream and when it calls `apply(_init_weights)` it visits modules in order, first `wte`, then `wpe`, then all transformer blocks, and finally `lm_head`, each visit draws random numbers and advances the RNG, even though `lm_head` is tied to `wte`, PyTorch still draws the random numbers for `lm_head` which overwrites `wte.weight`, we replicate this exact behavior so our C++ RNG stays synchronized with PyTorch's RNG for bit-exact reproducibility

---

## 2. How weight tying was implemented across our scripts

There are three different GPT-2 scripts in our codebase that handle weight
tying differently:

### Script 1, `master_gau_latest_ada_6000_sm89/gpt2_attn_navin.cpp` — the peak benchmark script

This was our highest-throughput single-GPU script. At construction time it
binds the two weights as ALIASES:

```cpp
// gpt2_attn_navin.cpp:309 (approximate)
lm_head->weight = wte.weight.transpose(0, 1);
```

This creates a plain view (no autograd node attached, because model
construction happens outside the gradient context). During every forward pass,
`lm_head->weight` is just a transposed view of `wte.weight`, no autograd
operation runs. During backward, gradient flows directly into `wte.weight.grad`
through the normal matmul backward path, no transpose backward involved.

**Key property**: this script hardcodes `world_size = 1` at line 448. It is
single-GPU only by design. It does NOT support DDP.

### Script 2, colleague's `BluTrain_new/gpt2_fmha_ddp_chck.cpp`

The colleague's DDP-capable test script does it differently:

```cpp
// gpt2_fmha_ddp_chck.cpp:1495 (approximate)
if (config.weight_tying) {
    Tensor w_T = autograd::transpose(wte.weight, 0, 1);
    logits = autograd::matmul(x, w_T);
}
```

Notice the `autograd::transpose` call. This creates an autograd node
(`TransposeBackward`) that gets registered in the computation graph. During
backward, the LM head's matmul produces a gradient `dL/d(w_T)`, then
`TransposeBackward::apply` transposes that gradient back to wte.weight's
orientation, then the gradient flows to `wte.weight.grad` via the
GradAccumulator.

### Script 3, our current `BluTrain/gpt2_fmha_ddp_chck.cpp`

Byte-for-byte identical to the colleague's script:

```cpp
// gpt2_fmha_ddp_chck.cpp:1495 (line 438-444 in the smaller test variant)
if (config.weight_tying) {
    Tensor w_T = autograd::transpose(wte.weight, 0, 1);
    logits = autograd::matmul(x, w_T);
}
```

Same `autograd::transpose` approach. Same DDP support. Same bug surface (until
we fixed it).

---

## 3. Why static binding fails under DDP

You might ask: why can't we just use the simpler peak-script approach
(`lm_head->weight = wte.weight.transpose(0, 1)`) in the DDP script too? The
answer is that **static binding breaks under DDP**.

When DDP initializes, it iterates over the model's registered parameters and
BROADCASTS them from rank 0 to all other ranks so every GPU starts with
identical weights. The broadcast iterates over each registered parameter
tensor.

If `lm_head->weight = wte.weight.transpose(0, 1)` is set at construction time,
then `lm_head->weight` is registered as a separate parameter pointing at the
transposed view. DDP sees `lm_head->weight` as its own thing and broadcasts
it, overwriting `wte.weight`'s underlying storage with rank 0's transposed
bytes. This corrupts the tied relationship — after the broadcast,
`lm_head->weight` and `wte.weight.transpose(0, 1)` are no longer the same
memory.

Worse, the optimizer treats `lm_head->weight` as a separate parameter and
tries to update it independently, but the underlying storage is supposed to be
shared with `wte.weight`. Updates compete, the model state becomes
inconsistent across ranks, training silently breaks.

**The only correct way** to do weight tying under DDP is to keep `wte.weight`
as the single owned parameter and rebuild the transposed view every forward
pass via `autograd::transpose`. This is what both the colleague's and our DDP
scripts do. It is not optional, it is mandatory for correctness under
multi-GPU training.

This is the same approach PyTorch uses — in `gpt2.py`:

```python
self.transformer.wte.weight = self.lm_head.weight  # tied
```

PyTorch's autograd handles the transpose via its own TransposeBackward node,
and DDP only ever sees one parameter (`wte.weight`).

---

## 4. The bug — where it originated in HANDOFF.md

The colleague filed `HANDOFF.md` on May 13 documenting their gradient
investigation. The headline finding was the **reduction bug** (bias gradients
were off by shape-dependent factors like 1/17, 1/6, 1/5 due to a multi-CTA
race condition in the reduction kernel).

But at the bottom of the document, they noted a second, separate issue:

> **Not Yet Done**
>
> - [ ] Investigate the **separate `wte.weight` issue** — cpp/py ratio ≈ 1/5760
>   (essentially zero). Looks like weight tying: in C++ `GPT::forward` the
>   LM-head matmul uses `autograd::transpose(wte.weight)` and the gradient
>   is expected to flow back to the same parameter tensor. Verify that the
>   transpose backward actually adds into `wte.weight.grad` and isn't being
>   dropped/redirected.

So the wte bug was flagged as "needs investigation" — separate from the
reduction bug. The cpp/py ratio of 1/5760 means: for a typical element, the
c++ gradient value was about 1/5760 of the corresponding PyTorch gradient
value, indicating the wte gradient was essentially uncorrelated noise compared
to what PyTorch produced.

This is what we picked up and worked on starting yesterday.

---

## 5. Root cause analysis — the three-piece chain

The wte gradient bug was caused by a chain of three things that individually
look harmless but together produce catastrophic gradient corruption. We
verified each step in the chain with code proofs.

### Piece 1, `TransposeBackward` returned a non-contiguous view

In `Tensor-Implementations/src/autograd/backward/TransposeBackward.cpp:25`,
the original code was:

```cpp
return {grads[0].transpose(dim0_, dim1_)};
```

`Tensor::transpose(0, 1)` returns a VIEW into the same storage with swapped
strides. So if the incoming `grads[0]` had shape `[768, 50304]` contiguous
(strides `[50304, 1]`), the returned tensor has metadata shape `[50304, 768]`
but strides `[1, 50304]` — non-contiguous, pointing at the same underlying
physical bytes laid out as `[768, 50304]`.

### Piece 2, `AutogradMeta::accumulate_grad` adopted the view as-is

In `Tensor-Implementations/src/core/AutogradMeta.cpp:102-109`, when a new
gradient arrives for a parameter that doesn't yet have a `grad_` allocated:

```cpp
void AutogradMeta::accumulate_grad(Tensor&& update) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!grad_) {
        grad_ = std::make_unique<Tensor>(std::move(update));  // ← MOVE, no contiguity check
    } else {
        *grad_ += update;
    }
}
```

The `std::move(update)` adopts the incoming tensor as-is, including its
strides. So `wte.weight.grad`'s metadata claims `[50304, 768]` but its
underlying storage is `[768, 50304]` in physical memory, exactly mirroring
the strided view from `TransposeBackward`.

This is the second link in the chain. PyTorch's equivalent
`AccumulateGrad::apply` has a "Gradient Layout Contract" check here that
catches this exact situation and materializes a properly-strided clone
before adopting. Ours doesn't (or didn't, until we fixed it — see
[Section 9](#9-permanent-fix--porting-pytorchs-gradient-layout-contract)).

### Piece 3, the optimizer is stride-blind

In `Tensor-Implementations/src/Kernels/cuda/optimizer/arch/MultiTensorKernels_sm89.cu:193-246`,
the multi-tensor Adam kernel walks parameters and gradients with flat linear
indexing:

```cpp
float* p = meta.params[tensor_idx];   // raw float*
float* g = meta.grads[tensor_idx];    // raw float*
...
for (int64_t i = vec_start + threadIdx.x * 4; i < vec_end; i += blockDim.x * 4) {
    float4* g4 = (float4*)(&g[i]);
    float4* p4 = (float4*)(&p[i]);
    ...
    p_out_j = pj - lr * (...);   // updates p[i] using g[i]
}
```

The kernel takes raw float pointers and walks them with flat linear index `i`,
assuming both `p` and `g` are contiguous row-major and that `g[i]` is the
gradient of `p[i]`. **No stride metadata. No layout check.**

### Putting it together — the scramble

For `wte.weight` (contiguous `[50304, 768]`, strides `[768, 1]`):

- `p[k]` is the logical entry at `(row = k/768, col = k%768)`

For the strided gradient storage S (physically `[768, 50304]` from
TransposeBackward):

- `g[k] = S[k]` is the gradient of the logical entry at `(row' = k%50304, col' = k/50304)`

So when the optimizer does `p[k] -= lr * g[k]`:

- It applies the gradient computed for word `(k%50304, k/50304)`
- To the parameter of word `(k/768, k%768)`

These row/col coordinates do not match for almost any `k`. The optimizer is
literally taking the gradient of word X and applying it to the weights of
word Y, scrambled deterministically by the permutation `k -> (k%50304, k/50304)`.

This is a **deterministic per-element permutation**, not random noise. Every
training step applies the same wrong mapping, scrambling 38 million parameter
positions.

---

## 6. Why loss kept descending despite scrambled gradients

If the gradient is scrambled per-element, you'd expect the loss to diverge
catastrophically. But it doesn't. The loss descends almost normally. Why?

Three reasons:

### Reason 1, the scramble is a fixed permutation of similar-magnitude values

All 50,304 wte rows start from the same `N(0, 0.02²)` initialization. So
their gradients have similar magnitudes. When the optimizer pulls the
gradient of word "Apple" and applies it to the weights of word "Car", the
magnitude is roughly right (both rows have ~0.02-scale gradients), just the
direction is wrong.

You're injecting wrong-direction updates of the right scale, not unbounded
noise. The grad norm stays at ~1-2 instead of exploding.

### Reason 2, wrong updates random-walk instead of compounding

Across many training steps, the scrambled gradient signals don't align in
any consistent direction. Row "Apple" gets row "Car"'s gradient one step,
then row "Banana"'s gradient another step, then row "Dog"'s gradient another.
These don't accumulate consistently, they random-walk around the original
initialization.

So `wte.weight` stays near its initial values, drifting around `N(0, 0.02²)`
which is harmless — it just doesn't actually learn useful embeddings.

### Reason 3, wte is only 31% of model params

GPT-2 124M has 124.5M total params. wte+lm_head (which are the same memory)
account for 50304 × 768 = 38.6M params, about 31% of the total.

The other 69% (12 attention blocks, 12 MLP blocks, LayerNorms, output norm)
train COMPLETELY CORRECTLY because their gradient paths don't go through
TransposeBackward. The attention layers, MLP layers, and norm layers all
work fine and carry the loss curve down.

The model effectively trains as if wte+lm_head were a frozen randomly-init
lookup table while everything else learns around it. The downstream layers
compensate. Loss descends. Validation loss looks OK. Token generation
produces sensible-looking text for thousands of steps.

The bug is **silent and invisible from training metrics alone**. The only
way to spot it is to bit-compare gradients against PyTorch (which we did).

---

## 7. Temporary fix — `.contiguous()` at TransposeBackward

The simplest fix is to materialize the strided view inside TransposeBackward
itself, before it ever reaches `AutogradMeta::accumulate_grad`.

In `Tensor-Implementations/src/autograd/backward/TransposeBackward.cpp:25`:

```cpp
// BEFORE (buggy):
return {grads[0].transpose(dim0_, dim1_)};

// AFTER (fix v1):
return {grads[0].transpose(dim0_, dim1_).contiguous()};
```

The `.contiguous()` call allocates fresh row-major storage of shape
`[50304, 768]` and copies the strided view's logical contents into it via
the `contiguous_strided_copy_cuda` kernel. The result is a properly
row-major tensor with strides `[768, 1]`.

When this contiguous tensor reaches `accumulate_grad` via `std::move`, the
adopted `grad_` is now correctly laid out. The optimizer's flat-indexing
walk then correctly pairs `p[k]` with the gradient of the corresponding
logical entry. No scramble.

**This is what the colleague had done** in their `BluTrain_new` tree as their
working fix, which is why their May 15 dumps (`cpp_grad_dumps_transpose_bckward/`)
showed cos = 1.0 vs PyTorch for wte.

This is a per-site fix — it works for the one site (TransposeBackward) but
provides no protection against any future backward op that might also return
a strided view. The right structural fix lives at AutogradMeta, which we
implemented next (see [Section 9](#9-permanent-fix--porting-pytorchs-gradient-layout-contract)).

---

## 8. Verification with gradient dumps — the proof

To verify any fix, we used the gradient dump pipeline the colleague had set
up. The flow is:

1. Run the C++ trainer with `GRAD_DUMP_DIR=<path>` env var
2. After every step's backward + DDP sync, dump every parameter's gradient
   as raw float32 to `<dir>/step_<NNNNNN>_<param_name>.bin`
3. On the PyTorch side, run `gpt2.py` with the same input data and dump
   gradients to a parallel directory with the same file naming
4. Run `compare_grad_dumps.py` to compute per-parameter cosine similarity,
   L2 norm ratio, and per-element ratio between C++ and PyTorch values

The script handles the C++ `[in, out]` vs PyTorch `[out, in]` Linear weight
convention by transposing the C++ side before comparison.

### Before fix dumps showed the scramble fingerprint

For `wte.weight` at step 0:

| metric | value | interpretation |
|---|---|---|
| L2 norm ratio | 1.0006 | gradient magnitudes match PyTorch |
| cosine similarity | 0.001 | direction is essentially random |
| median element ratio | 1/3472 to 1/5760 | typical element values are thousands × apart |

The colleague reported 1/5760 in HANDOFF.md, we measured 1/3472 in our run.
Both are in the same regime — essentially uncorrelated values.

### After the per-site `.contiguous()` fix

For `wte.weight` at step 0:

| metric | value |
|---|---|
| L2 norm ratio | 1.000061 |
| cosine similarity | 1.000000 |
| median element ratio | 0.999899 |
| MAE / mean(\|py\|) | 0.000163 (was 1.68 before fix) |

10,000× improvement on per-element accuracy. Bit-equivalent to PyTorch
within fp32 rounding noise.

Across all 148 parameters at both step 0 and step 1: **148/148 had cos > 0.9999**.

---

## 9. Permanent fix — porting PyTorch's Gradient Layout Contract

The `.contiguous()` per-site fix is correct but fragile. Any future backward
op that returns a strided view (e.g., a hypothetical `PermuteBackward`,
`SelectBackward`, etc.) would silently hit the same scramble. The structural
fix is to enforce the layout contract at the **autograd boundary** itself,
mirroring what PyTorch does.

### PyTorch's pattern

PyTorch's `torch/csrc/autograd/functions/accumulate_grad.h:179-226` does
this in three pieces:

```cpp
if (!variable_grad.defined()) {
    if (... && utils::obeys_layout_contract(new_grad, variable)) {
        update_grad(new_grad.detach());                            // safe to steal
    } else if (... sparse handling ...) {
        ...
    } else {
        // Case 1.5: Deep copies new_grad according to the "Gradient
        // Layout Contract."
        update_grad(utils::clone_obey_contract(new_grad, variable));   // FORCE LAYOUT
    }
}
```

And `obeys_layout_contract` in `grad_layout_contract.h`:

```cpp
inline bool obeys_layout_contract(const at::Tensor& grad, const at::Tensor& variable) {
    if (variable.is_non_overlapping_and_dense()) {
        for (idx in range(ndim)) {
            if (grad_sizes[idx] != 1) {
                if (grad_strides[idx] != variable_strides[idx]) return false;
            } else {
                if (grad_strides[idx] == 0) return false;
            }
        }
        return true;
    }
    return grad.is_contiguous(at::MemoryFormat::Contiguous);
}
```

And `clone_obey_contract`:

```cpp
inline at::Tensor clone_obey_contract(const at::Tensor& new_grad, const at::Tensor& variable) {
    return new_empty_strided_symint(variable.sym_sizes(), variable.sym_strides(), ...).copy_(new_grad);
}
```

### Our port

In `Tensor-Implementations/src/core/AutogradMeta.cpp:102-149`:

```cpp
void AutogradMeta::accumulate_grad(Tensor&& update) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!grad_) {
        // Gradient Layout Contract — mirrors PyTorch's AccumulateGrad Case 1.5
        // ...
        if (update.is_contiguous()) {
            grad_ = std::make_unique<Tensor>(std::move(update));
        } else {
            grad_ = std::make_unique<Tensor>(update.contiguous());
        }
    } else {
        *grad_ += update;
    }
}
```

Our version is the **MINIMAL** port because every parameter in our codebase
is rowmajor contiguous, so the contract reduces to "the stashed grad must
also be contiguous." We added a `TODO(Gautam_1926)` block listing 6 PyTorch
cases we intentionally skipped (sparse, MKLDNN, channels_last, double-backward,
ref-count safety, CHECK_RESULT tripwire) because the corresponding codepaths
don't exist in our library today. Each TODO has a precise trigger condition
for when to revisit.

### Reverting the per-site `.contiguous()`

With the contract in place at the autograd boundary, the per-site fix in
TransposeBackward becomes redundant — the contract will catch it. So we
reverted TransposeBackward back to the simple form and documented the
version history in code comments:

```cpp
// History — Gautam_1926:
//   v1 (buggy):       return {grads[0].transpose(dim0_, dim1_)};
//   v2 (workaround):  return {grads[0].transpose(dim0_, dim1_).contiguous()};
//   v3 (current):     return {grads[0].transpose(dim0_, dim1_)};
//                     ... structural fix at AutogradMeta handles materialization ...
return {grads[0].transpose(dim0_, dim1_)};
```

Both versions (v2 per-site and v3 contract) produce the same end behavior
(one materialization per non-contiguous gradient), v3 is just structurally
cleaner — protects future view-returning backwards automatically.

---

## 10. The fishy moment — same code, different gradients

After implementing the autograd contract, we ran the test and expected
cos = 1.0 vs PyTorch. Instead we got cos = -0.026 across all 148 parameters.
Worse, even reverting to the per-site `.contiguous()` fix (the colleague's
exact working code) ALSO produced cos = -0.026 on our system, while it
produced cos = 1.0 on the colleague's system.

You correctly observed "something is fishy bro" and asked to investigate
deeper.

We spent the entire day going through every diverging file between our
`BluTrain/` tree and the colleague's `BluTrain_new/` tree, organized into
4 tiers:

| tier | description | result |
|---|---|---|
| tier-1 | 10 small-diff files | mostly cosmetic, only sm89 attention IEEE rounding mattered |
| tier-2 | 5 medium-diff files | all cosmetic/dead-code |
| tier-3 | 4 large-diff files | all perf/refactor, no gradient impact |
| tier-4 | 5 massive-diff files | sm89 attention had real changes (IEEE rounding), GenMatmul had cuBLASLt optimization |

We synced all the substantive differences. Made the IEEE 754
round-to-nearest-even rounding fix in attention kernels (replacing the
biased round-half-up `+= 0x1000u` with proper `(b + 0x0FFFu + lsb) & 0xFFFFE000u`).
Even removed and re-added cuBLASLt for testing.

After all that work, cosine was still -0.026. We had exhausted every
identifiable file-level diff and were stuck.

---

## 11. The real divergence cause — missing `INIT_FROM_BIN`

You said "something is fishy" again and asked me to check the colleague's
dump folders directly. We compared

- the colleague's old dumps (`cpp_grad_dumps`, `cpp_grad_dumps_new`, May 13)
  which had cos = 0.001 vs PyTorch (broken state, like ours)
- the colleague's recent dumps (`cpp_grad_dumps_transpose_bckward`, May 15)
  which had cos = 1.0 vs PyTorch (fixed state)

Then we looked at his log file `BluTrain_new/may14_full_grad_check.txt` and
found this critical line near the top:

```
[init-load] loaded 148 / 148 tensors (124475904 floats, 474 MiB) from /mnt/.../gpt2_init.bin
[DIAG] wte.weight first 8: 0.013238 0.021739 -0.019373 -0.010006 0.057962 -0.021071 0.003657 0.008895
```

His run was loading the model's INITIAL WEIGHTS from a binary file called
`gpt2_init.bin` that was dumped from PyTorch. Our runs were using the C++
random initializer (`std::mt19937` with seed 1234).

### Why the seeds give different results

Both PyTorch and C++ used "seed 1234", but the underlying PRNGs are
completely different algorithms:

- **C++ `std::mt19937`** — Mersenne Twister (standard C++)
- **PyTorch `torch.Generator`** — Philox-based (custom)

Same seed, different algorithm, completely different random sequences from
step 0 onward. Initial weights were nowhere near PyTorch's initial weights.

### The script already supported INIT_FROM_BIN

The chck script's main function had this code path at line 1026:

```cpp
if (const char* p = std::getenv("INIT_FROM_BIN")) {
    if (p[0] != '\0') {
        try {
            load_init_bin(model, p, is_master);
        } catch (...) {
            ...
        }
    }
}
```

The script CHECKS for the `INIT_FROM_BIN` env var and if set, loads the
weights from the file path provided, OVERWRITING the C++ random init.

We had simply never set this env var in any of our test commands. So all
our runs used C++ random weights = different starting point from PyTorch =
different gradients = cos = 0 (because gradients are computed from
completely different initial weights, comparing them is apples vs oranges).

### Copying gpt2_init.bin and rerunning

```bash
cp /mnt/volgrp03/3rd_floor/final_merge_conflict_fix/BluTrain/gpt2_init.bin \
   /mnt/volgrp03/3rd_floor/Gautam_new/BluTrain/gpt2_init.bin

CUDA_VISIBLE_DEVICES=7 \
USE_PACKED_SDPA=1 \
INIT_FROM_BIN=/mnt/volgrp03/3rd_floor/Gautam_new/BluTrain/gpt2_init.bin \
GRAD_DUMP_DIR=cpp_grad_dumps_init_match \
GRAD_DUMP_UNTIL_STEP=2 \
make run-mpi FILE=gpt2_fmha_ddp_chck.cpp NP=1
```

Result: cos = 1.0 vs PyTorch across all 148 parameters. The "divergence" we
chased all day was a missing env var, not a code bug. All our actual code
changes were correct improvements (IEEE rounding, autograd contract, etc.)
the whole time.

---

## 12. Final verification — cos = 1.0 against PyTorch

After enabling `INIT_FROM_BIN`, all our changes produce gradients
bit-equivalent to PyTorch:

### Step 0 results

| param | C++ cos vs PY | C++ L2_ratio vs PY |
|---|---|---|
| wte.weight | 0.99999994 | 1.000061 |
| wpe.weight | 1.000000 | 1.000004 |
| All 12 layer params (attn, mlp, ln) | > 0.9999 | within 1e-5 of 1.0 |
| ln_f.weight, ln_f.bias | 1.000000 | within 1e-6 of 1.0 |

**148 / 148 parameters** at step 0 with cos > 0.9999.

### Step 1 results

After one optimizer step, slight accumulated fp32 rounding shows up:

| param | C++ cos vs PY |
|---|---|
| Best params (wte, embedding-adjacent) | 1.000000 |
| Worst params (h.11.mlp.c_proj.weight) | 0.999950 |

**148 / 148 parameters** at step 1 with cos > 0.9999. The minor dip at
step 1 is pure fp32 noise from accumulated operations across the optimizer
step + new forward + new backward.

### Initial loss confirmation

| | C++ random init (broken) | PyTorch init (fixed) |
|---|---|---|
| validation loss | 10.9968 | 10.9918 |
| step 0 train loss | 10.995877 | 10.993998 |
| step 0 grad norm | 15.7494 | 14.6707 |
| step 1 train loss | 10.206326 | 10.081621 |
| step 1 grad norm | 8.3845 | 5.2689 |

Loss dropped, grad norms dropped, all because PyTorch's init produces
slightly better starting weights for this architecture (closer to the
theoretical baseline `-ln(1/vocab) ≈ 10.825`).

---

## 13. Files changed — complete list

Tracking every file modified during the weight-tying bug fix work:

### Core autograd

| file | change | status |
|---|---|---|
| `Tensor-Implementations/src/autograd/backward/TransposeBackward.cpp` | added `.contiguous()` (v2), reverted to simple form (v3) once contract was in place | v3 active |
| `Tensor-Implementations/src/core/AutogradMeta.cpp` | added Gradient Layout Contract check, 6-case TODO block for future PyTorch parity | active |

### Optimizer / clip

| file | change | status |
|---|---|---|
| `Tensor-Implementations/src/core/TensorImpl.cpp` | replaced hardcoded `fill<float>(0.0f)` in `zero_grad` with `dispatch_by_dtype` (catches future fp16/bf16 cpu params) | active |
| `Tensor-Implementations/src/nn/optimizer/Optim.cpp` | removed dead `clip_grad_norm_async_` function definition | active |
| `Tensor-Implementations/include/nn/optimizer/Optim.h` | removed dead `clip_grad_norm_async_` declaration + comment block | active |

### Attention IEEE 754 rounding (discovered during the diagnostic)

| file | change | status |
|---|---|---|
| `Tensor-Implementations/src/Kernels/cuda/attention/AttentionForward.cu` | replaced biased `+= 0x1000u` with IEEE 754 round-half-to-even | active |
| `Tensor-Implementations/src/Kernels/cuda/attention/AttentionBackward.cu` | same IEEE rounding fix | active |
| `Tensor-Implementations/src/Kernels/cuda/attention/arch/AttentionForward_sm89.cu` | synced colleague's version with proper rounding + cleaner kernel | active |
| `Tensor-Implementations/src/Kernels/cuda/attention/arch/AttentionBackward_sm89.cu` | synced colleague's version with proper rounding + `exp2f → expf` for softmax bwd | active |

### Allocator

| file | change | status |
|---|---|---|
| `Tensor-Implementations/include/device/SizeClass.h` | removed unused `MB_1_5` and `MB_9` size class buckets, simpler bucket hierarchy | active |

### Matmul (the tug-of-war)

| file | change | status |
|---|---|---|
| `Tensor-Implementations/src/Kernels/cuda/matmul/GenMatmul.cu` | replaced with colleague's simpler version during diagnostic, then re-added our cuBLASLt EPILOGUE_BIAS fused path (sprint §17a) after confirming it's bit-equivalent to plain cuBLAS | active with cuBLASLt re-added |

---

## 14. Metric reference — L2 norm vs cosine vs element ratio

Throughout this investigation we used three metrics to compare gradient
tensors. They measure DIFFERENT things and you need them together to
understand what kind of bug you have.

### L2 norm ratio

```
L2_ratio = L2_norm(our_gradient) / L2_norm(pytorch_gradient)
```

Measures the SIZE/MAGNITUDE of the gradient relative to PyTorch.

- value = 1.0 → same magnitude
- value < 1.0 → our gradient is smaller (under-scale bug, e.g., bias reduction)
- value > 1.0 → our gradient is bigger (over-scale bug, e.g., double-counting)

### Cosine similarity

```
cosine = dot(our_gradient, pytorch_gradient) / (L2(our) × L2(py))
```

Measures the DIRECTION alignment between the two gradients as vectors.

- value = 1.0 → same direction (proportional)
- value = 0.0 → perpendicular (uncorrelated, random)
- value = -1.0 → opposite direction (sign flip)

### Per-element ratio

```
elem_ratio = median over i of (our_gradient[i] / pytorch_gradient[i])
```

For each element pair, what's the typical scaling? Median across all elements.

- value = 1.0 → each element matches
- value < 1.0 → typical element is smaller than PyTorch's
- value ≈ 1 / large_N → element values bear no relationship to PyTorch's

### The interpretation table — combinations

| L2 ratio | cosine | element ratio | what it means | example bug |
|---|---|---|---|---|
| ≈ 1.0 | ≈ 1.0 | ≈ 1.0 | perfect match | correct code |
| < 1.0 | ≈ 1.0 | ≈ same as L2 ratio | uniform under-scale, right direction wrong magnitude | bias reduction bug (each elem × 1/17) |
| > 1.0 | ≈ 1.0 | ≈ same as L2 ratio | uniform over-scale, right direction too big | double-counting bug |
| ≈ 1.0 | ≈ 0.0 | random small number | same total magnitude, wrong direction | wte scramble bug (elements permuted) |
| ≈ 0.0 | ≈ 0.0 | ≈ 0.0 | gradient is essentially zero | dead path, no gradient flowing |
| ≈ 1.0 | ≈ -1.0 | ≈ -1.0 | same magnitude, flipped sign | sign flip bug |

### Our two bugs had different fingerprints

**Weight tying bug (wte.weight)** = scramble fingerprint:

- L2 norm ratio: 1.0006 (matches PyTorch magnitude)
- cosine: 0.001 (random direction)
- element ratio: 1/3472 to 1/5760 (per-element values bear no relationship)

The 1/5760 number means: for a typical element, our cpp value was about
5760× smaller than PyTorch's corresponding value — but this is because the
elements were paired RANDOMLY by the scramble. Some elements were 100×
larger, some 100,000× smaller, the median of these random pairings happened
to be around 1/5760.

**Reduction bug (biases)** = uniform under-scale fingerprint:

- L2 norm ratio: 0.06 / 0.17 / 0.20 depending on layer (= 1/cpo for each shape)
- cosine: 1.0 (same direction, just shrunk)
- element ratio: matches L2 ratio (each element uniformly scaled by same factor)

The 1/17 number means: each element was exactly 17× smaller than
PyTorch's, uniformly. No scramble, just consistent under-counting because
only 1 of 17 CTAs' partial work survived the race.

### After both fixes

Both bugs now show:

- L2 norm ratio ≈ 1.000003 (within fp32 noise)
- cosine ≈ 1.000000 (perfect direction match)
- element ratio ≈ 1.000000 (each element matches)

Gradients are bit-equivalent to PyTorch within float32 rounding precision.

---

## Summary

The wte.weight scramble bug was a 3-piece chain:

1. `TransposeBackward` returned a strided view (the `transpose()` is a metadata-only view by design)
2. `AutogradMeta::accumulate_grad` adopted that view as `wte.weight.grad` without re-layout (the missing layout check)
3. The stride-blind multi-tensor Adam kernel walked `param[i]` and `grad[i]` as flat arrays, scrambling each weight update by a fixed permutation

Loss kept descending despite this because the scramble is a permutation
(not random noise) of similar-magnitude values, wte+lm_head is only 31% of
params, the remaining 69% trained correctly and carried the loss curve down.
Only bit-comparison against PyTorch gradient dumps could surface the bug.

Two fixes were applied:

- **v2 per-site fix** in TransposeBackward: append `.contiguous()` to
  materialize the view before it leaves the backward op
- **v3 structural fix** in AutogradMeta: enforce a Gradient Layout Contract
  at the autograd boundary, mirroring PyTorch's pattern, materialize any
  non-contiguous gradient before adopting

V3 is the active state. V2 was reverted because the contract catches the same
case more generally. Future view-returning backwards are automatically
protected, the optimizer never sees a stride-mismatched gradient.

Verified bit-equivalence with PyTorch via gradient dump comparison: all 148
parameters at both step 0 and step 1 have cosine > 0.9999 and L2 norm ratio
within fp32 noise of 1.0. The wte canary specifically went from cos = -0.026
(before INIT_FROM_BIN) to cos = 0.99999994 (matching PyTorch within fp32 noise).

The investigation also uncovered three other valuable improvements applied in
the same sweep: IEEE 754 round-to-nearest-even TF32 rounding in attention
kernels (replacing a biased round-half-up), dead `clip_grad_norm_async_` code
removal, and a dtype-dispatch fix in `TensorImpl::zero_grad` for future
fp16/bf16 CPU paths.
