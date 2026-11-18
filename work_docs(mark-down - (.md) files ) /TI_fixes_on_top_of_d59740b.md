# CP DPO: two silent TI bugs. How we found them, your fix, our fix

TI branch `madhu_post-training`, commit `c61ea5d` (your `d59740b` is included as-is).

---

## 1. How we found them (CP DPO trainer, 1.7B SFT model, T = 4096)

The CP library takes **fp32** Q/K/V, so the trainer casts `q_bf`, `k_bf`, `v_bf` to fp32 before the ring.

**First CP run: the step-0 check failed**
- Loss was **5.67** instead of ln 2 = 0.693. Log-probs were about **−13.5** instead of the reference **−1.3**.
- Traced to **V**. `v_bf = transpose(reshape(v))` is a **view**, and `Tensor::as_type` copies `numel` elements in flat memory order and **ignores strides** (`AsTypeTensor.cpp`, both the CUDA and the CPU path). V came out scrambled.
- → **Bug B: casting a view scrambles it.** This hits forward `to_dtype` **and** `ToDtypeBackward`.

**Second run (`autograd::contiguous()` added before every cast, to fix V): the loss passed (0.694), but `wK` and `k_gamma` gradients were exactly 0**
- Traced to **K**:
```
k_bf = output #1 of fused RoPE (grad_fn = RoPE, output_nr = 1)
autograd::contiguous(k_bf)  ->  Tensor::contiguous() returns *this (already tidy)
make_unary_op stamps ContiguousBackward on that SAME tensor, output_nr stays 1
ToDtypeBackward sends dK to (ContiguousBackward, slot 1)  ->  that node has only slot 0
ContiguousBackward reads slot 0, finds nothing  ->  dK dropped, no error
```
- `q_bf` was fine **by luck** (output 0). `v_bf` was fine (a view, so `contiguous` made a new tensor).
- → **Bug A: the autograd wrapper rewrites its input's `grad_fn` and keeps a stale `output_nr`.**

**You found Bug A independently** in SFT CP=2: `k_gamma` gradient 0, step-0 gradient norm 2.591 vs 2.800.

---

## 2. How you solved it: `d59740b`
- `autograd::contiguous` returns an already-tidy input **as-is**, with no new node (`ReshapeOps.cpp:64`). Same rule as `to_dtype`'s same-dtype no-op.
- ✅ Correct, and it removes the trigger for `contiguous`.
- Still open after it:
  - the **wrapper itself**: any other op whose forward returns its input hits the same drop;
  - a wrong slot is still **silent**;
  - **Bug B** (`as_type` strides) is not covered.

## 3. How we solved it (on top of yours)
| Layer | What | File |
|---|---|---|
| **Root rule** | In `make_unary_op` / `make_binary_op`: if `result` is the input's `TensorImpl`, then `result = result.detach()` (new impl, same storage, no copy). Then `set_grad_fn` + **`set_output_nr(0)`**. An op can never rewrite its input. | `ops_template.h:61-68`, `:125-129` |
| **Alarm** | `Node::num_outputs_` (default 1), raised automatically by `AutogradMeta::set_output_nr`. Both engines **throw** on a slot ≥ `num_outputs` (after cleaning up their state) instead of dropping it. | `Node.h`, `AutogradMeta.cpp:61-68`, `Engine.cpp:222-245` |
| **Bug B** | `as_type`: `if (!is_contiguous()) return contiguous().as_type(t);` This fixes both forward and backward. Contiguous slices keep the fast path (`data()` applies the offset). | `AsTypeTensor.cpp:17-24` |
| **Test** | `test_autograd_alias_slots.cu`: pass/fail, exits 1 on failure | `Tests/kernel_tests/` |

**Why this lasts:**
- The fix is at the source, so future ops are covered too.
- Any similar mistake now fails **loudly** with the node names, instead of silently training with a zero gradient.
- The CP trainer's workaround is removed (plain casts again).

---

## 4. Results (8× H200)
- **Our test: ALL PASS.**
  - Gradients on RoPE output #1 are identical with and without `contiguous` / a same-dtype `to_dtype` / a same-shape `reshape` (rel diff 0.00).
  - The alarm fires in both engines.
  - `as_type` on a view is bit-exact on CPU and GPU.
- **Your test:** A = B → q 238.746359 | k 156.991672 | q_gamma 502.862202 | k_gamma 350.833621.

**CP DPO trainer, workaround removed:**

| Check | Result |
|---|---|
| Step-0 loss diff vs ln 2: golden / CP1 / CP2 / CP2-contiguous | 1.9e-9 / 7.6e-8 / 4.45e-7 / 1.36e-7 (**same as before, to every digit**) |
| Gradient direction vs CP1: CP2 / CP2-contiguous | 2.79% / 0.92% (kernel-swap baseline 4.80%) |
| `wK` / `k_gamma` under CP2 | 2.95% / 2.65% (present; a dropped gradient shows about 100%) |
| 200 steps, CP2 vs golden (20-step windows) | loss within 0.002, margin within 0.005 |
| Weights on both CP ranks after 200 steps | identical (`d07f61385d4dad9f`) |
| Peak memory per GPU | 61.7 GiB (CP2) vs 85.7 GiB (golden) |

---

## 5. Open, your call
1. **`autograd::add_`** (`BinaryOps.cpp:19-51`) has the same stale `output_nr` pattern. One-line fix: `a.set_output_nr(0)`. No callers today.
2. **The version counter is never bumped** (`increment_version` has no callers). In-place changes to saved tensors go undetected.
3. **Your SFT CP code:** check any `to_dtype` of a **view**. Without Bug B's fix, the values are scrambled.
4. **Small items in `c61ea5d`:**
   - The test's line 90 needs `autograd::Node` → `Node` (not committed yet).
   - The commit also moved `BluBridge-BLAS` to `e392dbb`.
