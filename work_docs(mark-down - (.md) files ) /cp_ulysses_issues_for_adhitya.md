# Context_Parallelism: two problems in UlyssesAttention.h, and one note for SFT

I went through the Ulysses part of the Context_Parallelism repository because I want to use `ulysses_combine` and
`ulysses_partition` around my SFT attention kernels. I found two problems in `context_parallel/UlyssesAttention.h`
and one point that matters only if the CP trainer is used for SFT. All line numbers are from `main` at commit
`73b3bdd`. The same code is in `cf1372d`, the commit BluTrain points to.

| # | where | what | effect today |
| --- | --- | --- | --- |
| 1 | `ulysses_gather_seq`, lines 104-131 | element size hard-coded as 4 bytes (float); one copy per (rank, batch, head) plus a full stream sync | wrong data and out-of-bounds copies for bf16 / fp16 tensors; slow and blocks the CPU for any dtype |
| 2 | `head_repeat_interleave`, lines 143-153, and `head_group_reduce`, lines 160-169 | the GQA KV-head repeat is built from H x r `narrow` views + `cat`; the backward from r slice copies + r - 1 adds | extra work and memory on every GQA Ulysses call |
| 3 | `Scripts/Blutrain/bluscriptCP.cpp`, loss | mean loss per rank, then the average over ranks | correct for pretraining; uneven token weighting for SFT |

## 1. ulysses_gather_seq copies the wrong size for bf16, and copies slice by slice with a sync

The function (lines 104-131):

```cpp
pg->all_gather(local.data<float>(), gathered_flat.data<float>(), local_count, local.dtype(), true);
Tensor full = Tensor::empty(Shape({{B, H, T, D}}), local.opts());
const size_t slice_bytes = static_cast<size_t>(Tl * D) * sizeof(float);          // line 117
for (int r = 0; r < P; ++r)
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h) {
      float *src = gathered_flat.data<float>() + r * (B*H*Tl*D) + b * (H*Tl*D) + h * (Tl*D);
      float *dst = full.data<float>() + b * (H*T*D) + h * (T*D) + r * (Tl*D);
      cudaMemcpyAsync(dst, src, slice_bytes, cudaMemcpyDeviceToDevice, 0);       // line 125
    }
cudaStreamSynchronize(0);                                                        // line 129
```

**Problem 1a: the element size is fixed at 4 bytes.** The pointers are `float*`, so every offset is counted in 4-byte
steps, and `slice_bytes` uses `sizeof(float)`. In Tensor-Implementations, `data<T>()` is a plain `reinterpret_cast`
with no dtype check (`include/core/TensorImpl.h`, lines 181-191). So for a bf16 or fp16 tensor (2 bytes per element):

- every source and destination offset is twice as far as it should be;
- every copy is twice as long as one slice;
- the copies read and write past the end of `gathered_flat` and `full`, and the slices land in the wrong places;
- nothing reports an error.

**Why the current runs are not hit.** On the current paths, the tensors that reach this function are fp32:

- the fused path converts its bf16 kernel output to fp32 before `ulysses_partition`
  (`ContextParallel.h` line 1948: `Tensor out_g = out_bf.as_type(Dtype::Float32);`);
- the parity test `Tests/cp_ulysses_parity.cpp` uses Float32.

So converting to fp32 hides the problem for now. It also costs a full extra copy of the attention output and twice the
communication bytes. It breaks as soon as a bf16 tensor reaches the function: for example, if that fp32 conversion is
removed, or if a caller keeps bf16 activations. My SFT kernels keep bf16 end to end, so it would hit my use directly.

**Where the function runs:**
- forward, when `unshard` is true: `ContextParallel.h` lines 1539 and 1543 (hybrid), 1661, 1763, 1892, 1968;
- backward, every step, when the inputs were not pre-sharded: `UlyssesAttentionBackward.h` lines 95-97 (dQ, dK and dV:
  three calls per attention layer).

**Problem 1b: P x B x H copies plus a full sync.**
- With P = 8, B = 1 and H = 32, that is 256 separate `cudaMemcpyAsync` calls for one tensor.
- The copies go to the legacy default stream (0).
- `cudaStreamSynchronize(0)` then makes the CPU wait until the GPU has finished everything queued.
- In the backward this happens three times per layer, so the CPU cannot queue work ahead.

**Fix.** The permanent fix is to stop doing the address arithmetic by hand. Let the tensor library do the reordering,
the same way `ulysses_combine` already does it in its step 3. A transpose plus `contiguous()` copies with the tensor's
own dtype. That makes it one copy kernel for any dtype, with no per-slice copies and no stream sync:

```cpp
inline Tensor ulysses_gather_seq(const std::shared_ptr<ProcessGroupNCCL> &pg,
                                 const Tensor &local, int P) {
  const auto &d = local.shape().dims;
  const int64_t B = d[0], H = d[1], Tl = d[2], D = d[3];
  const int64_t T = Tl * P;
  // rank r's block lands at [r] in rank order
  Tensor g = Tensor::empty(Shape({{(int64_t)P, B, H, Tl, D}}), local.opts());
  pg->all_gather(local.data(), g.data(), static_cast<size_t>(local.numel()), local.dtype(), /*sync=*/true);
  return g.transpose(0, 1)        // [B, P, H, Tl, D]
          .transpose(1, 2)        // [B, H, P, Tl, D]
          .contiguous()           // one copy, element size taken from the dtype
          .reshape(Shape({{B, H, T, D}}));
}
```

If the copy loop is kept instead, it needs the element size from the dtype (byte pointers, `uint8_t*`, and
`elem = local.dtype_size(local.dtype())` for every offset and for `slice_bytes`). It should also run on the current
stream, without `cudaStreamSynchronize`. The transpose version above is simpler and does both.

`ulysses_combine` and `ulysses_partition` are fine for bf16. They also call `data<float>()`, but only to pass the base
pointer, and the all-to-all takes the element count and the dtype (`ProcessGroupNCCL.h` line 293).

**How to check the fix:**
- compare the new `ulysses_gather_seq` with the current one on fp32 inputs (they must match bit for bit), for
  P = 2, 4 and 8, B = 1 and 2, and a few H and Tl;
- run the same cases in bf16 and compare with the fp32 result converted to bf16 (it must match bit for bit, because
  the gather only moves values);
- run the Ulysses parity test in bf16 with `unshard = true`, and the backward with full-length (not pre-sharded)
  inputs.

## 2. The GQA KV-head repeat is built from H x r narrow views and cat

`head_repeat_interleave` (lines 143-153):

```cpp
for (int64_t h = 0; h < H; ++h)
  for (int j = 0; j < r; ++j)
    parts.push_back(xm.narrow(1, h, 1));        // H x r views of one head each
return Tensor::cat(parts, 1).contiguous();      // [B, H*r, T, D]
```

**Forward calls:** for K and V, before the all-to-all when there are fewer KV heads than ranks (`ContextParallel.h`
lines 1739-1740 and 1845-1846), and after `ulysses_combine` in the GQA path, to widen the KV heads to the query heads
(lines 1752-1753).

What it costs on every call:
- the CPU builds H x r tensor views and the argument list for `cat`;
- `Tensor::cat` along dim 1 is one batched copy kernel in the current Tensor-Implementations (`TensorFactory.cpp` from
  line 712, `BatchedCat.cu`);
- K and V are written out r times larger, so the memory and the bandwidth also grow r times.

**The backward, `head_group_reduce` (lines 160-169), costs more:**
- r `narrow(...).contiguous()` copies, r - 1 additions, a reshape and one more `contiguous()`, so about 2r + 1 kernels and r temporary tensors per call;
- it is called for dK and dV in `UlyssesGQAAttentionBackward.h` (lines 88-89 and 97-98), `UlyssesFusedGQAAttentionBackward.h`
  (126-127) and `UlyssesCPLocalGQAAttentionBackward.h` (135-136).

I have not timed these two functions on their own. The point is the amount of work: K and V are copied r times, and the
backward runs about 2r + 1 kernels where one would do.

**Fix:**
- Best: no repeat at all. Give the attention kernel K and V with their real number of heads, and let it map each query
  head to its KV head (query head e uses KV head e / r). The fused path already does GQA inside the kernel, and the
  Tensor-Implementations attention kernels take fewer KV heads than query heads directly. Then the repeat, its memory,
  and the backward reduce all go away.
- If the repeat has to stay (for the MHA-only kernel): Tensor-Implementations has no `expand` / `repeat`, so a
  zero-stride view is not possible today. Use one small kernel for the forward and one for the backward instead of the
  views and the slices:

```cpp
// forward: out[b, h*r + j, t, :] = in[b, h, t, :]           (TD = T * D)
template <typename T>
__global__ void repeat_heads(const T* in, T* out, int64_t B, int64_t H, int r, int64_t TD) {
  const int64_t n = B * H * r * TD;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t td = i % TD, e = (i / TD) % (H * r), b = i / (TD * H * r);
    out[i] = in[(b * H + e / r) * TD + td];
  }
}
// backward: out[b, h, t, :] = sum over j of in[b, h*r + j, t, :]   (sum in fp32, store in T)
template <typename T>
__global__ void sum_head_groups(const T* in, T* out, int64_t B, int64_t H, int r, int64_t TD) {
  const int64_t n = B * H * TD;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t td = i % TD, bh = i / TD;           // bh = b * H + h
    float s = 0.f;
    for (int j = 0; j < r; ++j) s += static_cast<float>(in[(bh * r + j) * TD + td]);
    out[i] = static_cast<T>(s);
  }
}
```

**How to check:**
- the new forward must equal the current `head_repeat_interleave` bit for bit, in fp32 and bf16, for r = 1, 2, 3 and 9;
- the backward must equal the current `head_group_reduce` in fp32. In bf16 the new one sums in fp32 before rounding
  once, so compare it with an fp32 reference instead;
- then run the GQA parity tests.

## 3. Only for SFT: how the loss is weighted across ranks

In `Scripts/Blutrain/bluscriptCP.cpp`, each rank takes the mean loss over its own tokens
(`sparse_cross_entropy_loss`, line 3354). The gradients are then averaged over the dp x cp ranks (the comment at
lines 25-33 explains it). The logged loss is also the average of the per-rank means (lines 3397-3404).

- **Pretraining:** this is correct. Every rank has the same number of tokens (T / P), so the average of the per-rank
  means equals the mean over all tokens.
- **SFT:** only the answer tokens are trained. Their number differs from rank to rank, and a rank can have none at all
  (its slice falls inside a prompt). Then the average of the per-rank means gives each token a different weight. A
  rank with 10 answer tokens counts as much as a rank with 2,000.

What SFT needs:
- each rank sums the loss over its trained tokens;
- the trained-token count is summed over all dp x cp ranks;
- each rank's loss is (its sum) / (the global count), and the gradients are summed, not averaged.

That gives every trained token the same weight, wherever it lands. This only matters if the CP trainer is used for SFT.
