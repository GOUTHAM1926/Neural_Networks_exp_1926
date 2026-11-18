# Verification reply: the loss-sum bug, the embedding backward, and context-parallel facts

**To:** Gautam (SFT and kernels; BluTrain / Tensor-Implementations branch `dist_side_gautam_lottn`)
**From:** Adhitya (checked by the Claude Code session on Adhitya's machine)
**Date:** 2026-10-30
**Replying to:** `cp_verification_request_2026_10_30.md`

## Scope and limits of this check

- Read only. Nothing was edited, committed or pushed. No GPU job was run, so Q6 was not done.
- Versions were read from the `.git` files directly. No git commands were run.
- **What was checked:** the local workstation copies. That is the Context_Parallelism source, the BluTrain Tensor-Implementations submodule source, and the local launch scripts.
- **What was NOT checked:** the library copies on the H100 servers (p22 / p32, under `/data/.../BluTrain/Tensor-Implementations`) and the server-side run logs.
  - Every library statement below is therefore about the local source, commit `1c84859`.
  - Whether the server builds match it is **UNKNOWN**.
- Line numbers refer to our versions: Context_Parallelism `009b3711`, Tensor-Implementations `1c84859`.

---

## 1. Summary table

| Q | Answer | Evidence (one line) |
| --- | --- | --- |
| Q1 | **YES, the bug is present.** Library: Tensor-Implementations `master` @ `1c84859` | `LossKernels.cu:174` loads one value per thread, no loop; stage 2 is `<<<1, 256>>>` (lines 486, 965) |
| Q2 | 3 call sites, all in `bluscriptCP.cpp` | train `:3354` and val `:3306`: N = B x T / CP; LongRoPE search `:2838`: N = B x min(16384, T / CP) |
| Q3 | **YES.** The backward uses only the saved stats and the denominator | `LossKernels.cu:704` `f_scale = *grad_output / *denom`, with `saved_max` / `saved_sum` per row |
| Q4 | **YES.** Evaluation uses its own kernel | `bluscriptCP.cpp:3189` calls `eval_nll_accumulate_row`; `EvalNLLKernel.cu:3` "One block per local position; no atomics anywhere." |
| Q5 | Only one hit: the LongRoPE search at CP=1, T=131072, before 2026-08-10 | Every training launcher resolves to N <= 32,768 (table in section 3) |
| Q6 | **NOT RUN** | GPU test; not authorised for this check |
| Q7 | **YES, already measured** on 2026-08-10: 7.55 printed vs 59.87 from our NLL kernel | Bug-predicted value: exp(ln 59.87 / 2) = 7.74 |
| Q8 | Default: fp32 + cooperative (deterministic). **Our long-context runs: bf16 + atomicAdd** | `run_lc8k.sh:20` sets `CP_BF16_EMBED=1`, so `wte` is built in bf16 (`bluscriptCP.cpp:616-617`) and the backward hits `EmbeddingKernels.cu:148` `atomicAdd` |
| Q9 | Context_Parallelism `main` @ `009b3711` | Your `cf1372d` is our previous commit ("forward inreadoc masking kernel") |
| Q10 | **YES.** Document masking on the ring path only | `CP_DOC_MASK=1` requires `CP_ATTN_MODE=ring` + `CP_ATTN_FUSION=split` (`bluscriptCP.cpp:1234-1242`) |
| Q11 | One offset per sequence; no per-document restart found | Document ids feed only `set_doc_ids` (the mask), `bluscriptCP.cpp:765-792` |
| Q12 | Ring forward is a fork of Hopper v62 `GQA_fwd_sm90.cu`; ring backward a fork of v44 `GQA_bwd_sm90.cu`. Ulysses on H100 needs `CP_ULYSSES_CPLOCAL=1` | `GQA_causal_cp_fwd_sm90.cu:1` and `GQA_causal_cp_bwd_sm90.cu` headers; `ContextParallel.h:1858` |
| Q13 | H100: B=1 x 16,384 per GPU (T=131072 / CP=8), 680M-class model, peak 74.1 GB | `probe128.out` "FITS  peak mem: 74103MB of 81559 MiB"; H200 / B300: UNKNOWN |

---

## 2. One section per question

### Q1. Library version and `sum_reduction_kernel`

**Version.** Read from `BluTrain/.git/modules/Tensor-Implementations/`:

- `HEAD`: `ref: refs/heads/master`
- `refs/heads/master`: `1c84859d5f13fbffbf377a2be1808d8e6b4a97fe`
- Last pull: 2026-09-26 15:02 IST (`logs/HEAD`)

The Makefile resolves `BLUTRAIN_ROOT := BluTrain` when `./BluTrain/Tensor-Implementations` exists (`Makefile:36-38`).

Caveat: the local `lib/libtensor.a` is dated 2026-08-18, which is older than the 2026-09-26 source. The server builds are UNKNOWN (see Scope).

**One value per thread, no loop.** `src/Kernels/cuda/loss/LossKernels.cu:166-190`:

```cpp
__global__ void sum_reduction_kernel(const TIn *input, TOut *output, int64_t n,
                                     const float *denom = nullptr) {
  ...
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? static_cast<float>(input[i]) : 0.0f;   // line 174
```

**Stage 2 is a single block.** `sparse_cross_entropy_forward_cuda_impl_vec`, lines 483-488. The save-stats variant has the same code at lines 962-967.

```cpp
      sum_reduction_kernel<float, float>
          <<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float),
             stream>>>(d_losses, d_partial, batch_size);
      sum_reduction_kernel<float, float>
          <<<1, reduce_threads, reduce_threads * sizeof(float), stream>>>(
              d_partial, loss_output, reduce_blocks, denom);
```

`reduce_threads = 256`, so stage 2 reads partials 0-255 only. Rows >= 65,536 are dropped.

**The denominator counts all rows.** `count_valid_targets_kernel`, line 109: `for (int64_t i = tid; i < batch_size; i += blockDim.x)`. So the printed loss is about true x 65,536 / N, as you described.

**Same pattern elsewhere in this file.** The two-stage sum is also used by the CCE (1226-1241), MSE (1335-1350), MAE (1409-1424) and BCE (1512-) forwards.

**Which forward the autograd entry point uses.** `src/autograd/operations/LossOps.cpp:315-321`:

- with grad enabled: `sparse_cross_entropy_forward_cuda_impl_vec_save_stats`;
- otherwise: `sparse_cross_entropy_forward_cuda_impl_vec`.

Both have the bug.

### Q2. Loss call sites and row count N

A search for `sparse_cross_entropy_loss`, `cross_entropy`, `mse_loss`, `bce_loss` and `categorical_cross` in `Scripts/Blutrain/bluscriptCP.cpp` finds three calls, all sparse CE. The other files in `Scripts/Blutrain/` (`gpt2_cp_test.cpp`, `dcp_weights_loader.cpp`) were not part of any run below.

| Call site | Purpose | N per call |
| --- | --- | --- |
| `:3354` `autograd::sparse_cross_entropy_loss(logits, shy.y_local)` | training micro-step | B x T / CP |
| `:3306` same call | periodic validation | B x T / CP |
| `:2838` `autograd::sparse_cross_entropy_loss(lc, yc)` | LongRoPE search fitness (`CP_LONGROPE_SEARCH`) | B x min(`CP_SEARCH_CE_CHUNK`, T / CP); default chunk 16384 (`:2759-2760`) |

Defaults:

- B = 2 (`bluscriptCP.cpp:161`), overridden by `CP_B` (`:1134`).
- CP = world size unless `CP_SIZE` is set (`:1022`).
- `GRAD_ACCUM` does not change N; each micro-step is a separate call.

The windowed validation you asked about (`win_loss`, `win_tok`) is the LongRoPE search loop at `:2831-2843`. Since 2026-08-10 it chunks the sequence, so N = B x 16384 per call. It is therefore safe only while **B <= 4**.

### Q3. The backward does not read the summed loss

`sparseCENormalize_from_stats`, `LossKernels.cu:671-705`, launched by `sparse_ce_backward_cuda_vec_with_stats` (`:979-993`):

```cpp
  const float final_max = saved_max[row];
  const float inv_sum = 1.0f / saved_sum[row];
  const float f_scale = static_cast<float>(*grad_output) / (*denom);   // line 704
```

The inputs are per-row `saved_max` / `saved_sum`, the upstream `grad_output` and the device `denom` from `count_valid_targets_kernel`. `loss_output` is never read. **Gradients are unaffected.**

### Q4. Evaluation uses its own kernel

`CP_EVAL_PPL` (per-position NLL CSV), `bluscriptCP.cpp:3183-3193`:

```cpp
                for (int bi = 0; bi < Bcur; ++bi) {
                    OwnTensor::cp::cuda::eval_nll_accumulate_row(
                        dptr + static_cast<size_t>(bi) * Tlocal * V,
```

`context_parallel/EvalNLLKernel.cu:3`: "One block per local position; no atomics anywhere." It launches at `:93` as `eval_nll_kernel<<<T, BLK>>>`.

The optional host path (`CP_EVAL_NLL_HOST=1`, `:3194-`) computes the log-sum-exp per row on the CPU in double. Neither path goes through `sum_reduction_kernel`.

The **only** evaluation-like path that does use the library loss is the LongRoPE search fitness (`:2838`, see Q2/Q7).

### Q5. Run inventory

See section 3. Method:

- B, T and CP were resolved from every local launch script that starts `bluscriptCP_exec`.
- Where `CP_SIZE` is unset, CP = `-np`.
- Server logs were not read.

### Q6. GPU test

**NOT RUN.** It was not authorised for this check. We can run it on request with V = 49,280 or 50,304, on an H100 node.

### Q7. Comparison against our own NLL kernel (already exists)

This was measured on 2026-08-10, during a reconciliation sweep between the LongRoPE search fitness and the `CP_EVAL_PPL` CSV path. Settings were CP=1, B=1, with identical windows, cache and checkpoint. From our run notes:

> T=16384 54.9950/54.994995, T=32768 51.0084/51.008415, T=65536 55.2463/55.246292 all agree, T=131072 59.87 vs 7.55 diverges 8x

At T=131072 the per-call N is 131,072, so the bug predicts a printed mean NLL of about ln(59.87) / 2 = 2.046. That is PPL **7.74**. The measured value was **7.55** (mean NLL 2.022, a ratio of 0.494 to the true 4.092).

The small gap is consistent with the dropped rows (positions 65,536-131,071) having a higher loss than the kept ones. That is an inference; per-half values were not logged.

At the time we blamed a "2^32 logit elements" limit. We now think that diagnosis was wrong (see the closing NEW finding 1).

### Q8. Embedding backward in our trainer

**Default (no flags).** The `wte` weight and its gradient are fp32:

- `bool bf16_embed = false;` (`bluscriptCP.cpp:218`)
- `wte(..., c.embed_dtype())` (`:616-617`)

For fp32 the library takes the cooperative, atomic-free kernel unless `BLUTRAIN_EMB_BWD_COOPERATIVE=0` (`EmbeddingKernels.cu:389-394`, `:417`):

```cpp
        return !(e && e[0] == '0');
```

**Our long-context runs.** All of these set `CP_BF16_EMBED=1`:

- the LC-8k arms (`lc8k_2026-10-24/run_lc8k.sh:20`);
- the D150 8k runs (`run_d150*.sh`, `lc_run.sh`);
- the D150 16k runs (`go16k_p32.sh`, `lc16_run.sh` via `run_d150_v4.sh`);
- their diagnostic scripts (20 launch scripts in total).

The quoted line from `run_lc8k.sh:20`:

```bash
  CP_BF16=1 CP_BF16_RESID=1 CP_BF16_EMBED=1 CP_BF16_LOGITS=1 \
```

With that flag, `nn::Embedding` is constructed in bf16, so its gradient is bf16.

- `EmbeddingBackward.cpp` dispatches on the gradient dtype to `launch_embedding_backward<__nv_bfloat16>`.
- The cooperative branch is `if constexpr (std::is_same_v<T, float>)`, so bf16 skips it.
- bf16 therefore always falls through to `embedding_backward_kernel_optimized`, `EmbeddingKernels.cu:148`:

```cpp
    atomicAdd(&grad_weight[(size_t)tok * grad_weight_stride_row + c * grad_weight_stride_col], grad);
```

**Plainly: our LC-8k and D150 8k / 16k runs used bf16 + atomicAdd for the embedding backward.** That path is non-deterministic and accumulates in bf16. The weights are tied in these runs (`CP_WEIGHT_TYING=1`), so `wte` is also the LM head.

Other points:

- Our trainer has no `BLU_EMB_BWD`, YAML `embedding.backward`, or fused / segmented embedding hook.
- Our library has no `BLUTRAIN_EMB_BWD` (only `BLUTRAIN_EMB_BWD_COOPERATIVE`). Your branch is newer than ours here.
- No log line names the kernel.

The size of the effect on our models is **not measured**.

### Q9. Context_Parallelism version

- **Context_Parallelism:** `.git/HEAD` = `ref: refs/heads/main`; `refs/heads/main` = `009b3711f40ed74a2e366f9808be9dc4eb874fe1`. Commit message: "after the intradoc masking kernel beating kernel changes".
- **Remote:** `https://github.com/adhityablubridge/Context_Parallelism.git`.
- **Commits since `cf1372d`:** one, `009b3711`. `cf1372d2` "forward inreadoc masking kernel" is the commit before it.
- **BluTrain:** branch `context-parallel-integration` @ `0df6ecd`.

### Q10. Document masking

**YES, on the ring path only.**

- `CP_DOC_MASK=1` derives document ids from an EOT token id (`CP_DOC_EOT`) on the GPU (`context_parallel/DocIdKernel.cu`, `bluscriptCP.cpp:765-792`).
- It installs them with `set_doc_ids(doc_full, /*lb=*/cfg.ring_mode)`.
- Attention is then restricted to same-document keys.

Restrictions, enforced with a hard error at `bluscriptCP.cpp:1234-1242`:

```cpp
        if (cfg.attn_fusion != "split")
            die("CP_DOC_MASK=1 requires CP_ATTN_FUSION=split -- the document mask is "
                "implemented only in the Stage-3 split kernels; ...
        if (cfg.attn_mode != "ring")
            die("CP_DOC_MASK=1 requires CP_ATTN_MODE=ring (got '" + cfg.attn_mode +
                "'). Ulysses and hybrid do not route through the split kernels.");
```

- The ids come from EOT scanning. There is no `cu_seqlens` / segment-id input.
- The split kernels require head_dim == 128 and are Hopper-only (see Q12).
- Our LC-8k arms and the 8k D150 runs trained with the mask on. The 16k stage trained with it off.

### Q11. RoPE positions

**One offset per sequence.** RoPE positions are global positions in the packed row, via the shard's `pos_local`. Document ids are used only for the attention mask. A search of `bluscriptCP.cpp` and `context_parallel/DocMask.h` found no code that restarts positions at document boundaries.

### Q12. Kernel lineage, and Ulysses on H100 / H200

**Ring forward.** `context_parallel/GQA_causal_cp_fwd_sm90.cu:1`:

> CP FORK of Hopper v62 (GQA_fwd_sm90.cu) -- persistent, warp-specialised wgmma/TMA causal-attention FORWARD, generalized for context-parallel ring steps

**Ring backward.** `GQA_causal_cp_bwd_sm90.cu`: "CP FORK of Hopper v44 (GQA_bwd_sm90.cu) -- TMA-reduce dQ causal-attention".

Both have these constraints:

- head_dim == 128, T_q % 256 == 0, T_k % 128 == 0;
- `__CUDA_ARCH__ >= 900 && < 1000`.

The change map is in `.mdfiles/hopper_v62_v44_CP_analysis.md`. The library commit the v62 / v44 bases came from is not recorded in these files: **UNKNOWN**.

**Ulysses on H100 / H200.**

- The default calls library `gqa_fused_flash_attn_forward`. That kernel is Blackwell-only after the library upgrade and fails on Hopper (`ContextParallel.h:1858`).
- With `CP_ULYSSES_CPLOCAL=1` it calls the CP-local WMMA fork through `sdpa_fused_forward_rope`. That fork is arch-portable and supports head_dim 64 and 128 (`ContextParallel.h:1857-1875`).

All our production runs used ring.

### Q13. Largest B x T per GPU

| GPU | Model | B x T_local per GPU | Setting | Peak memory |
| --- | --- | --- | --- | --- |
| H100 80 GB | 680M-class (d 1536, 24 layers, 12/4 heads, FFN 4096, tied) | 1 x 16,384 | T=131072, CP=8, ring (`probe128.sh`) | baseline: OOM at ~77.4 GB; `CP_SAVE_KV_BF16=1`: **74,103 MB**; + `CP_RING_BF16=1`: **71,031 MB** |
| H100 80 GB | 135M (d 512, 37 layers) | 8 x 4,096 = 32,768 | LC-8k, T=8192, CP=2 | not logged in the files checked |
| H100 80 GB | 135M | 4 x 8,192 / 2 x 16,384 = 32,768 | D150 8k / 16k, CP=1, DP=8 | not logged in the files checked |
| H200 | various | UNKNOWN | `mix64k_h200.sh`, `go_128k_h200.sh` exist; memory not checked | UNKNOWN |
| B300 | - | UNKNOWN | no B300 runs found | UNKNOWN |

The largest forward-only case (no gradients) is the LongRoPE search at CP=1, T=131072 (1 x 131,072 per GPU). Its memory was not checked.

---

## 3. Run inventory (Q5)

Sources and conventions:

- **Dates:** for LC / D150, the run dates in our notes. For the 64k / 128k rows, the script's last-modified date (marked *); the exact run dates are on the servers.
- **CP:** where `CP_SIZE` is unset, CP = `-np`.
- **Rows per call:** B x T / CP.

| run | date | purpose | B | T | CP | mode | rows per loss call | hit? | loss used for decisions? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LC-8k arms (`run_lc8k.sh`) | 2026-10-24.. | 135M long-context 8k continuation | 8 | 8192 | 2 | ring + doc mask | 32,768 | no | yes (train / val curves) |
| D150 LC (`lc_run.sh`, `run_d150_v4.sh`), DP=8 | 2026-10-26/27 | 135M long-context 8k, 3 runs | 4 | 8192 | 1 | ring (doc mask on / off) | 32,768 | no | yes |
| D150 16k (`go16k_p32.sh`, `lc16_run.sh`) | 2026-10-27/28 | 135M 16k stage | 2 | 16384 | 1 | ring | 32,768 | no | yes |
| t64 (`t64_train.sh`) | 2026-09-17* | 64k YaRN x16 extension | 1 | 65536 | 8 | ring | 8,192 | no | yes |
| cont64k / mixcont (`cont64k_b5_h100.sh`, `mixcont_h100.sh`) | 2026-09-15..23* | 64k continuation / data mix | 1 | 65536 | 8 | ring | 8,192 | no | yes |
| mix64k (`mix64k_h200.sh`) | 2026-09-23* | 64k data mix, H200, DP=2 | 1 | 65536 | 4 | ring | 16,384 | no | yes |
| ext64k_b5 (`ext64k_b5.sh`) | 2026-09-10* | 64k extension | 1 | 65536 | 4 | ring | 16,384 | no | yes |
| t128 / t128b / t128c | 2026-09-23* | 128k YaRN x32 extension | 1 | 131072 | 8 | ring | 16,384 | no | yes |
| probe128 (`probe128.sh`) | 2026-09-23* | 128k memory probe, 680M-class | 1 | 131072 | 8 | ring | 16,384 | no | no (memory only) |
| ext_32k, b4_full_epoch, b4_continue, p2_680m | 2026-09-10..15* | 32k extension runs | 1 | 32768 | 2 | ring | 16,384 | no | yes |
| **LongRoPE search, CP=1, before chunking** | up to 2026-08-10 | RoPE factor search fitness | 1 | 131072 | 1 | ring | **131,072** | **YES**, about 1/2 in log space | caught by the reconcile sweep; search re-run with chunking. Whether earlier CP=1 T=131072 searches fed a decision: **UNKNOWN** |
| LongRoPE search, after chunking | 2026-08-10.. | same | 1 | up to 131072 | 1-2 | ring | <= 16,384 | no | yes |
| CP_EVAL_PPL evals (`ppl_vs_pos.sh` etc.) | various | per-position NLL / PPL | 1 | up to 131072 | 1-8 | ring / ulysses | own kernel | n/a | yes |

---

## 4. Closing

### Affected printed losses

**Training and validation: none.** Every training launcher we found has N <= 32,768, below the 65,536 threshold.

**The one affected number** is the LongRoPE search fitness at CP=1, T=131072, before 2026-08-10. It printed about exp(true NLL x 0.5): PPL 7.55 where the truth was 59.87. It was caught at the time and the search was chunked.

Your fix (bit-identical for N <= 65,536) changes nothing in our existing results and is welcome.

### Embedding backward (Q8): needs attention

All our long-context runs (LC-8k and D150 8k / 16k) used `CP_BF16_EMBED=1`. That means a bf16 tied embedding with a bf16 atomicAdd backward:

- non-deterministic;
- each add rounded to bf16 on frequent tokens.

We have not measured how much this matters for our models. The cheapest check is one or two steps from the same checkpoint with `CP_BF16_EMBED=0` vs `=1`, comparing the embedding-gradient norm for frequent token rows. A separate question we have not checked: whether a bf16 `wte` under `CP_ZERO=1` keeps an fp32 master copy.

### NEW findings

1. **NEW: our 2026-08-10 diagnosis was wrong, and the fix holds only by a size condition.**
   - Comments at `bluscriptCP.cpp:2755-2757` and `:2817-2826` blame a "2^32 elements" limit.
   - The CE forward indexes rows in 64 bits (`LossKernels.cu:342` `int64_t row = blockIdx.x;`, with `row * vocab_size` in int64), so nothing overflows at 2^32 there.
   - The near-exact halving in log space (7.55 measured vs 7.74 predicted) is the 65,536-row bug.
   - Both hypotheses put the boundary between T=65536 and T=131072, so our sweep could not tell them apart. A run at B=1, T=81920, CP=1 would: 4.1e9 elements is under 2^32, but 81,920 rows is over 65,536.
   - Consequence: the chunk fix is correct only while B x `CP_SEARCH_CE_CHUNK` <= 65,536, i.e. B <= 4 at the default. Our search scripts use B=1.
   - We will correct the comments, and the condition goes away once your fix lands.
2. **NEW (ours): a stale comment.** `bluscriptCP.cpp:804-807` says "The embedding itself stays fp32". That is false whenever `CP_BF16_EMBED=1`, which builds `wte` in bf16 (`:616-617`).
3. **NEW (your library, relevant to SFT on B300): the sm103 forward does not honour `ignore_index`.** `LossKernels.cu:938`:

   > TODO(SFT): the sm103 kernel does NOT yet support ignore_index and on B300 , SFT label masking is silently a no-op and the model will train on prompt tokens

   This is in our copy `1c84859`; please check whether your branch still has it before any SFT run on B300.
4. **Note (our build hygiene):** the local `libtensor.a` (2026-08-18) predates the local library source (pulled 2026-09-26). This does not affect the conclusions above, but local binaries may not match the quoted source.
