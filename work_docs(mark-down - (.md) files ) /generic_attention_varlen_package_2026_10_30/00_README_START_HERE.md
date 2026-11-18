# Generic attention with causal mask and packed rows (varlen), for GPUs other than Hopper and B300

Gautam, 2026-10-30. Code: Tensor-Implementations, branch `dist_side_gautam_lottn`, commit `f7aa40f` (pushed
2026-10-27). BluTrain commit `5c0999c` on the same branch points at it.

---

## 1. What this is, in one paragraph

BluTrain trains on H200 (Hopper) with fast attention kernels written for that GPU. On every other GPU (Ampere such as
RTX 3060 / A100, Ada such as RTX 4090 / RTX 6000 Ada / L40), attention runs on older "generic" kernels.

Before this change those kernels had two gaps:

- **No causal mask.** For our model's head size, 128, they refused to run. For head size 64 they ran full (non-causal)
  attention, which is a silent wrong result.
- **No packed rows.** They could not take the SFT format where several chats share one row.

This change adds both, so SFT with packed rows now runs correctly on those GPUs. It is **correct but not optimized**
yet (section 7).

## 2. Background you need first

- **Causal attention:** token t may look only at tokens 0..t. Keys after the query get weight 0.
- **Packed row (SFT):** one row of T tokens holds several whole chats one after another, for example chat A (300
  tokens), then chat B (1,200), then a pad tail. Each chat must attend ONLY to itself: a token of chat B must not see
  chat A. This is "varlen" attention, the same idea as FlashAttention's `cu_seqlens`.
- **Segment:** one chat, or the row's pad tail. It is described by three numbers: row `b`, start `t0`, length `L`.
- **The chat plan (`GqaChatPlan`, `AttentionKernels.h`):** the trainer builds it once per micro-batch from the
  segment list. It goes to the GPU in one upload and holds the work lists for the Hopper kernels. This change adds one
  more list for the generic kernels: `gen_segs` = `{b, t0, L, 0}` per segment, plus `n_gen_segs` and `gen_max_len`.
  The new list is appended LAST, so the Hopper lists keep their place and bytes.
- **The generic kernels:**
  - `gqa_v2` (forward), in `GQA_fwd_generic.cu`;
  - `gqa_backward_dQ` and `gqa_backward_dKdV` (backward), in `GQA_bwd_generic.cu`.
  - History: written by Sanjay (2026-06-24, commit `339c4b4`, for sm103), then moved into the generic files by Madhu
    (2026-07-22, `465fd8b`).
  - Design: WMMA 16x16x16, one warp per block, 16 query rows per block, plain global loads (no TMA, no cp.async).
  - They need sm_80 or newer (bf16 WMMA).

## 3. What was changed (4 library files)

| file | change |
| --- | --- |
| `include/ops/helpers/AttentionKernels.h` | `GqaChatPlan` gets `gen_segs`, `n_gen_segs`, `gen_max_len` |
| `src/Kernels/cuda/attention/arch/GQA_fwd_generic.cu` | `gqa_v2` gets a `CAUSAL` template flag + segments; the plan builder fills `gen_segs`; dispatch routes causal calls here |
| `src/Kernels/cuda/attention/arch/GQA_bwd_generic.cu` | `gqa_backward_dQ` / `gqa_backward_dKdV` get `CAUSAL` + segments; dispatch the same |
| `src/autograd/operations/AttentionOps.cpp` | one comment |

How the kernels work now:

1. **One block = one (segment, head, query tile).** Tiles are counted from the SEGMENT START, not the row start. A
   chat inside a packed row therefore runs the same tiles in the same order as the same chat alone. That is why the
   results are bit-identical (section 5).
2. **The causal mask.** The key loop stops at the tile's last row. A key after the query gets score -inf, so its
   weight p is 0. Rows and keys past the segment end are read as zero and never written.
3. **The dK/dV backward** loops only over the query tiles that can see the block's keys.
4. **Plain causal rows** (no packing, pretraining) are the same code with one segment per row: t0 = 0, L = T.
5. **Launches** use the library's current stream (`getCurrentStream()`).

Dispatch rule, in both forward entries and both backward entries:

- `generic_ok = is_causal && !(the Hopper kernel takes it) && GPU is not B300`.
- The Hopper checks run first and are unchanged. B300 is untouched.
- To force the generic kernels on an H200 for testing, set `BLUTRAIN_NO_SM90_GQA=1`.

H200 safety (checked 2026-10-24):

- The six Hopper attention source files are byte-identical before and after this change.
- They do not include `AttentionKernels.h`.
- Compiled for sm_90a, their SASS is identical. So H200 training is not affected.

What the code did BEFORE this change, off Hopper:

- a chat plan: error;
- causal, head size 128: error;
- causal, head size 64: ran WITHOUT the causal mask, a silent wrong result.

## 4. How to build and run the test yourself

The test is `Tests/kernel_tests/test_varlen_attention.cu`.

- On a GPU that is not Hopper / B300 it tests the generic kernels automatically.
- It builds the references itself:
  - an fp64 "truth";
  - a "floor" with plain PyTorch-style bf16 math.
- Pass rule (FlashAttention's): `|kernel - truth| <= 2 x |floor - truth| + 2 x fp32 rounding`, per chat, for O, LSE,
  dQ, dK, dV.

1. Get the code: BluTrain branch `dist_side_gautam_lottn` with its Tensor-Implementations submodule (commit
   `f7aa40f`), or copy the files of `code/after_f7aa40f/` into their places in your tree (paths in section 3).
2. Build the library from the BluTrain root: `make -j$(nproc) tensor`. This also builds the profiler library the tests
   link. `SM_ARCH` is detected from `nvidia-smi`; set `SM_ARCH=86` (Ampere) or `SM_ARCH=89` (Ada) to force it.
3. Some groups need real chat lengths. Point the test at the labels file in this package:
   `export VARLEN_STAGE0_LABELS=<package>/data/stage0_ultrachat_5k_T4096_train_labels.bin`
   - It holds the labels of 5,000 real SFT chats, padded to 4,096 tokens.
   - The test only reads the chat lengths from it.
4. Run, from `Tensor-Implementations/`:
   - head size 128 (our model):
     `VARLEN_QUICK=1 make run-cuda FILE=Tests/kernel_tests/test_varlen_attention.cu`
   - head size 64:
     `VARLEN_QUICK=1 VARLEN_HD=64 make run-cuda FILE=Tests/kernel_tests/test_varlen_attention.cu`
   - without `VARLEN_QUICK=1`: the full run, with many more repeats (it was never run on a small GPU; expect it to be
     long).
5. Read the end of the output: `checks passed N, failed M`. Then read the `KNOWN` lines and `ALL RUN CHECKS PASS` or
   `FAILURES ABOVE`.

Other switches (the header of the test file lists all of them):

- `VARLEN_ONLY=G10` / `G11`: only the packed forward / backward groups.
- `VARLEN_CASE=<name part>`: only matching layouts.
- `VARLEN_HEADS=4/2`: other head counts.
- `VARLEN_ONLY=BENCH`: timing.

## 5. Results (RTX 3060, sm_86, quick mode, 2026-10-24; logs in `logs/`)

**Head size 128: 292 passed, 0 failed, 1 KNOWN** (`test_hd128_quick_rtx3060_2026-10-24.log`, 1,066 s).

What was covered:

- dense causal rows: T 256 / 4,096, heads 18/6 and 8/1;
- real SFT rows, large inputs, every chat alone;
- packed rows: layouts D2-D11 plus 16 real packed rows.

What was proven:

- **Packed == alone, BIT-EXACT,** in every layout (0 mismatched elements).
- **Isolation:** changing one chat leaves every other chat's outputs bit-identical.
- **Repeats:** 50x at 8 x 4,096 and 20x at 60 x 1,024, all five outputs bit-identical (no atomics, so no run-to-run
  noise).

The KNOWN case:

- dQ of chat `row 1 @664 L=303` is at 1.049x the limit.
- The same chat gave the same ratio on H200's own backward on 2026-10-10.
- The same chat run alone gives it too, so it is not caused by packing.
- Cause: dS (the softmax backward) is computed from the bf16-rounded P.

**Head size 64: 291 passed, 1 failed** (`test_hd64_quick_rtx3060_2026-10-24.log`, 741 s).

- The failure: dQ of chat `row 0 @2688 L=64`, at 1.175x the limit.
- Its packed-vs-alone check is bit-exact, so it is not packing.
- The cause is the same bf16-P one as above.
- Fix (not done yet): in the dQ kernel compute dS from the fp32 P, as the H200 kernel and FlashAttention-3 do. In that
  kernel P is used only for dS.
- `test_hd64_first_try_HARNESS_BUG_2026-10-24.log` is an earlier run that stopped with "illegal memory access". The
  cause was a bug in the TEST, not the kernels: two helpers had head size 128 written as a number. It was fixed before
  the run above. The log is kept so you can see what such a harness bug looks like.

**Speed (not optimized), RTX 3060, B = 8, T = 4,096, 18/6 heads:**

- forward 726 ms, backward 1,669 ms, about 0.9 TFLOP/s;
- packing skips work: a packed backward with 14% of the dense work took 12.5% of the dense time.

## 6. What has NOT been done (be honest about these when you use it)

- **The full (non-quick) test run.**
- **A run on an Ada GPU (sm_89).** Only the RTX 3060 (sm_86) was used. Ada should behave the same (same code path), but
  it is not tested.
- **An end-to-end SFT training run on a non-Hopper GPU.** The trainer build on the laptop (RTX 3060) did not complete,
  so only the kernel test ran.
- **The head-size-64 dQ fix** (section 5).
- **Any optimization** (section 7).

## 7. Why it is slow, and the optimization plan

The kernels:

- use one warp per block;
- issue plain global loads (no cp.async: 0 `LDGSTS`, 0 `ldmatrix` in the sm_86 SASS);
- wait for each load before computing.

cp.async exists from sm_80, so these GPUs have it. Our RoPE kernel already uses it and passes on the RTX 3060.

Plan, in the FlashAttention-2 style:

1. **4 warps per block,** each warp owning 16 query rows.
2. **K/V tiles double-buffered in shared memory with cp.async:** load tile j+1 while computing tile j.
3. **`ldmatrix` + `mma.sync`** instead of WMMA fragments.
4. **Compute D = rowsum(dO x O) once,** not per kernel.
5. **Fix the dK/dV register spills** at head size 128 (320 B).
6. **fp32 P for dS** (also fixes the head-size-64 case).

Keep the rules:

- packed == alone bit-exact;
- no atomics (bit-identical repeats);
- the test above must pass before and after every step.

## 8. What is in this package

| path | what |
| --- | --- |
| `00_README_START_HERE.md` / `.docx` | this document |
| `code/after_f7aa40f/` | the 4 library files + the test, exactly as pushed (same bytes as the files tested on 2026-10-24) |
| `code/before_bda3a34/` | the 4 library files before the change (the parent commit) |
| `diffs/library_changes_bda3a34_to_f7aa40f.diff` | the change as a diff (read this first) |
| `logs/` | the test runs of section 5 and the exact compile command used on the RTX 3060 |
| `data/stage0_ultrachat_5k_T4096_train_labels.bin` | real SFT labels, for `VARLEN_STAGE0_LABELS` |
| `MANIFEST_md5.txt` | checksums of every file |

Suggested reading order:

1. this README;
2. the diff;
3. `GQA_fwd_generic.cu`: the `CAUSAL` branch of `gqa_v2` and the `generic_ok` dispatch;
4. `GQA_bwd_generic.cu`: the same for dQ and dK/dV;
5. the test's header comment (groups, references, pass rule).
