# BluBLAS: my GEMM fixes and the symmetric-output change -- what each does, where they collide, the risks of keeping them apart, and how to put both on one path

Prepared 2026-11-11 for Aparna (and anyone who merges BluBLAS or Tensor-Implementations). Everything below was checked in
the repositories: the commits, the diffs, and a trial merge (appendix B shows how to repeat it).
**Updated 2026-11-11 evening: the merge is done, tested and pushed to my branch. Start with the next section.**

## Read first: the merge is done and pushed to my branch; please check it

Aparna, as you said ("just merge and push, I'll check"), I merged your symmetric change into my BluBLAS fixes and pushed
it to **my branch only**:

| repository | branch `dist_side_gautam_lottn` | what it is |
| --- | --- | --- |
| BluBridge-BLAS | **cfa0e92** | merge commit, parents a6d4292 (my fixes) and `master` e392dbb (your PR #4 = cb42c13) |
| Tensor-Implementations | 4d5c738 | only the BluBLAS pointer, now cfa0e92 |
| BluTrain | 597360b | only the Tensor-Implementations pointer, now 4d5c738 |

Nothing went to `master` or `preproduction-135M` (BluBLAS `master` is still e392dbb). After your check, everything goes
to `preproduction-135M` together. cfa0e92 descends from `master`, so `master` can fast-forward to it with no conflict.

**What I kept from your change and what I replaced** (details: section 8.5):

- Kept as is: your switch `blublasSetSymmetricOutput` (off by default), the context field, `sm90_tile_plan`, the mirror
  kernel, and the flag passed through the SM90 nt / tn variants and the addmm dispatch.
- Your tile filter `(r+1)*BM > c*BN`: moved into my GPU-built tile tables. The old 16-slot cache made the CPU wait
  (about 20 ms) on every new GEMM shape. Same tiles, same order (checked on 1,160 grids).
- Your `bias_row0` / `full_bias`: my FullBatched per-batch bias computes the same rows; yours stays in the file as
  comments.
- Your SM89 bias line: stays as a comment; after my addmm fix that kernel never receives a bias.
- Nothing of yours was deleted: everything active is live code, and the 4 replaced parts are in the files as comments
  with the reason next to each.

**What I tested** (details: section 9):

- With the switch off, nothing changes: every existing kernel is byte-identical to before the merge; the GEMM suite
  (7,220 cases) and the addmm suite pass with identical results; the 135M SFT trainer has the same loss and step time.
- With the switch on: 75 correctness cases pass on H200, including the batched Newton-Schulz addmm with a per-batch bias
  (lower triangle bit-identical to the full GEMM, upper triangle an exact mirror).
- Speed: at n = 2304 your mode is 30-38% faster; at n = 256 / 512 it is up to 38% slower (section 9.2). So it is better
  turned on only for large matrices.

**Please check on your side:**

1. Point your Tensor-Implementations (`coconut`) BluBridge-BLAS to cfa0e92, rebuild, run your Muon test (the 1.5B JIT
   runs), and compare with your earlier numbers.
2. `git diff e392dbb cfa0e92` shows everything that differs from your version: my three fix commits plus the resolution
   (section 8.5 lists each change to your code).
3. Heads-up for your compiler line: your Tensor-Implementations (`coconut`) and mine both define
   `newton_schulz_orthogonalize_batched`, with the same signature (section 8.3). When your line takes my
   Tensor-Implementations, only one may stay, or the build fails.

Tell me if anything is missing or wrong.

## 0. Summary

- BluBLAS has split into **two lines** from the same commit (ea666ab, PR #3):
  - **my line** (branch `dist_side_gautam_lottn`, head a6d4292): 3 commits, with **13 correctness fixes** and a CPU-stall fix;
  - **your line** (branch `master`, head e392dbb = PR #4, commit cb42c13 "symmetric change"): the opt-in **symmetric-output mode**.
- **Neither line has the other's work.**
  - Without my fixes, the old library failed **654 of 6,644** GEMM test cases (**197 of 270** test groups crashed). Several
    of these bugs sit on the path your symmetric mode uses.
  - Without your change, my line just lacks the symmetric speed-up (no correctness loss).
- A trial merge of the two heads gives **one conflicted file** (the SM90 warp-specialized template, 8 hunks in 3 groups),
  2 files merged automatically, and everything else clean. **The conflicts are real design overlaps, not text noise:**
  1. both lines added a **per-batch bias** to addmm, in different ways;
  2. your symmetric tile filter lives in the **old tile-order cache**, the one my line replaced because it stalled the CPU;
  3. both lines changed **how the launchers pick the tile map**.
- **Proposal:** build one merged BluBLAS commit (my fixes + your symmetric mode). Port your filter into my GPU-built tile
  tables, keep one per-batch bias, test both features together (including the batched Muon Newton-Schulz call where they
  meet), then point every Tensor-Implementations branch at that commit. Details in section 6.

## 1. The two lines at a glance

| | my line | your line |
| --- | --- | --- |
| BluBLAS branch | `dist_side_gautam_lottn` | `master` |
| commits after ea666ab | 297f372 (2026-11-03), 0b552f0 (2026-11-05), a6d4292 (2026-11-05) | cb42c13 (2026-11-05), merge e392dbb (2026-11-06) |
| size | 3 commits: 34 files, +1,224 / -412 lines | 1 commit: 7 files, +167 / -83 lines |
| Tensor-Implementations branches that point to it | `dist_side_gautam_lottn` (d59740b) | `madhu_post-training` (c61ea5d, the CP DPO work) |
| Tensor-Implementations `preproduction-135M` (a9e37e6) | points to ea666ab, the common base: it has **neither** line yet | (same) |
| tests | 6,644 GEMM + 13,302 addmm cases on 2x H200, 0 failed; 7,220 cases on an RTX 3060, 0 failed | none in the commit |

The BluBLAS `preproduction-135M` branch is still at 35e51a0 (August), older than both lines.

## 2. What my three commits change

### 2.1 297f372 -- tile-order tables without CPU stalls; deterministic split-K; sm_80/86/89 last-group fix

- **The CPU-stall fix.**
  - A GEMM with a new tile grid used to make the CPU wait for all queued GPU work. Examples: the SFT lm_head on trained rows
    only, MoE expert GEMMs, and on SM103 any two shapes of one kernel type.
  - The old caches built the table on the CPU, cudaMalloc'd it, copied it with a plain cudaMemcpy (which waits for the
    whole GPU queue) and cudaFree'd it when full (SM90: 16 slots).
  - Measured on H200: 20,011-20,074 us per call with 20 ms of GPU work queued.
- **New `blublas_tile_order_tables.cuh`:**
  - a row-first table depends only on grid_n, so one table per (GPU, order, key) serves every grid;
  - tables grow to the next power of two and are written on the GPU by a fill kernel on the GEMM's stream (no host buffer,
    no blocking copy);
  - mutex, GPU id in the key, never freed, limits 512 tables / 256 MB per GPU (then nullptr, and the caller falls back);
  - `static inline` on purpose: as plain `inline` the linker kept one SM103 copy, so on H200 every fill failed and every GEMM
    did cudaMalloc + cudaFree silently. The error path now prints once.
  - Used by SM90 warp-specialized, SwiGLU fused, cluster-NN persistent, SM103 cluster / addmm / dual / swap and FP8 SM103;
    the kernels themselves are unchanged.
- **Deterministic split-K:** the BN=64 TN split-K used atomicAdd, so dW GEMMs differed run to run. Even N now uses
  `Sm90Config_128x64x64_ws` (one fp32 slice per split + an ordered reduce): 1 distinct output in 20 runs.
- **sm_80/86/89 last swizzle group** (BF16 SM89 core and FP32 SM86 core): the last group of 8 tile rows was treated as full.
  When grid_y > 8 and grid_y % 8 != 0, some output tiles were never written. The last group now gets only the rows left.
- **Tests:**
  - SM90 kernels byte-identical except the 3 new `_ws` kernels;
  - H200, 59 shapes: same speed except split-K dW (10-15% faster at 16k);
  - new-shape stall 20 ms -> 3.7-5.2 us;
  - the real SFT trainer: 21 blocking table copies -> 0;
  - RTX 3060: the broken sm_86 cases are now correct, the normal ones bit-identical.

### 2.2 0b552f0 -- addmm: per-batch bias mode; every addmm path adds the bias like torch.addmm

- **SM90 bf16 addmm, new bias mode `FullBatched`** (bias_numel == batchCount*M*N, one [M,N] bias per batch element).
  - It is needed by the **batched Muon Newton-Schulz**: B_i = b*A_i + c*A_i@A_i.
  - Before, such a bias fell to the Runtime mode and was **not added at all**.
  - The FP16 SM90 header got the same enum line; SM103 addmm declines the per-batch case.
- **Fallback addmm** (bf16/fp16: K % 64 != 0 on H200, every call on sm_8x, strideB = 0).
  - It computed alpha*A@B + beta*C_old + bias (a scale pass, then a beta=1 GEMM). With beta = 1 the bias was **not added**,
    and the uninitialised output **was**.
  - Now it is torch.addmm's non-fused path: copy the bias into C, then ONE GEMM with alpha and beta.
- **beta == 0:** neither the bias nor C is read (torch: "nans and infs should not propagate").
- An unsupported bias size is reported on stderr instead of being dropped silently.
- **fp32 SM89 addmm:** a full [M,N] bias was read at row (batch % M) for every row; now per output row. fp32 SM86/SM89:
  per-row [M] bias supported.
- **Tested** on H200: BluBLAS addmm directly + the Tensor-Implementations addmm, every dtype, bias shape, layout and
  alpha/beta, against a double-precision reference.

### 2.3 a6d4292 -- GEMM correctness (found by a full GEMM test; every bug also existed before this commit)

| # | bug | when it hits | effect | fix |
| --- | --- | --- | --- | --- |
| 1 | **SM90 bf16/fp16 C stored through a 2-D [batch*M, N] TMA descriptor** (warp-specialized, 4 cluster-NN templates, SwiGLU fused, fp16 warp-specialized) | batch > 1 and M not a multiple of the tile height | the rows of the last tile row run past M and are **written into the next batch element**: wrong, and run-to-run different, results | 3-D C descriptor {N, M, batch} + 3-D store: rows past M are dropped for every batch element |
| 2 | **both operands transposed (TT)** | any TT call | bf16/fp16 had no kernel: tiles loaded along the wrong axis from misaligned addresses; fp32 sent TT to the NN kernel | new `blublas_gemm_generic.cuh` (a strided-batched GEMM correct for every input), used for TT, K == 0 and calls the SM8x vector kernels cannot load |
| 3 | SM103 kernels for non-packed leading dimensions | B300, lda/ldb/ldc not packed | wrong results (those kernels take no leading dimensions) | SM103 only for packed lda/ldb/ldc |
| 4 | fp32 SM89 kernel | Ada GPUs | TN K tail multiplied stale shared memory; misaligned float2 stores; 32x32 TN swizzle overwrote rows; BK = 32 skipped half of K | each fixed (alignment per batch element, zero-filled tails, aligned stores, swizzle kept in a 32-wide tile, static_assert BK == 16) |
| 5 | fp32 SM89 wrappers | small shapes; NT >= 256x128 | the split-K fallback added onto C and never applied beta; the NT kernel's prologue read data before cp.async landed | split-K only when there is a split; hand-written NT only for aligned calls; prologue always commits STAGES-1 groups |
| 6 | fp32 addmm fused kernel | operands it cannot load (16-byte) | wrong loads | the fused kernel only for loadable operands, else bias copy + one GEMM |
| 7 | **fp16 SM90 TN with M % 128 != 0** | such shapes | rows past the last 64-row chunk never loaded | declined, like bf16 |
| 8 | **bf16/fp16 gemv** | threads with no output | they returned before loading their part of the A tile and before the barrier | fixed; vector kernels only for aligned calls |
| 9 | **fp32 on sm_80/86** (A100, RTX 30xx) | busy GPU (batched calls) | the cp.async commit sat inside `if (tile to load)`, so the last K tile was read before it arrived: wrong results | one commit per stage, also an empty one (as the other SM86 kernels and CUTLASS) |

- **Totals:** 6,644 GEMM cases, 0 failed (the previous library: **654 failed, 197 of 270 groups crashed**); addmm 13,302
  cases, 0 failed; batched Newton-Schulz unchanged; cases both libraries pass are bit-identical (except fp32 split-K, whose
  atomic order varies); the SFT trainer's step time unchanged.
- **RTX 3060:** 7,220 cases, 0 failed. Before the sm_86 fix there were 152 fp32 failures, all batched; in 5 paired rounds the
  old kernels failed 148-160 times per round, the fixed ones 0.

## 3. What your change does (as I read it in cb42c13)

- **API:** `blublasSetSymmetricOutput(handle, enable)`, plus a context field `symmetric_output` (default 0). With enable != 0
  the caller guarantees that every following **square (M == N) bf16 GEMM / addmm result is symmetric**.
- **SM90:**
  - `sm90_tile_plan(M, N, BM, BN, symmetric)` keeps only the tiles on or below the diagonal: tile (r, c) is dropped when
    (r+1)*BM <= c*BN. The list is passed to the kernel as map_tiles x 1.
  - `sm90_mirror_lower_kernel` then copies C[b][j][i] = C[b][i][j] for i < j, in 64x64 tiles through shared memory, batch by
    batch (batch stride n*ldc).
  - The filter is built inside the **existing 16-slot `sm90_tile_map_cached`** (new key fields sym_bm / sym_bn and a tile count).
- **Per-batch bias, your version:** `bias_row0 = (bias_numel == M*N) ? 0 : batch*M`, a `full_bias` flag that covers M*N and
  batchCount*M*N, and the bias TMA map sized bias_numel / N rows.
- **SM89:** one line in `bgemm_scale_template_kernel`, the batchCount*M*N case in the bias chain.
- **nt / tn variants and SM90 addmm dispatch:** the symmetric flag passed through.
- **Off by default:** with the switch off the plan keeps every tile, so nothing changes.

## 4. Where they collide

### 4.1 Trial merge result (a6d4292 + e392dbb)

- **Conflicted:** `src/level3/batched_gemm/BF16/SM90/Bgemm_sm90_warpspecialized_template.cuh`, 8 hunks.
- **Merged automatically:** `BF16/SM89/Bgemm_core_template.cuh`, `BF16/SM90/Bgemm_sm90_tn_variants.cu`.
- **Clean:** everything else.

| hunks | my side | your side | what it means | proposed resolution |
| --- | --- | --- | --- | --- |
| 1, 2, 5, 6, 8 (bias) | `Sm90BiasMode::FullBatched`: bias row = (FullBatched ? batch*M : 0) + ..., bias map with batchCount*M rows | `bias_row0` offset, `full_bias` flag, bias map with bias_numel / N rows | **the same feature built twice**; they compute the same rows | keep **mine** (tested on 13,302 addmm cases against fp64, and the fallback and beta == 0 rules depend on it); drop `bias_row0` / `full_bias` |
| 3 (tile table) | the old cache replaced by the GPU-built shared tables (section 2.1) | the old 16-slot cache, extended with the symmetric filter | **design clash**: keeping yours brings the CPU stall back; keeping mine drops your filter | keep my tables and **port the filter into them**: a "symmetric" table kind keyed by (GPU, grid, BM, BN) that returns the kept-tile count |
| 4, 7 (tile map in the launchers) | the tall-grid row-first rule (`TALL_RATIO_THRESHOLD`) on my tables | `const int* tile_map_arg = plan.map;` from your plan | both changed the same lines | your `Sm90TilePlan` on top of my tables: the symmetric plan when on and M == N, else my rule |

### 4.2 Interactions that are not text conflicts (they need tests, not only a clean merge)

1. **The C store.** Your symmetric mode writes the computed lower-triangle tiles through the normal C store.
   - On your line that store is still the 2-D descriptor of bug 1 in section 2.3. With batch > 1 and n not a multiple of the
     tile height, the bottom tile row of element b writes into the first rows of element b+1, which are part of b+1's
     lower triangle.
   - `sm90_mirror_lower` then copies whatever is there into the upper triangle, so a corrupted lower triangle becomes a
     corrupted whole matrix.
   - After the merge my 3-D store fixes this, and it should be tested for that case.
2. **Both features in one call: batched Newton-Schulz.**
   - In B = b*A + c*A@A (A = X Xᵀ is symmetric, so A@A is too), the addmm has a **per-batch bias** (b*A_i, my FullBatched)
     **and** a **symmetric result** (your mode).
   - That combined path is exactly the Muon use case, and it was never tested on either line.
3. **The cache keys.** On your line every symmetric shape adds a new key to the 16-slot host cache. More keys mean more
   misses, and each miss is a blocking copy (20 ms with a busy GPU) plus a cudaFree when the cache is full. My tables have
   no such stall.
4. **The SM89 line.** My 0b552f0 changed the sm_8x addmm fallback to "copy the bias into C, then one GEMM", so the scale pass
   is no longer how the bias is added there. Your extra line is probably harmless in the merged code (never reached with a
   per-batch bias). The addmm suite must confirm that no bias is added twice.

## 5. If the two lines stay apart

### 5.1 For everything built on your line (`master`, e392dbb)

This includes Tensor-Implementations `madhu_post-training`, so the CP DPO runs. Every bug fixed in section 2 is still there:

| bug (section) | who hits it | effect |
| --- | --- | --- |
| 2-D C store spill (2.3 #1) | any batched SM90 bf16/fp16 GEMM with ragged M; **your symmetric mode with batch > 1** | wrong or run-to-run different rows in the next batch element; the mirror spreads it to the whole matrix |
| beta == 0 still reads C (2.2) | any addmm with beta = 0 on a fresh output buffer | NaN / Inf from uninitialised memory spread into the result |
| fallback: bias missing at beta = 1, uninitialised output added (2.2) | bf16/fp16 addmm with K % 64 != 0 on H200, **every** addmm on sm_8x | wrong addmm results |
| per-batch bias (2.2) | batched Newton-Schulz | your line adds the b*A_i term on the SM90 kernel path (your `bias_row0`); the fallback paths keep the beta = 1 problem of the row above, so a Newton-Schulz addmm that falls back computes a wrong update |
| CPU stall per new GEMM shape (2.1) | changing shapes: trained-rows lm_head, MoE, symmetric on/off for the same grid | the CPU waits for the whole GPU queue (20 ms per call measured) |
| non-deterministic BN=64 split-K dW (2.1) | weight gradients of those shapes | run-to-run differences in training |
| TT GEMM wrong (2.3 #2) | any call with both operands transposed | wrong results |
| fp16 SM90 TN, M % 128 (2.3 #7) | such shapes | rows never loaded |
| gemv early return (2.3 #8) | bf16/fp16 gemv | race, wrong values |
| sm_80/86 cp.async race (2.3 #9) | fp32 on A100 / RTX 30xx under load | wrong results |
| last swizzle group (2.1) | sm_80/86/89 with grid_y > 8, grid_y % 8 != 0 | output tiles never written |
| fp32 SM89 bugs (2.3 #4, #5) | Ada GPUs | wrong results |

In numbers: the old library failed 654 of 6,644 cases, and 197 of 270 test groups crashed.

### 5.2 For everything built on my line

There is no symmetric speed-up, and no correctness problem.

### 5.3 For the team

- Two Tensor-Implementations branches build two different GEMM libraries, so their results and speeds are not comparable.
- When the branches are merged into `preproduction-135M`, the BluBLAS pointer conflicts (a6d4292 vs e392dbb; the trial
  merge of the two Tensor-Implementations heads shows it as the only conflict).
  - Whoever resolves it by picking one side **silently drops the other's work**: either the 13 fixes, or the symmetric mode.
- The longer the lines stay apart, the harder the merge gets.

## 6. Putting both on one path: what, why, where, how

**Status 2026-11-11 evening:** steps 1-4 are done (section 9; the GEMM suite grew to 7,220 cases). Step 5 is done for my
branch only: BluBLAS `dist_side_gautam_lottn` = cfa0e92; `master` stays at e392dbb until you have checked it. Step 6 is
done for my Tensor-Implementations and BluTrain branches; `madhu_post-training` and `preproduction-135M` follow when
everything goes to `preproduction-135M`.

- **What:** one BluBLAS commit with my fixes and your symmetric mode.
- **Why:** sections 5.1-5.3.
- **Where:** the BluBridge-BLAS repository. Both branches (`master` and `dist_side_gautam_lottn`) then move to that commit,
  and every Tensor-Implementations branch points to it.

**How, step by step:**

1. **Branch** from my a6d4292 and merge `master` (e392dbb) into it.
2. **Resolve** the 8 hunks as in section 4.1:
   - bias: keep FullBatched, drop `bias_row0` / `full_bias`;
   - tile table: my GPU builder plus a symmetric table kind (the filter (r+1)*BM > c*BN, returning the tile count), keyed by
     (GPU, grid, BM, BN);
   - launchers: your `Sm90TilePlan` on my tables.
   - Keep your API, context field, mirror kernel and the pass-through in the variants. Keep or drop the SM89 line after
     reading the merged fallback (section 4.2 point 4).
3. **Test on H200:**
   - Symmetric **off**: my full GEMM suite (6,644 cases) and addmm suite (13,302 cases). Results must be bit-identical to
     a6d4292, since the switch is off.
   - Symmetric **on**, a new test:
     - square n, both multiples of 128 and not, batch 1 and > 1;
     - GEMM X Xᵀ and the addmm b*A + c*A@A with a per-batch bias (the batched Newton-Schulz call);
     - beta 0 with a NaN-filled C, and beta 1;
     - check: the computed lower triangle bit-identical to the full GEMM's lower triangle, and the upper triangle exactly
       its transpose.
   - **Speed:** symmetric vs full on the Newton-Schulz shapes; non-symmetric speed unchanged against a6d4292.
4. **Test on an RTX 3060:** the sm_86 suite (7,220 cases) for the SM8x files both lines touched.
5. **Push** the merged commit and move both BluBLAS branches to it.
6. **Point Tensor-Implementations at it:** `dist_side_gautam_lottn`, `madhu_post-training` and `preproduction-135M`, then
   BluTrain. After this, the `preproduction-135M` merge has no BluBLAS conflict.

**Who:** I can do steps 1-4, since I know my side. You review the port of your filter and plan, which is your design, or
you do the port on top of my line. Either way, we agree before anyone starts.

## 7. Questions for you

1. Which callers turn the symmetric mode on: Muon Newton-Schulz in your Tensor-Implementations or BluTrain code? Which
   branches?
2. Do you have speed measurements and correctness checks for it (shapes, batch sizes)?
3. Is your per-batch bias needed for anything my FullBatched mode does not cover?
4. Are you fine with the filter moving into the GPU-built tile tables (section 4.1, hunk 3)?

(2026-11-11 evening: question 1 is answered in section 8.2 and question 2 partly in section 8.4 (your 1.5B runs on
P032). The filter move of question 4 is done (section 8.5); please confirm it in your check.)

## 8. Added later on 2026-11-11: what I found in your branches

After your messages I read your branches on GitHub (fetched fresh) and your tree on P032 (`/localstorage/Aparna`,
read-only: nothing written or started there).

### 8.1 Where your work is

| repository | commit | branches | on P032 |
| --- | --- | --- | --- |
| BluBLAS | cb42c13 "symmetric change" (2026-11-05) | `symmetric_change`, merged into `master` as e392dbb | `BluTrain/Tensor-Implementations/BluBridge-BLAS`, no uncommitted edits |
| Tensor-Implementations | dc7414f "muon_optimization" (2026-11-05) | `coconut`, `infer-comp`, `jit-sft-nova` | `BluTrain/Tensor-Implementations`, branch `coconut` |
| BluTrain | bad5ecc "muon_optimization" (2026-11-05) | `master_v2_compiler`, `jit-sft-nova` | `BluTrain`, branch `master_v2_compiler` |

Nothing on GitHub is newer than cb42c13 on the BluBLAS side, so sections 3 and 4 still describe your BluBLAS change.

### 8.2 Who turns the symmetric mode on (answers question 1 of section 7)

- Only Tensor-Implementations `coconut` (dc7414f): `newton_schulz_orthogonalize_batched` computes A = X Xᵀ and
  B = b·A + c·A@A with `symmetric_out = true`, through `cuda_matmul` / `cuda_addmm` and a `BlublasSymmetricScope` that
  sets `blublasSetSymmetricOutput` for one call.
- No caller in Tensor-Implementations `dist_side_gautam_lottn`, `madhu_post-training` or `preproduction-135M`.
- So merging BluBLAS alone brings your code to the preproduction path, but not your speed-up: that also needs the
  symmetric flag in the Newton-Schulz that the preproduction trainers call (section 8.6).

### 8.3 The same Muon work on both lines

We both optimized the Muon Newton-Schulz in parallel (mine: Tensor-Implementations dbb4e5f, d98d6ee, e2af706 and
BluTrain 37514ff, 96f135c, 1bec9af, accc6c8, 2026-11-04/05).

| piece | my line | your line |
| --- | --- | --- |
| batched Newton-Schulz | `newton_schulz_orthogonalize_batched(G, steps, eps)`: one batched Frobenius-scale launch per stack (bit-identical to the single-matrix kernel), tall matrices transposed in and out, fp32 result; tested bitwise against the per-matrix function (`test_muon_batched_ns.cu`) | **the same name and signature**: per-matrix scale fused with the bf16 cast (3 launches per matrix), tall matrices without transposes (A = XᵀX, X_new = a·X + X@B), symmetric A and B, bf16 result; the single-matrix path also goes through it (n = 1) |
| update guard | decided on the GPU, one batched flag / ratio per chunk | decided on the GPU (`muon_flag_bad_update_cuda`, skip flag), check + norm in one read |
| momentum | batched prep + weight decay, one launch per chunk instead of three per matrix | fused per matrix (cast + prescale + decay + momentum) |
| apply | from fp32 | from bf16, publishes bf16 in the same pass |

Trial merges (read-only, `git merge-tree`):

- **Tensor-Implementations d59740b + dc7414f:** `Optim.cpp` merges with no conflict but then holds **two definitions of
  `newton_schulz_orthogonalize_batched`**, so the build fails. Conflicted: `Optim.h`, `MultiTensorKernels.h`,
  `MultiTensorKernels.cu` (5 hunks), `GenMatmul.cu`, `MuonKernels.cu` (2), `AttentionOps.cpp` (2), `ReshapeOps.cpp`, and
  the BluBLAS pointer. (Since the common base 02d0d18 my branch has 19 commits and yours 124; the attention /
  ReshapeOps conflicts come from the compiler line, not from Muon.)
- **BluTrain 572c3ac + bad5ecc:** 9 conflicted: the Tensor-Implementations pointer, `ShardedMuon.h`, `ShardedMuon.cpp`,
  `ZeroOptimizerBase.h`, `train/production_zero/bluscript_zero_ckpt.cpp` and 4 root `bluscript_zero_ckpt_*` trainers.

### 8.4 Your measurements on P032

| what | result |
| --- | --- |
| 1.5B, JIT ZeRO-1 Muon trainer, `BluTrain_base` vs `BluTrain`, 15 steps (2026-11-04) | 1 GPU 19,560 → 19,492 ms (−0.35%); 2 GPUs 9,850 → 9,815 ms (−0.36%); 8 GPUs 2,512 → 2,503 ms (−0.4%); loss the same to 4 decimals |
| 2026-11-06 | one 1-GPU run (19.43 s per step) and one DDP run |
| standalone correctness test of the symmetric GEMM | none found in the tree |

These runs were made before the commit time of cb42c13, and the logs do not show whether the symmetric BluBLAS was in
that build. They are on another trainer and model than mine, so the two lines' gains cannot be compared from the
records; only a run of both on the same trainer and GPU can.

### 8.5 How the merged BluBLAS keeps both (the resolution, in detail)

- **Bias (hunks 1, 2, 5, 6, 8):** my FullBatched mode. Your `bias_row0` line stays in the file as a comment (it computes
  the same rows).
- **Tile table (hunk 3):** my GPU-built tables, plus a new symmetric table next to them in
  `blublas_tile_order_tables.cuh`: keyed by (GPU, grid_m, grid_n, BM, BN, order), filled on the GEMM's stream by a
  kernel that walks the same order as your builder (Hilbert when grid_m == grid_n, else row by row) and keeps the same
  tiles ((r+1)·BM > c·BN); the kept-tile count comes from a formula on the host, so no copy back. Your list, in your
  order, without the CPU wait or the 16-slot limit. Your old cache body stays in the file as a comment.
- **Launchers (hunks 4, 7):** your `Sm90TilePlan` / `plan.map` / mirror call, with `sm90_tile_plan` given the stream so
  the non-symmetric branch is exactly my old rule and table (so with the switch off nothing changes).
- **Kept as they are:** your API, context field, mirror kernel, the pass-through in the nt / tn variants and the addmm
  dispatch, and your SM89 line (after the merge it is never reached with a bias: the fallback copies the bias into C
  first, section 2.2, so nothing is added twice).
- **Launchers that do not take the switch** (the split-K launcher `launch_bgemm_sm90_splitk`, which has no `symmetric`
  parameter and no tile map; cluster-NN; ping-pong; SM89; generic) compute the whole matrix: still correct, no speed-up.
  Only the warp-specialized launcher and its addmm use the shorter list.
- **Extra tests for the merge** (on top of section 6 step 3): a symmetric call that the dispatcher sends to split-K
  (must give the full, correct matrix); symmetric
  through the fallback addmm (bias copy + GEMM); the kept-tile fraction and speed for small matrices (512: 10 of 16
  tiles at 128x128, 6 of 8 at 128x256, plus one mirror launch) against large ones (2304: 171 of 324).

### 8.6 The Muon caller, and your compiler line

- **On the preproduction path:** add your `symmetric_out` pass-through to `cuda_matmul` / `cuda_addmm` and use it in the
  one `newton_schulz_orthogonalize_batched`; then A/B test each of your ideas (symmetric, tall without transposes,
  fused scale + cast, apply from bf16) on the 135M and 1.6B shapes, same GPU, same steps, and keep each one where it is
  faster (symmetric may need a minimum size: on small matrices the mirror launch can cost more than the skipped tiles).
- **When your compiler line takes this Tensor-Implementations:** keep one `newton_schulz_orthogonalize_batched` (the
  combined one) and remove the other, or the build fails (section 8.3).

### 8.7 A third BluBLAS line

Kathir's `MoE-Trials` (3 commits on the same base ea666ab, +13,148 / −277 lines: grouped GEMMs for MoE, "Fixed
Stalling and optimized kernels by kota", a pad-buffer cache) conflicts with my line in
`Bgemm_sm90_warpspecialized_template.cuh` and `Hgemm_sm90_warpspecialized_template.cuh`, and touches two of your files
(`Bgemm_addmm_sm90.cu`, the same template). It is a separate merge, whenever it goes to `master`.

## 9. The merge, done and tested (2026-11-11 evening)

Following your "just merge and push, I'll check": the merge commit has two parents, a6d4292 and `master` e392dbb (the
merge of your cb42c13; the same files as cb42c13), and is resolved as section 8.5 describes. Because it descends from
`master`, `master` can fast-forward to it, and Tensor-Implementations branches that point to e392dbb (yours) or to it
(mine) merge their BluBLAS pointer without a conflict. Your switch is unchanged: off by default, on only where a caller
sets it (your `BlublasSymmetricScope` in Tensor-Implementations `coconut`). Records: `profiling_reports/blublas_symmetric_merge_2026_11_11/` in my repository
(the patch against a6d4292, the 8 resolved files, every test source, log and script).

### 9.1 With the switch off (everything that runs today): nothing changes

| check | result |
| --- | --- |
| machine code (laptop, the whole library, every GPU architecture) | 5,453 of 5,453 existing kernels byte-identical to a6d4292; new: your mirror kernel and the symmetric-list fill kernel |
| GEMM suite, 7,220 cases (bf16 / fp16 / fp32 x NN / NT / TN / TT x 21 shapes x batch layouts x leading dimensions x beta 0 with a NaN-filled C / 0.5, + gemv), H200 | 0 failed, 0 crashed with both libraries; bitwise equal except 222 fp32 split-K cases, and running a6d4292 twice gives the same 222 differences (atomics) |
| addmm suite (13,302 cases: BluBLAS direct, Tensor-Implementations addmm, per-batch bias vs one by one), H200 | all parts pass with both, identical result lines |
| the 84 GEMM shapes of the 135M SFT step (forward / dX / dW in NN / NT / TN + the batched Newton-Schulz calls), H200 | bit-identical in all 4 runs (each library twice) |
| 135M SFT trainer, 30 steps, 4 runs per library on one GPU | same losses; step time 104.8-105.0 ms vs 104.2-105.1 ms |
| RTX 3060 (sm_86) | 7,220 cases, 0 failed, bitwise equal except fp32 split-K |

### 9.2 With the switch on (your mode)

- **Correctness**, 75 cases on H200 (X Xᵀ, Xᵀ X up to K = 8192, the Newton-Schulz addmm B = c A@A + b A with a per-batch
  bias, beta 0 with a NaN bias, the fallback addmm, ragged n 200 / 320 / 1000 / 1408, a padded output; batch 1 / 5 /
  37): all pass. The lower triangle is bit-identical to the full GEMM; the upper triangle is an exact mirror; nothing
  left as NaN; every sample within the fp64 reference bound. In the 7 cases where the full GEMM is not exactly symmetric
  in its last bit (the B = c A@A + b A addmm), the mirror is visible and correct, so the per-batch bias and the symmetric
  mode work together.
- **The symmetric tile list** equals your builder's list on 1,160 grids (718,335 entries: same tiles, same order).
- **Speed** (H200, median ms per call, the same Newton-Schulz shapes and batch counts as the trainers):

| matrices | n | A = X Xᵀ, ON / OFF | B = b A + c A@A, ON / OFF |
| --- | --- | --- | --- |
| 1.6B q/o, 52 | 2304 | 0.706 | 0.687 |
| 1.6B gate/up, 52 (K 6144) | 2304 | 0.627 | 0.691 |
| 1.6B down, 26 (K 6144) | 2304 | 0.617 | 0.696 |
| 1.6B k/v, 52 | 768 | 0.873 | 1.039 |
| 135M q/o, 74 | 512 | 1.270 | 1.058 |
| 135M k/v, 74 | 256 | 1.376 | 1.293 |
| 135M gate/up, 74 (K 1408) | 512 | 0.992 | 1.057 |
| 135M down, 37 (K 1408) | 512 | 0.902 | 0.966 |

  OFF run twice gives 0.986-1.021, so differences beyond about 2% are real. Your mode saves 30-38% on the large
  products and costs up to 38% on the small ones (the mirror launch and the shorter tile list cost more than the
  skipped tiles there). So a caller should turn it on only for large n (here n = 2304; n = 768 is mixed); I will add
  that threshold when the flag goes into the Newton-Schulz on my line (section 8.6).

## Appendix A. Commit ids

- Common base: ea666ab (PR #3, "optimized addmm").
- My line (`dist_side_gautam_lottn`): 297f372, 0b552f0, a6d4292.
- Your line (`master`): cb42c13 "symmetric change" (Aparna-BB, 2026-11-05), e392dbb "Merge pull request #4" (2026-11-06).
- Tensor-Implementations pointers:
  - `dist_side_gautam_lottn` d59740b -> BluBLAS a6d4292;
  - `madhu_post-training` c61ea5d -> BluBLAS e392dbb;
  - `preproduction-135M` a9e37e6 -> BluBLAS ea666ab (the common base).
- The BluBLAS branch named `preproduction-135M` is at 35e51a0 (August).
- **After the merge (pushed 2026-11-11 evening, branch `dist_side_gautam_lottn` only):** BluBLAS cfa0e92 (parents
  a6d4292 and e392dbb); Tensor-Implementations 4d5c738 -> BluBLAS cfa0e92; BluTrain 597360b -> Tensor-Implementations
  4d5c738. `master`, `preproduction-135M` and `madhu_post-training` unchanged.

## Appendix B. How to repeat the trial merge (read-only, needs git >= 2.38)

```
git clone --bare --filter=blob:none https://github.com/BlubridgeAI/BluBridge-BLAS.git blas.git && cd blas.git
git merge-tree --write-tree --name-only a6d4292 e392dbb        # exit 1 = conflicts; lists the conflicted file
T=$(git merge-tree --write-tree a6d4292 e392dbb | head -1)
git show $T:src/level3/batched_gemm/BF16/SM90/Bgemm_sm90_warpspecialized_template.cuh | grep -n '^<<<<<<<'   # the 8 hunks
git diff ea666ab a6d4292 --stat ; git diff ea666ab e392dbb --stat
```
