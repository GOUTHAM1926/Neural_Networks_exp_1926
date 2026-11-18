# BluBLAS tile-order fix: what changed, why, who it helps, and how to take it

Branch `dist_side_gautam_lottn` of `BlubridgeAI/BluBridge-BLAS`, one commit, **297f372**, on top of `master`
(`ea666ab`). Tensor-Implementations `dist_side_gautam_lottn` (d18f820) and BluTrain `dist_side_gautam_lottn` (7184fe1)
already point to it.

## 1. In short

- A dense BF16 GEMM that got a tile grid the library had not seen recently made the **CPU wait for all queued GPU
  work** (about 20 ms per call on H200 when 20 ms of work was queued). I removed that wait: the kernels are unchanged,
  only the host code that prepares the "tile-order table" changed.
- Same fix for the SwiGLU-fused, cluster-NN and all SM103 (B300) templates, which had their own copies of the problem.
- Two more fixes in the same commit: BN=64 TN split-K is now **bit-deterministic** for even N, and the sm_80/86/89
  cores no longer **skip output tiles** for some shapes.
- Fixed shapes run at the same speed as before. Changing shapes (trained-rows logits, variable batch, MoE-style
  per-expert dense GEMMs, anything on B300) no longer stall.

## 2. The problem

Several BluBLAS kernels read a small table, `tile_map[mn_tile] = (tile_row << 16) | tile_col`, that says in which order
the blocks visit the output tiles (row-first, column-first or Hilbert). The host built it before the launch:

| template | old behaviour |
| --- | --- |
| SM90 warp-specialized (nn / nt / tn, addmm) | cache of 16 tables keyed by (grid_m, grid_n, hilbert), round-robin. A miss: `cudaMalloc` + plain `cudaMemcpy` (waits for **all** work queued on the GPU) + `cudaFree` once the 16 slots were full (also waits for the whole GPU) |
| SM90 SwiGLU fused | its own copy of the same 16-slot cache |
| SM90 cluster-NN persistent | one static table |
| SM103 cluster / addmm / dual / swap, FP8 SM103 cluster | ONE table per kernel type, rebuilt (free + malloc + blocking copy) whenever the grid differed from the previous call |

So every new tile grid cost a full GPU drain. With fixed shapes the cache filled at start-up and nothing happened later.
With changing M it missed again and again. Measured (H200, 20 ms of GPU work queued, then one GEMM with a new row
count): **20,011-20,074 us per call**. In the real 135M SFT trainer with the trained-rows switch on: 21 blocking table
uploads in 99 micro-batches, up to 22 ms each. On B300 any two GEMMs of one kernel type with different shapes stalled,
even in normal training.

## 3. What I changed

| file | change |
| --- | --- |
| `src/level3/batched_gemm/blublas_tile_order_tables.cuh` (new) | one helper: `static inline const int* blublas_tile_order_table(BlublasTileOrder order, int grid_m, int grid_n, cudaStream_t stream)` with `BlublasTileOrder::{Rows, Cols, Hilbert}` |
| `BF16/SM90/Bgemm_sm90_warpspecialized_template.cuh` | the 16-slot cache replaced by the helper (both call sites pass the GEMM's stream); new config `Sm90Config_128x64x64_ws` |
| `BF16/SM90/Bgemm_sm90_swiglu_fused_template.cuh`, `Bgemm_sm90_cluster_nn_persistent_template.cuh` | use the helper |
| `BF16/SM103/Bgemm_sm103_cluster_template.cuh`, `_cluster_addmm_template`, `_cluster_dual_template`, `_swap_template`, `FP8/SM103/Bgemm_sm103_fp8_cluster_template.cuh` | use the helper; the row-first table as fallback (SM103 kernels always need a table); a clear error print if even that fails |
| `BF16/SM90/Bgemm_sm90_tn_variants.cu` | BN=64 TN split-K: even N uses `Sm90Config_128x64x64_ws` (each split writes its own fp32 slice, a reduce kernel adds them in a fixed order) instead of `atomicAdd`. Odd N unchanged |
| `BF16/SM89/Bgemm_core_template.cuh`, `FP32/SM86/Sgemm_core_template.cuh` | the last swizzle group gets only the rows left (`g_first = (block_id / (grid_x * G)) * G; sw = min(grid_y - g_first, G)`) |

Every change has a comment at its place (what, why, date).

## 4. Why it is built this way

- **The key fact.** In row-first order, entry i is `(i / grid_n, i % grid_n)` whatever grid_m is. So **one table per
  (GPU, grid_n) serves every row count**: the GEMM reads only the first `grid_m * grid_n` entries. Column-first likewise
  depends only on grid_m. Only Hilbert needs the exact grid.
- **Growth.** Row-first / column-first tables are made for the next power of two rows / columns and replaced by one at
  least twice as long only when a longer grid comes: a few times per run.
- **No host copy.** A new table is written **on the GPU** by a small fill kernel on the GEMM's own stream, so the CPU
  never waits. A GEMM on another stream gets `cudaStreamWaitEvent` on the table's event (a GPU-side wait).
- **Never freed** while the process runs (kernels already queued may still read a replaced table); tables are small
  (2 tile columns x 1,024 rows = 8 KB).
- **Limits.** 512 tables and 256 MB per GPU. Past them the helper returns `nullptr`, and the caller falls back to its
  computed order (SM90) or to the row-first table (SM103). A fallback changes only the visit order, never a result.
- **Thread safe and multi-GPU safe:** a mutex, and the GPU id is part of the key (the old caches had neither).
- **`static inline` on purpose.** The header is included by files compiled for different architectures (SM90 files:
  sm_90a only; SM103 files: sm_100a/103a only). As a plain `inline` the linker keeps ONE copy for the whole library,
  the first in link order: in Tensor-Implementations' build that was an SM103 object, so on H200 the fill kernel had no
  code, every launch failed, and every GEMM did `cudaMalloc` + `cudaFree`. The standalone BluBLAS build linked an SM90
  object first, so library tests could not see it. With `static`, every file keeps its own copy and its own small
  cache. The error path now prints `[BluBridge] tile-order table not made (...)` once instead of falling back silently.
- **Rejected alternative:** computing the tall order inside the kernel (CUTLASS-style). It was measurably slower on
  normal shapes (`down` dX +3-4%, 128k tall shapes +0.3-0.8%, three H200s), so the kernels stayed as they were.

## 5. Tests and results (H200 unless said)

| check | result |
| --- | --- |
| kernels | SM90: byte-identical (`nvcc -cubin` md5) except the 3 new `_ws` kernels; SM103: all identical |
| speed, 59 shapes (135M trainer shapes at 16k / 128k rows, lm_head full and trained-rows, 8 P1 shapes, square and wide) | equal to the old library, except split-K dW: 16k qkv / o / down dW 11 / 15 / 10% faster, 128k o dW +1.3% |
| precision | 56 of 59 bit-identical; the other 3 are the split-K shapes, which were not deterministic in the old library itself |
| split-K determinism (9 dW shapes x 20 runs) | old: 2-20 different outputs; new: 1 |
| stall, new row count with 20 ms queued | 20,011-20,074 us -> 3.7-5.2 us |
| tables vs the old builders | identical in all 257 cases |
| real 135M SFT trainer (nsys, own CPU cores, steps 10-59) | switch off: identical to the old library (GPU busy, GPU idle, every CUDA call count; 18,300 fewer memsets). Switch on: 21 blocking copies -> 0, about 0.8 ms less GPU idle per micro-batch |
| sm_86 fix (RTX 3060) | normal shapes bit-identical and same speed; the broken shapes (the old library left up to 25M output elements unwritten) now match cuBLAS |
| library memory | 8 -> 16 MiB (the split-K workspace) |
| B300 | compile and code review only (no B300 available) |

## 6. Which GEMMs benefit

| GEMM | what changes |
| --- | --- |
| SM90 BF16 dense (nn / nt / tn / addmm) whose M changes between calls: trained-rows lm_head, variable-batch inference, MoE models that run some expert or shared-expert GEMMs as dense GEMMs with per-step token counts | no CPU stall on a new shape |
| SM90 SwiGLU-fused GEMM with changing M | the same |
| SM90 cluster-NN persistent | the same |
| everything on SM103 (B300), BF16 and FP8 | no rebuild on every shape change (this hit normal training too) |
| BN=64 TN split-K with even N (dW with small output and huge K) | bit-deterministic, and faster at 16k |
| BF16 and FP32 cores on A100 / RTX 30xx / RTX 40xx / Ada (sm_80/86/89) | correct results for shapes with tile rows > G and tile rows % G != 0 |
| fixed-shape training on H200 | no change (same kernels, same speed) |

## 7. For the MoE grouped GEMMs (branch `MoE-Trials`)

- The grouped kernels in `MoE-Trials` (`src/level3/grouped_gemm/...`) have their **own** tile map
  (`Ggemm_sm90_tile_map.cu`, designed so no host sync is needed). They do not use the dense table cache, so this
  commit does not change them.
- It still helps an MoE model wherever **dense** BluBLAS GEMMs see changing shapes (any expert or shared-expert path
  that falls back to the dense GEMM, the SwiGLU-fused path, router or gate GEMMs with a varying row count, dW split-K)
  and everything on B300.
- Lessons that apply to the grouped code too:
  1. Never put a plain `cudaMemcpy`, `cudaFree` or `cudaMalloc` in a per-call path: `cudaMemcpy` and `cudaFree` wait
     for the whole GPU, and `cudaMalloc` can too.
  2. A buffer that grows with `cudaFree` + `cudaMalloc` stalls once per growth. `Ggemm_sm90_tile_map.cu` grows its
     scratch that way (lines ~339-350 in `MoE-Trials`); growing by at least 2x, or reserving the maximum once, keeps
     that rare.
  3. A header helper that launches a kernel and is included by files built for different architectures must be
     `static inline`, or the linker may keep a copy with no code for the running GPU.
  4. Check the CUDA call counts with nsys in the real trainer (`nsys profile --trace=cuda`, then count `cudaMalloc`,
     `cudaMemcpy`, `cudaFree` in the steady state), not only kernel times: the link-order bug above changed no result
     and no kernel time, only the call counts.

## 8. How to take the changes

I merged `dist_side_gautam_lottn` into `MoE-Trials` as a dry run (`git merge-tree`, nothing pushed): **clean, no
conflicts**. The only file both branches touch is `Bgemm_sm90_warpspecialized_template.cuh`, in different places
(`MoE-Trials` changes the epilogue options, this commit the tile-map cache, its call sites and one new config alias);
the template-parameter order the new `Sm90Config_128x64x64_ws` alias relies on is unchanged in the merged file.

```bash
cd <your BluBridge-BLAS clone>
git fetch origin dist_side_gautam_lottn
git log --oneline -1 origin/dist_side_gautam_lottn      # 297f372
git checkout MoE-Trials                                  # or your working branch
git merge origin/dist_side_gautam_lottn                  # or: git cherry-pick 297f372
```

Then:

1. Build: `make -j$(nproc)` in BluBLAS, or inside Tensor-Implementations `make -j$(nproc) blas SM_ARCH=90`.
2. Run your grouped-GEMM tests as usual; results of every existing kernel should be bit-identical.
3. The line `[BluBridge] tile-order table not made` must never appear in stderr.
4. Optional: nsys of a few steps of your workload; in the steady state `cudaMalloc`, `cudaMemcpy` and `cudaFree`
   should be 0 (or a handful, once per table growth).

To use the helper in new code:

```cpp
#include "../../blublas_tile_order_tables.cuh"   // path relative to your file
const int* map = blublas_tile_order_table(BlublasTileOrder::Rows, grid_m, grid_n, stream);  // stream = your launch stream
if (map == nullptr) { /* fall back to the kernel's own order */ }
```

## 9. Not covered yet

- B300 is not tested on hardware.
- The FP16 core (`FP16/Hgemm_core_template.cuh`) still has the sm_8x tile-skip bug (FP16 is not used by the trainer).
- BN=64 split-K with odd N still uses `atomicAdd`.
- The first table of a new kind still calls `cudaMalloc`, which can wait (13 ms measured with a busy GPU, once per
  growth); one arena reserved at start-up would remove it.
- The full library test suite above ran on the build just before the `static inline` change; the trainer tests ran on
  the final build. The `static inline` change touches host code only.
