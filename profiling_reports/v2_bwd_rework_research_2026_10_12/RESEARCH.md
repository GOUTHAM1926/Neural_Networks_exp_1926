# Backward rework (FA3 structure): research before any decision — 2026-10-12

No code was changed. Evidence comes from:
- the 10-12 ncu captures (`../v2_step6_benchmarks_2026_10_12/ncu_6c/`, dense B=8, 18/6 heads, T=4096, H200);
- reading FA2 / FA3 / our kernel source;
- a CPU emulation of the rounding (this folder).

## 1. Where our backward loses time (measured, ncu)

| | ours Vj1d | FA3 | cuDNN |
|---|---|---|---|
| kernel time | 3.626 ms | 2.494 ms | 2.788 ms |
| thread blocks × threads | 9216 × 128 (2 per SM) | 4608 × 384 | 132 × 384 |
| registers per thread | 220 | 168 (launch value; rebalanced to 24/240 at run time) | 168 |
| instructions | 888 M | 625 M | 809 M |
| tensor-core busy | 45.9 % | 69.7 % | 61.9 % |
| tensor-core busy **time** (busy % × kernel time) | **1.66 ms** | **1.74 ms** | 1.73 ms |
| shared-memory store instructions (SASS) | 41.9 M | 10.9 M | 10.3 M |
| L2 read requests from the SMs | 81.0 M | 40.9 M | 41.9 M |
| L2 reduce (add-back) requests | 76.7 M | 43.6 M | 77.9 M |
| DRAM read | 0.71 GB | 0.98 GB | 0.72 GB |

- All three kernels spend about the same time on tensor-core math. **Ours is slower only because its tensor cores sit idle longer** (1.96 ms idle vs FA3's 0.76 ms).
- L2 reads are 1.98× FA3's. That matches the loop arithmetic: our 64-key blocks visit 299,520 (block, 64-query tile) pairs; FA3's 128-key blocks visit 152,064 (1.97×). Each visit reloads 32 KB of Q+dO and adds back 32 KB of dQ.
- cuDNN does as many dQ add-backs as we do (77.9 M) and is still 23% faster than us. So the add-back volume alone is not the main cost; **how the threads wait around it is.**
- We are not DRAM-bound: we read less from DRAM than FA3.

## 2. Where the stall samples go (ours, 408,487 samples, per SASS instruction)

| place in our code | share | why |
|---|---|---|
| `__syncthreads` + fences before the dQ add-back (`GQA_bwd_vj1d_sm90.cu:609-611, 621-627`) | ~14 % (MEMBAR 9.3 % + 0.5 %, FENCE 2.5 % + 1.4 %) | all 128 threads stop at one barrier; warps 0-1 arrive late because they are loading the next tile's LSE/D |
| waiting for the GEMMs to finish (`WARPGROUP.DEPBAR`, after the dK/dQ GEMMs) | 9.8 % | only one group of 128 threads, so nothing else can run while it waits |
| P and dS round trip through shared memory (fence + arrive before the dV and dK/dQ GEMMs) | ~7 % | P is written as bf16, made visible to the tensor cores, then read back |
| next tile's LSE/D (`issue()`, a global load then a shared store) | ~4 % plus part of the barrier wait above | the load latency is exposed in the main loop |
| dQ staging stores (fp32 to shared) | ~4 % | 32 KB per (block, q-tile), twice as often as FA3 |
| softmax math (FMUL/FFMA/MUFU/F2FP) | ~14 % | real work |

## 3. How FA3 avoids each of these (code)

- **128 keys per block, 2 compute groups of 128 threads, 1 loader group.** Head dim 128, causal, Hopper: `kBlockM=64, kBlockN=128, NumMmaWarpGroups=2` (`flash_bwd_launch_template.h:351`). Registers are rebalanced 24 (loader) / 240 (compute) (`flash_bwd_kernel_sm90.h:64-65, 212, 242`). This halves the Q/dO reloads and dQ add-backs per key.
- **Loads are asynchronous and ahead of time.** The loader warp brings Q, dO, LSE and D (as dPsum) by TMA / bulk copy into 2 pipeline stages (`mainloop_bwd_sm90_tma_gmma_ws.hpp:511-555`).
- **A separate dQ writer.** Each compute group writes its dQ to shared memory, fences, and signals without blocking (`:953-957`). A writer warp issues `SM90_BULK_REDUCE_ADD` and frees the buffer (`:645-654`; warp 1 of the loader group, `flash_bwd_kernel_sm90.h:231-238`).
- **P never goes through shared memory.** The swapped score GEMM leaves Pᵀ and dSᵀ in registers in the layout the dV and dK GEMMs read directly (`Mma_dKV_is_RS`, `:75`; the shared P buffer has size 0, `:279`). Only dS is written to shared memory, for the dQ GEMM, once per 128 keys (`:918-923`).
- **dS is computed from fp32 P** (`:868, 893`); the bf16 copy of P is made only afterwards (`:898-900`). FA2 does the same (`flash_bwd_kernel.h:591` before the bf16 copy at `:548`).

## 4. Precision: what FA3's arithmetic would change (emulated on CPU)

Files: `bwd_precision_sim.py`, `sim_T2048_rms.txt`. The fp64 truth is checked against finite differences (`check_truth.py`, error ≤ 1.5e-9). Setup: 3 query heads on 1 KV head, causal, T = 2048. Errors are given as multiples of the unavoidable error (the exact answer rounded to bf16).

RMS error, × floor (normal inputs, 2 seeds; large inputs q×5 k×3 v×3, 2 seeds):

| variant | dQ | dK | dV |
|---|---|---|---|
| today | 1.74 / 1.81 ; 3.50 / 3.57 | 2.05 / 2.07 ; 3.51 / 3.61 | 1.77 / 1.71 ; 1.59 / 1.59 |
| dS from fp32 P | 1.44 / 1.48 ; 3.36 / 3.42 | 1.79 / 1.79 ; 3.37 / 3.47 | unchanged |
| fp32 head sum | unchanged | 1.75 / 1.80 ; 3.35 / 3.46 | 1.41 / 1.40 ; 1.23 / 1.22 |
| both (= FA3 arithmetic) | 1.44 / 1.48 ; 3.36 / 3.42 | 1.45 / 1.47 ; 3.21 / 3.31 | 1.41 / 1.40 ; 1.23 / 1.22 |

- With normal inputs, the FA3 arithmetic cuts our RMS error by about 17–30%. With large inputs the cut is about 4–23%.
- Most of the remaining error (still 1.4–3.4× the floor) comes from rounding P and dS to bf16 before the GEMMs. FA2, FA3 and cuDNN all do that too.
- **Precision is a real but modest gain, not a reason on its own to rebuild.**
- The fp32 head sum is independent of the FA3 structure. It can be done in today's kernel; it costs 2× larger partial buffers.

## 5. What is proven vs. what is an estimate

**Proven:**
- the gap is idle tensor time;
- the 2× L2 reads come from 64-key blocks;
- the stall shares in §2;
- FA3's mechanisms in §3 (code);
- the precision effect in §4 (emulation).

**Existence proof:** FA3 reaches 2.49 ms with this structure on this GPU and shape (packed middle rows: main kernel 1.08 ms vs our 1.50 ms).

**Not proven: that our version would reach it.** Our forward v62 already uses the FA3 structure (1 loader + 2 compute groups, register rebalancing, FA3-style barriers), yet it is still 1.22× FA3's forward (1086 vs 886 µs, ncu). An honest range for a reworked backward:
- if it lands where our forward lands relative to FA3: ~3.0 ms dense (−16% vs today);
- if it matches FA3: 2.49 ms (−31%).

## 6. Options for the decision

1. **Full rework (new kernel beside Vj1d, FA3 structure).** Several days. Range −16% to −31% on the backward kernel, plus the precision gain from §4.
   - Gate: build the dense version first and measure it. Continue to the packed modes (B, A) only if it clearly beats Vj1d.
   - Vj1d stays as the fallback.
2. **Targeted fixes inside Vj1d** (load LSE/D with the TMA; fp32 head sum). Smaller. The gain is unknown and probably a few percent each.
   - It cannot fix the 2× reload or the single-group waiting. Adding a writer warp would break the 2-blocks-per-SM register budget: today 220 × 128 × 2 = 56,320 of 65,536 registers; with one more warp, 220 × 160 × 2 = 70,400 > 65,536.
3. **Leave it.** Measure attention's share of the SFT step first (deferred by the user).

## 7. Scope: which runs the backward kernel serves, and what a rebuild would and would not cover (2026-10-12)

**The trainer path:**
- Attention backward runs `AttentionBackward.cpp:237` → `gqa_flash_attn_backward_qkv`.
- On H100/H200 it takes our fast kernel (Vj1d today) when ALL of these hold (`GQA_bwd_generic.cu:697-698`; forward: `GQA_fwd_generic.cu:338`):
  - causal;
  - head dim 128;
  - T a multiple of 64;
  - an H100/H200;
  - `BLUTRAIN_NO_SM90[_GQA]` not set.
- The head ratio is NOT a condition. The only head rule is "query heads a multiple of KV heads" (`:679`), and G is a run-time number (`hkv = hq / G`).

| dimension | today | a rebuild |
|---|---|---|
| head ratio (G) | any G. The test suite covers 18/6, 12/4, 8/8 (MHA), 32/8, 8/1 (MQA) (`test_varlen_attention.cu:1371`); all passed 10-12. Speed gap vs FA3 measured on 13 shapes (G 3 and 4, 12-32 query heads, batch 2-8): 1.32-1.42× on every one (`dense_tables_1253.txt`) | same kernel for every G, no per-ratio tuning. FA3 tunes per head dim only (`run_mha_bwd_hdim128`, `flash_bwd_launch_template.h:347-361`); GQA only changes the dK/dV write (`:92-94`) |
| head dim | only 128 has the fast Hopper kernel | 128 only; 64 would need its own tuned version (FA3 has a separate config per head dim) |
| sequence length | T % 64 == 0 | must keep T % 64 (a half-size last key block), or it would refuse lengths that work today |
| batch size | gap vs FA3 grows with batch: 1.32× at B=2, 1.42× at B=8 | gains less on small problems; 128-key blocks halve the block count, so very small B·H·T can leave SMs idle (not measured) |
| packed SFT, short chats | ours beats FA3 here (backward: real shortest 1.171 vs 1.227 ms, all-512 1.142 vs 1.167) | 128-key blocks could LOSE on short chats; keep the 64-key path selectable per plan and measure |
| B300 | head dim 128 causal throws on B300, forward and backward (`GQA_bwd_generic.cu:756-759`) | not affected |

## 8. Finding outside the rebuild: causal head dim 64 on H100/H200 is silently NON-causal (pre-existing, confirmed by code)

- Forward: causal + head dim 64 + not B300 → `launch_gqa_v2`. Its own comment says "Non-causal" (`GQA_fwd_generic.cu:59`) and "gqa_v2 has no causal mask … on other GPUs it approximates by full attention" (`:279-286`). The `_qkv` path the trainer uses does the same, without the comment (`:374-387`).
- Backward: → `launch_gqa_backward<16,32,64>` (`GQA_bwd_sm103.cu:400`), which has no mask. Only the B300 V17 kernel masks (`:1169`).
- Reachable from the production trainer by configuration:
  - `bluscript.cpp:73` defaults `head_dim = 64`. The yaml sets 128 (`bluscript_config.yaml:14`), so today's production config is fine.
  - The guard `bluscript.cpp:588` and `Config.h:270` both say head dim 64 has a causal fallback; that is true only on B300.
- Not reached by the GPT-2 trainer: it uses `mem_efficient_attn_forward_tc` (`AttentionOps.cpp:117, 216`), not this path.
- Not yet proven on the GPU. A 1-minute test (head dim 64, causal, H200, against a masked reference) would prove it.
- Fix options: refuse it (throw), like head dim 128 already does; or give head dim 64 a causal kernel on Hopper.
