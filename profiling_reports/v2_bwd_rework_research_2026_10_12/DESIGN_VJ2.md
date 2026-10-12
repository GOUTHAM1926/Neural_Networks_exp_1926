# Vj2: FA3-structured attention backward (Hopper, head dim 128) — design, 2026-10-12

Status: Stage 2 (dense mode only). Vj1d is NOT edited. Vj2 is opt-in with `USE_VJ2=1` until it has passed every test and is faster.

## Block layout (FA3 hdim-128 causal config, `flash_bwd_launch_template.h:351`)

- 384 threads = 3 warpgroups (WG = 128 threads). One block per (128-key block `kt`, query head `hq`, batch `b`).
- The grid is (ceil(S/128), Hq, B). `kt = blockIdx.x` puts the heaviest causal blocks first, as Vj1d does.
- **WG0, the loader group** (`setmaxnreg.dec 24`; FA3 uses 24 with 240, `flash_bwd_kernel_sm90.h:64-65`):
  - warp 0, lane 0: TMA of K and V once (2 × 32 KB). Then for each 64-query step, 2 stages ahead: TMA of Q and dO, plus a 1-D bulk copy of LSE·log2e and D (256 B each), all on one mbarrier `q_full[s]`. FA3 does the same with bulk copies of LSE and dPsum (`mainloop_bwd…:511-555`).
  - warp 1: the dQ writer. It waits on `dQ_full[c]`, issues the TMA reduce-add of compute group c's fp32 dQ, waits until the shared buffer has been read, then signals `dQ_empty[c]`. This is FA3's `store_dq` (`mainloop_bwd…:645-654`).
  - warps 2-3: nothing.
- **WG1 and WG2, compute groups c = 0, 1** (`setmaxnreg.inc 240`). Group c owns keys `k0 + 64c … k0 + 64c + 63`.

## One 64-query step, per compute group (FA3 order, `mainloop_bwd…:837-999`)

| # | operation | operands (smem SW128 unless said) | proven pattern copied from |
|---|---|---|---|
| 1 | Sᵀ = K_c · Qᵀ (m64n64, 8 k-steps over head dim) | A = sK rows 64c..64c+63, K-major; B = sQ[s], K-major | Vj1d S GEMM with A/B swapped (`GQA_bwd_vj1d_sm90.cu:189-200`) |
| 2 | dPᵀ = V_c · dOᵀ | A = sV rows 64c..; B = sdO[s] | same |
| 3 | P = ex2(Sᵀ·scale·log2e − LSE·log2e[query]); causal mask on the first two steps (key > query → 0) | 16 LSE values per thread from sLSE[s] | Vj1d `fused_p_stsm` arithmetic (`:259-269`) |
| 4 | dSᵀ = P · (dPᵀ − D[query]) in **fp32 P** | 16 D values from sD[s] | FA3 `:893`; precision gain shown in `RESEARCH.md` §4 |
| 5 | pack P and dS into bf16 A-fragments (16 regs each) | accumulator → A-fragment mapping | forward v62 `GQA_fwd_sm90.cu:636-643` (P·V) |
| 6 | stmatrix dSᵀ into sdS[s] rows 64c.. (row = key, 64 queries = one 128-B swizzled row) | | Vj1d `fused_p_stsm` stmatrix addressing (`:281-293`) |
| 7 | dV += Pᵀ · dO (register A, 2 halves of 64 head cols × 4 k-steps over queries) | B = sdO[s], MN-major (trans-b) | forward `wgmma_pv_half_x8` (`:305-330`) |
| 8 | `fence.proxy.async`, then `bar.sync 1, 256` (both groups' dS written) | | FA3 `:117-118` |
| 9 | dQ_c = dS · K[:, 64c..64c+63] (m64n64, 8 k-steps over 128 keys) | A = sdS[s] MN-major (trans-a); B = sK head-half c MN-major (trans-b) | Vj1d dK GEMM `wgmma_m64n64k16_tAtB` (`:361-376`) |
| 10 | dK += dSᵀ · Q (register A) | B = sQ[s] MN-major | same as 7 |
| 11 | `wgmma.wait_group 1` (dV and dQ done); `bar.sync dQ_empty[c]` (not on the first step); stage dQ·scale as fp32 into sdQ[c]; `fence.proxy.async`; `bar.arrive dQ_full[c]` | | Vj1d `store_acc_sw128_f32` (`:390-406`); FA3 `:953-957` |
| 12 | `wgmma.wait_group 0`; one thread per group arrives on `q_empty[s]` | | |

## Budgets

- **Shared memory:** sK 32 + sV 32 + sQ 2×16 + sdO 2×16 + sdS 2×16 + sdQ 2×16 KB = 192 KB. sLSE/sD 1 KB. With barriers and 1 KB for manual 1024-B alignment, about 194 KB, under the 227 KB limit.
- **Registers per compute thread:**
  - always live: dV 64 + dK 64;
  - during the softmax: S/P 32 + dP/dS 32 + 16 LSE values, then the 16 D values (loaded after P, so they never overlap);
  - peak about 208 plus addressing, under the 240 limit. ptxas must confirm 0 spills.
- **Named barriers:** 0 `__syncthreads` (only before the role split); 1 = both compute groups (256); 2/3 = `dQ_full[c]` (160 = 128 + writer warp); 4/5 = `dQ_empty[c]` (160); 6/7 = per-group epilogue (128).

## Correctness arguments

- **dS double buffer:** sdS[s] is rewritten 2 steps later. Both groups pass barrier 1 every step, so the other group's dQ GEMM of step i has finished before step i+2 writes the buffer. FA3 relies on the same argument (`mainloop_bwd…:92-95` in the fetched excerpt).
- **dQ handoff counts:**
  - each `dQ_full[c]` instance gets 128 arrives plus 32 syncs;
  - each `dQ_empty[c]` instance gets 32 writer arrives (after steps 0..n-2) plus 128 group syncs (before steps 1..n-1). Equal counts, no barrier is left pending at exit.
- **No deadlock:** the writer's progress on group c at step i needs only group c's arrive at step i. Group c's step i needs only the writer's arrive at step i-1 and the other group at barrier 1.
- **Causal mask on the first two steps:**
  - step 0 (queries k0..k0+63): group 0 is diagonal; group 1 (keys k0+64..) is fully masked.
  - step 1: group 0 needs no mask; group 1 is diagonal.
- **S % 128 == 64 is routed to Vj1d** (`GQA_bwd_generic.cu`, the `Vj2_Fa3` case). The forward's fast path needs S % 256 == 0 (`gqa_v62_shape_ok`), so training never sends such an S, and the forward+backward tests cannot reach Vj2's half-size last block. The handling below stays in the code but does not run until it can be tested.
- **S % 128 == 64 (if it were run):** the last block has one step. Group 1's keys are ≥ S, so they are fully masked by the causal rule (every key is after every query). Their K/V rows are loaded but never used, and their dK/dV are not stored.
- **Determinism:** unchanged from Vj1d. dK/dV are one-writer bf16 partials per query head, then the fixed-order head sum. dQ is an fp32 reduce-add (non-deterministic, as today).
- **Precision:** dS from fp32 P (item 1 of `RESEARCH.md` §4). The fp32 head sum (item 2) is a later, separate choice.

## Proof plan for Stage 2

1. Laptop: ptxas `-v` must show 0 spills, compute-group registers ≤ 240, shared memory as computed. Vj1d's SASS must be identical before and after (only the router file and the new file change).
2. GPU, after OK:
   - dense correctness on all 5 head setups and T = 1024, 2048, 4096, 8192, 4032;
   - error against fp64 next to Vj1d on the same inputs;
   - speed on the 13 dense shapes against Vj1d, FA3 and cuDNN.
3. Gate: at least 10% faster than Vj1d on every dense shape (provisional).

## Stage 2 laptop/box proofs (done 2026-10-12)

| check | result |
|---|---|
| ptxas, laptop CUDA 13.0 and box CUDA 13.4 | 0 spills, 0 stack. The first build had a 4-B spill of the warp number across `setmaxnreg`; fixed by computing it after the re-split. |
| ptxas warnings (`-warn-spills`, `-warn-lmem-usage`) | none |
| SASS per step | 40 HGMMA: 16 smem-A no-transpose (Sᵀ, dPᵀ), 8 smem-A trans-A/B (dQ), 16 register-A trans-B (dV, dK); 5 `WARPGROUP.ARRIVE`, 4 `DEPBAR` = exactly the 5 fences / 4 waits written; no ptxas-inserted syncs |
| untouched files | Vj1d deadc500, Vj1 26f6468c, forward 698d2705 / b71b6dc5: md5 identical |
| router file's WMMA kernels | 9,360 SASS instruction lines identical before/after. The previous router was rebuilt from the edits, md5 04255500 = the original. |
| harness | `bwd_mode_name` knows `USE_VJ2` ("Vj2 (USE_VJ2=1)"); G4 checks dK/dV bitwise for Vj2, as for Vj1d. Packed groups expect a refusal under `USE_VJ2=1`. |
| box | library rebuilt 19:53 (only the router and Vj2 recompiled); `tva_vj2` built with the exact `make run-cuda` line |

## Stage 2 GPU results (2026-10-12 evening, H200 GPU 7; logs in `gpu_runs/`)

**Correctness:**
- Smoke (quick CLASSIC): 144/144.
- Full CLASSIC (G1-G7, 5 head setups, G4 1000× repeat, G5 large inputs): **605/605**, the same 605 checks Vj1d passes. G9/G11 refuse packed rows under `USE_VJ2=1` as intended.
- G4: dK/dV bit-identical over all repeats (0 differing elements), as Vj1d.

**Precision against Vj1d on identical inputs** (fixed seeds; O/LSE/dV identical in all 89 checks; all 30 single-chat checks have identical reference floors). Worst ratio (error ÷ limit) per check, `precision_vj1d_vs_vj2.txt`:

| | Vj2 lower | Vj2 higher | equal | median Vj2/Vj1d | worst case Vj1d → Vj2 |
|---|---|---|---|---|---|
| dQ (84) | 57 | 16 | 11 | 0.91 | 1.156 → 0.891 |
| dK (84) | 61 | 16 | 7 | 0.91 | 0.834 → 0.874 |

The only KNOWN issue of the Vj1d run (G6 D10 row 1022 dQ, 1.156, caused by bf16 P in dS) does not occur under Vj2.

**Speed:** the harness BENCHDENSE ran in the order Vj1d, Vj2, Vj1d, Vj2, then FA3/FA4/cuDNN in the same session (`vj2_speed_1012/bwd_table.txt`). Backward per call, ms:
- Vj2 vs Vj1d: **5.2% to 13.0% faster** on all 13 shapes, mean −10.6%. The two rounds agree within 1-3%; the unchanged forward varies up to ±5% (noise check).
- Vj2 vs FA3: 1.20-1.28× (Vj1d was 1.32-1.43×).
- Vj2 vs cuDNN: 0.95-1.11× (Vj1d was 1.05-1.24× on 10-12).
- Vj2 vs FA3-deterministic: about ±4%.
- **Gate (≥10% on every shape): met on 9 of 13.** Missed on B2 12/4 (9.1%), B2 16/4 (9.7%), B2 24/8 (5.2%), B4 12/4 (8.3%).
- These are per-call times, which include the unchanged helper kernels (D/LSE prep, head sum, dQ convert, memset). The kernel-only gain is not measured yet; that needs nsys.
- Below the research range (−16% to −31% on the kernel). Why is not known yet; that needs ncu.

## Profiling of Vj2 v1 (nsys + ncu, dense B=8 18/6 T=4096; `gpu_runs/vj2_prof_1012/`)

**Kernel time (nsys median):**

| | main kernel | helpers |
|---|---|---|
| Vj1d | 3615 µs | 299 µs |
| Vj2 v1 | 3197 µs (−11.6%) | 313 µs |
| FA3 (10-12) | 2495 µs | 255 µs |

**Data movement now matches FA3:**
- L2 reads 42.7M (FA3 40.9M, Vj1d 81.0M);
- dQ reduce requests 38.9M (FA3 43.6M);
- L2 busy 48% (FA3 64%, Vj1d 79%);
- block-barrier stall ratio 0.71 (Vj1d 1.56).

**Remaining gap: instructions.** 1,096M executed, against FA3's 625M (Vj1d 888M). Two causes, found per SASS instruction:
1. Every shared-memory access except stmatrix was GENERIC: 0 LDS/STS, 19.5M `LD.E.64` (sLSE/sD), 25.4M `ST.E*` (dQ staging, epilogue). They drew `lg_throttle` stalls (0.41 per issue, zero in Vj1d/FA3), about 11% of samples. Cause: the shared base was rounded to 1024 B through `(uintptr_t)` and back, so the compiler lost the address space.
2. Descriptors were rebuilt from pointers for every wgmma: 5.4 uniform/R2UR instructions per HGMMA, 23% of all instructions.

**Fix (v2 of the file):**
- the base is rounded by pointer arithmetic on `smem_raw`;
- base descriptors are made once per block, and siblings come from constant 32-bit adds (`desc_add`, a copy of the forward's `desc_add_lo`).

SASS, same flags, before → after: 2520 → 1816 lines; `LD.E`/`ST.E` 16/145 → 0/0; `LDS`/`STS` 0/0 → 16/80; IMAD 219 → 89, LOP3 433 → 233, LEA 65 → 19, uniform-int 540 → 364; HGMMA 40 = 40; 0 spills. No arithmetic changed.
