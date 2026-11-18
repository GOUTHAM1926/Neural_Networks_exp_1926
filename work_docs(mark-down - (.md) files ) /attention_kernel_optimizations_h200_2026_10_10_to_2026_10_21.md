# Attention kernel optimizations on H200, 2026-10-10 to 2026-10-21

Forward: v62 → QPU. Backward: Vj1d → Vj2. Every change I tried is listed, kept or not, with its date, its timings and the
reason for the decision. Part 4 explains each one in detail. Source of every number: the SFT work log
`sft_work_documentation.md` §40–§45.62 and the run folders named there.

---

## How everything was measured

- **GPU:** NVIDIA H200, head size 128, causal attention, bf16 inputs.
- **"Dense" shape:** the model's own attention: B = 8 rows × T = 4,096 tokens, 18 query heads / 6 KV heads (GQA, G = 3).
  Some races also used 13 dense shapes (B 2–8, heads 12/4 to 32/8).
- **"Packed" shape:** `real_rows_752`: 8 real packed SFT rows of typical chats, T = 4,096 (several chats per row, each
  attending only to itself).
- **Race:** two or more library builds run one after another in the same session, 6–12 rounds, free clocks (as in
  training). A % is the paired mean change against the previous best build (minus = faster). "8/8" = faster in 8 of 8
  rounds. t = paired t-statistic (|t| above ~3 = a real effect).
- **Correctness gate before every race:** the varlen attention test suite (`test_varlen_attention.cu`: quick 144 or 296
  checks, full 605). It includes "every chat in a packed row = the same chat run alone, bit for bit". A change that breaks
  this is rejected even if it is faster.
- **Scoreboards against FA3 / FA4 / cuDNN:** kernel time from nsys, free clocks, same box and session, steady state. A ratio
  is BluTrain time / rival time, so below 1 = BluTrain faster.
- **Power:** every run sits at the H200's 700 W cap. So a faster kernel often makes the NEXT kernel (unchanged code) a
  little slower in the same loop: the shared clock drops. Those small +0.1–0.5% moves of unchanged kernels are this effect.

---

## Part 1 — Backward: Vj1d → Vj2

Main kernel, dense: **3,626 µs → 2,406.7 µs (−33.6%)**; from 1.46× FA3's time to 0.97×.

| Variant name | Date | Change | Correctness | Speed result | Kept? |
| --- | --- | --- | --- | --- | --- |
| Vj1d (baseline) | 2026-10-12 (measured) | the old default backward: 64 keys per block, one warpgroup does loads, math and the dQ add-back | — | main kernel 3,626 µs; FA3 2,494 µs (1.46×) | — |
| Vj1d sync fix | 2026-10-10 | one `__syncthreads` so all warps see the first query tile's LSE values (a race) | 289/289 | 12/4 −0.04%, 18/6 +1.71% (t 2.0) = noise | **Yes** (correctness) |
| Vj2 (first version) | 2026-10-12 | new kernel in FA3's layout: 128 keys per block, loader + dQ-writer warps + 2 compute warpgroups, P/dS in registers, dS from the fp32 P | 605/605; dK/dV bit-identical every run | **5.2–13.0% faster than Vj1d** (mean 10.6%); kernel 3,615 → 3,197 µs | **Yes** |
| Vj2 fixed (memory path + descriptors) | built 2026-10-12, tested 2026-10-13 | shared memory used through real shared loads/stores (they had compiled to generic loads); GEMM descriptors built once | 605/605, same values | **18–22% faster than Vj1d** (mean 20%); kernel 2,761 µs; instructions 1,096M → 840M | **Yes** |
| Fix 1 — dQ path | 2026-10-13 | dQ in thread order with 128-bit stores + one 1-D bulk reduce-add per group (FA3's scheme) | 144/144, same values | **−5.7%** (13/13); 2,778 → 2,599 µs; bank conflicts 56.3M → 18.3M | **Yes** |
| Fix 2 — LSE order | 2026-10-13 | LSE loads moved between the S and dP GEMM issues | 144/144 | +0.3% (4/13 faster) = no change | No (solved later by G3) |
| Fix 4 — block order | 2026-10-13 | heaviest key blocks first, 8 (batch, head) pairs per L2 section | 144/144 | **−3.3%** (10/13; batch 2: −5 to −10%); 2,599 → 2,590 µs | **Yes** |
| Fix 5 — epilogue swizzle | 2026-10-13 | dK/dV staging in the 128-byte swizzle, swizzled TMA store | 144/144 | **−1.1%** (12/13); 2,590 → 2,545 µs; conflicts 18.3M → 1.79M | **Yes** |
| Fix 4 with 4-pair sections | 2026-10-13 | fix 4 with a 20 MB L2 budget | — | −0.6%, batch 8 flat | No |
| G2 — warp-uniform index | 2026-10-13 | warpgroup index broadcast with `__shfl_sync` | 144/144 | 0.0% (59M fewer instructions, same time) | Yes (neutral) |
| G1 — full-width dV/dK GEMMs | 2026-10-13 | dV and dK as one `m64n128k16` instead of 2 × `m64n64k16` | 144/144, bit-identical | −0.3%; 2,547 → 2,532 µs | **Yes** |
| G3 — LSE before the S GEMM | 2026-10-13 | the 8 LSE loads before the first S HGMMA | 144/144 | −0.6% (9/13); 2,532 → 2,512 µs; LSE wait 11.3% → 3.1% | **Yes** |
| B-2a — FA3's wait placement | 2026-10-14 | wait only for dV (not dQ) before starting the dK GEMM | 144/144 | **−1.9%** (13/13, t −8.7); 2,526 → 2,478 µs = FA3 (2,481) | **Yes** |
| G4 — dQ scale once | 2026-10-14 | dQ stored unscaled; scale applied once in the convert kernel | 144/144 | **−2.7%** (13/13, t −8.0); 2,478 → **2,406.7 µs** (FA3 2,481.4: **3.0% faster**) | **Yes** |
| B-4a — D loads early | 2026-10-14 | D loads moved next to the LSE loads, before the S/dP GEMMs | 144/144 | **+2.3% slower** (13/13) | **No, reverted** |
| B-1 — dQ zeroed in prep | 2026-10-14 | the fp32 dQ buffer zeroed inside the prep kernel (no separate memset) | 144/144 | own race never ran (no free GPU); expected ≤ −20 µs (−0.7%) | In the code |
| Vj2 packed rows (design A) + Vj2 default | 2026-10-15 (built, made default); 2026-10-16 (GPU proof) | Vj2 takes the per-chat plan; Vj2 replaces Vj1d as the default | G11 94/94; packed = alone bit-exact | **26% / 22% / 12% faster than Vj1d** on long / middle / short-chat rows | **Yes** |
| Mask fix v1 | 2026-10-16 | causal mask moved to its own block after the exps (it ran on every step) | 296/296 | instructions −12.5%; dense −0.08% (no change); **packed +0.58% slower** (0/12) | No |
| Mask fix v3 | 2026-10-16 | two copies of the exp loop behind one branch (masked copy only on the 2 diagonal steps) | 296/296 | instructions −12.4%; dense +0.11%, packed +0.22%; the forward after it −0.3% / −0.6% | Kept (user's call: neutral over forward + backward) |

---

## Part 2 — Forward: v62 → QPU

Baseline v62 (measured 2026-10-13 and 2026-10-16): 1.375× FA3's time dense, 1.28× packed (kernel only, free clocks).
Same clock (ncu): 1,084 µs vs FA3 877 µs; tensor cores busy 64% vs 83%.

### 2026-10-16

| Variant name | Date | Change | Correctness | Dense | Packed | Kept? |
| --- | --- | --- | --- | --- | --- | --- |
| G1 (forward) | 2026-10-16 | P·V as 8 full-width `m64n128k16` instead of 16 `m64n64k16` | 296/296 | **−3.34%** (12/12) | **−3.22%** (12/12) | **Yes** |
| F1 | 2026-10-16 | FA3's loop order: softmax of tile n while P·V(n−1) runs; removes a redundant QK | 296/296 | −0.50% (12/12) | **−1.82%** (12/12) | **Yes** |
| G5 | 2026-10-16 | dynamic work order: persistent scheduler, heaviest query pair first, L2 sections, work counter | 296/296 | **−2.63%** (12/12) | **−3.55%** (12/12) | **Yes** |
| F2 | 2026-10-16 | always rescale O (no vote/branch) + unit and stage broadcast (no per-HGMMA register moves) | 296/296 | **−3.84%** (12/12) | −1.54% (12/12) | **Yes** |
| F3 | 2026-10-16 | FA3's unit hand-over + FA3's 24/240 register split | **hung** (packed deadlock) | — | — | No |
| F3b | 2026-10-16 | F3 with the deadlock fixed | 296/296 | **+1.11% slower** (0/12) | −0.69% | No (replaced by F3c) |
| F3c | 2026-10-16 (raced again 2026-10-17) | F3's hand-over with the old 40/232 registers | 296/296 | +0.13% / −0.10% (no change) | **−1.55% / −0.84%** | **Yes** |
| E1 | 2026-10-16 | FA3's L2 cache hints (Q evict-first, K/V evict-last) on F2 | 296/296 | −0.17% | −0.12% | No (see C1) |
| C1 (F3c + E1) | 2026-10-17 | the two combined | 296/296 | −0.07% vs F3c (no change) | +0.05% vs F3c; the next backward **+0.18%** (0/12) | No |

Day total (chained races): dense 1.2398 → 1.1157 ms (**−10.0%**), packed 0.5750 → 0.5157 ms (**−10.3%**).

### 2026-10-17 (K-series)

| Variant name | Date | Change | Correctness | Dense | Packed | Kept? |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | 2026-10-17 | GEMM descriptors built once | — | no effect (ptxas rebuilt them in the loop) | — | No |
| K1 | 2026-10-17 | P·V wait pinned after the whole softmax + diagonal tile peeled + two P register sets + one K/V wait per tile | 296/296 | **−0.95%** (8/8) | 0.00% | Yes (base) |
| K1s | 2026-10-17 | K1 with one P register set | — | +0.94% worse | +2.81% worse | No |
| K1n2 | 2026-10-17 | K1 with 2 K/V stages (as FA3) | — | **+39% worse** | **+22% worse** | No |
| K2 | 2026-10-17 | O store through shared memory (needs 2 K/V stages) | — | — | — | No (dropped with K1n2) |
| K3h / K3q / K3e | 2026-10-17 | split softmax: 32 / 16 / 8 exps before the P·V wait | — | K3h −1.26%, K3q −1.45% (K3e not raced) | K3h −1.97%, K3q −0.86% | Folded into K5q |
| K4 | 2026-10-17 | O staged in the last tile's K buffer (fast epilogue, keeps 3 stages) | quick suite of K4h **hung** (handshake bug) | −5.30% | −7.33% | Replaced by K4b |
| K4b | 2026-10-17 | K4 with a counter-based handshake (cannot hang) | 296/296 | **−5.26%** (8/8) | **−7.37%** (8/8) | **Yes** |
| K4hb | 2026-10-17 | K4b + split 32 | 296/296 | −2.58% | **+8.38% worse** (spills) | No |
| K5 | 2026-10-17 | K4b + split 32 + one P register set | 296/296 | +0.84% | −4.63% | No |
| **K5q** | 2026-10-17 | K4b + split 16 + one P register set | 296/296 | −0.36% (7/8) | **−4.33%** (8/8) | **Yes (best of the day)** |
| K6 | 2026-10-17 | K5 + next unit's descriptor loaded during the last P·V | 296/296 | +0.95% | −4.20% | No |
| K5e | 2026-10-17 | split 8 | 296/296 | +1.22% worse | +0.39% | No |
| K7 / K7e | 2026-10-17 | FA4's conditional O rescale (/ + split 8) | 296/296 | −0.49% / −0.81% | +0.48% / +0.63% worse | No |
| K8 / K9 / K9h | 2026-10-17 | different softmax math per mode / polynomial exp | **failed** bit-exact packed = alone (278, 277, 274 of 296) | — | — | **No** |
| Knock-outs (KOexp, KOmax, KOres, KOall) | 2026-10-17 | measurement only: K5q without the exps / row max / O rescale / all three | wrong on purpose | −6.8% / −4.1% / −2.4% / −11.2% | −6.8% / −3.5% / −2.2% / −11.6% | — |
| W1 / W1s / W2 | 2026-10-17 | CUTLASS-style barrier wait / spin wait / separate K and V barriers | — | −0.07% / +0.14% / −0.11% | −0.04% / +0.44% / +0.10% | No |
| P8w / P4w | 2026-10-17 | every 8th / 4th exp by a polynomial | — | **+2.69% / +7.07% worse** | +3.40% / +7.44% worse | No |

Day total: K5q vs F3c about **−6.5% dense / −11.4% packed**.

### 2026-10-19

| Variant name | Date | Change | Correctness | Dense | Packed | Kept? |
| --- | --- | --- | --- | --- | --- | --- |
| Scoreboard with K5q (morning) | 2026-10-19 | — | — | 1.084× FA3 | 0.936× FA3 | — |
| L1b | built 2026-10-17, raced 2026-10-19 | Q descriptors built once, passed through a shuffle | 296/296 | +0.36% | +0.90% | No |
| R8 / R16 / R32 | 2026-10-19 | row-split softmax (exps of row 0 first) | 296/296 | +0.34% / +0.76% / +0.12% | −0.07% / 0.00% / +2.54% | No |
| KOkv | 2026-10-19 | measurement only: no K/V wait at all (upper bound for any K/V pipeline change) | wrong on purpose | −0.30% | −1.79% | — |
| SKV | 2026-10-19 | FA3's split K/V pipeline (separate K and V barriers and rings) | 296/296 | +0.20% | +0.87% | No |
| LH / LHO | 2026-10-19 | FA3's L2 hints (+ O stores evict-first), forward-only race | 296/296 | −1.18% / −1.49% | +0.56% / +0.68% worse | No (packed worse) |
| LHOd | 2026-10-19 | the hints in the dense kernel only (packed = K5q) | 296/296 | −0.34% (forward + backward loop; the backward also −0.33%) | 0.00% | **Yes** |
| SM3 | 2026-10-19 | exps spaced like FA3 (row by row, max of the next row between exps) | 296/296 | **−3.05%** (8/8) | +0.45% | Replaced by SM3m |
| SM3m | 2026-10-19 | SM3 + FA3's causal-mask order | 296/296 | **−3.51%** (8/8) | **−4.48%** (8/8) | **Yes** |
| PF | 2026-10-19 | first K/V tile loaded before waiting for Q to be free (FA3's producer order) | 296/296 | **−2.60%** (8/8) | **−3.01%** (8/8) | **Yes** |
| PFT | 2026-10-19 | O written by TMA store (FA3's epilogue) | 296/296 | **+1.60% slower** (0/8) | 0.00% | No |
| QP | 2026-10-19 | Q prefetched into L2 when a pair starts | 296/296 | +0.08% | −0.24% | No (alone) |
| **QPU** | 2026-10-19 | QP + the producer publishes the decoded unit | 296/296 | **−0.81%** (7/8) | −0.13% | **Yes (final best)** |
| QPD | 2026-10-19 | QP + diagonal tile first | 296/296 | +0.32% | −0.43% | No |
| QPUD | 2026-10-19 | all three | 296/296 | −0.31% | −0.44% | No |

---

## Part 3 — Where it ended (scoreboard 2026-10-19 evening, BluTrain = QPU forward + Vj2 backward)

| BluTrain / rival (below 1 = BluTrain faster) | dense fwd | dense bwd | dense fwd+bwd | packed fwd | packed bwd | packed fwd+bwd |
| --- | --- | --- | --- | --- | --- | --- |
| FA3 | **1.004** (984.2 vs 980.4 µs) | 0.968 | 0.976 | 0.865 | 0.953 | 0.933 |
| FA4 | 0.919 | 0.989 | 0.972 | 0.781 | 0.971 | 0.923 |
| cuDNN | 0.863 | 0.891 | 0.884 | 0.827 | 0.899 | 0.883 |

- BluTrain wins 11 of 12 matchups. The one left is the dense forward vs FA3: a tie (+0.4%).
- Forward vs FA3: **1.375× → 1.004×** (dense), **1.28× → 0.865×** (packed).

**In the production pretraining trainer (2026-10-21, 1x H200, ported into `dist_side_gautam_lottn`):**

- **Attention alone vs BLUCCL's old kernels:** forward **+18.9%** faster, backward **+40.5%** faster.
- **Whole step**, together with the loss-kernel fixes ported in the same push: **−5.98%** step time, **+6.4% tok/s**
  (45,936 → 48,855), MFU 51.1% → 54.3%.
- **Loss values:** step 0 differs by 4e-6 (the forward's rounding); later steps stay within the old kernels' own
  run-to-run noise.

**SFT packed-rows design, for reference (2026-10-10 to 2026-10-15):**

- The per-chat list (cu_seqlens style, design A) beat one-number-per-token (seg_start, design B): forward **2–16% faster**,
  backward **3–5% faster** on middle and long chats. Design B was removed on 2026-10-15.
- Packed attention costs about half of dense on typical rows, and about a third on short-chat rows.

---

## Part 4 — Each optimization explained

Each entry: **date**, the **state before**, **what changed and how**, **why**, the **result**, and the **decision with its reason**.

### 4.1 Backward

#### Vj1d (baseline)

- **Date:** measured 2026-10-12 (the default backward since before October).
- **State:** 64 keys per CTA; one warpgroup issues the loads, does the softmax math, the GEMMs and the dQ add-back; a block-wide
  `__syncthreads` plus fences before every dQ reduce; P makes a round trip through shared memory.
- **Measured (ncu, dense):** main kernel 3,626 µs vs FA3 2,494 µs and cuDNN 2,788 µs. Tensor-core busy time is the same in all
  three (1.66 / 1.74 / 1.73 ms), so the whole gap is time with idle tensor cores. L2 reads 81.0M vs FA3 40.9M (64-key blocks
  visit the query tiles twice as often: 299,520 vs 152,064 visits). Instructions 888M vs 625M. Stall samples: ~14% at the
  fences before the dQ add-back, ~10% waiting for GEMMs, ~7% the P/dS shared-memory round trip.
- **Conclusion that day:** small fixes could not close a 1.46× gap; it needed FA3's structure (a new kernel, Vj2).

#### Vj1d sync fix

- **Date:** 2026-10-10.
- **State:** in Vj1d (and Vj1) threads 0–63 write the first query tile's LSE values into shared memory, and all 4 warps read
  them before any barrier: a race (the only wait in between was the TMA barrier for Q/dO).
- **What / how:** one `__syncthreads()` after the first `issue(0)`, once per CTA, not per tile. SASS: +1 `BAR.SYNC` only.
- **Why:** correctness, not speed.
- **Result:** quick suite 289/289; 1,000 repeats bit-identical for O/LSE/dK/dV. Overhead in a 12-round race: 12/4 heads −0.04%,
  18/6 +1.71% (t 2.0, within noise).
- **Decision:** kept (approved by the user): a correctness fix with no measurable cost.

#### Vj2 (first version)

- **Date:** 2026-10-12.
- **State before:** Vj1d as above, 1.46× FA3.
- **What / how:** a new file `GQA_bwd_vj2_sm90.cu` in FA3's layout: 128 keys per CTA and 384 threads (3 warpgroups). Warpgroup 0
  is the loader (warp 0 issues TMA loads of K/V once and Q/dO/LSE/D two steps ahead; warp 1 is the dQ writer). Two compute
  warpgroups own 64 keys each; `setmaxnreg` 24/240. The score GEMMs are "swapped" (Sᵀ = K·Qᵀ), so P and dS stay in registers
  for the dV and dK GEMMs; only dS goes to shared memory (for dQ). dS is computed from the fp32 P. The compute groups hand dQ
  to the writer warp with named barriers (no block-wide barrier in the loop). Kept from Vj1d: TMA maps and swizzle, the fp32 dQ
  reduce-add, the one-writer bf16 dK/dV partials with the fixed-order head sum (so dK/dV stay bit-identical), the dQ convert.
- **Why:** FA3's source shows exactly this layout; 128 keys need ~256 registers for dK/dV in one warpgroup (over the 255 limit),
  which is why the bigger block needs 3 warpgroups.
- **Result:** full suite 605/605; dK/dV bit-identical over all repeats; precision better than Vj1d (dQ lower error in 57 of 84
  checks, median 0.91×). Speed: 5.2–13.0% faster than Vj1d on 13 shapes (mean 10.6%); kernel 3,615 → 3,197 µs (FA3 2,495).
- **Decision:** kept (opt-in at first with `USE_VJ2=1`), and profiled at once.

#### Vj2 fixed (memory path + descriptors)

- **Date:** fix written 2026-10-12, tested 2026-10-13 (file md5 `bbd977be`).
- **State before:** first Vj2, 1.28× FA3. ncu: data movement already matched FA3 (L2 reads 42.7M vs 40.9M), but instructions were
  1,096M vs 625M.
- **Why (two causes found in SASS):** (1) every shared-memory read/write except stmatrix compiled to GENERIC loads/stores (0 LDS/STS;
  ~11% of stall samples on `lg_throttle`): the base pointer was rounded to 1,024 B through an integer cast, so the compiler lost the
  address space. (2) GEMM descriptors were rebuilt from pointers for every wgmma: 5.4 extra instructions per HGMMA, 23% of all.
- **What / how:** round the base by pointer arithmetic; build base descriptors once and get siblings by constant adds. SASS
  2,520 → 1,816 lines; generic LD/ST 161 → 0; LDS/STS 0 → 96.
- **Result:** 605/605, all 435 error values identical to the first Vj2 (no math change). 18–22% faster than Vj1d (mean 20%);
  1.05–1.15× FA3; faster than cuDNN on 12/13 shapes. Kernel 2,761 µs (nsys); instructions 1,096M → 840M; tensor busy 51.4% → 59.6%.
- **Decision:** kept.

#### Fix 1 — dQ path (Phase 1.5)

- **Date:** 2026-10-13.
- **State before:** fixed Vj2 (2,778 µs ncu). Line-by-line against FA3: same shared loads, but shared STORES 21.8M vs 10.9M and
  store bank conflicts **56.3M vs 1.8M**; tensor busy 59.6% vs 68.8%.
- **Why:** each thread wrote 16 `STS.64` per step in the SW128 layout of a 2-D TMA reduce; in each half-warp rows r and r^1 hit the
  same 8 banks (4 wavefronts instead of 2; checked exactly: 96,731,136 wavefronts = ncu's number). FA3 writes dQ in thread order
  with 128-bit stores and sends 16 KB with one 1-D bulk reduce-add.
- **What / how:** FA3's scheme: 8 `STS.128` per thread in thread order, one 1-D `cp.reduce.async.bulk` per group, a global fp32
  accumulator in fragment order, and a new convert kernel `convert_dq_frag_vj2` (padded shared memory, coalesced writes).
- **Result:** 144/144, identical values; −5.7% (faster on 13/13); 2,778 → 2,599 µs; conflicts 56.3M → 18.3M; tensor busy 65.2%.
- **Decision:** kept.

#### Fix 2 — LSE order

- **Date:** 2026-10-13.
- **Why:** the first FFMA that uses each loaded LSE value stalled (9.7% of samples): the LSE arrived late.
- **What / how:** LSE loads moved between the S and dP GEMM issues.
- **Result:** 144/144; mean 1.003 (faster on 4/13 only); the wait did not move.
- **Decision:** removed. The problem was solved later by G3 (loads before the first S GEMM).
- (Cause 3 of the Phase 1.5 list, the dQ hand-off wait, got no separate fix in the record.)

#### Fix 4 — block order

- **Date:** 2026-10-13.
- **State before:** SMs busy 96.9% of elapsed cycles (FA3 98.6%): a tail at the end of the launch.
- **Why:** FA3 uses `SingleTileBwdLPTScheduler`: heaviest key blocks first, 8 heads per L2 section.
- **What / how:** a 1-D grid, heaviest key blocks first, 8 (batch, head) pairs per L2 section (FA3's sizing: 40 MB / 4 MB per head);
  a script checked on 336 shapes that every block runs exactly once.
- **Result:** 144/144 identical; 0.904–1.010, mean −3.3%, faster on 10/13 (batch 2: −5 to −10%; batch 8 about flat); SMs busy
  96.9% → 98.8%. Side effect: dQ's run-to-run differences grew (G4 B8 max 4.9e-4 → 2.0e-3; limit 1.56e-2), because more
  blocks add into the same dQ tile at once; dK/dV stay bitwise.
- **Decision:** kept.

#### Fix 5 — epilogue swizzle

- **Date:** 2026-10-13.
- **Why:** the bf16 dK/dV staging wrote plain 128-byte rows: 8-way bank conflicts (2.2% of samples).
- **What / how:** stage dK/dV in the 128-byte swizzle; dK/dV TMA store maps set to `SWIZZLE_128B`.
- **Result:** 144/144 identical; mean −1.1%, faster on 12/13; 2,590 → 2,545 µs; store conflicts 18.3M → 1.79M (FA3 1.81M).
- **Decision:** kept.

#### Fix 4 with 4-pair sections

- **Date:** 2026-10-13.
- **What:** fix 4 with a 20 MB L2 budget (4 pairs per section) instead of 8.
- **Result:** mean −0.6%, batch 8 flat; DRAM 745 → 724 MB.
- **Decision:** not kept (no real gain over 8 pairs).

#### G2 — warp-uniform warpgroup index

- **Date:** 2026-10-13.
- **Why:** GEMM descriptors were built in per-thread registers: R2UR 59M in the backward (FA3 2M), 40.2M in the forward (FA3 0.7M).
- **What / how:** `c = __shfl_sync(0xffffffff, threadIdx.x / 128, 0) - 1` in v62 and Vj2 (FA3's `warp_uniform`), so ptxas
  knows the index is the same for the whole warp.
- **Result:** 144/144 identical. Backward R2UR 59.1M → 0.66M, IMAD 74M → 41M, but uniform-datapath instructions 137M → 212M and the
  time unchanged (2,545 → 2,547 µs); forward −1% (noise).
- **Decision:** kept at the user's request (a correct clean-up with no cost). Lesson: neither kernel is limited by instruction
  issue; the time is lost while tensor cores wait.

#### G1 — full-width dV/dK GEMMs (backward)

- **Date:** 2026-10-13.
- **Why:** the register-A GEMMs for dV and dK ran as 2 × `m64n64k16` per step (16 instructions); FA3/FA4/cuDNN use one `m64n128k16`.
- **What / how:** dV and dK as one `m64n128k16` per k-step; B descriptor LBO = 8,192 B (CUTLASS `Major::MN` B128 layout).
- **Result:** 144/144 identical (dK/dV bitwise); −0.3%; 2,547 → 2,532 µs; 8 + 24 HGMMA per step = FA3's mix.
- **Decision:** kept.

#### G3 — LSE before the S GEMM

- **Date:** 2026-10-13.
- **Why:** the exp FFMAs waited for LSE 11.8% of samples vs FA3 5.5%; FA3 issues its LSE loads right after its first S HGMMA.
- **What / how:** the 8 LSE `LDS.64` placed before the first S HGMMA.
- **Result:** 144/144 identical; −0.6% (faster on 9/13); 2,532 → 2,512 µs; LSE wait 11.3% → 3.1%; tensor busy 67.0%.
- **Decision:** kept. After G3 the largest wait is on the tensor cores themselves ("warpgroup arrive" 16.8%).

#### B-2a — FA3's wait placement between the dQ and dK GEMMs

- **Date:** 2026-10-14.
- **State before:** g123 (G1 + G2 + G3), main kernel 2,526 µs on that day's GPU; warps waiting at the dK GEMM start 215.5
  µs-equivalent (FA3 141.1).
- **What / how:** one `wgmma_wait1()` (dV done, dQ still running) before the dK GEMM is sent; the wait after dK now waits for dQ only.
  FA3 does the same (`mainloop_bwd_sm90_tma_gmma_ws.hpp:941-951`).
- **Result:** 144/144, every error value identical. Race: −1.9% (t −8.7, faster on 13/13); forward + backward −1.4%. ncu: 2,526.3 →
  **2,478.0 µs**, FA3 2,481.4 = equal. The dK-start wait 215.5 → 148.2 (FA3 141.1), as predicted.
- **Decision:** kept.

#### G4 — dQ scale applied once

- **Date:** 2026-10-14.
- **Why:** the backward ran 80.3M FMUL vs FA3's 41.3M. The extra 39M were the dQ scale: every step multiplied all 32 dQ values of
  every thread by the softmax scale. FA3 stores dQ unscaled and multiplies once in its convert kernel
  (`flash_bwd_postprocess_kernel.h:213`).
- **What / how:** `store_acc_frag_f32` stores dQ unscaled; `convert_dq_frag_vj2(in, out, scale)` multiplies each value once before the
  bf16 conversion. Main kernel FMUL 129 → 97 per step.
- **Result:** 144/144 (error values identical; dQ run-to-run differences got smaller). Race: −2.7% (t −8.0, faster on 13/13); B8 18/6
  3.221 → 3.119 ms. ncu: 2,478.0 → **2,406.7 µs**, **3.0% faster than FA3** (2,481.4); float instructions now equal to FA3's.
- **Decision:** kept.

#### B-4a — D loads before the GEMMs

- **Date:** 2026-10-14.
- **Why:** after G4 the largest single stall was the first dS FADD waiting 85 µs-eq for a D value; G3 had gained by moving the LSE
  loads early, so the same move was tried for D.
- **What / how:** the 8 D loads moved next to the 8 LSE loads, before the S/dP GEMM issue.
- **Result:** 144/144; **+2.3% slower on 13/13** (B8 18/6 3.041 → 3.095 ms). ncu: the D wait vanished, but one warpgroup now reached
  the dS barrier much earlier and waited there (+166), the 16 loads congested the shared-memory queue (+90), the dP wait grew (+51).
- **Decision:** reverted. The time only moved somewhere else.

#### B-1 — dQ buffer zeroed inside the prep kernel

- **Date:** 2026-10-14.
- **Why:** nsys of one backward call: a separate memset of the fp32 dQ buffer (67.9 µs) + the prep kernel (101.8 µs) = 169.7 µs vs
  FA3's single preprocess that also zeroes dQaccum (149.2 µs).
- **What / how:** `bwd_prep_vj2` also zeroes the dQ accumulator; no separate `cudaMemset`.
- **Result:** quick suite 144/144. Its own speed race (`b1_1014/t2_speed.sh`) never ran (no free GPU); expected at most −20 µs
  per call (~0.7%). The later nsys (2026-10-16) shows the helpers at parity with FA3 (369 vs 374 µs).
- **Decision:** it is in the current code.

#### Vj2 packed rows (design A) + Vj2 as the default

- **Date:** built and made the default 2026-10-15; proven on the GPU 2026-10-16.
- **State before:** packed SFT rows still ran on Vj1d (Vj2 refused them), 1.18× FA3 varlen.
- **What / how:** Vj2 takes the per-chat plan: blocks of 128 keys per (chat, query head), cut from each chat's start, plus the
  padded dQ tiles; `USE_VJ2=0` selects Vj1d.
- **Result:** G11 94/94 (every segment = the same segment alone; isolation holds; 50 repeats bit-identical O/LSE/dK/dV). Backward on
  real packed rows: 2.90 / 1.47 / 1.02 ms (long / middle / short) vs Vj1d's 3.90 / 1.90 / 1.16 ms (2026-10-12, another box): about
  26% / 22% / 12% faster. End-to-end SFT run: step-0 loss identical to the unpacked v1 run (1.445038).
- **Decision:** kept; Vj2 is the default backward.

#### Mask fix v1 (causal mask in its own block)

- **Date:** 2026-10-16.
- **Why:** ncu showed FSEL = 38.9M = exactly one per exp: the causal mask ran on EVERY step, although only the 2 diagonal steps of
  each key block need it (`if (diag)` inside the unrolled loop became predicated code).
- **What / how:** the exps run with no mask code; then `if (diag)` one block of 73 instructions applies the mask (only on the 2
  diagonal steps). FA3's two-loop way was tried first and dropped: ptxas spilled 128 B per step.
- **Result:** 296/296. Instructions really dropped (705M → 617M, −12.5%; FA3 626M), but cycles did not (+0.3% dense, +1.4% packed).
  Race: dense −0.08% (no change), **packed +0.58% slower in all 12 rounds**: on the diagonal steps the mask used to hide in the exps'
  latency; now it runs after them, and packed rows have 30–40% diagonal steps.
- **Decision:** not kept (packed is the SFT path).

#### Mask fix v3 (two exp copies)

- **Date:** 2026-10-16.
- **What / how:** two copies of the exp loop behind one branch: diagonal steps run the original interleaved exp + mask code; all other
  steps run a plain 65-instruction exp loop.
- **Result:** 296/296; instructions −12.4%; dense backward +0.11% (t 1.8), packed +0.22% (t 4.8); the forward timed after it −0.30% /
  −0.62% (less power drawn by the backward).
- **Decision:** kept by the user (fewer instructions; a wash over a forward + backward pair). Lesson: this kernel is not issue-bound;
  instruction cuts alone convert to almost no time.

### 4.2 Forward

#### v62 (baseline)

- **State (2026-10-13 / 2026-10-16):** persistent kernel, 132 CTAs; each CTA = one loader warpgroup + 2 compute groups of 64 query rows
  (128 × 128 tiles, as FA3); 3 K/V stages; static work split (CTA i takes units i, i + 132, ...); P·V as 16 half-width GEMMs per
  key tile; O written with 64 four-byte global stores per thread.
- **Measured:** dense kernel 1.202 ms vs FA3 0.874 (free clocks: 1.375×); packed 0.554 vs 0.432 (1.28×). Same clock: 1,084 vs 877 µs;
  tensor busy 64% vs 83%; SMs busy 95.7% vs 99.6%. The real difference was the loop order: BluTrain waited for QK(n), ran the softmax
  with nothing on the tensor cores, then started P·V(n); FA3 runs the softmax of tile n while P·V(n−1) is still on the tensor cores.

#### G1 (forward) — full-width P·V

- **Date:** 2026-10-16.
- **What / how:** P·V as 8 `m64n128k16` per key tile (new `wgmma_pv128_x8`, V descriptor LBO 16,384 B) instead of 16 `m64n64k16`. The
  tile does not change; the accumulator layout equals the two old halves side by side, so the softmax and epilogue are unchanged.
- **Why:** FA3 / FA4 / cuDNN all use the full width; the same change in the backward on 2026-10-13 was bit-identical.
- **Result:** 296/296; dense −3.34% (1.2398 → 1.1972 ms, 12/12, t −61.8), packed −3.22% (0.5750 → 0.5564 ms, 12/12). ncu: cycles −4.0%,
  instructions 349M → 326M, tensor busy 64.1% → 66.6%.
- **Decision:** kept.

#### F1 — FA3's loop order

- **Date:** 2026-10-16.
- **What / how:** per key tile n ≥ 1: wait K/V(n); baton; start QK(n) and P·V(n−1); baton to the other group; wait for QK(n) only;
  softmax(n); wait for P·V(n−1); release tile n−1; rescale O and pack P(n). The math and its order are unchanged. The old redundant
  QK after the last key tile is gone.
- **Why:** to overlap the softmax with the previous P·V, as FA3 (IntraWGOverlap).
- **Result:** 296/296; dense −0.50% (12/12), packed −1.82% (12/12). ncu: dense cycles unchanged; the gain is mostly the removed
  redundant QK (bigger share in packed rows, which have few key tiles per query tile).
- **Decision:** kept (faster on both, correct).

#### G5 — dynamic work order

- **Date:** 2026-10-16.
- **Why:** 2,304 equal units over 132 SMs = 17.45 rounds: 60 SMs did 18 units, 72 did 17 (SMs busy 95.7% vs FA3 99.6%). A simulation
  of 132 SMs gave slowest/average 1.031 (static) vs 1.002 (single pairs, heaviest first, dynamic).
- **What / how:** FA3's persistent scheduler for causal: a unit = one query pair (128 rows), heaviest first, in sections of (batch,
  head) pairs (FA3's L2 rule), taken through a work counter.
- **Result:** 296/296, no hang; dense −2.63% (12/12), packed −3.55% (12/12); SMs busy 96.7% → 98.6% (dense), 94.4% → 97.7% (packed).
  Cost found: dense L2 hit rate 92.2% → 83.0%, DRAM reads 254 → 526 MB (a 48-head section is too big for the L2).
- **Decision:** kept.

#### F2 — always rescale O + broadcast the unit and stage

- **Date:** 2026-10-16.
- **Why (instruction-level comparison with FA3):** about 75% of the remaining gap was the softmax's instruction ORDER: a vote +
  branch skipped the O rescale when no row max changed, so the exps ran back to back and 29 packs came in a row at the end. Also the
  K/V stage index and (after G5) the unit number sat in vector registers, so every HGMMA needed its descriptor moved over first (32
  R2UR + ~30 moves per tile).
- **What / how:** (a) O rescaled on every tile, as FA3 (O *= 1.0 changes nothing); (b) the unit number and stage index broadcast from
  lane 0 with `__shfl_sync`, so ptxas knows they are warp-uniform. SASS: R2UR in the issue phase 32 → 2 per tile; exps now interleaved
  like FA3's.
- **Result:** 296/296; dense −3.84% (12/12), packed −1.54% (12/12). ncu: cycles −3.8%, tensor busy 67.6% → 70.8%; 1.18× FA3 at the
  locked clock.
- **Decision:** kept. Day total after F2: dense −10.0%, packed −9.8%.

#### F3 / F3b — FA3's unit hand-over + 24/240 registers

- **Date:** 2026-10-16.
- **What / how:** (A) the producer publishes the next unit as soon as the current unit's loads are issued; the consumers read it right
  after their last P·V, before the epilogue (FA3's position). (B) `setmaxnreg` 24/240 (FA3's split).
- **Why:** the unit start was a dependent chain (shared load → shuffle → stage shuffle) before the K/V wait could start.
- **Result:** F3 **hung** in packed rows: a group whose query tile lies past its chat's end fetched the next unit before its inactive
  tail, where it releases this unit's K/V stages; the producer waited for those releases (deadlock). Proven on the CPU with a new
  protocol model (`protocol_sim.py`): 1,124 deadlocks in 3,000 random unit lists for F3, 0 for G5 and for the fixed F3. F3b (fixed):
  296/296, but dense **+1.11% slower** (0/12), packed −0.69%.
- **Decision:** not kept. New rule from this: every change to the hand-over or barriers runs the CPU protocol model before any GPU run.

#### F3c — the hand-over with the old 40/232 registers

- **Date:** 2026-10-16 (raced again 2026-10-17).
- **Why:** to separate F3b's two parts: was the dense loss from the 24-register loader?
- **Result:** 2026-10-16 race: dense +0.13% (no change), packed **−1.55%** (12/12). 2026-10-17 race on another box: dense −0.10%,
  packed −0.84% (12/12).
- **Decision:** kept. FA3's 24/240 split hurts this kernel; the hand-over itself helps packed.

#### E1 — FA3's L2 cache hints (on F2)

- **Date:** 2026-10-16.
- **Why:** FA3 loads K/V EVICT_LAST and Q EVICT_FIRST; BluTrain gave no hint and read 524 MB from DRAM vs FA3's 367 MB.
- **Result:** 296/296; dense −0.17% (10/12), packed −0.12% (9/12); dense DRAM 524 → 454 MB.
- **Decision:** folded into C1 (below).

#### C1 — F3c + E1

- **Date:** 2026-10-17.
- **Result:** vs F3c: dense −0.07% (no change), packed forward +0.05%, but the NEXT kernel (the backward, identical code) **+0.18%
  slower** on packed (0/12). DRAM −14%, cycles −0.13%: the dense forward is not memory-bound.
- **Decision:** E1 not taken; F3c stays the base.

#### L1 — descriptors built once

- **Date:** 2026-10-17.
- **What:** build the Q descriptors once per unit.
- **Result:** no change at all: ptxas rebuilt the "once" values inside the loop from the live shared-memory base.
- **Decision:** dropped (L1b later added the shuffle that stops it).

#### K1 — the P·V wait pinned after the softmax

- **Date:** 2026-10-17.
- **Why:** SASS showed ptxas moved the wait for P·V(n−1) up to just after 61 instructions of the softmax, so the rest of the softmax
  (328 instructions) did not overlap the GEMM. Many attempts did not move it (peeled diagonal, more max chains, row by row, two P
  sets, moved arrives).
- **What / how:** an mbarrier arrive placed before the wait whose ADDRESS depends on a value computed at the end of the softmax
  (two dummy barriers nobody waits on); ptxas then keeps the wait after it. Also: diagonal tile peeled out of the loop, two P register
  sets, only group 0 waits for each tile's K/V, no O rescale in the prologue. Result: 314 instructions before the wait (the whole
  softmax), 95 after.
- **Result:** 296/296; dense **−0.95%** (8/8), packed 0.00%.
- **Decision:** kept as the base of the K-series.

#### K1s and K1n2

- **Date:** 2026-10-17.
- **K1s** (one P register set): +0.94% / +2.81% worse. Not kept.
- **K1n2** (2 K/V stages, as FA3): **+39% dense / +22% packed worse**. One BluTrain stage holds K AND V of a tile and stays busy two
  steps (V(n) is read in step n+1); FA3 has separate K and V pipelines, so its 2 + 2 stages have the slack of BluTrain's 3. Not kept.

#### K2 — O store through shared memory

- **Date:** 2026-10-17.
- **What:** normalise + convert, 8 stmatrix per thread into a 16 KB block, then 16-byte global stores (8 instead of 32 per thread).
- **Decision:** dropped as built: it needed 32 KB of shared memory, only available with 2 K/V stages, and K1n2 showed 2 stages are far
  slower. The idea came back as K4 (staged in the last K buffer).

#### K3h / K3q / K3e — split softmax

- **Date:** 2026-10-17.
- **Why:** a timing trace of K1 showed the softmax (+pack) took 1,413 cycles per step (the code alone ~750): the two groups' softmaxes
  overlap on the exp unit. A step must fit in ~1,536 cycles to be tensor-bound; K1 was at ~1,710.
- **What / how:** part A (mask, max, rescale factors and the first 32 / 16 / 8 exps) before the P·V wait, part B after it,
  interleaved with the rescale and packs as in FA3.
- **Result (vs K1):** K3h dense −1.26%, packed −1.97%; K3q −1.45% / −0.86%. K3e built, not raced.
- **Decision:** the split was carried into K5q (split 16 on top of K4b).

#### K4 / K4b — O staged in the last tile's K buffer

- **Date:** 2026-10-17.
- **Why:** the epilogue (unit end) cost 4,230 cycles per pair (9% dense, 21% packed).
- **What / how:** after a group's last P·V, wait until the other group's last QK is done (the K buffer is then free), stmatrix O into
  the group's half of the last tile's K buffer, then 16-byte global stores, then release that stage. Keeps the 3 stages.
- **Result:** K4: dense −5.30%, packed −7.33%, but the first handshake (an mbarrier with parity waits) could deadlock (an inactive
  group could run a whole pair ahead; CPU model: 9 / 385 deadlocks dense / packed). **K4b** uses a monotonic counter (release-store,
  acquire-spin: a count that only grows cannot be overtaken; CPU model 0 / 0): 296/296; dense **−5.26%** (8/8), packed **−7.37%** (8/8).
- **Decision:** K4b kept.

#### K4hb, K5, K5q, K6

- **Date:** 2026-10-17.
- **K4hb** = K4b + split 32: dense −2.58%, packed **+8.38% worse**: it spilled ~6× more (108 / 76 B) at every unit's start and end;
  packed has many short units. Not kept.
- **K5** = K4b + split 32 + ONE P register set (with the split, all of P(n) is packed after the wait, so the second set only cost
  registers): 0 B spills; dense +0.84%, packed −4.63%.
- **K5q** = the same with split 16: dense −0.36% (7/8), packed **−4.33%** (8/8). **Kept: best of the day.** A spill-free trace showed
  the epilogue fell to 1,411 cycles per pair (3.2%, from 9%): the remaining dense gap is now the loop step.
- **K6** = K5 + the next unit's descriptor loaded during the last P·V: dense +0.95%, packed −4.20% (the prefetch did not help). Not kept.

#### K5e, K7, K7e

- **Date:** 2026-10-17.
- **K5e** (split 8): dense +1.22% worse. Not kept.
- **K7** = K5q + FA4's conditional rescale (keep the row max unless it grows by more than 8 in log2 units; the 64-FMUL O rescale runs
  only when needed): dense −0.49%, packed +0.48% worse. **K7e** (+ split 8): dense −0.81%, packed +0.63% worse. The conditional rescale
  pays on long rows (the max settles) and costs on short chats (the max keeps growing).
- **Decision:** not kept: packed SFT is the main path, and a per-mode build is impossible (next entry).

#### K8 / K9 / K9h — per-mode math and polynomial exp

- **Date:** 2026-10-17.
- **What:** K8 = a different softmax per mode (dense = K7e's code, packed = K5q's); K9 / K9h = K8 + every 4th / 2nd exp by a degree-3
  polynomial.
- **Result:** quick suites 278 / 277 / 274 of 296: every failed check is a BIT-EXACT one ("every packed chat = the same chat alone
  through the dense kernel"). With different math per mode, packed ≠ alone bit for bit (differences of rounding size).
- **Decision:** rejected. Rule kept since: dense and packed must use the same softmax arithmetic.

#### Knock-outs (measurement only)

- **Date:** 2026-10-17.
- **What:** K5q builds with one part of the softmax removed (results wrong on purpose, timing only).
- **Result:** without the 64 exps per tile −6.8% dense / −6.8% packed; without the row max −4.1% / −3.5%; without the O rescale −2.4% /
  −2.2%; all three −11.2% / −11.6%. The parts do not add up (13.3 > 11.2): removing one shows the next limit.
- **What it proved:** the softmax IS on the critical path.

#### W1 / W1s / W2 — wait forms and separate K/V barriers

- **Date:** 2026-10-17.
- **Why:** a timing trace showed the step-start wait was bimodal (~100–150 or ~275–300 cycles); FA3 / FA4 / cuDNN use CUTLASS's wait
  (try-wait + a sleep hint), BluTrain a plain try-wait loop; FA3 also keeps K and V on separate barriers.
- **Result:** W1 (CUTLASS wait) −0.07% / −0.04% (no change); W1s (spin wait) +0.14% / +0.44% worse; W2 (W1 + separate K/V barriers)
  −0.11% / +0.10% (no change). The wait shrank but its time reappeared in the softmax phases: it was slack.
- **Decision:** none kept.

#### P8w / P4w — polynomial exp

- **Date:** 2026-10-17.
- **What:** W1 + every 8th / every 4th exp computed by a polynomial instead of the exp unit (FA4's idea for Blackwell).
- **Result:** P8w +2.69% / +3.40% worse; P4w +7.07% / +7.44% worse (~0.035% per added instruction per tile: the softmax phase is
  issue-bound on Hopper).
- **Decision:** not kept.

#### L1b — Q descriptors once, through a shuffle

- **Date:** built 2026-10-17, raced 2026-10-19.
- **Why:** ncu of K5q vs FA3 (2026-10-17): dense 1.08× FA3's cycles; the QK issue phase took 72 instructions vs FA3's 41 (Q AND K
  descriptors rebuilt for every HGMMA).
- **What / how:** pass each Q descriptor half once through `__shfl_sync(.., 0)` (stops ptxas rebuilding it); K/V = stage-0 descriptor +
  stage offset. QK issue 72 → 42 instructions.
- **Result:** 296/296; dense **+0.36%**, packed **+0.90%** (0/8). On the new box's CUDA 13.4, K5q itself already compiled to 59 / 40
  issue instructions.
- **Decision:** rejected. Lesson: the removed instructions sat in time the warps would have waited anyway; judge a change by the
  tensor pipe's idle time, not by stall shares.

#### R8 / R16 / R32 — row-split softmax

- **Date:** 2026-10-19.
- **Why:** BluTrain computed BOTH rows' max first (62 FMNMX), then part A's exps nearly back to back (25 of 66 exps with 0–1
  instructions between them); FA3 runs row 1's exps while computing row 2's max.
- **What / how:** part A = the first 8 / 16 / 32 scores of ROW 0 only; same values (bit-identical expected).
- **Result:** 296/296; dense +0.34% / +0.76% / +0.12%, packed −0.07% / 0.00% / +2.54%.
- **Decision:** rejected: changing which exps sit before the wait did not shorten the step. (ptxas still put most of row 1's max
  first.)

#### KOkv — knock-out of the K/V wait (measurement only)

- **Date:** 2026-10-19.
- **What:** K5q without group 0's in-loop K/V wait (results wrong on purpose): the upper bound of ANY K/V pipeline change.
- **Result:** dense −0.30%, packed −1.79% (3 rounds).
- **What it proved:** FA3's split K/V pipeline is not the lever for the dense gap (~8%).

#### SKV — FA3's split K/V pipeline

- **Date:** 2026-10-19.
- **What / how:** separate rings for K and V (new full_v / empty_v barriers); the loader sends K(n) then V(n−1), as FA3; K(n) released
  right after its QK (except the unit's last tile, which K4 uses for O). Protocol checked on the CPU first.
- **Result:** 296/296; dense +0.20%, packed **+0.87%** (0/8).
- **Decision:** rejected (correct, but no faster dense and slower packed, as KOkv's bound predicted).

#### LH / LHO / LHOd — FA3's L2 hints

- **Date:** 2026-10-19.
- **Why:** the forward runs at the 700 W cap (691–695 W in every run). FA3 tells the L2 to keep K/V and drop Q first; BluTrain read 520
  MB from DRAM vs FA3's 365 MB (L2 hit 80.6% vs 86.6%).
- **What / how:** LH = Q loads evict-first, K/V evict-last (CUTLASS's policy values); LHO = LH + O stores evict-first; LHOd = the hints in
  the DENSE kernel only (packed = K5q exactly).
- **Result:** forward-only race: LH dense −1.18% (8/8), packed **+0.56% worse**; LHO dense −1.49%, packed +0.68% worse. Dense DRAM 520 →
  432 MB. A clock/power check showed the hints save ENERGY, not cycles: same cycles, the clock rises ~1% at the cap. In packed they cost
  cycles at each unit's first K/V wait and save nothing (packed DRAM is already FA3's). LHOd in a forward + backward race: dense
  −0.34%, and the unchanged backward −0.33% (the higher clock helps both); packed 0.00%.
- **Decision:** LHOd kept; LH / LHO not (packed worse).

#### SM3 / SM3m — exps spaced like FA3

- **Date:** 2026-10-19.
- **Why (ncu PC sampling, FA3 vs BluTrain, same box):** both run the same instructions per step (66 exps, 68 FMNMX, ...), but the exps
  had +120 cycles of "wait" stalls per step (367 vs 247): an exp issued 0–2 instructions after the previous one waits ~7–9 cycles
  for the exp unit. FA3's source (`softmax.h`) computes the max row by row and runs row n's exps while row n+1's max chain runs, so
  ~2 FMNMX sit between exps.
- **What / how:** SM3 = the max and exps row by row as FA3 (exps of one row interleaved with the next row's max). SM3m = SM3 + FA3's
  causal-mask order (row by row against one per-row limit), because SM3's mask had turned into a 64-bit bitmask (64 LOP3 per masked
  step) that slowed packed rows.
- **Result:** SM3: dense **−3.05%** (8/8), packed +0.45% (the mask bitmask). SM3m: dense **−3.51%** (8/8), packed **−4.48%** (8/8); exp
  "wait" 367 → 245 cycles per step (FA3 247); ncu cycles dense −4.0%, packed −6.3%. The loop step now equals FA3's.
- **Decision:** SM3m kept.

#### PF — first K/V tile before the Q-empty wait

- **Date:** 2026-10-19.
- **Why:** after SM3m the remaining dense gap was at the unit boundaries: ONE wait, for tile 0's K/V, cost 1,498 cycles per unit (FA3:
  50). BluTrain's producer waited for both groups to finish the previous unit's last QK (qempty), then loaded Q, then K/V(0): the 64 KB
  were requested too late. FA3's producer loads the first K as soon as a K slot is free, THEN waits for Q to be free, then loads Q.
- **What / how:** the producer loads the unit's first K/V tile (waiting only for its slot) BEFORE the qempty waits, then Q, then the
  rest (FA3's order). Checked first on a CPU protocol model (3,000 + 3,000 random unit lists, no deadlock).
- **Result:** 296/296; dense **−2.60%** (8/8, t −50), packed **−3.01%** (8/8); the tile-0 wait 1,498 → 48 cycles per unit (FA3 50).
  Same-box scoreboard: dense forward **1.025× FA3** (was 1.084×), packed 0.866×.
- **Decision:** kept.

#### PFT — O written by TMA store

- **Date:** 2026-10-19.
- **Why:** the epilogue cost 2,442 cycles per unit vs FA3's 2,059 (+383); FA3 writes O to shared memory and one lane issues a TMA store.
- **What / how:** stmatrix in the TMA layout, fence, warps 1–3 arrive and go on, one lane of warp 0 issues the TMA store (dense only;
  varlen cannot use TMA for O).
- **Result:** 296/296; dense **+1.60% slower** (0/8); packed identical. ncu: the epilogue really got 484 cycles cheaper, but the next
  unit's start grew +668 (the store warp's wait) and the steady loop +40 cycles per step (the compiler reordered it).
- **Decision:** rejected.

#### QP / QPU / QPD / QPUD — the last unit-start candidates

- **Date:** 2026-10-19.
- **What / how (each a switch on PF):**
  - **QP:** Q prefetched into L2 (`cp.async.bulk.prefetch.tensor`) when a pair starts, before the K/V and the qempty wait (FA3 mainloop
    :681). The steady loops are identical to PF's.
  - **QPU:** QP + the producer publishes the DECODED unit (batch, head, pair) in a slot next to the unit counter, so the consumers do no
    integer divisions at the unit start.
  - **QPD:** QP + the diagonal tile first (FA3 walks the keys backwards): no masked step at the end.
  - **QPUD:** all three.
- **Result (5-way race on the 8x node):** QP +0.08% / −0.24%; **QPU −0.81% (7/8) / −0.13%**; QPD +0.32% / −0.43%; QPUD −0.31% / −0.44%.
  ncu: QP makes Q land early (wait 275 → 171 cycles per unit) but gains nothing alone (the divisions were then on the path); QPU's
  unit start is 222 cycles cheaper and its reordered loop 19 cycles per step faster; QPD removes the masked last step (−333 per unit)
  but its masked first tile costs +121 and its loop +19 per step: no net dense gain. QPU vs FA3 in cycles: +0.31% (PF +1.85%).
- **Scoreboard with QPU (same node):** dense forward **1.004× FA3**; 11 of 12 matchups won (Part 3).
- **Decision:** QPU kept as the final forward; ported to the production pretraining branch on 2026-10-21.

---

## Lessons that apply to the next kernel work

1. Under the 700 W cap, time follows energy. Cutting instructions or cycles converts to time only partly; a faster kernel often makes
   the next one slower (shared clock). Always race at free clocks, forward + backward together.
2. Removing instructions that sit in waiting time changes nothing (G2, mask fix v1, L1b). Judge a change by what it does to the
   tensor pipe's idle time.
3. Compare with FA3 line by line (source + SASS + PC sampling) before building; most wins (Vj2 layout, fix 1, B-2a, G4, F1, F2, SM3m,
   PF, QPU) copied a specific FA3 detail found that way. Copying FA3's register split (F3b) or pipelines (K1n2, SKV) did not fit
   this kernel.
4. Anything that touches barriers or the unit hand-over is first checked on a CPU protocol model (F3 and K4 hung; F3b and K4b did not).
5. Dense and packed must use the same softmax arithmetic, so a packed chat stays bit-identical to the same chat alone (K8/K9 failed this).
