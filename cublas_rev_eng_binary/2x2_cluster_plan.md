# P2 — 2×2 cluster (4-CTA) implementation plan (reference-verified 2026-07-04)

Target file: `BLUBLAS_kota/BluBridge-BLAS/src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_template.cuh`
Grounded in: PTX ISA 9.3, CUTLASS (sm100_mma_warpspecialized / copy_sm100_tma / sm100_pipeline),
DeepGEMM (sm100_bf16_gemm / tma_copy), ThunderKittens.

## Core model (settled)
- NO 4-CTA MMA. Still `tcgen05.mma.cta_group::2` (a 2-CTA pair along M).
- 2×2 = same pair replicated as a 2nd pair along N. A is multicast across the 2 N-pairs;
  B stays per-pair (each pair loads its own N-block). Only A traffic is halved.
- Use `__cluster_dims__(4,1,1)` (NOT (2,2,1)) → keeps CLC 1-D. Identical masks.
- rank = `%cluster_ctarank` (flat 0..3). cta_m = rank&1 (M-partner, drives DP rows),
  cta_n = rank>>1 (N-block). pair_leader = rank&~1. n_peer = rank^2.
- A-multicast mask = 0b0101<<cta_m (0b0101 rank0 / 0b1010 rank1).
- MMA-commit / peer mask = 0b0011<<(2*cta_n) (pair0=0b0011, pair1=0b1100). NOT 0b1111.
- Per-CTA M ≤ 128 (BM=128 assert stays). TMEM read at DP base 0 per CTA (unchanged, §28.7).

## Distinguish the two "cluster" factors (STOP overloading CLUSTER_SIZE)
- CLUSTER_M = 2  → MMA pair size. Everything that meant "the MMA pair" stays 2:
  B_STAGE_BYTES (BN/2), i_desc M height (2*BM=256), full_mbar init, tmem_empty init,
  TMA cta_group::2, dealloc cta_group::2.
- CLUSTER_N = 2  → number of pairs (N-blocks per cluster).
- CLUSTER_CTAS = 4 → CTAs per cluster. Everything that meant "grid stride / CLC count":
  clc_empty arrive count, __cluster_dims__, grid stride, ctaid/CLUSTER decode.

## Bring-up milestones (B300)
1. **Redundant-A** (recommended first): (4,1,1) cluster, scheduler split into 2 N-blocks,
   per-pair leader — but every CTA loads its OWN A+B (NO multicast). Each pair is
   byte-for-byte the current 2-CTA kernel. Only new logic = scheduler + per-pair leader
   + CLC counts. Validate correctness first. (Cost: 2× A DRAM traffic.)
2. **Turn on A multicast** (cta_n==0 issues A, mask 0b0101<<cta_m, a_mbar bit24 cleared) →
   same results, lower DRAM.
3. Sweep BN=128/256, batch, NN/NT/TN, all 42 Qwen shapes.

## Edit checklist (file:line today → change)
- Config: add CLUSTER_M=2, CLUSTER_N=2, CLUSTER_CTAS=4. CTA_MASK per-pair rank-dependent.
- CLC_EMPTY_ARRIVE: NUM_WARPS*CLUSTER_SIZE → NUM_WARPS*CLUSTER_CTAS (×4).
- __cluster_dims__(2,1,1) → (4,1,1).
- get_cluster_cta_rank(): %cluster_ctaid.x → %cluster_ctarank. Derive cta_m/cta_n/pair_leader/n_peer.
- full_mbar init: stays 1*CLUSTER_M (2). tmem_empty init: stays NUM_EPI_WARPS*CLUSTER_M (2).
- clc_empty init: ×4 (via CLC_EMPTY_ARRIVE).
- tma_mbar_base: cta_m==0 ? own : mapa(full, pair_leader).
- cta0_clc_empty: all 4 → rank0 (CLC is cluster-global): cta_rank==0 ? own : mapa(...,0).
- init_tile_idx = ctaid/CLUSTER_CTAS (÷4). CLC new_tile_idx = new_ctaid/4.
  Decode: n_cluster = cluster_idx % ceil(grid_n/2); row_tile = cluster_idx / ceil(grid_n/2).
  block_row = row_tile*CLUSTER_M + cta_m; block_col = n_cluster*2 + cta_n.
- Producer: B all 4 CTAs (own N-half, cta_group::2, unchanged). A: milestone1 all-load;
  milestone2 only cta_n==0 multicast (mask 0b0101<<cta_m, a_mbar=mbar&0xFEFFFFFF).
  expect_tx: every CTA mbarrier_arrive_expect(leader_mbar, copy_size).
- MMA leader gate: if(cta_rank==0) → if(cta_m==0) (ranks 0 AND 2 drive their pair).
- MMA commit mask: CTA_MASK → 0b0011<<(2*cta_n).
- Scheduler CLC arm: fan out expect_tx=16 to r=0..3 (all 4 CTAs' clc_full via mapa).
- Epilogue store coords: block_col = n_cluster*2+cta_n; block_row = row_tile*CLUSTER_M+cta_m.
- Launcher grid: total_cluster_tiles*batch*CLUSTER_CTAS, total_cluster_tiles = grid_m*ceil(grid_n/2).
- Keep static_assert(BM==128).

## Hurdles + resolution
1. CLC↔cluster dim: use (4,1,1) → first_ctaid 1-D, no query change. (biggest feared risk → mechanical)
2. clc_empty count must be ×4 or scheduler deadlocks. #1 hang suspect.
3. CLC expect_tx must be armed on all 4 CTAs (multicast::cluster::all broadcasts response).
4. A-multicast mbar peer-bit (bit24 clear) so A bytes land on leader (CUTLASS copy_sm100_tma.hpp:318).
5. Audit every CLUSTER_SIZE use (pair vs cluster meaning).
6. BM==128 assert unchanged.
7. HW-verify: plain 2SM TMA in a 4-CTA cluster still pairs {0,1},{2,3} — redundant-A milestone must
   be bit-exact vs two separate 2-CTA launches; if pair1 wrong, fall back to (2,2,1) + CUTLASS peer mask.

## FIX APPLIED (2026-07-04) — dual empty-barrier per CUTLASS

Root cause of the 2×2 deadlock (confirmed on a near-idle B300, contention ruled out): we used ONE
empty-barrier with the PAIR-LOCAL mask (0b1100) for BOTH smem lifetimes. CUTLASS
(sm100_pipeline.hpp) keeps TWO:
- accumulator/tmem-empty → PAIR-LOCAL (calculate_umma_peer_mask): 0b0011 / 0b1100, count = pair.
- operand-smem-empty → CLUSTER-WIDE (calculate_multicast_mask<kRowCol>): for atom(2,1) every CTA
  participates → mask = full cluster (0b11 @ 2×1, 0xF @ 2×2); count = one arrive per pair-leader
  = CM/atomM + CN/atomN - 1 = CLUSTER_N.
A is multicast across N (shared by both pairs), so its smem may only be refilled after EVERY
pair-leader is done → the release must span the cluster. Pair-local release → pair1's producer never
saw pair0's release → deadlock.

Changed in Bgemm_sm103_cluster_template.cuh:
- empty_mbar init count 1 → CLUSTER_N (line ~706).
- consumer operand-empty commit mask pair_mask → operand_empty_mask = (1<<CLUSTER_CTAS)-1 (line ~993).
- tmem_full (accumulator) commit KEEPS pair_mask (line ~1001) — pair-local, already correct.
CLUSTER_N=1 (2×1) → count 1, mask 0b11 = old behaviour exactly (no regression).
Generalises: same formula is correct for 1×2, 4×1 (the single "kNumMulticast"-style knob).

## FIX #2 APPLIED (2026-07-04) — multicast mbar parity (the P2-M2 hang)

Result of fix #1 on B300: P2 redundant-A 2×2 = 21/21 bit-exact, deadlock GONE. P2-M2 (multicast-A)
still hung on its first kernel. Root cause found in the PTX ISA 9.3 spec itself
(cp.async.bulk.tensor, ptx_docs/ptx_isa_9.3.txt ~line 17169):

> .cta_group::2 : ... for each destination CTA specified in the ctaMask, the mbarrier signal is
> sent either to the destination CTA or its peer-CTA **based on the %cluster_ctarank parity of the
> shared memory where the mbarrier object mbar resides.**

i.e. completion routing is decided by the PARITY OF THE MBAR OPERAND'S ADDRESS — it is NOT an
automatic route-to-leader. Our multicast call passed the raw LOCAL full_mbar. For the odd issuer
(rank1, mask 0b1010) the mbar resides in an odd CTA → the +A tx multicast to the ODD CTA of each
destination pair (rank1/rank3's own barriers — nobody waits there) → both pair leaders starved of
A_bytes every stage → hang. CUTLASS avoids this by forcing even parity on every 2SM load's mbar
(& Sm100MmaPeerBitMask = clear bit 24, copy_sm100_tma.hpp:45, used at :240/:266/:292/:318).

Fix (one-line semantics): issue_A_load multicast path now uses the LEADER mbar (tma_mbar_base —
even-parity residency; already mapa'd for the odd partner) instead of the raw local mbar; the unused
local_mbar param removed from issue_A_load + its 3 call sites. Accounting now balances: each pair
leader collects 2×A (both A-multicasts route even) + 2×B (own pair's B loads) = expected 2×(A+B).
Compile-verified sm_103a (frontend+ptxas clean).

### Staged validation (needs B300)
1. Redundant-A first (RUN_2X2=1, P2 mcastA=0): full-barrier accounting unchanged & known-correct;
   the empty-barrier fix should clear the deadlock. Must be bit-exact vs cuBLAS (MaxRelErr 0.00e+00).
2. Then multicast-A (P2-M2, mcastA=1): each CTA still ends with a+b bytes (A via multicast, own B),
   so the full-barrier tx MAY already balance — verify. Only if a tx imbalance/hang appears, apply
   operand-FULL fix#2 (leader-only expect for the joint 2SM transaction, CUTLASS A.4).
3. Barriers are pure sync → ZERO precision impact; perf is neutral-to-positive (enables A-DRAM halving).
