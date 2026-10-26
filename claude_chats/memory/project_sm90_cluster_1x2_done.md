---
name: project_sm90_cluster_1x2_done
description: SM90 1x2 cluster (redundant-A + A-multicast) validated 12/12 bit-exact on H200
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-07-24T15:10:09.368Z
---

SM90 (H200/Hopper) BF16 GEMM 1x2 cluster is DONE & validated on H200 (2026-07-23):
NT layout, both redundant-A and A-multicast (leader-only) forms, 12/12 bit-exact vs
cuBLAS via `test_bgemm_sm90_cluster_1x2_gate` (2-milestone gate). Real target shape
M6144×N2048×K4096 ≈ 89-90% of cuBLAS.

Key design (in `Bgemm_sm90_warpspecialized_template.cuh`):
- `MulticastA`/`MulticastB` template knobs (like SM103 §30.8): false + CLUSTER>1 =
  redundant cluster (each CTA loads own operand); true = shared via multicast.
- Multicast is PARTITIONED (§H14.7, the PERMANENT fix, validated H200): each CTA
  loads its OWN 1/N slice (A: BM/CLUSTER_N rows; B: BN/CLUSTER_M cols) via a SLICED
  descriptor (box=tile/N) + multicasts it to all peers. Symmetric work → CTAs stay
  in lockstep → RACE-FREE by construction; zero perf cost; 1× DRAM. Proven from
  CUTLASS box-truncation (cute/atom/copy_traits_sm90_tma.hpp:996).
- SUPERSEDED: the earlier LEADER-ONLY multicast (only cluster_local_n==0 issues
  full-A) had a cross-CTA drift RACE — asymmetric work → leader↔non-leader drift →
  mbarrier phase corruption → flaky crash, but ONLY at large-K + persistent scale
  (proven by bisection: E1 large-K-alone PASS, E2 persistent-alone PASS, E3 both
  CRASH; and crash vanished under all 3 sanitizers = Heisenbug). A per-tile
  cluster_sync band-aid bounded it (validated) but was replaced by partitioned.
  Lesson: leader-only asymmetric multicast is latently racy even for NT; use
  partitioned (symmetric) always.
- FULL count=1 (NT), expect_tx=one CTA-tile(A+B) satisfied by N partial slice-
  completes; EMPTY count=NUM_CONSUMER_WG×(SHARE_M+SHARE_N−1) with local + cross-CTA
  remote (mapa) arrives.
- Also: multicast is a shared `issue_operand_4d` helper called from all layout
  branches (NT + NN's direct-A wired; NN-B/TN raw-path multicast = future step 3,
  guarded by static_assert).

Two wrong paths burned real time (see analyzations_h200.md §H12-H14): (1) a standalone
multicast repro whose crash was a repro-only missing-`__grid_constant__`, irrelevant to
the real kernel; (2) a "leader-only fix still hangs" run that was actually a STALE BUILD
— see [[project_reduction_build_gotcha]], the Makefile has `-include *.d` but no
`-MMD -MP` so `.cuh` edits never rebuild. After any header-only kernel edit, force a
rebuild (touch a `.cu` / rm the `.o` / make clean) before trusting a run.

Next: NN/TN layouts, other cluster shapes (2x1/1x4/4x2, 2x2 needs B-multicast), then
benchmark sweep → dispatcher. Ties to [[feedback_diff_against_proven_code_first]] and
the B300 milestone-bisection process (analyzations_b300.md §30.6-30.11).
