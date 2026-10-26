---
name: project_sm103_cuh_locked
description: SM103 cluster template — all knobs (2x1 h/v, 2x2, mcastA, 2x2f dynamic) validated 336/336 on B300; remaining variants + key gotchas
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
---

`Bgemm_sm103_cluster_template.cuh` (production single-source-of-truth, B300/sm_103a BF16 GEMM) — as of 2026-07-04 ALL knobs validated bit-exact on the B300 (336/336 with RUN_2X2=1 RUN_2X2_MCAST=1 RUN_2X2F=1): tile (128x128/128x256), `VerticalWave`, `CLUSTER_N` (2x1/2x2), `MulticastA`, `DynamicCluster` ("2x2f" = preferred-4/fallback-2 launch). NO `__cluster_dims__` anymore — every launch goes through cudaLaunchKernelEx with cluster launch attributes (CUTLASS pattern).

Bugs fixed this arc (all root-caused from CUTLASS/PTX-ISA/nsys/device-probe evidence, analyzations.md §30–31):
- Dual empty-barrier: operand-smem empty is CLUSTER-WIDE (mask (1<<ctas)-1, count=CLUSTER_N); tmem empty stays PAIR-LOCAL (0b11<<2*cta_n_grant). One barrier for both = 2x2 deadlock.
- Multicast mbar PARITY rule (PTX ~line 17169): cta_group::2 completion routes by the parity of where the mbar RESIDES → multicast must use the LEADER mbar (even parity), not raw local. Was the P2-M2 hang.
- CLC ONE-outstanding try_cancel protocol (CUTLASS-matched): 2 in-flight cancels can orphan a VALID steal at end-of-grid → zero-filled tiles (was the flaky LM-head failures; worst-element dump in the test exposed it as missing tiles, NOT tolerance noise — earlier cancellation-noise diagnosis was wrong).
- Dynamic CLC contract MEASURED via device probe (examples/test_clc_dynamic_probe.cu, §31.3): cancellation units are REQUESTER-SIZED; stolen piece's N-block = ((new_ctaid % CLUSTER_CTAS)>>1) + cta_n_grant (degenerates to old decode for rigid launches).

TMEM gotcha (§28.7): TMEM = 128 DP × 512 col, PER-CTA; cta_group::2 epilogue reads own half at DP base 0 (no cross-CTA lane offset); BM must be 128 (static_assert). cta_group::2 f16 M ∈ {128,256}.

Perf OFFICIAL BASELINE (B5 run 3, locked 1400MHz, warmup-fixed, analyzations §47.3): best-of-variants avg 82.3% of cuBLAS; BEAT on 1/42 (8b-Down 102.9%); near-parity ≥94% on 7 more (7b-Down 99.7, 8b-Gate 97.6...). Earlier ">100% LM-head" claims RETRACTED = boost-clock order bias (§47.1-2: whoever measures first on an unlocked GPU reads ~30% high; ALWAYS lock clocks low enough to avoid the power cap — 1400MHz on this B300 — and warm up EVERY timing pass). Diagnostic: our kernels scale EXACTLY with clock (compute/issue-bound); cuBLAS LM-head is memory-bound → close gaps via per-SM efficiency (epilogue vectorization), not more tiles. Skinny-K 42-68% = structural gap (nvjet 16-stage pipeline). Dispatcher winners: 128x256_v/h dominate, 128x128 h/v, 2x2 family, 64x128_2x2 for tiny-N, 192x128; retired (code kept): 64x64_2x2, 160x128m, swap160x192, 1x2_128x192. B4 sw32 tiles NOT worth building.

Qwen coverage 42/42 family-level; EXACT-tile 36/42 after B1+B2 (07/07): composite-M knob SILICON-VALIDATED — B1 = 2x M256 atoms (256x256_2x2 tile, 512x512 cluster, EPI_PIPE_DEPTH=1 forced, P7 24/24) and B2 = MIXED M256+M128 Layout-B tail (192x128 tile = nvjet 128x192 family, quadrant epilogue, 64-row tail C-tmap, P8 36/36) — NO public library implements mixed-shape composites; derivation in analyzations.md §41 (R7-R12 research, twice-verified) + §42-§44. Pieces derive from BM (no new template param); traps: idesc M-field must encode PER-ATOM M (512>>4 overflows 5-bit field); TMEM cols = per-stage SUM over atoms; tail store needs its OWN 64-row tmap (128-row box would overwrite next tile — TMA clips only at tensor edge). Batched partial-M is library-wide unsupported (store clips at TENSOR edge → bleeds into next batch). 1x2 swap-AB 160x192 also validated (composite M'=192=128+64 ::1, 64B-swizzle operand). Remaining tiles: 128x160 masked (B2.5), 64x64/128x64 (B3 sw64 B-slice), 160x192_2x2/320x192_2x2f (B4 sw32, benchmark-gated — DeepGEMM never uses narrow swizzle on operands). ALL documentation goes in analyzations.md ONLY (user rule; old plan files merged as §38-§40). NEXT: B2.5/B3 → B5 clean benchmark → dispatcher (LAST), addmm (§32).

Test/run: server dharun@server27:/mnt/raid0/gautam_k/blublas (CUDA 13.2); local box RTX 3060 = compile-check only (`nvcc -std=c++17 -gencode arch=compute_103a,code=sm_103a`). `make test TEST=examples/<name>.cu`; force `rm -f build/<name>` after header-only edits. Sync via file copy + grep-count marker check (both machines same-looking paths but separate checkouts). Watch out: user's IDE buffer can clobber freshly-patched files on disk — re-verify marker counts after any "compare/overwrite" prompt. See [[project_active_repo]], [[feedback_test_log_format]].
