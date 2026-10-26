# Answer to "GEMM ping-pong cross-check" (Gautam, 2026-10-26)

From: the Claude session on Harshini's machine. Date: 2026-10-26.

**How this was checked.** Everything marked VERIFIED-NOW was read or run in this session:
- **Code:** Nova at `ac2ab83` (branch `current`, working tree clean).
- **Build:** a local `nova-opt` rebuilt at HEAD (16:33 today). The binary from my earlier reply was built 10-24 18:57, before ac2ab83, so it was stale; every compiled fact here is redone on the fresh build.
- **Tools:** CUDA 13.0 `ptxas` / `cuobjdump`, and the CUTLASS `main` sources fetched today.
- **Artifacts** (MLIR, PTX, cubin, SASS): `reports/pingpong_crosscheck_2026-10-26/`.

FROM-EARLIER items come from written records made at the time; each one names its file. The 10-23 instance (95.133.252.52) has since been reassigned (its host key changed), so its raw ncu/nsys files are gone; only the local write-ups and CSVs remain.

**Six corrections to my earlier reply (Appendix A),** all explained under F1–F3:
1. **c54a485 is Aparna's commit,** not Harshini's.
2. **"−3.4% on the 26-layer step" is wrong.** −3.39% was the ping-pong GEMMs' ncu cycles. The measured step result is **−1.19%**.
3. **F1:** CUTLASS cooperative has the same 40/232 register split.
4. **F2:** CUTLASS's turn barrier is an mbarrier, not `BAR.ARV`.
5. **The SASS spelling is `USETMAXREG.TRY_ALLOC`.**
6. **v1's slowdown** was not shown to come from strict ordering alone.

---

## Section 1–7 claims I checked first

| Gautam's claim | Check | Result |
|---|---|---|
| CUTLASS cooperative has 40/232 (§2.4) | fetched `sm90_gemm_tma_warpspecialized_cooperative.hpp` | ✓ `LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;` `MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;` plus `warpgroup_reg_dealloc` / `warpgroup_reg_alloc`. No `MathWarpGroupOrderBarrier`. (It does have `using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;`, an ordering inside the producer, not between consumers.) |
| OrderedSequenceBarrier is an mbarrier (§2.3) | fetched `sm90_pipeline.hpp` | ✓ `using Barrier = cutlass::arch::ClusterBarrier;`. `wait()` calls `get_barrier_for_current_stage(params_.group_id).wait(stage_.phase())`, and `arrive()` signals `(group_id + 1) % Length` |
| Remote BluBLAS = Kota's `ea666ab` (§3) | local submodule | ✓ `Tensor-Implementations/.gitmodules` url `https://github.com/kathir-23s/BluBridge-BLAS`. Local checkout `ea666ab 2026-08-31 14:35:51 +0530 Kathir \| Merge pull request #3 from BlubridgeAI/production_blas` |
| Kota's ping-pong template is never compiled (§4.2.2) | `git -C BluBridge-BLAS grep -n -i "include.*pingpong" HEAD`; symbol grep; build-file grep | ✓ all three return nothing |

---

## A. Where Nova's ping-pong came from

**ID: A1**
Answer: I can't tell which files Harshini personally looked at. The written record of the Nova implementation (10-21) cites CUTLASS concepts only and names no BluBLAS file. Separately, on 10-10 my session reviewed BluBridge-BLAS `ea666ab`, including Kota's ping-pong template.
Evidence:
- **10-21 record** (`reports/p022_2026-10-21/progress_2026-10-21_p022_pingpong.md`):
  - row 1a: "Fix: named-barrier turn between the K loops (barriers 1/2, CUTLASS ordered math barrier)";
  - job notes: "wait_group 1, release the previous stage: CUTLASS's K_PIPE_MMAS = 1".
- **Code comment** at `include/Transforms/Config/NovaGPULoweringConfigUtils.h:153 @ ac2ab83`: "Registers per consumer thread of two warp-specialized consumer warpgroups (tuning: CUTLASS's count)."
- **10-10 review** (memory note `blublas-eager-c-handling`): "`Bgemm_sm90_pingpong_template.cuh` is never included (it hung on hardware)… An sm_90a compile probe (128x128x64 NN) gives 168 regs and 0 spills… the header's explanation of the hang is unproven."
Status: FROM-EARLIER (both records written at the time). Whether Harshini used other material: NOT-AVAILABLE, so ask her.

**ID: A2**
Answer: No record shows Kota's file being used for Nova. Nova's design is structurally different from it (named-barrier turn, no epilogue turn, early hand-off, compiler passes rather than a CUDA template). On 10-10 it was already known that the file is never included and that it hung.
Evidence:
- Code differences: B2 and B5 below (VERIFIED-NOW).
- Knowledge on 10-10: memory note `blublas-eager-c-handling` (FROM-EARLIER).
Status: VERIFIED-NOW for the design differences; FROM-EARLIER for the history.

**ID: A3**
Answer:
- **Followed:** CUTLASS (`KernelTmaWarpSpecializedPingpong`: two consumer warpgroups, ordered math turn, 232 registers, `K_PIPE_MMAS = 1`).
- **Primitive reused:** the named-barrier turn already in Nova's flash attention (`pingPongFlashConsumers`, the FA3-style turn from Mahesh's commit fbda5fe), generalized into `emitConsumerTurns`.
- **Not in any record:** a CUTLASS commit or version. The CUTLASS / PyTorch / FA3 quotes in my earlier reply were fetched from GitHub `main` / the web on 10-26. ThunderKittens and DeepGEMM are not in any record.
Evidence:
- `include/Transforms/Tiling/NovaGPUForallMappingUtils.h @ ac2ab83`, with the comment: "Two consumer warpgroups' named-barrier turns… warpgroup 1 hands warpgroup 0 (`isFirst`) the first one on barrier `first`."
- That helper is used by both flash and GEMM (B2).
Status: VERIFIED-NOW (code); FROM-EARLIER (history).

**ID: A4**
Answer: In compile (JIT) mode the whole forward+backward graph is compiled by Nova, so every model GEMM is a Nova kernel. On the current build (ac2ab83), with the 1.58B one-layer graph (QKV packed), **9 of 15 GEMMs use ping-pong** and all 15 are persistent:

| GEMM | M × K × N | Kernel |
|---|---|---|
| QKV fwd | 32768 × 2304 × 3840 | ping-pong 128x192 S5 |
| O fwd, O dX | 32768 × 2304 × 2304 (×2) | ping-pong 128x192, S4 and S5 |
| gate/up fwd | 32768 × 2304 × 12288 | ping-pong 128x192 S5 |
| down fwd | 32768 × 6144 × 2304 | ping-pong 128x192 S4 |
| LM-head fwd | 32768 × 2304 × 50304 | ping-pong 128x192 S5 |
| QKV dX | 32768 × 3840 × 2304 | ping-pong 128x192 S5 |
| down dX | 32768 × 2304 × 6144 | ping-pong 128x192 S5 |
| gate/up dX | 32768 × 12288 × 2304 | ping-pong 128x192 S5 |
| all 5 dW | 2304/6144 × 32768 × … | cooperative 128x256 S4 or 128x192 S5 |
| LM-head dX | 32768 × 50304 × 2304 | cooperative 128x256 S4 |

All of them launch with grid 132 × 1 × 1 (persistent).

Evidence:
- Command: `nova-opt swiglu.mlir --nova-gpu-optimized-pipeline=arch=sm_90 --mlir-print-ir-before=nova-gpu-configure-tensor-layouts`, parsed with `reports/turns_calib/gemm_configs.py`. Ping-pong means each consumer warpgroup owns the whole tile: `subgroup == workgroup` and `wg_subgroup = [1,1,0]`.
- That `swiglu.mlir` is a 1-layer 1.58B graph (`tensor<32768x2304xbf16>, tensor<2304x3840xbf16>`; file dated 10-24).
- "No BluBLAS/cuBLAS in compile mode" comes from the 10-22 nsys compile runs: 48 `main_kernel_*` per micro-step (`reports/progress_2026-10-22_inst101_eager_vs_compile.md`). This is FROM-EARLIER on an older build.
Status: VERIFIED-NOW (per-GEMM map); FROM-EARLIER (no library GEMM in compile mode). Optimizer GEMMs under Muon were not checked (NOT-AVAILABLE).

---

## B. Code proof

**ID: B1**
Answer: Both commits exist.
- `c54a485` is by **Aparna**, on branches `current` and `origin/water`.
- `ac2ab83` is Harshini's, on `current`.
- Harshini's ping-pong commit is `961e4ff`, on `current`, `origin/kidd` and `origin/water`.

Evidence:
```
c54a485 2026-10-24 15:55:22 +0530 Aparna-BB | persistence, stream-k and ping-pong | branches: current remotes/origin/water
ac2ab83 2026-10-26 14:03:13 +0530 harshini-v-m | rope+norm backward fusion | branches: current
961e4ff 2026-10-23 15:37:10 +0530 harshini-v-m | seed-tray,ping-pong,grid size check | branches: current remotes/origin/kidd remotes/origin/water

$ git show --stat c54a485
 include/Dialect/NovaNVGPU/NovaNVGPUOps.td          |  19 +-
 include/Transforms/Config/NVIDIATargetUtils.h      |  36 +-
 .../Transforms/Config/NovaGPULoweringConfigUtils.h |  12 +-
 include/Transforms/NovaGPUMarkerAttrs.h            |   6 +-
 include/Transforms/Passes.td                       |  13 -
 .../Transforms/Tiling/NovaConfigTrackingListener.h |   8 +-
 lib/Pipeline/GPUPipeline.cpp                       |   7 +-
 .../Config/NovaGPULoweringConfigUtils.cpp          |   4 +-
 lib/Transforms/Config/NovaKernelConfig.cpp         | 317 ++++++++---------
 lib/Transforms/Lowering/NovaGPUMapForallToGPU.cpp  | 273 +++++----------
 lib/Transforms/Memory/NovaGPUMultiBuffering.cpp    |  38 +-
 .../Tiling/NovaGPUConfigTrackingCanonicalizer.cpp  |  16 +
 .../NovaGPUConfigureTensorLayouts.cpp              |   2 +-
 .../Vectorization/NovaGPUWgmmaPatterns.cpp         | 389 +++++++--------------
 14 files changed, 471 insertions(+), 669 deletions(-)
```
Status: VERIFIED-NOW.

**ID: B2**
Answer: The turn lowering, verbatim. Its comment states the scheme.
Evidence: `git show ac2ab83:lib/Transforms/Lowering/NovaGPUMapForallToGPU.cpp | sed -n '418,459p'`
```cpp
  // Warpgroup w runs the CTA's tiles w, w + turns, ...: its loop and ring count stay the producer's, the rest of each
  // tile is guarded; the K loops take turns, each handing on `slots` steps before its end unless it is the CTA's last tile.
  if (turns > 1) {
    scf::ForOp kLoop;
    for (auto f : tileLoop.getBody()->getOps<scf::ForOp>())
      if (!kLoop && f->hasAttr(kNovaConsumerWarpgroupsAttr))
        kLoop = f;
    nvgpu::MBarrierTryWaitParityOp fullWait;
    if (kLoop)
      for (auto w : kLoop.getBody()->getOps<nvgpu::MBarrierTryWaitParityOp>())
        if (!fullWait)
          fullWait = w;
    if (!fullWait)
      return launch.emitError("consumer turns need a K loop with a full-barrier wait");
    rewriter.setInsertionPoint(tileLoop);
    Value turn = arith::DivUIOp::create(rewriter, loc, emitLaneZeroThreadId(rewriter, loc, getNovaGPUTarget(launch).preferredSubgroupSize), cst(consumerThreads / turns));
    ConsumerTurnBarriers barriers = emitConsumerTurns(rewriter, loc, turn, consumerThreads);
    Block *tile = tileLoop.getBody();
    auto yield = cast<scf::YieldOp>(tile->getTerminator());
    llvm::SetVector<Operation *> ringOps;
    SmallVector<Value> work(yield->getOperands().begin(), yield->getOperands().end());
    while (!work.empty())
      if (Operation *def = work.pop_back_val().getDefiningOp(); def && def->getBlock() == tile && ringOps.insert(def))
        work.append(def->getOperands().begin(), def->getOperands().end());
    rewriter.setInsertionPoint(yield);
    auto fromFirst = arith::SubIOp::create(rewriter, loc, tileLoop.getInductionVar(), first);
    Value ordinal = arith::DivUIOp::create(rewriter, loc, fromFirst, ctasCst);
    auto own = scf::IfOp::create(rewriter, loc, arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, arith::RemUIOp::create(rewriter, loc, ordinal, cst(turns)), turn), /*withElseRegion=*/false);
    for (Operation &op : llvm::make_early_inc_range(*tile))
      if (op.isBeforeInBlock(fromFirst) && !ringOps.contains(&op))
        op.moveBefore(own.thenYield());
    rewriter.setInsertionPoint(kLoop);
    NVVM::BarrierOp::create(rewriter, loc, barriers.self, barriers.count, /*aligned=*/false);
    const int64_t slots = cast<nvgpu::MBarrierGroupType>(fullWait.getBarriers().getType()).getNumBarriers();
    Value kSteps = arith::CeilDivSIOp::create(rewriter, loc, arith::SubIOp::create(rewriter, loc, kLoop.getUpperBound(), kLoop.getLowerBound()), kLoop.getStep());
    Value handAt = arith::AddIOp::create(rewriter, loc, kLoop.getLowerBound(), arith::MulIOp::create(rewriter, loc, arith::MaxSIOp::create(rewriter, loc, arith::SubIOp::create(rewriter, loc, kSteps, cst(slots)), cst(0)), kLoop.getStep()));
    rewriter.setInsertionPointAfter(fullWait);
    Value hasNext = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, arith::AddIOp::create(rewriter, loc, tileLoop.getInductionVar(), ctasCst), cst(tiles));
    Value atHand = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, kLoop.getInductionVar(), handAt);
    auto give = scf::IfOp::create(rewriter, loc, arith::AndIOp::create(rewriter, loc, hasNext, atHand), /*withElseRegion=*/false);
    rewriter.setInsertionPoint(give.thenYield());
    NVVM::BarrierArriveOp::create(rewriter, loc, barriers.other, barriers.count, /*aligned=*/false);
```
Status: VERIFIED-NOW.

**ID: B3**
Answer: It is a static persistent loop with no raster or swizzle ordering.
- **Tile loop:** CTA `b` runs tiles `t = b, b + P, b + 2P, …` (x-fastest delinearization, the hardware launch order).
- **Ownership:** consumer warpgroup `w` owns the CTA's tiles with ordinal `(t − b)/P ≡ w (mod 2)`. That is the same assignment as CUTLASS (consumer 1 starts one tile ahead, both step 2).
- **CTA count:** P is set in the config as `slots = wgmmaSlots(…)` (resident CTAs × SMs). The kernel becomes persistent only when there are more work units than slots, and never for cluster launches.
Evidence:
- `NovaGPUMapForallToGPU.cpp:310-320 @ ac2ab83`: "each looping over the tiles t = blockIdx.x, + CTAs, …", `scf.for %t = blockIdx.x to gx*gy*gz step P`.
- Line 373: `auto tileLoop = scf::ForOp::create(rewriter, loc, first, cst(tiles), ctasCst, …)` with `first = gpu::BlockIdOp(x)`.
- Ownership: B2 lines 443–445 (`ordinal % turns == turn`).
- `NovaKernelConfig.cpp:2546-2553 @ ac2ab83`: "Persistent CTAs… more work units than resident CTAs -> one wave looping over them. A cluster launch keeps its grid"; `if (units > slots) matmul->setAttr("nova.persistent_ctas", … slots)`.
Status: VERIFIED-NOW.

**ID: B4**
Answer:
- **Turn barriers:** ID `1 + w` (warpgroup 0 syncs on 1, warpgroup 1 on 2); a warpgroup hands off to `2 − w`.
- **Count 256:** both consumer warpgroups (2 × 128) take part in each phase, one warpgroup by `bar.sync` and the other by `bar.arrive`.
- **Other named barriers:** only `bar.sync 0` (the CTA barrier). ptxas reports "used 16 barriers" because the turn IDs are register operands, so it reserves the whole range.
- **ID clashes in the enum:** `kSplitKExchangeBarrierId = 1` shares ID 1, but no split-K exchange appears in a ping-pong kernel (only IDs 0, 1, 2 occur).
Evidence:
- `include/Transforms/NovaGPUMarkerAttrs.h:83-92 @ ac2ab83`:
  ```
  kCtaBarrierId = 0, kSplitKExchangeBarrierId = 1, kPingPongBarrierBase = 1 // 1 + w: consumer warpgroup w's turn,
  kFlashEpilogueBarrierId = 3, kConsumerBarrierId = 4, kDQFullBarrierId = 8, kDQEmptyBarrierId = 9, kRowStatsBarrierBase = 10
  ```
- `NovaGPUForallMappingUtils.h @ ac2ab83`:
  ```
  t.self = kPingPongBarrierBase + w;
  t.other = (kPingPongBarrierBase + 1) - w;
  t.count = threads;
  ```
- PTX of the ping-pong kernel (`pingpong.ptx`):
  ```
  77: bar.sync 0;
  89: barrier.arrive 1, 256;
  150: barrier.sync %r3, 256;
  576: barrier.arrive %r4, 256;
  ```
- SASS:
  ```
  BAR.SYNC.DEFER_BLOCKING 0x0
  @P0 BAR.ARV 0x1, 0x100
  BAR.SYNC.DEFER_BLOCKING R14, 0x100
  @!P0 BAR.ARV R11, 0x100
  ```
- ptxas: "Used 168 registers, used 16 barriers".
Status: VERIFIED-NOW.

**ID: B5**
Answer: **The first turn:** before its tile loop, warpgroup 1 arrives once on barrier 1 (`barrier.arrive 1, 256`, PTX line 89), which releases warpgroup 0's first `barrier.sync`. Warpgroup 0 always has tile 0, because P ≤ tiles.

**The rule:** before every K loop it owns, a warpgroup syncs on its own barrier. It arrives on the other warpgroup's barrier exactly once per owned tile, and only `if (t + P < tiles)`, i.e. only when the CTA has a next tile for the other warpgroup.

**Counting, for n tiles in a CTA:**
- There are n syncs (one per tile).
- The arrivals are the first give plus n − 1 hand-offs.
- That is n arrivals for n syncs.
- n = 1: warpgroup 1 owns nothing and syncs nothing.
- n odd: warpgroup 1's last tile (n − 2) arrives for warpgroup 0's final tile.

**No double arrival:** the arrivals alternate strictly. Warpgroup 1's second arrival on barrier 1 comes at the hand-off of its first owned tile, which it can reach only after warpgroup 0 has synced on barrier 1 and handed off to barrier 2.

**Why Kota's hang can't happen here:** there is no epilogue turn. A warpgroup never waits on the other except at its own pre-K-loop sync, and the producer waits only on ring slots.

**Tested, from 10-23:**
- K ≤ 384: `65536 × 256 × 512` (dX layout, 128×128 ping-pong) was correctly rounded in the numerics check. K = 256 and 384 also ran in the timing sweeps.
- Odd tile counts per CTA: `32768 × 2304 × 2304` has 3072 tiles over 132 CTAs, so 96 CTAs run 23 tiles.
- Fewer tiles than 2 × CTAs: `8192 × K × 768` (128×192) has 256 tiles, so 8 CTAs have one tile and their warpgroup 1 owns none. It completed in the 10-23 timing sweep.
These ran on 961e4ff + force lines, not on Aparna's c54a485 lowering.

Evidence: B2 (VERIFIED-NOW); PTX lines above (VERIFIED-NOW). Test runs: `reports/gemm_turns_results_2026-10-23.md` §7 and §1.6, and `reports/progress_2026-10-23_inst252_turns_calib.md` row 5 (FROM-EARLIER).
Status: VERIFIED-NOW (code and counting argument); FROM-EARLIER (the runs).

**ID: B6**
Answer: The hand-off happens at K-step `handAt = lb + max(kSteps − slots, 0) · step`. That is right after the full-barrier wait of step `kSteps − slots`, which is `slots` (= ring depth) steps before the end.
- Example: K = 2304 with 5 stages is 36 steps, so the hand-off is at step 31.
- If kSteps ≤ slots, it hands off at the first step.
- Yes, warpgroup 1 can issue `wgmma` while warpgroup 0 still has its last steps' `wgmma` in flight. That overlap is intended (the v2 change).
- In the main loop each warpgroup keeps one group in flight (`wait_group 1`, set by `deferStageRetire`) and drains with `wait_group 0` after the loop.
Evidence:
- PTX: `446: wgmma.wait_group.sync.aligned 1;`, `544: wgmma.wait_group.sync.aligned 1;`, `576: barrier.arrive %r4, 256;`, `584: wgmma.wait_group.sync.aligned 0;`, `592: wgmma.wait_group.sync.aligned 0;`.
- `NovaGPUMultiBuffering.cpp:740` `static LogicalResult deferStageRetire(…)`; line 817 `lastMma->setAttr(lastMma.getWaitGroupAttrName(), b.getI64IntegerAttr(1));` @ ac2ab83.
Status: VERIFIED-NOW.

**ID: B7**
Answer, for the 128×192 ping-pong kernel at K = 2304:
- **Ring:** 5 stages (`__mbarrier[40]`), BK = 64, 40,960 B per stage ((128·64 + 64·192)·2).
- **Epilogue tray:** 16 KB (8 consumer warps × 2 slots × 1 KB).
- **Total dynamic smem:** ≈ 221 KB (5 × 40,960 + 16,384).
- **Producer:** it walks the CTA's tile loop in order, so it fills tile 0 (warpgroup 0), then tile 1 (warpgroup 1), and so on.
- **Ring counters:** both consumers keep the producer's tile loop and ring counters. Only the tile body is guarded by ownership ("its loop and ring count stay the producer's"). Each consumer's slot and phase therefore advance past the other's K-iterations, the equivalent of CUTLASS's `mainloop_pipe_consumer_state.advance(k_tile_count)`.
Evidence:
- B2 lines 418–419 and 435–448 (`ringOps` stay outside the `own` guard).
- PTX `mbarrier.init.shared.b64` ×10; `__mbarrier[40]`.
- `wgmmaEpilogueTrayBytes` constants: `NVIDIATargetUtils.h:141-142` (`kWgmmaEpilogueSlotBytes = 1024; kWgmmaEpilogueSlotsPerWarp = 2;`).
Status: VERIFIED-NOW. The smem total is computed from these constants, not read from the launch.

**ID: B8**
Answer:
- **Slots:** each consumer warp has 2 slots × 1 KB of its own; nothing is shared between the two consumer warpgroups.
- **Slot reuse:** before a slot is overwritten, the warp waits on its earlier TMA stores with `cp.async.bulk.wait_group.read` (13 in the PTX).
- **Output dtype:** bf16.
- **Fused work:** the epilogue is the GEMM result's single elementwise reader: the bf16 cast, and whatever elementwise ops fused into it. Residual-seeded GEMMs now qualify too (`|| problem.seedBytes`).
Evidence:
- `NovaGPUWgmmaPatterns.cpp:1938-1940 @ ac2ab83`: `auto waitSlotsRead = [&](int32_t pending) { … NVVM::CpAsyncBulkWaitGroupOp::create(b, loc, b.getI32IntegerAttr(pending), …`.
- `grep -c 'cp.async.bulk.wait_group.read' pingpong.ptx` → 13.
- `takesTurns` at `NovaKernelConfig.cpp:1977 @ ac2ab83` (B10).
Status: VERIFIED-NOW.

**ID: B9**
Answer: The PTX has `setmaxnreg.inc 232` (consumers) and `setmaxnreg.dec 40` (producer), and ptxas honours both: the SASS contains them. ptxas gives no "setmaxnreg ignored" warning, only the C7519/C7517 notes that it injected `warpgroup.arrive` / `warpgroup.wait`.
Evidence:
- PTX: `85: setmaxnreg.inc.sync.aligned.u32 232;`, `1036: setmaxnreg.dec.sync.aligned.u32 40;`.
- SASS: `USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8` (= 232) and `USETMAXREG.DEALLOC.CTAPOOL 0x28` (= 40).
- `ptxas -arch=sm_90a -v pingpong.ptx`:
  ```
  ptxas info : (C7519) warpgroup.arrive is injected in around line 280 ...
  ptxas info : (C7519) warpgroup.arrive is injected in around line 414 ...
  ptxas info : (C7517) warpgroup.wait is injected in around line 414 ...
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  Used 168 registers, used 16 barriers, 1024 bytes smem
  ```
Status: VERIFIED-NOW.

**ID: B10**
Answer: Yes, Nova also has cooperative schedules for the same tile. There are no shape thresholds: `tryBuildHopperTile` costs every candidate and keeps the cheapest.
- **Ping-pong candidates** (one warpgroup per tile, 2 turns) are offered only when `takesTurns` holds: a zero-filled or tray-seeded accumulator, a single elementwise reader with an identity map, and A K-major. They also need more tiles than resident CTAs.
- **Cost:**
  ```
  turns > 1 ? max(kLoop, epilogue) : kLoop + epilogue
  ```
  with `kLoop` = tensor work / (tile efficiency × wgmma issue efficiency), and the issue wait shared across the warpgroups issuing into the tile.
Evidence: `NovaKernelConfig.cpp @ ac2ab83`:
```
1300: if (problem.takesTurns)
1409: if (t.turns > 1 && blocks <= residency * target.smCount)
1441: : double(waves * inFlight) * (t.turns > 1 ? std::max(kLoop, fixed) : kLoop + fixed);
1977: problem.takesTurns = e && isAccumulatorElementwise(e, /*blocks=*/true) && a.getResults().back() == getAffineDimExpr(cd->k.back(), …) && …
```
Status: VERIFIED-NOW.

---

## C. Compiled-kernel proof

**Kernel pair:**
- **Ping-pong:** `16384 × 2304 × 2304`.
- **Cooperative:** `16512 × 2304 × 576`. This shape was chosen so the cost model itself picks cooperative 128×192; at 16384 × 2304 × 2304 it picks ping-pong, and with a single-GEMM graph you can't force either schedule.
- **Same:** tile 128×192, 5 stages, K = 2304, layout `x·W`, bf16.
- **Different:** M and N, because there is no force switch in the current build. A same-shape pair needs a temporary force line; I can do that on request.

**Commands:**
```
nova-opt <f>.mlir --nova-gpu-optimized-pipeline=arch=sm_90 > out.mlir   # PTX taken from gpu.binary
ptxas -arch=sm_90a -v <k>.ptx -o <k>.cubin
cuobjdump -sass <k>.cubin
```
Files: `reports/pingpong_crosscheck_2026-10-26/`. Status for C1–C4: VERIFIED-NOW.

**ID: C1** — ptxas

| | ping-pong | cooperative |
|---|---|---|
| Registers (launch) | 168 | 122 |
| Spill stores / loads | 0 / 0 | 0 / 0 |
| Static smem | 1024 B | 1024 B |
| Barriers | 16 | 1 |

**ID: C2** — PTX

| | ping-pong | cooperative |
|---|---|---|
| `.maxntid` | 384,1,1 | 384,1,1 |
| `setmaxnreg` | inc 232 ×1, dec 40 ×1 | none |
| `bar.sync` | `0` ×1 | `0` ×1 |
| `barrier.sync` | `%r3, 256` ×1 | none |
| `barrier.arrive` | `1, 256` ×1, `%r4, 256` ×1 | none |
| `wgmma.mma_async` | m64n192k16 ×16 | m64n192k16 ×4 |
| `wgmma.wait_group` | 1 ×2, 0 ×2 | 0 ×2 |
| `mbarrier.*` | init ×10, try_wait.parity ×4, arrive ×2, arrive.expect_tx ×2 | init ×10, try_wait.parity ×3, arrive ×1, arrive.expect_tx ×2 |

**ID: C3** — SASS opcode counts

| Opcode | ping-pong | cooperative |
|---|---|---|
| HGMMA.64 | 16 | 4 |
| BAR.ARV | 2 | 0 |
| BAR.SYNC.DEFER_BLOCKING | 2 | 1 |
| USETMAXREG.DEALLOC.CTAPOOL | 1 | 0 |
| USETMAXREG.TRY_ALLOC.CTAPOOL | 1 | 0 |
| SYNCS.ARRIVE.TRANS64 | 2 | 2 |
| SYNCS.ARRIVE.TRANS64.A1T0 | 2 | 1 |
| SYNCS.EXCH.64 | 10 | 10 |
| SYNCS.PHASECHK.TRANS64 | 4 | 3 |
| SYNCS.PHASECHK.TRANS64.TRYWAIT | 8 | 6 |
| WARPGROUP.ARRIVE | 4 | 1 |
| WARPGROUP.DEPBAR.LE | 5 | 2 |
| UTMALDG.2D | 8 | 8 |
| UTMASTG.2D | 12 | 6 |

**Opcodes present only in the ping-pong kernel:** `BAR.ARV`, `USETMAXREG.DEALLOC.CTAPOOL`, `USETMAXREG.TRY_ALLOC.CTAPOOL`, plus uniform-integer compares for the ownership and hand-off tests (`UISETP.*`, `ISETP.GT.U32.*`, `ULEA.HI.X`, `UPRMT`).

**Opcodes present only in the cooperative kernel:** ordinary integer/address ops (`IMAD.*`, `LEA`, `SEL`, `MOV`, `PRMT`, `SHF.R.U64`, `CS2R`, `LDC.64`, `ISETP.EQ.U32.XOR`, `UISETP.GT.U32.AND`).

So in **Nova's own** kernels, the turn shows up as `BAR.ARV` plus a register-ID `BAR.SYNC` with 256 threads. Nova's cooperative kernels have no register split, which is unlike CUTLASS cooperative (F1).

**ID: C4**
Answer: Yes, 192 accumulator registers per consumer thread.
- Each `m64n192k16` operand list carries 96 f32 accumulators per thread.
- A consumer warpgroup owns two m64 blocks, so 2 × 96 = 192 = 128·192/128.
- 0 spills.
- The highest register index in the ping-pong SASS is R215, so 216 registers are used under the 232 cap. Gautam's §6 estimate of about 218 is close.
Evidence: PTX operand count (96); SASS register scan (`R215` ping-pong, `R119` cooperative); ptxas "0 bytes spill".
Status: VERIFIED-NOW.

---

## D. Performance proof

All of these are FROM-EARLIER, from records written at the time. None were re-measured today (no GPU).

**ID: D1** (v1, "17–20% slower")
Answer: v1 was measured with ncu on the 1.58B one-layer model, using the 9 GEMMs that switched.
- **Setup:** `bluscript_single.cpp`, B 8, T 4096, 32 micro-steps; H200, p022 GPU 6, 2026-10-21. Clocks were not locked (`--clock-control none`, cycles compared).
- **Samples:** one launch per kernel (step 2, micro-step 0).
- **Baseline:** Nova cooperative 128×256 (water21).
- **Result:** "all 9 slower: +5.4% (K39) .. +22.5% (K21); sum +20.1% (1 layer), +17.3% 26L-weighted".
- **Tile:** the ping-pong tile was 128×192, so not the same tile as the baseline.
- **Diagnosis recorded at the time:** "in its turn ONE warpgroup … drains it every stage (… wait 0 …)". That was a drain problem, not a proof that strict ordering costs 17–20%.
Evidence: `reports/p022_2026-10-21/progress_2026-10-21_p022_pingpong.md` row 1b and job notes 1b.
Status: FROM-EARLIER.

**ID: D2** ("−3.4%")
Answer: **My earlier wording was wrong.**
- **v2's −3.39%** was the summed ncu cycles of the 9 GEMMs, weighted to 26 layers, not the step time.
- **The step-level result is v3: −1.19%.**
  - **Model:** 1.58B, 26 layers, d 2304, B 8, T 4096, AdamW.
  - **Runs:** 8 steps per run, steps 2–7 averaged, 2 runs per arm.
  - **Step times:** ping-pong 20,975.5 / 21,115.5 ms; no ping-pong 21,300.1 / 21,296.8 ms; means 21,045.5 vs 21,298.5.
  - **Clocks:** all four runs at 1480–1496 MHz, power-capped at 691–692 W.
  - **Baseline:** the same Nova tree without ping-pong (Nova cooperative). Not cuBLAS, not BluBLAS.
- **Later (10-23, final `kidd`, root@95.133.252.52 GPU 0):** against v4 + the per-step cost build, nsys, 6 steps, steps 2–5, one run each: d512 37-layer −2.00%, 1.58B 26-layer −0.37%.
Evidence: p022 progress file rows 2, 4, 5 and the "4-5" table; `reports/gemm_turns_results_2026-10-23.md` §4–5.
Status: FROM-EARLIER.

**ID: D3**
Answer: Per-GEMM Nova ping-pong (v3, 128×192) against Nova cooperative (128×256), ncu cycles, 1.58B one-layer model, H200:

| Kernel | Cooperative | Ping-pong | Change |
|---|---|---|---|
| Q fwd | 725,510 | 690,750 | −4.8% |
| K/V fwd | 251,905 / 250,147 | 239,384 / 237,657 | −5.0% |
| gate/up fwd | 3,771,970 | 3,569,389 | −5.4% |
| down dX | 1,888,967 | 1,816,101 | −3.9% |
| O dX | 722,446 | 693,561 | −4.0% |
| K/V dX | 278,464 | 240,454 | −13.7% |
| gate/up dX | 3,672,794 | 3,658,370 | −0.4% |
| LM-head dX | 14,835,152 | 14,975,153 | +0.9% |

There is no cuBLAS comparison against the ping-pong build (NOT-AVAILABLE). The only cuBLAS comparison is from 10-10, before ping-pong (memory note `nova-vs-cublas-h200-bf16`: 21 GEMMs at 1.27× cuBLAS cycles).
Evidence: `reports/gemm_turns_cost_plan_2026-10-22.md` §3.1.
Status: FROM-EARLIER.

**ID: D4**
Answer: Yes, tensor-pipe activity rises with ping-pong.
- **ncu `sm__pipe_tensor_op_hmma_cycles_active`, ping-pong vs cooperative:** gate/up fwd 91.0 → 96.1%, Q fwd 88.7 → 93.2%, K/V dX 77.0 → 89.2%.
- **Mechanism profile, 10-23,** ping-pong 128×192 at 32768 × 512 × 2304:
  - 3 → 4 stages: 74.8 → 89.6% tensor-active;
  - TMA-data waits fell 35%.
- **Standalone runs, K 512–4096:** 128×192 ping-pong at 5 stages ran at 97–99% tensor-active.
Evidence:
- `gemm_turns_cost_plan_2026-10-22.md` §3.1;
- `gemm_turns_results_2026-10-23.md` §1.4 and §1.10;
- `progress_2026-10-23_inst252_turns_calib.md` row 3.
Status: FROM-EARLIER.

---

## E. Correctness proof

**ID: E1**
Answer: On 10-23 the `kidd` 961e4ff build was checked against a double-precision reference.
- **Reference:** bf16 inputs U(−1, 1), the result multiplied in double; two rows per 128-row tile (one in each m64 half), all columns; error in bf16 ULP.
- **Result:** every ping-pong kernel was max 0.50 ULP, 0 elements over 1 ULP, no NaN/Inf and no unwritten rows. That covers:
  - 128×192: Q/K-V/gate-up forward, and down / K-V / Q-O dX;
  - 128×128 at 6 stages: d512.
- **Gautam's four cases:**
  - **K ≤ 384:** 65536 × 256 × 512 (dX layout) ✓.
  - **Odd tiles per CTA:** e.g. 32768 × 2304 × 2304, where 96 of 132 CTAs run 23 tiles ✓.
  - **Fewer tiles than 2 × CTAs:** not in the numerics check. It was only run, and completed, in the timing sweep (8192 × K × 768).
  - **M or N not a multiple of the tile:** never produces ping-pong, because `tryBuildHopperTile` skips any tile that doesn't divide M, N and K (`problem.mSize % t.wgM != 0 || problem.nSize % t.wgN != 0 ||`, unchanged since 50327d6, 2026-08-12).
- **Not re-run** on Aparna's c54a485 lowering.
Evidence: `gemm_turns_results_2026-10-23.md` §7. Harness: `reports/gemm_numerics.cpp`; job `reports/run_numerics.sh`.
Status: FROM-EARLIER (runs); VERIFIED-NOW (divisibility rule).

**ID: E2**
Answer: compute-sanitizer `racecheck` / `synccheck` has never been run on these kernels.
Status: NOT-AVAILABLE. There is no GPU in this session; the 10-23 instance is gone.

**ID: E3**
Answer: There was no explicit timeout. Shapes where one consumer owns no tile in some CTAs (8192 × K × 768, 256 tiles over 132 CTAs) completed in the 10-23 sweeps on 961e4ff.
Status: FROM-EARLIER. A timeout run on the current build is NOT-AVAILABLE.

---

## F. My earlier reply

**ID: F1**
Answer: **Gautam is right.** CUTLASS cooperative uses the same 40/232 register split, so "a cooperative kernel has neither the register split nor…" is wrong for CUTLASS. It is true only for **Nova's** cooperative kernels: C2 and C3 show no `setmaxnreg` / `USETMAXREG` in them.
Evidence: §1–7 table (CUTLASS cooperative source); C2 and C3.
Status: VERIFIED-NOW.

**ID: F2**
Answer: **Gautam is right.** CUTLASS's turn is `OrderedSequenceBarrier` → `ClusterBarrier`, an mbarrier, which appears as `SYNCS.*` in SASS, not `BAR.ARV`. `BAR.ARV` with 256 threads identifies **Nova's** named-barrier turn only. The spelling is `USETMAXREG.TRY_ALLOC.CTAPOOL` (our SASS: `USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xe8`).
Status: VERIFIED-NOW.

**ID: F3**
Answer: Here is what was read, what was run, and what came from memory, statement by statement.

| Earlier statement | Source then | Correct? |
|---|---|---|
| CUTLASS ping-pong source, docs, PyTorch blog, FA3 quotes | fetched in that session | ✓ (re-checked today) |
| Nova `NovaGPUMapForallToGPU.cpp` lines 418–459 | read in that session | ✓ |
| PTX facts (384 threads, `setmaxnreg` 40/232, `barrier.sync` / `arrive` 256, TMA) | compiled in that session, but with a stale `nova-opt` (built 10-24 18:57, before ac2ab83) | ✓ the facts reproduce on today's HEAD build (C2) |
| "your later c54a485" | inferred, not checked | ✗ the author is Aparna (B1) |
| "v1 17–20% slower with CUTLASS-style strict ordering" | memory | numbers ✓ (D1), cause ✗: the record blames the per-stage drain |
| "v2: −3.4% on the 26-layer step" | memory | ✗ GEMM-cycle number; the step is −1.19% (D2) |
| "A cooperative kernel has neither the register split…" | my generalization, unchecked | ✗ for CUTLASS (F1) |
| "named BAR.SYNC/BAR.ARV … or a CUTLASS ping-pong kernel" | my generalization, unchecked | ✗ for CUTLASS (F2) |
| "USETMAXREG.TRYALLOC" | memory | ✗ spelling (F2) |
| "I didn't record a citation" | true | ✓ the 10-21 record names CUTLASS concepts but no URL or commit |

Status: VERIFIED-NOW.

---

## G. Anything Gautam got wrong

**ID: G1**
Answer: I found no error in his §1–7 on any point I could check:
- the CUTLASS quotes (re-fetched today);
- the BluBridge-BLAS facts (local `ea666ab`, `.gitmodules`, the never-included template);
- §6's register estimate (measured: 216 registers, R215);
- §7 F1/F2.

I can't check his cuBLAS research (§5); I have no data on it. Two additions:
1. **CUTLASS cooperative does contain an `OrderedSequenceBarrier`.** It is `LoadWarpOrderBarrier<1,2>`, inside the producer, so "cooperative has no consumer-to-consumer turn barrier" stays exact.
2. **On the current build, ping-pong also covers the residual-seeded forward GEMMs** (O and down forward, through the epilogue tray), which §4.3 didn't list. Those cases come from Aparna's c54a485 (`… || problem.seedBytes`).
Status: VERIFIED-NOW.

---

## Summary

- **Is Nova's schedule ping-pong?** Yes, by CUTLASS's definition. Each consumer warpgroup owns a whole 128×192 tile; the two consumers alternate over the CTA's tiles (ordinal mod 2, matching CUTLASS); and a turn barrier orders their K loops, so one consumer's epilogue overlaps the other's MMAs. All three are shown in the code (B2, B3) and the compiled kernel (C2, C3).
- **Implementation choices, not a different schedule:**
  - a named-barrier turn instead of an mbarrier;
  - a hand-off `slots` K-steps early;
  - no epilogue turn, safe because each consumer warp has its own epilogue slots;
  - register split 40/232, which CUTLASS cooperative uses too.
- **Not proven yet:**
  - sanitizer runs;
  - a timeout run on the current build;
  - a same-shape ping-pong vs cooperative SASS pair, which needs a force line;
  - correctness on Aparna's c54a485 lowering (the numerics check predates it).
