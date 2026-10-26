# GEMM "ping-pong" cross-check: BluBLAS (Gautam), Kota's BluBridge-BLAS, and Nova (Harshini)

From: Gautam (BluBridge)
Date: 2026-10-26 (Monday), IST
For: the Claude session on Harshini's system (and Harshini)

---

## 0. Why I am sending this, and what I need back

Harshini said Nova (the compiler) now has a ping-pong GEMM schedule, and that she took the
reference from "our BluBLAS in remote BluTrain". I asked you (the Claude on her system) about it,
and you gave the reply that is copied in full in **Appendix A**.

I checked that reply against:
- the CUTLASS source,
- the CUTLASS docs,
- the PyTorch blog,
- the team BluBridge-BLAS repo (all 15 remote branches),
- my own SM90 BluBLAS work and research notes from 2026-07 to 2026-09.

This document has four parts:
1. Everything I found, with proof: file, line, commit and verbatim quotes, so you can check each item yourself.
2. Two statements in your earlier reply that I believe are wrong (Section 7). Please confirm or refute them with code.
3. A numbered list of questions about Nova's code, compiled kernels, timings and tests (Section 8).
4. The answer format I need (Section 9), so I can compare everything in one pass.

Please answer only from files you read or commands you run in this session. Mark anything you
take from memory as such. If I got something wrong anywhere in Sections 1-7, say so with proof.

My own questions, in my words (typos fixed, meaning unchanged), are in **Appendix B**.

---

## 1. Three different things are called "ping-pong"

| # | Name | What it is | Where |
|---|---|---|---|
| 1 | **GEMM ping-pong schedule** (the one in question) | 1 producer warpgroup + 2 consumer (math) warpgroups. **Each consumer owns a different output tile.** One consumer's epilogue runs while the other consumer's MMAs run. | CUTLASS `KernelTmaWarpSpecializedPingpong`, `sm90_gemm_tma_warpspecialized_pingpong.hpp` |
| 2 | **Attention ping-pong** (FlashAttention-3 §3.1 "Pingpong scheduling") | Two warpgroups in an attention kernel take turns, with `bar.sync`, so one does softmax while the other does its GEMMs | FA3, arXiv 2407.08608. My attention forward kernel also has this kind of turn passing between warpgroups |
| 3 | **Buffer ping-pong** (double buffering) | Two buffers used alternately, for example epilogue smem slots or two TMEM accumulators | Many places; this is not a schedule |

Only meaning #1 is in question here. Meaning #3 appears in Kota's B300 code (Section 4.2.3).

---

## 2. The official definition (CUTLASS), verbatim

My local CUTLASS checkout is at commit `546c3ef` (2026-05-20, "Fix examples and pytest, run ruff
(#3230)"). Line numbers may differ in your version, so I also give the text to search for.

### 2.1 CUTLASS docs: `media/docs/cpp/efficient_gemm.md` (lines 240-245 in 546c3ef)

> **Warp-Specialized Persistent Ping-Pong kernel design**
> The third kernel design is the *Warp-Specialized Persistent Ping-Pong* kernel.
> Like the Warp-Specialized Persistent Cooperative, kernel the concepts of warp groups, barrier
> synchronization between warp groups, and the shape of the grid launch remain the same in the
> persistent ping-pong design.
> The distinctive feature of the Warp-Specialized Persistent Ping-Pong kernel is the following :
> * The two *consumer* warp groups are assigned a different output tile using the Tile Scheduler.
>   This allows for *epilogue* of one *consumer* warp group to be overlapped with the math
>   operations of the other *consumer* warp group - thus maximizing tensor core utilization.
> * The *producer* warp group synchronizes using the Ordered Sequence Barrier to fill buffers of
>   the two *consumer* warp groups one after the other in order.

The same file, describing **Cooperative**:

> * Presence of two *consumer* warp groups cooperating on the same output tile by splitting the
>   tile in half across the M dimension. This allows for larger tile sizes to be enabled - since
>   the register pressure per *consumer* warp group is reduced - and hence improving performance.

**So the defining property is tile ownership:**
- Ping-pong: each consumer has its own tile.
- Cooperative: both consumers share one tile.

### 2.2 CUTLASS source: `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp`

**Structure and registers** (lines 132-158):
```cpp
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  ...
  static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
  static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    (size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator))
    / (NumMMAThreads * sizeof(uint32_t));
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
  ...
  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;
```

**Tile assignment**: consumer 1 starts one tile ahead (lines 558-566):
```cpp
    if (warp_group_role == WarpGroupRole::Consumer1) {
      if constexpr (not IsSchedDynamicPersistent) {
        // Advance 2nd Math WG to the next work tile for the startup
        scheduler.advance_to_next_work();
      }
      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
```

After each tile, both consumers step by 2 (line 952):
```cpp
        scheduler.advance_to_next_work(NumMmaWarpGroups);
```

**MMA turn** (lines 849-880). The arrive comes after `mma()` has issued all MMAs, and before
`mma_tail()` waits for them:
```cpp
        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();
        ...
          collective_mainloop.mma( ... );
        ...
        // Cue for next Math WG's MMA to start
        math_wg_order_barrier.arrive();

        // Make sure the math instructions are done and free buffers before entering the epilogue
        collective_mainloop.mma_tail( ... );
```

**Epilogue turn** (lines 898-936):
```cpp
        // Order two Math WG's Epilogue one after the other
        math_wg_order_barrier.wait();
        // Epilogue and write to gD
        ...
        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();
```

### 2.3 The turn barrier is an mbarrier, not a named barrier

`include/cutlass/pipeline/sm90_pipeline.hpp` (lines 1255-1258):
```cpp
template<int SequenceDepth, int SequenceLength>
struct OrderedSequenceBarrierSharedStorage {
  using Barrier = cutlass::arch::ClusterBarrier;
  Barrier barrier_[SequenceDepth][SequenceLength];
};
```

`ClusterBarrier` is a shared-memory mbarrier. In SASS, mbarrier operations appear as `SYNCS.*`
instructions, **not** as `BAR.SYNC` / `BAR.ARV` (those are named barriers).

### 2.4 CUTLASS **cooperative** has the same register split

`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`:
- lines 146-151:
  ```cpp
    static constexpr bool HeavyRegisterPressure =
      IsSm120BlockScaled<DispatchPolicy>::value ? (RegsPerThread >= 128) : (RegsPerThread >= 208);

    static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
    static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
  ```
- line 561: `cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();` (producer)
- line 771: `cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();` (consumers)
- line 372: `static_assert(NumMMAThreads == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");`
  It is also 1 producer warpgroup + 2 consumer warpgroups, so 384 threads.
- `grep -c MathWarpGroupOrderBarrier` on this file returns **0**: cooperative has no consumer-to-consumer turn barrier.

**Conclusion:** 384 threads and setmaxnreg 40/232 appear in **both** schedules. Neither one proves
ping-pong. Only two things are specific to ping-pong:
- each consumer gets its own tile, and the tiles alternate;
- a turn barrier between the two consumers.

### 2.5 PyTorch blog (checked 2026-10-26)

"Deep Dive on CUTLASS Ping-Pong GEMM Kernel", Less Wright and Adnan Hoque, 2024-11-01,
https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/ . Verbatim quotes:

> "...give up some of their register space to the consumer warps (keeping only 40 registers,
> while consumers will get 232)."

> "Producers will reduce their max register counts by 40, vs consumers will increase their max
> register count by 232, an effect we can see in the CUTLASS source and corresponding SASS:
> Unique to Ping-Pong, each consumer works on separate C output tiles. (For reference, the
> cooperative kernel is largely equivalent to Ping-Pong, but both consumer groups work on the
> same C output tile)."

### 2.6 FlashAttention-3 §3.1

FA3 (Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao; arXiv 2407.08608), §3.1 "Pingpong
scheduling", uses `bar.sync` named barriers to make one warpgroup's GEMMs run before the other's.
That way one warpgroup does softmax while the other does GEMMs. It is about attention (meaning #2
in Section 1), but its named-barrier primitive is a valid way to build a GEMM turn barrier too.

---

## 3. What the remote BluBLAS is (checked 2026-10-26 with `git ls-remote` and a bare clone)

**How it is wired:**
- BluTrain (`github.com/BlubridgeAI/BluTrain`) uses Tensor-Implementations as a submodule, and that uses **BluBridge-BLAS** as a submodule.
- From `Tensor-Implementations/.gitmodules`:
  ```
  [submodule "BluBridge-BLAS"]
      path = BluBridge-BLAS
      url  = https://github.com/kathir-23s/BluBridge-BLAS
      branch = master
  ```
  (`github.com/BlubridgeAI/BluBridge-BLAS` returns the same refs.)

**What master is:**
- Remote `master` = `ea666ab`: "Merge pull request #3 from BlubridgeAI/production_blas", Kathir, 2026-08-31 14:35:51 +0530.
- Harshini's screenshot shows this same commit.
- Its 4 changed files (+247 / -22) come from `969fda4` "optimized addmm kernel", dhanakumarK777, 2026-08-27.
- The SM90 code in this repo was written by **Kota Dhanakumar** (git name `dhanakumarK777`).

**All 15 remote branches** (`for-each-ref` on a fresh bare clone):

| Branch | Head | Date | Author | Subject |
|---|---|---|---|---|
| master | ea666ab | 2026-08-31 | Kathir | Merge pull request #3 from BlubridgeAI/production_blas |
| production_blas | 969fda4 | 2026-08-27 | dhanakumarK777 | optimized addmm kernel |
| prod_V | 7d1f1b9 | 2026-08-29 | root | 29/aug |
| preproduction-135M | 35e51a0 | 2026-08-24 | Kathir | Merge pull request #2 from BlubridgeAI/production_blas |
| present-25/8 | 35e51a0 | 2026-08-24 | Kathir | Merge pull request #2 from BlubridgeAI/production_blas |
| checkkkkkk | 2b96901 | 2026-09-08 | Vishnu-BB | addmm + residual add fused kernel +gelu and GEMM+ loss ... |
| madhu_tried_compiler_fusions_gpt2 | 2b96901 | 2026-09-08 | Vishnu-BB | (same commit as checkkkkkk) |
| master_du | 0ca2e29 | 2026-08-22 | vishn-BB | 22/8 |
| dhana | 80547e5 | 2026-08-01 | dhanakumarK777 | Remove cluster/coopB kernel routing from SM90 BF16 dispatcher |
| production_dhana | 650bc5c | 2026-08-01 | root | clusster 2 errors while running script fixed |
| production_kathir | 3a2e494 | 2026-07-31 | kathir-23s | naive run - week 1 |
| muon_blas | 86ebe2e | 2026-08-04 | vishnu-BB | fixed convert_type_cuda kernel & nan/inf errors... |
| master_v2 | fed19b6 | 2026-06-10 | kathir-23s | Repo revamp |
| master-instance-move-test | 16a4539 | 2026-06-13 | sairupesh36 | Changes 407k run |
| master-instance | 1be8c44 | 2026-04-28 | sairupesh36 | fixed run on server |

None of these branches is my work. My hybrid BluBLAS (Section 4.1) was never pushed to this repo
or to BluTrain.

---

## 4. What each side has

### 4.1 Mine (Gautam): SM90 BluBLAS and the "hybrid": **no ping-pong**

**My SM90 BF16 kernel template** (`Bgemm_sm90_warpspecialized_template.cuh`, header lines 27-32):
```
// Warp-group layout (384 threads/CTA = 12 warps):
//   warps [0, 4)   consumer warpgroup 0 — wgmma + epilogue for M-rows [0, WG_M)
//   warps [4, 8)   consumer warpgroup 1 — wgmma + epilogue for M-rows [WG_M, 2*WG_M)
//   warps [8, 12)  producer warpgroup   — TMA issue only (lane 0 of warp 8)
// Both consumer warpgroups cooperate on ONE output tile (CUTLASS's "cooperative"
// scheme, not "pingpong") — higher throughput for large dense batched GEMM.
```

**The hybrid:**
- It is Kota's production BLAS at `ea666ab` with my kernel template and registry merged in.
- It is in my personal repository only (last commit 2026-09-18). It is not in the team repo or in BluTrain.
- Hybrid steps:
  - 2026-09-02: snapshot of `ea666ab` as the baseline.
  - 2026-09-02: removed the 11 SM90 files that are not compiled at all (5,586 lines). This **included Kota's `Bgemm_sm90_pingpong_template.cuh`**.
  - 2026-09-02: removed coopB / coopB_v2.
  - 2026-09-10: the hybrid dispatcher became the only path.
  - 2026-09-11: last hybrid kernel change.
- So the hybrid has no ping-pong schedule either.

**My research notes record ping-pong as not built:**
- §H17: "both coopB(odd-tiles) and pingpong are undone".
- §H158: "**We have no Pingpong** ... **Pingpong is the named, documented feature to build if BM=64 is ever wanted.**"
- Full quotes are in Appendix C.

### 4.2 Kota's (team BluBridge-BLAS, remote `master` = `ea666ab`)

#### 4.2.1 Production SM90 kernels are cooperative

`Bgemm_sm90_swiglu_fused_template.cuh:104` and `FP16/SM90/Hgemm_sm90_warpspecialized_template.cuh:104`
both say: `scheme, not "pingpong") — higher throughput for large dense batched GEMM.` That is the
same sentence as in my template header.

#### 4.2.2 `src/level3/batched_gemm/BF16/SM90/Bgemm_sm90_pingpong_template.cuh` (680 lines): a real ping-pong design, never compiled

**History:**
- Added in `577d5e0` (2026-08-17); last changed in `526d8b4` (2026-08-24). Both by dhanakumarK777.
- Present on 8 of the 15 branches: master, production_blas, prod_V, preproduction-135M, present-25/8, checkkkkkk, madhu_tried_compiler_fusions_gpt2, master_du.

**Never compiled, on any branch.** I checked every remote branch that has the file. **No other
file includes it or uses any of its symbols** (`Sm90PingpongConfig`, `bgemm_sm90_pingpong_kernel`,
`launch_bgemm_sm90_pingpong`, the include guard). No build file (Makefile / CMake / .mk / .txt) on any branch mentions it. You can verify:
```
git grep -n -i "include.*pingpong" ea666ab -- .
git grep -l "Sm90PingpongConfig\|bgemm_sm90_pingpong_kernel\|launch_bgemm_sm90_pingpong" ea666ab -- . | grep -v pingpong_template
```
Both commands return nothing. So the file is never part of the build and never runs.

**It is still a true CUTLASS-style ping-pong.** All three ping-pong parts are in the code:

| Part | Kota's file (ea666ab), line | CUTLASS pingpong.hpp, line |
|---|---|---|
| Each consumer owns alternating tiles | 410-412: `for (int task_id = (int)blockIdx.x + g * (int)gridDim.x; task_id < total_tiles; task_id += 2 * (int)gridDim.x)` | 558-562 (consumer 1 starts one tile ahead), 952 (`advance_to_next_work(NumMmaWarpGroups)`) |
| MMA turn between consumers | 460: `mbarrier_wait(mainloop_turn_addr + g * 8, ...)` before the K-loop; 514: `mbarrier_arrive(mainloop_turn_addr + (1 - g) * 8)` after it | 850 wait, 877 arrive |
| Epilogue turn between consumers | 521: `mbarrier_wait(epilogue_turn_addr + g * 8, ...)`; 600: `mbarrier_arrive(epilogue_turn_addr + (1 - g) * 8)` | 899 wait, 936 arrive |
| Register split | 321: `setmaxnreg_dec<Config::PRODUCER_REGS>()`; 364: `setmaxnreg_inc<Config::CONSUMER_REGS>()`; defaults 24/240 (line 137) | 147-148 (24/240 is CUTLASS's heavy-register value) |
| Turn barrier type | mbarriers (lines 279-280, 291-294) | mbarrier (`ClusterBarrier`) |

**Differences from CUTLASS:**
- **Full drain every k-step.** Line 502 (`wgmma_wait_group<0>()` inside the K-loop) waits for all MMAs at every k-step. CUTLASS keeps several MMAs in flight and drains only in `mma_tail`. The schedule is the same; the mainloop is slower.
- **One shared epilogue buffer.** Both consumers share a single epilogue store buffer, so the epilogue turn is needed for correctness. The file's own comment at lines 586-600:
  > "Enabling it requires PER-WARPGROUP store buffers, doubling STORE_BUF_BYTES 32768 -> 65536 and
  > cutting LOAD_PIPE_DEPTH from 6 to (232448-65536)/32768 = 5. That also removes the need for
  > epilogue_turn entirely. Deliberately NOT done in this pass"

**Its own validation status (header, lines 112-118, verbatim):**
```
// VALIDATION STATUS (corrected 2026-08-24). This file HAS been run on
// hardware and it HUNG -- see the mainloop_turn note above. The hang was
// diagnosed and fixed by making the ordering reference-faithful, and the fixed
// version has NEVER BEEN RE-RUN. So: not "untested", but "tested, failed,
// fixed, unretested" -- a materially different state, ...
```

**Why it hung (header, lines 60-66, verbatim):**
```
// producer can never run far enough ahead for BOTH warpgroups to hold
// resident tile data at once. At BM=128/BN=128 STAGE_BYTES is 32768, so
// depth is ~6. That assumption holds at K=2048 (k_iters=32) and FAILS at
// K<=384 (k_iters<=6): both warpgroups enter MMA together, the turn-passing
// that was supposed to emerge from backpressure never happens, and with
// epilogue_turn still ordering the downstream stage you get a lock-order
// inversion -- the hang.
```

**Its motivation (header, line 8 onward):** dated 2026-08-06. Kota's best kernel (`bn256_rs`)
"still trails the baseline by a uniform 0.4-9%", and "ncu measured 15-18pp of unused tensor pipe
at K=1024".

**Summary for 4.2.2:**
- It is the real CUTLASS ping-pong structure.
- It hung on its first hardware run.
- It was fixed but never run again, and it is never compiled.
- It never contributed to any measured BluBLAS number.
- My hybrid deleted it on 2026-09-02, recording "VERDICT RECORDED: none. Motivation only — abandoned or unfinished, not measured-and-rejected."

#### 4.2.3 B300 (SM103): a different thing with the same name

`src/level3/batched_gemm/BF16/SM103/Bgemm_sm103_cluster_dual_template.cuh` (added in 526d8b4,
2026-08-24), lines 24 and 129-132:
```
// existing epilogue/MMA wave-to-wave overlap (EPI_PIPE_DEPTH=1 here, vs the
// single-consumer kernel's EPI_PIPE_DEPTH=2 ping-pong) ...
    // EPI_PIPE_DEPTH=2 restores the single-consumer kernel's wave-to-wave
    // ping-pong (next tile's MMA can start while this tile's epilogue is
    // still draining TMEM) at the cost of halving BN to stay within the
    // 512-column TMEM budget (NUM_CONSUMERS * BN * EPI_PIPE_DEPTH <= 512).
```
This is two TMEM accumulator buffers on Blackwell. The goal is the same overlap, but there are no
two math warpgroups taking turns. It is meaning #3 in Section 1, and it does not apply to Hopper.

### 4.3 Nova (Harshini), as described in your earlier reply. **Not verified by me.**

From Appendix A:
- Commit `c54a485` "persistence, stream-k and ping-pong"; tree at `ac2ab83`.
- File `NovaGPUMapForallToGPU.cpp`, lines 418-459.

| Property | CUTLASS ping-pong | Nova (your description) | My comment |
|---|---|---|---|
| Warpgroups / threads | 1 + 2, 384 | `.maxntid 384` | Same in cooperative, so not a marker |
| Register split | 40/232 | setmaxnreg.dec 40 / .inc 232 | Same in cooperative, so not a marker |
| Tile per consumer | own tile | each warpgroup owns a whole 128×192 tile (m64n192 × 2 m-blocks) | **This is the defining property.** Needs code proof |
| Tile assignment | consumer 1 one tile ahead, both step 2 | warpgroup w runs the CTA's tiles w, w+2, … (lines 418-448) | Matches, if the code shows it |
| Turn barrier | mbarrier (`OrderedSequenceBarrier`) | `barrier.sync self, 256` / `barrier.arrive other, 256` (lines 450, 459) | FA3-style named barrier. A valid alternative |
| Turn hand-off point | after all MMAs are issued, before `mma_tail` | some K-steps before the end of the K-loop (line 453) | Earlier hand-off; needs the exact value and a safety argument |
| Epilogue order | ordered (second turn) | not ordered: each consumer has its own epilogue slots | This is what Kota's file (lines 586-600) said was needed to drop the epilogue turn |
| Performance | — | v1 (strict ordering) 17-20% slower than cooperative; v2 −3.4% on the 26-layer step | No measurement detail given |

**My reading:**
- If Nova's code really gives each consumer its own tile, alternates the tiles and passes a turn
  between the consumers, then it **is** ping-pong by the CUTLASS definition (Section 2.1).
- The early hand-off, the unordered epilogue and the named barriers are implementation choices, not
  a different schedule.
- Nova's design is **not** a copy of Kota's file. Kota uses mbarriers, an ordered epilogue, and hands
  off after a full drain. Nova uses named barriers, an unordered epilogue, and hands off early.

---

## 5. My cuBLAS research (2026-07 to 2026-09): what it proved and what it did not

Full quotes are in Appendix C. In short:

1. **The definition was found and quoted.** §H24 quotes `efficient_gemm` for both schedules. CUTLASS issue #2181 ("why both schedules?") has no maintainer answer, so there is no NVIDIA guidance on which schedule to choose.
2. **cuBLAS uses ping-pong. Proven by register arithmetic, not by an instruction.**
   - Every nvjet sm90 kernel runs with 384 threads (ncu Block Size (384,1,1)).
   - Untagged kernels: blocks per warpgroup = the full grid, so each consumer owns a whole tile. `coopA`/`coopB` kernels: grid/2.
   - "max register used = (blocks/WG × wgmmaN/2) + 26" is exact on 12 of 12 kernels across both families.
   - §H23d: "NO SUFFIX -> PING-PONG : each consumer owns a whole tile".
3. **The dispatch rule in our 163-shape sweep:**
   - Ping-pong for every tile ≤ 22,528 elements. The largest is 128×176 in BM×BN; cuBLAS's name says 176x128 because its token is N×M.
   - Cooperative for every tile ≥ 24,576 elements.
   - 25 of 163 cuBLAS picks were ping-pong tiles.
4. **SASS could not tell the two schedules apart.**
   - §H24: all kernels have identical barrier structure (46 BAR.SYNC, no BAR.ARV).
   - §H25: an opcode census of all 18 unique kernels found only 3 opcodes that differ between kernels:
     - `WARPGROUP.WAIT` (follows the layout);
     - `STSM.16.MT88` vs `STSM.16.M88` (follows the layout);
     - `USETMAXREG.TRY_ALLOC/.DEALLOC.CTAPOOL` (in 10/18 kernels, exactly when max registers ≥ 154, in both families).
   - None of the three follows the schedule.
5. **The blind spot in my own SASS check.**
   - §H24 counted only named barriers (`BAR.*`).
   - §H25 compared which opcodes are **present**, not how many times they occur.
   - A CUTLASS-style turn barrier is an mbarrier (`SYNCS.*`), and `SYNCS.*` is already present in every kernel for the TMA pipeline.
   - So my check could not have seen an mbarrier turn barrier. "Not found" there does not mean "not there".
   - This cannot be re-checked on my laptop: 0 of the 1,597 sm_90 cubins in my local libcublasLt 13.1 contain an `nvjet` name.
6. **Why I did not build ping-pong.** It was not lack of knowledge:
   - It only matters for matching performance, not for correctness (§H17).
   - It covers only 25/163 cuBLAS picks, all small tiles (§H27e).
   - It was priority #4, behind a uniform 10-14% gap that was worth more (recommended-order plan).
   - It does not explain the V5 gap: R3, "needs 288 accumulator registers vs 232 available". 288 = 192×192/128, the accumulator count for one warpgroup holding a whole 192×192 tile.
   - It is the feature to build if BM=64 is ever wanted (§H158).

---

## 6. A register-budget point about Nova's tile

**Nova's per-consumer accumulator:**
- Each consumer warpgroup (128 threads) holds a whole **128×192** fp32 tile.
- That is 128·192/128 = **192 accumulator registers per thread**.
- With the 232-register cap, everything else gets 40 registers.

**Compared with cuBLAS:**
- cuBLAS's largest ping-pong tile in our sweep is 22,528 elements.
- At that tile cuBLAS used max 202 registers per thread (accumulators + 26).
- Nova's 24,576-element tile is **above every ping-pong tile cuBLAS picked** in that sweep.
- With the same +26 overhead it would need ~218 registers, so it fits under 232 only if Nova's overhead is small.

**What I need:** the ptxas register and spill report (Q-C1). It is the quickest proof that this tile
is healthy.

---

## 7. Two statements in your earlier reply that I believe are wrong

Please confirm or refute each one with code or SASS.

**F1. "A cooperative kernel has neither the register split nor the consumer-to-consumer barriers."**
- The first half is wrong.
- CUTLASS cooperative uses the same 40/232 split (Section 2.4: lines 150-151, 561, 771).
- The PyTorch blog says cooperative is "largely equivalent to Ping-Pong, but both consumer groups work on the same C output tile".
- In cuBLAS, `USETMAXREG` appears in cooperative (`coopA`/`coopB`) kernels too (§H25).
- The second half is right for CUTLASS: cooperative has no `MathWarpGroupOrderBarrier`.

**F2. "Look for named BAR.SYNC / BAR.ARV with 256 threads … in any of our ping-pong kernels (or a CUTLASS ping-pong kernel)."**
- This is wrong for CUTLASS.
- CUTLASS's turn barrier is `OrderedSequenceBarrier` → `ClusterBarrier`, an mbarrier (Section 2.3). In SASS that is `SYNCS.*`, not `BAR.ARV`. Your own table row said "OrderedSequenceBarrier (mbarrier-based)".
- cuBLAS's ping-pong kernels also show no `BAR.ARV` (§H24).
- So BAR.ARV with 256 threads can only check **Nova's own** named-barrier design. It is not a general ping-pong signature.

**Minor:** the SASS spelling is `USETMAXREG.TRY_ALLOC.CTAPOOL` / `USETMAXREG.DEALLOC.CTAPOOL`, not "USETMAXREG.TRYALLOC".

---

## 8. Questions (please answer every ID)

### A. Where Nova's ping-pong came from

- **A1.** Which exact files did Harshini / Nova use from "BluBLAS in remote BluTrain"? Give the paths and the BluBridge-BLAS commit (I expect `ea666ab`).
- **A2.** Was `src/level3/batched_gemm/BF16/SM90/Bgemm_sm90_pingpong_template.cuh` one of them? If yes, which parts were reused (ideas, structure or code)? Was it known that this file is never compiled and that its fixed version was never run?
- **A3.** What other references were used (CUTLASS commit/version, FA3, PyTorch blog, ThunderKittens, DeepGEMM)? Which one did the turn-barrier design follow?
- **A4.** At runtime, are all GEMMs in the Nova-compiled model generated by Nova, or are any still sent to BluBLAS or cuBLAS? Which GEMMs use the ping-pong schedule?

### B. Code proof

Paste the code verbatim with `path:line @ commit`; use `git show <commit>:<path> | sed -n 'a,bp'`.

- **B1.** Do commits `c54a485` and `ac2ab83` exist, and on which branch? Give `git show --stat c54a485`.
- **B2.** `NovaGPUMapForallToGPU.cpp` lines 418-459, verbatim: the tile loop (w, w+2, …), the `barrier.sync` / `barrier.arrive` emission, and the hand-off point.
- **B3.** How tiles map to CTAs and then to the two consumers. Is it a static persistent loop, and what is the formula? What is the grid size rule (number of CTAs)? Is there raster or swizzle ordering?
- **B4.** The named-barrier IDs for the turn, and why the count is 256. List every other named barrier in the kernel with its ID, to show there is no clash (ID 0 is `__syncthreads`).
- **B5.** The first turn: how consumer 0 gets the first turn.
  The last turn: when a CTA has an odd number of tiles, or fewer than 2, one consumer has one tile
  fewer. Show that the arrive/sync count still matches and the kernel cannot hang. Kota's version
  hung at K ≤ 384 with this kind of turn passing.
- **B6.** The early hand-off: the exact number of K-steps before the end, and how it is computed.
  Can consumer 1 issue `wgmma` while consumer 0 still has `wgmma` in flight? Show the
  `wgmma.wait_group N` values in the mainloop (your reply mentioned "wait_group 1").
- **B7.** The producer and the shared ring:
  - ring depth (stages) and bytes per stage;
  - the BK value;
  - total smem;
  - whether the producer strictly alternates consumer 0's tile, then consumer 1's;
  - how each consumer's stage/phase counter skips the other consumer's k-iterations (CUTLASS does `mainloop_pipe_consumer_state.advance(k_tile_count)` at startup).
- **B8.** The epilogue:
  - size of each consumer's smem slots;
  - how slot reuse is protected (`cp.async.bulk.wait_group.read` before overwrite?);
  - the output dtype;
  - any fused bias or activation.
- **B9.** setmaxnreg: show the PTX lines. Does ptxas honour them? Show the `ptxas -v` output, and any "setmaxnreg ignored" warning. Kota's other SM90 file recorded ptxas ignoring `setmaxnreg(40,232)` for that kernel.
- **B10.** Does Nova also have a cooperative schedule for the same tile? What rule chooses cooperative or ping-pong per GEMM (shape thresholds)?

### C. Compiled-kernel proof

Use one real compiled ping-pong kernel and the matching cooperative kernel: same tile, same shape.

- **C1.** `ptxas -v` for both: registers per thread, spill stores and loads, smem bytes.
- **C2.** PTX counts or lines for both:
  - `.maxntid`;
  - `setmaxnreg.*`;
  - `barrier.sync` / `bar.sync`, `barrier.arrive` / `bar.arrive` (with IDs and thread counts);
  - `wgmma.mma_async` shape;
  - `wgmma.wait_group` values;
  - `mbarrier.*`.
- **C3.** SASS opcode counts for both (`cuobjdump -sass` on the cubin), side by side:
  - `HGMMA.*` (with shape);
  - `BAR.SYNC`, `BAR.ARV`;
  - `USETMAXREG.*`;
  - every `SYNCS.*` variant;
  - `WARPGROUP.ARRIVE`, `WARPGROUP.WAIT`;
  - `UTMALDG`, `UTMASTG`.

  Also say which instructions differ between the two kernels, and give the exact command used.
- **C4.** The register budget from Section 6: confirm 192 accumulator registers per consumer thread and 0 spills.

### D. Performance proof

- **D1.** v1 "17-20% slower than cooperative":
  - which shapes (M, N, K, layout);
  - which GPU (H100 or H200);
  - clocks free or locked;
  - how many repetitions;
  - timing method (CUDA events / nsys / ncu);
  - same tile for both?
- **D2.** v2 "−3.4% on the 26-layer step":
  - model size (layers, hidden, sequence, batch);
  - metric (step ms or tokens/s);
  - which step window, excluding warm-up;
  - how many steps averaged and how many runs;
  - GPU and clocks;
  - and above all: **what the baseline was** (Nova cooperative? cuBLAS? BluBLAS?).
- **D3.** Per-GEMM kernel times (nsys) for each GEMM shape of that model: Nova ping-pong v2 vs Nova cooperative vs cuBLAS.
- **D4.** If available, ncu tensor-pipe active % (`sm__pipe_tensor_op_hmma_cycles_active` or equivalent) on one shape, for ping-pong vs cooperative. This directly shows whether the claimed epilogue/MMA overlap happens.

### E. Correctness proof

- **E1.** The test list with max error vs a reference. It should include:
  - small K (K ≤ 384);
  - an odd number of tiles per CTA;
  - total tiles < 2 × number of CTAs;
  - M or N not a multiple of the tile.
- **E2.** `compute-sanitizer --tool racecheck` and `--tool synccheck` results on the ping-pong kernel.
- **E3.** Was the kernel run with a timeout on shapes where one consumer gets 0 tiles, as a hang check?

### F. Your earlier reply

- **F1, F2.** Confirm or refute Section 7.
- **F3.** Which statements in your earlier reply came from reading files or running commands in that session, and which came from memory? You wrote that you "didn't record a citation".

### G. Anything I got wrong

- **G1.** If any statement in Sections 1-6 is wrong, quote it and show the proof.

---

## 9. Answer format I need

For every ID (A1 … G1):

```
ID: <A1>
Answer: <one or two plain sentences>
Evidence: <verbatim code / command + output, with path:line @ commit>
Status: VERIFIED-NOW (read or ran in this session) | FROM-EARLIER (not re-checked) | NOT-AVAILABLE (say why)
```

Please do not paraphrase code: paste it. Do not estimate numbers: give the measured value and the
command that produced it, or say NOT-AVAILABLE.

---

## Appendix A: your earlier reply (verbatim, as pasted to me)

```
What "ping-pong" means, from primary sources
NVIDIA CUTLASS source, include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp, selected by the schedule tag KernelTmaWarpSpecializedPingpong:
Structure: NumMmaWarpGroups = 2, NumLoadWarpGroups = 1, "Pingpong kernel must have 384 threads in total".
Register split: LoadRegisterRequirement = 40, MmaRegisterRequirement = 232 (24/240 under heavy register pressure), applied with warpgroup_reg_dealloc / warpgroup_reg_alloc, which compile to setmaxnreg.
Ordering: OrderedSequenceBarrier with two stages: "// Order two Math WG's MMA one after the other, helps hide Epilogue", then "// Order two Math WG's Epilogue one after the other".
Tile assignment: the 2nd math warpgroup starts one tile ahead (advance_to_next_work()), and each warpgroup then steps advance_to_next_work(NumMmaWarpGroups).
NVIDIA CUTLASS docs (efficient_gemm.md, "Warp-Specialized Persistent Ping-Pong kernel design"): "The two consumer warp groups are assigned a different output tile… This allows for epilogue of one consumer warp group to be overlapped with the math operations of the other consumer warp group." The Cooperative design is the contrast: both consumers work on the same tile, split along M.
PyTorch's "Deep Dive on CUTLASS Ping-Pong GEMM Kernel" says the same independently: producers keep 40 registers and consumers get 232, which it shows in CUTLASS source and SASS; "each consumer works on separate C output tiles".
FlashAttention-3 (Shah, …, Vijay Thakkar, Pradeep Ramani (NVIDIA CUTLASS), Tri Dao), §3.1 "Pingpong scheduling": bar.sync barriers force one warpgroup's GEMMs before the other's, so one does softmax while the other does GEMMs.
cuBLAS is closed source, so the "pingpong" in its kernel names can't be read beyond its SASS. CUTLASS is the only public, inspectable definition.
Nova against CUTLASS
Checked on your current local tree (ac2ab83, which includes your later c54a485 "persistence, stream-k and ping-pong"), from a compiled ping-pong GEMM's PTX and from NovaGPUMapForallToGPU.cpp:
Defining property
CUTLASS ping-pong
Nova
Same?
Warpgroups / threads
1 producer + 2 consumers, 384
.maxntid 384
✓
Register split
dealloc 40 / alloc 232
setmaxnreg.dec 40, setmaxnreg.inc 232
✓ identical
Output tile per consumer
each consumer its own tile
each warpgroup owns a whole 128×192 tile (m64n192 ×2 m-blocks)
✓
Tile assignment
consumer 1 starts one tile ahead, both step 2
warpgroup w runs the CTA's tiles w, w+2, … (lines 418–448)
✓
Operand feed
one TMA producer fills both consumers' buffers in order, shared ring
one producer, one shared ring, TMA loads (cp.async.bulk.tensor…global)
✓
Epilogue
TMA store
TMA store (cp.async.bulk.tensor…global.shared::cta)
✓
Mainloop ordering
OrderedSequenceBarrier (mbarrier-based)
named barriers: barrier.sync self, 256 / barrier.arrive other, 256 (lines 450, 459)
same effect, FA3-style primitive
When the turn passes
after the warpgroup has issued all its MMAs (arrive after mma(), before mma_tail)
slots K-steps before the end of its K loop (line 453)
different: an earlier hand-off
Epilogue ordering
epilogues also alternate (stage 2)
not ordered: each consumer warp has its own epilogue buffer slots
different
The two differences, and why:
The early hand-off is the v1 → v2 change. With CUTLASS-style strict ordering, our first version was 17–20% slower than cooperative. Handing on early, together with wait_group 1, is what turned it into a win (v2: −3.4% on the 26-layer step).
The unordered epilogue is safe because Nova's consumers don't share epilogue shared memory.
How to see it in SASS yourself: in any of our ping-pong kernels (or a CUTLASS ping-pong kernel), look for:
the register split: USETMAXREG dealloc 40 / alloc 232 (USETMAXREG.TRYALLOC);
384 threads per block;
named BAR.SYNC / BAR.ARV with 256 threads between the two consumer warpgroups.
A cooperative kernel has neither the register split nor the consumer-to-consumer barriers.
So yes: the structure Nova implements under the name ping-pong is CUTLASS's KernelTmaWarpSpecializedPingpong: one producer and two consumer warpgroups, 40/232 registers, a separate output tile per consumer alternating over the CTA's tiles, and an ordered barrier so one consumer's epilogue overlaps the other's MMA. It differs in two deliberate details, the earlier turn hand-off and the unordered epilogue, and uses FlashAttention-3's named-barrier primitive instead of CUTLASS's mbarrier-based one.
Sources:
CUTLASS sm90_gemm_tma_warpspecialized_pingpong.hpp
CUTLASS docs, efficient_gemm.md (Hopper Warp Specialization)
PyTorch blog: Deep Dive on CUTLASS Ping-Pong GEMM Kernel
FlashAttention-3, arXiv 2407.08608, §3.1
```

---

## Appendix B: my questions, in order (typos fixed, meaning unchanged)

**B-1 (first question).**
When I worked on the SM90 matmul module (level 3) in BluBLAS:
- Kota's work and mine were merged into a hybrid BluBLAS library, and it was not pushed anywhere, right?
- While researching, we looked for the ping-pong scheduling optimization and could not find anything
  proving it, right? We documented that what we built is not ping-pong, and we had proofs for that,
  right?
- In Kota's module he did some other optimization and named it ping-pong schedule, and we got confused, right?
- The BluBLAS module on remote GitHub is not our hybrid BluBLAS, right? We have not pushed any of our matmul work to the remote, right?

Harshini (compiler team) mentioned a ping-pong optimization and said she took the reference from
our BluBLAS in remote BluTrain. I told her it may be Kota's, not mine and not my hybrid:
- Even our hybrid BluBLAS has no ping-pong, right?
- Is what Kota documented really ping-pong?

I asked the Claude on her system, and it gave the reply in Appendix A. Is what it said correct?
Compare it with our old research, and explain each point in detail.

**B-2 (follow-up).**
We had proofs too: we even compared the SASS of cuBLAS and could not find exactly which kernel is
ping-pong. Is the BluBLAS on the remote Kota's work or our hybrid? Check that. The colleague showed
a screenshot: BluBLAS master at `ea666ab`, "Merge pull request #3 from BlubridgeAI/production_blas
— optimized addmm kernel", Kathir, 31 August 2026 14:35, 4 files changed, +247 / −22.

**B-3 (follow-up).**
So we know what the ping-pong schedule is, and we have the exact reference from the CUTLASS docs.
But when we cross-checked cuBLAS SASS, we could not find that order of instructions as CUTLASS
documents it, so we wrote that we did not know, and we did not implement it. Is that right?
- Is what Harshini implemented different from what ping-pong is?
- Is what Kota implemented different from what ping-pong is, or is it the actual thing?

I want solid proof for each statement.

**What my own checking concluded (my answers to B-1 to B-3, all with proof in this document):**

1. Remote BluBLAS = Kota's (`ea666ab`, the same commit as Harshini's screenshot). My hybrid is not on the remote. (Section 3)
2. Neither my SM90 module nor my hybrid has ping-pong; both are cooperative. (Section 4.1)
3. We **did** have the definition: §H24 quotes CUTLASS.
   - What we could not do was see ping-pong in cuBLAS SASS. My check was blind to mbarrier-based ordering.
   - We **did** conclude that cuBLAS uses ping-pong for small tiles, from register arithmetic.
   - We did not skip ping-pong for lack of knowledge. We skipped it because of priority and coverage (25/163). (Section 5)
4. Kota's SM90 file **is** real ping-pong: all three parts match CUTLASS line for line. But it hung once, its fix was never re-run, and it is never compiled. Kota's B300 "ping-pong" is TMEM double buffering, a different thing with the same name. (Section 4.2)
5. Harshini's design, **as described**, is ping-pong by definition, with three implementation differences. It is not proven until the Section 8 items come back. (Section 4.3)
6. Your earlier reply had two real errors (F1, F2) and one spelling slip. Everything else in it about CUTLASS, the docs, the PyTorch blog and FA3 checks out verbatim. (Sections 2 and 7)

---

## Appendix C: my research notes, verbatim (from my BluBLAS SM90 analysis log)

**§H8.7 (2026-07-24, first look; later corrected by §H23d):**
```
- coopA: only 128x256, 256x128 tiles
- coopB: only 128x192, 192x128, 192x192, 320x128 tiles
- plain (no schedule suffix): only smaller tiles — 64x64, 112x128, 128x64, 128x128,
  144x128, 168x128, 176x128
Zero overlap between groups — cooperative scheduling is reserved for the larger tile shapes in
this dataset; smaller tiles use the plain baseline schedule.
```

**§H17 (our state):**
```
OUR STATE (corrected): our kernel is cooperative-ONLY and EVEN-tile-only (WG_M=BM/2, ATOMS_PER_WG=
WG_M/64 must be integer -> BM%128==0). So we CANNOT instantiate 192/320 tiles (coopB) NOR the Pingpong
schedule (M=64 / alternate). These are UNDONE structural variants (perf-matching, not correctness):
our even-tile cooperative kernel is CORRECT for all shapes; coopB/pingpong give the dispatcher more
tile/schedule options to match cuBLAS per-shape. "coopA/coopB done" was wrong — both coopB(odd-tiles)
and pingpong are undone.
```

**§H23d (complete model, SASS + ncu):**
```
CORRECTION 1: Block Size is (384,1,1) for EVERY kernel -- coop AND non-coop. So all of them
run 3 warpgroups = 1 producer + 2 consumers. My H23c inference that non-coop meant a SINGLE
consumer warpgroup was WRONG.

CORRECTION 2: that makes the no-suffix family PING-PONG, not "no cooperation". It matches the
documented distinction exactly (PyTorch/CUTLASS): ping-pong = each consumer owns its OWN C
tile; cooperative = both consumers share ONE C tile. So blocks/WG = the FULL grid for
ping-pong, and grid/2 for cooperative -- which is precisely what the registers show.

VERIFIED ON EVERY KERNEL (acc-base count = blocks/WG, spacing = CREGS = wgmmaN/2):
  family      token     N     M   Mat  Nch  grid  blocks/WG  spacing  wgmmaN/2
  pingpong    176x128  176  128    2    1     2       2         88       88
  pingpong    144x128  144  128    2    1     2       2         72       72
  pingpong    112x128  112  128    2    1     2       2         56       56
  pingpong    128x64   128   64    1    2     2       2         32       32
  pingpong     64x64    64   64    1    1     1       1          -       32
  coop        256x128  256  128    2    2     4       2         64       64
  coop        192x192  192  192    3    2     6       3         48       48
  coop        320x128  320  128    2    5    10       5         32       32
8/8 on the register spacing, and the blocks/WG rule holds for BOTH families. No outliers.

THE COMPLETE DECODED MODEL:
  * every kernel: 384 threads = 1 producer warpgroup + 2 consumer warpgroups
  * tile token = N x M  (NOT M x N)
  * wgmma = m64 x wgmmaN x k16;  CREGS = wgmmaN/2 accumulator regs per thread per block
  * work grid = (M/64) M-atoms  x  (N/wgmmaN) N-chunks
  * NO SUFFIX      -> PING-PONG   : each consumer owns a whole tile, blocks/WG = grid
  * coopA / coopB  -> COOPERATIVE : consumers share one tile,       blocks/WG = grid/2
  * coopA <=> blocks/WG EVEN (2,2);  coopB <=> blocks/WG ODD (3,5,3,3)
  * split axis when cooperative: M when M-atoms even, N when odd (forced by factorisation)

IN OUR NAMING (BM x BN) the cuBLAS catalog is:
  cooperative: 128x256 (have), 256x128 (have), 128x320, 128x192, 192x192, 192x128
  ping-pong  : 128x176, 128x144, 128x112, 64x128, 64x64
```

**§H24 (ping-pong vs cooperative, cross-verified against official docs and our own data):**
```
OFFICIAL SOURCE (CUTLASS docs, docs.nvidia.com/cutlass efficient_gemm.html), verbatim:
  Cooperative: "Presence of two consumer warp groups cooperating on the same output tile by
               splitting the tile in half across the M dimension." ... "This allows for
               larger tile sizes to be enabled - since the register pressure per consumer
               warp group is reduced."
  Ping-Pong  : "The two consumer warp groups are assigned a different output tile using the
               Tile Scheduler." ... "This allows for epilogue of one consumer warp group to
               be overlapped with the math operations of the other consumer warp group."
The PyTorch deep-dive adds: 1 producer + 2 consumer warpgroups; producer does TMA. It does
NOT state tile-M constraints or which shapes suit which -- flagged as not covered.
NOTE the CUTLASS GitHub issue #2181 asking exactly "why both schedules?" is UNANSWERED (no
maintainer reply), so there is no vendor guidance beyond the doc text above.

OUR MEASUREMENT CONFIRMS ALL OF IT, and quantifies what the docs only assert:
 * ncu: Block Size = (384,1,1) on EVERY kernel = 1 producer + 2 consumer warpgroups. Matches
   the documented structure exactly.
 * THE KEY RELATION, exact on 12/12 kernels:  max register used = (blocks/WG * wgmmaN/2) + 26
       pingpong    64x64 32->57   128x64 64->90   112x128 112->138  144x128 144->170
                   168x128 168->194   176x128 176->202
       cooperative 128x192 96->122  192x128 96->122  256x128 128->154  128x256 128->154
                   192x192 144->170  320x128 160->186

QUANTIFIED BENEFIT/LOSS (the docs assert this; these are OUR numbers):
                       biggest tile   area     acc regs/thread   total regs
      ping-pong        176x128       22528          176             202
      cooperative      320x128       40960          160             186
   Cooperative buys 1.8x the tile area for FEWER registers, because two consumers share one
   tile. Ping-pong tops out at 176x128 because a single consumer holding a WHOLE tile hits
   the register wall (202 of the 255 hardware max).
   Tile areas are CLEANLY SEPARATED with zero overlap: cuBLAS uses cooperative for every
   tile >= 24576 and ping-pong for every tile <= 22528. That is the dispatch rule, measured.

 2. I predicted ping-pong would show extra named barriers for its MMA/epilogue alternation.
    It does NOT -- all kernels have identical barrier structure (46 BAR.SYNC, same BAR.LE /
    BAR.SYNC ops, no BAR.ARV). So the alternation is NOT observable in the barrier mix. That
    test was inconclusive, neither confirming nor refuting; the register/block evidence is
    what carries the identification.

DIVERGENCE FROM CUTLASS worth remembering: CUTLASS cooperative splits the tile "across the M
dimension" ALWAYS, which forces TileShape_M % 128 == 0 and is why CUTLASS has no 192/320
tiles. cuBLAS splits N instead when M/64 is odd (192x192, 128x192 -> 3 M-atoms x 1 N-chunk).
```

**§H25 (exhaustive opcode census of all 18 unique kernels):**
```
  discriminator                       present in   actually correlates with
  WARPGROUP.WAIT                        10/18      LAYOUT: every _TNT has it, no _TNN does
  STSM.16.MT88.* vs STSM.16.M88.*       9 vs 8     LAYOUT: MT (transposed epilogue store)
                                                   <=> TNT;  M <=> TNN
  USETMAXREG.TRY_ALLOC/.DEALLOC.CTAPOOL 10/18      REGISTER PRESSURE (see threshold below)

USETMAXREG THRESHOLD, exact across all 18, no exceptions:
    max R used  57 58 58 90 90 122 122 138  -> USETMAXREG absent
    max R used  154 154 154 170 170 170 186 186 194 202 -> USETMAXREG present
Clean break between 138 and 154, and it cuts ACROSS all three families (coopA, coopB and
pingpong each appear on both sides). So register redistribution is a pure register-pressure
response, orthogonal to the schedule.

CORRECTION TO H24: I reported "no SETMAXNREG anywhere, so cuBLAS does not redistribute
registers". WRONG -- I grepped the PTX mnemonic. In SASS it is USETMAXREG.TRY_ALLOC.CTAPOOL /
USETMAXREG.DEALLOC.CTAPOOL, and it appears in 10 of 18 kernels. cuBLAS DOES use dynamic
register redistribution, precisely when the consumer's register demand is high (>=154).
```

**§H27e (coverage):**
```
WRONG. Both are ping-pong tiles in cuBLAS (no coop tag), and 64x64 has a 1-block grid which
CANNOT be split between 2 warpgroups at all -- our picker correctly refuses it (no legal
WGMMA_N -> static_assert). Those tiles need the PING-PONG schedule, a different feature.
Corrected numbers:
    cooperative schedule (what this feature implements) : 138/163 = 85% of cuBLAS's picks
      of which STRICTLY need the N-split (odd M-atoms)  :  34/163 = 21%  (192x192 x33, 192x128 x1)
    ping-pong tiles (need the other, unbuilt feature)   :  25/163
```

**Plan, recommended order (same log):**
```
1. 320x128 -- 52 shapes, zero new features ...
2. 128x64 -- 4 shapes, zero new features ...
3. Masked BM as a knob -- ... unlocks four tiles and 13 shapes at once.
4. Ping-pong for 64x64 -- the only genuinely new structure needed.

The uniform 10-14% gap runs in parallel with all of this and is worth more than items 3 and 4
combined; it should not wait behind them.
```

**R3 (hypotheses killed while explaining the V5 gap):**
```
| R3 | ping-pong | DEAD — needs 288 accumulator registers vs 232 available |
```

**§H65 (2026-09-02, hybrid step 1: removing the 11 uncompiled files):**
```
1. pingpong_template.cuh (680) — ping-pong scheduling of the non-clustered warp-specialised
kernel. Motivation 2026-08-06: his best real-shape kernel bn256_rs
(Sm90Config_128x256x64_regsplit) already beat both his cluster kernels at every real 1.7B shape
but still trailed the reference by a uniform 0.4–9%; the reference ships TWO non-clustered SM90
kernels with the same 1-producer/2-consumer structure, "cooperative" and ping-pong.
VERDICT RECORDED: none. Motivation only — abandoned or unfinished, not measured-and-rejected.
```

**§H158 (2026-09-08):**
```
The real concept behind the name is CUTLASS's Cooperative schedule — 2 consumer warpgroups
split ONE tile's M (AtomLayoutMNK Shape<2,1,1>), requires TileShape_M ≥ 128, falls back to
Pingpong at M=64. ... We have no Pingpong, so at BM=64 we must pick between half-width and half
the warpgroups. Pingpong is the named, documented feature to build if BM=64 is ever wanted.
```
