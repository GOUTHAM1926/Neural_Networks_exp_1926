---
name: project_bwd_rework_research_1012
description: "Backward FA3-structure rework — RESEARCH ONLY (no code), decision pending; gap = idle tensor time, 2x L2 reloads, precision gain modest (emulated)"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-12T13:45:04.861Z
---

On 2026-10-12 the user said: no code before proofs and deep research, then we decide. Record: `profiling_reports/v2_bwd_rework_research_2026_10_12/RESEARCH.md`, which also holds the emulation scripts.

**Proven** (10-12 ncu, dense B8 18/6 T4096):
- Ours 3.626 ms vs FA3 2.494 ms.
- Tensor busy TIME is equal (1.66 vs 1.74 ms), so the whole gap is idle tensor time.
- L2 reads are 1.98× FA3's, which matches 64-key vs 128-key blocks (299,520 vs 152,064 q-tile visits).
- cuDNN has the same dQ reduce count as ours and is still 23% faster, so the waiting around the reduce, not its volume, is the issue.

**Stalls in ours:**
- ~14% at `__syncthreads` + fences before the dQ reduce (`:609-611, 621-627`);
- 9.8% waiting on GEMMs;
- ~7% P/dS shared-memory round trip;
- ~4% LSE/D loads in `issue()`;
- ~14% softmax math.

**Precision (CPU emulation, fp64 truth checked by finite differences):** the FA3 arithmetic cuts RMS error 17–30% (normal inputs) and 4–23% (large inputs). The rest comes from bf16 P/dS in the GEMMs, which every library has. So it is modest.

**Estimate, not proven:**
- Our v62 forward already has the FA3 structure yet is 1.22× FA3, so a reworked backward lands between −16% and −31% on the kernel.
- Proposed gate: dense prototype first, then B/A modes only if it clearly wins.

**Why:** the user rejected coding on assumption ("dont just assume, proofs first, then decide").

**How to apply:** don't start the kernel until the user picks option 1/2/3 in RESEARCH.md §6. Related: [[project_cudnn_varlen_reachable]], [[feedback_verify_claims_and_benchmark]].

**Scope, checked later on 10-12:**
- The fast path does not depend on the head ratio: any G, 5 ratios tested, 13 shapes all 1.32-1.42× vs FA3.
- It needs head dim 128 + T%64 + Hopper.
- A rebuild must keep T%64 and keep the 64-key path for short chats (FA3 loses there).

**FOUND:** causal head dim 64 on H100/H200 is silently NON-causal, forward (`gqa_v2`) and backward (`launch_gqa_backward`).
- Reachable via `bluscript.cpp:73` (default `head_dim = 64`); the guard at `:588` and `Config.h:270` wrongly assume a causal fallback.
- Production yaml uses 128, so production is OK today.
- Not GPU-proven yet. See RESEARCH.md §8.

**BUILT + TESTED 10-12 evening:**
- Vj2 = `GQA_bwd_vj2_sm90.cu` (laptop + box `~/gautam_k/v2_work`), opt-in `USE_VJ2=1`, dense only, S%128==0 (else Vj1d).
- Correctness: CLASSIC 605/605; dK/dV bitwise; the known G6 D10 dQ case is gone; dQ/dK worst-ratio median 0.91× Vj1d.
- Speed: −5.2..−13.0% vs Vj1d (mean −10.6%); 1.20-1.28× FA3 (was 1.32-1.43×); 0.95-1.11× cuDNN.
- Gate ≥10% met on 9/13 shapes (misses the small ones).
- Next: nsys/ncu to find the remaining gap BEFORE porting the packed modes. Records: `DESIGN_VJ2.md`, `gpu_runs/`.

**STOPPED 10-12 ~21:05 by the user ("continue tomorrow").** Nothing was running; GPU 7 idle.

State:
- The fixed Vj2 (shared address space + `desc_add`, file md5 bbd977be) is built into the box library (21:00, 0 spills). It is NOT yet run on the GPU.
- First thing tomorrow, after the user's OK:
  - run 1 = quick CLASSIC smoke (~3.5 min);
  - run 2 = BENCHDENSE Vj1d/Vj2 ABAB + nsys + ncu (~5 min), only if run 1 passes;
  - then ASK about run 3 (full CLASSIC, ~23 min). The user said: ask for 3.
- Script draft: logs dir `vj2c_1012`, absolute paths (the earlier relative-path bug skipped the library run).
- Pre-fix numbers: kernel 3197 µs vs Vj1d 3615 vs FA3 2495; instructions 1,096M vs FA3 625M.

Also pending:
- the causal hd64 non-causal bug (1-min proof test + refusal fix);
- `sft/` untracked in git (user to commit);
- Stage 3 (packed modes) after the speed is settled.
