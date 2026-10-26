---
name: project_state_2026_10_12
description: "STATE end 10-12 + resume plan for 10-13. Step 6 done, A default + B kept; Vj2 (FA3-style bwd) built, 605/605, −5..−13% vs Vj1d; fixed Vj2 built but not GPU-run; docs §41-42"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-12T16:02:56.590Z
---

End of 2026-10-12. Work doc §41 (Step 6, cuDNN varlen §41.7) and §42 (A/B decision, who calls which kernel, research, Vj2, scope, hd64 bug, SFT preprocessing paths) are written.

Records:
- `profiling_reports/v2_step6_benchmarks_2026_10_12/`
- `profiling_reports/v2_bwd_rework_research_2026_10_12/` (RESEARCH.md, DESIGN_VJ2.md, gpu_runs/, code_snapshot/)

`profiling_reports/` is in `.gitignore`, so records need `git add -f` with explicit paths. The 10-10 and 10-12 folders were not committed as of 10-12 evening; the push block was given to the user.

**Resume 10-13, in order, each GPU run only after the user's OK:**
1. Fixed Vj2 (md5 bbd977be, box library built 21:00):
   - run 1 = quick CLASSIC (~3.5 min);
   - run 2 = BENCHDENSE ABAB + nsys + ncu (~5 min), only if run 1 passes;
   - then ASK about run 3 = full CLASSIC (~23 min).
   - Use absolute paths in scripts.
2. Stage 3: port A/B to Vj2 (A's plan needs 128-key backward tiles), G8-G13, packed benchmark. Keep Vj1d's 64-key path for short chats if Vj2 loses there.
2b. ADD Meta xFormers (user asked 10-12 night) to the library benchmark:
   - its own packed mode (BlockDiagonalCausalMask), fwd + bwd, same segments, output check vs FA3;
   - run next to FA2/3/4, cuDNN, PyTorch varlen/Flex/SDPA;
   - pip install into `~/gautam_k/benv` only after the user's OK.
3. hd64 causal bug: 1-min proof test + refusal fix ([[project_bwd_rework_research_1012]] §8).
4. Step 7: `attn_pack_mode` config setting, packing script, loader, attention node, end-to-end gradient test, packed SFT run vs v1.

**Open items:**
- `sft/` is untracked in the BluTrain_sft git (the user should commit it).
- Code on BluTrain `dist_side_merge` is not pushed (v2 A/B since 10-10, race fix, Vj2). Pushing to the team branch is the user's call.

Related: [[project_v2_build_progress]], [[project_cudnn_varlen_reachable]].
