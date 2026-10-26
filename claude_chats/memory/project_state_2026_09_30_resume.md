---
name: project-state-2026-09-30-resume
description: STATE at end of 2026-09-30 + what to do first on 2026-10-01 (TOMORROW list = loss doc §26); F4 pushed on BLUCCL; Navin's a3aba3e bugs; box released
metadata:
  type: project
---
**Done 2026-09-30:** F4 fixed + pushed on BluTrain `dist_side_merge_BLUCCL` (`149249d`, TI pointer `3298e74` -> b975be5), see [[project-f4-fix-plan]]. Whole session documented: `work_docs*/embedding_optimizer_zero_1_parallelism.md` (master `~/Downloads/sft_research/`), loss doc §26 TOMORROW list + §31, SFT doc §36. The H200 box (round-earth-echoes-fin-03, 86.38.182.174) was cleared to release; everything copied to `profiling_reports/f4_2026_09_30/`.

**REMIND FIRST THING ON 2026-10-01 MORNING (user: "remind me tmrw mrng, we will do these"), in this order** — same list
at the top of the loss doc §26 ("TOMORROW MORNING"):
1. share the findings with the team: send Navin the message (new doc §13): test_muon_rsag random at 1 gpu since a3aba3e
   (fails 6/6, e6eaeba passes 6/6, same before/after F4), TI pointer fixed by 3298e74, stale test scripts, 2 symlinks
2. confirm the rope_theta of Navin's checkpoint (his trainer 10000, ours 500000) BEFORE the sft task 1 test
3. measure the real size of the d1, d2, d3 roundings on real data (the re-scoped ablation)
4. decide the embedding setup for sft: fp32 embedding, or port Kruthik's sort-based kernel (EmbeddingBwdFused; its hook
   ignores N — fix first) to dist_side_merge
5. run the sft task 1 test on h200 (weights-only load, BLU_INIT_FROM, SFT doc §35; expect 262/262 CRC, step-0 val ~1.87)
6. continue with the next sft tasks (2-7)
Then (not urgent): Navin's muon-bug lead, optional F4 trainer A/B (new box), local fetch of BLUCCL, F4 on dist_side_merge,
multi-gpu send-dtype research, rotate the token. The §26 reorder (loss doc, master + work_docs) is NOT committed yet: put
it in tomorrow's first docs commit.

**Still-valid state from 09-29:** SFT continues on `dist_side_merge` in `~/Downloads/BluTrain_sft` (task 1 `BLU_INIT_FROM` patch applied locally, untested, uncommitted; `.orig_init` untracked, never add). SFT runs `BLU_ALL_BF16=1`. No pre-flight dtype check (comment only), see [[feedback-no-redundant-boundary-fixes]].
