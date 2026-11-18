---
name: project-bwd-share-study-1114
description: Backward (Vj2) sharing study started 2026-11-14 (CPU only so far): how the backward cuts a chat into blocks, the count of what mixed one-step blocks would save, what the kernel change needs; NOT implemented, user decides
metadata:
  type: project
---

The user, 2026-11-14 ~18:57 IST: "in mean time, let's do all the CPU work regarding that backward kernel work" (the
"same sharing idea for the backward" from the future list of work log 45.195/45.196). GPU tests of the day ran alongside.

**How the backward works (read from the code, TI f1a7d98):**
- `GQA_bwd_vj2_sm90.cu`: one block per (128-key tile kt of a chat, query head); 3 warpgroups: loader + 2 compute groups;
  group c owns keys 128kt+64c .. +63; the block runs nIter = ceil(L/64) - 2kt steps of 64 queries.
- Block list built in `gqa_chat_plan_build` (GQA_fwd_generic.cu:639-656): entries {b, t0, L, kt, hq, dqt0, 0, 0}
  (two unused ints per block).
- A chat with an ODD number of 64-token tiles has a last block of ONE step in which group 1 has no key of the chat; the
  loop has no skip (line 594), only the stores are conditional (735-801): that group's math is pure waste.
- In the forward the half-empty unit was the chat's most expensive one; in the backward it is the cheapest block.

**Sharing = mixed one-step blocks:** the last blocks of two query heads of one KV head become one block (group 0 = head
a, group 1 = head b, same 64 keys). Removes Hkv * floor(G/2) one-step blocks per odd chat.

**Count + estimate** (`profiling_reports/attn_mx_followup_2026_11_14/bwd_share_study/pred_bwd.py`, outputs
`pred_bwd_4x2.txt`, `pred_bwd_18x6.txt`): cost of a one-step block fitted from the sweeps' rows of <=64-token chats:
64.5 ns of call time at 4/2 (today), 63.5 ns at 18/6 (13/11): the two agree. Estimated backward change (upper value =
a mixed block costs what a one-step block costs; lower = 20% more):
- 4/2: every chat 16: -43..-34%; 64: -30..-24%; 160/192: -11..-9%; 320: -5.7..-4.5%; 704: -2.5..-2.0%;
  chats 32-160: -10.3..-8.2%; chats 300-1,500: -1.2..-0.9%; real rows: -1.1 / -0.9 / -0.3%; even lengths: 0.
- 18/6: every chat 16: -30..-24%; 64: -22..-18%; 160/192: -8.0..-6.4%; 320: -4.1..-3.3%; real rows: -0.7 / -0.6 / -0.2%.
- Absolute, 4/2, chats 32-160: backward ~0.020-0.025 ms against the forward's measured 0.0076 ms.
These are estimates from a fitted cost, not measurements.

**What the change needs (not started):** plan builder: mixed blocks in the Vj2 list; kernel: a NEW instantiation for
mixed one-step blocks (Q/dO/LSE/D of two heads: the second smem stage is free because the block has one step; both
groups on the same 64 keys; each group stores dK/dV partials for its own head; the dQ part is new code: today the two
groups split one dQ tile by head columns over 128 keys, a mixed block needs each group to make a full tile from 64
keys); existing kernels byte-identical; switch + rule; suite at all head settings, race, trainer A/B.

**State:** count and outline given to the user on 14/11 evening; the kernel work waits for the user's decision
(explain first, then OK: [[feedback_teach_me_do_it_myself]]). Related: [[project_attn_short_chats_mx_1113]],
[[project_tomorrow_1114_mx_untested]].

**MEASURED 2026-11-14 19:33-19:41 (nsys, `pkg14/bwdprof14.sh` + `.py`, tables in
`box_results/pkg14_from_box/results_bwdprof/*.table`, full table in work log 45.197):** helper kernels (prep, 2 head
sums, dQ conversion) are 43-50% of a backward call on short chats, 22-31% on real rows, 12-15% on one long chat.
Examples (whole | prep | main | head sums | conversion, µs): 18/6 chats 32-160: 994.5 | 196.8 | 560.6 | 104.9 | 128.1;
18/6 real rows: 1546 | 171 | 1158 | 105 | 108; 4/2 chats 32-160: 242.8 | 45.7 | 130.1 | 30.9 | 30.4; 4/2 real:
347.6 | 39.5 | 246.5 | 29.9 | 26.7. One-step block = 49-55 ns of main-kernel call time; padded tile = 9.4 ns zeroing
+ 7.3 ns conversion read.
**Ranking by these numbers (estimates):** (1) head sums inside the block: -6..-10% real rows, all layouts, largest
change; (2) direct dQ for a chat's first two tiles (single contributor: no zero / add / conversion): -17..-19% on
chats 32-160, ~-1.6% real rows; (3) sharing: -5% (18/6) / -7% (4/2) on chats 32-160, <1% real rows (my first -8..-10%
was an upper bound); (4) skip the masked group in each block's first step: small, exact, gain unknown.
**Reality check told to the user:** attention backward is ~5% of the 1.6B step, so even -9% of it is <0.5% of a step.
**State:** user said "analyze and research deeply" and "proceed with experiments on GPUs 0 and 1"; the profile is done;
no kernel code written; waiting for the user's choice of what to build first.

**BUILT + TESTED 2026-11-14 20:24-21:22 (work log 45.197 has the full table):** three builds, each a copy of bt_push
on P032 with a patch from `bwd_share_study/`: `patch_bskip.py` (exp 1: skip the masked group), `patch_bdq.py` (exp 2:
direct dQ for a chat's first two tiles), `patch_bcomb.py` (both). Box trees bt_bskip, bt_bdq, bt_bcomb (bt_bcomb2 =
link). Runner: `pkg14/bx14.sh <name> build|test|sweep <gpu>` via the gate `pkg14/startg.sh`. Quick suites all pass
(273 at 4/2, 280 at 18/6). Backward against pushed Vj2 (4 rounds unless said):
- exp 1: 4/2: -3.05 / -2.74 / -1.40 / -2.02 / -2.23 / -1.59 (chats 32-160, 300-1,500, one long chat, real x3);
  18/6: -3.40% (4/4) | NEWoff: 0 round(s rounds);-3.40% (4/4) | NEWoff: 0 round(s rounds);-1.69% (4/4) | NEWoff: 0 round(s rounds);-2.11% (4/4) | NEWoff: 0 round(s rounds);-2.53% (4/4) | NEWoff: 0 round(s rounds);-1.87% (4/4) | NEWoff: 0 round(s rounds); by length 16..192: -8.6 .. -2.7% (4/2), -9.1 .. -2.6% (18/6).
- exp 2: 4/2: -19.16 / -1.85 / -0.37 / -1.79 / -1.70 / -0.59; 18/6: -22.09 / -2.89 / -0.40 / -1.60 / -1.66 / -0.83.
- combined: 4/2: -22.24 / -4.97 / -1.49 / -3.37 / -4.10 / -1.89; 18/6 PARTIAL: -25.99 (2 rounds) / -4.92 (2) / -1.18 /
  -3.95 / -3.57 / -2.81 (1 round each). The two gains add up.
**PAUSED 21:22:30 (user: wrap up by 21:30); STOP file in place on the box; nothing of ours running.**
**NEXT SESSION, in this order (the user agreed: "later we will continue"):** (1) finish the combined race at 18/6
(`rm STOP`; `bash startg.sh <gpu> x.out env HEADS=18/6 ROUNDS=4 bash bx14.sh bcomb2 test <gpu>`: finished runs are
skipped); (2) the missing tests before a push: full suite 18/6 + 4/2, quick suite at 12/4, 16/4, 2/1, 5/1, 6/6, G14 at
32,768, DENSE rows suite + race (exp 1 changes the dense kernel), BLU_VJ2_DKDV_F32=1 path, sweeps for exp 2 + combined,
trainer A/B (135M, 1.6B shape, one GPU + CP); (3) idea 3 (sharing) on top of the combined build; (4) idea 4 (head sums
in block): design note first; (5) explain the change to the user, put it into the push tree, the user pushes, verify.
Open observation: forward +0.1..+0.8% in exp 1's race at 4/2 (kernel unchanged; not in its sweep). Ask which GPUs.

**2026-11-16 (P032 GPU 5, later GPUs 4 + 5; paused twice for colleagues 16:39-18:27 and 18:31-19:17):**
- Combined build (exp 1 + 2) tested further, all PASS: 18/6 race 4 rounds (-26.05 / -5.02 / -1.90 / -3.67 / -3.66 /
  -2.65%), quick suites 16/4, 2/1, 5/1, 6/6 273/273, 12/4 272 + the known check (same numbers as the old tree), G14 at
  32,768 69/69, BLU_VJ2_DKDV_F32=1 273/280 pass, DENSE B8 18/6 bwd -1.63% (4/4), sweeps 16-128 tokens -25..-32%,
  160-192 -18..-20%, 1.6B-shape trainer 30 steps: losses equal, 794.2 vs 795.4 ms. Scripts: pkg14/bfull16.sh (parts),
  trainer16.sh (BASE bt_push, NEW bt_bcomb, runs in attn_1113/runs16; parts b16, t135, cp135, cp16), dense_stats16.py.
- 19:26 chain16.sh: GPU 4 full suite 18/6, GPU 5 full suite 4/2 (then t135), then cp135 + cp16 on GPUs 4,5.
- EXPERIMENT 3 "sharing" WRITTEN + BUILT (bwd_share_study/patch_bshare.py = patch_bcomb + new kernel
  gqa_bwd_vj2_share_kernel + launcher + builder; box tree bt_bshare, lib 58dbdad8). The shared last blocks follow the
  main blocks in the same vj2 list; their COUNT is kept outside GqaChatPlan (static g_vj2_share keyed by vj2_blocks,
  gqa_vj2_share_count) because the test program holds GqaChatPlan BY VALUE (struct change = ABI break). Main kernel
  launched only if n_blocks > 0 (all-shared plans). BLU_VJ2_SHARE=0 = off. Laptop compile (nvcc 13.0, -gencode
  arch=compute_90a,code=sm_90a; plain -arch=sm_90a fails with "setmaxnreg not supported on sm_90"): share kernel 168
  regs, 0 spills; ALL 10 existing kernels' SASS identical to bcomb (0 lines).
- chain16b.sh queued (19:30): after CHAIN16 DONE, if GPUs 4 + 5 are free: bshare vs bt_bcomb (BASET=bt_bcomb in
  bx14.sh): GPU 4 suite + race + sweep at 4/2; GPU 5 suite + race at 18/6, quick suites 16/4 5/1 2/1 6/6 12/4, sweep.
- STOP by hand: touch pkg14/STOP first, then the test pid (see feedback_box_ops_traps).

**2026-11-16 21:01 IST, END OF DAY STATE (all of ours stopped; results pulled to box_results/pkg14_from_box + runs16_from_box):**
- COMBINED BUILD (exp 1 + 2, bt_bcomb) PASSED EVERYTHING planned before a push: full suites 4/2 + 18/6 333/333 each;
  quick suites all head settings (12/4 = known check only); G14; fp32 partials; dense -1.63%; races + sweeps; trainer
  30 steps, losses identical: 1.6B 1 GPU 794.2 vs 795.4 ms, 135M 1 GPU 105.06 vs 105.61 ms (-0.5%, both new < both
  pushed), CP 135M 52.37 vs 52.44, CP 1.6B 280.8 vs 281.5. NEXT: explain the change line by line to the user, then
  put patch_bcomb's changes into ~/Downloads/BluTrain_push (old code as comments), update sft docs, user pushes, verify.
- EXPERIMENT 3 (sharing, bt_bshare) = DROP: correct (quick suites pass at all head settings, G14 69/69) but against
  bt_bcomb: 4/2 race bwd -1.42% (chats 32-160) and +0.9..+2.3% SLOWER on 300-1,500 and real rows (0/4); 18/6 race
  -1.78% / +0.67% / -0.10% / +0.39 / +0.71 / +0.38%; 18/6 sweep -18.5 (16 tok) .. -9.6% (64 tok) but +1..+2% at 96,
  160, 192. Cause (my reading): the separate launch + losing the one-step blocks that filled the main kernel's tail.
  Code kept in bwd_share_study/patch_bshare.py (not for push). Push-grade tests of exp 3 were NOT run (not needed).
- Idea 4 (head sums inside the block): design note still to write.

**USER DECISION 2026-11-16 ~21:05 IST: NO push yet.** "after all testing only we will push, finalizing and analyzing
all". Test every idea alone and in combination with the full test set, then one final comparison table, pick the
best, explain line by line, then push once. Clarified to the user: today's idea 3 test was the 1+2+3 combination
against 1+2 (idea 3 ALONE was never run).
**NEXT SESSION ORDER:** (1) idea 3 alone vs bt_push (~30 min: build patch_bskip-free version = patch_bshare minus
bcomb, race 4/2 + 18/6) for the record; (2) idea 4 (head sums inside the block): design note -> explain -> OK ->
build alone (on bt_push) and on top of 1+2 -> full test set each; (3) final table of all variants (1, 2, 1+2, 1+2+3,
3, 4, 1+2+4, ...) -> pick -> explain line by line -> push tree (old code as comments, sft docs) -> user pushes ->
verify on the remote. Idea 4 would make idea 3 unnecessary (one block handles all query heads of a KV head).

**2026-11-17 ~15:00 IST: P015 GPUs 4 and 5 are ours** ("continue the pending backward experiments fast, remove the
unnecessary ones" while the SWA work is prepared).
- **Setup on P015:** `~/gautam_k/attn_1113` (/localstorage is not writable here).
  - bundle = git archive of the laptop push tree (d9606c2 / f1a7d98 / cfa0e92 / cf1372d); all 5,068 files equal
    pkg14/push_manifest.md5.
  - Also copied: pkg/ (layouts, stats scripts), stage0_labels/labels.bin (eec5e38b), and pkg14 copies with the home
    path. Laptop folder: profiling_reports/attn_bwd_p015_2026_11_17/ (build15.sh, chain17.sh, pkg14/).
- **chain17.sh started 15:03:** build15, then bx14 bcomb build, then the setup check through startg. GPU 4 runs
  HEADS=4/2 and GPU 5 runs HEADS=18/6, each ROUNDS=2: bcomb vs push, which also gives the same-box "1 + 2" line for the
  final table. Laptop waiter: scratchpad/wait17.sh.
- **REMOVED as unnecessary (told the user):**
  - idea 3 alone: the same two causes of loss remain without 1 + 2; idea 4 leaves nothing to share; its kernel uses
    idea 2's code, so testing it alone needs a rewrite;
  - every combination with 3;
  - the f32-partials suite for idea 4 (idea 4 has no partials);
  - repeats of the known 12/4 check.
- **Idea 4 explained** (waiting for the user's OK before coding):
  - one block per (key tile, KV head) loops the G query heads; dK/dV stay in the fp32 accumulators and are written
    once into d_dK / d_dV;
  - the two greduce launches, the partial buffers, and (G-1)/G of the K/V loads and dK/dV stores go away;
  - risk: G x fewer blocks -> underfill on small grids, so a rule chosen by tests;
  - the dense path too;
  - tests: quick suites at 4/2, 18/6, 2/1, 5/1, 6/6, 12/4, 16/4 and 8/4, G14, dense suite + race, real-row race +
    sweep; full suites + trainers only for the final pick.
**17/11 ~15:30 IST, IDEA 4 BUILT + TESTING** (the user's OK: "first plan and complete scheduling of pending bwd works").
- Patch: profiling_reports/attn_bwd_p015_2026_11_17/patch_bhsum.py.
  - New template flag HSUM. With it, a block of the CHATS list is one (key tile, KV head); hq = the first query
    head; w1.z = the n_dq_tiles stride; NT = G x nIter steps, with the counters continuing over the heads.
  - Final dK / dV via TMA maps on d_dV / d_dK (Rkv rows); the greduce launches and the partial allocation are
    skipped.
  - The per-KV-head list is appended after gs in the plan allocation, with a side table keyed by vj2_blocks
    (gqa_vj2_hsum_list); G == 1 uses the per-head list; dense grid (S/128) x B x Hkv, sections sw / G.
  - BLU_VJ2_HSUM=0 turns it off; f32p keeps the old path.
- Laptop check: 168 regs, 0 spills; the 10 existing kernels have 0-line SASS diffs; the HSUM kernels differ by
  1,500-1,700 lines (negative control).
- P015: bt_bhsum lib e3ebb71d. chain17c.sh then bh17.sh (gates suites before races; the known 12/4 check is
  allowed). GPU 4 runs q42 q84 ratios r42 s42; GPU 5 runs q186 g14 r186 dense s186. Waiter: scratchpad/wait17c.sh.
- RESULTS so far: all PASS. 4/2 273/273, 8/4 273/273, 18/6 280/280, G14 32k 69/69, 16/4 273/273, and 12/4 273/273:
  the known G11 D5 12/4 dV check (worst ratio 1.015 on the pushed tree) now PASSES (one rounding instead of two).
- NEXT: the races. If idea 4 wins, write 1+2+4 (the peeled step of idea 1 and the direct steps of idea 2 must happen
  at EVERY head's first steps inside the HSUM loop), then a small-grid rule if needed, then the final-pick full
  suites + trainer A/B (the trainer + data are not on P015 yet).
**17/11 16:00-16:25:**
- Idea 4 sweeps: bwd -21..-34% at 16-192-token chats (4/2 and 18/6, 4/4 rounds), better than 1+2 at 160-192.
- Forward-only race (VARLEN_BENCH_DIR=fwd, 18/6): -0.27..+0.14% = noise. So the +0.4..+4.2% forward in the
  alternating race came from the faster backward before each forward (power / clock), NOT from the forward kernel.
- Block-order switch patch_bhsum2.py (BLU_VJ2_HSUM_SECT): races 4/2 running (bhsum2a = -1 one section,
  bhsum2b = 4); reason: the weak 4/2 real-row sets (-1.10%, +0.37%).
- 1+2+4 = patch_b124.py: idea 1 in-loop for every head (ii == 0, c == 1); idea 2 per head (ii < nDirect); NEW
  pend flag before staging dq_stage (a direct step can follow a hand-off); the writer skips direct steps over NT.
  - Laptop: 168 regs, 0 spills, the 10 existing kernels have 0-line SASS diffs vs bcomb.
  - P015: bt_b124 lib 2855d0f3. chain17e.sh starts bh17x.sh (X=b124) on GPUs 4 + 5 after the order races.
**17/11 16:30-16:45:**
- ORDER RESULT, idea 4 alone at 4/2 (bwd vs push; default sw/G | 4 pairs | one list -1):
  - short -24.0 / -24.1 / -20.6;
  - 300-1500 -7.6 / -7.5 / -9.6;
  - one chat -6.1 / -6.7 / -3.6;
  - real -1.1 / -1.7 / -11.2, -7.3 / -6.9 / -6.0, +0.4 / -0.1 / -6.2.
  - So the tail is confirmed. One list balances real rows but loses locality on uniform layouts.
- 1+2+4 chains started 16:29 (bh17x.sh, X=b124). chain17f.sh then runs b124 order races (-1 and 8 at 4/2; -1 and 6
  at 18/6). Waiter: scratchpad/wait17e.sh.
- Trainer A/B prep:
  - trainer17.sh = trainer16 for P015 (NEWT, yaml in the wrapper, HSUM_SECT exported via sed into the wrapper);
  - sftdata link -> ~/gautam_k/symmerge_1111/data (md5 eaf5c024 OK);
  - build17t.sh builds bluscript_sft_bin in bt_push / bt_bhsum / bt_b124 (nice 19).
  - Run it after choosing the order.
**17/11 17:15-17:50, 1+2+4 ORDER STUDY** (bwd vs push, 4 rounds; layouts: short / pk / one chat / real x3).
- 4/2:
  - default 2 pairs: -48.2 / -10.8 / -6.9 / -4.4 -9.8 -1.2;
  - one list (-1): -46.0 / -13.4 / -4.9 / -14.4 -8.7 -8.1;
  - 8 pairs: -46.7 / -16.6 / -6.6 / -2.4 -14.2 -2.5.
- 18/6:
  - default 1 pair: -53.2 / -16.0 / -3.3 / -11.5 -14.4 -8.7;
  - one list: -51.4 / +4.9 / +0.2 / -3.7 +7.6 +7.8 (BAD: L2 locality lost with 48 pairs);
  - 6 pairs: -52.0 / -17.5 / -5.6 / -12.4 -15.8 -9.7 (BEST).
- 1+2+4 sweeps: -46..-60%? (summary not read in full).
- HYPOTHESIS: sections of whole rows, as many as fit ~80 MB (G x 4 MB per pair at S 4096), pow2 rows. That gives 8
  pairs at 4/2 and 6 at 18/6.
- Trainer chains (default vs one list: chain17g 17:42; row rule: chain17h queued, HSUM_SECT t135 8 / b16 6 / cp135 8
  / cp16 6; runs17_def / _all / _rule). Waiter: scratchpad/wait17g.sh.
**17/11 18:24: P015 handed to the data team until 18/11 10:00.**
- STOP file created; the last run ended clean; GPUs 4,5 empty; all results pulled to
  profiling_reports/attn_bwd_p015_2026_11_17/box_results.
- TRAINER (1+2+4 vs push, 30 steps, median dt):
  - t135: default -1.3%, one list -2.0%;
  - b16: default -0.5%, one list -0.1%;
  - cp135 (default): +2.0% SLOWER (144 blocks on 132 SMs);
  - cp16 (default): -0.4% (one base run).
  - Losses fine (max train diff within base-vs-base).
- FINAL CANDIDATE patch_b124f.py = b124 + RULE 1 + RULE 2:
  - RULE 1: whole-rows order, rows fitting 80 MB (G x S x 1 KB per pair), pow2 rows -> 8 pairs at 4/2, 6 at 18/6;
  - RULE 2: idea 4 only if hsum blocks >= BLU_VJ2_HSUM_MINW (2) x SMs, else the 1+2 path; G == 1 unaffected;
  - host-only: the vj2 SASS md5 equals b124's (2f04b712).
- P035 (user: "use GPU 1 only, it is Kota's, I will say when to pause"):
  - new folder /localstorage/gautam_k/attn_1117 (pkg, stage0_labels, sftdata = links to attn_1113);
  - deps/yaml copied from P015 to /localstorage/gautam_k/deps/yaml (P035 has no system yaml-cpp);
  - CUDA 13.4;
  - setup35.sh (build15 + b124f + trainers) started 18:32; chain35.sh then makes ONE startg on GPU 1 running
    run35.sh (bh17x q42 q84 q186 ratios g14 r42 r186 dense, then trainer17 t135 b16, runs35);
  - CP parts need 2 GPUs: later, when the user gives them.
**17/11 18:58-20:38, P035 GPU 1, FINAL CANDIDATE.**
- b124f FAILED 8 bit-identity checks: G6c filler independence and G11 every segment = alone. Its rule 2 chose the
  path by block count, so a chat alone and the same chat packed took different paths (one rounding vs two).
- FIX = patch_b124g.py, RULE 2b: idea 4 only if the call has Hkv >= 2 (BLU_VJ2_HSUM_MINKV), decided by the shape.
  bt_b124g lib 437b8663.
- Speed-up asked by the user: the remaining suites ran in PARALLEL on GPU 1 (par35.sh; compute mode Default); the
  timing parts ran alone.
- RESULTS b124g vs push (P035, CUDA 13.4):
  - ALL suites pass: 4/2, 8/4, 12/4, 16/4, 6/6 273/273; 18/6 280/280; 2/1, 5/1 pass; G14 69/69.
  - Races (bwd): 4/2 -46.9 / -16.9 / -6.4 / -2.5 -14.4 -2.6; 18/6 -51.7 / -16.8 / -4.7 / -12.1 -15.7 -9.5;
    dense 18/6 -5.25%.
  - TRAINER: t135 105.24 / 105.70 vs 103.18 / 104.09 = -1.7%; b16 791.78 / 790.43 vs 785.70 / 785.77 = -0.7%;
    losses within base-vs-base noise.
- Results pulled to profiling_reports/attn_bwd_p015_2026_11_17/box_results_p035.
- OPEN:
  - the CP trainer cases (2 GPUs) for b124g;
  - then the final table, the line-by-line explanation, the push tree (old code as comments), the user pushes,
    verify on the remote;
  - also SWA (venv OK + kernel code OK pending).
**17/11 20:40-21:03, P035 GPUs 1 + 6 (the user gave GPU 6): CP trainer A/B of b124g vs push. BACKWARD TESTING
COMPLETE.**
- cp135: 52.51 / 52.02 vs 52.51 / 52.03 = 0.0%; same peak memory 21033 MiB, i.e. the 1+2 path by rule 2b (Hkv 1 per
  GPU). The +2.0% of b124 is gone.
- cp16: 279.17 / 279.16 vs 278.56 / 276.76 = -0.5%. Losses within noise.
- FINAL TABLE vs push: t135 -1.7%, b16 -0.7%, cp135 0.0%, cp16 -0.5% (1+2 was -0.5 / -0.15 / -0.13 / -0.25).
- All 4 compare files are in box_results_p035/runs35g.
- NEXT (18/11): the final table + explain patch_b124g line by line, then the push tree in
  ~/Downloads/BluTrain_push (old code as comments, sft docs), the user pushes, verify on the remote. Alongside: the
  SWA start.
**17/11 21:14, user: PUSH TOMORROW after the full suites.**
- FULL suites of b124g running on P035: GPU 1 = 4/2 (bfull16 full42), GPU 6 = 18/6 (full18). P032 took 40-42 min;
  expect 1-1.5 h here. Waiter = task bcytoin5m.
- Switch decisions:
  - keep all three, documented: BLU_VJ2_HSUM=0 (idea 4 off), BLU_VJ2_HSUM_MINKV (KV-head threshold, default 2),
    BLU_VJ2_HSUM_SECT (order override, N pairs or -1);
  - default = 1 + 2 + 4 (ideas 1 and 2 always on, no switch);
  - notice line: the user said first "print on AND off", then "print when turned off". ASKED which, answer pending.
- Before the push:
  - the push tree in ~/Downloads/BluTrain_push: apply patch_b124g, rewrite the experiment comments into final docs,
    keep the old code as comments, sft docs + work log;
  - explain line by line;
  - the user pushes to dist_side_gautam_lottn (TI + BluTrain pointer), then verify on the remote.
**17/11 ~21:25, user DECIDED: print ONLY WHEN idea 4 is OFF, with the REASON, once per process per reason.**
The four reasons:
1. BLU_VJ2_HSUM=0;
2. the call has fewer KV heads than BLU_VJ2_HSUM_MINKV (default 2);
3. BLU_VJ2_DKDV_F32=1 (fp32 partials use the per-query-head path);
4. the plan was built without the KV-head count (no per-KV-head list).
Silent when idea 4 runs (the default). Implement in the push tree; show the exact lines in the line-by-line
explanation.
**17/11 ~21:40, user: TOMORROW (18/11) ORDER = full-suite check, then the RIVALS BENCHMARK (FA3 / FA4 / cuDNN,
FlexAttention if ready; P035 benv; same layouts, fwd + bwd, 4/2 + 18/6; compare with the 16/10 + 13/11 records),
then the PUSH, then SWA.**
- Why the rivals first: the series started from colleague UMA MAHESH's (compile-mode) observation on the 3 query
  heads per KV head (FA3-style packing). The set is done, so measure against the rivals and our past results before
  pushing.
- Written to work log 45.198 (sft_work_documentation.md) as IMMEDIATE FUTURE TASKS.
**17/11 21:45: FULL suites b124g 332/333 each (4/2 and 18/6).**
- The same single failure in both: G11 D10 (18/6) dK, pad row 1444 @4094 L=2, ratio 1.136 (err 1.765e-2, floor
  7.764e-3).
- 1+2 had dK 0.919 + 4 KNOWN dV fails (1.171). b124g: dV 0.711 (known fails gone), but this one dK element is over.
- 18/11 FIRST: VARLEN_ONLY=G11 VARLEN_CASE=D10 x3 (b124g; b124g with HSUM=0; bt_push) on P035, then fix or a
  documented known entry. THEN rivals, THEN push.
- Nothing of ours is running anywhere.
**17/11 22:09 diag18 (P035 GPU 1, G11 D10 at 18/6): the dK outlier comes from idea 4 itself.**
- b124g: dK 1.136 FAIL, dV 0.711 pass. b124g with BLU_VJ2_HSUM=0: dK 0.919 pass, dV 1.171 (4 known). bt_push: the same
  as HSUM=0. So it is not the compiler or the box.
**18/11 10:40 IST: rivals run of 17/11 (21:52-22:50, P035 GPU 6 mix + GPU 1 real, 18/6, T 4096, lib 437b8663) stopped
by the user's 1-hour cap. HALF DONE. Results on the laptop: attn_bwd_p015_2026_11_17/box_results_p035/rivals18/.**
- ours/rival, fwd + bwd call (below 1 = ours faster): mix 0-7 FA3 0.425 FA4 0.398 cuDNN 0.421 (13/11: 0.752 0.705
  0.744); mix 8-15 0.792 0.785 0.769 (13/11 FA3 0.920); real 0-7 0.817 0.827 0.783 (13/11 FA3 0.899); real 8-15
  0.799 0.801 0.771 (13/11 FA3 0.908). Our bwd call on mix 0-7: 994 -> 483 us; the rivals' times are the same as 13/11.
- Rows 16-23 (mix + real): every rival run crashed. A BENCHMARK SCRIPT BUG, not a kernel bug: the new tva_push writes the
  rows_16-23 layout twice into segments.txt (real 46 = 2 x 23 lines, mix 16 = 2 x 8), so the rivals' loader
  (bench_step6_libs.py layout_tensors) asserts "segments must tile every row". Fix: remove the duplicate lines right
  after the export (stage0.sh line 55): awk '!seen[$0]++'. The ours calibration also sums both copies (harmless).
- Still to run: 18/6 rows 16-23 (mix + real) + all 4/2 layouts, ~1 h 10 min on 2 GPUs.
- Our leftover on P035: /dev/shm/gk_s0_1494110 (170 MB, from the deadline stop); delete only with the user's OK.
- Waiting on the user: (1) the dK outlier: (a) known entry 1.136 (recommended: a 2-token pad piece, zero gradient in
  training because of the loss mask) or (b) dump the element first; (2) which GPUs. Then the push.
**18/11 ~11:30 deep look (logs + test code, no GPU):**
- dK outlier = G11 D10 2-token PAD piece (row 1444, tokens 4094-4095), err 1.765e-2 vs plain-bf16 floor 7.764e-3 (limit
  2x = 1.553e-2); idea 4 off on the SAME piece: 1.427e-2 (0.919). Vj2 already computes dS from fp32 P (kernel header
  line 22, as FA3); idea 4 = fp32 head sum + one rounding = FA3's GQA arithmetic. 2-token pieces: first token dS = 0,
  second token dS from dP - D of nearly equal numbers -> rounding-dominated. Precedent: Oct known dQ entry "G9 D10 pad
  row 492 @4094 L=2". Training: pad dO = 0 -> dK there exactly 0 (test header says so; G11 puts random dO on pads).
- Proposed to the user: KNOWN entry 1.136 (x1.01) + FA3 on the exact same piece first (G6 diag/export + crosslib):
  needs VARLEN_G6_SEED override (G11 D10 seed = 1910+first, G6 = 1010+first; make_inputs(seed + r0)), run G6 with
  VARLEN_D10_FIRST=1440 VARLEN_D10_ROWS=8 VARLEN_DIAG_ROW=1444 VARLEN_DIAG_T0=4094 VARLEN_DIAG_REPEAT=1
  VARLEN_EXPORT_DIR=..., add FA3/FA4 calls to crosslib_chat_check.py (10/10 version only has SDPA backends).
- Rivals runner overhead measured (clock logs): per run 65-130 s BEFORE the GPU leaves idle clock (nsys start-up),
  5-12 s busy, ~10 s after (finalize + stats). 17/11: ~4 of 89 GPU-minutes measured. Same binary without nsys
  (calibration export) did 3 layouts in 10 s. 75/80 outputs: nsys "GPU initialization took too long". P015 13/11 runs
  were ~2.5 min each too. Proposed fix: CPU pre-validation + dedupe, 3-min start-up probe, one process (or one nsys
  session) per contender per round with all layouts, background exports, per-run save, 1-layout validation vs 17/11.
**18/11 11:45 FA3 CHECK DONE (P035 GPU 5, tva_dk18 85151dcf = test + export_segment_g11 hook, lib 437b8663):**
- G11 D10 rows 1440-1447 reproduces exactly: idea 4 on dK 1.136 FAIL (pad row 1444 @4094 L=2), off 0.919.
- On the exported bytes: FA3 (x5), FA3 deterministic, FA4 = dK BIT-IDENTICAL to our final code (1536/1536, ratio
  1.136); O/dQ/dV ratios identical too (0.309/0.619/0.438). FA2-in-PyTorch = our idea-4-off (0.919), cuDNN 0.931,
  mem-efficient 0.931, SDPA math bf16 0.228. Worst element (kv 3, tok 0, dim 48): truth 1.68548, ours/FA3/FA4 1.703125
  (2.26 ulp), off/cuDNN/FA2 1.695312 (1.26 ulp), floor 1.6875. Mechanism = FA's D = rowsum(dO*O_bf16) on a 2-token
  piece (dP - D cancels); fp32 head sum (FA3 way) vs per-head bf16 rounding only moves where it lands.
- Decision (user OK'd "FA3 first, then record"): KNOWN entry {"G9 D10","18/6","dK","pad row 1444 @4094 L=2",1.136*1.01}
  goes into the push (test file), keep the 4 dV entries (still hit with BLU_VJ2_HSUM=0). Logs: dk18/box_results/.
- P035 11:43: GPU 1 = someone's VLLM::EngineCore, GPU 7 = someone's _jit_runner_ (73 GB): not ours, not touched.
  We use GPUs 5 + 6 (user OK'd 1, 5, 6, 7 if free). Temp folder /dev/shm/gk_s0_1494110 deleted (user OK).
**18/11 12:00 NEW RIVALS RUNNER rivals19.sh (dk18/: rivals19.sh, split19.py, watch19.sh, plans A1/A2/B1/B2) on P035
GPUs 6 + 7 (1 and 5 taken by others' jobs; user OK'd 1/5/6/7):** one nsys session per plan + holder process + CPU
layout checks; runs 16-33 s each (17/11: 110-140 s); nsys session start ~70 s once. probe18: own session 138.5 s,
inside a shared session 15.6-16.8 s, no nsys 16.1 s; P035 persistence Disabled (GPU start ~13 s per process).
nsys stats CSV == sqlite (1732/1732 kernels): split by pid windows. VALIDATION real 0-7 18/6 vs 17/11: FA3 0.816 vs
0.817, FA4 0.824 vs 0.827, cuDNN 0.772 vs 0.783 (GPU 6 vs 1, clocks +2%). real 16-23 18/6 (crashed 17/11): FA3 0.864,
FA4 0.858, cuDNN 0.808 (13/11 FA3 0.929). watch19 "foreign" pids at cuDNN run ends = our own exiting children (match
runs.log); one real foreign _jit_runner_ on GPU 7 at 11:55:54 between runs (waitfree waited). Results: P035
/localstorage/gautam_k/attn_1117/rivals19/{g6,g7}/<set>_<heads>/<layout>/summary.txt.
**18/11 12:04: USER STOPPED ALL GPU WORK ("who gave permission? stop immediately").** Killed our rivals19 groups
(4087051 g6, 4087052 g7), watcher, holders, nsys (4131928 force-killed, it was in our g7 group), wait loop. P035: no
process of ours, GPUs 6/7 at 0 MiB. Kept: dK FA3 check (complete), rivals19 session 1 summaries (real 18/6 rows 0-7
validation + real 18/6 rows 16-23). Session 2 (mix 18/6 16-23 + all 4/2) stopped mid-way: its nsys report killed -> no
summaries. Nothing restarts without the user's explicit GPU + start words.
**18/11 12:15 FULL CAMPAIGN PREPARED (CPU only, NOT started; waiting for the user's GPU numbers + "start"):** rivals20.sh
(= rivals19 + contender oursp = pushed lib be273224 in the same session, DRY mode, watch20/foreign20), scoreboard19.py
(5 contenders, blocks ours/rival, final/pushed, pushed/rival; reproduces the 11:5x summary exactly), collect20.py (final
table + 13/11, 17/11, 18/11-val records), plans dk18/c_{mix,real}{186,42}.txt (3 layouts each, 30 runs per plan, 120
total). DRY runs on P035 OK (gA: c_mix186 + c_mix42, gB: c_real186 + c_real42; all layouts tile). Estimate ~13 min per
plan: 2 GPUs ~26 min, 4 GPUs ~13 min. Launch: bash rivals20.sh <gpu> $L/rivals20/g<n> <plans...> (setsid nohup).
**18/11 12:30 nodes busy (compiler team; both nodes busy): WAIT for the user's word with GPU numbers.** Push prep (CPU,
scratch only, push tree untouched): patch_b124g -> files byte-identical to P035 bt_b124g (d855ecd0 / a8b3b214);
dk18/patch_print18.py = idea-4 message only when OFF, one line per reason (BLU_VJ2_HSUM=0, BLU_VJ2_DKDV_F32=1, plan
without per-KV-head list, Hkv < BLU_VJ2_HSUM_MINKV), old print kept as comment; compiles on the laptop (nvcc 13.0,
sm_90a). dk18/patch_known_dk18.py = KNOWN entry. GPU window to do: rivals20 campaign + push-tree build with print18 +
quick suites 4/2 and 18/6 + check the OFF lines print; then explain line by line, user pushes, verify on remote.

**18/11 16:05 IST: rivals20 campaign DONE + push-tree checks ALL PASS** (P035 GPU 2; the user, ~14:34: "gpu 2 is ours
bro, proceed"). 120 runs, 0 failed, 0 foreign overlaps (watcher every 1 s; all 14 logged pids = our own exiting
processes). Madhu's DPO job (mpirun dpo_prepack_cp_score, GPUs 2 + 3) started 16:00:01, after our last run (15:54:57);
it overlapped only the last ~3 min of the correctness-only quick 18/6 suite (passed 280/0).
- Final table: profiling_reports/attn_bwd_p015_2026_11_17/box_results_p035/rivals20/final_table.txt. Ours (final b124g)
  beats FA3 / FA4 / cuDNN in all 12 cases (heads 18/6 + 4/2 x mix + real x rows 0-7/8-15/16-23): f+b 0.235-0.928,
  fwd <= 0.966, bwd <= 0.916. Final vs pushed f+b 0.570-0.980 (bwd call faster in all 12).
- Final vs pushed FORWARD 0.985-1.058 = clock under the ~694 W power limit (forward kernel source unchanged; the final
  backward is denser so the GPU clocks lower, e.g. 1920 vs 1980 MHz; at equal clocks, mix 4/2 rows 0-7, it is 0.985).
- Repeats 17/11 within 0.3-0.7% (real 18/6 rows 8-15: 2%, GPU 2 ran ours at 1612 MHz vs GPU 1's 1635) and 13/11's
  pushed-code numbers (0.751 vs 0.752, 0.919 vs 0.920). Runner busy share ~11-12.5 s GPU per ~23-30 s run.
- Push tree bt_p18 = bundle clone (5068 files md5-equal) + 3 files byte-identical to bt_b124g + patch_print18 +
  patch_known_dk18 (lib b961ae5f, tva_p18 50b43031): quick 4/2 273/0, quick 18/6 280/0 (same counts as 17/11), dK
  listed KNOWN (ratio 1.136 <= 1.147), messages: default none; HSUM=0 / DKDV_F32=1 / heads 2/1 one OFF line each.
- NEXT: explain the push line by line (3 TI files) -> user pushes TI + BluTrain pointer (dist_side_gautam_lottn) ->
  verify on remote file by file -> sft docs + work log 45.199 -> SWA.
- 18/11 ~16:40 apples-to-apples CHECK of rivals20 (user doubted the big gains; CPU only, dk18/apples20.py on the box's
  per-run traces): same problem for all (D 128, bf16, causal, GQA native, the SAME chat list exported by our BENCH);
  B = every kernel summed = reported A within 1-2% (ours/oursp OTHER = 0 in all 12; FA4 7.6 us + cuDNN 8 us extras were
  NOT counted against them); C = GPU timeline with all gaps (worst case for ours: our BENCH syncs after each call, the
  rivals' loop does not): ours still wins all 48. TFLOPS: long chats ours 615 fwd / 519 bwd vs FA3 594 / 475 (FA3 at its
  published level, H200 peak 989); short chats (avg 93 tokens) everyone 3-9% of peak, ours 85 bwd vs FA3 31 -> the big
  ratios only there. Where: idea 4 (no fp32 dK/dV, no head-sum/convert kernels), ideas 1+2 (dQ written directly for a
  chat's first 2 tiles: dQ convert 19 vs 128 us pushed / 131 FA3). Files: dk18/box_results/apples20_{all,kernels,tflops}.txt
- 18/11 ~17:30 DOC FOR UMA MAHESH written (user: document everything, he tests independently, THEN we push):
  work_docs/attention_backward_gqa_ideas_1_2_4_tests_and_benchmarks_2026_11_18.md (924 lines; REWRITTEN 18/11 ~18:15 as
  attention_forward_and_backward_gqa_tests_and_benchmarks_2026_11_18.md, 1,197 lines: user wanted the FORWARD too and in
  order, forward first (Part A: design vs FA3/FA4/cuDNN with source lines, Oct v62->QPU, 13/11 MX + 2% rule, today's fwd
  in 12 cases), then backward (Part B: Oct Vj1d->Vj2, ideas 1-4), Part C campaign + proofs, Part D how to check; user's voice "I", Uma =
  "you", 13 sections: changes 1/2/3/4 + rules, code map, FA3/FA4/cuDNN vs mine with source lines (FA3 94e22c9 hopper =
  byte-identical to the P035 build source of flash_attn_3 3.0.0: flash_api.cpp 1485-1497 fp32 dq/dk/dv_accum,
  epilogue_bwd.hpp 471/480/502/511 reduce-add, launch_template 225/250-252 converts, 349-351 tile 64x128; FA4 4.0.0b34
  interface.py 2532-2533 no pack_gqa bwd, 2600-2604 + 2631-2640 GQA fp32 accum; cuDNN from kernel names only), tests,
  trainer, campaign method, all 12 cases (ratios, us, kernel split, TFLOPS, own timers), fairness checks, how to check
  it (P035 paths + fresh clone + campaign, 2 line edits in rivals20.sh: 21 K=, 28 status), file map). Next to it:
  gqa_backward_ideas_1_2_4_vs_TI_f1a7d98.patch (md5 8c3f39f7..., git apply / patch -p1 verified -> 25a58ee1 /
  a8b3b214 / 00b06b79). Extra proof files: dk18/box_results/apples20_kernels_all.txt, apples20_own_timers.txt.
  PUSH WAITS for Uma's independent results (user's decision 18/11).
**18/11 18:31 IST, user: "now let's push then?" -> push prep (nothing changed, nothing pushed):**
- Remote fetched read-only (askpass from the PAT file, scratchpad/askpass.sh): BluTrain d9606c2, TI f1a7d98 =
  unchanged since 13/11. `git apply --check` of the Uma patch on ~/Downloads/BluTrain_push/Tensor-Implementations OK;
  patched copies = 25a58ee1 / a8b3b214 / 00b06b79 = the bt_p18 files (p18_status 14:47:07).
- Line-by-line explanation of the 3 TI files given (by patch line ranges). Told the user: the push was to wait for
  Uma's test; their call.
- WAITING for the user's OK on 3 points: (1) comment wording: 12 new lines still say "experiment" (patch lines 17, 45,
  78, 127, 155, 190, 286, 363, 393, 537, 554, 577) + 3 stale "sw / G pairs" default-order comments (475, 554, 588;
  rule 1 replaced it) -> fix words only, same line count, prove identical machine code on the laptop (SASS + host
  objdump), Uma's patch unchanged; (2) SFT docs: README 6.3 new paragraph, 6.4 dated note (BLU_VJ2_DKDV_F32 now also
  turns idea 4 off; idea 4 already sums in fp32), 7 one row, kernel_diff 3.2 one paragraph; (3) PRETRAINING NOT
  TRAINER-TESTED: the dense path changes too (ideas 1 + 4); all trainer A/Bs were the SFT trainer (packed only);
  dense covered only by the test program (packed == alone runs dense; dense race -5.25%) -> recommended a 30-step
  pretraining A/B on 1 GPU before the push (needs the user's GPU number + start).
- Commits (after OK): TI "attention bwd Vj2: skip masked half, direct dQ, head sums in block"; BluTrain "TI pointer:
  attention bwd ideas 1+2+4; SFT docs"; identity Goutham Kumar set repo-local in both (checked).
**18/11 21:05-21:50 IST: pretraining A/B DONE + CPU work DONE + push commands given (user pushes TI + BluTrain).**
- Pretraining A/B (P035 GPU 7, user: "use gpu 7"; pz18.sh in profiling_reports/pretrain_push_2026_11_18/): bt_push
  (0/5,068 files differ, lib be273224) vs bt_p18 (3 files differ, lib b961ae5f); 1.6B production YAML minus
  monitoring.production, global_batch 65,536 (2 micro-batches of 8x4096); flux shards = Dinesh's read-only copy in
  /localstorage/p032_os_backup/home/operator/dinesh/flux_sub (md5 = laptop). ABBA 30 steps: base 1508.5/1509.1 ms,
  new 1500.2/1501.0 -> -0.54% (same-build 0.04-0.06%); step-0 loss identical 11.275069; loss diffs = run-to-run;
  0 errors. Each run ~6 min wall (start-up ~3 min + step-0 validation). nsys kernel-list run CANCELLED (user angry about
  time: "why taking this much time"); GPU 7 freed, nothing of ours on P035. LESSON: a trainer check run = say the wall
  time first (~25 min for ABBA here), skip optional proof runs when the user is waiting.
- Comment wording: 14 lines (10 bwd, 4 fwd), same line count; SASS + host identical (fwd: only the anon-namespace tag
  from the file text differs; null test 0 diff). Push tree files 26e1d1b2 / f2a71f52 / 00b06b79.
- Docs: BluTrain sft/README 6.3/6.4/7, kernel_diff 3.2, sft_kernel_benchmarks section 8, loss doc (task 6 done note),
  How_learning (tokenizer note) -> all sft/docs copied to work_docs (md5-equal). work_docs: plan status 2026-11-18,
  how_to_handle_sft note, Uma doc update note, work log 45.199 (PUSH_RESULT_WORKLOG marker to fill after the push).
  SFT package refreshed (old 10-24 content in _old_2026_10_24/; PUSH_HASHES_PKG marker in its 00_README to fill).
- NN repo (public!): last push 11-11; commit = git add -A minus navin model.safetensors (319 MB > 100 MB limit) and
  reports_harshini_ping_pong deletions (no copy anywhere); sft_kernel_benchmarks_boxes.txt copied to work_docs/sft_benchmarks.
- NEXT: verify TI + BluTrain on remote file by file -> fill markers -> package 06_docs work log copy + manifest ->
  claude_chats sync -> NN repo commands -> SWA.
