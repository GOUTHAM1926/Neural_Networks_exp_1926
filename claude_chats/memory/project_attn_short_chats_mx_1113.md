---
name: project_attn_short_chats_mx_1113
description: "11-13 attention forward MX (2 query heads of a KV head share the half-empty last pair of a chat) + 2% rule: PUSHED + VERIFIED on remote (TI f1a7d98, BluTrain d9606c2, dist_side_gautam_lottn); fwd -21..-26% at <=64-token chats, real rows -0.5..-1.3%, SFT step unchanged; FA3 fwd still wins at ~40 tokens and below; P032 folder attn_1113 (4.2 GB) cleanup pending"
metadata:
  type: project
---

**What.** Started from Uma Mahesh's Nova varlen forward study (`work_docs/varlen-forward-study-nov13.md`: his kernel
1.28-1.44x slower than FA3 on rows of 32-160-token chats, cause named "one tile per query head"). Our forward tiles per
query head too and that case had never been timed. The user: measure, try the fixes one by one, push what holds.

**Full record:** work log `sft_work_documentation.md` section 45.194 (every table, incidents, commands to continue).
Study folder `profiling_reports/attn_short_chats_2026_11_13/` (PLAN.md with a STATE block, cands/, patches/, pkg/,
box_results/). Note for the colleague, updated with the measurements (sections 10-11):
`work_docs/attention_packed_rows_our_experiments_for_nova_forward_study_2026_11_13.md`.

**Results (P032 GPUs 4/5, 8 rows x 4,096, heads 18/6, hd 128; ratio = ours / rival):**
- chats of 32-160: fwd 0.875 FA3 / 0.503 FA4 / 0.607 cuDNN; fwd+bwd 0.763 / 0.715 / 0.755. FA3 = 94e22c9 with PackGQA
  ON (read from its kernel template args). So we do NOT lose on short chats.
- chats of 300-1,500: fwd 0.909 FA3 (FA3/FA4 one round), fwd+bwd 0.922. Real group 2: fwd 0.907; group 1: 0.916 (1 round).
- one forward unit costs 19.1 ns (40-token chats) / 19.6 (64) / 25.7 (120): nearly all fixed cost.
- **MX** (in a chat with an odd number of 64-row tiles, the idle second compute group takes the same last tile of
  another query head of the same KV head; plan builder + unit decode only; switch `MX_UNITS`, on per run with
  `BLU_GQA_MX_HKV=<KV heads>`): quick suite 245 passed / 0 failed on both legs; race 4 rounds: fwd -8.01% (32-160),
  -2.68% (300-1,500), -0.71 / -1.05 / -0.39% (real groups), 0 on one long chat; MX code with the old plan costs
  +0.5..+1.1% on many-unit rows; ncu: cycles -9.9%, instructions -0.9%, DRAM read -8.5%. Dense kernel SASS identical.

**Not done (paused 17:01 IST 11-13 by the user):** one long chat vs rivals; real group 3 vs FA3; real group 1 round 2;
pk round 2 for FA3/FA4; race rounds 5-8; the 4 suite groups that need stage-0 labels.bin; full 605 suite; MXon vs
rivals; heads 4/2; trainer A/B; stage 2 (FA3-style row packing, not built; bound >= 55% on 40-token chats, but only
0.93% of our tokens are in chats <= 160). MX NOT adopted, NOT pushed, master files untouched.

**Box:** `/localstorage/gautam_k/attn_1113` on P032 kept (1.6 GB, built: libs BASE/MX under work/libs, bin/tva,
pkg/stage0_b.sh = the script with the fixes). Nothing of ours running there.

**For adoption (the user decides):** Hkv as a real argument of gqa_chat_plan_build/make (+ trainer) instead of the env
var + registry; two kernel instantiations picked at launch (plans without mixed units keep today's kernel); full suite;
trainer A/B; explain -> OK -> push -> remote check.

**UPDATE 11-13 evening (resumed 18:27 IST on P032 GPUs 4/5, later 4/6; full record to be written into work log 45.194/45.195):**
- Push-ready ("adoption") form built: `adopt/` by `patches/make_mx_adopt.py` (Hkv as a real argument of
  gqa_chat_plan_build/make, third kernel instantiation MODE_CHATS_MX, harness + SFT trainer pass Hkv, BLU_GQA_MX=0 off
  switch). SASS: dense and plain packed = master, mixed-unit kernel = the raced candidate (0 lines, laptop + P032).
- Passed on it: quick suites 280 / 280 (MX off) / 273 (heads 4/2), full suite 333 (18/6), G14 8192: 69; trainer A/B
  (135M, 30 steps): identical step-0 losses, step time unchanged (106.1-106.5 ms); trainer launches the MX kernel
  (5,432 of 5,432 forward launches); race real rows 8 rounds: fwd -1.33 / -1.32 / -0.52%.
- Length sweep (20 lengths, 3 rounds): fwd -21..-26% at <=64 tokens, -8..-10% at 160/192, -1..-7% at longer odd
  bands, 0 at even bands; ONE loss: rows of 96-token chats + a 64-token leftover: +0.92% (MX kernel runs, almost
  nothing shared). Fix prepared, NOT tested, NOT approved yet: "2% rule" (`adopt2/`, `patches/make_mx_rule.py`): mix
  only when mixed units >= 2% of the plain list's units (BLU_GQA_MX_MIN_PCT).
- vs FA3 by length (new build, fwd | fwd+bwd): 32: 1.146 | 0.643 (FA3's forward wins at ~40 tokens and below);
  64: 0.620 | 0.645; 128: 0.656 | 0.783; 192: 0.702 | 0.771; 704: 0.825 | 0.882; 2048: 0.911 | 0.948;
  32-160 mixed: 0.802 | 0.752; real rows: 0.897-0.919 | 0.899-0.929. Fwd+bwd on real rows MX vs today ~ -0.05..-0.25%.
- **EOD of 13/11 (user, 21:15 and 21:18: "remind these two tables, in eod we will put those two"): put BOTH tables of
  my 21:16 message into the EOD stats**, next to the other benchmark tables: (1) the ratio table per chat layout
  (columns: MX against current kernel forward; current / FA3 forward; MX / FA3 forward; current / FA3 forward +
  backward; MX / FA3 forward + backward; rows: every chat 32 / 64 / 128 / 192 / 704 / 2,048, chats 32-160 mixed,
  300-1,500 mixed, one chat of 4,096, real groups 1-3); (2) the forward-times table (current kernel, MX, FA3, in µs).
  The numbers are recomputed by the script in my 21:14 tool call from box_results/pkg_from_box (results, results_c,
  results_b5, results_mx, results_mx2, results_len, results_sweep).
- 2% rule: the user said "proceed" at 21:18; its tree bt_adopt2 built on P032 (GATE2 passed: all three kernels identical
  to bt_adopt's), its checks (stage4_rule.sh) running on GPU 4 from 21:19. Nothing pushed; master files untouched.

**FINAL 11-13 22:01 IST: PUSHED by the user ~21:57, verified on the remote file by file** (TI 4d5c738..f1a7d98, 4 files;
BluTrain d9d3251..d9606c2: bluscript_sft.cpp + TI pointer + sft/README.md + sft/docs/kernel_diff_sft_pretraining.md;
each code file on the remote = the tested adopt2 file by md5). 2% rule tests passed (bt_adopt2: kernels identical to
bt_adopt; 96-token rows -> today's kernel, -0.02%; 160/192 keep -9.7/-9.0%; quick suite 280/280; trainer 5,717 forward
launches all MODE_CHATS_MX, step-0 losses identical). Table extras measured: current kernel vs FA3 fwd at 32/64/192/704 =
1.536 / 0.788 / 0.774 / 0.859; new build vs FA3 on the 300-1,500 mix 0.897 (f+b 0.920). Both EOD tables final:
`profiling_reports/attn_short_chats_2026_11_13/box_results/full_table.txt` (script full_table.py). Full record: work log
45.194 + 45.195; colleague note sections 10-12. Nothing of ours runs on P032; folder
`/localstorage/gautam_k/attn_1113` (4.2 GB) still there: the user has not said keep or delete.
Open/future: FA3-style 3-head packing (needed to pass FA3's forward at <= ~40 tokens); the same sharing for the
backward; cheaper unit decode in the MX kernel; full suite at heads 4/2 (stopped at 295 PASS lines, 0 FAIL), G14 at
32,768, CP with MX, 1.6B trainer A/B: not run.

Related: [[project_dense_fwd_gap_1017]], [[feedback_box_ops_traps]], [[feedback_guard_our_gpus_during_runs]].
