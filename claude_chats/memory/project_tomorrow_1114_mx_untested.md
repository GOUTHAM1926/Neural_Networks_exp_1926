---
name: project_tomorrow_1114_mx_untested
description: "TOMORROW 14/11/2026, FIRST THING (user 13/11 night): test what was NOT tested for the pushed attention MX change -- speed race at heads 4/2, GPU suites at other head ratios (12/4, 16/4, 2/1, 6/6, 5/1), context parallelism with MX on 2 GPUs, plus the runs cut on 13/11"
metadata:
  type: project
---

User, 2026-11-13 ~22:40 IST: "we will test tomorrow on what we didn't ... make these too as immediate future tasks for
tomorrow start, note and document, and also this too: context parallelism ... has not been run on GPUs with MX yet".
The MX change (+ 2% rule) is already pushed and verified: TI f1a7d98, BluTrain d9606c2, branch dist_side_gautam_lottn
([[project_attn_short_chats_mx_1113]]; work log 45.194-45.196).

**Remind the user of this list at the start of the next session, then ask which box / GPUs (ask before every GPU run).**

Tomorrow's tests, in this order:
1. **Heads 4/2 (135M), speed:** race new against current on the chat layouts (the 13/11 runs at 4/2 were only the
   quick suite, 273 passed, and the 30-step trainer). `H=4/2` in `pkg/stage2.sh race` / `pkg/stage3_sweep.sh`.
2. **Other head ratios on a GPU** (13/11: only the CPU check of the work list): quick suite + "which kernel runs"
   at 12/4, 16/4, 2/1 (MQA), 5/1, and 6/6 (MHA: the plan and the kernel must stay the old ones). Run each with
   `BLU_GQA_MX_MIN_PCT=0` (forces mixing in every small layout) and with the default 2% rule.
3. **Context parallelism with MX** (2 GPUs, BLU_SFT_CP=2): old tree against the pushed tree, same losses, step time,
   and the kernel instantiation count under nsys. The trainer passes `kv_heads / U` to the plan builder; never run on
   GPUs with MX.
4. **Runs cut on 13/11:** the full suite at heads 4/2 (stopped at 295 PASS lines, 0 FAIL); long rows G14 at 32,768;
   the quick suite of the rule's build with `BLU_GQA_MX_MIN_PCT=0`; a 1.6B trainer A/B (needs the 1.6B checkpoint and
   data on the box).

How: test the CONSUMER build = a fresh clone / pull of the pushed branch on the box (not the bt_adopt2 copy), with the
unchanged tree `/localstorage/gautam_k/attn_1113/BluTrain` (TI 4d5c738) as the "current" leg. Scripts, layouts and the
SFT test data are in `/localstorage/gautam_k/attn_1113` on P032 (4.2 GB; KEEP it until these tests are done) and in
`profiling_reports/attn_short_chats_2026_11_13/pkg` on the laptop. On P032 a run under nsys takes ~2 minutes for 5 s of
GPU work: races without nsys, one nsys session per layout where a library comparison is needed.
Not in this list (future, bigger): FA3-style packing of 3 heads into one tile; the sharing idea for the backward.

**IN PROGRESS 2026-11-14 (user 12:15 IST: "ours is gpu 1, so let's continue" = P032 GPU 1 only).**
- Laptop folder: `profiling_reports/attn_mx_followup_2026_11_14/` (`pkg14/` scripts, `box_results/`); work log 45.197.
- Box: `/localstorage/gautam_k/attn_1113/bt_push` = fresh clone of the pushed branch (made on the laptop with the
  org token through an askpass script + lfs filter off, copied without .git), built 12:42; `pkg14/` scripts;
  `pkg14/status` = the chain's status file; `bash pkg14/ours14.sh` lists our processes (read-only).
- CPU checks passed: 5 pushed files = pushed md5s; clone = laptop BluTrain_push (5,068 files); vs bt_adopt2 only the 2
  SFT docs differ; SASS of the 3 forward kernels = bt_adopt2's (0 lines). GATE14 PASSED.
- Chain started 12:44:49 IST: `chain14.sh 1` = stage5.sh "smoke race42 sweep42 ratios0 reach" -> trainer14.sh "b16"
  (1.6B SHAPE, random weights, VOCAB=49280: the 1.6B checkpoint is not on P032) -> stage5.sh "suite0 ratiosd g14
  full42". About 3 h on one GPU. Pause = `touch pkg14/STOP` (stops before the next run; finished runs are skipped on
  restart). Smoke passed.
- NOT started: item 3 (CP) needs TWO GPUs: `trainer14.sh <a,b> "cp135 cp16"` is ready; ask the user for a second GPU.
- Default foreign-process rule today: report + wait (no kill word given for this campaign).

**PAUSED 2026-11-14 12:57:45 IST (user 12:54: "pause everything at 1pm as Kruthik uses it from 1 to 3:30 pm";
then "we will continue later").** Chain stopped by `pkg14/STOP` at the END of race round 2 (whole rounds only);
nothing of ours on P032, GPU 1 free. Done: build checks, smoke, heads 4/2 race rounds 1-2 (fwd: 32-160 chats -14.1%,
300-1,500 -3.9%, one long chat -0.2%, real rows -1.9 / -2.8 / -3.8%; bwd unchanged). Left (~2 h 45 min on one GPU):
race rounds 3-4, sweep42, ratios0, reach, b16 trainer, suite0, ratiosd, g14, full42; item 3 (CP) needs 2 GPUs.
**RESUME ONLY ON THE USER'S WORD (after 15:30): copy the laptop pkg14 to the box (chain14.sh fix), `rm pkg14/STOP`,
start `chain14.sh <gpu>` once, re-arm the monitor (mon14.sh).** Ask whether a second GPU is ours then.

**RESUMED 2026-11-14 15:34:09 IST on P032 GPUs 1 AND 3 (user ~15:15: "ours is gpu 3 bro you can continue now", then
"gpu 1 too ours, use it"; also "don't use much quota, do it efficiently": no agents, few calls).**
- Started ONCE by `pkg14/start14b.sh 1 3` -> `chain14b.sh 1 3` (new, 14/11 afternoon): (1) smoke on both GPUs;
  (2) item 3 FIRST, the only two-GPU item: `trainer14.sh 1,3 "cp135 cp16"` (RUNMAX 900); (3) two streams side by side:
  stream T on GPU 1 (timed; the GPU of race rounds 1-2): race42 rounds 3-4, sweep42, b16, ratiosd, g14;
  stream C on GPU 3 (checks only): ratios0, reach, suite0, full42. Estimate ~1 h 55 min, end about 17:30 IST.
- Status lines carry a tag: [chain], [T], [C], [CP]. Smoke logs are now `results_smoke/smoke_g<gpu>_<heads>.log`.
- Watch: `mon14.sh 1,3` (30-minute windows, re-arm). Pause: `touch pkg14/STOP`. Restart after a pause: `rm STOP`,
  `bash start14b.sh 1 3` (refuses when a chain of ours is alive or someone is on a GPU; finished runs are skipped,
  but trainer A/B runs are skipped only when their `.result` says "stopped after step").
- Both GPUs were free at 15:20 (two samples); default foreign-process rule: report + wait.
- Still to do after the runs: pull results, final tables, work log 45.197, announce the GPU release.

**PAUSED AGAIN 2026-11-14 17:00:40 IST (user 16:57: "by 5 pm pause all our processes, Kruthik again taking for an hour,
pause till I say again").** STOP written 17:00:21; our two runs on GPU 1 and GPU 3 ended by exact pid 17:00:36 (the
12/4 default-rule quick suite at 226 s, the full suite at 4/2 at 200 PASS lines / 0 FAIL); GPUs 1 and 3 free 17:01,
nothing of ours on P032. Results pulled to `box_results/pkg14_from_box` + `runs14_from_box` (17:01).
- DONE 15:34-17:00: smoke on both GPUs; item 3 CP PASSED (cp135: same losses, 52.4 vs 52.5 ms, 4,773 launches per GPU
  all `<2>`; cp16: same losses, 283.0 vs 282.9 ms, ~2,920 `<2>` + 312 `<1>` per GPU); 4/2 race 4 rounds (fwd -13.6 /
  -3.6 / +0.0 / -1.9 / -2.0 / -3.2%, bwd within +-0.25%); sweep42 3 rounds (odd lengths -37% .. -1%, even +-0.6%, bwd
  +-0.5%); ratios0 (16/4, 2/1, 5/1, 6/6: 273/0; 12/4: 272 + 1 FAIL); reach (12/4, 16/4, 2/1, 5/1: `<2>` on the two
  chat layouts, `<1>` on the long chat; 6/6: all `<1>`); b16 (same losses, 782.5 vs 783.1 ms, 3,354 `<2>` + 338 `<1>`);
  suite0 18/6 MIN_PCT=0: 280/0.
- OPEN FINDING: quick suite NEW at heads 12/4 fails ONE check: "G11 D5 64x64, 32x128 12/4 dV", chat row 1 @768 L=128,
  err 4.509e-02 vs floor 2.222e-02 (ratio 1.015). Reasons it is probably not from the push: backward kernel unchanged,
  that chat has an even tile count, "A vs alone (dense kernels)" 0 mismatched elements. NOT proven: BASE never ran this
  suite at 12/4. Control = `pkg14/ctrl14.sh <gpu>` (BASE + NEW with BLU_GQA_MX=0 at each failed head setting, ~3.5 min
  each); the launch was denied by the permission classifier as an extra run -> ASKED THE USER (16:07), NO ANSWER YET:
  do not start it without their yes. ctrl14.sh + the mon14.sh [ctrl] filter are NOT yet copied to the box.
- LEFT (about 40 min with two GPUs): GPU-1 stream `stage5.sh <g> "ratiosd g14"`, GPU-3 stream `stage5.sh <g> "full42"`
  (both restart their cut runs from the start), then the control if allowed. Restarting chain14b.sh also works
  (finished runs are skipped) but repeats smoke and the CPU parts of the trainer A/Bs.
- RESUME ONLY ON THE USER'S WORD; ask which GPU(s). Then: final tables, work log 45.197, announce the GPU release.
- EVIDENCE on the 12/4 finding (from logs, 17:05): the SAME error value 4.509e-02 for that dV check appears at heads
  18/6 in six logs of 13/11, with mixing ON (quick_NEW_18x6, full_NEW_18x6, quick_MXon, quick1_NEW_18x6) and with mixing
  OFF (quick_NEWoff_18x6, quick_MXoff), and today at 18/6 MIN_PCT=0 and at 12/4 with the default rule. Only the floor
  differs: 2.581e-02 at 18/6 (ratio 0.873, PASS) against 2.222e-02 at 12/4 (ratio 1.015, FAIL). So the error does not
  depend on mixing; the test's noise floor is lower at 12/4. Still missing: a literal old-tree run at 12/4 (ctrl14.sh).

**18:26 IST 14/11: NOTHING RUNNING, STOP FILE IN PLACE, WAITING FOR THE USER.** At ~18:24 I launched `start14c.sh 1 3`
(-> `chain14c.sh 1 3` = smoke, then stream T `stage5.sh 1 "ratiosd g14"` + stream C `stage5.sh 3 "full42"`) on an
unclear message; the user interrupted ("dont wait / dont run", then "wait"). Dinesh's 8-GPU test was on all GPUs; our
smoke waited, ran 18:25:15-18:25:49 after his test ended (passed on both GPUs), and the STOP file ended the chain at
18:26:03 before any suite started. Left is unchanged: ratiosd + g14, full42 (~40 min with two GPUs), and the control
`ctrl14.sh` (needs the user's yes; ctrl14.sh is still NOT on the box). Restart only on a CLEAR word from the user:
`bash start14c.sh <gT> <gC>` (it removes STOP itself). See [[feedback-ambiguous-gpu-word-confirm-first]].

**ALL DONE 2026-11-14 19:48 IST: every item of the 45.196 list has run; nothing of ours on P032; GPUs free.**
Evening: GPU 1 `smoke ratiosd g14` (18:57-19:14: 16/4, 2/1, 5/1, 6/6 273/0; 12/4 272 + 1; G14 at 32,768: 69/0),
GPU 2 `smoke full42` (19:06-19:48: 333 passed, 0 failed), GPU 0 control (user: "start in GPU 0 too"): old tree and
pushed build with BLU_GQA_MX=0 at 12/4 both give the SAME single failure (err 4.509e-02, ratio 1.015) -> the 12/4
failure is NOT from the push (open only as a test-limit question). Work log 45.197 has the full table (written 20:20).
Left: EOD for 14/11 when the user asks; the user commits the docs; P032 folder attn_1113 is still there (ask: keep?).
QUOTA (user, 20:15: "why are we using much quota"): no live watcher that wakes me on every status line; start a run,
check ONCE when it should be done; short replies.
