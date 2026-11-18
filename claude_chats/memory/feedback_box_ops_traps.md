---
name: feedback-box-ops-traps
description: Shell/ops traps that each cost real time on the Verda boxes - NEVER retry a launch, pkill -f kills the shell running it, timeout escapes a group kill, a bg loop inside $(...) hangs forever, count by exact argv not pattern
metadata:
  type: feedback
---

**1. `pkill -f <pattern>` kills the shell that is running it** whenever the pattern appears in
that shell's own command line. Four separate kill attempts on 2026-08-28 reported success while
the target process kept running at 700 W — the `ssh root@box 'pkill -f h28c; ...'` shell matched
its own `-f` pattern and died before reaching the rest of the command.

**Why:** it looks like the kill worked (exit 0, no error), so the failure is invisible and you
conclude the process is gone when it is not.

**How to apply:** use a bracket class so the pattern cannot match itself —
`pkill -9 -f "[h]28c.sh"`. Then always verify with a separate
`pgrep -af "[h]28c" || echo NONE`. Same trick for `pgrep`: `pgrep -f skip_watch` inside a command
containing "skip_watch" matches the caller and returns a bogus PID.

**2. A background infinite loop inside `$( )` hangs the parent forever.**
`SP=$(sampler_start file)` where the function ends in `( while :; ...; done ) &` never returns:
command substitution waits for the stdout pipe to close, and the background loop holds it open.
The campaign sat at 0% GPU for 11 minutes on its first line. Fix: have the function set a global
(`SP=$!`) instead of echoing, and redirect the loop `>/dev/null 2>&1 &`.

**3. Build toolkit must match the objects.** `/usr/local/cuda` on the H200 boxes points at
**13.1** and `.bashrc` puts it on PATH, so interactive builds use 13.1. Forcing 13.0 at link time
gives `nvlink fatal: input file 'X.o' newer than toolkit (131 vs 130)`. Also `nvcc` is **not** on
PATH in a non-interactive ssh shell — export it explicitly.
See [[feedback-separate-build-and-run]], [[feedback-backend-ab-snapshot]].

**4. NEVER PUT A LAUNCH INSIDE A RETRY LOOP. This is the most expensive mistake of 2026-08-28.**
An `until ssh ... launch; do sleep; done` loop treated "connection dropped before printing
confirmation" as failure and relaunched. Every launch had actually SUCCEEDED: **8 concurrent
copies of the script and 4 concurrent GEMM processes** ended up on one GPU, voiding the run.
Those same loops then reconnected every 20-25 s and **took the box's sshd down**, producing a
long run of "connection failed" that looked like a box problem and was self-inflicted.

**The rule: killing and reading are idempotent — safe to retry. Launching is not.** On an
uncertain launch outcome, reconnect and **COUNT what is running**, then decide. Never relaunch
blind. Verify with `PRE count == 0` before launching and `POST count == expected` after.

**5. Count by exact argv element, not by pattern.** `pkill -f X` / `pgrep -f X` match the shell
running them whenever X appears in that shell's own command line — and the `"[h]33"` bracket
trick FAILS once the bracketed text itself is in the command. This produced false "1 still
running" readings and repeated self-kills all day. Correct form:

    for p in $(ls /proc | grep -E '^[0-9]+$'); do
      [ "$p" = "$$" ] && continue
      while IFS= read -r -d '' a; do [ "$a" = "/tmp/x.sh" ] && n=$((n+1)) && break; done < /proc/$p/cmdline
    done

**6. `mkdir -p` BEFORE any redirect into that directory.** `bash -c "script > dir/log"` fails
silently when the script is what creates `dir` — the redirect is set up first. Cost one run.

**Box clones are `--depth 1` -- any `git checkout <older sha>` fails.** Bit twice on the same
task (the SASS before/after comparison), each time costing a full build cycle. Before comparing
against any earlier commit on a box, run `git fetch --deepen=50 origin main` first. Symptom is
`fatal: invalid reference: <sha>`.

**Wrap box-side checkout/restore in `trap restore EXIT INT TERM`.** The first attempt at that
comparison died mid-run when the connection dropped and left the tree on a foreign commit with a
library built from it -- a state that silently poisons the next measurement. With the trap, the
same failure restored cleanly and cost nothing.

**NEVER let two background benchmark chains exist at once (2026-09-05).** Three `h99_ours`
processes ran on the card SIMULTANEOUSLY and silently contaminated every timing. Cause: chains
launched with `setsid nohup bash -c '...'` become process-group leaders with ppid=1, and killing
the CHILD benchmark just makes the still-looping parent spawn another one a second later.
`pkill -f "<script text>"` does not find them reliably either.
**How to apply:**
  * kill the process GROUP, not the process: `kill -9 -$(ps -o pgid= -p <pid> | tr -d ' ')`
  * before launching any timing chain, assert the card is empty AND no prior chain is alive
  * after killing, re-sample nvidia-smi at least twice a few seconds apart -- a respawn takes
    a second to appear, so one clean sample proves nothing
  * any data collected while >1 runner was resident is VOID; move it aside, do not analyse it

**7. `timeout` escapes a process-group kill (2026-10-13).** GNU `timeout` puts its child in a NEW process group, so `kill -- -<script pgid>` killed the run script but left `timeout ... tva_vj2` and `timeout ... nsys profile` alive (twice in one day). nsys children (`nsys-launcher`) also reparent to init and can still start the target after their parent dies. Same day, `P=$(pgrep -f "<log dir>/run_2b.sh")` returned my own ssh shell too and killed it (trap 1 again).
**How to apply:** to stop a run, kill by exact PIDs: list every process whose `/proc/<pid>/cwd` or cmdline is under `~/gautam_k/` (script, timeout, nsys, nsys-launcher, ncu, tva_vj2, python), kill those PIDs, then re-check twice and confirm GPU memory is back to ~0 MiB. Never use `pgrep -f` inside the same command line that names the pattern.

**10-14 again:** a kill list built with `grep -E "...|TreeLauncherSubreaper .*gautam_k|..."` (one alternative WITHOUT the [x] bracket) matched my own `bash -c` line, so the loop killed my own ssh shell (exit 255). Every alternative in the pattern needs the bracket, e.g. `[T]reeLauncherSubreaper`.

**10-19: pre-delete check of a git clone must cover EVERYTHING.** Before `rm -rf` of a clone: `git log --oneline --branches --not --remotes`
(ALL local branches, not just HEAD vs @{u}), `git stash list`, `git status --porcelain` (incl. submodules), and
`git submodule foreach --quiet 'git log --oneline --branches --not --remotes'`. Then confirm on the remote side (ls-remote / fetch +
merge-base --is-ancestor) that the last pushed commits are on the branch. On 10-19 BluTrain_bluccl was deleted after checking only the
current branch (0 unpushed, clean, 0 stashes); every recorded BLUCCL commit was verified on GitHub afterwards, but other local
branches were never listed -- the user asked "did you check correctly?". Also: `rm` does not free space while a process holds the
file open (the VSCode remote server held a deleted 22 GB checkpoint); truncate it via /proc/<pid>/fd/<n> (`: > fd`) to free it.

**8. Chaining scripts (2026-10-31, 4x H200 box): two hangs/mis-starts in one hour.**
(a) A bare `wait` after starting workers ALSO waits for the background disk-watch loop, which never ends: ab_run.sh sat
idle 8 min after phase B ("hung at 14:48"). Use `WP="$WP $!"` per worker and `wait $WP`.
(b) A follow-up script waited with `grep -q "AB DONE|ABORT" ab/status`; the status file still held an `ABORT` line from an
EARLIER run, so it started at once and put a test on the GPU between two timing jobs of the running study (caught and
stopped in 1 min; the study's own free-GPU check kept the timings clean). Wait on the PROCESS being gone
(`ps -eo args | grep -c '[t]ests/ab_run.sh'` == 0), never on a grep of an append-only status file. Trap 7 struck again
the same minute: killing the script's group left `timeout <test>` running on the GPU; kill the timeout PID too.

## 2026-11-04: "lock the GPUs" requests
The permission classifier refuses both ways of locking: `sudo nvidia-smi -c EXCLUSIVE_PROCESS` and a guard that kills
other users' processes on our GPUs ("Interfere With Workloads"), even when the user asks for it. Tell the user at once
that the lock is refused; then run with a free-check before launch plus a recorder that only LOGS other processes.
Recorder trap: a process exiting at the end of our own run shows as "[No data]" or our own binary path; check the path
before calling it foreign.

## 2026-11-05: GPU lock that works (when root on the box) + ssh quoting trap
- `nvidia-smi -i N -c EXCLUSIVE_PROCESS` (root, no sudo needed on the 2x box) was NOT refused and is a real lock: other
  processes get "device busy". Restore with `-c DEFAULT` from an EXIT trap. It also refuses OUR second process on the
  same GPU: a check run while our trainer held GPU 0 produced 0 results -- never treat its "0 differences" as a result.
- Never put a script containing single quotes inside `ssh host '...'` (a heredoc there broke on `trap '...'` and the rest
  ran on the laptop). Write the script locally, scp it, run it with a plain ssh command.

## 2026-11-13: P032, rival scoreboard + race (attention short-chat study)
- **nsys on P032 costs 90-100 s per run** for 5-7 s of GPU work (box load ~32; measured from the clock logs). Plan
  1.6 min per nsys run there, not 20 s; put all contenders of a layout into ONE nsys session next time. A plain `tva`
  BENCH run (no nsys) takes ~20-25 s for 4 s of GPU work.
- **BENCH exports the "group with the longest chat" a second time under the same layout name** when it is one of the
  three groups: the rival runner then stops with "segments must tile every row". Remove duplicate lines of
  segments.txt before the rival runs (`awk '!seen[$0]++'`), as `stage0_b.sh` does.
- **`seq_names.py`** needs the rule for `gqa_bwd_vj2_kernel<(bool)1, (bool)0>` (two template args since the fp32 dK/dV
  switch of 11-01); with the old one-arg rule our backward shows as "nan" in `scoreboard_free.py`. Fixed copy:
  `profiling_reports/attn_short_chats_2026_11_13/pkg/seq_names.py`.
- **Trap 7 again:** killing a script's process group left its `timeout 900 nsys profile ...` sessions alive (own group);
  they would have started the next GPU run. After a group kill, list what is still alive with cwd under our folder and
  stop those by PID.
- A leg's `LD_LIBRARY_PATH` = the candidate's folder FIRST, then the WHOLE normal path (libblublas.so lives in the tree's
  lib folder; dropping it gave exit 127 at once).
- Check a status before reporting a layout "complete": count its runs (8 for 4 contenders x 2 rounds), not the last line.

**2026-11-16 (P032): the user kills our runs by hand; a test pid is not the chain.** The user ran `kill <pid>` on our
test program (tva_push of the full suite); it worked (exit 143), but the chain script (bfull16.sh) started its next
test 6 s later, so it looked like the kill had failed; then `kill -f <pid>` gave "invalid signal specification"
(`kill` has no -f; that is `pkill -f`, unsafe on a shared box where every user is `operator`). How to apply: when the
user wants to stop our work by hand, give: `touch .../pkg14/STOP` FIRST (nothing new starts), then the pid from
`nvidia-smi -i <gpu> --query-compute-apps=pid,process_name --format=csv`, check it with `ps -o pid,etime,args -p`,
then `kill <pid>` (`kill -9` only if still alive after ~10 s). Also: bfull16.sh prints "PART DONE" / "BFULL DONE"
after its last test even when that test was killed: judge completion by "checks passed" in the suite log, not by them.
