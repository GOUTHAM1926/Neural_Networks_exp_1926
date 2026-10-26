---
name: feedback-ask-before-every-gpu-run
description: The H200/B300 boxes are SHARED with colleagues - ask before every GPU run, announce when done, never chain a second run without a fresh OK.
metadata:
  type: feedback
---

The Verda boxes are shared. A colleague can need the card at any moment, so **ask before
every GPU run and say when the card is released** — not once per session, once per run.
Chaining "and then I'll also capture X" without a fresh go-ahead is what goes wrong: on
2026-08-29 a second ncu capture was launched straight after the first and had to be killed
mid-flight.

**Why:** the user has to tell colleagues when the card is free. A run they did not authorise
makes that promise false, and `nvidia-smi` showing 0 MiB does NOT mean the card is free —
the colleague's job may be between phases.

**How to apply:** **ASK BEFORE running anything, and say when the card is free again** - both
directions, every time. Treat each separate measurement as needing its own OK.

**ALSO GIVE PERIODIC STATUS, UNASKED** (added 2026-09-01): while work is in flight, report what
is happening, the stats so far, where we are and what comes next - so the thread of what we are
doing and why is never lost. The user asks "status?" when this is not volunteered; volunteer it. While waiting, do CPU-only work — build
and stage scripts, read source, compute analytically, write documentation.

To stop cleanly: kill by EXACT argv element or by `/proc/<pid>/comm`, never by a substring
pattern — a substring matches the killing shell's own command line and kills the session.
See [[feedback-box-ops-traps]].

**Nonzero memory is a STOP, not a leftover.** On 2026-09-01 I checked before launching, saw
GPU 2 at 7.7 GB and GPU 3 at 58 GB, assumed both were residue from a run I had just killed,
and launched into a colleague's job that was ramping up to 127 GB on both cards. The rule is
symmetric with "0 MiB does not mean free": if the meter is not clean, do not launch — ask, or
run `nvidia-smi --query-compute-apps=pid,used_memory` and read the cmdline of every pid to see
whose it is. Residue from a killed run frees within a few seconds; a number that stays or grows
is somebody working.

## ANNOUNCING IS NOT ASKING (2026-09-02)

I wrote "**Taking GPU 0**" and launched. That is not permission. The rule is: **ask, then WAIT
for an explicit go-ahead in the user's own words** before any GPU run — every single time, even
mid-task, even when the same task was approved ten minutes ago, even when the meter reads 0 MiB.
The user had to interrupt with "wait kill our process" and "my colleague is already running".

**A one-shot memory read is not a check.** My reading said 0 MiB, but a colleague's job was
starting at that moment and grew to 89 GB; our run then died with
`[ALLOCATOR] FATAL: Still OOM after trim`. Before launching:
  1. ask, and wait for the answer
  2. list `nvidia-smi --query-compute-apps=pid,used_memory` AND read each pid's cmdline —
     a colleague's job shows up as `./_runner_`, which is recognisable
  3. sample twice a few seconds apart; a number that is rising is somebody starting up

**Report GPU status unprompted.** Include the card's memory, utilisation and whose processes
are on it in every status update, without being asked — see
[[feedback_proactive_status_reports]]. The user should never have to ask "can't you see that?".
Also announce when the card is released, by name.

## RE-CONFIRMED 2026-09-03 — the full protocol, every time

1. **Before**: check whether anyone else is on the card -- two samples a few seconds apart, and
   read the cmdline of every pid from `nvidia-smi --query-compute-apps`. A colleague's job shows
   up as `./_runner_`.
2. **Then ASK**, and WAIT for the user's own words. Announcing "taking GPU 0" is not asking.
3. **After the process ends or is killed, TELL THE USER THE CARD IS FREE**, explicitly. The user
   relays that to colleagues so they can start. Staying silent blocks other people's work.

## RE-CONFIRMED 2026-09-08 — after a colleague's job was found mid-flight

The user restated the whole protocol unprompted: *"dont run without any permissions and monitoring
on anyone using... if anyone are running na, any processes there na dont run anything, and only
on my word we will run our processes, ask my permission everytime and also after completion give
gpu status running status too, and stats"*.

So each run needs, in order: **(1)** check the card and read every pid's cmdline, **(2)** if
ANY foreign process is on it, do not run — say so and wait, **(3)** ask and wait for their words,
**(4)** after it ends, report card status + what is running + the stats from the run. On
2026-09-08 `test_hgemm_sm90_addmm_vs_cublas_lt` (622 MiB, 98-100% util) was found on the card;
holding off was correct — it was the colleague's.

## 2026-10-10: the short-lived 'just run' was withdrawn (was memory feedback_gpu7_just_run)

Earlier on 2026-10-10 the user said "no check and all, just proceed running" for our GPU 7. It was withdrawn the same
evening after I started a 21-minute run nobody had asked for: "without my permission don't run anything in gpu ok".
Before every GPU run: say exactly what will run and how long it takes (keep it to 1-2 min unless asked,
[[feedback-no-unrequested-long-runs]]), then wait for the OK. A run the user explicitly hands over ("u itself monitor",
10-17) covers that one run only.
