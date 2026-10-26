---
name: feedback-proactive-status-reports
description: Give a full 360-degree status report — stats, where we are, where we are heading — periodically and unprompted, never only when asked.
metadata:
  type: feedback
---

The user wants running status without having to ask for it, and has asked for this more than
once ("time to time u should give me status on whats happening, whats the status, what are the
stats and where we are and where we are heading to so that we wont get deviated", and again on
2026-09-02: "everytime i cant say that to u, u should remind and do it urself").

**Why:** long GPU campaigns drift. Without a periodic recap of the numbers and the plan, work
wanders off the thing that was actually asked for — which has already happened once, when a
short targeted sweep turned into a multi-stage measurement campaign nobody wanted.

**How to apply:** give a status block unprompted at every natural checkpoint — a run finishing,
a long run passing its midpoint, a phase completing, or before starting anything new. Cover all
of:
- **stats** — the actual numbers, with their basis (steps averaged, locked or free clock, which
  box, which tree). Never quote a peak step as if it were the result.
- **status** — what is running right now, what finished, what is blocked and on what.
- **where we are** — the number against its comparison point, and whether it is believed yet.
- **where we are heading** — the next step and the one after, so scope drift is visible early.
Also state which artefacts have been pulled local ([[feedback_pull_run_artifacts_immediately]]).
Keep it dense and factual — tables over prose, no padding.

## REINFORCED 2026-09-03 — two additions the user asked for explicitly

**Say whether we are DIVERGING from the plan.** Every status block must state where we are in the
agreed plan and whether the current work is on it or has drifted off it. The user's words: "where
we are, where we are heading to, and whether we are diverting or not from our path". This is the
whole point of the report -- it exists to catch drift early, which has already happened once when
a short targeted sweep turned into a multi-stage campaign nobody asked for.

**Always include GPU stats in the status**, unprompted: memory used, utilisation, and WHOSE
processes are on the card (read each pid's cmdline, do not just report the number).

**Announce when the card is released, by name and immediately** -- the user forwards that to
colleagues so they can start their own work. A run that finished without me saying the GPU is free
leaves other people blocked for no reason.

**Document as we go, not only at the end.** Numbered sections in analyzations_h200.md at each
milestone, plus memory for anything that would otherwise have to be rediscovered.

## REINFORCED AGAIN 2026-09-10 — THIRD time he has had to ask

*"u should give me status time to time bro? note this too in ur memory notes and dont forget and
do it properly time to time ok va"* -- said after I went quiet during a long build and only
reported when he asked "build completed aah?" twice.

**The failure mode is silence during long-running work.** A build or a run that takes 20-40
minutes is exactly when he needs updates, and it is exactly when I stop talking because I am
"waiting". Waiting is not a reason to go silent.

**Concrete rule: while anything of mine is running on the box, report roughly every 5-10 minutes,
unprompted, even when nothing has changed.** "Still compiling, 90 objects, 0 errors, 12 min in,
4 compilers left, load 1.4" is a valid and wanted update. Do not wait for a state change.

**Mechanism that works:** arm a Monitor with a periodic heartbeat (not only change-triggered) so
a notification arrives on a timer, and report each time one lands. A change-only filter goes
silent for 20 minutes during a slow tail -- which is precisely the case that annoyed him.

Never make him ask "is it done yet". If he asks, I have already failed the rule.

## ALWAYS REPORT TIMES IN IST (added 2026-09-10)

The Verda boxes run **UTC** (`Etc/UTC`); the user is in **IST (UTC+5:30)**. Quoting raw box time
confused him -- I said "05:47" when his clock read 11:18.

**Rule: every timestamp shown to the user is IST.** Convert with `TZ=Asia/Kolkata date` on the box,
or add 5:30 to box time. In scripts that log timestamps for him to read, emit
`TZ=Asia/Kolkata date +%H:%M:%S`. If a raw UTC time from a log must be quoted, give the IST
equivalent beside it. The local workstation is already IST, so `date` here is safe -- it is only
the box that needs converting.
