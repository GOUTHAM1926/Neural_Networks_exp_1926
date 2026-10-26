---
name: feedback-lock-clocks-before-timing
description: Free-clock is the DEFAULT measurement regime; locked clocks only when a question specifically needs equal-clock comparison
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-16T13:14:19.163Z
---

**Free clocks are the default. Run locked ONLY when the question specifically needs an
equal-clock comparison, and when you do run locked, run free alongside it.**

**Why:** production never locks clocks. A locked-clock win that disappears at free clocks is
not a win the trainer ever sees. Locked answers "who is faster at equal clock" -- a diagnostic
question, not the shipping question.

**How to apply:**
  * default every A/B and every benchmark to free clocks
  * if a locked run is genuinely needed (isolating a power/clock effect from an instruction
    effect), do BOTH and report the two tables side by side -- they can disagree
  * when locking, still CHECK the lock actually held UNDER SUSTAINED LOAD, not at idle
    (`clocks.applications.graphics` at idle proves nothing -- that mistake was made once)
  * a box change voids all prior absolute timings either way -- see [[feedback-gpu-pin-device-1]]

**Re-stated by the user 2026-10-16 (attention work):** every stat I report and every keep / revert
verdict must be at FREE clocks (real training conditions). ncu's locked-clock cycles are only for
diagnosis or a side comparison, never the headline or the deciding number. When I give "the stats",
lead with free-clock times (same session for us vs rivals), and say when a number is locked-clock.

Supersedes the earlier "LOCK CLOCKS before any timing" instruction, which had the default
backwards. The free-clock gap being measurement ORDER (cold card measured first) is still true
and is handled by ABBA crossover, not by locking -- see [[feedback-paired-null-instrument]].
