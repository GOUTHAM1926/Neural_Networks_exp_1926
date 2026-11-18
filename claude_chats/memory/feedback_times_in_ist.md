---
name: feedback_times_in_ist
description: Write every time in the user's local time (IST, the laptop clock); the Verda 2x H200 box clock is UTC (IST - 5:30), convert it; scripts print IST via TZ=Asia/Kolkata
metadata:
  type: feedback
---

2026-10-16 the user: "use my timings, not the timings in another country". I had copied the 2x H200 box's clock (UTC,
Finland, `ripe-song-lies-fin-02`) into the work doc and chat ("06:49-06:51" = 12:19-12:21 IST).

**Why:** the user reads all notes and EODs in India time; mixed clocks make the record wrong.

**How to apply:** every time in chat, docs, memory and EOD = IST (laptop `date`). Box timestamps: the 2x H200 box is UTC
-> add 5:30; the old 2-node box (103.42.51.252) already shows IST (checked against the 10-15 transcript). Every new box
script: `say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" ...; }` and `TZ=Asia/Kolkata nvidia-smi ...` for
logged timestamps.

**Broken again 2026-11-05 (pretraining check on the 2x box 86.38.238.51):** I wrote every status in the box's UTC
("08:26 UTC") and the user had to ask: "say in this time I am following ... foreign time". From now on: box scripts
log with `TZ=Asia/Kolkata date`, and every time in chat, docs and reports is IST (convert box UTC +5:30 before
writing). Same day: the user interrupted two of my multi-minute blocking waits (`for ... sleep` loops over ssh) with
"status?" -- keep each check short (seconds), answer status at once, never block the chat for minutes; let the box
scripts run on their own and look in briefly.
