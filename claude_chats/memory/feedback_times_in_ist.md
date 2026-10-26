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
