---
name: feedback_reports_state_reasons_not_mistakes
description: "In benchmark reports for others, never frame a setup detail as \"my mistake\" — state the reason in one plain line"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-21T14:09:37.775Z
---

In reports meant for colleagues or leads (benchmarks, results), do not write "this is a mistake in how I set up X" or similar
self-blame. Just state the reason plainly: "the gap is there because the two runs validate on slightly different val chats
(the loader drops the last incomplete micro-batch: ...)".

**Why:** 2026-10-21, the SFT benchmark report for Kathirvel: I wrote the val-set difference of test 1 as "a mistake in how I set
up test 1"; the user: "it's a benchmarks thing, not to show our mistakes, just a line showing it's because of that is enough".
In chat with the user, owning a mistake is still fine.

**How to apply:** in any report/doc for others, explain causes neutrally (what happened and why), no "mistake" wording.
Related: [[feedback_reports_in_my_voice_gdoc]].
