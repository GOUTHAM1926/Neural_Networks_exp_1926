---
name: feedback_simple_english_no_blabber
description: "ALWAYS write in simple plain English — no fancy words, no long build-ups, no blabbering. Applies to EVERY response, every time."
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-23T10:04:06.099Z
---

Write every response in simple, plain English. Short sentences. Plain words.

Banned: fancy or dramatic phrasing, long wind-ups before the point, restating the
same thing in nicer words, dressing up a small finding as a big one, "here's the
thing" style build-ups, and any word where a common one works.

**Why:** the user has told me this many times and I keep slipping back into fancy
writing. It wastes their time and hides the actual answer.

**How to apply:** say the answer first, in the plainest words. Cut every sentence
that does not carry a fact. Do not build drama around a finding — just state what
it is and how big it is. This applies to EVERY response, not just reports.

Also: answer the question that was ASKED. Do not drift to a nearby topic I find
more interesting and report on that instead.

Pairs with [[feedback_proactive_status_reports]] (stats + status every time) and
[[feedback_eod_report_format]].

**2026-09-30:** when the user asks "explain", they do not know the topic yet. Explain ONE thing at a time, from zero, step by step in full sentences, with one concrete example followed all the way through. Do not answer with summary tables or several topics at once; the user was angry that tables and 4 topics in one reply were unreadable. Stop after the one topic and check before moving on.

**2026-10-23 (repeated, user annoyed: "whenever I ask you to explain, that means I don't know that"):** never use a
symbol (k, k+1, n, x) or a term (micro-batch, step k) that has not been defined from zero in the same answer. Build up
from the raw objects (rows -> micro-batches -> steps), write out real numbers row by row (step 0 = micro-batches
0,1,2,3,4, ...) BEFORE any general rule, and only then state the rule. Great depth, simple words, no analogies.

**2026-10-05 — do ONLY the change named in the latest message.** I had proposed a 4-step plan (copy, cleanup,
all remaining edits, verify); the user said "ok modify in this md file" meaning ONLY the AlpacaEval fix they were
discussing. I started bundling all my proposals and got stopped: "dont apply all those remaining edits, just do
what i said, just that alpacaeval". A generic "ok modify/proceed" approves the specific thing under discussion,
not my earlier multi-step proposal. When unsure which items are approved, do the one in focus and ask about the rest.
