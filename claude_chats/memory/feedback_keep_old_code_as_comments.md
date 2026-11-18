---
name: feedback_keep_old_code_as_comments
description: "User 2026-11-05: NEVER delete or replace code silently -- keep the old code verbatim as comments next to the new code, and explain in comments there WHAT changed and WHY (dated, work-doc ref)"
metadata:
  type: feedback
---

User, 2026-11-05 15:45 IST (item 9, get_arch / attention Hopper check): "don't remove or delete code ok va, just
comment and tell what we did and why we did that optimization ... if you added something new, comment there too ...
don't remove anything ... comment and document in comments explaining things there itself".
I had replaced get_arch()'s body without keeping the old one, and shortened the old attention lines in a comment.
**Why:** the user and colleagues read the code to learn what changed; the old version must stay visible for review
and comparison.
**How to apply:** for every code change: keep the old lines VERBATIM as comments ("Old code (kept, not deleted):"),
put a dated block above saying what the code is for, how it was, what changed, why (numbers, sources), and short
notes on new lines; check with a script that every removed code line survives as a comment. Also, when the user asks
"explain", explain from zero, step by step, for someone with no background (see [[feedback_simple_english_no_blabber]]).
