---
name: don't tag every edit with author name
description: Only put "Gautam_Reddy_1926" attribution in comments when there's a TODO or research-followup item, not on every modification.
type: feedback
originSessionId: 037f68e7-08f8-4537-ba3a-080a79e5360a
---
Don't write `edits: Gautam_Reddy_1926` (or any author tag) on every code change. Only add the user's name when:
- The comment is a TODO / research-followup that the user owns
- The user explicitly asks for attribution on a specific block

**Why:** the user finds attribution-on-every-edit noisy and visually distracting in the diff. They want clean, technical comments — no signature footers per change.

**How to apply:** when writing new explanatory comments, default to NO author tag. If the comment is something like "TODO: research X" or "Review whether Y is still needed", that's when to mention them.

**2026-09-25 update:** the user asked for "edit done: gautam" on every change, then minutes later said to remove it: "no need of our name". So **no name tags at all** in code comments. What they DO want on every change: a comment saying what changed, what it was before, and why, with the evidence (file:line, test result).
