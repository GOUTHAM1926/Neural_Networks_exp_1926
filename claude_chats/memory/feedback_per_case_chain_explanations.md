---
name: feedback-per-case-chain-explanations
description: Explain numeric/dtype flows as a per-case step chain (one list per case, each step "rounding: yes/no" + file:line); NEVER mix two cases in one paragraph
metadata:
  type: feedback
---
The format the user understood "clearly, in detail" (2026-09-30): one list per case (e.g. fp32 table / bf16 table), every step of the chain on its own line, each saying what happens and whether it rounds, with the code line as proof, then a TOTAL line.

**Why:** a paragraph that mixed both cases ("sent as fp32" + "converted to bf16 before sending") made the user think I contradicted myself and got them angry; the per-case chain fixed it at once. Also: say "the F1 FIX is in place", never "F1 is still there" (read as the bug still existing).

**How to apply:** for any "where does precision/dtype change" question, write the chain per case, never interleave cases, and name exactly which case each sentence belongs to. Related: [[feedback-simple-english-no-blabber]].
