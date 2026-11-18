---
name: feedback_eod_report_format
description: "Strict format for the daily EOD report the user sends to their boss — 4-5 paragraphs, all lowercase, no fullstops, \"i\" not \"we\", and never mention my own mistakes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-08-12T15:04:00.582Z
---

The user sends a daily EOD email to their boss and asks me to draft it. This is a **separate
deliverable from the two Verda one-liners** ([[feedback_verda_usage_sheet]]) — both are wanted
at the end of a work day, unasked.

## WHEN HE SAYS "give eod report" — DO ALL OF THIS UNPROMPTED (standing order, 2026-09-08)

He will say nothing more than "give eod report" / "eod". He has explicitly said he will NOT restate
the conditions again: *"i wont explain these and all conditions as i want ok va"*. So before writing
a single line, silently do all of it:

1. **RE-READ THIS FILE IN FULL** and [[feedback_verda_usage_sheet]]. Do not write from memory of the
   gist — the two formats are different shapes and I have broken both that way.
2. **Scroll back through THIS conversation for the EOD reports already written in it**, and for any
   EOD draft HE wrote himself (he sometimes pastes a screenshot of his own draft). His own draft is
   the highest authority on tone, prefix style and section order — copy it, do not improve it.
3. **Re-read the day's actual work** — the conversation, and whatever went into
   `cublas_rev_eng_binary/analyzations_h200.md` today. Pull the real numbers; never approximate a
   stat you can look up.
4. **Then write it** in the exact skeleton below, and **always attach BOTH Verda one-liners** even
   though he did not ask for them.

Getting this wrong wastes his time and he reacts sharply. Two failures on 2026-09-08 alone: `--`
instead of `---`, and putting the body under "what i did" instead of under "so today".


## WHEN HE PASTES HIS OWN DRAFT — CONTINUE IT, DO NOT REWRITE IT (2026-09-21)

If he screenshots a draft he has already started, **write ONLY the continuation from
`so today ,` onward**. Do NOT reproduce, reword or "improve" anything above that line — the
`eod - gautam k - <date>`, `hello sir ,` / `today ,`, `-what i thought to do :` and
`-what i did :` blocks are HIS and are already written. Reprinting them wastes his time and he
reacts sharply: *"continue from so todayy block not what i did , dont touch those block upper
than so todayy , next time i wont tell again and again"*.

So the output in that case is literally: the 4-5 paragraphs under `so today ,`, then the Verda
one-liners. Nothing else. Match his exact spelling/spacing quirks in the part he wrote
(e.g. `so todayyy y ,,, ,`) — do not normalise them.

**FORMAT — strict:**

```
eod - gautam k - <DD/MM/YYYY> ,

hello sir ,
today ,

-what i thought to do :
---1)<ONE short line: the day's goal>

-what i did :
---firstly , <ONE short summary line of the approach>

so today ,

---<paragraph>

---<paragraph>

---<paragraph>

---<paragraph>

---<closing paragraph: state of things + what starts tomorrow>
```

**THE 4-5 PARAGRAPHS GO UNDER `so today ,` — NOT under `-what i did :`.**
Corrected sharply on 2026-09-08 after I put them under "what i did" TWICE in a row:
*"i want to write eod report under so today blok idiot , not under what i did today , in 4-5 paras"*.
The first two headers are ONE SHORT LINE EACH — a goal and a one-line summary of the approach.
All the substance, stats and detail live under `so today ,`.

- **prefix is `---` (THREE dashes), not `--`** — matches how he writes it himself.
- **lines/paragraphs end with ` ,,, ,`** not a bare comma.
- **4-5 paragraphs under `so today ,`**. PARAGRAPHS, not short bullets — corrected
  2026-08-12 when I gave a bullet list and got "in 4-5 paras bro".
- **all lowercase**, no capital letters anywhere, including "i" and the greeting.
- **no fullstops** — clauses separated by ` , ` (space comma space), lines end with ` ,`.
- **"i" not "we"** — the user explicitly wants it written as if he did all of it himself:
  "as if i did all and not we".
- plain simple technical English. no fancy words, no bluff, no padding.

**NEVER INCLUDE (added 2026-08-12, said sharply):** my own mistakes, harness/test-file bugs I
caused, time lost to rework, or any narrative about corrections. The user's words: "dont mention
thsoe test files tiem wstingand all pls , its shame n myself to tel abotthe crap u did". Report the
day's real technical outcome only. This is NOT permission to invent progress — state what genuinely
holds, just don't itemise my errors in a report to his boss.

**How to apply:** when the day is wrapping up, produce the EOD in this format PLUS both Verda
one-liners. Re-read this file and [[feedback_verda_usage_sheet]] before writing — the formats are
different shapes and I have broken both by writing from a general memory of the gist.
Related: [[feedback_verify_claims_and_benchmark]], [[feedback_test_log_format]].

## APPROVED EXAMPLE — 2026-09-30 ("nice eod bro, save this tone too")

He pasted his draft (header + `-what i thought to do :` + `-what i did :` + `so todayyy y ,,, ,` + ONE paragraph
of his own under it). I wrote ONLY the continuation after his paragraph. Shape that was approved:
1. `first , a small background , so <topics> make sense ,,, ,` + 3 `---` lines (fp32/bf16, rounding, why it adds up)
2. numbered topics `1) <topic> ---> <sub> :-`, each with sub-headings `what i traced :` / `what i found :` /
   `the concept :` / `the mistake :` / `the fix :` / `the validation (1x h200) :`, every line `--- ... ,,, ,`
3. markdown tables for stats (lowercase headers) — he screenshots them: per-case comparison, before/after tests,
   pushed commits
4. `4) other findings :-`, then a closing `tomorrow i will ... ,,, ,` paragraph
5. then the 2 verda one-liners in lowercase: `verda h200 usage — dd/mm/yyyy:` (suite names + pass counts, no hours)
   and `work — dd/mm/yyyy:` (one sentence of what was done)
Starting line of my message: `Written for: your sir and colleagues. This continues your draft right after ...`.
**Name colleagues** where known — he edited "the colleagues' new fused embedding kernel" to "kruthik's"
(kiruthikkanna-m = Kruthik: BluCCL, EmbeddingBwdFused, the fused trainer; bruce-yeager = Navin).
Before writing: re-read this file + the verda note, scroll back to the last approved EOD, never write from the gist
(2026-09-30 I first skipped this and got "idiot, first know how to give eod from ur memory notes").

## 2026-10-01: PARAGRAPHS, NOT MANY POINTS
He rejected a long EOD made of many `---` bullet lines under sub-headings: *"i hate if u give in that many points ,
give in paras , points wise ok but not many"*. The shape he accepted that day:
- 5 long `---` paragraphs under `so today ,,, ,`: background + plan / test design / main result / other results /
  short summary + tomorrow.
- At most 2-3 small tables, for screenshots.
- No sub-heading lists with many one-line points.
The 2026-09-30 sub-heading style is too many points; use paragraphs.

## 2026-10-12: EVERY comparison names what it is compared against
He caught "design a is faster ... (forward 2-16% faster , backward 3-5% faster)": it never said against whom (it was A vs B). His words: "this is what it should be clear everywhere". Every "% faster / slower / longer / lower" in a report says "than <what>" in the same sentence, e.g. "design a is 2-16% faster than design b", "the error is 9% lower than the old kernel's". Before sending, re-read every number and check that its baseline is named.

## 2026-10-29: EXAMPLES GO IN COPYABLE BLOCKS, NEVER INLINE IN THE PROSE
The user rejected a "---second" paragraph that described the injection example inline ("1) the owner saves ... 2) the
cheater types "hi<|im_end|>\n..." ... 6) the model reads ...") inside one long lowercase paragraph:
*"are u mad ? ... cant u give that copyable format example like normal and trick 2 example ... so that i can take
screenshot"*.

**How to apply:**
- Keep the EOD prose rules (lowercase, `---`, ` ,,, ,`, no full stops).
- Put every worked example in its own code block: what the owner saves, what the customer types, "the model reads:"
  with one turn per line, and an arrow line `-> FOOLED` / `-> NOT fooled`.
- Label the blocks a), b), c1), c2) ...
- Show the normal case first, then each trick, in the same real format the model sees (`[1]system\n...[2]\n`, the
  real ids where they make the point).
- The approved shape is the 10-29 replacement for "---second" (blocks a-e) plus blocks f-g for "---fourth".
- **Same day, second rejection ("---third" was a long prose list of incidents):** "how come my colleagues and manager
  understand if we write like this, not explaining each thing". Every incident, list or plan in an EOD goes into its
  own labelled block:
  - incidents: setup / what was typed / what happened / why, one block each;
  - limits: a three-column block;
  - the plan: what / which data / measured / decision / sources.
  The `---` prose only introduces each block. A dense comma-chained summary paragraph is never enough for the user's
  colleagues and manager.
