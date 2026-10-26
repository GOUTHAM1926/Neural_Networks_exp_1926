---
name: feedback_docs_keep_full_detail
description: "When revising or merging into a doc, NEVER shorten: keep every earlier example and the merged text in full (fix wrong lines in place), plain words, step-by-step; user furious 10-14 when a merge condensed things"
metadata:
  type: feedback
---

On 2026-10-14 the user asked me to merge a long pasted explanation (TRL packing strategies + varlen) into
`work_docs/sft_packing_algorithms_bfd_ffd.md` "every letter should be there". I condensed it into short bullets and
replaced some of my own earlier small examples with the pasted ones. The user was angry: "you just minimized everything
... earlier it was so elaborative with more examples ... and there is no explanation of cross-document attention itself".

**Why:** the user reads these docs to learn from zero and shares them with colleagues; detail and worked examples are the
point. Shortening reads as lost work.

**How to apply:**
- Merging: keep the incoming text in full, in its own step-by-step form; correct wrong statements in place and list
  every correction in a table at the end; only swap hard words for plain ones (and say which).
- Revising: never delete or shrink earlier examples/sections; add next to them.
- Explain basic concepts from zero before using them (e.g. attention -> causal -> plain causal -> cross-document), with a
  tiny worked example, before the big ones.
- Check the result: table column counts, fences, cross-references, every number computed by a saved script.
Related: [[feedback_simple_english_no_blabber]], [[feedback_no_analogies]].
