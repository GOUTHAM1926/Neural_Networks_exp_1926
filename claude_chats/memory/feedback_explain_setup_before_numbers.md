---
name: feedback_explain_setup_before_numbers
description: "Before any table or number, explain the setup, every column and every term; answer history questions in time order with each team's intention; one topic at a time"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T13:02:09.812Z
---

When I show a table, a number or a demo, I first say:
- what was run (setup: model, data, what was measured, where);
- what each column means;
- what each id or term is (e.g. "id 2 = im_end, the end token");
- why the result came out that way.

For "how did X evolve" questions, answer in time order: what each team did, when, and the intention they stated. Do
not list everything at once.

**Why:** on 2026-10-29 the user could not follow my 10-28 answer:
- part 5 (injection) had no step-by-step mechanism, no cases and no examples;
- part 9 (a loss/chance table) never said what the setup was, what "id 2" was, or what loss and chance meant.

The user: "u r telling everything without explaining in detail what it is, and why it is and how it is". They asked for
a chronological, cause-and-effect order.

**Repeated 2026-10-29 evening:** the user still could not follow injection after 4.5 and 4.17 ("how many times I should
have asked"). They wanted it as if they knew nothing: every word defined, one concrete scenario (app, developer, system
message, user), the exact ids at each step, what happened before ChatML and what happens now, and exactly how the 2 ids
fix it (and what they do not fix). §4.24 of the plan file is the model answer.

**How to apply:**
- Each concept gets: the words → the setup → the steps → examples with real ids → which cases → what it means for us.
- For a mechanism the user has never seen: assume zero background; define every term before using it; one running
  example from start to end; show the ids at every step; state what is still unsolved.
- This is still simple English, with no analogies ([[feedback_no_analogies]]) and no filler ([[feedback_simple_english_no_blabber]]); detail is not padding.
- Related: [[feedback_per_case_chain_explanations]].

**Repeated 2026-11-02 21:10 (tile-map analysis):** I gave tables of "Test 1 / 2 / 3" with only a one-line description
each. The user, angry: "i didnt even understand what those 3 tests and all that we did ... explain in simple
understandable english elaboratively, as to someone who doesn't know anything or no context". For EVERY test or
experiment: why it was needed, exactly what the program does step by step, what is measured and how, what result was
expected in each case, what the result was, what it proves and what it does not. Define CPU / GPU / queue / wait /
tile / nsys etc. before using them. A short summary written for me is not an explanation for the user.
