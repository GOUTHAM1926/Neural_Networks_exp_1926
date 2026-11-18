---
name: feedback_keep_sft_package_docs_updated
description: "User rule 2026-10-30: every SFT change also updates the SFT package docs (sft/README.md + sft/docs/), the SFT grid plan and the work doc, in detail and simple English, incl. effects on pretraining"
metadata:
  type: feedback
---

User, 2026-10-30: "document this in our plan sft_grid search, our sft package too, our work documentation and all
related docs, and keep on updating our sft package files as we keep on modifying sft related files, in detail in simple
English ... continuing those effects on pretraining or other things".

**Why:** the SFT package in the repo is what other teams read (DPO team, data team, Adhitya); it had fallen behind the
code (README still said "chats longer than T are dropped, future task" and GPT-2 end token).

**How to apply:** with every change to an SFT file, update in the same round:
- `sft/README.md` (the relevant section + "what is where" + tests + limits);
- the matching `sft/docs/*.md`, then copy it to `work_docs/` (the two copies must stay md5-equal);
- `plan_grid_search_sft.md` (Q&A record, future experiments);
- `sft_work_documentation.md`.

Say what the change means for pretraining and other users (e.g. the loss-sum fix also hit pretraining). Simple
English, no analogies ([[feedback_simple_english_no_blabber]]).
