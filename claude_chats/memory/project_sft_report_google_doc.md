---
name: project_sft_report_google_doc
description: "SFT report master is now the user's Google Doc (PDF export 2026-10-05, 36 pages); local .md/.docx from 09-28 are stale; edits go as Gemini instructions"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-05T06:31:47.736Z
---

**Where the SFT report lives now:** the master is the user's Google Doc. They export it to `work_docs*/sft_report.pdf` and `~/Downloads/sft_report.pdf`. The latest export is 2026-10-05 11:02: 36 pages, PDF producer "Google Docs Renderer".

The local `profiling_reports/sft_report_2026_09_28/sft_report.md` and `~/Downloads/sft_report.docx` (09-28) are **stale**. The user edited the doc in Google Docs after uploading it (the title changed, for example). Never rebuild from them.

**How to edit it:** I have no Google Docs access. When the user wants changes, give exact instructions for Gemini in Google Docs. That is how 2026-10-05 was done: `profiling_reports/sft_report_2026_10_05/gemini_edit_instructions.md`. Each edit gives:
- the page and line in the latest PDF;
- the Ctrl+F anchor text;
- for number fixes, Ctrl+H steps;
- for new text, the text in fenced code blocks.

Use code blocks because the IDE's markdownlint auto-fix rewrites bare URLs as `<…>` and reformats tables outside fences. Mention "Paste from Markdown" as the manual route.

**Done 2026-10-05:**
- The Zephyr number was wrong in 4 places. 7.34 is the final model (Table 1), not the matched DPO-only comparison: the fair pair is 7.00 vs 4.76, and AlpacaEval 86.07% vs 30.76% (Table 3).
- Added a new §1.8 with the 3 papers in detail, a §3 note on the libraries, §3.2 template paragraphs, and a Sources line.
- Checked by a 5-agent adversarial pass (112 claims, 28 fixed).
- Facts learned:
  - Chu's rewards are 5 / −1 / −2 / −3 (never 0), and RL without SFT drove the success rate down to ~0.
  - DeepSeek-R1 v2 Appendix G.1: RL-from-base failed on its 7B and 16B bases. This strongly supports SFT-first for our 1.5B base.

**How to apply:**
- Before any report edit, get the latest PDF export (or a fresh .docx download) and anchor against it.
- Report style rules stay as in [[feedback-reports-in-my-voice-gdoc]].

**Update 2026-10-05 16:22:**
- The user exported the Google Doc as Markdown. The master copy is now `~/Downloads/sft_research/sft_report.md`, mirrored in `work_docs*/sft_report.md`.
- Applied so far: 7.34→7.00 in 4 places (the user did this in Google Docs), and AlpacaEval 90.60→86.07 in 2 places (I did this in the md; the user copied it into Google Docs).
- Not applied, and only on request: C2, the D-H edits, the code-box/heading cleanup.
- `~/Downloads/sft_report.md` (16:03) is a stale duplicate that still has 90.60.

**Update 2026-10-15: NEW clean markdown copy = `work_docs/post_training_research_sft.md`** (+ `post_training_research_sft_images/`, 5 PNGs).
- Google Doc file name = "PostTraining - Research and Planning", tab "SFT : what it is". Link: https://docs.google.com/document/d/1nJ4utik9Jm7RT9YuaWD8UXcPpJRAj2WUrXnEdjJ_jlg/edit?tab=t.rl6mp65w3cm3 (shared by link).
- I can READ it without a connector: `export?format=txt&tab=...` and `export?format=md&tab=...` via curl (md export embeds images as base64, escapes everything, leaves `|` in table cells unescaped, and drops headings that are only bold text, e.g. 1.6).
- Built from the live export + 29 fixes listed in `sft_work_documentation.md` §45.2; verified word by word against the txt export (no word lost), tables, fences, all 60 links (52 arXiv titles via the arXiv API, 8 others HTTP 200).
- **Gemini in Google Docs is unreliable for edits:** rewrote paragraphs, invented token ids, rewrote the title/intro unasked. The user now updates the Google Doc BY HAND from the md. Never route edits through Gemini again; give FIND/REPLACE text for manual paste.
- The old `~/Downloads/sft_research/sft_report.md` (10-05) is stale.
