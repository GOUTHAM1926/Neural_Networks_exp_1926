---
name: feedback-reports-in-my-voice-gdoc
description: "Reports for the team lead must read as if the user wrote them: first person \"I\" (never \"we\"), simple Google-Doc style, editable .docx, no footer/name/date/report furniture"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-28T04:05:33.200Z
---

Write reports and write-ups (for the team lead or colleagues) **as if the user wrote them himself**:
- first person "I did / I plan"; never "we/us/our", except inside quotes from papers;
- plain Google-Doc style: a title, simple headings, paragraphs, a few tables;
- no page footer, no name/date/team line, no "how to read this" section, no note boxes, no "checked" tags on references;
- deliver an **editable .docx** (it opens in Google Docs), not a PDF. PDFs cannot be edited.

Keep the depth, and keep a source next to every number; those were liked ("content is nice").

**Why:** on 2026-09-28 the user rejected a 42-page reportlab PDF of the SFT report. It was "too much", did "not look like a human wrote it", and had footers with his name and date everywhere. He also wanted to edit it himself, which a PDF does not allow.

**How to apply:** use the source `profiling_reports/sft_report_2026_09_28/sft_report.md`. Run `build_docx.py` (Markdown → HTML → LibreOffice → .docx; there is no pandoc and no python-docx on this machine). After an edit, grep the text for "we/our" that is outside quotes. The EOD style is separate, see [[feedback_eod_report_format]].
