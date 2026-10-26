#!/usr/bin/env python3
"""<report>.md -> <report>.docx with python-docx (2026-10-21; the laptop has no LibreOffice / pandoc any more).
Open the .docx in Google Docs to edit. Handles what the report uses: #/##/### headings, paragraphs, - and 1. lists,
| tables |, ``` code blocks, **bold**, `code`, and "![caption](file.png)" images (embedded in the .docx).

Usage: PYTHONPATH=<python-docx> python3 build_report_docx.py <report.md> [out.docx]
"""
import os, re, sys
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

SRC = os.path.abspath(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(SRC)[0] + ".docx"


def runs(par, text, size=None):
    # **bold** and `code` spans; everything else plain.
    for part in re.split(r"(\*\*.+?\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Courier New"
        else:
            r = par.add_run(part)
        if size:
            r.font.size = Pt(size)


def split_row(line):
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    cells, cur, in_code = [], "", False
    for ch in body:
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append(cur.strip()); cur = ""
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


def main():
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(11)
    for s in ("Heading 1", "Heading 2", "Heading 3", "Title"):
        doc.styles[s].font.name = "Arial"; doc.styles[s].font.color.rgb = RGBColor(0, 0, 0)
    lines = open(SRC, encoding="utf-8").read().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1; block = []
            while j < len(lines) and not lines[j].startswith("```"):
                block.append(lines[j]); j += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(block)); r.font.name = "Courier New"; r.font.size = Pt(8)
            i = j + 1; continue
        m = re.match(r"^!\[(.*)\]\((.+)\)\s*$", line)
        if m:
            doc.add_picture(os.path.join(os.path.dirname(SRC), m.group(2)), width=Inches(6.5))
            p = doc.add_paragraph(); r = p.add_run(m.group(1)); r.italic = True; r.font.size = Pt(9)
            i += 1; continue
        m = re.match(r"^(#{1,3}) (.*)$", line)
        if m:
            lvl = len(m.group(1))
            h = doc.add_heading(level=0 if lvl == 1 else lvl - 1)
            runs(h, m.group(2))
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            head = split_row(rows[0]); body = [split_row(r) for r in rows[2:]]
            t = doc.add_table(rows=1 + len(body), cols=len(head))
            t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.LEFT
            for c, txt in enumerate(head):
                cell = t.rows[0].cells[c]; cell.text = ""
                runs(cell.paragraphs[0], f"**{txt}**" if txt else "", size=9)
            for r, row in enumerate(body, 1):
                row = row + [""] * (len(head) - len(row))
                for c, txt in enumerate(row[:len(head)]):
                    cell = t.rows[r].cells[c]; cell.text = ""
                    runs(cell.paragraphs[0], txt, size=9)
            doc.add_paragraph()
            continue
        if re.match(r"^- ", line) or re.match(r"^\d+\. ", line):
            ordered = bool(re.match(r"^\d+\. ", line))
            pat = r"^\d+\. " if ordered else r"^- "
            while i < len(lines) and re.match(pat, lines[i]):
                p = doc.add_paragraph(style="List Number" if ordered else "List Bullet")
                runs(p, re.sub(pat, "", lines[i], count=1)); i += 1
            continue
        if line.strip():
            runs(doc.add_paragraph(), line)
        i += 1
    doc.save(OUT)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
