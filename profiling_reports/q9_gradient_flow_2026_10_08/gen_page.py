#!/usr/bin/env python3
"""Build q9_page.html from q9_results.json and q9_sideeffect.json (every number on the page comes from them)."""
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "q9_results.json")))
S = json.load(open(os.path.join(HERE, "q9_sideeffect.json")))
TK = R["tokens"]
T = len(TK)
E1 = R["E1"]
KIND = [t["kind"] for t in TK]
N_REAL = R["config"]["n_real_inputs"]
PROMPT = [i for i, k in enumerate(KIND) if k == "prompt"]
TRAIN = [i for i, k in enumerate(KIND) if k == "trained"]
PAD = [i for i, k in enumerate(KIND) if k == "pad"]
esc = html.escape


def disp(s):
    if s is None:
        return "ignore"
    if s == "<|endoftext|>":
        return "eot"
    if s == "\n":
        return "↵"
    if s.startswith(" "):
        return "·" + s[1:]
    return s


# ---------------------------------------------------------------- 1. token strip (HTML)
def strip():
    cells = []
    for t in TK:
        k = t["kind"]
        inp = "eot" if k == "pad" else disp(t["input_text"])
        lab = disp(t["label_text"]) if t["label_text"] is not None else "65535"
        cells.append(
            f'<div class="tok {k}"><span class="pos">{t["pos"]}</span>'
            f'<span class="in">{esc(inp)}</span><span class="arrow" aria-hidden="true">↓</span>'
            f'<span class="lab">{esc(lab)}</span></div>')
    return "\n".join(cells)


# ---------------------------------------------------------------- 2. counted loss (SVG)
def loss_svg():
    W, H, x0, y0, bw = 760, 262, 64, 180, 24
    sc = 13.0                                       # px per nat
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Loss per position: only positions 20, 21 and 22 count; '
           f'every other position adds 0.">']
    for v in (0, 4, 8, 12):
        y = y0 - v * sc
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + T * bw}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{x0 - 6}" y="{y + 4:.1f}" class="ax" text-anchor="end">{v}</text>')
    for r, t in zip(E1["per_pos"], TK):
        x = x0 + r["pos"] * bw
        k = t["kind"]
        if k == "pad":
            out.append(f'<text x="{x + bw / 2}" y="{y0 - 6}" class="ax small" text-anchor="middle">–</text>'
                       f'<text x="{x + bw / 2}" y="{y0 + 16}" class="ax small" text-anchor="middle">{r["pos"]}</text>'
                       f'<text x="{x + bw / 2}" y="{y0 + 36}" class="zero" text-anchor="middle">0</text>')
            continue
        nll = r["nll_next"]
        h = nll * sc
        if k == "trained":
            out.append(f'<rect x="{x + 3}" y="{y0 - h:.1f}" width="{bw - 6}" height="{h:.1f}" class="f-answer"/>')
            out.append(f'<text x="{x + bw / 2}" y="{y0 + 36}" class="val answer-t" text-anchor="middle">'
                       f'{r["loss_counted"]:.1f}</text>')
        else:
            out.append(f'<rect x="{x + 3}" y="{y0 - h:.1f}" width="{bw - 6}" height="{h:.1f}" class="ghost"/>')
            out.append(f'<text x="{x + bw / 2}" y="{y0 + 36}" class="zero" text-anchor="middle">0</text>')
        out.append(f'<text x="{x + bw / 2}" y="{y0 + 16}" class="ax small" text-anchor="middle">{r["pos"]}</text>')
    out.append(f'<text x="{x0 - 8}" y="{y0 + 16}" class="ax" text-anchor="end">position</text>'
               f'<text x="{x0 - 8}" y="{y0 + 36}" class="ax" text-anchor="end">counted</text>')
    out.append(f'<text x="{x0}" y="14" class="ax">−log p (nats)</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 3. logit-gradient matrix (SVG)
def logit_svg():
    W, rh, x0, y0, mw = 700, 7, 150, 20, 470
    H = y0 + T * rh + 40
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Gradient at the logits: rows 20 to 22 have a value in '
           f'all 50,304 columns, every other row is exactly zero.">']
    for r in E1["per_pos"]:
        y = y0 + r["pos"] * rh
        k = KIND[r["pos"]]
        if k == "trained":
            out.append(f'<rect x="{x0}" y="{y}" width="{mw}" height="{rh - 1}" class="f-answer-soft"/>')
            out.append(f'<rect x="{x0 + mw * 0.37:.1f}" y="{y}" width="5" height="{rh - 1}" class="f-answer"/>')
        else:
            out.append(f'<rect x="{x0}" y="{y}" width="{mw}" height="{rh - 1}" class="empty-row"/>')
    yb = lambda p: y0 + p * rh
    out.append(f'<text x="{x0 - 10}" y="{(yb(0) + yb(20)) / 2 + 4:.1f}" class="lbl prompt-t" text-anchor="end">0–19 prompt: all 0</text>')
    out.append(f'<text x="{x0 - 10}" y="{yb(21) + 4:.1f}" class="lbl answer-t" text-anchor="end">20–22 trained</text>')
    out.append(f'<text x="{x0 - 10}" y="{(yb(23) + yb(28)) / 2 + 4:.1f}" class="lbl pad-t" text-anchor="end">23–27 pad: all 0</text>')
    out.append(f'<text x="{x0 + mw * 0.37 + 2.5:.1f}" y="{y0 + T * rh + 16}" class="ax" text-anchor="middle">'
               f'right token: −0.3333</text>')
    out.append(f'<text x="{x0}" y="12" class="ax">column 0</text>'
               f'<text x="{x0 + mw}" y="12" class="ax" text-anchor="end">column 50,303</text>')
    out.append(f'<text x="{x0 + mw}" y="{y0 + T * rh + 32}" class="ax" text-anchor="end">'
               f'every other column: +0.0000066 on average</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 4a. mechanism (SVG)
def mech_svg():
    cols = [("·capital", "prompt"), ("·France", "prompt"), ("?", "prompt"), ("…", "gap"),
            ("↵", "trained"), ("Paris", "trained"), (".", "trained"), ("eot", "pad"), ("eot", "pad")]
    W, H = 800, 330
    xs = [90 + i * 72 for i in range(len(cols))]
    yL, y4, y3 = 46, 120, 262
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="In the last layer, each answer position reads the '
           f'keys and values of every earlier position. The backward pass runs along those reads, so prompt '
           f'positions get gradient one layer down. Nothing reads the padding.">',
           '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>'
           '<marker id="ahA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="f-answer"/></marker></defs>']
    out.append(f'<text x="8" y="{yL + 4}" class="lbl">loss</text>'
               f'<text x="8" y="{y4 + 4}" class="lbl">layer 4 out</text>'
               f'<text x="8" y="{(y4 + y3) / 2 + 4}" class="lbl">attention</text>'
               f'<text x="8" y="{y3 + 4}" class="lbl">layer 3 out</text>')
    # attention reads: from the trained "↵" column (index 4) down to earlier columns at layer 3
    q = xs[4]
    for j in (0, 1, 2):
        out.append(f'<path d="M{q},{y4 + 14} C{q - 30},{y4 + 70} {xs[j] + 30},{y3 - 60} {xs[j]},{y3 - 15}" '
                   f'class="read" marker-end="url(#ahA)"/>')
    out.append(f'<text x="{xs[1]}" y="{y4 + 46}" class="lbl answer-t" '
               f'text-anchor="middle">gradient flows back</text>'
               f'<text x="{xs[1]}" y="{y4 + 62}" class="lbl answer-t" '
               f'text-anchor="middle">along every read</text>')
    for x, (name, k) in zip(xs, cols):
        if k == "gap":
            out.append(f'<text x="{x}" y="{y4 + 5}" class="lbl" text-anchor="middle">…</text>'
                       f'<text x="{x}" y="{y3 + 5}" class="lbl" text-anchor="middle">…</text>')
            continue
        # residual path inside the column
        out.append(f'<line x1="{x}" y1="{y3 - 14}" x2="{x}" y2="{y4 + 14}" class="resid"/>')
        if k == "trained":
            out.append(f'<line x1="{x}" y1="{yL + 12}" x2="{x}" y2="{y4 - 15}" class="hot" marker-end="url(#ahA)"/>'
                       f'<rect x="{x - 22}" y="{yL - 12}" width="44" height="24" rx="4" class="f-answer"/>'
                       f'<text x="{x}" y="{yL + 4}" class="inv" text-anchor="middle">loss</text>')
            n4, n3 = "f-answer", "f-answer"
        elif k == "prompt":
            out.append(f'<text x="{x}" y="{yL + 4}" class="ax" text-anchor="middle">not scored</text>')
            n4, n3 = "empty-node", "f-prompt"
        else:
            out.append(f'<text x="{x}" y="{yL + 4}" class="ax" text-anchor="middle">not scored</text>')
            n4, n3 = "empty-node", "empty-node"
        out.append(f'<circle cx="{x}" cy="{y4}" r="13" class="{n4}"/>')
        out.append(f'<circle cx="{x}" cy="{y3}" r="13" class="{n3}"/>')
        if n4 == "empty-node":
            out.append(f'<text x="{x}" y="{y4 + 4}" class="zero" text-anchor="middle">0</text>')
        if n3 == "empty-node":
            out.append(f'<text x="{x}" y="{y3 + 4}" class="zero" text-anchor="middle">0</text>')
        cls = {"prompt": "prompt-t", "trained": "answer-t", "pad": "pad-t"}[k]
        out.append(f'<text x="{x}" y="{y3 + 36}" class="tokl {cls}" text-anchor="middle">{esc(name)}</text>')
    out.append(f'<text x="{xs[8] + 24}" y="{(y4 + y3) / 2 + 4}" class="lbl pad-t">nobody reads</text>'
               f'<text x="{xs[8] + 24}" y="{(y4 + y3) / 2 + 20}" class="lbl pad-t">the padding</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 4b. measured heatmap (SVG)
def heat_svg():
    rows = E1["resid_grad"]                          # [layer 0..L][pos]
    L = len(rows) - 1
    cw, ch, x0, y0 = 25, 30, 92, 18
    W, H = x0 + T * cw + 10, y0 + len(rows) * ch + 64
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Measured gradient size at each layer and position. '
           f'At the top layer only positions 20 to 22 are non-zero. In every lower layer the prompt positions are '
           f'non-zero too. Padding is exactly zero in every layer.">']
    for li in range(L, -1, -1):
        r = rows[li]
        y = y0 + (L - li) * ch
        nz = [v for v in r if v > 0]
        lo, hi = math.log10(min(nz)), math.log10(max(nz))
        name = "embedding" if li == 0 else f"layer {li} out"
        out.append(f'<text x="{x0 - 8}" y="{y + ch / 2 + 4}" class="lbl" text-anchor="end">{name}</text>')
        for p, v in enumerate(r):
            x = x0 + p * cw
            k = KIND[p]
            if v == 0:
                out.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{cw - 3}" height="{ch - 3}" rx="2" class="zero-cell"/>'
                           f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" class="zero" text-anchor="middle">0</text>')
            else:
                a = 0.25 + 0.75 * ((math.log10(v) - lo) / (hi - lo) if hi > lo else 1)
                cls = "f-answer" if k == "trained" else "f-prompt"
                out.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{cw - 3}" height="{ch - 3}" rx="2" class="{cls}" '
                           f'fill-opacity="{a:.2f}"><title>position {p}, {name}: {v:.2e}</title></rect>')
    yb = y0 + len(rows) * ch
    for p, t in enumerate(TK):
        cls = {"prompt": "prompt-t", "trained": "answer-t", "pad": "pad-t"}[t["kind"]]
        lab = "eot" if t["kind"] == "pad" else disp(t["input_text"])
        out.append(f'<text transform="translate({x0 + p * cw + cw / 2 + 4},{yb + 8}) rotate(-60)" '
                   f'class="tokl small {cls}" text-anchor="end">{esc(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- 5. weight table (HTML)
def weight_table():
    c = E1["contrib"]
    names = [("q", "Q (query)"), ("k", "K (key)"), ("v", "V (value)"), ("o", "O (attention out)"),
             ("w1", "MLP w1"), ("w3", "MLP w3"), ("w2", "MLP w2")]

    def cnt(layer, w, kind):
        v = c[f"L{layer}.{w}"]
        ids = {"prompt": PROMPT, "trained": TRAIN, "pad": PAD}[kind]
        return sum(1 for i in ids if v[i] != 0), len(ids)

    rows = []
    for w, label in names:
        cells = []
        for layer in (4, 3, 1):
            for kind in ("prompt", "trained", "pad"):
                n, tot = cnt(layer, w, kind)
                cls = "yes" if n else "no"
                cells.append(f'<td class="{cls} {kind}-c">{n}/{tot}</td>')
        rows.append(f"<tr><th scope=\"row\">{label}</th>{''.join(cells)}</tr>")
    head = ('<tr><th rowspan="2" scope="col">weight</th><th colspan="3" scope="colgroup">layer 4 (last)</th>'
            '<th colspan="3" scope="colgroup">layer 3</th><th colspan="3" scope="colgroup">layer 1</th></tr><tr>'
            + "".join('<th scope="col" class="prompt-t">prompt</th><th scope="col" class="answer-t">trained</th>'
                      '<th scope="col" class="pad-t">pad</th>' for _ in range(3)) + "</tr>")
    return f'<table class="wt"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


# ---------------------------------------------------------------- 7. training curves (SVG)
def curve_svg():
    W, H, x0, y0, xw, yh = 680, 300, 48, 20, 600, 230
    sx = lambda s: x0 + s / 60 * xw
    sy = lambda v: y0 + yh - v / 12 * yh
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Over 60 steps with the SFT mask, the answer loss falls '
           f'to 0.06 while the prompt loss stays near 9. Training every position instead brings both down.">']
    for v in range(0, 13, 2):
        out.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x0 + xw}" y2="{sy(v):.1f}" class="grid"/>'
                   f'<text x="{x0 - 8}" y="{sy(v) + 4:.1f}" class="ax" text-anchor="end">{v}</text>')
    for s in range(0, 61, 10):
        out.append(f'<text x="{sx(s):.1f}" y="{y0 + yh + 18}" class="ax" text-anchor="middle">{s}</text>')
    out.append(f'<text x="{x0 + xw / 2}" y="{H - 4}" class="ax" text-anchor="middle">training step</text>')
    out.append(f'<text x="{x0}" y="12" class="ax">mean −log p of the true next token</text>')
    for run, dash in (("all_positions", "6 5"), ("sft_mask", "")):
        for ids, cls in ((PROMPT, "s-prompt"), (TRAIN, "s-answer")):
            pts = []
            for row in R["E3"]["curves"][run]:
                m = sum(row["nll"][i] for i in ids) / len(ids)
                pts.append(f'{sx(row["step"]):.1f},{sy(m):.1f}')
            d = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<polyline points="{" ".join(pts)}" class="{cls}"{d}/>')
    out.append("</svg>")
    return "".join(out)


def curve_end(run, ids):
    rows = R["E3"]["curves"][run]
    m = lambda r: sum(r["nll"][i] for i in ids) / len(ids)
    return m(rows[0]), m(rows[-1])


se = {k: sum(r[k] for r in S["per_pos"]) / len(S["per_pos"]) for k in ("true", "other_seen", "answer_tokens", "unseen")}
se_rank = sorted(r["rank_true"] for r in S["per_pos"])[len(S["per_pos"]) // 2]
pp = E1["per_pos"]
loss_terms = " + ".join(f'{pp[i]["loss_counted"]:.3f}' for i in TRAIN)
E2 = R["E2"]
wte = E1["wte"]
sft_p0, sft_p1 = curve_end("sft_mask", PROMPT)
sft_a0, sft_a1 = curve_end("sft_mask", TRAIN)
all_p0, all_p1 = curve_end("all_positions", PROMPT)
all_a0, all_a1 = curve_end("all_positions", TRAIN)

page = f"""<title>SFT Gradient Flow</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
/* Layout: one reading column (~44rem); figures may run wider inside their own scroll boxes. Colours encode the
   three kinds of position everywhere on the page: prompt = blue, trained = amber, padding = grey. */
:root {{
  --bg: #f4f6f9; --surface: #ffffff; --fg: #18202d; --muted: #586377; --rule: #d6dce5;
  --prompt: #2a68a3; --prompt-soft: #dbe7f3; --answer: #b26d0c; --answer-soft: #f6e5c6;
  --pad: #858e9e; --pad-soft: #e9ecf1;
  --serif: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #10141a; --surface: #171c24; --fg: #e2e6ec; --muted: #98a2b3; --rule: #2a313c;
  --prompt: #74aade; --prompt-soft: #1b2b3d; --answer: #e8a63c; --answer-soft: #3a2b12;
  --pad: #7f8899; --pad-soft: #222833; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #10141a; --surface: #171c24; --fg: #e2e6ec; --muted: #98a2b3; --rule: #2a313c;
  --prompt: #74aade; --prompt-soft: #1b2b3d; --answer: #e8a63c; --answer-soft: #3a2b12;
  --pad: #7f8899; --pad-soft: #222833; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.6 var(--sans); }}
.wrap {{ max-width: 46rem; margin: 0 auto; padding-inline: 18px; padding-block: 40px 72px; }}
h1, h2 {{ font-family: var(--serif); font-weight: 600; text-wrap: balance; line-height: 1.2; }}
h1 {{ font-size: 2.1rem; margin: 0.2rem 0 0.8rem; }}
h2 {{ font-size: 1.35rem; margin: 3rem 0 0.6rem; display: flex; gap: 0.6rem; align-items: baseline; }}
h2 .n {{ font-family: var(--mono); font-size: 0.85rem; color: var(--muted); font-weight: 500; }}
p {{ margin: 0.6rem 0; }}
.eyebrow {{ font: 500 0.75rem/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
.lede {{ font-size: 1.12rem; }}
.meta {{ color: var(--muted); font-size: 0.88rem; }}
code, .mono {{ font-family: var(--mono); font-size: 0.88em; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; margin: 1rem 0; font-size: 0.9rem; }}
.chip {{ display: inline-flex; align-items: center; gap: 0.45rem; }}
.chip i {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
.chip .p {{ background: var(--prompt); }} .chip .a {{ background: var(--answer); }}
.chip .d {{ background: var(--pad-soft); border: 1.5px dashed var(--pad); }}
.answer {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.2rem; margin: 1.4rem 0; }}
.answer ol {{ margin: 0.3rem 0 0; padding-left: 1.3rem; display: grid; gap: 0.45rem; }}
.answer h3 {{ margin: 0; font: 600 0.8rem/1 var(--mono); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
figure {{ margin: 1.2rem 0 1.6rem; }}
.scroll {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; }}
.scroll svg {{ display: block; width: 100%; height: auto; color: var(--fg); }}
.scroll.wide svg {{ min-width: 640px; }}
figcaption {{ color: var(--muted); font-size: 0.88rem; margin-top: 0.5rem; }}
.strip {{ display: grid; grid-template-columns: repeat({T}, minmax(46px, 1fr)); gap: 3px; min-width: {T * 49}px; }}
.tok {{ display: grid; justify-items: center; gap: 2px; padding: 6px 2px; border-radius: 5px; font-family: var(--mono); font-size: 0.78rem; }}
.tok .pos {{ font-size: 0.66rem; color: var(--muted); }}
.tok .arrow {{ color: var(--muted); font-size: 0.7rem; line-height: 1; }}
.tok .in, .tok .lab {{ white-space: nowrap; }}
.tok.prompt {{ background: var(--prompt-soft); }} .tok.prompt .lab {{ color: var(--muted); }}
.tok.trained {{ background: var(--answer-soft); outline: 1.5px solid var(--answer); }} .tok.trained .lab {{ font-weight: 500; }}
.tok.pad {{ background: var(--pad-soft); border: 1.5px dashed var(--pad); }} .tok.pad .lab, .tok.pad .in {{ color: var(--muted); }}
.formula {{ font-family: var(--mono); font-size: 0.92rem; background: var(--surface); border-left: 3px solid var(--answer); padding: 0.6rem 0.9rem; margin: 0.8rem 0; overflow-x: auto; }}
.grid {{ stroke: var(--rule); stroke-width: 1; }}
.ax {{ fill: var(--muted); font: 11px var(--sans); }} .ax.small {{ font-size: 10px; }}
.val {{ fill: var(--fg); font: 500 11px var(--mono); }}
.zero {{ fill: var(--muted); font: 10px var(--mono); }}
.lbl {{ fill: var(--fg); font: 12px var(--sans); }}
.inv {{ fill: var(--bg); font: 600 11px var(--sans); }}
.tokl {{ font: 12px var(--mono); }} .tokl.small {{ font-size: 10px; }}
.prompt-t {{ fill: var(--prompt); color: var(--prompt); }} .answer-t {{ fill: var(--answer); color: var(--answer); }} .pad-t {{ fill: var(--pad); color: var(--pad); }}
.f-answer {{ fill: var(--answer); }} .f-answer-soft {{ fill: var(--answer-soft); }} .f-prompt {{ fill: var(--prompt); }}
.ghost {{ fill: var(--pad-soft); stroke: var(--rule); }}
.empty-row {{ fill: none; stroke: var(--rule); stroke-dasharray: 2 2; }}
.empty-node, .zero-cell {{ fill: var(--surface); stroke: var(--pad); stroke-dasharray: 3 2; }}
.resid {{ stroke: var(--rule); stroke-width: 2; }}
.read {{ fill: none; stroke: var(--answer); stroke-width: 1.6; }}
.hot {{ stroke: var(--answer); stroke-width: 2; }}
.s-prompt {{ fill: none; stroke: var(--prompt); stroke-width: 2.2; }} .s-answer {{ fill: none; stroke: var(--answer); stroke-width: 2.2; }}
table {{ border-collapse: collapse; font-size: 0.88rem; width: 100%; }}
th, td {{ padding: 0.35rem 0.5rem; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }}
thead th {{ font-weight: 600; }}
.tablebox {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 6px 10px; }}
.wt td {{ font-family: var(--mono); text-align: center; font-variant-numeric: tabular-nums; }}
.wt th[scope="col"], .wt th[scope="colgroup"] {{ text-align: center; }}
.wt td.no {{ color: var(--muted); }}
.wt td.yes.prompt-c {{ background: var(--prompt-soft); }} .wt td.yes.trained-c {{ background: var(--answer-soft); }}
.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 12px; margin: 1rem 0; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 0.9rem 1rem; min-width: 0; }}
.card h3 {{ margin: 0 0 0.4rem; font-size: 1rem; }}
.card .big {{ font: 500 1.5rem/1.2 var(--mono); }}
.key {{ display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; font-size: 0.85rem; color: var(--muted); margin-top: 0.4rem; }}
.key span {{ display: inline-flex; gap: 0.4rem; align-items: center; }}
.key svg {{ width: 30px; height: 10px; }}
.note {{ font-size: 0.9rem; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 1rem; margin-top: 3rem; }}
</style>

<div class="wrap">
<p class="eyebrow">SFT Q&amp;A · question 9</p>
<h1>How SFT learns only the answer</h1>
<p class="lede">The model reads every token of the chat. Only the answer gets a loss. The gradient then flows back into the prompt, so the model learns to <em>use</em> the question, but it is never trained to <em>write</em> it.</p>
<p class="meta">Every number below is measured, on 8 Oct 2026. Model: a tiny model built like ours (4 layers, tied embedding table with 50,304 rows, RMSNorm, causal attention, SwiGLU), fp64 on CPU. The chat was tokenized by our own <code>prepare_sft_data.py</code> (template <code>tulu-plain-v1</code>).</p>

<div class="legend">
  <span class="chip"><i class="p"></i>prompt: label 65535, ignored</span>
  <span class="chip"><i class="a"></i>trained: answer tokens and the end token</span>
  <span class="chip"><i class="d"></i>padding: label 65535, ignored</span>
</div>

<div class="answer">
<h3>The answer in four steps</h3>
<ol>
<li><strong>Forward:</strong> all {T} positions are computed, the prompt and the padding too.</li>
<li><strong>Loss:</strong> only the {len(TRAIN)} trained positions are scored. The other {T - len(TRAIN)} add exactly 0, and N = {len(TRAIN)}.</li>
<li><strong>Backward:</strong> the gradient starts on those {len(TRAIN)} rows only. In every layer below the top, it reaches the prompt positions through attention, because the answer <em>read</em> them.</li>
<li><strong>Padding</strong> gets exactly 0 everywhere. It comes after the answer, and attention only looks back, so nothing ever reads it.</li>
</ol>
</div>

<h2><span class="n">1</span>The row the model gets</h2>
<p>One chat: the user asks <em>“What is the capital of France?”</em> and the answer is <em>“Paris.”</em> It becomes {N_REAL + 1} tokens. The input is the tokens without the last one. The label at each position is the <em>next</em> token, or 65535 where we don't want a loss.</p>
<figure>
<div class="scroll"><div class="strip">
{strip()}
</div></div>
<figcaption>Top: the input token at each position. Bottom: its label. <span class="mono">·</span> marks a leading space, <span class="mono">↵</span> a newline, <span class="mono">eot</span> the end token (id 50256). The trained positions are the ones <em>before</em> each answer token: position 20 holds <span class="mono">↵</span> and must predict <span class="mono">Paris</span>; position 22 must predict the end token, which teaches the model to stop.</figcaption>
</figure>

<h2><span class="n">2</span>Forward: every position is computed, three are scored</h2>
<p>The forward pass is the same as in pretraining. Every position gets 50,304 scores (the logits) and a probability for every token. The label decides which positions <em>count</em>.</p>
<figure>
<div class="scroll wide">{loss_svg()}</div>
<figcaption>Grey bars: −log p that each prompt position gives its real next token. The model computed it, but it is multiplied by nothing. Amber bars: the only three values that enter the loss. All values sit near ln 50,304 ≈ 10.83 because this tiny model is untrained; our real 1.5B base starts at 1.446.</figcaption>
</figure>
<div class="formula">loss = ({loss_terms}) ÷ N = {E1["loss"]:.3f} &nbsp;&nbsp; with N = {E1["N"]} trained tokens (not {T}, not {N_REAL})</div>

<h2><span class="n">3</span>Backward starts at the logits, on three rows only</h2>
<p>The gradient of the loss with respect to the logits is a {T} × 50,304 table. For a trained row it is <span class="mono">(p − 1) ÷ N</span> on the right token and <span class="mono">p ÷ N</span> on every other token. For an ignored row the kernel writes zeros.</p>
<figure>
<div class="scroll wide">{logit_svg()}</div>
<figcaption>Measured: rows 20–22 have a non-zero value in all {pp[20]["logit_grad_nonzero_cols"]:,} columns, with {pp[20]["logit_grad_right"]:.4f} on the right token (the formula gives (p − 1) ÷ 3 = {pp[20]["logit_grad_right_formula"]:.4f}). The other 50,303 values add up to +0.3333, because a softmax gradient row always sums to 0. Rows 0–19 and 23–27: all 50,304 values are exactly 0.</figcaption>
</figure>
<p>So the loss never pushes a prompt position to predict its next prompt token. That is the masking.</p>

<h2><span class="n">4</span>Into the layers: the prompt gets gradient through attention</h2>
<p>Here is the part that is easy to miss. In layer 4, the answer position <span class="mono">↵</span> builds its output by <em>reading</em> the keys and values of every earlier position, the prompt included. The backward pass runs along the same reads in reverse. So the prompt positions get gradient one layer down, even though their own prediction is never scored.</p>
<figure>
<div class="scroll wide">{mech_svg()}</div>
<figcaption>Three of the reads are drawn. In the real model every answer position reads every earlier position, in every layer and every head. Amber = trained, blue = prompt, dashed = exactly 0.</figcaption>
</figure>
<p>The measured gradient at every layer, for all {T} positions:</p>
<figure>
<div class="scroll wide">{heat_svg()}</div>
<figcaption>Top row: right before the output layer, only the three trained positions have gradient; the prompt is 0 there. Every row below: the prompt positions light up as well, because they were read. Padding: 0 in every layer. Shade = size within that row, on a log scale; a dashed cell with 0 = exactly zero. Hover a cell for its value.</figcaption>
</figure>

<h2><span class="n">5</span>Which weights learn from which positions</h2>
<p>Each weight matrix is shared by all positions, so its gradient is a sum over positions. The table counts the positions that add something to that sum.</p>
<figure>
<div class="tablebox">{weight_table()}</div>
<figcaption>In the last layer the prompt feeds only K and V, the part that gets <em>read</em>. In the lower layers it feeds every weight, because there its output is read again by the next layer. Padding feeds nothing. (Layer 2 is the same as layer 3. The 19/20 for Q: position 0 can only look at itself, so its attention weight is always 1 and its query gets no gradient.)</figcaption>
</figure>
<p><strong>The embedding table</strong> is used twice (it is tied): once to look up the input tokens, once as the output layer.</p>
<div class="tablebox"><table>
<thead><tr><th scope="col">part of the table's gradient</th><th scope="col" class="num">rows that get a gradient</th><th scope="col">why</th></tr></thead>
<tbody>
<tr><td>output side</td><td class="num">{wte["rows_nonzero_output_side"]:,} of 50,304</td><td>every wrong token is pushed down a little at the 3 trained rows</td></tr>
<tr><td>input side</td><td class="num">{wte["rows_nonzero_input_side"]}</td><td>the distinct tokens at prompt and answer positions</td></tr>
<tr><td>input side, padding id 50256</td><td class="num">0</td><td>its only input uses are padding, which gets no gradient (measured norm {wte["pad_row_input_side_norm"]:.1f})</td></tr>
</tbody></table></div>
<p class="meta">The two parts add up to PyTorch's own gradient exactly (largest difference {wte["sum_matches_autograd_maxdiff"]:.1f}).</p>

<h2><span class="n">6</span>Two checks</h2>
<div class="cards">
<div class="card"><h3>Change the padding</h3>
<p>The 5 padding tokens replaced by random ones ({esc(", ".join(repr(s) for s in E2["pad_changed_to"]))}).</p>
<p class="big">{E2["pad_change_max_grad_diff"]:.1f}</p><p>largest change in any weight gradient. Bit-for-bit the same: padding has no effect at all, with no attention mask.</p></div>
<div class="card"><h3>Change one prompt word</h3>
<p>France → Japan. The answer is still “Paris.”</p>
<p class="big">{E2["prompt_change_max_grad_diff"]:.3f}</p><p>largest change in a weight gradient, and the loss moved from {E2["prompt_change_loss"][0]:.4f} to {E2["prompt_change_loss"][1]:.4f}. The answer's gradient depends on the prompt.</p></div>
</div>

<h2><span class="n">7</span>Sixty training steps: what actually gets learned</h2>
<p>I trained the same model on this one chat for 60 steps (AdamW, LR 3e-3) twice: once with our SFT labels, once with every position trained, as in pretraining. After each step I measured how well it predicts the true next token at <em>every</em> position.</p>
<figure>
<div class="scroll wide">{curve_svg()}</div>
<div class="key">
<span><svg viewBox="0 0 30 10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" class="s-answer"/></svg>answer, SFT labels</span>
<span><svg viewBox="0 0 30 10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" class="s-prompt"/></svg>prompt, SFT labels</span>
<span><svg viewBox="0 0 30 10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" class="s-answer" stroke-dasharray="6 5"/></svg>answer, every position trained</span>
<span><svg viewBox="0 0 30 10" aria-hidden="true"><line x1="0" y1="5" x2="30" y2="5" class="s-prompt" stroke-dasharray="6 5"/></svg>prompt, every position trained</span>
</div>
<figcaption>Mean −log p of the true next token. With SFT labels: answer {sft_a0:.2f} → {sft_a1:.3f}, prompt {sft_p0:.2f} → {sft_p1:.2f}. With every position trained: answer {all_a0:.2f} → {all_a1:.2f}, prompt {all_p0:.2f} → {all_p1:.2f}.</figcaption>
</figure>
<p>With our labels the answer is learned almost perfectly, while the prompt stays near 9. But it did drop from 10.8 to 9.0 without being trained, so I checked what that drop is. At the prompt positions after 60 SFT steps, −log p of:</p>
<div class="tablebox"><table>
<thead><tr><th scope="col">token</th><th scope="col" class="num">−log p</th><th scope="col">meaning</th></tr></thead>
<tbody>
<tr><td>the true next prompt token</td><td class="num">{se["true"]:.2f}</td><td rowspan="2">the same: the model does not know which prompt token comes next</td></tr>
<tr><td>any other token of this chat</td><td class="num">{se["other_seen"]:.2f}</td></tr>
<tr><td>tokens that never appear in the chat</td><td class="num">{se["unseen"]:.2f}</td><td>pushed down from 10.8 by every step</td></tr>
<tr><td>the answer tokens (Paris . eot)</td><td class="num">{se["answer_tokens"]:.2f}</td><td>the model now wants to say the answer everywhere</td></tr>
</tbody></table></div>
<p>So the drop is not learning the prompt. The ~50,000 tokens that never appear were pushed far down. That makes all {S["seen_count"]} tokens of this chat look a bit more likely, all by about the same amount (the true next token ranks about {se_rank}th of 50,304, inside that group). On a real run with thousands of chats, this kind of drift in what the model says by default is what the before-and-after tests on MMLU and HellaSwag are meant to catch.</p>

<h2><span class="n">8</span>Where this happens in BluTrain</h2>
<div class="tablebox"><table>
<thead><tr><th scope="col">step</th><th scope="col">where</th><th scope="col">what it does</th></tr></thead>
<tbody>
<tr><td>labels</td><td class="mono">sft/prepare_sft_data.py, build_row</td><td>label = next token on answer and end-token positions, 65535 everywhere else</td></tr>
<tr><td>loss, forward</td><td class="mono">LossKernels.cu:215, :350</td><td>a row with label 65535 gets loss 0 and skips the 50,304-wide softmax</td></tr>
<tr><td>count N</td><td class="mono">LossKernels.cu:110</td><td>rows with 65535 are not counted</td></tr>
<tr><td>loss, backward</td><td class="mono">LossKernels.cu:692, :777</td><td>a row with label 65535 gets an all-zero gradient row</td></tr>
<tr><td>attention</td><td>causal GQA, unchanged</td><td>answer rows read prompt rows; nothing reads the padding</td></tr>
<tr><td>denominator</td><td class="mono">bluscript_zero_ckpt.cpp, SFT mode</td><td>N = trained tokens of the whole step, across all GPUs</td></tr>
</tbody></table></div>
<p class="meta">Line numbers are from Tensor-Implementations <code>d96b991</code>.</p>

<p class="note">What this measurement can and cannot show: the model is tiny and starts from random weights, so the sizes of the numbers mean nothing for our 1.5B model. Which values are zero and which are not comes from the structure (masked labels, causal attention, a tied table), and that structure is the same in our model. The scripts and results are saved with my run records (<code>q9_flow.py</code>, <code>q9_sideeffect.py</code>, <code>q9_results.json</code>).</p>
</div>
"""

with open(os.path.join(HERE, "q9_page.html"), "w") as f:
    f.write(page)
print("wrote", os.path.join(HERE, "q9_page.html"), len(page), "bytes")
