#!/usr/bin/env python3
# summarize_sftb.py (2026-10-21): the tables of run_all_sftb.sh + run_extra_sftb.sh, from the pulled runs folder.
#   A5  RMSNorm + RoPE: ours (nopos / identity / packed) and the rivals, F / B / F+B, ratio ours / rival (time; < 1 = ours faster)
#   B1  attention per layout: ours packed (plan A) and ours dense full-row next to every library, fwd / bwd / fwd+bwd; and
#       padding vs packing for the same chats: padded = (chats in the layout) x (dense full-row time per row), packed = plan A
#       time of the 8 packed rows. Chat counts: uniform_L exact (8 x 4096 / L); real rows = the attention test's own packing
#       (chat length = last trained label + 1 of each padded row, First-Fit Decreasing), recomputed here from labels.bin
#   A3b the trainer's in-place loss backward vs the out-of-place entry (+ checksums)
#   A1 / A4 / B2 / S1: their own summaries, copied
# usage: python3 summarize_sftb.py <runs folder> <padded train labels.bin>
import os, re, statistics, struct, sys

R = sys.argv[1]; LAB = sys.argv[2]; T = 4096


def lines(p):
    return open(p, errors="replace").read().splitlines() if os.path.exists(p) else []


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def section(t):
    print(f"\n## {t}\n")


# ---- A5 ------------------------------------------------------------------------------------------------------------
section("A5 fused QK-RMSNorm + RoPE, q [8,18,4096,128] + k [8,6,4096,128] fp32 (ms, median)")
ours = {kv(l)["case"]: kv(l) for l in lines(f"{R}/A5_rope/ours.log") if l.startswith("ROPE case=")}
riv = [kv(l) for l in lines(f"{R}/A5_rope/rivals.log") if l.startswith("ROPE impl=")]
for l in lines(f"{R}/A5_rope/ours.log"):
    if "identity_vs_nopos" in l or l.startswith("packed rows"): print(l)
print(f"\n| impl | case | F ms | B ms | F+B ms | F+B ours / impl | kept MiB | max err dq |")
print("|---|---|---|---|---|---|---|---|")
for c, o in ours.items():
    print(f"| ours (fused, 1 kernel each way) | {c} | {float(o['fwd_med_ms']):.4f} | {float(o['bwd_med_ms']):.4f} | "
          f"{float(o['fwd_med_ms']) + float(o['bwd_med_ms']):.4f} | 1.00 | rstd only | (test_rope_positions) |")
for r in riv:
    o = ours.get(r["case"])
    fb_o = float(o["fwd_med_ms"]) + float(o["bwd_med_ms"]) if o else float("nan")
    print(f"| {r['impl']} | {r['case']} | {float(r['fwd_med_ms']):.4f} | {float(r['bwd_med_ms']):.4f} | {float(r['fb_med_ms']):.4f} | "
          f"{fb_o / float(r['fb_med_ms']):.3f} | {r['kept_MiB']} | {r['err_dq']} |")
for l in lines(f"{R}/A5_rope/rivals.log"):
    if l.startswith(("FAIL", "SKIP")): print("  ", l)
tl = lines(f"{R}/A5_rope/test_rope_positions.log")
print("\ntest_rope_positions:", " | ".join(l.strip() for l in tl if re.search(r"passed|PASS|FAIL", l))[-400:])

# ---- B1 ------------------------------------------------------------------------------------------------------------
section("B1 attention per layout, B 8 x T 4096, 18/6 heads, hd 128, bf16, causal (ms, median)")
bench = {}
for f in ("ours.log", "rivals.log"):
    for l in lines(f"{R}/B1_attn_layouts/{f}"):
        if l.startswith("BENCH layout="):
            d = kv(l); bench.setdefault(d["layout"], {}).setdefault(d["impl"], {})[d["dir"]] = float(d["med_ms"])
# chats per layout: the test's packing, recomputed
raw = open(LAB, "rb").read(); n = len(raw) // (2 * T); L = []
for r in range(n):
    row = struct.unpack_from(f"<{T}H", raw, r * T * 2)
    L.append(next((t + 1 for t in range(T - 1, -1, -1) if row[t] != 65535), 0))
rows, used = [], []
for x in sorted(L, reverse=True):
    i = next((i for i, u in enumerate(used) if u + x <= T), None)
    if i is None: rows.append([]); used.append(0); i = len(rows) - 1
    rows[i].append(x); used[i] += x
chats = {}
for f0 in (0, len(rows) // 2 // 8 * 8, (len(rows) - 8) // 8 * 8):
    chats[f"real_rows_{f0}-{f0 + 7}"] = sum(len(r) for r in rows[f0:f0 + 8])
for Lu in (512, 1024, 2048, 4096): chats[f"uniform_{Lu}"] = 8 * T // Lu
print(f"test packing: {n} padded rows -> {len(rows)} packed rows; chats per layout: {chats}\n")
impls = sorted({i for v in bench.values() for i in v}, key=lambda s: (not s.startswith("ours"), s))
for dname in ("fwd", "bwd", "f+b"):
    print(f"\n{dname}: time in ms; in brackets ours packed (ours_A_plan) / impl (< 1 = ours faster)\n")
    print("| layout | chats | " + " | ".join(impls) + " |")
    print("|---|---|" + "---|" * len(impls))
    for lay, v in bench.items():
        def t(i):
            x = v.get(i, {})
            return x["fwd"] + x["bwd"] if dname == "f+b" and "fwd" in x and "bwd" in x else x.get(dname)
        a = t("ours_A_plan"); cells = []
        for i in impls:
            x = t(i)
            cells.append("-" if x is None else (f"{x:.3f}" + (f" ({a / x:.2f})" if a and i != "ours_A_plan" else "")))
        print(f"| {lay} | {chats.get(lay, '?')} | " + " | ".join(cells) + " |")
print("\npadding vs packing, the same chats (ours): padded = chats x dense full-row time per row (dense B8 / 8); packed = plan A on the 8 packed rows\n")
print("| layout | chats | padded fwd+bwd ms | packed fwd+bwd ms | packed / padded |")
print("|---|---|---|---|---|")
for lay, v in bench.items():
    d, a = v.get("ours_dense_fullrow", {}), v.get("ours_A_plan", {})
    if {"fwd", "bwd"} <= d.keys() and {"fwd", "bwd"} <= a.keys() and lay in chats:
        pad = chats[lay] * (d["fwd"] + d["bwd"]) / 8; pk = a["fwd"] + a["bwd"]
        print(f"| {lay} | {chats[lay]} | {pad:.3f} | {pk:.3f} | {pk / pad:.3f} |")

# ---- A3b -----------------------------------------------------------------------------------------------------------
section("A3b loss backward: the trainer's in-place entry vs the out-of-place entry (ms, median)")
print("| labels | case | B out-of-place | B in-place | in/out | F+B out | F+B in | gradient |")
print("|---|---|---|---|---|---|---|---|")
for s in ("padded", "packed"):
    for l in lines(f"{R}/A3b_loss_inplace/{s}.log"):
        if l.startswith("INPLACE"):
            d = kv(l)
            print(f"| {s} | {d['case']} | {float(d['B_med_ms']):.4f} | {float(d['Bi_med_ms']):.4f} | "
                  f"{float(d['Bi_med_ms']) / float(d['B_med_ms']):.3f} | {float(d['FB_med_ms']):.4f} | {float(d['FBi_med_ms']):.4f} | {d['gradient']} |")

# ---- copies --------------------------------------------------------------------------------------------------------
for t, p, grab in (("A1 attention scoreboard, steady state (summary_steady.txt)", "A1_scoreboard/summary_steady.txt", None),
                   ("A4 loss OLD (BLUCCL) vs NEW (master)", "A4_loss_old_new/summary.txt", None),
                   ("B2 loss NEW on the packed rows' labels (SUMMARY of the bench)", "B2_loss_packed/new_packed.log", "SUMMARY"),
                   ("A3 ours at the SFT shape (SUMMARY of the bench)", "A3_loss_rivals/ours.log", "SUMMARY")):
    section(t)
    ls = lines(f"{R}/{p}")
    if grab:
        i = next((k for k, l in enumerate(ls) if l.startswith(grab)), None)
        ls = ls[i:i + 14] if i is not None else ["(missing)"]
    print("```\n" + "\n".join(ls) + "\n```")
