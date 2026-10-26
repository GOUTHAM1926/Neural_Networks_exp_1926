#!/usr/bin/env python3
# 2026-10-17: ours (bench_sparse_ce_sft.cu -> ours.log) next to the other libraries (loss_rivals_1017.py -> rivals.log), per case:
# F / B / F+B (median ms over the timed calls), F+B as a multiple of ours, memory (kept = what the forward holds for the
# backward; peak = the most above the inputs during F+B), the loss (Part 1: same logits and labels, so it must equal ours).
# Ours' memory is counted from its buffers: kept = the saved row max + sum (2 x N x 4 B), peak = + the logits' gradient (N x V bf16).
# Part 2 (lm_head + loss): ours = the cuBLAS GEMMs (gemm_only) + ours' loss pair; the trainer runs our BLAS (about cuBLAS speed).
# usage: python3 loss_compare.py <run folder holding ours.log and rivals.log>
import re, sys

L = sys.argv[1]
N, V, D = 8 * 4096, 50304, 2304
MiB = 2**20

ours, cur = {}, None
for line in open(f"{L}/ours.log"):
    m = re.match(r"== (\w+)\s", line)
    if m:
        cur = m.group(1); ours[cur] = {}
        continue
    if cur is None:
        continue
    m = re.match(r"\s+trained rows : (\d+) of", line)
    if m:
        ours[cur]["trained"] = int(m.group(1))
    for key, pat in (("F", r"\s+F   fwd save_stats\s+median\s+([0-9.]+)"), ("B", r"\s+B   bwd with stats\s+median\s+([0-9.]+)"),
                     ("FB", r"\s+F\+B trainer pair\s+median\s+([0-9.]+)")):
        m = re.match(pat, line)
        if m:
            ours[cur][key] = float(m.group(1))
    m = re.match(r"\s+checks\s+: loss (\S+)", line)
    if m:
        ours[cur]["loss"] = float(m.group(1))
        ours[cur]["pass"] = "=> PASS" in line

riv = []
for line in open(f"{L}/rivals.log"):
    if line.startswith("LOSSBENCH"):
        riv.append(dict(kv.split("=", 1) for kv in line.split()[1:]))
    elif line.startswith(("SKIP", "FAIL", "note:", "device", "logit fill")):
        print(line.rstrip())


def fnum(x):
    return float(x) if x not in (None, "", "nan") else float("nan")


def row(name, F, B, FB, ref, kept, peak, loss, dloss="", gd=""):
    x = f"{FB / ref:5.2f}x" if ref and FB == FB else "   - "
    print(f"  {name:28s} {F:8.4f} {B:8.4f} {FB:8.4f}  {x}  {kept:9.1f} {peak:9.1f}  {loss:11.6f} {dloss:>10s} {gd:>10s}")


hdr = f"  {'':28s} {'F ms':>8s} {'B ms':>8s} {'F+B ms':>8s}  {'/ours':>6s}  {'kept MiB':>9s} {'peak MiB':>9s}  {'loss':>11s} {'d loss':>10s} {'grad diff':>10s}"
print("\nPART 1 -- loss only (logits given), median of the timed calls")
for case, o in ours.items():
    print(f"\n{case}: {o.get('trained', '?')} of {N} rows trained" + ("" if o.get("pass", True) else "   (ours' checks FAILED)"))
    print(hdr)
    ref = o.get("FB")
    row("OURS (save_stats pair)", o["F"], o["B"], o["FB"], ref, 2 * N * 4 / MiB, (2 * N * 4 + N * V * 2) / MiB, o["loss"])
    for r in riv:
        if r.get("part") == "1" and r.get("case") == case:
            l = fnum(r["loss"])
            dl = f"{l - o['loss']:+.2e}" if l == l and o["loss"] == o["loss"] else ""
            gd = r.get("grad_maxdiff_rel", "")
            row(r["impl"], fnum(r["F"]), fnum(r["B"]), fnum(r["FB"]), ref, fnum(r["kept_MB"]), fnum(r["peak_MB"]), l, dl, gd)
print("\n  F / B alone are not like for like: Liger computes the gradient in its forward (its B is a scale), PyTorch saves the "
      "whole log-softmax; compare F+B and memory.")

g = next((r for r in riv if r.get("part") == "2" and r.get("impl") == "gemm_only"), None)
p2 = [r for r in riv if r.get("part") == "2" and r.get("impl") != "gemm_only"]
if g or p2:
    print("\nPART 2 -- lm_head (X [N, 2304] x W^T) + loss, hidden states given")
    if g:
        print(f"  cuBLAS GEMMs alone: F {fnum(g['F']):.4f} ms (X W^T), B {fnum(g['B']):.4f} ms (dX + dW), F+B {fnum(g['FB']):.4f} ms")
    for case in dict.fromkeys(r["case"] for r in p2):
        o = ours.get(case, {})
        print(f"\n{case}: {o.get('trained', '?')} of {N} rows trained")
        print(hdr)
        ref = None
        if g and o:
            F_, B_, FB_ = fnum(g["F"]) + o["F"], fnum(g["B"]) + o["B"], fnum(g["FB"]) + o["FB"]
            ref = FB_
            row("OURS (cuBLAS GEMM + ours)", F_, B_, FB_, ref, fnum(g["kept_MB"]) + 2 * N * 4 / MiB,
                fnum(g["peak_MB"]) + 2 * N * 4 / MiB, float("nan"))
        tl = next((fnum(r["loss"]) for r in p2 if r["case"] == case and r["impl"] == "torch"), None)
        for r in p2:
            if r["case"] == case:
                l = fnum(r["loss"])
                dl = f"{l - tl:+.2e}" if tl is not None and r["impl"] != "torch" else ""
                row(r["impl"], fnum(r["F"]), fnum(r["B"]), fnum(r["FB"]), ref, fnum(r["kept_MB"]), fnum(r["peak_MB"]), l, dl)
    print("\n  Part 2 losses are on random X / W (not Part 1's logits): 'd loss' is against torch on the same X, W, labels.")
