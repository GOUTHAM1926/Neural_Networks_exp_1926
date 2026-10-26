#!/usr/bin/env python3
# compare_samesteps.py (2026-10-21): the padded run (C_P_padded) against the packed run with the same chats in every step
# (C_S_samesteps, build_samesteps.py + run_samesteps.sh), step by step.
#   figure  the same 3 graphs as plot_sft_runs.py: (a) train loss per step, (b) val loss per epoch, (c) val loss vs minutes
#   table   trained tokens per step (must be equal at every step), train loss and grad norm differences (step 0, max, mean),
#           val loss at every val step, epoch means, step time, whole-run time, peak memory
# usage: PYTHONPATH=<matplotlib> python3 compare_samesteps.py <runs folder> [out prefix]
import csv, re, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else f"{R}/sft_padding_vs_packing_same_chats"
VAL = re.compile(r"\[val\] step (\d+) \| val_loss ([0-9.]+) \| (\d+) trained tokens")


def load(name):
    rows = list(csv.DictReader(open(f"{R}/{name}/sft_adamw.csv")))
    vals = {int(m.group(1)): (float(m.group(2)), int(m.group(3)))
            for m in (VAL.search(l) for l in open(f"{R}/{name}/run.log", errors="replace")) if m}
    mem = []
    for r in csv.reader(open(f"{R}/{name}/gpu.csv")):
        try: mem.append(float(r[1].split()[0]))
        except (IndexError, ValueError): pass
    f = lambda k: [float(r[k]) for r in rows]
    return {"loss": f("train_loss"), "norm": f("grad_norm"), "dt": f("dt_ms"), "el": f("elapsed_min"),
            "tok": [int(r["trained_tokens"]) for r in rows], "vals": vals, "peak": max(mem)}


P, S = load("C_P_padded"), load("C_S_samesteps")
n = len(P["loss"]); assert len(S["loss"]) == n, (n, len(S["loss"]))
spe = n / 2.0
lines = []
say = lambda s="": (lines.append(s), print(s))
same_tok = P["tok"] == S["tok"]
say(f"steps: {n} in both | trained tokens equal at every step: {same_tok} ({min(P['tok']):,}-{max(P['tok']):,} per step)")
dl = [abs(a - b) for a, b in zip(P["loss"], S["loss"])]; dn = [abs(a - b) for a, b in zip(P["norm"], S["norm"])]
say(f"step 0: train loss padded {P['loss'][0]:.6f} packed {S['loss'][0]:.6f} (diff {dl[0]:.1e}) | grad norm {P['norm'][0]:.4f} / {S['norm'][0]:.4f}")
say(f"train loss |padded - packed| over all {n} steps: mean {statistics.mean(dl):.2e}, max {max(dl):.2e} (step {dl.index(max(dl))})")
for a, b in ((0, 10), (10, 50), (50, 100), (100, n)):
    say(f"   steps {a}-{b - 1}: mean diff {statistics.mean(dl[a:b]):.2e}, max {max(dl[a:b]):.2e}")
say(f"grad norm |diff|: mean {statistics.mean(dn):.2e}, max {max(dn):.2e} (relative max {max(d / p for d, p in zip(dn, P['norm'])):.2%})")
h = n // 2
say(f"train loss mean epoch 1 / 2: padded {statistics.mean(P['loss'][:h]):.4f} / {statistics.mean(P['loss'][h:]):.4f}, "
    f"packed {statistics.mean(S['loss'][:h]):.4f} / {statistics.mean(S['loss'][h:]):.4f}")
say("val loss (same 496 val chats in both runs):")
say(f"   {'step':>5s} {'padded':>10s} {'packed':>10s} {'diff':>9s}   val tokens")
for s in sorted(P["vals"]):
    vp, tp = P["vals"][s]; vs, ts = S["vals"].get(s, (float("nan"), 0))
    say(f"   {s:5d} {vp:10.6f} {vs:10.6f} {vs - vp:+9.6f}   {tp:,} / {ts:,}")
mp = lambda r: statistics.median(r["dt"][10:]) / 1000
say(f"step time median: padded {mp(P):.3f} s, packed {mp(S):.3f} s | whole run (trainer clock): padded {P['el'][-1]:.2f} min, "
    f"packed {S['el'][-1]:.2f} min | peak GPU memory: padded {P['peak']:.0f} MiB, packed {S['peak']:.0f} MiB")
open(OUT + ".txt", "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
runs = (("padding (1 chat per row, 64 rows/step)", P, "#d62728", "-", 2.2),
        ("packing, same 64 chats per step (28 rows/step)", S, "#1f77b4", "--", 1.6))
for label, r, col, ls, lw in runs:
    ep = [s / spe for s in range(n)]
    ax[0].plot(ep, r["loss"], color=col, ls=ls, lw=lw, label=label)
    vs = sorted(r["vals"])
    ax[1].plot([s / spe for s in vs], [r["vals"][s][0] for s in vs], marker="o", ms=4, color=col, ls=ls, lw=lw, label=label)
    ax[2].plot([r["el"][min(s, n - 1)] for s in vs], [r["vals"][s][0] for s in vs], marker="o", ms=4, color=col, ls=ls, lw=lw,
               label=f"{label}: {r['el'][-1]:.1f} min total")
ax[0].set(title="(a) train loss per optimizer step (no smoothing)", xlabel="epoch (5,000 chats each)", ylabel="loss (answer tokens)")
ax[1].set(title="(b) validation loss (the same 496 val chats)", xlabel="epoch", ylabel="val loss")
ax[2].set(title="(c) validation loss vs wall-clock time", xlabel="minutes since start (1x H200)", ylabel="val loss")
for a in ax: a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle("Same chats in every step: padding vs packing, from blumodel_step_8000, AdamW 2e-5 cosine, 2 epochs, 156 steps, "
             "1x H200, BluTrain SFT kernels", fontsize=11)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=140)
print(f"figure: {OUT}.png")
