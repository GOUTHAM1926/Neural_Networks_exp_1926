#!/usr/bin/env python3
# plot_sft_runs.py (2026-10-21): padding vs packing, the two SFT runs of run_all_sftb.sh (phase C), from their run.log,
# sft_adamw.csv and gpu.csv (nvidia-smi every second).
#   figure  (a) train loss per step vs epoch, (b) val loss vs epoch, (c) val loss vs wall-clock minutes
#   table   steps, steps per epoch, step time (median of steps >= 10; steps 0-9 hold the start-up and the trainer's
#           empty_cache() steps), training time per epoch, wall time of the whole run, tokens computed per second, trained
#           (answer) tokens per second, chats per second, peak GPU memory (nvidia-smi), val loss at the start and the end
# usage: PYTHONPATH=<matplotlib> python3 plot_sft_runs.py <runs folder> [out prefix]
import csv, re, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = 4096
CHATS = 5000
RUNS = [("C_P_padded", "padding (1 chat per row, B 8, 64 rows/step)", 64, "#d62728"),
        ("C_K_packed", "packing (First-Fit Decreasing, B 4, 20 rows/step)", 20, "#1f77b4")]
STEP = re.compile(r"^step\s+(\d+) \| train ([0-9.]+) \| val (\S+) \| tok/s ([0-9.]+) \| norm ([0-9.]+) \| dt ([0-9.]+) ms")
VAL = re.compile(r"\[val\] step (\d+) \| val_loss ([0-9.]+)")


def load(d, rows_per_step):
    steps, vals = {}, {}
    for line in open(f"{d}/run.log", errors="replace"):
        m = STEP.match(line)
        if m: steps[int(m.group(1))] = {"train": float(m.group(2)), "dt": float(m.group(6)) / 1000.0}
        m = VAL.search(line)
        if m: vals[int(m.group(1))] = float(m.group(2))
    rows = list(csv.DictReader(open(f"{d}/sft_adamw.csv")))
    for r in rows:
        s = int(r["step"])
        if s in steps:
            steps[s]["elapsed_min"] = float(r["elapsed_min"])
            steps[s]["trained"] = float(r.get("trained_tokens") or "nan")
    mem = []
    for r in csv.reader(open(f"{d}/gpu.csv")):
        try: mem.append(float(r[1].split()[0]))
        except (IndexError, ValueError): pass
    n = len(steps)
    spe = n / 2.0   # 2 epochs
    late = [v["dt"] for s, v in steps.items() if s >= 10]
    dt = statistics.median(late)
    trained = [v["trained"] for v in steps.values() if v.get("trained") == v.get("trained")]
    last = max(steps)
    return {"steps": steps, "vals": vals, "n": n, "spe": spe, "dt": dt, "dt_mean": statistics.mean(late),
            "wall_min": steps[last].get("elapsed_min", float("nan")), "peak_mib": max(mem) if mem else float("nan"),
            "tok_s": rows_per_step * T / dt, "trained_s": statistics.mean(trained) / dt if trained else float("nan"),
            "chats_s": CHATS / spe / dt, "epoch_min": spe * dt / 60.0, "rows": rows_per_step}


def main():
    root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{root}/sft_padding_vs_packing"
    res = {name: load(f"{root}/{name}", rps) for name, _, rps, _ in RUNS}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for name, label, _, col in RUNS:
        r = res[name]; st = sorted(r["steps"])
        ep = [s / r["spe"] for s in st]
        tr = [r["steps"][s]["train"] for s in st]
        ax[0].plot(ep, tr, color=col, alpha=0.25, lw=0.8)
        k = 5
        sm = [statistics.mean(tr[max(0, i - k + 1):i + 1]) for i in range(len(tr))]
        ax[0].plot(ep, sm, color=col, lw=2, label=f"{label} (5-step mean)")
        vs = sorted(r["vals"])
        ax[1].plot([s / r["spe"] for s in vs], [r["vals"][s] for s in vs], "o-", color=col, ms=4, lw=1.6, label=label)
        wall = [r["steps"].get(min(s, max(st)), {}).get("elapsed_min", float("nan")) for s in vs]
        ax[2].plot(wall, [r["vals"][s] for s in vs], "o-", color=col, ms=4, lw=1.6,
                   label=f"{label}: {r['wall_min']:.1f} min total")
    ax[0].set(title="(a) train loss per optimizer step", xlabel="epoch (5,000 chats each)", ylabel="loss (answer tokens)")
    ax[1].set(title="(b) validation loss (500 held-out chats)", xlabel="epoch", ylabel="val loss")
    ax[2].set(title="(c) validation loss vs wall-clock time", xlabel="minutes since start (1x H200)", ylabel="val loss")
    for a in ax: a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("SFT padding vs packing: same 5,000 UltraChat chats, from blumodel_step_8000, AdamW 2e-5 cosine, 2 epochs, "
                 "~64 chats per step, 1x H200, BluTrain SFT kernels", fontsize=11)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=140)
    P, K = res["C_P_padded"], res["C_K_packed"]
    lines = [f"{'':34s} {'padding':>12s} {'packing':>12s} {'packing / padding':>18s}"]
    def row(lbl, a, b, fmt="{:12.2f}", ratio=True):
        lines.append(f"{lbl:34s} {fmt.format(a)} {fmt.format(b)} " + (f"{b / a:18.3f}" if ratio else ""))
    row("optimizer steps", P["n"], K["n"], "{:12.0f}")
    row("steps per epoch", P["spe"], K["spe"], "{:12.1f}")
    row("rows per step", P["rows"], K["rows"], "{:12.0f}")
    row("step time, median (s)", P["dt"], K["dt"], "{:12.3f}")
    row("step time, mean of steps>=10 (s)", P["dt_mean"], K["dt_mean"], "{:12.3f}")
    row("training time per epoch (min)", P["epoch_min"], K["epoch_min"])
    row("whole run incl. val + load (min)", P["wall_min"], K["wall_min"])
    row("tokens computed per s", P["tok_s"], K["tok_s"], "{:12.0f}")
    row("trained (answer) tokens per s", P["trained_s"], K["trained_s"], "{:12.0f}")
    row("chats per s", P["chats_s"], K["chats_s"])
    row("peak GPU memory (MiB, nvidia-smi)", P["peak_mib"], K["peak_mib"], "{:12.0f}")
    v0 = lambda r: r["vals"][min(r["vals"])]; v1 = lambda r: r["vals"][max(r["vals"])]
    row("val loss at the start", v0(P), v0(K), "{:12.6f}", False)
    row("val loss at the end", v1(P), v1(K), "{:12.6f}", False)
    txt = "\n".join(lines)
    open(out + ".txt", "w").write(txt + "\n")
    print(txt)
    print(f"figure: {out}.png")


if __name__ == "__main__":
    main()
