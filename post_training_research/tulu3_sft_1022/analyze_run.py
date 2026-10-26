# Tulu 3 SFT run (2026-10-22/23): joins the three run logs into one file, rebuilds the final step-by-step trajectory,
# draws the loss charts and prints the numbers used for the overfitting reading
# (work_docs/overfitting_check_val_dev_suite_and_unseen_suite.md, sections 5 and 10).
#
#   Logs (box -> laptop ~/Downloads/box_backup_2026_10_22/tulu3_run_final/):
#     run.log            steps 0-8723     old box, GPU 0, 10-22 15:51-20:43 IST (stopped)
#     run_resume1.log    steps 8000-9194  same box, resumed from checkpoint 8000, 10-22 20:48-21:30 IST (stopped)
#     run_resume_4x.log  steps 9000-11589 4x box, GPU 3, resumed from checkpoint 9000, 10-23 13:15-14:42 IST (exit 0)
#   The final model's steps: 0-7999 from run.log, 8000-8999 from run_resume1.log, 9000-11589 from run_resume_4x.log.
#
#   "[val] step N" is computed at the top of step N, before its update: the weights after N updates, the same weights
#   as checkpoint blumodel_step_N (bluscript_sft.cpp:2304, :2857).
#   Rows are read in the same order in every epoch (sft/SftDataLoader.h:25-26, no shuffle); one epoch = 28,976
#   micro-batches of 4 rows = 5,795.2 steps of 5 micro-batches, so epoch-2 step 5795+k holds 4 of the 5 micro-batches
#   of epoch-1 step k.
#
# usage: PYTHONPATH=<dir with matplotlib> python3 analyze_run.py <dir with the three logs> <output dir>
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGDIR, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
PARTS = [("run.log", 0, 8000), ("run_resume1.log", 8000, 9000), ("run_resume_4x.log", 9000, 11590)]
STEPS_PER_EPOCH = 28976 / 5            # 5795.2
E2 = 5795                              # first step holding epoch-2 rows

STEP_RE = re.compile(r"^step +(\d+) \| train ([0-9.]+) \| val ([0-9.-]+) \| tok/s ([0-9.]+) \| norm ([0-9.]+) \| "
                     r"dt ([0-9.]+) ms \| mfu ([0-9.]+)% .*?\| lr ([0-9.e+-]+) .*?\| trained (\d+)")
VAL_RE = re.compile(r"^\s*\[val\] step (\d+) \| val_loss ([0-9.]+)")

def parse(path):
    steps, vals = {}, {}
    for line in open(path, errors="replace"):
        m = STEP_RE.match(line)
        if m:
            s = int(m.group(1))
            steps[s] = dict(train=float(m.group(2)), toks=float(m.group(4)), norm=float(m.group(5)), dt=float(m.group(6)),
                            mfu=float(m.group(7)), lr=float(m.group(8)), trained=int(m.group(9)))
            continue
        m = VAL_RE.match(line)
        if m: vals[int(m.group(1))] = float(m.group(2))
    return steps, vals

parsed = {name: parse(os.path.join(LOGDIR, name)) for name, _, _ in PARTS}

# ---------------------------------------------------------------- 1. the joined log, every line as it was
joined = os.path.join(OUT, "tulu3_sft_1022_full_run.log")
with open(joined, "w") as f:
    f.write("# Tulu 3 SFT run, 2026-10-22/23: the run's three log files joined as they are (no line changed or removed).\n"
            "#   part 1  run.log            steps 0-8723     old box 86.38.182.101, GPU 0, 10-22 15:51-20:43 IST (stopped)\n"
            "#   part 2  run_resume1.log    steps 8000-9194  same box, resumed from checkpoint 8000, 10-22 20:48-21:30 IST (stopped)\n"
            "#   part 3  run_resume_4x.log  steps 9000-11589 4x box 95.133.252.52, GPU 3, resumed from checkpoint 9000,\n"
            "#                              10-23 13:15-14:42 IST (finished, exit 0)\n"
            "# The final model's steps: 0-7999 from part 1, 8000-8999 from part 2, 9000-11589 from part 3. After each resume the\n"
            "# steps after the checkpoint were trained again, so parts 1 and 2 end with steps that part 2 / part 3 repeat.\n")
    for i, (name, a, b) in enumerate(PARTS, 1):
        f.write(f"\n########## part {i}: {name} ##########\n")
        f.write(open(os.path.join(LOGDIR, name), errors="replace").read())

# ---------------------------------------------------------------- 2. the final trajectory
steps, vals, src = {}, {}, {}
for name, a, b in PARTS:
    st, va = parsed[name]
    for s in range(a, b):
        if s in st: steps[s] = st[s]; src[s] = name
    for s, v in va.items():
        if a <= s < b or (name == PARTS[-1][0] and s == 11589): vals[s] = v
missing = [s for s in range(11590) if s not in steps]
S = np.arange(11590)
train = np.array([steps[s]["train"] for s in S]); lr = np.array([steps[s]["lr"] for s in S])
vsteps = np.array(sorted(vals)); vloss = np.array([vals[s] for s in vsteps])
with open(os.path.join(OUT, "trajectory.csv"), "w") as f:
    f.write("step,train_loss,val_loss_if_measured,lr,grad_norm,trained_tokens,tok_per_s,mfu_pct,source_log\n")
    for s in S:
        d = steps[s]
        f.write(f"{s},{d['train']},{vals.get(s, '')},{d['lr']},{d['norm']},{d['trained']},{d['toks']},{d['mfu']},{src[s]}\n")

def smooth(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0)); out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out
tr100 = smooth(train, 100)

# ---------------------------------------------------------------- 3. numbers
print(f"steps in the final trajectory: {len(steps)} (missing {len(missing)}), val points: {len(vals)}")
print("consistency at the resume points (same checkpoint -> val must match):")
for s, a, b in [(8000, "run.log", "run_resume1.log"), (9000, "run_resume1.log", "run_resume_4x.log")]:
    va, vb = parsed[a][1].get(s), parsed[b][1].get(s)
    print(f"  val @ {s}: {a} {va} | {b} {vb} | {'IDENTICAL' if va == vb else 'differ'}")
    ta, tb = parsed[a][0][s]["train"], parsed[b][0][s]["train"]
    print(f"  train @ {s}: {ta} | {tb} | {'IDENTICAL' if ta == tb else 'differ'}")
print("\nval loss along the run (weights after N updates):")
for s in [0, 200, 400, 600, 1000, 2000, 3000, 4000, 5000, 5800, 6000, 7000, 8000, 9000, 10000, 11000, 11589]:
    print(f"  step {s:5d}  epoch {1 if s < E2 else 2}  val {vals[s]:.6f}")
def val_at(x): return float(np.interp(x, vsteps, vloss))
v_e1 = val_at(STEPS_PER_EPOCH)
print(f"\nval at the end of epoch 1 (interpolated, step 5795.2): {v_e1:.6f}")
print(f"val drop epoch 1 (0 -> 5795): {vals[0] - v_e1:.6f} | epoch 2 (5795 -> 11589): {v_e1 - vals[11589]:.6f}")
mn = int(vsteps[np.argmin(vloss)]); print(f"lowest val: {vloss.min():.6f} at step {mn}")
rises = [(int(vsteps[i]), float(vloss[i] - vloss[i - 1])) for i in range(1, len(vloss)) if vloss[i] > vloss[i - 1]]
print(f"val points that rose over the previous one: {len(rises)} of {len(vloss) - 1}; largest rise "
      f"{max(r for _, r in rises) if rises else 0:.6f}" + (f" (at step {max(rises, key=lambda x: x[1])[0]})" if rises else ""))
print("val change per 1000 steps in epoch 2:")
for a in range(6000, 11000, 1000):
    b = min(a + 1000, 11589); print(f"  {a:5d} -> {b:5d}: {vals[b] - vals[a]:+.6f}")
for nm, a, b in [("epoch 1, first 500 steps", 0, 500), ("epoch 1, last 500 steps", 5295, 5795),
                 ("epoch 2, first 500 steps", 5795, 6295), ("epoch 2, last 500 steps", 11090, 11590)]:
    print(f"train mean, {nm} ({a}-{b - 1}): {train[a:b].mean():.4f}")
print(f"gap val - train (500-step train mean): end of epoch 1 {v_e1 - train[5295:5795].mean():+.4f} | "
      f"end of epoch 2 {vals[11589] - train[11090:11590].mean():+.4f}")
print(f"lr: step 0 {lr[0]:.3e}, peak {lr.max():.3e} at step {int(lr.argmax())}, end {lr[-1]:.3e}")
for name, a, b in PARTS:
    seg = [steps[s] for s in range(max(a, 5) if a == 0 else a + 1, b)]
    print(f"speed {name:18s}: mean {np.mean([d['toks'] for d in seg]):,.0f} tok/s, mean MFU {np.mean([d['mfu'] for d in seg]):.2f}%")

# memorization: epoch-1 step k and epoch-2 step 5795+k share 4 of 5 micro-batches. Over a window of W steps,
#   rows_drop = mean train(epoch 1 window) - mean train(epoch 2 window)   (the same rows, before vs after training on them once)
#   val_drop  = val at the epoch-1 window's middle - val at the epoch-2 window's middle   (improvement on chats never trained on)
#   extra     = rows_drop - val_drop  = how much more the model improved on the rows it had trained on (memorization)
W = 200
mem = []
for k in range(0, 11590 - E2 - W + 1, W):
    e1, e2 = train[k:k + W].mean(), train[E2 + k:E2 + k + W].mean()
    c1, c2 = k + W / 2, E2 + k + W / 2
    rows_drop, val_drop = e1 - e2, val_at(c1) - val_at(c2)
    mem.append((E2 + k, rows_drop, val_drop, rows_drop - val_drop))
mem = np.array(mem)
print(f"\nmemorization measure, {len(mem)} windows of {W} steps (epoch-2 steps {E2}-{int(mem[-1, 0]) + W - 1}):")
for lo in [0, 0.5]:
    sel = mem[mem[:, 0] >= E2 + lo * (11590 - E2)]
    print(f"  {'all windows' if lo == 0 else 'second half of epoch 2'}: rows_drop {sel[:, 1].mean():.4f}, "
          f"val_drop {sel[:, 2].mean():.4f}, extra (memorization) {sel[:, 3].mean():+.4f}")
first = mem[mem[:, 0] >= E2 + 1000][0]
print(f"  e.g. window from epoch-2 step {int(first[0])}: rows_drop {first[1]:.4f}, val_drop {first[2]:.4f}, extra {first[3]:+.4f}")
np.savetxt(os.path.join(OUT, "memorization_windows.csv"), mem, delimiter=",", fmt="%.6f",
           header="epoch2_window_start,rows_drop,val_drop,extra_memorization", comments="")

# ---------------------------------------------------------------- 4. charts (reference palette, light surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 13, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
def frame(ax):
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
def epoch_line(ax, ytext):
    ax.axvline(STEPS_PER_EPOCH, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(STEPS_PER_EPOCH + 80, ytext, "epoch 2 starts\n(step 5795)", color=INK2, fontsize=10, va="top")

# (a) train + val
fig, ax = plt.subplots(figsize=(13, 6.5)); frame(ax)
ax.plot(S, train, color=BLUE, alpha=0.18, linewidth=0.6, label="train loss, every step")
ax.plot(S, tr100, color=BLUE, linewidth=2, label="train loss, mean of the last 100 steps")
ax.plot(vsteps, vloss, color=ORANGE, linewidth=2, marker="o", markersize=4, label="val loss (2,000 held-out chats, every 200 steps)")
ax.set_ylim(0.55, 1.25); ax.set_xlim(0, 11700)
epoch_line(ax, 1.23)
ax.annotate(f"start {vals[0]:.4f}", (0, vals[0]), xytext=(250, 1.20), color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2))
ax.annotate(f"end {vals[11589]:.4f}", (11589, vals[11589]), xytext=(10300, 0.95), color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2))
ax.set_xlabel("optimizer step (20 packed rows of 4,096 tokens each)"); ax.set_ylabel("loss (nats per trained token)")
ax.set_title("Tulu 3 SFT, 1.7B (fullchess base): train and val loss over 2 epochs")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_train_val.png"), dpi=150); plt.close(fig)

# (b) val only: full range and the zoom on the flat part
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5)); frame(a1); frame(a2)
for ax in (a1, a2):
    ax.plot(vsteps, vloss, color=ORANGE, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("val loss")
a1.set_title("Val loss, whole run"); epoch_line(a1, vloss.max())
z = vsteps >= 3000
a2.set_title("Val loss from step 3000 (zoomed)"); a2.set_xlim(2900, 11700)
lo_, hi_ = vloss[z].min(), vloss[z].max(); a2.set_ylim(lo_ - 0.002, hi_ + 0.003); epoch_line(a2, hi_ + 0.0025)
for s in (6000, 8000, 10000, 11589):
    off, ha = ((8, -14), "left") if s == 6000 else ((0, 9), "center")   # 6000 sits on the epoch line: label below-right
    a2.annotate(f"{vals[s]:.4f}", (s, vals[s]), xytext=off, textcoords="offset points", ha=ha, fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "loss_val.png"), dpi=150); plt.close(fig)

# (c) memorization measure
fig, ax = plt.subplots(figsize=(13, 5.8)); frame(ax)
x = mem[:, 0] + W / 2
ax.plot(x, mem[:, 1], color=BLUE, linewidth=2, marker="o", markersize=4, label="drop on the SAME rows (epoch-1 loss - epoch-2 loss)")
ax.plot(x, mem[:, 2], color=ORANGE, linewidth=2, marker="o", markersize=4, label="drop on NEVER-trained chats (val loss, same time span)")
ax.fill_between(x, mem[:, 2], mem[:, 1], color=AQUA, alpha=0.18, label="gap = extra gain on rows already trained on (memorization)")
ax.axhline(0, color=INK2, linewidth=0.8)
ax.set_xlabel("epoch-2 step (window of 200 steps; the epoch-1 window is 5795 steps earlier)"); ax.set_ylabel("loss drop (nats per trained token)")
ax.set_title("Memorization check: how much more the model improved on the rows it had already seen")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "memorization_check.png"), dpi=150); plt.close(fig)
print(f"\nwrote {joined}, trajectory.csv, memorization_windows.csv, loss_train_val.png, loss_val.png, memorization_check.png")
