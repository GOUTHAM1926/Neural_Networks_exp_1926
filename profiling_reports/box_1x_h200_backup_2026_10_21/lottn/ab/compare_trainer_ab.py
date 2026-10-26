# 2026-10-21: summary of ab_trainer.sh (OLD vs NEW trainer A/B, runs old_r1 new_r1 old_r2 new_r2).
# Reads <run>.log (the trainer's per-step lines) and <run>.smi.csv (nvidia-smi every 500 ms) from the run folder.
#   precision: train loss, val loss and grad norm per step; |new - old| next to |old_r1 - old_r2| (the run-to-run noise:
#              OLD's backward dQ is not bit-reproducible, so two OLD runs drift apart too). Step 0 is forward-only for
#              the loss (identical weights), its norm is the first backward.
#   speed:     mean dt over the SAME steps in every run: 3 .. STOP without the validation steps (step % val_freq == 0).
#   memory:    peak memory.used of GPU 0 during each run (the caching allocator's reserve + the CUDA context).
# usage: python3 compare_trainer_ab.py <run folder> <stop step> [val_freq, default 4]
import csv, re, statistics, sys

R, STOP = sys.argv[1], int(sys.argv[2]); VF = int(sys.argv[3]) if len(sys.argv) > 3 else 4
RUNS = ["old_r1", "new_r1", "old_r2", "new_r2"]
PAT = re.compile(r"^step\s+(\d+) \| train (\S+) \| val (\S+) \| tok/s (\S+) \| norm (\S+) \| dt (\S+) ms")
steps = {}
for r in RUNS:
    steps[r] = {}
    try:
        for line in open(f"{R}/{r}.log", errors="replace"):
            m = PAT.match(line)
            if m:
                s = int(m.group(1))
                steps[r][s] = dict(train=float(m.group(2)), val=float(m.group(3)), toks=float(m.group(4)),
                                   norm=float(m.group(5)), dt=float(m.group(6)))
    except FileNotFoundError:
        pass
print(f"run folder {R} | stop step {STOP} | validation every {VF} steps")
for r in RUNS:
    print(f"  {r}: {len(steps[r])} step lines" + ("" if steps[r] else "  <-- NO STEP LINES (see the log)"))
common = sorted(set.intersection(*(set(steps[r]) for r in RUNS))) if all(steps[r] for r in RUNS) else []
if not common:
    print("no step present in all 4 runs: nothing to compare"); sys.exit(1)

def row(key, fmt, s):
    return " ".join(fmt.format(steps[r][s][key]) for r in RUNS)
print(f"\nper step (columns {' '.join(RUNS)})")
print(f"{'step':>4} | {'train loss':^43} | {'grad norm':^31} | {'dt ms':^39}")
for s in common:
    print(f"{s:4d} | {row('train', '{:10.6f}', s)} | {row('norm', '{:7.4f}', s)} | {row('dt', '{:9.1f}', s)}")
vsteps = [s for s in common if s % VF == 0]
if vsteps:
    print(f"\nvalidation loss (refreshed at steps {vsteps})")
    for s in vsteps:
        print(f"{s:4d} | {row('val', '{:10.6f}', s)}")

def maxdiff(key, a, b, ss):
    return max(abs(steps[a][s][key] - steps[b][s][key]) for s in ss)
print("\nprecision: largest difference over the common steps")
for key, ss in (("train", common), ("norm", common), ("val", vsteps or common)):
    noise = maxdiff(key, "old_r1", "old_r2", ss)
    d1, d2 = maxdiff(key, "new_r1", "old_r1", ss), maxdiff(key, "new_r2", "old_r2", ss)
    nn = maxdiff(key, "new_r1", "new_r2", ss)
    print(f"  {key:5s}: |new - old| {d1:.6f} (r1) {d2:.6f} (r2)   |old_r1 - old_r2| {noise:.6f}   |new_r1 - new_r2| {nn:.6f}")
s0 = common[0]
print(f"  step {s0} train loss (forward only, same weights): {row('train', '{:.6f}', s0)}"
      f"  -> {'all equal' if len({steps[r][s0]['train'] for r in RUNS}) == 1 else 'NOT all equal'}")

win = [s for s in common if 3 <= s <= STOP and s % VF != 0]
print(f"\nspeed: mean dt over steps {win}")
mdt = {r: statistics.mean(steps[r][s]["dt"] for s in win) for r in RUNS} if win else {}
for r in RUNS:
    if win:
        print(f"  {r}: dt {mdt[r]:9.1f} ms | tok/s {statistics.mean(steps[r][s]['toks'] for s in win):10.1f}")
if win:
    old, new = (mdt["old_r1"] + mdt["old_r2"]) / 2, (mdt["new_r1"] + mdt["new_r2"]) / 2
    print(f"  NEW / OLD step time {new / old:.4f} ({(1 - new / old) * 100:+.2f}% faster) | pairs: "
          f"r1 {mdt['new_r1'] / mdt['old_r1']:.4f}, r2 {mdt['new_r2'] / mdt['old_r2']:.4f}")

print("\nGPU 0 during each run (nvidia-smi every 500 ms)")
peaks = {}
for r in RUNS:
    mem, clk, pw = [], [], []
    try:
        for row_ in csv.reader(open(f"{R}/{r}.smi.csv")):
            try:
                mem.append(float(row_[1].replace("MiB", ""))); clk.append(float(row_[2].replace("MHz", "")))
                pw.append(float(row_[3].replace("W", "")))
            except (ValueError, IndexError):
                pass
    except FileNotFoundError:
        pass
    if mem:
        busy = [i for i, p in enumerate(pw) if p > 300]   # samples while training (not startup / teardown)
        peaks[r] = max(mem)
        print(f"  {r}: peak memory {max(mem):8.0f} MiB | while busy: SM clock {statistics.mean(clk[i] for i in busy) if busy else 0:6.0f} MHz, "
              f"power {statistics.mean(pw[i] for i in busy) if busy else 0:5.0f} W ({len(busy)} samples)")
if len(peaks) == 4:
    o, n = (peaks["old_r1"] + peaks["old_r2"]) / 2, (peaks["new_r1"] + peaks["new_r2"]) / 2
    print(f"  peak memory NEW - OLD: {n - o:+.0f} MiB ({o:.0f} -> {n:.0f})")
